#!/usr/bin/env python3
"""
Train Orbital Bee Foraging with Per-Bee Retasking Board (v2)
=============================================================
Each bee has its own retask board; when a bee's battery dies, it broadcasts
its assigned tasks to the nearest active bee. Tasks propagate via communication
chains - if a bee can't perform a task, it holds it until close enough to
another bee to pass it on.

Key Features:
- Per-bee retasking board (not shared)
- Communication chain propagation (hops tracking)
- Nearest-neighbor task broadcast on battery death
- Orbital motion with battery/recharge dynamics
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from bee_policy import Actor, CentralizedCritic
from bees_env import BeeForagingEnv
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train_utils import load_config, save_models


# -----------------------------
# Seed helper
# -----------------------------
def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Reward Tracker
# -----------------------------
class RewardTracker:
    def __init__(self, window_size=100):
        self.episode_rewards = []
        self.window_size = window_size

    def add_episode(self, reward):
        self.episode_rewards.append(reward)
        if len(self.episode_rewards) > self.window_size:
            self.episode_rewards.pop(0)

    def mean_reward(self):
        return np.mean(self.episode_rewards) if self.episode_rewards else 0.0

    def best_reward(self):
        return max(self.episode_rewards) if self.episode_rewards else 0.0


# -----------------------------
# Tensor utilities
# -----------------------------
def _ensure_tensor(x, device="cpu"):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def obs_to_tensor(obs_dict: dict, device: str = "cpu") -> dict:
    """Convert observation dict to tensor dict for actor."""
    return {
        "position": _ensure_tensor(obs_dict["position"], device).unsqueeze(0),
        "status": _ensure_tensor(obs_dict["status"], device).unsqueeze(0),
        "flowers": _ensure_tensor(obs_dict["flowers"], device).unsqueeze(0),
        "step_count": _ensure_tensor(obs_dict["step_count"], device).unsqueeze(0),
        "consensus": _ensure_tensor(obs_dict["consensus"], device).unsqueeze(0),
        "retask_board": _ensure_tensor(obs_dict.get("retask_board", np.zeros(15)), device).unsqueeze(0),
        "action_availability": _ensure_tensor(
            obs_dict.get("action_availability", np.ones(3)), device
        ).unsqueeze(0),
    }


def obs_to_global_state(obs_dict: dict, num_bees: int, device: str = "cpu") -> torch.Tensor:
    """Convert all bee observations to a single global state tensor for centralized critic."""
    all_features = []
    for i in range(num_bees):
        ob = obs_dict.get(f"bee_{i}", {})
        features = np.concatenate([
            np.array(ob.get("position", [0, 0, 0]), dtype=np.float32),
            np.array(ob.get("status", [0, 0]), dtype=np.float32),
            np.array(ob.get("flowers", np.zeros(72)), dtype=np.float32),
            np.array(ob.get("step_count", [0]), dtype=np.float32),
            np.array(ob.get("consensus", np.zeros(num_bees)), dtype=np.float32),
            np.array(ob.get("retask_board", np.zeros(10)), dtype=np.float32),
        ])
        all_features.append(features)
    
    global_state = np.concatenate(all_features)
    return torch.tensor(global_state, dtype=torch.float32, device=device).unsqueeze(0)


def collect_rollout(
    env: BeeForagingEnv,
    actor: nn.Module,
    critic: nn.Module,
    rollout_len: int,
    gamma: float,
    lam: float,
    device: str,
    stochastic: bool = True,
) -> dict:
    """
    Collect a rollout of `rollout_len` steps and compute GAE advantages.
    Returns batch dict with observations, actions, rewards, values, advantages, returns.
    """
    obs = env.reset()
    num_bees = len(env.bees)

    # Storage
    all_obs = []
    all_actions = []
    all_rewards = []
    all_values = []
    all_logprobs = []
    all_dones = []

    for step in range(rollout_len):
        step_obs = {}
        step_actions = {}
        step_logprobs = []

        # Get global state for critic (once per step)
        global_state = obs_to_global_state(obs, num_bees, device)
        with torch.no_grad():
            step_value = critic(global_state).item()

        for i in range(num_bees):
            agent = f"bee_{i}"
            ob = obs[agent]
            ob_tensor = obs_to_tensor(ob, device)

            with torch.no_grad():
                # Get action logits from actor
                action_logits = actor(ob_tensor)
                
                # Apply action masking if available
                # NOTE: action_availability = [can_harvest, can_groom, can_do_nothing]
                # But action space is: 0=DONOTHING, 1=HARVEST, 2=GROOM
                # So we need to reorder the mask to match action indices
                if "action_availability" in ob:
                    avail = ob["action_availability"]
                    # Reorder: [donothing, harvest, groom] to match action indices
                    mask = _ensure_tensor([avail[2], avail[0], avail[1]], device)
                    # Set unavailable actions to very low logits
                    action_logits = action_logits + (mask - 1.0) * 1e8
                
                probs = F.softmax(action_logits, dim=-1)
                
                if stochastic:
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                else:
                    action = torch.argmax(probs, dim=-1)
                    logprob = torch.log(probs.gather(-1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)

            step_obs[agent] = ob
            step_actions[agent] = int(action.item())
            step_logprobs.append(logprob.item())

        # Step environment
        next_obs, rewards, terminated, truncated, infos, _ = env.step(step_actions)

        # Store (use same value for all bees since centralized)
        all_obs.append(step_obs)
        all_actions.append(step_actions)
        all_rewards.append([rewards[f"bee_{i}"] for i in range(num_bees)])
        all_values.append([step_value] * num_bees)  # Centralized value
        all_logprobs.append(step_logprobs)
        all_dones.append(any(terminated.values()) or any(truncated.values()))

        obs = next_obs

        if all_dones[-1]:
            break

    # Compute GAE advantages
    T = len(all_rewards)
    advantages = np.zeros((T, num_bees), dtype=np.float32)
    returns = np.zeros((T, num_bees), dtype=np.float32)
    
    # Bootstrap value for last step
    last_values = np.zeros(num_bees, dtype=np.float32)
    if not all_dones[-1]:
        global_state = obs_to_global_state(obs, num_bees, device)
        with torch.no_grad():
            last_value = critic(global_state).item()
        last_values[:] = last_value  # Same value for all bees (centralized)

    gae = np.zeros(num_bees, dtype=np.float32)
    for t in reversed(range(T)):
        if t == T - 1:
            next_values = last_values
            next_nonterminal = 1.0 - float(all_dones[-1])
        else:
            next_values = np.array(all_values[t + 1], dtype=np.float32)
            next_nonterminal = 1.0 - float(all_dones[t])

        rewards_t = np.array(all_rewards[t], dtype=np.float32)
        values_t = np.array(all_values[t], dtype=np.float32)
        
        delta = rewards_t + gamma * next_values * next_nonterminal - values_t
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae
        returns[t] = gae + values_t

    return {
        "obs": all_obs,
        "actions": all_actions,
        "rewards": np.array(all_rewards, dtype=np.float32),
        "values": np.array(all_values, dtype=np.float32),
        "logprobs": np.array(all_logprobs, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "episode_reward": float(np.sum(all_rewards)),
        "episode_len": T,
    }


def ppo_update(
    actor: nn.Module,
    critic: nn.Module,
    opt_actor: optim.Optimizer,
    opt_critic: optim.Optimizer,
    batch: dict,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    device: str,
    num_epochs: int = 4,
    minibatch_size: int = 64,
) -> dict:
    """Perform PPO update on actor and critic."""
    obs_list = batch["obs"]
    actions_list = batch["actions"]
    old_logprobs = torch.tensor(batch["logprobs"], dtype=torch.float32, device=device)
    advantages = torch.tensor(batch["advantages"], dtype=torch.float32, device=device)
    returns = torch.tensor(batch["returns"], dtype=torch.float32, device=device)

    T, num_bees = advantages.shape

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    for epoch in range(num_epochs):
        indices = np.random.permutation(T)
        
        for start in range(0, T, minibatch_size):
            end = min(start + minibatch_size, T)
            mb_indices = indices[start:end]

            policy_loss_sum = 0.0
            value_loss_sum = 0.0
            entropy_sum = 0.0
            count = 0

            for t in mb_indices:
                # Get global state for this timestep (for critic)
                global_state = obs_to_global_state(obs_list[t], num_bees, device)
                
                for i in range(num_bees):
                    agent = f"bee_{i}"
                    ob = obs_list[t][agent]
                    ob_tensor = obs_to_tensor(ob, device)
                    action = actions_list[t][agent]

                    # Forward pass
                    action_logits = actor(ob_tensor)
                    
                    # Apply action masking
                    # NOTE: action_availability = [can_harvest, can_groom, can_do_nothing]
                    # But action space is: 0=DONOTHING, 1=HARVEST, 2=GROOM
                    # So we need to reorder the mask to match action indices
                    if "action_availability" in ob:
                        avail = ob["action_availability"]
                        mask = _ensure_tensor([avail[2], avail[0], avail[1]], device)
                        action_logits = action_logits + (mask - 1.0) * 1e8

                    probs = F.softmax(action_logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    
                    new_logprob = dist.log_prob(torch.tensor([action], device=device))
                    entropy = dist.entropy()
                    value = critic(global_state)  # Use global state for centralized critic

                    # PPO ratio
                    old_lp = old_logprobs[t, i]
                    ratio = torch.exp(new_logprob - old_lp)
                    
                    # Clipped surrogate objective
                    adv = advantages[t, i]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
                    policy_loss = -torch.min(surr1, surr2)

                    # Value loss
                    ret = returns[t, i]
                    value_loss = F.mse_loss(value.squeeze(-1), ret.view(1))

                    policy_loss_sum += policy_loss.mean()
                    value_loss_sum += value_loss
                    entropy_sum += entropy.mean()
                    count += 1

            if count == 0:
                continue

            # Average losses
            policy_loss_avg = policy_loss_sum / count
            value_loss_avg = value_loss_sum / count
            entropy_avg = entropy_sum / count

            # Total loss
            loss = policy_loss_avg + value_coef * value_loss_avg - entropy_coef * entropy_avg

            # Optimize
            opt_actor.zero_grad()
            opt_critic.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            opt_actor.step()
            opt_critic.step()

            total_policy_loss += policy_loss_avg.item()
            total_value_loss += value_loss_avg.item()
            total_entropy += entropy_avg.item()
            num_updates += 1

    return {
        "policy_loss": total_policy_loss / max(1, num_updates),
        "value_loss": total_value_loss / max(1, num_updates),
        "entropy": total_entropy / max(1, num_updates),
    }


def train_orbital_v2(args):
    """Main training loop for orbital bee foraging with per-bee retasking."""
    
    # Load config
    train_cfg, env_cfg, out_dir = load_config(args.config)
    
    # Override with command line args
    if args.output:
        out_dir = args.output
    
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[train_orbital_v2] device={device}, output={out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # Create environment
    env = BeeForagingEnv(
        num_bees=env_cfg.get("num_bees", 5),
        num_flowers=env_cfg.get("num_flowers", 12),
        grid_size=env_cfg.get("grid_size", 20),
        max_steps=env_cfg.get("max_steps", 800),
        retask_board_size=env_cfg.get("retask_board_size", 3),
        harvest_radius=env_cfg.get("harvest_radius", 3.0),
        orbit_scale=env_cfg.get("orbit_scale", 1.2),
        battery_min_steps=env_cfg.get("battery_min_steps", 250),
        battery_max_steps=env_cfg.get("battery_max_steps", 450),
        recharge_steps=env_cfg.get("recharge_steps", 30),
        verbose=env_cfg.get("verbose", False),  # Quiet by default during training
    )

    print(f"[env] num_bees={env.num_bees}, num_flowers={env.num_flowers}, "
          f"retask_board_size={env.retask_board_size}")

    # Create networks
    hidden_dim = train_cfg.get("hidden_dim", 256)
    actor = Actor(
        num_bees=env.num_bees,
        num_flowers=env.num_flowers,
        retask_board_size=env.retask_board_size,
        hidden_dim=hidden_dim,
    ).to(device)

    # Calculate global state size for critic (position + status per bee + flower features)
    # position(3) + status(2) + flowers(num_flowers*12) + step(1) + consensus(num_bees) + retask_board(5*size)
    obs_dim_per_bee = 3 + 2 + (env.num_flowers * 12) + 1 + env.num_bees + (5 * env.retask_board_size)
    global_state_size = obs_dim_per_bee * env.num_bees  # Centralized critic sees all bees

    critic = CentralizedCritic(
        global_state_size=global_state_size,
        num_bees=env.num_bees,
        hidden_dim=hidden_dim,
    ).to(device)

    opt_actor = optim.Adam(actor.parameters(), lr=train_cfg.get("lr_actor", 3e-4))
    opt_critic = optim.Adam(critic.parameters(), lr=train_cfg.get("lr_critic", 3e-4))

    # Resume from checkpoint if specified
    start_update = getattr(args, 'start_update', 1) or 1
    if hasattr(args, 'resume') and args.resume:
        resume_dir = args.resume
        actor_path = os.path.join(resume_dir, "outputs_actor.pt")
        critic_path = os.path.join(resume_dir, "outputs_critic.pt")
        
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            print(f"[RESUME] Loading checkpoint from {resume_dir}")
            actor.load_state_dict(torch.load(actor_path, map_location=device))
            critic.load_state_dict(torch.load(critic_path, map_location=device))
            print(f"[RESUME] Loaded actor and critic weights")
            
            # Try to infer start_update from directory name (e.g., upd400 -> 401)
            if start_update == 1:
                dir_name = os.path.basename(resume_dir)
                if dir_name.startswith("upd"):
                    try:
                        start_update = int(dir_name[3:]) + 1
                        print(f"[RESUME] Inferred start_update={start_update} from checkpoint name")
                    except ValueError:
                        pass
        else:
            print(f"[RESUME] Warning: Checkpoint not found at {resume_dir}, starting fresh")

    # TensorBoard
    writer = SummaryWriter(out_dir)

    # Training params
    num_updates = train_cfg.get("num_updates", 1500)
    rollout_len = train_cfg.get("rollout_len", 200)
    gamma = train_cfg.get("gamma", 0.99)
    lam = train_cfg.get("lam", 0.95)
    clip_ratio = train_cfg.get("clip_ratio", 0.2)
    value_coef = train_cfg.get("value_coef", 0.5)
    entropy_coef = train_cfg.get("entropy_coef", 0.01)

    reward_tracker = RewardTracker(window_size=100)
    best_mean_reward = -float("inf")

    remaining_updates = num_updates - start_update + 1
    print(f"\n[train_orbital_v2] Starting training for {remaining_updates} updates (from {start_update} to {num_updates})...")
    print(f"  rollout_len={rollout_len}, gamma={gamma}, lam={lam}")
    print(f"  clip_ratio={clip_ratio}, value_coef={value_coef}, entropy_coef={entropy_coef}")
    print()

    for update in tqdm(range(start_update, num_updates + 1), desc="Training", initial=start_update-1, total=num_updates):
        # Collect rollout
        batch = collect_rollout(
            env, actor, critic, rollout_len, gamma, lam, device, stochastic=True
        )

        # PPO update
        losses = ppo_update(
            actor, critic, opt_actor, opt_critic, batch,
            clip_ratio, value_coef, entropy_coef, device,
            num_epochs=2, minibatch_size=64
        )

        # Track reward
        reward_tracker.add_episode(batch["episode_reward"])
        mean_reward = reward_tracker.mean_reward()

        # Log to TensorBoard
        writer.add_scalar("reward/episode", batch["episode_reward"], update)
        writer.add_scalar("reward/mean_100", mean_reward, update)
        writer.add_scalar("loss/policy", losses["policy_loss"], update)
        writer.add_scalar("loss/value", losses["value_loss"], update)
        writer.add_scalar("loss/entropy", losses["entropy"], update)
        writer.add_scalar("episode/length", batch["episode_len"], update)

        # Print progress
        if update % 50 == 0:
            print(
                f"[upd {update:4d}] reward={batch['episode_reward']:.1f}, "
                f"mean100={mean_reward:.1f}, len={batch['episode_len']}, "
                f"policy_loss={losses['policy_loss']:.4f}"
            )

        # Save best model
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            save_models(actor, critic, out_dir, "best")
            print(f"  [BEST] New best mean reward: {best_mean_reward:.2f}")

        # Periodic save
        if update % 100 == 0:
            save_models(actor, critic, out_dir, f"upd{update}")

    # Final save
    save_models(actor, critic, out_dir, "final")
    writer.close()

    print(f"\n[train_orbital_v2] Training complete!")
    print(f"  Best mean reward: {best_mean_reward:.2f}")
    print(f"  Models saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train orbital bee foraging with per-bee retasking (v2)"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU execution"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from (e.g., upd400)"
    )
    parser.add_argument(
        "--start-update", type=int, default=1,
        help="Starting update number when resuming"
    )
    
    args = parser.parse_args()
    train_orbital_v2(args)
