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

import numpy as np
import torch
import torch.nn.functional as F
from bee_policy import Actor, CentralizedCritic
from bees_env import BeeForagingEnv
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train_utils import load_config, save_models

# HRL imports
from hrl_policy import (
    ManagerPolicy,
    GoalConditionedWorker,
    HRLCritic,
    build_manager_obs,
)


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


def _batch_obs_to_tensor(obs: dict, num_bees: int, device: str) -> dict:
    """Convert all bee observations into a single batched tensor dict.
    Optimized: single CPU→GPU transfer via one contiguous numpy buffer.
    """
    keys = ["position", "status", "battery", "flowers", "step_count", "consensus", "retask_board", "action_availability"]
    defaults = {
        "battery": np.zeros(2),
        "retask_board": np.zeros(15),
        "action_availability": np.ones(3),
    }
    # Determine dims from first bee
    sample = obs.get("bee_0", {})
    dims = {}
    total_dim = 0
    for key in keys:
        val = sample.get(key, defaults.get(key, np.zeros(1)))
        if isinstance(val, (int, float)):
            dims[key] = 1
        else:
            dims[key] = np.asarray(val).ravel().shape[0]
        total_dim += dims[key]

    # Single contiguous buffer
    buf = np.zeros((num_bees, total_dim), dtype=np.float32)
    col = 0
    slices = {}
    for key in keys:
        d = dims[key]
        slices[key] = (col, col + d)
        default_val = defaults.get(key)
        for i in range(num_bees):
            ob = obs.get(f"bee_{i}", {})
            val = ob.get(key, default_val)
            if val is None:
                pass
            elif isinstance(val, (int, float)):
                buf[i, col] = float(val)
            else:
                arr = np.asarray(val, dtype=np.float32).ravel()
                n = min(len(arr), d)
                buf[i, col:col+n] = arr[:n]
        col += d

    # Single CPU→GPU transfer
    buf_t = torch.from_numpy(buf).to(device)
    batch = {}
    for key in keys:
        s, e = slices[key]
        batch[key] = buf_t[:, s:e]
    return batch


def obs_to_tensor(obs_dict: dict, device: str = "cpu") -> dict:
    """Convert observation dict to tensor dict for actor."""
    return {
        "position": _ensure_tensor(obs_dict["position"], device).unsqueeze(0),
        "status": _ensure_tensor(obs_dict["status"], device).unsqueeze(0),
        "battery": _ensure_tensor(obs_dict.get("battery", np.zeros(2)), device).unsqueeze(0),
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
            np.array(ob.get("battery", [1.0, 0.0]), dtype=np.float32),
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
    BATCHED: processes all bees in a single forward pass per step.
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
        # Batch all bees into single tensors — ONE forward pass for all bees
        batched_obs = _batch_obs_to_tensor(obs, num_bees, device)

        # Get global state for critic (once per step)
        global_state = obs_to_global_state(obs, num_bees, device)
        
        with torch.no_grad():
            step_value = critic(global_state).item()

            # Single batched forward pass for ALL bees
            action_logits = actor(batched_obs)  # (num_bees, action_dim)

            # Apply action masking (batched)
            if "action_availability" in batched_obs:
                avail = batched_obs["action_availability"]  # (num_bees, 3)
                # Reorder: [can_harvest, can_groom, can_do_nothing] -> [donothing, harvest, groom]
                mask = torch.stack([avail[:, 2], avail[:, 0], avail[:, 1]], dim=-1)
                action_logits = action_logits + (mask - 1.0) * 1e8

            probs = F.softmax(action_logits, dim=-1)  # (num_bees, 3)

            if stochastic:
                dist = torch.distributions.Categorical(probs)
                actions_t = dist.sample()       # (num_bees,)
                logprobs_t = dist.log_prob(actions_t)  # (num_bees,)
            else:
                actions_t = torch.argmax(probs, dim=-1)
                logprobs_t = torch.log(probs.gather(-1, actions_t.unsqueeze(-1)) + 1e-8).squeeze(-1)

        # Build action dict for env
        step_actions = {f"bee_{i}": int(actions_t[i].item()) for i in range(num_bees)}

        # Step environment (chain relay handles retasking internally)
        next_obs, rewards, terminated, truncated, infos, _ = env.step(step_actions)

        all_obs.append(obs)
        all_actions.append(step_actions)
        all_rewards.append([rewards[f"bee_{i}"] for i in range(num_bees)])
        all_values.append([step_value] * num_bees)
        all_logprobs.append(logprobs_t.cpu().numpy().tolist())
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
        "harvest_rate": sum(1 for f in env.flowers if f.harvested) / max(1, len(env.flowers)),
    }


def _prebuild_batched_obs(obs_list, num_bees, device):
    """Pre-convert all timestep observations into stacked numpy arrays for fast indexing.
    Returns dict of arrays, each shaped (T, num_bees, feat_dim).
    """
    T = len(obs_list)
    keys = ["position", "status", "battery", "flowers", "step_count", "consensus", "retask_board", "action_availability"]
    defaults = {"battery": np.zeros(2), "retask_board": np.zeros(15), "action_availability": np.ones(3)}
    
    # First pass: determine dimensions
    sample_ob = obs_list[0].get("bee_0", {})
    dims = {}
    for key in keys:
        val = sample_ob.get(key, defaults.get(key, np.zeros(1)))
        if isinstance(val, (int, float)):
            dims[key] = 1
        else:
            dims[key] = np.asarray(val).ravel().shape[0]
    
    # Pre-allocate arrays
    arrays = {key: np.zeros((T, num_bees, dims[key]), dtype=np.float32) for key in keys}
    actions_flat = np.zeros((T, num_bees), dtype=np.int64)
    
    for t in range(T):
        for i in range(num_bees):
            ob = obs_list[t].get(f"bee_{i}", {})
            for key in keys:
                val = ob.get(key, defaults.get(key, np.zeros(dims[key])))
                if isinstance(val, (int, float)):
                    arrays[key][t, i, 0] = float(val)
                else:
                    arr = np.asarray(val, dtype=np.float32).ravel()
                    arrays[key][t, i, :len(arr)] = arr[:dims[key]]
    
    return arrays, dims


def _prebuild_global_states(obs_list, num_bees, device):
    """Pre-build all global states as a single tensor (T, global_dim)."""
    states = []
    for t in range(len(obs_list)):
        gs = obs_to_global_state(obs_list[t], num_bees, device)
        states.append(gs.squeeze(0))
    return torch.stack(states)  # (T, global_dim)


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
    """Perform PPO update on actor and critic. BATCHED for performance."""
    obs_list = batch["obs"]
    actions_list = batch["actions"]
    old_logprobs = torch.tensor(batch["logprobs"], dtype=torch.float32, device=device)
    advantages = torch.tensor(batch["advantages"], dtype=torch.float32, device=device)
    returns = torch.tensor(batch["returns"], dtype=torch.float32, device=device)

    T, num_bees = advantages.shape

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Pre-build obs arrays and global states for fast indexing
    obs_arrays, dims = _prebuild_batched_obs(obs_list, num_bees, device)
    global_states = _prebuild_global_states(obs_list, num_bees, device)  # (T, global_dim)

    # Pre-build actions tensor (T, num_bees)
    actions_tensor = torch.zeros((T, num_bees), dtype=torch.long, device=device)
    for t in range(T):
        for i in range(num_bees):
            actions_tensor[t, i] = actions_list[t].get(f"bee_{i}", 0)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    for epoch in range(num_epochs):
        indices = np.random.permutation(T)
        
        for start in range(0, T, minibatch_size):
            end = min(start + minibatch_size, T)
            mb_indices = indices[start:end]
            mb_size = len(mb_indices)

            # Build batched obs for all bees across all timesteps in minibatch
            # Shape: (mb_size * num_bees, feat_dim) for each key
            mb_obs = {}
            for key in obs_arrays:
                # (mb_size, num_bees, dim) -> (mb_size * num_bees, dim)
                arr = obs_arrays[key][mb_indices]  # (mb_size, num_bees, dim)
                mb_obs[key] = torch.tensor(
                    arr.reshape(mb_size * num_bees, -1), dtype=torch.float32, device=device
                )

            # Batched actor forward — ONE call for all bees across all timesteps
            action_logits = actor(mb_obs)  # (mb_size * num_bees, action_dim)

            # Apply action masking (batched)
            if "action_availability" in mb_obs:
                avail = mb_obs["action_availability"]  # (mb_size*num_bees, 3)
                mask = torch.stack([avail[:, 2], avail[:, 0], avail[:, 1]], dim=-1)
                action_logits = action_logits + (mask - 1.0) * 1e8

            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            # Get actions for this minibatch: (mb_size, num_bees) -> (mb_size * num_bees,)
            mb_actions = actions_tensor[mb_indices].reshape(-1)
            new_logprobs = dist.log_prob(mb_actions)
            entropy = dist.entropy()

            # Critic: one forward pass per timestep in minibatch
            mb_global = global_states[mb_indices]  # (mb_size, global_dim)
            values = critic(mb_global).squeeze(-1)  # (mb_size,)
            # Expand to (mb_size, num_bees) since centralized
            values_expanded = values.unsqueeze(1).expand(mb_size, num_bees).reshape(-1)

            # PPO ratio
            mb_old_lp = old_logprobs[mb_indices].reshape(-1)  # (mb_size*num_bees,)
            ratio = torch.exp(new_logprobs - mb_old_lp)

            # Clipped surrogate
            mb_adv = advantages[mb_indices].reshape(-1)  # (mb_size*num_bees,)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            mb_ret = returns[mb_indices].reshape(-1)  # (mb_size*num_bees,)
            value_loss = F.mse_loss(values_expanded, mb_ret)

            # Entropy
            entropy_mean = entropy.mean()

            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean

            # Optimize
            opt_actor.zero_grad()
            opt_critic.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            opt_actor.step()
            opt_critic.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_mean.item()
            num_updates += 1

    return {
        "policy_loss": total_policy_loss / max(1, num_updates),
        "value_loss": total_value_loss / max(1, num_updates),
        "entropy": total_entropy / max(1, num_updates),
    }


# -----------------------------
# HRL Training Functions
# -----------------------------
def collect_hrl_rollout(
    env: BeeForagingEnv,
    manager: nn.Module,
    worker: nn.Module,
    critic: nn.Module,
    rollout_len: int,
    manager_interval: int,
    gamma: float,
    lam: float,
    device: str,
    stochastic: bool = True,
) -> dict:
    """
    Collect rollout with hierarchical policies.
    Manager sets goals every manager_interval steps.
    Worker executes actions conditioned on goals.
    """
    obs = env.reset()
    num_bees = len(env.bees)
    
    # Storage
    all_obs = []
    all_actions = []
    all_goals = []
    all_rewards = []
    all_worker_values = []
    all_manager_values = []
    all_logprobs = []
    all_goal_logprobs = []
    all_dones = []
    all_is_manager_step = []
    
    current_goals = torch.zeros(num_bees, dtype=torch.long, device=device)
    total_episode_reward = 0.0
    harvest_count = 0
    # Track harvest rate across ALL sub-episodes in the rollout
    total_harvested_across_episodes = 0
    total_flowers_across_episodes = 0
    num_episodes_in_rollout = 0
    
    for step in range(rollout_len):
        is_manager_step = (step % manager_interval == 0)
        
        # Manager decision every manager_interval steps
        if is_manager_step:
            manager_obs = build_manager_obs(env, device)
            with torch.no_grad():
                new_goals, goal_logprobs = manager.get_goals(manager_obs, stochastic=stochastic)
                current_goals = new_goals.squeeze(0)  # (num_bees,)
                goal_lp = goal_logprobs.squeeze(0)  # (num_bees,)
            
            # Set goals in environment
            env.set_goals(current_goals.cpu().numpy())
            
            # Manager value estimate
            global_state = obs_to_global_state(obs, num_bees, device)
            with torch.no_grad():
                manager_value = critic.forward_manager(global_state).item()
        else:
            goal_lp = torch.zeros(num_bees, device=device)
            manager_value = 0.0  # Only used at manager steps
        
        step_obs = {}
        step_actions = {}
        step_logprobs = []
        step_worker_values = []
        
        # Per-bee worker values and actions
        for i in range(num_bees):
            agent = f"bee_{i}"
            ob = obs[agent]
            ob_tensor = obs_to_tensor(ob, device)
            goal_i = current_goals[i:i+1]  # (1,)
            
            with torch.no_grad():
                # Worker value: use global state + goals
                global_state = obs_to_global_state(obs, num_bees, device)
                worker_val = critic.forward_worker(global_state, current_goals).item()
                
                # Worker action
                action_logits = worker(ob_tensor, goal_i)
                
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
            step_worker_values.append(worker_val)
        
        # Step environment
        next_obs, rewards, terminated, truncated, infos, _ = env.step(step_actions)
        
        # Add goal-based reward shaping
        for i in range(num_bees):
            agent = f"bee_{i}"
            goal_reward = env.get_goal_reward(i, step_actions[agent])
            rewards[agent] += goal_reward
        
        # Track harvests
        harvest_count = sum(1 for f in env.flowers if f.harvested)
        
        step_rewards = [rewards[f"bee_{i}"] for i in range(num_bees)]
        total_episode_reward += sum(step_rewards)
        
        done = any(terminated.values()) or any(truncated.values())
        
        # Store
        all_obs.append(step_obs)
        all_actions.append(step_actions)
        all_goals.append(current_goals.cpu().numpy().copy())
        all_rewards.append(step_rewards)
        all_worker_values.append(step_worker_values)
        all_manager_values.append(manager_value)
        all_logprobs.append(step_logprobs)
        all_goal_logprobs.append(goal_lp.cpu().numpy().copy())
        all_dones.append(done)
        all_is_manager_step.append(is_manager_step)
        
        obs = next_obs
        
        if done:
            # Record harvest stats for the completed episode
            total_harvested_across_episodes += sum(1 for f in env.flowers if f.harvested)
            total_flowers_across_episodes += len(env.flowers)
            num_episodes_in_rollout += 1
            # Auto-reset for remaining rollout steps
            obs = env.reset()
            current_goals = torch.zeros(num_bees, dtype=torch.long, device=device)
    
    # Compute GAE advantages for WORKER
    T = len(all_rewards)
    worker_advantages = np.zeros((T, num_bees), dtype=np.float32)
    worker_returns = np.zeros((T, num_bees), dtype=np.float32)
    
    # Bootstrap last value
    last_worker_values = np.zeros(num_bees, dtype=np.float32)
    if not all_dones[-1]:
        global_state = obs_to_global_state(obs, num_bees, device)
        with torch.no_grad():
            last_val = critic.forward_worker(global_state, current_goals).item()
        last_worker_values[:] = last_val
    
    gae = np.zeros(num_bees, dtype=np.float32)
    for t in reversed(range(T)):
        if t == T - 1:
            next_values = last_worker_values
        else:
            next_values = np.array(all_worker_values[t + 1], dtype=np.float32)
        
        next_nonterminal = 0.0 if all_dones[t] else 1.0
        rewards_t = np.array(all_rewards[t], dtype=np.float32)
        values_t = np.array(all_worker_values[t], dtype=np.float32)
        
        delta = rewards_t + gamma * next_values * next_nonterminal - values_t
        gae = delta + gamma * lam * next_nonterminal * gae
        worker_advantages[t] = gae
        worker_returns[t] = gae + values_t
    
    # Compute GAE advantages for MANAGER (only at manager decision steps)
    # Manager gets sum of rewards between its decisions as its reward signal
    manager_steps = [t for t in range(T) if all_is_manager_step[t]]
    manager_advantages = np.zeros(T, dtype=np.float32)
    manager_returns = np.zeros(T, dtype=np.float32)
    
    if len(manager_steps) > 0:
        # Aggregate rewards between manager steps
        manager_rewards = []
        manager_vals = []
        for idx, ms in enumerate(manager_steps):
            next_ms = manager_steps[idx + 1] if idx + 1 < len(manager_steps) else T
            # Sum all bee rewards between this manager step and next
            interval_reward = sum(
                sum(all_rewards[t]) for t in range(ms, next_ms)
            ) / num_bees  # Average per bee
            manager_rewards.append(interval_reward)
            manager_vals.append(all_manager_values[ms])
        
        # Bootstrap
        last_mgr_val = 0.0
        if not all_dones[-1]:
            global_state = obs_to_global_state(obs, num_bees, device)
            with torch.no_grad():
                last_mgr_val = critic.forward_manager(global_state).item()
        
        # GAE for manager
        mgr_gae = 0.0
        for idx in reversed(range(len(manager_steps))):
            if idx == len(manager_steps) - 1:
                next_val = last_mgr_val
                next_nonterminal = 0.0 if all_dones[-1] else 1.0
            else:
                next_val = manager_vals[idx + 1]
                # Check if done between this and next manager step
                next_ms = manager_steps[idx + 1]
                next_nonterminal = 0.0 if any(all_dones[t] for t in range(manager_steps[idx], next_ms)) else 1.0
            
            delta = manager_rewards[idx] + gamma * next_val * next_nonterminal - manager_vals[idx]
            mgr_gae = delta + gamma * lam * next_nonterminal * mgr_gae
            
            ms = manager_steps[idx]
            manager_advantages[ms] = mgr_gae
            manager_returns[ms] = mgr_gae + manager_vals[idx]
    
    # Include the final (possibly partial) episode in stats
    total_harvested_across_episodes += sum(1 for f in env.flowers if f.harvested)
    total_flowers_across_episodes += len(env.flowers)
    num_episodes_in_rollout += 1
    harvest_rate = total_harvested_across_episodes / max(1, total_flowers_across_episodes)
    
    return {
        "obs": all_obs,
        "actions": all_actions,
        "goals": np.array(all_goals, dtype=np.int64),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "worker_values": np.array(all_worker_values, dtype=np.float32),
        "logprobs": np.array(all_logprobs, dtype=np.float32),
        "goal_logprobs": np.array(all_goal_logprobs, dtype=np.float32),
        "worker_advantages": worker_advantages,
        "worker_returns": worker_returns,
        "manager_advantages": manager_advantages,
        "manager_returns": manager_returns,
        "is_manager_step": np.array(all_is_manager_step, dtype=bool),
        "episode_reward": total_episode_reward,
        "episode_len": T,
        "harvest_rate": harvest_rate,
        "num_episodes": num_episodes_in_rollout,
    }


def train_hrl(args):
    """Training loop for Hierarchical RL (Manager + Worker)."""
    
    # Load config
    train_cfg, env_cfg, out_dir = load_config(args.config)
    
    if args.output:
        out_dir = args.output
    out_dir = out_dir + "_hrl"  # Separate output for HRL
    
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[train_hrl] device={device}, output={out_dir}")
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
        verbose=env_cfg.get("verbose", False),
    )
    
    print(f"[env] num_bees={env.num_bees}, num_flowers={env.num_flowers}")
    print(f"[hrl] manager_interval={args.manager_interval}")
    
    hidden_dim = train_cfg.get("hidden_dim", 256)
    
    # Create HRL networks
    manager = ManagerPolicy(
        num_bees=env.num_bees,
        num_flowers=env.num_flowers,
        hidden_dim=hidden_dim,
        manager_interval=args.manager_interval,
    ).to(device)
    
    worker = GoalConditionedWorker(
        num_bees=env.num_bees,
        num_flowers=env.num_flowers,
        retask_board_size=env.retask_board_size,
        hidden_dim=hidden_dim,
        num_goals=ManagerPolicy.NUM_GOALS,
        grid_size=env.grid_size,
    ).to(device)
    
    # Critic for HRL
    obs_dim_per_bee = 3 + 2 + 2 + (env.num_flowers * 12) + 1 + env.num_bees + (5 * env.retask_board_size)
    global_state_size = obs_dim_per_bee * env.num_bees
    
    critic = HRLCritic(
        global_state_size=global_state_size,
        num_bees=env.num_bees,
        num_goals=ManagerPolicy.NUM_GOALS,
        hidden_dim=hidden_dim,
        grid_size=env.grid_size,
    ).to(device)
    
    # Optimizers
    opt_manager = optim.Adam(manager.parameters(), lr=train_cfg.get("lr_actor", 3e-4))
    opt_worker = optim.Adam(worker.parameters(), lr=train_cfg.get("lr_actor", 3e-4))
    opt_critic = optim.Adam(critic.parameters(), lr=train_cfg.get("lr_critic", 3e-4))
    
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
    
    print(f"\n[train_hrl] Starting HRL training for {num_updates} updates...")
    print(f"  rollout_len={rollout_len}, manager_interval={args.manager_interval}")
    print()
    
    for update in tqdm(range(1, num_updates + 1), desc="HRL Training"):
        # Collect rollout
        batch = collect_hrl_rollout(
            env, manager, worker, critic, rollout_len, args.manager_interval,
            gamma, lam, device, stochastic=True
        )
        
        obs_list = batch["obs"]
        actions_list = batch["actions"]
        goals = torch.tensor(batch["goals"], dtype=torch.long, device=device)
        old_logprobs = torch.tensor(batch["logprobs"], dtype=torch.float32, device=device)
        old_goal_logprobs = torch.tensor(batch["goal_logprobs"], dtype=torch.float32, device=device)
        worker_advantages = torch.tensor(batch["worker_advantages"], dtype=torch.float32, device=device)
        worker_returns = torch.tensor(batch["worker_returns"], dtype=torch.float32, device=device)
        manager_advantages_np = batch["manager_advantages"]
        manager_returns_np = batch["manager_returns"]
        is_manager_step = batch["is_manager_step"]
        
        T = len(obs_list)
        num_bees = env.num_bees
        
        # Normalize worker advantages
        worker_advantages = (worker_advantages - worker_advantages.mean()) / (worker_advantages.std() + 1e-8)
        
        # ===== WORKER PPO UPDATE (with minibatches and multiple epochs) =====
        num_ppo_epochs = 4
        minibatch_size = max(1, T // 4)
        
        for epoch in range(num_ppo_epochs):
            indices = np.random.permutation(T)
            
            for mb_start in range(0, T, minibatch_size):
                mb_end = min(mb_start + minibatch_size, T)
                mb_idx = indices[mb_start:mb_end]
                
                worker_loss_sum = 0.0
                entropy_sum = 0.0
                value_loss_sum = 0.0
                count = 0
                
                for t in mb_idx:
                    global_state = obs_to_global_state(obs_list[t], num_bees, device)
                    goals_t = goals[t]
                    
                    # Critic value for this timestep
                    value = critic.forward_worker(global_state, goals_t)
                    target = worker_returns[t].mean()
                    value_loss_sum += F.mse_loss(value.squeeze(), target)
                    
                    for i in range(num_bees):
                        agent = f"bee_{i}"
                        ob = obs_list[t][agent]
                        ob_tensor = obs_to_tensor(ob, device)
                        action = actions_list[t][agent]
                        goal_i = goals_t[i:i+1]
                        
                        action_logits = worker(ob_tensor, goal_i)
                        probs = F.softmax(action_logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        
                        new_logprob = dist.log_prob(torch.tensor([action], device=device))
                        entropy = dist.entropy()
                        
                        ratio = torch.exp(new_logprob - old_logprobs[t, i])
                        adv = worker_advantages[t, i]
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
                        
                        worker_loss_sum += -torch.min(surr1, surr2).mean()
                        entropy_sum += entropy.mean()
                        count += 1
                
                if count > 0:
                    total_loss = (worker_loss_sum / count) + value_coef * (value_loss_sum / len(mb_idx)) - entropy_coef * (entropy_sum / count)
                    
                    opt_worker.zero_grad()
                    opt_critic.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(worker.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    opt_worker.step()
                    opt_critic.step()
        
        # ===== MANAGER PPO UPDATE =====
        manager_steps = [t for t in range(T) if is_manager_step[t]]
        
        if len(manager_steps) > 1:
            mgr_adv = torch.tensor([manager_advantages_np[t] for t in manager_steps], dtype=torch.float32, device=device)
            mgr_ret = torch.tensor([manager_returns_np[t] for t in manager_steps], dtype=torch.float32, device=device)
            
            # Normalize manager advantages
            if mgr_adv.numel() > 1:
                mgr_adv = (mgr_adv - mgr_adv.mean()) / (mgr_adv.std() + 1e-8)
            
            for epoch in range(num_ppo_epochs):
                manager_loss_sum = 0.0
                mgr_value_loss_sum = 0.0
                mgr_entropy_sum = 0.0
                mgr_count = 0
                
                for idx, t in enumerate(manager_steps):
                    # Rebuild manager obs
                    # We use the stored obs to reconstruct global state
                    global_state = obs_to_global_state(obs_list[t], num_bees, device)
                    
                    # Manager value
                    mgr_value = critic.forward_manager(global_state)
                    mgr_value_loss_sum += F.mse_loss(mgr_value.squeeze(), mgr_ret[idx])
                    
                    # Manager policy: recompute goal logprobs
                    # Build manager obs from stored observations
                    # We need to recompute from env state - but we don't have it
                    # So use the old goal_logprobs with PPO ratio
                    goals_t = goals[t]  # (num_bees,)
                    old_glp = old_goal_logprobs[t]  # (num_bees,)
                    
                    # We need the manager forward pass - reconstruct from stored obs
                    # Use obs_to_global_state-style reconstruction for manager
                    bee_states = []
                    for i in range(num_bees):
                        ob = obs_list[t][f"bee_{i}"]
                        pos = ob["position"]
                        status = ob["status"]
                        battery_ratio = status[1] if len(status) > 1 else 0.5
                        load_ratio = status[0] if len(status) > 0 else 0.0
                        is_active = 1.0
                        bee_states.append([
                            pos[0] / env.grid_size, pos[1] / env.grid_size, pos[2] / env.grid_size,
                            status[0] / 2.0, battery_ratio, load_ratio, is_active
                        ])
                    
                    # Flower features from first bee's obs (same for all)
                    fl = obs_list[t]["bee_0"]["flowers"]
                    fl_reshaped = np.array(fl).reshape(env.num_flowers, 12)
                    
                    step_norm = obs_list[t]["bee_0"]["step_count"][0] if hasattr(obs_list[t]["bee_0"]["step_count"], '__len__') else float(obs_list[t]["bee_0"]["step_count"])
                    harvest_ratio = sum(1 for f in fl_reshaped if f[3] > 0.5) / max(1, env.num_flowers)
                    active_ratio = sum(1 for bs in bee_states if bs[-1] > 0.5) / max(1, num_bees)
                    
                    mgr_obs_rebuilt = {
                        "bee_states": torch.tensor([bee_states], dtype=torch.float32, device=device),
                        "flowers": torch.tensor([fl_reshaped.tolist()], dtype=torch.float32, device=device),
                        "context": torch.tensor([[step_norm, harvest_ratio, active_ratio]], dtype=torch.float32, device=device),
                    }
                    
                    goal_logits = manager(mgr_obs_rebuilt)  # (1, num_bees, NUM_GOALS)
                    goal_dist = torch.distributions.Categorical(logits=goal_logits.squeeze(0))
                    new_goal_lp = goal_dist.log_prob(goals_t)  # (num_bees,)
                    mgr_entropy = goal_dist.entropy().mean()
                    
                    # PPO ratio per bee, then average
                    ratio = torch.exp(new_goal_lp - old_glp)
                    adv = mgr_adv[idx]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
                    
                    manager_loss_sum += -torch.min(surr1, surr2).mean()
                    mgr_entropy_sum += mgr_entropy
                    mgr_count += 1
                
                if mgr_count > 0:
                    mgr_total = (manager_loss_sum / mgr_count) + value_coef * (mgr_value_loss_sum / mgr_count) - entropy_coef * (mgr_entropy_sum / mgr_count)
                    
                    opt_manager.zero_grad()
                    opt_critic.zero_grad()
                    mgr_total.backward()
                    torch.nn.utils.clip_grad_norm_(manager.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    opt_manager.step()
                    opt_critic.step()
        
        # Track reward
        reward_tracker.add_episode(batch["episode_reward"])
        mean_reward = reward_tracker.mean_reward()
        harvest_rate = batch.get("harvest_rate", 0.0)
        
        # Log
        writer.add_scalar("reward/episode", batch["episode_reward"], update)
        writer.add_scalar("reward/mean_100", mean_reward, update)
        writer.add_scalar("episode/length", batch["episode_len"], update)
        writer.add_scalar("harvest/rate", harvest_rate, update)
        
        if update % 50 == 0:
            n_eps = batch.get("num_episodes", "?")
            print(
                f"[upd {update:4d}] reward={batch['episode_reward']:.1f}, "
                f"mean100={mean_reward:.1f}, harvest={harvest_rate:.0%}, "
                f"eps={n_eps}, len={batch['episode_len']}"
            )
        
        # Save best
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            torch.save(manager.state_dict(), os.path.join(out_dir, "best_manager.pt"))
            torch.save(worker.state_dict(), os.path.join(out_dir, "best_worker.pt"))
            torch.save(critic.state_dict(), os.path.join(out_dir, "best_critic.pt"))
            print(f"  [BEST] New best mean reward: {best_mean_reward:.2f}")
        
        # Periodic save
        if update % 100 == 0:
            ckpt_dir = os.path.join(out_dir, f"upd{update}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(manager.state_dict(), os.path.join(ckpt_dir, "manager.pt"))
            torch.save(worker.state_dict(), os.path.join(ckpt_dir, "worker.pt"))
            torch.save(critic.state_dict(), os.path.join(ckpt_dir, "critic.pt"))
    
    # Final save
    torch.save(manager.state_dict(), os.path.join(out_dir, "final_manager.pt"))
    torch.save(worker.state_dict(), os.path.join(out_dir, "final_worker.pt"))
    torch.save(critic.state_dict(), os.path.join(out_dir, "final_critic.pt"))
    writer.close()
    
    print(f"\n[train_hrl] Training complete!")
    print(f"  Best mean reward: {best_mean_reward:.2f}")
    print(f"  Models saved to: {out_dir}")


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
        grid_size=env.grid_size,
    ).to(device)

    # Calculate global state size for critic (position + status per bee + flower features)
    # position(3) + status(2) + battery(2) + flowers(num_flowers*12) + step(1) + consensus(num_bees) + retask_board(5*size)
    obs_dim_per_bee = 3 + 2 + 2 + (env.num_flowers * 12) + 1 + env.num_bees + (5 * env.retask_board_size)
    global_state_size = obs_dim_per_bee * env.num_bees  # Centralized critic sees all bees

    critic = CentralizedCritic(
        global_state_size=global_state_size,
        num_bees=env.num_bees,
        hidden_dim=hidden_dim,
        grid_size=env.grid_size,
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
    
    # Early stopping settings
    early_stopping = train_cfg.get("early_stopping", True)
    patience = train_cfg.get("patience", 200)
    min_harvest_rate = train_cfg.get("min_harvest_rate", 0.95)
    convergence_window = train_cfg.get("convergence_window", 50)
    
    # Track harvest rates for early stopping
    harvest_rates = []
    best_harvest_rate = 0.0
    no_improvement_count = 0
    converged = False

    remaining_updates = num_updates - start_update + 1
    print(f"\n[train_orbital_v2] Starting training for up to {remaining_updates} updates...")
    print(f"  rollout_len={rollout_len}, gamma={gamma}, lam={lam}")
    print(f"  clip_ratio={clip_ratio}, value_coef={value_coef}, entropy_coef={entropy_coef}")
    if early_stopping:
        print(f"  Early stopping: patience={patience}, target_harvest_rate={min_harvest_rate:.0%}")
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
        
        # Track harvest rate for early stopping
        harvest_rate = batch.get("harvest_rate", 0.0)
        harvest_rates.append(harvest_rate)
        if len(harvest_rates) > convergence_window:
            harvest_rates.pop(0)
        mean_harvest_rate = np.mean(harvest_rates) if harvest_rates else 0.0

        # Log to TensorBoard
        writer.add_scalar("reward/episode", batch["episode_reward"], update)
        writer.add_scalar("reward/mean_100", mean_reward, update)
        writer.add_scalar("loss/policy", losses["policy_loss"], update)
        writer.add_scalar("loss/value", losses["value_loss"], update)
        writer.add_scalar("loss/entropy", losses["entropy"], update)
        writer.add_scalar("episode/length", batch["episode_len"], update)
        writer.add_scalar("performance/harvest_rate", harvest_rate, update)
        writer.add_scalar("performance/mean_harvest_rate", mean_harvest_rate, update)

        # Print progress
        if update % 50 == 0:
            print(
                f"[upd {update:4d}] reward={batch['episode_reward']:.1f}, "
                f"mean100={mean_reward:.1f}, harvest={harvest_rate:.0%}, "
                f"mean_harvest={mean_harvest_rate:.0%}"
            )

        # Save best model (based on harvest rate, not just reward)
        if mean_harvest_rate > best_harvest_rate and len(harvest_rates) >= convergence_window // 2:
            best_harvest_rate = mean_harvest_rate
            best_mean_reward = mean_reward
            save_models(actor, critic, "best", out_dir)
            print(f"  [BEST] harvest_rate={best_harvest_rate:.1%}, reward={best_mean_reward:.2f}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Periodic save
        if update % 100 == 0:
            save_models(actor, critic, f"upd{update}", out_dir)
        
        # Early stopping check
        if early_stopping and len(harvest_rates) >= convergence_window:
            # Check if we've hit target and are stable
            if mean_harvest_rate >= min_harvest_rate:
                print(f"\n  [CONVERGED] Target harvest rate {min_harvest_rate:.0%} achieved!")
                print(f"  Mean harvest rate: {mean_harvest_rate:.1%} over {convergence_window} episodes")
                converged = True
                break
            
            # Check if no improvement for too long
            if no_improvement_count >= patience:
                print(f"\n  [EARLY STOP] No improvement for {patience} updates")
                print(f"  Best harvest rate: {best_harvest_rate:.1%}")
                break

    # Final save
    save_models(actor, critic, "final", out_dir)
    writer.close()

    print(f"\n[train_orbital_v2] Training complete!")
    print(f"  Status: {'CONVERGED' if converged else 'STOPPED'}")
    print(f"  Best harvest rate: {best_harvest_rate:.1%}")
    print(f"  Best mean reward: {best_mean_reward:.2f}")
    print(f"  Total updates: {update}")
    print(f"  Models saved to: {out_dir}")
    print(f"\n  Use 'outputs/best_actor.pt' for the best model (not final!)")


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
    parser.add_argument(
        "--hrl", action="store_true",
        help="Use Hierarchical RL (Manager + Worker policies)"
    )
    parser.add_argument(
        "--manager-interval", type=int, default=10,
        help="Steps between manager decisions (HRL only)"
    )
    
    args = parser.parse_args()
    
    if args.hrl:
        train_hrl(args)
    else:
        train_orbital_v2(args)
