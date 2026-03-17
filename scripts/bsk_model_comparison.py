"""
BSK Model Comparison: Gradient Policy (original env) vs Bee_Fi Actor (BSK env)

Runs 100+ episodes of each model in BSK mode and compares:
- Harvest rate (flowers completed / total)
- Pollen collected
- Satellite failures (energy deaths)
- Task completion by priority

The gradient policy uses its own env (bees_env_std from satellite_constellation_scheduling)
with the SatelliteToBeeMapper to map BSK positions. The Bee_Fi Actor uses BeeForagingEnv
directly with BSK integration.
"""

import os
import sys
import json
import math
import time
import random
import argparse
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Any


import numpy as np
import torch

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
SCS_ROOT = PROJECT_ROOT / "satellite_constellation_scheduling"

# ── Import Bee_Fi Actor model FIRST (before adding SCS paths) ─
sys.path.insert(0, str(PROJECT_ROOT))
from bee_policy import Actor
from bees_env import BeeForagingEnv as BeeFiEnv
from hrl_policy import ManagerPolicy, GoalConditionedWorker, HRLCritic, build_manager_obs

# ── Now add SCS paths for gradient policy imports ────────────
sys.path.insert(0, str(SCS_ROOT))
sys.path.insert(0, str(SCS_ROOT / "bees_drl_app"))
sys.path.insert(0, str(SCS_ROOT / "rl_models"))
sys.path.insert(0, str(SCS_ROOT / "model_repository"))

from rl_models.bee_gradient_policy import BeeGradientAgent, ReTaskModel, GossipMessage
from rl_models.bee_policy import BeePolicyWorker, TaskManager, ActionSpace

# Import bee_flower_state from bees_drl_app explicitly (not rl_models)
_bfs_spec = importlib.util.spec_from_file_location(
    "bees_drl_app_bee_flower_state",
    str(SCS_ROOT / "bees_drl_app" / "bee_flower_state.py")
)
_bfs_module = importlib.util.module_from_spec(_bfs_spec)
_bfs_spec.loader.exec_module(_bfs_module)
GlobalStateExtractor = _bfs_module.GlobalStateExtractor
BeeStateWrapper = _bfs_module.BeeStateWrapper

# Gradient env (the original one from satellite_constellation_scheduling)
# We import it under a different name to avoid collision
_gradient_env_spec = importlib.util.spec_from_file_location(
    "gradient_bees_env",
    str(SCS_ROOT / "bees_drl_app" / "bees_env.py")
)
_gradient_env_module = importlib.util.module_from_spec(_gradient_env_spec)
_gradient_env_spec.loader.exec_module(_gradient_env_module)
GradientBeeForagingEnv = _gradient_env_module.BeeForagingEnv


# Only these keys go to the Actor (must match training loop & bsk_evaluator)
_ACTOR_OBS_KEYS = ["position", "status", "flowers", "step_count", "consensus", "retask_board", "action_availability"]


def obs_to_tensor(obs_dict, device):
    """Convert Bee_Fi env observation dict to tensors for Actor (filtered keys)."""
    out = {}
    for k in _ACTOR_OBS_KEYS:
        v = obs_dict.get(k)
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            out[k] = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        elif isinstance(v, (list, tuple)):
            out[k] = torch.tensor(np.array(v, dtype=np.float32), device=device).unsqueeze(0)
        else:
            out[k] = torch.tensor([v], dtype=torch.float32, device=device).unsqueeze(0)
    return out


# ═══════════════════════════════════════════════════════════════
#  Single-episode worker functions (for multiprocessing)
# ═══════════════════════════════════════════════════════════════

def _run_single_actor_episode(args_tuple):
    """Run a single Actor-Critic episode. Designed for ProcessPoolExecutor."""
    (ep, model_path, num_sats, num_tasks, grid_size, max_steps,
     bsk_dt, bsk_meters_per_unit, seed) = args_tuple

    import random as _random
    np.random.seed(seed + ep)
    torch.manual_seed(seed + ep)
    _random.seed(seed + ep)

    device = torch.device("cpu")  # CPU in workers to avoid CUDA fork issues

    env = BeeFiEnv(
        num_bees=num_sats, num_flowers=num_tasks, grid_size=grid_size,
        max_steps=max_steps, retask_board_size=3, use_basilisk=True,
        bsk_dt_sec=bsk_dt, bsk_meters_per_unit=bsk_meters_per_unit,
        low_battery_chance=0.30, verbose=False,
    )
    obs = env.reset()

    actor = Actor(num_bees=num_sats, action_dim=3, num_flowers=num_tasks,
                  retask_board_size=3, grid_size=grid_size).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    actor.load_state_dict(ckpt)
    actor.eval()

    total_reward = 0.0
    step = 0
    while True:
        actions = {}
        with torch.no_grad():
            for agent_id, agent_obs in obs.items():
                if agent_obs is None:
                    continue
                obs_t = obs_to_tensor(agent_obs, device)
                logits = actor(obs_t)
                avail = obs_t["action_availability"]
                mask = torch.stack([avail[:, 2], avail[:, 0], avail[:, 1]], dim=-1)
                logits = logits + (mask - 1.0) * 1e8
                action = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
                actions[agent_id] = action
        obs, rewards, dones, truncs, infos, _global = env.step(actions)
        total_reward += sum(rewards.values())
        step += 1
        if all(dones.values()) or all(truncs.values()):
            break

    harvested = sum(1 for f in env.flowers if f.harvested)
    expired = sum(1 for f in env.flowers if f.expired)
    alive = sum(1 for i in range(num_sats) if env._battery[i] > 0)
    hard_total = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'HARD')
    soft_total = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'SOFT')
    none_total = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'NONE')
    hard_done = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'HARD' and f.harvested)
    soft_done = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'SOFT' and f.harvested)
    none_done = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'NONE' and f.harvested)
    env.close()

    result = {
        "episode": ep, "harvest_rate": harvested / num_tasks,
        "harvested": harvested, "expired": expired, "total_flowers": num_tasks,
        "alive_sats": alive, "dead_sats": num_sats - alive,
        "total_reward": total_reward, "steps": step,
        "hard_total": hard_total, "hard_done": hard_done,
        "soft_total": soft_total, "soft_done": soft_done,
        "none_total": none_total, "none_done": none_done,
    }
    print(f"  [Actor-Critic] Ep {ep+1}: harvest={harvested}/{num_tasks} "
          f"({100*harvested/num_tasks:.0f}%), alive={alive}/{num_sats}, dead={num_sats-alive}", flush=True)
    return result


def _run_single_gradient_episode(args_tuple):
    """Run a single Gradient episode in BSK env. Designed for ProcessPoolExecutor."""
    (ep, num_sats, num_tasks, grid_size, max_steps,
     bsk_dt, bsk_meters_per_unit, seed) = args_tuple

    import random as _random
    np.random.seed(seed + ep)
    torch.manual_seed(seed + ep)
    _random.seed(seed + ep)

    device = torch.device("cpu")

    # Load gradient model
    model_path = SCS_ROOT / "bees_drl_app" / "models" / "bee_gradient_policy_hierarchical_vfrl_best_ole.pth"
    gradient_policy = torch.load(model_path, map_location=device, weights_only=False)
    training_config = gradient_policy['metadata']

    retask_model = ReTaskModel()
    retask_model.load_state_dict(gradient_policy['retask_model'])
    retask_model.eval()

    worker_policy = BeePolicyWorker(
        num_groom_types=training_config['num_groom_types'],
        num_flowers=num_tasks, device=device,
    )
    worker_policy.load_state_dict(gradient_policy['worker_policy'])
    worker_policy.eval()

    coordinator_state = gradient_policy.get('coordinators')
    mapper = _FlattenNormMapper(grid_size)

    env = BeeFiEnv(
        num_bees=num_sats, num_flowers=num_tasks, grid_size=grid_size,
        max_steps=max_steps, retask_board_size=3, use_basilisk=True,
        bsk_dt_sec=bsk_dt, bsk_meters_per_unit=bsk_meters_per_unit,
        low_battery_chance=0.30, verbose=False,
    )
    obs = env.reset()

    bee_agents = [
        BeeGradientAgent(
            bee_id=i, worker_policy=worker_policy, retask_reassignment=retask_model,
            device=device, grid_size=grid_size, flower_feature_dim=6,
            max_steps=max_steps, embed_dim=64,
        ) for i in range(num_sats)
    ]
    for agent in bee_agents:
        agent.coordinator.load_state_dict(coordinator_state)
        agent.coordinator.eval()

    task_manager = TaskManager(num_flowers=num_tasks)
    total_reward = 0.0
    step = 0

    while True:
        actions = {}
        for i in range(num_sats):
            bee = env.bees[i]
            agent = bee_agents[i]
            agent.energy = env._battery[i] / max(1.0, env._battery_max[i])
            agent.pollen_load = bee.load
            is_available = not bee.terminated and env._recharge_until[i] <= env.steps
            agent.update_agent_status(is_available)

            assigned = getattr(bee, 'assigned_flowers', [])
            assigned_task_dict = {}
            for fj in assigned:
                f = env.flowers[fj]
                assigned_task_dict[f"flower_{fj}"] = {
                    'position': [f.x, f.y], 'pollen_amount': f.pollen,
                    'priority_level': f.priority, 'min_step': f.window_start,
                    'max_step': f.window_end, 'hard_window': f.window_type == "HARD",
                    'harvested': f.harvested,
                }
            agent.update_task_dict(assigned_task_dict)

            if not agent.active:
                actions[f"bee_{i}"] = 0
                continue

            try:
                state_dict, bee_pos = _build_gradient_state_dict(env, i, mapper, device)
                num_assigned = len(assigned)
                action_mask = _build_gradient_action_mask(env, i, num_assigned, device)
                action_idx, info, _ = agent.select_hierarchical_action(
                    state_dict=state_dict, current_step=env.steps,
                    action_mask=action_mask, current_bee_positions=bee_pos,
                    is_training=False,
                )
                action = _gradient_action_to_bsk(action_idx, bee, env)
            except Exception:
                action = 0

            actions[f"bee_{i}"] = action

        obs, rewards, dones, truncs, infos, _global = env.step(actions)
        total_reward += sum(rewards.values())
        step += 1
        if all(dones.values()) or all(truncs.values()):
            break

    harvested = sum(1 for f in env.flowers if f.harvested)
    expired = sum(1 for f in env.flowers if getattr(f, 'expired', False))
    alive = sum(1 for i in range(num_sats) if env._battery[i] > 0)
    hard_total = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'HARD')
    soft_total = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'SOFT')
    none_total = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'NONE')
    hard_done = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'HARD' and f.harvested)
    soft_done = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'SOFT' and f.harvested)
    none_done = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'NONE' and f.harvested)
    env.close()

    result = {
        "episode": ep, "harvest_rate": harvested / num_tasks,
        "harvested": harvested, "expired": expired, "total_flowers": num_tasks,
        "dead_sats": num_sats - alive, "alive_sats": alive,
        "total_reward": total_reward, "steps": step,
        "hard_total": hard_total, "hard_done": hard_done,
        "soft_total": soft_total, "soft_done": soft_done,
        "none_total": none_total, "none_done": none_done,
    }
    print(f"  [Gradient+BSK] Ep {ep+1}: harvest={harvested}/{num_tasks} "
          f"({100*harvested/num_tasks:.0f}%), alive={alive}/{num_sats}, dead={num_sats-alive}", flush=True)
    return result





# ═══════════════════════════════════════════════════════════════
#  Bee_Fi Actor evaluation (BSK mode)
# ═══════════════════════════════════════════════════════════════
def run_beefi_actor_episodes(
    model_path: str,
    num_episodes: int,
    num_sats: int = 25,
    num_tasks: int = 50,
    grid_size: int = 75,
    max_steps: int = 1200,
    bsk_dt: float = 5.0,
    bsk_meters_per_unit: float = 200_000.0,
    seed: int = 42,
    device: torch.device = None,
    verbose: bool = False,
) -> List[Dict]:
    """Run N episodes of the Bee_Fi Actor model in BSK mode."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for ep in range(num_episodes):
        np.random.seed(seed + ep)
        torch.manual_seed(seed + ep)
        random.seed(seed + ep)

        # Create environment
        env = BeeFiEnv(
            num_bees=num_sats,
            num_flowers=num_tasks,
            grid_size=grid_size,
            max_steps=max_steps,
            retask_board_size=3,
            use_basilisk=True,
            bsk_dt_sec=bsk_dt,
            bsk_meters_per_unit=bsk_meters_per_unit,
            low_battery_chance=0.30,
            verbose=False,
        )
        obs = env.reset()

        # Load actor
        actor = Actor(
            num_bees=num_sats,
            action_dim=3,
            num_flowers=num_tasks,
            retask_board_size=3,
            grid_size=grid_size,
        ).to(device)

        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        actor.load_state_dict(ckpt)
        actor.eval()

        total_reward = 0.0
        step = 0

        while True:
            actions = {}
            with torch.no_grad():
                for agent_id, agent_obs in obs.items():
                    if agent_obs is None:
                        continue
                    obs_t = obs_to_tensor(agent_obs, device)
                    logits = actor(obs_t)
                    # Apply hard action masking (same reorder as training loop & bsk_evaluator)
                    # action_availability = [can_harvest, can_groom, can_do_nothing]
                    # Actor logits   = [DONOTHING,    HARVEST,   GROOM]
                    avail = obs_t["action_availability"]  # (1, 3)
                    mask = torch.stack([avail[:, 2], avail[:, 0], avail[:, 1]], dim=-1)
                    logits = logits + (mask - 1.0) * 1e8
                    action = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
                    actions[agent_id] = action

            obs, rewards, dones, truncs, infos, _global = env.step(actions)
            total_reward += sum(rewards.values())
            step += 1

            if all(dones.values()) or all(truncs.values()):
                break

        # Collect results
        harvested = sum(1 for f in env.flowers if f.harvested)
        expired = sum(1 for f in env.flowers if f.expired)
        alive = sum(1 for i in range(num_sats) if env._battery[i] > 0)

        # Task-type breakdowns
        hard_total = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'HARD')
        soft_total = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'SOFT')
        none_total = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'NONE')
        hard_done = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'HARD' and f.harvested)
        soft_done = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'SOFT' and f.harvested)
        none_done = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'NONE' and f.harvested)

        results.append({
            "episode": ep,
            "harvest_rate": harvested / num_tasks,
            "harvested": harvested,
            "expired": expired,
            "total_flowers": num_tasks,
            "alive_sats": alive,
            "dead_sats": num_sats - alive,
            "total_reward": total_reward,
            "steps": step,
            "hard_total": hard_total, "hard_done": hard_done,
            "soft_total": soft_total, "soft_done": soft_done,
            "none_total": none_total, "none_done": none_done,
        })

        print(f"  [BeeFi] Ep {ep+1}/{num_episodes}: harvest={harvested}/{num_tasks} "
              f"({100*harvested/num_tasks:.0f}%), alive={alive}/{num_sats}", flush=True)

        env.close()

    return results


# ═══════════════════════════════════════════════════════════════
#  Bee_Fi HRL evaluation (Manager + Worker, BSK mode)
# ═══════════════════════════════════════════════════════════════
# HRL obs keys for worker (same as Actor + battery)
_HRL_OBS_KEYS = ["position", "status", "battery", "flowers", "step_count", "consensus", "retask_board", "action_availability"]


def hrl_obs_to_tensor(obs_dict, device):
    """Convert Bee_Fi env observation dict to tensors for GoalConditionedWorker."""
    out = {}
    for k in _HRL_OBS_KEYS:
        v = obs_dict.get(k)
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            out[k] = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        elif isinstance(v, (list, tuple)):
            out[k] = torch.tensor(np.array(v, dtype=np.float32), device=device).unsqueeze(0)
        else:
            out[k] = torch.tensor([v], dtype=torch.float32, device=device).unsqueeze(0)
    return out


def run_hrl_episodes(
    manager_path: str,
    worker_path: str,
    num_episodes: int,
    num_sats: int = 25,
    num_tasks: int = 50,
    grid_size: int = 75,
    max_steps: int = 1200,
    manager_interval: int = 10,
    bsk_dt: float = 5.0,
    bsk_meters_per_unit: float = 200_000.0,
    seed: int = 42,
    device: torch.device = None,
    verbose: bool = False,
) -> List[Dict]:
    """Run N episodes of the HRL (Manager + Worker) model in BSK mode."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for ep in range(num_episodes):
        np.random.seed(seed + ep)
        torch.manual_seed(seed + ep)
        random.seed(seed + ep)

        env = BeeFiEnv(
            num_bees=num_sats,
            num_flowers=num_tasks,
            grid_size=grid_size,
            max_steps=max_steps,
            retask_board_size=3,
            use_basilisk=True,
            bsk_dt_sec=bsk_dt,
            bsk_meters_per_unit=bsk_meters_per_unit,
            verbose=False,
        )
        obs = env.reset()

        # Load manager
        manager = ManagerPolicy(
            num_bees=num_sats,
            num_flowers=num_tasks,
            hidden_dim=256,
            manager_interval=manager_interval,
        ).to(device)
        manager.load_state_dict(torch.load(manager_path, map_location=device, weights_only=True))
        manager.eval()

        # Load worker
        worker = GoalConditionedWorker(
            num_bees=num_sats,
            action_dim=3,
            num_flowers=num_tasks,
            retask_board_size=3,
            hidden_dim=256,
            num_goals=ManagerPolicy.NUM_GOALS,
            grid_size=grid_size,
        ).to(device)
        worker.load_state_dict(torch.load(worker_path, map_location=device, weights_only=True))
        worker.eval()

        current_goals = torch.zeros(num_sats, dtype=torch.long, device=device)
        total_reward = 0.0
        step = 0

        while True:
            # Manager decision every manager_interval steps
            if step % manager_interval == 0:
                with torch.no_grad():
                    manager_obs = build_manager_obs(env, device)
                    new_goals, _ = manager.get_goals(manager_obs, stochastic=False)
                    current_goals = new_goals.squeeze(0)  # (num_sats,)
                    env.set_goals(current_goals.cpu().numpy())

            # Worker actions for each satellite
            actions = {}
            with torch.no_grad():
                for agent_id, agent_obs in obs.items():
                    if agent_obs is None:
                        continue
                    i = int(agent_id.split("_")[1])
                    ob_t = hrl_obs_to_tensor(agent_obs, device)
                    goal_i = current_goals[i:i+1]

                    logits = worker(ob_t, goal_i)

                    # Apply action masking (same reorder as training)
                    avail = ob_t["action_availability"]  # (1, 3)
                    mask = torch.stack([avail[:, 2], avail[:, 0], avail[:, 1]], dim=-1)
                    logits = logits + (mask - 1.0) * 1e8
                    action = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
                    actions[agent_id] = action

            obs, rewards, dones, truncs, infos, _global = env.step(actions)
            total_reward += sum(rewards.values())
            step += 1

            if all(dones.values()) or all(truncs.values()):
                break

        harvested = sum(1 for f in env.flowers if f.harvested)
        expired = sum(1 for f in env.flowers if f.expired)
        alive = sum(1 for i in range(num_sats) if env._battery[i] > 0)

        # Task-type breakdowns
        hard_total = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'HARD')
        soft_total = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'SOFT')
        none_total = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'NONE')
        hard_done = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'HARD' and f.harvested)
        soft_done = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'SOFT' and f.harvested)
        none_done = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'NONE' and f.harvested)

        results.append({
            "episode": ep,
            "harvest_rate": harvested / num_tasks,
            "harvested": harvested,
            "expired": expired,
            "total_flowers": num_tasks,
            "alive_sats": alive,
            "dead_sats": num_sats - alive,
            "total_reward": total_reward,
            "steps": step,
            "hard_total": hard_total, "hard_done": hard_done,
            "soft_total": soft_total, "soft_done": soft_done,
            "none_total": none_total, "none_done": none_done,
        })

        print(f"  [HRL] Ep {ep+1}/{num_episodes}: harvest={harvested}/{num_tasks} "
              f"({100*harvested/num_tasks:.0f}%), alive={alive}/{num_sats}", flush=True)

        env.close()

    return results


# ═══════════════════════════════════════════════════════════════
#  Gradient Policy evaluation IN BSK ENV (via LinearKeplerMapper)
# ═══════════════════════════════════════════════════════════════

# Import LinearKeplerMapper — self-contained, no env-specific deps
_linear_spec = importlib.util.spec_from_file_location(
    "linear_mapper",
    str(SCS_ROOT / "bee_swarm_package" / "linear.py"),
)
# We only need LinearKeplerMapper; avoid executing the whole module
# (it does sys.path manipulation and imports the SCS env).
# Instead, inline the flatten_norm projection — it's just math.

class _FlattenNormMapper:
    """Minimal re-implementation of LinearKeplerMapper(strategy='flatten_norm')."""
    def __init__(self, grid_size: int):
        self.grid_size = int(grid_size)
        self._half = self.grid_size / 2.0

    def map_position(self, fx: float, fy: float, fz: float):
        """(fx,fy,fz) → [x_grid, y_grid] in grid units, clipped."""
        dx = fx - self._half
        dy = fy - self._half
        dz = fz  # no grid-centre offset in z
        r = math.sqrt(dx*dx + dy*dy + dz*dz)
        if r < 1e-9:
            return [self._half, self._half]
        scale = self._half
        x = self._half + (dx / r) * scale
        y = self._half + (dy / r) * scale
        x = float(np.clip(x, 0.0, self.grid_size - 1))
        y = float(np.clip(y, 0.0, self.grid_size - 1))
        return [x, y]


class GossipProtocol:
    """Manages gossip communication between bees (from run_test.py)"""
    def __init__(self, num_bees: int, gossip_interval: int = 3):
        self.num_bees = num_bees
        self.gossip_interval = gossip_interval
        self.message_queues = {i: [] for i in range(num_bees)}

    def should_gossip(self, step: int) -> bool:
        return step % self.gossip_interval == 0

    def broadcast_messages(self, agents, current_step, bee_positions, bee_state, device):
        messages = []
        for agent in agents:
            if agent.active:
                pos = bee_positions.get(f"bee_{agent.bee_id}", (0, 0))
                msg = agent.create_gossip_message(current_step, pos, bee_state)
                messages.append(msg)
        for agent in agents:
            for msg in messages:
                if msg.sender_id != agent.bee_id:
                    agent.receive_gossip(msg)


def update_bee_agent(bee_vfrl_agents, bee_feature_state: Dict):
    """Update model agents with bee information (from run_test.py)"""
    agents_list = []
    terminated_bees = []
    for i, vfrl_agent in enumerate(bee_vfrl_agents):
        bee_id = f"bee_{i}"
        bee_state = bee_feature_state.get(bee_id, {})
        vfrl_agent.energy = bee_state.get('energy_level', 1.0)
        vfrl_agent.pollen_load = bee_state.get('pollen_collected', 0.0)
        vfrl_agent.update_agent_status(bee_state.get('is_available', False))
        vfrl_agent.update_task_dict(bee_state.get('assigned_task', {}))
        if not vfrl_agent.active:
            terminated_bees.append(i)
        agents_list.append(vfrl_agent)
    return agents_list, terminated_bees


def run_gradient_episodes(
    num_episodes: int,
    num_bees: int = 20,
    num_flowers: int = 60,
    grid_size: int = 20,
    max_steps: int = 500,
    max_energy_level: int = 2000,
    max_bee_capacity: int = 100,
    seed: int = 42,
    device: torch.device = None,
    verbose: bool = False,
) -> List[Dict]:
    """Run N episodes of the gradient policy using its own env."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load gradient model
    model_path = SCS_ROOT / "bees_drl_app" / "models" / "bee_gradient_policy_hierarchical_vfrl_best_ole.pth"
    gradient_policy = torch.load(model_path, map_location=device, weights_only=False)
    training_config = gradient_policy['metadata']

    retask_model = ReTaskModel()
    retask_model.load_state_dict(gradient_policy['retask_model'])
    retask_model.eval()

    worker_policy = BeePolicyWorker(
        num_groom_types=training_config['num_groom_types'],
        num_flowers=num_flowers,
        device=device
    )
    worker_policy.load_state_dict(gradient_policy['worker_policy'])
    worker_policy.eval()

    coordinator_state = gradient_policy.get('coordinators')

    # Test datasets always have 10 scenarios — cycle through them
    num_scenarios = 10

    results = []

    for ep in range(num_episodes):
        scenario_num = (ep % num_scenarios) + 1  # Cycle through scenarios 1-10
        np.random.seed(seed + ep)
        torch.manual_seed(seed + ep)
        random.seed(seed + ep)

        # Create fresh agents each episode
        bee_agents = [
            BeeGradientAgent(
                bee_id=i,
                worker_policy=worker_policy,
                retask_reassignment=retask_model,
                device=device,
                grid_size=grid_size,
                flower_feature_dim=6,
                max_steps=max_steps,
                embed_dim=64
            )
            for i in range(num_bees)
        ]
        for agent in bee_agents:
            agent.coordinator.load_state_dict(coordinator_state)
            agent.coordinator.eval()

        # Create gradient env
        env = GradientBeeForagingEnv(
            dataset_type="test_data",
            render_mode=None,
            num_bees=num_bees,
            grid_size=grid_size,
            num_flowers=num_flowers,
            max_steps=max_steps,
            num_scenario=num_scenarios,
            max_energy_level=max_energy_level,
            bee_capacity=max_bee_capacity,
        )
        env.reset(scenario_num=scenario_num)

        extractor = GlobalStateExtractor(env)
        env_wrapper = BeeStateWrapper(env, num_groom_types=training_config['num_groom_types'], extractor=extractor)
        gossip_protocol = GossipProtocol(num_bees=num_bees)
        task_manager = TaskManager(num_flowers=num_flowers)

        scenario_reward = 0.0
        done = False
        step = 1

        while not done:
            bee_info = extractor.get_bee_info()
            vfrl_agents, _ = update_bee_agent(bee_agents, bee_feature_state=bee_info)
            bee_current_positions, all_bee_states = extractor.get_bees_states_current_positions(step)

            actions_list = {}
            try:
                gossip_protocol.broadcast_messages(vfrl_agents, step, bee_current_positions, all_bee_states, device)
            except Exception:
                pass

            for agent in vfrl_agents:
                if agent.active:
                    try:
                        state_dict, bee_pos = extractor.get_global_state(agent_id=agent.agent_id, current_step=step)
                        action_mask = env_wrapper.get_action_mask(agent.agent_id)
                        reassigned_task_dict, _ = agent.perform_task_reassignment(
                            agent.agent_id, state_dict, step,
                            bee_pos, task_manager.orphaned_tasks,
                            is_training=False
                        )
                        action, _, _ = agent.select_hierarchical_action(
                            state_dict, step, action_mask,
                            bee_pos, task_manager.orphaned_tasks,
                            is_training=False
                        )
                        actions_list[agent.agent_id] = {'action': action, 'assigned_task': agent.my_tasks.keys()}
                    except Exception as e:
                        actions_list[agent.agent_id] = {'action': 0, 'assigned_task': {}}
                else:
                    actions_list[agent.agent_id] = {'action': 0, 'assigned_task': {}}

            try:
                rewards, terminations, truncations, failed_bees = env_wrapper.step(actions_list, step)
            except Exception as e:
                rewards = {a: 0 for a in actions_list}
                terminations = {a: False for a in actions_list}
                truncations = {a: False for a in actions_list}
                failed_bees = {a: False for a in actions_list}

            scenario_reward += sum(rewards.values())

            # Handle failed bees
            for agent in vfrl_agents:
                if failed_bees and failed_bees.get(agent.agent_id, False):
                    agent.update_agent_status(False)
                    task_manager.orphan_agent_tasks(agent)

            done = (failed_bees and all(failed_bees.values())) or (step >= max_steps)
            step += 1

        # Collect results
        harvested = sum(1 for f in env.flower_dict.values() if getattr(f, 'harvested', False))
        total = len(env.flower_dict)
        dead_bees = sum(1 for b in env.possible_bee_agents.values() if getattr(b, 'terminated', False))
        total_pollen = sum(
            getattr(f, 'pollen_amount', 0) for f in env.flower_dict.values()
            if getattr(f, 'harvested', False)
        )

        results.append({
            "episode": ep,
            "scenario": scenario_num,
            "harvest_rate": harvested / total if total > 0 else 0,
            "harvested": harvested,
            "total_flowers": total,
            "dead_bees": dead_bees,
            "alive_bees": num_bees - dead_bees,
            "total_reward": scenario_reward,
            "total_pollen": total_pollen,
            "steps": step,
        })

        print(f"  [Gradient] Ep {ep+1}/{num_episodes}: harvest={harvested}/{total} "
              f"({100*harvested/total:.0f}%), dead={dead_bees}/{num_bees}", flush=True)

        env.close()

    return results


# ═══════════════════════════════════════════════════════════════
#  Main comparison
# ═══════════════════════════════════════════════════════════════
def print_comparison(ac_results: List[Dict], gradient_results: List[Dict]):
    """Print side-by-side comparison table (Actor-Critic vs Gradient)."""
    def stats(results, key):
        vals = [r.get(key, 0) for r in results]
        return np.mean(vals), np.std(vals), np.min(vals), np.max(vals)

    width = 72

    print("\n" + "=" * width)
    print("BSK MODEL COMPARISON  —  Actor-Critic (Kepler) vs Gradient (2D)")
    print("=" * width)
    header = f"{'Metric':<30} {'Actor-Critic':>18} {'Gradient':>18}"
    print(header)
    print("-" * width)

    for label, key in [
        ("Task Completion (%)", "harvest_rate"),
        ("Tasks Completed", "harvested"),
        ("Total Reward", "total_reward"),
        ("Steps Used", "steps"),
    ]:
        am, a_s, _, _ = stats(ac_results, key)
        gm, gs, _, _ = stats(gradient_results, key)
        if key == "harvest_rate":
            am, a_s, gm, gs = am*100, a_s*100, gm*100, gs*100
        print(f"  {label:<28} {am:>7.1f} ± {a_s:>4.1f}   {gm:>7.1f} ± {gs:>4.1f}")

    # Dead satellites
    am_dead = np.mean([r.get("dead_sats", 0) for r in ac_results])
    gm_dead = np.mean([r.get("dead_sats", 0) for r in gradient_results])
    print(f"  {'Dead Sats (avg)':<28} {am_dead:>7.2f}            {gm_dead:>7.2f}")

    # Task-type breakdowns
    print("-" * width)
    print("  Task-Type Breakdown:")
    for wtype in ["hard", "soft", "none"]:
        at = np.mean([r.get(f"{wtype}_total", 0) for r in ac_results])
        ad = np.mean([r.get(f"{wtype}_done", 0) for r in ac_results])
        gt = np.mean([r.get(f"{wtype}_total", 0) for r in gradient_results])
        gd = np.mean([r.get(f"{wtype}_done", 0) for r in gradient_results])
        ar = 100*ad/at if at > 0 else 0
        gr = 100*gd/gt if gt > 0 else 0
        print(f"    {wtype.upper():<8} completed       {ad:>5.1f}/{at:>4.1f} ({ar:>4.0f}%)   {gd:>5.1f}/{gt:>4.1f} ({gr:>4.0f}%)")

    print("-" * width)

    # Per-model summaries
    ahr = [r["harvest_rate"] * 100 for r in ac_results]
    ghr = [r["harvest_rate"] * 100 for r in gradient_results]
    print(f"\n  Actor-Critic:     avg {np.mean(ahr):.1f}%, best {np.max(ahr):.0f}%, worst {np.min(ahr):.0f}%")
    print(f"  Gradient Policy:  avg {np.mean(ghr):.1f}%, best {np.max(ghr):.0f}%, worst {np.min(ghr):.0f}%")

    # Episodes with deaths
    ac_death_eps = sum(1 for r in ac_results if r.get("dead_sats", 0) > 0)
    gr_death_eps = sum(1 for r in gradient_results if r.get("dead_sats", 0) > 0)
    print(f"\n  Episodes with sat deaths:  Actor-Critic {ac_death_eps}/{len(ac_results)}, Gradient {gr_death_eps}/{len(gradient_results)}")
    print()


# ═══════════════════════════════════════════════════════════════
#  Gradient Policy running INSIDE BSK env (same env as HRL)
# ═══════════════════════════════════════════════════════════════

def _build_gradient_state_dict(env, bee_idx, mapper, device):
    """
    Build the state_dict that BeeGradientAgent.select_hierarchical_action expects,
    using the BSK env's internal state + LinearKeplerMapper for positions.
    """
    bee = env.bees[bee_idx]
    mapped_pos = mapper.map_position(bee.fx, bee.fy, bee.fz)

    # Bee features: [norm_x, norm_y, norm_pollen, terminated, norm_energy]
    battery_ratio = env._battery[bee_idx] / max(1.0, env._battery_max[bee_idx])
    bee_features = [[
        mapped_pos[0] / env.grid_size,
        mapped_pos[1] / env.grid_size,
        bee.load / max(1.0, bee.capacity),
        1.0 if bee.terminated else 0.0,
        battery_ratio,
    ]]
    bee_mask = [0.0 if bee.terminated else 1.0]
    bee_positions = [[int(mapped_pos[0]), int(mapped_pos[1])]]

    # Flower features for assigned flowers
    PRIORITY_MAP = {0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.6, 5: 0.6, 6: 0.6, 7: 0.99, 8: 0.99, 9: 0.99, 10: 0.99}
    flower_features = []
    flower_positions = []
    flower_mask = []

    assigned = getattr(bee, 'assigned_flowers', [])
    for fj in assigned:
        f = env.flowers[fj]
        pri_int = int(f.priority * 10)
        pri_val = PRIORITY_MAP.get(pri_int, 0.6)
        flower_features.append([
            f.x / env.grid_size,
            f.y / env.grid_size,
            1.0 if f.harvested else 0.0,
            f.window_start / max(1, env.max_steps),
            f.window_end / max(1, env.max_steps),
            pri_val,
        ])
        flower_positions.append([f.x, f.y])
        flower_mask.append(0.0 if f.harvested else 1.0)

    # If no assigned flowers, add a dummy so tensor shape is valid
    if len(flower_features) == 0:
        flower_features = [[0.0, 0.0, 1.0, 0.0, 1.0, 0.3]]
        flower_positions = [[0, 0]]
        flower_mask = [0.0]

    state_dict = {
        'bee_features': torch.tensor([bee_features], dtype=torch.float32, device=device),
        'flower_features': torch.tensor([flower_features], dtype=torch.float32, device=device),
        'bee_mask': torch.tensor([bee_mask], dtype=torch.float32, device=device),
        'flower_mask': torch.tensor([flower_mask], dtype=torch.float32, device=device),
        'flower_positions': torch.tensor(flower_positions, dtype=torch.float32, device=device),
        'bee_positions': torch.tensor(bee_positions, dtype=torch.int32, device=device),
    }
    return state_dict, mapped_pos


def _build_gradient_action_mask(env, bee_idx, num_assigned, device):
    """Create action mask for gradient agent, matching LinearModel.create_action_mask."""
    NUM_FIXED = 3  # DO_NOTHING, GROOM_OFFLOAD, GROOM_RECHARGE
    total_actions = NUM_FIXED + max(1, num_assigned)
    mask = torch.zeros(1, 1, total_actions, dtype=torch.bool, device=device)

    bee = env.bees[bee_idx]
    if bee.terminated:
        mask[0, 0, :] = True
        return mask

    # Block groom_offload if low pollen
    if bee.load <= bee.capacity / 3:
        mask[0, 0, 1] = True  # GROOM_OFFLOAD
    # Block groom_recharge if sufficient energy
    battery_ratio = env._battery[bee_idx] / max(1.0, env._battery_max[bee_idx])
    if battery_ratio > 0.33:
        mask[0, 0, 2] = True  # GROOM_RECHARGE

    # Mask harvested flowers
    assigned = getattr(bee, 'assigned_flowers', [])
    for local_idx, fj in enumerate(assigned):
        f = env.flowers[fj]
        if f.harvested:
            mask[0, 0, NUM_FIXED + local_idx] = True
        elif f.window_type == "HARD":
            if env.steps < f.window_start or env.steps > f.window_end:
                mask[0, 0, NUM_FIXED + local_idx] = True

    return mask


def _gradient_action_to_bsk(action_idx, bee, env):
    """Convert gradient agent action index → BSK env action (0=NOTHING, 1=HARVEST, 2=GROOM)."""
    if action_idx == 0:
        return 0  # DO_NOTHING
    elif action_idx in (1, 2):
        return 2  # GROOM
    else:
        return 1  # HARVEST (any flower-specific action maps to generic HARVEST)


def run_gradient_in_bsk_episodes(
    num_episodes: int,
    num_sats: int = 25,
    num_tasks: int = 50,
    grid_size: int = 75,
    max_steps: int = 1200,
    bsk_dt: float = 5.0,
    bsk_meters_per_unit: float = 200_000.0,
    seed: int = 42,
    device: torch.device = None,
    verbose: bool = False,
) -> List[Dict]:
    """
    Run the gradient policy INSIDE the BSK env using LinearKeplerMapper
    to project 3D orbital positions → 2D grid for the gradient agents.
    Same environment and flower layout as the HRL model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load gradient model
    model_path = SCS_ROOT / "bees_drl_app" / "models" / "bee_gradient_policy_hierarchical_vfrl_best_ole.pth"
    gradient_policy = torch.load(model_path, map_location=device, weights_only=False)
    training_config = gradient_policy['metadata']

    retask_model = ReTaskModel()
    retask_model.load_state_dict(gradient_policy['retask_model'])
    retask_model.eval()

    worker_policy = BeePolicyWorker(
        num_groom_types=training_config['num_groom_types'],
        num_flowers=num_tasks,
        device=device,
    )
    worker_policy.load_state_dict(gradient_policy['worker_policy'])
    worker_policy.eval()

    coordinator_state = gradient_policy.get('coordinators')

    mapper = _FlattenNormMapper(grid_size)

    results = []

    for ep in range(num_episodes):
        np.random.seed(seed + ep)
        torch.manual_seed(seed + ep)
        random.seed(seed + ep)

        # Create BSK env — SAME config as HRL evaluation
        env = BeeFiEnv(
            num_bees=num_sats,
            num_flowers=num_tasks,
            grid_size=grid_size,
            max_steps=max_steps,
            retask_board_size=3,
            use_basilisk=True,
            bsk_dt_sec=bsk_dt,
            bsk_meters_per_unit=bsk_meters_per_unit,
            low_battery_chance=0.30,
            verbose=False,
        )
        obs = env.reset()

        # Fresh gradient agents each episode
        bee_agents = [
            BeeGradientAgent(
                bee_id=i,
                worker_policy=worker_policy,
                retask_reassignment=retask_model,
                device=device,
                grid_size=grid_size,
                flower_feature_dim=6,
                max_steps=max_steps,
                embed_dim=64,
            )
            for i in range(num_sats)
        ]
        for agent in bee_agents:
            agent.coordinator.load_state_dict(coordinator_state)
            agent.coordinator.eval()

        task_manager = TaskManager(num_flowers=num_tasks)
        total_reward = 0.0
        step = 0

        while True:
            actions = {}
            for i in range(num_sats):
                bee = env.bees[i]
                agent = bee_agents[i]

                # Update agent state from BSK env
                agent.energy = env._battery[i] / max(1.0, env._battery_max[i])
                agent.pollen_load = bee.load
                is_available = not bee.terminated and env._recharge_until[i] <= env.steps
                agent.update_agent_status(is_available)

                # Build assigned task dict for gradient agent
                assigned = getattr(bee, 'assigned_flowers', [])
                assigned_task_dict = {}
                for fj in assigned:
                    f = env.flowers[fj]
                    assigned_task_dict[f"flower_{fj}"] = {
                        'position': [f.x, f.y],
                        'pollen_amount': f.pollen,
                        'priority_level': f.priority,
                        'min_step': f.window_start,
                        'max_step': f.window_end,
                        'hard_window': f.window_type == "HARD",
                        'harvested': f.harvested,
                    }
                agent.update_task_dict(assigned_task_dict)

                if not agent.active:
                    actions[f"bee_{i}"] = 0
                    continue

                try:
                    state_dict, bee_pos = _build_gradient_state_dict(env, i, mapper, device)
                    num_assigned = len(assigned)
                    action_mask = _build_gradient_action_mask(env, i, num_assigned, device)

                    action_idx, info, _ = agent.select_hierarchical_action(
                        state_dict=state_dict,
                        current_step=env.steps,
                        action_mask=action_mask,
                        current_bee_positions=bee_pos,
                        is_training=False,
                    )
                    action = _gradient_action_to_bsk(action_idx, bee, env)
                except Exception:
                    action = 0

                actions[f"bee_{i}"] = action

            obs, rewards, dones, truncs, infos, _global = env.step(actions)
            total_reward += sum(rewards.values())
            step += 1

            if all(dones.values()) or all(truncs.values()):
                break

        harvested = sum(1 for f in env.flowers if f.harvested)
        expired = sum(1 for f in env.flowers if getattr(f, 'expired', False))
        alive = sum(1 for i in range(num_sats) if env._battery[i] > 0)

        # Task-type breakdowns
        hard_total = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'HARD')
        soft_total = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'SOFT')
        none_total = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'NONE')
        hard_done = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'HARD' and f.harvested)
        soft_done = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'SOFT' and f.harvested)
        none_done = sum(1 for f in env.flowers if getattr(f, 'window_type', 'NONE') == 'NONE' and f.harvested)

        results.append({
            "episode": ep,
            "harvest_rate": harvested / num_tasks,
            "harvested": harvested,
            "expired": expired,
            "total_flowers": num_tasks,
            "dead_sats": num_sats - alive,
            "alive_sats": alive,
            "total_reward": total_reward,
            "steps": step,
            "hard_total": hard_total, "hard_done": hard_done,
            "soft_total": soft_total, "soft_done": soft_done,
            "none_total": none_total, "none_done": none_done,
        })

        print(f"  [Gradient+BSK] Ep {ep+1}/{num_episodes}: harvest={harvested}/{num_tasks} "
              f"({100*harvested/num_tasks:.0f}%), alive={alive}/{num_sats}", flush=True)

        env.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Actor-Critic (Kepler) vs Gradient Policy in BSK")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes per model")
    parser.add_argument("--flat_model", default="outputs/best_actor.pt", help="Path to flat Actor model (Kepler-trained)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default="bsk_comparison_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Episodes per model: {args.episodes}")

    flat_actor_path = os.path.abspath(args.flat_model)

    # ── Run Gradient Policy IN BSK ENV ────────────────────────
    print(f"\n[1/2] Running Gradient Policy ({args.episodes} episodes)...", flush=True)
    t0 = time.time()
    gradient_results = run_gradient_in_bsk_episodes(
        num_episodes=args.episodes, num_sats=25, num_tasks=50,
        grid_size=75, max_steps=1200, bsk_dt=5.0,
        bsk_meters_per_unit=200_000.0, seed=args.seed,
        device=device, verbose=args.verbose,
    )
    t_grad = time.time() - t0
    print(f"  Gradient done in {t_grad:.1f}s ({t_grad/60:.1f} min)", flush=True)

    # ── Run Actor-Critic (Kepler-trained) IN BSK ENV ──────────
    print(f"\n[2/2] Running Actor-Critic ({args.episodes} episodes)...", flush=True)
    print(f"  Model: {flat_actor_path}", flush=True)
    t0 = time.time()
    flat_results = run_beefi_actor_episodes(
        model_path=flat_actor_path, num_episodes=args.episodes,
        num_sats=25, num_tasks=50, grid_size=75, max_steps=1200,
        bsk_dt=5.0, bsk_meters_per_unit=200_000.0,
        seed=args.seed, device=device, verbose=args.verbose,
    )
    t_flat = time.time() - t0
    print(f"  Actor-Critic done in {t_flat:.1f}s ({t_flat/60:.1f} min)", flush=True)

    # ── Compare ──────────────────────────────────────────────
    print_comparison(flat_results, gradient_results)

    # ── Save results ─────────────────────────────────────────
    output = {
        "beefi_hrl": flat_results,
        "gradient_policy": gradient_results,
        "config": {
            "episodes": args.episodes,
            "actor_critic": {"model": flat_actor_path, "trained_with": "Kepler", "num_sats": 25, "num_tasks": 50, "grid": 75, "steps": 1200},
            "gradient": {"num_sats": 25, "num_tasks": 50, "grid": 75, "steps": 1200, "mapper": "flatten_norm", "env": "BSK"},
            "seed": args.seed,
            "low_battery_chance": 0.30,
        }
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
