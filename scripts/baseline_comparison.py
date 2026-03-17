"""
Baseline comparison script for Bee Swarm RL.

Compares trained RL policy against classical baselines:
- NFG (Nearest First Greedy) - Eq.18
- PWG (Priority Weighted Greedy) - Eq.19
- TWG (Time Window Aware Greedy) - Eq.20
- Hungarian (Optimal Assignment)
"""

import numpy as np
import torch
import argparse
from scipy.optimize import linear_sum_assignment
from bees_env import BeeForagingEnv
from bee_policy import Actor

# =============================================================================
# BASELINE IMPLEMENTATIONS
# =============================================================================

def get_bee_position(bee):
    """Extract 3D position from bee state."""
    return np.array([bee.fx, bee.fy, bee.fz])

def get_flower_position(flower):
    """Extract 2D position from flower (flowers are on ground plane)."""
    return np.array([flower.x, flower.y, 0.0])

def distance_3d(pos1, pos2, lambda_z=0.5):
    """Euclidean distance in 3D with z-weighting."""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dz = pos1[2] - pos2[2]
    return np.sqrt(dx*dx + dy*dy + lambda_z * dz*dz)

def get_available_flowers(env):
    """Get list of flowers that haven't been harvested."""
    available = []
    for i, flower in enumerate(env.flowers):
        if not flower.harvested:
            available.append((i, flower))
    return available

def is_window_open(flower, current_step):
    """Check if flower's time window is currently open."""
    window_type = getattr(flower, 'window_type', 'NONE')
    if window_type == "NONE":
        return True
    window_start = getattr(flower, 'window_start', 0)
    window_end = getattr(flower, 'window_end', float('inf'))
    return window_start <= current_step <= window_end

def nearest_first_greedy(bee_idx, env):
    """
    NFG (Eq.18): Select action for bee based on nearest available flower.
    Returns: action (0=DONOTHING, 1=HARVEST, 2=GROOM)
    """
    bee = env.bees[bee_idx]
    bee_pos = get_bee_position(bee)
    current_step = env.steps
    
    # Check if bee is terminated/truncated
    if bee.truncated or bee.terminated:
        return 0  # DONOTHING
    
    # Check if bee needs to groom (deposit pollen or recharge)
    load_ratio = bee.load / bee.capacity if bee.capacity > 0 else 0
    if load_ratio > 0.8:
        return 2  # GROOM - deposit pollen
    
    battery_ratio = bee.battery / bee.battery_capacity if bee.battery_capacity > 0 else 1
    if battery_ratio < 0.3:
        return 2  # GROOM - recharge battery
    
    # Find available flowers assigned to this bee or unassigned
    available = []
    for fj in bee.assigned_flowers:
        if fj < len(env.flowers) and not env.flowers[fj].harvested:
            available.append((fj, env.flowers[fj]))
    
    # Also consider unassigned flowers in retask board
    for entry in getattr(bee, 'retask_board', []):
        fj = entry.get('flower_id', -1)
        if 0 <= fj < len(env.flowers) and not env.flowers[fj].harvested:
            if (fj, env.flowers[fj]) not in available:
                available.append((fj, env.flowers[fj]))
    
    if not available:
        return 0  # DONOTHING - no flowers available
    
    # Find nearest flower
    nearest_flower = None
    nearest_idx = -1
    nearest_dist = float('inf')
    
    for idx, flower in available:
        flower_pos = get_flower_position(flower)
        dist = distance_3d(bee_pos, flower_pos)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_flower = flower
            nearest_idx = idx
    
    if nearest_flower is None:
        return 0  # DONOTHING
    
    # Check if we can harvest (in range and window open)
    harvest_radius = getattr(env, 'harvest_radius', 3.0) + getattr(env, 'reach_margin', 0.5)
    if nearest_dist <= harvest_radius:
        if is_window_open(nearest_flower, current_step):
            return 1  # HARVEST
    
    # Not in range yet, keep orbiting towards it
    return 0  # DONOTHING


def priority_weighted_greedy(bee_idx, env, alpha=0.5):
    """
    PWG (Eq.19): Select action based on pollen * priority / distance ratio.
    """
    bee = env.bees[bee_idx]
    bee_pos = get_bee_position(bee)
    current_step = env.steps
    
    # Check if bee is terminated/truncated
    if bee.truncated or bee.terminated:
        return 0
    
    # Check if bee needs to groom
    load_ratio = bee.load / bee.capacity if bee.capacity > 0 else 0
    if load_ratio > 0.8:
        return 2  # GROOM
    
    battery_ratio = bee.battery / bee.battery_capacity if bee.battery_capacity > 0 else 1
    if battery_ratio < 0.3:
        return 2  # GROOM
    
    # Find available flowers
    available = []
    for fj in bee.assigned_flowers:
        if fj < len(env.flowers) and not env.flowers[fj].harvested:
            available.append((fj, env.flowers[fj]))
    
    for entry in getattr(bee, 'retask_board', []):
        fj = entry.get('flower_id', -1)
        if 0 <= fj < len(env.flowers) and not env.flowers[fj].harvested:
            if (fj, env.flowers[fj]) not in available:
                available.append((fj, env.flowers[fj]))
    
    if not available:
        return 0  # DONOTHING
    
    # Score each flower
    best_flower = None
    best_idx = -1
    best_score = -float('inf')
    best_dist = float('inf')
    
    for idx, flower in available:
        flower_pos = get_flower_position(flower)
        dist = distance_3d(bee_pos, flower_pos)
        
        # Priority: HARD=3, SOFT=2, NONE=1
        window_type = getattr(flower, 'window_type', 'NONE')
        priority = {"HARD": 3, "SOFT": 2, "NONE": 1}.get(window_type, 1)
        
        # Score = (pollen * priority^alpha) / (distance + 0.1)
        pollen = getattr(flower, 'pollen', 1.0)
        score = (pollen * (priority ** alpha)) / (dist + 0.1)
        
        if score > best_score:
            best_score = score
            best_flower = flower
            best_idx = idx
            best_dist = dist
    
    if best_flower is None:
        return 0  # DONOTHING
    
    # Check if we can harvest
    harvest_radius = getattr(env, 'harvest_radius', 3.0) + getattr(env, 'reach_margin', 0.5)
    if best_dist <= harvest_radius:
        if is_window_open(best_flower, current_step):
            return 1  # HARVEST
    
    return 0  # DONOTHING


def time_window_aware_greedy(bee_idx, env, beta=0.7):
    """
    TWG (Eq.20): Extends priority_weighted with urgency for HARD windows.
    """
    bee = env.bees[bee_idx]
    bee_pos = get_bee_position(bee)
    current_step = env.steps
    
    # Check if bee is terminated/truncated
    if bee.truncated or bee.terminated:
        return 0
    
    # Check if bee needs to groom
    load_ratio = bee.load / bee.capacity if bee.capacity > 0 else 0
    if load_ratio > 0.8:
        return 2  # GROOM
    
    battery_ratio = bee.battery / bee.battery_capacity if bee.battery_capacity > 0 else 1
    if battery_ratio < 0.3:
        return 2  # GROOM
    
    # Find available flowers
    available = []
    for fj in bee.assigned_flowers:
        if fj < len(env.flowers) and not env.flowers[fj].harvested:
            available.append((fj, env.flowers[fj]))
    
    for entry in getattr(bee, 'retask_board', []):
        fj = entry.get('flower_id', -1)
        if 0 <= fj < len(env.flowers) and not env.flowers[fj].harvested:
            if (fj, env.flowers[fj]) not in available:
                available.append((fj, env.flowers[fj]))
    
    if not available:
        return 0  # DONOTHING
    
    # Score each flower with urgency
    best_flower = None
    best_idx = -1
    best_score = -float('inf')
    best_dist = float('inf')
    
    for idx, flower in available:
        flower_pos = get_flower_position(flower)
        dist = distance_3d(bee_pos, flower_pos)
        
        # Priority: HARD=3, SOFT=2, NONE=1
        window_type = getattr(flower, 'window_type', 'NONE')
        priority = {"HARD": 3, "SOFT": 2, "NONE": 1}.get(window_type, 1)
        pollen = getattr(flower, 'pollen', 1.0)
        
        # Calculate urgency for HARD windows
        urgency = 0.0
        if window_type == "HARD":
            window_start = getattr(flower, 'window_start', 0)
            window_end = getattr(flower, 'window_end', env.max_steps)
            window_duration = max(1, window_end - window_start)
            time_remaining = window_end - current_step
            if time_remaining > 0:
                urgency = max(0, 1 - (time_remaining / window_duration))
            else:
                urgency = 1.0  # Window passed, maximum urgency
        
        # Score = (pollen * priority * (1 + beta * urgency)) / (distance + 0.1)
        score = (pollen * priority * (1 + beta * urgency)) / (dist + 0.1)
        
        if score > best_score:
            best_score = score
            best_flower = flower
            best_idx = idx
            best_dist = dist
    
    if best_flower is None:
        return 0  # DONOTHING
    
    # Check if we can harvest
    harvest_radius = getattr(env, 'harvest_radius', 3.0) + getattr(env, 'reach_margin', 0.5)
    if best_dist <= harvest_radius:
        if is_window_open(best_flower, current_step):
            return 1  # HARVEST
    
    return 0  # DONOTHING


def hungarian_optimal(env):
    """
    Centralized optimal assignment using scipy.optimize.linear_sum_assignment.
    Returns dict of {agent_name: action}
    """
    current_step = env.steps
    num_bees = env.num_bees
    
    # Get all available (unharvested) flowers
    available = get_available_flowers(env)
    
    # Initialize all bees to DONOTHING
    actions = {f"bee_{i}": 0 for i in range(num_bees)}
    
    if not available:
        return actions  # No flowers, all DONOTHING
    
    # Check each bee for grooming needs first
    bee_needs_groom = {}
    for i in range(num_bees):
        bee = env.bees[i]
        if bee.truncated or bee.terminated:
            bee_needs_groom[i] = True
            continue
            
        load_ratio = bee.load / bee.capacity if bee.capacity > 0 else 0
        battery_ratio = bee.battery / bee.battery_capacity if bee.battery_capacity > 0 else 1
        
        if load_ratio > 0.8:
            actions[f"bee_{i}"] = 2  # GROOM
            bee_needs_groom[i] = True
        elif battery_ratio < 0.3:
            actions[f"bee_{i}"] = 2  # GROOM
            bee_needs_groom[i] = True
        else:
            bee_needs_groom[i] = False
    
    # Get bees that are available for assignment
    available_bees = [i for i in range(num_bees) if not bee_needs_groom[i]]
    
    if not available_bees:
        return actions  # All bees need grooming
    
    # Build cost matrix: rows = available bees, cols = available flowers
    num_available_bees = len(available_bees)
    num_available_flowers = len(available)
    
    # Handle case where we have more bees than flowers or vice versa
    size = max(num_available_bees, num_available_flowers)
    cost_matrix = np.full((size, size), 1e6)  # Large cost for dummy assignments
    
    for bi, bee_idx in enumerate(available_bees):
        bee = env.bees[bee_idx]
        bee_pos = get_bee_position(bee)
        
        for fi, (flower_idx, flower) in enumerate(available):
            flower_pos = get_flower_position(flower)
            dist = distance_3d(bee_pos, flower_pos)
            
            # Adjust cost based on window urgency
            window_type = getattr(flower, 'window_type', 'NONE')
            if window_type == "HARD":
                window_end = getattr(flower, 'window_end', env.max_steps)
                time_remaining = window_end - current_step
                if time_remaining < 0:
                    cost_matrix[bi, fi] = 1e6  # Window missed, very high cost
                else:
                    # Lower cost for urgent flowers
                    urgency_bonus = max(0, 1 - time_remaining / 100)
                    cost_matrix[bi, fi] = dist - urgency_bonus * 10
            else:
                cost_matrix[bi, fi] = dist
    
    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Assign actions based on Hungarian result
    harvest_radius = getattr(env, 'harvest_radius', 3.0) + getattr(env, 'reach_margin', 0.5)
    
    for bi, fi in zip(row_ind, col_ind):
        if bi >= num_available_bees or fi >= num_available_flowers:
            continue  # Dummy assignment
        
        bee_idx = available_bees[bi]
        bee = env.bees[bee_idx]
        bee_pos = get_bee_position(bee)
        
        flower_idx, flower = available[fi]
        flower_pos = get_flower_position(flower)
        dist = distance_3d(bee_pos, flower_pos)
        
        # Check if bee can harvest assigned flower
        if dist <= harvest_radius:
            if is_window_open(flower, current_step):
                actions[f"bee_{bee_idx}"] = 1  # HARVEST
    
    return actions


# =============================================================================
# POLICY WRAPPERS
# =============================================================================

class NFGPolicy:
    """Nearest First Greedy Policy"""
    def __init__(self, env):
        self.env = env
        self.name = "NFG (Eq.18)"
    
    def get_actions(self, observations):
        actions = {}
        for i in range(self.env.num_bees):
            agent_name = f"bee_{i}"
            actions[agent_name] = nearest_first_greedy(i, self.env)
        return actions


class PWGPolicy:
    """Priority Weighted Greedy Policy"""
    def __init__(self, env, alpha=0.5):
        self.env = env
        self.alpha = alpha
        self.name = f"PWG (Eq.19, α={alpha})"
    
    def get_actions(self, observations):
        actions = {}
        for i in range(self.env.num_bees):
            agent_name = f"bee_{i}"
            actions[agent_name] = priority_weighted_greedy(i, self.env, self.alpha)
        return actions


class TWGPolicy:
    """Time Window Aware Greedy Policy"""
    def __init__(self, env, beta=0.7):
        self.env = env
        self.beta = beta
        self.name = f"TWG (Eq.20, β={beta})"
    
    def get_actions(self, observations):
        actions = {}
        for i in range(self.env.num_bees):
            agent_name = f"bee_{i}"
            actions[agent_name] = time_window_aware_greedy(i, self.env, self.beta)
        return actions


class HungarianPolicy:
    """Hungarian Optimal Assignment Policy"""
    def __init__(self, env):
        self.env = env
        self.name = "Hungarian"
    
    def get_actions(self, observations):
        return hungarian_optimal(self.env)


class TrainedModelPolicy:
    """Trained RL Policy"""
    def __init__(self, env, model_path, device="cpu"):
        self.env = env
        self.device = device
        self.name = "Trained RL"
        
        # Load model with correct architecture (must match trained model!)
        num_bees = env.num_bees
        num_flowers = env.num_flowers
        action_dim = 3  # Fixed: DONOTHING, HARVEST, GROOM
        retask_board_size = getattr(env, "retask_board_size", 2)
        
        self.actor = Actor(
            num_bees=num_bees,
            action_dim=action_dim,
            num_flowers=num_flowers,
            retask_board_size=retask_board_size,
            grid_size=getattr(env, "grid_size", 50),
        ).to(device)
        self.actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.actor.eval()
    
    def get_actions(self, observations):
        """Get actions from trained actor network."""
        actions = {}
        with torch.no_grad():
            for agent_name, obs in observations.items():
                # Convert observation dict to tensor format expected by Actor
                if isinstance(obs, dict):
                    obs_tensor = {
                        k: torch.FloatTensor(v).unsqueeze(0).to(self.device)
                        for k, v in obs.items()
                    }
                    logits = self.actor(obs_tensor)
                else:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    logits = self.actor(obs_tensor)
                
                action = torch.argmax(logits, dim=-1).item()
                actions[agent_name] = action
        return actions


# =============================================================================
# EVALUATION
# =============================================================================

def create_episode_metrics(num_bees, num_flowers):
    """Initialize metrics dictionary for an episode."""
    return {
        # Task completion
        'flowers_harvested': 0,
        'total_flowers': num_flowers,
        'tcr': 0.0,  # Task Completion Rate = harvested/total * 100
        
        # Episode info
        'episode_length': 0,
        'total_reward': 0.0,
        
        # Time windows
        'hard_windows_total': 0,
        'hard_windows_completed': 0,
        'dsr': 0.0,  # Deadline Satisfaction Rate = completed/total * 100
        
        # Fault tolerance (for low-battery episodes)
        'battery_deaths': 0,
        'orphaned_tasks': 0,
        'orphaned_completed': 0,
        'frr': 0.0,  # Fault Recovery Rate = orphaned_completed/orphaned * 100
        
        # Per-agent tracking
        'per_agent_harvested': [0] * num_bees,
        'per_agent_deaths': [0] * num_bees,
        'per_agent_retasks_claimed': [0] * num_bees,
    }


def evaluate_policy(policy, env, num_episodes=10, max_steps=500):
    """Evaluate a policy over multiple episodes with comprehensive metrics."""
    
    # Aggregate results across episodes
    all_episode_metrics = []
    
    for ep in range(num_episodes):
        observations = env.reset()
        num_bees = env.num_bees
        num_flowers = env.num_flowers
        
        # Initialize episode metrics
        metrics = create_episode_metrics(num_bees, num_flowers)
        
        # Track state for orphan/retask detection
        prev_battery = {f"bee_{i}": getattr(env.bees[i], 'battery', 100) for i in range(num_bees)}
        prev_assigned = {}  # bee_id -> set of assigned flower indices
        orphaned_flowers = set()  # Track flowers orphaned due to battery death
        
        # Initialize previous assignments
        for i in range(num_bees):
            bee = env.bees[i]
            prev_assigned[i] = set(getattr(bee, 'assigned_flowers', []))
        
        # Count HARD window flowers
        for flower in env.flowers:
            if getattr(flower, 'window_type', 'NONE') == "HARD":
                metrics['hard_windows_total'] += 1
        
        for step in range(max_steps):
            # Get actions from policy
            actions = policy.get_actions(observations)
            
            # Step environment
            observations, rewards, terminations, truncations, infos, _ = env.step(actions)
            
            # Sum rewards
            metrics['total_reward'] += sum(rewards.values())
            metrics['episode_length'] += 1
            
            # Track per-agent harvests and battery deaths
            for i in range(num_bees):
                bee = env.bees[i]
                agent_name = f"bee_{i}"
                current_battery = getattr(bee, 'battery', 100)
                
                # Check for battery death (transition from >0 to <=0)
                if prev_battery[agent_name] > 0 and current_battery <= 0:
                    metrics['battery_deaths'] += 1
                    metrics['per_agent_deaths'][i] += 1
                    
                    # Track orphaned tasks (flowers that were assigned to this bee)
                    assigned = getattr(bee, 'assigned_flowers', [])
                    for fj in assigned:
                        if fj < len(env.flowers) and not env.flowers[fj].harvested:
                            orphaned_flowers.add(fj)
                            metrics['orphaned_tasks'] += 1
                
                prev_battery[agent_name] = current_battery
                
                # Track retask claims (new flowers appearing in assignment)
                current_assigned = set(getattr(bee, 'assigned_flowers', []))
                new_claims = current_assigned - prev_assigned[i]
                
                # Check if any new claims came from retask board (orphaned flowers)
                for fj in new_claims:
                    if fj in orphaned_flowers:
                        metrics['per_agent_retasks_claimed'][i] += 1
                
                prev_assigned[i] = current_assigned
            
            # Check if done
            if all(terminations.values()) or all(truncations.values()):
                break
        
        # Compute final metrics
        metrics['flowers_harvested'] = sum(1 for f in env.flowers if f.harvested)
        metrics['tcr'] = (metrics['flowers_harvested'] / metrics['total_flowers'] * 100) if metrics['total_flowers'] > 0 else 0.0
        
        # Count per-agent harvests
        for i in range(num_bees):
            bee = env.bees[i]
            # Count harvested flowers that were assigned to this bee
            assigned = getattr(bee, 'assigned_flowers', [])
            for fj in assigned:
                if fj < len(env.flowers) and env.flowers[fj].harvested:
                    metrics['per_agent_harvested'][i] += 1
        
        # DSR: Check HARD window completion (harvested before deadline)
        for flower in env.flowers:
            if getattr(flower, 'window_type', 'NONE') == "HARD":
                harvested_step = getattr(flower, 'harvested_step', None)
                if flower.harvested and harvested_step is not None:
                    if harvested_step <= getattr(flower, 'window_end', float('inf')):
                        metrics['hard_windows_completed'] += 1
        
        if metrics['hard_windows_total'] > 0:
            metrics['dsr'] = metrics['hard_windows_completed'] / metrics['hard_windows_total'] * 100
        else:
            metrics['dsr'] = 100.0  # No HARD windows = 100% DSR
        
        # FRR: Check how many orphaned tasks were completed
        for fj in orphaned_flowers:
            if fj < len(env.flowers) and env.flowers[fj].harvested:
                metrics['orphaned_completed'] += 1
        
        if metrics['orphaned_tasks'] > 0:
            metrics['frr'] = metrics['orphaned_completed'] / metrics['orphaned_tasks'] * 100
        else:
            metrics['frr'] = 100.0  # No orphans = 100% FRR
        
        all_episode_metrics.append(metrics)
        
        # Print episode summary
        status = "✓ SUCCESS" if metrics['tcr'] == 100.0 else f"TCR={metrics['tcr']:.1f}%"
        fault_info = f"Deaths={metrics['battery_deaths']}, Orphans={metrics['orphaned_tasks']}" if metrics['battery_deaths'] > 0 else ""
        print(f"  Ep {ep+1}: {status}, Steps={metrics['episode_length']}, Reward={metrics['total_reward']:.1f} {fault_info}")
    
    return all_episode_metrics


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

EXPERIMENTS = {
    'normal': {
        'num_episodes': 100,
        'low_battery_mode': False,
        'description': 'Normal battery conditions'
    },
    'fault': {
        'num_episodes': 100,
        'low_battery_mode': True,  # Force low battery: 1/3 capacity, 2-3x drain
        'description': 'Low-battery fault tolerance test'
    },
    'scalability': {
        'num_episodes': 100,
        'agent_counts': [3, 5, 7, 10],
        'description': 'Varying agent count'
    }
}

METHODS = ['ours', 'nfg', 'pwg', 'twg', 'hungarian']


def create_policies(env, model_path, device="cpu"):
    """Create all policy instances for comparison."""
    return {
        'ours': TrainedModelPolicy(env, model_path, device),
        'nfg': NFGPolicy(env),
        'pwg': PWGPolicy(env, alpha=0.5),
        'twg': TWGPolicy(env, beta=0.7),
        'hungarian': HungarianPolicy(env),
    }


def run_experiment_normal(model_path, num_episodes=100, max_steps=500, device="cpu"):
    """Run normal battery conditions experiment."""
    print("\n" + "="*80)
    print("EXPERIMENT: NORMAL BATTERY CONDITIONS")
    print("="*80)
    
    env = BeeForagingEnv(
        num_bees=5,
        num_flowers=12,
        max_steps=max_steps,
        verbose=False,
        retask_board_size=2,
        # Normal battery settings (use defaults)
    )
    
    policies = create_policies(env, model_path, device)
    
    all_results = {}
    for method_name in METHODS:
        policy = policies[method_name]
        print(f"\n  Evaluating: {policy.name} ({num_episodes} episodes)...")
        results = evaluate_policy(policy, env, num_episodes, max_steps)
        all_results[policy.name] = results
    
    return all_results


def run_experiment_fault(model_path, num_episodes=100, max_steps=500, device="cpu"):
    """Run low-battery fault tolerance experiment."""
    print("\n" + "="*80)
    print("EXPERIMENT: LOW-BATTERY FAULT TOLERANCE (AGGRESSIVE)")
    print("="*80)
    print("  Mode: EXTREME battery stress - bees WILL die!")
    print("  Settings: 30-60 step battery, 3x drain rate")
    
    env = BeeForagingEnv(
        num_bees=5,
        num_flowers=12,
        max_steps=max_steps,
        verbose=True,  # Show deaths
        retask_board_size=2,
        # EXTREME low battery settings - guaranteed deaths
        battery_min_steps=30,   # Very short battery life (30 steps)
        battery_max_steps=60,   # Max 60 steps before death
        drain_per_step=3.0,     # 3x drain rate - dies in 10-20 steps!
        recharge_steps=50,      # Long recharge time
    )
    
    policies = create_policies(env, model_path, device)
    
    all_results = {}
    for method_name in METHODS:
        policy = policies[method_name]
        print(f"\n  Evaluating: {policy.name} ({num_episodes} episodes)...")
        results = evaluate_policy(policy, env, num_episodes, max_steps)
        all_results[policy.name] = results
    
    return all_results


def run_experiment_scalability(model_path, num_episodes=100, max_steps=500, device="cpu"):
    """Run scalability experiment with varying agent counts."""
    print("\n" + "="*80)
    print("EXPERIMENT: SCALABILITY (VARYING AGENT COUNT)")
    print("="*80)
    
    agent_counts = EXPERIMENTS['scalability']['agent_counts']
    all_results = {}
    
    for num_bees in agent_counts:
        print(f"\n  --- {num_bees} Bees ---")
        
        # Scale flowers proportionally (roughly 2-3 flowers per bee)
        num_flowers = num_bees * 2 + 2
        
        env = BeeForagingEnv(
            num_bees=num_bees,
            num_flowers=num_flowers,
            max_steps=max_steps,
            verbose=False,
            retask_board_size=2,
        )
        
        # For scalability, only run baselines (trained model is fixed to 5 bees)
        # Create baseline policies only
        baseline_policies = {
            'nfg': NFGPolicy(env),
            'pwg': PWGPolicy(env, alpha=0.5),
            'twg': TWGPolicy(env, beta=0.7),
            'hungarian': HungarianPolicy(env),
        }
        
        # Add trained model only if num_bees matches training config (5)
        if num_bees == 5:
            try:
                baseline_policies['ours'] = TrainedModelPolicy(env, model_path, device)
            except RuntimeError as e:
                print(f"    [WARN] Trained model not compatible with {num_bees} bees: {e}")
        
        scale_results = {}
        for method_name, policy in baseline_policies.items():
            print(f"    Evaluating: {policy.name}...")
            results = evaluate_policy(policy, env, num_episodes, max_steps)
            scale_results[policy.name] = results
        
        all_results[f"{num_bees}_bees"] = scale_results
    
    return all_results


def print_summary_table(all_results, experiment_name=""):
    """Print comprehensive metrics summary table."""
    
    print(f"\n{'='*100}")
    print(f"RESULTS: {experiment_name}")
    print('='*100)
    
    # Header row
    print(f"{'Method':<25} {'TCR%':>7} {'DSR%':>7} {'FRR%':>7} {'Reward':>9} {'±Std':>7} {'Steps':>7} {'Deaths':>7} {'Orphans':>8}")
    print('-'*100)
    
    for name, episode_metrics in all_results.items():
        # Aggregate metrics across episodes
        tcr_list = [m['tcr'] for m in episode_metrics]
        dsr_list = [m['dsr'] for m in episode_metrics]
        frr_list = [m['frr'] for m in episode_metrics]
        reward_list = [m['total_reward'] for m in episode_metrics]
        length_list = [m['episode_length'] for m in episode_metrics]
        deaths_list = [m['battery_deaths'] for m in episode_metrics]
        orphans_list = [m['orphaned_tasks'] for m in episode_metrics]
        
        tcr = np.mean(tcr_list)
        dsr = np.mean(dsr_list)
        frr = np.mean(frr_list)
        reward = np.mean(reward_list)
        std = np.std(reward_list)
        length = np.mean(length_list)
        deaths = np.mean(deaths_list)
        orphans = np.mean(orphans_list)
        
        print(f"{name:<25} {tcr:>6.1f}% {dsr:>6.1f}% {frr:>6.1f}% {reward:>9.1f} {std:>7.1f} {length:>7.1f} {deaths:>7.2f} {orphans:>8.2f}")
    
    print('='*100)
    print("TCR = Task Completion Rate | DSR = Deadline Satisfaction Rate (HARD windows)")
    print("FRR = Fault Recovery Rate (orphaned tasks completed) | Deaths = Battery failures")
    print('='*100)


def print_scalability_table(scale_results):
    """Print scalability experiment results."""
    
    print(f"\n{'='*120}")
    print("SCALABILITY RESULTS")
    print('='*120)
    
    # Get method names from first scale
    first_scale = list(scale_results.values())[0]
    method_names = list(first_scale.keys())
    
    # Header
    print(f"{'Bees':<8}", end="")
    for method in method_names:
        print(f" | {method:<20}", end="")
    print()
    print("-"*120)
    
    # TCR row
    print(f"{'TCR%':<8}", end="")
    for scale_name, methods in scale_results.items():
        for method_name, metrics in methods.items():
            tcr = np.mean([m['tcr'] for m in metrics])
            print(f" | {tcr:>18.1f}%", end="")
        break  # Only print header once
    print()
    
    for scale_name, methods in scale_results.items():
        num_bees = scale_name.replace("_bees", "")
        print(f"{num_bees:<8}", end="")
        for method_name, metrics in methods.items():
            tcr = np.mean([m['tcr'] for m in metrics])
            print(f" | {tcr:>19.1f}", end="")
        print()
    
    print('='*120)


def print_per_agent_breakdown(all_results, method_name="Trained RL"):
    """Print per-agent breakdown for a specific method."""
    
    if method_name not in all_results:
        return
    
    print(f"\n{'='*60}")
    print(f"PER-AGENT BREAKDOWN ({method_name})")
    print('='*60)
    
    rl_metrics = all_results[method_name]
    num_bees = len(rl_metrics[0]['per_agent_harvested'])
    
    print(f"{'Agent':<12} {'Harvested':>10} {'Deaths':>10} {'Retasks':>10}")
    print('-'*42)
    
    for i in range(num_bees):
        harvested = np.mean([m['per_agent_harvested'][i] for m in rl_metrics])
        deaths = np.mean([m['per_agent_deaths'][i] for m in rl_metrics])
        retasks = np.mean([m['per_agent_retasks_claimed'][i] for m in rl_metrics])
        print(f"Bee {i:<8} {harvested:>10.2f} {deaths:>10.2f} {retasks:>10.2f}")
    
    print('='*60)


def save_results_json(all_results, filename):
    """Save results to JSON file."""
    import json
    
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(filename, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\nResults saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Baseline Comparison for Bee Swarm RL')
    parser.add_argument('--model', type=str, default='best', help='Model tag (final, best, upd100, etc.)')
    parser.add_argument('--experiment', type=str, default='all', 
                        choices=['all', 'normal', 'fault', 'scalability'],
                        help='Which experiment to run')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes per policy')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--cpu', action='store_true', help='Force CPU execution')
    parser.add_argument('--save', type=str, default=None, help='Save results to JSON file')
    args = parser.parse_args()
    
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = f'{args.model}/outputs_actor.pt'
    
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Episodes per method: {args.episodes}")
    
    all_experiment_results = {}
    
    # Run selected experiments
    if args.experiment in ['all', 'normal']:
        results = run_experiment_normal(model_path, args.episodes, args.max_steps, device)
        print_summary_table(results, "NORMAL BATTERY CONDITIONS")
        print_per_agent_breakdown(results, "Trained RL")
        all_experiment_results['normal'] = results
    
    if args.experiment in ['all', 'fault']:
        results = run_experiment_fault(model_path, args.episodes, args.max_steps, device)
        print_summary_table(results, "LOW-BATTERY FAULT TOLERANCE")
        print_per_agent_breakdown(results, "Trained RL")
        all_experiment_results['fault'] = results
    
    if args.experiment in ['all', 'scalability']:
        results = run_experiment_scalability(model_path, args.episodes, args.max_steps, device)
        print_scalability_table(results)
        all_experiment_results['scalability'] = results
    
    # Save results if requested
    if args.save:
        save_results_json(all_experiment_results, args.save)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()