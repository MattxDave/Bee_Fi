"""
Training Curves Analysis for Bee Swarm RL.

Extracts performance metrics from saved checkpoints and generates training curves.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

from bees_env import BeeForagingEnv
from bee_policy import Actor


def get_checkpoint_dirs():
    """Find all checkpoint directories."""
    checkpoints = []
    
    # Look for upd* directories
    for item in os.listdir('.'):
        if os.path.isdir(item):
            if item.startswith('upd') and os.path.exists(f'{item}/outputs_actor.pt'):
                try:
                    update_num = int(item.replace('upd', ''))
                    checkpoints.append((update_num, item))
                except ValueError:
                    pass
            elif item == 'best' and os.path.exists(f'{item}/outputs_actor.pt'):
                checkpoints.append((-1, item))  # Special marker for best
            elif item == 'final' and os.path.exists(f'{item}/outputs_actor.pt'):
                checkpoints.append((999999, item))  # Special marker for final
    
    # Sort by update number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def load_actor(model_path, num_bees=5, action_dim=3, hidden_dim=256, num_flowers=12, retask_board_size=2):
    """Load an actor from checkpoint."""
    actor = Actor(
        num_bees=num_bees,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_flowers=num_flowers,
        retask_board_size=retask_board_size
    )
    actor.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    actor.eval()
    return actor


def evaluate_checkpoint(actor, env, num_episodes=20, max_steps=500, device='cpu'):
    """Evaluate a checkpoint and return metrics."""
    metrics = {
        'rewards': [],
        'tcr': [],
        'dsr': [],
        'episode_length': [],
        'flowers_harvested': []
    }
    
    for ep in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Get actions from actor
            actions = {}
            with torch.no_grad():
                for agent_name, agent_obs in obs.items():
                    # Convert observation dict to tensor format expected by Actor
                    if isinstance(agent_obs, dict):
                        obs_tensor = {
                            k: torch.FloatTensor(v).unsqueeze(0).to(device)
                            for k, v in agent_obs.items()
                        }
                    else:
                        obs_tensor = torch.FloatTensor(agent_obs).unsqueeze(0).to(device)
                    
                    action_probs = actor(obs_tensor)
                    action = torch.argmax(action_probs, dim=-1).item()
                    actions[agent_name] = action
            
            # Step environment
            result = env.step(actions)
            if len(result) == 6:
                obs, rewards, terms, truncs, infos, _ = result
            else:
                obs, rewards, terms, truncs, infos = result
            episode_reward += sum(rewards.values())
            
            if any(terms.values()) or any(truncs.values()):
                break
        
        # Calculate metrics
        harvested = sum(1 for f in env.flowers if f.harvested)
        tcr = harvested / len(env.flowers) * 100
        
        # DSR
        hard_total = sum(1 for f in env.flowers if f.window_type == "HARD")
        if hard_total > 0:
            hard_completed = sum(1 for f in env.flowers 
                                if f.window_type == "HARD" and f.harvested)
            dsr = hard_completed / hard_total * 100
        else:
            dsr = 100.0
        
        metrics['rewards'].append(episode_reward)
        metrics['tcr'].append(tcr)
        metrics['dsr'].append(dsr)
        metrics['episode_length'].append(step + 1)
        metrics['flowers_harvested'].append(harvested)
    
    return {
        'reward_mean': np.mean(metrics['rewards']),
        'reward_std': np.std(metrics['rewards']),
        'tcr_mean': np.mean(metrics['tcr']),
        'tcr_std': np.std(metrics['tcr']),
        'dsr_mean': np.mean(metrics['dsr']),
        'dsr_std': np.std(metrics['dsr']),
        'length_mean': np.mean(metrics['episode_length']),
        'length_std': np.std(metrics['episode_length']),
        'harvested_mean': np.mean(metrics['flowers_harvested']),
        'harvested_std': np.std(metrics['flowers_harvested'])
    }


def generate_training_curves(num_episodes=20, save_path='training_curves.json'):
    """Generate training curves by evaluating all checkpoints."""
    
    # Create environment
    env = BeeForagingEnv(retask_board_size=2, verbose=False)
    env.reset()
    
    # Find checkpoints
    checkpoints = get_checkpoint_dirs()
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Evaluate each checkpoint
    results = {
        'updates': [],
        'checkpoint_names': [],
        'reward_mean': [],
        'reward_std': [],
        'tcr_mean': [],
        'tcr_std': [],
        'dsr_mean': [],
        'dsr_std': [],
        'length_mean': [],
        'length_std': [],
        'harvested_mean': [],
        'harvested_std': []
    }
    
    for update_num, checkpoint_dir in checkpoints:
        print(f"\nEvaluating checkpoint: {checkpoint_dir} (update {update_num})")
        
        # Load actor
        model_path = f'{checkpoint_dir}/outputs_actor.pt'
        try:
            actor = load_actor(model_path, num_bees=5)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue
        
        # Evaluate
        metrics = evaluate_checkpoint(actor, env, num_episodes=num_episodes)
        
        # Store results
        results['updates'].append(update_num if update_num >= 0 else 0)
        results['checkpoint_names'].append(checkpoint_dir)
        
        for key in ['reward_mean', 'reward_std', 'tcr_mean', 'tcr_std', 
                    'dsr_mean', 'dsr_std', 'length_mean', 'length_std',
                    'harvested_mean', 'harvested_std']:
            results[key].append(metrics[key])
        
        print(f"  Reward: {metrics['reward_mean']:.1f} ± {metrics['reward_std']:.1f}")
        print(f"  TCR: {metrics['tcr_mean']:.1f}% ± {metrics['tcr_std']:.1f}%")
        print(f"  DSR: {metrics['dsr_mean']:.1f}% ± {metrics['dsr_std']:.1f}%")
        print(f"  Steps: {metrics['length_mean']:.1f} ± {metrics['length_std']:.1f}")
    
    # Save results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")
    
    return results


def plot_training_curves(results, save_path='training_curves.png'):
    """Plot training curves from results."""
    
    # Filter out 'best' and 'final' for the main curve (they're not sequential updates)
    updates = []
    reward_mean = []
    reward_std = []
    tcr_mean = []
    tcr_std = []
    length_mean = []
    length_std = []
    
    for i, name in enumerate(results['checkpoint_names']):
        if name.startswith('upd'):
            updates.append(results['updates'][i])
            reward_mean.append(results['reward_mean'][i])
            reward_std.append(results['reward_std'][i])
            tcr_mean.append(results['tcr_mean'][i])
            tcr_std.append(results['tcr_std'][i])
            length_mean.append(results['length_mean'][i])
            length_std.append(results['length_std'][i])
    
    updates = np.array(updates)
    reward_mean = np.array(reward_mean)
    reward_std = np.array(reward_std)
    tcr_mean = np.array(tcr_mean)
    tcr_std = np.array(tcr_std)
    length_mean = np.array(length_mean)
    length_std = np.array(length_std)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Reward
    ax1 = axes[0]
    ax1.plot(updates, reward_mean, 'b-', linewidth=2, label='Mean Reward')
    ax1.fill_between(updates, reward_mean - reward_std, reward_mean + reward_std, 
                     alpha=0.3, color='blue', label='±1 Std')
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title('Training Curves - Bee Swarm RL', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Task Completion Rate
    ax2 = axes[1]
    ax2.plot(updates, tcr_mean, 'g-', linewidth=2, label='Mean TCR')
    ax2.fill_between(updates, tcr_mean - tcr_std, tcr_mean + tcr_std, 
                     alpha=0.3, color='green', label='±1 Std')
    ax2.set_ylabel('Task Completion Rate (%)', fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Perfect')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Episode Length
    ax3 = axes[2]
    ax3.plot(updates, length_mean, 'orange', linewidth=2, label='Mean Length')
    ax3.fill_between(updates, length_mean - length_std, length_mean + length_std, 
                     alpha=0.3, color='orange', label='±1 Std')
    ax3.set_xlabel('Training Updates', fontsize=12)
    ax3.set_ylabel('Episode Length (steps)', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    return fig


def print_summary_table(results):
    """Print a summary table of all checkpoints."""
    print("\n" + "="*90)
    print("TRAINING PROGRESS SUMMARY")
    print("="*90)
    print(f"{'Checkpoint':<12} {'Update':>8} {'Reward':>12} {'TCR%':>10} {'DSR%':>10} {'Steps':>10}")
    print("-"*90)
    
    for i, name in enumerate(results['checkpoint_names']):
        update = results['updates'][i]
        reward = f"{results['reward_mean'][i]:.1f}±{results['reward_std'][i]:.1f}"
        tcr = f"{results['tcr_mean'][i]:.1f}±{results['tcr_std'][i]:.1f}"
        dsr = f"{results['dsr_mean'][i]:.1f}±{results['dsr_std'][i]:.1f}"
        length = f"{results['length_mean'][i]:.0f}±{results['length_std'][i]:.0f}"
        
        print(f"{name:<12} {update:>8} {reward:>12} {tcr:>10} {dsr:>10} {length:>10}")
    
    print("="*90)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training curves for Bee Swarm RL')
    parser.add_argument('--episodes', type=int, default=20, 
                        help='Episodes per checkpoint for evaluation')
    parser.add_argument('--save-data', type=str, default='training_curves.json',
                        help='Path to save JSON results')
    parser.add_argument('--save-plot', type=str, default='training_curves.png',
                        help='Path to save plot')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting (useful for headless servers)')
    args = parser.parse_args()
    
    # Generate curves
    results = generate_training_curves(
        num_episodes=args.episodes,
        save_path=args.save_data
    )
    
    # Print summary
    print_summary_table(results)
    
    # Plot if not disabled
    if not args.no_plot:
        try:
            plot_training_curves(results, save_path=args.save_plot)
        except Exception as e:
            print(f"Could not plot (display issue?): {e}")
            print("Data saved to JSON - you can plot later.")
