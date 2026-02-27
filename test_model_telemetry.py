"""
Test trained model performance against real telemetry data.
Evaluates how well the RL model handles:
1. Real satellite positions and states
2. Failed satellite scenarios
3. Task reassignment from failed satellites

Usage:
    python test_model_telemetry.py --compare --episodes 10
    python test_model_telemetry.py --model outputs/best_actor.pt --episodes 20
"""

import os
import json
import numpy as np
import torch
from datetime import datetime
from typing import List, Optional

from bees_env import BeeForagingEnv
from bee_policy import Actor
from train_orbital_v2 import obs_to_tensor


class TelemetryModelTester:
    """Test trained RL models against real telemetry data."""
    
    def __init__(
        self,
        telemetry_path: str = "telemetrybridge.json",
        model_path: Optional[str] = None,
        num_episodes: int = 10,
        max_steps_per_episode: int = 500,
        verbose: bool = True,
    ):
        self.telemetry_path = telemetry_path
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.max_steps = max_steps_per_episode
        self.verbose = verbose
        
        # Model state
        self.model = None
        self.model_type = "random"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Metrics storage
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.tasks_completed: List[int] = []
        self.tasks_failed: List[int] = []
        self.reassignment_success: List[int] = []
        self.battery_deaths: List[int] = []
        
        # Metadata
        self.metadata: dict = {}
        
    def create_env(self) -> BeeForagingEnv:
        """Create environment and load telemetry."""
        env = BeeForagingEnv(verbose=False)
        env.reset()
        self.metadata = env.load_from_telemetry(self.telemetry_path)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("ENVIRONMENT LOADED FROM TELEMETRY")
            print(f"{'='*60}")
            print(f"  Satellites (bees): {env.num_bees}")
            print(f"  Tasks (flowers): {env.num_flowers}")
            print(f"  Active satellites: {self.metadata.get('num_active_bees', 0)}")
            print(f"  Failed satellites: {self.metadata.get('failed_satellites', [])}")
            print(f"  Tasks reassigned: {self.metadata.get('total_tasks_moved', 0)}")
            print(f"  Unassigned tasks: {self.metadata.get('num_unassigned_flowers', 0)}")
            print(f"{'='*60}\n")
        
        return env
    
    def load_model(self, model_path):
        """Load the trained model from the specified path"""
        # Infer model shape from checkpoint
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        # Infer num_bees from consensus layer weight shape
        num_bees = state_dict.get('cons_fc.0.weight', torch.zeros(64, 5)).shape[1]
        # Infer num_flowers from flower transformer input
        num_flowers = state_dict.get('flowers_fc.0.weight', torch.zeros(128, 12)).shape[0] // 128 * 12
        # Check for retask board
        has_retask = 'retask_board_fc.0.weight' in state_dict
        retask_size = 0
        if has_retask:
            retask_input = state_dict['retask_board_fc.0.weight'].shape[1]
            retask_size = retask_input // 5
        # Infer hidden_dim from trunk
        hidden_dim = state_dict.get('trunk.0.weight', torch.zeros(256, 480)).shape[0]

        self.model = Actor(
            num_bees=num_bees,
            num_flowers=num_flowers // 12 if num_flowers > 12 else num_flowers,
            hidden_dim=hidden_dim,
            retask_board_size=retask_size,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.model_type = "pytorch"
    
    def get_actions(self, env: BeeForagingEnv, obs: dict) -> dict:
        """Get actions from model or random policy."""
        actions = {}
        
        if self.model is not None and self.model_type == "pytorch":
            # PyTorch actor model
            try:
                with torch.no_grad():
                    for agent_name, agent_obs in obs.items():
                        # Flatten observation if it's a dict
                        if isinstance(agent_obs, dict):
                            flat_obs = []
                            for key in sorted(agent_obs.keys()):
                                val = agent_obs[key]
                                if isinstance(val, np.ndarray):
                                    flat_obs.extend(val.flatten().tolist())
                                elif isinstance(val, (int, float)):
                                    flat_obs.append(float(val))
                            obs_tensor = torch.FloatTensor(flat_obs).unsqueeze(0).to(self.device)
                        else:
                            obs_tensor = torch.FloatTensor(agent_obs).unsqueeze(0).to(self.device)
                        
                        # Convert per-agent observation dict to Actor input
                        if isinstance(agent_obs, dict):
                            obs_tensor = obs_to_tensor(agent_obs, self.device)
                        
                        # Get action logits from model
                        action_logits = self.model(obs_tensor)
                        action = torch.argmax(action_logits, dim=-1).squeeze(0).cpu().item()
                        
                        actions[agent_name] = action
            except Exception as e:
                if self.verbose:
                    print(f"Model inference error: {e}, using random")
                actions = {f"bee_{i}": np.random.randint(0, 3) 
                          for i in range(env.num_bees)}
        else:
            # Random policy
            actions = {f"bee_{i}": np.random.randint(0, 3) 
                      for i in range(env.num_bees)}
        
        return actions
    
    def run_episode(self, env: BeeForagingEnv, episode_num: int) -> dict:
        """Run a single episode and collect metrics."""
        # Load telemetry directly (skip reset which creates default bees)
        try:
            # Create fresh env and load telemetry
            env = BeeForagingEnv(verbose=False)
            # Don't call reset - go straight to telemetry load
            env.agents = []
            env.bees = []
            env.flowers = []
            env.steps = 0
            env.last_actions = {}
            self.metadata = env.load_from_telemetry(self.telemetry_path)
        except Exception as e:
            if self.verbose:
                print(f"  Load error: {e}")
            return {
                "episode": episode_num,
                "reward": 0,
                "steps": 0,
                "tasks_completed": 0,
                "tasks_total": 0,
                "tasks_expired": 0,
                "battery_deaths": 0,
                "reassigned_completed": 0,
                "terminated": True,
                "truncated": False,
            }
        
        obs = env._get_observations()
        
        episode_reward = 0.0
        steps = 0
        
        # Track initial state
        initial_unharvested = sum(1 for f in env.flowers if not f.harvested)
        failed_bee_ids = set()
        for sat_id in self.metadata.get("failed_satellites", []):
            bee_id = self.metadata.get("sat_id_map", {}).get(sat_id, -1)
            if bee_id >= 0:
                failed_bee_ids.add(bee_id)
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < self.max_steps:
            # Get actions
            actions = self.get_actions(env, obs)
            
            # Step environment
            try:
                result = env.step(actions)
                if len(result) == 6:
                    obs, rewards, terms, truncs, infos, _ = result
                else:
                    obs, rewards, terms, truncs, infos = result
            except Exception as e:
                if self.verbose:
                    print(f"  Step error: {e}")
                break
            
            # Aggregate rewards
            if isinstance(rewards, dict):
                step_reward = sum(rewards.values())
            else:
                step_reward = float(rewards)
            
            episode_reward += step_reward
            steps += 1
            
            # Check termination
            if isinstance(terms, dict):
                terminated = terms.get("__all__", all(terms.values()))
            else:
                terminated = bool(terms)
            
            if isinstance(truncs, dict):
                truncated = truncs.get("__all__", all(truncs.values()))
            else:
                truncated = bool(truncs)
        
        # Calculate final metrics
        final_unharvested = sum(1 for f in env.flowers if not f.harvested)
        tasks_completed = initial_unharvested - final_unharvested
        tasks_expired = sum(1 for f in env.flowers 
                          if not f.harvested and getattr(f, 'window_missed', False))
        
        # Battery deaths (excluding initially failed)
        battery_deaths = sum(1 for i, b in enumerate(env.bees) 
                            if b.terminated and i not in failed_bee_ids)
        
        # Reassignment tracking
        reassigned_completed = 0
        for f in env.flowers:
            if f.harvested and f.assigned_bee is not None:
                if f.assigned_bee in failed_bee_ids:
                    reassigned_completed += 1
        
        return {
            "episode": episode_num,
            "reward": episode_reward,
            "steps": steps,
            "tasks_completed": tasks_completed,
            "tasks_total": env.num_flowers,
            "tasks_expired": tasks_expired,
            "battery_deaths": battery_deaths,
            "reassigned_completed": reassigned_completed,
            "terminated": terminated,
            "truncated": truncated,
        }
    
    def run_evaluation(self) -> dict:
        """Run full evaluation across multiple episodes."""
        if self.verbose:
            print("\n" + "="*60)
            print("TELEMETRY MODEL EVALUATION")
            print("="*60)
        
        # Create environment
        env = self.create_env()
        
        # Load model
        self.load_model(env)
        
        if self.verbose:
            print(f"\nRunning {self.num_episodes} episodes with {self.model_type} policy...")
            print("-"*60)
        
        all_results = []
        
        for ep in range(self.num_episodes):
            result = self.run_episode(env, ep + 1)
            all_results.append(result)
            
            # Store metrics
            self.episode_rewards.append(result["reward"])
            self.episode_lengths.append(result["steps"])
            self.tasks_completed.append(result["tasks_completed"])
            self.tasks_failed.append(result["tasks_expired"])
            self.battery_deaths.append(result["battery_deaths"])
            self.reassignment_success.append(result["reassigned_completed"])
            
            # Progress output
            if self.verbose:
                print(f"Episode {ep+1:3d}: "
                      f"Reward={result['reward']:8.2f}, "
                      f"Steps={result['steps']:4d}, "
                      f"Tasks={result['tasks_completed']:2d}/{result['tasks_total']}, "
                      f"Expired={result['tasks_expired']:2d}")
        
        # Compute summary statistics
        summary = self._compute_summary(env)
        
        # Print summary
        if self.verbose:
            self._print_summary(summary)
        
        # Save results
        self._save_results(all_results, summary)
        
        return summary
    
    def _compute_summary(self, env: BeeForagingEnv) -> dict:
        """Compute summary statistics."""
        def stats(arr):
            arr = np.array(arr)
            if len(arr) == 0:
                return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
            }
        
        total_tasks = env.num_flowers
        total_bees = env.num_bees
        
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "num_episodes": self.num_episodes,
            "telemetry_file": self.telemetry_path,
            "failed_satellites": self.metadata.get("failed_satellites", []),
            "total_satellites": total_bees,
            "total_tasks": total_tasks,
            "metrics": {
                "episode_reward": stats(self.episode_rewards),
                "episode_length": stats(self.episode_lengths),
                "tasks_completed": stats(self.tasks_completed),
                "tasks_expired": stats(self.tasks_failed),
                "battery_deaths": stats(self.battery_deaths),
                "reassignment_success": stats(self.reassignment_success),
            },
            "task_completion_rate": np.mean(self.tasks_completed) / max(1, total_tasks),
            "survival_rate": 1.0 - np.mean(self.battery_deaths) / max(1, total_bees - len(self.metadata.get("failed_satellites", []))),
        }
    
    def _print_summary(self, summary: dict):
        """Print formatted summary."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\nPolicy: {summary['model_type']}")
        if summary['model_path']:
            print(f"Model: {summary['model_path']}")
        print(f"Episodes: {summary['num_episodes']}")
        print(f"Satellites: {summary['total_satellites']} ({len(summary['failed_satellites'])} failed)")
        print(f"Tasks: {summary['total_tasks']}")
        
        print("\n--- Performance Metrics ---")
        metrics = summary["metrics"]
        
        print(f"\nReward:")
        print(f"  Mean: {metrics['episode_reward']['mean']:,.2f} ± {metrics['episode_reward']['std']:.2f}")
        print(f"  Range: [{metrics['episode_reward']['min']:.2f}, {metrics['episode_reward']['max']:.2f}]")
        
        print(f"\nTasks Completed:")
        print(f"  Mean: {metrics['tasks_completed']['mean']:.1f} ± {metrics['tasks_completed']['std']:.1f}")
        print(f"  Completion Rate: {summary['task_completion_rate']*100:.1f}%")
        
        print(f"\nTasks Expired (Deadline Missed):")
        print(f"  Mean: {metrics['tasks_expired']['mean']:.1f} ± {metrics['tasks_expired']['std']:.1f}")
        
        print(f"\nBattery Deaths (excluding initial failures):")
        print(f"  Mean: {metrics['battery_deaths']['mean']:.1f}")
        print(f"  Survival Rate: {summary['survival_rate']*100:.1f}%")
        
        print(f"\nReassigned Task Completion:")
        print(f"  Mean: {metrics['reassignment_success']['mean']:.1f}")
        
        print("\n" + "="*60)
    
    def _save_results(self, all_results: List[dict], summary: dict):
        """Save results to JSON file."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "episodes": all_results,
        }
        
        output_path = "telemetry_evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        if self.verbose:
            print(f"\nResults saved to: {output_path}")


class BaselineComparison:
    """Compare different policies against telemetry scenario."""
    
    def __init__(self, telemetry_path: str = "telemetrybridge.json"):
        self.telemetry_path = telemetry_path
        self.results = {}
        self.metadata = {}
    
    def run_random_baseline(self, num_episodes: int = 10) -> dict:
        """Run random policy baseline."""
        print("\n" + "="*60)
        print(">>> RANDOM BASELINE")
        print("="*60)
        
        tester = TelemetryModelTester(
            telemetry_path=self.telemetry_path,
            model_path=None,
            num_episodes=num_episodes,
            verbose=True,
        )
        self.results["random"] = tester.run_evaluation()
        self.metadata = tester.metadata
        return self.results["random"]
    
    def run_greedy_baseline(self, num_episodes: int = 10) -> dict:
        """Run greedy harvest-when-possible policy."""
        print("\n" + "="*60)
        print(">>> GREEDY BASELINE (Always Harvest)")
        print("="*60)
        
        env = BeeForagingEnv(verbose=False)
        
        rewards = []
        tasks_done = []
        steps_list = []
        
        print(f"\nRunning {num_episodes} episodes...")
        print("-"*60)
        
        for ep in range(num_episodes):
            try:
                env.reset()
                self.metadata = env.load_from_telemetry(self.telemetry_path)
            except Exception as e:
                print(f"  Reset error: {e}")
                continue
            
            ep_reward = 0
            steps = 0
            
            for step in range(500):
                # Greedy: always try to harvest (action=1)
                actions = {f"bee_{i}": 1 for i in range(env.num_bees)}
                
                try:
                    result = env.step(actions)
                    if len(result) == 6:
                        obs, rews, terms, truncs, _, _ = result
                    else:
                        obs, rews, terms, truncs, _ = result
                    
                    if isinstance(rews, dict):
                        ep_reward += sum(rews.values())
                    else:
                        ep_reward += float(rews)
                    
                    steps += 1
                    
                    if isinstance(terms, dict):
                        done = terms.get("__all__", all(terms.values()))
                    else:
                        done = bool(terms)
                    
                    if done:
                        break
                except Exception as e:
                    print(f"  Error: {e}")
                    break
            
            completed = sum(1 for f in env.flowers if f.harvested)
            rewards.append(ep_reward)
            tasks_done.append(completed)
            steps_list.append(steps)
            
            print(f"Episode {ep+1:3d}: Reward={ep_reward:8.2f}, Steps={steps:4d}, Tasks={completed:2d}/{env.num_flowers}")
        
        self.results["greedy"] = {
            "model_type": "greedy",
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_tasks": float(np.mean(tasks_done)),
            "std_tasks": float(np.std(tasks_done)),
            "mean_steps": float(np.mean(steps_list)),
            "completion_rate": float(np.mean(tasks_done)) / env.num_flowers,
        }
        
        print(f"\nGreedy Summary: Reward={np.mean(rewards):.2f}±{np.std(rewards):.2f}, "
              f"Tasks={np.mean(tasks_done):.1f}±{np.std(tasks_done):.1f}")
        
        return self.results["greedy"]
    
    def run_do_nothing_baseline(self, num_episodes: int = 10) -> dict:
        """Run do-nothing policy baseline."""
        print("\n" + "="*60)
        print(">>> DO-NOTHING BASELINE")
        print("="*60)
        
        env = BeeForagingEnv(verbose=False)
        
        rewards = []
        tasks_done = []
        
        print(f"\nRunning {num_episodes} episodes...")
        print("-"*60)
        
        for ep in range(num_episodes):
            try:
                env.reset()
                self.metadata = env.load_from_telemetry(self.telemetry_path)
            except Exception as e:
                print(f"  Reset error: {e}")
                continue
            
            ep_reward = 0
            
            for step in range(500):
                # Do nothing (action=0)
                actions = {f"bee_{i}": 0 for i in range(env.num_bees)}
                
                try:
                    result = env.step(actions)
                    if len(result) == 6:
                        obs, rews, terms, truncs, _, _ = result
                    else:
                        obs, rews, terms, truncs, _ = result
                    
                    if isinstance(rews, dict):
                        ep_reward += sum(rews.values())
                    else:
                        ep_reward += float(rews)
                    
                    if isinstance(terms, dict):
                        done = terms.get("__all__", all(terms.values()))
                    else:
                        done = bool(terms)
                    
                    if done:
                        break
                except:
                    break
            
            completed = sum(1 for f in env.flowers if f.harvested)
            rewards.append(ep_reward)
            tasks_done.append(completed)
            
            print(f"Episode {ep+1:3d}: Reward={ep_reward:8.2f}, Tasks={completed:2d}/{env.num_flowers}")
        
        self.results["do_nothing"] = {
            "model_type": "do_nothing",
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_tasks": float(np.mean(tasks_done)),
            "completion_rate": float(np.mean(tasks_done)) / env.num_flowers,
        }
        
        return self.results["do_nothing"]
    
    def run_trained_model(self, model_path: str, num_episodes: int = 10) -> dict:
        """Run trained model evaluation."""
        print("\n" + "="*60)
        print(f">>> TRAINED MODEL: {model_path}")
        print("="*60)
        
        tester = TelemetryModelTester(
            telemetry_path=self.telemetry_path,
            model_path=model_path,
            num_episodes=num_episodes,
            verbose=True,
        )
        self.results["trained"] = tester.run_evaluation()
        return self.results["trained"]
    
    def compare_all(self, model_path: Optional[str] = None, num_episodes: int = 10):
        """Run all baselines and compare."""
        print("\n" + "="*70)
        print("BASELINE COMPARISON: Real Telemetry Scenario")
        print("="*70)
        print(f"Telemetry: {self.telemetry_path}")
        print(f"Episodes per policy: {num_episodes}")
        
        # Run baselines
        self.run_do_nothing_baseline(num_episodes)
        self.run_random_baseline(num_episodes)
        self.run_greedy_baseline(num_episodes)
        
        # Run trained model if available
        if model_path is None:
            # Try to find model
            candidates = ["outputs/best_actor.pt", "models/ppo_bees.zip"]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break
        
        if model_path and os.path.exists(model_path):
            self.run_trained_model(model_path, num_episodes)
        
        # Print comparison table
        self._print_comparison()
        
        # Save comparison
        self._save_comparison()
    
    def _print_comparison(self):
        """Print comparison table."""
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"\n{'Policy':<15} {'Mean Reward':>15} {'Std':>10} {'Tasks':>10} {'Completion':>12}")
        print("-"*62)
        
        best_reward = float('-inf')
        best_policy = None
        
        for name, result in self.results.items():
            if "metrics" in result:
                reward = result["metrics"]["episode_reward"]["mean"]
                std = result["metrics"]["episode_reward"]["std"]
                tasks = result["metrics"]["tasks_completed"]["mean"]
                completion = result["task_completion_rate"] * 100
            else:
                reward = result.get("mean_reward", 0)
                std = result.get("std_reward", 0)
                tasks = result.get("mean_tasks", 0)
                completion = result.get("completion_rate", 0) * 100
            
            print(f"{name:<15} {reward:>15.2f} {std:>10.2f} {tasks:>10.1f} {completion:>11.1f}%")
            
            if reward > best_reward:
                best_reward = reward
                best_policy = name
        
        print("="*70)
        
        if best_policy:
            print(f"\nBest performing policy: {best_policy.upper()}")
    
    def _save_comparison(self):
        """Save comparison results."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "telemetry_file": self.telemetry_path,
            "results": self.results,
        }
        
        output_path = "telemetry_comparison_results.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nComparison saved to: {output_path}")


def main():
    """Main entry point for telemetry testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model against telemetry data")
    parser.add_argument("--telemetry", default="telemetrybridge.json", 
                       help="Path to telemetry JSON")
    parser.add_argument("--model", default=None, 
                       help="Path to trained model (.pt or .zip)")
    parser.add_argument("--episodes", type=int, default=10, 
                       help="Number of episodes per policy")
    parser.add_argument("--compare", action="store_true", 
                       help="Run full baseline comparison")
    parser.add_argument("--max-steps", type=int, default=500,
                       help="Max steps per episode")
    
    args = parser.parse_args()
    
    if args.compare:
        comparison = BaselineComparison(args.telemetry)
        comparison.compare_all(args.model, args.episodes)
    else:
        tester = TelemetryModelTester(
            telemetry_path=args.telemetry,
            model_path=args.model,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
        )
        tester.run_evaluation()


if __name__ == "__main__":
    main()
