#!/usr/bin/env python3
"""
Mission Generator - Creates random test missions for the bee swarm.

Each mission is an "episode" with:
- Random flower spawn points, priorities, and time windows
- Random bee starting positions (orbital parameters)
- Optional failure events (probability-based)
- Mission metadata (difficulty, expected steps, etc.)

Missions are saved to JSON and loaded by mission_runner.py

Usage:
    # Generate 10 missions with 30% failure chance
    python3 mission_generator.py --count 10 --failure-chance 0.3 --output missions/

    # Generate specific difficulty missions
    python3 mission_generator.py --count 5 --difficulty hard --output missions/hard/
"""

import argparse
import json
import os
import random
from datetime import datetime
from typing import Optional


def generate_flower(
    flower_id: int,
    grid_size: float,
    max_steps: int,
    difficulty: str,
    rng: random.Random,
) -> dict:
    """Generate a single flower configuration."""
    
    # Random position
    x = rng.uniform(0, grid_size)
    y = rng.uniform(0, grid_size)
    
    # Random pollen based on difficulty
    if difficulty == "easy":
        pollen = rng.uniform(3.0, 8.0)
    elif difficulty == "hard":
        pollen = rng.uniform(5.0, 10.0)  # Higher value = more important
    else:
        pollen = rng.uniform(1.0, 10.0)
    
    # Random priority (0.0 - 1.0)
    priority = rng.random()
    
    # Window type distribution based on difficulty
    if difficulty == "easy":
        weights = [0.1, 0.3, 0.6]  # 10% HARD, 30% SOFT, 60% NONE
    elif difficulty == "hard":
        weights = [0.5, 0.3, 0.2]  # 50% HARD, 30% SOFT, 20% NONE
    else:
        weights = [0.3, 0.4, 0.3]  # Balanced
    
    window_type = rng.choices(["HARD", "SOFT", "NONE"], weights=weights)[0]
    
    # Generate time window
    if window_type == "NONE":
        window_start = 0
        window_end = max_steps
    else:
        if difficulty == "hard":
            # Shorter, tighter windows
            duration = rng.randint(30, 80)
        elif difficulty == "easy":
            # Longer, more forgiving windows
            duration = rng.randint(100, 250)
        else:
            duration = rng.randint(50, 150)
        
        window_start = rng.randint(0, max(1, max_steps - duration - 20))
        window_end = window_start + duration
    
    return {
        "id": flower_id,
        "x": round(x, 2),
        "y": round(y, 2),
        "pollen": round(pollen, 2),
        "priority": round(priority, 3),
        "window_type": window_type,
        "window_start": window_start,
        "window_end": window_end,
    }


def generate_bee(
    bee_id: int,
    grid_size: float,
    rng: random.Random,
) -> dict:
    """Generate random orbital parameters for a bee."""
    
    # Orbital parameters
    semi_major = rng.uniform(grid_size * 0.25, grid_size * 0.5)
    eccentricity = rng.uniform(0.0, 0.25)
    inclination_deg = rng.uniform(5, 45)
    omega_deg = rng.uniform(0, 360)
    Omega_deg = rng.uniform(0, 360)
    true_anomaly_deg = rng.uniform(0, 360)
    
    # Battery capacity variation
    battery_capacity = rng.randint(300, 500)
    
    return {
        "id": bee_id,
        "semi_major": round(semi_major, 2),
        "eccentricity": round(eccentricity, 3),
        "inclination_deg": round(inclination_deg, 1),
        "omega_deg": round(omega_deg, 1),
        "Omega_deg": round(Omega_deg, 1),
        "true_anomaly_deg": round(true_anomaly_deg, 1),
        "battery_capacity": battery_capacity,
    }


def generate_failure_event(
    num_bees: int,
    max_steps: int,
    rng: random.Random,
) -> Optional[dict]:
    """Generate a failure event (bee malfunction)."""
    
    bee_id = rng.randint(0, num_bees - 1)
    
    # Failure happens somewhere in the middle of the mission
    failure_step = rng.randint(max_steps // 5, max_steps * 4 // 5)
    
    # Failure types
    failure_type = rng.choice([
        "battery_drain",      # Battery depletes instantly
        "communication_loss", # Can't broadcast for N steps
        "motor_failure",      # Can't move for N steps
    ])
    
    # Duration for recoverable failures
    if failure_type == "battery_drain":
        duration = 30  # Recharge time
        recoverable = True
    elif failure_type == "communication_loss":
        duration = rng.randint(20, 50)
        recoverable = True
    else:  # motor_failure
        duration = rng.randint(30, 80)
        recoverable = rng.random() < 0.7  # 70% chance recoverable
    
    return {
        "bee_id": bee_id,
        "step": failure_step,
        "type": failure_type,
        "duration": duration,
        "recoverable": recoverable,
    }


def generate_mission(
    mission_id: str,
    num_bees: int = 5,
    num_flowers: int = 12,
    grid_size: float = 30.0,
    max_steps: int = 400,
    difficulty: str = "normal",
    failure_chance: float = 0.3,
    max_failures: int = 2,
    seed: Optional[int] = None,
) -> dict:
    """Generate a complete mission configuration."""
    
    rng = random.Random(seed)
    
    # Generate flowers
    flowers = [
        generate_flower(i, grid_size, max_steps, difficulty, rng)
        for i in range(num_flowers)
    ]
    
    # Generate bees
    bees = [
        generate_bee(i, grid_size, rng)
        for i in range(num_bees)
    ]
    
    # Generate failure events (probabilistic)
    failures = []
    num_failures = 0
    while num_failures < max_failures and rng.random() < failure_chance:
        failure = generate_failure_event(num_bees, max_steps, rng)
        # Avoid multiple failures for same bee
        if not any(f["bee_id"] == failure["bee_id"] for f in failures):
            failures.append(failure)
            num_failures += 1
    
    # Calculate mission stats
    hard_windows = sum(1 for f in flowers if f["window_type"] == "HARD")
    soft_windows = sum(1 for f in flowers if f["window_type"] == "SOFT")
    total_pollen = sum(f["pollen"] for f in flowers)
    avg_priority = sum(f["priority"] for f in flowers) / num_flowers
    
    # Estimate difficulty score
    difficulty_score = (
        hard_windows * 2 +
        soft_windows * 1 +
        len(failures) * 3 +
        (1 if difficulty == "hard" else 0) * 2
    )
    
    return {
        "mission_id": mission_id,
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "config": {
            "num_bees": num_bees,
            "num_flowers": num_flowers,
            "grid_size": grid_size,
            "max_steps": max_steps,
            "difficulty": difficulty,
        },
        "flowers": flowers,
        "bees": bees,
        "failures": failures,
        "stats": {
            "hard_windows": hard_windows,
            "soft_windows": soft_windows,
            "total_pollen": round(total_pollen, 2),
            "avg_priority": round(avg_priority, 3),
            "num_failures": len(failures),
            "difficulty_score": difficulty_score,
        },
    }


def generate_mission_batch(
    count: int,
    output_dir: str,
    num_bees: int = 5,
    num_flowers: int = 12,
    grid_size: float = 30.0,
    max_steps: int = 400,
    difficulty: str = "normal",
    failure_chance: float = 0.3,
    max_failures: int = 2,
    base_seed: Optional[int] = None,
) -> list[str]:
    """Generate multiple missions and save to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    mission_files = []
    
    for i in range(count):
        # Generate unique ID
        mission_id = f"mission_{datetime.now().strftime('%Y%m%d')}_{i+1:03d}"
        
        # Seed for reproducibility
        seed = base_seed + i if base_seed else None
        
        mission = generate_mission(
            mission_id=mission_id,
            num_bees=num_bees,
            num_flowers=num_flowers,
            grid_size=grid_size,
            max_steps=max_steps,
            difficulty=difficulty,
            failure_chance=failure_chance,
            max_failures=max_failures,
            seed=seed,
        )
        
        # Save to file
        filename = f"{mission_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(mission, f, indent=2)
        
        mission_files.append(filepath)
        
        # Summary
        failures_str = f"{len(mission['failures'])} failures" if mission['failures'] else "no failures"
        print(f"  [{i+1}/{count}] {mission_id}: {mission['stats']['hard_windows']}H/{mission['stats']['soft_windows']}S windows, {failures_str}")
    
    return mission_files


def main():
    parser = argparse.ArgumentParser(description="Generate test missions for bee swarm")
    parser.add_argument("--count", type=int, default=5, help="Number of missions to generate")
    parser.add_argument("--output", "-o", type=str, default="missions/", help="Output directory")
    parser.add_argument("--bees", type=int, default=5, help="Number of bees")
    parser.add_argument("--flowers", type=int, default=12, help="Number of flowers")
    parser.add_argument("--steps", type=int, default=400, help="Max steps per mission")
    parser.add_argument("--difficulty", choices=["easy", "normal", "hard"], default="normal")
    parser.add_argument("--failure-chance", type=float, default=0.3, 
                        help="Probability of failure events (0.0-1.0)")
    parser.add_argument("--max-failures", type=int, default=2, help="Max failures per mission")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MISSION GENERATOR")
    print("=" * 60)
    print(f"Generating {args.count} missions...")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Failure chance: {args.failure_chance*100:.0f}%")
    print(f"  Output: {args.output}")
    print()
    
    files = generate_mission_batch(
        count=args.count,
        output_dir=args.output,
        num_bees=args.bees,
        num_flowers=args.flowers,
        max_steps=args.steps,
        difficulty=args.difficulty,
        failure_chance=args.failure_chance,
        max_failures=args.max_failures,
        base_seed=args.seed,
    )
    
    print(f"\nGenerated {len(files)} mission files in {args.output}")
    print("\nTo run a mission:")
    print(f"  python3 mission_runner.py {files[0]}")


if __name__ == "__main__":
    main()
