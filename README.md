# Bee RL — Orbital Foraging with Battery-Based Task Reassignment

A multi-agent reinforcement learning environment where robot bees perform coordinated pollen harvesting on orbital paths. Features battery-driven task reassignment, time-constrained windows, and decentralized pool claiming for fault-tolerant multi-agent coordination.

## Features

### **Core Systems**

- **Battery-Based Task Reassignment**: 
  - Bees have limited battery capacity (randomized per episode)
  - **Battery=0 triggers automatic task release**: All assigned flowers return to pool
  - After recharging, bees can claim tasks from the unassigned pool
  - **Curriculum learning**: 30% of episodes use low-battery mode (1/3 capacity, 2-3x drain) to force retasking
  
- **Strict Assignment + Pool Claiming**:
  - Initial round-robin assignment by pollen amount
  - Bees can only harvest: their assigned flowers OR unassigned pool flowers
  - One bee per flower (strict exclusion)
  - Decentralized claiming via retask board (closest bee wins)

- **Time Window System**:
  - **HARD windows**: One-time only, severe penalty if missed (-50 reward)
  - **SOFT windows**: Repeating opportunities
  - **NONE**: Always available
  - 12-feature flower observations include time-to-window and harvestability

- **Individual Truncation**:
  - Bees truncate when completing all assignments
  - Truncated bees can resume by claiming pool tasks
  - Episode only ends when: all flowers done OR (all bees truncated AND no queue) OR max steps

- **Orbital Motion**: Bees move on Keplerian elliptical orbits; optional SGP4 real-satellite propagation.

- **Multi-Agent Coordination**: PPO with transformer attention over flowers, centralized critic for value estimation.

## Quick Start

### Installation

```bash
cd /mnt/w/Bee/bee1/test
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # or pip install torch numpy gymnasium pettingzoo pyyaml toml matplotlib pygame
```

### Train a Model

```bash
python train_model_policy.py --config config.yaml --device cpu
```

Training includes:
- **Progress bar** with live metrics (V(s0), mean/last/best rewards, episode length)
- **Episode summaries** showing battery deaths and completion status
- **Mixed episode types**: 70% normal, 30% low-battery for retasking curriculum
- **Automatic checkpointing**: Saves best and final models

### Visualize with Enhanced HUD

```bash
# 3D Matplotlib with diagnostic HUD panel
python bee_orbits_3d.py --policy --model_tag best --stochastic
```

The HUD displays:
- **Bee Status**: Battery %, load, assignments, harvested count, status (DEAD, CHARGING, TRUNCATED)
- **Flower Summary**: Total/done/pending/queue counts
- **Unharvested Flowers Detail**: Position, assigned bee, time window status, and diagnostic reasons:
  - `UNASSIGNED`: In pool, no bee claimed it
  - `BEE-DEAD`: Assigned bee has zero battery
  - `BEE-CHARGE`: Assigned bee is recharging
  - `FAR(dist)`: Bee too far from flower
  - `WIN-Xs`: Waiting X seconds for time window
  - `WIN-CLOSED`: Window permanently closed
  - `FULL`: Bee capacity exceeded
- **Battery Summary**: Episode type (NORMAL/LOW-BATTERY), dead/low/ok counts

### Interactive Pygame Visualization

```bash
python run_app.py
```

### Run Tests

```bash
python3 smoke_test_vfl.py
```

## Configuration

Edit `config.yaml` to tune parameters:

```yaml
env:
  num_bees: 5
  num_flowers: 12
  grid_size: 30
  max_steps: 800
  
  # Retask board (pool claiming)
  retask_board_size: 3         # Top-N unassigned flowers exposed to actors
  retask_timeout_steps: 30     # Unused (legacy VFL parameter)
  count_idle_as_silent: false  # Unused (legacy VFL parameter)
  
  # Battery system
  battery_min_steps: 250       # Min battery capacity (normal episodes)
  battery_max_steps: 450       # Max battery capacity (normal episodes)
  recharge_steps: 30           # Steps to fully recharge
  drain_per_step: 1.0          # Battery drain per step
  
  # Time windows
  time_window_min: 50          # Min window duration
  time_window_max: 200         # Max window duration
  
  # Capacity & rewards
  bee_capacity: 10.0           # Max pollen load per bee
  shaping_weight: 0.1          # Reward shaping factor
  anti_spam_pen: 0.1           # Penalty for repeated invalid actions

training:
  run_name: "bee_run_fixed"
  seed: 42
  episodes: 1000
  rollout_len: 256
  updates: 100                 # Training iterations
  lr_actor: 1.0e-4
  lr_critic: 2.0e-4
  
output:
  path: "output_fixed"         # Model checkpoint directory
```

### Key Parameters

- **Low-Battery Episodes**: Automatically randomized (30% probability)
  - Battery capacity: 1/3 of normal (50-150 steps)
  - Drain rate: 2-3x normal
  - Forces battery deaths for retasking curriculum

- **Flower Observations**: 12 features per flower
  ```
  x, y, pollen, harvested, assigned_to_me, busy, 
  reachable, fits_capacity, distance_3d, 
  is_harvestable_now, is_hard_window, time_to_window
  ```

- **Global State**: 235 values
  - All bee positions, batteries, loads, modes
  - All flower states, assignments, windows
  - Retask board (unassigned flower queue)

## Architecture

### Local Actor (per-bee)

**Transformer-based attention over flowers**

- **Inputs**:
  - Own position (fx, fy, fz), mode, load, battery, step count
  - Neighbor consensus (average actions)
  - **12-feature flower observations** (x, y, pollen, harvested, assigned_to_me, busy, reachable, fits_capacity, distance_3d, is_harvestable_now, is_hard_window, time_to_window)
  - Retask board (top-M unassigned flowers with priority scores)
  
- **Architecture**:
  - Trunk MLP processes bee state + consensus
  - 2-layer transformer encoder with 4 attention heads
  - Mean pooling over flower tokens
  - Separate heads for actions and claims
  
- **Outputs**:
  - **Action logits** (3 actions): DONOTHING, HARVEST, GROOM
  - **Claim logits** (M+1 options): Retask board slots 0..M-1, or no-claim
  - **Hard action masking**: Invalid actions get -inf logits (e.g., can't harvest if battery=0)

### Centralized Critic

- Observes **235-value global state** (all bees + all flowers + retask board)
- Estimates state value V(s) for PPO advantage computation
- 2-layer MLP (hidden_dim=256 by default)

### Battery-Based Retasking Flow

**Each step:**
1. Bees drain battery (1.0 per step)
2. If battery=0:
   - Bee stops working and broadcasting
   - **All assigned flowers released to pool** (assigned_bee=None)
   - Bee enters recharge mode (30 steps)
3. After recharge:
   - Bee picks 1 flower from pool if available
   - Can claim more via retask board
4. Truncated bees automatically attempt to claim pool tasks

**Conflict resolution**: Closest bee (3D orbit distance) wins flower claim

### Centralized Critic

- Observes **235-value global state** (all bees + all flowers + retask board)
- Estimates state value V(s) for PPO advantage computation
- 2-layer MLP (hidden_dim=256 by default)

### Battery-Based Retasking Flow

**Each step:**
1. Bees drain battery (1.0 per step)
2. If battery=0:
   - Bee stops working and broadcasting
   - **All assigned flowers released to pool** (assigned_bee=None)
   - Bee enters recharge mode (30 steps)
3. After recharge:
   - Bee picks 1 flower from pool if available
   - Can claim more via retask board
4. Truncated bees automatically attempt to claim pool tasks

**Conflict resolution**: Closest bee (3D orbit distance) wins flower claim

## Training Results

### Episode Completion
- **Success rate**: Near 100% (12/12 flowers harvested)
- **Episode length**: 
  - Normal episodes: ~200-300 steps
  - Low-battery episodes: ~450-480 steps (due to recharge delays)
- **Battery deaths**: 0-5 per episode in low-battery mode
- **Retasking**: Bees successfully claim pool tasks after battery recovery

### Observable Behaviors
- Bees prioritize nearby flowers
- Strategic grooming when load full
- Respects time windows (waits for SOFT, abandons closed HARD)
- Automatic task claiming from pool when truncated
- Coordinated retasking when batteries die

### Training Metrics
```
Best mean reward: ~80-100 (varies by config)
Training time: ~30-60 minutes on CPU (100 updates)
Convergence: Typically within 50-100 updates
```

## Security Notes

**Model Loading**

This project uses `torch.load()` for checkpoints. `torch.load()` deserializes pickled objects and **can execute arbitrary code**.

- Only load checkpoints from trusted sources.
- For production, consider using `torch.load(..., weights_only=True)` (PyTorch ≥1.13).
- Or save/load only `model.state_dict()` and recreate the model class.

## File Organization

```
.
├── bee_state.py                 # Bee and Flower classes (Keplerian + SGP4 orbit)
├── bee_policy.py                # Actor (transformer attention) and Critic networks
├── bees_env.py                  # Multi-agent environment with battery, windows, retasking
├── train_model_policy.py        # PPO training loop with progress bar
=/                    # Generated episode data (TOML format)
├── logs/                        # Training logs and HUD snapshots
└── README.md                    # This file
```

## Performance Tips

1. **Battery curriculum**: The 30% low-battery episodes ensure retasking behavior is learned
2. **Reduce grid size** or **flower count** for faster iteration during development
3. **Disable SGP4** if not needed (`use_sgp4: false` in config)
4. **Use GPU**: set `--device cuda` for faster training (if available)
5. **Adjust reward shaping** (`shaping_weight`, `anti_spam_pen`) for convergence
6. **Tune retask board size**: 
   - Smaller (M=1-3): Faster, less flexibility
   - Larger (M=5-10): More options, higher compute cost
7. **Monitor HUD**: Use visualization to diagnose why flowers aren't harvested

## Debugging Tips

### Flowers Not Harvesting?
Use the 3D visualization HUD to diagnose:
- **UNASSIGNED**: No bee claimed it (check retask board)
- **BEE-DEAD**: Battery=0 (wait for recharge + pool claiming)
- **BEE-CHARGE**: Bee recharging (wait 30 steps)
- **FAR(X.X)**: Distance > harvest_radius (bee needs to move closer)
- **WIN-Xs**: Time window not open (wait for SOFT window or check HARD deadlines)
- **FULL**: Bee capacity exceeded (need strategic groom)

### Training Not Converging?
- Check episode completion rate in training output
- Increase `rollout_len` for more samples per update
- Reduce `lr_actor` and `lr_critic` if loss oscillates
- Verify low-battery episodes are triggering (look for battery death messages)
- Ensure flowers have reasonable window sizes (`time_window_min`/`max`)

## Model Checkpoint Management

Large model checkpoints (`.pt` files) should not be committed to git:

```bash
# Add to .gitignore
echo "models/*.pt" >> .gitignore
echo "!models/actor_best.pt" >> .gitignore

# Or use DVC / external storage
dvc add models/
git add models/.gitignore models.dvc
```

Keep only the best model (e.g., `actor_best.pt`, `critic_best.pt`) in version control, or store in a model registry (S3, HuggingFace Hub).

## References

- **Battery-Based Task Reassignment**: Fault-tolerant multi-agent coordination through energy constraints
- **Transformer Attention**: Vaswani et al., "Attention Is All You Need" (2017)
- **PPO**: Schulman et al., "Proximal Policy Optimization" (2017)
- **PettingZoo**: Multi-agent parallel environment API
- **SGP4**: Simplified General Perturbations (TLE-based satellite propagation)
- **Gymnasium**: Standard RL environment API

## What Makes This Unique

1. **Battery-driven curriculum learning**: Automatic randomization of battery constraints
2. **Individual truncation with resume**: Bees can stop/restart independently
3. **Decentralized pool claiming**: No centralized task allocator
4. **Time-constrained harvesting**: HARD/SOFT/NONE window semantics
5. **Diagnostic HUD**: Real-time explanation of system state
6. **Fault tolerance**: System recovers from battery failures through retasking

## License

MIT

## Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Update `config.yaml` defaults if adding hyperparameters
3. Document battery/window behavior in docstrings
4. Test with both normal and low-battery episodes

---

**Last updated**: November 16, 2025
