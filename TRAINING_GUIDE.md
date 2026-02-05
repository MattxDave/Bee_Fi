# BEE RL — Training & 3D Visualization Guide

##  Completed Tasks

### 1. Enhanced 3D Orbital Visualization HUD
The `bee_orbits_3d.py` visualization now displays **comprehensive bee status** in the right-side HUD panel:

**New Status Displays:**
- **Battery %**: Real-time battery level (0-100%)
- **Recharge Status**: Shows "RECHARGING" when bee is charging
- **Critical Battery**: Displays "CRIT(XX%)" warning when battery < 20%
- **Silent Status**: Shows "SILENT" when bee hasn't broadcast for > retask_timeout_steps
- **Idle-Silent**: Shows "IDLE-SILENT" when bee is IDLE and count_idle_as_silent is enabled
- **Mode**: Displays bee operational mode (IDLE, HARVEST, GROOM)

**HUD Format:**
```
step=0800  (grid=30)
bee  action    battery  load   harvested  status
  0  HARVEST   89.3%    7.5/10.0   3     OK
  1  DONOTHING  45.2%    0.0/10.0   1     CRIT(45%)
  2  GROOM      22.1%    10.0/10.0  2     CRIT(22%),RECHARGING
  3  HARVEST    78.5%    5.2/10.0   4     OK
  4  IDLE       15.0%    2.5/10.0   0     SILENT,IDLE-SILENT
```

---

## Model Training

### Training Started
**Command:**
```bash
python3 train_model_policy.py --config config.yaml
```

**Configuration (config.yaml):**
- **Agents**: 5 bees
- **Flowers**: 12 flowers
- **Episodes**: 1000
- **Rollout Length**: 512 steps per collection
- **PPO Updates**: 1000
- **VFL Retask Board Size**: 3 (top-3 orphan flowers per bee)
- **Battery**: 250-450 steps capacity
- **Retask Timeout**: 30 steps (silent detection)

### Training Monitoring

**Option 1: Watch live log**
```bash
tail -f training_output.log
```

**Option 2: Interactive monitor script**
```bash
python3 monitor_training.py
```

**Option 3: Check process**
```bash
ps aux | grep train_model_policy
```

### Training Checkpoints

Checkpoints saved to `output_fixed/`:
- `actor_50.pt`, `actor_100.pt`, ... (every 50 episodes)
- `critic_50.pt`, `critic_100.pt`, ...
- `actor_best.pt`, `critic_best.pt` (best reward episode)
- `actor_final.pt`, `critic_final.pt` (final checkpoint)

### Estimated Training Time

- **Total Episodes**: 1000
- **Steps per Episode**: 512 avg
- **Total Steps**: ~512,000
- **Approx Duration**: 2-4 hours (on CPU)
  - With GPU: 30-60 minutes

---

## Visualizing Results

### View Latest Training Checkpoint
```bash
python3 bee_orbits_3d.py --policy --model_tag 150 --episodes 1
```
Replaces `150` with the latest checkpoint number.

### View Best Checkpoint
```bash
python3 bee_orbits_3d.py --policy --model_tag best --episodes 1
```

### View Final Checkpoint
```bash
python3 bee_orbits_3d.py --policy --model_tag final --episodes 1
```

### Customize Visualization
```bash
python3 bee_orbits_3d.py --policy --model_tag best \
  --episodes 3 \
  --fps 60 \
  --trail_len 100 \
  --orbit_alpha 0.8 \
  --save bee_episode.mp4
```

**Options:**
- `--episodes N`: Number of episodes to visualize
- `--fps N`: Frames per second (default 30)
- `--trail_len N`: Bee trail length in frames (default 60)
- `--orbit_alpha F`: Orbit line transparency (0-1)
- `--stochastic`: Use stochastic policy (random sampling) instead of argmax
- `--save FILE.mp4`: Save animation to file

---

## Interactive Visualization

**Real-time Policy Display** (left panel - 3D orbital scene):
- Grid plane (xy-plane at z=0)
- Bee orbits (colored ribbons)
- Bee positions (colored dots with ID labels)
- Bee trails (colored paths showing recent motion)
- Flowers (orange = available, green = harvested, grey = expired)

**HUD Status Panel** (right panel):
- Step counter & grid size
- Per-bee: action, battery %, load, harvest count, status flags
- Status includes: OK, SILENT, RECHARGING, CRIT (critical battery), IDLE-SILENT
- Dimmed visuals for terminated bees

**Mouse Controls:**
- Drag to rotate 3D view
- Scroll to zoom

---

## Training Metrics

The training loop logs these metrics:

### Per-Episode:
- `total_reward`: Sum of all bee rewards
- `bees_harvested_total`: Total flowers harvested
- `avg_load_dropped`: Average pollen dropped per bee
- `avg_battery_remaining`: Average battery at episode end

### Per-Update:
- `actor_loss`: Policy gradient loss
- `critic_loss`: Value function loss
- `entropy`: Policy entropy (exploration)
- `advantage_mean`: Average advantage estimate
- `explained_var`: Critic's explained variance ratio

### Tensorboard Logs:
```bash
tensorboard --logdir output_fixed
# Open http://localhost:6006 in browser
```

---

## VFL Retasking in Action

During training with VFL enabled:

1. **Orphan Detection** (every step):
   - Identifies flowers assigned to silent bees
   - Identifies flowers assigned to recharging bees
   - Identifies flowers assigned to unreachable bees

2. **Retask Board Generation**:
   - Top-3 orphan flowers selected by priority & distance
   - Encoded as compact 5-feature vector (x, y, priority, reachable, assigned)
   - Exposed in each local bee's observation

3. **Claim Resolution**:
   - Local actor outputs claim logits over 4 slots (3 board + 1 no-claim)
   - Actor samples claim selection during rollout
   - Conflicts resolved by minimum orbit distance
   - Winner assigned to claimed flower

---

## Tips & Tricks

### Speed Up Training
- Reduce `rollout_len` in config.yaml (256 instead of 512)
- Reduce `episodes` (100 instead of 1000)
- Use GPU: `--device cuda`
- Disable visualizations during training

### Improve Convergence
- Increase `entropy_coef` in config.yaml (0.05 instead of 0.02)
- Reduce learning rates: `lr_actor: 5e-5`, `lr_critic: 1e-4`
- Increase `gae_lambda` to 0.98
- Tune `retask_board_size`: Larger = more flexibility, more compute

### Debug Issues
- Check log for environment errors: `grep "\[ENV\]" training_output.log`
- Watch for NaN/Inf: `grep -i "nan\|inf" training_output.log`
- Monitor memory: `watch -n 1 'ps aux | grep train_model_policy'`
- Verify checkpoints saved: `ls -lh output_fixed/actor_*.pt | tail -5`

### Visualize Specific Bee
Edit `bee_orbits_3d.py` to highlight a particular bee:
```python
# Line ~380: make target bee's trail thicker
if i == 0:  # Bee 0
    trail_lines[i].set_linewidth(3.0)
    trail_lines[i].set_color("red")
```

---

## Configuration Reference

**Key Parameters in config.yaml:**

```yaml
env:
  num_bees: 5                    # Number of agents
  num_flowers: 12                # Number of tasks
  grid_size: 30                  # World grid size
  max_steps: 800                 # Episode length
  retask_board_size: 3           # VFL board size
  retask_timeout_steps: 30       # Silent detection window
  battery_min_steps: 250         # Min battery capacity
  battery_max_steps: 450         # Max battery capacity

training:
  episodes: 1000                 # Total episodes
  rollout_len: 512               # Rollout steps per collection
  updates: 1000                  # PPO update iterations
  lr_actor: 1.0e-4               # Actor learning rate
  lr_critic: 2.0e-4              # Critic learning rate
  entropy_coef: 0.02             # Exploration bonus
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Training very slow | Use GPU (`--device cuda`), reduce rollout_len |
| NaN loss | Reduce learning rates, check for inf/nan in log |
| Memory error | Reduce batch_size, reduce num_bees, reduce episodes |
| Visualization shows all grayed out | Check that checkpoint exists, verify model_tag |
| HUD shows "---" for all values | Likely checkpoint loading issue, check log |
| Training crashes | Check `training_output.log`, verify config.yaml syntax |

---

## next Steps

1. **Monitor Training**: Use `monitor_training.py` or `tail -f training_output.log`
2. **Visualize Progress**: Periodically run with latest checkpoint: `bee_orbits_3d.py --policy --model_tag <N>`
3. **Analyze Results**: Check tensorboard: `tensorboard --logdir output_fixed`
4. **Fine-tune**: Adjust config.yaml parameters based on performance
5. **Deploy**: Copy best.pt checkpoint for production use

---

**Last Updated**: November 14, 2025

Training started at $(date).
Current process: PID 35529, using 345% CPU.
Estimated completion: ~2-4 hours from start.
