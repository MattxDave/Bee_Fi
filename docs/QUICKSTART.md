# 🐝 Bee Swarm RL - Quick Start Guide

## Overview

Multi-agent reinforcement learning system simulating bee-like orbital agents harvesting flowers. Uses PPO with centralized critic and attention-based actor networks.

## Files Included

```
bee_swarm_package/
├── bee_state.py          # Bee agent with orbital mechanics
├── bees_env.py           # PettingZoo multi-agent environment
├── bee_policy.py         # Actor (attention) + Centralized Critic
├── bee_orbits_3d.py      # 3D visualization with HUD
├── train_orbital_v2.py   # PPO training script
├── train_utils.py        # Training utilities
├── utils.py              # General utilities
├── requirements.txt      # Python dependencies
├── README.md             # Full documentation
├── ARCHITECTURE.md       # System architecture diagrams (Mermaid)
├── TRAINING_GUIDE.md     # Detailed training guide
└── best/
    ├── best_actor.pt     # Trained actor weights
    └── best_critic.pt    # Trained critic weights
```

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- PettingZoo

## Run Simulation (with trained model)

```bash
# Run visualization with trained policy
python bee_orbits_3d.py --policy --model_tag best --stochastic --fps 10 --steps 400
```

### Command Options
| Flag | Description |
|------|-------------|
| `--policy` | Use trained neural network policy |
| `--model_tag best` | Load from `best/` folder |
| `--stochastic` | Sample actions stochastically (more realistic) |
| `--fps 10` | Animation speed |
| `--steps 400` | Max simulation steps |

## Train New Model

```bash
# Train with 16 parallel environments
python train_orbital_v2.py --output my_model --num-envs 16

# Train on CPU only
python train_orbital_v2.py --output my_model --num-envs 16 --cpu
```

### Training Output
Models saved to `<output>/`:
- `<output>_actor.pt` - Actor network weights
- `<output>_critic.pt` - Critic network weights

## Key Concepts

### Actions
| ID | Action | Description |
|----|--------|-------------|
| 0 | DONOTHING | Continue orbiting |
| 1 | HARVEST | Collect pollen from nearby flower |
| 2 | GROOM | Deposit pollen (reset load) |

### Communication
- Each bee has its own **retask board**
- When a bee dies (battery=0), it broadcasts tasks to nearest active bee
- Receiving bee either claims task or hands off to next bee

### Rewards
- Successful harvest: +5.0 + pollen bonus
- Strategic groom (load ≥ 80%): +5.0 + 10×load_ratio
- Invalid actions: penalty

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed Mermaid diagrams of:
- Actor/Critic neural networks
- Per-bee retask board communication
- PPO training loop
- Environment structure

## Example Output

```
Step 42: Bee 0 harvested flower 5 (+7.5 pollen)
Step 43: Bee 2 battery = 0 → DIED
Step 43: Retask: Flower 7 orphaned, broadcast to Bee 0
Step 48: Bee 0 claimed flower 7 from retask board
Step 85: SUCCESS! All 12 flowers harvested
```
