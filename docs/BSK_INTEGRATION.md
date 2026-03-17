# Basilisk Integration — How the RL Model Works in the BSK Simulator

## Architecture Overview

```
┌──────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌───────────┐
│  BSKInterface │────▶│  bees_env.py │────▶│  Observations   │────▶│  Actor    │
│  (Basilisk)   │     │  (PettingZoo) │     │  (per bee)      │     │  (policy) │
└──────┬───────┘     └──────┬───────┘     └────────┬────────┘     └─────┬─────┘
       │                    │                      │                     │
  ECI positions        maps to grid           12-feat/flower         action 0/1/2
  velocity             battery sync           battery frac           DONOTHING
  battery frac         task expiration        retask board           HARVEST
  solar power          gossip relay           consensus              GROOM
       │                    │                      │                     │
       └────────────────────┴──────────────────────┴─────────────────────┘
                              ▲                                   │
                              └───────────── env.step(actions) ───┘
```

## Components

### 1. BSKInterface (`bsk_interface.py`)

Thin wrapper around a multi-satellite Basilisk simulation. Creates N spacecraft with:

- **Gravity model**: Earth central body (includes J2 perturbations)
- **Battery**: `SimpleBattery` — 200 Wh capacity, 80% initial charge
- **Power sink**: `SimplePowerSink` — 3W constant draw
- **Solar panel** (optional): `SimpleSolarPanel` — 0.32 m², 35% efficiency
- **Navigation**: `SimpleNav` for clean position output

Default constellation: **Walker-delta LEO** — 550 km altitude, 53° inclination, 3 orbital planes.

Key methods:
- `initialize()` — Builds and starts the Basilisk sim
- `step()` → `list[SatState]` — Advances sim by 1 second, returns per-satellite state
- `get_positions_grid(grid_size, meters_per_unit)` — Maps ECI positions to grid coordinates

### 2. BeeForagingEnv (`bees_env.py`) — BSK Mode

The PettingZoo multi-agent environment has a `use_basilisk=True` toggle. When active:

#### Reset (Initialization)

1. BSKInterface creates N spacecraft in Basilisk
2. `get_positions_grid()` maps ECI metres → grid coords (centered on constellation centroid, scaled by `meters_per_unit = 500,000`)
3. **Kepler element sync**: Bee orbital elements (a, e, i, Ω, ω) are overwritten from BSK orbital elements so that flower placement + reachability calculations use the real BSK orbits
4. Flowers are placed along the synced orbits via `_ensure_reachable_flower_placement()`
5. Each flower gets task metadata: `task_id`, `task_description`, `deadline_step`, `status`
6. Round-robin assignment by descending pollen value

#### Step (Each Tick)

| Phase | What Happens |
|-------|-------------|
| Action override | Battery-dead bees forced to DONOTHING |
| Proactive groom | If HARVEST requested but load can't fit smallest in-range flower → auto-groom first |
| GROOM resolution | Battery recharge (if dead) or pollen offload (if ≥80% capacity) |
| HARVEST intent | Finds closest reachable, non-expired, assigned/unassigned flower within `harvest_radius` |
| Exclusivity | If two bees claim same flower, closest wins |
| **BSK position update** | `bsk.step()` advances Basilisk by 1s → new ECI positions → `get_positions_grid()` → updates `bee.fx/fy/fz` |
| **BSK battery sync** | `bsk_states[i].battery_frac × battery_max[i]` → overwrites env battery |
| Task expiration | Flowers past `deadline_step` → expired, unassigned, penalty to owning bee |
| Gossip propagation | Dead/overloaded bees broadcast orphan tasks via physics-based gossip relay |
| Build observations | Per-bee observation dict constructed from BSK-updated state |

### 3. Actor Policy (`bee_policy.py`)

Transformer-attention architecture:

- **Flower encoder**: Each flower → 12-feature vector → Linear(12, 128) → 1-layer Transformer (4 heads) → 128-dim summary
- **Feature encoders**: Position(64), Status(32), Battery(32), Step(32), Consensus(64), Retask(128), ActionAvail(32)
- **Trunk**: Concatenated features → Linear(512, 256) → Linear(256, 256) → policy logits (3 actions)
- **Claim head** (optional): For retask board slot selection

### 4. BSK Evaluator (`bsk_evaluator.py`)

Runs the trained policy in BSK mode and produces `telemetrybridge.json`-format output:

```bash
PYTHONPATH="basilisk/dist3:$PYTHONPATH" python bsk_evaluator.py \
    --model outputs/best_actor.pt \
    --num_sats 25 --num_tasks 50 \
    --steps 1200 --snapshot_interval 60 \
    --output bsk_telemetry_eval.json
```

Output schema per satellite:
- `satellite_id`, `status`, `orbit_type` (NEO/MEO/GEO)
- `position_eci` (x, y, z in metres), `velocity_eci` (vx, vy, vz in m/s)
- `position_rn` (range, lat, lon), `velocity_rn` (speed, heading)
- `assigned_tasks[]` — each with `task_id`, `task_description`, `priority`, `distance_to_task_m`, `task_status`, `deadline_step`
- `fuel_mass`, `battery_level`, `solar_panel_power`
- `communication_status`, `simulation_time`
- Gossiper section: per-satellite neighbors, their tasks, TTL, LOS status

---

## Data Flow: BSK → Environment → Policy → Action

### Per-Bee Observation Vector

| Field | Source in BSK Mode | Dimension |
|-------|-------------------|-----------|
| `position` | `bee.fx, fy, fz` from `bsk.get_positions_grid()` | 3 |
| `status` | `[bee.mode, load_fraction]` | 2 |
| `battery` | `[bsk_battery_frac, is_recharging]` | 2 |
| `flowers` | 12 features × 50 flowers | 600 |
| `step_count` | `steps / max_steps` | 1 |
| `consensus` | Last action of each bee / 2.0 | 25 |
| `retask_board` | 5 features × 3 slots (from gossiper inbox) | 15 |
| `action_availability` | `[can_harvest, can_groom, can_do_nothing]` | 3 |

### Per-Flower Features (12 each)

| # | Feature | Description |
|---|---------|-------------|
| 0 | `x` | Normalized grid x position |
| 1 | `y` | Normalized grid y position |
| 2 | `pollen` | Pollen amount / max capacity |
| 3 | `harvested` | 1.0 if already harvested |
| 4 | `mine` | 1.0 if assigned to this bee |
| 5 | `busy` | 1.0 if another bee is harvesting it |
| 6 | `reachable` | 1.0 if on this bee's orbital path |
| 7 | `fits` | 1.0 if pollen fits in remaining capacity |
| 8 | `dist_norm` | 3D distance / harvest_radius |
| 9 | `harvestable_now` | 1.0 if within time window |
| 10 | `hard_window` | 1.0 if HARD time window type |
| 11 | `time_to_window` | Steps until next window / max_steps |

### Retask Board Slot Features (5 each, 3 slots)

| Feature | Description |
|---------|-------------|
| `x` | Normalized orphan flower x |
| `y` | Normalized orphan flower y |
| `pollen` | Pollen / max capacity |
| `can_fit` | 1.0 if fits in bee's remaining capacity |
| `hops` | Gossip hop count / 10 |

---

## What BSK Controls vs What the RL Policy Controls

| Aspect | BSK Simulator | RL Policy |
|--------|--------------|-----------|
| Satellite **position** | ✅ Full N-body orbital mechanics | ❌ Position overwritten by BSK |
| **Velocity** | ✅ Computed by BSK | ❌ In BSK state but not in obs |
| **Battery drain** | ✅ SimplePowerSink drains SimpleBattery | ❌ Env battery synced FROM BSK |
| **Solar recharge** | ✅ SimpleSolarPanel (if enabled) | ❌ Automatic in BSK |
| **Orbital manoeuvring** | ❌ No thruster model | ❌ Satellites follow passive orbits |
| **Task selection** | ❌ | ✅ HARVEST / GROOM / DONOTHING |
| **Task assignment** | ❌ | ✅ Retask board + gossip relay |
| **Groom timing** | ❌ | ✅ When to offload pollen or recharge |
| **Task expiration response** | ❌ | ✅ Must harvest before deadline_step |
| **Fuel mass** | Static (50 kg, no thruster model) | Not used |

**Key insight**: The RL policy does NOT control orbital motion. Satellites follow their Basilisk-computed orbits passively. The policy's job is purely **task scheduling** — given where each satellite IS right now (from BSK), choose whether to harvest, groom, or wait.

---

## Task System (Inspired by satellite_constellation_scheduling)

### Task Metadata on Each Flower

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Unique ID: `"TASK-{id:03d}-{x}-{y}"` |
| `task_description` | `str` | Cycles through: Capture imagery, Relay communication, Data collection, Sensor calibration, Harvest pollen |
| `status` | `str` | `unassigned` → `assigned` → `completed` or `expired` |
| `deadline_step` | `int` | Absolute step deadline (priority-scaled: high priority = tighter deadline) |
| `created_step` | `int` | Step when task was created (0 at reset) |
| `priority` | `float` | `pollen / capacity` (0–1) |

### Task Expiration

- Each tick, any unharvested flower past its `deadline_step` is marked `expired`
- Expired tasks are unassigned from their owning bee (−1.0 penalty)
- Expired flowers are excluded from: harvest scans, retask board, action availability
- Episode can end when all flowers are **done** (harvested OR expired)

### Gossip-Based Re-tasking

- When a bee dies (battery = 0), its orphan tasks are broadcast as `GossipMessage`
- Messages propagate one hop per step to nearest reachable bee (range-limited to `harvest_radius × 1.5`)
- Receiving bee evaluates: orbit reachability, capacity, deadline, workload vs priority
- If rejected, message forwards to next unseen reachable bee (with TTL decrement)

---

## BSK Element Sync Fix (0% → 54% Harvest)

### The Problem

During `reset()`, flowers were placed using default Bee Kepler elements (`a=90, e=0.85`), then BSK overwrote positions to a tight Walker constellation. Flowers and satellites existed in different grid regions.

### The Fix

After BSK initializes, sync Bee Kepler elements from BSK orbital elements:

```python
for i, b in enumerate(self.bees):
    oe = bsk._orbital_elements[i]
    b.a = oe["a_m"] / self.bsk_meters_per_unit  # metres → grid units
    b.e = oe["e"]
    b.i = math.radians(oe["i_deg"])
    b.Omega = math.radians(oe["Omega_deg"])
    b.omega = math.radians(oe["omega_deg"])
```

Then re-place flowers along the corrected orbits + re-compute reachability. Result: **0% → 54% harvest rate**.

---

## Performance

| Mode | Harvest Rate | Notes |
|------|-------------|-------|
| Keplerian (training mode) | 93.9% | Spread-out orbits, trained model |
| BSK Walker constellation | 54% | Same model, tighter Walker geometry |

The gap exists because the model was trained on spread-out Keplerian orbits but evaluated on BSK's tight Walker cluster. Retraining specifically for BSK geometry would close this gap.

---

## Running the Evaluation

```bash
# Set Basilisk path
export PYTHONPATH="/home/matthew/projects/Bee_Fi/basilisk/dist3:$PYTHONPATH"

# Run BSK evaluation with trained model
python bsk_evaluator.py \
    --model outputs/best_actor.pt \
    --num_sats 25 --num_tasks 50 \
    --steps 1200 --snapshot_interval 60 \
    --output bsk_telemetry_eval.json

# Run without BSK (Keplerian fallback)
python bsk_evaluator.py \
    --model outputs/best_actor.pt \
    --num_sats 25 --num_tasks 50 \
    --steps 1200 --no_basilisk \
    --output keplerian_eval.json
```

---

## File Map

| File | Role |
|------|------|
| `bsk_interface.py` | Basilisk wrapper — creates spacecraft, steps sim, returns SatState |
| `bees_env.py` | RL environment — BSK toggle, position sync, battery sync, task system |
| `bee_state.py` | Bee + Flower classes with task metadata (task_id, deadline, status) |
| `bee_policy.py` | Transformer-attention Actor network |
| `gossiper.py` | Physics-based gossip relay for task re-assignment |
| `bsk_evaluator.py` | Runs policy in BSK mode, outputs telemetrybridge.json |
| `train_orbital_v2.py` | PPO training loop |
| `config.yaml` | All training + environment parameters |
