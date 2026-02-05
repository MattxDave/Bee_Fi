#!/usr/bin/env python3

import math
import random
from datetime import datetime, timedelta, timezone

import bee_state
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

# Optional; present in your tree. We keep import for compatibility.
try:
    from telemetry_replay import TelemetryStream  # noqa: F401
except Exception:
    TelemetryStream = None

# Telemetry bridge mapper
try:
    from telemetry_mapper import TelemetryMapper
except Exception:
    TelemetryMapper = None

# Quiet mode for environment prints. Set to True to suppress verbose per-step prints
# during training (helps keep logs small). Assign to False to re-enable prints.
ENV_QUIET = False
if ENV_QUIET:
    import builtins as _builtins

    _builtins_print = _builtins.print
    _builtins.print = lambda *a, **k: None


class BeeForagingEnv(ParallelEnv):
    """
    Orbital-bee foraging with assignments, exclusivity, capacity + grooming logic,
    SGP4 toggles, broadcast + re-tasking, battery + recharge with task reassignment.

    BATTERY & TASK QUEUE SYSTEM:
    - When battery depletes: bee releases all assigned flowers to pool (assigned_bee=None)
    - Dead bee stops broadcasting, flowers appear in retask_board for other bees
    - Other bees can harvest unassigned flowers from the pool
    - When battery recharges: bee gets reassigned available flowers from pool
    - Time windows (HARD/SOFT/NONE) with bonuses and penalties
    - Episode terminates when all flowers harvested OR all bees truncated

    ENERGY COSTS:
    - Orbital movement drains battery per step (drain_per_step + drain_per_unit * distance)
    - HARVEST action costs 5 energy units

    GROOM ACTION (dual-purpose):
    - When load >= 80% capacity: offloads pollen (empties load)
    - When battery <= 30% max: initiates battery recharge
    - If both conditions met, prioritizes battery recharge

    PER-BEE RETASKING BOARD (communication chain):
    - Each bee maintains its own retask_board of pending tasks
    - When a bee fails/dies, its tasks broadcast to nearest active bee
    - Receiving bee evaluates task acceptance based on:
      * Orbit reachability (can bee's orbit reach the flower?)
      * Capacity (does bee have room for pollen?)
      * Deadline (can bee reach flower before HARD window closes?)
      * Workload vs Priority (is task worth taking given current assignments?)
    - If bee REJECTS task, it holds until close to another active bee
    - Task then handed off to next bee for evaluation (hops tracking)
    - Chain continues until: task accepted, deadline missed, or episode ends

    Actions:
      0 = DONOTHING, 1 = HARVEST, 2 = GROOM
    """

    metadata = {"render.modes": ["human"], "name": "bee_foraging_orbits_capacity_v5"}

    def __init__(
        self,
        num_bees=5,
        num_flowers=12,
        grid_size=20,
        max_steps=800,
        time_window_min=10,
        time_window_max=80,
        harvest_radius=35.0,  # Increased to cover orbital motion range
        lambda_z=0.1,
        knn_k=3,
        orbit_scale=1.2,
        spawn_on_orbit_ratio=0.8,
        shaping_weight=0.05,
        anti_spam_pen=-0.005,
        reach_margin=5.0,  # Increased margin for orbital reachability
        reach_samples=120,
        bee_capacity: float = 10.0,
        # ---------- SGP4 toggles ----------
        use_sgp4: bool = False,
        sgp4_km_per_unit: float = 300.0,
        sgp4_time_stagger_s: int = 60,
        tle_lines: list | None = None,
        # ---------- Broadcast / re-tasking (VFRL-ish) ----------
        retask_timeout_steps: int = 30,
        # size of the fixed retask board exposed to each local actor
        retask_board_size: int = 3,
        count_idle_as_silent: bool = True,
        # ---------- Battery / Recharge ----------
        battery_min_steps: int = 250,
        battery_max_steps: int = 450,
        recharge_steps: int = 30,
        drain_per_step: float = 1.0,
        drain_per_unit: float = 0.0,
        # ---------- Logging ----------
        verbose: bool = True,
    ):
        super().__init__()

        # Logging flag
        self.verbose = verbose

        # Base config
        self.num_bees = int(num_bees)
        self.num_flowers = int(num_flowers)
        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.time_window_min = int(time_window_min)
        self.time_window_max = int(time_window_max)
        self.harvest_radius = float(harvest_radius)
        self.lambda_z = float(lambda_z)
        self.knn_k = int(knn_k)
        self.orbit_scale = float(orbit_scale)
        self.spawn_on_orbit_ratio = float(spawn_on_orbit_ratio)
        self.shaping_weight = float(shaping_weight)
        self.anti_spam_pen = float(anti_spam_pen)
        self.reach_margin = float(reach_margin)
        self.reach_samples = int(reach_samples)
        self.bee_capacity = float(bee_capacity)

        # SGP4
        self.use_sgp4 = bool(use_sgp4)
        self.sgp4_km_per_unit = float(sgp4_km_per_unit)
        self.sgp4_time_stagger_s = int(sgp4_time_stagger_s)
        self.tle_lines = tle_lines if isinstance(tle_lines, list) else None

        # Broadcast / Re-tasking
        self.retask_timeout_steps = int(retask_timeout_steps)
        self.retask_board_size = int(retask_board_size)
        self.count_idle_as_silent = bool(count_idle_as_silent)

        # Battery settings
        self.battery_min_steps = int(battery_min_steps)
        self.battery_max_steps = int(battery_max_steps)
        self.recharge_steps = int(recharge_steps)
        self.drain_per_step = float(drain_per_step)
        self.drain_per_unit = float(drain_per_unit)

        # Track if this is a low-battery training episode
        self._is_low_battery_episode = False

        # Agent lists
        self.possible_agents = [f"bee_{i}" for i in range(self.num_bees)]
        self.agents = []
        self._agent_name_mapping = {a: i for i, a in enumerate(self.possible_agents)}

        # Observations: position(3) + status(2) + flowers(num_flowers*7) + step(1) + consensus(num_bees)
        # plus a compact retask board: per-slot (x_norm, y_norm, priority, reachable_flag, assigned_flag)
        retask_slot_dim = 5 * self.retask_board_size
        obs_dim = 3 + 2 + (self.num_flowers * 7) + 1 + self.num_bees + retask_slot_dim
        obs_high = np.inf * np.ones((obs_dim,), dtype=np.float32)
        self.observation_spaces = {
            a: spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
            for a in self.possible_agents
        }

        # Actions
        self.action_spaces = {a: spaces.Discrete(3) for a in self.possible_agents}

        # Runtime state (initialized in reset)
        self.last_actions = {}
        self.bees: list[bee_state.Bee] = []
        self.flowers: list[bee_state.Flower] = []
        self.assignments: list[list[int]] = []
        # Retask board runtime (list of dicts with compact info for top-M orphan flowers)
        self.retask_board: list[dict] = []

        # Reachability matrix (B x F)
        self.reachable = np.zeros((self.num_bees, self.num_flowers), dtype=bool)

        # Broadcast / heartbeat state
        self._last_broadcast_step = np.zeros(self.num_bees, dtype=int)
        self._idle_steps = np.zeros(self.num_bees, dtype=int)

        # Battery state
        self._battery_max = np.zeros(self.num_bees, dtype=float)
        self._battery = np.zeros(self.num_bees, dtype=float)
        self._recharge_until = np.full(self.num_bees, -1, dtype=int)
        self._last_pos = np.zeros((self.num_bees, 3), dtype=float)

        # For orbit-change checks
        self.last_effective_inclination = [0.0] * self.num_bees
        self.last_effective_yaw = [0.0] * self.num_bees

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.steps = 0
        self.last_actions = {a: 0 for a in self.possible_agents}

        # Check if we should preserve telemetry-loaded state
        preserve_telemetry = getattr(self, '_telemetry_loaded', False) and len(getattr(self, 'bees', [])) > 0
        
        if preserve_telemetry:
            # TELEMETRY MODE: Preserve loaded bees and flowers, just reset their state
            for b in self.bees:
                b.load = 0.0
                b.mode = bee_state.Bee.IDLE
                b.groom_cooldown = 0
                # Don't reset position - keep telemetry positions
            
            # Reset flower harvest state but keep positions and windows
            for f in self.flowers:
                f.harvested = False
            
            # Skip normal bee/flower creation
        else:
            # NORMAL MODE: Create new random bees and flowers
            # Bees - USE CONFIGURED ORBIT SCALE
            self.bees = []
            for i in range(self.num_bees):
                # Use the configured orbit_scale instead of hardcoded value
                b = bee_state.Bee(
                    i, self.grid_size, orbit_scale=self.orbit_scale, capacity=self.bee_capacity
                )

                # Ensure the orbit is properly sized
                b.a = b.a_units = self.grid_size * self.orbit_scale
                b.update_position()
                b.load = 0.0
                b.mode = bee_state.Bee.IDLE
                b.groom_cooldown = 0
                self.bees.append(b)

            # SGP4 toggle for all bees (safe if methods absent)
            self._apply_sgp4_mode(
                use_sgp4=self.use_sgp4,
                tle_lines=self.tle_lines,
                km_per_unit=self.sgp4_km_per_unit,
                stagger_seconds=self.sgp4_time_stagger_s,
            )

            # Flowers with improved reachable placement
            self.flowers = []
            self._ensure_reachable_flower_placement()

            # Unique pollen amounts up to capacity
            base = np.arange(1, self.num_flowers + 1, dtype=float)
            perm = np.random.permutation(base)
            scale = self.bee_capacity / self.num_flowers
            amounts = perm * scale
            for j, f in enumerate(self.flowers):
                f.pollen = float(amounts[j])
                f.priority = f.pollen / self.bee_capacity

        # Round-robin assignments by descending pollen (for both modes)
        order = sorted(range(self.num_flowers), key=lambda j: self.flowers[j].pollen, reverse=True)
        self.assignments = [[] for _ in range(self.num_bees)]
        for idx, fj in enumerate(order):
            bid = idx % self.num_bees
            self.assignments[bid].append(fj)
            self.flowers[fj].assigned_bee = bid
        for i, b in enumerate(self.bees):
            b.assigned_flowers = self.assignments[i][:]

        # Orbit-change trackers
        self.last_effective_inclination = [
            float(b.i + getattr(b, "inclination_delta", 0.0)) for b in self.bees
        ]
        self.last_effective_yaw = [float(b.Omega + getattr(b, "yaw_delta", 0.0)) for b in self.bees]

        # Broadcast heartbeat
        self._last_broadcast_step[:] = 0
        self._idle_steps[:] = 0

        # Battery init with random low-battery episodes for retasking learning
        rng = np.random.default_rng()

        # 30% chance of "low battery episode" to force retasking scenarios
        if rng.random() < 0.3:
            # Low battery episode: batteries will die mid-episode, forcing task reassignment
            self._is_low_battery_episode = True
            battery_min = max(50, self.battery_min_steps // 3)  # Much shorter battery
            battery_max = max(100, self.battery_max_steps // 3)
            self._battery_max = rng.integers(
                battery_min, battery_max + 1, size=self.num_bees
            ).astype(float)

            # Also increase drain rate for low-battery episodes to ensure batteries die
            self.drain_per_step = self.drain_per_step * rng.uniform(2.0, 3.0)

            if self.verbose:
                print(
                    f"[BATTERY]  Low-battery training episode - max={battery_max}, drain={self.drain_per_step:.1f}x"
                )
                print("[BATTERY] Batteries WILL die during episode to practice retasking!")
        else:
            # Normal episode: standard battery capacity
            self._is_low_battery_episode = False
            self._battery_max = rng.integers(
                self.battery_min_steps, self.battery_max_steps + 1, size=self.num_bees
            ).astype(float)

        self._battery = self._battery_max.copy()
        self._recharge_until[:] = -1
        for i, b in enumerate(self.bees):
            self._last_pos[i] = np.array([float(b.fx), float(b.fy), float(b.fz)], dtype=float)

        # Reachability + harvestable mask
        self._mark_reachable_per_bee()
        self.harvestable = np.zeros(self.num_flowers, dtype=bool)
        for j, f in enumerate(self.flowers):
            ab = int(f.assigned_bee) if f.assigned_bee is not None else -1
            self.harvestable[j] = ab >= 0 and self.reachable[ab, j]

        # Debug reachability (optional - comment out for production)
        # self.debug_reachability()

        # Initialize retask board for the start of an episode
        try:
            self._update_retask_board()
        except Exception:
            # don't fail reset if update logic has an unexpected issue
            self.retask_board = []

        return self._get_observations()

    # ----------------------------------------------------------
    # Telemetry Loading
    # ----------------------------------------------------------
    def load_from_telemetry(self, telemetry_path: str, start_step: int = 0) -> dict:
        """
        Initialize environment state from a telemetry JSON file.

        Args:
            telemetry_path: Path to telemetrybridge.json
            start_step: Step to start from (default 0)

        Returns:
            metadata dict with mapping info
        """
        if TelemetryMapper is None:
            raise ImportError("TelemetryMapper not available. Check telemetry_mapper.py")
        
        # Set flag to preserve telemetry state during reset
        self._telemetry_loaded = True
        self._telemetry_path = telemetry_path

        # Create mapper with environment config
        # ECI coordinates are in METERS, scale appropriately
        mapper = TelemetryMapper({
            "grid_size": self.grid_size,
            "meters_per_unit": 500_000,  # 500km per grid unit - fits LEO/MEO/GEO
            "seconds_per_step": 1.0,     # 1 second per step for responsive control
            "max_steps": self.max_steps,
            "battery_capacity": self._battery_max[0] if len(self._battery_max) > 0 else 100.0,
            "pollen_capacity": self.bee_capacity,
        })

        # Load and map telemetry
        bees, flowers, metadata = mapper.map_telemetry(telemetry_path)

        # Update environment dimensions if different
        if len(bees) != self.num_bees or len(flowers) != self.num_flowers:
            self.num_bees = len(bees)
            self.num_flowers = len(flowers)
            self.possible_agents = [f"bee_{i}" for i in range(self.num_bees)]
            self._agent_name_mapping = {a: i for i, a in enumerate(self.possible_agents)}

            # Rebuild observation and action spaces for new dimensions
            retask_slot_dim = 5 * self.retask_board_size
            obs_dim = 3 + 2 + (self.num_flowers * 7) + 1 + self.num_bees + retask_slot_dim
            obs_high = np.inf * np.ones((obs_dim,), dtype=np.float32)
            self.observation_spaces = {
                a: spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
                for a in self.possible_agents
            }
            self.action_spaces = {a: spaces.Discrete(3) for a in self.possible_agents}

        # Replace bees
        self.bees = bees
        self.agents = self.possible_agents[:]

        # Replace flowers
        self.flowers = flowers

        # Rebuild assignments from flower.assigned_bee
        self.assignments = [[] for _ in range(self.num_bees)]
        for j, f in enumerate(flowers):
            if f.assigned_bee is not None and 0 <= f.assigned_bee < self.num_bees:
                self.assignments[f.assigned_bee].append(j)

        # Sync bee.assigned_flowers
        for i, bee in enumerate(self.bees):
            bee.assigned_flowers = self.assignments[i][:]

        # Reset battery arrays to match new bee count
        self._battery_max = np.array([b.battery_capacity for b in self.bees], dtype=float)
        self._battery = np.array([b.battery for b in self.bees], dtype=float)
        self._recharge_until = np.full(self.num_bees, -1, dtype=int)
        self._last_pos = np.array([[b.fx, b.fy, b.fz] for b in self.bees], dtype=float)

        # Broadcast/idle state
        self._last_broadcast_step = np.zeros(self.num_bees, dtype=int)
        self._idle_steps = np.zeros(self.num_bees, dtype=int)

        # Orbit-change trackers (must match num_bees)
        self.last_effective_inclination = [
            float(b.i + getattr(b, "inclination_delta", 0.0)) for b in self.bees
        ]
        self.last_effective_yaw = [
            float(b.Omega + getattr(b, "yaw_delta", 0.0)) for b in self.bees
        ]

        # Set step counter
        self.steps = start_step
        self.last_actions = {a: 0 for a in self.possible_agents}

        # For telemetry mode: All flowers are "reachable" by their assigned bee
        # (orbital mechanics don't apply the same way to telemetry scenarios)
        self.reachable = np.zeros((self.num_bees, self.num_flowers), dtype=bool)
        for j, f in enumerate(self.flowers):
            if f.assigned_bee is not None and 0 <= f.assigned_bee < self.num_bees:
                self.reachable[f.assigned_bee, j] = True
            else:
                # Unassigned flowers are reachable by all non-terminated bees
                for i, b in enumerate(self.bees):
                    if not b.terminated:
                        self.reachable[i, j] = True

        # Harvestable mask
        self.harvestable = np.zeros(self.num_flowers, dtype=bool)
        for j, f in enumerate(self.flowers):
            ab = int(f.assigned_bee) if f.assigned_bee is not None else -1
            self.harvestable[j] = ab >= 0 and self.reachable[ab, j]

        # Update retask board
        try:
            self._update_retask_board()
        except Exception:
            self.retask_board = []

        # Store telemetry metadata
        self._telemetry_metadata = metadata

        if self.verbose:
            print(f"[TELEMETRY] Loaded {len(bees)} bees, {len(flowers)} flowers from {telemetry_path}")
            print(f"[TELEMETRY] Failed satellites: {metadata.get('failed_satellites', [])}")
            print(f"[TELEMETRY] Tasks moved: {metadata.get('total_tasks_moved', 0)}")

        return metadata

    def _ensure_reachable_flower_placement(self):
        """Ensure flowers are placed in locations reachable by at least one bee"""
        occupied = set()

        for j in range(self.num_flowers):
            attempts = 0
            max_attempts = 50

            while attempts < max_attempts:
                # Try to spawn near orbits with higher probability
                if random.random() < self.spawn_on_orbit_ratio:
                    # Spawn near orbital paths
                    bee_id = random.randint(0, self.num_bees - 1)
                    bee = self.bees[bee_id]

                    # Sample points along the orbit
                    nu = random.uniform(0, 2 * math.pi)
                    a, e = float(bee.a), float(bee.e)
                    i = float(bee.i)
                    Om = float(bee.Omega)
                    om = float(bee.omega)

                    R = self._R3(Om) @ self._R1(i) @ self._R3(om)
                    b_semi = a * math.sqrt(max(1e-12, 1 - e * e))

                    r_local = np.array([a * math.cos(nu), b_semi * math.sin(nu), 0.0])
                    r_world = R @ r_local

                    cx = self.grid_size / 2.0
                    cy = self.grid_size / 2.0

                    # Add some random offset but keep within grid
                    offset_x = random.uniform(-self.harvest_radius, self.harvest_radius)
                    offset_y = random.uniform(-self.harvest_radius, self.harvest_radius)

                    x = int(np.clip(cx + r_world[0] + offset_x, 0, self.grid_size - 1))
                    y = int(np.clip(cy + r_world[1] + offset_y, 0, self.grid_size - 1))
                else:
                    # Random spawn anywhere in grid
                    x, y = self._sample_scatter_cell(self.grid_size)

                if (x, y) not in occupied:
                    # Check if this position is reachable by any bee
                    reachable_by_any = False
                    flower_center_x, flower_center_y = x + 0.5, y + 0.5  # flower center

                    for bee in self.bees:
                        min_dist = self._min_distance_to_orbit(
                            bee, flower_center_x, flower_center_y
                        )
                        if min_dist <= (self.harvest_radius + self.reach_margin):
                            reachable_by_any = True
                            break

                    if reachable_by_any:
                        occupied.add((x, y))
                        f = self._make_flower(j, x, y)
                        self.flowers.append(f)
                        break

                attempts += 1

            # If we couldn't find a reachable spot after many attempts, place randomly
            if attempts >= max_attempts:
                x, y = self._sample_scatter_cell(self.grid_size)
                while (x, y) in occupied:
                    x, y = self._sample_scatter_cell(self.grid_size)
                occupied.add((x, y))
                f = self._make_flower(j, x, y)
                self.flowers.append(f)

    # ----------------------------------------------------------
    # Step (existing step method remains the same)
    # ----------------------------------------------------------
    def step(self, actions, claims: dict | None = None):
        if not self.agents:
            return {}, {}, {}, {}, {}, None

        rewards = {a: 0.0 for a in self.agents}
        terminated = {a: False for a in self.agents}
        truncated = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        # record actions for HUD/obs
        self.last_actions.update(actions)

        # Battery: silence recharging bees (force DONOTHING)
        for i in range(self.num_bees):
            if self._recharge_until[i] > self.steps:
                actions[f"bee_{i}"] = 0

        # Truncated bees: force DONOTHING unless they can pick up tasks from queue
        for i in range(self.num_bees):
            bee = self.bees[i]
            if bee.truncated:
                # Check if there are unassigned flowers in the pool that this bee could work on
                available_tasks = [
                    j
                    for j, f in enumerate(self.flowers)
                    if not f.harvested and f.assigned_bee is None
                ]

                if available_tasks and self._recharge_until[i] <= self.steps:
                    # Bee can resume work - claim a task from queue
                    # Sort by pollen to prioritize high-value tasks
                    available_tasks.sort(key=lambda j: self.flowers[j].pollen, reverse=True)
                    fj = available_tasks[0]
                    self.flowers[fj].assigned_bee = i
                    bee.assigned_flowers.append(fj)
                    bee.truncated = False  # Resume working
                    print(
                        f"[RESUME] Bee {i} picked up flower {fj} from queue - no longer truncated"
                    )
                else:
                    # No tasks available - force DONOTHING
                    actions[f"bee_{i}"] = 0

        # VFRL heartbeat update - REWARD DONOTHING WHEN APPROPRIATE
        for i in range(self.num_bees):
            a = int(actions.get(f"bee_{i}", 0))
            if a in (1, 2):  # HARVEST or GROOM
                self._last_broadcast_step[i] = self.steps
                self._idle_steps[i] = 0
            else:
                self._idle_steps[i] += 1

                # **NEW: Smart DONOTHING rewards**
                bee = self.bees[i]

                # Default: assume doing nothing is acceptable
                should_do_nothing = True

                # Reason 1: If bee is at/near capacity it MUST groom, so DONOTHING is bad
                if bee.load >= bee.capacity * 0.95:  # At 95% capacity
                    should_do_nothing = False

        # Claim resolution: process retask board claims and assign flower to winner
        if claims:
            self._resolve_claims(claims, rewards)

        # Re-task silent bees
        self._retask_silent_bees()

        # Per-bee retask board propagation (v2)
        self._propagate_retask_board()

        # clear per-step busy flags
        for f in self.flowers:
            f.busy_by = None

        # Helper: is there an in-range assigned flower that would NOT fit?
        def needs_groom_for_fit(bee, bee_id):
            fx, fy, fz = bee.fx, bee.fy, bee.fz
            for j in range(self.num_flowers):
                f = self.flowers[j]
                if f.harvested:
                    continue
                cx, cy = f.center_xy
                dx, dy, dz = fx - cx, fy - cy, fz
                d = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
                if d <= self.harvest_radius and (bee.load + f.pollen) > bee.capacity + 1e-9:
                    return True
            return False

        # -------------------- explicit GROOM action --------------------
        # GROOM serves two purposes:
        #   1. Offload pollen (when load >= 80% capacity)
        #   2. Recharge battery (when battery <= 30% max)
        for agent, a in actions.items():
            if a != 2:
                continue
            i = int(agent.split("_")[1])
            bee = self.bees[i]

            # Check both GROOM conditions
            needs_pollen_offload = bee.load >= bee.capacity * 0.8
            battery_ratio = self._battery[i] / self._battery_max[i] if self._battery_max[i] > 0 else 1.0
            needs_battery_recharge = battery_ratio <= 0.3

            # Already recharging? Can't groom again
            if self._recharge_until[i] > self.steps:
                rewards[agent] -= 0.1  # Small penalty for trying to groom while recharging
                if self.verbose:
                    print(f"[ENV] Bee {i} tried to groom while already recharging")
                continue

            # Priority: Battery recharge takes precedence over pollen offload
            if needs_battery_recharge:
                # Initiate battery recharge via GROOM
                old_battery = self._battery[i]
                self._recharge_until[i] = self.steps + self.recharge_steps
                bee.mode = bee_state.Bee.GROOMING
                bee.groom_cooldown = 1

                # Reward based on how proactive the recharge is (more reward for not waiting until 0)
                recharge_reward = 8.0 + 5.0 * battery_ratio  # 8-9.5 reward (proactive is better)
                rewards[agent] += recharge_reward

                if self.verbose:
                    print(
                        f"[ENV] Bee {i} GROOM→RECHARGE (battery {old_battery:.1f}/{self._battery_max[i]:.1f}, reward: {recharge_reward:.1f})"
                    )

            elif needs_pollen_offload:
                # Offload pollen via GROOM
                old_load = float(bee.load)
                load_ratio = old_load / bee.capacity
                groom_reward = 5.0 + 10.0 * load_ratio  # 5-15 reward based on load

                bee.mode = bee_state.Bee.GROOMING
                bee.groom_cooldown = 1
                rewards[agent] += groom_reward
                bee.load = 0.0

                # Reset harvest failure counter on successful groom
                if hasattr(bee, "_consecutive_harvest_fails"):
                    bee._consecutive_harvest_fails = 0

                if self.verbose:
                    print(
                        f"[ENV] Bee {i} GROOM→OFFLOAD (was {old_load:.1f}/{bee.capacity:.1f}, reward: {groom_reward:.1f})"
                    )
            else:
                # Neither condition met - unnecessary groom
                rewards[agent] -= 0.3  # Moderate penalty for unnecessary groom
                bee.mode = bee_state.Bee.IDLE
                bee.groom_cooldown = 0
                if self.verbose:
                    print(
                        f"[ENV] Bee {i} unnecessary groom (load {bee.load:.1f}/{bee.capacity:.1f}, battery {self._battery[i]:.1f}/{self._battery_max[i]:.1f})"
                    )
        # -------------------- intend HARVEST --------------------
        intents = {}  # bee_id -> (flower_id, dist)
        attempted_harvest = set()
        for agent, a in actions.items():
            i = int(agent.split("_")[1])
            bee = self.bees[i]
            if a == 1:
                # Charging guard
                if self._recharge_until[i] > self.steps:
                    rewards[agent] -= 0.05
                    continue

                attempted_harvest.add(agent)
                if bee.mode == bee_state.Bee.GROOMING:
                    rewards[agent] -= 0.1
                else:
                    fx, fy, fz = bee.fx, bee.fy, bee.fz
                    best, best_d = None, 1e9
                    in_range_any = False

                    for j in range(self.num_flowers):
                        f = self.flowers[j]
                        if f.harvested:
                            continue

                        # Can harvest: flowers assigned to this bee OR unassigned flowers (released by dead bees)
                        if f.assigned_bee is not None and f.assigned_bee != i:
                            continue  # Skip flowers assigned to other active bees

                        cx, cy = f.center_xy
                        dx, dy, dz = fx - cx, fy - cy, fz
                        d = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
                        if d <= self.harvest_radius:
                            current_d = self._min_distance_to_orbit(bee, cx, cy)
                            if current_d <= (self.harvest_radius + self.reach_margin):
                                in_range_any = True
                                if (bee.load + f.pollen) <= bee.capacity + 1e-9:
                                    if d < best_d:
                                        best, best_d = j, d

                    if best is not None:
                        intents[i] = (best, best_d)
                    else:
                        if in_range_any:
                            bee.mode = bee_state.Bee.GROOMING
                            bee.groom_cooldown = 1
                            rewards[agent] += 5.0
                            bee.load = 0.0
                        else:
                            rewards[
                                agent
                            ] -= 1.0  # Stronger penalty to discourage harvest spam (was -0.2)

        # Resolve exclusivity (closest bee wins per flower)
        claims = {}
        for i, (fj, d) in intents.items():
            if fj not in claims or d < claims[fj][1]:
                claims[fj] = (i, d)
        for fj, (iwin, _) in claims.items():
            self.flowers[fj].busy_by = iwin

        # --- Advance orbital motion with battery drain / recharge ---
        for i, b in enumerate(self.bees):
            bee = self.bees[i]
            prev = self._last_pos[i].copy()

            # FIRST: Check if recharge just completed - refill battery BEFORE movement
            if (self._recharge_until[i] == self.steps) and (self._battery[i] == 0.0):
                self._battery[i] = self._battery_max[i]

                # BATTERY RECHARGED: Reassign available unharvested flowers
                available_flowers = [
                    j
                    for j, f in enumerate(self.flowers)
                    if not f.harvested and f.assigned_bee is None
                ]
                if available_flowers:
                    # Sort by pollen descending to prioritize high-value flowers
                    available_flowers.sort(key=lambda j: self.flowers[j].pollen, reverse=True)

                    # Give this bee some flowers (round-robin style)
                    num_to_assign = min(
                        len(available_flowers), max(1, len(available_flowers) // self.num_bees)
                    )
                    for fj in available_flowers[:num_to_assign]:
                        self.flowers[fj].assigned_bee = i
                        if fj not in bee.assigned_flowers:
                            bee.assigned_flowers.append(fj)

                    if self.verbose:
                        print(
                            f"[BATTERY] Bee {i} recharged to {self._battery[i]:.1f} - reassigned {num_to_assign} flowers from pool"
                        )
                else:
                    if self.verbose:
                        print(
                            f"[BATTERY] Bee {i} recharged to {self._battery[i]:.1f} - no flowers available to reassign"
                        )

            if self._recharge_until[i] > self.steps:
                # Charging: freeze position
                b.fx, b.fy, b.fz = prev[0], prev[1], prev[2]
            else:
                b.update_position()
                dx = float(b.fx) - prev[0]
                dy = float(b.fy) - prev[1]
                dz = float(b.fz) - prev[2]
                dist = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
                if dist > 0.0:
                    self._battery[i] -= self.drain_per_step + self.drain_per_unit * dist

                # Battery must reach exactly 0 before bee stops working
                if self._battery[i] <= 0.0:
                    self._battery[i] = 0.0  # Clamp to exactly 0, never negative
                    self._recharge_until[i] = self.steps + self.recharge_steps

                    # =========================================================
                    # NEW: Per-bee retasking - broadcast tasks to nearest bee
                    # =========================================================
                    if bee.assigned_flowers:
                        # Find nearest active bee
                        nearest_bee_id = None
                        min_dist = float("inf")
                        
                        for j in range(self.num_bees):
                            if i == j:
                                continue
                            other = self.bees[j]
                            if other.truncated or self._recharge_until[j] > self.steps:
                                continue  # Skip truncated or recharging bees
                            
                            # Calculate 3D distance
                            dx = other.fx - bee.fx
                            dy = other.fy - bee.fy
                            dz = other.fz - bee.fz
                            dist = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
                            
                            if dist < min_dist:
                                min_dist = dist
                                nearest_bee_id = j
                        
                        # Broadcast to nearest bee with full task metadata
                        tasks_broadcast = 0
                        if nearest_bee_id is not None:
                            for fj in bee.assigned_flowers:
                                if fj < len(self.flowers) and not self.flowers[fj].harvested:
                                    f = self.flowers[fj]
                                    task = {
                                        "flower_id": fj,
                                        "source_bee": i,
                                        "hops": 1,
                                        "received_step": self.steps,
                                        # Full metadata for decision-making
                                        "pollen": f.pollen,
                                        "priority": f.priority,
                                        "window_type": f.window_type,
                                        "window_start": f.window_start,
                                        "window_end": f.window_end,
                                        "x": f.x,
                                        "y": f.y,
                                    }
                                    self.bees[nearest_bee_id].retask_board.append(task)
                                    tasks_broadcast += 1
                            
                            self.bees[nearest_bee_id].last_received_from = i
                            bee.last_broadcast_to = nearest_bee_id
                            
                            if self.verbose:
                                print(
                                    f"[RETASK] Bee {i} dead → broadcast {tasks_broadcast} tasks "
                                    f"to nearest bee {nearest_bee_id} (dist: {min_dist:.2f})"
                                )
                        else:
                            # No active bee found - release to global pool
                            for fj in bee.assigned_flowers:
                                if fj < len(self.flowers) and not self.flowers[fj].harvested:
                                    self.flowers[fj].assigned_bee = None
                                    if self.flowers[fj].busy_by == i:
                                        self.flowers[fj].busy_by = None
                            if self.verbose:
                                print(
                                    f"[RETASK] Bee {i} dead → no active bees, "
                                    f"releasing {len(bee.assigned_flowers)} to global pool"
                                )
                        
                        bee.assigned_flowers = []  # Clear assignments

                    if self.verbose:
                        print(
                            f"[BATTERY] Bee {i} battery = 0 - STOPS working/broadcasting"
                        )

            self._last_pos[i] = np.array([float(b.fx), float(b.fy), float(b.fz)], dtype=float)

        # If orbits changed, refresh reachability + harvestable
        orbit_changed = False
        for i, b in enumerate(self.bees):
            current_i = float(b.i + getattr(b, "inclination_delta", 0.0))
            current_y = float(b.Omega + getattr(b, "yaw_delta", 0.0))
            if abs(current_i - self.last_effective_inclination[i]) > math.radians(0.5) or abs(
                current_y - self.last_effective_yaw[i]
            ) > math.radians(0.5):
                orbit_changed = True
                break

        if orbit_changed:
            self._mark_reachable_per_bee()
            self.harvestable = np.zeros(self.num_flowers, dtype=bool)
            for j, f in enumerate(self.flowers):
                ab = int(f.assigned_bee) if f.assigned_bee is not None else -1
                self.harvestable[j] = ab >= 0 and self.reachable[ab, j]
            for i, b in enumerate(self.bees):
                self.last_effective_inclination[i] = float(
                    b.i + getattr(b, "inclination_delta", 0.0)
                )
                self.last_effective_yaw[i] = float(b.Omega + getattr(b, "yaw_delta", 0.0))

        # Harvest outcomes - FIXED VERSION
        success_map = {a: False for a in self.agents}
        for agent, a in actions.items():
            i = int(agent.split("_")[1])
            bee = self.bees[i]
            if a != 1 or bee.mode == bee_state.Bee.GROOMING:
                continue

            # Check if this bee successfully claimed a flower
            if i in intents:
                fj, _ = intents[i]
                f = self.flowers[fj]

                # Verify this bee actually won the flower and it's in range
                if f.busy_by == i:
                    cx, cy = f.center_xy
                    current_d = self._min_distance_to_orbit(bee, cx, cy)

                    # NEW: Check time window before allowing harvest
                    in_time_window = f.is_harvestable_at_time(self.steps)

                    # Double-check it's actually reachable and in time window
                    if (
                        current_d <= (self.harvest_radius + self.reach_margin)
                        and not f.harvested
                        and in_time_window
                    ):
                        # SUCCESSFUL HARVEST
                        f.harvested = True
                        f.harvested_step = self.steps
                        bee.mode = bee_state.Bee.HARVESTING
                        old_load = bee.load
                        bee.load = min(bee.capacity, bee.load + f.pollen)
                        actual_gain = bee.load - old_load

                        # HARVEST costs 5 energy
                        self._battery[i] = max(0.0, self._battery[i] - 5.0)

                        # If this was an unassigned flower from the pool, claim it
                        if f.assigned_bee is None:
                            f.assigned_bee = i
                            if fj not in bee.assigned_flowers:
                                bee.assigned_flowers.append(fj)
                            print(f"[POOL] Bee {i} claimed unassigned flower {fj} from pool")

                        # BALANCED REWARD: proportional to pollen with efficiency bonuses
                        base_reward = f.pollen * 0.5
                        efficiency_bonus = 2.0 if actual_gain == f.pollen else 1.0
                        capacity_bonus = 3.0 * (1.0 - bee.load / bee.capacity)
                        # NEW: Bonus for hitting time windows
                        window_bonus = (
                            5.0
                            if f.window_type == "HARD"
                            else (2.0 if f.window_type == "SOFT" else 0.0)
                        )
                        total_reward = (
                            base_reward + efficiency_bonus + capacity_bonus + window_bonus
                        )

                        rewards[agent] += total_reward
                        success_map[agent] = True
                        print(
                            f"[ENV] Bee {i} harvested flower {fj} (+{actual_gain:.1f}pollen, reward: {total_reward:.1f}, now {bee.load:.1f}/{bee.capacity:.1f})"
                        )
                    elif not in_time_window:
                        # NEW: Attempted harvest outside time window
                        rewards[agent] -= 2.0
                        print(
                            f"[ENV] Bee {i} MISSED TIME WINDOW for flower {fj} (type: {f.window_type})"
                        )
                        # Mark HARD window as permanently missed
                        if f.window_type == "HARD":
                            f.window_missed = True
                            bee.missed_hard_windows.add(fj)
                    else:
                        # Too far or already harvested - simple fixed penalty
                        rewards[agent] -= 0.5
                        print(
                            f"[ENV] Bee {i} failed harvest - distance {current_d:.2f} > threshold {(self.harvest_radius + self.reach_margin):.2f}"
                        )
                else:
                    # Lost the claim to another bee
                    rewards[agent] -= 0.1
                    print(f"[ENV] Bee {i} lost flower {fj} to bee {f.busy_by}")
            else:
                # No valid target found - simple fixed penalty
                if agent in attempted_harvest:
                    rewards[agent] -= 0.5
                    print(f"[ENV] Bee {i} attempted harvest but no valid target")

        # DONOTHING: no explicit reward/penalty - let harvest success/failure drive behavior
        # (removed progressive penalty tracking)

        # Groom cool down
        for b in self.bees:
            if b.groom_cooldown > 0:
                b.groom_cooldown -= 1
                if b.groom_cooldown == 0:
                    b.mode = bee_state.Bee.IDLE

        # Termination & truncation with NEW conditions

        # TERMINATED: All flowers in the field harvested (global success)
        all_flowers_harvested = all(f.harvested for f in self.flowers)

        # Check per-bee truncation conditions
        for agent in self.agents:
            i = int(agent.split("_")[1])
            bee = self.bees[i]

            # TRUNCATED CONDITION 1: All flowers assigned to this bee are harvested
            if bee.assigned_flowers:
                my_flowers_done = all(self.flowers[fj].harvested for fj in bee.assigned_flowers)
                if my_flowers_done:
                    bee.truncated = True
                    print(f"[TRUNCATED] Bee {i}: All assigned flowers completed")

            # TRUNCATED CONDITION 2: Missed HARD window(s) permanently
            for fj in bee.assigned_flowers:
                f = self.flowers[fj]
                if f.window_type == "HARD" and not f.harvested:
                    # Check if window is permanently missed
                    if self.steps > f.window_end:
                        if fj not in bee.missed_hard_windows:
                            bee.missed_hard_windows.add(fj)
                            bee.truncated = True
                            rewards[agent] -= 50.0  # Severe penalty for missing HARD window
                            print(f"[TRUNCATED] Bee {i}: Missed HARD window for flower {fj}")

            # TRUNCATED CONDITION 3: All SOFT windows for assigned flowers are permanently unreachable
            if bee.assigned_flowers:
                soft_flowers_unreachable = True
                for fj in bee.assigned_flowers:
                    f = self.flowers[fj]
                    if f.harvested:
                        continue
                    if f.window_type != "SOFT":
                        soft_flowers_unreachable = False
                        continue
                    # Check if SOFT window can ever be hit again
                    time_to_next = f.time_until_next_window(self.steps)
                    remaining_steps = self.max_steps - self.steps
                    if time_to_next >= 0 and time_to_next <= remaining_steps:
                        soft_flowers_unreachable = False
                        break

                if soft_flowers_unreachable and not bee.truncated:
                    bee.truncated = True
                    print(f"[TRUNCATED] Bee {i}: No more opportunities for SOFT window flowers")

        # Episode terminates only if:
        # 1. All flowers harvested (success), OR
        # 2. All bees truncated AND no tasks in queue (complete failure)
        all_bees_truncated = all(self.bees[i].truncated for i in range(self.num_bees))
        tasks_in_queue = any(not f.harvested and f.assigned_bee is None for f in self.flowers)

        # Only fail if all bees stopped AND no way to recover (no queue tasks)
        episode_failed = all_bees_truncated and not tasks_in_queue

        terminated = {a: (all_flowers_harvested or episode_failed) for a in self.agents}
        # Individual bee truncation does NOT end episode - just that bee does DONOTHING until queue tasks available
        truncated = {
            a: (self.steps >= self.max_steps) for a in self.agents
        }  # Only max_steps truncates episode

        # Print episode summary when terminating
        if all_flowers_harvested or episode_failed or self.steps >= self.max_steps:
            episode_type = "LOW-BATTERY " if self._is_low_battery_episode else "NORMAL"
            batteries_died = sum(1 for b in self.bees if b.battery <= 0)
            flowers_harvested = sum(1 for f in self.flowers if f.harvested)
            print(f"\n[EPISODE END] Type: {episode_type} | Steps: {self.steps}/{self.max_steps}")
            print(
                f"[EPISODE END] Flowers: {flowers_harvested}/{len(self.flowers)} | Batteries died: {batteries_died}/{self.num_bees}"
            )
            if all_flowers_harvested:
                print("[EPISODE END]  SUCCESS - All flowers harvested!")
            elif episode_failed:
                print("[EPISODE END] FAILED - All bees truncated, no queue tasks")
            elif self.steps >= self.max_steps:
                print("[EPISODE END]   TIMEOUT - Max steps reached")

        # Step count
        self.steps += 1

        # Refresh compact retask board before exposing observations
        try:
            self._update_retask_board()
        except Exception:
            self.retask_board = []

        next_obs = self._get_observations()
        next_global_state = self._get_global_state()
        return next_obs, rewards, terminated, truncated, infos, next_global_state

    # ----------------------------------------------------------
    # Observations
    # ----------------------------------------------------------
    def _get_observations(self):
        obs = {}
        consensus = np.array(
            [self.last_actions[f"bee_{i}"] / 2.0 for i in range(self.num_bees)], dtype=np.float32
        )

        max_pollen = float(max(1, self.bee_capacity))
        for i, bee in enumerate(self.bees):
            fx, fy, fz = float(bee.fx), float(bee.fy), float(bee.fz)
            load_frac = np.clip(bee.load / max(1e-9, bee.capacity), 0.0, 1.0)
            flowers_feat = []
            # Per flower: x, y, pollen_norm, harvested, assigned_to_me, busy_flag, reachable, fits, dist_norm,
            #             is_harvestable_now, is_hard_window, time_to_window (12 features)
            for fj, f in enumerate(self.flowers):
                x = (f.x + 0.5) / self.grid_size
                y = (f.y + 0.5) / self.grid_size
                pol = float(f.pollen) / max_pollen
                harvested = 1.0 if f.harvested else 0.0
                mine = 1.0 if f.assigned_bee == i else 0.0
                busy = 1.0 if f.busy_by is not None else 0.0
                reachable = (
                    1.0
                    if (
                        hasattr(self, "reachable")
                        and self.reachable.shape == (self.num_bees, self.num_flowers)
                        and self.reachable[i, fj]
                    )
                    else 0.0
                )
                # fits: 1.0 if reachable AND fits in bee's remaining capacity
                fits = (
                    1.0
                    if (reachable >= 0.5 and bee.load + f.pollen <= bee.capacity + 1e-9)
                    else 0.0
                )

                # CRITICAL: Calculate 3D distance
                cx, cy = f.center_xy
                dx, dy, dz = fx - cx, fy - cy, fz
                dist_3d = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
                dist_norm = dist_3d / self.harvest_radius  # Normalized 3D distance

                # NEW: Time window features
                is_harvestable_now = 1.0 if f.is_harvestable_at_time(self.steps) else 0.0
                is_hard_window = 1.0 if f.window_type == "HARD" else 0.0
                time_to_window = f.time_until_next_window(self.steps) / max(1.0, self.max_steps)

                flowers_feat.extend(
                    [
                        x,
                        y,
                        pol,
                        harvested,
                        mine,
                        busy,
                        reachable,
                        fits,
                        dist_norm,
                        is_harvestable_now,
                        is_hard_window,
                        time_to_window,
                    ]
                )

            # Build per-bee retask board feature (each bee has its own board now)
            retask_feat = []
            bee_retask_board = bee.retask_board if hasattr(bee, "retask_board") else []
            for task in bee_retask_board[: self.retask_board_size]:
                fj = task.get("flower_id", -1)
                if fj < 0 or fj >= len(self.flowers):
                    retask_feat.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                    continue
                f = self.flowers[fj]
                cx, cy = f.center_xy
                dx, dy, dz = fx - cx, fy - cy, fz
                dist = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
                dist_norm = dist / max(1.0, self.harvest_radius)
                can_fit = 1.0 if (bee.load + f.pollen) <= bee.capacity + 1e-9 else 0.0
                retask_feat.extend(
                    [
                        (f.x + 0.5) / self.grid_size,  # x normalized
                        (f.y + 0.5) / self.grid_size,  # y normalized
                        float(f.pollen) / max_pollen,   # pollen normalized
                        can_fit,                        # can fit in capacity
                        float(task.get("hops", 0)) / 10.0,  # hops normalized (communication chain depth)
                    ]
                )
            # pad to fixed size
            expected_len = 5 * getattr(self, "retask_board_size", 0)
            if len(retask_feat) < expected_len:
                retask_feat.extend([0.0] * (expected_len - len(retask_feat)))

            # compute action_availability: [can_harvest, can_groom, can_do_nothing]
            can_harvest = 0.0
            can_groom = 0.0
            can_do_nothing = 1.0  # Default: can always do nothing

            has_assigned_harvestable = False
            has_large_assigned = False

            for fj, f in enumerate(self.flowers):
                if f.harvested:
                    continue
                reachable = bool(
                    hasattr(self, "reachable")
                    and self.reachable.shape == (self.num_bees, self.num_flowers)
                    and self.reachable[i, fj]
                )
                if not reachable:
                    continue

                # CRITICAL FIX: Check real-time distance using dist_norm from observation
                # dist_norm is at index 8 in the 12-feature vector
                flower_obs_start = fj * 12
                dist_norm = flowers_feat[flower_obs_start + 8]  # Get dist_norm for this flower

                # NEW: Check time window availability
                is_harvestable_now = flowers_feat[flower_obs_start + 9]  # Time window check
                in_time_window = is_harvestable_now >= 0.5

                # Only allow harvest if actually in range (dist_norm < 1.0 means within harvest_radius)
                in_range = dist_norm < 1.0

                # Can harvest flowers assigned to this bee OR unassigned flowers (from pool)
                is_mine = f.assigned_bee == i
                is_unassigned = f.assigned_bee is None  # Released from dead bee's queue

                # Can harvest if: (mine OR unassigned), reachable, in range, in time window, and fits
                if (
                    (is_mine or is_unassigned)
                    and in_range
                    and in_time_window
                    and (bee.load + f.pollen) <= bee.capacity + 1e-9
                ):
                    can_harvest = 1.0
                    if f.assigned_bee == i:
                        has_assigned_harvestable = True

                # Check if there's a large assigned flower that won't fit
                if f.assigned_bee == i and (bee.load + f.pollen) > bee.capacity + 1e-9:
                    has_large_assigned = True

            # Can groom if: (1) loaded >=80%
            # But NOT if load is negligible (<= 5% capacity)
            if bee.load <= bee.capacity * 0.05:
                can_groom = 0.0  # Don't groom with negligible load
            elif bee.load >= bee.capacity * 0.8:
                can_groom = 1.0

            action_availability = np.array(
                [can_harvest, can_groom, can_do_nothing], dtype=np.float32
            )

            obs[f"bee_{i}"] = {
                "position": np.array([fx, fy, fz], dtype=np.float32),
                "status": np.array([float(bee.mode), load_frac], dtype=np.float32),
                "flowers": np.array(flowers_feat, dtype=np.float32),
                "step_count": np.array([self.steps / max(1.0, self.max_steps)], dtype=np.float32),
                "consensus": np.array(consensus, dtype=np.float32),
                "retask_board": np.array(retask_feat, dtype=np.float32),
                "action_availability": action_availability,
            }
        return obs

    # ----------------------------------------------------------
    # Helpers (reachability & global state)
    # ----------------------------------------------------------
    def _make_flower(self, fid, x, y):
        start = np.random.randint(self.time_window_min, self.time_window_max)
        end = start + np.random.randint(15, 40)
        f = bee_state.Flower(fid, self.grid_size, window_start=start, window_end=end)
        f.x = int(np.clip(x, 0, self.grid_size - 1))
        f.y = int(np.clip(y, 0, self.grid_size - 1))
        return f

    def _sample_scatter_cell(self, n):
        return int(np.random.randint(0, n)), int(np.random.randint(0, n))

    def _R1(self, a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

    def _R3(self, a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

    def _min_distance_to_orbit(self, bee, x_center, y_center):
        # In telemetry mode, use direct position distance (no orbital mechanics)
        if getattr(self, '_telemetry_loaded', False):
            dx = bee.fx - x_center
            dy = bee.fy - y_center
            dz = bee.fz
            return math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
        
        cx, cy = self.grid_size / 2.0, self.grid_size / 2.0
        px, py, pz = float(x_center), float(y_center), 0.0
        a, e = float(bee.a), float(bee.e)
        i = float(bee.i + getattr(bee, "inclination_delta", 0.0))
        Om = float(bee.Omega + getattr(bee, "yaw_delta", 0.0))
        om = float(bee.omega)
        R = self._R3(Om) @ self._R1(i) @ self._R3(om)
        bsemi = a * math.sqrt(max(1e-12, 1 - e * e))
        best = np.inf
        for k in range(self.reach_samples):
            nu = 2.0 * math.pi * (k / self.reach_samples)
            r_local = np.array([a * math.cos(nu), bsemi * math.sin(nu), 0.0], float)
            r_world = R @ r_local
            ox, oy, oz = cx + r_world[0], cy + r_world[1], r_world[2]
            dx, dy, dz = px - ox, py - oy, pz - oz
            d = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
            if d < best:
                best = d
        return best

    def _mark_reachable_per_bee(self):
        """Mark which flowers are reachable by each bee with improved detection"""
        B, F = self.num_bees, self.num_flowers
        self.reachable = np.zeros((B, F), dtype=bool)
        thresh = self.harvest_radius + self.reach_margin

        for j, f in enumerate(self.flowers):
            cx, cy = f.center_xy
            for i, b in enumerate(self.bees):
                # Use both current position and orbit minimum for reachability
                current_dx = b.fx - cx
                current_dy = b.fy - cy
                current_dz = b.fz
                current_dist = math.sqrt(
                    current_dx * current_dx
                    + current_dy * current_dy
                    + self.lambda_z * current_dz * current_dz
                )

                orbit_min_dist = self._min_distance_to_orbit(b, cx, cy)

                # Consider reachable if either current position or orbit path is within range
                self.reachable[i, j] = (current_dist <= thresh) or (orbit_min_dist <= thresh)

    def _debug_harvest_conditions(self, bee, flower_id):
        """Debug helper to see why harvest might fail"""
        f = self.flowers[flower_id]
        cx, cy = f.center_xy

        # Current position distance
        dx = bee.fx - cx
        dy = bee.fy - cy
        dz = bee.fz
        current_dist = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)

        # Orbit minimum distance
        orbit_min = self._min_distance_to_orbit(bee, cx, cy)

        print(f"[DEBUG] Flower {flower_id} at ({cx:.1f},{cy:.1f})")
        print(f"  Current distance: {current_dist:.2f} (radius: {self.harvest_radius:.2f})")
        print(
            f"  Orbit min distance: {orbit_min:.2f} (threshold: {self.harvest_radius + self.reach_margin:.2f})"
        )
        print(f"  Harvestable: {current_dist <= self.harvest_radius}")
        print(f"  Reachable: {orbit_min <= (self.harvest_radius + self.reach_margin)}")
        print(f"  Assigned to bee: {f.assigned_bee}")
        print(f"  Already harvested: {f.harvested}")

    def debug_reachability(self):
        """Print reachability matrix for debugging"""
        print("\n=== REACHABILITY MATRIX ===")
        print("Rows: Bees, Columns: Flowers")
        print("Grid size:", self.grid_size)
        print("Harvest radius:", self.harvest_radius)
        print("Reach margin:", self.reach_margin)

        for i, bee in enumerate(self.bees):
            print(f"Bee {i}: a={bee.a:.1f}, e={bee.e:.2f}")

        print("\nReachable matrix:")
        for i in range(self.num_bees):
            row = []
            for j in range(self.num_flowers):
                row.append("X" if self.reachable[i, j] else ".")
            print(f"Bee {i}: {''.join(row)}")

        # Count reachable flowers per bee
        print("\nReachable flowers per bee:")
        for i in range(self.num_bees):
            count = np.sum(self.reachable[i, :])
            print(f"Bee {i}: {count}/{self.num_flowers}")

        # Count bees that can reach each flower
        print("\nBees that can reach each flower:")
        for j in range(self.num_flowers):
            count = np.sum(self.reachable[:, j])
            f = self.flowers[j]
            print(f"Flower {j} at ({f.x},{f.y}): {count} bees")

    def _update_retask_board(self):
        """
        Build a compact fixed-size retask board listing the top-M orphan flowers.
        Each entry is a small dict with fields: flower (id), x, y (normalized),
        priority, reachable (bool), assigned (bee id or -1), min_dist (orbit min).
        The board is sorted by priority desc then min_dist asc and padded to M slots.
        """
        M = max(0, int(getattr(self, "retask_board_size", 0)))
        board = []
        if M == 0:
            self.retask_board = []
            return

        timeout = int(self.retask_timeout_steps)
        for j, f in enumerate(self.flowers):
            if getattr(f, "harvested", False):
                continue

            ab = f.assigned_bee
            orphan = False
            if ab is None:
                orphan = True
            else:
                silent_by_idle = self.count_idle_as_silent and self._idle_steps[ab] >= timeout
                silent_by_heartbeat = (self.steps - self._last_broadcast_step[ab]) >= timeout
                recharging = self._recharge_until[ab] > self.steps
                cannot_reach = not bool(self.reachable[ab, j])
                if silent_by_idle or silent_by_heartbeat or recharging or cannot_reach:
                    orphan = True

            if orphan:
                cx, cy = f.center_xy
                try:
                    min_dist = min([self._min_distance_to_orbit(b, cx, cy) for b in self.bees])
                except Exception:
                    min_dist = float("inf")
                reachable_any = bool(self.reachable[:, j].any())
                board.append(
                    {
                        "flower": int(j),
                        "x": float((f.x + 0.5) / max(1, self.grid_size)),
                        "y": float((f.y + 0.5) / max(1, self.grid_size)),
                        "priority": float(getattr(f, "priority", 0.0)),
                        "min_dist": float(min_dist),
                        "reachable": bool(reachable_any),
                        "assigned": -1 if ab is None else int(ab),
                    }
                )

        # sort by priority desc, then min_dist asc
        board.sort(key=lambda e: (-e["priority"], e["min_dist"]))
        # take top-M and pad
        selected = board[:M]
        while len(selected) < M:
            selected.append(
                {
                    "flower": -1,
                    "x": 0.0,
                    "y": 0.0,
                    "priority": 0.0,
                    "min_dist": float("inf"),
                    "reachable": False,
                    "assigned": -1,
                }
            )
        self.retask_board = selected

    def _get_global_state(self):
        """Enhanced global state with assignments, time windows, and bee modes"""
        bee_feats = []
        for b in self.bees:
            # Extended bee features: position, load, battery, mode
            bee_feats.extend(
                [
                    float(b.fx),
                    float(b.fy),
                    float(b.fz),
                    float(b.load),
                    float(b.capacity),
                    float(getattr(b, "battery", 100.0)),
                ]
            )

        flower_feats = []
        for f in self.flowers:
            # Extended flower features: position, pollen, state, time window info
            flower_feats.extend(
                [float(f.x) + 0.5, float(f.y) + 0.5, float(f.pollen), 1.0 if f.harvested else 0.0]
            )

        # NEW: Assignment matrix (which bee owns which flower)
        assignment_feats = []
        for i in range(self.num_bees):
            for j in range(self.num_flowers):
                is_assigned = self.flowers[j].assigned_bee == i
                assignment_feats.append(1.0 if is_assigned else 0.0)

        # NEW: Time window status for each flower
        time_window_feats = []
        for f in self.flowers:
            time_window_feats.extend(
                [
                    1.0 if f.window_type == "HARD" else 0.0,
                    1.0 if f.is_harvestable_at_time(self.steps) else 0.0,
                    f.time_until_next_window(self.steps) / max(1.0, self.max_steps),  # Normalized
                ]
            )

        # NEW: Reachability matrix
        reachability_feats = []
        if hasattr(self, "reachable") and self.reachable is not None:
            reachability_feats = self.reachable.flatten().tolist()
        else:
            reachability_feats = [0.0] * (self.num_bees * self.num_flowers)

        # Temporal progress
        time_feat = [self.steps / max(1.0, self.max_steps)]

        all_feats = (
            bee_feats
            + flower_feats
            + assignment_feats
            + time_window_feats
            + reachability_feats
            + time_feat
        )
        return np.nan_to_num(np.array(all_feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # ---------- SGP4 bulk toggler ----------
    def _apply_sgp4_mode(
        self, use_sgp4: bool, tle_lines: list | None, km_per_unit: float, stagger_seconds: int
    ):
        # If Bee doesn't have helpers, nothing to do
        if not hasattr(self.bees[0], "set_tle") or not hasattr(self.bees[0], "disable_tle"):
            return

        if not use_sgp4:
            for b in self.bees:
                try:
                    b.disable_tle()
                except Exception:
                    pass
            if self.verbose:
                print("[SGP4] disabled for all bees")
            return

        if not tle_lines or not isinstance(tle_lines, list):
            if self.verbose:
                print("[SGP4] No TLE lines provided; keeping Kepler mode.")
            return

        now = datetime.now(timezone.utc)
        for i, b in enumerate(self.bees):
            try:
                if i < len(tle_lines):
                    l1, l2 = tle_lines[i]
                else:
                    l1, l2 = tle_lines[0]
                start_t = now + timedelta(seconds=i * max(0, int(stagger_seconds)))
                b.set_tle(str(l1), str(l2), start_utc=start_t, km_per_unit=float(km_per_unit))
            except Exception as e:
                if self.verbose:
                    print(f"[SGP4] enable failed for bee_{i}: {e}")

    # ---------- Claim resolution helper ----------
    def _resolve_claims(self, claims: dict, rewards: dict | None = None):
        """
        Process claims from retask board.
        claims: {agent_name -> board_slot_idx or None}
        rewards: reward dict to apply anti-spam penalty

        For each claimed board slot, resolve conflicts (closest bee wins or highest logit).
        Winner gets temporarily assigned to that flower if it's orphaned.
        """
        if not claims or not self.retask_board:
            return

        # Temporarily disable blanket anti-spam penalty - it was discouraging valid claims
        # Instead rely on DONOTHING reward and high harvest bonus to guide behavior
        if rewards is None:
            rewards = {}

        # Group claims by flower
        claims_by_flower = {}  # flower_idx -> list of (bee_id, agent_name)

        for agent, slot_idx in claims.items():
            if slot_idx is None:
                continue  # no-claim

            # Get bee_id from agent name
            try:
                bee_id = int(agent.split("_")[1])
            except (ValueError, IndexError):
                continue

            # Get flower from retask board
            if 0 <= slot_idx < len(self.retask_board):
                slot = self.retask_board[slot_idx]
                flower_idx = slot.get("flower", -1)
                if flower_idx >= 0 and flower_idx < self.num_flowers:
                    if flower_idx not in claims_by_flower:
                        claims_by_flower[flower_idx] = []
                    claims_by_flower[flower_idx].append((bee_id, agent))

        # Resolve conflicts and assign winners
        for flower_idx, claimants in claims_by_flower.items():
            f = self.flowers[flower_idx]

            # Check if flower is still orphaned / unclaimed
            if getattr(f, "harvested", False):
                continue

            orphan = False
            if f.assigned_bee is None:
                orphan = True
            else:
                timeout = int(self.retask_timeout_steps)
                silent_by_idle = (
                    self.count_idle_as_silent and self._idle_steps[f.assigned_bee] >= timeout
                )
                silent_by_heartbeat = (
                    self.steps - self._last_broadcast_step[f.assigned_bee]
                ) >= timeout
                recharging = self._recharge_until[f.assigned_bee] > self.steps
                cannot_reach = not bool(self.reachable[f.assigned_bee, flower_idx])
                if silent_by_idle or silent_by_heartbeat or recharging or cannot_reach:
                    orphan = True

            if not orphan:
                continue  # flower is not orphaned, skip

            # Winner is closest bee (by orbit min distance)
            winner_bee_id = None
            best_dist = float("inf")
            for bee_id, agent in claimants:
                try:
                    cx, cy = f.center_xy
                    dist = self._min_distance_to_orbit(self.bees[bee_id], cx, cy)
                    if dist < best_dist:
                        best_dist = dist
                        winner_bee_id = bee_id
                except Exception:
                    pass

            if winner_bee_id is not None:
                # Assign flower to winner
                f.assigned_bee = winner_bee_id
                if winner_bee_id not in self.assignments:
                    self.assignments[winner_bee_id] = []
                if flower_idx not in self.assignments[winner_bee_id]:
                    self.assignments[winner_bee_id].append(flower_idx)
                    self.bees[winner_bee_id].assigned_flowers.append(flower_idx)

    # ---------- Per-bee retask board propagation (v2) ----------
    def _propagate_retask_board(self):
        """
        Enhanced task propagation via communication chain between nearby bees.
        
        DECISION LOGIC:
        Each bee evaluates incoming tasks based on:
        1. Can it reach the flower (orbit reachability)?
        2. Does it have capacity for the pollen?
        3. Can it meet the deadline (time window)?
        4. Is it worth taking given current workload vs priority?
        
        If a bee CAN'T accept a task, it holds it until close to another active bee,
        then passes the task along. This continues until:
        - A bee accepts the task, OR
        - The task expires (deadline missed), OR
        - The episode ends
        """
        comm_threshold = self.harvest_radius * 2.0  # bees within this distance can communicate
        
        for i, bee in enumerate(self.bees):
            if not bee.retask_board or bee.truncated:
                continue
            if self._recharge_until[i] > self.steps:
                continue  # Recharging bees don't process retask board
            
            # Filter out stale tasks (older than 100 steps) or expired deadlines
            valid_tasks = []
            for t in bee.retask_board:
                if self.steps - t["received_step"] >= 100:
                    continue  # Too old
                fj = t["flower_id"]
                if fj >= len(self.flowers):
                    continue
                f = self.flowers[fj]
                if f.harvested:
                    continue
                # Check if deadline permanently missed (HARD window)
                if f.window_type == "HARD" and self.steps > f.window_end:
                    continue  # Deadline missed, discard task
                valid_tasks.append(t)
            bee.retask_board = valid_tasks
            
            # Try to perform tasks from retask board
            tasks_accepted = []
            tasks_to_handoff = []
            
            for task_idx, task in enumerate(bee.retask_board):
                fj = task["flower_id"]
                f = self.flowers[fj]
                cx, cy = f.center_xy
                
                # ========== DECISION CRITERIA ==========
                
                # 1. REACHABILITY: Can this bee's orbit reach the flower?
                orbit_dist = self._min_distance_to_orbit(bee, cx, cy)
                can_reach = orbit_dist <= (self.harvest_radius + self.reach_margin)
                
                # 2. CAPACITY: Does bee have room for pollen?
                can_fit = (bee.load + f.pollen) <= bee.capacity + 1e-9
                
                # 3. DEADLINE: Can bee reach flower before deadline?
                can_meet_deadline = True
                time_to_window = f.time_until_next_window(self.steps)
                if f.window_type == "HARD":
                    # Estimate steps to reach flower (rough approximation)
                    steps_remaining = f.window_end - self.steps
                    # Conservative: assume bee needs ~10 steps minimum to get there
                    if steps_remaining < 10:
                        can_meet_deadline = False
                    elif time_to_window < 0:  # Window already missed
                        can_meet_deadline = False
                
                # 4. WORKLOAD vs PRIORITY: Should bee take on more work?
                # Compute a score: high priority + urgent deadline = should accept
                priority_score = task.get("priority", f.priority)
                urgency_score = 0.0
                if f.window_type == "HARD" and f.window_end > self.steps:
                    # Higher urgency as deadline approaches
                    urgency_score = 1.0 - (f.window_end - self.steps) / self.max_steps
                
                task_score = priority_score + urgency_score * 2.0  # Weight urgency higher
                
                # Current workload: how many pending flowers does bee have?
                current_workload = len([
                    fk for fk in bee.assigned_flowers 
                    if fk < len(self.flowers) and not self.flowers[fk].harvested
                ])
                
                # Accept if: can reach, can fit, can meet deadline, AND
                # (workload is low OR task is high-priority/urgent)
                workload_ok = current_workload < 5  # Max 5 pending tasks
                priority_ok = task_score >= 1.0  # Accept high-priority regardless
                
                should_accept = can_reach and can_fit and can_meet_deadline and (workload_ok or priority_ok)
                
                if should_accept:
                    # ACCEPT: This bee claims the task
                    f.assigned_bee = i
                    if fj not in bee.assigned_flowers:
                        bee.assigned_flowers.append(fj)
                    tasks_accepted.append(task_idx)
                    if self.verbose:
                        print(
                            f"[COMM] Bee {i} ACCEPTS task (flower {fj}) "
                            f"[priority={priority_score:.1f}, urgency={urgency_score:.1f}, "
                            f"hops={task['hops']}, src_bee={task['source_bee']}]"
                        )
                else:
                    # REJECT: Cannot or should not accept - mark for handoff
                    reason = []
                    if not can_reach:
                        reason.append("unreachable")
                    if not can_fit:
                        reason.append("no_capacity")
                    if not can_meet_deadline:
                        reason.append("deadline_miss")
                    if not workload_ok and not priority_ok:
                        reason.append("overloaded")
                    
                    task["reject_reason"] = ",".join(reason)
                    tasks_to_handoff.append(task_idx)
                    bee.awaiting_handoff = True
            
            # Remove accepted tasks from board
            for idx in sorted(tasks_accepted, reverse=True):
                if idx < len(bee.retask_board):
                    bee.retask_board.pop(idx)
            
            # ========== HANDOFF TO NEARBY BEE ==========
            if bee.awaiting_handoff and tasks_to_handoff:
                # Find nearest active bee to pass tasks to
                nearest_bee_id = None
                min_dist = float("inf")
                
                for j in range(self.num_bees):
                    if i == j:
                        continue
                    other = self.bees[j]
                    if other.truncated or self._recharge_until[j] > self.steps:
                        continue
                    
                    dx = other.fx - bee.fx
                    dy = other.fy - bee.fy
                    dz = other.fz - bee.fz
                    dist = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
                    
                    if dist < comm_threshold and dist < min_dist:
                        min_dist = dist
                        nearest_bee_id = j
                
                # If found nearby bee, handoff tasks
                if nearest_bee_id is not None:
                    tasks_passed = 0
                    last_task = None
                    for idx in sorted(tasks_to_handoff, reverse=True):
                        if idx < len(bee.retask_board):
                            task = bee.retask_board.pop(idx)
                            task["hops"] += 1
                            task["received_step"] = self.steps
                            # Preserve flower metadata for next bee's decision
                            fj = task["flower_id"]
                            if fj < len(self.flowers):
                                f = self.flowers[fj]
                                task["priority"] = f.priority
                                task["window_type"] = f.window_type
                                task["window_end"] = f.window_end
                                task["pollen"] = f.pollen
                            self.bees[nearest_bee_id].retask_board.append(task)
                            tasks_passed += 1
                            last_task = task
                    
                    if tasks_passed > 0:
                        self.bees[nearest_bee_id].last_received_from = i
                        bee.last_broadcast_to = nearest_bee_id
                        if self.verbose:
                            print(
                                f"[COMM] Bee {i} → Bee {nearest_bee_id}: passed {tasks_passed} tasks "
                                f"(dist: {min_dist:.2f}, hops: {last_task['hops'] if last_task else 0})"
                            )
                    
                    bee.awaiting_handoff = False
                # else: Keep tasks on this bee's board until a nearby bee is found

    # ---------- Re-task helper ----------
    def _retask_silent_bees(self):
        timeout = int(self.retask_timeout_steps)
        if timeout <= 0:
            return
        for i in range(self.num_bees):
            silent_by_idle = self.count_idle_as_silent and self._idle_steps[i] >= timeout
            silent_by_heartbeat = (self.steps - self._last_broadcast_step[i]) >= timeout
            if not (silent_by_idle or silent_by_heartbeat):
                continue
            for fj in list(self.assignments[i]):
                f = self.flowers[fj]
                if not getattr(f, "harvested", False):
                    f.assigned_bee = None
            self.assignments[i].clear()
