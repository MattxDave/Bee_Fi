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

# Physics-based gossip relay (replaces chain relay v3)
try:
    from gossiper import Gossiper
except Exception:
    Gossiper = None

# Basilisk orbital dynamics interface
try:
    from bsk_interface import BSKInterface
except Exception:
    BSKInterface = None

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

    PER-BEE GOSSIP RELAY (physics-based communication):
    - When a bee dies, orphan tasks are broadcast via GossipMessage
    - Messages propagate one hop per step to nearest reachable bee
    - Range-limited: bees must be within harvest_radius * 1.5
    - Line-of-sight: optional Earth occlusion (ray-sphere test)
    - Receiving bee evaluates task acceptance based on:
      * Orbit reachability (can bee's orbit reach the flower?)
      * Capacity (does bee have room for pollen?)
      * Deadline (can bee reach flower before HARD window closes?)
      * Workload vs Priority (is task worth taking given current assignments?)
    - If bee REJECTS: message forwarded to next unseen reachable bee
    - Messages expire after max_hops or max_age steps

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
        time_window_max=150,
        harvest_radius=18.0,  # Euclidean distance from current bee position to flower
        lambda_z=0.1,
        knn_k=3,
        orbit_scale=1.2,
        spawn_on_orbit_ratio=0.8,
        shaping_weight=0.05,
        anti_spam_pen=-0.005,
        reach_margin=5.0,  # Increased margin for orbital reachability
        reach_samples=48,
        bee_capacity: float = 10.0,
        # ---------- SGP4 toggles ----------
        use_sgp4: bool = False,
        sgp4_km_per_unit: float = 300.0,
        sgp4_time_stagger_s: int = 60,
        tle_lines: list | None = None,
        # ---------- Basilisk toggles ----------
        use_basilisk: bool = False,
        bsk_dt_sec: float = 5.0,
        bsk_battery_wh: float = 200.0,
        bsk_power_draw_w: float = 3.0,
        bsk_orbital_elements: list | None = None,
        bsk_meters_per_unit: float = 200_000.0,
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
        low_battery_chance: float = 0.05,  # Probability of low-battery training episode
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

        # Basilisk
        self.use_basilisk = bool(use_basilisk)
        self.bsk_dt_sec = float(bsk_dt_sec)
        self.bsk_battery_wh = float(bsk_battery_wh)
        self.bsk_power_draw_w = float(bsk_power_draw_w)
        self.bsk_orbital_elements = bsk_orbital_elements
        self.bsk_meters_per_unit = float(bsk_meters_per_unit)
        self._bsk: object | None = None  # BSKInterface instance

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
        self.low_battery_chance = float(low_battery_chance)
        self._episode_drain_per_step = float(drain_per_step)  # Per-episode drain (set in reset)

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

        # HRL: Current goals for each bee (set by manager policy)
        # Goals: 0=IDLE, 1=HARVEST, 2=GROOM, 3=ASSIST
        self._current_goals = np.zeros(self.num_bees, dtype=int)
        self._goal_set_step = np.zeros(self.num_bees, dtype=int)  # Step when goal was set

        # Gossiper (physics-based relay, replaces chain relay v3)
        if Gossiper is not None:
            self.gossiper = Gossiper(
                env=self,
                max_range=self.harvest_radius * 1.5,
                max_hops=self.num_bees,
                max_age=200,
                cooldown_steps=3,
                earth_radius=0.0,  # flat-grid mode; set >0 for Earth occlusion
            )
            self.gossiper.reset(self.num_bees)
        else:
            self.gossiper = None

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

            # Populate task metadata for each flower
            for j, f in enumerate(self.flowers):
                f.task_id = f"TASK-{f.id:03d}-{f.x}-{f.y}"
                f.task_description = bee_state.TASK_DESCRIPTIONS[j % len(bee_state.TASK_DESCRIPTIONS)]
                f.created_step = 0
                f.status = "unassigned"
                # Deadline scaled by priority: high-priority → tighter deadline
                # Range: 70% (highest priority) to 95% (lowest priority) of max_steps
                f.deadline_step = int(self.max_steps * (0.7 + 0.25 * (1.0 - f.priority)))

        # Round-robin assignments by descending pollen (for both modes)
        order = sorted(range(self.num_flowers), key=lambda j: self.flowers[j].pollen, reverse=True)
        self.assignments = [[] for _ in range(self.num_bees)]
        for idx, fj in enumerate(order):
            bid = idx % self.num_bees
            self.assignments[bid].append(fj)
            self.flowers[fj].assigned_bee = bid
            self.flowers[fj].status = "assigned"
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

        # Curriculum learning: 30% of episodes use low-battery mode
        if self.low_battery_chance > 0 and random.random() < self.low_battery_chance:
            # Use local variable instead of mutating class attribute
            episode_drain_per_step = self.drain_per_step * random.uniform(2.0, 3.0)
            episode_battery_min_steps = int(self.battery_min_steps * 0.33)
            episode_battery_max_steps = int(self.battery_max_steps * 0.33)
            self.episode_type = "LOW_BATTERY"
        else:
            episode_drain_per_step = self.drain_per_step
            episode_battery_min_steps = self.battery_min_steps
            episode_battery_max_steps = self.battery_max_steps

        # Apply episode-local battery settings
        for i in range(self.num_bees):
            cap = random.randint(episode_battery_min_steps, episode_battery_max_steps)
            self._battery_max[i] = float(cap)
            self._battery[i] = float(cap)
        self._episode_drain_per_step = episode_drain_per_step  # Store for use in step()
        self._recharge_until[:] = -1
        for i, b in enumerate(self.bees):
            self._last_pos[i] = np.array([float(b.fx), float(b.fy), float(b.fz)], dtype=float)

        # Reachability + harvestable mask
        self._cached_orbit_points = None  # reset orbit cache for new episode
        self._mark_reachable_per_bee()
        self.harvestable = np.zeros(self.num_flowers, dtype=bool)
        for j, f in enumerate(self.flowers):
            ab = int(f.assigned_bee) if f.assigned_bee is not None else -1
            self.harvestable[j] = ab >= 0 and self.reachable[ab, j]

        # Debug reachability (optional - comment out for production)
        # self.debug_reachability()

        # HRL: Reset goals to IDLE (0) for all bees
        self._current_goals = np.zeros(self.num_bees, dtype=int)
        self._goal_set_step = np.zeros(self.num_bees, dtype=int)

        # Initialize retask board for the start of an episode
        try:
            self._update_retask_board()
        except Exception:
            # don't fail reset if update logic has an unexpected issue
            self.retask_board = []

        # Reset gossiper for new episode
        if getattr(self, 'gossiper', None) is not None:
            self.gossiper.reset(self.num_bees)

        # Initialize / reset Basilisk simulation
        if self.use_basilisk and BSKInterface is not None and BSKInterface.is_available():
            try:
                bsk = BSKInterface(
                    num_sats=self.num_bees,
                    dt_sec=self.bsk_dt_sec,
                    battery_wh=self.bsk_battery_wh,
                    power_draw_w=self.bsk_power_draw_w,
                )
                if self.bsk_orbital_elements is not None:
                    bsk.configure_orbits(self.bsk_orbital_elements)
                bsk.initialize()

                # Set initial bee positions from BSK
                grid_pos = bsk.get_positions_grid(
                    self.grid_size, self.bsk_meters_per_unit
                )
                for i, b in enumerate(self.bees):
                    b.fx = float(grid_pos[i, 0])
                    b.fy = float(grid_pos[i, 1])
                    b.fz = float(grid_pos[i, 2])
                    self._last_pos[i] = grid_pos[i]

                # ─── Sync Bee Kepler elements from BSK orbital elements ───
                # This ensures _precompute_orbit_points(), flower placement,
                # and reachability all use the REAL BSK orbits, not defaults.
                bsk_oe = bsk._orbital_elements  # list[dict] with a_m, e, i_deg, ...
                for i, b in enumerate(self.bees):
                    oe = bsk_oe[i]
                    # Convert semi-major axis: metres → grid units
                    b.a = oe["a_m"] / self.bsk_meters_per_unit
                    b.a_units = b.a
                    b.e = oe["e"]
                    b.i = math.radians(oe["i_deg"])
                    b.Omega = math.radians(oe["Omega_deg"])
                    b.omega = math.radians(oe["omega_deg"])
                    b.inclination_delta = 0.0
                    b.yaw_delta = 0.0
                    # Update derived Kepler state
                    b.a_km = max(1e-6, b.a * b.km_per_unit)
                    b.n = math.sqrt(b.mu / (b.a_km ** 3))
                    b.M = math.radians(oe["f_deg"])  # approx: use true anomaly as mean

                # Invalidate orbit cache so new elements are used
                self._cached_orbit_points = None
                self._bsk = bsk  # Set early so flower placement can detect BSK mode

                # Re-place flowers using BSK-synced orbits
                self.flowers = []
                self._ensure_reachable_flower_placement()

                # Re-assign pollen amounts
                base = np.arange(1, self.num_flowers + 1, dtype=float)
                perm = np.random.permutation(base)
                scale = self.bee_capacity / self.num_flowers
                amounts = perm * scale
                for j, f in enumerate(self.flowers):
                    f.pollen = float(amounts[j])
                    f.priority = f.pollen / self.bee_capacity

                # Populate task metadata for BSK-created flowers
                for j, f in enumerate(self.flowers):
                    f.task_id = f"TASK-{f.id:03d}-{f.x}-{f.y}"
                    f.task_description = bee_state.TASK_DESCRIPTIONS[j % len(bee_state.TASK_DESCRIPTIONS)]
                    f.created_step = 0
                    f.status = "unassigned"
                    # Deadline scaled by priority: high-priority → tighter deadline
                    # Range: 70% (highest priority) to 95% (lowest priority) of max_steps
                    f.deadline_step = int(self.max_steps * (0.7 + 0.25 * (1.0 - f.priority)))

                # Re-do round-robin assignments
                order = sorted(
                    range(self.num_flowers),
                    key=lambda j: self.flowers[j].pollen,
                    reverse=True,
                )
                self.assignments = [[] for _ in range(self.num_bees)]
                for idx, fj in enumerate(order):
                    bid = idx % self.num_bees
                    self.assignments[bid].append(fj)
                    self.flowers[fj].assigned_bee = bid
                    self.flowers[fj].status = "assigned"
                for i_b, b in enumerate(self.bees):
                    b.assigned_flowers = self.assignments[i_b][:]

                # Re-compute reachability with BSK-synced orbits
                self._cached_orbit_points = None
                self._mark_reachable_per_bee()
                self.harvestable = np.zeros(self.num_flowers, dtype=bool)
                for j, f in enumerate(self.flowers):
                    ab = int(f.assigned_bee) if f.assigned_bee is not None else -1
                    self.harvestable[j] = ab >= 0 and self.reachable[ab, j]

                if self.verbose:
                    n_reach = int(self.reachable.any(axis=0).sum())
                    print(
                        f"[BSK] Initialized {self.num_bees}-sat Basilisk simulation  "
                        f"({n_reach}/{self.num_flowers} flowers reachable)"
                    )
            except Exception as e:
                if self.verbose:
                    print(f"[BSK] Init failed, falling back to Keplerian: {e}")
                self._bsk = None
        else:
            self._bsk = None

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
            "meters_per_unit": 200_000,  # 200km per grid unit - orbits span the grid
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

        # Reinitialize arrays that depend on num_bees/num_flowers
        self._current_goals = np.zeros(self.num_bees, dtype=int)

        # Store telemetry metadata
        self._telemetry_metadata = metadata

        if self.verbose:
            print(f"[TELEMETRY] Loaded {len(bees)} bees, {len(flowers)} flowers from {telemetry_path}")
            print(f"[TELEMETRY] Failed satellites: {metadata.get('failed_satellites', [])}")
            print(f"[TELEMETRY] Tasks moved: {metadata.get('total_tasks_moved', 0)}")

        return metadata

    def _ensure_reachable_flower_placement(self):
        """Ensure flowers are placed in locations reachable by at least one bee.
        
        In BSK mode, flowers are placed in the central region of the grid
        (the 'ground' area) while bees orbit around it.
        In non-BSK mode, flowers are placed near orbital paths.
        """
        occupied = set()

        # In BSK mode, place flowers in the central region so bees orbit around them
        use_central_placement = getattr(self, '_bsk', None) is not None

        for j in range(self.num_flowers):
            attempts = 0
            max_attempts = 50

            while attempts < max_attempts:
                if use_central_placement:
                    # BSK mode: flowers in central 60% of grid (the "ground")
                    margin = self.grid_size * 0.2  # 20% margin on each side
                    x = int(random.uniform(margin, self.grid_size - margin))
                    y = int(random.uniform(margin, self.grid_size - margin))
                elif random.random() < self.spawn_on_orbit_ratio:
                    # Non-BSK: Spawn near orbital paths
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
                    if use_central_placement:
                        # BSK mode: skip orbit reachability check — orbital dynamics
                        # handle access windows; all central flowers are valid targets
                        reachable_by_any = True
                    else:
                        # Check if this position is reachable by any bee's orbit path
                        reachable_by_any = False
                        flower_center_x, flower_center_y = x + 0.5, y + 0.5

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
    def step(self, actions, claims: dict | None = None, targets: dict | None = None, gossip_claims: dict | None = None):
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
                    if self.verbose:
                        print(
                            f"[RESUME] Bee {i} picked up flower {fj} from queue - no longer truncated"
                        )
                else:
                    # No tasks available - force DONOTHING
                    actions[f"bee_{i}"] = 0

        # VFRL heartbeat update - REWARD DONOTHING WHEN APPROPRIATE
        for i in range(self.num_bees):
            agent_id = f"bee_{i}"
            a = int(actions.get(agent_id, 0))
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

                # Check if a valid harvest exists right now for this bee
                has_harvest_now = False
                fx, fy, fz = bee.fx, bee.fy, bee.fz
                for fj, f in enumerate(self.flowers):
                    if f.harvested or f.expired:
                        continue
                    if f.assigned_bee is not None and f.assigned_bee != i:
                        continue
                    reachable = bool(
                        hasattr(self, "reachable")
                        and self.reachable.shape == (self.num_bees, self.num_flowers)
                        and self.reachable[i, fj]
                    )
                    if not reachable:
                        continue
                    cx, cy = f.center_xy
                    dx, dy, dz = fx - cx, fy - cy, fz
                    d = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
                    in_range = d <= self.harvest_radius
                    in_time_window = f.is_harvestable_at_time(self.steps)
                    fits = (bee.load + f.pollen) <= bee.capacity + 1e-9
                    if in_range and in_time_window and fits:
                        has_harvest_now = True
                        break

                battery_ratio = (
                    self._battery[i] / self._battery_max[i]
                    if self._battery_max[i] > 0
                    else 1.0
                )
                needs_groom = (bee.load >= bee.capacity * 0.8) or (self._battery[i] <= 0.0)

                if needs_groom or not should_do_nothing:
                    rewards[agent_id] -= 0.02
                elif has_harvest_now:
                    rewards[agent_id] -= 0.01
                else:
                    rewards[agent_id] += 0.01

        # Re-task silent bees
        self._retask_silent_bees()

        # Gossip relay: advance messages one hop per step
        # If gossip_claims provided, use learned policy decisions instead of heuristic
        if getattr(self, 'gossiper', None) is not None:
            learned = None
            if gossip_claims is not None:
                # Convert agent-name keys to int bee ids for gossiper
                learned = {}
                for agent, slot in gossip_claims.items():
                    try:
                        bid = int(agent.split("_")[1]) if isinstance(agent, str) else int(agent)
                    except (ValueError, IndexError):
                        continue
                    learned[bid] = slot
            self.gossiper.propagate(self.steps, learned_claims=learned)
        else:
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
        #   2. Recharge battery (when battery is DEAD, i.e. reaches 0)
        for agent, a in actions.items():
            if a != 2:
                continue
            i = int(agent.split("_")[1])
            bee = self.bees[i]

            # Check both GROOM conditions
            needs_pollen_offload = bee.load >= bee.capacity * 0.8
            needs_battery_recharge = self._battery[i] <= 0.0  # Battery must be dead (at 0)

            # Already recharging (or completion step)? Can't groom again
            if self._recharge_until[i] >= self.steps:
                rewards[agent] -= 0.1  # Small penalty for trying to groom while recharging
                if self.verbose:
                    if self.verbose:
                        print(f"[ENV] Bee {i} tried to groom while already recharging")
                continue

            # Priority: Battery recharge takes precedence over pollen offload
            if needs_battery_recharge:
                # Battery is dead (0) — initiate recharge via GROOM
                self._recharge_until[i] = self.steps + self.recharge_steps
                bee.mode = bee_state.Bee.GROOMING
                bee.groom_cooldown = 1

                # Reward for grooming to recharge when battery is dead
                recharge_reward = 8.0
                rewards[agent] += recharge_reward

                if self.verbose:
                    print(
                        f"[ENV] Bee {i} GROOM→RECHARGE (battery DEAD, reward: {recharge_reward:.1f})"
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
                rewards[agent] -= 0.1  # Small penalty for unnecessary groom
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

                    # ── Proactive groom: if load is too high for the smallest
                    #    reachable unharvested flower, offload FIRST ──
                    smallest_reachable_pollen = float('inf')
                    for j in range(self.num_flowers):
                        f = self.flowers[j]
                        if f.harvested or f.expired:
                            continue
                        if f.assigned_bee is not None and f.assigned_bee != i:
                            continue
                        cx, cy = f.center_xy
                        dx, dy, dz = fx - cx, fy - cy, fz
                        d = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
                        if d <= self.harvest_radius and f.pollen < smallest_reachable_pollen:
                            smallest_reachable_pollen = f.pollen

                    if (smallest_reachable_pollen < float('inf')
                            and (bee.load + smallest_reachable_pollen) > bee.capacity + 1e-9):
                        # Can't fit even the smallest in-range flower → offload first
                        old_load = float(bee.load)
                        load_ratio = old_load / bee.capacity if bee.capacity > 0 else 0
                        groom_reward = 5.0 + 10.0 * load_ratio
                        bee.mode = bee_state.Bee.GROOMING
                        bee.groom_cooldown = 1
                        bee.load = 0.0
                        rewards[agent] += groom_reward
                        if self.verbose:
                            print(
                                f"[ENV] Bee {i} PROACTIVE GROOM before harvest "
                                f"(was {old_load:.1f}/{bee.capacity:.1f}, "
                                f"smallest flower needs {smallest_reachable_pollen:.1f}, "
                                f"reward: {groom_reward:.1f})"
                            )
                        continue  # skip harvest intent — will harvest next step

                    # ============================================================
                    # FLOWER TARGETING: Use policy-selected target if provided,
                    # otherwise fall back to nearest valid flower.
                    # ============================================================
                    target_fj = None
                    if targets is not None:
                        target_fj = targets.get(agent, targets.get(i, None))

                    best, best_d = None, 1e9
                    in_range_any = False

                    # If the policy chose a specific flower, validate it first
                    if target_fj is not None and 0 <= target_fj < self.num_flowers:
                        f = self.flowers[target_fj]
                        if not f.harvested and not f.expired:
                            if f.assigned_bee is None or f.assigned_bee == i:
                                cx, cy = f.center_xy
                                dx, dy, dz = fx - cx, fy - cy, fz
                                d = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
                                if d <= self.harvest_radius:
                                    in_range_any = True
                                    if (bee.load + f.pollen) <= bee.capacity + 1e-9:
                                        best, best_d = target_fj, d

                    # Fallback: if chosen target is invalid, pick nearest valid
                    if best is None:
                        for j in range(self.num_flowers):
                            f = self.flowers[j]
                            if f.harvested or f.expired:
                                continue

                            # Can only harvest flowers assigned to this bee OR
                            # unassigned flowers (claimed from retask board)
                            if f.assigned_bee is not None and f.assigned_bee != i:
                                continue  # Skip flowers assigned to other bees

                            cx, cy = f.center_xy
                            dx, dy, dz = fx - cx, fy - cy, fz
                            # Use Euclidean distance from current position
                            d = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
                            if d <= self.harvest_radius:
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
                            ] -= 0.2  # Mild penalty for harvest when nothing in range

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
                if self.verbose:
                    print(f"[BATTERY] Bee {i} recharged to {self._battery[i]:.1f}")

            if self._recharge_until[i] > self.steps:
                # Charging: freeze position
                b.fx, b.fy, b.fz = prev[0], prev[1], prev[2]
            elif getattr(self, '_bsk', None) is not None:
                # Basilisk mode: positions updated in batch below
                pass
            else:
                b.update_position()
                dx = float(b.fx) - prev[0]
                dy = float(b.fy) - prev[1]
                dz = float(b.fz) - prev[2]
                dist = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
                if dist > 0.0:
                    self._battery[i] -= self._episode_drain_per_step + self.drain_per_unit * dist

                # Battery must reach exactly 0 before bee stops working
                if self._battery[i] <= 0.0:
                    self._battery[i] = 0.0  # Clamp to exactly 0, never negative
                    self._recharge_until[i] = self.steps + self.recharge_steps

                    # =========================================================
                    # GOSSIP RELAY: Dead bee broadcasts orphan tasks
                    # =========================================================
                    if bee.assigned_flowers:
                        orphan_flowers = [
                            fj for fj in bee.assigned_flowers
                            if fj < len(self.flowers) and not self.flowers[fj].harvested
                        ]
                        if orphan_flowers:
                            if getattr(self, 'gossiper', None) is not None:
                                self.gossiper.broadcast_tasks(i, orphan_flowers, self.steps)
                            else:
                                # Fallback: just unassign flowers to pool
                                for fj in orphan_flowers:
                                    self.flowers[fj].assigned_bee = None
                                if self.verbose:
                                    print(
                                        f"[RELAY] Bee {i} dead → {len(orphan_flowers)} "
                                        f"flowers released to pool (no gossiper)"
                                    )

                        bee.assigned_flowers = []  # Clear dead bee's assignments
                        if i in self.assignments:
                            self.assignments[i] = []

                    if self.verbose:
                        print(
                            f"[BATTERY] Bee {i} battery = 0 - STOPS working/broadcasting"
                        )

            self._last_pos[i] = np.array([float(b.fx), float(b.fy), float(b.fz)], dtype=float)

        # ── Basilisk batch position update ──
        if getattr(self, '_bsk', None) is not None:
            try:
                bsk_states = self._bsk.step()
                grid_pos = self._bsk.get_positions_grid(
                    self.grid_size, self.bsk_meters_per_unit
                )
                for i, b in enumerate(self.bees):
                    if self._recharge_until[i] > self.steps:
                        continue  # charging bees stay frozen
                    prev = self._last_pos[i].copy()
                    b.fx = float(grid_pos[i, 0])
                    b.fy = float(grid_pos[i, 1])
                    b.fz = float(grid_pos[i, 2])
                    # Battery drain: use episode-local drain (respects LOW_BATTERY mode)
                    dx = float(b.fx) - prev[0]
                    dy = float(b.fy) - prev[1]
                    dz = float(b.fz) - prev[2]
                    dist = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
                    if dist > 0.0:
                        self._battery[i] -= self._episode_drain_per_step + self.drain_per_unit * dist
                    if self._battery[i] <= 0.0:
                        self._battery[i] = 0.0
                        if self._recharge_until[i] <= self.steps:
                            self._recharge_until[i] = self.steps + self.recharge_steps
                            bee = self.bees[i]
                            if bee.assigned_flowers:
                                orphan_flowers = [
                                    fj for fj in bee.assigned_flowers
                                    if fj < len(self.flowers) and not self.flowers[fj].harvested
                                ]
                                if orphan_flowers:
                                    if getattr(self, 'gossiper', None) is not None:
                                        self.gossiper.broadcast_tasks(i, orphan_flowers, self.steps)
                                    else:
                                        for fj in orphan_flowers:
                                            self.flowers[fj].assigned_bee = None
                                bee.assigned_flowers = []
                                if i in self.assignments:
                                    self.assignments[i] = []
                    self._last_pos[i] = np.array(
                        [float(b.fx), float(b.fy), float(b.fz)], dtype=float
                    )
            except Exception as e:
                if self.verbose:
                    print(f"[BSK] Step failed: {e}")

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
            self._cached_orbit_points = None  # invalidate orbit cache
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
                    # Use Euclidean distance from current position (not orbit-min)
                    dx = bee.fx - cx
                    dy = bee.fy - cy
                    dz = bee.fz
                    current_d = math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)

                    # NEW: Check time window before allowing harvest
                    in_time_window = f.is_harvestable_at_time(self.steps)

                    # Double-check it's actually in range (Euclidean) and in time window
                    if (
                        current_d <= self.harvest_radius
                        and not f.harvested
                        and in_time_window
                    ):
                        # SUCCESSFUL HARVEST
                        f.harvested = True
                        f.harvested_step = self.steps
                        f.status = "completed"
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
                            if self.verbose:
                                print(f"[POOL] Bee {i} claimed unassigned flower {fj} from pool")

                        # BALANCED REWARD: proportional to pollen with efficiency bonuses
                        base_reward = f.pollen * 0.5
                        efficiency_bonus = 2.0 if actual_gain == f.pollen else 1.0
                        capacity_bonus = 3.0 * (1.0 - bee.load / bee.capacity)
                        # NEW: Bonus for hitting time windows
                        window_bonus = (
                            15.0
                            if f.window_type == "HARD"
                            else (2.0 if f.window_type == "SOFT" else 0.0)
                        )
                        total_reward = (
                            base_reward + efficiency_bonus + capacity_bonus + window_bonus
                        )

                        rewards[agent] += total_reward
                        success_map[agent] = True
                        if self.verbose:
                            print(
                                f"[ENV] Bee {i} harvested flower {fj} (+{actual_gain:.1f}pollen, reward: {total_reward:.1f}, now {bee.load:.1f}/{bee.capacity:.1f})"
                            )
                    elif not in_time_window:
                        # NEW: Attempted harvest outside time window
                        rewards[agent] -= 0.5
                        if self.verbose:
                            print(
                                f"[ENV] Bee {i} MISSED TIME WINDOW for flower {fj} (type: {f.window_type})"
                            )
                        # Mark HARD window as permanently missed
                        if f.window_type == "HARD":
                            f.window_missed = True
                            bee.missed_hard_windows.add(fj)
                    else:
                        # Too far or already harvested - simple fixed penalty
                        rewards[agent] -= 0.2
                        if self.verbose:
                            print(
                                f"[ENV] Bee {i} failed harvest - distance {current_d:.2f} > threshold {self.harvest_radius:.2f}"
                            )
                else:
                    # Lost the claim to another bee
                    rewards[agent] -= 0.1
                    if self.verbose:
                        print(f"[ENV] Bee {i} lost flower {fj} to bee {f.busy_by}")
            else:
                # No valid target found - simple fixed penalty
                if agent in attempted_harvest:
                    rewards[agent] -= 0.2
                    if self.verbose:
                        print(f"[ENV] Bee {i} attempted harvest but no valid target")

        # DONOTHING: no explicit reward/penalty - let harvest success/failure drive behavior
        # (removed progressive penalty tracking)

        # Groom cool down
        for b in self.bees:
            if b.groom_cooldown > 0:
                b.groom_cooldown -= 1
                if b.groom_cooldown == 0:
                    b.mode = bee_state.Bee.IDLE

        # ── Task expiration sweep ────────────────────────────────
        # Tasks past their deadline_step are marked expired, unassigned,
        # and removed from bee assigned_flowers lists.
        for j, f in enumerate(self.flowers):
            if f.harvested or f.expired:
                continue
            if f.deadline_step is not None and self.steps > f.deadline_step:
                f.expired = True
                f.status = "expired"
                # Unassign from owning bee
                owner = f.assigned_bee
                if owner is not None:
                    bee_owner = self.bees[owner]
                    if j in bee_owner.assigned_flowers:
                        bee_owner.assigned_flowers.remove(j)
                    if owner < len(self.assignments) and j in self.assignments[owner]:
                        self.assignments[owner].remove(j)
                    # Penalty to the bee that let the task expire
                    agent_id = f"bee_{owner}"
                    if agent_id in rewards:
                        rewards[agent_id] -= 1.0
                    f.assigned_bee = None
                if self.verbose:
                    print(
                        f"[EXPIRED] Task {getattr(f, 'task_id', f'flower_{j}')} "
                        f"expired at step {self.steps} (deadline was {f.deadline_step})"
                    )

        # Termination & truncation with NEW conditions

        # TERMINATED: All actionable flowers harvested or expired (nothing left to do)
        all_flowers_done = all(f.harvested or f.expired for f in self.flowers)
        all_flowers_harvested = all(f.harvested for f in self.flowers)

        # Check per-bee truncation conditions
        for agent in self.agents:
            i = int(agent.split("_")[1])
            bee = self.bees[i]

            # TRUNCATED CONDITION 1: All flowers assigned to this bee are harvested
            if bee.assigned_flowers:
                my_flowers_done = all(self.flowers[fj].harvested for fj in bee.assigned_flowers)
                if my_flowers_done and not bee.truncated:
                    bee.truncated = True
                    if self.verbose:
                        print(f"[TRUNCATED] Bee {i}: All assigned flowers completed")

            # PENALTY CONDITION: Missed HARD window(s) permanently (no truncation)
            for fj in bee.assigned_flowers:
                f = self.flowers[fj]
                if f.window_type == "HARD" and not f.harvested:
                    # Check if window is permanently missed
                    if self.steps > f.window_end:
                        if fj not in bee.missed_hard_windows:
                            bee.missed_hard_windows.add(fj)
                            rewards[agent] -= 5.0  # Penalty for missing HARD window
                            if self.verbose:
                                print(f"[PENALTY] Bee {i}: Missed HARD window for flower {fj} (-5.0)")

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
                    if self.verbose:
                        print(f"[TRUNCATED] Bee {i}: No more opportunities for SOFT window flowers")

        # Episode terminates only if:
        # 1. All flowers harvested (success), OR
        # 2. All actionable flowers done — harvested or expired (partial success), OR
        # 3. All bees truncated AND no tasks in queue (complete failure)
        all_bees_truncated = all(self.bees[i].truncated for i in range(self.num_bees))
        tasks_in_queue = any(
            not f.harvested and not f.expired and f.assigned_bee is None
            for f in self.flowers
        )

        # Only fail if all bees stopped AND no way to recover (no queue tasks)
        episode_failed = all_bees_truncated and not tasks_in_queue

        terminated = {a: (all_flowers_done or episode_failed) for a in self.agents}
        # Individual bee truncation does NOT end episode - just that bee does DONOTHING until queue tasks available
        truncated = {
            a: (self.steps >= self.max_steps) for a in self.agents
        }  # Only max_steps truncates episode

        # Print episode summary when terminating
        if all_flowers_done or episode_failed or self.steps >= self.max_steps:
            if self.verbose:
                episode_type = "LOW-BATTERY " if self._is_low_battery_episode else "NORMAL"
                batteries_died = sum(1 for b in self.bees if b.battery <= 0)
                flowers_harvested = sum(1 for f in self.flowers if f.harvested)
                flowers_expired = sum(1 for f in self.flowers if f.expired)
                print(f"\n[EPISODE END] Type: {episode_type} | Steps: {self.steps}/{self.max_steps}")
                print(
                    f"[EPISODE END] Flowers: {flowers_harvested}/{len(self.flowers)} harvested, "
                    f"{flowers_expired} expired | Batteries died: {batteries_died}/{self.num_bees}"
                )
                if all_flowers_harvested:
                    print("[EPISODE END]  SUCCESS - All flowers harvested!")
                elif all_flowers_done:
                    print("[EPISODE END]  PARTIAL - All flowers resolved (some expired)")
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
            #             is_harvestable_now, is_hard_window, time_to_window, priority, deadline_urgency (14 features)
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

                # Priority (normalized 0-1)
                priority = float(getattr(f, 'priority', 0.0))

                # Deadline urgency: how close is the deadline relative to remaining time
                # 1.0 = deadline imminent, 0.0 = plenty of time or no deadline
                deadline_step = getattr(f, 'deadline_step', None)
                if deadline_step is not None and not f.harvested and not f.expired:
                    remaining = max(0, deadline_step - self.steps)
                    total_remaining = max(1, self.max_steps - self.steps)
                    deadline_urgency = 1.0 - (remaining / total_remaining)
                    deadline_urgency = max(0.0, min(1.0, deadline_urgency))
                else:
                    deadline_urgency = 0.0

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
                        priority,
                        deadline_urgency,
                    ]
                )

            # Build per-bee retask board feature (gossiper inbox or legacy board)
            if getattr(self, 'gossiper', None) is not None:
                retask_feat = self.gossiper.get_retask_obs(i, self.retask_board_size)
            else:
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
                            float(task.get("hops", 0)) / 10.0,  # hops normalized
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
                if getattr(f, "expired", False):
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
                flower_obs_start = fj * 14
                dist_norm = flowers_feat[flower_obs_start + 8]  # Get dist_norm for this flower

                # NEW: Check time window availability
                is_harvestable_now = flowers_feat[flower_obs_start + 9]  # Time window check
                in_time_window = is_harvestable_now >= 0.5

                # Only allow harvest if actually in range (dist_norm < 1.0 means within harvest_radius)
                in_range = dist_norm < 1.0

                # Can only harvest flowers assigned to this bee or unassigned (from retask board)
                is_mine = f.assigned_bee == i
                is_unassigned = f.assigned_bee is None

                # Can harvest if: (mine OR unassigned), in range, in time window, and fits
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

            # HRL: Current goal for this bee (one-hot encoded)
            goal_onehot = np.zeros(4, dtype=np.float32)
            goal_onehot[self._current_goals[i]] = 1.0

            # Battery observation: fraction remaining + recharging flag
            battery_frac = self._battery[i] / self._battery_max[i] if self._battery_max[i] > 0 else 1.0
            is_recharging = 1.0 if self._recharge_until[i] > self.steps else 0.0

            obs[f"bee_{i}"] = {
                "position": np.array([fx, fy, fz], dtype=np.float32),
                "status": np.array([float(bee.mode), load_frac], dtype=np.float32),
                "battery": np.array([battery_frac, is_recharging], dtype=np.float32),
                "flowers": np.array(flowers_feat, dtype=np.float32),
                "step_count": np.array([self.steps / max(1.0, self.max_steps)], dtype=np.float32),
                "consensus": np.array(consensus, dtype=np.float32),
                "retask_board": np.array(retask_feat, dtype=np.float32),
                "action_availability": action_availability,
                "goal": goal_onehot,  # HRL goal
            }
        return obs

    # ----------------------------------------------------------
    # HRL: Goal Management
    # ----------------------------------------------------------
    def set_goals(self, goals: np.ndarray):
        """
        Set goals for all bees (called by Manager policy).
        
        Args:
            goals: (num_bees,) array of goal indices
                0 = IDLE, 1 = HARVEST, 2 = GROOM, 3 = ASSIST
        """
        self._current_goals = np.array(goals, dtype=int)
        self._goal_set_step[:] = self.steps
    
    def get_goal_reward(self, bee_id: int, action: int) -> float:
        """
        Compute reward shaping based on goal alignment.
        
        Returns bonus/penalty based on whether action aligns with current goal.
        Kept small so goal-shaping doesn't dominate the task reward signal.
        """
        goal = self._current_goals[bee_id]
        bee = self.bees[bee_id]
        
        # Goal 0: IDLE - reward DONOTHING when waiting is appropriate
        if goal == 0:
            if action == 0:  # DONOTHING
                return 0.1  # Bonus for following IDLE goal
            else:
                return -0.05  # Penalty for acting when told to idle
        
        # Goal 1: HARVEST - reward harvesting
        elif goal == 1:
            if action == 1:  # HARVEST
                return 0.2  # Bonus for attempting harvest when told to
            elif action == 0:  # DONOTHING
                return -0.1  # Penalty for idling when should harvest
            else:
                return 0.0
        
        # Goal 2: GROOM - reward grooming
        elif goal == 2:
            if action == 2:  # GROOM
                return 0.3  # Bonus for grooming when told to
            else:
                return -0.1 if bee.load >= bee.capacity * 0.5 else 0.0
        
        # Goal 3: ASSIST - reward claiming from retask board
        elif goal == 3:
            # Check if bee took a task from retask board
            if hasattr(bee, 'retask_board') and len(bee.retask_board) > 0:
                if action == 1:  # HARVEST (attempting to help)
                    return 0.15
            return 0.0
        
        return 0.0

    # ----------------------------------------------------------
    # Helpers (reachability & global state)
    # ----------------------------------------------------------
    def _make_flower(self, fid, x, y):
        # Randomly assign time window type so the manager must schedule
        wtype_roll = np.random.random()
        if wtype_roll < 0.15:
            wtype = "HARD"   # 15% hard deadline (rare, high-pressure)
        elif wtype_roll < 0.45:
            wtype = "SOFT"   # 30% soft/repeating window
        else:
            wtype = "NONE"   # 55% always available

        if wtype == "SOFT":
            # SOFT: repeating window with period. start/end are RELATIVE to period.
            period = np.random.randint(40, 120)  # repeats every 40-120 steps
            # Window occupies 40-70% of the period (meaningful on/off pattern)
            duty = np.random.uniform(0.4, 0.7)
            duration = max(10, int(period * duty))
            start = np.random.randint(0, period - duration + 1)
            end = start + duration
        elif wtype == "HARD":
            # HARD: one-shot absolute window
            start = np.random.randint(self.time_window_min, self.time_window_max)
            end = start + np.random.randint(80, 160)
            period = None
        else:
            # NONE: always available, window values don't matter
            start = 0
            end = self.max_steps
            period = None

        f = bee_state.Flower(
            fid, self.grid_size,
            window_start=start, window_end=end,
            window_type=wtype, window_period=period,
        )
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

    def _precompute_orbit_points(self):
        """Precompute orbit sample points for all bees (vectorized).
        Cached and reused until orbits change. Returns (B, S, 3) world-frame points."""
        B = self.num_bees
        S = self.reach_samples
        cx, cy = self.grid_size / 2.0, self.grid_size / 2.0
        nu = np.linspace(0, 2 * np.pi, S, endpoint=False)  # (S,)
        cos_nu = np.cos(nu)  # (S,)
        sin_nu = np.sin(nu)  # (S,)

        all_points = np.empty((B, S, 3), dtype=np.float64)
        for idx, bee in enumerate(self.bees):
            a = float(bee.a)
            e = float(bee.e)
            i_ = float(bee.i + getattr(bee, "inclination_delta", 0.0))
            Om = float(bee.Omega + getattr(bee, "yaw_delta", 0.0))
            om = float(bee.omega)
            R = self._R3(Om) @ self._R1(i_) @ self._R3(om)  # (3,3)
            bsemi = a * math.sqrt(max(1e-12, 1 - e * e))
            # Local orbit points: (3, S)
            local = np.vstack([a * cos_nu, bsemi * sin_nu, np.zeros(S)])
            world = R @ local  # (3, S)
            all_points[idx, :, 0] = cx + world[0]
            all_points[idx, :, 1] = cy + world[1]
            all_points[idx, :, 2] = world[2]
        self._cached_orbit_points = all_points
        return all_points

    def _min_distance_to_orbit(self, bee, x_center, y_center):
        # In telemetry mode, use direct position distance (no orbital mechanics)
        if getattr(self, '_telemetry_loaded', False):
            dx = bee.fx - x_center
            dy = bee.fy - y_center
            dz = bee.fz
            return math.sqrt(dx * dx + dy * dy + self.lambda_z * dz * dz)
        # Find bee index
        bee_idx = next((i for i, b in enumerate(self.bees) if b is bee), 0)
        if not hasattr(self, '_cached_orbit_points') or self._cached_orbit_points is None:
            self._precompute_orbit_points()
        pts = self._cached_orbit_points[bee_idx]  # (S, 3)
        dx = float(x_center) - pts[:, 0]
        dy = float(y_center) - pts[:, 1]
        dz = 0.0 - pts[:, 2]
        dists = np.sqrt(dx*dx + dy*dy + self.lambda_z * dz*dz)
        return float(dists.min())

    def _min_distance_to_orbit_batch(self, flower_centers):
        """Compute min orbit distance for ALL bees × ALL flowers at once.
        flower_centers: (F, 2) array of (x, y).
        Returns: (B, F) array of minimum distances.
        """
        if not hasattr(self, '_cached_orbit_points') or self._cached_orbit_points is None:
            self._precompute_orbit_points()
        pts = self._cached_orbit_points  # (B, S, 3)
        B, S = pts.shape[0], pts.shape[1]
        F = flower_centers.shape[0]
        # Flower positions: (F, 3) with z=0
        fp = np.zeros((F, 3), dtype=np.float64)
        fp[:, 0] = flower_centers[:, 0]
        fp[:, 1] = flower_centers[:, 1]
        # Vectorized: (B, S, 3) vs (F, 3) -> (B, F, S)
        # dx[b,f,s] = fp[f,0] - pts[b,s,0]
        dx = fp[np.newaxis, :, np.newaxis, 0] - pts[:, np.newaxis, :, 0]  # (B,F,S)
        dy = fp[np.newaxis, :, np.newaxis, 1] - pts[:, np.newaxis, :, 1]  # (B,F,S)
        dz = fp[np.newaxis, :, np.newaxis, 2] - pts[:, np.newaxis, :, 2]  # (B,F,S)
        dists = np.sqrt(dx*dx + dy*dy + self.lambda_z * dz*dz)  # (B,F,S)
        return dists.min(axis=2)  # (B, F)

    def _mark_reachable_per_bee(self):
        """Mark which flowers are reachable by each bee (VECTORIZED)."""
        B, F = self.num_bees, self.num_flowers
        self.reachable = np.zeros((B, F), dtype=bool)
        thresh = self.harvest_radius + self.reach_margin

        # Current position distances: (B, F)
        bee_pos = np.array([[b.fx, b.fy, b.fz] for b in self.bees], dtype=np.float64)  # (B,3)
        flower_centers = np.array([f.center_xy for f in self.flowers], dtype=np.float64)  # (F,2)
        dx = bee_pos[:, np.newaxis, 0] - flower_centers[np.newaxis, :, 0]  # (B,F)
        dy = bee_pos[:, np.newaxis, 1] - flower_centers[np.newaxis, :, 1]  # (B,F)
        dz = bee_pos[:, np.newaxis, 2]  # (B,1) broadcast to (B,F)
        current_dist = np.sqrt(dx*dx + dy*dy + self.lambda_z * dz*dz)  # (B,F)

        # Orbit minimum distances: (B, F) — fully vectorized
        self._precompute_orbit_points()
        orbit_min_dist = self._min_distance_to_orbit_batch(flower_centers)  # (B,F)

        self.reachable = (current_dist <= thresh) | (orbit_min_dist <= thresh)

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
            if getattr(f, "harvested", False) or getattr(f, "expired", False):
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
                    if hasattr(self, '_cached_orbit_points') and self._cached_orbit_points is not None:
                        fc = np.array([[cx, cy]], dtype=np.float64)
                        all_dists = self._min_distance_to_orbit_batch(fc)  # (B,1)
                        min_dist = float(all_dists.min())
                    else:
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
                # Note: recharging bees come back in ~40 steps, NOT orphaned
                cannot_reach = not bool(self.reachable[f.assigned_bee, flower_idx])
                if silent_by_idle or silent_by_heartbeat or cannot_reach:
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
                # Clean up old owner's assignment lists
                old_owner = f.assigned_bee
                if old_owner is not None and old_owner != winner_bee_id:
                    if old_owner in self.assignments and flower_idx in self.assignments[old_owner]:
                        self.assignments[old_owner].remove(flower_idx)
                    if flower_idx in self.bees[old_owner].assigned_flowers:
                        self.bees[old_owner].assigned_flowers.remove(flower_idx)

                # Assign flower to winner
                f.assigned_bee = winner_bee_id
                if winner_bee_id not in self.assignments:
                    self.assignments[winner_bee_id] = []
                if flower_idx not in self.assignments[winner_bee_id]:
                    self.assignments[winner_bee_id].append(flower_idx)
                    self.bees[winner_bee_id].assigned_flowers.append(flower_idx)

    # ---------- Helper: Clear task from all retask boards ----------
    def _clear_task_from_all_boards(self, flower_id: int):
        """Remove a flower from all bees' retask boards (when accepted or harvested)."""
        for bee in self.bees:
            bee.retask_board = [t for t in bee.retask_board if t["flower_id"] != flower_id]

    # ---------- Per-bee chain relay (v3) ----------
    def _propagate_retask_board(self):
        """
        Chain relay: each bee with tasks on its board evaluates them.
        - ACCEPT: task removed from board, flower assigned to this bee.
          Bee is then LOCKED (can't claim from board again until a NEW
          traveling board arrives to refresh it).
        - REJECT: task stays on board for handoff to nearest unseen bee.
        
        One hop per step: after evaluation, rejected tasks pass to the
        nearest active bee that hasn't seen them yet (tracked via seen_by).
        """
        for i, bee in enumerate(self.bees):
            if not bee.retask_board or bee.truncated:
                continue
            if self._recharge_until[i] > self.steps:
                continue  # Recharging bees don't process boards

            # ---- Filter stale / harvested / expired tasks ----
            valid_tasks = []
            for t in bee.retask_board:
                fj = t["flower_id"]
                if fj >= len(self.flowers):
                    continue
                f = self.flowers[fj]
                if f.harvested:
                    continue
                if f.window_type == "HARD" and self.steps > f.window_end:
                    continue
                # Drop tasks older than 200 steps (stale)
                if self.steps - t.get("received_step", 0) >= 200:
                    continue
                valid_tasks.append(t)
            bee.retask_board = valid_tasks

            if not bee.retask_board:
                continue

            # ---- Evaluate: accept or reject each task ----
            tasks_accepted_idx = []
            tasks_to_pass = []

            for task_idx, task in enumerate(bee.retask_board):
                # Ensure seen_by exists
                if "seen_by" not in task:
                    task["seen_by"] = set()

                # If this bee already saw this task, skip (can't claim again)
                if i in task["seen_by"]:
                    tasks_to_pass.append(task_idx)
                    continue

                # Mark this bee as having seen the task
                task["seen_by"].add(i)

                fj = task["flower_id"]
                f = self.flowers[fj]
                cx, cy = f.center_xy

                # 1. REACHABILITY
                orbit_dist = self._min_distance_to_orbit(bee, cx, cy)
                can_reach = orbit_dist <= (self.harvest_radius + self.reach_margin)

                # 2. CAPACITY
                can_fit = (bee.load + f.pollen) <= bee.capacity + 1e-9

                # 3. DEADLINE
                can_meet_deadline = True
                if f.window_type == "HARD":
                    steps_remaining = f.window_end - self.steps
                    if steps_remaining < 10:
                        can_meet_deadline = False

                # 4. WORKLOAD
                current_workload = sum(
                    1 for fk in bee.assigned_flowers
                    if fk < len(self.flowers) and not self.flowers[fk].harvested
                )
                workload_ok = current_workload < 5

                should_accept = can_reach and can_fit and can_meet_deadline and workload_ok

                if should_accept:
                    # ACCEPT: assign flower to this bee, remove from board
                    f.assigned_bee = i
                    if fj not in bee.assigned_flowers:
                        bee.assigned_flowers.append(fj)
                    if i not in self.assignments:
                        self.assignments[i] = []
                    if fj not in self.assignments[i]:
                        self.assignments[i].append(fj)
                    # Clear from ALL bees' boards (it's been claimed)
                    self._clear_task_from_all_boards(fj)
                    tasks_accepted_idx.append(task_idx)
                    if self.verbose:
                        print(
                            f"[CHAIN] Bee {i} ACCEPTS flower {fj} "
                            f"(hops={task['hops']}, src={task['source_bee']})"
                        )
                else:
                    tasks_to_pass.append(task_idx)

            # ---- Remove accepted tasks (already cleared by _clear_task_from_all_boards) ----
            # Rebuild board with only tasks to pass
            remaining = []
            for idx in tasks_to_pass:
                if idx < len(bee.retask_board):
                    remaining.append(bee.retask_board[idx])

            # ---- Handoff remaining tasks to nearest UNSEEN active bee ----
            if remaining:
                # For each remaining task, find nearest bee not in seen_by
                tasks_still_here = []
                for task in remaining:
                    seen = task.get("seen_by", set())
                    nearest_id = None
                    min_d = float("inf")
                    for j in range(self.num_bees):
                        if j == i or j in seen:
                            continue
                        other = self.bees[j]
                        if other.truncated or self._recharge_until[j] > self.steps:
                            continue
                        dx2 = other.fx - bee.fx
                        dy2 = other.fy - bee.fy
                        dz2 = other.fz - bee.fz
                        d = math.sqrt(dx2*dx2 + dy2*dy2 + self.lambda_z*dz2*dz2)
                        if d < min_d:
                            min_d = d
                            nearest_id = j

                    if nearest_id is not None:
                        target = self.bees[nearest_id]
                        existing = {t["flower_id"] for t in target.retask_board}
                        fj = task["flower_id"]
                        if fj not in existing:
                            task["hops"] += 1
                            task["received_step"] = self.steps
                            target.retask_board.append(task)
                            if self.verbose:
                                print(
                                    f"[CHAIN] Bee {i} \u2192 Bee {nearest_id}: "
                                    f"passed flower {fj} (hops={task['hops']})"
                                )
                    else:
                        # All bees have seen this task — cycle complete, keep on board
                        # (it will expire via staleness check next cycle)
                        tasks_still_here.append(task)

                bee.retask_board = tasks_still_here
            else:
                bee.retask_board = []

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
