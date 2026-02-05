import math
import random
from datetime import datetime, timedelta, timezone

import numpy as np

# Optional SGP4 (TLE) support
try:
    from sgp4.api import Satrec, jday

    _HAS_SGP4 = True
except Exception:
    _HAS_SGP4 = False


class Bee:
    IDLE, HARVESTING, GROOMING = 0, 1, 2

    def __init__(
        self,
        bee_id: int,
        grid_size: int,
        *,
        orbit_scale: float = 1.2,
        capacity: float = 10.0,
        # --- physics params ---
        mu_km3_s2: float = 398600.4418,  # Earth GM
        dt_s: float = 1.0,  # physics step (seconds)
        km_per_unit: float = 10.0,  # how many km per 1 grid unit
        # --- NEW: battery & broadcast params (added) ---
        battery_capacity: float = 100.0,
        initial_battery: float | None = None,
        battery_drain_per_sec: float = 0.10,
        recharge_seconds: float = 30.0,
    ):
        self.id = bee_id
        self.grid_size = grid_size

        # ---- Kepler (two-body) defaults ----
        self.a_units = grid_size * float(orbit_scale)  # semi-major axis in GRID UNITS
        self.a = self.a_units
        self.e = 0.25 + 0.05 * bee_id
        self.i = math.radians(10 + 12 * bee_id)
        self.Omega = math.radians((31 * bee_id) % 360)
        self.omega = math.radians((57 * bee_id) % 360)

        # For backward-compat visuals if needed
        self.period = 360 + 40 * bee_id
        self.phase = math.radians((23 * bee_id) % 360)

        # Kepler timing state
        self.mu = float(mu_km3_s2)
        self.dt_s = float(dt_s)
        self.km_per_unit = float(km_per_unit)
        self.a_km = max(1e-6, self.a_units * self.km_per_unit)
        self.n = math.sqrt(self.mu / (self.a_km**3))  # rad/s
        self.M = 0.0  # mean anomaly at epoch

        # Runtime state
        self.ticks = 0
        self.fx = self.fy = self.fz = 0.0
        self.x = self.y = 0

        self.mode = Bee.IDLE
        self.groom_cooldown = 0
        self.capacity = float(capacity)
        self.load = 0.0
        self.speed_scale = 1.0
        self.inclination_delta = 0.0
        self.yaw_delta = 0.0

        self.assigned_flowers = []
        self.terminated = False
        self.truncated = False  # NEW: Track if this bee should be truncated
        self.missed_hard_windows = set()  # NEW: Track flowers with permanently missed HARD windows

        # ---- SGP4 (optional real-satellite mode) ----
        self._sgp4_sat = None  # Satrec or None
        self._sgp4_t_utc = None  # datetime in UTC
        self._sgp4_scale_km_per_unit = float(self.km_per_unit)
        self._sgp4_dt_s = float(self.dt_s)
        self._last_r_km = [0.0, 0.0, 0.0]

        # =====================================================================
        # NEW: Battery & “broadcast” support (non-breaking additions)
        # =====================================================================
        self.battery_capacity = float(battery_capacity)
        if initial_battery is None:
            # randomize start between 40–100% to diversify behavior
            frac = random.uniform(0.40, 1.00)
            self.battery = frac * self.battery_capacity
        else:
            self.battery = float(initial_battery)
        self.battery_drain_per_sec = float(battery_drain_per_sec)
        self.recharge_seconds = float(recharge_seconds)
        self._recharge_left_s = 0.0  # >0 while recharging (no motion / no broadcast)

        # =====================================================================
        # Per-bee retasking board and communication chain (v2)
        # =====================================================================
        # Each bee holds a local retasking board: list of task dicts
        # {flower_id, source_bee, hops, received_step, can_perform}
        self.retask_board: list[dict] = []
        self.last_broadcast_to: int | None = None  # last bee ID this bee broadcast to
        self.last_received_from: int | None = None  # last bee ID this bee received from
        self.awaiting_handoff: bool = False  # waiting to pass task to nearby bee

    # ----------------- Rotation helpers -----------------
    def _R1(self, a):
        ca, sa = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

    def _R3(self, a):
        ca, sa = math.cos(a), math.sin(a)
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)

    def _rotation_matrix(self):
        return (
            self._R3(self.Omega + self.yaw_delta)
            @ self._R1(self.i + self.inclination_delta)
            @ self._R3(self.omega)
        )

    # ----------------- Kepler solver -----------------
    def _solve_kepler(self, M, e, tol=1e-10, iters=20):
        """Solve M = E - e sin E for E with Newton's method."""
        E = M if e < 0.8 else math.pi
        for _ in range(iters):
            f = E - e * math.sin(E) - M
            fp = 1 - e * math.cos(E)
            dE = -f / fp
            E += dE
            if abs(dE) < tol:
                break
        return E

    # ----------------- SGP4 helpers -----------------
    @staticmethod
    def _datetime_from_jd(jd: float) -> datetime:
        """Convert Julian Date to timezone-aware UTC datetime."""
        unix_days = jd - 2440587.5  # JD of Unix epoch
        seconds = unix_days * 86400.0
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)

    def set_tle(
        self,
        line1: str,
        line2: str,
        *,
        start_utc: datetime | None = None,
        km_per_unit: float | None = None,
    ):
        """
        Enable real-satellite propagation using SGP4 for this bee.
        If start_utc is None, starts at the TLE epoch (best for matching trackers).
        """
        if not _HAS_SGP4:
            raise RuntimeError("sgp4 not installed. Run: pip install sgp4")
        self._sgp4_sat = Satrec.twoline2rv(line1.strip(), line2.strip())
        if start_utc is None:
            jd_epoch = float(self._sgp4_sat.jdsatepoch) + float(self._sgp4_sat.jdsatepochF)
            start_utc = self._datetime_from_jd(jd_epoch)
        self._sgp4_t_utc = start_utc.astimezone(timezone.utc)
        if km_per_unit is not None:
            self._sgp4_scale_km_per_unit = float(km_per_unit)

    def _sgp4_step_and_project(self):
        """Advance SGP4 by dt and map TEME r[km] to grid units (centered)."""
        if self._sgp4_sat is None or self._sgp4_t_utc is None:
            return  # not in SGP4 mode

        t = self._sgp4_t_utc
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        err, r_km, v_km_s = self._sgp4_sat.sgp4(jd, fr)  # TEME
        if err != 0:
            # keep last good position on warnings/errors
            r_km = self._last_r_km
        else:
            self._last_r_km = r_km

        # advance internal time by dt * speed_scale
        dt = self._sgp4_dt_s * max(1e-6, self.speed_scale)
        self._sgp4_t_utc = self._sgp4_t_utc + timedelta(seconds=dt)

        # map km -> grid units and center
        xk, yk, zk = r_km
        cx = self.grid_size / 2.0
        cy = self.grid_size / 2.0
        s = self._sgp4_scale_km_per_unit
        self.fx = cx + (xk / s)
        self.fy = cy + (yk / s)
        self.fz = zk / s

        # clamp to grid integers for legacy fields
        self.x = int(np.clip(round(self.fx), 0, self.grid_size - 1))
        self.y = int(np.clip(round(self.fy), 0, self.grid_size - 1))

    def sgp4_enabled(self) -> bool:
        return (self._sgp4_sat is not None) and (self._sgp4_t_utc is not None)

    def disable_tle(self):
        """Turn off SGP4 mode and fall back to Kepler updates."""
        self._sgp4_sat = None
        self._sgp4_t_utc = None

    # =========================
    # NEW: Battery & broadcast
    # =========================
    @property
    def is_recharging(self) -> bool:
        return self._recharge_left_s > 0.0

    @property
    def can_broadcast(self) -> bool:
        """Broadcast is suppressed while recharging."""
        return not self.is_recharging

    def _drain_battery(self, dt_s: float):
        """Consume battery while moving (scaled by speed_scale). Speed is constant regardless of battery level."""
        drain = self.battery_drain_per_sec * max(1e-6, self.speed_scale) * max(0.0, dt_s)
        self.battery = max(0.0, self.battery - drain)  # Battery stops at exactly 0, never negative

        # Only enter recharge when battery reaches exactly 0 (not before)
        if self.battery == 0.0 and self._recharge_left_s <= 0.0:
            # Battery depleted: enter recharge state
            self._recharge_left_s = float(self.recharge_seconds)

    def _tick_recharge(self, dt_s: float):
        """Advance recharge timer and refill when done."""
        if self._recharge_left_s > 0.0:
            self._recharge_left_s = max(0.0, self._recharge_left_s - max(0.0, dt_s))
            if self._recharge_left_s <= 0.0:
                self.battery = self.battery_capacity  # full charge

    def broadcast_payload(self) -> dict:
        """
        Lightweight “what I'm doing” message for neighborhood retasking.
        Env can read this each step; no coupling required.
        """
        return {
            "bee_id": int(getattr(self, "id", 0)),
            "mode": int(self.mode),  # 0/1/2
            "can_broadcast": bool(self.can_broadcast),
            "battery": float(self.battery),
            "battery_capacity": float(self.battery_capacity),
            "is_recharging": bool(self.is_recharging),
            "recharge_left_s": float(self._recharge_left_s),
            "position": {"x": float(self.fx), "y": float(self.fy), "z": float(self.fz)},
            "load": float(self.load),
            "capacity": float(self.capacity),
        }

    # ----------------- Unified step -----------------
    def update_position(self):
        """
        Advance one step at CONSTANT SPEED (never slows down based on battery level):
          - If recharging (battery = 0), do not move; only tick recharge timer.
          - Else, move at full constant speed via SGP4 (if enabled) or Kepler.
          - Battery drains proportionally to distance traveled.
          - When battery reaches exactly 0, bee stops and enters recharge mode.
          - Speed remains constant until battery = 0, then bee stops completely.
        """
        # If currently recharging, don't move—only tick timer.
        if self._recharge_left_s > 0.0:
            self._tick_recharge(self.dt_s)
            self.ticks += 1
            # keep grid ints consistent with current fx/fy
            self.x = int(np.clip(round(self.fx), 0, self.grid_size - 1))
            self.y = int(np.clip(round(self.fy), 0, self.grid_size - 1))
            return

        # Drain battery for the step; may enter recharge immediately.
        self._drain_battery(self.dt_s)
        if self._recharge_left_s > 0.0:
            # started recharge this tick; don't move
            self._tick_recharge(self.dt_s)
            self.ticks += 1
            self.x = int(np.clip(round(self.fx), 0, self.grid_size - 1))
            self.y = int(np.clip(round(self.fy), 0, self.grid_size - 1))
            return

        # SGP4 short-circuit (movement allowed)
        if self._sgp4_sat is not None and self._sgp4_t_utc is not None:
            self._sgp4_step_and_project()
            self.ticks += 1
            return

        # ---- Keplerian timing (two-body) ----
        self.ticks += 1

        # Advance mean anomaly by n * dt (allow speed_scale to time-dilate)
        dt = self.dt_s * max(1e-6, self.speed_scale)
        self.M = (self.M + self.n * dt) % (2.0 * math.pi)

        # Solve Kepler -> true anomaly and radius
        E = self._solve_kepler(self.M, self.e)
        cosE, sinE = math.cos(E), math.sin(E)
        nu = 2.0 * math.atan2(math.sqrt(1 + self.e) * sinE, math.sqrt(1 - self.e) * (1.0 + cosE))
        r_km = self.a_km * (1.0 - self.e * cosE)

        # Perifocal coordinates in km
        x_p = r_km * math.cos(nu)
        y_p = r_km * math.sin(nu)
        z_p = 0.0

        # Rotate to world using your existing orientation (Omega, i, omega)
        R = self._rotation_matrix()
        xw, yw, zw = R @ np.array([x_p, y_p, z_p], float)

        # Convert km -> grid units and center in your grid
        cx = self.grid_size / 2.0
        cy = self.grid_size / 2.0
        self.fx = cx + (xw / self.km_per_unit)
        self.fy = cy + (yw / self.km_per_unit)
        self.fz = zw / self.km_per_unit

        self.x = int(np.clip(round(self.fx), 0, self.grid_size - 1))
        self.y = int(np.clip(round(self.fy), 0, self.grid_size - 1))

    def to_dict(self) -> dict:
        """Serialize Bee to plain-Python types for TOML/JSON."""
        return {
            "id": int(getattr(self, "id", 0)),
            "grid_size": int(self.grid_size),
            # orbit / motion
            "a": float(getattr(self, "a", getattr(self, "a_units", 0.0))),
            "e": float(getattr(self, "e", 0.0)),
            "i": float(getattr(self, "i", 0.0)),
            "Omega": float(getattr(self, "Omega", 0.0)),
            "omega": float(getattr(self, "omega", 0.0)),
            "M": float(getattr(self, "M", 0.0)),
            "speed_scale": float(getattr(self, "speed_scale", 1.0)),
            # position (continuous + grid ints)
            "fx": float(getattr(self, "fx", 0.0)),
            "fy": float(getattr(self, "fy", 0.0)),
            "fz": float(getattr(self, "fz", 0.0)),
            "x": int(getattr(self, "x", 0)),
            "y": int(getattr(self, "y", 0)),
            # capacity / load / mode
            "capacity": float(getattr(self, "capacity", 10.0)),
            "load": float(getattr(self, "load", 0.0)),
            "mode": int(getattr(self, "mode", Bee.IDLE)),
            # env deltas used by reachability
            "inclination_delta": float(getattr(self, "inclination_delta", 0.0)),
            "yaw_delta": float(getattr(self, "yaw_delta", 0.0)),
            # bookkeeping
            "ticks": int(getattr(self, "ticks", 0)),
            "terminated": bool(getattr(self, "terminated", False)),
            # NEW: battery state
            "battery": float(getattr(self, "battery", 0.0)),
            "battery_capacity": float(getattr(self, "battery_capacity", 100.0)),
            "battery_drain_per_sec": float(getattr(self, "battery_drain_per_sec", 0.10)),
            "recharge_seconds": float(getattr(self, "recharge_seconds", 30.0)),
            "recharge_left_s": float(getattr(self, "_recharge_left_s", 0.0)),
            # optional label
            "bee_id": str(getattr(self, "bee_id", f"bee_{getattr(self, 'id', 0)}")),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Bee":
        """Rebuild Bee from a dict previously saved by to_dict()."""
        b = cls(
            int(d.get("id", 0)),
            int(d.get("grid_size", 20)),
            battery_capacity=float(d.get("battery_capacity", 100.0)),
            initial_battery=float(d.get("battery", 100.0)),
            battery_drain_per_sec=float(d.get("battery_drain_per_sec", 0.10)),
            recharge_seconds=float(d.get("recharge_seconds", 30.0)),
        )
        b.a = float(d.get("a", getattr(b, "a", getattr(b, "a_units", 0.0))))
        b.e = float(d.get("e", getattr(b, "e", 0.0)))
        b.i = float(d.get("i", getattr(b, "i", 0.0)))
        b.Omega = float(d.get("Omega", getattr(b, "Omega", 0.0)))
        b.omega = float(d.get("omega", getattr(b, "omega", 0.0)))
        b.M = float(d.get("M", 0.0))
        b.speed_scale = float(d.get("speed_scale", getattr(b, "speed_scale", 1.0)))

        b.fx = float(d.get("fx", 0.0))
        b.fy = float(d.get("fy", 0.0))
        b.fz = float(d.get("fz", 0.0))
        b.x = int(d.get("x", 0))
        b.y = int(d.get("y", 0))

        b.capacity = float(d.get("capacity", getattr(b, "capacity", 10.0)))
        b.load = float(d.get("load", 0.0))
        b.mode = int(d.get("mode", Bee.IDLE))

        b.inclination_delta = float(d.get("inclination_delta", 0.0))
        b.yaw_delta = float(d.get("yaw_delta", 0.0))
        b.ticks = int(d.get("ticks", 0))
        b.terminated = bool(d.get("terminated", False))
        b.bee_id = d.get("bee_id", f"bee_{b.id}")

        # NEW: restore recharge remaining if present
        b._recharge_left_s = float(d.get("recharge_left_s", 0.0))
        return b


class Flower:
    def __init__(
        self,
        flower_id: int,
        grid_size: int,
        window_start=0,
        window_end=100,
        window_type="NONE",
        window_period=None,
    ):
        self.id = flower_id
        self.x = np.random.randint(0, grid_size)
        self.y = np.random.randint(0, grid_size)

        self.pollen = 0.0
        self.priority = 0.0

        self.harvested = False
        self.assigned_bee = None
        self.harvested_step = None
        self.window_start = window_start
        self.window_end = window_end
        self.expired = False

        self.busy_by = None

        # NEW: Time window attributes
        self.window_type = window_type  # 'NONE', 'HARD', 'SOFT'
        self.window_period = window_period  # For SOFT: repeat every N steps (e.g., 100 = daily)
        self.window_missed = False  # Track if HARD window permanently missed

    @property
    def center_xy(self):
        return (self.x + 0.5, self.y + 0.5)

    def is_harvestable_at_time(self, current_step: int) -> bool:
        """Check if flower is within harvest window at current timestep"""
        if self.harvested:
            return False

        if self.window_type == "NONE":
            return True

        if self.window_type == "HARD":
            # One-time window: must be within start-end range
            in_window = self.window_start <= current_step <= self.window_end
            if not in_window and current_step > self.window_end:
                self.window_missed = True  # Permanently missed
            return in_window

        if self.window_type == "SOFT":
            # Repeating window (e.g., 10:00-12:00 every day)
            if self.window_period is None or self.window_period <= 0:
                return True  # No period defined, treat as always open
            time_of_day = current_step % self.window_period
            return self.window_start <= time_of_day <= self.window_end

        return False

    def time_until_next_window(self, current_step: int) -> float:
        """Calculate steps until next harvestable window. Returns -1 if no future window."""
        if self.harvested or self.window_type == "NONE":
            return 0.0

        if self.window_type == "HARD":
            if current_step < self.window_start:
                return float(self.window_start - current_step)
            elif current_step <= self.window_end:
                return 0.0  # Currently in window
            else:
                return -1.0  # Window permanently missed

        if self.window_type == "SOFT":
            if self.window_period is None or self.window_period <= 0:
                return 0.0
            time_of_day = current_step % self.window_period
            if self.window_start <= time_of_day <= self.window_end:
                return 0.0  # Currently in window
            elif time_of_day < self.window_start:
                return float(self.window_start - time_of_day)
            else:
                # Next occurrence is tomorrow
                return float(self.window_period - time_of_day + self.window_start)

        return -1.0

    def to_dict(self) -> dict:
        """Serialize Flower to plain-Python types for TOML/JSON."""
        return {
            "id": int(getattr(self, "id", 0)),
            # grid position
            "x": int(getattr(self, "x", 0)),
            "y": int(getattr(self, "y", 0)),
            "position": [int(getattr(self, "x", 0)), int(getattr(self, "y", 0))],
            # time windows (support both names used in your codebase)
            "min_step": int(getattr(self, "min_step", getattr(self, "window_start", 0))),
            "max_step": int(getattr(self, "max_step", getattr(self, "window_end", 0))),
            "window_start": int(getattr(self, "window_start", getattr(self, "min_step", 0))),
            "window_end": int(getattr(self, "window_end", getattr(self, "max_step", 0))),
            # attributes
            "priority": float(getattr(self, "priority", 0.0)),
            "pollen": float(getattr(self, "pollen", getattr(self, "pollen_amount", 0.0))),
            "pollen_amount": float(getattr(self, "pollen_amount", getattr(self, "pollen", 0.0))),
            # status / assignment
            "harvested": bool(getattr(self, "harvested", False)),
            "assigned_bee": (
                int(self.assigned_bee) if getattr(self, "assigned_bee", None) is not None else None
            ),
            "harvested_step": (
                int(self.harvested_step)
                if getattr(self, "harvested_step", None) is not None
                else None
            ),
            "expired": bool(getattr(self, "expired", False)),
            "busy_by": (int(self.busy_by) if getattr(self, "busy_by", None) is not None else None),
            # optional label
            "flower_id": str(getattr(self, "flower_id", f"flower_{getattr(self,'id',0)}")),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Flower":
        g = int(d.get("grid_size", 20))
        f = cls(
            int(d.get("id", 0)),
            g,
            window_start=int(d.get("min_step", d.get("window_start", 0))),
            window_end=int(d.get("max_step", d.get("window_end", 0))),
        )
        pos = d.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            f.x, f.y = int(pos[0]), int(pos[1])
        else:
            f.x, f.y = int(d.get("x", 0)), int(d.get("y", 0))
        f.priority = float(d.get("priority", 0.0))
        pol = d.get("pollen", d.get("pollen_amount", 0.0))
        f.pollen = float(pol)
        f.harvested = bool(d.get("harvested", False))
        f.harvested_step = d.get("harvested_step")
        f.assigned_bee = d.get("assigned_bee")
        f.expired = bool(d.get("expired", False))
        f.busy_by = d.get("busy_by")
        f.flower_id = d.get("flower_id", f"flower_{f.id}")
        return f
