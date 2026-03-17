"""
Basilisk orbital dynamics interface for BeeForagingEnv.

Wraps a multi-satellite Basilisk simulation and exposes a simple
step-based API that feeds live ECI positions, battery levels, and
fuel mass into the RL environment each step.

Usage:
    from bsk_interface import BSKInterface

    bsk = BSKInterface(num_sats=25, dt_sec=1.0)
    bsk.configure_orbits(orbital_elements)   # optional custom orbits
    bsk.initialize()

    # Each env step:
    state = bsk.step()  # returns dict per satellite
    # state[i] = {"r_m": [x,y,z], "v_ms": [vx,vy,vz],
    #             "battery_frac": 0.8, "fuel_kg": 45.0, ...}

Requires Basilisk built from source:
    export PYTHONPATH="/path/to/basilisk/dist3:$PYTHONPATH"
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Basilisk imports (deferred to allow graceful fallback) ───
_BSK_AVAILABLE = False
try:
    from Basilisk.utilities import (
        SimulationBaseClass,
        macros,
        orbitalMotion,
        unitTestSupport,
        simIncludeGravBody,
    )
    from Basilisk.simulation import (
        spacecraft,
        simpleNav,
        simpleBattery,
        simpleSolarPanel,
        simplePowerSink,
    )

    _BSK_AVAILABLE = True
except ImportError:
    pass


# ── Orbital element presets ──────────────────────────────────
# Default: 3-shell Walker constellation (LEO)
def _default_walker(n: int) -> list[dict]:
    """Generate Walker-delta orbital elements for *n* satellites."""
    elements = []
    # Distribute across 3 planes with staggered true anomalies
    planes = max(1, n // 3)
    sats_per_plane = math.ceil(n / planes)
    alt_m = 550_000.0 + 6_371_000.0  # 550 km LEO

    idx = 0
    for p in range(planes):
        raan_deg = 360.0 / planes * p
        for s in range(sats_per_plane):
            if idx >= n:
                break
            f_deg = 360.0 / sats_per_plane * s
            elements.append({
                "a_m": alt_m,
                "e": 0.001,
                "i_deg": 53.0,           # Starlink-like inclination
                "Omega_deg": raan_deg,
                "omega_deg": 0.0,
                "f_deg": f_deg,
            })
            idx += 1
    return elements


@dataclass
class SatState:
    """Snapshot of one satellite's state at the current sim step."""
    r_m: np.ndarray          # ECI position [m] (3,)
    v_ms: np.ndarray         # ECI velocity [m/s] (3,)
    battery_frac: float      # 0..1
    battery_j: float         # current charge [J]
    battery_capacity_j: float  # max charge [J]
    fuel_kg: float           # remaining propellant [kg]
    sim_time_s: float        # simulation time [s]
    solar_panel_power_w: float = 0.0  # instantaneous solar panel output [W]

    # ── derived quantities ────────────────────────────────
    @property
    def position_rn(self) -> dict:
        """Range / latitude / longitude from ECI position."""
        x, y, z = self.r_m
        r = float(np.linalg.norm(self.r_m))
        lat = math.degrees(math.asin(z / max(r, 1e-6)))
        lon = math.degrees(math.atan2(y, x))
        return {"r": r, "lat": lat, "lon": lon}

    @property
    def velocity_rn(self) -> dict:
        """Speed and heading from ECI velocity."""
        speed = float(np.linalg.norm(self.v_ms))
        vx, vy, vz = self.v_ms
        heading = math.degrees(math.atan2(vy, vx))
        return {"speed": speed, "heading": heading}

    @property
    def battery_level_pct(self) -> float:
        """Battery charge as percentage (0-100)."""
        return self.battery_frac * 100.0


class BSKInterface:
    """
    Thin wrapper around a Basilisk multi-satellite simulation.

    The interface creates N spacecraft with gravity, battery, and
    optional solar panels.  Each call to ``step()`` advances the
    simulation by ``dt_sec`` seconds and returns a list of
    ``SatState`` objects.
    """

    def __init__(
        self,
        num_sats: int = 25,
        dt_sec: float = 1.0,
        battery_wh: float = 200.0,
        battery_init_frac: float = 0.8,
        power_draw_w: float = 3.0,
        sat_mass_kg: float = 750.0,
        fuel_mass_kg: float = 50.0,
        add_solar_panel: bool = False,
        panel_area_m2: float = 0.32,
        panel_efficiency: float = 0.35,
    ):
        if not _BSK_AVAILABLE:
            raise ImportError(
                "Basilisk not available. Build from source and set PYTHONPATH "
                "to include basilisk/dist3."
            )

        self.num_sats = num_sats
        self.dt_sec = dt_sec
        self.dt_ns = macros.sec2nano(dt_sec)
        self.battery_wh = battery_wh
        self.battery_init_frac = battery_init_frac
        self.power_draw_w = power_draw_w
        self.sat_mass_kg = sat_mass_kg
        self.fuel_mass_kg = fuel_mass_kg
        self.add_solar_panel = add_solar_panel
        self.panel_area_m2 = panel_area_m2
        self.panel_efficiency = panel_efficiency

        # Will be populated by initialize()
        self._sim: Any = None
        self._sats: list = []
        self._navs: list = []
        self._batts: list = []
        self._pos_recs: list = []
        self._batt_recs: list = []
        self._mu: float = 0.0
        self._step_count: int = 0
        self._orbital_elements: list[dict] | None = None
        self._initialized = False

    # ── configuration ────────────────────────────────────────
    def configure_orbits(self, elements: list[dict]):
        """
        Set custom orbital elements before ``initialize()``.

        Each element dict should have:
            a_m, e, i_deg, Omega_deg, omega_deg, f_deg
        """
        if len(elements) != self.num_sats:
            raise ValueError(
                f"Expected {self.num_sats} orbital elements, got {len(elements)}"
            )
        self._orbital_elements = elements

    # ── initialization ───────────────────────────────────────
    def initialize(self):
        """Build and initialize the Basilisk simulation."""
        if self._initialized:
            raise RuntimeError("BSKInterface already initialized. Call reset() first.")

        sim = SimulationBaseClass.SimBaseClass()
        proc = sim.CreateNewProcess("dynProc")
        proc.addTask(sim.CreateNewTask("dynTask", self.dt_ns))

        # ── Gravity ──
        grav_factory = simIncludeGravBody.gravBodyFactory()
        earth = grav_factory.createEarth()
        earth.isCentralBody = True
        self._mu = earth.mu

        # ── Orbital elements ──
        if self._orbital_elements is None:
            self._orbital_elements = _default_walker(self.num_sats)

        # ── Per-satellite setup ──
        sats, navs, batts = [], [], []
        pos_recs, batt_recs = [], []

        panel_recs = []

        for i in range(self.num_sats):
            oe_dict = self._orbital_elements[i]

            # ── Spacecraft ──
            sc = spacecraft.Spacecraft()
            sc.ModelTag = f"sat-{i}"
            sc.hub.mHub = self.sat_mass_kg
            sc.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(
                [900.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 600.0]
            )
            sc.gravField.gravBodies = spacecraft.GravBodyVector(
                list(grav_factory.gravBodies.values())
            )

            # ── Orbit IC ──
            oe = orbitalMotion.ClassicElements()
            oe.a = oe_dict["a_m"]
            oe.e = oe_dict["e"]
            oe.i = oe_dict["i_deg"] * macros.D2R
            oe.Omega = oe_dict["Omega_deg"] * macros.D2R
            oe.omega = oe_dict["omega_deg"] * macros.D2R
            oe.f = oe_dict["f_deg"] * macros.D2R

            r_n, v_n = orbitalMotion.elem2rv(self._mu, oe)
            sc.hub.r_CN_NInit = r_n
            sc.hub.v_CN_NInit = v_n
            sc.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
            sc.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]

            sim.AddModelToTask("dynTask", sc)
            sats.append(sc)

            # ── Navigation (gives clean position output) ──
            nav = simpleNav.SimpleNav()
            nav.ModelTag = f"nav-{i}"
            nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
            sim.AddModelToTask("dynTask", nav)
            navs.append(nav)

            # ── Power: drain ──
            sink = simplePowerSink.SimplePowerSink()
            sink.ModelTag = f"sink-{i}"
            sink.nodePowerOut = -self.power_draw_w  # negative = consumption
            sim.AddModelToTask("dynTask", sink)

            # ── Power: optional solar panel ──
            panel_msg = None
            if self.add_solar_panel:
                panel = simpleSolarPanel.SimpleSolarPanel()
                panel.ModelTag = f"panel-{i}"
                panel.stateInMsg.subscribeTo(sc.scStateOutMsg)
                panel.setPanelParameters(
                    [0, 0, 1],
                    self.panel_area_m2,
                    self.panel_efficiency,
                )
                sim.AddModelToTask("dynTask", panel)
                panel_msg = panel.nodePowerOutMsg

            # ── Battery ──
            batt = simpleBattery.SimpleBattery()
            batt.ModelTag = f"batt-{i}"
            cap_j = self.battery_wh * 3600.0  # W·hr → Joules
            batt.storageCapacity = cap_j
            batt.storedCharge_Init = cap_j * self.battery_init_frac
            batt.addPowerNodeToModel(sink.nodePowerOutMsg)
            if panel_msg is not None:
                batt.addPowerNodeToModel(panel_msg)
            sim.AddModelToTask("dynTask", batt)
            batts.append(batt)

            # ── Recorders ──
            pos_rec = sc.scStateOutMsg.recorder(self.dt_ns)
            sim.AddModelToTask("dynTask", pos_rec)
            pos_recs.append(pos_rec)

            batt_rec = batt.batPowerOutMsg.recorder(self.dt_ns)
            sim.AddModelToTask("dynTask", batt_rec)
            batt_recs.append(batt_rec)

            # Solar panel power recorder (if panel exists)
            if self.add_solar_panel and panel_msg is not None:
                panel_rec = panel_msg.recorder(self.dt_ns)
                sim.AddModelToTask("dynTask", panel_rec)
                panel_recs.append(panel_rec)
            else:
                panel_recs.append(None)

        # ── Store refs ──
        self._sim = sim
        self._sats = sats
        self._navs = navs
        self._batts = batts
        self._pos_recs = pos_recs
        self._batt_recs = batt_recs
        self._panel_recs = panel_recs
        self._step_count = 0

        # ── Initialize simulation ──
        sim.InitializeSimulation()
        self._initialized = True

        # Run one step so recorders have initial position data
        self.step()

    # ── stepping ─────────────────────────────────────────────
    def step(self) -> list[SatState]:
        """
        Advance simulation by one ``dt_sec`` step.
        Returns a list of SatState, one per satellite.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() first.")

        self._step_count += 1
        stop_ns = self._step_count * self.dt_ns
        self._sim.ConfigureStopTime(stop_ns)
        self._sim.ExecuteSimulation()

        states = []
        sim_time_s = self._step_count * self.dt_sec

        for i in range(self.num_sats):
            r = np.array(self._pos_recs[i].r_BN_N[-1], dtype=np.float64)
            v = np.array(self._pos_recs[i].v_BN_N[-1], dtype=np.float64)

            batt_level = float(self._batt_recs[i].storageLevel[-1])
            batt_cap = float(self._batts[i].storageCapacity)
            batt_frac = batt_level / max(1e-12, batt_cap)

            # Solar panel power
            panel_w = 0.0
            if self._panel_recs[i] is not None:
                try:
                    panel_w = float(self._panel_recs[i].netPower[-1])
                except (IndexError, AttributeError):
                    panel_w = 0.0

            states.append(SatState(
                r_m=r,
                v_ms=v,
                battery_frac=np.clip(batt_frac, 0.0, 1.0),
                battery_j=max(0.0, batt_level),
                battery_capacity_j=batt_cap,
                fuel_kg=self.fuel_mass_kg,  # static for now (no thruster model)
                sim_time_s=sim_time_s,
                solar_panel_power_w=panel_w,
            ))

        return states

    # ── reset ────────────────────────────────────────────────
    def reset(self):
        """Tear down and reinitialize the simulation."""
        self._sim = None
        self._sats = []
        self._navs = []
        self._batts = []
        self._pos_recs = []
        self._batt_recs = []
        self._panel_recs = []
        self._step_count = 0
        self._initialized = False
        self.initialize()

    # ── utility ──────────────────────────────────────────────
    def get_positions_grid(
        self, grid_size: float, meters_per_unit: float = 500_000.0
    ) -> np.ndarray:
        """
        Get current satellite positions as grid coordinates.
        Centers on the constellation centroid.

        Returns: (num_sats, 3) array in grid units.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() first.")

        positions_m = np.zeros((self.num_sats, 3), dtype=np.float64)
        for i in range(self.num_sats):
            positions_m[i] = self._pos_recs[i].r_BN_N[-1]

        # Center on centroid and scale to grid
        centroid = positions_m.mean(axis=0)
        relative = positions_m - centroid
        grid_coords = relative / meters_per_unit + grid_size / 2.0

        # Clamp to grid
        grid_coords[:, :2] = np.clip(grid_coords[:, :2], 0.0, grid_size - 1.0)
        # Z axis: keep relative (altitude spread), scale down
        grid_coords[:, 2] = relative[:, 2] / meters_per_unit

        return grid_coords.astype(np.float32)

    @property
    def sim_time_s(self) -> float:
        return self._step_count * self.dt_sec

    @property
    def initialized(self) -> bool:
        return self._initialized

    @staticmethod
    def is_available() -> bool:
        return _BSK_AVAILABLE
