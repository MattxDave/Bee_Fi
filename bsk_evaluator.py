"""
BSK Evaluator — Run the trained bee model inside a Basilisk simulation
and emit telemetry in the telemetrybridge.json schema.

Usage:
    PYTHONPATH="basilisk/dist3:$PYTHONPATH" python bsk_evaluator.py \
        --model outputs/best_actor.pt \
        --num_sats 18 --num_tasks 36 \
        --steps 600 --snapshot_interval 60 \
        --output bsk_telemetry_eval.json

Output matches the telemetrybridge.json schema:
    telemetry-bridge.controller.satellites[*]
        satellite_id, status, orbit_type,
        position_eci, velocity_eci, position_rn, velocity_rn,
        assigned_tasks, active_tasks,
        fuel_mass, battery_level, solar_panel_power,
        communication_status, last_update, simulation_time
    telemetry-bridge.gossiper.{sat_id}
        neighbors, their tasks, ttl, lastUpdated
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from bee_policy import Actor
from bees_env import BeeForagingEnv

# ── Orbit type classification by altitude ────────────────────
EARTH_RADIUS_M = 6_371_000.0


def classify_orbit(r_m: np.ndarray) -> str:
    """Classify orbit as NEO / MEO / GEO from ECI position vector."""
    alt_km = (np.linalg.norm(r_m) - EARTH_RADIUS_M) / 1000.0
    if alt_km < 2_000:
        return "NEO"
    elif alt_km < 35_786:
        return "MEO"
    else:
        return "GEO"


def satellite_id(orbit_type: str, index_in_type: int) -> str:
    """Generate SAT-{type}-{NNN} identifier."""
    return f"SAT-{orbit_type}-{index_in_type + 1:03d}"


# ── Flower → Task descriptor ────────────────────────────────

def flower_to_task(
    flower,
    bee_pos_eci: np.ndarray | None,
    task_idx: int,
    sim_start: datetime,
    step: int,
    dt_sec: float,
) -> dict:
    """
    Convert a Flower object into the telemetrybridge task schema.

    Maps:
        flower.id         → task_id
        flower.priority   → priority
        flower (x,y,z=0)  → location_task (kept in grid coords; 
                             scaled to meters if ECI transform available)
        flower.harvested / flower.assigned_bee → task_status
        flower.window_start / .window_end     → ReleaseTime / Deadline
    """
    # Task status
    if flower.harvested:
        status = "completed"
    elif getattr(flower, "expired", False):
        status = "expired"
    elif flower.assigned_bee is not None:
        status = "executing"
    else:
        status = "available"

    # Use flower's own status field if available
    flower_status = getattr(flower, "status", None)
    if flower_status in ("completed", "expired"):
        status = flower_status

    # Distance (Euclidean grid → metres via meters_per_unit)
    dist_m = 0.0
    if bee_pos_eci is not None:
        # Use grid distance * scale  (flower positions are grid coords)
        dx = flower.x - 0.0   # placeholder: real mapping done by caller
        dy = flower.y - 0.0
        dist_m = math.sqrt(dx * dx + dy * dy)

    # Time mapping: step → sim_start + step * dt
    release_dt = sim_start + timedelta(seconds=flower.window_start * dt_sec)
    dl_step = getattr(flower, "deadline_step", None)
    if dl_step is not None:
        deadline_dt = sim_start + timedelta(seconds=dl_step * dt_sec)
    else:
        deadline_dt = sim_start + timedelta(seconds=flower.window_end * dt_sec)

    # Location in task ECI (keep as grid coords for now — caller can transform)
    loc = {"x": float(flower.x), "y": float(flower.y), "z": 0.0}

    # Use flower's own task_id if available
    tid = getattr(flower, "task_id", f"TASK-{task_idx + 1:03d}-{flower.x}-{flower.y}")
    uid = f"{tid}-{loc['x']:.1f}-{loc['y']:.1f}-0.0"
    desc = getattr(flower, "task_description", _task_description(flower))

    return {
        "task_id": tid,
        "task_description": desc,
        "priority": int(flower.priority),
        "distance_to_task_m": round(dist_m, 1),
        "task_status": status,
        "location_task": loc,
        "UniqueID": uid,
        "ReleaseTime": release_dt.strftime("%m/%d/%Y %H:%M:%S:%f")[:-3],
        "Deadline": deadline_dt.strftime("%m/%d/%Y %H:%M:%S:%f")[:-3],
        "created_step": int(getattr(flower, "created_step", 0)),
        "deadline_step": int(dl_step) if dl_step is not None else None,
    }


def _task_description(flower) -> str:
    """Fallback: map flower window_type to a human-readable task description.
    Prefer flower.task_description if available (set during reset)."""
    desc = getattr(flower, "task_description", None)
    if desc:
        return desc
    wt = getattr(flower, "window_type", "NONE")
    if wt == "HARD":
        return "Time-critical observation"
    elif wt == "SOFT":
        return "Periodic measurement"
    else:
        return "Monitor space debris"


# ── Snapshot builder ─────────────────────────────────────────

def build_snapshot(
    env: BeeForagingEnv,
    step: int,
    sim_start: datetime,
    dt_sec: float,
    meters_per_unit: float,
) -> dict:
    """
    Build a single telemetry-bridge snapshot from current env state.

    Returns the full JSON-serialisable dict matching telemetrybridge.json.
    """
    bsk = env._bsk  # BSKInterface or None
    gossiper = getattr(env, "gossiper", None)
    now = sim_start + timedelta(seconds=step * dt_sec)
    now_str = now.strftime("%m/%d/%Y %H:%M:%S:%f")[:-3]

    # ── Classify satellites by orbit type ────────
    orbit_types: list[str] = []
    type_counters: dict[str, int] = {}

    for i in range(env.num_bees):
        if bsk is not None and bsk.initialized:
            r_m = np.array(bsk._pos_recs[i].r_BN_N[-1], dtype=np.float64)
        else:
            # Fallback: reconstruct ECI from grid pos
            b = env.bees[i]
            r_m = np.array([b.fx, b.fy, b.fz]) * meters_per_unit
        ot = classify_orbit(r_m)
        orbit_types.append(ot)
        type_counters.setdefault(ot, 0)

    # Assign SAT-{TYPE}-{NNN} IDs in type-order
    sat_ids: list[str] = []
    for ot in orbit_types:
        idx = type_counters[ot]
        sat_ids.append(satellite_id(ot, idx))
        type_counters[ot] = idx + 1

    # ── Per-satellite telemetry ──────────────────
    satellites: list[dict] = []

    for i in range(env.num_bees):
        bee = env.bees[i]

        # ── Position / velocity from BSK or fallback ──
        if bsk is not None and bsk.initialized:
            r_m = np.array(bsk._pos_recs[i].r_BN_N[-1], dtype=np.float64)
            v_ms = np.array(bsk._pos_recs[i].v_BN_N[-1], dtype=np.float64)
            batt_level_j = float(bsk._batt_recs[i].storageLevel[-1])
            batt_cap_j = float(bsk._batts[i].storageCapacity)
            batt_pct = (batt_level_j / max(1e-12, batt_cap_j)) * 100.0
            fuel = bsk.fuel_mass_kg
            sim_time = bsk.sim_time_s

            # Solar panel power
            solar_w = 0.0
            if hasattr(bsk, "_panel_recs") and bsk._panel_recs[i] is not None:
                try:
                    solar_w = float(bsk._panel_recs[i].netPower[-1])
                except (IndexError, AttributeError):
                    solar_w = 0.0
        else:
            # Keplerian fallback
            r_m = np.array([bee.fx, bee.fy, bee.fz]) * meters_per_unit
            v_ms = np.array([0.0, 0.0, 0.0])
            batt_pct = (env._battery[i] / max(1e-12, env._battery_max[i])) * 100.0
            fuel = 50.0
            sim_time = step * dt_sec
            solar_w = 0.0

        # Derived: position_rn
        r_mag = float(np.linalg.norm(r_m))
        lat = math.degrees(math.asin(r_m[2] / max(r_mag, 1e-6)))
        lon = math.degrees(math.atan2(r_m[1], r_m[0]))

        # Derived: velocity_rn
        speed = float(np.linalg.norm(v_ms))
        heading = math.degrees(math.atan2(v_ms[1], v_ms[0])) if speed > 0 else 0.0

        # ── Status ──
        status = "Inactive" if bee.truncated else "Active"

        # ── Communication status ──
        comm = "Offline" if bee.truncated else "Online"

        # ── Tasks (assigned flowers) ──
        assigned_tasks = []
        active_tasks = []
        for ti, fj in enumerate(bee.assigned_flowers):
            if fj >= len(env.flowers):
                continue
            f = env.flowers[fj]

            # Distance from bee to flower (in metres)
            dx = (f.x - bee.fx) * meters_per_unit
            dy = (f.y - bee.fy) * meters_per_unit
            dz = (0.0 - bee.fz) * meters_per_unit
            dist_m = math.sqrt(dx * dx + dy * dy + dz * dz)

            # Task location in ECI-like coords (scale grid→metres)
            task_loc = {
                "x": float(f.x) * meters_per_unit,
                "y": float(f.y) * meters_per_unit,
                "z": 0.0,
            }

            task_status = "completed" if f.harvested else ("expired" if getattr(f, "expired", False) else "executing")
            # Use flower's own status if available
            f_status = getattr(f, "status", None)
            if f_status in ("completed", "expired"):
                task_status = f_status

            release_dt = sim_start + timedelta(seconds=f.window_start * dt_sec)
            dl_step = getattr(f, "deadline_step", None)
            if dl_step is not None:
                deadline_dt = sim_start + timedelta(seconds=dl_step * dt_sec)
            else:
                deadline_dt = sim_start + timedelta(seconds=f.window_end * dt_sec)

            tid = getattr(f, "task_id", f"TASK-{fj + 1:03d}-{f.x}-{f.y}")
            task_dict = {
                "task_id": tid,
                "task_description": getattr(f, "task_description", _task_description(f)),
                "priority": int(f.priority),
                "distance_to_task_m": round(dist_m, 1),
                "task_status": task_status,
                "location_task": task_loc,
                "UniqueID": f"{tid}-{task_loc['x']:.1f}-{task_loc['y']:.1f}-{task_loc['z']:.1f}",
                "ReleaseTime": release_dt.strftime("%m/%d/%Y %H:%M:%S:%f")[:-3],
                "Deadline": deadline_dt.strftime("%m/%d/%Y %H:%M:%S:%f")[:-3],
                "created_step": int(getattr(f, "created_step", 0)),
                "deadline_step": int(dl_step) if dl_step is not None else None,
            }
            assigned_tasks.append(task_dict)
            if task_status == "executing":
                active_tasks.append(task_dict)

        sat_entry = {
            "satellite_id": sat_ids[i],
            "status": status,
            "orbit_type": orbit_types[i],
            "position_eci": {
                "x": float(r_m[0]),
                "y": float(r_m[1]),
                "z": float(r_m[2]),
            },
            "velocity_eci": {
                "vx": float(v_ms[0]),
                "vy": float(v_ms[1]),
                "vz": float(v_ms[2]),
            },
            "position_rn": {
                "r": r_mag,
                "lat": lat,
                "lon": lon,
            },
            "velocity_rn": {
                "speed": speed,
                "heading": heading,
            },
            "assigned_tasks": assigned_tasks,
            "active_tasks": active_tasks,
            "fuel_mass": float(fuel),
            "battery_level": round(batt_pct, 10),
            "solar_panel_power": round(solar_w, 10),
            "communication_status": comm,
            "last_update": now_str,
            "simulation_time": sim_time,
        }
        satellites.append(sat_entry)

    # ── Gossiper section ─────────────────────────
    gossiper_section: dict = {}
    if gossiper is not None:
        for i in range(env.num_bees):
            bee = env.bees[i]
            if bee.truncated:
                continue

            neighbors: dict = {}
            for j in range(env.num_bees):
                if j == i:
                    continue
                other = env.bees[j]
                if other.truncated:
                    continue
                if not gossiper._can_communicate(i, j):
                    continue

                # Neighbor position / velocity
                if bsk is not None and bsk.initialized:
                    nb_pos = np.array(bsk._pos_recs[j].r_BN_N[-1], dtype=np.float64)
                    nb_vel = np.array(bsk._pos_recs[j].v_BN_N[-1], dtype=np.float64)
                else:
                    nb_pos = np.array([other.fx, other.fy, other.fz]) * meters_per_unit
                    nb_vel = np.array([0.0, 0.0, 0.0])

                # Neighbor's assigned tasks
                nb_tasks = []
                for fj in other.assigned_flowers:
                    if fj >= len(env.flowers):
                        continue
                    f = env.flowers[fj]
                    task_loc = {
                        "x": float(f.x) * meters_per_unit,
                        "y": float(f.y) * meters_per_unit,
                        "z": 0.0,
                    }

                    dx = (f.x - other.fx) * meters_per_unit
                    dy = (f.y - other.fy) * meters_per_unit
                    dist_m = math.sqrt(dx * dx + dy * dy)

                    task_status = "completed" if f.harvested else ("expired" if getattr(f, "expired", False) else "executing")
                    f_status = getattr(f, "status", None)
                    if f_status in ("completed", "expired"):
                        task_status = f_status
                    release_dt = sim_start + timedelta(seconds=f.window_start * dt_sec)
                    dl_step = getattr(f, "deadline_step", None)
                    if dl_step is not None:
                        deadline_dt = sim_start + timedelta(seconds=dl_step * dt_sec)
                    else:
                        deadline_dt = sim_start + timedelta(seconds=f.window_end * dt_sec)

                    tid = getattr(f, "task_id", f"TASK-{fj + 1:03d}-{f.x}-{f.y}")
                    nb_tasks.append({
                        "task_id": tid,
                        "task_description": getattr(f, "task_description", _task_description(f)),
                        "priority": int(f.priority),
                        "distance_to_task_m": round(dist_m, 1),
                        "task_status": task_status,
                        "location_task": task_loc,
                        "UniqueID": f"{tid}-{task_loc['x']:.1f}-{task_loc['y']:.1f}-{task_loc['z']:.1f}",
                        "ReleaseTime": release_dt.strftime("%m/%d/%Y %H:%M:%S:%f")[:-3],
                        "Deadline": deadline_dt.strftime("%m/%d/%Y %H:%M:%S:%f")[:-3],
                        "created_step": int(getattr(f, "created_step", 0)),
                        "deadline_step": int(dl_step) if dl_step is not None else None,
                    })

                # Gossip TTL = max remaining hops in any message this neighbor holds
                nb_ttl = 0
                if j < len(gossiper.inboxes):
                    for msg in gossiper.inboxes[j]:
                        nb_ttl = max(nb_ttl, msg.ttl)

                neighbors[sat_ids[j]] = {
                    "position": {
                        "x": float(nb_pos[0]),
                        "y": float(nb_pos[1]),
                        "z": float(nb_pos[2]),
                    },
                    "velocity": {
                        "vx": float(nb_vel[0]),
                        "vy": float(nb_vel[1]),
                        "vz": float(nb_vel[2]),
                    },
                    "tasks": nb_tasks,
                    "ttl": nb_ttl,
                    "lastUpdated": now_str,
                }

            if neighbors:
                gossiper_section[sat_ids[i]] = {
                    "timestamp": now_str,
                    "neighbors": neighbors,
                }

    # ── Assemble top-level ───────────────────────
    snapshot = {
        "telemetry-bridge": {
            "controller": {
                "type": "telemetry",
                "generated_at": now_str,
                "satellites": satellites,
            },
            "gossiper": gossiper_section,
        },
    }

    return snapshot


# ── Evaluation metrics ───────────────────────────────────────

def compute_metrics(env: BeeForagingEnv, step: int) -> dict:
    """Compute evaluation metrics from current env state."""
    total = len(env.flowers)
    harvested = sum(1 for f in env.flowers if f.harvested)
    harvest_rate = harvested / max(1, total)

    alive = sum(1 for b in env.bees if not b.truncated)
    dead = env.num_bees - alive

    total_load = sum(b.load for b in env.bees)

    gossiper = getattr(env, "gossiper", None)
    gossip_stats = {}
    if gossiper is not None:
        gossip_stats = {
            "total_messages_sent": gossiper.total_messages_sent,
            "total_messages_accepted": gossiper.total_messages_accepted,
            "total_messages_expired": gossiper.total_messages_expired,
            "active_messages": sum(len(inbox) for inbox in gossiper.inboxes),
        }

    bsk = env._bsk
    bsk_stats = {}
    if bsk is not None and bsk.initialized:
        fuel_masses = [bsk.fuel_mass_kg for _ in range(env.num_bees)]
        batt_fracs = []
        for i in range(env.num_bees):
            lvl = float(bsk._batt_recs[i].storageLevel[-1])
            cap = float(bsk._batts[i].storageCapacity)
            batt_fracs.append(lvl / max(1e-12, cap))
        bsk_stats = {
            "sim_time_s": bsk.sim_time_s,
            "mean_battery_pct": np.mean(batt_fracs) * 100,
            "min_battery_pct": np.min(batt_fracs) * 100,
            "mean_fuel_kg": np.mean(fuel_masses),
        }

    return {
        "step": step,
        "harvest_rate": round(harvest_rate, 4),
        "harvested": harvested,
        "total_flowers": total,
        "alive_satellites": alive,
        "dead_satellites": dead,
        "total_load": round(total_load, 2),
        "gossiper": gossip_stats,
        "basilisk": bsk_stats,
    }


# ── Main evaluation loop ────────────────────────────────────

def evaluate(
    model_path: str,
    num_sats: int = 18,
    num_tasks: int = 36,
    grid_size: int = 75,
    max_steps: int = 600,
    snapshot_interval: int = 60,
    output_path: str = "bsk_telemetry_eval.json",
    use_basilisk: bool = True,
    bsk_dt_sec: float = 1.0,
    bsk_battery_wh: float = 200.0,
    bsk_power_draw_w: float = 3.0,
    bsk_meters_per_unit: float = 500_000.0,
    add_solar_panel: bool = True,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Run the trained policy in BSK mode and produce telemetrybridge.json output.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim_start = datetime.now(tz=timezone.utc)

    # ── Create environment ───────
    env = BeeForagingEnv(
        num_bees=num_sats,
        num_flowers=num_tasks,
        grid_size=grid_size,
        max_steps=max_steps,
        retask_board_size=3,
        use_basilisk=use_basilisk,
        bsk_dt_sec=bsk_dt_sec,
        bsk_battery_wh=bsk_battery_wh,
        bsk_power_draw_w=bsk_power_draw_w,
        bsk_meters_per_unit=bsk_meters_per_unit,
        verbose=verbose,
    )
    obs = env.reset()

    bsk_active = env._bsk is not None
    if verbose:
        print(f"BSK active: {bsk_active}")
        print(f"Satellites: {num_sats}, Tasks: {num_tasks}")
        print(f"Grid: {grid_size}x{grid_size}, Steps: {max_steps}")

    # ── Load policy ──────────────
    actor = None
    try:
        actor = Actor(
            num_bees=num_sats,
            action_dim=3,
            num_flowers=num_tasks,
            retask_board_size=3,
            grid_size=grid_size,
        ).to(device)
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        actor.load_state_dict(ckpt)
        actor.eval()
        if verbose:
            print(f"Loaded policy: {model_path}")
    except Exception as e:
        print(f"[WARNING] Could not load policy ({e}), using random actions")
        actor = None

    # ── Run episode ──────────────
    snapshots: list[dict] = []
    metrics_log: list[dict] = []
    step = 0

    if verbose:
        print(f"\n{'=' * 60}")
        print("EVALUATION START")
        print(f"{'=' * 60}")

    t0 = time.time()

    while step < max_steps and env.agents:
        # ── Select actions ──
        actions = {}
        if actor is not None:
            for agent in env.agents:
                ob = obs[agent]
                ob_tensor = {
                    "position": torch.tensor(ob["position"], dtype=torch.float32).unsqueeze(0).to(device),
                    "status": torch.tensor(ob["status"], dtype=torch.float32).unsqueeze(0).to(device),
                    "flowers": torch.tensor(ob["flowers"], dtype=torch.float32).unsqueeze(0).to(device),
                    "step_count": torch.tensor(ob["step_count"], dtype=torch.float32).unsqueeze(0).to(device),
                    "consensus": torch.tensor(ob["consensus"], dtype=torch.float32).unsqueeze(0).to(device),
                    "retask_board": torch.tensor(ob["retask_board"], dtype=torch.float32).unsqueeze(0).to(device),
                    "action_availability": torch.tensor(ob["action_availability"], dtype=torch.float32).unsqueeze(0).to(device),
                }
                with torch.no_grad():
                    logits = actor(ob_tensor)
                    # Apply hard action masking (same as training loop)
                    avail = ob_tensor["action_availability"]          # (1, 3) = [can_harvest, can_groom, can_idle]
                    mask = torch.stack([avail[:, 2], avail[:, 0], avail[:, 1]], dim=-1)  # reorder to [DONOTHING, HARVEST, GROOM]
                    logits = logits + (mask - 1.0) * 1e8
                    action = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
                actions[agent] = action
        else:
            import random as _rng
            for agent in env.agents:
                actions[agent] = _rng.choices([0, 1, 2], weights=[0.2, 0.7, 0.1])[0]

        # ── Step ──
        result = env.step(actions)
        obs = result[0]
        step += 1

        # ── Snapshot at interval ──
        if step % snapshot_interval == 0 or step == 1 or step == max_steps:
            snap = build_snapshot(env, step, sim_start, bsk_dt_sec, bsk_meters_per_unit)
            snapshots.append(snap)

            m = compute_metrics(env, step)
            metrics_log.append(m)

            if verbose:
                print(
                    f"  Step {step:>5d}  |  "
                    f"harvest {m['harvest_rate'] * 100:5.1f}%  |  "
                    f"alive {m['alive_satellites']:>2d}/{num_sats}  |  "
                    f"load {m['total_load']:6.1f}  |  "
                    f"gossip_sent {m['gossiper'].get('total_messages_sent', 0)}"
                )

    elapsed = time.time() - t0

    # ── Final metrics ────────────
    final = compute_metrics(env, step)

    if verbose:
        print(f"\n{'=' * 60}")
        print("EVALUATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Steps run:      {step}")
        print(f"  Wall time:      {elapsed:.1f}s")
        print(f"  Harvest rate:   {final['harvest_rate'] * 100:.1f}%")
        print(f"  Alive sats:     {final['alive_satellites']}/{num_sats}")
        print(f"  Dead sats:      {final['dead_satellites']}")
        if final["gossiper"]:
            g = final["gossiper"]
            print(f"  Gossip sent:    {g['total_messages_sent']}")
            print(f"  Gossip accepted:{g['total_messages_accepted']}")
            print(f"  Gossip expired: {g['total_messages_expired']}")
        if final["basilisk"]:
            b = final["basilisk"]
            print(f"  BSK sim time:   {b['sim_time_s']:.1f}s")
            print(f"  Mean battery:   {b['mean_battery_pct']:.1f}%")
            print(f"  Min battery:    {b['min_battery_pct']:.1f}%")

    # ── Write output ─────────────
    output = {
        "evaluation": {
            "model": str(model_path),
            "num_sats": num_sats,
            "num_tasks": num_tasks,
            "max_steps": max_steps,
            "steps_run": step,
            "wall_time_s": round(elapsed, 2),
            "seed": seed,
            "use_basilisk": bsk_active,
            "bsk_dt_sec": bsk_dt_sec,
        },
        "final_metrics": final,
        "metrics_log": metrics_log,
        "snapshots": snapshots,
    }

    out = Path(output_path)
    out.write_text(json.dumps(output, indent=2))
    if verbose:
        n_snaps = len(snapshots)
        print(f"\nWrote {n_snaps} snapshot(s) → {out}")

    return output


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained bee model in Basilisk simulation"
    )
    parser.add_argument("--model", default="outputs/best_actor.pt",
                        help="Path to trained actor .pt checkpoint")
    parser.add_argument("--num_sats", type=int, default=18)
    parser.add_argument("--num_tasks", type=int, default=36)
    parser.add_argument("--grid_size", type=int, default=75)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--snapshot_interval", type=int, default=60,
                        help="Emit telemetry snapshot every N steps")
    parser.add_argument("--output", default="bsk_telemetry_eval.json")
    parser.add_argument("--no_basilisk", action="store_true",
                        help="Run with Keplerian fallback (no BSK)")
    parser.add_argument("--bsk_dt", type=float, default=1.0)
    parser.add_argument("--bsk_battery_wh", type=float, default=200.0)
    parser.add_argument("--bsk_power_draw_w", type=float, default=3.0)
    parser.add_argument("--bsk_meters_per_unit", type=float, default=500_000.0)
    parser.add_argument("--solar_panel", action="store_true", default=True,
                        help="Enable solar panel on each satellite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        num_sats=args.num_sats,
        num_tasks=args.num_tasks,
        grid_size=args.grid_size,
        max_steps=args.steps,
        snapshot_interval=args.snapshot_interval,
        output_path=args.output,
        use_basilisk=not args.no_basilisk,
        bsk_dt_sec=args.bsk_dt,
        bsk_battery_wh=args.bsk_battery_wh,
        bsk_power_draw_w=args.bsk_power_draw_w,
        bsk_meters_per_unit=args.bsk_meters_per_unit,
        add_solar_panel=args.solar_panel,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
