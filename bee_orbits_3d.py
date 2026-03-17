#!/usr/bin/env python3
"""
Bee orbit visualizer with POLICY mode + right-side HUD panel.

Adds SGP4 toggles via CLI:
  --use_sgp4, --sgp4-scale, --sgp4-stagger, --tle, --tle-repeat

Adds Telemetry replay:
  --telemetry <file.jsonl> [--telemetry-speed 1.0] [--telemetry-prefix SAT-BEE-]

When --telemetry is provided, bee marker positions are driven by the telemetry
stream (NDJSON) each frame. Orbit ribbons are hidden to avoid clutter.
"""

import argparse
import math
import os
from collections import deque
from datetime import timedelta
from typing import List, Tuple

import matplotlib
if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    try:
        matplotlib.use("TkAgg")
    except ImportError:
        pass  # fall back to default

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.lines import Line2D


# ---------- math helpers ----------
def r1(a: float):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], float)


def r3(a: float):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], float)


def set_axes_equal(ax):
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xr, yr, zr = abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]), abs(zlim[1] - zlim[0])
    r = 0.5 * max(xr, yr, zr)
    xm, ym, zm = np.mean(xlim), np.mean(ylim), np.mean(zlim)
    ax.set_xlim3d([xm - r, xm + r])
    ax.set_ylim3d([ym - r, ym + r])
    ax.set_zlim3d([zm - r, zm + r])


def draw_grid_plane_centered(
    ax, grid_size: int, cx: float, cy: float, z: float = 0.0, alpha: float = 0.12
):
    xs = np.linspace(cx - grid_size / 2, cx + grid_size / 2, 11)
    ys = np.linspace(cy - grid_size / 2, cy + grid_size / 2, 11)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, z)
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False, alpha=alpha)


# ---------- env-bee orbit sampling ----------
def orbit_samples_for_env_bee(env_bee, grid_size: int, samples: int = 360):
    """
    Return (xs,ys,zs) to draw the orbit path.
    - If the bee is in SGP4 mode, sample its TLE ephemeris over time.
    - Else, draw the Kepler ellipse from (a,e,i,Ω,ω).
    """
    # --- SGP4 path (real satellites) ---
    if (
        hasattr(env_bee, "_sgp4_sat")
        and env_bee._sgp4_sat is not None
        and getattr(env_bee, "_sgp4_t_utc", None)
    ):
        try:
            from sgp4.api import jday
        except Exception:
            pass
        else:
            sat = env_bee._sgp4_sat
            t0 = env_bee._sgp4_t_utc
            s = float(getattr(env_bee, "_sgp4_scale_km_per_unit", 300.0))
            step_sec = 60.0
            cx = grid_size / 2.0
            cy = grid_size / 2.0
            xs, ys, zs = [], [], []
            half = max(1, samples // 2)
            for k in range(-half, half):
                t = t0 + timedelta(seconds=k * step_sec)
                jd, fr = jday(
                    t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6
                )
                err, r_km, _ = sat.sgp4(jd, fr)
                if err != 0:
                    continue
                xk, yk, zk = r_km
                xs.append(cx + xk / s)
                ys.append(cy + yk / s)
                zs.append(zk / s)
            return np.array(xs), np.array(ys), np.array(zs)

    # --- Kepler ellipse (synthetic bees) ---
    a, e = float(env_bee.a), float(env_bee.e)
    i = float(env_bee.i + getattr(env_bee, "inclination_delta", 0.0))
    Om = float(env_bee.Omega + getattr(env_bee, "yaw_delta", 0.0))
    om = float(env_bee.omega)
    R = r3(Om) @ r1(i) @ r3(om)
    b = a * math.sqrt(max(1e-12, 1 - e * e))
    nus = np.linspace(0, 2 * math.pi, samples, endpoint=False)
    cx = grid_size / 2.0
    cy = grid_size / 2.0
    pts = []
    for nu in nus:
        r_local = np.array([a * math.cos(nu), b * math.sin(nu), 0.0], float)
        r_world = R @ r_local
        pts.append([cx + r_world[0], cy + r_world[1], r_world[2]])
    arr = np.array(pts, float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


# ---------- TLE file reader ----------
def read_tle_pairs(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not path or not os.path.isfile(path):
        return pairs
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    buf = []
    for ln in lines:
        if ln.startswith("1 ") or ln.startswith("2 "):
            buf.append(ln)
            if len(buf) == 2 and buf[0].startswith("1 ") and buf[1].startswith("2 "):
                pairs.append((buf[0], buf[1]))
                buf = []
        else:
            continue
    return pairs


# ---------- infer hidden width from checkpoint ----------
def infer_hidden_from_checkpoint(actor_path: str, default_hidden: int = 128) -> int:
    try:
        sd = torch.load(actor_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        for k, v in sd.items():
            if k.endswith("trunk.0.weight") and v.ndim == 2:
                return int(v.shape[0])
        for k, v in sd.items():
            if ".0.weight" in k and v.ndim == 2:
                return int(v.shape[0])
    except Exception as e:
        print(f"[infer_hidden] fallback to default due to: {e}")
    return default_hidden


def infer_global_state_size_from_checkpoint(critic_path: str, default_size: int = 235) -> int:
    """Infer global_state_size from trunk.0.weight shape."""
    try:
        sd = torch.load(critic_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        for k, v in sd.items():
            if k.endswith("trunk.0.weight") and v.ndim == 2:
                return int(v.shape[1])
    except Exception as e:
        print(f"[infer_gsize] fallback to default due to: {e}")
    return default_size


# ---------- detect trained num_flowers & adapt obs ----------
def trained_num_flowers_from_ckpt(actor_path: str, default_n: int = 12) -> int:
    """Infer num_flowers from flowers_fc.weight.
    Newer checkpoints use 9 features per flower (x,y,pollen,harvested,mine,busy,reachable,fits,load_frac).
    Fallback gracefully if shape not found.
    """
    try:
        sd = torch.load(actor_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        for k, v in sd.items():
            if k.endswith("flowers_fc.0.weight") or k.endswith("flowers_fc.weight"):
                if v.ndim == 2:
                    in_dim = int(v.shape[1])
                    # Prefer 9-feature stride, fallback to 6 if divisible
                    if in_dim % 9 == 0:
                        return in_dim // 9
                    if in_dim % 6 == 0:
                        return in_dim // 6
    except Exception as e:
        print(f"[flowers_detect] fallback to default due to: {e}")
    return default_n


def adapt_flowers_vec(
    flowers_vec: torch.Tensor, trained_nflowers: int, stride: int = 9
) -> torch.Tensor:
    """Adapt flower feature vector to expected length (trained_nflowers * stride).
    Runtime env may provide 9 features per flower; older checkpoints might use 6.
    We detect stride from current tensor dimension when possible.
    """
    flowers_vec = torch.nan_to_num(flowers_vec, nan=0.0, posinf=0.0, neginf=0.0)
    B, D = flowers_vec.shape
    # If D matches either 9*env_n or 6*env_n keep stride accordingly for padding logic
    if D % trained_nflowers == 0:
        current_stride = D // trained_nflowers
    else:
        current_stride = stride
    expect = trained_nflowers * current_stride
    if expect == D:
        return flowers_vec
    if expect > D:
        pad = flowers_vec.new_zeros((B, expect - D))
        return torch.cat([flowers_vec, pad], dim=-1)
    return flowers_vec[:, :expect]


# ---------- action decode ----------
AC_MAP = {0: "DONOTHING", 1: "HARVEST", 2: "GROOM"}


def choose_actions_hrl(manager, worker, env, obs_dicts, current_goals, step, manager_interval,
                      stochastic: bool, num_bees: int = 25, device: str = "cpu"):
    """HRL action selection: manager sets goals, worker executes."""
    import torch.nn.functional as F
    from train_orbital_v2 import _batch_obs_to_tensor
    from hrl_policy import build_manager_obs

    is_manager_step = (step % manager_interval == 0)

    if is_manager_step:
        manager_obs = build_manager_obs(env, device)
        with torch.no_grad():
            new_goals, _ = manager.get_goals(manager_obs, stochastic=stochastic)
            current_goals[:] = new_goals.squeeze(0)
        env.set_goals(current_goals.cpu().numpy())

    actions = {}
    claims = {agent: None for agent in obs_dicts}

    with torch.no_grad():
        batched = _batch_obs_to_tensor(obs_dicts, num_bees, device)
        logits = worker(batched, current_goals)  # (num_bees, action_dim)

        if "action_availability" in batched:
            avail = batched["action_availability"]
            mask = torch.stack([avail[:, 2], avail[:, 0], avail[:, 1]], dim=-1)
            logits = logits + (mask - 1.0) * 1e8

        probs = F.softmax(logits, dim=-1)

        if stochastic:
            dist = torch.distributions.Categorical(probs)
            actions_t = dist.sample()
        else:
            actions_t = torch.argmax(probs, dim=-1)

        for agent in obs_dicts:
            idx = int(agent.split("_")[1])
            actions[agent] = int(actions_t[idx].item())

    return actions, claims, current_goals


def choose_actions(actor, obs_dicts, stochastic: bool, trained_nflowers: int, num_bees: int = 25, device: str = "cpu"):
    """Batched action selection matching training pipeline exactly."""
    import torch.nn.functional as F
    from train_orbital_v2 import _batch_obs_to_tensor

    actions = {}
    claims = {agent: None for agent in obs_dicts}

    with torch.no_grad():
        batched = _batch_obs_to_tensor(obs_dicts, num_bees, device)
        logits = actor(batched)  # (num_bees, action_dim)

        # Apply action masking (same as training)
        if "action_availability" in batched:
            avail = batched["action_availability"]  # (num_bees, 3)
            # Reorder: [can_harvest, can_groom, can_do_nothing] -> [donothing, harvest, groom]
            mask = torch.stack([avail[:, 2], avail[:, 0], avail[:, 1]], dim=-1)
            logits = logits + (mask - 1.0) * 1e8

        probs = F.softmax(logits, dim=-1)

        if stochastic:
            dist = torch.distributions.Categorical(probs)
            actions_t = dist.sample()  # (num_bees,)
        else:
            actions_t = torch.argmax(probs, dim=-1)

        for agent in obs_dicts:
            idx = int(agent.split("_")[1])
            actions[agent] = int(actions_t[idx].item())

    return actions, claims


# ---------- flowers to arrays (colored by harvester) ----------
def flower_arrays(env, bee_colors):
    xs, ys, zs, cs = [], [], [], []
    for f in env.flowers:
        xs.append(f.x + 0.5)
        ys.append(f.y + 0.5)
        zs.append(0.0)
        if f.harvested:
            if getattr(f, "assigned_bee", None) is not None:
                cs.append(bee_colors[int(f.assigned_bee) % len(bee_colors)])
            else:
                cs.append("tab:green")
        elif getattr(f, "expired", False):
            cs.append("0.7")  # grey
        else:
            cs.append("tab:orange")
    return np.array(xs), np.array(ys), np.array(zs), cs


def make_model_with_hidden(
    ActorCls, CriticCls, num_bees, action_dim, gsize, hidden_dim, num_flowers,
    retask_board_size=0, grid_size=75
):
    actor = ActorCls(
        num_bees=num_bees, action_dim=action_dim, hidden_dim=hidden_dim,
        num_flowers=num_flowers, retask_board_size=retask_board_size, grid_size=grid_size
    )
    critic = CriticCls(
        global_state_size=gsize, num_bees=num_bees, action_dim=action_dim, hidden_dim=hidden_dim,
        grid_size=grid_size
    )
    return actor, critic


# ==========================
# Main
# ==========================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--orbit_alpha", type=float, default=0.9)
    p.add_argument("--orbit_width", type=float, default=1.2)
    p.add_argument("--orbit_samples", type=int, default=400)
    p.add_argument("--trail_len", type=int, default=60, help="frames kept in the tail/trail")
    p.add_argument("--save", type=str, default="")
    # policy controls
    p.add_argument("--policy", action="store_true")
    p.add_argument("--hrl", action="store_true", help="Use HRL (Manager+Worker) instead of flat Actor")
    p.add_argument("--model_tag", type=str, default="best")
    p.add_argument("--model_dir", type=str, default="", help="Override model directory (default: from config.yaml)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--stochastic", action="store_true")
    # ---------- SGP4 toggles ----------
    p.add_argument("--use_sgp4", action="store_true", help="Enable SGP4 for all bees")
    p.add_argument("--sgp4-scale", type=float, default=300.0, help="km per grid unit")
    p.add_argument(
        "--sgp4-stagger", type=int, default=60, help="seconds to offset start time per bee"
    )
    p.add_argument("--tle", type=str, default="", help="Path to text file with 2-line TLE pairs")
    p.add_argument(
        "--tle-repeat",
        action="store_true",
        help="Reuse first TLE for all bees if file has fewer pairs",
    )
    # ---------- Telemetry replay ----------
    p.add_argument(
        "--telemetry",
        type=str,
        default="",
        help="Path to telemetry NDJSON from generate_orbit_telemetry.py",
    )
    p.add_argument(
        "--telemetry-speed", type=float, default=1.0, help="Seconds of telemetry per env step"
    )
    p.add_argument(
        "--telemetry-prefix",
        type=str,
        default="SAT-BEE-",
        help="ID prefix used to map telemetry to bees",
    )
    p.add_argument(
        "--snapshots-dir",
        type=str,
        default="",
        help="Directory to save HUD snapshots (png) every snapshot-interval steps",
    )
    p.add_argument(
        "--snapshot-interval", type=int, default=50, help="Save a HUD snapshot every N steps"
    )
    args = p.parse_args()

    colors = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()
        .get("color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"])
    )

    if not args.policy:
        print("Run with --policy to view the model in action.")
        return

    # --------- POLICY MODE ---------
    from bee_policy import Actor, CentralizedCritic
    from bees_env import BeeForagingEnv
    from train_utils import load_config, load_models

    # Optional: Telemetry stream
    telemetry_enabled = bool(args.telemetry)
    telemetry = None
    telemetry_ids: List[str] = []
    if telemetry_enabled:
        try:
            from telemetry_replay import TelemetryStream

            telemetry = TelemetryStream(args.telemetry)
            # prefer IDs that match the prefix, preserve index order
            all_ids = telemetry.ids()
            pref = [sid for sid in all_ids if sid.startswith(args.telemetry_prefix)]
            others = [sid for sid in all_ids if sid not in pref]
            telemetry_ids = pref + others
        except Exception as e:
            print(f"[telemetry] failed to open {args.telemetry}: {e}")
            telemetry_enabled = False

    # config + env
    cfg = load_config("config.yaml")
    if isinstance(cfg, tuple) and len(cfg) >= 2:
        _, env_cfg, out_dir = cfg
    else:
        d = cfg or {}
        env_cfg = d.get("env", {}) or {}
        out_dir = d.get("output", {}).get(
            "path", d.get("training", {}).get("model_output_path", "output")
        )
    # CLI override for model directory
    if args.model_dir:
        out_dir = args.model_dir

    # overlay SGP4 from CLI (optional)
    if args.use_sgp4:
        env_cfg = dict(env_cfg)
        env_cfg["use_sgp4"] = True
        env_cfg["sgp4_km_per_unit"] = float(args.sgp4_scale)
        env_cfg["sgp4_time_stagger_s"] = int(args.sgp4_stagger)
        tle_pairs = read_tle_pairs(args.tle) if args.tle else []
        if tle_pairs:
            env_cfg["tle_lines"] = tle_pairs
        else:
            env_cfg["tle_lines"] = []

    # construct env (only passing known keys)
    env = BeeForagingEnv(
        **{
            k: v
            for k, v in env_cfg.items()
            if k
            in [
                "num_bees",
                "num_flowers",
                "grid_size",
                "max_steps",
                "time_window_min",
                "time_window_max",
                "harvest_radius",
                "lambda_z",
                "knn_k",
                "orbit_scale",
                "spawn_on_orbit_ratio",
                "shaping_weight",
                "anti_spam_pen",
                "reach_margin",
                "reach_samples",
                "bee_capacity",
                "retask_board_size",
                "retask_timeout_steps",
                "count_idle_as_silent",
                "battery_min_steps",
                "battery_max_steps",
                "recharge_steps",
                "drain_per_step",
                "use_sgp4",
                "sgp4_km_per_unit",
                "sgp4_time_stagger_s",
                "tle_lines",
            ]
        },
        low_battery_chance=0.0,  # Disable low-battery episodes for visualization
    )
    _ = env.reset()
    num_bees = len(env.bees)
    ACTION_DIM = 3

    # ── HRL or flat model loading ──
    use_hrl = getattr(args, 'hrl', False)
    trained_nflowers = getattr(env, "num_flowers", 12)
    rtb_size = getattr(env, "retask_board_size", 0)

    if use_hrl:
        from hrl_policy import ManagerPolicy, GoalConditionedWorker, HRLCritic, build_manager_obs
        NUM_GOALS = 4
        MANAGER_INTERVAL = 10

        # Checkpoint paths
        m_path = os.path.join(out_dir, f"{args.model_tag}_manager.pt")
        w_path = os.path.join(out_dir, f"{args.model_tag}_worker.pt")
        c_path = os.path.join(out_dir, f"{args.model_tag}_critic.pt")

        # Infer hidden_dim and global_state_size from critic checkpoint
        hrl_hidden = 256
        hrl_gsize = env._get_global_state().shape[0]
        try:
            _sd = torch.load(c_path, map_location="cpu", weights_only=True)
            hrl_hidden = _sd["manager_trunk.0.weight"].shape[0]
            hrl_gsize  = _sd["manager_trunk.0.weight"].shape[1]
            del _sd
            print(f"[HRL] inferred from critic ckpt: hidden={hrl_hidden}, gsize={hrl_gsize}")
        except Exception as e:
            print(f"[HRL] could not infer critic dims, using defaults: {e}")

        manager = ManagerPolicy(
            num_bees=num_bees, num_flowers=trained_nflowers, hidden_dim=hrl_hidden,
        )
        worker = GoalConditionedWorker(
            num_bees=num_bees, action_dim=ACTION_DIM, num_goals=NUM_GOALS,
            num_flowers=trained_nflowers, retask_board_size=rtb_size,
            grid_size=env.grid_size, hidden_dim=hrl_hidden,
        )
        hrl_critic = HRLCritic(
            global_state_size=hrl_gsize, num_bees=num_bees,
            num_goals=NUM_GOALS, hidden_dim=hrl_hidden,
        )

        # Load HRL checkpoints
        try:
            manager.load_state_dict(torch.load(m_path, map_location="cpu", weights_only=True))
            print(f"[HRL] loaded {m_path}")
        except Exception as e:
            print(f"[HRL] warning: could not load manager: {e}")
        try:
            worker.load_state_dict(torch.load(w_path, map_location="cpu", weights_only=True))
            print(f"[HRL] loaded {w_path}")
        except Exception as e:
            print(f"[HRL] warning: could not load worker: {e}")
        try:
            hrl_critic.load_state_dict(torch.load(c_path, map_location="cpu", weights_only=True))
            print(f"[HRL] loaded {c_path}")
        except Exception as e:
            print(f"[HRL] warning: could not load critic: {e}")

        manager.eval(); worker.eval(); hrl_critic.eval()
        current_goals = torch.zeros(num_bees, dtype=torch.long)
        hrl_step_counter = [0]  # mutable counter for closure
        actor = None  # not used in HRL mode
        critic = None
        print(f"[HRL] Manager+Worker loaded. Goals every {MANAGER_INTERVAL} steps.")
    else:
        # ── Flat Actor-Critic loading ──
        a_path = os.path.join(out_dir, f"{args.model_tag}_actor.pt")
        c_path = os.path.join(out_dir, f"{args.model_tag}_critic.pt")
        hid = infer_hidden_from_checkpoint(a_path, 128)
        gsize = infer_global_state_size_from_checkpoint(c_path, env._get_global_state().shape[0])
        print(f"[policy] inferred hidden: {hid}, gsize={gsize}, trained_nflowers={trained_nflowers}")

        actor, critic = make_model_with_hidden(
            Actor,
            CentralizedCritic,
            num_bees=num_bees,
            action_dim=ACTION_DIM,
            gsize=gsize,
            hidden_dim=hid,
            num_flowers=trained_nflowers,
            retask_board_size=rtb_size,
            grid_size=env.grid_size,
        )
        try:
            _ = load_models(actor, critic, args.model_tag, out_dir)
        except Exception as e:
            print(f"[load_models] warning: failed to load checkpoint {args.model_tag}: {e}")
            print("[load_models] continuing with random-initialized models for visualization")
        manager = None
        worker = None

    # Map telemetry IDs -> bees (first N)
    if telemetry_enabled:
        if len(telemetry_ids) < num_bees:
            print(f"[telemetry] warning: only {len(telemetry_ids)} streams for {num_bees} bees.")
        telemetry_map = {}
        for i in range(num_bees):
            sid = telemetry_ids[i] if i < len(telemetry_ids) else telemetry_ids[-1]
            telemetry_map[i] = sid
        print("[telemetry] mapping:", telemetry_map)

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        if use_hrl:
            hrl_step_counter[0] = 0
            current_goals.zero_()
        if telemetry_enabled and telemetry is not None:
            telemetry.reset()

        # ---- Clean Dashboard Layout ----
        fig = plt.figure(figsize=(18, 10), facecolor='#1a1a2e')
        gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[2.5, 1, 1], height_ratios=[1, 1, 1], 
                              wspace=0.15, hspace=0.2, left=0.05, right=0.98, top=0.92, bottom=0.05)

        # 3D visualization takes up left 2/3
        ax = fig.add_subplot(gs[:, 0], projection="3d")
        ax.set_facecolor('#0a0a15')
        fig.suptitle(f"🐝 Bee Mission Dashboard — Episode {ep}", fontsize=16, fontweight='bold', color='white')

        # Top right: Bee Status Panel
        bee_ax = fig.add_subplot(gs[0, 1:])
        bee_ax.set_facecolor('#16213e')
        bee_ax.set_xticks([])
        bee_ax.set_yticks([])
        for spine in bee_ax.spines.values():
            spine.set_color('#3a5a8a')
            spine.set_linewidth(2)

        # Middle right: Mission Status
        mission_ax = fig.add_subplot(gs[1, 1:])
        mission_ax.set_facecolor('#1a1a2e')
        mission_ax.set_xticks([])
        mission_ax.set_yticks([])
        for spine in mission_ax.spines.values():
            spine.set_color('#3a5a8a')
            spine.set_linewidth(2)

        # Bottom right: Event Log
        event_ax = fig.add_subplot(gs[2, 1:])
        event_ax.set_facecolor('#0f3460')
        event_ax.set_xticks([])
        event_ax.set_yticks([])
        for spine in event_ax.spines.values():
            spine.set_color('#3a5a8a')
            spine.set_linewidth(2)

        # Event log storage (max 12 events)
        event_log = []
        MAX_EVENTS = 12

        # Track previous state for detecting changes
        prev_battery_dead = [False] * env.num_bees
        prev_assignments = {i: set() for i in range(env.num_bees)}
        prev_retask_board = []

        def log_event(step, event_type, message, color='#333'):
            """Add event to the log"""
            event_log.append({
                'step': step,
                'type': event_type,
                'msg': message,
                'color': color
            })
            while len(event_log) > MAX_EVENTS:
                event_log.pop(0)

        # grid
        cx = env.grid_size / 2.0
        cy = env.grid_size / 2.0
        draw_grid_plane_centered(ax, env.grid_size, cx, cy, 0.0, alpha=0.12)

        # Orbits (skip when telemetry drives motion)
        orbit_lines: List = []

        def rebuild_orbits():
            for ln in orbit_lines:
                ln.remove()
            orbit_lines.clear()
            if telemetry_enabled:
                return  # hide the orbit ribbons in telemetry mode
            for i, b in enumerate(env.bees):
                xs, ys, zs = orbit_samples_for_env_bee(b, env.grid_size, args.orbit_samples)
                (ln,) = ax.plot(
                    xs,
                    ys,
                    zs,
                    linewidth=args.orbit_width,
                    alpha=args.orbit_alpha,
                    solid_capstyle="round",
                    color=colors[i % len(colors)],
                    label=f"Bee {i} orbit",
                )
                orbit_lines.append(ln)

        rebuild_orbits()

        # flowers, colored by harvester
        fx, fy, fz, fc = flower_arrays(env, colors)
        flower_scatter = ax.scatter(
            fx, fy, fz, s=30, depthshade=False, c=fc, marker="o", edgecolors="k", linewidths=0.4
        )

        # axes limits
        try:
            max_r = max(
                getattr(b, "a", env.grid_size) * (1 + getattr(b, "e", 0.0)) for b in env.bees
            )
        except ValueError:
            max_r = env.grid_size
        lim = max(env.grid_size / 2.0, 0.65 * max_r)
        ax.set_xlim(cx - lim, cx + lim)
        ax.set_ylim(cy - lim, cy + lim)
        ax.set_zlim(-lim, lim)
        set_axes_equal(ax)
        ax.set_xlabel("X", color='white')
        ax.set_ylabel("Y", color='white')
        ax.set_zlabel("Z", color='white')
        ax.tick_params(colors='white')

        # Legend on the 3D plot (not in HUD)
        legend_handles = []
        if not telemetry_enabled:
            legend_handles = [
                Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=f"Bee {i}")
                for i in range(num_bees)
            ]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="tab:orange",
                lw=0,
                markeredgecolor="k",
                markeredgewidth=0.4,
                label="Flowers",
            )
        )
        ax.legend(handles=legend_handles, loc='upper left', framealpha=0.7, fontsize=8, 
                  facecolor='#1a1a2e', labelcolor='white')

        # bee markers + trails + status trackers
        scatters, labels, trails, trail_lines = [], [], [], []
        claim_lines = []
        retask_handles = []
        retask_texts = []
        provisional_claim_lines = []
        done_step = [-1] * num_bees

        for i, b in enumerate(env.bees):
            s = ax.scatter(
                [b.fx], [b.fy], [b.fz], s=36, depthshade=True, color=colors[i % len(colors)]
            )
            scatters.append(s)
            labels.append(ax.text(b.fx, b.fy, b.fz, f"{i}", fontsize=9, weight="bold"))
            tr = deque(maxlen=args.trail_len)
            tr.append((b.fx, b.fy, b.fz))
            trails.append(tr)
            (ln,) = ax.plot(
                [b.fx], [b.fy], [b.fz], lw=1.0, alpha=0.75, color=colors[i % len(colors)]
            )
            trail_lines.append(ln)

        # HUD text elements for each panel
        bee_text = bee_ax.text(
            0.03,
            0.92,
            "",
            transform=bee_ax.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            color="#e8e8e8",
        )
        bee_ax.text(0.5, 1.08, "🐝 BEE STATUS", transform=bee_ax.transAxes, 
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='#00d4ff')
        
        mission_text = mission_ax.text(
            0.03,
            0.92,
            "",
            transform=mission_ax.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            color="#e8e8e8",
        )
        mission_ax.text(0.5, 1.08, "🎯 MISSION STATUS", transform=mission_ax.transAxes,
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#ffd700')
        
        trunc_text = bee_ax.text(
            0.97,
            0.92,
            "",
            transform=bee_ax.transAxes,
            va="top",
            ha="right",
            family="monospace",
            fontsize=10,
            color="#ff4444",
            weight="bold",
        )

        # Event log text element
        event_text = event_ax.text(
            0.03,
            0.92,
            "",
            transform=event_ax.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=8,
            color="#b8d4e8",
        )
        event_ax.text(0.5, 1.08, "📡 COMMUNICATION LOG", transform=event_ax.transAxes,
                      ha='center', va='bottom', fontsize=11, fontweight='bold', color='#00ff88')

        # Communication arrows storage (bee_from -> bee_to for task handoffs)
        comm_arrows = []

        last_incl = [float(b.i + getattr(b, "inclination_delta", 0.0)) for b in env.bees]
        last_yaw = [float(b.Omega + getattr(b, "yaw_delta", 0.0)) for b in env.bees]

        # Prepare HUD CSV logging
        snapshots_dir = getattr(args, "snapshots_dir", "").strip()
        logs_dir = snapshots_dir or "logs"
        os.makedirs(logs_dir, exist_ok=True)
        hud_csv_path = os.path.join(logs_dir, "hud_logs.csv")
        write_header = not os.path.isfile(hud_csv_path)
        if write_header:
            try:
                with open(hud_csv_path, "w", encoding="utf-8") as f:
                    f.write("step,bee,action,claim_slot,claim_flower,battery_pct,load,status\n")
            except Exception:
                pass

        def build_bee_status_string():
            """Build the bee status panel content"""
            episode_type = "⚡ LOW-BATTERY" if getattr(env, "_is_low_battery_episode", False) else "🔋 NORMAL"
            
            lines = []
            lines.append(f"Step: {env.steps:4d}/{env.max_steps}  |  Mode: {episode_type}")
            lines.append("")
            lines.append("ID  Battery   Load      Tasks  Done  Status")
            lines.append("─" * 48)

            for i, b in enumerate(env.bees):
                battery_pct = 100.0 * b.battery / max(1e-6, b.battery_capacity)
                
                # Battery bar visualization
                bar_len = 6
                filled = int(battery_pct / 100.0 * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)
                
                load_str = f"{b.load:.0f}/{b.capacity:.0f}"
                assigned_count = sum(1 for f in env.flowers if getattr(f, "assigned_bee", None) == i and not f.harvested)
                harvested_count = sum(1 for f in env.flowers if getattr(f, "assigned_bee", None) == i and f.harvested)

                # Status icons
                if b.battery <= 0:
                    status = "💀 DEAD"
                elif hasattr(b, "_recharge_left_s") and b._recharge_left_s > 0.0:
                    status = "🔌 CHARGING"
                elif battery_pct < 20.0:
                    status = "⚠️ LOW"
                elif b.truncated:
                    status = "✓ DONE"
                else:
                    mode_icons = {0: "💤", 1: "🌸", 2: "✨"}
                    mode_icon = mode_icons.get(int(b.mode), "?")
                    mode_names = {0: "IDLE", 1: "HARVEST", 2: "GROOM"}
                    mode_name = mode_names.get(int(b.mode), "?")
                    status = f"{mode_icon} {mode_name}"

                lines.append(f"{i:2d}  {bar} {battery_pct:>4.0f}%  {load_str:>7s}  {assigned_count:>3d}   {harvested_count:>3d}   {status}")

            return "\n".join(lines)

        def build_mission_status_string():
            """Build the mission status panel content"""
            harvested = sum(1 for f in env.flowers if f.harvested)
            total = len(env.flowers)
            pending = sum(1 for f in env.flowers if not f.harvested and getattr(f, "assigned_bee", None) is not None)
            orphaned = sum(1 for f in env.flowers if not f.harvested and getattr(f, "assigned_bee", None) is None)
            
            # Progress bar
            progress = harvested / max(1, total)
            bar_len = 20
            filled = int(progress * bar_len)
            progress_bar = "█" * filled + "░" * (bar_len - filled)
            
            lines = []
            lines.append(f"Progress: [{progress_bar}] {harvested}/{total}")
            lines.append("")
            
            # Battery summary
            alive = sum(1 for b in env.bees if b.battery > 0)
            dead = env.num_bees - alive
            low_bat = sum(1 for b in env.bees if 0 < b.battery < 0.2 * b.battery_capacity)
            lines.append(f"Bees:  🟢 {alive} active  |  ⚠️ {low_bat} low  |  💀 {dead} dead")
            lines.append(f"Tasks: ✅ {harvested} done  |  ⏳ {pending} pending  |  📋 {orphaned} queued")
            lines.append("")
            
            # Retask queue
            retask_board = getattr(env, 'retask_board', []) or []
            valid_slots = [s for s in retask_board if s.get('flower', -1) >= 0]
            
            if valid_slots:
                lines.append(f"📋 RETASK QUEUE ({len(valid_slots)} orphaned):")
                for idx, slot in enumerate(valid_slots[:4]):
                    fid = slot.get('flower', -1)
                    prio = slot.get('priority', 0)
                    reach = "✓" if slot.get('reachable', False) else "✗"
                    lines.append(f"   #{idx}: Flower {fid}  (prio: {prio:.1f})  reach: {reach}")
            else:
                lines.append("📋 RETASK QUEUE: Empty")
            
            return "\n".join(lines)

        def build_event_log_string():
            """Build the event log display string"""
            if not event_log:
                return "Waiting for events...\n\n📡 Events tracked:\n  • 💀 Battery deaths\n  • 🔄 Task transfers\n  • 📋 Queue changes\n  • ✅ Harvests"
            
            lines = []
            # Event type icons
            icons = {
                'DEATH': '💀',
                'TRANSFER': '🔄', 
                'PICKUP': '📥',
                'ORPHAN': '📋',
                'HARVEST': '✅'
            }
            
            for evt in reversed(event_log[-MAX_EVENTS:]):
                icon = icons.get(evt['type'], '•')
                step_str = f"[{evt['step']:3d}]"
                lines.append(f"{step_str} {icon} {evt['msg']}")
            return "\n".join(lines)

        def detect_events():
            """Detect and log communication/retask events"""
            nonlocal prev_battery_dead, prev_assignments, prev_retask_board
            
            # Clear old communication arrows
            for arrow in comm_arrows:
                try:
                    arrow.remove()
                except Exception:
                    pass
            comm_arrows.clear()
            
            # Detect battery deaths
            for i, b in enumerate(env.bees):
                is_dead = b.battery <= 0
                if is_dead and not prev_battery_dead[i]:
                    # Bee just died!
                    assigned_flowers = [j for j, f in enumerate(env.flowers) 
                                        if f.assigned_bee == i and not f.harvested]
                    log_event(env.steps, "DEATH", 
                              f"Bee {i} DIED! {len(assigned_flowers)} tasks orphaned", 
                              '#c00')
                    prev_battery_dead[i] = True
                    
                    # Mark which flowers are being transferred
                    for fid in assigned_flowers:
                        if not hasattr(env, '_transfer_source'):
                            env._transfer_source = {}
                        env._transfer_source[fid] = i  # flower came from bee i
            
            # Detect assignment changes (task transfers)
            for i in range(env.num_bees):
                current_assigned = set(j for j, f in enumerate(env.flowers) 
                                       if f.assigned_bee == i and not f.harvested)
                
                # New assignments
                new_tasks = current_assigned - prev_assignments[i]
                for task_id in new_tasks:
                    # Check if this was from retask board (orphan pickup)
                    was_orphan = any(entry.get('flower') == task_id 
                                    for entry in prev_retask_board)
                    if was_orphan:
                        # Get the source bee if tracked
                        source_bee = getattr(env, '_transfer_source', {}).get(task_id, None)
                        if source_bee is not None:
                            log_event(env.steps, "TRANSFER", 
                                      f"Flower {task_id}: Bee {source_bee} → Bee {i}", 
                                      '#0a0')
                            # Draw communication arrow from dead bee to new bee
                            try:
                                src_b = env.bees[source_bee]
                                dst_b = env.bees[i]
                                (arrow_line,) = ax.plot(
                                    [src_b.fx, dst_b.fx],
                                    [src_b.fy, dst_b.fy],
                                    [src_b.fz, dst_b.fz],
                                    linestyle='-',
                                    linewidth=2.5,
                                    color='lime',
                                    alpha=0.9,
                                    marker='>',
                                    markersize=8,
                                    markevery=[1],  # marker at end
                                )
                                comm_arrows.append(arrow_line)
                            except Exception:
                                pass
                        else:
                            log_event(env.steps, "PICKUP", 
                                      f"Bee {i} claimed orphan flower {task_id}", 
                                      '#080')
                
                prev_assignments[i] = current_assigned
            
            # Detect retask board changes
            current_retask = list(getattr(env, 'retask_board', []))
            current_orphans = set(e['flower'] for e in current_retask if e.get('flower', -1) >= 0)
            prev_orphans = set(e['flower'] for e in prev_retask_board if e.get('flower', -1) >= 0)
            
            new_orphans = current_orphans - prev_orphans
            for orphan_id in new_orphans:
                log_event(env.steps, "ORPHAN", 
                          f"Flower {orphan_id} added to retask queue", 
                          '#880')
            
            prev_retask_board = current_retask
            
            # Detect harvests
            for j, f in enumerate(env.flowers):
                if f.harvested and j not in getattr(env, '_logged_harvests', set()):
                    if not hasattr(env, '_logged_harvests'):
                        env._logged_harvests = set()
                    env._logged_harvests.add(j)
                    harvester = f.assigned_bee if f.assigned_bee is not None else '?'
                    log_event(env.steps, "HARVEST", 
                              f"Flower {j} harvested by Bee {harvester}", 
                              '#00a')

        def update(_frame_idx):
            nonlocal obs
            if use_hrl:
                actions, claims, _ = choose_actions_hrl(
                    manager, worker, env, obs, current_goals,
                    hrl_step_counter[0], MANAGER_INTERVAL,
                    stochastic=args.stochastic, num_bees=num_bees, device="cpu"
                )
                hrl_step_counter[0] += 1
            else:
                actions, claims = choose_actions(
                    actor, obs, stochastic=args.stochastic, trained_nflowers=trained_nflowers,
                    num_bees=num_bees, device="cpu"
                )
            # draw provisional claim arrows (before env resolves)
            for pl in provisional_claim_lines:
                try:
                    pl.remove()
                except Exception:
                    pass
            provisional_claim_lines.clear()
            for agent, slot_idx in claims.items():
                if slot_idx is None:
                    continue
                try:
                    bee_id = int(agent.split("_")[1])
                except Exception:
                    continue
                if not getattr(env, "retask_board", None):
                    continue
                if slot_idx < 0 or slot_idx >= len(env.retask_board):
                    continue
                slot = env.retask_board[slot_idx]
                if slot.get("flower", -1) < 0:
                    continue
                tx = float(slot.get("x", 0.0)) * env.grid_size
                ty = float(slot.get("y", 0.0)) * env.grid_size
                tz = 0.0
                b = env.bees[bee_id]
                try:
                    (pln,) = ax.plot(
                        [b.fx, tx],
                        [b.fy, ty],
                        [b.fz, tz],
                        linestyle="--",
                        color="magenta",
                        alpha=0.85,
                    )
                    provisional_claim_lines.append(pln)
                except Exception:
                    pass

            next_obs, rewards, terminated, truncated, infos, _ = env.step(actions, claims=claims)
            obs = next_obs

            # Telemetry-driven positions (if enabled)
            if telemetry_enabled and telemetry is not None:
                t = env.steps * float(args.telemetry_speed)
                for i, b in enumerate(env.bees):
                    sid = telemetry_map.get(i)
                    if sid:
                        p = telemetry.get_at_time(sid, t)
                        if p is not None:
                            b.fx, b.fy, b.fz = float(p[0]), float(p[1]), float(p[2])

            # move bees + trails
            for i, (b, s, lbl, tr, ln) in enumerate(
                zip(env.bees, scatters, labels, trails, trail_lines)
            ):
                s._offsets3d = ([b.fx], [b.fy], [b.fz])
                lbl.set_position((b.fx, b.fy))
                lbl.set_3d_properties(b.fz)
                tr.append((b.fx, b.fy, b.fz))
                xs = [p[0] for p in tr]
                ys = [p[1] for p in tr]
                zs = [p[2] for p in tr]
                ln.set_data(xs, ys)
                ln.set_3d_properties(zs)

                # Dim or highlight bees based on status
                # e.g., recharging bees get a semi-transparent marker
                if hasattr(b, "_recharge_left_s") and b._recharge_left_s > 0.0:
                    s.set_alpha(0.6)
                else:
                    s.set_alpha(1.0)

            # update flowers
            fx, fy, fz, fc = flower_arrays(env, colors)
            flower_scatter._offsets3d = (fx, fy, fz)
            flower_scatter.set_color(fc)

            # remove previous claim lines and retask handles
            for ln in claim_lines:
                try:
                    ln.remove()
                except Exception:
                    pass
            claim_lines.clear()
            for h in retask_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            retask_handles.clear()
            for t in retask_texts:
                try:
                    t.remove()
                except Exception:
                    pass
            retask_texts.clear()

            # draw claim lines: for assigned (but not harvested) flowers, draw a thin line from bee to flower
            for j, f in enumerate(env.flowers):
                if getattr(f, "harvested", False):
                    continue
                if getattr(f, "assigned_bee", None) is None:
                    continue
                bid = int(f.assigned_bee)
                if bid < 0 or bid >= num_bees:
                    continue
                b = env.bees[bid]
                try:
                    lx = [b.fx, f.x + 0.5]
                    ly = [b.fy, f.y + 0.5]
                    lz = [b.fz, 0.0]
                    (ln,) = ax.plot(lx, ly, lz, lw=1.2, color=colors[bid % len(colors)], alpha=0.65)
                    claim_lines.append(ln)
                except Exception:
                    pass

            # draw retask board markers (top-M orphan flowers)
            for slot_i, slot in enumerate(getattr(env, "retask_board", []) or []):
                try:
                    if slot.get("flower", -1) >= 0:
                        wx = float(slot.get("x", 0.0)) * env.grid_size
                        wy = float(slot.get("y", 0.0)) * env.grid_size
                        wz = 0.15
                        h = ax.scatter(
                            [wx],
                            [wy],
                            [wz],
                            s=60,
                            depthshade=False,
                            marker="s",
                            color="cyan",
                            edgecolors="k",
                            linewidths=0.5,
                            alpha=0.9,
                        )
                        retask_handles.append(h)
                        txt = ax.text(
                            wx,
                            wy,
                            wz + 0.05,
                            f"#{slot_i}:{int(slot.get('priority',0))}",
                            fontsize=7,
                            color="navy",
                        )
                        retask_texts.append(txt)
                except Exception:
                    pass

            # mark newly done bees; dim their visuals
            for i in range(num_bees):
                if terminated.get(f"bee_{i}", False) and done_step[i] < 0:
                    done_step[i] = env.steps
                    scatters[i].set_alpha(0.3)
                    trail_lines[i].set_alpha(0.25)
                    for ln in orbit_lines:
                        ln.set_alpha(0.25)

            # if any tilt/yaw changed, rebuild orbits (only meaningful when not telemetry)
            if not telemetry_enabled:
                changed = False
                for i, b in enumerate(env.bees):
                    cur_i = float(b.i + getattr(b, "inclination_delta", 0.0))
                    cur_y = float(b.Omega + getattr(b, "yaw_delta", 0.0))
                    if abs(cur_i - last_incl[i]) > math.radians(0.5) or abs(
                        cur_y - last_yaw[i]
                    ) > math.radians(0.5):
                        last_incl[i] = cur_i
                        last_yaw[i] = cur_y
                        changed = True
                if changed:
                    rebuild_orbits()

            # Update all HUD panels
            bee_text.set_text(build_bee_status_string())
            mission_text.set_text(build_mission_status_string())
            trunc_text.set_text("⚠️ TRUNCATED" if any(truncated.values()) else "")
            
            # Detect and log communication/retask events
            detect_events()
            event_text.set_text(build_event_log_string())

            # Append per-bee HUD CSV log for this step
            try:
                with open(hud_csv_path, "a", encoding="utf-8") as f:
                    for i in range(num_bees):
                        agent = f"bee_{i}"
                        action = actions.get(agent, "")
                        claim_slot = claims.get(agent, "")
                        claim_flower = ""
                        try:
                            if (
                                claim_slot is not None
                                and getattr(env, "retask_board", None)
                                and claim_slot >= 0
                                and claim_slot < len(env.retask_board)
                            ):
                                claim_flower = env.retask_board[claim_slot].get("flower", "")
                        except Exception:
                            claim_flower = ""
                        b = env.bees[i]
                        battery_pct = 100.0 * b.battery / max(1e-6, b.battery_capacity)
                        load = b.load
                        status_flags = []
                        if hasattr(b, "_recharge_left_s") and b._recharge_left_s > 0.0:
                            status_flags.append("RECHARGING")
                        if battery_pct < 20.0:
                            status_flags.append(f"CRIT({battery_pct:.0f}%)")
                        heartbeat_age = (
                            env.steps - getattr(env, "_last_broadcast_step", [0] * num_bees)[i]
                        )
                        if heartbeat_age > env.retask_timeout_steps:
                            status_flags.append("SILENT")
                        mode_names = {0: "IDLE", 1: "HARVEST", 2: "GROOM"}
                        mode_name = mode_names.get(int(b.mode), "?")
                        if mode_name == "IDLE" and env.count_idle_as_silent:
                            status_flags.append("IDLE-SILENT")
                        status_str = ",".join(status_flags) if status_flags else "OK"
                        f.write(
                            f"{env.steps},{i},{action},{claim_slot if claim_slot is not None else ''},{claim_flower},{battery_pct:.1f},{load:.2f},{status_str}\n"
                        )
            except Exception:
                pass

            # stop only when ALL bees are terminated or any truncated
            if all(terminated.values()) or any(truncated.values()):
                return []
            # optionally save a HUD snapshot every N steps (if snapshots_dir set)
            try:
                snapshots_dir = getattr(args, "snapshots_dir", "").strip()
                snapshot_interval = max(1, int(getattr(args, "snapshot_interval", 50)))
                if snapshots_dir and (env.steps % snapshot_interval == 0):
                    try:
                        os.makedirs(snapshots_dir, exist_ok=True)
                    except Exception:
                        pass
                    fname = os.path.join(snapshots_dir, f"hud_step_{env.steps:05d}.png")
                    try:
                        fig.savefig(fname, dpi=150)
                    except Exception:
                        pass
            except Exception:
                pass

            return (
                scatters
                + labels
                + trail_lines
                + orbit_lines
                + [flower_scatter, bee_text, mission_text, trunc_text, event_text]
            )

        interval_ms = int(1000.0 / max(1, args.fps))
        ani = animation.FuncAnimation(
            fig, update, frames=env.max_steps, interval=interval_ms, blit=False
        )

        if args.save:
            ani.save(args.save, fps=args.fps)
            print(f"Saved: {args.save}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
