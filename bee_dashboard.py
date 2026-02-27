#!/usr/bin/env python3
"""
BEE-FI Orbital Mission Dashboard
=================================
Interactive 3D visualization of the trained multi-agent satellite
task-scheduling model with real-time telemetry panels.

Usage:
    python bee_dashboard.py                   # default (stochastic, port 8050)
    python bee_dashboard.py --deterministic   # greedy actions
    python bee_dashboard.py --port 8080       # custom port
"""

import argparse
import math
import os
from collections import deque

import numpy as np
import plotly.graph_objects as go
import torch
from dash import Dash, callback_context, dcc, html
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc

# ── project imports (lazy where possible) ──────────────────────────
from bee_orbits_3d import (
    choose_actions,
    infer_global_state_size_from_checkpoint,
    infer_hidden_from_checkpoint,
    orbit_samples_for_env_bee,
)
from bee_policy import Actor, CentralizedCritic
from bees_env import BeeForagingEnv
from train_utils import load_config, load_models

# ═══════════════════════════════════════════════════════════════════
#  THEME  –  deep-space dark palette
# ═══════════════════════════════════════════════════════════════════
DARK_BG = "#0b0b1e"
CARD_BG = "#111128"
CARD_BORDER = "#1e3a5f"
TEXT_COLOR = "#d8dce6"
TEXT_DIM = "#667088"
ACCENT_CYAN = "#00d4ff"
ACCENT_RED = "#ff6b6b"
ACCENT_GOLD = "#ffd93d"
ACCENT_GREEN = "#6bcb77"
ACCENT_BLUE = "#4a9eff"
ACCENT_PURPLE = "#b388ff"
ACCENT_ORANGE = "#ff8a65"

FLOWER_COLORS = {
    "NONE": ACCENT_GREEN,
    "SOFT": ACCENT_GOLD,
    "HARD": ACCENT_RED,
    "HARVESTED": ACCENT_BLUE,
}

FLOWER_SYMBOLS = {
    "NONE": "circle",
    "SOFT": "diamond",
    "HARD": "cross",
    "HARVESTED": "circle-open",
}

# 25 distinct, high-contrast bee colours
BEE_PALETTE = [
    "#ff6b6b", "#ffd93d", "#6bcb77", "#4a9eff", "#b388ff",
    "#ff8a65", "#4dd0e1", "#81c784", "#f06292", "#9575cd",
    "#64b5f6", "#4db6ac", "#dce775", "#ff8a80", "#80deea",
    "#a5d6a7", "#ce93d8", "#ffcc80", "#80cbc4", "#ef9a9a",
    "#90caf9", "#c5e1a5", "#f48fb1", "#b39ddb", "#fff59d",
]


# ═══════════════════════════════════════════════════════════════════
#  SIMULATION STATE  –  holds env + model + history
# ═══════════════════════════════════════════════════════════════════
class SimState:
    def __init__(self):
        self.env = None
        self.actor = None
        self.obs = None
        self.step = 0
        self.running = False
        self.done = False
        self.num_bees = 0
        self.num_flowers = 0
        self.stochastic = True

        # history
        self.harvest_history = []   # [(step, count)]
        self.events = []            # [{step, type, msg}]
        self.max_events = 60

        # visuals
        self.trails = {}            # bee_id → deque[(x,y,z)]
        self.trail_len = 60
        self.orbit_data = {}        # bee_id → (xs, ys, zs)

        # event detection
        self.prev_harvested = set()
        self.prev_dead = set()
        self.prev_recharging = set()

    # ── initialise env + model ──
    def initialize(self, config_path="config.yaml", model_tag="best",
                   model_dir="", stochastic=True):
        self.stochastic = stochastic

        cfg = load_config(config_path)
        if isinstance(cfg, tuple) and len(cfg) >= 2:
            _, env_cfg, out_dir = cfg
        else:
            d = cfg or {}
            env_cfg = d.get("env", {}) or {}
            out_dir = d.get("output", {}).get("path", "output")
        if model_dir:
            out_dir = model_dir

        allowed_keys = {
            "num_bees", "num_flowers", "grid_size", "max_steps",
            "time_window_min", "time_window_max", "harvest_radius",
            "lambda_z", "knn_k", "orbit_scale", "spawn_on_orbit_ratio",
            "shaping_weight", "anti_spam_pen", "reach_margin", "reach_samples",
            "bee_capacity", "retask_board_size", "retask_timeout_steps",
            "count_idle_as_silent", "battery_min_steps", "battery_max_steps",
            "recharge_steps", "drain_per_step",
        }
        self.env = BeeForagingEnv(
            **{k: v for k, v in env_cfg.items() if k in allowed_keys},
            low_battery_chance=0.0,
        )
        self.obs = self.env.reset()
        self.num_bees = len(self.env.bees)
        self.num_flowers = len(self.env.flowers)

        # build + load model
        a_path = os.path.join(out_dir, f"{model_tag}_actor.pt")
        c_path = os.path.join(out_dir, f"{model_tag}_critic.pt")
        hid = infer_hidden_from_checkpoint(a_path, 128)
        gsize = infer_global_state_size_from_checkpoint(
            c_path, self.env._get_global_state().shape[0]
        )
        rtb = getattr(self.env, "retask_board_size", 0)

        self.actor = Actor(
            num_bees=self.num_bees, num_flowers=self.num_flowers,
            retask_board_size=rtb, hidden_dim=hid, grid_size=self.env.grid_size,
        )
        critic = CentralizedCritic(
            global_state_size=gsize, num_bees=self.num_bees,
            hidden_dim=hid, grid_size=self.env.grid_size,
        )
        try:
            load_models(self.actor, critic, model_tag, out_dir)
            print(f"[dashboard] loaded {model_tag} from {out_dir}")
        except Exception as e:
            print(f"[dashboard] warning: {e}  – using random weights")
        self.actor.eval()

        self._post_reset()
        print(f"[dashboard] ready: {self.num_bees} bees, "
              f"{self.num_flowers} flowers, grid={self.env.grid_size}")

    # ── post-reset bookkeeping ──
    def _post_reset(self):
        self.step = 0
        self.done = False
        self.running = False
        self.harvest_history = [(0, 0)]
        self.prev_harvested = set()
        self.prev_dead = set()
        self.prev_recharging = set()
        self.events = []
        for i, b in enumerate(self.env.bees):
            self.trails[i] = deque(maxlen=self.trail_len)
            self.trails[i].append((b.fx, b.fy, b.fz))
        for i, b in enumerate(self.env.bees):
            xs, ys, zs = orbit_samples_for_env_bee(b, self.env.grid_size, 180)
            self.orbit_data[i] = (xs, ys, zs)

    # ── advance one step ──
    def advance(self):
        if self.done or self.env is None:
            return
        actions, claims = choose_actions(
            self.actor, self.obs, stochastic=self.stochastic,
            trained_nflowers=self.num_flowers,
            num_bees=self.num_bees, device="cpu",
        )
        next_obs, rewards, terminated, truncated, infos, _ = self.env.step(
            actions, claims=claims
        )
        self.obs = next_obs
        self.step = self.env.steps

        # trails
        for i, b in enumerate(self.env.bees):
            self.trails[i].append((b.fx, b.fy, b.fz))

        # harvest counter
        harvested = sum(1 for f in self.env.flowers if f.harvested)
        self.harvest_history.append((self.step, harvested))

        # detect events
        self._detect_events()

        if all(terminated.values()) or any(truncated.values()):
            self.done = True
            self.running = False
            self._log("COMPLETE",
                      f"Episode ended – {harvested}/{self.num_flowers} harvested "
                      f"({100*harvested/max(1,self.num_flowers):.0f}%)")

    # ── event detection ──
    def _detect_events(self):
        for j, f in enumerate(self.env.flowers):
            if f.harvested and j not in self.prev_harvested:
                self.prev_harvested.add(j)
                wt = getattr(f, "window_type", "NONE")
                bid = f.assigned_bee if f.assigned_bee is not None else "?"
                self._log("HARVEST", f"Flower {j} ({wt}) → Bee {bid}")
        for i, b in enumerate(self.env.bees):
            if b.battery <= 0 and i not in self.prev_dead:
                self.prev_dead.add(i)
                self._log("DEATH", f"Bee {i} battery depleted")
            recharging = hasattr(b, "_recharge_left_s") and b._recharge_left_s > 0
            if recharging and i not in self.prev_recharging:
                self.prev_recharging.add(i)
                self._log("RECHARGE", f"Bee {i} recharging")
            if not recharging and i in self.prev_recharging:
                self.prev_recharging.discard(i)
                self._log("RECHARGED", f"Bee {i} fully charged")

    def _log(self, etype, msg):
        self.events.append({"step": self.step, "type": etype, "msg": msg})
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    # ── reset ──
    def reset(self):
        if self.env is None:
            return
        self.obs = self.env.reset()
        self._post_reset()


# ── singleton ──
sim = SimState()


# ═══════════════════════════════════════════════════════════════════
#  FIGURE BUILDERS
# ═══════════════════════════════════════════════════════════════════

def _empty_3d():
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def build_3d_figure():
    """Full 3-D viewport rebuild (orbits, bees, flowers, trails, assignments)."""
    if sim.env is None:
        return _empty_3d()

    env = sim.env
    gs = env.grid_size
    cx, cy = gs / 2.0, gs / 2.0
    traces = []

    # ── ground grid (combined into 2 traces) ──
    grid_n = 9
    xs_lines, ys_lines, zs_lines = [], [], []
    for v in np.linspace(0, gs, grid_n):
        xs_lines += [v, v, None]
        ys_lines += [0, gs, None]
        zs_lines += [0, 0, None]
    for v in np.linspace(0, gs, grid_n):
        xs_lines += [0, gs, None]
        ys_lines += [v, v, None]
        zs_lines += [0, 0, None]
    traces.append(go.Scatter3d(
        x=xs_lines, y=ys_lines, z=zs_lines,
        mode="lines", line=dict(color="rgba(80,100,180,0.12)", width=1),
        showlegend=False, hoverinfo="skip",
    ))

    # ── orbits (one trace per bee, thin & dim) ──
    for i in range(sim.num_bees):
        oxs, oys, ozs = sim.orbit_data.get(i, ([], [], []))
        if len(oxs) == 0:
            continue
        # close the loop
        oxs_c = np.append(oxs, oxs[0])
        oys_c = np.append(oys, oys[0])
        ozs_c = np.append(ozs, ozs[0])
        traces.append(go.Scatter3d(
            x=oxs_c, y=oys_c, z=ozs_c,
            mode="lines",
            line=dict(color=BEE_PALETTE[i % len(BEE_PALETTE)], width=1.5),
            opacity=0.18, showlegend=False, hoverinfo="skip",
        ))

    # ── trails ──
    for i in range(sim.num_bees):
        pts = list(sim.trails.get(i, []))
        if len(pts) < 2:
            continue
        traces.append(go.Scatter3d(
            x=[p[0] for p in pts], y=[p[1] for p in pts], z=[p[2] for p in pts],
            mode="lines",
            line=dict(color=BEE_PALETTE[i % len(BEE_PALETTE)], width=3),
            opacity=0.55, showlegend=False, hoverinfo="skip",
        ))

    # ── assignment lines (combined with None separators) ──
    ax_l, ay_l, az_l = [], [], []
    for f in env.flowers:
        if f.harvested or f.assigned_bee is None:
            continue
        bid = int(f.assigned_bee)
        if 0 <= bid < sim.num_bees:
            b = env.bees[bid]
            ax_l += [b.fx, f.x + 0.5, None]
            ay_l += [b.fy, f.y + 0.5, None]
            az_l += [b.fz, 0.0, None]
    if ax_l:
        traces.append(go.Scatter3d(
            x=ax_l, y=ay_l, z=az_l,
            mode="lines",
            line=dict(color="rgba(150,180,255,0.25)", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

    # ── flowers (grouped by type) ──
    flower_groups = {}
    for j, f in enumerate(env.flowers):
        wt = getattr(f, "window_type", "NONE")
        key = "HARVESTED" if f.harvested else wt
        flower_groups.setdefault(key, {"x": [], "y": [], "z": [], "text": []})
        flower_groups[key]["x"].append(f.x + 0.5)
        flower_groups[key]["y"].append(f.y + 0.5)
        flower_groups[key]["z"].append(0.0)
        bid_str = str(f.assigned_bee) if f.assigned_bee is not None else "–"
        flower_groups[key]["text"].append(
            f"Flower {j} ({wt})<br>Assigned: {bid_str}<br>"
            f"Priority: {f.priority:.1f}"
        )
    for wt in ("NONE", "SOFT", "HARD", "HARVESTED"):
        grp = flower_groups.get(wt)
        if not grp:
            continue
        color = FLOWER_COLORS[wt]
        sym = FLOWER_SYMBOLS[wt]
        sz = 4 if wt == "HARVESTED" else 6
        opa = 0.45 if wt == "HARVESTED" else 0.92
        traces.append(go.Scatter3d(
            x=grp["x"], y=grp["y"], z=grp["z"],
            mode="markers",
            marker=dict(color=color, size=sz, symbol=sym, opacity=opa,
                        line=dict(color="white", width=0.5)),
            name=wt, showlegend=True,
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=grp["text"],
        ))

    # ── retask board markers ──
    for slot_i, slot in enumerate(getattr(env, "retask_board", []) or []):
        if slot.get("flower", -1) >= 0:
            wx = float(slot.get("x", 0)) * gs
            wy = float(slot.get("y", 0)) * gs
            traces.append(go.Scatter3d(
                x=[wx], y=[wy], z=[0.15],
                mode="markers+text",
                marker=dict(color="cyan", size=7, symbol="square",
                            line=dict(color="white", width=1)),
                text=[f"Q{slot_i}"], textfont=dict(size=7, color="cyan"),
                textposition="top center",
                showlegend=False, hoverinfo="skip",
            ))

    # ── bees (single trace with per-point colours) ──
    bx, by, bz, bc, bsz, bt = [], [], [], [], [], []
    mode_icons = {0: "IDLE", 1: "HARVEST", 2: "GROOM"}
    for i, b in enumerate(env.bees):
        bx.append(b.fx); by.append(b.fy); bz.append(b.fz)
        col = BEE_PALETTE[i % len(BEE_PALETTE)]
        if b.battery <= 0:
            col = "#555555"
        bc.append(col)
        recharging = hasattr(b, "_recharge_left_s") and b._recharge_left_s > 0
        bsz.append(5 if b.battery <= 0 else (7 if recharging else 9))
        bpct = 100.0 * b.battery / max(1e-6, b.battery_capacity)
        mn = mode_icons.get(int(b.mode), "?")
        bt.append(f"Bee {i}<br>Battery: {bpct:.0f}%<br>Mode: {mn}<br>"
                  f"Load: {b.load:.0f}/{b.capacity:.0f}")
    traces.append(go.Scatter3d(
        x=bx, y=by, z=bz,
        mode="markers+text",
        marker=dict(color=bc, size=bsz, symbol="circle",
                    line=dict(color="white", width=1)),
        text=[str(i) for i in range(sim.num_bees)],
        textfont=dict(size=7, color="white"),
        textposition="top center",
        showlegend=False,
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=bt,
    ))

    # ── layout ──
    max_r = max(
        (getattr(b, "a", gs) * (1 + getattr(b, "e", 0.0)) for b in env.bees),
        default=gs,
    )
    lim = max(gs / 2.0, 0.65 * max_r)
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[cx - lim, cx + lim], showbackground=False,
                       showgrid=False, showticklabels=False, title=""),
            yaxis=dict(range=[cy - lim, cy + lim], showbackground=False,
                       showgrid=False, showticklabels=False, title=""),
            zaxis=dict(range=[-lim, lim], showbackground=False,
                       showgrid=False, showticklabels=False, title=""),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.7),
                        up=dict(x=0, y=0, z=1)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(15,15,40,0.85)",
            font=dict(color=TEXT_COLOR, size=10),
            bordercolor=CARD_BORDER, borderwidth=1,
            orientation="h",
        ),
        uirevision="keep-camera",
    )
    return fig


def build_battery_chart():
    """Horizontal bar chart of battery levels for all bees."""
    if sim.env is None:
        return go.Figure()
    bees = sim.env.bees
    ids = list(range(len(bees)))
    bats = [100.0 * b.battery / max(1e-6, b.battery_capacity) for b in bees]
    bar_colors = []
    for pct in bats:
        if pct <= 0:
            bar_colors.append("#444")
        elif pct < 20:
            bar_colors.append(ACCENT_RED)
        elif pct < 50:
            bar_colors.append(ACCENT_GOLD)
        else:
            bar_colors.append(ACCENT_GREEN)
    fig = go.Figure(go.Bar(
        x=bats, y=[f"B{i}" for i in ids], orientation="h",
        marker_color=bar_colors,
        marker_line=dict(color="rgba(255,255,255,0.15)", width=0.5),
        hovertemplate="Bee %{y}<br>Battery: %{x:.0f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=28, r=8, t=4, b=4),
        xaxis=dict(range=[0, 105], showgrid=True,
                   gridcolor="rgba(100,100,200,0.1)", color="#556",
                   tickfont=dict(size=8), ticksuffix="%"),
        yaxis=dict(showgrid=False, color="#556", tickfont=dict(size=7),
                   autorange="reversed"),
        bargap=0.15, height=max(180, sim.num_bees * 14 + 30),
    )
    return fig


def build_harvest_chart():
    """Area chart of harvested flower count over time."""
    if not sim.harvest_history:
        return go.Figure()
    steps = [h[0] for h in sim.harvest_history]
    counts = [h[1] for h in sim.harvest_history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=counts, mode="lines", fill="tozeroy",
        line=dict(color=ACCENT_CYAN, width=2),
        fillcolor="rgba(0,212,255,0.12)",
        hovertemplate="Step %{x}<br>Harvested: %{y}<extra></extra>",
    ))
    if sim.num_flowers > 0:
        fig.add_hline(y=sim.num_flowers, line_dash="dash",
                      line_color=ACCENT_GOLD, opacity=0.5,
                      annotation_text=f"Target {sim.num_flowers}",
                      annotation_font=dict(color=ACCENT_GOLD, size=9))
    max_step = sim.env.max_steps if sim.env else 1200
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=8, t=4, b=20),
        xaxis=dict(range=[0, max_step], showgrid=False, color="#556",
                   tickfont=dict(size=8)),
        yaxis=dict(range=[0, sim.num_flowers + 3], showgrid=True,
                   gridcolor="rgba(100,100,200,0.1)", color="#556",
                   tickfont=dict(size=8)),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
#  HTML COMPONENT BUILDERS
# ═══════════════════════════════════════════════════════════════════

def _card_style():
    return {"background": CARD_BG, "border": f"1px solid {CARD_BORDER}",
            "borderRadius": 6}


def build_metric_cards():
    if sim.env is None:
        return html.Div()
    harvested = sum(1 for f in sim.env.flowers if f.harvested)
    total = sim.num_flowers
    pct = 100.0 * harvested / max(1, total)
    alive = sum(1 for b in sim.env.bees if b.battery > 0)
    dead = sim.num_bees - alive
    charging = sum(1 for b in sim.env.bees
                   if hasattr(b, "_recharge_left_s") and b._recharge_left_s > 0)

    # counts by flower type
    h_none = sum(1 for f in sim.env.flowers if f.harvested and f.window_type == "NONE")
    h_soft = sum(1 for f in sim.env.flowers if f.harvested and f.window_type == "SOFT")
    h_hard = sum(1 for f in sim.env.flowers if f.harvested and f.window_type == "HARD")
    t_none = sum(1 for f in sim.env.flowers if f.window_type == "NONE")
    t_soft = sum(1 for f in sim.env.flowers if f.window_type == "SOFT")
    t_hard = sum(1 for f in sim.env.flowers if f.window_type == "HARD")

    pct_color = ACCENT_GREEN if pct >= 90 else (ACCENT_GOLD if pct >= 50 else ACCENT_RED)

    def _metric(icon, label, value, color, detail=""):
        children = [
            html.Div(f"{icon} {label}",
                     style={"fontSize": 10, "color": TEXT_DIM,
                            "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Div(str(value),
                     style={"fontSize": 20, "fontWeight": "bold", "color": color,
                            "fontFamily": "monospace", "lineHeight": "1.2"}),
        ]
        if detail:
            children.append(html.Div(detail, style={"fontSize": 9, "color": TEXT_DIM}))
        return dbc.Col(dbc.Card(dbc.CardBody(children, style={"padding": "8px 10px"}),
                                style=_card_style()),
                       width=True, className="px-1")

    return dbc.Row([
        _metric("🌾", "Harvested", f"{harvested}/{total}", pct_color),
        _metric("📊", "Rate", f"{pct:.0f}%", pct_color),
        _metric("🟢", "Active", str(alive),
                ACCENT_GREEN if dead == 0 else ACCENT_GOLD,
                f"🔌 {charging} charging" if charging else ""),
        _metric("💀", "Dead", str(dead), ACCENT_RED if dead else "#556"),
        _metric("●", "NONE", f"{h_none}/{t_none}", ACCENT_GREEN),
        _metric("◆", "SOFT", f"{h_soft}/{t_soft}", ACCENT_GOLD),
        _metric("✕", "HARD", f"{h_hard}/{t_hard}", ACCENT_RED),
    ], className="g-1 mb-2")


def build_event_log():
    if not sim.events:
        return html.Div("Waiting for events…",
                        style={"color": "#445", "fontStyle": "italic", "padding": 8})
    icons = {
        "HARVEST": ("✅", ACCENT_GREEN), "DEATH": ("💀", ACCENT_RED),
        "RECHARGE": ("🔌", ACCENT_GOLD), "RECHARGED": ("⚡", ACCENT_CYAN),
        "COMPLETE": ("🏁", ACCENT_PURPLE), "TRANSFER": ("🔄", ACCENT_CYAN),
    }
    items = []
    for evt in reversed(sim.events[-25:]):
        icon, color = icons.get(evt["type"], ("•", TEXT_DIM))
        items.append(html.Div([
            html.Span(f"[{evt['step']:4d}] ", style={"color": "#445"}),
            html.Span(f"{icon} "),
            html.Span(evt["msg"], style={"color": color}),
        ], style={"padding": "1px 0",
                  "borderBottom": "1px solid rgba(255,255,255,0.03)"}))
    return html.Div(items)


def build_bee_table():
    if sim.env is None:
        return html.Div()

    header = html.Tr([
        html.Th("ID", style={"width": 30}), html.Th("Battery"),
        html.Th("Mode"), html.Th("Load"), html.Th("Assigned"),
        html.Th("Harvested"),
    ], style={"color": ACCENT_PURPLE, "fontSize": 10,
              "borderBottom": f"1px solid {CARD_BORDER}"})

    rows = []
    for i, b in enumerate(sim.env.bees):
        bpct = 100.0 * b.battery / max(1e-6, b.battery_capacity)
        bc = "#444" if bpct <= 0 else (ACCENT_RED if bpct < 20 else
             (ACCENT_GOLD if bpct < 50 else ACCENT_GREEN))
        bar = html.Div(
            html.Div(style={"width": f"{max(0, min(100, bpct)):.0f}%",
                            "height": "7px", "background": bc,
                            "borderRadius": 2}),
            style={"width": 50, "height": 7, "background": "#1a1a30",
                   "borderRadius": 2, "display": "inline-block",
                   "verticalAlign": "middle"},
        )
        m_icons = {0: "💤", 1: "🌸", 2: "✨"}
        mode = "💀" if b.battery <= 0 else (
            "🔌" if hasattr(b, "_recharge_left_s") and b._recharge_left_s > 0
            else m_icons.get(int(b.mode), "?"))
        assigned = sum(1 for f in sim.env.flowers
                       if getattr(f, "assigned_bee", None) == i and not f.harvested)
        done = sum(1 for f in sim.env.flowers
                   if getattr(f, "assigned_bee", None) == i and f.harvested)
        color = BEE_PALETTE[i % len(BEE_PALETTE)]
        rows.append(html.Tr([
            html.Td(str(i), style={"color": color, "fontWeight": "bold"}),
            html.Td([bar, html.Span(f" {bpct:.0f}%", style={"fontSize": 9})]),
            html.Td(mode, style={"textAlign": "center"}),
            html.Td(f"{b.load:.0f}/{b.capacity:.0f}"),
            html.Td(str(assigned), style={"textAlign": "center"}),
            html.Td(str(done), style={"color": ACCENT_GREEN if done else "#445",
                                       "textAlign": "center"}),
        ], style={"borderBottom": "1px solid rgba(255,255,255,0.03)",
                  "fontSize": 10, "color": TEXT_COLOR}))

    return html.Table([html.Thead(header), html.Tbody(rows)],
                      style={"width": "100%"})


# ═══════════════════════════════════════════════════════════════════
#  DASH APP LAYOUT
# ═══════════════════════════════════════════════════════════════════

def create_app():
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.SUPERHERO],
        suppress_callback_exceptions=True,
        title="BEE-FI Dashboard",
    )

    controls_bar = dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button("▶  Play", id="play-btn", color="success",
                           size="sm", className="me-1",
                           style={"fontWeight": "bold", "letterSpacing": "0.5px"}),
                dbc.Button("⏸  Pause", id="pause-btn", color="warning",
                           size="sm", className="me-1"),
                dbc.Button("⏭  Step", id="step-btn", color="info",
                           size="sm", className="me-1"),
                dbc.Button("🔄  Reset", id="reset-btn", color="danger",
                           size="sm"),
            ]),
        ], width=4),
        dbc.Col([
            html.Div([
                html.Span("Speed ", style={"color": TEXT_DIM, "fontSize": 12,
                                            "marginRight": 6}),
                dcc.Slider(id="speed-slider", min=0.5, max=10, step=0.5,
                           value=2,
                           marks={1: "1×", 2: "2×", 5: "5×", 10: "10×"},
                           tooltip={"always_visible": False}),
            ], style={"display": "flex", "alignItems": "center"}),
        ], width=5),
        dbc.Col(html.Div(id="status-badge",
                         style={"textAlign": "right", "paddingTop": 4}),
                width=3),
    ], className="g-0", style={
        "background": CARD_BG, "padding": "6px 14px",
        "borderRadius": 6, "border": f"1px solid {CARD_BORDER}",
    })

    app.layout = dbc.Container([
        # ── header ──
        dbc.Row([
            dbc.Col([
                html.H3("🐝  BEE-FI  ORBITAL  MISSION  DASHBOARD",
                         style={"color": ACCENT_CYAN, "fontWeight": "bold",
                                "margin": 0, "letterSpacing": "1.5px"}),
                html.Span("Multi-Agent Satellite Task-Scheduling Visualization",
                          style={"color": TEXT_DIM, "fontSize": 13}),
            ], width=8),
            dbc.Col(html.Div(id="step-display",
                             style={"fontSize": 22, "fontWeight": "bold",
                                    "color": TEXT_COLOR, "textAlign": "right",
                                    "fontFamily": "monospace", "paddingTop": 6}),
                    width=4),
        ], className="mt-2 mb-2"),

        # ── controls ──
        controls_bar,
        html.Div(style={"height": 8}),

        # ── metric cards ──
        html.Div(id="metric-cards"),

        # ── main row: 3-D + right panels ──
        dbc.Row([
            # left: 3-D viewport
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dcc.Graph(id="viewport-3d",
                                  style={"height": "58vh"},
                                  config={"scrollZoom": True,
                                          "displayModeBar": True,
                                          "modeBarButtonsToRemove": ["toImage"]}),
                        style={"padding": 2},
                    ),
                    style=_card_style(),
                ),
                width=7,
            ),
            # right column
            dbc.Col([
                # battery
                dbc.Card([
                    dbc.CardHeader("🔋  BATTERY LEVELS", style={
                        "color": ACCENT_GOLD, "fontWeight": "bold", "fontSize": 12,
                        "background": CARD_BG,
                        "borderBottom": f"1px solid {CARD_BORDER}",
                        "padding": "5px 10px",
                    }),
                    dbc.CardBody(
                        dcc.Graph(id="battery-chart",
                                  style={"height": "25vh"},
                                  config={"displayModeBar": False}),
                        style={"padding": 2},
                    ),
                ], style=_card_style(), className="mb-2"),
                # event log
                dbc.Card([
                    dbc.CardHeader("📡  EVENT LOG", style={
                        "color": ACCENT_GREEN, "fontWeight": "bold", "fontSize": 12,
                        "background": CARD_BG,
                        "borderBottom": f"1px solid {CARD_BORDER}",
                        "padding": "5px 10px",
                    }),
                    dbc.CardBody(
                        html.Div(id="event-log",
                                 style={"height": "24vh", "overflowY": "auto",
                                        "fontFamily": "monospace", "fontSize": 11}),
                        style={"padding": "4px 8px"},
                    ),
                ], style=_card_style()),
            ], width=5),
        ], className="mb-2"),

        # ── bottom row: harvest chart + bee table ──
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("📈  HARVEST TIMELINE", style={
                        "color": ACCENT_CYAN, "fontWeight": "bold", "fontSize": 12,
                        "background": CARD_BG,
                        "borderBottom": f"1px solid {CARD_BORDER}",
                        "padding": "5px 10px",
                    }),
                    dbc.CardBody(
                        dcc.Graph(id="harvest-chart",
                                  style={"height": "20vh"},
                                  config={"displayModeBar": False}),
                        style={"padding": 2},
                    ),
                ], style=_card_style()),
                width=5,
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("🐝  BEE STATUS", style={
                        "color": ACCENT_PURPLE, "fontWeight": "bold", "fontSize": 12,
                        "background": CARD_BG,
                        "borderBottom": f"1px solid {CARD_BORDER}",
                        "padding": "5px 10px",
                    }),
                    dbc.CardBody(
                        html.Div(id="bee-table",
                                 style={"height": "20vh", "overflowY": "auto",
                                        "fontFamily": "monospace", "fontSize": 10}),
                        style={"padding": "4px 8px"},
                    ),
                ], style=_card_style()),
                width=7,
            ),
        ]),

        # ── tick interval ──
        dcc.Interval(id="interval", interval=200, disabled=True),
    ], fluid=True, style={"background": DARK_BG, "minHeight": "100vh",
                          "padding": "0 10px"})

    return app


# ═══════════════════════════════════════════════════════════════════
#  CALLBACKS
# ═══════════════════════════════════════════════════════════════════

def register_callbacks(app):

    # ── play / pause ──
    @app.callback(
        Output("interval", "disabled"),
        [Input("play-btn", "n_clicks"),
         Input("pause-btn", "n_clicks")],
        State("interval", "disabled"),
        prevent_initial_call=True,
    )
    def toggle_play(play_clicks, pause_clicks, currently_disabled):
        ctx = callback_context
        if not ctx.triggered:
            return True
        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        if btn == "play-btn":
            sim.running = True
            return False
        sim.running = False
        return True

    # ── speed ──
    @app.callback(
        Output("interval", "interval"),
        Input("speed-slider", "value"),
    )
    def update_speed(speed):
        return max(30, int(200 / max(0.1, speed)))

    # ── main update ──
    @app.callback(
        [Output("viewport-3d", "figure"),
         Output("battery-chart", "figure"),
         Output("harvest-chart", "figure"),
         Output("metric-cards", "children"),
         Output("event-log", "children"),
         Output("bee-table", "children"),
         Output("step-display", "children"),
         Output("status-badge", "children")],
        [Input("interval", "n_intervals"),
         Input("step-btn", "n_clicks"),
         Input("reset-btn", "n_clicks")],
        prevent_initial_call=False,
    )
    def update_all(n_intervals, step_clicks, reset_clicks):
        ctx = callback_context
        triggered = (ctx.triggered[0]["prop_id"].split(".")[0]
                     if ctx.triggered else "")

        if triggered == "reset-btn" and reset_clicks:
            sim.reset()
        elif triggered in ("interval", "step-btn"):
            if not sim.done:
                sim.advance()

        # build visuals
        fig_3d = build_3d_figure()
        fig_bat = build_battery_chart()
        fig_harv = build_harvest_chart()
        cards = build_metric_cards()
        events = build_event_log()
        bee_tbl = build_bee_table()

        max_steps = sim.env.max_steps if sim.env else 0
        step_text = f"Step  {sim.step:4d} / {max_steps}"

        if sim.done:
            badge = dbc.Badge("✓ COMPLETE", color="success",
                              className="p-2", style={"fontSize": 13})
        elif sim.running:
            badge = dbc.Badge("● RUNNING", color="primary",
                              className="p-2", style={"fontSize": 13})
        else:
            badge = dbc.Badge("⏸ PAUSED", color="warning",
                              className="p-2", style={"fontSize": 13})

        return fig_3d, fig_bat, fig_harv, cards, events, bee_tbl, step_text, badge


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="BEE-FI Orbital Mission Dashboard")
    p.add_argument("--model_tag", type=str, default="best")
    p.add_argument("--model_dir", type=str, default="")
    p.add_argument("--stochastic", action="store_true", default=True)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    sim.initialize(
        config_path="config.yaml",
        model_tag=args.model_tag,
        model_dir=args.model_dir,
        stochastic=not args.deterministic,
    )

    app = create_app()
    register_callbacks(app)

    print(f"\n{'═' * 54}")
    print(f"  🐝  BEE-FI Dashboard → http://localhost:{args.port}")
    print(f"  Press ▶ Play to start the simulation")
    print(f"{'═' * 54}\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
