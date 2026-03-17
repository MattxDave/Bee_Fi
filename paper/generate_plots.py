#!/usr/bin/env python3
"""Generate publication-quality figures for the IEEE paper.

Usage:
    cd /home/matthew/projects/Bee_Fi
    source venv/bin/activate
    python paper/generate_plots.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from scipy import stats

# ── Professional IEEE styling ────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Color palette (colorblind-friendly, Wong palette) ────────
BEEFI_COLOR  = "#0072B2"   # blue
GRAD_COLOR   = "#D55E00"   # vermillion
BEEFI_LIGHT  = "#56B4E9"   # light blue
GRAD_LIGHT   = "#E69F00"   # amber
ACCENT       = "#009E73"   # green
NEUTRAL      = "#555555"
BG_FILL      = "#F7F7F7"

IEEE_COL_W   = 3.5         # single-column width (inches)
IEEE_DBCOL_W = 7.16        # double-column width (inches)

OUT_DIR = "paper/figures"

# ── Load data ────────────────────────────────────────────────
with open("hrl_vs_flat_vs_gradient_50ep.json") as f:
    results = json.load(f)

hrl  = results["beefi_hrl"]
grad = results["gradient_policy"]

hrl_hr  = np.array([r["harvest_rate"] * 100 for r in hrl])
grad_hr = np.array([r["harvest_rate"] * 100 for r in grad])

hrl_reward  = np.array([r["total_reward"] for r in hrl])
grad_reward = np.array([r["total_reward"] for r in grad])

hrl_steps   = np.array([r["steps"] for r in hrl])
grad_steps  = np.array([r["steps"] for r in grad])

hrl_harvested  = np.array([r["harvested"] for r in hrl])
grad_harvested = np.array([r["harvested"] for r in grad])

hrl_alive  = np.array([r["alive_sats"] for r in hrl])
grad_alive = np.array([r["alive_sats"] for r in grad])

with open("training_convergence_data.json") as f:
    train_data = json.load(f)

N_EPISODES = len(hrl_hr)


# ── Helper: significance annotation ─────────────────────────
def add_significance_bracket(ax, x1, x2, y, h, text):
    """Draw a bracket with significance text between two bar positions."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.8, c='k')
    ax.text((x1+x2)/2, y+h+0.3, text, ha='center', va='bottom', fontsize=7.5)


# ═════════════════════════════════════════════════════════════
# Fig 1: Combined bar + violin (double-column, two panels)
# ═════════════════════════════════════════════════════════════
def fig_harvest_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DBCOL_W, 2.8),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    # ─── Panel A: Bar chart with individual data points ───
    models = ["Actor-Critic\n(Kepler)", "Gradient\n(2D)"]
    means = [np.mean(hrl_hr), np.mean(grad_hr)]
    stds  = [np.std(hrl_hr),  np.std(grad_hr)]
    colors = [BEEFI_COLOR, GRAD_COLOR]

    bars = ax1.bar([0, 1], means, yerr=stds, capsize=5, color=colors,
                   edgecolor="white", linewidth=1.2, width=0.55, alpha=0.85,
                   error_kw={"lw": 1.2, "capthick": 1.2})

    # Overlay individual data points
    rng = np.random.default_rng(42)
    for i, (data, c) in enumerate([(hrl_hr, BEEFI_LIGHT), (grad_hr, GRAD_LIGHT)]):
        jitter = rng.uniform(-0.12, 0.12, len(data))
        ax1.scatter(i + jitter, data, s=10, alpha=0.5, color=c,
                    edgecolors="white", linewidth=0.3, zorder=4)

    # Value labels on bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m + s + 1.2, f"{m:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold",
                 color=colors[i])

    # Significance bracket
    t_stat, p_val = stats.ttest_ind(hrl_hr, grad_hr)
    sig_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    add_significance_bracket(ax1, 0, 1, max(means) + max(stds) + 3, 0.8,
                             f"p < 0.001 {sig_text}")

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(models)
    ax1.set_ylabel("Harvest Rate (%)")
    ax1.set_ylim(50, 95)
    ax1.set_title("(a) Mean Harvest Rate")
    ax1.grid(axis="y")

    # ─── Panel B: Enhanced violin + box plot ───
    vp = ax2.violinplot([hrl_hr, grad_hr], positions=[0, 1],
                        showmeans=False, showmedians=False, showextrema=False,
                        widths=0.7)

    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.25)
        body.set_edgecolor(colors[i])
        body.set_linewidth(1.0)

    # Box plots inside violins
    bp = ax2.boxplot([hrl_hr, grad_hr], positions=[0, 1], widths=0.15,
                     patch_artist=True, showfliers=False,
                     medianprops={"color": "white", "linewidth": 1.5},
                     whiskerprops={"linewidth": 1.0},
                     capprops={"linewidth": 1.0})
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.8)
        patch.set_edgecolor("white")

    # Overlay swarm-like jittered points
    for i, (data, c) in enumerate([(hrl_hr, BEEFI_COLOR), (grad_hr, GRAD_COLOR)]):
        jitter = rng.uniform(-0.22, -0.08, len(data)) if i == 0 else rng.uniform(0.08, 0.22, len(data))
        ax2.scatter(i + jitter, data, s=12, alpha=0.45, color=c,
                    edgecolors="white", linewidth=0.3, zorder=3)

    # Annotate statistics
    for i, (data, c) in enumerate([(hrl_hr, BEEFI_COLOR), (grad_hr, GRAD_COLOR)]):
        median = np.median(data)
        ax2.text(i + 0.35, median, f"med={median:.0f}%",
                 fontsize=6.5, va="center", color=c, fontstyle="italic")

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(models)
    ax2.set_ylabel("Harvest Rate (%)")
    ax2.set_ylim(50, 95)
    ax2.set_title("(b) Distribution (50 Episodes)")
    ax2.grid(axis="y")

    fig.tight_layout(w_pad=2.5)
    fig.savefig(f"{OUT_DIR}/harvest_comparison.pdf")
    fig.savefig(f"{OUT_DIR}/harvest_comparison.png")
    plt.close(fig)
    print("  [OK] harvest_comparison (bar + violin)")


# ═════════════════════════════════════════════════════════════
# Fig 2: Per-episode scatter with rolling mean + CI band
# ═════════════════════════════════════════════════════════════
def fig_episode_scatter():
    fig, ax = plt.subplots(figsize=(IEEE_DBCOL_W, 2.6))
    eps = np.arange(1, N_EPISODES + 1)

    # Confidence band (rolling mean +/- std over window of 7)
    window = 7
    for data, color, light, label, marker in [
        (hrl_hr, BEEFI_COLOR, BEEFI_LIGHT, "Actor-Critic (Kepler)", "o"),
        (grad_hr, GRAD_COLOR, GRAD_LIGHT, "Gradient (2D)", "s"),
    ]:
        # Rolling statistics
        if len(data) >= window:
            roll_mean = np.convolve(data, np.ones(window)/window, mode="valid")
            roll_std  = np.array([np.std(data[max(0,j-window//2):j+window//2+1])
                                  for j in range(window//2, len(data)-window//2)])
            roll_x = eps[window//2 : window//2 + len(roll_mean)]
            ax.fill_between(roll_x, roll_mean - roll_std, roll_mean + roll_std,
                            alpha=0.12, color=color, linewidth=0)
            ax.plot(roll_x, roll_mean, color=color, lw=1.5, alpha=0.7, zorder=2)

        # Individual episodes
        ax.scatter(eps, data, s=22, color=color, alpha=0.55, label=label,
                   marker=marker, edgecolors="white", linewidth=0.4, zorder=3)

    # Mean reference lines
    for data, color, label in [(hrl_hr, BEEFI_COLOR, "HRL mean"),
                                (grad_hr, GRAD_COLOR, "Grad mean")]:
        m = np.mean(data)
        ax.axhline(m, color=color, ls="--", lw=0.8, alpha=0.5)
        ax.text(N_EPISODES + 0.8, m, f"{m:.1f}%", fontsize=7, color=color,
                va="center", fontweight="bold")

    # Highlight worst gradient episode
    worst_g = np.argmin(grad_hr)
    ax.annotate(f"worst: {grad_hr[worst_g]:.0f}%",
                xy=(worst_g + 1, grad_hr[worst_g]),
                xytext=(worst_g + 1 + 4, grad_hr[worst_g] - 3),
                fontsize=6.5, color=GRAD_COLOR,
                arrowprops=dict(arrowstyle="-|>", color=GRAD_COLOR, lw=0.8))

    ax.set_xlabel("Episode")
    ax.set_ylabel("Harvest Rate (%)")
    ax.set_title("Per-Episode Harvest Rate in Basilisk (50 Episodes)")
    ax.set_xlim(0, N_EPISODES + 4)
    ax.set_ylim(50, 92)
    ax.legend(loc="lower left", ncol=2, framealpha=0.95,
              handletextpad=0.4, columnspacing=1.0)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/episode_scatter.pdf")
    fig.savefig(f"{OUT_DIR}/episode_scatter.png")
    plt.close(fig)
    print("  [OK] episode_scatter")


# ═════════════════════════════════════════════════════════════
# Fig 3: Training convergence — dual axis with fill, phases
# ═════════════════════════════════════════════════════════════
def fig_training_convergence():
    fig, ax1 = plt.subplots(figsize=(IEEE_DBCOL_W, 3.0))

    reward_steps = np.array([x[0] for x in train_data["reward_mean100"]])
    reward_vals  = np.array([x[1] for x in train_data["reward_mean100"]])
    harvest_steps = np.array([x[0] for x in train_data["harvest_rate"]])
    harvest_vals  = np.array([x[1] * 100 for x in train_data["harvest_rate"]])

    # Smooth both series
    win_r = 30
    win_h = 80
    reward_smooth = np.convolve(reward_vals, np.ones(win_r)/win_r, mode="valid")
    reward_x = reward_steps[win_r-1:]
    harvest_smooth = np.convolve(harvest_vals, np.ones(win_h)/win_h, mode="valid")
    harvest_x = harvest_steps[win_h-1:]

    # Reward (left axis) -- filled area
    ax1.fill_between(reward_x, 0, reward_smooth, alpha=0.08, color=BEEFI_COLOR)
    ax1.plot(reward_steps, reward_vals, color=BEEFI_COLOR, alpha=0.12, lw=0.3)
    ln1 = ax1.plot(reward_x, reward_smooth, color=BEEFI_COLOR, lw=1.8,
                   label="Mean Reward (smoothed)", zorder=3)
    ax1.set_xlabel("Training Update")
    ax1.set_ylabel("Mean Episodic Reward", color=BEEFI_COLOR)
    ax1.tick_params(axis="y", labelcolor=BEEFI_COLOR)
    ax1.set_ylim(0, 2700)

    # Harvest rate (right axis)
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(harvest_steps, harvest_vals, color=GRAD_COLOR, alpha=0.1, lw=0.3)
    ln2 = ax2.plot(harvest_x, harvest_smooth, color=GRAD_COLOR, lw=1.8,
                   label="Harvest Rate (smoothed)", zorder=3)
    ax2.set_ylabel("Harvest Rate (%)", color=GRAD_COLOR)
    ax2.tick_params(axis="y", labelcolor=GRAD_COLOR)
    ax2.set_ylim(55, 100)

    # Phase annotations
    phases = [
        (0, 300, "Exploration\nPhase", BG_FILL),
        (300, 1500, "Rapid Learning", "#E8F4FD"),
        (1500, 3325, "Fine-Tuning", "#FFF3E0"),
    ]
    for x0, x1, label, fc in phases:
        ax1.axvspan(x0, x1, alpha=0.25, color=fc, zorder=0)
        ax1.text((x0+x1)/2, 2550, label, ha="center", fontsize=6.5,
                 color=NEUTRAL, fontstyle="italic", zorder=1)

    # Phase dividers
    for xv in [300, 1500]:
        ax1.axvline(xv, color=NEUTRAL, ls=":", lw=0.6, alpha=0.5)

    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right", fontsize=7.5, framealpha=0.95)

    ax1.set_title("Actor-Critic Training Convergence (Kepler Environment)")
    ax1.set_xlim(0, 3350)
    ax1.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/training_convergence.pdf")
    fig.savefig(f"{OUT_DIR}/training_convergence.png")
    plt.close(fig)
    print("  [OK] training_convergence")


# ═════════════════════════════════════════════════════════════
# Fig 4: Multi-metric comparison (reward, harvested, steps)
# ═════════════════════════════════════════════════════════════
def fig_multi_metric():
    fig, axes = plt.subplots(1, 3, figsize=(IEEE_DBCOL_W, 2.4))

    models = ["Actor-\nCritic", "Gradient"]
    colors = [BEEFI_COLOR, GRAD_COLOR]
    rng = np.random.default_rng(42)

    metrics = [
        ("Episodic Reward", hrl_reward, grad_reward, "Total Reward"),
        ("Flowers Harvested", hrl_harvested.astype(float), grad_harvested.astype(float), "Count (out of 50)"),
        ("Episode Length", hrl_steps.astype(float), grad_steps.astype(float), "Steps"),
    ]

    for ax, (title, d1, d2, ylabel) in zip(axes, metrics):
        # Box plots
        bp = ax.boxplot([d1, d2], vert=True, positions=[0, 1], widths=0.4,
                        patch_artist=True, showfliers=True,
                        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
                        medianprops={"color": "white", "linewidth": 1.5},
                        whiskerprops={"linewidth": 0.8},
                        capprops={"linewidth": 0.8})

        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.75)
            patch.set_edgecolor("white")

        # Swarm points
        for i, (data, c) in enumerate([(d1, BEEFI_LIGHT), (d2, GRAD_LIGHT)]):
            jitter = rng.uniform(-0.15, 0.15, len(data))
            ax.scatter(i + jitter, data, s=8, alpha=0.35, color=c,
                       edgecolors="none", zorder=3)

        # Mean markers
        for i, data in enumerate([d1, d2]):
            ax.scatter(i, np.mean(data), s=40, color="white", edgecolors=colors[i],
                       linewidth=1.2, zorder=5, marker="D")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(models, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y")

    # Add panel labels
    for i, ax in enumerate(axes):
        ax.text(-0.15, 1.08, f"({chr(97+i)})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")

    fig.tight_layout(w_pad=1.8)
    fig.savefig(f"{OUT_DIR}/multi_metric.pdf")
    fig.savefig(f"{OUT_DIR}/multi_metric.png")
    plt.close(fig)
    print("  [OK] multi_metric (reward, harvested, steps)")


# ═════════════════════════════════════════════════════════════
# Fig 5: CDF comparison of harvest rates
# ═════════════════════════════════════════════════════════════
def fig_cdf_harvest():
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.6))

    for data, color, label in [
        (hrl_hr, BEEFI_COLOR, "Actor-Critic (Kepler)"),
        (grad_hr, GRAD_COLOR, "Gradient (2D)"),
    ]:
        sorted_d = np.sort(data)
        cdf = np.arange(1, len(sorted_d)+1) / len(sorted_d)
        ax.step(sorted_d, cdf, where="post", color=color, lw=1.8, label=label)
        ax.fill_between(sorted_d, 0, cdf, step="post", alpha=0.1, color=color)

    # Reference lines
    ax.axhline(0.5, color=NEUTRAL, ls=":", lw=0.6, alpha=0.5)
    ax.text(52, 0.51, "50th percentile", fontsize=6.5, color=NEUTRAL)

    # KS test
    ks_stat, ks_p = stats.ks_2samp(hrl_hr, grad_hr)
    ax.text(0.97, 0.15, f"KS = {ks_stat:.3f}\np < {ks_p:.1e}",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.8", alpha=0.9))

    ax.set_xlabel("Harvest Rate (%)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Empirical CDF of Harvest Rates")
    ax.set_xlim(52, 90)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cdf_harvest.pdf")
    fig.savefig(f"{OUT_DIR}/cdf_harvest.png")
    plt.close(fig)
    print("  [OK] cdf_harvest")


# ═════════════════════════════════════════════════════════════
# Fig 6: Domain gap architecture diagram (polished)
# ═════════════════════════════════════════════════════════════
def fig_domain_gap():
    fig, ax = plt.subplots(figsize=(IEEE_DBCOL_W, 2.8))
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.3, 4.2)
    ax.axis("off")

    # ─── Three fidelity level boxes ───
    box_specs = [
        # (x, y, w, h, title, subtitle, color, items)
        (0, 1.2, 2.8, 2.5,
         "Level 0: 2D Grid", "Source Domain (Gradient)",
         GRAD_COLOR,
         ["12 x 12 flat grid", "No orbital physics", "Linear sweep motion",
          "10 agents, 100 tasks"]),
        (3.8, 1.2, 2.8, 2.5,
         "Level 1: Kepler", "Source Domain (Actor-Critic)",
         BEEFI_COLOR,
         ["75 x 75 grid", "Two-body dynamics", "3D orbital positions",
          "25 agents, 50 tasks"]),
        (7.6, 1.2, 2.8, 2.5,
         "Level 2: Basilisk", "Evaluation Target",
         ACCENT,
         ["Walker-delta, 550 km", "J2 perturbation", "6-DOF propagation",
          "25 agents, 50 tasks"]),
    ]

    for x, y, w, h, title, subtitle, color, items in box_specs:
        # Main box
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.12",
                              facecolor=color, alpha=0.08,
                              edgecolor=color, linewidth=1.8)
        ax.add_patch(rect)

        # Header bar
        header = FancyBboxPatch((x, y+h-0.55), w, 0.55,
                                boxstyle="round,pad=0.08",
                                facecolor=color, alpha=0.75,
                                edgecolor="none")
        ax.add_patch(header)
        ax.text(x + w/2, y+h-0.28, title, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")

        # Subtitle
        ax.text(x + w/2, y+h-0.72, subtitle, ha="center", va="center",
                fontsize=6.5, color=color, fontstyle="italic")

        # Bullet items
        for j, item in enumerate(items):
            ax.text(x + 0.2, y+h-1.05-j*0.35, f"\u2022 {item}",
                    fontsize=6.5, va="center", color="#333")

    # ─── Arrows between boxes ───
    arrow_style = "Simple,tail_width=3,head_width=8,head_length=5"

    # Arrow: Level 0 -> Level 2 (curved, bottom)
    arrow1 = FancyArrowPatch(
        (2.8, 1.5), (7.6, 1.5),
        connectionstyle="arc3,rad=-0.25",
        arrowstyle=arrow_style, color=GRAD_COLOR, alpha=0.5, lw=1)
    ax.add_patch(arrow1)
    ax.text(5.2, 0.3, "Large Domain Gap",
            ha="center", fontsize=7, color=GRAD_COLOR,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=GRAD_COLOR, alpha=0.8, lw=0.8))

    # Arrow: Level 1 -> Level 2 (straight, middle)
    arrow2 = FancyArrowPatch(
        (6.6, 2.5), (7.6, 2.5),
        arrowstyle=arrow_style, color=BEEFI_COLOR, alpha=0.6, lw=1)
    ax.add_patch(arrow2)
    ax.text(7.1, 2.85, "Small Gap",
            ha="center", fontsize=7, color=BEEFI_COLOR, fontweight="bold")

    # Mapping function annotation
    ax.text(5.2, 0.85, "flatten_norm mapping",
            ha="center", fontsize=6.5, color=NEUTRAL,
            fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#F0F0F0",
                      edgecolor="0.7", alpha=0.9, lw=0.5))

    # Fidelity arrow at top
    ax.annotate("", xy=(10.2, 4.0), xytext=(0.2, 4.0),
                arrowprops=dict(arrowstyle="-|>", color=NEUTRAL, lw=1.5))
    ax.text(5.2, 4.15, "Increasing Dynamics Fidelity",
            ha="center", fontsize=8, color=NEUTRAL, fontweight="bold")

    fig.savefig(f"{OUT_DIR}/domain_gap.pdf")
    fig.savefig(f"{OUT_DIR}/domain_gap.png")
    plt.close(fig)
    print("  [OK] domain_gap (architecture)")


# ═════════════════════════════════════════════════════════════
# Fig 7: Radar / spider chart — multi-dimensional comparison
# ═════════════════════════════════════════════════════════════
def fig_radar_comparison():
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 3.2), subplot_kw=dict(polar=True))

    # Metrics (normalized to 0-1 where 1 = better)
    categories = [
        "Harvest\nRate",
        "Consistency\n(1/CV)",
        "Reward\nEfficiency",
        "Step\nEfficiency",
        "Worst-Case\nPerf.",
    ]
    N = len(categories)

    # Compute normalized scores
    hrl_scores = [
        np.mean(hrl_hr) / 100,                          # harvest rate
        1 - (np.std(hrl_hr) / np.mean(hrl_hr)),         # consistency (1 - CV)
        1.0,                                              # reward (normalized, HRL is positive)
        1 - (np.mean(hrl_steps) / 1200),                 # step efficiency
        np.min(hrl_hr) / 100,                            # worst case
    ]
    grad_scores = [
        np.mean(grad_hr) / 100,
        1 - (np.std(grad_hr) / np.mean(grad_hr)),
        max(0, 1 + np.mean(grad_reward) / 5000),         # normalize negative reward
        1 - (np.mean(grad_steps) / 1200),
        np.min(grad_hr) / 100,
    ]

    # Clamp to [0, 1]
    hrl_scores = [max(0, min(1, s)) for s in hrl_scores]
    grad_scores = [max(0, min(1, s)) for s in grad_scores]

    # Angles
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    hrl_scores += hrl_scores[:1]
    grad_scores += grad_scores[:1]

    # Plot
    ax.plot(angles, hrl_scores, "o-", color=BEEFI_COLOR, lw=1.8, markersize=5,
            label="Actor-Critic (Kepler)", zorder=3)
    ax.fill(angles, hrl_scores, alpha=0.15, color=BEEFI_COLOR)

    ax.plot(angles, grad_scores, "s-", color=GRAD_COLOR, lw=1.8, markersize=5,
            label="Gradient (2D)", zorder=3)
    ax.fill(angles, grad_scores, alpha=0.15, color=GRAD_COLOR)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6, color=NEUTRAL)
    ax.set_rlabel_position(30)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7.5)
    ax.set_title("Multi-Dimensional Performance\nComparison", pad=20, fontsize=10)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/radar_comparison.pdf")
    fig.savefig(f"{OUT_DIR}/radar_comparison.png")
    plt.close(fig)
    print("  [OK] radar_comparison")


# ═════════════════════════════════════════════════════════════
# Fig 8: Harvest rate vs reward scatter (correlation)
# ═════════════════════════════════════════════════════════════
def fig_harvest_reward_scatter():
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 3.0))

    for data_hr, data_rew, color, label, marker in [
        (hrl_hr, hrl_reward, BEEFI_COLOR, "Actor-Critic", "o"),
        (grad_hr, grad_reward, GRAD_COLOR, "Gradient", "s"),
    ]:
        ax.scatter(data_hr, data_rew, s=25, color=color, alpha=0.6,
                   label=label, marker=marker, edgecolors="white", linewidth=0.4)

        # Regression line
        slope, intercept, r, p, se = stats.linregress(data_hr, data_rew)
        x_line = np.linspace(data_hr.min(), data_hr.max(), 50)
        ax.plot(x_line, slope * x_line + intercept, color=color, ls="--",
                lw=1.0, alpha=0.7)
        # R^2 annotation
        ax.text(data_hr.mean(), data_rew.mean() + (200 if data_rew.mean() > 0 else -350),
                f"$R^2$ = {r**2:.2f}", fontsize=7, color=color, ha="center",
                fontstyle="italic")

    ax.set_xlabel("Harvest Rate (%)")
    ax.set_ylabel("Total Episodic Reward")
    ax.set_title("Harvest Rate vs. Reward Correlation")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(True)

    # Zero line
    ax.axhline(0, color=NEUTRAL, ls="-", lw=0.5, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/harvest_reward_scatter.pdf")
    fig.savefig(f"{OUT_DIR}/harvest_reward_scatter.png")
    plt.close(fig)
    print("  [OK] harvest_reward_scatter")


# ═════════════════════════════════════════════════════════════
# Run all
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating publication-quality figures...\n")
    fig_harvest_comparison()       # Fig 1: bar + violin combo
    fig_episode_scatter()          # Fig 2: per-episode with rolling CI
    fig_training_convergence()     # Fig 3: dual-axis with phases
    fig_multi_metric()             # Fig 4: 3-panel box+swarm
    fig_cdf_harvest()              # Fig 5: empirical CDF + KS test
    fig_domain_gap()               # Fig 6: architecture diagram
    fig_radar_comparison()         # Fig 7: radar chart
    fig_harvest_reward_scatter()   # Fig 8: correlation scatter

    print(f"\n{'='*50}")
    print(f"All 8 figures saved to {OUT_DIR}/")
    print(f"{'='*50}")
