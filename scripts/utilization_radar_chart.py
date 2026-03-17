#!/usr/bin/env python3
"""
Multi-Dimensional Utilization Radar Chart
==========================================
Compares HRL vs Gradient across 6 scheduling-utilization axes
using data from hrl_vs_flat_vs_gradient_50ep.json (50 episodes each).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

# ── Load data ──────────────────────────────────────────────────────────────────
with open("hrl_vs_flat_vs_gradient_50ep.json") as f:
    data = json.load(f)

hrl_eps = data["beefi_hrl"]
grad_eps = data["gradient_policy"]
NUM_SATS = 25  # from config

# ── Compute per-episode metrics ───────────────────────────────────────────────
def compute_metrics(episodes):
    harvest_rates = []
    hard_window_rates = []
    soft_window_rates = []
    none_window_rates = []
    sat_survival_rates = []
    step_efficiencies = []

    for ep in episodes:
        harvest_rates.append(ep["harvest_rate"])
        hard_window_rates.append(ep["hard_done"] / ep["hard_total"] if ep["hard_total"] > 0 else 0)
        soft_window_rates.append(ep["soft_done"] / ep["soft_total"] if ep["soft_total"] > 0 else 0)
        none_window_rates.append(ep["none_done"] / ep["none_total"] if ep["none_total"] > 0 else 0)
        sat_survival_rates.append(ep["alive_sats"] / NUM_SATS)
        # Normalize step efficiency: harvested per 100 steps (higher = better)
        step_efficiencies.append(ep["harvested"] / ep["steps"] if ep["steps"] > 0 else 0)

    return {
        "Harvest\nRate": harvest_rates,
        "Hard Window\nSuccess": hard_window_rates,
        "Soft Window\nSuccess": soft_window_rates,
        "No-Window\nSuccess": none_window_rates,
        "Satellite\nSurvival": sat_survival_rates,
        "Step\nEfficiency": step_efficiencies,
    }

hrl_metrics = compute_metrics(hrl_eps)
grad_metrics = compute_metrics(grad_eps)

# ── Axis labels and mean values ───────────────────────────────────────────────
labels = list(hrl_metrics.keys())
N = len(labels)

hrl_means = [np.mean(hrl_metrics[k]) for k in labels]
grad_means = [np.mean(grad_metrics[k]) for k in labels]
hrl_stds = [np.std(hrl_metrics[k]) for k in labels]
grad_stds = [np.std(grad_metrics[k]) for k in labels]

# Normalize Step Efficiency to 0-1 scale (divide by max across both)
eff_idx = labels.index("Step\nEfficiency")
max_eff = max(hrl_means[eff_idx] + hrl_stds[eff_idx],
              grad_means[eff_idx] + grad_stds[eff_idx])
if max_eff > 0:
    hrl_means[eff_idx] /= max_eff
    grad_means[eff_idx] /= max_eff
    hrl_stds[eff_idx] /= max_eff
    grad_stds[eff_idx] /= max_eff

# ── Radar chart ───────────────────────────────────────────────────────────────
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
# Close the polygon
hrl_vals = hrl_means + [hrl_means[0]]
grad_vals = grad_means + [grad_means[0]]
hrl_std_vals = hrl_stds + [hrl_stds[0]]
grad_std_vals = grad_stds + [grad_stds[0]]
angles += [angles[0]]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("#fafafa")
ax.set_facecolor("#fafafa")

# Style
HRL_COLOR = "#1f77b4"
GRAD_COLOR = "#d35400"

# Grid styling
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                   fontsize=9, color="#666666")
ax.yaxis.grid(True, color="#cccccc", linewidth=0.5, linestyle="--")
ax.xaxis.grid(True, color="#cccccc", linewidth=0.5)
ax.spines["polar"].set_visible(False)

# Plot HRL
ax.plot(angles, hrl_vals, "o-", color=HRL_COLOR, linewidth=2.5,
        markersize=7, label="HRL (Kepler)", zorder=5)
ax.fill(angles, [max(0, v - s) for v, s in zip(hrl_vals, hrl_std_vals)],
        alpha=0.0, color=HRL_COLOR)  # invisible lower bound
hrl_upper = [min(1.05, v + s) for v, s in zip(hrl_vals, hrl_std_vals)]
hrl_lower = [max(0, v - s) for v, s in zip(hrl_vals, hrl_std_vals)]
ax.fill_between(angles, hrl_lower, hrl_upper, alpha=0.12, color=HRL_COLOR,
                zorder=2)

# Plot Gradient
ax.plot(angles, grad_vals, "s--", color=GRAD_COLOR, linewidth=2.5,
        markersize=7, label="Gradient (Kepler)", zorder=5)
grad_upper = [min(1.05, v + s) for v, s in zip(grad_vals, grad_std_vals)]
grad_lower = [max(0, v - s) for v, s in zip(grad_vals, grad_std_vals)]
ax.fill_between(angles, grad_lower, grad_upper, alpha=0.12, color=GRAD_COLOR,
                zorder=2)

# Axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11, fontweight="bold", color="#333333")

# Shift labels outward
for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
    if angle in (0, np.pi):
        label.set_horizontalalignment("center")
    elif 0 < angle < np.pi:
        label.set_horizontalalignment("left")
    else:
        label.set_horizontalalignment("right")

# Title
ax.set_title("Multi-Dimensional Utilization\nPerformance Comparison",
             fontsize=18, fontweight="bold", color="#222222",
             pad=30, linespacing=1.3)

# Legend
legend = ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12),
                   fontsize=12, frameon=True, fancybox=True,
                   shadow=True, borderpad=1.0)
legend.get_frame().set_facecolor("#ffffff")
legend.get_frame().set_edgecolor("#cccccc")

# ── Add annotation box with raw numbers ───────────────────────────────────────
raw_labels_short = ["Harvest", "Hard Win.", "Soft Win.", "No-Win.", "Sat. Surv.", "Step Eff."]
# Recover un-normalized step efficiency for display
hrl_raw = [np.mean(hrl_metrics[k]) for k in hrl_metrics]
grad_raw = [np.mean(grad_metrics[k]) for k in grad_metrics]

text_lines = "  Metric            HRL     Grad    Delta\n"
text_lines += "  " + "─" * 44 + "\n"
for i, lbl in enumerate(raw_labels_short):
    h = hrl_raw[i]
    g = grad_raw[i]
    delta = h - g
    sign = "+" if delta >= 0 else ""
    if i == 5:  # step efficiency — show as flowers/step
        text_lines += f"  {lbl:<16s} {h:.4f}  {g:.4f}  {sign}{delta:.4f}\n"
    else:
        text_lines += f"  {lbl:<16s} {h:.3f}   {g:.3f}   {sign}{delta:.3f}\n"

fig.text(0.50, -0.02, text_lines, fontsize=9, fontfamily="monospace",
         ha="center", va="top", color="#444444",
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f0f0",
                   edgecolor="#cccccc", alpha=0.9))

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("utilization_radar_chart.png", dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.savefig("utilization_radar_chart.pdf", bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: utilization_radar_chart.png / .pdf")
plt.show()
