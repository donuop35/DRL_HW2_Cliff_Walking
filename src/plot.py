"""
plot.py — All visualization functions for the Cliff Walking comparison.

Outputs high-resolution PNG files to results/figures/.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "qlearning": "#E05C5C",   # coral red
    "sarsa":     "#5B8DD9",   # steel blue
    "fill_ql":   "#F5A0A0",
    "fill_sa":   "#A0BCEE",
    "cliff":     "#2B2B2B",
    "start":     "#27AE60",
    "goal":      "#F39C12",
    "grid_bg":   "#F7F9FC",
}

LABELS = {"qlearning": "Q-Learning (off-policy)", "sarsa": "SARSA (on-policy)"}
ARROW_CHARS = {0: "↑", 1: "→", 2: "↓", 3: "←"}
DPI = 150
FIGSIZE_WIDE = (12, 5)
FIGSIZE_SQ   = (8, 6)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ── 1. Learning Curves ────────────────────────────────────────────────────────
def plot_learning_curves(
    results: Dict[str, Dict],
    out_dir: str,
    window: int = 50,
) -> None:
    """Raw + smoothed reward curves for Q-learning and SARSA."""
    _ensure_dir(out_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle("Learning Curves: Q-Learning vs SARSA (Cliff Walking)", fontsize=14, fontweight="bold")

    for ax, (title, smoothed) in zip(axes, [("Raw Episode Reward", False), ("Smoothed (MA-50)", True)]):
        for algo, res in results.items():
            color = COLORS[algo]
            fill_color = COLORS.get(f"fill_{algo[:2]}", color)
            n_ep = res["n_episodes"]
            x = np.arange(n_ep)

            if smoothed:
                y = res["ma_mean"]
                ax.plot(x, y, color=color, lw=2, label=LABELS[algo])
            else:
                y = res["mean_curve"]
                ax.plot(x, y, color=color, lw=1.2, alpha=0.85, label=LABELS[algo])

            y_lo = res["q25_curve"]
            y_hi = res["q75_curve"]
            ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.18)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel("Total Reward", fontsize=10)
        ax.axhline(-13, color="gray", ls="--", lw=0.8, alpha=0.6, label="Optimal (−13)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, n_ep)

    plt.tight_layout()
    path = os.path.join(out_dir, "01_learning_curves.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {path}")


# ── 2. Convergence Comparison ─────────────────────────────────────────────────
def plot_convergence(
    results: Dict[str, Dict],
    out_dir: str,
) -> None:
    """Box + jitter plot of convergence episodes across seeds."""
    _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    algos = list(results.keys())
    data = [results[a]["convergence_episodes"] for a in algos]
    colors = [COLORS[a] for a in algos]

    bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                    medianprops=dict(color="white", lw=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Jitter overlay
    for i, (d, color) in enumerate(zip(data, colors), start=1):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(d))
        ax.scatter([i + j for j in jitter], d, color=color, alpha=0.5, s=20, zorder=5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([LABELS[a] for a in algos], fontsize=10)
    ax.set_ylabel("Convergence Episode", fontsize=10)
    ax.set_title("Convergence Speed Comparison (threshold: MA-50 ≥ −20 for 50 eps)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    for i, a in enumerate(algos, start=1):
        m = results[a]["convergence_mean"]
        ax.annotate(f"μ={m:.0f}", xy=(i, m), xytext=(i + 0.25, m),
                    fontsize=9, color=COLORS[a], fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "02_convergence.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {path}")


# ── 3. Final Reward Bar Chart ─────────────────────────────────────────────────
def plot_final_reward_bar(
    results: Dict[str, Dict],
    out_dir: str,
) -> None:
    """Bar chart of final-100-episode reward mean ± std per algorithm."""
    _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(7, 5))
    algos = list(results.keys())
    means = [results[a]["final_mean"] for a in algos]
    stds  = [results[a]["final_std"]  for a in algos]
    colors = [COLORS[a] for a in algos]

    bars = ax.bar([LABELS[a] for a in algos], means, yerr=stds,
                  color=colors, alpha=0.8, capsize=8, error_kw={"lw": 2})

    ax.axhline(-13, color="gray", ls="--", lw=1, label="Optimal (−13)")
    ax.set_ylabel("Mean Final-100 Reward", fontsize=10)
    ax.set_title("Final Performance (last 100 episodes × 20 seeds)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean - std - 1.5,
                f"{mean:.2f}", ha="center", va="top", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "03_final_reward.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {path}")


# ── 4. Policy Grid Visualization ──────────────────────────────────────────────
def plot_policy_grid(
    policy: np.ndarray,
    algo: str,
    out_dir: str,
    highlight_path: bool = True,
) -> None:
    """
    Draw the greedy policy as arrows on the 4×12 grid.
    Highlights: Start (green), Goal (gold), Cliff (dark).
    Optionally traces the greedy path from Start.
    """
    _ensure_dir(out_dir)

    ROWS, COLS = 4, 12
    CLIFF_CELLS = {(3, c) for c in range(1, 11)}
    START = (3, 0)
    GOAL  = (3, 11)
    DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.5, ROWS - 0.5)
    ax.invert_yaxis()

    # Draw grid cells
    for r in range(ROWS):
        for c in range(COLS):
            cell = (r, c)
            if cell in CLIFF_CELLS:
                fc = COLORS["cliff"]
            elif cell == START:
                fc = COLORS["start"]
            elif cell == GOAL:
                fc = COLORS["goal"]
            else:
                fc = COLORS["grid_bg"]

            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                  facecolor=fc, edgecolor="#CCCCCC", lw=0.8)
            ax.add_patch(rect)

            # Labels for special cells
            if cell == START:
                ax.text(c, r, "S", ha="center", va="center",
                        color="white", fontsize=13, fontweight="bold")
            elif cell == GOAL:
                ax.text(c, r, "G", ha="center", va="center",
                        color="white", fontsize=13, fontweight="bold")
            elif cell in CLIFF_CELLS:
                ax.text(c, r, "✕", ha="center", va="center",
                        color="#FF4444", fontsize=9)
            else:
                state = r * COLS + c
                act = policy[state]
                ax.text(c, r, ARROW_CHARS[act], ha="center", va="center",
                        color=COLORS[algo], fontsize=14)

    # Trace greedy path from Start
    if highlight_path:
        path_cells = []
        r, c = START
        seen = set()
        for _ in range(ROWS * COLS):
            if (r, c) == GOAL or (r, c) in seen:
                break
            seen.add((r, c))
            path_cells.append((r, c))
            state = r * COLS + c
            act = policy[state]
            dr, dc = DELTAS[act]
            r = max(0, min(ROWS - 1, r + dr))
            c = max(0, min(COLS - 1, c + dc))

        # Draw path with colored trail
        xs = [c for _, c in path_cells]
        ys = [r for r, _ in path_cells]
        ax.plot([x for x in xs], [y for y in ys],
                color=COLORS[algo], lw=2.5, alpha=0.6, zorder=5,
                marker="o", markersize=4)

    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xticklabels(range(COLS), fontsize=8)
    ax.set_yticklabels(range(ROWS), fontsize=8)
    ax.set_title(f"Greedy Policy: {LABELS.get(algo, algo)}", fontsize=12, fontweight="bold")

    legend_items = [
        mpatches.Patch(facecolor=COLORS["start"], label="Start (S)"),
        mpatches.Patch(facecolor=COLORS["goal"],  label="Goal (G)"),
        mpatches.Patch(facecolor=COLORS["cliff"], label="Cliff"),
        mpatches.Patch(facecolor=COLORS[algo],    label="Policy arrow / path", alpha=0.6),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, f"04_policy_{algo}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {path}")


# ── 5. Epsilon Sensitivity ────────────────────────────────────────────────────
def plot_epsilon_sensitivity(
    eps_results: Dict[float, Dict[str, Dict]],
    out_dir: str,
) -> None:
    """
    Grouped bar chart: final reward per (epsilon, algo).
    eps_results: {eps_val: {"qlearning": eval_dict, "sarsa": eval_dict}}
    """
    _ensure_dir(out_dir)

    epsilons = sorted(eps_results.keys())
    algos = ["qlearning", "sarsa"]
    x = np.arange(len(epsilons))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, algo in enumerate(algos):
        means = [eps_results[e][algo]["final_mean"] for e in epsilons]
        stds  = [eps_results[e][algo]["final_std"]  for e in epsilons]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=LABELS[algo],
               color=COLORS[algo], alpha=0.8, capsize=6, error_kw={"lw": 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels([f"ε = {e}" for e in epsilons], fontsize=10)
    ax.set_ylabel("Final-100 Reward (mean ± std)", fontsize=10)
    ax.set_title("Epsilon Sensitivity: Final Performance", fontsize=12)
    ax.axhline(-13, color="gray", ls="--", lw=0.8, label="Optimal (−13)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "05_epsilon_sensitivity.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {path}")


# ── 6. Rolling Std (Stability) ────────────────────────────────────────────────
def plot_stability(
    results: Dict[str, Dict],
    out_dir: str,
    window: int = 50,
) -> None:
    """Rolling standard deviation across seeds to show stability over time."""
    _ensure_dir(out_dir)

    from src.utils import moving_average

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for algo, res in results.items():
        color = COLORS[algo]
        n_ep = res["n_episodes"]
        x = np.arange(n_ep)
        rolling_std = moving_average(res["std_curve"].tolist(), window=window)
        ax.plot(x, rolling_std, color=color, lw=2, label=LABELS[algo])

    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel(f"Rolling Std of Reward (window={window})", fontsize=10)
    ax.set_title("Reward Stability Over Training (lower = more stable)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, list(results.values())[0]["n_episodes"])

    plt.tight_layout()
    path = os.path.join(out_dir, "06_stability.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {path}")
