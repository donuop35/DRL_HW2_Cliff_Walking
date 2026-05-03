"""
evaluate.py — Compute summary statistics and convergence metrics.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List
from src.utils import moving_average, compute_aulc, find_convergence_episode, summary_stats


def evaluate_experiment(
    matrix: np.ndarray,
    algo: str,
    convergence_threshold: float = -20.0,
    convergence_window: int = 50,
    convergence_sustained: int = 50,
    aulc_episodes: int = 500,
    final_window: int = 100,
) -> Dict:
    """
    Given reward matrix (n_seeds × n_episodes), compute all comparison metrics.

    Returns a dict with:
      - algo: algorithm name
      - mean_curve: mean rewards over seeds
      - std_curve: std over seeds
      - ma_mean: moving average of mean_curve
      - final_mean: mean of last `final_window` episodes (averaged over seeds first)
      - final_std: std of per-seed final-window means
      - convergence_episodes: list of convergence episode per seed
      - convergence_mean: mean convergence episode
      - convergence_std: std
      - aulc: Area Under Learning Curve (first aulc_episodes), per seed, then mean
      - cliff_falls: not directly computable from rewards alone (placeholder)
    """
    n_seeds, n_episodes = matrix.shape
    stats = summary_stats(matrix)

    # Per-seed final window means
    final_means_per_seed = matrix[:, -final_window:].mean(axis=1)

    # Per-seed convergence episode
    conv_eps = [
        find_convergence_episode(
            matrix[i],
            threshold=convergence_threshold,
            window=convergence_window,
            sustained=convergence_sustained,
        )
        for i in range(n_seeds)
    ]

    # AULC per seed
    aulc_vals = [compute_aulc(matrix[i], aulc_episodes) for i in range(n_seeds)]

    return {
        "algo": algo,
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
        "mean_curve": stats["mean"],
        "std_curve": stats["std"],
        "q25_curve": stats["q25"],
        "q75_curve": stats["q75"],
        "ma_mean": moving_average(stats["mean"].tolist(), window=50),
        "final_mean": float(np.mean(final_means_per_seed)),
        "final_std": float(np.std(final_means_per_seed)),
        "convergence_episodes": conv_eps,
        "convergence_mean": float(np.mean(conv_eps)),
        "convergence_std": float(np.std(conv_eps)),
        "aulc_values": aulc_vals,
        "aulc_mean": float(np.mean(aulc_vals)),
        "aulc_std": float(np.std(aulc_vals)),
    }


def compare_algos(results: Dict[str, Dict]) -> Dict:
    """Print a comparison table and return comparison dict."""
    print("\n" + "=" * 65)
    print(f"{'Metric':<35} {'Q-Learning':>14} {'SARSA':>14}")
    print("=" * 65)

    ql = results.get("qlearning", {})
    sa = results.get("sarsa", {})

    rows = [
        ("Final Reward (mean ± std)",
         f"{ql.get('final_mean', float('nan')):.2f} ± {ql.get('final_std', float('nan')):.2f}",
         f"{sa.get('final_mean', float('nan')):.2f} ± {sa.get('final_std', float('nan')):.2f}"),
        ("Convergence Episode (mean ± std)",
         f"{ql.get('convergence_mean', float('nan')):.1f} ± {ql.get('convergence_std', float('nan')):.1f}",
         f"{sa.get('convergence_mean', float('nan')):.1f} ± {sa.get('convergence_std', float('nan')):.1f}"),
        ("AULC-500 (mean ± std)",
         f"{ql.get('aulc_mean', float('nan')):.2f} ± {ql.get('aulc_std', float('nan')):.2f}",
         f"{sa.get('aulc_mean', float('nan')):.2f} ± {sa.get('aulc_std', float('nan')):.2f}"),
    ]

    for label, ql_val, sa_val in rows:
        print(f"  {label:<33} {ql_val:>14} {sa_val:>14}")

    print("=" * 65)
    return {"qlearning": ql, "sarsa": sa}
