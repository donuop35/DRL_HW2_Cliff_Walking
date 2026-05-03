"""
run_experiments.py — Master experiment orchestrator.

Usage:
    python scripts/run_experiments.py

Runs:
  1. Baseline experiment: Q-learning vs SARSA, 20 seeds × 1000 episodes
  2. Epsilon sensitivity: ε ∈ {0.01, 0.1, 0.2}, 10 seeds × 500 episodes
  3. Saves all raw results to results/raw/
  4. Generates all figures to results/figures/
  5. Prints comparison table
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np

# Make sure src/ is importable regardless of working directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.train import run_experiment, get_final_policy
from src.evaluate import evaluate_experiment, compare_algos
from src.plot import (
    plot_learning_curves,
    plot_convergence,
    plot_final_reward_bar,
    plot_policy_grid,
    plot_epsilon_sensitivity,
    plot_stability,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(ROOT, "results", "raw")
FIG_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
ALPHA   = 0.1
GAMMA   = 0.9
EPSILON = 0.1
N_EPISODES_MAIN = 1000
N_EPISODES_EPS  = 500
SEEDS_MAIN = list(range(20))   # 20 seeds for main experiment
SEEDS_EPS  = list(range(10))   # 10 seeds for epsilon sensitivity
EPSILONS   = [0.01, 0.1, 0.2]
ALGOS = ["qlearning", "sarsa"]


def save_matrix(matrix: np.ndarray, algo: str, tag: str = "baseline") -> None:
    path = os.path.join(RAW_DIR, f"rewards_{algo}_{tag}.npy")
    np.save(path, matrix)
    # Also save as CSV for human inspection (seeds as rows, episodes as cols)
    csv_path = os.path.join(RAW_DIR, f"rewards_{algo}_{tag}.csv")
    np.savetxt(csv_path, matrix, delimiter=",", fmt="%.4f")
    print(f"  [save] {path}")


def load_matrix(algo: str, tag: str = "baseline") -> np.ndarray | None:
    path = os.path.join(RAW_DIR, f"rewards_{algo}_{tag}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Baseline Experiment
# ══════════════════════════════════════════════════════════════════════════════
def run_baseline() -> dict:
    print("\n" + "═" * 60)
    print("Phase 1: Baseline Experiment")
    print(f"  alpha={ALPHA}, gamma={GAMMA}, epsilon={EPSILON}")
    print(f"  {N_EPISODES_MAIN} episodes × {len(SEEDS_MAIN)} seeds")
    print("═" * 60)

    matrices = {}
    for algo in ALGOS:
        cached = load_matrix(algo, "baseline")
        if cached is not None:
            print(f"  [{algo}] loaded from cache.")
            matrices[algo] = cached
        else:
            print(f"\n  Running {algo} ...")
            m = run_experiment(
                algo=algo,
                alpha=ALPHA,
                gamma=GAMMA,
                epsilon=EPSILON,
                n_episodes=N_EPISODES_MAIN,
                seeds=SEEDS_MAIN,
            )
            matrices[algo] = m
            save_matrix(m, algo, "baseline")

    return matrices


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Epsilon Sensitivity
# ══════════════════════════════════════════════════════════════════════════════
def run_epsilon_sensitivity() -> dict:
    print("\n" + "═" * 60)
    print("Phase 2: Epsilon Sensitivity Analysis")
    print(f"  epsilons={EPSILONS}, {N_EPISODES_EPS} episodes × {len(SEEDS_EPS)} seeds")
    print("═" * 60)

    eps_matrices = {}  # {eps: {algo: matrix}}
    for eps in EPSILONS:
        eps_matrices[eps] = {}
        for algo in ALGOS:
            tag = f"eps{str(eps).replace('.','p')}"
            cached = load_matrix(algo, tag)
            if cached is not None:
                print(f"  [{algo}, ε={eps}] loaded from cache.")
                eps_matrices[eps][algo] = cached
            else:
                print(f"\n  Running {algo}, ε={eps} ...")
                m = run_experiment(
                    algo=algo,
                    alpha=ALPHA,
                    gamma=GAMMA,
                    epsilon=eps,
                    n_episodes=N_EPISODES_EPS,
                    seeds=SEEDS_EPS,
                )
                eps_matrices[eps][algo] = m
                save_matrix(m, algo, tag)

    return eps_matrices


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Evaluate + Plot
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_and_plot(matrices: dict, eps_matrices: dict) -> dict:
    print("\n" + "═" * 60)
    print("Phase 3: Evaluation & Visualization")
    print("═" * 60)

    # Baseline evaluation
    eval_results = {}
    for algo in ALGOS:
        eval_results[algo] = evaluate_experiment(
            matrices[algo], algo,
            convergence_threshold=-20.0,
            convergence_window=50,
            convergence_sustained=50,
            aulc_episodes=500,
            final_window=100,
        )

    compare_algos(eval_results)

    # Save summary JSON
    summary = {}
    for algo, res in eval_results.items():
        summary[algo] = {
            "final_mean": res["final_mean"],
            "final_std": res["final_std"],
            "convergence_mean": res["convergence_mean"],
            "convergence_std": res["convergence_std"],
            "aulc_mean": res["aulc_mean"],
            "aulc_std": res["aulc_std"],
            "convergence_episodes": res["convergence_episodes"],
            "aulc_values": res["aulc_values"],
        }
    json_path = os.path.join(RAW_DIR, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  [save] {json_path}")

    # Epsilon sensitivity evaluation
    eps_eval = {}  # {eps: {algo: eval_dict}}
    for eps in EPSILONS:
        eps_eval[eps] = {}
        for algo in ALGOS:
            eps_eval[eps][algo] = evaluate_experiment(
                eps_matrices[eps][algo], algo,
                final_window=100,
                aulc_episodes=N_EPISODES_EPS,
            )

    # Plots
    print("\n  Generating figures...")
    plot_learning_curves(eval_results, FIG_DIR)
    plot_convergence(eval_results, FIG_DIR)
    plot_final_reward_bar(eval_results, FIG_DIR)
    plot_stability(eval_results, FIG_DIR)
    plot_epsilon_sensitivity(eps_eval, FIG_DIR)

    # Policy grids
    print("\n  Computing final policies...")
    for algo in ALGOS:
        policy = get_final_policy(
            algo=algo, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
            n_episodes=N_EPISODES_MAIN, seed=0,
        )
        np.save(os.path.join(RAW_DIR, f"policy_{algo}.npy"), policy)
        plot_policy_grid(policy, algo, FIG_DIR)

    return eval_results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    matrices = run_baseline()
    eps_matrices = run_epsilon_sensitivity()
    eval_results = evaluate_and_plot(matrices, eps_matrices)

    print("\n" + "═" * 60)
    print("All experiments complete!")
    print(f"  Raw data : {RAW_DIR}")
    print(f"  Figures  : {FIG_DIR}")
    print("═" * 60)
