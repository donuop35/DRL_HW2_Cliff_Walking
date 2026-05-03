"""
utils.py — Seed management and helper functions.
"""

from __future__ import annotations

import numpy as np
from typing import List


def set_global_seed(seed: int) -> np.random.Generator:
    """Create a reproducible numpy RNG from a seed."""
    return np.random.default_rng(seed)


def moving_average(data: List[float], window: int = 50) -> np.ndarray:
    """Compute centered moving average with edge padding."""
    arr = np.array(data, dtype=float)
    kernel = np.ones(window) / window
    # Use 'same' convolution with edge-value padding
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def compute_aulc(rewards: np.ndarray, n_episodes: int = 500) -> float:
    """Area Under Learning Curve for the first n_episodes (trapezoidal)."""
    y = rewards[:n_episodes]
    return float(np.trapz(y)) / len(y)


def find_convergence_episode(
    rewards: np.ndarray,
    threshold: float = -20.0,
    window: int = 50,
    sustained: int = 50,
) -> int:
    """
    Return the first episode e such that MA[e:e+sustained] >= threshold throughout.
    Returns len(rewards) if convergence never achieved.
    """
    ma = moving_average(list(rewards), window)
    n = len(ma)
    for e in range(n - sustained + 1):
        if np.all(ma[e : e + sustained] >= threshold):
            return e
    return n


def summary_stats(matrix: np.ndarray) -> dict:
    """
    matrix shape: (n_seeds, n_episodes)
    Returns mean, std, median, q25, q75 over seeds per episode.
    """
    return {
        "mean": matrix.mean(axis=0),
        "std": matrix.std(axis=0),
        "median": np.median(matrix, axis=0),
        "q25": np.percentile(matrix, 25, axis=0),
        "q75": np.percentile(matrix, 75, axis=0),
    }
