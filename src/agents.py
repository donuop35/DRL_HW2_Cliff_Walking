"""
agents.py — Tabular Q-Learning and SARSA agents sharing a unified skeleton.

Both algorithms use ε-greedy action selection.
The only difference is the update target:
  Q-learning (off-policy): target = r + γ * max_{a'} Q(s', a')
  SARSA      (on-policy):  target = r + γ * Q(s', a')  where a' ~ ε-greedy(s')

Usage:
    agent = TabularAgent(algo="qlearning", alpha=0.1, gamma=0.9, epsilon=0.1,
                         n_states=48, n_actions=4, seed=42)
    agent = TabularAgent(algo="sarsa", ...)
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class TabularAgent:
    """Unified tabular RL agent supporting Q-learning and SARSA."""

    SUPPORTED_ALGOS = {"qlearning", "sarsa"}

    def __init__(
        self,
        algo: str,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        n_states: int = 48,
        n_actions: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        if algo not in self.SUPPORTED_ALGOS:
            raise ValueError(f"algo must be one of {self.SUPPORTED_ALGOS}, got '{algo}'")
        self.algo = algo
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, state: int) -> int:
        """ε-greedy action selection."""
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------
    # Update rules
    # ------------------------------------------------------------------
    def update(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        a_next: Optional[int] = None,
        done: bool = False,
    ) -> None:
        """
        Update Q-table.

        For terminal states: target = r (no bootstrap).
        Q-learning: a_next is ignored; uses max_Q(s').
        SARSA:      a_next must be provided; uses Q(s', a_next).
        """
        if done:
            td_target = r
        elif self.algo == "qlearning":
            td_target = r + self.gamma * np.max(self.Q[s_next])
        else:  # sarsa
            if a_next is None:
                raise ValueError("SARSA requires a_next to be provided.")
            td_target = r + self.gamma * self.Q[s_next, a_next]

        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])

    # ------------------------------------------------------------------
    # Policy extraction
    # ------------------------------------------------------------------
    def greedy_policy(self) -> np.ndarray:
        """Return greedy action (argmax Q) for every state."""
        return np.argmax(self.Q, axis=1)

    def reset_qtable(self) -> None:
        """Reset Q-table to zeros (used between different seed runs)."""
        self.Q[:] = 0.0
        self._rng = np.random.default_rng()  # reseed handled externally

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
