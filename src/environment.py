"""
environment.py — Custom minimal Cliff Walking environment (4×12 GridWorld).

This is a faithful fallback implementation matching the Gymnasium CliffWalking-v0
specification, used when Gymnasium is unavailable or for unit-testing purposes.

Grid layout (row 0 = top):
  Row 3, Col 0  → Start (S)
  Row 3, Col 11 → Goal  (G)
  Row 3, Col 1..10 → Cliff (C)

Actions: 0=Up, 1=Right, 2=Down, 3=Left
Rewards: -1 per step; -100 on cliff fall (reset to S, episode continues); +0 on reaching Goal (episode ends)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


class CliffWalkingEnv:
    """Minimal 4×12 Cliff Walking GridWorld."""

    ROWS = 4
    COLS = 12
    N_STATES = ROWS * COLS  # 48
    N_ACTIONS = 4            # 0=Up, 1=Right, 2=Down, 3=Left

    START = (3, 0)   # bottom-left
    GOAL  = (3, 11)  # bottom-right
    CLIFF_COLS = set(range(1, 11))  # row 3, cols 1-10

    # Action deltas (row_delta, col_delta)
    _DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        self._state: int = self._encode(*self.START)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def _encode(self, row: int, col: int) -> int:
        return row * self.COLS + col

    def _decode(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.COLS)

    def _is_cliff(self, row: int, col: int) -> bool:
        return row == 3 and col in self.CLIFF_COLS

    def _is_goal(self, row: int, col: int) -> bool:
        return row == self.GOAL[0] and col == self.GOAL[1]

    # ------------------------------------------------------------------
    # Gymnasium-compatible interface
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[int, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._encode(*self.START)
        return self._state, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        row, col = self._decode(self._state)
        dr, dc = self._DELTAS[action]
        new_row = max(0, min(self.ROWS - 1, row + dr))
        new_col = max(0, min(self.COLS - 1, col + dc))

        if self._is_cliff(new_row, new_col):
            # Fall into cliff: penalty and reset to Start
            self._state = self._encode(*self.START)
            return self._state, -100.0, False, False, {}

        if self._is_goal(new_row, new_col):
            self._state = self._encode(new_row, new_col)
            return self._state, 0.0, True, False, {}

        self._state = self._encode(new_row, new_col)
        return self._state, -1.0, False, False, {}

    @property
    def observation_space_n(self) -> int:
        return self.N_STATES

    @property
    def action_space_n(self) -> int:
        return self.N_ACTIONS

    def render_grid(self) -> str:
        """ASCII render for debugging."""
        row_cur, col_cur = self._decode(self._state)
        lines = []
        for r in range(self.ROWS):
            row_chars = []
            for c in range(self.COLS):
                if r == row_cur and c == col_cur:
                    row_chars.append("X")
                elif (r, c) == self.START:
                    row_chars.append("S")
                elif (r, c) == self.GOAL:
                    row_chars.append("G")
                elif self._is_cliff(r, c):
                    row_chars.append("C")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))
        return "\n".join(lines)
