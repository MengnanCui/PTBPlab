"""Charge mixing for the SCC cycle (Anderson / Pulay-type)."""
from __future__ import annotations

import numpy as np

from .backend import ops


class AndersonMixer:
    def __init__(self, beta: float = 0.2, history: int = 8):
        self.beta = beta
        self.history = history
        self.xs: list[np.ndarray] = []
        self.fs: list[np.ndarray] = []

    def mix(self, q_in: np.ndarray, q_out: np.ndarray) -> np.ndarray:
        f = q_out - q_in
        self.xs.append(q_in.copy())
        self.fs.append(f.copy())
        if len(self.xs) > self.history:
            self.xs.pop(0)
            self.fs.pop(0)
        if len(self.xs) == 1:
            return q_in + self.beta * f
        df = np.array([self.fs[k + 1] - self.fs[k] for k in range(len(self.fs) - 1)])
        dx = np.array([self.xs[k + 1] - self.xs[k] for k in range(len(self.xs) - 1)])
        a = df @ df.T
        a += 1e-12 * np.trace(a) * np.eye(len(a))
        gam = ops.solve(a, df @ f)
        x_bar = q_in - gam @ dx
        f_bar = f - gam @ df
        return x_bar + self.beta * f_bar
