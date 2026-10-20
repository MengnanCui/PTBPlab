"""Charge mixing for the SCC cycle.

* :class:`BroydenMixer` - modified Broyden mixing exactly as DFTB+'s default
  (``mixer/broydenmixer.F90``; Johnson, PRB 38, 12807 (1988)).
* :class:`AndersonMixer` - a Pulay/DIIS-type mixer (pydftb's original choice;
  note it is *not* DFTB+'s "Anderson" mixer).

``mix(q_in, q_out)`` returns the next input charges.  The mixed quantity is
the population of every SCC site (atom, or shell when shell-resolved).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .backend import ops


class BroydenMixer:
    """DFTB+ modified Broyden mixer (defaults = DFTB+ defaults)."""

    def __init__(self, alpha: float = 0.2, omega0: float = 0.01, min_weight: float = 1.0,
                 max_weight: float = 1.0e5, weight_factor: float = 1.0e-2):
        self.alpha = alpha
        self.omega0 = omega0
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weight_factor = weight_factor
        self.n = 0
        self.q_last = self.f_last = None
        self.w: list[float] = []          # weights of the stored iterations
        self.df: list[np.ndarray] = []    # normalised dF vectors
        self.u: list[np.ndarray] = []
        self.a: list[list[float]] = []    # overlaps <dF_i|dF_j>

    def mix(self, q_in: np.ndarray, q_out: np.ndarray) -> np.ndarray:
        f = q_out - q_in
        self.n += 1
        if self.n == 1:
            self.q_last, self.f_last = q_in.copy(), f.copy()
            return q_in + self.alpha * f
        norm_f = np.sqrt(f @ f)
        w = self.weight_factor / norm_f if norm_f > self.weight_factor / self.max_weight \
            else self.max_weight
        w = max(w, self.min_weight)
        dfu = f - self.f_last
        inv_norm = 1.0 / max(np.sqrt(dfu @ dfu), np.finfo(float).eps)
        dfu = inv_norm * dfu
        m = len(self.df)                       # previous vectors: nn - 2 in DFTB+
        row = [self.df[i] @ dfu for i in range(m)]
        for i in range(m):
            self.a[i].append(row[i])
        self.a.append(row + [1.0])
        self.w.append(w)
        ww = np.array(self.w)
        aa = np.array(self.a)
        cc = np.array([ww[i] * (self.df[i] @ f) for i in range(m)] + [w * (dfu @ f)])
        beta = ww[:, None] * ww[None, :] * aa
        beta[np.diag_indices_from(beta)] += self.omega0 ** 2
        cc = sla.solve(beta.T, cc)
        self.df.append(dfu)
        u = self.alpha * dfu + inv_norm * (q_in - self.q_last)
        q_new = q_in + self.alpha * f
        for i in range(m):
            q_new = q_new - ww[i] * cc[i] * self.u[i]
        q_new = q_new - w * cc[m] * u
        self.u.append(u)
        self.q_last, self.f_last = q_in.copy(), f.copy()
        return q_new


class AndersonMixer:
    """Pulay/DIIS-type mixing over the last ``history`` iterations."""

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


MIXERS = {"broyden": BroydenMixer, "pulay": AndersonMixer, "anderson": AndersonMixer}


def make_mixer(kind: str = "broyden", mixing: float | None = None):
    cls = MIXERS[kind.lower()]
    if mixing is None:
        return cls()
    return cls(mixing)
