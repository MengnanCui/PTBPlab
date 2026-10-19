"""Electronic occupations: Fermi-Dirac smearing (or T = 0 aufbau)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

SPIN = 2.0


def fill(evals: list[np.ndarray], weights: np.ndarray, nelec: float, kt: float):
    """Occupations (incl. spin factor 2), Fermi level and T*S (Hartree)."""
    if kt < 1e-12:
        return _fill_zero(evals, weights, nelec)
    allv = np.concatenate(evals)

    def count(mu):
        return sum(w * SPIN * expit(-(e - mu) / kt).sum() for e, w in zip(evals, weights)) - nelec

    lo, hi = allv.min() - 50 * kt - 1.0, allv.max() + 50 * kt + 1.0
    mu = brentq(count, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=500)
    occ, ts = [], 0.0
    for e, w in zip(evals, weights):
        f = expit(-(e - mu) / kt)
        occ.append(SPIN * f)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -(np.where(f > 0, f * np.log(f), 0.0) + np.where(f < 1, (1 - f) * np.log(1 - f), 0.0))
        ts += w * SPIN * kt * s.sum()
    return occ, mu, ts


def _fill_zero(evals, weights, nelec):
    allv = np.concatenate(evals)
    allw = np.concatenate([np.full(len(e), w * SPIN) for e, w in zip(evals, weights)])
    order = np.argsort(allv, kind="stable")
    cum = np.cumsum(allw[order])
    ihomo = int(np.searchsorted(cum, nelec - 1e-10))
    mu = allv[order][min(ihomo, len(allv) - 1)]
    # fill strictly below mu, share remaining electrons among degenerate levels at mu
    tol = 1e-8
    occ = []
    below = sum((w * SPIN * (e < mu - tol)).sum() for e, w in zip(evals, weights))
    at = sum((w * SPIN * (np.abs(e - mu) <= tol)).sum() for e, w in zip(evals, weights))
    frac = (nelec - below) / at if at > 0 else 0.0
    for e, w in zip(evals, weights):
        f = np.where(e < mu - tol, SPIN, np.where(np.abs(e - mu) <= tol, SPIN * frac, 0.0))
        occ.append(f)
    return occ, mu, 0.0
