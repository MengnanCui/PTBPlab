"""Electronic occupations: Fermi-Dirac, Gaussian and Methfessel-Paxton smearing.

Reproduces DFTB+ (``dftb/etemp.F90`` and ``main.F90:getFillingsAndBandEnergies``):

* spin-unpolarised: N/2 electrons are distributed over states holding at most
  one electron, then occupations, band energy and TS are doubled;
* T below ``MIN_TEMP`` (1e-8 Ha) is raised to it, so "T = 0" is Fermi filling
  at a tiny temperature, with the Fermi level in the middle of the gap;
* the Fermi level is first tried at mid-gap (integer electron counts), then
  found by DFTB+'s bisection, which is deterministic even though N(mu) is not
  monotonic for Methfessel-Paxton;
* one deliberate difference: the mid-gap search allows for rounding in the summed
  k-point weights.  DFTB+'s ``middleGap`` adds the weights until they reach N; with
  weights such as 1/13 the sum can land one rounding step below N, the gap is missed,
  and the bisection puts the Fermi level of an insulator next to the conduction band
  (the occupations are the same either way);
* Gaussian = Methfessel-Paxton order 0.

Entropy convention for Methfessel-Paxton order N >= 1: DFTB+ uses
TS = kT sum w 1/2 sum_{l=0..N} A_l H_2l(x) e^{-x^2} ("dftb+", default, for
parity).  The variationally consistent choice keeps only the l = N term
("variational"); only then is the Mermin energy the exact antiderivative of
the forces.  Both agree for Fermi and Gaussian smearing.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial, pi, sqrt

import numpy as np
from scipy.special import erfc, expit

from .constants import ELEC_TOL_MAX, KB_HARTREE, MIN_TEMP

ELEC_TOL = 1.0e-15
CUM_WEIGHT_TOL = 1.0e-9               # rounding allowance of the summed k-point weights
EPS2 = 2.0 * np.finfo(float).eps
EPS_SINGLE = float(np.finfo(np.float32).eps)
M_EXP_ARG = 36.043653389117154        # DFTB+ exp() trap: f = 0 beyond this


@dataclass(frozen=True)
class Smearing:
    """Occupation function. ``temperature`` in K (DFTB+ "Temperature [K]")."""
    kind: str = "fermi"                 # "fermi" | "gaussian" | "mp"
    temperature: float = 0.0
    order: int = 1                      # Methfessel-Paxton order (>= 1)
    ts_convention: str = "dftb+"        # "dftb+" | "variational"
    # "dftb+": with an integer electron count the Fermi level is put mid-gap if the count
    # there is within ELEC_TOL_MAX (1e-7) of it, as DFTB+ does; "exact": always the
    # bisection to the exact count, so that F is the fixed-N Mermin free energy (whose
    # gradients pydftb.autodiff gives in mode "hf")
    fermi_level: str = "dftb+"          # "dftb+" | "exact"

    def __post_init__(self):
        if self.kind not in ("fermi", "gaussian", "mp"):
            raise ValueError(f"unknown smearing {self.kind!r}")
        if self.kind == "mp" and self.order < 1:
            raise ValueError("Methfessel-Paxton order must be >= 1 (order 0 is 'gaussian')")
        if self.ts_convention not in ("dftb+", "variational"):
            raise ValueError(f"unknown ts_convention {self.ts_convention!r}")
        if self.fermi_level not in ("dftb+", "exact"):
            raise ValueError(f"unknown fermi_level {self.fermi_level!r}")

    @property
    def variational(self) -> bool:
        """Whether F = E - TS is the fixed-N Mermin free energy (stationary in the density):
        Methfessel-Paxton needs ``ts_convention="variational"``."""
        return self.kind != "mp" or self.ts_convention == "variational"

    @property
    def kt(self) -> float:
        """kT in Hartree, raised to MIN_TEMP like DFTB+."""
        return max(self.temperature * KB_HARTREE, MIN_TEMP)

    @property
    def mp_order(self) -> int | None:
        return {"fermi": None, "gaussian": 0, "mp": self.order}[self.kind]

    @classmethod
    def from_spec(cls, spec) -> "Smearing":
        """Smearing | ("fermi", T) | ("gaussian", T) | ("mp", T, order) | T (Fermi)."""
        if isinstance(spec, Smearing):
            return spec
        if spec is None:
            return cls()
        if np.isscalar(spec):
            return cls("fermi", float(spec))
        kind = spec[0].lower()
        kind = {"methfesselpaxton": "mp", "methfessel-paxton": "mp"}.get(kind, kind)
        if kind == "mp":
            return cls("mp", float(spec[1]), int(spec[2]) if len(spec) > 2 else 1)
        return cls(kind, float(spec[1]))


def _mp_weights(order: int) -> np.ndarray:
    """A_n = (-1)^n / (n! 4^n sqrt(pi)), n = 0..order."""
    return np.array([(-1) ** n / (factorial(n) * 4 ** n * sqrt(pi)) for n in range(order + 1)])


def _hermite(nmax: int, x: np.ndarray) -> list[np.ndarray]:
    """Physicists' Hermite polynomials H_0..H_nmax at x."""
    h = [np.ones_like(x)]
    if nmax > 0:
        h.append(2.0 * x)
    for i in range(2, nmax + 1):
        h.append(2.0 * (x * h[i - 1] - (i - 1) * h[i - 2]))
    return h


def _occ_mp(x: np.ndarray, order: int) -> np.ndarray:
    a = _mp_weights(order)
    h = _hermite(2 * order, x)
    g = np.exp(-x * x)
    occ = 0.5 * erfc(x)
    for n in range(1, order + 1):
        occ = occ + a[n] * h[2 * n - 1] * g
    return occ


def _electron_count(mu, evals, kw, sm: Smearing, kt) -> float:
    x = (evals - mu) / kt
    if sm.mp_order is None:
        f = np.where(x <= M_EXP_ARG, expit(-x), 0.0)
    else:
        f = _occ_mp(x, sm.mp_order)
    return float((kw[:, None] * f).sum())


def _middle_gap_levels(evals, kw, nfill) -> tuple[int, int]:
    """Flat indices (into ``evals.ravel()``) of the last filled and the next level."""
    flat = evals.ravel()
    wts = np.repeat(kw, evals.shape[1])
    order = np.argsort(flat, kind="stable")
    cum = np.cumsum(wts[order])
    # first index with cum >= nfill, allowing for rounding in the sum (DFTB+ has no allowance)
    i_last = int(np.searchsorted(cum, nfill - CUM_WEIGHT_TOL, side="left"))
    i_next = min(i_last + 1, len(flat) - 1)
    return int(order[i_last]), int(order[i_next])


def _middle_gap(evals, kw, nfill) -> float:
    """DFTB+ middleGap: half-way between the last filled and the next level."""
    flat = evals.ravel()
    a, b = _middle_gap_levels(evals, kw, nfill)
    return 0.5 * (flat[a] + flat[b])


def _electron_fill(mu, evals, kw, sm: Smearing, kt):
    """Occupations (max 1), TS and band energy per channel, as DFTB+ electronFill."""
    x = (evals - mu) / kt
    if sm.mp_order is None:
        f = np.where(x <= M_EXP_ARG, expit(-x), 0.0)
        # DFTB+ stops at the first (sorted) level with f <= 1e-15 per k-point
        below = f <= ELEC_TOL
        cut = np.cumsum(below, axis=1) > 0
        f = np.where(cut, 0.0, f)
        mid = (f > np.finfo(float).eps) & (f < 1.0 - np.finfo(float).eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(mid, -(f * np.log(f) + (1.0 - f) * np.log(1.0 - f)), 0.0)
        ts = kt * float((kw[:, None] * s).sum())
    else:
        n = sm.mp_order
        f = _occ_mp(x, n)
        a = _mp_weights(n)
        h = _hermite(2 * n, x)
        g = np.exp(-x * x)
        terms = range(n + 1) if sm.ts_convention == "dftb+" else [n]
        s = sum(0.5 * a[m] * h[2 * m] * g for m in terms)
        ts = kt * float((kw[:, None] * s).sum())
    eband = float((kw[:, None] * f * evals).sum())
    return f, ts, eband


def fermi_rule(evals, weights, nelec: float, smearing=None, spin_factor: float = 2.0):
    """How :func:`fill` sets the Fermi level for these levels: ``("count",)`` (bisection to
    the electron count), ``("midgap", a, b)`` (DFTB+'s mid-gap rule: mu is the mean of the
    levels with flat indices a and b of ``evals``) or ``("fixed",)`` (no electrons, or all
    levels full: mu outside the spectrum).  :mod:`pydftb.autodiff` differentiates mu by it."""
    sm = Smearing.from_spec(smearing)
    ev = np.asarray(evals, float)
    if ev.ndim == 1:
        ev = ev[None]
    kw = np.asarray(weights, float)
    nfill = nelec / spin_factor
    if nfill < np.finfo(float).eps or ev.shape[1] * round(kw.sum()) <= nfill:
        return ("fixed",)
    if sm.fermi_level == "dftb+" and abs(nfill - round(nfill)) <= ELEC_TOL:
        mu = _middle_gap(ev, kw, nfill)
        if abs(nfill - _electron_count(mu, ev, kw, sm, sm.kt)) <= ELEC_TOL_MAX:
            return ("midgap",) + _middle_gap_levels(ev, kw, nfill)
    return ("count",)


def fill(evals, weights, nelec: float, smearing=None, spin_factor: float = 2.0):
    """Occupations (incl. spin factor), Fermi level and T*S (Hartree).

    ``evals``: list/array (nk, nlevels) of ascending eigenvalues per k-point;
    ``weights``: k-point weights summing to 1; ``smearing``: :class:`Smearing`
    or spec accepted by :meth:`Smearing.from_spec` (a number = Fermi at T K).
    """
    sm = Smearing.from_spec(smearing)
    kt = sm.kt
    ev = np.asarray(evals, float)
    if ev.ndim == 1:
        ev = ev[None]
    kw = np.asarray(weights, float)
    nfill = nelec / spin_factor
    nk, nlev = ev.shape

    if nfill < np.finfo(float).eps:
        mu = ev.min() - 1000.0 * (kt + EPS_SINGLE)
        return [np.zeros(nlev) for _ in range(nk)], mu, 0.0
    if nlev * round(kw.sum()) <= nfill:       # all levels full; kw sums to 1 per spin channel
        mu = ev.max() + 1000.0 * (kt + EPS_SINGLE)
        f, ts, _ = _electron_fill(mu, ev, kw, sm, kt)
        return [spin_factor * fk for fk in f], mu, spin_factor * ts

    if sm.fermi_level == "dftb+" and abs(nfill - round(nfill)) <= ELEC_TOL:
        mu = _middle_gap(ev, kw, nfill)
        if abs(nfill - _electron_count(mu, ev, kw, sm, kt)) <= ELEC_TOL_MAX:
            f, ts, _ = _electron_fill(mu, ev, kw, sm, kt)
            return [spin_factor * fk for fk in f], mu, spin_factor * ts

    # bisection exactly as DFTB+ Efilling
    upper = ev[:, -1].max() + 0.01
    lower = ev[:, 0].min() - 0.01
    n_max = _electron_count(upper, ev, kw, sm, kt)
    n_min = _electron_count(lower, ev, kw, sm, kt)
    while n_min > nfill:
        lower = 2.0 * (lower - upper) + lower
        n_min = _electron_count(lower, ev, kw, sm, kt)
    while n_max < nfill:
        upper = 2.0 * (upper - lower) + lower
        n_max = _electron_count(upper, ev, kw, sm, kt)
    mu = 0.5 * (upper + lower)
    n_el = _electron_count(mu, ev, kw, sm, kt)
    while abs(nfill - n_el) > ELEC_TOL and abs(upper - lower) >= max(abs(mu) * EPS2, EPS2):
        if (n_max >= n_min) == (nfill >= n_el):
            lower, n_min = mu, n_el
        else:
            upper, n_max = mu, n_el
        mu = 0.5 * (upper + lower)
        n_el = _electron_count(mu, ev, kw, sm, kt)
    if abs(nfill - n_el) > ELEC_TOL_MAX:
        raise RuntimeError("Fermi level search did not converge")
    f, ts, _ = _electron_fill(mu, ev, kw, sm, kt)
    if nfill > EPS2 and abs(nfill - n_el) > EPS2:
        f = f * nfill / n_el
    return [spin_factor * fk for fk in f], mu, spin_factor * ts
