"""Finite-temperature densities without eigenpairs (milestone M7b): the recursive Fermi-operator
expansion (A. M. N. Niklasson, Phys. Rev. B 68, 233104 (2003)).

* The map x -> x^2 / (x^2 + (1 - x)^2) takes f(y) = 1/(1 + e^y) to f(2y) exactly.  From
  X0 = 1/2 - T/4, T = (H~ - mu0) / (kT 2^n) (f to first order), n doublings give f(Y'), with
  Y' = 2^(n+1) artanh(T/2) = Y + Y^3/(12 4^n) + Y^5/(80 16^n) + ...  (Y = (H~ - mu0)/kT): the
  occupations differ from f(Y) by at most about 0.1/4^n (n = 22: 6e-15).
* The recursion runs on D = X - 1/2: D <- 2 D (1 + 4 D^2)^-1 (the Cholesky factor of 1 + 4 D^2,
  eigenvalues in [1, 2]).  Its rounding is relative to the small early D, about beta rho eps in
  all (rho: the spectral radius of H~ - mu0); on X it would grow as 2^n eps.
* The chemical potential: P(mu0 + d) = P [P + e^(-d/kT) (1 - P)]^-1 exactly (mu is a multiple of
  1), accurate within about 5 kT of mu0 (farther, the occupations far from mu0 have lost their
  relative precision): Newton in that window, otherwise a step to a new recursion, inside the
  Gershgorin bounds.  The SCC starts each search from the previous chemical potential.
* The entropy of the computed P from the recursion's log-determinants:
  S = N ln 2 + sum_k 2^(n-1-k) tr ln(1 + 4 D_k^2) - 2^n tr ln(1 + T/2) + tr(P Y')
  (the terms 2^n N ln 2 cancel analytically; early log-determinants by trace series).
* W~ = f(H~) H~ (they commute): (P~ H~ + H~ P~)/2.  rho = g Z^+ P~ Z, W = g Z^+ W~ Z with
  Z = S^-1/2 (M7a's Newton-Schulz iteration, once per geometry).

Matrices are NumPy arrays or torch tensors; the chemical potential is the exact electron count
(pydftb's ``fermi_level="exact"``: DFTB+'s rule of the mid-gap level needs the eigenvalues).
"""
from __future__ import annotations

import math

import numpy as np

from .purification import _ct, _equal, _eye, _is_torch, _like, _stack, _trace, inverse_sqrt

STEPS = 22                   # doublings
WINDOW = 5.0                 # kT: the exact shift's range
N_TOL = 1e-11                # electron count: enough in a gap, where N(mu) is flat to rounding
N_EXACT = 1e-14              # electron count (relative): the chemical potential found
SERIES_BELOW = 1e-2          # ||4 D^2|| below it: tr ln(1 + 4 D^2) by the series
ISR_TOLERANCE = 1e-10        # S^-1/2: the Newton-Schulz test (one more step after it)


# ---- array helpers: NumPy arrays or torch tensors --------------------------------------------
def _cholesky(a, device=None):
    """Lower Cholesky factor (torch: on ``device`` if given, the result back on a's device)."""
    if _is_torch(a):
        import torch
        b = a if device is None else a.to(device)
        l, info = torch.linalg.cholesky_ex(b)
        if int(info) > 0:
            raise np.linalg.LinAlgError("a Fermi-operator matrix is not positive definite")
        return l
    return np.linalg.cholesky(a)


def _cho_solve(l, b):
    """A^-1 b for A = L L^H."""
    if _is_torch(l):
        import torch
        return torch.cholesky_solve(b.to(l.device), l).to(b.device)
    import scipy.linalg as sla
    return sla.cho_solve((l, True), b, check_finite=False)


def _logdet(l) -> float:
    """ln det A from its Cholesky factor."""
    if _is_torch(l):
        return 2.0 * float(l.diagonal().real.log().sum())
    return 2.0 * float(np.log(np.diag(l).real).sum())


def _hermitian(a):
    return 0.5 * (a + _ct(a))


def _norm1(a) -> float:
    if _is_torch(a):
        return float(a.abs().sum(dim=0).max())
    return float(np.abs(a).sum(axis=0).max())


def _gershgorin(h) -> tuple[float, float]:
    if _is_torch(h):
        d = h.diagonal().real
        r = h.abs().sum(dim=0) - h.diagonal().abs()
        return float((d - r).min()), float((d + r).max())
    d = np.diag(h).real
    r = np.abs(h).sum(axis=0) - np.abs(np.diag(h))
    return float((d - r).min()), float((d + r).max())


def _tr_log1p(a, definite: bool = True, max_terms: int = 60) -> float:
    """tr ln(1 + A) for a Hermitian A with a small norm, by the series, until the term is below
    1e-19: its trace for a positive semidefinite A (the terms then alternate and decrease), the
    bound N ||A||_1^j / j for an indefinite one (whose odd powers can have a vanishing trace)."""
    nb, q = a.shape[0], (None if definite else _norm1(a))
    out, ak = 0.0, a
    for j in range(1, max_terms + 1):
        t = _trace(ak)
        out += (-1) ** (j + 1) * t / j
        if (abs(t) if definite else nb * q ** j / j) < 1e-19:
            break
        ak = ak @ a
    return out


# ---- the recursion ------------------------------------------------------------------------------
class Recursion:
    """f((H~ - mu0)/kT) by ``steps`` doublings (more if ||Y||_1 > 2^(steps - 8), so that
    ||T||_1 <= 2^-8: X0 inside (0, 1), the series of tr ln(1 + T/2) short): D (P = 1/2 + D), the
    implied tr ln(1 + e^-Y') and what tr(P Y') needs."""

    def __init__(self, hw, mu0: float, kt: float, steps: int = STEPS, linalg_device=None):
        self.mu0, self.kt = float(mu0), float(kt)
        self.linalg_device = linalg_device
        nb = hw.shape[0]
        eye = _eye(nb, hw)
        self.y = (hw - self.mu0 * eye) * (1.0 / kt)                   # Y
        ny = _norm1(self.y)
        n = self.steps = max(int(steps), math.ceil(math.log2(ny)) + 8 if ny > 1.0 else 0)
        t = self.y * (1.0 / 2.0 ** n)
        d = -0.25 * t
        s_terms = 0.0
        for k in range(n):
            a = 4.0 * (d @ d)
            l = _cholesky(eye + a, linalg_device)
            ld = _tr_log1p(a) if _norm1(a) < SERIES_BELOW else _logdet(l)
            s_terms += 2.0 ** (n - 1 - k) * ld
            d = _hermitian(2.0 * _cho_solve(l, d))
        self.d = d
        # tr ln(1 + e^-Y'): N ln 2 + sum_k ... - 2^n tr ln(1 + T/2)
        self.trl = nb * math.log(2.0) + s_terms - 2.0 ** n * _tr_log1p(0.5 * t, definite=False)

    def shifted(self, d: float):
        """(P, ln det[P0 + c (1 - P0)]) at mu0 + d, c = e^(-d/kT)."""
        eye = _eye(self.d.shape[0], self.d)
        if d == 0.0:
            return 0.5 * eye + self.d, 0.0
        c = math.exp(-d / self.kt)
        m = (0.5 * (1.0 + c)) * eye + (1.0 - c) * self.d
        l = _cholesky(m, self.linalg_device)
        dn = _hermitian(_cho_solve(l, (0.25 * (1.0 - c)) * eye + (0.5 * (1.0 + c)) * self.d))
        return 0.5 * eye + dn, _logdet(l)

    def entropy(self, p, d: float, ld: float) -> float:
        """S (units of k) of P = the shifted density at mu0 + d (``ld`` from :meth:`shifted`)."""
        n = self.steps
        nb = p.shape[0]
        yp = self.y @ p
        y2 = self.y @ self.y
        y2yp = y2 @ yp
        tr_ypp = (_trace(yp) + _trace(y2yp) / (12.0 * 4.0 ** n)
                  + _trace(y2 @ y2yp) / (80.0 * 16.0 ** n))                  # tr(Y' P)
        trl = self.trl + nb * d / self.kt + ld                               # at mu0 + d
        return trl + tr_ypp - d / self.kt * _trace(p)


def _count(recs, w, d):
    """(N, dN/dmu) of the shifted densities, k-weights ``w``."""
    n, dn = 0.0, 0.0
    for r, wk in zip(recs, w):
        p, _ = r.shifted(d)
        n += wk * _trace(p)
        dn += wk * (_trace(p) - _dot_self(p)) / r.kt
    return n, dn


def _dot_self(p) -> float:
    """tr(P P) for a Hermitian P."""
    if _is_torch(p):
        return float((p.conj() * p).sum().real)
    return float(np.vdot(p, p).real)


def chemical_potential(hws, w, target: float, kt: float, mu_start=None, steps: int = STEPS,
                       linalg_device=None):
    """(mu, recursions at mu0, d, recursion count): sum_k w_k tr f((H~_k - mu)/kT) = ``target``
    to N_TOL, mu = mu0 + d with |d| <= WINDOW kT."""
    lo = min(_gershgorin(h)[0] for h in hws) - 40.0 * kt
    hi = max(_gershgorin(h)[1] for h in hws) + 40.0 * kt
    mu0 = 0.5 * (lo + hi) if mu_start is None else min(max(float(mu_start), lo), hi)
    win = WINDOW * kt
    count = 0
    for _ in range(200):
        recs = [Recursion(h, mu0, kt, steps, linalg_device) for h in hws]
        count += 1
        n0, dn0 = _count(recs, w, 0.0)
        if abs(n0 - target) < N_EXACT * max(1.0, target):     # e.g. the previous SCC iteration's
            return mu0, recs, 0.0, count
        n_m, dn_m = _count(recs, w, -win)
        n_p, dn_p = _count(recs, w, win)
        if n_m <= target <= n_p:
            d = _newton_in_window(recs, w, target, -win, win, n0, dn0)
            return mu0 + d, recs, d, count
        if abs(n0 - target) <= N_TOL:        # e.g. in a gap, where N(mu) is flat to rounding
            return mu0, recs, 0.0, count
        if target > n_p:
            lo = max(lo, mu0 + win)
            new = mu0 + win + (target - n_p) / dn_p if dn_p > 1e-300 else math.inf
        else:
            hi = min(hi, mu0 - win)
            new = mu0 - win - (n_m - target) / dn_m if dn_m > 1e-300 else -math.inf
        if not lo < new < hi:
            new = 0.5 * (lo + hi)
        if hi - lo < 1e-13:                  # the count cannot be resolved further
            recs = [Recursion(h, new, kt, steps, linalg_device) for h in hws]
            return new, recs, 0.0, count + 1
        mu0 = new
    raise RuntimeError("Fermi-operator expansion: no chemical potential found")


def _newton_in_window(recs, w, target, lo, hi, n, dn) -> float:
    """d in [lo, hi] with N(d) = target, Newton with bisection (N at d = 0 given)."""
    d = 0.0
    for _ in range(200):
        if abs(n - target) < N_EXACT * max(1.0, target):
            return d
        if n < target:
            lo = d
        else:
            hi = d
        new = d + (target - n) / dn if dn > 1e-300 else 0.5 * (lo + hi)
        if not lo <= new <= hi:
            new = 0.5 * (lo + hi)
        if abs(new - d) < 1e-16:
            return d
        d = new
        n, dn = _count(recs, w, d)
    return d


# ---- the solver ---------------------------------------------------------------------------------
class FermiOperator:
    """Fermi densities at T > 0 without eigenpairs (``solver="foe"``): the recursive
    Fermi-operator expansion with ``steps`` doublings at least (22: the occupations to about
    1e-14; more at low temperature, when ||H~ - mu||_1 / kT > 2^steps).  The rounding grows as
    ||H~ - mu|| / kT eps (about 5e-13 at 300 K): near T = 0 use ``"ntpoly"``.

    One recursion per k-point and spin channel and SCC iteration once the chemical potential
    settles; a shared chemical potential over the k-points (metals) and, with
    ``relax_total_spin``, the spin channels; the exact electron count (DFTB+'s mid-gap rule needs
    eigenvalues).  ``recursions`` counts the recursions of the last density call."""
    name = "foe"
    eigenpairs = False
    finite_temperature = True

    def __init__(self, steps: int = STEPS, threads=None):
        if int(steps) < 1:
            raise ValueError("steps must be >= 1")
        self.steps = int(steps)
        self.threads = None
        self.linalg_device = None        # torch: where Cholesky runs (set by the calculator)
        self._mu = {}
        self.recursions = 0

    def __repr__(self):
        return f"FermiOperator(steps={self.steps})"

    def prepare(self, s):
        """S^-1/2 of every matrix of the stack, once per geometry (a repeated half, the spin
        channels' copy of the k-points, is reused)."""
        n = len(s)
        half = n // 2
        same = n % 2 == 0 and n > 1 and _equal(s[:half], s[half:])
        zs = [inverse_sqrt(s[i], ISR_TOLERANCE, 0.0)[0] for i in range(half if same else n)]
        self.z = zs + zs if same else zs

    def density(self, h, kw, n_occ: float, spin_factor: float, need_w: bool, offset: int = 0,
                kt: float | None = None, mu: float | None = None, independent_k: bool = False,
                channels: int = 1):
        """(rho stack, W stack or None, chemical potential, T S) of the matrices ``h`` (``channels``
        copies of the k-points ``kw``, from ``offset`` in the prepared stack): sum_c sum_k w_k tr P
        = ``n_occ`` (w normalised over the k-points), ``spin_factor`` electrons per level;
        ``mu``: a fixed chemical potential; ``independent_k``: every k-point holds ``n_occ`` on its
        own (the chemical potential and T S are then the k-weighted sums)."""
        if kt is None:
            raise ValueError("the Fermi-operator expansion needs kT")
        kw = np.asarray(kw, float)
        w = np.tile(kw / kw.sum(), channels)
        hws = []
        for i, hk in enumerate(h):
            z = _like(self.z[offset + i], hk)
            hws.append(_hermitian(z @ hk @ _ct(z)))
        self.recursions = 0
        if independent_k:
            groups = [([i], [1.0], n_occ) for i in range(len(hws))]
        else:
            groups = [(list(range(len(hws))), list(w), n_occ)]
        ps, mus, ts = [None] * len(hws), [], 0.0
        for idx, ww, target in groups:
            key = (offset, len(hws), channels, independent_k, idx[0])
            sub = [hws[i] for i in idx]
            if mu is not None:
                recs = [Recursion(hh, mu, kt, self.steps, self.linalg_device) for hh in sub]
                m, d = float(mu), 0.0
                self.recursions += 1
            else:
                m, recs, d, count = chemical_potential(sub, ww, target, kt, self._mu.get(key),
                                                       self.steps, self.linalg_device)
                self.recursions += count
                self._mu[key] = m
            mus.append(m)
            for i, r in zip(idx, recs):
                p, ld = r.shifted(d)
                ps[i] = p
                ts += w[i] * kt * spin_factor * r.entropy(p, d, ld)
        rho, wdm = [], []
        for i, (p, hw) in enumerate(zip(ps, hws)):
            z = _like(self.z[offset + i], hw)
            rho.append(_ct(z) @ p @ z * spin_factor)
            if need_w:
                wdm.append(_ct(z) @ _hermitian(p @ hw) @ z * spin_factor)
        mu_out = (float(np.dot(kw / kw.sum(), mus)) if independent_k else mus[0])
        return _stack(rho), (_stack(wdm) if need_w else None), mu_out, float(ts)

    def solve(self, h, kw, nelec, smearing, need_w=True):
        """The closed-shell interface of the other solvers: (rho, W, None, mu, T S)."""
        rho, wdm, mu, ts = self.density(h, kw, nelec / 2.0, 2.0, need_w, kt=smearing.kt)
        self.last_occupations = None
        return rho, wdm, None, mu, ts
