"""Density matrices by purification, without eigenvalues (milestone M7a).

A port of NTPoly 2.7's dense algorithms as DFTB+ uses them through ELSI 2.9
(``Solver = NTPoly``; NTPoly ``DensityMatrixSolversModule``, ``SquareRootSolversModule``,
ELSI ``elsi_ntpoly.f90``), step for step, including where NTPoly flushes small elements
(``truncation``), its convergence test and its chemical potential:

* Z = S^-1/2 by the coupled Newton-Schulz iteration of Taylor order 5 (threshold
  max(0.01 truncation, 1e-15)), once per geometry;
* X from H~ = Z H Z^T by PM, TRS2, TRS4 or HPCP, stopped when |E - E_previous| <= tolerance
  with E = 2 tr(X H~) (TRS4 also when |tr G| < 1e-14);
* the chemical potential: bisection for the energy that the composed purification polynomial
  maps to 1/2, over all steps but the last;
* P = g Z^T X Z and W = P H P / g (g = 2 closed shell, 1 per spin channel or spinor).

Matrices are NumPy arrays, torch tensors (any device) or SciPy CSR matrices (``sparse=True``,
milestone M7c: DFTB+'s ``Sparse = Yes``, see :mod:`pydftb.csr`); scalars of the control flow are
Python floats.  ``NTPoly`` is the solver object of :class:`pydftb.DFTB` (``solver="ntpoly"``):
no eigenvalues, the T = 0 density, one purification per k-point and spin channel.
"""
from __future__ import annotations

import contextlib
import warnings

import numpy as np

from . import csr as _csr

METHODS = {"pm": 0, "trs2": 1, "trs4": 2, "hpcp": 3}
NAMES = {v: k for k, v in METHODS.items()}
SIGMA_MIN, SIGMA_MAX = 0.0, 6.0          # NTPoly TRS4
TRACE_G_EXIT = 1e-14                     # NTPoly TRS4: |tr G| below it ends the iteration
MIXED_FULL_STEPS = 4                     # precision="mixed": full-precision steps at most
MIXED_SCC_TOL = 1e-5                     # precision="mixed": SCC tolerances below it are not met


# ---- array helpers: NumPy arrays, torch tensors or CSR matrices ------------------------------
def _is_torch(a) -> bool:
    return type(a).__module__.startswith("torch")


_is_sparse = _csr.is_sparse


def _eye(n: int, like):
    if _is_torch(like):
        import torch
        return torch.eye(n, dtype=like.dtype, device=like.device)
    if _is_sparse(like):
        return _csr.eye(n, like.dtype)
    return np.eye(n, dtype=like.dtype)


def _ct(a):
    """Conjugate transpose."""
    if _is_torch(a):
        return a.mH
    if _is_sparse(a):
        return _csr.ct(a)
    return a.conj().T


def _trace(a) -> float:
    if _is_torch(a):
        return float(a.diagonal(dim1=-2, dim2=-1).sum().real)
    if _is_sparse(a):
        return _csr.trace(a)
    return float(np.trace(a).real)


def _dot(a, b) -> float:
    """Re sum(conj(a) * b) = tr(A^H B), NTPoly's DotMatrix (the real case)."""
    if _is_torch(a):
        return float((a.conj() * b).sum().real)
    if _is_sparse(a):
        return _csr.dot(a, b)
    return float(np.vdot(a, b).real)


def _drop(a, threshold: float):
    """NTPoly's flushing: elements with |a| < threshold set to zero."""
    if _is_sparse(a):
        return _csr.drop(a, threshold)
    if threshold <= 0.0:
        return a
    if _is_torch(a):
        import torch
        return torch.where(a.abs() < threshold, torch.zeros_like(a), a)
    return np.where(np.abs(a) < threshold, 0.0, a).astype(a.dtype, copy=False)


def _mm(a, b, threshold: float):
    if _is_sparse(a):
        return _csr.matmul(a, b, threshold)
    return _drop(a @ b, threshold)


def _abs_column_sums(a):
    if _is_torch(a):
        return a.abs().sum(dim=0)
    if _is_sparse(a):
        return _csr.abs_column_sums(a)
    return np.abs(a).sum(axis=0)


def gershgorin(h) -> tuple[float, float]:
    """Gershgorin bounds of a Hermitian matrix (NTPoly GershgorinBounds)."""
    if _is_torch(h):
        d = h.diagonal().real
        r = h.abs().sum(dim=0) - h.diagonal().abs()
        return float((d - r).min()), float((d + r).max())
    if _is_sparse(h):
        return _csr.gershgorin(h)
    d = np.diag(h).real
    r = np.abs(h).sum(axis=0) - np.abs(np.diag(h))
    return float((d - r).min()), float((d + r).max())


def _cast(a, single: bool):
    """float32/complex64 (``single``) or float64/complex128 copies."""
    if _is_torch(a):
        import torch
        if a.is_complex():
            return a.to(torch.complex64 if single else torch.complex128)
        return a.to(torch.float32 if single else torch.float64)
    if np.iscomplexobj(a):
        return a.astype(np.complex64 if single else np.complex128)
    return a.astype(np.float32 if single else np.float64)


# ---- S^-1/2 --------------------------------------------------------------------------------
_ISR = {}


def _isr_coefficients():
    """NTPoly NewtonSchultzISRTaylor, order 5."""
    if not _ISR:
        aa, bb, cc, dd = -40.0 / 35.0, 48.0 / 35.0, -64.0 / 35.0, 128.0 / 35.0
        a = (aa - 1.0) / 2.0
        b = bb * (a + 1.0) - cc - a * (a + 1.0) ** 2
        c = bb - b - a * (a + 1.0)
        d = dd - b * c
        _ISR.update(a=a, b=b, c=c, d=d)
    return _ISR


def inverse_sqrt(s, tolerance: float, threshold: float, max_iterations: int = 100):
    """S^-1/2 by NTPoly's coupled Newton-Schulz iteration (Taylor order 5): (Z, iterations).
    The norm of X = Z Y - I (the largest column sum of |X|) is tested before each update and
    the iteration ends after the update that follows a norm <= ``tolerance``."""
    co = _isr_coefficients()
    eye = _eye(s.shape[0], s)
    e_min, e_max = gershgorin(s)
    lam = 1.0 / max(abs(e_min), abs(e_max))
    z = eye
    y = s * lam
    it = 0
    for it in range(1, max_iterations + 1):
        x = _mm(z, y, threshold) - eye
        norm = float(_abs_column_sums(x).max())
        temp = _mm(x, x, threshold) + co["a"] * x
        temp2 = co["b"] * eye + x + temp
        temp = temp + co["c"] * eye
        x = (_mm(temp2, temp, threshold) + co["d"] * eye) * (35.0 / 128.0)
        z = _mm(x, z, threshold)
        y = _mm(y, x, threshold)
        if norm <= tolerance:
            break
    return z * float(np.sqrt(lam)), it


# ---- the four purifications ----------------------------------------------------------------
class _Purification:
    """One NTPoly method: the starting matrix, one step and the scalar map of a step (for
    the chemical potential)."""

    def start(self, hw, nel):
        raise NotImplementedError

    def step(self, x, nel, threshold):
        """(next X or None to stop now, sigma)."""
        raise NotImplementedError

    def scalar(self, sigma, v):
        raise NotImplementedError

    def mu(self, mid):
        raise NotImplementedError


class _TRS2(_Purification):
    def start(self, hw, nel):
        self.e_min, self.e_max = gershgorin(hw)
        return (self.e_max * _eye(hw.shape[0], hw) - hw) * (1.0 / (self.e_max - self.e_min))

    def step(self, x, nel, threshold):
        sigma = -1.0 if nel * 0.5 - _trace(x) < 0.0 else 1.0
        x2 = _mm(x, x, threshold)
        return (_drop(2.0 * x - x2, threshold) if sigma > 0.0 else x2), sigma

    def scalar(self, sigma, v):
        return v * v if sigma < 0.0 else 2.0 * v - v * v

    def mu(self, mid):
        return self.e_max + (self.e_min - self.e_max) * mid


class _TRS4(_TRS2):
    def step(self, x, nel, threshold):
        x2 = _mm(x, x, threshold)
        fx = 4.0 * x - 3.0 * x2
        gx = _eye(x.shape[0], x) - 2.0 * x + x2
        trace_f, trace_g = _dot(x2, fx), _dot(x2, gx)
        if abs(trace_g) < TRACE_G_EXIT:
            return None, None
        sigma = (nel * 0.5 - trace_f) / trace_g
        if sigma > SIGMA_MAX:
            return 2.0 * x - x2, sigma
        if sigma < SIGMA_MIN:
            return x2, sigma
        return _mm(x2, fx + sigma * gx, threshold), sigma

    def scalar(self, sigma, v):
        if sigma > SIGMA_MAX:
            return 2.0 * v - v * v
        if sigma < SIGMA_MIN:
            return v * v
        return v * v * (4.0 * v - 3.0 * v * v) + sigma * v * v * (1.0 - v) * (1.0 - v)


class _PM(_Purification):
    def start(self, hw, nel):
        n = hw.shape[0]
        self.n = n
        self.e_min, self.e_max = gershgorin(hw)
        self.lam = _trace(hw) / n
        self.alpha = min(nel * 0.5 / (self.e_max - self.lam), (n - nel * 0.5) / (self.lam - self.e_min))
        return hw * (-self.alpha / n) + _eye(n, hw) * ((self.alpha * self.lam + nel * 0.5) / n)

    def step(self, x, nel, threshold):
        x2 = _mm(x, x, threshold)
        x3 = _mm(x, x2, threshold)
        tmp = _drop(x - x2, threshold)
        tv = _trace(tmp)
        sigma = 1.0 if tv <= np.finfo(float).tiny else _dot(tmp, x) / tv
        if sigma > 0.5:
            a1, a2, a3 = 0.0, 1.0 + 1.0 / sigma, -1.0 / sigma
        else:
            a1 = (1.0 - 2.0 * sigma) / (1.0 - sigma)
            a2 = (1.0 + sigma) / (1.0 - sigma)
            a3 = -1.0 / (1.0 - sigma)
        return _drop(_drop(a1 * x + a2 * x2, threshold) + a3 * x3, threshold), sigma

    def scalar(self, sigma, v):
        if sigma > 0.5:
            return ((1.0 + sigma) * v * v - v ** 3) / sigma
        return ((1.0 - 2.0 * sigma) * v + (1.0 + sigma) * v * v - v ** 3) / (1.0 - sigma)

    def mu(self, mid):
        return self.lam - (self.n * mid - self.nel * 0.5) / self.alpha


class _HPCP(_Purification):
    def start(self, hw, nel):
        n = hw.shape[0]
        e_min, e_max = gershgorin(hw)
        self.mu0 = _trace(hw) / n
        sigma_bar = (n - 0.5 * nel) / n
        sigma = 1.0 - sigma_bar
        self.beta_1 = sigma
        self.beta_2 = min(sigma / (e_max - self.mu0), sigma_bar / (self.mu0 - e_min))
        eye = _eye(n, hw)
        return eye * self.beta_1 + (self.mu0 * eye - hw) * self.beta_2

    def step(self, x, nel, threshold):
        dh = _eye(x.shape[0], x) - x
        ddh = _mm(x, dh, threshold)
        d2dh = _mm(x, ddh, threshold)
        sigma = _trace(d2dh) / _trace(ddh)
        return x + 2.0 * d2dh - (2.0 * sigma) * ddh, sigma

    def scalar(self, sigma, v):
        return v + 2.0 * (v * v * (1.0 - v) - sigma * v * (1.0 - v))

    def mu(self, mid):
        return self.mu0 + (self.beta_1 - mid) / self.beta_2


_CLASSES = {0: _PM, 1: _TRS2, 2: _TRS4, 3: _HPCP}


def method_index(method) -> int:
    """0-3 from a name ("pm", "trs2", "trs4", "hpcp") or DFTB+'s PurificationMethod."""
    if isinstance(method, str):
        key = method.lower()
        if key not in METHODS:
            raise ValueError(f"unknown purification method {method!r}: use one of "
                             f"{sorted(METHODS)} or DFTB+'s 0-3")
        return METHODS[key]
    m = int(method)
    if m not in _CLASSES:
        raise ValueError(f"unknown purification method {method!r}: DFTB+'s 0-3")
    return m


def purify(hw, nel: float, method=2, tolerance: float = 1e-5, threshold: float = 1e-10,
           max_iterations: int = 100, mixed: bool = False):
    """X (trace nel/2), its band energy 2 tr(X H~), the chemical potential and the number of
    steps, for the orthogonalised H~ = ``hw``.  ``mixed``: the steps in single precision
    until NTPoly's test holds or the energy changes stop shrinking, then in the precision of
    ``hw`` (not part of NTPoly)."""
    m = _CLASSES[method_index(method)]()
    m.nel = nel
    x = m.start(hw, nel)
    work = _cast(hw, True) if mixed else hw
    if mixed:
        x = _cast(x, True)
    sigmas = []
    energy = 0.0
    total = max_iterations             # NTPoly: total_iterations = outer_counter - 1
    single = mixed
    noise = 10.0 * hw.shape[0] * float(np.finfo(np.float32).eps)
    near = 1e-2 * float(np.sqrt(hw.shape[0]))
    idem = []                          # mixed: tr(X - X^2) of the single-precision steps
    full_steps = 0
    for it in range(1, max_iterations + 1):
        x_new, sigma = m.step(x, nel, threshold)
        if x_new is None:              # TRS4: |tr G| < 1e-14
            total = it - 1
            break
        sigmas.append(sigma)
        x = x_new
        e_old, energy = energy, 2.0 * _dot(x, work)
        if single:
            # mixed: leave single precision at its noise floor, or when tr(X - X^2) stops
            # shrinking over two steps once it is small (pydftb's rule after Kruchinin et al.;
            # early steps may raise it)
            idem.append(abs(_trace(x) - _dot(x, x)))
            if idem[-1] <= noise or (len(idem) > 3 and idem[-1] >= idem[-3]
                                     and idem[-1] < near):
                single = False
                x, work = _cast(x, False), hw
            continue
        if mixed:
            full_steps += 1
        # NTPoly's test; mixed: at most MIXED_FULL_STEPS steps in full precision (the
        # single-precision subspace error stays, further steps do not remove it)
        if abs(energy - e_old) <= tolerance or full_steps >= MIXED_FULL_STEPS:
            total = it - 1
            break
    if total == max_iterations and not mixed:
        warnings.warn(f"the {NAMES[method_index(method)].upper()} purification did not converge "
                      f"in {max_iterations} steps (no gap at this occupation?)", RuntimeWarning,
                      stacklevel=4)
    mid = _chemical_potential_point(m, sigmas[:total], tolerance, max_iterations)
    if mixed and x.dtype != hw.dtype:          # ended in single precision
        x = _cast(x, False)
        if x.dtype != hw.dtype:
            x = x.to(hw.dtype) if _is_torch(x) else x.astype(hw.dtype)
    return x, energy, m.mu(mid), len(sigmas)


def _chemical_potential_point(m: _Purification, sigmas, tolerance: float,
                              max_iterations: int) -> float:
    """NTPoly: bisection on [0, 1] for the point the composed step maps to 1/2."""
    a, b = 0.0, 1.0
    mid = 0.0
    for _ in range(max_iterations):
        mid = (b - a) / 2.0 + a
        v = mid
        for s in sigmas:
            v = m.scalar(s, v)
        if v < 0.5:
            a = mid
        else:
            b = mid
        if abs(v - 0.5) < tolerance:
            break
    return mid


# ---- the solver ------------------------------------------------------------------------------
class NTPoly:
    """DFTB+'s ``Solver = NTPoly`` (ELSI): densities by purification, no eigenvalues, T = 0.

    ``method``: "pm", "trs2", "trs4" (DFTB+'s default) or "hpcp", or DFTB+'s 0-3;
    ``tolerance`` (DFTB+ 1e-5) and ``truncation`` (DFTB+ 1e-10): NTPoly's convergence test and
    the flushing of small elements; ``precision="mixed"``: the purification steps in single
    precision first (pydftb's, not NTPoly's; the charges are then good to about 1e-6).
    Each k-point and spin channel (or spinor matrix) is purified at its own integer occupation;
    the chemical potential of several k-points is their weighted mean.

    ``sparse=True`` (M7c, DFTB+'s ``Sparse = Yes``): the calculator keeps H0, S, H, rho and W on
    the pattern of the neighbour blocks and hands CSR matrices to the solver, whose products are
    then sparse with the same flushing (:mod:`pydftb.csr`); ``threads``: the threads of the CSR
    products (None: all cores); ``dense_fill``: two factors filled above it are multiplied
    dense (NTPoly's 0.1; 1 or more: always sparse).  ``last_fill``: the mean number of elements
    per row of the last density's X."""
    name = "ntpoly"
    eigenpairs = False

    def __init__(self, method="trs4", tolerance: float = 1e-5, truncation: float = 1e-10,
                 precision: str | None = None, max_iterations: int = 100, threads=None,
                 sparse: bool = False, dense_fill: float = _csr.DENSE_FILL):
        self.method = method_index(method)
        self.tolerance = float(tolerance)
        self.truncation = float(truncation)
        if precision not in (None, "double", "mixed"):
            raise ValueError(f"precision must be None, 'double' or 'mixed', not {precision!r}")
        self.mixed = precision == "mixed"
        self.max_iterations = int(max_iterations)
        self.sparse = bool(sparse)
        if self.sparse and self.mixed:
            raise ValueError("precision='mixed' is not used with sparse=True: the CSR products "
                             "are not faster in single precision (milestone M7c)")
        self.product_threads = int(threads) if self.sparse and threads is not None else None
        self.dense_fill = float(dense_fill)
        self.threads = None            # one purification after another
        self.last_iterations = []
        self.isr_iterations = []
        self.last_fill = None

    def __repr__(self):
        return (f"NTPoly(method={NAMES[self.method]!r}, tolerance={self.tolerance:g}, "
                f"truncation={self.truncation:g}"
                + (", precision='mixed'" if self.mixed else "")
                + (", sparse=True" if self.sparse else "") + ")")

    def _products(self):
        """The threads and the dense fill of the CSR products (M7c)."""
        if not self.sparse:
            return contextlib.nullcontext()
        return _csr.options(self.product_threads, self.dense_fill)

    def prepare(self, s):
        """S^-1/2 for every matrix of the stack (a repeated half, the spin channels' copy of
        the k-points, is not computed twice), once per geometry; ``last_iterations`` then
        collects the purification steps of this geometry's SCC, a list per density call.
        A list of CSR matrices: the sparse path (M7c)."""
        self.last_iterations = []
        thr = max(0.01 * self.truncation, 1e-15)
        n = len(s)
        half = n // 2
        if isinstance(s, list):
            same = n % 2 == 0 and n > 1 and all(a is b or _equal(a, b)
                                                for a, b in zip(s[:half], s[half:]))
        else:
            same = n % 2 == 0 and n > 1 and _equal(s[:half], s[half:])
        zs, its = [], []
        with self._products():
            for i in range(half if same else n):
                z, it = inverse_sqrt(s[i], self.tolerance, thr, self.max_iterations)
                zs.append(z)
                its.append(it)
        self.z = zs + zs if same else zs
        self.isr_iterations = its

    def density(self, h, kw, n_occ: float, spin_factor: float, need_w: bool, offset: int = 0):
        """(P stack, W stack or None, chemical potential) of the matrices ``h`` (k-points of
        one channel), each with ``n_occ`` occupied levels of ``spin_factor`` electrons;
        ``offset``: the first matrix's index in the prepared stack.  CSR matrices (M7c): lists
        of CSR matrices."""
        n_int = round(float(n_occ))
        if abs(n_occ - n_int) > 1e-8:
            raise ValueError(
                f"the purification needs a whole number of occupied levels per spin channel, "
                f"not {n_occ:g}: use spin polarisation (DFTB+'s NTPoly rounds the electron "
                "count to an even number)")
        thr = self.truncation
        nel = 2.0 * n_int
        ps, ws, mus = [], [], []
        its, fills = [], []
        with self._products():
            for i, hk in enumerate(h):
                z = _like(self.z[offset + i], hk)
                zh = _ct(z)
                hw = _mm(_mm(z, hk, thr), zh, thr)
                x, _, mu, it = purify(hw, nel, self.method, self.tolerance, thr,
                                      self.max_iterations, self.mixed)
                p = _mm(_mm(zh, x, thr), z, thr) * spin_factor
                ps.append(p)
                if need_w:
                    ws.append(_mm(_mm(p, hk, thr), p, thr) * (1.0 / spin_factor))
                mus.append(mu)
                its.append(it)
                if _is_sparse(x):
                    fills.append(x.nnz / x.shape[0])
        self.last_iterations.append(its)
        if fills:
            self.last_fill = float(np.mean(fills))
        mu = float(np.dot(np.asarray(kw, float), mus) / np.sum(kw))
        return _stack(ps), (_stack(ws) if need_w else None), mu

    def solve(self, h, kw, nelec, smearing, need_w=True):
        """The closed-shell interface of the other solvers: (rho, W, None, mu, 0)."""
        rho, wdm, mu = self.density(h, kw, nelec / 2.0, 2.0, need_w)
        self.last_occupations = None
        return rho, wdm, None, mu, 0.0


def _like(z, h):
    """Z in the dtype of H (torch: the backend's working precision; NumPy promotes)."""
    if _is_torch(z) and z.dtype != h.dtype:
        return z.to(h.dtype)
    return z


def _equal(a, b) -> bool:
    if _is_torch(a):
        import torch
        return bool(torch.equal(a, b))
    if _is_sparse(a):
        return _csr.equal(a, b)
    return bool(np.array_equal(a, b))


def _stack(parts):
    if _is_torch(parts[0]):
        import torch
        return torch.stack(parts)
    if _is_sparse(parts[0]):
        return list(parts)
    return np.stack(parts)
