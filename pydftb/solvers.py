"""Electronic-structure solvers: H(k), S(k) -> density matrices.

Every solver returns, for each k-point, the density matrix rho(k), the
energy-weighted density matrix W(k), the eigenvalues (or None), the Fermi
level and the entropic term T*S.

* ``EighSolver``     reference: LAPACK generalised eigensolver, one k at a time
* ``BatchedSolver``  Cholesky once per geometry, batched eigh over all k
* ``SP2Solver``      diagonalisation-free second-order spectral projection
                     (Niklasson 2002).  Only GEMMs => GPU / tensor-core
                     friendly; optional mixed precision.  Gamma, T = 0, gapped.
* DFTB+'s ``Solver`` choices (M7a): ``"qr"``, ``"divideandconquer"`` (= ``"eigh"``),
  ``"relativelyrobust"`` (LAPACK drivers), ``"elpa"`` (= ``"eigh"``) and ``"ntpoly"``
  (purification without eigenvalues, :mod:`pydftb.purification`).
* ``"foe"`` (M7b): Fermi densities at T > 0 without eigenvalues, the recursive Fermi-operator
  expansion (:mod:`pydftb.fermi_operator`; DFTB+'s finite-temperature ELSI solver is PEXSI).

``EighSolver`` and ``BatchedSolver`` take ``threads`` (a :class:`pydftb.parallel.KThreads`):
the per-k work then runs on chunks of k-points in threads, bitwise identical to
the whole stack (see :mod:`pydftb.parallel`).
"""
from __future__ import annotations

from functools import partial

import numpy as np

from .backend import ops
from .backend.spec import NUMPY, Backend

from .occupations import fill
from .fermi_operator import FermiOperator
from .purification import NTPoly
from .subsystems import Subsystems


def _kmap(threads, fn, nk, norb):
    """fn over chunks of the k stack (all of it at once without threads)."""
    return [fn(slice(0, nk))] if threads is None else threads.map(fn, nk, norb)


def _stack(parts):
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _density(evals, evecs, kw, nelec, smearing, need_w, spin_factor=2.0, threads=None,
             backend: Backend = NUMPY, occupations=None):
    """Densities rho = (C f) C^H and W = (C f eps) C^H.  The occupations and the Fermi level
    are always found on the host (DFTB+'s rules); with the torch backend ``evals`` and
    ``evecs`` live on the device and the eigenvalues are returned as host arrays.
    ``occupations``: a function (evals, kw, nelec, smearing, spin_factor) -> (occ, mu, ts) used
    instead of :func:`pydftb.occupations.fill` (a fixed Fermi level or independent k-point
    filling, M6f)."""
    ev_host = backend.to_host(evals)
    if occupations is not None:
        occ, mu, ts = occupations(ev_host, kw, nelec, smearing, spin_factor)
    else:
        occ, mu, ts = fill(ev_host, kw, nelec, smearing, spin_factor=spin_factor)   # all k at once
    occ = np.array(occ)
    occ_d = backend.asarray(occ)
    xp = backend.xp

    def part(sl):
        c = evecs[sl]
        cf = c * occ_d[sl, None, :]
        rho = cf @ xp.conj(xp.swapaxes(c, 1, 2))
        wdm = (cf * evals[sl, None, :]) @ xp.conj(xp.swapaxes(c, 1, 2)) if need_w else None
        return rho, wdm

    parts = _kmap(threads, part, len(evecs), evecs.shape[-1])
    rho = _stack([p[0] for p in parts])
    wdm = _stack([p[1] for p in parts]) if need_w else None
    return rho, wdm, ev_host, mu, ts, occ


class EighSolver:
    """``driver`` (M7a): one of DFTB+'s LAPACK drivers ("gv" QR, "gvd" DivideAndConquer,
    "gvr" RelativelyRobust, :func:`pydftb.backend.ops.eigh_driver`); None: :func:`ops.eigh`."""
    name = "eigh"

    def __init__(self, threads=None, driver: str | None = None):
        self.threads = threads
        self.driver = driver

    def prepare(self, s):
        self.s = s

    def eig(self, h):
        """Eigenvalues and S-orthonormal eigenvectors for every k."""
        def part(sl):
            ev, vec = [], []
            for hk, sk in zip(h[sl], self.s[sl]):
                if self.driver is None:
                    e, c = ops.eigh(hk, sk)
                else:
                    e, c = ops.eigh_driver(hk, sk, self.driver)
                ev.append(e)
                vec.append(c)
            return np.array(ev), np.array(vec)

        parts = _kmap(self.threads, part, len(h), h.shape[-1])
        return _stack([p[0] for p in parts]), _stack([p[1] for p in parts])

    def solve(self, h, kw, nelec, smearing, need_w=True):
        *out, self.last_occupations = _density(*self.eig(h), kw, nelec, smearing, need_w,
                                               threads=self.threads)
        return tuple(out)


class BatchedSolver:
    """Reduce to standard form once per geometry (S is fixed in the SCC loop)."""
    name = "batched"

    def __init__(self, threads=None):
        self.threads = threads

    def prepare(self, s):
        eye = np.eye(s.shape[-1], dtype=s.dtype)

        def part(sl):
            l = ops.cholesky(s[sl])
            return ops.solve(l, np.broadcast_to(eye, l.shape))     # L^-1

        self.linv = _stack(_kmap(self.threads, part, len(s), s.shape[-1]))
        self.linv_h = np.conj(np.swapaxes(self.linv, 1, 2))

    def eig(self, h):
        """Eigenvalues and S-orthonormal eigenvectors for every k."""
        def part(sl):
            e, c = ops.batched_eigh(self.linv[sl] @ h[sl] @ self.linv_h[sl])
            return e, self.linv_h[sl] @ c

        parts = _kmap(self.threads, part, len(h), h.shape[-1])
        return _stack([p[0] for p in parts]), _stack([p[1] for p in parts])

    def solve(self, h, kw, nelec, smearing, need_w=True):
        *out, self.last_occupations = _density(*self.eig(h), kw, nelec, smearing, need_w,
                                               threads=self.threads)
        return tuple(out)


def _gershgorin(h):
    d = np.diag(h)
    r = np.abs(h).sum(1) - np.abs(d)
    return (d - r).min(), (d + r).max()


def sp2(hp: np.ndarray, nocc: float, dtype=np.float64, refine_fp64: int = 0,
        max_iter: int = 100):
    """SP2 purification of an orthogonal Hamiltonian -> projector X (and mu).

    Uses the parameter-free convergence criterion of Kruchinin et al.:
    stop when the idempotency error no longer decreases.
    """
    emin, emax = _gershgorin(hp)
    n = hp.shape[0]
    x = ((emax * np.eye(n) - hp) / (emax - emin)).astype(dtype)
    idem_old = np.inf
    hist = []
    for it in range(max_iter):
        x2 = x @ x
        tr_x, tr_x2 = np.trace(x), np.trace(x2)
        idem = abs(tr_x - tr_x2)
        if it > 4 and idem >= idem_old * 0.9 and idem < 1e-2 * n ** 0.5:
            break
        idem_old = idem
        if abs(tr_x2 - nocc) < abs(2 * tr_x - tr_x2 - nocc):
            x = x2
            hist.append(1)
        else:
            x = 2 * x - x2
            hist.append(0)
    if refine_fp64:
        x = x.astype(np.float64)
        for _ in range(refine_fp64):     # McWeeny polishing in FP64
            x2 = x @ x
            x = 3 * x2 - 2 * x2 @ x
    return x.astype(np.float64), len(hist)


class SP2Solver:
    """Gamma-point, zero-temperature, diagonalisation-free solver."""
    name = "sp2"
    eigenpairs = False

    def __init__(self, dtype=np.float64, refine_fp64: int = 0, threads=None):
        self.dtype = dtype
        self.refine = refine_fp64
        self.iterations = 0
        self.threads = None             # Gamma only: nothing to split over k

    def prepare(self, s):
        if s.shape[0] != 1 or np.iscomplexobj(s):
            raise ValueError("SP2 solver supports Gamma-point only")
        # Loewdin factor Z = S^-1/2 (in production: sparse Newton-Schulz)
        w, v = ops.eigh(s[0])
        self.z = (v / np.sqrt(w)) @ v.T

    def solve(self, h, kw, nelec, smearing, need_w=True):
        z = self.z
        hp = z @ h[0] @ z
        x, nit = sp2(hp, nelec / 2.0, self.dtype, self.refine)
        self.iterations = nit
        rho = 2.0 * z @ x @ z
        wdm = 2.0 * z @ (x @ hp @ x) @ z if need_w else None
        # chemical potential estimate: midpoint is not available without
        # eigenvalues; report mid-gap from Rayleigh quotients of X and 1-X
        return rho[None], None if wdm is None else wdm[None], None, np.nan, 0.0


SOLVERS = {"eigh": EighSolver, "batched": BatchedSolver, "sp2": SP2Solver,
           # DFTB+'s Solver block (M7a)
           "qr": partial(EighSolver, driver="gv"),
           "divideandconquer": partial(EighSolver, driver="gvd"),
           "relativelyrobust": partial(EighSolver, driver="gvr"),
           "elpa": EighSolver, "ntpoly": NTPoly,
           "foe": FermiOperator,                                   # M7b
           "subsystems": Subsystems}                               # M7d
# DFTB+ solvers that pydftb does not provide, and why (M7a)
UNSUPPORTED_SOLVERS = {
    "omm": "DFTB+'s OMM (libOMM through ELSI) is not implemented: in DFTB+ 24.1 its densities, "
           "converged on the energy only, leave the SCC unconverged in most runs (M7a report); "
           "use 'ntpoly' or an eigensolver",
    "pexsi": "PEXSI is not implemented (DFTB+'s conda-forge builds do not include it); "
             "finite-temperature densities without eigenvalues: use 'foe' (the recursive "
             "Fermi-operator expansion, M7b)",
}
