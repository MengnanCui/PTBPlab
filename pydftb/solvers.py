"""Electronic-structure solvers: H(k), S(k) -> density matrices.

Every solver returns, for each k-point, the density matrix rho(k), the
energy-weighted density matrix W(k), the eigenvalues (or None), the Fermi
level and the entropic term T*S.

* ``EighSolver``     reference: LAPACK generalised eigensolver, one k at a time
* ``BatchedSolver``  Cholesky once per geometry, batched eigh over all k
* ``SP2Solver``      diagonalisation-free second-order spectral projection
                     (Niklasson 2002).  Only GEMMs => GPU / tensor-core
                     friendly; optional mixed precision.  Gamma, T = 0, gapped.
"""
from __future__ import annotations

import numpy as np

from .backend import ops

from .occupations import fill


def _density(evals, evecs, kw, nelec, smearing, need_w, spin_factor=2.0):
    occ, mu, ts = fill(evals, kw, nelec, smearing, spin_factor=spin_factor)
    occ = np.array(occ)
    cf = evecs * occ[:, None, :]
    rho = cf @ np.conj(np.swapaxes(evecs, 1, 2))
    wdm = (cf * evals[:, None, :]) @ np.conj(np.swapaxes(evecs, 1, 2)) if need_w else None
    return rho, wdm, evals, mu, ts, occ


class EighSolver:
    name = "eigh"

    def prepare(self, s):
        self.s = s

    def eig(self, h):
        """Eigenvalues and S-orthonormal eigenvectors for every k."""
        ev, vec = [], []
        for hk, sk in zip(h, self.s):
            e, c = ops.eigh(hk, sk)
            ev.append(e)
            vec.append(c)
        return np.array(ev), np.array(vec)

    def solve(self, h, kw, nelec, smearing, need_w=True):
        *out, self.last_occupations = _density(*self.eig(h), kw, nelec, smearing, need_w)
        return tuple(out)


class BatchedSolver:
    """Reduce to standard form once per geometry (S is fixed in the SCC loop)."""
    name = "batched"

    def prepare(self, s):
        l = ops.cholesky(s)
        eye = np.broadcast_to(np.eye(s.shape[-1], dtype=s.dtype), s.shape)
        self.linv = ops.solve(l, eye)                       # L^-1
        self.linv_h = np.conj(np.swapaxes(self.linv, 1, 2))

    def eig(self, h):
        """Eigenvalues and S-orthonormal eigenvectors for every k."""
        e, c = ops.batched_eigh(self.linv @ h @ self.linv_h)
        return e, self.linv_h @ c

    def solve(self, h, kw, nelec, smearing, need_w=True):
        *out, self.last_occupations = _density(*self.eig(h), kw, nelec, smearing, need_w)
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

    def __init__(self, dtype=np.float64, refine_fp64: int = 0):
        self.dtype = dtype
        self.refine = refine_fp64
        self.iterations = 0

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


SOLVERS = {"eigh": EighSolver, "batched": BatchedSolver, "sp2": SP2Solver}
