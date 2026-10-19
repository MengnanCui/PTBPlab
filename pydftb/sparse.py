"""Linear-scaling (sparse) solvers: Newton-Schulz S^-1/2 and thresholded SP2.

Prototype on scipy.sparse (single-threaded CSR).  In production these map
onto blocked-sparse libraries (DBCSR / BML / NTPoly) or GPU SpGEMM.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _drop(m: sp.csr_matrix, thresh: float) -> sp.csr_matrix:
    m = m.tocsr()
    m.data[np.abs(m.data) < thresh] = 0.0
    m.eliminate_zeros()
    return m


def to_csr(pb, values, norb, diag):
    m = sp.coo_matrix((values, (pb.rows, pb.cols)), shape=(norb, norb)).tocsr()
    return (m + sp.diags(diag)).tocsr()


def inv_sqrt_newton_schulz(s: sp.csr_matrix, thresh=1e-7, tol=1e-6, max_iter=50):
    """Coupled Newton-Schulz iteration for S^-1/2 (Higham).

    Stops when ||T - 1|| < tol or stagnates at the truncation floor.
    """
    n = s.shape[0]
    err_old = np.inf
    c = (abs(s).sum(axis=1).A1).max()            # Gershgorin bound on lambda_max
    y = (s / c).tocsr()
    z = sp.identity(n, format="csr")
    eye = sp.identity(n, format="csr")
    for it in range(max_iter):
        t = (1.5 * eye - 0.5 * (z @ y)).tocsr()
        y = _drop(y @ t, thresh)
        z = _drop(t @ z, thresh)
        err = abs(t - eye).max()
        if err < tol or (it > 3 and err > 0.5 * err_old):
            break
        err_old = err
    return (z / np.sqrt(c)).tocsr(), it + 1


def sp2_sparse(hp: sp.csr_matrix, nocc: float, thresh=1e-5, max_iter=100):
    n = hp.shape[0]
    d = hp.diagonal()
    r = abs(hp).sum(axis=1).A1 - np.abs(d)
    emin, emax = (d - r).min(), (d + r).max()
    x = ((emax * sp.identity(n) - hp) / (emax - emin)).tocsr()
    idem_old = np.inf
    nit = 0
    for nit in range(1, max_iter + 1):
        x2 = _drop(x @ x, thresh)
        tr_x, tr_x2 = x.diagonal().sum(), x2.diagonal().sum()
        idem = abs(tr_x - tr_x2)
        if nit > 5 and idem >= idem_old * 0.9 and idem < 1e-2 * np.sqrt(n):
            break
        idem_old = idem
        if abs(tr_x2 - nocc) < abs(2 * tr_x - tr_x2 - nocc):
            x = x2
        else:
            x = (2 * x - x2).tocsr()
    return x, nit
