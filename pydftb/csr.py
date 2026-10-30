"""Sparse matrices for the linear-scaling purification (milestone M7c, DFTB+'s ``Solver = NTPoly
{ Sparse = Yes }``).

* SciPy CSR matrices with NTPoly's flushing: after every product the elements with
  |a| < threshold are dropped (pydftb's dense ``_drop`` rule).
* Products (:func:`matmul`): when both factors are filled above ``dense_fill`` (NTPoly's
  ``sparsity_threshold`` 0.1) a dense GEMM, otherwise SciPy's CSR product with the rows of the
  left factor in panels on a thread pool (SciPy releases the GIL; every row is computed as in
  the serial product, so the result is the same to the bit).
* The array helpers of :mod:`pydftb.purification` for CSR matrices: identity, conjugate
  transpose, trace, tr(A^H B), column sums of |A|, Gershgorin bounds, equality.
* :class:`Pattern`: the CSR structure of the H0/S neighbour blocks (and the diagonal), the
  images of a periodic system folded onto the central cell, as DFTB+ hands its packed matrices
  to ELSI.  The calculator keeps H0, S, H, rho and W as values on it (one row per k-point or
  spin channel): folding the pair elements, CSR matrices that share its indices, the values of
  a product at its places, row sums (Mulliken), the values at the pair elements (forces).
"""
from __future__ import annotations

import contextvars
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp

DENSE_FILL = 0.1                 # NTPoly GemmMatrix: both factors filled above it -> dense GEMM
PANELS_PER_THREAD = 4            # row panels per thread of a CSR product
SMALL_PRODUCT = 20000            # nnz of the left factor below it: one serial product

_THREADS = contextvars.ContextVar("pydftb_sparse_threads", default=None)
_DENSE_FILL = contextvars.ContextVar("pydftb_sparse_dense_fill", default=DENSE_FILL)
_POOLS: dict = {}
_POOL_LOCK = threading.Lock()


def is_sparse(a) -> bool:
    return sp.issparse(a)


def default_threads() -> int:
    """The cores this process may use."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                       # not Linux
        return os.cpu_count() or 1


class options:
    """Context manager: the threads of the CSR products (None: :func:`default_threads`) and the
    fill above which both factors are multiplied dense (1 or more: always sparse)."""

    def __init__(self, threads: int | None = None, dense_fill: float = DENSE_FILL):
        self.threads, self.dense_fill = threads, float(dense_fill)

    def __enter__(self):
        self._tokens = (_THREADS.set(self.threads), _DENSE_FILL.set(self.dense_fill))
        return self

    def __exit__(self, *exc):
        _THREADS.reset(self._tokens[0])
        _DENSE_FILL.reset(self._tokens[1])


def _pool(threads: int) -> ThreadPoolExecutor:
    with _POOL_LOCK:
        pool = _POOLS.get(threads)
        if pool is None:
            pool = _POOLS[threads] = ThreadPoolExecutor(threads,
                                                        thread_name_prefix="pydftb-sparse")
        return pool


def fill(a) -> float:
    """The fraction of stored elements."""
    n, m = a.shape
    return a.nnz / (n * m) if n and m else 0.0


def _flush(c, threshold: float):
    """Drop |c| < threshold in place (and the explicit zeros) of a fresh CSR matrix and sort its
    column indices (what SciPy's arithmetic needs)."""
    if threshold > 0.0:
        c.data[np.abs(c.data) < threshold] = 0.0
    c.eliminate_zeros()
    c.sort_indices()
    return c


def _panel(a, b, lo: int, hi: int, threshold: float):
    """Rows lo:hi of A B, flushed (in a worker thread: the untruncated product is wider than the
    result, so flushing and sorting there keep the serial part small)."""
    return _flush(a[lo:hi] @ b, threshold)


def drop(a, threshold: float):
    """NTPoly's flushing: a copy of ``a`` without the elements |a| < threshold."""
    a = a.tocsr()
    if threshold <= 0.0:
        return a
    small = np.abs(a.data) < threshold
    if not small.any():
        return a
    a = a.copy()
    a.data[small] = 0.0
    a.eliminate_zeros()
    return a


def matmul(a, b, threshold: float):
    """A B, flushed at ``threshold``: a dense GEMM when both factors are filled above the dense
    fill (NTPoly), else the CSR product in row panels on the thread pool."""
    if min(fill(a), fill(b)) > _DENSE_FILL.get():
        c = a.toarray() @ b.toarray()
        if threshold > 0.0:
            c[np.abs(c) < threshold] = 0.0
        return sp.csr_matrix(c)
    a, b = a.tocsr(), b.tocsr()
    threads = _THREADS.get() or default_threads()
    n = a.shape[0]
    if threads <= 1 or a.nnz < SMALL_PRODUCT or n < PANELS_PER_THREAD * threads:
        return _flush((a @ b).tocsr(), threshold)
    cuts = np.linspace(0, n, PANELS_PER_THREAD * threads + 1).astype(int)
    parts = list(_pool(threads).map(lambda k: _panel(a, b, cuts[k], cuts[k + 1], threshold),
                                    range(len(cuts) - 1)))
    c = sp.vstack(parts, format="csr")
    c.has_sorted_indices = True                  # every panel sorted, stacked by rows
    return c


def eye(n: int, dtype):
    return sp.identity(n, dtype=dtype, format="csr")


def ct(a):
    """Conjugate transpose (CSR)."""
    return (a.conj() if np.iscomplexobj(a.data) else a).T.tocsr()


def trace(a) -> float:
    return float(a.diagonal().sum().real)


def dot(a, b) -> float:
    """Re tr(A^H B) = Re sum(conj(a) b), NTPoly's DotMatrix."""
    if np.iscomplexobj(a.data):
        a = a.conj()
    return float(a.multiply(b).sum().real)


def abs_column_sums(a) -> np.ndarray:
    return np.asarray(abs(a).sum(axis=0)).ravel()


def gershgorin(h) -> tuple[float, float]:
    """Gershgorin bounds of a Hermitian CSR matrix (NTPoly GershgorinBounds)."""
    d = h.diagonal()
    r = abs_column_sums(h) - np.abs(d)
    return float((d.real - r).min()), float((d.real + r).max())


def equal(a, b) -> bool:
    return a.shape == b.shape and (a != b).nnz == 0


# ---- the pattern of the H0/S neighbour blocks --------------------------------------------------
class Pattern:
    """The CSR structure of the pair elements of ``pb`` and the diagonal (``norb`` orbitals), the
    images folded onto the central cell: ``indptr``, ``indices``, the row and column of every
    entry (``rows``, ``cols``), the entry of every pair element (``elem``) and of its transpose
    (``elem_t``), the diagonal entries (``diag``)."""

    def __init__(self, pb, norb: int):
        n = int(norb)
        rows = np.concatenate([pb.rows, np.arange(n)]).astype(np.int64)
        cols = np.concatenate([pb.cols, np.arange(n)]).astype(np.int64)
        keys, inv = np.unique(rows * n + cols, return_inverse=True)
        inv = inv.ravel()
        self.n = n
        self.nnz = len(keys)
        self.rows = keys // n
        self.cols = keys % n
        self.indices = self.cols.astype(np.int32)
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(self.rows, minlength=n))]
                                     ).astype(np.int32)
        ne = len(pb.rows)
        self.elem = inv[:ne]
        self.diag = inv[ne:]
        self.elem_t = np.searchsorted(keys, cols[:ne] * n + rows[:ne])
        if ne and not np.array_equal(keys[self.elem_t], cols[:ne] * n + rows[:ne]):
            raise ValueError("the pair elements are not symmetric (an ordered pair without its "
                             "partner)")
        self.shift = pb.shift
        self.pair_of = pb.pair_of

    def phases(self, kpt) -> np.ndarray:
        """e^(2 pi i k.T) of every pair element at the fractional k-point ``kpt``."""
        return np.exp(2j * np.pi * (self.shift @ np.asarray(kpt, float)))[self.pair_of]

    def fold(self, values, diag=None, kpts=None) -> np.ndarray:
        """sum_T M(T) e^(2 pi i k.T) of the pair element ``values`` (plus ``diag`` on the
        diagonal) at every k-point: (1, nnz) real at the Gamma point (``kpts`` None),
        (nk, nnz) complex otherwise."""
        if kpts is None:
            out = np.bincount(self.elem, weights=values, minlength=self.nnz)[None]
        else:
            out = np.empty((len(kpts), self.nnz), complex)
            for i, k in enumerate(kpts):
                v = values * self.phases(k)
                out[i] = (np.bincount(self.elem, weights=v.real, minlength=self.nnz)
                          + 1j * np.bincount(self.elem, weights=v.imag, minlength=self.nnz))
        if diag is not None:
            out[:, self.diag] += diag
        return out

    def matrix(self, data):
        """The CSR matrix of values ``data`` (nnz,) on the pattern."""
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))

    def sample(self, m) -> np.ndarray:
        """The values of the CSR matrix ``m`` at the pattern's entries (0 where it has none)."""
        m = m.tocsr()
        m.sort_indices()
        return np.asarray(m[self.rows, self.cols]).ravel()

    def row_sums(self, x) -> np.ndarray:
        """sum over the entries of every row of the values ``x`` (nnz,)."""
        return np.bincount(self.rows, weights=x, minlength=self.n)

    def at_pairs(self, data, weights, kpts=None) -> np.ndarray:
        """Re sum_c w_c M_c[nu, mu] e^(2 pi i k_c.T) at every pair element (mu, nu, T), the
        real-space value of the channels ``data`` (nc, nnz) with weights (nc,) and k-points
        (nc, 3) (None: the Gamma point)."""
        out = np.zeros(len(self.elem_t))
        for c, (row, w) in enumerate(zip(data, weights)):
            v = row[self.elem_t]
            if kpts is not None:
                v = (v * self.phases(kpts[c])).real
            out += w * v.real
        return out
