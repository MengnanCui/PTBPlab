"""Thin layer over the array operations that differ between backends.

All physics modules call the few non-portable operations through here
(scatter/segment sums, dense eigensolvers, factorisations).  Today only the
NumPy/SciPy implementation exists; the PyTorch backend (milestone M4) will
provide the same functions, so porting a physics module is mostly a matter
of swapping the array namespace instead of rewriting it.

Portable, standard operations (``@``, ``einsum``, ``exp`` ...) are used
directly from the array namespace and do not belong here.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def namespace(*arrays):
    """Array namespace of the inputs (NumPy only for now)."""
    return np


def scatter_add(target: np.ndarray, index, values) -> np.ndarray:
    """Return ``target`` with ``values`` added at ``index`` (repeats accumulate).

    Functional: the input is not modified (as required by autodiff backends).
    ``index`` may be an integer array or a tuple of integer arrays (one per
    leading axis of ``target``), exactly like ``np.add.at``.
    """
    out = np.array(target, copy=True)
    np.add.at(out, index, values)
    return out


def segment_sum(values: np.ndarray, segment_ids: np.ndarray, num_segments: int) -> np.ndarray:
    """Sum ``values`` (n, ...) into ``num_segments`` bins along axis 0."""
    values = np.asarray(values)
    if not np.iscomplexobj(values):
        if values.ndim == 1:
            return np.bincount(segment_ids, values, minlength=num_segments)
        if values.ndim == 2:   # bincount per column is much faster than add.at
            return np.stack([np.bincount(segment_ids, values[:, k], minlength=num_segments)
                             for k in range(values.shape[1])], axis=1)
    out = np.zeros((num_segments,) + values.shape[1:], dtype=values.dtype)
    np.add.at(out, segment_ids, values)
    return out


def eigh(a: np.ndarray, b: np.ndarray | None = None):
    """Hermitian (generalised if ``b`` is given) eigenproblem, ascending."""
    if b is None:
        return np.linalg.eigh(a)
    return sla.eigh(a, b, driver="gvd")


def batched_eigh(a: np.ndarray):
    """Standard Hermitian eigenproblem for a stack of matrices (..., n, n)."""
    return np.linalg.eigh(a)


def cholesky(a: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor(s), batched over leading axes."""
    return np.linalg.cholesky(a)


def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``a x = b`` (batched over leading axes)."""
    return np.linalg.solve(a, b)
