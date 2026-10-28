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
    """Array namespace of the inputs: the numpy module, or NumPy names over torch
    (:mod:`pydftb.backend.torch_xp`) if any input is a torch tensor.  Decided from the
    types alone, so NumPy-only code never imports torch."""
    for a in arrays:
        if type(a).__module__.startswith("torch"):
            from . import torch_xp
            return torch_xp
    return np


def to_host(x):
    """A torch tensor as a NumPy array (float64/complex128); anything else unchanged."""
    if _is_torch(x):
        import torch
        t = x.detach().resolve_conj()
        return t.to(torch.complex128 if t.is_complex() else torch.float64).cpu().numpy()
    return x


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def _real_dtype(t):
    import torch
    return {torch.complex128: torch.float64, torch.complex64: torch.float32}.get(t.dtype, t.dtype)


# ---- array creation that follows an existing array (milestone M4) -------------------------
# NumPy branches make exactly the calls the NumPy code made before.
def zeros(shape, like, dtype=None):
    """Zeros with the dtype (or ``dtype``) and device of ``like``."""
    if _is_torch(like):
        import torch
        return torch.zeros(shape, dtype=like.dtype if dtype is None else dtype,
                           device=like.device)
    return np.zeros(shape, dtype=like.dtype if dtype is None else dtype)


def ones(shape, like):
    if _is_torch(like):
        import torch
        return torch.ones(shape, dtype=like.dtype, device=like.device)
    return np.ones(shape, dtype=like.dtype)


def asarray(a, like):
    """A constant (NumPy array or number) next to ``like``: unchanged for NumPy, a tensor in
    the real dtype of ``like`` on its device for torch."""
    if _is_torch(like) and a is not None:
        import torch
        if _is_torch(a):
            return a.to(device=like.device)
        arr = np.asarray(a)
        dt = _real_dtype(like) if arr.dtype.kind == "f" else None
        return torch.as_tensor(arr, dtype=dt, device=like.device)
    return a


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
        # float64 as bincount gives it; with no values at all bincount returns int zeros
        if values.ndim == 1:
            return np.bincount(segment_ids, values, minlength=num_segments).astype(
                np.float64, copy=False)
        if values.ndim == 2:   # bincount per column is much faster than add.at
            return np.stack([np.bincount(segment_ids, values[:, k], minlength=num_segments)
                             .astype(np.float64, copy=False)
                             for k in range(values.shape[1])], axis=1)
    out = np.zeros((num_segments,) + values.shape[1:], dtype=values.dtype)
    np.add.at(out, segment_ids, values)
    return out


def eigh(a: np.ndarray, b: np.ndarray | None = None):
    """Hermitian (generalised if ``b`` is given) eigenproblem, ascending."""
    if b is None:
        return np.linalg.eigh(a)
    return sla.eigh(a, b, driver="gvd")


def eigh_driver(a: np.ndarray, b: np.ndarray, driver: str):
    """The generalised eigenproblem with DFTB+'s LAPACK drivers (M7a): "gv" (QR), "gvd"
    (DivideAndConquer, what :func:`eigh` uses) or "gvr" (RelativelyRobust: DFTB+'s hegvr,
    Cholesky, hegst, heevr with abstol = the safe minimum, triangular back-substitution)."""
    if driver in ("gv", "gvd"):
        return sla.eigh(a, b, driver=driver)
    if driver != "gvr":
        raise ValueError(f"unknown LAPACK driver {driver!r}: 'gv', 'gvd' or 'gvr'")
    cplx = np.iscomplexobj(a) or np.iscomplexobj(b)
    dt = np.complex128 if cplx else np.float64
    a = np.asarray(a, dt)
    b = np.asarray(b, dt)
    prefix = "z" if cplx else "d"
    potrf, = sla.get_lapack_funcs(("potrf",), (b,))
    hegst = getattr(sla.lapack, prefix + ("hegst" if cplx else "sygst"))
    heevr = getattr(sla.lapack, prefix + ("heevr" if cplx else "syevr"))
    u, info = potrf(b, lower=0, clean=1)               # S = U^H U (DFTB+: uplo "L" of the
    if info:                                          # transposed storage, the same factor)
        raise np.linalg.LinAlgError("the overlap matrix is not positive definite")
    c, info = hegst(a, u, itype=1, lower=0)
    if info:
        raise np.linalg.LinAlgError(f"{prefix}hegst failed (info {info})")
    c = np.triu(c) + np.triu(c, 1).conj().T            # hegst fills one triangle
    safe_min = sla.lapack.dlamch("S")
    w, v, m, _, info = heevr(c, compute_v=1, range="A", lower=0, abstol=safe_min)
    if info:
        raise np.linalg.LinAlgError(f"{prefix}{'heevr' if cplx else 'syevr'} failed (info {info})")
    return w[:m], sla.solve_triangular(u, v[:, :m], lower=False)


def batched_eigh(a: np.ndarray):
    """Standard Hermitian eigenproblem for a stack of matrices (..., n, n)."""
    return np.linalg.eigh(a)


def cholesky(a: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor(s), batched over leading axes."""
    return np.linalg.cholesky(a)


def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``a x = b`` (batched over leading axes)."""
    return np.linalg.solve(a, b)
