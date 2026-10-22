"""Generalised eigensolver and density matrices on a torch device (milestone M4a).

The algorithm of :class:`pydftb.solvers.BatchedSolver` and of LAPACK's sygvd (SciPy
``driver="gvd"``): the Cholesky factor S = L L^H once per geometry, the standard problem
L^-1 H L^-H for every k, ``eigh``, and the back-transformation C = L^-H Y.  L^-1 comes
from a triangular solve; batched LU on the CPU is avoided (pytorch#142815).  The factor
and L^-1 are computed in double precision also in float32 mode and only then cast.
Occupations and the Fermi level are found on the host with DFTB+'s rules
(:func:`pydftb.occupations.fill`).
"""
from __future__ import annotations

import numpy as np
import torch

from ..solvers import _density
from .spec import Backend


class TorchSolver:
    name = "torch"

    def __init__(self, backend: Backend, threads=None):
        self.backend = backend
        self.threads = None                  # no k-point threads on torch

    def prepare(self, s):
        """``s``: the overlap stack in double precision on the device."""
        l, info = torch.linalg.cholesky_ex(s)
        if int(info.max()) > 0:
            raise np.linalg.LinAlgError("the overlap matrix is not positive definite")
        eye = torch.eye(s.shape[-1], dtype=s.dtype, device=s.device).expand(s.shape)
        linv = self.backend.cast(torch.linalg.solve_triangular(l, eye, upper=False))
        self.linv = linv
        self.linv_h = linv.mH

    def eig(self, h):
        """Eigenvalues and S-orthonormal eigenvectors for every k (on the device)."""
        e, y = torch.linalg.eigh(self.linv @ h @ self.linv_h)
        return e, self.linv_h @ y

    def solve(self, h, kw, nelec, smearing, need_w=True):
        *out, self.last_occupations = _density(*self.eig(h), kw, nelec, smearing, need_w,
                                               backend=self.backend)
        return tuple(out)
