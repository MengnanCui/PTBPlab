"""The SCC map in torch and its implicit differentiation (milestone M4b).

:class:`SCCMap` is one SCC step without mixing, Phi(q, theta): mixed populations ->
potentials -> H(k) -> densities (:class:`~pydftb.autodiff.density.DensityMatrix`) ->
Mulliken populations, built from :class:`~pydftb.autodiff.params.Params` and the geometry
with the topology of a calculation the NumPy engine already converged.

At the engine's fixed point q* = Phi(q*, theta) the implicit function theorem gives
dq*/dtheta = (I - J)^-1 dPhi/dtheta with J = dPhi/dq.  :func:`attach` returns

    q_hat = q* + Phi(q*, theta) - stopgrad(Phi(q*, theta))       (numerically q*)

with a hook that turns the gradient g arriving at q_hat into y = (I - J^T)^-1 g, so that
autograd through q_hat yields the implicit derivative.  J^T comes from vector-Jacobian
products of Phi: densely for up to ``DENSE_MAX`` populations (PTBP has 2-16), else GMRES.
A fixed-point iteration would not do: for metals the spectral radius of J exceeds 1.
"""
from __future__ import annotations

import numpy as np
import torch

from ..hamiltonian import shift_index
from .density import DensityMatrix, Occupation
from .model import (gamma_matrix, onsite_vector, pair_terms, site_hubbard,
                    site_hubbard_derivs, spin_matrix, third_matrix)

DENSE_MAX = 64          # populations up to which J is built densely
GMRES_TOL = 1e-10       # relative residual of the adjoint equation
GMRES_MAXITER = 200
DENSE_FALLBACK = 512    # GMRES failed: dense solve up to this size


class Bloch:
    """Folding of pair values to the k-point matrices of a calculation, and back."""

    def __init__(self, pb, norb: int, kpts, gamma_only: bool):
        shifts, t_of = shift_index(pb)
        self.norb, self.gamma = norb, gamma_only
        self.rows, self.cols = torch.as_tensor(pb.rows), torch.as_tensor(pb.cols)
        self.t_of, self.nt, self.nk = torch.as_tensor(t_of), len(shifts), len(kpts)
        if not gamma_only:
            self.phase = torch.as_tensor(np.exp(2j * np.pi * (kpts @ shifts.T)))   # (nk, nT)

    def fold(self, values, diag):
        """M(k) = sum_T M(T) exp(2 pi i k.T) + diag, (nk, N, N)."""
        n = self.norb
        if self.gamma:
            m = values.new_zeros((n, n)).index_put((self.rows, self.cols), values,
                                                   accumulate=True)
            return (m + torch.diag(diag))[None]
        mt = values.new_zeros((self.nt, n, n)).index_put((self.t_of, self.rows, self.cols),
                                                         values, accumulate=True)
        m = (self.phase @ mt.reshape(self.nt, -1).to(self.phase.dtype)).reshape(self.nk, n, n)
        return m + torch.diag(diag).to(m.dtype)

    def pairs(self, mats, weights):
        """P_e = Re sum_k w_k M_k[col, row] exp(2 pi i k.T) at every pair element, for a stack
        of matrices (k-points, repeated per spin) with one weight each."""
        w = torch.as_tensor(np.asarray(weights, float), dtype=torch.float64)
        if self.gamma:
            return w @ mats[:, self.cols, self.rows].real
        n = self.norb
        ph = (self.phase.repeat(len(w) // self.nk, 1) * w[:, None]).T          # (nT, n_mat)
        r_t = (ph @ mats.reshape(len(w), -1).to(ph.dtype)).reshape(self.nt, n, n)
        return r_t[self.t_of, self.cols, self.rows].real


def _segment_sum(x, index, n):
    return x.new_zeros(n).index_add(0, torch.as_tensor(index), x)


class SCCMap:
    """Phi(q, theta) of the engine ``eng`` for its setup ``st`` (see the module docstring);
    ``pos``/``cell`` are torch tensors (Bohr), ``need_w`` adds the energy-weighted density."""

    def __init__(self, eng, st, params, pos, cell, need_w: bool = False):
        self.eng, self.st = eng, st
        b = st.basis
        self.norb, self.nk = b.norb, len(st.kw)
        self.bloch = Bloch(st.pairs, b.norb, st.kpts, st.gamma_only)
        h, s = pair_terms(st, eng.sk, params, pos, cell)
        # Hermitian parts (equal to the matrices to rounding): the two copies of an integral
        # that the files A-B and B-A both store then get equal shares of its gradient, as in
        # the surrogate of mode "hf" (eigh and Cholesky read one triangle, Mulliken sums
        # rho_mu,nu S_nu,mu into orbital mu only)
        h0 = self.bloch.fold(h, onsite_vector(b, params))
        smat = self.bloch.fold(s, h.new_ones(b.norb))
        self.h0, self.smat = 0.5 * (h0 + h0.mH), 0.5 * (smat + smat.mH)
        chol = torch.linalg.cholesky(self.smat)
        eye = torch.eye(b.norb, dtype=chol.dtype).expand_as(chol)
        self.linv = torch.linalg.solve_triangular(chol, eye, upper=False)
        self.gamma = self.third = self.wspin = None
        if eng.scc:
            u = site_hubbard(b, params, eng.sk.shell_resolved)
            self.gamma = gamma_matrix(st.coulomb, pos, cell, 3.2 * u)
            if st.third is not None:
                ud = site_hubbard_derivs(b, params, eng.sk)
                self.third = third_matrix(st.third, st.coulomb, pos, cell, u, ud)
        nspin = 1
        if st.spin is not None:
            self.wspin = spin_matrix(b, params, h)
            nspin = 2
        self.nspin = nspin
        self.kw_mat = np.tile(st.kw, nspin)              # weight of every stacked matrix
        nk = self.nk
        if st.spin is None:
            groups = [(np.arange(nk), st.nelec, 2.0)]
        elif st.spin.shared_ef:
            groups = [(np.arange(2 * nk), st.nelec, 1.0)]
        else:
            groups = [(np.arange(nk), st.spin.n_up, 1.0), (np.arange(nk, 2 * nk), st.spin.n_down, 1.0)]
        self.occ = Occupation(self.kw_mat, groups, eng.smearing, need_w)
        self.q0 = torch.as_tensor(st.q0, dtype=torch.float64)

    # ---- populations and potentials (the engine's _charge_and_magnetisation / _potentials)
    def charge_and_magnetisation(self, q):
        st = self.st
        if st.spin is None:
            return q - self.q0, None
        b = st.basis
        nsh = len(b.shell_l)
        up, down = q[:nsh], q[nsh:]
        qs = up + down
        if not self.eng.sk.shell_resolved:
            qs = _segment_sum(qs, b.shell_atom, len(st.species))
        return qs - self.q0, up - down

    def third_shift(self, dq):
        """E3 and dE3/dq (pydftb.thirdorder.ThirdOrder.energy_shift)."""
        th = self.st.third
        natom = th.natom
        dq_atom = _segment_sum(dq, th.site_atom, natom)
        qa = dq_atom[torch.as_tensor(th.site_atom)]
        gq = self.third @ dq
        e = (qa * dq * gq).sum() / 3.0
        v = (qa * gq + self.third.T @ (qa * dq)
             + _segment_sum(dq * gq, th.site_atom, natom)[torch.as_tensor(th.site_atom)]) / 3.0
        return e, v

    def potentials(self, q):
        """Charge potential per orbital and spin potential per orbital (or None)."""
        st = self.st
        if self.gamma is None:
            return q.new_zeros(self.norb), None
        dq, m = self.charge_and_magnetisation(q)
        v_site = self.gamma @ dq
        if self.third is not None:
            v_site = v_site + self.third_shift(dq)[1]
        vs = None if m is None else (self.wspin @ m)[torch.as_tensor(st.basis.orb_shell)]
        return v_site[torch.as_tensor(st.site_of_orb)], vs

    # ---- one SCC step
    def solve(self, q):
        """Densities of H(q): dict with rho, wdm (stacked [up k, down k] with spin), evals,
        mu (per Fermi-level group), q_orb (per orbital, summed over spins), q_out (the mixed
        vector, = Phi(q))."""
        st = self.st
        v, vs = self.potentials(q)
        vv = (v[:, None] + v[None, :]).to(self.smat.dtype)
        h = self.h0 + 0.5 * self.smat * vv
        linv = self.linv
        if vs is not None:
            hs = 0.5 * self.smat * (vs[:, None] + vs[None, :]).to(self.smat.dtype)
            h = torch.cat([h + hs, h - hs])
            linv = torch.cat([linv, linv])
        hp = linv @ h @ linv.mH
        rho_p, w_p, evals, mu = DensityMatrix.apply(hp, self.occ)
        rho = linv.mH @ rho_p @ linv
        wdm = None if w_p is None else linv.mH @ w_p @ linv
        smat = self.smat if vs is None else torch.cat([self.smat, self.smat])
        kw = torch.as_tensor(self.kw_mat, dtype=torch.float64)
        per = (rho * smat.mT).sum(-1).real                     # (rho S)_mu,mu per matrix
        b = st.basis
        if vs is None:
            q_orb = kw @ per
            q_out = _segment_sum(q_orb, st.site_of_orb, len(st.q0))
        else:
            nk, nsh = self.nk, len(b.shell_l)
            q_up, q_dn = kw[:nk] @ per[:nk], kw[nk:] @ per[nk:]
            q_orb = q_up + q_dn
            q_out = torch.cat([_segment_sum(q_up, b.orb_shell, nsh),
                               _segment_sum(q_dn, b.orb_shell, nsh)])
        return dict(rho=rho, wdm=wdm, evals=evals, mu=mu, q_orb=q_orb, q_out=q_out)

    def __call__(self, q):
        return self.solve(q)["q_out"]


class _Adjoint:
    """Hook on q_hat: y = (I - J^T)^-1 g with J^T v from the graph of Phi(q_c)."""

    def __init__(self, phi_out, q_c):
        self.out, self.q_c = phi_out, q_c
        self.a = None
        self.residual = None

    def jtv(self, v):
        (g,) = torch.autograd.grad(self.out, self.q_c, v, retain_graph=True)
        return g

    def dense(self):
        if self.a is None:
            n = self.q_c.numel()
            eye = torch.eye(n, dtype=torch.float64)
            jac = torch.stack([self.jtv(eye[j]) for j in range(n)])     # row j: dPhi_j/dq
            self.a = eye - jac.T
        return self.a

    def __call__(self, g):
        g = g.detach()
        if not torch.any(g):
            return g
        n = g.numel()
        if n <= DENSE_MAX:
            y = torch.linalg.solve(self.dense(), g)
        else:
            y, ok = gmres(lambda v: v - self.jtv(v), g, GMRES_TOL, GMRES_MAXITER)
            if not ok:
                if n > DENSE_FALLBACK:
                    raise RuntimeError("the adjoint SCC equation did not converge (GMRES)")
                y = torch.linalg.solve(self.dense(), g)
        r = g - (y - self.jtv(y))
        self.residual = float(r.norm() / g.norm())
        return y


def gmres(matvec, b, tol: float, maxiter: int):
    """Plain GMRES (no restart) for a small linear system given by its matrix-vector
    product; returns (x, converged)."""
    n = b.numel()
    beta = b.norm()
    q = [b / beta]
    h = torch.zeros((maxiter + 1, maxiter), dtype=b.dtype)
    x = torch.zeros_like(b)
    for k in range(min(maxiter, n)):
        w = matvec(q[k])
        for i in range(k + 1):                       # modified Gram-Schmidt
            h[i, k] = torch.dot(q[i], w)
            w = w - h[i, k] * q[i]
        h[k + 1, k] = w.norm()
        e1 = torch.zeros(k + 2, dtype=b.dtype)
        e1[0] = beta
        yk = torch.linalg.lstsq(h[:k + 2, :k + 1], e1[:, None]).solution[:, 0]
        res = (h[:k + 2, :k + 1] @ yk - e1).norm() / beta
        x = torch.stack(q[:k + 1], 1) @ yk
        if res < tol or h[k + 1, k] < 1e-300:
            return x, True
        q.append(w / h[k + 1, k])
    return x, False


def attach(phi: SCCMap, q_star):
    """q_hat (see the module docstring) and the hook's solver (its ``residual`` is the
    relative residual of the last adjoint solve)."""
    q_c = torch.as_tensor(np.asarray(q_star, float)).clone().requires_grad_(True)
    p1 = phi(q_c)
    q_hat = q_c.detach() + (p1 - p1.detach())
    adj = _Adjoint(p1, q_c)
    q_hat.register_hook(adj)
    return q_hat, adj


def unrolled(phi: SCCMap, q_start, mixing: float = 0.2, iterations: int = 300):
    """The fixed point by plain linear mixing with every step on the autograd graph: an
    independent reference for the implicit derivative on small systems (tests; the memory
    grows with the iterations, and metals may need many)."""
    q = torch.as_tensor(np.asarray(q_start, float))
    for _ in range(iterations):
        q = q + mixing * (phi(q) - q)
    return q
