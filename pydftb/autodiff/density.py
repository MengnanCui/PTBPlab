"""Density matrices of an orthogonal eigenproblem as an autograd Function (milestone M4b).

:class:`DensityMatrix` maps a stack of Hermitian matrices H_k (one per k-point and spin
channel, in an orthonormal basis) to

    rho_k = C_k f_k C_k^H,   W_k = C_k (f_k eps_k) C_k^H,   eps_k,   mu (one per group)

with the occupations f = f((eps - mu) / kT) of the engine's smearing and one Fermi level per
group of matrices that holds a fixed number of electrons.  The forward pass finds the
occupations on the host with :func:`pydftb.occupations.fill` (DFTB+'s rules), so the values
are the engine's.

The backward pass never differentiates eigenvectors one by one (that is singular for
degenerate levels): with X = C^H dH C,

    d rho = C (D o X) C^H - C diag(f') C^H d mu,    D_ij = (f_i - f_j) / (eps_i - eps_j),
    D_ii = f'_i (Daleckii-Krein), and d mu from the group's rule:
    fixed electron count   d mu = sum w f' X_ii / sum w f'
    DFTB+'s mid-gap rule    d mu = (X_HH + X_LL) / 2   (the two levels of the gap).

W uses the divided differences of g = f eps.  The divided differences of nearby levels are
taken in stable forms (Fermi: -f_a (1 - f_b) expm1(d) / d; Gaussian / Methfessel-Paxton:
the expansion about the mid-point), so degenerate levels and kT -> 0 stay finite.

A single eigenvalue of a degenerate level has no derivative (a perturbation splits the
level in a direction-dependent, non-linear way); the gradient of the eigenvalues is
averaged over each degenerate level (within DEGENERATE): the exact derivative of every
symmetric function of the level's eigenvalues, such as their sum or mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial, pi, sqrt

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from ..occupations import Smearing, fermi_rule, fill

NEAR_FERMI = 1.0     # |x_a - x_b| below which Fermi differences use the expm1 form
NEAR_POLY = 1e-3     # ... and Gaussian / Methfessel-Paxton ones the mid-point expansion
DEGENERATE = 1e-9    # Hartree: eigenvalues this close form one degenerate level


@dataclass
class Occupation:
    """What :class:`DensityMatrix` needs besides H: the weight of every matrix (its
    k-point weight), the groups of matrices that share a Fermi level -- (matrix indices,
    electrons, spin factor) -- the smearing, and whether W is wanted."""
    kw: np.ndarray
    groups: list
    smearing: Smearing
    need_w: bool = True


def _mp_weights(order: int) -> list:
    return [(-1) ** n / (factorial(n) * 4 ** n * sqrt(pi)) for n in range(order + 1)]


def _hermite(nmax: int, x):
    h = [torch.ones_like(x)]
    if nmax > 0:
        h.append(2.0 * x)
    for i in range(2, nmax + 1):
        h.append(2.0 * (x * h[i - 1] - (i - 1) * h[i - 2]))
    return h


def occupation(x, sm: Smearing):
    """f(x) without the spin factor (x = (eps - mu) / kT), as pydftb.occupations."""
    if sm.mp_order is None:
        return torch.sigmoid(-x)
    n = sm.mp_order
    a = _mp_weights(n)
    h = _hermite(2 * n, x)
    g = torch.exp(-x * x)
    occ = 0.5 * torch.special.erfc(x)
    for m in range(1, n + 1):
        occ = occ + a[m] * h[2 * m - 1] * g
    return occ


def _poly_derivs(x, sm: Smearing):
    """df/dx and d3f/dx3 of the Gaussian / Methfessel-Paxton occupation."""
    n = sm.mp_order
    a = _mp_weights(n)
    h = _hermite(2 * n + 2, x)
    g = torch.exp(-x * x)
    d1 = -sum(a[m] * h[2 * m] for m in range(n + 1)) * g
    d3 = -sum(a[m] * h[2 * m + 2] for m in range(n + 1)) * g
    return d1, d3


def entropy(x, sm: Smearing):
    """The entropy term s(x) of pydftb.occupations._electron_fill (TS = kT sum w s)."""
    if sm.mp_order is None:
        # -f ln f - (1 - f) ln(1 - f) with ln f = -softplus(x): no log(0), finite gradients
        f = torch.sigmoid(-x)
        s = f * torch.nn.functional.softplus(x) + torch.sigmoid(x) * torch.nn.functional.softplus(-x)
        eps = np.finfo(float).eps
        mid = (f > eps) & (f < 1.0 - eps)
        return torch.where(mid, s, torch.zeros_like(s))
    n = sm.mp_order
    a = _mp_weights(n)
    h = _hermite(2 * n, x)
    g = torch.exp(-x * x)
    terms = range(n + 1) if sm.ts_convention == "dftb+" else [n]
    return sum(0.5 * a[m] * h[2 * m] * g for m in terms)


def divided_differences(x, sm: Smearing):
    """(f(x_i) - f(x_j)) / (x_i - x_j) for every pair of one matrix's levels, f'(x_i) on
    the diagonal (no spin factor, per unit of x).  x: (..., n) -> (..., n, n)."""
    d = x[..., :, None] - x[..., None, :]
    safe = torch.where(d == 0, torch.ones_like(d), d)
    if sm.mp_order is None:
        f, fm = torch.sigmoid(-x), torch.sigmoid(x)
        ratio = torch.where(d.abs() < 1e-8, 1.0 + 0.5 * d, torch.expm1(d) / safe)
        near = -f[..., :, None] * fm[..., None, :] * ratio
        far = (f[..., :, None] - f[..., None, :]) / safe
        out = torch.where(d.abs() < NEAR_FERMI, near, far)
    else:
        f = occupation(x, sm)
        mid = 0.5 * (x[..., :, None] + x[..., None, :])
        d1, d3 = _poly_derivs(mid, sm)
        near = d1 + d3 * d * d / 24.0
        far = (f[..., :, None] - f[..., None, :]) / safe
        out = torch.where(d.abs() < NEAR_POLY, near, far)
    return 0.5 * (out + out.transpose(-1, -2))


def _level_mean(g, e):
    """``g`` averaged over every degenerate level of ``e`` (both (n, N), ``e`` ascending)."""
    new_level = torch.ones_like(e, dtype=torch.bool)
    new_level[:, 1:] = (e[:, 1:] - e[:, :-1]) > DEGENERATE
    label = torch.cumsum(new_level.to(torch.int64), dim=1) - 1            # level of each
    sums = g.new_zeros(e.shape).scatter_add(1, label, g)
    counts = g.new_zeros(e.shape).scatter_add(1, label, torch.ones_like(g))
    return (sums / counts.clamp(min=1)).gather(1, label)


class DensityMatrix(torch.autograd.Function):
    """``DensityMatrix.apply(h, occ)`` -> rho, W (None unless ``occ.need_w``), eigenvalues,
    Fermi levels (one per group).  ``h``: (n, N, N) Hermitian, real or complex."""

    @staticmethod
    def forward(ctx, h, occ: Occupation):
        e, c = torch.linalg.eigh(h)
        e_host = e.detach().cpu().numpy()
        f = np.zeros_like(e_host)
        mus, rules = [], []
        for idx, nelec, sf in occ.groups:
            o, mu, _ = fill(e_host[idx], occ.kw[idx], nelec, occ.smearing, spin_factor=sf)
            f[idx] = np.array(o)
            mus.append(mu)
            rules.append(fermi_rule(e_host[idx], occ.kw[idx], nelec, occ.smearing, sf))
        ft = torch.as_tensor(f, dtype=e.dtype, device=e.device)
        cf = c * ft.to(c.dtype)[:, None, :]
        rho = cf @ c.mH
        wdm = (cf * e.to(c.dtype)[:, None, :]) @ c.mH if occ.need_w else None
        mu = torch.as_tensor(mus, dtype=e.dtype, device=e.device)
        ctx.occ, ctx.rules = occ, rules
        ctx.save_for_backward(e, c, ft, mu)
        return rho, wdm, e, mu

    @staticmethod
    @once_differentiable
    def backward(ctx, g_rho, g_w, g_e, g_mu):
        e, c, f, mu = ctx.saved_tensors
        occ = ctx.occ
        sm = occ.smearing
        kt = sm.kt
        n, nlev = e.shape
        y = torch.zeros((n, nlev, nlev), dtype=c.dtype, device=c.device)

        def eig_basis(g):
            if g is None:
                return None
            m = c.mH @ g @ c
            return 0.5 * (m + m.mH)

        gr, gw = eig_basis(g_rho), eig_basis(g_w if occ.need_w else None)
        fp = torch.zeros_like(e)                 # df/d eps, with the spin factor
        for (idx, nelec, sf), m in zip(occ.groups, mu):
            ii = torch.as_tensor(idx, device=e.device)
            x = (e[ii] - m) / kt
            dd = divided_differences(x, sm) * (sf / kt)
            fp[ii] = torch.diagonal(dd, dim1=-2, dim2=-1)
            if gr is not None:
                y[ii] += dd.to(c.dtype) * gr[ii]
            if gw is not None:
                ei = e[ii]
                dw = (0.5 * (f[ii][:, :, None] + f[ii][:, None, :])
                      + 0.5 * (ei[:, :, None] + ei[:, None, :]) * dd)
                y[ii] += dw.to(c.dtype) * gw[ii]
        if g_e is not None:
            y += torch.diag_embed(_level_mean(g_e, e).to(c.dtype))
        # the Fermi levels: rho and W depend on mu (-f' and -f' eps), and mu on H
        for gi, ((idx, nelec, sf), rule) in enumerate(zip(occ.groups, ctx.rules)):
            ii = torch.as_tensor(idx, device=e.device)
            a = torch.zeros((), dtype=e.dtype, device=e.device)
            if gr is not None:
                a = a + (fp[ii] * torch.diagonal(gr[ii], dim1=-2, dim2=-1).real).sum()
            if gw is not None:
                a = a + (fp[ii] * e[ii] * torch.diagonal(gw[ii], dim1=-2, dim2=-1).real).sum()
            c_mu = -a + (g_mu[gi] if g_mu is not None else 0.0)
            if rule[0] == "count":
                w = torch.as_tensor(occ.kw[idx], dtype=e.dtype, device=e.device)[:, None] * fp[ii]
                z = w.sum()
                if z != 0:                     # f' < 0: z < 0 (z = 0: mu in a gap, no coupling)
                    y[ii] += torch.diag_embed((c_mu * w / z).to(c.dtype))
            elif rule[0] == "midgap":
                for flat in rule[1:]:
                    k, lev = divmod(flat, nlev)
                    y[idx[k], lev, lev] += 0.5 * c_mu
        return c @ y @ c.mH, None
