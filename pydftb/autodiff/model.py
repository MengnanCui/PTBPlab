"""Geometry- and parameter-dependent quantities in torch (milestone M4b).

Recomputed from :class:`~pydftb.autodiff.params.Params` and the atomic positions (and
cell) with the topology of a calculation that the NumPy engine already did: pair lists,
charge sites, cut-offs, the Ewald alpha and the integer G vectors are host data and treated
as constants; everything else uses the same kernels as the engine (``pair_values``,
``_short``, ``dgamma_du`` ...), so autograd differentiates the engine's own formulas.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..coulomb import _gamma0, _short
from ..hamiltonian import pair_values
from ..thirdorder import dgamma_du, dgamma_du_onsite


def deform(positions, cell, strain):
    """Positions and cell under the strain ``strain`` (r -> (1 + strain) r); None: none."""
    if strain is None:
        return positions, cell
    f = torch.eye(3, dtype=positions.dtype, device=positions.device) + strain
    return positions @ f.T, (None if cell is None else cell @ f.T)


def bond_vectors(pos, cell, i, j, shift):
    vec = pos[j] - pos[i]
    if cell is not None:
        vec = vec + torch.as_tensor(np.asarray(shift, dtype=float), dtype=pos.dtype,
                                    device=pos.device) @ cell
    return vec


def norm(vec):
    return torch.sqrt((vec * vec).sum(-1))


# ---- one-centre parameters -----------------------------------------------------------------
def onsite_vector(basis, params):
    """On-site energy of every orbital."""
    parts = []
    for lab, shells in zip(basis.species, basis.shells):
        ls = np.repeat(shells, [2 * l + 1 for l in shells])
        parts.append(params.onsite[lab][ls])
    return torch.cat(parts)


def site_hubbard(basis, params, shell_resolved: bool):
    """Hubbard U of every charge site (shells, or atoms with the s-shell value)."""
    if shell_resolved:
        return torch.stack([params.hubbard[basis.species[a]][l]
                            for a, l in zip(basis.shell_atom, basis.shell_l)])
    return torch.stack([params.hubbard[lab][0] for lab in basis.species])


def site_hubbard_derivs(basis, params, sk):
    """Hubbard derivative of every site (one value per type, or per shell)."""
    ud = params.hubbard_derivs
    if not sk.shell_resolved:
        return torch.stack([ud[lab][0] for lab in basis.species])
    out = []
    for a, l in zip(basis.shell_atom, basis.shell_l):
        v = ud[basis.species[a]]
        out.append(v[0] if len(v) == 1 else v[sk.shells(basis.species[a]).index(l)])
    return torch.stack(out)


def spin_matrix(basis, params, like):
    """Spin constants W over all shells (block-diagonal per atom), as the engine."""
    nsh = len(basis.shell_l)
    w = torch.zeros((nsh, nsh), dtype=like.dtype, device=like.device)
    for a, lab in enumerate(basis.species):
        idx = np.nonzero(basis.shell_atom == a)[0]
        v = params.spin_constants[lab]
        blk = v.reshape(-1)[0].expand(len(idx), len(idx)) if v.numel() == 1 else \
            v.reshape(len(idx), len(idx))
        w = w.index_put((torch.as_tensor(np.repeat(idx, len(idx))),
                         torch.as_tensor(np.tile(idx, len(idx)))), blk.reshape(-1))
    return w


# ---- two-centre terms ------------------------------------------------------------------------
def pair_terms(st, sk, params, pos, cell, fd_delta=None):
    """H0/S of every pair element (and DFTB+'s finite differences with ``fd_delta``)."""
    pb = st.pairs
    vec = bond_vectors(pos, cell, pb.i, pb.j, pb.shift)
    return pair_values(pb, sk, vec, tables=params.tables, fd_delta=fd_delta)


def _damping(coul, i, j, sp_pair, sp_a, sp_b, tau, d_sp):
    """H-X damping factor of every site pair (1 where no damped atom is involved)."""
    fac = torch.ones_like(d_sp)
    if not coul.damped.any():
        return fac
    m = (coul.damped[i] | coul.damped[j])[sp_pair]
    if not m.any():
        return fac
    u = (tau[sp_a[m]] + tau[sp_b[m]]) / 6.4
    r = d_sp[m]
    return fac.index_put((torch.as_tensor(np.nonzero(m)[0]),), torch.exp(-u ** coul.damp_exp * r * r))


def gamma_matrix(coul, pos, cell, tau):
    """The site gamma matrix of :class:`pydftb.coulomb.Coulomb` (same topology)."""
    sa = coul.site_atom
    same_atom = torch.as_tensor(sa[:, None] == sa[None, :], device=pos.device)
    g = torch.where(same_atom, _gamma0(tau[:, None], tau[None, :]), torch.zeros((), dtype=pos.dtype))
    i, j, shift, _, d_host = coul.nl
    vec = bond_vectors(pos, cell if coul.periodic else None, i, j, shift)
    d = norm(vec)
    d_sp = d[coul.sp_pair]
    s, _ = _short(d_sp, tau[coul.sp_a], tau[coul.sp_b])
    s = s * _damping(coul, i, j, coul.sp_pair, coul.sp_a, coul.sp_b, tau, d_sp)
    g = g.index_put((torch.as_tensor(coul.sp_a), torch.as_tensor(coul.sp_b)), -s, accumulate=True)
    n = coul.natom
    eye = torch.eye(n, dtype=pos.dtype, device=pos.device)
    if not coul.periodic:
        diff = pos[None, :, :] - pos[:, None, :]
        dist = torch.sqrt((diff * diff).sum(-1) + eye)          # 1 on the diagonal: no 0/0
        return g + ((1.0 - eye) / dist)[sa][:, sa]
    a = coul.alpha
    m = d_host < coul.rcut_ewald
    real = torch.zeros((n, n), dtype=pos.dtype, device=pos.device).index_put(
        (torch.as_tensor(i[m]), torch.as_tensor(j[m])),
        torch.special.erfc(a * d[m]) / d[m], accumulate=True)
    rec_basis = 2.0 * math.pi * torch.linalg.inv(cell).T
    gvec = torch.as_tensor(coul.gint, dtype=pos.dtype, device=pos.device) @ rec_basis
    vol = torch.abs(torch.linalg.det(cell))
    g2 = (gvec * gvec).sum(1)
    gfac = 4.0 * math.pi / vol * torch.exp(-g2 / (4 * a * a)) / g2
    phase = pos @ gvec.T
    cg, sg = torch.cos(phase), torch.sin(phase)
    rec = (cg * gfac) @ cg.T + (sg * gfac) @ sg.T
    atom_mat = real + rec - math.pi / (vol * a * a) - eye * (2.0 * a / math.sqrt(math.pi))
    return g + atom_mat[sa][:, sa]


def third_matrix(third, coul, pos, cell, u, ud):
    """Gamma of :class:`pydftb.thirdorder.ThirdOrder` (same topology)."""
    sites = coul.atom_sites
    pa = np.repeat(sites, sites.shape[1], axis=1).ravel()
    pb = np.tile(sites, (1, sites.shape[1])).ravel()
    ok = (pa >= 0) & (pb >= 0)
    pa, pb = pa[ok], pb[ok]
    mat = torch.zeros((third.nsite, third.nsite), dtype=pos.dtype, device=pos.device)
    mat = mat.index_put((torch.as_tensor(pa), torch.as_tensor(pb)),
                        dgamma_du_onsite(u[pa], u[pb]) * ud[pa])
    if third.nl is not None:
        i, j, shift = third.nl[0], third.nl[1], third.nl[2]
        d = norm(bond_vectors(pos, cell if third.periodic else None, i, j, shift))
        damped = (coul.damped[i] | coul.damped[j])[third.sp_pair]
        g_ab, _ = dgamma_du(u[third.sp_a], u[third.sp_b], d[third.sp_pair], damped,
                            coul.damp_exp)
        mat = mat.index_put((torch.as_tensor(third.sp_a), torch.as_tensor(third.sp_b)),
                            g_ab * ud[third.sp_a], accumulate=True)
    return mat


def third_energy(mat, dq, site_atom, natom):
    """E3 = 1/3 sum_ab Gamma_ab dQ_A(a) dq_a dq_b (the engine's ThirdOrder.energy_shift)."""
    dq_atom = torch.zeros(natom, dtype=dq.dtype, device=dq.device).index_add(
        0, torch.as_tensor(site_atom), dq)
    qa = dq_atom[site_atom]
    return (qa * dq * (mat @ dq)).sum() / 3.0


def repulsive_energy(engine, st, params, pos, cell):
    """Pair repulsive energy with the coefficients of ``params`` (the engine's pairs)."""
    nl, _ = engine._repulsive_pairs(st)
    e = torch.zeros((), dtype=pos.dtype, device=pos.device)
    if nl is None:
        return e
    i, j, shift = nl[0], nl[1], nl[2]
    d = norm(bond_vectors(pos, cell, i, j, shift))
    spec = np.array(st.species)
    for a in sorted(set(st.species)):
        for b in sorted(set(st.species)):
            rep = engine.sk.repulsive(a, b, engine.repulsive)
            m = (spec[i] == a) & (spec[j] == b)
            if rep is None or not m.any():
                continue
            e = e + 0.5 * rep.energy(d[m], **params.repulsive.get((a, b), {})).sum()
    return e
