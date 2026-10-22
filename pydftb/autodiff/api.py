"""Derivatives of DFTB results with respect to parameters and geometry (milestone M4b).

    from pydftb.autodiff import Params, evaluate
    params = Params.from_skset(calc.sk, requires_grad=("tables", "hubbard"))
    res = evaluate(calc, species, positions, cell, pbc, params)      # mode="hf"
    res.free_energy.backward()                                       # -> params.tables[..].grad

Both modes start from the engine's converged calculation (NumPy, DFTB+'s rules): the
values returned are the engine's own numbers, the modes only add derivatives.

``mode="hf"`` (default) differentiates the Mermin free energy F through its variational
property: the converged densities enter a surrogate energy as constants, whose gradient is
exactly dF/dtheta, the forces and the stress (M4b design, section 2).  Only F (and E at
T ~ 0) carries a graph.

``mode="implicit"`` recomputes one SCC step in torch at the converged populations and
attaches the fixed point by the implicit function theorem (:mod:`pydftb.autodiff.scc`):
every returned quantity -- E, F, charges, eigenvalues, Fermi levels, forces, stress -- has
its full derivative (dE/dtheta with smearing, d forces/dtheta for force matching ...).

The hf gradient is the one of the fixed-N Mermin functional.  Two DFTB+ conventions of the
engine depart from it: Methfessel-Paxton smearing with ``ts_convention="dftb+"`` (F is not
variational, so mode "hf" refuses dF: use ``ts_convention="variational"`` or mode
"implicit"), and the mid-gap Fermi level for an integer electron count, accepted when the
count there is within 1e-7 (the gradient of the engine's F then differs by mu dN/dtheta,
up to ~1e-5 relative in the checks; ``Smearing(fermi_level="exact")`` removes it; mode
"implicit" differentiates the rule itself).
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass

import numpy as np
import torch

from ..backend import ops
from ..constants import MIN_TEMP
from .density import entropy
from ..hamiltonian import pair_values
from .model import (bond_vectors, deform, gamma_matrix, onsite_vector, pair_terms,
                    repulsive_energy, site_hubbard, site_hubbard_derivs, spin_matrix,
                    third_energy, third_matrix)
from .params import Params
from .scc import SCCMap, attach

MODES = ("hf", "implicit")


class GradientUnavailable(RuntimeError):
    """The requested derivative cannot be given (e.g. the SCC did not converge)."""


@dataclass
class Result:
    energy: torch.Tensor           # total energy E (Hartree)
    free_energy: torch.Tensor      # Mermin free energy F = E - TS
    forces: torch.Tensor | None    # (n_atom, 3) Hartree/Bohr
    stress: torch.Tensor | None    # (3, 3) Hartree/Bohr^3, DFTB+ sign convention
    charges: torch.Tensor          # Mulliken electron populations per atom
    eigenvalues: torch.Tensor | None   # (n_spin * n_k, n_orb)
    fermi: torch.Tensor            # one Fermi level per group (spin channels unless shared)
    raw: object                    # the engine's DFTBResult
    mode: str
    adjoint: object = None         # mode "implicit": the SCC adjoint solver (its residual)


class _External(torch.autograd.Function):
    """A term computed outside torch (dispersion): its value with its derivatives with
    respect to the positions, the cell (at fixed Cartesian positions) and the strain, from
    the gradient and the virial the library gives (constant in the parameters)."""

    @staticmethod
    def forward(ctx, pos, cell, strain, energy, grad, dcell, virial):
        ctx.save_for_backward(grad, dcell, virial)
        return energy.clone()

    @staticmethod
    def backward(ctx, g):
        grad, dcell, virial = ctx.saved_tensors
        need = ctx.needs_input_grad              # e.g. no strain for the energy of mode
        return (g * grad if need[0] else None,   # "implicit" (failed before M4d)
                g * dcell if need[1] and dcell is not None else None,
                g * virial if need[2] and virial is not None else None, None, None, None, None)


class _NoGrad(torch.autograd.Function):
    """A value whose gradient mode "hf" does not know (E with smearing, F when it is not
    variational): asking for it raises GradientUnavailable."""

    @staticmethod
    def forward(ctx, value, carrier, why):
        ctx.why = why
        return value.clone()

    @staticmethod
    def backward(ctx, g):
        raise GradientUnavailable(ctx.why)


def _engine_for(calc, params: Params):
    """A copy of ``calc`` whose SK set (and DFTB3 / spin constants) hold the values of
    ``params``; NumPy backend (the autodiff part needs host densities)."""
    eng = copy.copy(calc)
    eng.sk = params.to_skset()
    if params.hubbard_derivs is not None:
        eng.hubbard_derivs = params.host_hubbard_derivs()
    if params.spin_constants is not None:
        eng.spin_constants = params.host_spin_constants()
    return eng


def _blas_limit(n: int):
    from threadpoolctl import threadpool_limits
    return threadpool_limits(n, user_api="blas")


def _tensor(x, dtype=torch.float64, device=None):
    """A tensor of ``x`` (on ``device``; a tensor keeps its graph when moved)."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x if device is None else x.to(device)
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=dtype, device=device)


def _with_value(value, t):
    """``value`` (the engine's number) carrying the derivatives of the tensor ``t``."""
    v = torch.as_tensor(np.asarray(value, dtype=float), dtype=torch.float64, device=t.device)
    return v + (t - t.detach()) if t.requires_grad else v


def evaluate(calc, species, positions, cell=None, pbc=None, params: Params | None = None,
             mode: str = "hf", forces: bool = True, stress: bool | None = None,
             q_init=None) -> Result:
    """Energy, free energy, forces and stress of ``calc`` (a :class:`pydftb.DFTB`) with
    gradients.  ``positions``/``cell`` in Bohr, NumPy or torch (torch tensors that require
    gradients receive them).  ``params`` defaults to the engine's own values, without
    gradients.  ``stress`` defaults to ``forces`` for periodic systems."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r} (one of {MODES})")
    if getattr(calc, "pauli", False):
        raise NotImplementedError("pydftb.autodiff does not support two-component calculations "
                                  "(non-collinear spin, spin-orbit coupling): NumPy engine only "
                                  "(milestone M6a)")
    if getattr(calc, "block_terms", False):
        raise NotImplementedError("pydftb.autodiff does not support DFTB+U or onsite "
                                  "corrections: NumPy engine only (milestone M6b)")
    if params is None:
        params = Params.from_skset(calc.sk, calc.repulsive, calc.hubbard_derivs,
                                   calc.spin_constants)
    dev = params.device                      # the derivatives are computed there
    pos_in, cell_in = _tensor(positions, device=dev), _tensor(cell, device=dev)
    periodic = cell_in is not None and (pbc is None or np.any(pbc))
    if stress is None:
        stress = forces and periodic
    if stress and not periodic:
        raise ValueError("stress needs a periodic system")
    eng = _engine_for(calc, params)
    pos_h = pos_in.detach().cpu().numpy()
    cell_h = None if cell_in is None else cell_in.detach().cpu().numpy()
    # the engine's default outputs (forces, and the stress of a periodic system), so that
    # Result.raw is what calc.calculate returns; forces/stress only choose what carries a graph
    limit = (contextlib.nullcontext() if eng.k_threads.n == 1
             else _blas_limit(1))                # as DFTB.calculate: k threads take the cores
    with limit:
        res, st, sc = eng._run(species, pos_h, cell_h, pbc, True, q_init, None)
    if eng.scc and not sc.converged:
        err = GradientUnavailable(f"the SCC did not converge (max |dq| = {sc.error:.1e})")
        err.raw = res                   # the engine's results all the same (PyDFTB.evaluate)
        raise err
    pc = cell_in if periodic else None
    implicit = mode == "implicit"
    phi = out = adj = None
    if implicit:
        phi = SCCMap(eng, st, params, pos_in, pc, need_w=forces or stress)
        q_hat, adj = attach(phi, sc.q_in)
        out = phi.solve(q_hat)
        dens = _torch_densities(phi, out)
    else:
        dens = _host_densities(eng, st, sc, dev)
    e_sur, f_t, s_t = None, None, None
    if not implicit or forces or stress:
        e_sur, f_t, s_t = _explicit(eng, st, params, pos_in, pc, dens, forces, stress,
                                    create_graph=implicit)
    f_val = None if not forces else _with_value(res.forces, f_t if implicit else f_t.detach())
    s_val = None if not stress else _with_value(res.stress, s_t if implicit else s_t.detach())

    if implicit:
        e_t, ts_t = _implicit_energies(eng, st, phi, out, params, pos_in, pc)
        energy = _with_value(res.energy, e_t)
        free = _with_value(res.free_energy, e_t - ts_t)
        qa = out["q_orb"].new_zeros(len(st.species)).index_add(
            0, torch.as_tensor(st.basis.orb_atom, device=dev), out["q_orb"])
        charges = _with_value(res.charges, qa)
        ev = _with_value(np.array(res.eigenvalues), out["evals"])
        fermi = _with_value(_fermi_levels(res, st), out["mu"])
        return Result(energy, free, f_val, s_val, charges, ev, fermi, res, mode, adj)

    sm = eng.smearing
    kt_zero = sm.kt <= MIN_TEMP * (1 + 1e-12)
    shift = e_sur - e_sur.detach()
    f0 = torch.as_tensor(res.free_energy, dtype=torch.float64, device=dev)
    e0 = torch.as_tensor(res.energy, dtype=torch.float64, device=dev)
    if sm.variational:
        free = f0 + shift
    else:
        free = _NoGrad.apply(f0, shift, "with Methfessel-Paxton smearing and DFTB+'s TS "
                             "(ts_convention='dftb+') F is not variational: use "
                             "Smearing(..., ts_convention='variational') or mode='implicit'")
    if kt_zero and sm.variational:
        energy = e0 + shift
    else:
        energy = _NoGrad.apply(e0, shift, "with smearing, mode='hf' gives the gradient of "
                               "the free energy only (use free_energy, or mode='implicit')")
    ev = (None if res.eigenvalues is None
          else torch.as_tensor(np.array(res.eigenvalues), device=dev))
    return Result(energy, free, f_val, s_val, torch.as_tensor(res.charges, device=dev), ev,
                  torch.as_tensor(_fermi_levels(res, st), device=dev), res, mode)


def _fermi_levels(res, st):
    if st.spin is not None and not st.spin.shared_ef:
        return np.array(res.fermi_spin, dtype=float)
    return np.array([res.fermi], dtype=float)


def _explicit(eng, st, params, pos_in, cell, dens, forces, stress, create_graph):
    """The surrogate energy with the densities ``dens`` and its derivatives at fixed
    densities: the forces (an internal leaf for the positions) and the stress (a strain);
    the caller's own position tensor stays connected.  With graphs (mode "implicit") and
    the engine's ``fd_delta``, the H0/S part of the forces and stress uses DFTB+'s finite
    differences, as the engine does, so that they are the derivatives of its forces."""
    pos_leaf = pos_in.detach().requires_grad_(True) if forces else pos_in
    pos_base = pos_leaf
    if forces and pos_in.requires_grad:
        pos_base = pos_leaf + (pos_in - pos_in.detach())
    strain = (torch.zeros((3, 3), dtype=torch.float64, device=pos_in.device,
                          requires_grad=True) if stress else None)
    pos, cell_t = deform(pos_base, cell, strain)
    fd_band = create_graph and eng.fd_delta is not None and (forces or stress)
    e_sur = _surrogate(eng, st, params, pos, cell_t, dens, pairs=not fd_band)
    # the library's virial already includes the strain of the positions: undeformed here
    e_sur = e_sur + _dispersion(st, pos_base, cell, strain)
    wants = [x for x in (pos_leaf if forces else None, strain) if x is not None]
    f_t = s_t = None
    if wants:
        grads = list(torch.autograd.grad(e_sur, wants, retain_graph=True,
                                         create_graph=create_graph, allow_unused=True))
        grads = [torch.zeros_like(x) if g is None else g for x, g in zip(wants, grads)]
        if fd_band:
            g_band, w_band = _band_gradient_fd(eng, st, params, pos_base, cell, dens)
            if forces:
                grads[0] = grads[0] + g_band
            if stress:
                grads[-1] = grads[-1] + w_band
        if forces:
            f_t = -grads[0]
        if stress:
            vol = torch.abs(torch.linalg.det(cell.detach()))
            s_t = -0.5 * (grads[-1] + grads[-1].T) / vol
    if fd_band:          # the energy itself (for parameter gradients) has the H0/S terms
        e_sur = _surrogate(eng, st, params, pos, cell_t, dens) + _dispersion(st, pos_base,
                                                                            cell, strain)
    return e_sur, f_t, s_t


def _band_gradient_fd(eng, st, params, pos, cell, d):
    """dE/dR and the virial of the H0/S terms with DFTB+'s finite-difference derivatives
    of the pair blocks (the engine's _gradient with ``fd_delta``)."""
    pb = st.pairs
    vec = bond_vectors(pos, cell, pb.i, pb.j, pb.shift)
    _, _, dh, ds = pair_values(pb, eng.sk, vec, tables=params.tables, fd_delta=eng.fd_delta)
    ge = d["p_e"][:, None] * dh + _coef_s(st, d)[:, None] * ds
    idx = lambda a: torch.as_tensor(a, device=ge.device)            # noqa: E731
    gp = ge.new_zeros((len(pb.i), 3)).index_add(0, idx(pb.pair_of), ge)
    natom = len(st.species)
    grad = (ge.new_zeros((natom, 3)).index_add(0, idx(pb.j), gp)
            - ge.new_zeros((natom, 3)).index_add(0, idx(pb.i), gp))
    return grad, gp.T @ vec


def _host_onsite_density(st, sc):
    """sum_k w_k Re rho_k[mu, mu] per orbital, summed over spins."""
    nk = len(st.kw)
    out = 0.0
    for s0 in range(0, len(sc.rho), nk):
        rho = ops.to_host(sc.rho[s0:s0 + nk])
        out = out + np.einsum("k,kii->i", st.kw, rho).real
    return out


def _host_densities(eng, st, sc, device=None):
    """The converged densities of the engine as constant tensors (mode "hf")."""
    p_e, w_e, p_m = eng._pair_densities(st, sc)
    v_orb, vs_orb = eng._potentials(st, sc.q_out)
    dq, m = eng._charge_and_magnetisation(st, sc.q_out)
    t = lambda a: _tensor(a, device=device)                         # noqa: E731
    return dict(p_e=t(p_e), w_e=t(w_e), p_m=t(p_m), v=t(v_orb), vs=t(vs_orb),
                onsite=t(_host_onsite_density(st, sc)), dq=t(dq), m=t(m))


def _torch_densities(phi, out):
    """The same densities from the torch SCC step (mode "implicit": with graphs)."""
    b, kw = phi.bloch, phi.kw_mat
    rho = out["rho"]
    p_m = None
    if phi.nspin == 2:
        p_m = b.pairs(rho, kw * np.repeat([1.0, -1.0], phi.nk))
    v, vs = phi.potentials(out["q_out"])
    dq, m = phi.charge_and_magnetisation(out["q_out"])
    kwt = torch.as_tensor(kw, dtype=torch.float64, device=rho.device)
    onsite = kwt @ torch.diagonal(rho, dim1=-2, dim2=-1).real
    return dict(p_e=b.pairs(rho, kw), w_e=None if out["wdm"] is None else b.pairs(out["wdm"], kw),
                p_m=p_m, v=v, vs=vs, onsite=onsite, dq=dq, m=m)


def _coef_s(st, d):
    """The density multiplying each overlap element in E_sur: P (V_mu + V_nu) / 2 - W
    (+ the magnetisation density times the spin potentials)."""
    pb = st.pairs
    dev = d["p_e"].device
    rows, cols = torch.as_tensor(pb.rows, device=dev), torch.as_tensor(pb.cols, device=dev)
    coef_s = d["p_e"] * 0.5 * (d["v"][rows] + d["v"][cols])
    if d["w_e"] is not None:
        coef_s = coef_s - d["w_e"]
    if d["p_m"] is not None:
        coef_s = coef_s + d["p_m"] * 0.5 * (d["vs"][rows] + d["vs"][cols])
    return coef_s


def _surrogate(eng, st, params, pos, cell, d, pairs: bool = True):
    """E_sur: the energy with the densities, potentials and charges ``d`` held fixed (M4b
    design, section 2).  Its gradient is the analytic gradient of the engine.  ``pairs``:
    with the H0/S terms."""
    e = (d["onsite"] * onsite_vector(st.basis, params)).sum()
    if pairs:
        h, s = pair_terms(st, eng.sk, params, pos, cell)
        e = e + (d["p_e"] * h).sum() + (_coef_s(st, d) * s).sum()
    if eng.scc:
        u = site_hubbard(st.basis, params, eng.sk.shell_resolved)
        gam = gamma_matrix(st.coulomb, pos, cell, 3.2 * u)
        e = e + 0.5 * d["dq"] @ gam @ d["dq"]
        if st.third is not None:
            ud = site_hubbard_derivs(st.basis, params, eng.sk)
            mat = third_matrix(st.third, st.coulomb, pos, cell, u, ud)
            e = e + third_energy(mat, d["dq"], st.coulomb.site_atom, st.coulomb.natom)
    if d["m"] is not None:
        e = e + 0.5 * d["m"] @ spin_matrix(st.basis, params, d["m"]) @ d["m"]
    e = e + repulsive_energy(eng, st, params, pos, cell)
    return e


def _implicit_energies(eng, st, phi, out, params, pos, cell):
    """E and TS of the torch SCC step (the engine's _energies)."""
    kw = torch.as_tensor(phi.kw_mat, dtype=torch.float64, device=out["rho"].device)
    h0 = phi.h0 if phi.nspin == 1 else torch.cat([phi.h0, phi.h0])
    e = kw @ (out["rho"] * h0.mT).sum((-1, -2)).real                    # Tr(rho H0)
    dq, m = phi.charge_and_magnetisation(out["q_out"])
    if phi.gamma is not None:
        e = e + 0.5 * dq @ phi.gamma @ dq
        if phi.third is not None:
            e = e + phi.third_shift(dq)[0]
    if m is not None:
        e = e + 0.5 * m @ phi.wspin @ m
    e = e + repulsive_energy(eng, st, params, pos, cell)
    e = e + _dispersion(st, pos, cell, None)
    sm = phi.occ.smearing
    ts = e.new_zeros(())
    for (idx, _, sf), mu in zip(phi.occ.groups, out["mu"]):
        x = (out["evals"][idx] - mu) / sm.kt
        ts = ts + sf * sm.kt * (kw[idx] @ entropy(x, sm)).sum()
    return e, ts


def _dispersion(st, pos, cell, strain):
    """The dispersion energy as an :class:`_External` term.  Its cell derivative C follows
    from the virial W = dE/d(strain) = g^T R + C^T H (H: cell rows): C = H^-T (W - g^T R)^T."""
    if st.disp is None:
        return torch.zeros((), dtype=pos.dtype, device=pos.device)
    d = st.disp
    grad = np.asarray(d.gradient, dtype=float)
    vir = dcell = None
    if d.virial is not None and st.cell is not None:     # a molecule has no cell derivative
        vir = np.asarray(d.virial, dtype=float)
        dcell = np.linalg.solve(np.asarray(st.cell, float).T, (vir - grad.T @ st.pos).T)
    t = lambda a: None if a is None else torch.as_tensor(a, device=pos.device)  # noqa: E731
    return _External.apply(pos, cell, strain, torch.as_tensor(float(d.energy), dtype=torch.float64,
                                                               device=pos.device),
                           t(grad), t(dcell), t(vir))
