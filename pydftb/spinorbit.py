"""Two-component (spinor) DFTB: non-collinear spin and spin-orbit coupling (milestone M6a).

Conventions of DFTB+ (``dftb/sparse2dense.F90``, ``dftb/spinorbit.F90``,
``math/angmomentum.F90``):

* the spinor basis is [all orbitals spin up, all orbitals spin down]; the overlap is S x 1;
* matrices are kept as four N x N Pauli components, M = M0 x 1 + Mx sx + My sy + Mz sz,
  i.e. the blocks are uu = M0 + Mz, dd = M0 - Mz, ud = Mx - i My, du = Mx + i My;
* the density's components are n = Tr_s(rho), m_a = Tr_s(sigma_a rho):
  n = rho_uu + rho_dd, m_x = rho_ud + rho_du, m_y = i (rho_ud - rho_du), m_z = rho_uu - rho_dd;
* spin-orbit coupling H_SO = xi L.S with one xi per shell.  In DFTB+'s real orbitals
  (p: y, z, x) L_a is purely imaginary; Lambda^a = Im(xi/2 L_a) is real antisymmetric and
  block diagonal (atoms, shells).  The onsite model adds i Lambda^a x sigma_a on the atom
  blocks; the dual model (DFTB+'s default) adds i (S Lambda^a + Lambda^a S)/2 x sigma_a,
  Lambda^a being an imaginary block potential applied like the DFTB+U shift.

Design notes: ``docs/design/M6a-spin-orbit.html``.
"""
from __future__ import annotations

import numpy as np

from .hamiltonian import Basis


def l_matrices(l: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(L_x, L_y, L_z) for angular momentum ``l`` in DFTB+'s real orbitals (m = -l..l,
    p: y, z, x), complex (2l + 1, 2l + 1).  Port of ``angmomentum.F90:localLOperators``:
    the operators of the complex spherical harmonics, transformed as U X U^dagger."""
    n = 2 * l + 1
    lz = np.diag(np.arange(-l, l + 1)).astype(complex)
    lplus = np.zeros((n, n), complex)
    lminus = np.zeros((n, n), complex)
    for m in range(-l, l):
        lplus[l + m + 1, l + m] = np.sqrt(l * (l + 1) - m * (m + 1))
    for m in range(-l + 1, l + 1):
        lminus[l + m - 1, l + m] = np.sqrt(l * (l + 1) - m * (m - 1))
    lx = (lplus + lminus) / 2.0
    ly = (lplus - lminus) / 2j
    u = np.zeros((n, n), complex)
    for m in range(1, l + 1):
        u[l + m, l + m] = np.sqrt(0.5) * (1.0 if m % 2 == 0 else -1.0)
        u[l + m, l - m] = np.sqrt(0.5)
        u[l - m, l + m] = -np.sqrt(0.5) * 1j * (1.0 if m % 2 == 1 else -1.0)
        u[l - m, l - m] = -np.sqrt(0.5) * 1j
    u[l, l] = 1.0
    return tuple(u @ x @ u.conj().T for x in (lx, ly, lz))


def check_xi(spin_orbit: dict, sk, labels) -> dict:
    """The spin-orbit constants of these type labels as {label: (n_shell,) array}, checked as
    DFTB+ does: one value per shell, zero for s shells.  Types with only s shells may be
    left out (zero)."""
    out = {}
    for lab in dict.fromkeys(labels):
        shells = sk.shells(lab)
        if lab not in spin_orbit:
            if any(shells):
                raise ValueError(f"spin_orbit has no entry for {lab!r} (one xi per shell, "
                                 f"{len(shells)} shells)")
            out[lab] = np.zeros(len(shells))
            continue
        xi = np.atleast_1d(np.asarray(spin_orbit[lab], float))
        if xi.shape != (len(shells),):
            raise ValueError(f"spin_orbit[{lab!r}] needs one value per shell "
                             f"({len(shells)}), got {xi.size}")
        for l, x in zip(shells, xi):
            if l == 0 and x != 0.0:
                raise ValueError(f"spin_orbit[{lab!r}]: an s shell cannot have a spin-orbit "
                                 "constant (DFTB+ stops on this too)")
        out[lab] = xi
    return out


def lambda_matrices(basis: Basis, xi: dict) -> np.ndarray:
    """Lambda^a = Im(xi/2 L_a) over all orbitals: (3, N, N) real, block diagonal."""
    norb = basis.norb
    lam = np.zeros((3, norb, norb))
    first = np.concatenate([[0], np.cumsum(2 * basis.shell_l + 1)[:-1]])
    first_shell = np.searchsorted(basis.shell_atom, np.arange(len(basis.species)))
    lmats = {}
    for ish, (atom, l) in enumerate(zip(basis.shell_atom, basis.shell_l)):
        if l == 0:
            continue
        x = xi[basis.species[atom]][ish - first_shell[atom]]
        if x == 0.0:
            continue
        if l not in lmats:
            lmats[l] = [m.imag for m in l_matrices(int(l))]
        o = first[ish]
        for a in range(3):
            lam[a, o:o + 2 * l + 1, o:o + 2 * l + 1] = 0.5 * x * lmats[l][a]
    return lam


def l_imag(basis: Basis) -> np.ndarray:
    """Im L_a over all orbitals (no xi): (3, N, N) real, block diagonal (orbital moments)."""
    norb = basis.norb
    out = np.zeros((3, norb, norb))
    first = np.concatenate([[0], np.cumsum(2 * basis.shell_l + 1)[:-1]])
    for ish, l in enumerate(basis.shell_l):
        if l == 0:
            continue
        o = first[ish]
        for a, m in enumerate(l_matrices(int(l))):
            out[a, o:o + 2 * l + 1, o:o + 2 * l + 1] = m.imag
    return out


def spinor(m0, mx, my, mz):
    """Spinor matrix (..., 2N, 2N) from its four Pauli components (..., N, N); a component
    may be 0."""
    top = np.concatenate(np.broadcast_arrays(m0 + mz, mx - 1j * my), axis=-1)
    bottom = np.concatenate(np.broadcast_arrays(mx + 1j * my, m0 - mz), axis=-1)
    return np.concatenate([top, bottom], axis=-2)


def components(rho):
    """The four Pauli components (n, m_x, m_y, m_z) of spinor matrices (..., 2N, 2N):
    (4, ..., N, N), as DFTB+ packs its density (packHSPauli)."""
    n = rho.shape[-1] // 2
    uu, ud = rho[..., :n, :n], rho[..., :n, n:]
    du, dd = rho[..., n:, :n], rho[..., n:, n:]
    return np.stack([uu + dd, ud + du, 1j * (ud - du), uu - dd])


def overlap2(smat):
    """S x 1 in the spinor basis: blockdiag(S, S) for every k, (nk, 2N, 2N)."""
    nk, n, _ = smat.shape
    s2 = np.zeros((nk, 2 * n, 2 * n), dtype=smat.dtype)
    s2[:, :n, :n] = smat
    s2[:, n:, n:] = smat
    return s2


def soc_hamiltonian(lam, smat=None):
    """H_SO for every k: i X^a x sigma_a with X^a = Lambda^a (onsite model, ``smat`` None:
    (1, 2N, 2N)) or (S_k Lambda^a + Lambda^a S_k)/2 (dual model: (nk, 2N, 2N))."""
    if smat is None:
        x = [lam[a][None].astype(complex) for a in range(3)]
    else:
        x = [0.5 * (smat @ lam[a] + lam[a] @ smat) for a in range(3)]
    return spinor(0.0, 1j * x[0], 1j * x[1], 1j * x[2])


def closed_under_inversion(kpts, kweights, decimals: int = 8) -> bool:
    """Whether a k-point set contains -k (mod reciprocal lattice) with the same weight for
    every k, i.e. was not reduced by time reversal."""
    k = np.round(np.mod(np.asarray(kpts, float), 1.0), decimals) % 1.0
    km = np.round(np.mod(-np.asarray(kpts, float), 1.0), decimals) % 1.0
    w = np.asarray(kweights, float)
    weight = {}
    for p, x in zip(map(tuple, k), w):
        weight[p] = weight.get(p, 0.0) + x
    return all(abs(weight.get(tuple(q), -1.0) - x) <= 1e-12 * max(1.0, abs(x))
               for q, x in zip(km, w))


def block_populations(comp, smat, kw, basis: Basis):
    """Dual (block) populations of every atom, Q = 1/2 sum_k w_k (m_k S_k + S_k m_k)_AA, for
    the Pauli components ``comp`` (4, nk, N, N): a list over atoms of (4, n_a, n_a) complex
    blocks; the real part is DFTB+'s qBlock, the imaginary part its skew block qiBlock."""
    out = []
    for a in range(len(basis.species)):
        o = basis.offsets[a]
        sl = slice(o, o + basis.norb_atom[a])
        ms = np.einsum("k,skij,kjl->sil", kw, comp[:, :, sl, :], smat[:, :, sl])
        sm = np.einsum("k,kij,skjl->sil", kw, smat[:, sl, :], comp[:, :, :, sl])
        out.append(0.5 * (ms + sm))
    return out


def dual_soc_energy(blocks, lam, basis: Basis) -> float:
    """E_LS of the dual model from the skew block populations, as DFTB+
    (getDualSpinOrbitEnergy): -sum_A sum_a sum_mu,nu Qi^a_{nu mu} Lambda^a_{mu nu}."""
    e = 0.0
    for a, q in enumerate(blocks):
        o = basis.offsets[a]
        sl = slice(o, o + basis.norb_atom[a])
        for c in range(3):
            e -= float(np.sum(q[c + 1].imag.T * lam[c][sl, sl]))
    return e


def orbital_moments(lim, onsite_imag, basis: Basis) -> np.ndarray:
    """<L_a> of every shell, (n_shell, 3): -sum_{mu,nu in shell} Im(L_a)_{mu nu} X_{nu mu},
    with X the imaginary part of the atom block of the charge density (onsite model,
    DFTB+ getLOnsite) or of its dual population (getLDual); ``onsite_imag`` is a list of
    the atoms' (n_a, n_a) blocks."""
    first = np.concatenate([[0], np.cumsum(2 * basis.shell_l + 1)[:-1]])
    out = np.zeros((len(basis.shell_l), 3))
    for ish, (atom, l) in enumerate(zip(basis.shell_atom, basis.shell_l)):
        o = first[ish]
        lo = o - basis.offsets[atom]
        sl = slice(o, o + 2 * l + 1)
        x = onsite_imag[atom][lo:lo + 2 * l + 1, lo:lo + 2 * l + 1]
        for a in range(3):
            out[ish, a] = -float(np.sum(lim[a][sl, sl] * x.T))
    return out
