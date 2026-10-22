"""Construction of the non-SCC Hamiltonian H0 and overlap S (real space).

Matrices are kept as a list of pair blocks (sparse, real space).  They are
folded into dense Gamma-point or Bloch (k-point) matrices on demand.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .backend import ops
from .backend.spec import NUMPY, Backend
from .skf import N_INTEGRALS, SKSet
from .slako import bond_frames, rotation_blocks, sk_block
from .neighbors import neighbor_list

COMPLEX_STEP = 1e-30


@dataclass
class Basis:
    species: list[str]
    offsets: np.ndarray       # first orbital of each atom
    norb_atom: np.ndarray
    shells: list[list[int]]   # angular momenta of each atom
    shell_atom: np.ndarray    # atom of each shell
    shell_l: np.ndarray
    orb_shell: np.ndarray     # shell of each orbital
    orb_atom: np.ndarray

    @property
    def norb(self) -> int:
        return int(self.norb_atom.sum())

    @classmethod
    def build(cls, species: list[str], sk: SKSet) -> "Basis":
        norb = np.array([sk.norb(s) for s in species])
        offsets = np.concatenate([[0], np.cumsum(norb)[:-1]])
        shells = [sk.shells(s) for s in species]
        shell_atom, shell_l, orb_shell, orb_atom = [], [], [], []
        ish = 0
        for ia, shl in enumerate(shells):
            for l in shl:
                shell_atom.append(ia)
                shell_l.append(l)
                orb_shell += [ish] * (2 * l + 1)
                orb_atom += [ia] * (2 * l + 1)
                ish += 1
        return cls(list(species), offsets, norb, shells, np.array(shell_atom),
                   np.array(shell_l), np.array(orb_shell), np.array(orb_atom))


@dataclass
class PairBlocks:
    """Real-space H0 / S blocks for every ordered neighbour pair."""
    i: np.ndarray
    j: np.ndarray
    shift: np.ndarray
    vec: np.ndarray
    dist: np.ndarray
    rows: np.ndarray       # flattened orbital row index of each element
    cols: np.ndarray
    pair_of: np.ndarray    # pair index of each element
    h: np.ndarray          # values (flattened, aligned with rows/cols)
    s: np.ndarray
    dh: np.ndarray | None = None   # (nelem, 3) derivative wrt bond vector
    ds: np.ndarray | None = None
    groups: list | None = None     # (label a, label b, pair indices) in element order


def build_pairs(species, pos, cell, pbc, sk: SKSet, basis: Basis,
                derivatives: bool = False, fd_delta: float | None = None) -> PairBlocks:
    """Real-space H0/S pair blocks (and their derivatives w.r.t. the bond vector).

    Derivatives are exact (complex step) unless ``fd_delta`` is given; then
    they are central finite differences with that step (Bohr), exactly like
    DFTB+'s ``Differentiation = FiniteDiff { Delta = ... }`` (``nonscc.F90``),
    which allows bit-level comparisons with DFTB+ forces.
    """
    i, j, shift, vec, dist = neighbor_list(pos, cell, pbc, sk.cutoff)
    rows, cols, pidx, hv, sv, dhv, dsv, groups = [], [], [], [], [], [], [], []
    spec = np.array(species)
    elems = sorted(set(species))
    for a in elems:
        for b in elems:
            mask = (spec[i] == a) & (spec[j] == b)
            sel = np.nonzero(mask)[0]
            if len(sel) == 0:
                continue
            groups.append((a, b, sel))
            shells_a, shells_b = sk.shells(a), sk.shells(b)
            na, nb = sk.norb(a), sk.norb(b)

            def blocks(v):
                d = np.sqrt((v * v).sum(1))
                rot = bond_frames(v)
                dm = rotation_blocks(rot, max(max(shells_a), max(shells_b)))
                fab, fba = sk.files[(a, b)], sk.files[(b, a)]
                iab = fab.evaluate(d, full=False)
                iba = fba.evaluate(d, full=False)
                cab, cba = fab.colmap, fba.colmap
                hb = sk_block(v, iab, iba, shells_a, shells_b, 0, dm, cab, cba)
                sb = sk_block(v, iab, iba, shells_a, shells_b, N_INTEGRALS, dm, cab, cba)
                return hb, sb

            v = vec[sel]
            hb, sb = blocks(v)
            if derivatives:
                dh = np.empty(hb.shape + (3,))
                ds = np.empty(sb.shape + (3,))
                for k in range(3):
                    if fd_delta is None:
                        vc = v.astype(complex)
                        vc[:, k] += 1j * COMPLEX_STEP
                        hc, sc = blocks(vc)
                        dh[..., k] = hc.imag / COMPLEX_STEP
                        ds[..., k] = sc.imag / COMPLEX_STEP
                    else:
                        vp, vm = v.copy(), v.copy()
                        vp[:, k] += fd_delta
                        vm[:, k] -= fd_delta
                        (hp, sp_), (hm, sm) = blocks(vp), blocks(vm)
                        dh[..., k] = (hp - hm) / (2.0 * fd_delta)
                        ds[..., k] = (sp_ - sm) / (2.0 * fd_delta)
                dhv.append(dh.reshape(-1, 3))
                dsv.append(ds.reshape(-1, 3))
            r0 = basis.offsets[i[sel]]
            c0 = basis.offsets[j[sel]]
            rr = (r0[:, None, None] + np.arange(na)[None, :, None]) + 0 * np.arange(nb)[None, None, :]
            cc = (c0[:, None, None] + np.arange(nb)[None, None, :]) + 0 * np.arange(na)[None, :, None]
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            pidx.append(np.repeat(sel, na * nb))
            hv.append(hb.ravel())
            sv.append(sb.ravel())
    cat = lambda x, dt: np.concatenate(x) if x else np.zeros(0, dt)
    pb = PairBlocks(i, j, shift, vec, dist, cat(rows, int), cat(cols, int), cat(pidx, int),
                    cat(hv, float), cat(sv, float), groups=groups)
    if derivatives:
        pb.dh = np.concatenate(dhv) if dhv else np.zeros((0, 3))
        pb.ds = np.concatenate(dsv) if dsv else np.zeros((0, 3))
    return pb


def pair_values(pb: PairBlocks, sk: SKSet, vec, tables=None, fd_delta: float | None = None):
    """H0/S values of every pair element of ``pb`` (the order of ``pb.h``) for the bond
    vectors ``vec`` (n_pair, 3), NumPy or torch (milestone M4b), interpolated from
    ``tables[(a, b)]`` (default: the SKF tables; torch leaves make them differentiable).
    With ``fd_delta`` also DFTB+'s central differences (n_elem, 3) w.r.t. the bond vector,
    as :func:`build_pairs`; returns ``h, s`` or ``h, s, dh, ds``."""
    xp = ops.namespace(vec)
    hv, sv, dhv, dsv = [], [], [], []
    for a, b, sel in pb.groups:
        shells_a, shells_b = sk.shells(a), sk.shells(b)
        lmax = max(max(shells_a), max(shells_b))
        fab, fba = sk.files[(a, b)], sk.files[(b, a)]
        tab = fab.table if tables is None else tables[(a, b)]
        tba = fba.table if tables is None else tables[(b, a)]
        cab, cba = fab.colmap, fba.colmap

        def blocks(v):
            d = xp.sqrt((v * v).sum(1))
            dm = rotation_blocks(bond_frames(v), lmax)
            iab, iba = fab.evaluate_table(d, tab), fba.evaluate_table(d, tba)
            return (sk_block(v, iab, iba, shells_a, shells_b, 0, dm, cab, cba),
                    sk_block(v, iab, iba, shells_a, shells_b, N_INTEGRALS, dm, cab, cba))

        v = vec[sel]
        hb, sb = blocks(v)
        hv.append(hb.reshape(-1))
        sv.append(sb.reshape(-1))
        if fd_delta is not None:
            dh, ds = [], []
            for k in range(3):
                step = ops.zeros(3, like=v)
                step[k] = fd_delta
                (hp, sp_), (hm, sm) = blocks(v + step), blocks(v - step)
                dh.append(((hp - hm) / (2.0 * fd_delta)).reshape(-1))
                ds.append(((sp_ - sm) / (2.0 * fd_delta)).reshape(-1))
            dhv.append(xp.stack(dh, axis=-1))
            dsv.append(xp.stack(ds, axis=-1))
    h, s = xp.concatenate(hv), xp.concatenate(sv)
    if fd_delta is None:
        return h, s
    return h, s, xp.concatenate(dhv), xp.concatenate(dsv)


def onsite_energies(basis: Basis, sk: SKSet) -> np.ndarray:
    eps = np.empty(basis.norb)
    for ia, el in enumerate(basis.species):
        o = basis.offsets[ia]
        for l in sk.shells(el):
            eps[o + l * l: o + (l + 1) ** 2] = sk.onsite[el][l]
    return eps


def fold(pb: PairBlocks, values: np.ndarray, norb: int, kpt: np.ndarray | None,
         diag: np.ndarray | None = None) -> np.ndarray:
    """Dense Bloch matrix  M(k)_{mu nu} = sum_T M_{mu nu}(T) exp(2 pi i k.T).

    ``kpt`` is in fractional reciprocal coordinates (None => Gamma, real).
    """
    if kpt is None or not np.any(kpt):
        m = sp.coo_matrix((values, (pb.rows, pb.cols)), shape=(norb, norb)).toarray()
    else:
        phase = np.exp(2j * np.pi * (pb.shift @ kpt))[pb.pair_of]
        m = sp.coo_matrix((values * phase, (pb.rows, pb.cols)), shape=(norb, norb)).toarray()
    if diag is not None:
        m[np.diag_indices(norb)] += diag
    return m


def shift_index(pb: PairBlocks):
    """Unique lattice translations T and, for each matrix element, its T index."""
    shifts, inv = np.unique(pb.shift, axis=0, return_inverse=True)
    return shifts, inv.ravel()[pb.pair_of]


def fold_all(pb: PairBlocks, values: np.ndarray, norb: int, kpts: np.ndarray,
             diag: np.ndarray | None = None, backend: Backend = NUMPY):
    """Bloch matrices for all k at once: M(k) = sum_T M(T) exp(2 pi i k.T).

    Elements are first summed into one small real-space matrix per
    translation T, then folded with a single (nk x nT) @ (nT x N^2) product.
    With the torch backend only that product (and the diagonal) runs on the device,
    in double precision; the real-space matrices and the phases are the same host arrays.
    """
    shifts, t_of = shift_index(pb)
    mt = np.zeros((len(shifts), norb, norb))
    mt = ops.scatter_add(mt, (t_of, pb.rows, pb.cols), values)
    phase = np.exp(2j * np.pi * (kpts @ shifts.T))                 # (nk, nT)
    phase = backend.asarray(phase, precise=True)
    mt = backend.asarray(mt, precise=True, complex_=True)
    m = (phase @ mt.reshape(len(shifts), -1)).reshape(len(kpts), norb, norb)
    if diag is not None:
        idx = backend.index(np.arange(norb))
        m[:, idx, idx] += backend.asarray(diag, precise=True)
    return m
