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
    shell_local: np.ndarray | None = None   # index of each shell on its atom (M6f-5)

    @property
    def norb(self) -> int:
        return int(self.norb_atom.sum())

    @classmethod
    def build(cls, species: list[str], sk: SKSet) -> "Basis":
        norb = np.array([sk.norb(s) for s in species])
        offsets = np.concatenate([[0], np.cumsum(norb)[:-1]])
        shells = [sk.shells(s) for s in species]
        shell_atom, shell_l, orb_shell, orb_atom, shell_local = [], [], [], [], []
        ish = 0
        for ia, shl in enumerate(shells):
            for k, l in enumerate(shl):
                shell_atom.append(ia)
                shell_l.append(l)
                shell_local.append(k)
                orb_shell += [ish] * (2 * l + 1)
                orb_atom += [ia] * (2 * l + 1)
                ish += 1
        return cls(list(species), offsets, norb, shells, np.array(shell_atom),
                   np.array(shell_l), np.array(orb_shell), np.array(orb_atom),
                   np.array(shell_local, dtype=int))


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


def _grouped_blocks(sk: SKSet, a: str, b: str):
    """The H0/S block function of a label pair whose shells come from several SK files (DFTB+
    ``SelectedShells``, milestone M6f-5): one :func:`sk_block` per pair of shell groups, with that
    pair's A-B and B-A files, placed at the groups' orbitals (DFTB+ ``getFullTable``: per shell
    pair, l_a <= l_b from file (A_i, B_j), else from (B_j, A_i))."""
    ga, gb = sk.shell_groups(a), sk.shell_groups(b)
    files_ab, files_ba = sk.group_files[(a, b)], sk.group_files[(b, a)]
    lmax = max(max(sk.shells(a)), max(sk.shells(b)))
    off_a = np.cumsum([0] + [sum(2 * l + 1 for l in ls) for _, ls in ga])
    off_b = np.cumsum([0] + [sum(2 * l + 1 for l in ls) for _, ls in gb])

    def blocks(v):
        d = np.sqrt((v * v).sum(1))
        dm = rotation_blocks(bond_frames(v), lmax)
        hb = sb = None
        for i, (_, la) in enumerate(ga):
            for j, (_, lb) in enumerate(gb):
                fab, fba = files_ab[i][j], files_ba[j][i]
                iab = fab.evaluate(d, full=False)
                iba = fba.evaluate(d, full=False)
                h = sk_block(v, iab, iba, la, lb, 0, dm, fab.colmap, fba.colmap)
                s = sk_block(v, iab, iba, la, lb, N_INTEGRALS, dm, fab.colmap, fba.colmap)
                if hb is None:
                    hb = np.zeros((len(v), off_a[-1], off_b[-1]), dtype=h.dtype)
                    sb = np.zeros((len(v), off_a[-1], off_b[-1]), dtype=s.dtype)
                hb[:, off_a[i]:off_a[i + 1], off_b[j]:off_b[j + 1]] = h
                sb[:, off_a[i]:off_a[i + 1], off_b[j]:off_b[j + 1]] = s
        return hb, sb

    return blocks


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

            if (a, b) in sk.group_files:      # M6f-5: shells from several files (SelectedShells)
                blocks = _grouped_blocks(sk, a, b)
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


def _type_pair_blocks(v, sk: SKSet, a: str, b: str):
    """H0 and S blocks of pairs a -> b (type labels) for the bond vectors ``v``: the operations
    of :func:`build_pairs`' ``blocks``, in the same order (so the values are bitwise the same)."""
    shells_a, shells_b = sk.shells(a), sk.shells(b)
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


def build_mixed_pairs(species, pos, cell, pbc, sk: SKSet, basis: Basis, candidates: dict,
                      column: dict, weights: np.ndarray, derivatives: bool = False,
                      fd_delta: float | None = None, keep: np.ndarray | None = None):
    """:func:`build_pairs` for atoms whose parameters are mixtures of type labels (milestone
    M5a, :mod:`pydftb.typeswitch`): the block of the pair A-B is sum_tu w_A,t w_B,u B^tu(R) over
    the candidate labels t of A and u of B (``candidates[species label]``, weight column
    ``column[label]`` of ``weights`` (n_atom, n_labels)), its derivative w.r.t. the bond vector
    likewise (weights fixed).  Pairs are grouped by ``species`` (labels of equal basis); every
    type pair is computed with build_pairs' operations on the pairs where w_A,t w_B,u != 0, so
    with weights 0 and 1 the values are build_pairs' (bitwise).

    Returns the PairBlocks and, with ``derivatives`` and ``keep`` (n_atom bool), the raw
    type-pair blocks of the pairs with an atom in ``keep``: a list of (column of t, column of u,
    pair indices, element indices, h, s), h and s flattened like the elements of ``pb.h``."""
    i, j, shift, vec, dist = neighbor_list(pos, cell, pbc, sk.cutoff)
    rows, cols, pidx, hv, sv, dhv, dsv, groups, kept = [], [], [], [], [], [], [], [], []
    spec = np.array(species)
    elems = sorted(set(species))
    offset = 0
    for a in elems:
        for b in elems:
            mask = (spec[i] == a) & (spec[j] == b)
            sel = np.nonzero(mask)[0]
            if len(sel) == 0:
                continue
            groups.append((a, b, sel))
            na, nb = sk.norb(a), sk.norb(b)
            hb = np.zeros((len(sel), na, nb))
            sb = np.zeros((len(sel), na, nb))
            dh = np.zeros((len(sel), na, nb, 3)) if derivatives else None
            ds = np.zeros((len(sel), na, nb, 3)) if derivatives else None
            v_all = vec[sel]
            w_i, w_j = weights[i[sel]], weights[j[sel]]
            touch = None if (keep is None or not derivatives) else keep[i[sel]] | keep[j[sel]]
            for t in candidates[a]:
                for u in candidates[b]:
                    c = w_i[:, column[t]] * w_j[:, column[u]]
                    sub = np.nonzero(c)[0]
                    if len(sub) == 0:
                        continue
                    v = v_all[sub]
                    bh, bs = _type_pair_blocks(v, sk, t, u)
                    cc = c[sub][:, None, None]
                    hb[sub] += cc * bh
                    sb[sub] += cc * bs
                    if derivatives:
                        bdh = np.empty(bh.shape + (3,))
                        bds = np.empty(bs.shape + (3,))
                        for k in range(3):
                            if fd_delta is None:
                                vc = v.astype(complex)
                                vc[:, k] += 1j * COMPLEX_STEP
                                hc, sc = _type_pair_blocks(vc, sk, t, u)
                                bdh[..., k] = hc.imag / COMPLEX_STEP
                                bds[..., k] = sc.imag / COMPLEX_STEP
                            else:
                                vp, vm = v.copy(), v.copy()
                                vp[:, k] += fd_delta
                                vm[:, k] -= fd_delta
                                (hp, sp_), (hm, sm) = (_type_pair_blocks(vp, sk, t, u),
                                                       _type_pair_blocks(vm, sk, t, u))
                                bdh[..., k] = (hp - hm) / (2.0 * fd_delta)
                                bds[..., k] = (sp_ - sm) / (2.0 * fd_delta)
                        dh[sub] += cc[..., None] * bdh
                        ds[sub] += cc[..., None] * bds
                    if touch is not None:
                        kk = np.nonzero(touch[sub])[0]
                        if len(kk):
                            loc = sub[kk]
                            elem = (offset + loc[:, None] * (na * nb)
                                    + np.arange(na * nb)[None, :]).ravel()
                            kept.append((column[t], column[u], sel[loc], elem, bh[kk].ravel(),
                                         bs[kk].ravel()))
            if derivatives:
                dhv.append(dh.reshape(-1, 3))
                dsv.append(ds.reshape(-1, 3))
            r0 = basis.offsets[i[sel]]
            c0 = basis.offsets[j[sel]]
            rr = (r0[:, None, None] + np.arange(na)[None, :, None]) + 0 * np.arange(nb)[None, None, :]
            cc_ = (c0[:, None, None] + np.arange(nb)[None, None, :]) + 0 * np.arange(na)[None, :, None]
            rows.append(rr.ravel())
            cols.append(cc_.ravel())
            pidx.append(np.repeat(sel, na * nb))
            hv.append(hb.ravel())
            sv.append(sb.ravel())
            offset += len(sel) * na * nb
    cat = lambda x, dt: np.concatenate(x) if x else np.zeros(0, dt)    # noqa: E731
    pb = PairBlocks(i, j, shift, vec, dist, cat(rows, int), cat(cols, int), cat(pidx, int),
                    cat(hv, float), cat(sv, float), groups=groups)
    if derivatives:
        pb.dh = np.concatenate(dhv) if dhv else np.zeros((0, 3))
        pb.ds = np.concatenate(dsv) if dsv else np.zeros((0, 3))
    return pb, kept


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
    if not hv:                       # no pair within the cut-off (e.g. an isolated atom)
        empty = ops.zeros(0, like=vec)
        return (empty, empty) if fd_delta is None else (empty, empty, ops.zeros((0, 3), like=vec),
                                                        ops.zeros((0, 3), like=vec))
    h, s = xp.concatenate(hv), xp.concatenate(sv)
    if fd_delta is None:
        return h, s
    return h, s, xp.concatenate(dhv), xp.concatenate(dsv)


def onsite_energies(basis: Basis, sk: SKSet) -> np.ndarray:
    eps = np.empty(basis.norb)
    for ia, el in enumerate(basis.species):
        o = basis.offsets[ia]
        if el in sk.groups:                  # M6f-5: one value per shell, shells in any order
            for k, l in enumerate(sk.shells(el)):
                eps[o: o + 2 * l + 1] = sk.onsite[el][k]
                o += 2 * l + 1
            continue
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
        idx = backend.index(np.arange(norb), precise=True)
        m[:, idx, idx] += backend.asarray(diag, precise=True)
    return m
