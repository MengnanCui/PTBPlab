"""Block (on-site) potentials: DFTB+U and onsite corrections (milestone M6b).

Both act on whole atomic orbital blocks through the dual (block) Mulliken populations of
every spin component, Q^A = 1/2 sum_k w_k (rho_k S_k + S_k rho_k)_AA, and enter H as
1/2 (S V + V S) with V block diagonal (DFTB+ ``shift.F90:addShift``); the charge and spin
potentials are the special case of a diagonal V.

* DFTB+U (DFTB+ ``OrbitalPotential``, ``dftb/dftbplusu.F90``): for every group of shells
  with a value U - J, V^s = -(U-J)/2 Q^s on the group's orbital block of each spin component
  s, plus (U-J)/2 on the charge component's diagonal for the FLL functional; the energy is
  -(U-J)/4 sum_s |Q^s|^2 (+ (U-J)/2 Tr Q^q, FLL), i.e. (U-J)/2 sum_sigma Tr(n - n n) (FLL) or
  -(U-J)/2 sum_sigma Tr(n n) (pSIC) in the occupation matrices of the two spins.
* onsite corrections (DFTB+ ``OnSiteCorrection``, ``dftb/onscorrection.F90``; Dominguez,
  Niehaus, Frauenheim, J. Phys. Chem. A 119, 3535 (2015)): f+ = f_uu + f_ud for the charge,
  f- = f_uu - f_ud for the magnetisation, spread over the orbital pairs of the shells;
  W^s = f^s o (Q^s - q0 on the charge diagonal), minus each shell's mean on the diagonal;
  E = 1/2 sum_s sum W^s o Q^s.

Spin components are DFTB+'s: (q) without spin, (q, m) for collinear spin, (q, m_x, m_y, m_z)
for two-component calculations; with the dual spin-orbit model the blocks also have a skew
(imaginary) part Qi, and the potentials an imaginary part Vi that enters H as
i 1/2 (S Vi + Vi S) on the same spin component.

The SCC mixes what DFTB+ mixes (``main.F90:reduceCharges``): the orbital populations summed
over equivalence classes, then the blocks' off-diagonal elements 1/2 (Q_ij + Q_ji), then the
skew elements 1/2 (Qi_ij - Qi_ji); collinear spin in the up/down representation.  See
:class:`BlockTerms`.  Design notes: ``docs/design/M6b-dftbu-onsite.html``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hamiltonian import Basis

# DFTB+ manual, appendix "Atomic onsite constants" (Garcia's thesis, PBE): f_uu and f_ud over
# the shells (H: s; C, N, O: s p; S, Ti, Au: s p d), Hartree.  "shells" are the angular
# momenta the matrices belong to: a parameter set with other shells is rejected.
ONSITE_PBE = {
    "H": {"shells": [0], "uu": [[0.0]], "ud": [[0.0]]},
    "C": {"shells": [0, 1], "uu": [[0.0, 0.04973], [0.04973, -0.01203]],
          "ud": [[0.0, 0.10512], [0.10512, 0.02643]]},
    "N": {"shells": [0, 1], "uu": [[0.0, 0.06816], [0.06816, -0.00879]],
          "ud": [[0.0, 0.12770], [0.12770, 0.03246]]},
    "O": {"shells": [0, 1], "uu": [[0.0, 0.08672], [0.08672, -0.00523]],
          "ud": [[0.0, 0.14969], [0.14969, 0.03834]]},
    "S": {"shells": [0, 1, 2],
          "uu": [[0.0, 0.07501, 0.00398], [0.07501, 0.00310, 0.01100], [0.00398, 0.01100, -0.01792]],
          "ud": [[0.0, 0.11653, 0.03915], [0.11653, 0.03058, 0.04979], [0.03915, 0.04979, 0.01582]]},
    "Ti": {"shells": [0, 1, 2],
           "uu": [[0.0, 0.02659, -0.00587], [0.02659, -0.01297, -0.00523],
                  [-0.00587, -0.00523, -0.00750]],
           "ud": [[0.0, 0.06881, 0.01239], [0.06881, 0.01640, 0.01144], [0.01239, 0.01144, 0.02604]]},
    "Au": {"shells": [0, 1, 2],
           "uu": [[0.0, 0.03752, 0.00073], [0.03752, -0.00505, -0.00002],
                  [0.00073, -0.00002, 0.00531]],
           "ud": [[0.0, 0.06928, 0.01339], [0.06928, 0.01677, 0.01228], [0.01339, 0.01228, 0.02519]]},
}
FUNCTIONALS = ("fll", "psic")
UJ_MIN = 1e-10                       # DFTB+: U - J <= 1e-10 is "too small"


# ---- options ----------------------------------------------------------------------------
def shell_indices(spec, shells: list, label: str) -> tuple:
    """Shells of one +U group: angular momentum letters ("d", "pd") or 0-based indices."""
    if isinstance(spec, str):
        out = []
        for ch in spec.lower():
            if ch not in "spdf":
                raise ValueError(f"plus_u[{label!r}]: unknown shell {ch!r} (use s, p, d, f or "
                                 "0-based shell indices)")
            hits = [i for i, l in enumerate(shells) if l == "spdf".index(ch)]
            if not hits:
                raise ValueError(f"plus_u[{label!r}]: {label} has no {ch} shell "
                                 f"(shells {''.join('spdf'[l] for l in shells)})")
            if len(hits) > 1:
                raise ValueError(f"plus_u[{label!r}]: {label} has {len(hits)} {ch} shells; give "
                                 "0-based shell indices instead")
            out += hits
    else:
        out = [int(i) for i in np.atleast_1d(spec)]
        for i in out:
            if not 0 <= i < len(shells):
                raise ValueError(f"plus_u[{label!r}]: shell index {i} out of range (0-based, "
                                 f"{label} has {len(shells)} shells)")
    return tuple(sorted(out))


def plus_u_pairs(spec) -> list:
    """The (shells, U - J) groups of one type: a list of pairs, one pair, or a
    {shells: U - J} dict."""
    if isinstance(spec, dict):
        return list(spec.items())
    if (isinstance(spec, (tuple, list)) and len(spec) == 2 and np.ndim(spec[1]) == 0
            and not isinstance(spec[1], (tuple, list))):
        return [tuple(spec)]
    return [tuple(p) for p in spec]


def parse_plus_u(plus_u: dict, sk, labels) -> dict:
    """{type label: [(shell indices, U - J), ...]} for these labels, checked as DFTB+ does:
    U - J > 1e-10 Hartree and no shell in two groups of one type.  Per type the groups are
    a list of (shells, U - J) pairs, one pair, or a {shells: U - J} dict; shells are
    angular momentum letters ("d", "pd") or 0-based shell indices."""
    out = {}
    for lab in dict.fromkeys(labels):
        if lab not in plus_u:
            continue
        pairs = plus_u_pairs(plus_u[lab])
        shells = list(sk.shells(lab))
        groups, seen = [], []
        for shell_spec, uj in pairs:
            idx = shell_indices(shell_spec, shells, lab)
            uj = float(uj)
            if uj < 0.0:
                raise ValueError(f"plus_u[{lab!r}]: negative U - J ({uj:g})")
            if uj <= UJ_MIN:
                raise ValueError(f"plus_u[{lab!r}]: U - J = {uj:g} is too small (DFTB+ needs "
                                 "> 1e-10 Hartree)")
            seen += idx
            groups.append((idx, uj))
        if len(set(seen)) != len(seen):
            raise ValueError(f"plus_u[{lab!r}]: a shell is in more than one group (DFTB+ stops "
                             "on this too)")
        out[lab] = groups
    return out


def parse_onsite(onsite: dict, sk, labels) -> dict:
    """{type label: (f_uu, f_ud)} shell matrices for these labels.  As in DFTB+ every type
    needs an entry (zeros for H, e.g. {"uu": [[0]], "ud": [[0]]}); an entry is a dict
    {"uu": ..., "ud": ...} (optionally "shells", the angular momenta it was made for, as
    in :data:`ONSITE_PBE`) or a pair (uu, ud); matrices are n_shell x n_shell, Hartree."""
    out = {}
    for lab in dict.fromkeys(labels):
        if lab not in onsite:
            raise ValueError(f"onsite_corrections has no entry for {lab!r} (DFTB+ needs every "
                             "type; use zero matrices, e.g. {'uu': [[0]], 'ud': [[0]]} for H)")
        spec = onsite[lab]
        shells = list(sk.shells(lab))
        if isinstance(spec, dict):
            if spec.get("shells") is not None and list(spec["shells"]) != shells:
                raise ValueError(
                    f"onsite_corrections[{lab!r}] is for shells "
                    f"{''.join('spdf'[l] for l in spec['shells'])}, but {lab} has "
                    f"{''.join('spdf'[l] for l in shells)} in this parameter set")
            uu, ud = spec["uu"], spec["ud"]
        else:
            uu, ud = spec
        mats = []
        for name, m in (("uu", uu), ("ud", ud)):
            m = np.asarray(m, float)
            n = len(shells)
            if m.size != n * n:
                raise ValueError(f"onsite_corrections[{lab!r}][{name!r}] needs {n} x {n} values "
                                 f"(one per pair of the {n} shells), got {m.size}")
            mats.append(m.reshape(n, n))
        out[lab] = tuple(mats)
    return out


def onsite_orbital_matrix(shell_l, f) -> np.ndarray:
    """A shell matrix spread over the orbital pairs of an atom (DFTB+ ``getOnsME``): the
    block of shells (i, j) is f_ij, the same s shell's own element is not used, and the result
    is symmetrised."""
    sizes = [2 * l + 1 for l in shell_l]
    first = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
    n = sum(sizes)
    m = np.zeros((n, n))
    for i, (li, oi, ni) in enumerate(zip(shell_l, first, sizes)):
        for j, (oj, nj) in enumerate(zip(first, sizes)):
            if i == j and li == 0:
                continue
            m[oj:oj + nj, oi:oi + ni] = f[i, j]
    return 0.5 * (m + m.T)


# ---- the block terms of one calculation ------------------------------------------------------
@dataclass
class BlockTerms:
    """Block potentials and the SCC mixing vector of one calculation.

    ``atoms`` are the atoms that have blocks (+U groups, or all with onsite corrections);
    lists named per atom follow ``atoms``.  ``rep``: "q" (one channel), "ud" (collinear spin,
    channels up and down) or "pauli" (q, m_x, m_y, m_z); potentials and energies use the
    spin components (q), (q, m) or (q, m_x, m_y, m_z).

    The mixing vector is [class sums of the orbital populations | block elements | skew
    elements]: an orbital class is (channel, charge site) as in the existing paths, except
    that +U orbitals (and with onsite corrections all orbitals) are classes of their own;
    classes are numbered by their first (channel, atom, orbital), as DFTB+'s
    ``OrbitalEquiv_merge``.  Block elements are 1/2 (Q_ij + Q_ji) of the pairs i > j in
    the order of DFTB+'s ``blockIndx`` (+U) or ``ons_blockIndx`` (onsite).
    """
    basis: Basis
    rep: str
    atoms: np.ndarray                  # atoms with blocks
    slices: list                       # their orbital slices
    u_groups: list                     # per atom: [(shells, local orbital indices, U - J), ...]
    fll: bool
    ons: list                          # per atom: (f+, f-) orbital matrices, or None
    shell_ranges: list                 # per atom: local (start, stop) of every shell
    q0: np.ndarray                     # (N,) reference populations per orbital
    imaginary: bool                    # skew blocks (dual spin-orbit coupling)
    orb_class: np.ndarray              # (n_chan, N) class of every channel orbital
    first: np.ndarray                  # (n_class,) flat (channel * N + orbital) of the first
    pairs: np.ndarray                  # (n_pair, 4): channel, atom position, i, j (i > j)

    @property
    def n_chan(self) -> int:
        return {"q": 1, "ud": 2, "pauli": 4}[self.rep]

    @property
    def n_class(self) -> int:
        return len(self.first)

    @property
    def size(self) -> int:
        """Length of the mixing vector."""
        return self.n_class + len(self.pairs) * (2 if self.imaginary else 1)

    @property
    def has_u(self) -> bool:
        return any(self.u_groups)

    @classmethod
    def build(cls, basis: Basis, sk, rep: str, site_of_orb, plus_u: dict | None,
              functional: str, onsite: dict | None, imaginary: bool,
              shell_occupation=None) -> "BlockTerms":
        """``site_of_orb``: the charge site (atom or shell) of every orbital of the existing
        mixing; with spin the classes are shells anyway.  ``plus_u``/``onsite``: parsed
        options (:func:`parse_plus_u`, :func:`parse_onsite`).  ``shell_occupation``: the
        reference population of every shell when it is not the SKF's (DFTB+
        ``CustomisedOccupations``, M6f)."""
        natom = len(basis.species)
        norb = basis.norb
        n_chan = {"q": 1, "ud": 2, "pauli": 4}[rep]
        first_orb = np.concatenate([[0], np.cumsum(2 * basis.shell_l + 1)[:-1]]).astype(int)
        first_shell = np.searchsorted(basis.shell_atom, np.arange(natom))
        plus_u = plus_u or {}
        unique = np.zeros(norb, bool)          # orbitals that are classes of their own
        u_groups, ons, ranges, atoms = [], [], [], []
        for a, lab in enumerate(basis.species):
            o, n = int(basis.offsets[a]), int(basis.norb_atom[a])
            nsh = len(basis.shells[a])
            local = [(int(first_orb[first_shell[a] + s] - o),
                      int(first_orb[first_shell[a] + s] - o + 2 * basis.shells[a][s] + 1))
                     for s in range(nsh)]
            groups = []
            for shells, uj in plus_u.get(lab, []):
                idx = np.concatenate([np.arange(*local[s]) for s in shells])
                groups.append((shells, idx, uj))
                unique[o + idx] = True
            f = None
            if onsite is not None:
                fuu, fud = (onsite_orbital_matrix(basis.shells[a], m) for m in onsite[lab])
                f = (fuu + fud, fuu - fud)
                unique[o:o + n] = True
            if groups or f is not None:
                atoms.append(a)
                u_groups.append(groups)
                ons.append(f)
                ranges.append(local)
        # orbital classes: (channel, unique orbital) or (channel, site); spin -> shells
        site = basis.orb_shell if rep != "q" else np.asarray(site_of_orb)
        cls_of, orb_class, first = {}, np.zeros((n_chan, norb), int), []
        for c in range(n_chan):
            for mu in range(norb):
                key = (c, "orb", mu) if unique[mu] else (c, "site", int(site[mu]))
                if key not in cls_of:
                    cls_of[key] = len(first)
                    first.append(c * norb + mu)
                orb_class[c, mu] = cls_of[key]
        # block elements, in DFTB+'s order
        pairs = []
        for c in range(n_chan):
            for k, a in enumerate(atoms):
                if onsite is not None:
                    n = int(basis.norb_atom[a])
                    pairs += [(c, k, j, i) for i in range(n) for j in range(i + 1, n)]
                    continue
                for shells, _, _ in u_groups[k]:
                    for s1 in shells:
                        for s2 in shells:
                            for kk in range(*ranges[k][s1]):
                                pairs += [(c, k, ll, kk) for ll in range(*ranges[k][s2]) if ll > kk]
        q0 = np.zeros(norb)
        if shell_occupation is not None:          # custom reference occupations (M6f)
            occ = np.asarray(shell_occupation, float)
            for a in range(natom):
                for s, l in enumerate(basis.shells[a]):
                    o = first_orb[first_shell[a] + s]
                    q0[o:o + 2 * l + 1] = occ[first_shell[a] + s] / (2 * l + 1)
        else:
            for a, lab in enumerate(basis.species):
                for s, l in enumerate(basis.shells[a]):
                    o = first_orb[first_shell[a] + s]
                    q0[o:o + 2 * l + 1] = sk.occupation[lab][l] / (2 * l + 1)
        slices = [slice(int(basis.offsets[a]), int(basis.offsets[a] + basis.norb_atom[a]))
                  for a in atoms]
        return cls(basis, rep, np.array(atoms, int), slices, u_groups,
                   functional.lower() == "fll", ons, ranges, q0, bool(imaginary), orb_class,
                   np.array(first, int), np.array(pairs, int).reshape(-1, 4))

    # ---- vector <-> populations ---------------------------------------------------------
    def _pairs_of(self, k):
        """Positions in ``pairs`` of atom ``atoms[k]`` and their (channel, i, j)."""
        sel = np.nonzero(self.pairs[:, 1] == k)[0]
        return sel, self.pairs[sel, 0], self.pairs[sel, 2], self.pairs[sel, 3]

    def reduce(self, q_orb, blocks, skew=None) -> np.ndarray:
        """Mixing vector from the orbital populations (n_chan, N) and the blocks of
        ``atoms`` ((n_chan, n_a, n_a) each, real; ``skew`` their imaginary parts), all in
        the channel representation (DFTB+ ``reduceCharges``)."""
        q_orb = np.asarray(q_orb, float)
        sums = np.bincount(self.orb_class.ravel(), weights=q_orb.ravel(), minlength=self.n_class)
        sym = np.zeros(len(self.pairs))
        asym = np.zeros(len(self.pairs)) if self.imaginary else None
        for k in range(len(self.slices)):
            sel, c, i, j = self._pairs_of(k)
            sym[sel] = 0.5 * (blocks[k][c, i, j] + blocks[k][c, j, i])
            if asym is not None:
                asym[sel] = 0.5 * (skew[k][c, i, j] - skew[k][c, j, i])
        return np.concatenate([sums, sym] + ([asym] if asym is not None else []))

    def expand(self, vec):
        """(orbital populations (n_chan, N), blocks, skew blocks or None) from a mixing
        vector, in the channel representation (DFTB+ ``expandCharges``): a class's total
        sits on its first orbital; the blocks' diagonals are the populations of their
        (unique) orbitals, the elements of the vector fill both triangles (skew: with
        opposite signs)."""
        vec = np.asarray(vec, float)
        if vec.shape != (self.size,):
            raise ValueError(f"the mixing vector of this calculation has {self.size} entries, "
                             f"got shape {vec.shape}")
        norb = self.basis.norb
        q_orb = np.zeros(self.n_chan * norb)
        q_orb[self.first] = vec[:self.n_class]
        q_orb = q_orb.reshape(self.n_chan, norb)
        blocks = [np.zeros((self.n_chan, sl.stop - sl.start, sl.stop - sl.start))
                  for sl in self.slices]
        skew = [np.zeros_like(b) for b in blocks] if self.imaginary else None
        sym = vec[self.n_class:self.n_class + len(self.pairs)]
        asym = vec[self.n_class + len(self.pairs):]
        for k, sl in enumerate(self.slices):
            idx = self._diagonal(k)
            blocks[k][:, idx, idx] = q_orb[:, sl][:, idx]
            sel, c, i, j = self._pairs_of(k)
            blocks[k][c, i, j] = blocks[k][c, j, i] = sym[sel]
            if skew is not None:
                skew[k][c, i, j] = asym[sel]
                skew[k][c, j, i] = -asym[sel]
        return q_orb, blocks, skew

    def _diagonal(self, k) -> np.ndarray:
        """Local orbitals of atom ``atoms[k]`` whose block diagonal is mixed (unique)."""
        if self.ons[k] is not None:
            return np.arange(self.slices[k].stop - self.slices[k].start)
        return np.concatenate([idx for _, idx, _ in self.u_groups[k]])

    def components(self, blocks):
        """Blocks in the spin components (q, m) from the up/down channels (collinear spin);
        unchanged otherwise."""
        if self.rep != "ud" or blocks is None:
            return blocks
        return [np.stack([b[0] + b[1], b[0] - b[1]]) for b in blocks]

    # ---- potentials and energies (spin components) --------------------------------------
    def potentials(self, blocks, skew=None, onsite: bool = True):
        """Block potentials (V, Vi) of ``atoms`` from blocks in the spin components:
        lists of (n_comp, n_a, n_a); Vi is None without skew blocks.  ``onsite=False`` leaves
        the onsite corrections out (DFTB+'s first SCC iteration)."""
        v_out, vi_out = [], [] if skew is not None else None
        for k, q in enumerate(blocks):
            v = np.zeros_like(q)
            vi = np.zeros_like(q) if skew is not None else None
            for _, idx, uj in self.u_groups[k]:
                ix = np.ix_(idx, idx)
                if self.fll:
                    v[0][idx, idx] += 0.5 * uj
                for s in range(len(q)):
                    v[s][ix] -= 0.5 * uj * q[s][ix]
                    if vi is not None:
                        vi[s][ix] -= 0.5 * uj * skew[k][s][ix]
            if onsite and self.ons[k] is not None:
                w = self._onsite_shift(k, q)
                v += w
                if vi is not None:
                    vi += self._onsite_skew(k, skew[k])
            v_out.append(v)
            if vi_out is not None:
                vi_out.append(vi)
        return v_out, vi_out

    def _onsite_shift(self, k, q) -> np.ndarray:
        """W^s of DFTB+'s ``addOnsShift`` for atom ``atoms[k]``."""
        fp, fm = self.ons[k]
        sl = self.slices[k]
        w = np.empty_like(q)
        for s in range(len(q)):
            t = q[s].copy()
            if s == 0:
                t[np.diag_indices_from(t)] -= self.q0[sl]
            t *= fp if s == 0 else fm
            for start, stop in self.shell_ranges[k]:          # rotational invariance
                d = np.arange(start, stop)
                t[d, d] -= t[d, d].sum() / (stop - start)
            w[s] = t
        return w

    def _onsite_skew(self, k, qi) -> np.ndarray:
        fp, fm = self.ons[k]
        w = np.empty_like(qi)
        for s in range(len(qi)):
            t = qi[s].copy()
            t[np.diag_indices_from(t)] = 0.0
            w[s] = t * (fp if s == 0 else fm)
        return w

    def energies(self, blocks, skew=None) -> tuple[float, float]:
        """(E_U, E_onsite) from blocks in the spin components (DFTB+ ``getEnergy``,
        ``getEons``)."""
        e_u = e_ons = 0.0
        for k, q in enumerate(blocks):
            for _, idx, uj in self.u_groups[k]:
                ix = np.ix_(idx, idx)
                e_u -= 0.25 * uj * sum(float(np.sum(q[s][ix] ** 2)) for s in range(len(q)))
                if skew is not None:
                    e_u -= 0.25 * uj * sum(float(np.sum(skew[k][s][ix] ** 2))
                                           for s in range(len(q)))
                if self.fll:
                    e_u += 0.5 * uj * float(np.trace(q[0][ix]))
            if self.ons[k] is not None:
                e_ons += 0.5 * float(np.sum(self._onsite_shift(k, q) * q))
                if skew is not None:
                    e_ons += 0.5 * float(np.sum(self._onsite_skew(k, skew[k]) * skew[k]))
        return e_u, e_ons

    def atom_energies(self, blocks, skew=None) -> tuple[np.ndarray, np.ndarray]:
        """The terms of :meth:`energies` atom by atom, (E_U, E_onsite) of every atom of the
        structure, zero for atoms without blocks (milestone M6f-4a; DFTB+ ``getEnergy``/
        ``getEons`` with per-atom output)."""
        n = len(self.basis.species)
        e_u, e_ons = np.zeros(n), np.zeros(n)
        for k, q in enumerate(blocks):
            a = self.atoms[k]
            for _, idx, uj in self.u_groups[k]:
                ix = np.ix_(idx, idx)
                e_u[a] -= 0.25 * uj * sum(float(np.sum(q[s][ix] ** 2)) for s in range(len(q)))
                if skew is not None:
                    e_u[a] -= 0.25 * uj * sum(float(np.sum(skew[k][s][ix] ** 2))
                                              for s in range(len(q)))
                if self.fll:
                    e_u[a] += 0.5 * uj * float(np.trace(q[0][ix]))
            if self.ons[k] is not None:
                e_ons[a] += 0.5 * float(np.sum(self._onsite_shift(k, q) * q))
                if skew is not None:
                    e_ons[a] += 0.5 * float(np.sum(self._onsite_skew(k, skew[k]) * skew[k]))
        return e_u, e_ons

    # ---- matrices ---------------------------------------------------------------------
    def shift(self, smat, blocks, skew: bool = False):
        """1/2 (S B + B S) for the block-diagonal B of ``blocks`` ((n_a, n_a) per atom, real
        symmetric, or antisymmetric with ``skew``), for every k: (nk, N, N).  Computed
        column-block-wise, S B = S[:, A] B_A, and B S = (S B)^dagger (minus for skew)."""
        sb = np.zeros(smat.shape, dtype=np.result_type(smat.dtype, float))
        for sl, b in zip(self.slices, blocks):
            sb[:, :, sl] = smat[:, :, sl] @ b
        adj = np.conj(sb).transpose(0, 2, 1)
        return 0.5 * (sb - adj) if skew else 0.5 * (sb + adj)

    def pair_coefficient(self, real_space, ti, ci, ri, blocks, skew: bool = False):
        """Coefficient of the overlap derivative at the pair elements (T, row, col):
        1/2 (B R(T) + R(T) B)[col, row] (real part), or for imaginary potentials
        -1/2 (B I(T) + I(T) B)[col, row] with I = Im R (DFTB+ ``derivative_blockEuclidean``,
        ``derivative_iBlock``).  ``real_space``: (nT, N, N) density of one spin component."""
        r = real_space.imag if skew else real_space
        m = np.zeros(r.shape, dtype=r.dtype)
        for sl, b in zip(self.slices, blocks):
            m[:, sl, :] += b @ r[:, sl, :]
            m[:, :, sl] += r[:, :, sl] @ b
        out = 0.5 * m[ti, ci, ri]
        return -out.real if skew else out.real


def block_populations(comp, smat, kw, slices) -> list:
    """Dual populations 1/2 sum_k w_k (m_k S_k + S_k m_k)_AA of the components ``comp``
    (n_comp, nk, N, N) for the orbital slices of some atoms: (n_comp, n_a, n_a) complex each
    (real part: DFTB+ qBlock, imaginary part: its skew block qiBlock)."""
    out = []
    for sl in slices:
        ms = np.einsum("k,ckij,kjl->cil", kw, comp[:, :, sl, :], smat[:, :, sl])
        sm = np.einsum("k,kij,ckjl->cil", kw, smat[:, sl, :], comp[:, :, :, sl])
        out.append(0.5 * (ms + sm))
    return out
