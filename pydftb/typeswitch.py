"""Smooth switching between several parameter sets of one element (milestone M5a).

DFTB+ gives every atom one type label (one parameter set); pydftb's labels do the same
(``Cu``, ``Cu2``: files ``Cu-Cu2.skf`` etc., DFTB+'s ``Type2FileNames`` convention).  Here every
atom A of a *switched* element gets weights w_A,t >= 0, sum_t w_A,t = 1, over the element's
types, as a smooth function of its coordination number:

* coordination number CN_A = sum_B f(r_AB) over all neighbours B (periodic images included;
  optionally only some elements), with the counting function f = 1 for r <= r_on,
  f = 1 - S((r - r_on) / (r_off - r_on)) in between, 0 for r >= r_off, where
  S(x) = 6x^5 - 15x^4 + 10x^3 (C2, exactly 0 and 1 outside [0, 1]); by default
  r_on/off = 1.1/1.3 x (R_A + R_B) with ASE's covalent radii;
* weights: the types t_0 .. t_m-1 are ordered by increasing CN; window k = 1 .. m-1 (centre
  c_k, width d_k) switches from t_k-1 to t_k with s_k = S((CN - c_k + d_k/2) / d_k);
  w_0 = 1 - s_1, w_k = s_k - s_k+1, w_m-1 = s_m-1 (windows must not overlap);
* mixing: H0_AB = sum_tu w_A,t w_B,u H^tu_AB(R), S likewise, E_rep = 1/2 sum over ordered pairs
  of sum_tu w_A,t w_B,u V^tu(r_AB).  On-site energies, Hubbard values, occupations, basis
  (and the per-label options Ud, W, xi, DFTB+U) must be equal for all types of an element, so
  nothing else depends on the weights; dispersion and the other element-keyed terms do not
  see the types;
* forces and stress: the usual terms with the mixed H0/S at fixed weights, plus
  sum_A sum_t dE/dw_A,t dw_A,t/dCN_A dCN_A/dR, where (variational, densities fixed)
  dE/dw_A,t = sum over the pair elements of A of [rho_e H^tu_e + c_e S^tu_e] w_B,u (+ the
  repulsive's V^tu w_B,u / 2 per ordered pair), c_e the coefficient of dS in the force code
  (SCC shift times density minus energy-weighted density, with spin, spin-orbit and block
  terms); the virial gets sum over CN pairs of dE/dCN_A f'(r)/r r (x) r.

The parameters of a switch (windows and cut-offs) are hyperparameters, never fitted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.data import atomic_numbers, covalent_radii

from .constants import BOHR
from .neighbors import neighbor_list
from .system import LABEL_FORBIDDEN, element_of_label

DEFAULT_RADIUS_SCALE = (1.1, 1.3)    # (r_on, r_off) / (R_cov,A + R_cov,B)
_ONSITE_LINE_SIZES = (10, 13)        # values of an SKF on-site line (simple, extended format)


def smoothstep(x):
    """S(x) = 6x^5 - 15x^4 + 10x^3 on [0, 1] (0 below, 1 above) and dS/dx; S is clipped to
    [0, 1] (round-off puts it 2e-16 above 1 just below x = 1)."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    s = np.clip(x * x * x * (x * (6.0 * x - 15.0) + 10.0), 0.0, 1.0)
    return s, 30.0 * x * x * (x - 1.0) ** 2


def _as_tuple(value, n: int, name: str) -> tuple:
    vals = tuple(float(v) for v in np.atleast_1d(np.asarray(value, float)).ravel())
    if len(vals) == 1 and n > 1:
        raise ValueError(f"{name}: one value per switching window ({n} for {n + 1} types)")
    if len(vals) != n:
        raise ValueError(f"{name} needs {n} value(s) (one per switching window), not {len(vals)}")
    if not all(np.isfinite(vals)):
        raise ValueError(f"{name} must be finite")
    return vals


def _check_window(name: str, window) -> tuple[float, float]:
    r_on, r_off = (float(x) for x in window)
    if not (0.0 < r_on < r_off and np.isfinite(r_off)):
        raise ValueError(f"{name}: needs 0 < r_on < r_off (Bohr), not ({r_on}, {r_off})")
    return r_on, r_off


@dataclass(frozen=True)
class TypeSwitch:
    """Smooth switching of the atoms of ``element`` between the type labels ``types`` (parameter
    sets) as a function of their coordination number CN.

    element : chemical symbol, e.g. ``"Cu"``; :meth:`DFTB.calculate` takes this symbol as the
              label of its atoms (not a type label).
    types   : the type labels, ordered by increasing CN: below the first window an atom is
              ``types[0]``, above the last ``types[-1]`` (e.g. ``("Cu2", "Cu")`` for under-
              coordinated surface atoms ``Cu2`` and bulk ``Cu``).
    centre, width : CN at the middle and width of each switching window (one per pair of
              consecutive types; a number for two types): between centre - width/2 and
              centre + width/2 the weights go from one type to the next (C2); outside they are
              exactly 0 or 1.
    cutoff  : the counting function's (r_on, r_off) in Bohr: neighbours closer than r_on count
              1, farther than r_off 0; one pair for every neighbour element, or a dict
              {element: (r_on, r_off)} whose keys are the elements that count; None (default):
              every element, r_on/r_off = 1.1/1.3 x the sum of the two covalent radii (ASE's).
    """
    element: str
    types: tuple
    centre: object
    width: object
    cutoff: object = None

    def __post_init__(self):
        if self.element not in atomic_numbers:
            raise ValueError(f"TypeSwitch: {self.element!r} is not a chemical symbol")
        types = tuple(str(t) for t in self.types)
        if len(types) < 2 or len(set(types)) != len(types):
            raise ValueError(f"TypeSwitch({self.element}): needs at least two different type "
                             f"labels, not {types}")
        for t in types:
            if any(c in t for c in LABEL_FORBIDDEN):
                raise ValueError(f"type label {t!r} must not contain any of {LABEL_FORBIDDEN!r}")
            if element_of_label(t) != self.element:
                raise ValueError(f"TypeSwitch({self.element}): type label {t!r} is not "
                                 f"{self.element} (element_of_label)")
        n = len(types) - 1
        centre = _as_tuple(self.centre, n, "centre")
        width = _as_tuple(self.width, n, "width")
        if any(w <= 0.0 for w in width):
            raise ValueError(f"TypeSwitch({self.element}): widths must be positive")
        lo = [c - 0.5 * w for c, w in zip(centre, width)]
        hi = [c + 0.5 * w for c, w in zip(centre, width)]
        if any(hi[k] > lo[k + 1] for k in range(n - 1)):
            raise ValueError(f"TypeSwitch({self.element}): the switching windows must be "
                             "ordered and must not overlap")
        cutoff = self.cutoff
        if isinstance(cutoff, dict):
            if not cutoff:
                raise ValueError(f"TypeSwitch({self.element}): empty cutoff dict")
            out = {}
            for el, win in cutoff.items():
                if el not in atomic_numbers:
                    raise ValueError(f"TypeSwitch({self.element}) cutoff: {el!r} is not a "
                                     "chemical symbol")
                out[str(el)] = _check_window(f"TypeSwitch({self.element}) cutoff[{el!r}]", win)
            cutoff = out
        elif cutoff is not None:
            cutoff = _check_window(f"TypeSwitch({self.element}) cutoff", cutoff)
        object.__setattr__(self, "types", types)
        object.__setattr__(self, "centre", centre)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "cutoff", cutoff)

    @classmethod
    def make(cls, spec) -> "TypeSwitch":
        """A TypeSwitch, or a dict of its fields."""
        if isinstance(spec, TypeSwitch):
            return spec
        if isinstance(spec, dict):
            unknown = set(spec) - {"element", "types", "centre", "width", "cutoff"}
            if unknown:
                raise ValueError(f"type switch: unknown keys {sorted(unknown)} (use 'element', "
                                 "'types', 'centre', 'width', 'cutoff')")
            return cls(**spec)
        raise TypeError(f"a type switch must be a TypeSwitch or a dict, not {type(spec).__name__}")

    @property
    def reference(self) -> str:
        """The label the engine gives the element's atoms for everything that is equal for all
        types (basis, on-site data, per-label options): the element symbol if it is one of the
        types, else the first type."""
        return self.element if self.element in self.types else self.types[0]

    @property
    def windows(self) -> list:
        """(lower, upper) CN of every switching window."""
        return [(c - 0.5 * w, c + 0.5 * w) for c, w in zip(self.centre, self.width)]

    def weights(self, cn):
        """Weights (n, n_types) and their derivatives dw/dCN (n, n_types) for CN values."""
        cn = np.atleast_1d(np.asarray(cn, float))
        s, ds = [], []
        for lo, hi in self.windows:
            v, dv = smoothstep((cn - lo) / (hi - lo))
            s.append(v)
            ds.append(dv / (hi - lo))
        m = len(self.types)
        w = np.empty((len(cn), m))
        dw = np.empty((len(cn), m))
        w[:, 0] = 1.0 - s[0]
        dw[:, 0] = -ds[0]
        for k in range(1, m - 1):
            w[:, k] = s[k - 1] - s[k]
            dw[:, k] = ds[k - 1] - ds[k]
        w[:, m - 1] = s[m - 2]
        dw[:, m - 1] = ds[m - 2]
        return w, dw

    def windows_of(self, neighbour: str) -> tuple[float, float] | None:
        """(r_on, r_off) in Bohr of the counting function for a neighbour element, or None if
        that element does not count."""
        if isinstance(self.cutoff, dict):
            return self.cutoff.get(neighbour)
        if self.cutoff is not None:
            return self.cutoff
        rsum = (covalent_radii[atomic_numbers[self.element]]
                + covalent_radii[atomic_numbers[neighbour]]) / BOHR
        return DEFAULT_RADIUS_SCALE[0] * rsum, DEFAULT_RADIUS_SCALE[1] * rsum

    def coordination_numbers(self, elements, pos, cell=None, pbc=None, derivatives=False):
        """CN of every atom of ``element`` (0 for the others), positions in Bohr; with
        ``derivatives`` also the pairs of its gradient: (i, j, vec, f'(r)/r), i the atom whose
        CN it is, vec = r_j (+ lattice vector) - r_i."""
        elements = list(elements)
        pos = np.asarray(pos, float)
        natom = len(elements)
        cn = np.zeros(natom)
        win = {el: self.windows_of(el) for el in set(elements)}
        win = {el: w for el, w in win.items() if w is not None}
        empty = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3)), np.zeros(0))
        if not win or self.element not in elements:
            return (cn, empty) if derivatives else cn
        rc = max(w[1] for w in win.values())
        periodic = cell is not None and pbc is not None and np.any(pbc)
        i, j, _, vec, d = neighbor_list(pos, cell if periodic else None,
                                        pbc if periodic else None, rc)
        el = np.array(elements)
        r_on = np.array([win.get(e, (0.0, 0.0))[0] for e in el])
        r_off = np.array([win.get(e, (0.0, 0.0))[1] for e in el])
        keep = (el[i] == self.element) & (d < r_off[j])
        i, j, vec, d = i[keep], j[keep], vec[keep], d[keep]
        span = r_off[j] - r_on[j]
        s, ds = smoothstep((d - r_on[j]) / span)
        cn = np.bincount(i, 1.0 - s, minlength=natom)
        if not derivatives:
            return cn
        return cn, (i, j, vec, -ds / span / d)


def check_switches(switches) -> list:
    """The switches as TypeSwitch objects: at most one per element, no label in two."""
    out = [TypeSwitch.make(s) for s in switches]
    if not out:
        raise ValueError("type_switching needs at least one TypeSwitch")
    elements = [s.element for s in out]
    if len(set(elements)) != len(elements):
        raise ValueError("type_switching: one TypeSwitch per element")
    return out


def onsite_line_files(sk, switches) -> list:
    """The hetero-label files (t-u, t != u, both types of one switched element) whose second
    line is an on-site line: DFTB+ and pydftb read the format from the labels, so such a file
    (hotcent writes one for two atoms of one symbol) would be misread silently."""
    from .skf import _expand_fortran, _open_text
    bad = []
    for sw in switches:
        for t in sw.types:
            for u in sw.types:
                if t == u:
                    continue
                path = sk.file_name(t, u)
                with _open_text(path) as fh:
                    first = fh.readline()
                    if first.lstrip().startswith("@"):
                        fh.readline()
                    line = fh.readline()
                if len(_expand_fortran(line)) in _ONSITE_LINE_SIZES:
                    bad.append(path)
    return bad


def unequal_type_quantities(sk, switches) -> list:
    """What differs between the types of a switch in an SK set: basis, on-site energies,
    Hubbard values, occupations, masses (all must be equal)."""
    problems = []
    for sw in switches:
        ref = sw.reference
        for t in (x for x in sw.types if x != ref):
            if sk.max_l[t] != sk.max_l[ref]:
                problems.append(f"basis (l_max) of {t} and {ref}")
                continue
            for name in ("onsite", "hubbard", "occupation"):
                if not np.array_equal(getattr(sk, name)[t], getattr(sk, name)[ref]):
                    problems.append(f"{name} of {t} and {ref}")
            if sk.mass[t] != sk.mass[ref]:
                problems.append(f"mass of {t} and {ref}")
    return problems


def unequal_options(options: dict, switches) -> list:
    """Per-label options (``name -> {label: value}``) whose values differ between the types of a
    switch (labels missing from an option are not compared)."""
    problems = []
    for name, spec in options.items():
        if not spec:
            continue
        for sw in switches:
            given = [t for t in sw.types if t in spec]
            for t in given[1:]:
                a, b = spec[given[0]], spec[t]
                try:
                    same = np.array_equal(np.asarray(a, float), np.asarray(b, float))
                except (TypeError, ValueError):
                    same = a == b
                if not same:
                    problems.append(f"{name} of {t} and {given[0]}")
    return problems


@dataclass
class RepulsiveTerms:
    """The mixed repulsive of one geometry: energy, dE/dR at fixed weights and its virial,
    and dE/dw (n_atom, n_labels)."""
    energy: float
    gradient: np.ndarray | None
    virial: np.ndarray | None
    dedw: np.ndarray


@dataclass
class TypeMixing:
    """The type weights of one geometry and what the gradient needs (``_Setup.switching``)."""
    labels: list                  # type label of every weight column (sorted)
    weights: np.ndarray           # (n_atom, n_labels)
    dweights: np.ndarray          # (n_atom, n_labels): dw/dCN (zero for unswitched atoms)
    cn: np.ndarray                # (n_atom,) coordination numbers (NaN for unswitched atoms)
    switched: np.ndarray          # (n_atom,) bool
    candidates: dict              # internal species label -> its candidate type labels
    cn_pairs: tuple | None = None  # (i, j, vec, f'(r)/r) of every switched atom's CN
    blocks: list | None = None    # raw type-pair blocks of the pairs of active atoms
    rep: RepulsiveTerms | None = None

    @property
    def active(self) -> np.ndarray:
        """Atoms whose weights change with the geometry here (dw/dCN != 0)."""
        return np.any(self.dweights != 0.0, axis=1)

    @property
    def column(self) -> dict:
        return {lab: k for k, lab in enumerate(self.labels)}


def internal_species(switches, species) -> list:
    """The labels the engine uses for atoms given with ``species``: an atom of a switched element
    must carry the element symbol and becomes the switch's reference type (its types share the
    basis and on-site data); other labels are kept."""
    by_element = {sw.element: sw for sw in switches}
    out = []
    for lab in species:
        el = element_of_label(lab)
        if el in by_element:
            if lab != el:
                raise ValueError(f"atoms of the switched element {el} must be given as {el!r}, "
                                 f"not as the type label {lab!r} (the weights choose the types)")
            lab = by_element[el].reference
        out.append(lab)
    return out


def setup(switches, species, pos, cell, pbc, derivatives: bool):
    """(internal species, TypeMixing or None) for the labels a calculation was given
    (:func:`internal_species`); None when no atom is switched."""
    by_element = {sw.element: sw for sw in switches}
    species = list(species)
    internal = internal_species(switches, species)
    elements = [element_of_label(lab) for lab in species]
    switched = np.array([el in by_element for el in elements], bool)
    if not switched.any():
        return species, None
    present = [sw for sw in switches if sw.element in elements]
    candidates = {lab: [lab] for lab, s in zip(species, switched) if not s}
    for sw in present:
        candidates[sw.reference] = list(sw.types)
    labels = sorted({t for c in candidates.values() for t in c})
    col = {lab: k for k, lab in enumerate(labels)}
    natom = len(species)
    w = np.zeros((natom, len(labels)))
    dw = np.zeros_like(w)
    cn = np.full(natom, np.nan)
    pairs = []
    for a, (lab, s) in enumerate(zip(species, switched)):
        if not s:
            w[a, col[lab]] = 1.0
    for sw in present:
        atoms = np.flatnonzero(np.array(elements) == sw.element)
        out = sw.coordination_numbers(elements, pos, cell, pbc, derivatives=derivatives)
        c = out[0] if derivatives else out
        ww, dd = sw.weights(c[atoms])
        cols = [col[t] for t in sw.types]
        w[np.ix_(atoms, cols)] = ww
        dw[np.ix_(atoms, cols)] = dd
        cn[atoms] = c[atoms]
        if derivatives:
            pairs.append(out[1])
    cn_pairs = None
    if derivatives:
        cn_pairs = tuple(np.concatenate([p[k] for p in pairs]) for k in range(4))
    return internal, TypeMixing(labels, w, dw, cn, switched, candidates, cn_pairs)


def result_weights(switches, mix: TypeMixing | None, natom: int) -> dict:
    """``DFTBResult.type_weights``: {type label: (n_atom,) weights} for every type of every
    switch (zero for atoms of other elements)."""
    out = {}
    for sw in switches:
        for t in sw.types:
            if mix is not None and t in mix.labels:
                v = mix.weights[:, mix.column[t]].copy()
                v[~mix.switched] = 0.0       # an unswitched label is never a switch's type
                out[t] = v
            else:
                out[t] = np.zeros(natom)
    return out


def mixed_repulsive(sk, mode: str, mix: TypeMixing, pos, cell, pbc, grad: bool) -> RepulsiveTerms:
    """E_rep = 1/2 sum over ordered pairs of sum_tu w_it w_ju V^tu(r), its gradient and virial at
    fixed weights and dE/dw.  Labels with a non-zero weight somewhere, in sorted order, as
    :meth:`DFTB._repulsive_pairs`: with weights 0/1 the sums are those of the plain labels."""
    natom = len(pos)
    dedw = np.zeros_like(mix.weights)
    present = sorted(lab for lab, k in mix.column.items() if np.any(mix.weights[:, k] != 0.0))
    rc = max((r.cutoff for a in present for b in present
              if (r := sk.repulsive(a, b, mode)) is not None), default=0.0)
    gradient = np.zeros((natom, 3)) if grad else None
    virial = np.zeros((3, 3)) if grad else None
    if rc == 0.0:
        return RepulsiveTerms(0.0, gradient, virial, dedw)
    periodic = cell is not None and pbc is not None and np.any(pbc)
    i, j, _, vec, d = neighbor_list(pos, cell if periodic else None, pbc if periodic else None, rc)
    w, col = mix.weights, mix.column
    need_dw = bool(mix.active.any())
    e_tot = 0.0
    for a in present:
        for b in present:
            rep = sk.repulsive(a, b, mode)
            if rep is None:
                continue
            c = w[i, col[a]] * w[j, col[b]]
            m = c != 0.0
            if not m.any():
                continue
            cm, dm = c[m], d[m]
            e, de = rep(dm)
            e_tot += 0.5 * (cm * e).sum()
            if grad:
                g = (0.5 * cm * de / dm)[:, None] * vec[m]
                np.add.at(gradient, i[m], -g)
                np.add.at(gradient, j[m], g)
                virial += g.T @ vec[m]
            if need_dw:
                im, jm = i[m], j[m]
                np.add.at(dedw[:, col[a]], im, 0.5 * w[jm, col[b]] * e)
                np.add.at(dedw[:, col[b]], jm, 0.5 * w[im, col[a]] * e)
    return RepulsiveTerms(e_tot, gradient, virial, dedw)


def weight_gradient(mix: TypeMixing, pb, p_e, coef_s, virial: bool):
    """The part of dE/dR (n_atom, 3) and of the virial (3, 3, or None) that comes from the
    weights' dependence on the geometry: sum_A dE/dCN_A dCN_A/dR, with dE/dw from the raw
    type-pair blocks contracted with the force code's coefficients (``p_e`` of dH0, ``coef_s``
    of dS) and the repulsive's dE/dw."""
    natom = len(mix.weights)
    dedw = mix.rep.dedw.copy() if mix.rep is not None else np.zeros_like(mix.weights)
    w = mix.weights
    for it, iu, pairs, elems, hb, sb in mix.blocks or ():
        n = len(pairs)
        e_p = (p_e[elems] * hb + coef_s[elems] * sb).reshape(n, -1).sum(1)
        i, j = pb.i[pairs], pb.j[pairs]
        np.add.at(dedw[:, it], i, w[j, iu] * e_p)
        np.add.at(dedw[:, iu], j, w[i, it] * e_p)
    dedcn = (dedw * mix.dweights).sum(1)
    i, j, vec, fr = mix.cn_pairs
    g = (dedcn[i] * fr)[:, None] * vec           # dE/dvec of every CN pair
    grad = np.zeros((natom, 3))
    np.add.at(grad, j, g)
    np.add.at(grad, i, -g)
    return grad, (g.T @ vec if virial else None), dedw
