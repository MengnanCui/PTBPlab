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

A series (A0 plan, section 4), two generalisations, both opt-in (M5a's results are unchanged):

* descriptor-driven weights (channel E2, :mod:`pydftb.r0mix`): a switch with a ``mixing``
  method supplies the weights of its element's atoms from any descriptor
  (:class:`pydftb.descriptors.Features`), and their gradient through ``state.chain``;
* per-type on-site energies and Hubbard U (channel E1, :class:`OnsiteSwitch`, or an R0Switch
  with ``mix_onsite``): eps_A,l = sum_t w_A,t eps^t_l and U_A,l = sum_t w_A,t U^t_l per atom,
  the SKF values plus fitted shifts; dE/dw then also has sum_mu D_mu,mu eps^t (the on-site
  density) and sum_sites dE/dU U^t (:meth:`pydftb.coulomb.Coulomb.hubbard_gradient`).

Switches that are fitted models (:mod:`pydftb.envmodel` protocol, engine keyword
``type_switching``) get dF/d(their vector) in ``DFTBResult.switching`` (:class:`SwitchTerms`).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import ClassVar

import numpy as np
from ase.data import atomic_numbers, covalent_radii

from .constants import BOHR
from .neighbors import neighbor_list
from .system import LABEL_FORBIDDEN, element_of_label

DEFAULT_RADIUS_SCALE = (1.1, 1.3)    # (r_on, r_off) / (R_cov,A + R_cov,B)
_ONSITE_LINE_SIZES = (10, 13)        # values of an SKF on-site line (simple, extended format)
SHELL_LETTERS = "spdf"


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
        """A TypeSwitch, a dict of its fields, another switch object (a subclass, or one with a
        ``mixing`` method such as :class:`pydftb.r0mix.R0Switch`), or a model dict with a
        ``kind`` (:func:`pydftb.envmodel.model_from_dict`)."""
        if isinstance(spec, TypeSwitch) or hasattr(spec, "mixing"):
            return spec
        if isinstance(spec, dict):
            if "kind" in spec:
                from .envmodel import model_from_dict
                return model_from_dict(spec)
            unknown = set(spec) - {f.name for f in fields(cls)}
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


# ---- per-type on-site energies and Hubbard U (A2 channel E1) -----------------------------------
def shift_table(spec, types, name: str) -> tuple:
    """Shifts ``{type label: {shell letter: Hartree}}`` (or ((label, letter, value), ...)) as a
    tuple ((label, letter, value), ...) in the order of ``types`` and of s, p, d, f."""
    if spec is None:
        return ()
    if isinstance(spec, dict):
        items = []
        for lab, shells in spec.items():
            if not isinstance(shells, dict):
                raise ValueError(f"{name}[{lab!r}] must be a dict {{shell letter: Hartree}}")
            items += [(str(lab), str(letter), float(v)) for letter, v in shells.items()]
    else:
        items = [(str(a), str(b), float(c)) for a, b, c in spec]
    for lab, letter, v in items:
        if lab not in types:
            raise ValueError(f"{name}: {lab!r} is not one of the types {tuple(types)}")
        if letter not in SHELL_LETTERS:
            raise ValueError(f"{name}[{lab!r}]: shells are letters s, p, d, f, not {letter!r}")
        if not np.isfinite(v):
            raise ValueError(f"{name}[{lab!r}][{letter!r}] must be finite")
    keys = [(lab, letter) for lab, letter, _ in items]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name}: a (type, shell) is given twice")
    order = {t: k for k, t in enumerate(types)}
    return tuple(sorted(items, key=lambda x: (order[x[0]], SHELL_LETTERS.index(x[1]))))


def shift_dict(table) -> dict:
    """The dict form {label: {letter: value}} of a :func:`shift_table`."""
    out = {}
    for lab, letter, v in table:
        out.setdefault(lab, {})[letter] = v
    return out


class ShiftedOnsite:
    """The per-type shifts of the on-site energies and Hubbard U of a switch whose types may
    differ in both (A2 channel E1): ``onsite_shift`` and ``hubbard_shift`` tables
    (:func:`shift_table`) added to the SKF values.  Mixed into :class:`OnsiteSwitch` and
    :class:`pydftb.r0mix.R0Switch`."""

    def gradient_from(self, result):
        """dF/d(:meth:`vector`) in Hartree from a calculation with this switch
        (``DFTBResult.switching``); zeros for a structure without the element; None when the
        result has no gradient (``gradient`` off, or another engine)."""
        n = len(self.vector())
        terms = (getattr(result, "switching", None) or {}).get(self.element)
        if terms is not None and terms.gradient is not None and len(terms.gradient) == n:
            return np.asarray(terms.gradient, float)
        tw = getattr(result, "type_weights", None) or {}
        if self.types and all(t in tw for t in self.types) \
                and not any(np.any(tw[t]) for t in self.types):
            return np.zeros(n)                   # no atom of the element in this structure
        return None

    def shift_names(self) -> list:
        return ([f"{t}.onsite.{letter}" for t, letter, _ in self.onsite_shift]
                + [f"{t}.hubbard.{letter}" for t, letter, _ in self.hubbard_shift])

    def shift_vector(self) -> np.ndarray:
        return np.array([v for *_, v in self.onsite_shift] + [v for *_, v in self.hubbard_shift],
                        float)

    def shifts_from_vector(self, x) -> dict:
        """{onsite_shift, hubbard_shift} tables with the values of x (the order of
        :meth:`shift_vector`)."""
        x = np.asarray(x, float)
        n = len(self.onsite_shift)
        if len(x) != n + len(self.hubbard_shift):
            raise ValueError(f"needs {n + len(self.hubbard_shift)} shift values, not {len(x)}")
        return {"onsite_shift": tuple((t, l, float(v)) for (t, l, _), v in zip(self.onsite_shift, x[:n])),
                "hubbard_shift": tuple((t, l, float(v))
                                       for (t, l, _), v in zip(self.hubbard_shift, x[n:]))}

    def onsite_values(self, sk, label: str) -> np.ndarray:
        """On-site energies (per l, Hartree) of a type: the SKF's plus its shifts."""
        v = np.array(sk.onsite[label], float)
        for t, letter, x in self.onsite_shift:
            if t != label:
                continue
            l = SHELL_LETTERS.index(letter)
            if l >= len(v):
                raise ValueError(f"onsite_shift[{t!r}][{letter!r}]: {t} has no {letter} shell in "
                                 "this basis")
            v[l] += x
        return v

    def hubbard_values(self, sk, label: str) -> np.ndarray:
        """Hubbard U (per l, Hartree) of a type: the SKF's (or the customised) plus its shifts;
        with atom-resolved SCC (one U per atom) one shift of any letter moves that U."""
        v = np.array(sk.hubbard[label], float)
        given = [(letter, x) for t, letter, x in self.hubbard_shift if t == label]
        if not sk.shell_resolved:
            if len(given) > 1:
                raise ValueError(f"hubbard_shift[{label!r}]: atom-resolved SCC has one U per atom: "
                                 "give one shift")
            for _, x in given:
                v = v + x
            return v
        for letter, x in given:
            l = SHELL_LETTERS.index(letter)
            if l >= len(v):
                raise ValueError(f"hubbard_shift[{label!r}][{letter!r}]: {label} has no {letter} "
                                 "shell in this basis")
            v[l] += x
        return v


@dataclass(frozen=True)
class OnsiteSwitch(ShiftedOnsite, TypeSwitch):
    """M5a's coordination-number switch whose types may differ in on-site energies and Hubbard U
    (A2 channel E1): every atom gets eps_l = sum_t w_t (eps^t_l + onsite_shift[t][l]) and
    U_l = sum_t w_t (U^t_l + hubbard_shift[t][l]), with exact forces and stress.  The shifts
    (Hartree) are the fitted parameters (:mod:`pydftb.envmodel` protocol: ``vector()`` in the
    order of the types and s, p, d, f, names like ``Li2.onsite.s``); dF/d(shifts) comes with
    every calculation (``DFTBResult.switching``).  The windows and cut-offs stay
    hyperparameters.  Not with DFTB3, two-component spin, DFTB+U/onsite corrections, the H-X
    damping or H5, or the sparse solvers."""
    onsite_shift: object = ()
    hubbard_shift: object = ()
    gradient: bool = True                 # dF/d(shifts) in DFTBResult.switching
    engine_keyword: ClassVar[str] = "type_switching"
    mixes_onsite: ClassVar[bool] = True
    needs_dedw: ClassVar[bool] = False
    kind: ClassVar[str] = "typeswitch"

    def __post_init__(self):
        TypeSwitch.__post_init__(self)
        object.__setattr__(self, "onsite_shift",
                           shift_table(self.onsite_shift, self.types, "onsite_shift"))
        object.__setattr__(self, "hubbard_shift",
                           shift_table(self.hubbard_shift, self.types, "hubbard_shift"))

    @property
    def wants_gradient(self) -> bool:
        return bool(self.gradient) and bool(self.onsite_shift or self.hubbard_shift)

    # -- pydftb.envmodel protocol ------------------------------------------------------------
    def vector(self) -> np.ndarray:
        return self.shift_vector()

    def parameter_names(self) -> list:
        return self.shift_names()

    def with_vector(self, x) -> "OnsiteSwitch":
        return replace(self, **self.shifts_from_vector(x))

    def to_dict(self) -> dict:
        cutoff = self.cutoff
        if isinstance(cutoff, dict):
            cutoff = {el: list(w) for el, w in cutoff.items()}
        elif cutoff is not None:
            cutoff = list(cutoff)
        return {"kind": self.kind, "element": self.element, "types": list(self.types),
                "centre": list(self.centre), "width": list(self.width), "cutoff": cutoff,
                "onsite_shift": shift_dict(self.onsite_shift),
                "hubbard_shift": shift_dict(self.hubbard_shift), "gradient": self.gradient}


def onsite_switch_from_dict(d: dict) -> OnsiteSwitch:
    d = {k: v for k, v in d.items() if k != "kind"}
    c = d.get("cutoff")
    if isinstance(c, dict):
        d["cutoff"] = {el: tuple(w) for el, w in c.items()}
    elif c is not None:
        d["cutoff"] = tuple(c)
    d["types"] = tuple(d["types"])
    return OnsiteSwitch(**d)


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
    Hubbard values, occupations, masses (all must be equal; a switch that mixes on-site
    energies and U per atom, channel E1, allows those two to differ)."""
    problems = []
    for sw in switches:
        ref = sw.reference
        names = (("occupation",) if getattr(sw, "mixes_onsite", False)
                 else ("onsite", "hubbard", "occupation"))
        for t in (x for x in sw.types if x != ref):
            if sk.max_l[t] != sk.max_l[ref]:
                problems.append(f"basis (l_max) of {t} and {ref}")
                continue
            for name in names:
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
    # A series: descriptor-driven switches (r0mix) and fitted switches (E1, E2)
    states: list = field(default_factory=list)     # (switch, weight columns, state) per switch
    extra_active: np.ndarray | None = None         # (n_atom,) bool: weights moved by a state
    gradient_switches: list = field(default_factory=list)   # switches that want dF/d(vector)
    atoms_of: dict = field(default_factory=dict)   # element -> atom indices of its switch
    onsite: "OnsiteMixing | None" = None           # E1: per-atom on-site energies and U

    @property
    def active(self) -> np.ndarray:
        """Atoms whose weights change with the geometry here (dw/dCN != 0), or with the
        coefficients of a descriptor-driven switch."""
        a = np.any(self.dweights != 0.0, axis=1)
        if self.extra_active is not None:
            a = a | self.extra_active
        return a

    @property
    def needs_dedw(self) -> bool:
        """Whether dF/dw of every active atom is needed after the SCC (a fitted switch whose
        weights depend on its coefficients)."""
        return any(getattr(sw, "needs_dedw", False) for sw in self.gradient_switches)

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
    pairs, states, atoms_of = [], [], {}
    extra = np.zeros(natom, bool)
    for a, (lab, s) in enumerate(zip(species, switched)):
        if not s:
            w[a, col[lab]] = 1.0
    for sw in present:
        atoms = np.flatnonzero(np.array(elements) == sw.element)
        atoms_of[sw.element] = atoms
        cols = [col[t] for t in sw.types]
        if hasattr(sw, "mixing"):               # descriptor-driven weights (r0mix, E2)
            st = sw.mixing(elements, pos, cell, pbc, derivatives)
            if not np.array_equal(np.asarray(st.atoms), atoms):
                raise ValueError(f"the switch of {sw.element} must give weights to every "
                                 f"{sw.element} atom")
            w[np.ix_(atoms, cols)] = st.weights
            extra[atoms] |= np.asarray(st.active, bool)
            states.append((sw, cols, st))
            continue
        out = sw.coordination_numbers(elements, pos, cell, pbc, derivatives=derivatives)
        c = out[0] if derivatives else out
        ww, dd = sw.weights(c[atoms])
        w[np.ix_(atoms, cols)] = ww
        dw[np.ix_(atoms, cols)] = dd
        cn[atoms] = c[atoms]
        if derivatives:
            pairs.append(out[1])
    cn_pairs = None
    if derivatives:
        if pairs:
            cn_pairs = tuple(np.concatenate([p[k] for p in pairs]) for k in range(4))
        else:
            cn_pairs = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3)), np.zeros(0))
    grad_sw = [sw for sw in present if getattr(sw, "wants_gradient", False)]
    return internal, TypeMixing(labels, w, dw, cn, switched, candidates, cn_pairs,
                                states=states, extra_active=extra if states else None,
                                gradient_switches=grad_sw, atoms_of=atoms_of)


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


def weight_derivatives(mix: TypeMixing, pb, p_e, coef_s, extra_dedw=None) -> np.ndarray:
    """dE/dw (n_atom, n_labels) at fixed density: the raw type-pair blocks contracted with the
    force code's coefficients (``p_e`` of dH0, ``coef_s`` of dS), the repulsive's dE/dw and
    ``extra_dedw`` (E1: :func:`onsite_dedw`).  Complete for the atoms of ``mix.active`` (the
    blocks of pairs without such an atom are not kept)."""
    dedw = mix.rep.dedw.copy() if mix.rep is not None else np.zeros_like(mix.weights)
    if extra_dedw is not None:
        dedw = dedw + extra_dedw
    w = mix.weights
    for it, iu, pairs, elems, hb, sb in mix.blocks or ():
        n = len(pairs)
        e_p = (p_e[elems] * hb + coef_s[elems] * sb).reshape(n, -1).sum(1)
        i, j = pb.i[pairs], pb.j[pairs]
        np.add.at(dedw[:, it], i, w[j, iu] * e_p)
        np.add.at(dedw[:, iu], j, w[i, it] * e_p)
    return dedw


def weight_gradient(mix: TypeMixing, pb, p_e, coef_s, virial: bool, extra_dedw=None):
    """The part of dE/dR (n_atom, 3) and of the virial (3, 3, or None) that comes from the
    weights' dependence on the geometry: sum_A dE/dCN_A dCN_A/dR with dE/dw of
    :func:`weight_derivatives`, plus the descriptor-driven switches' chains (``state.chain``)."""
    natom = len(mix.weights)
    dedw = weight_derivatives(mix, pb, p_e, coef_s, extra_dedw)
    dedcn = (dedw * mix.dweights).sum(1)
    i, j, vec, fr = mix.cn_pairs
    g = (dedcn[i] * fr)[:, None] * vec           # dE/dvec of every CN pair
    grad = np.zeros((natom, 3))
    np.add.at(grad, j, g)
    np.add.at(grad, i, -g)
    vir = g.T @ vec if virial else None
    for _, cols, st in mix.states:
        gs, ws = st.chain(dedw[np.asarray(st.atoms)][:, cols], natom, virial)
        grad = grad + gs
        if virial:
            vir = vir + ws
    return grad, vir, dedw


# ---- E1: the per-atom on-site energies and Hubbard U, and the fitted switches' gradients -------
@dataclass
class OnsiteMixing:
    """Per-atom on-site energies and Hubbard U of the atoms of the switches that mix them (E1)."""
    eps: np.ndarray               # (n_orb,) on-site energies of every orbital
    hub: np.ndarray               # (n_site,) Hubbard U of every charge site
    hub_cut: np.ndarray           # (n_site,) U for the short-range cut-offs: min over the types
    parts: list                   # (switch, atoms, cols, eps_t (n_types, n_l), u_t (n_types, n_l))
    orb_l: np.ndarray             # (n_orb,) angular momentum of every orbital
    orb_atom: np.ndarray          # (n_orb,)
    site_l: np.ndarray | None     # (n_site,) l of every site (shell-resolved), None: atom sites
    site_atom: np.ndarray         # (n_site,)


def onsite_mixing(switches, mix: TypeMixing, basis, sk, eps0, hub, site_atom):
    """The on-site energies (n_orb,) and site U (n_site,) with the switched atoms of the E1
    switches mixed (``eps0``, ``hub``: the engine's values of the reference labels), or None
    when no present switch mixes them."""
    parts_sw = [sw for sw in switches if getattr(sw, "mixes_onsite", False)
                and sw.element in mix.atoms_of]
    if not parts_sw:
        return None
    orb_l = np.asarray(basis.shell_l)[np.asarray(basis.orb_shell)]
    orb_atom = np.asarray(basis.orb_atom)
    site_atom = np.asarray(site_atom)
    site_l = np.asarray(basis.shell_l) if sk.shell_resolved else None
    eps, u, ucut = np.array(eps0, float), np.array(hub, float), np.array(hub, float)
    parts = []
    for sw in parts_sw:
        atoms = mix.atoms_of[sw.element]
        cols = [mix.column[t] for t in sw.types]
        eps_t = np.array([sw.onsite_values(sk, t) for t in sw.types])
        u_t = np.array([sw.hubbard_values(sk, t) for t in sw.types])
        wts = mix.weights[np.ix_(atoms, cols)]
        orbs = np.flatnonzero(np.isin(orb_atom, atoms))
        eps[orbs] = np.einsum("ot,to->o", wts[_rows(atoms, orb_atom[orbs])], eps_t[:, orb_l[orbs]])
        sites = np.flatnonzero(np.isin(site_atom, atoms))
        sl = site_l[sites] if site_l is not None else np.zeros(len(sites), int)
        u[sites] = np.einsum("st,ts->s", wts[_rows(atoms, site_atom[sites])], u_t[:, sl])
        ucut[sites] = u_t[:, sl].min(0)
        parts.append((sw, atoms, cols, eps_t, u_t))
    return OnsiteMixing(eps, u, ucut, parts, orb_l, orb_atom, site_l, site_atom)


def _rows(atoms, idx_atom):
    row = np.full(int(max(np.max(atoms, initial=-1), np.max(idx_atom, initial=-1))) + 1, -1)
    row[atoms] = np.arange(len(atoms))
    return row[idx_atom]


def onsite_dedw(om: OnsiteMixing, mix: TypeMixing, d_onsite, dedu) -> np.ndarray:
    """E1's part of dE/dw (n_atom, n_labels): sum over the atom's orbitals of D_mu,mu eps^t
    plus sum over its sites of dE/dU U^t (``d_onsite`` (n_orb,), ``dedu`` (n_site,))."""
    out = np.zeros_like(mix.weights)
    for _, atoms, cols, eps_t, u_t in om.parts:
        c = np.zeros((len(atoms), len(cols)))
        orbs = np.flatnonzero(np.isin(om.orb_atom, atoms))
        np.add.at(c, _rows(atoms, om.orb_atom[orbs]),
                  np.asarray(d_onsite)[orbs][:, None] * eps_t[:, om.orb_l[orbs]].T)
        sites = np.flatnonzero(np.isin(om.site_atom, atoms))
        sl = om.site_l[sites] if om.site_l is not None else np.zeros(len(sites), int)
        np.add.at(c, _rows(atoms, om.site_atom[sites]),
                  np.asarray(dedu)[sites][:, None] * u_t[:, sl].T)
        out[np.ix_(atoms, cols)] += c
    return out


def shift_gradient(sw, om: OnsiteMixing, mix: TypeMixing, d_onsite, dedu) -> np.ndarray:
    """dF/d(the switch's shifts) in the order of ``sw.shift_vector()``: an on-site shift of
    (t, l) gives sum_A w_A,t sum_{mu of shell l} D_mu,mu, a U shift sum_A w_A,t dE/dU of the
    atom's sites of shell l (atom-resolved SCC: its one site)."""
    atoms = mix.atoms_of[sw.element]
    wts = mix.weights[np.ix_(atoms, [mix.column[t] for t in sw.types])]
    k_of = {t: k for k, t in enumerate(sw.types)}
    orbs = np.flatnonzero(np.isin(om.orb_atom, atoms))
    orow = _rows(atoms, om.orb_atom[orbs])
    sites = np.flatnonzero(np.isin(om.site_atom, atoms))
    srow = _rows(atoms, om.site_atom[sites])
    d_onsite, dedu = np.asarray(d_onsite), np.asarray(dedu)
    out = []
    for t, letter, _ in sw.onsite_shift:
        sel = om.orb_l[orbs] == SHELL_LETTERS.index(letter)
        out.append(float((wts[orow[sel], k_of[t]] * d_onsite[orbs[sel]]).sum()))
    for t, letter, _ in sw.hubbard_shift:
        if om.site_l is None:
            sel = np.ones(len(sites), bool)
        else:
            sel = om.site_l[sites] == SHELL_LETTERS.index(letter)
        out.append(float((wts[srow[sel], k_of[t]] * dedu[sites[sel]]).sum()))
    return np.array(out, float)


@dataclass
class SwitchTerms:
    """``DFTBResult.switching[element]``: what a fitted switch (E1, E2) needs from a calculation."""
    element: str
    atoms: np.ndarray                  # (n_c,) the element's atoms
    labels: list                       # the switch's types (weight columns below)
    weights: np.ndarray                # (n_c, n_types)
    gradient: np.ndarray | None        # dF/d(switch.vector()) (Hartree)
    dedw: np.ndarray | None = None     # (n_c, n_types) dF/dw at fixed density (Hartree)
    u: np.ndarray | None = None        # (n_c, n_axes) node coordinates (r0mix)
    r0: np.ndarray | None = None       # (n_c, 2) sum_t w_t (r0_w, r0_d)_t (r0mix, Bohr)


def switch_terms(mix: TypeMixing, dedw, d_onsite, dedu) -> dict:
    """{element: SwitchTerms} of the fitted switches: their gradients in the order of their
    ``vector()`` (descriptor-driven coefficients first, then the E1 shifts)."""
    out = {}
    by_switch = {id(sw): st for sw, _, st in mix.states}
    for sw in mix.gradient_switches:
        atoms = mix.atoms_of[sw.element]
        cols = [mix.column[t] for t in sw.types]
        st = by_switch.get(id(sw))
        parts = []
        rows = None
        if dedw is not None:
            rows = dedw[np.ix_(atoms, cols)]
        if st is not None and hasattr(st, "theta_gradient"):
            parts.append(st.theta_gradient(rows) if getattr(sw, "fit_theta", True)
                         else np.zeros(0))
        if getattr(sw, "onsite_shift", ()) or getattr(sw, "hubbard_shift", ()):
            parts.append(shift_gradient(sw, mix.onsite, mix, d_onsite, dedu))
        grad = np.concatenate(parts) if parts else np.zeros(0)
        out[sw.element] = SwitchTerms(sw.element, atoms, list(sw.types),
                                      mix.weights[np.ix_(atoms, cols)].copy(), grad, rows,
                                      getattr(st, "u", None), getattr(st, "r0", None))
    return out


def _register():
    from . import envmodel
    envmodel.register("typeswitch", onsite_switch_from_dict)


_register()
