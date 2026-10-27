"""PTBP's parameters by type (milestone M5b).

PTBP parametrises an element with three numbers: r0_w and r0_d, hotcent's confinement radii of
the wave functions and of the density, and sigma_rep, the size parameter of physrep's repulsive.
A *type* is a label with an element (the label names the SKF files, ``Cu2-Cu.skf``).  By default
every element of the dataset is one type (label = element symbol), so a dataset of N elements
has N parameter sets: 2N radii for the outer optimiser and N sigma_rep for the inner search.

* ``--shared-parameters``: one set for all elements, as PTBP did before M5b.  That layout, and
  any dataset of one element, run the code of before M5b (:attr:`TypeModel.legacy`), bit for bit.
* Several types of one element (``Cu2``, ``Cu``): every atom of the element switches smoothly
  between them with its coordination number (pydftb's milestone M5a,
  :class:`pydftb.typeswitch.TypeSwitch`); three parameters per type, the switching windows are
  fixed hyperparameters (never fitted).  Needs the pydftb engine.
* ``--multi_element E ...``: only the types of these elements are fitted; the others keep the
  parameters PTBP published (``utils.parameters_set``).

Order: elements by atomic number, the types of a switched element in the switch's order
(increasing coordination number).  The outer optimisers see (r0_w, r0_d) for every parameter
set in :attr:`TypeModel.sets` order.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

from ase.data import atomic_numbers, covalent_radii
from ase.units import Bohr

DEFAULT_P = 2.0                      # confinement power (Loss.Loss_EnergyGeometry)
R0_W_LOW, R0_D_LOW = 2.1, 3.4        # BO bounds (Loss._bo_dimensions_and_cost): r0_w in
R0_W_SCALE, R0_D_SCALE = 7.0, 9.0    # [2.1, 7 r_cov], r0_d in [3.4, 9 r_cov]
SIGMA_BOUNDS = (0.1, 0.8)            # sigma_rep search (Loss.Repulsion_optimization)
TYPED_LOSS_VERSION = 3               # utils.loss.LOSS_VERSION of the per-type layout
DEFAULT_N_CALLS = 11                 # --n-calls of before: the seed and skopt's 10 random points


def default_n_calls(n_dims: int) -> int:
    """``--n-calls`` when it is not given (M5b-2): 11 for up to 2 optimised dimensions (one
    parameter set: one element, or --shared-parameters, as before); 10 + 5 x dims above, so that
    BO gets guided steps after skopt's 10 random points (two elements, 4 dimensions: 30; three:
    40).  An explicit value (command line, YAML) always wins."""
    n_dims = int(n_dims)
    return DEFAULT_N_CALLS if n_dims <= 2 else 10 + 5 * n_dims


def energygeometry_warning(model, mode) -> str | None:
    """The warning for several types of one element with the energygeometry loss (M5b-2), or
    None.  The run goes on: the loss can still be used, but it may not constrain every type."""
    if not getattr(model, "switches", None) or str(mode or "").lower() != "energygeometry":
        return None
    which = "; ".join(f"{sw.element}: {', '.join(sw.types)}" for sw in model.switches)
    return ("WARNING: several types of one element (" + which + ") with the energygeometry "
            "loss.  energygeometry compares energies only within each EOS group, and with "
            "Boltzmann weights it drops the groups whose weight is below 0.1 (3000 K), so a type "
            "(often the low-coordination one) may get no data and be fixed by the penalty "
            "alone.  For type switching use a loss in which every structure enters and energies "
            "are compared across structures: a reaction-type loss (--mode reaction, or --mode "
            "dataset; an EOS dataset can be fitted that way too, every point then counts).  "
            "Continuing with energygeometry.")


def heaviest_radius(dataset, opt_elem=()) -> float:
    """utils.loss.penalty_radius (kept here so that this module has no import of utils.loss)."""
    if len(opt_elem) > 0:
        el = opt_elem[0]
        z = atomic_numbers[el] if isinstance(el, str) else int(el)
    else:
        zs = sorted({int(z) for atoms in dataset for z in atoms.numbers})
        if not zs:
            raise ValueError("cannot choose a penalty radius for an empty dataset")
        z = zs[-1]
    return covalent_radii[z] / Bohr


def r0_penalty(r0_w: float, r0_d: float, cov_radius: float) -> float:
    """utils.loss.r0_penalty: minimum 4 at r0_w = 2 r_cov, r0_d = 4 r_cov."""
    return ((r0_w / (2 * cov_radius) - 1)**2 + 1)**2 * ((r0_d / (4 * cov_radius) - 1)**2 + 2)**2


def r0_dlnp(r0_w: float, r0_d: float, cov_radius: float) -> tuple:
    """(d ln P / d r0_w, d ln P / d r0_d) of :func:`r0_penalty` (1/Bohr)."""
    a = r0_w / (2 * cov_radius) - 1
    b = r0_d / (4 * cov_radius) - 1
    return 2 * a / (cov_radius * (a * a + 1)), b / (cov_radius * (b * b + 2))


@dataclass(frozen=True)
class PType:
    """One type: its label, element, and the parameters of a type that is not fitted."""
    label: str
    element: str
    fixed: tuple | None = None        # (r0_w, r0_d, sigma_rep); None: fitted

    @property
    def number(self) -> int:
        return atomic_numbers[self.element]

    @property
    def radius(self) -> float:
        """ASE's covalent radius in Bohr (the radius of the penalty and the bounds)."""
        return covalent_radii[self.number] / Bohr


@dataclass(frozen=True)
class ParameterSet:
    """(r0_w, r0_d, sigma_rep) of one or more types."""
    name: str                         # the type's label, or "shared"
    labels: tuple
    radius: float                     # Bohr: bounds and penalty
    start: tuple                      # (r0_w, r0_d): the optimisers' starting point


# ---------------------------------------------------------------------------
# declarations of switched types (CLI, YAML)
def switch_from_declaration(decl: dict):
    """A :class:`pydftb.typeswitch.TypeSwitch` from a declaration
    ``{"element", "labels", "centre", "width", "cutoff"?, "start"?}`` (``types`` is accepted
    for ``labels``); the windows and cut-offs are in pydftb's units (CN, Bohr)."""
    from pydftb.typeswitch import TypeSwitch
    unknown = set(decl) - {"element", "labels", "types", "centre", "width", "cutoff", "start"}
    if unknown:
        raise ValueError(f"type declaration: unknown keys {sorted(unknown)} (use element, labels, "
                         "centre, width, cutoff, start)")
    labels = decl.get("labels", decl.get("types"))
    if labels is None or "element" not in decl or "centre" not in decl or "width" not in decl:
        raise ValueError("a type declaration needs element, labels, centre and width, e.g. "
                         "{'element': 'Cu', 'labels': ['Cu2', 'Cu'], 'centre': 11.0, 'width': 1.0}")
    cutoff = decl.get("cutoff")
    if isinstance(cutoff, dict):
        cutoff = {str(k): tuple(float(x) for x in v) for k, v in cutoff.items()}
    elif cutoff is not None:
        cutoff = tuple(float(x) for x in cutoff)
    return TypeSwitch(str(decl["element"]), tuple(str(t) for t in labels),
                      centre=decl["centre"], width=decl["width"], cutoff=cutoff)


def declaration_of(sw) -> dict:
    """The declaration (JSON-able) of a TypeSwitch."""
    out = {"element": sw.element, "labels": list(sw.types), "centre": list(sw.centre),
           "width": list(sw.width)}
    if isinstance(sw.cutoff, dict):
        out["cutoff"] = {k: list(v) for k, v in sw.cutoff.items()}
    elif sw.cutoff is not None:
        out["cutoff"] = list(sw.cutoff)
    return out


def parse_switch_args(items) -> list:
    """Declarations from ``--switch ELEMENT LABELS CENTRE WIDTH`` (each a list of 4 strings,
    LABELS/CENTRE/WIDTH comma-separated), e.g. ``Cu Cu2,Cu 11.0 1.0``."""
    out = []
    for item in items or ():
        if len(item) != 4:
            raise ValueError(f"--switch takes ELEMENT LABELS CENTRE WIDTH, not {item}")
        el, labels, centre, width = item
        out.append({"element": el, "labels": [s for s in labels.split(",") if s],
                    "centre": [float(c) for c in centre.split(",") if c],
                    "width": [float(w) for w in width.split(",") if w]})
    return out


def parse_switch_text(text: str) -> list:
    """Declarations from the compact form ``ELEMENT:LABELS:CENTRE:WIDTH[; ...]`` (the TUI),
    e.g. ``Cu:Cu2,Cu:11.0:1.0``."""
    items = []
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        fields = [f.strip() for f in part.split(":")]
        if len(fields) != 4:
            raise ValueError(f"switched types: ELEMENT:LABELS:CENTRE:WIDTH, not {part!r}")
        items.append(fields)
    return parse_switch_args(items)


def parse_starts(items) -> dict:
    """{label: (r0_w, r0_d)} from ``--type-start LABEL R0_W R0_D`` items."""
    out = {}
    for item in items or ():
        if len(item) != 3:
            raise ValueError(f"--type-start takes LABEL R0_W R0_D, not {item}")
        out[str(item[0])] = (float(item[1]), float(item[2]))
    return out


def load_declarations(value) -> list:
    """Declarations from YAML/JSON: a list of dicts, a JSON string of one, or None."""
    if value is None or value == "" or value == []:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        value = [value]
    return [dict(d) for d in value]


# ---------------------------------------------------------------------------
class TypeModel:
    """The types of a dataset and their parameter sets (see the module docstring)."""

    def __init__(self, types, sets, switches=(), shared: bool = False):
        self.types = tuple(types)
        self.sets = tuple(sets)
        self.switches = tuple(switches)
        self.shared = bool(shared)
        labels = [t.label for t in self.types]
        if len(set(labels)) != len(labels):
            raise ValueError(f"type labels must be unique: {labels}")
        self._set_of = {lab: k for k, s in enumerate(self.sets) for lab in s.labels}

    # ---- construction ------------------------------------------------------------
    @classmethod
    def from_dataset(cls, dataset, *, shared: bool = False, declarations=None, starts=None,
                     multi_element=None) -> "TypeModel":
        """The types of ``dataset`` (ASE Atoms): its elements, split into the declared switched
        types; ``multi_element``: the elements whose types are fitted (the others are fixed at
        PTBP's table); ``starts``: {label: (r0_w, r0_d)} starting points.  ValueError for
        layouts that cannot run, among them ``shared`` with a ``multi_element`` that lists every
        element of a dataset of several elements (M5b-4)."""
        from pydftb.typeswitch import check_switches

        from utils.parameters_set import get_PTBP
        symbols = sorted({s for a in dataset for s in a.get_chemical_symbols()},
                         key=lambda s: atomic_numbers[s])
        if not symbols:
            raise ValueError("the dataset has no atoms")
        decls = load_declarations(declarations)
        switches = check_switches([switch_from_declaration(d) for d in decls]) if decls else []
        by_element = {sw.element: sw for sw in switches}
        starts = dict(starts or {})
        for d in decls:
            for lab, v in (d.get("start") or {}).items():
                starts.setdefault(str(lab), tuple(float(x) for x in v))
        missing = sorted(set(by_element) - set(symbols))
        if missing:
            raise ValueError(f"types declared for {missing}, which the dataset does not contain")
        if shared and switches:
            raise ValueError("--shared-parameters gives every type the same parameters: the "
                             "declared types of " + ", ".join(by_element) + " would be identical")
        fitted = list(dict.fromkeys(multi_element)) if multi_element else list(symbols)
        bad = [e for e in fitted if e not in symbols]
        if bad:
            raise ValueError(f"--multi_element: {bad} not in the dataset ({symbols})")
        if shared and multi_element and len(symbols) > 1 and set(fitted) == set(symbols):
            # M5b-4: nothing left at PTBP's parameters is the layout of before M5b (legacy), whose
            # code reads --multi_element in its old sense and gives the confinement radii to the
            # first listed element only: the run could not work
            raise ValueError(
                f"--shared-parameters cannot be combined with a --multi-element that lists every "
                f"element of the dataset ({', '.join(symbols)}): nothing would keep PTBP's "
                f"parameters, and the code of one parameter set would give the radii to "
                f"{fitted[0]} only.  Use --shared-parameters alone (one set for all elements), "
                f"--multi-element alone (one set per listed element), or list only the elements "
                f"to fit (with --shared-parameters they share one set, the others keep PTBP's "
                f"published parameters)")
        types = []
        for el in symbols:
            fixed = None
            if el not in fitted:
                entry = get_PTBP().get(el)
                if entry is None:
                    raise ValueError(f"{el} keeps PTBP's parameters (--multi_element), but PTBP's "
                                     "table has no entry for it")
                fixed = (float(entry["r0_w"]), float(entry["r0_d"]), float(entry["sigma_rep"]))
            for lab in (by_element[el].types if el in by_element else (el,)):
                types.append(PType(lab, el, fixed))
        unknown = sorted(set(starts) - {t.label for t in types})
        if unknown:
            raise ValueError(f"starting values for unknown types {unknown}")
        fitted_types = [t for t in types if t.fixed is None]
        if shared:
            radius = heaviest_radius(dataset, multi_element or ())
            els = sorted({t.element for t in fitted_types})
            if len(els) == 1 and els[0] in get_PTBP():
                p = get_PTBP()[els[0]]
                start = (float(p["r0_w"]), float(p["r0_d"]))
            else:
                start = (radius * 3, radius * 5)
            given = [starts[t.label] for t in fitted_types if t.label in starts]
            if given:
                start = given[0]
            sets = [ParameterSet("shared", tuple(t.label for t in fitted_types), radius,
                                 _clip_start(start, radius))]
        else:
            sets = []
            for t in fitted_types:
                if t.label in starts:
                    start = starts[t.label]
                elif t.element in get_PTBP():
                    p = get_PTBP()[t.element]
                    start = (float(p["r0_w"]), float(p["r0_d"]))
                else:
                    start = (t.radius * 3, t.radius * 5)
                sets.append(ParameterSet(t.label, (t.label,), t.radius, _clip_start(start, t.radius)))
        return cls(types, sets, switches, shared=shared)

    # ---- properties --------------------------------------------------------------
    @property
    def legacy(self) -> bool:
        """One parameter set for every type, nothing fixed, no switching: the layout of PTBP
        before M5b (one element, or --shared-parameters), which runs that code bit for bit."""
        return (not self.switches and len(self.sets) == 1
                and all(t.fixed is None for t in self.types))

    @property
    def labels(self) -> list:
        return [t.label for t in self.types]

    @property
    def element_of(self) -> dict:
        return {t.label: t.element for t in self.types}

    @property
    def number_of(self) -> dict:
        return {t.label: t.number for t in self.types}

    @property
    def n_params(self) -> int:
        return 2 * len(self.sets)

    @property
    def n_sigma(self) -> int:
        return len(self.sets)

    @property
    def layout(self) -> str:
        return "shared" if self.shared else "per_type"

    def set_index(self, label: str):
        """Index of the parameter set of a type (None: a fixed type)."""
        return self._set_of.get(label)

    def labels_of_element(self, element: str) -> list:
        return [t.label for t in self.types if t.element == element]

    # ---- parameter vectors -------------------------------------------------------
    def dimension_names(self) -> list:
        if len(self.sets) == 1:
            return ["r0_w", "r0_d"]
        return [f"{p}_{s.name}" for s in self.sets for p in ("r0_w", "r0_d")]

    def bounds(self) -> tuple:
        lo, hi = [], []
        for s in self.sets:
            lo += [R0_W_LOW, R0_D_LOW]
            hi += [R0_W_SCALE * s.radius, R0_D_SCALE * s.radius]
        return lo, hi

    def seed(self) -> list:
        return [float(v) for s in self.sets for v in s.start]

    def _check(self, x, n, what):
        x = [float(v) for v in x]
        if len(x) != n:
            raise ValueError(f"{what}: {n} values expected ({self.describe_sets()}), got {len(x)}")
        return x

    def r0(self, x) -> dict:
        """{label: (r0_w, r0_d)} of every type at the outer parameters ``x``."""
        x = self._check(x, self.n_params, "confinement radii")
        out = {}
        for t in self.types:
            k = self.set_index(t.label)
            out[t.label] = (t.fixed[0], t.fixed[1]) if k is None else (x[2 * k], x[2 * k + 1])
        return out

    def sigma(self, svec) -> dict:
        """{label: sigma_rep} of every type for the inner variables ``svec`` (one per set)."""
        svec = self._check(list(_flat(svec)), self.n_sigma, "sigma_rep")
        return {t.label: (t.fixed[2] if self.set_index(t.label) is None
                          else svec[self.set_index(t.label)]) for t in self.types}

    def confinement(self, x) -> tuple:
        """(confinement_r0 {label: [r0_w, r0_d]}, confinement_p {label: p}) for hotpy."""
        r0 = self.r0(x)
        return ({lab: [w, d] for lab, (w, d) in r0.items()},
                {lab: DEFAULT_P for lab in r0})

    def penalties(self, x) -> list:
        x = self._check(x, self.n_params, "confinement radii")
        return [r0_penalty(x[2 * k], x[2 * k + 1], s.radius) for k, s in enumerate(self.sets)]

    def penalty(self, x) -> float:
        """The regularisation factor: the geometric mean of the sets' penalties (one set: its
        penalty; minimum 4 in every case)."""
        p = self.penalties(x)
        return p[0] if len(p) == 1 else math.prod(p) ** (1.0 / len(p))

    def dlnp(self, x) -> list:
        """d ln P / dx (1/Bohr), x in :meth:`dimension_names` order."""
        x = self._check(x, self.n_params, "confinement radii")
        out = []
        for k, s in enumerate(self.sets):
            dw, dd = r0_dlnp(x[2 * k], x[2 * k + 1], s.radius)
            out += [dw / len(self.sets), dd / len(self.sets)]
        return out

    def directory_name(self, x, digits: int = 3) -> str:
        """The BO step's directory: opt_<r0_w>_<r0_d>[_<r0_w>_<r0_d> ...] (sets order)."""
        x = self._check(x, self.n_params, "confinement radii")
        return "opt_" + "_".join(f"{v:.{digits}f}" for v in x)

    # ---- text --------------------------------------------------------------------
    def describe_sets(self) -> str:
        return ", ".join(f"{s.name}: r0_w r0_d" for s in self.sets)

    def describe(self) -> str:
        lines = [f"types ({'shared parameters' if self.shared else 'parameters per type'}):"]
        for t in self.types:
            k = self.set_index(t.label)
            what = (f"fixed at PTBP's r0_w={t.fixed[0]}, r0_d={t.fixed[1]}, "
                    f"sigma_rep={t.fixed[2]}" if k is None else f"set {self.sets[k].name}")
            lines.append(f"  {t.label:<8s} {t.element:<3s} r_cov {t.radius:.4f} Bohr  {what}")
        for sw in self.switches:
            lines.append(f"  switch {sw.element}: {' -> '.join(sw.types)} at CN windows "
                         f"{[tuple(round(v, 4) for v in w) for w in sw.windows]}"
                         + ("" if sw.cutoff is None else f", cutoff {sw.cutoff}"))
        lo, hi = self.bounds()
        for k, s in enumerate(self.sets):
            lines.append(f"  set {s.name}: r0_w {lo[2 * k]} ~ {hi[2 * k]:.4f}, r0_d {lo[2 * k + 1]} ~ "
                         f"{hi[2 * k + 1]:.4f}, start {tuple(round(v, 4) for v in s.start)}")
        return "\n".join(lines)

    def par_header(self) -> str:
        cols = [f"{p}_{lab}" for lab in self.labels for p in ("r0_w", "r0_d", "sigma_rep")]
        return "#" + ",".join(cols + ["penalty", "mu"]) + "\n"

    def log_header(self, mode: str) -> str:
        cols = [f"{p}_{lab}" for lab in self.labels for p in ("r0_w", "r0_d", "sigma_rep")]
        tail = ["mu_E", "mu_V", "mu_B"] if mode == "energygeometry" else ["mu_E", "mu_F", "mu_B"]
        return "#" + ",".join(cols + tail + ["penalty", "mu"]) + "\n"

    def _columns(self, x, svec) -> str:
        r0, sig = self.r0(x), self.sigma(svec)
        return ", ".join(f"{r0[lab][0]:15.8f}, {r0[lab][1]:15.8f}, {sig[lab]:15.8f}"
                         for lab in self.labels)

    def par_line(self, x, svec, penalty: float, mu: float) -> str:
        return f"{self._columns(x, svec)}, {penalty:15.5f}, {mu:25.11f}\n"

    def log_line(self, x, svec, mu_e, mu_v, mu_b, penalty: float, mu: float) -> str:
        return (f"{self._columns(x, svec)}, {mu_e:15.8f}, {mu_v:20.8f}, {mu_b: 20.8f}, "
                f"{penalty:15.5f}, {mu:25.11f}\n")

    def parameter_text(self, x, svec) -> str:
        r0, sig = self.r0(x), self.sigma(svec)
        return " ".join(f"{lab}:r0_w={r0[lab][0]:.6f},r0_d={r0[lab][1]:.6f},sigma_rep={sig[lab]:.6f}"
                        for lab in self.labels)

    # ---- checkpoints -------------------------------------------------------------
    def meta(self, mode: str) -> dict:
        """What a checkpoint of this layout records (utils.loss, loss version 3)."""
        return {"ptbp_loss_version": TYPED_LOSS_VERSION, "mode": mode, "layout": self.layout,
                "penalty": "geometric mean over the parameter sets",
                "sets": [{"name": s.name, "labels": list(s.labels), "radius_bohr": s.radius}
                         for s in self.sets],
                "fixed": {t.label: list(t.fixed) for t in self.types if t.fixed is not None},
                "switches": [declaration_of(sw) for sw in self.switches]}

    def matches_meta(self, meta) -> bool:
        if not isinstance(meta, dict) or meta.get("ptbp_loss_version") != TYPED_LOSS_VERSION:
            return False
        mine = self.meta(meta.get("mode", ""))
        if meta.get("layout") != mine["layout"] or meta.get("switches") != mine["switches"]:
            return False
        if meta.get("fixed") != mine["fixed"]:
            return False
        sets = meta.get("sets") or []
        return (len(sets) == len(mine["sets"])
                and all(a.get("name") == b["name"] and a.get("labels") == b["labels"]
                        and abs(a.get("radius_bohr", -1.0) - b["radius_bohr"]) < 1e-5
                        for a, b in zip(sets, mine["sets"])))


def _clip_start(start, radius) -> tuple:
    """A starting point inside the BO box (PTBP's table has r0_d below 3.4 Bohr for some
    light elements, e.g. N 2.538)."""
    w = min(max(float(start[0]), R0_W_LOW), R0_W_SCALE * radius)
    d = min(max(float(start[1]), R0_D_LOW), R0_D_SCALE * radius)
    return (w, d)


def _flat(svec):
    try:
        return [float(v) for v in svec]
    except TypeError:
        return [float(svec)]
