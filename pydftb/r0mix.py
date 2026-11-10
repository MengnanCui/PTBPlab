"""Continuous confinement radii per atom: SK tables mixed between r0 nodes (A2 channel E2).

The idea of DOVE (environment-dependent r0 decides the SK tables), with exact forces and stress.
Every atom A of ``element`` gets a node coordinate u_A from a descriptor D_A of its environment:

    u_A = clamp(u_ref + f(D_A; theta))          f a pydftb.regressors regressor,

clamped smoothly (C3) to the range where the cubic B-splines of the nodes sum to one.  The node
labels t (element + digits, DFTB+'s Type2FileNames convention) are SK types with their own
confinement radii (r0_w, r0_d)_t, one SK set holding the files of every label pair
(:class:`utils.skgrid.TypedSKCache` writes them).  The atom's weights are the cubic (C2)
B-spline basis of the nodes,

    w_A,t = B3(u_A - t)        (1-D: nodes on a line; 2-D: tensor products over a grid),

non-negative and summing to one, and M5a's mixing does the rest (:mod:`pydftb.typeswitch`):
H0_AB = sum_tu w_A,t w_B,u H^tu_AB, S likewise, the repulsive likewise.  So the tables of an
atom are the B-spline quasi-interpolant of the node tables at its u: at u = k they are
(T_k-1 + 4 T_k + T_k+1) / 6, at most four nodes (sixteen on a grid) contribute, and with linearly
spaced nodes the mean radius sum_t w_t r0_t is exactly the linear r0(u).

Forces and stress: dE/dw (variational, at fixed density) times dw/du_raw (the B-splines and the
clamp) times df/dD (the regressor) through the descriptor's gradient pairs
(:meth:`pydftb.descriptors.Features.chain`).  Fitting: dF/dtheta = sum_A,t dF/dw_A,t dw_A,t/du_A
du_A/dtheta comes with every calculation (``DFTBResult.switching``), so an R0Switch is an
:mod:`pydftb.envmodel` model (engine keyword ``type_switching``, kind ``r0mix``).  With
``mix_onsite`` (or E1 shifts) the nodes may also differ in on-site energies and Hubbard U, mixed
per atom (:class:`pydftb.typeswitch.ShiftedOnsite`); PTBP's SK files have the free atom's values
at every r0, so this is normally off.

    sw = R0Switch.around("Li", r0=(4.2, 9.1), spacing=(0.3, 0.6), n=7)    # labels Li1 .. Li7
    calc = DFTB(sk, type_switching=[sw])                                  # sk: every label pair
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from .descriptors import CoordinationNumber, descriptor_from_dict
from .regressors import Linear, regressor_from_dict
from .system import element_of_label
from .typeswitch import ShiftedOnsite, shift_dict, shift_table, smoothstep


# ---- the cubic B-spline basis and the clamp ------------------------------------------------------
def bspline3(x):
    """The centred cubic B-spline B3(x) (support [-2, 2], C2) and its derivative."""
    x = np.asarray(x, float)
    a = np.abs(x)
    b = np.zeros_like(x)
    db = np.zeros_like(x)
    m = a < 1.0
    b[m] = 2.0 / 3.0 - a[m] ** 2 + 0.5 * a[m] ** 3
    db[m] = -2.0 * x[m] + 1.5 * x[m] * a[m]
    m = (a >= 1.0) & (a < 2.0)
    b[m] = (2.0 - a[m]) ** 3 / 6.0
    db[m] = -np.sign(x[m]) * 0.5 * (2.0 - a[m]) ** 2
    return b, db


def smooth_clamp(x, lo: float, hi: float, d: float):
    """x clamped to [lo, hi], C3: the identity on [lo + d, hi - d], constant beyond lo - d and
    hi + d; in between its slope is M5a's smoothstep, y = lo + 2d P(t) with P = integral of S.
    Returns y and dy/dx."""
    if not hi - lo >= 2.0 * d > 0.0:
        raise ValueError("smooth_clamp needs hi - lo >= 2 d > 0")
    x = np.asarray(x, float)
    y = x.copy()
    dy = np.ones_like(x)

    def p(t):
        return t ** 4 * (t * (t - 3.0) + 2.5)

    lo_band = x < lo + d
    t = np.clip((x[lo_band] - (lo - d)) / (2.0 * d), 0.0, 1.0)
    y[lo_band] = lo + 2.0 * d * p(t)
    dy[lo_band] = smoothstep(t)[0]
    hi_band = x > hi - d
    t = np.clip(((hi + d) - x[hi_band]) / (2.0 * d), 0.0, 1.0)
    y[hi_band] = hi - 2.0 * d * p(t)
    dy[hi_band] = smoothstep(t)[0]
    return y, dy


# ---- the state of one geometry -------------------------------------------------------------------
@dataclass
class R0State:
    """The weights of one geometry (``typeswitch.setup``) and what forces and fits need."""
    switch: "R0Switch"
    atoms: np.ndarray            # (n_c,) the element's atoms
    x: np.ndarray                # (n_c, n_in) descriptor values
    features: object             # pydftb.descriptors.Features (pairs with derivatives)
    u: np.ndarray                # (n_c, n_axes) clamped node coordinates
    weights: np.ndarray          # (n_c, n_types)
    dwdu: np.ndarray             # (n_c, n_types, n_axes) dw / du_raw (clamp included)
    r0: np.ndarray               # (n_c, 2) sum_t w_t (r0_w, r0_d)_t

    @property
    def active(self) -> np.ndarray:
        """Atoms whose weights move with the geometry or the coefficients."""
        return np.any(self.dwdu != 0.0, axis=(1, 2))

    def _dedu(self, dedw):
        return np.einsum("ct,cta->ca", np.asarray(dedw, float), self.dwdu)

    def chain(self, dedw, natom: int, virial: bool):
        """dE/dR and the virial from dE/dw (n_c, n_types) of these atoms."""
        sw = self.switch
        dedu = self._dedu(dedw)
        dedx = np.einsum("ca,cai->ci", dedu, sw.regressor.d_input(self.x, sw.theta_array))
        return self.features.chain(dedx, natom, virial)

    def theta_gradient(self, dedw) -> np.ndarray:
        """dE/dtheta from dE/dw (n_c, n_types) of these atoms."""
        sw = self.switch
        if dedw is None:
            raise ValueError("dE/dw is needed for the coefficients' gradient")
        dedu = self._dedu(dedw)
        return np.einsum("ca,cak->k", dedu, sw.regressor.d_theta(self.x, sw.theta_array))


# ---- the model -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class R0Switch(ShiftedOnsite):
    """Environment-dependent confinement radii of the atoms of ``element`` (A2 channel E2).

    types      : the node labels (element + digits), in node order (1-D) or row-major over
                 ``grid`` (2-D: index a * n_d + b for node (a, b)).
    nodes      : (r0_w, r0_d) of every node in Bohr (the radii of the SK set's types; reports).
    descriptor : a :mod:`pydftb.descriptors` descriptor of ``element`` (default: CN).
    regressor  : a :mod:`pydftb.regressors` regressor, n_in = descriptor.size, n_out = the
                 number of node axes (default: Linear).
    theta      : its coefficients (default: ``regressor.initial()``, u = u_ref everywhere).
    u_ref      : the node coordinate(s) at f = 0 (default: the middle of the nodes).
    grid       : (n_w, n_d) for a tensor-product grid of nodes; None: one line of nodes.
    margin     : the width d of the clamp to [1, n - 2] per axis (the range where the
                 B-splines sum to one; nodes 0 and n - 1 are guards).
    mix_onsite : the nodes may differ in on-site energies and U (mixed per atom, E1); also on
                 when shifts are given.
    onsite_shift, hubbard_shift : E1 shifts per node label (Hartree), fitted as well.
    fit_theta  : the coefficients are in ``vector()`` (False: only the shifts are fitted).
    gradient   : dF/d(vector) in every result (``DFTBResult.switching``; off for production MD).
    """
    element: str
    types: tuple
    nodes: tuple
    descriptor: object = None
    regressor: object = None
    theta: tuple | None = None
    u_ref: tuple | float | None = None
    grid: tuple | None = None
    margin: float = 0.5
    mix_onsite: bool = False
    onsite_shift: object = ()
    hubbard_shift: object = ()
    fit_theta: bool = True
    gradient: bool = True
    engine_keyword: ClassVar[str] = "type_switching"
    kind: ClassVar[str] = "r0mix"

    def __post_init__(self):
        types = tuple(str(t) for t in self.types)
        if len(set(types)) != len(types):
            raise ValueError("R0Switch: the node labels must differ")
        for t in types:
            if element_of_label(t) != self.element:
                raise ValueError(f"R0Switch({self.element}): node label {t!r} is not "
                                 f"{self.element} (element_of_label)")
        if self.element in types:
            raise ValueError(f"R0Switch({self.element}): the atoms carry the label "
                             f"{self.element!r}; name the nodes {self.element}1, ... instead")
        nodes = tuple(tuple(float(v) for v in n) for n in self.nodes)
        if len(nodes) != len(types) or any(len(n) != 2 for n in nodes):
            raise ValueError("R0Switch: one (r0_w, r0_d) per node label")
        shape = (len(types),) if self.grid is None else tuple(int(n) for n in self.grid)
        if int(np.prod(shape)) != len(types) or len(shape) not in (1, 2):
            raise ValueError(f"R0Switch: grid {shape} does not hold {len(types)} nodes")
        if not self.margin > 0.0 or any(n - 3 < 2.0 * self.margin for n in shape):
            raise ValueError("R0Switch: every axis needs n >= 3 + 2 margin nodes (the cubic "
                             "B-splines sum to one on [1, n - 2])")
        desc = self.descriptor if self.descriptor is not None else CoordinationNumber(self.element)
        if desc.element != self.element:
            raise ValueError(f"R0Switch({self.element}): the descriptor describes {desc.element}")
        reg = self.regressor if self.regressor is not None else Linear(desc.size, len(shape))
        if reg.n_in != desc.size or reg.n_out != len(shape):
            raise ValueError(f"R0Switch: the regressor maps {reg.n_in} -> {reg.n_out}, the "
                             f"descriptor gives {desc.size} values and the nodes have "
                             f"{len(shape)} axes")
        theta = reg.initial() if self.theta is None else np.asarray(self.theta, float)
        if theta.shape != (reg.size,):
            raise ValueError(f"R0Switch: theta needs {reg.size} values")
        u_ref = (tuple(0.5 * (n - 1) for n in shape) if self.u_ref is None
                 else tuple(float(v) for v in np.atleast_1d(self.u_ref)))
        if len(u_ref) != len(shape):
            raise ValueError("R0Switch: u_ref needs one value per node axis")
        for name, value in (("types", types), ("nodes", nodes), ("descriptor", desc),
                            ("regressor", reg), ("theta", tuple(float(v) for v in theta)),
                            ("u_ref", u_ref), ("grid", None if self.grid is None else shape)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "onsite_shift",
                           shift_table(self.onsite_shift, types, "onsite_shift"))
        object.__setattr__(self, "hubbard_shift",
                           shift_table(self.hubbard_shift, types, "hubbard_shift"))

    # -- constructors ----------------------------------------------------------------------------
    @classmethod
    def around(cls, element: str, r0, spacing, n: int = 7, grid: bool = False,
               descriptor=None, regressor=None, **kw) -> "R0Switch":
        """Nodes centred on the radii ``r0`` = (r0_w, r0_d) in Bohr (e.g. an A1 best point).
        1-D: n nodes r0 + (k - (n-1)/2) spacing, spacing = (d_w, d_d) (a number: d_w, with
        d_d = d_w r0_d / r0_w), labels <element>1 .. <element>n; ``grid``: n x n nodes with
        (r0_w + (a - c) d_w, r0_d + (b - c) d_d), labels <element><a+1><b+1> (n <= 9).
        With an odd n the middle node is r0 and u_ref sits on it."""
        r0 = tuple(float(v) for v in r0)
        if np.ndim(spacing) == 0:
            spacing = (float(spacing), float(spacing) * r0[1] / r0[0])
        dw, dd = (float(v) for v in spacing)
        c = 0.5 * (n - 1)
        if not grid:
            nodes = tuple((r0[0] + (k - c) * dw, r0[1] + (k - c) * dd) for k in range(n))
            labels = tuple(f"{element}{k + 1}" for k in range(n))
            shape = None
        else:
            if n > 9:
                raise ValueError("R0Switch.around(grid=True): at most 9 nodes per axis")
            nodes = tuple((r0[0] + (a - c) * dw, r0[1] + (b - c) * dd)
                          for a in range(n) for b in range(n))
            labels = tuple(f"{element}{a + 1}{b + 1}" for a in range(n) for b in range(n))
            shape = (n, n)
        if min(min(v) for v in nodes) <= 0.0:
            raise ValueError("R0Switch.around: a node radius is not positive")
        return cls(element, labels, nodes, descriptor=descriptor, regressor=regressor,
                   grid=shape, **kw)

    # -- what typeswitch and the calculator need -------------------------------------------------
    @property
    def reference(self) -> str:
        """The label of the element's atoms inside the engine (basis and on-site data)."""
        return self.types[0]

    @property
    def shape(self) -> tuple:
        return (len(self.types),) if self.grid is None else tuple(self.grid)

    @property
    def mixes_onsite(self) -> bool:
        return bool(self.mix_onsite or self.onsite_shift or self.hubbard_shift)

    @property
    def needs_dedw(self) -> bool:
        return bool(self.fit_theta)

    @property
    def wants_gradient(self) -> bool:
        return bool(self.gradient) and len(self.vector()) > 0

    @property
    def theta_array(self) -> np.ndarray:
        return np.asarray(self.theta, float)

    def coordinates(self, x):
        """Node coordinates u (n, n_axes), clamped, and du/du_raw, for descriptor values x."""
        raw = np.asarray(self.u_ref)[None, :] + self.regressor.value(x, self.theta_array)
        u = np.empty_like(raw)
        du = np.empty_like(raw)
        for a, n in enumerate(self.shape):
            u[:, a], du[:, a] = smooth_clamp(raw[:, a], 1.0, n - 2.0, self.margin)
        return u, du

    def spline_weights(self, u):
        """Weights (n, n_types) and dw/du (n, n_types, n_axes) of node coordinates u."""
        u = np.atleast_2d(np.asarray(u, float))
        bases = []
        for a, n in enumerate(self.shape):
            b, db = bspline3(u[:, a][:, None] - np.arange(n)[None, :])
            bases.append((b, db))
        if len(bases) == 1:
            b, db = bases[0]
            return b, db[:, :, None]
        (bw, dbw), (bd, dbd) = bases
        n = len(u)
        w = (bw[:, :, None] * bd[:, None, :]).reshape(n, -1)
        dw = np.stack([(dbw[:, :, None] * bd[:, None, :]).reshape(n, -1),
                       (bw[:, :, None] * dbd[:, None, :]).reshape(n, -1)], axis=2)
        return w, dw

    def mixing(self, elements, pos, cell=None, pbc=None, derivatives: bool = False) -> R0State:
        """The weights of the element's atoms in one geometry (positions in Bohr)."""
        if derivatives and not self.descriptor.has_derivatives:
            raise ValueError(f"R0Switch({self.element}): forces need a descriptor with "
                             "derivatives (this one only drives band fits at frozen positions)")
        f = self.descriptor.compute(elements, pos, cell, pbc, derivatives=derivatives)
        x = np.asarray(f.values, float)
        u, du = self.coordinates(x)
        w, dwdu = self.spline_weights(u)
        dwdu = dwdu * du[:, None, :]
        r0 = w @ np.asarray(self.nodes)
        return R0State(self, np.asarray(f.atoms), x, f, u, w, dwdu, r0)

    def evaluate(self, atoms):
        """(u (n_c, n_axes), r0 (n_c, 2) Bohr, atom indices) of an ASE Atoms object."""
        from .constants import BOHR
        periodic = bool(np.any(atoms.pbc))
        st = self.mixing(atoms.get_chemical_symbols(), atoms.positions / BOHR,
                         atoms.cell.array / BOHR if periodic else None,
                         np.asarray(atoms.pbc, bool) if periodic else None)
        return st.u, st.r0, st.atoms

    def r0_of(self, result):
        """(atoms, r0 (n_c, 2) Bohr) of a calculation with this switch (``gradient`` on)."""
        terms = (getattr(result, "switching", None) or {}).get(self.element)
        if terms is None or terms.r0 is None:
            return None
        return terms.atoms, terms.r0

    # -- pydftb.envmodel protocol ----------------------------------------------------------------
    def vector(self) -> np.ndarray:
        parts = [self.theta_array] if self.fit_theta else []
        parts.append(self.shift_vector())
        return np.concatenate(parts)

    def parameter_names(self) -> list:
        names = ([f"{self.element}.r0mix.{n}" for n in self.regressor.names]
                 if self.fit_theta else [])
        return names + self.shift_names()

    def with_vector(self, x) -> "R0Switch":
        x = np.asarray(x, float)
        if len(x) != len(self.vector()):
            raise ValueError(f"R0Switch: needs {len(self.vector())} values, not {len(x)}")
        k = self.regressor.size if self.fit_theta else 0
        new = dict(self.shifts_from_vector(x[k:]))
        if self.fit_theta:
            new["theta"] = tuple(float(v) for v in x[:k])
        return replace(self, **new)

    def with_gradient(self, on: bool) -> "R0Switch":
        return replace(self, gradient=bool(on))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "element": self.element, "types": list(self.types),
                "nodes": [list(n) for n in self.nodes], "descriptor": self.descriptor.to_dict(),
                "regressor": self.regressor.to_dict(), "theta": list(self.theta),
                "u_ref": list(self.u_ref), "grid": None if self.grid is None else list(self.grid),
                "margin": self.margin, "mix_onsite": self.mix_onsite,
                "onsite_shift": shift_dict(self.onsite_shift),
                "hubbard_shift": shift_dict(self.hubbard_shift), "fit_theta": self.fit_theta,
                "gradient": self.gradient}


def r0mix_from_dict(d: dict) -> R0Switch:
    d = {k: v for k, v in d.items() if k != "kind"}
    d["types"] = tuple(d["types"])
    d["nodes"] = tuple(tuple(n) for n in d["nodes"])
    d["descriptor"] = descriptor_from_dict(d["descriptor"])
    d["regressor"] = regressor_from_dict(d["regressor"])
    d["theta"] = tuple(d["theta"])
    d["u_ref"] = tuple(d["u_ref"])
    if d.get("grid") is not None:
        d["grid"] = tuple(d["grid"])
    return R0Switch(**d)


def node_radii(switches, fixed: dict | None = None) -> dict:
    """{label: (element, r0_w, r0_d)} of the node labels of R0Switches, plus ``fixed``
    {label: (element, r0_w, r0_d)} for the other labels of the SK set (e.g. an unswitched O):
    the input of :class:`utils.skgrid.TypedSKCache`."""
    out = dict(fixed or {})
    for sw in switches:
        if isinstance(sw, R0Switch):
            for t, (w, d) in zip(sw.types, sw.nodes):
                out[t] = (sw.element, w, d)
    return out


def labels_max_l(switches, max_l: dict) -> dict:
    """``max_l`` keyed by element (DFTB+ MaxAngularMomentum, e.g. {"Li": "s"}) extended to the
    node labels of the switches (an SK set of type labels needs one entry per label)."""
    out = dict(max_l or {})
    for sw in switches:
        if sw.element in out:
            for t in sw.types:
                out.setdefault(t, out[sw.element])
    return out


def _register():
    from . import envmodel
    envmodel.register("r0mix", r0mix_from_dict)


_register()
