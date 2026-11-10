"""Descriptors of atomic environments for the environment-dependent models of the A series
(A0 plan, section 4: the descriptor ladder L1 coordination numbers, L2 radial and angular
functions, L3 SOAP, L4 MACE features).

Contract (shared with :mod:`pydftb.regressors`, :mod:`pydftb.envmodel`, :mod:`pydftb.adaptive`
and :mod:`pydftb.r0mix`).  Positions, cells and distances are in Bohr, as inside the engine.

* A descriptor describes the atoms of one ``element`` (the centre atoms) with ``size`` numbers
  named ``names``.  ``compute(elements, pos, cell, pbc, derivatives)`` returns
  :class:`Features`.
* ``Features.pairs`` (with ``derivatives``) holds the gradient as one entry per (centre atom i,
  neighbour atom j, image): ``vec = r_j + T - r_i`` and ``grad[p] = d values[row(i)] / d vec``
  of shape (size, 3).  Every descriptor here depends on the positions only through such
  vectors, so dE/dR follows from dE/dvalues by :meth:`Features.chain` (dE/dr_j += g,
  dE/dr_i -= g, virial += g (x) vec, with g = dE/dvalues[row] . grad), the convention of
  :func:`pydftb.adaptive.coordinate_gradient`.
* ``has_derivatives`` False (SOAP or MACE features without their derivatives): the features
  can drive band-only fits (frozen positions) but not forces, relaxations or MD.
* ``to_dict()`` and :func:`descriptor_from_dict` round-trip a descriptor through JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from ase.data import atomic_numbers, covalent_radii

from .constants import BOHR
from .neighbors import neighbor_list
from .typeswitch import smoothstep


@dataclass
class Features:
    """Descriptor values of the centre atoms and (optionally) their gradient pairs."""
    atoms: np.ndarray                    # (n_c,) global indices of the centre atoms
    values: np.ndarray                   # (n_c, size)
    pairs: tuple | None = None           # (i, j, vec, grad): (n_p,), (n_p,), (n_p, 3), (n_p, size, 3)
    names: list = field(default_factory=list)
    virial_ok: bool = True               # False: the pairs give forces but not the virial
                                         # (derivatives w.r.t. atoms, images summed: SOAP)

    @property
    def size(self) -> int:
        return self.values.shape[1]

    def rows(self, natom: int) -> np.ndarray:
        """(natom,) row of every atom in ``values`` (-1 for atoms that are not centres)."""
        out = np.full(natom, -1, int)
        out[self.atoms] = np.arange(len(self.atoms))
        return out

    def chain(self, dedv, natom: int, virial: bool = False):
        """dE/dR (natom, 3) and the virial (3, 3, or None) from dE/dvalues (n_c, size)."""
        if self.pairs is None:
            raise ValueError("these features have no gradient pairs (compute with derivatives)")
        i, j, vec, grad = self.pairs
        if virial and not self.virial_ok:
            raise NotImplementedError("these features give forces but not the virial (stress)")
        grad_r = np.zeros((natom, 3))
        if len(i) == 0:
            return grad_r, (np.zeros((3, 3)) if virial else None)
        row = self.rows(natom)[i]
        g = np.einsum("pk,pkx->px", np.asarray(dedv, float)[row], grad)
        np.add.at(grad_r, j, g)
        np.add.at(grad_r, i, -g)
        return grad_r, (g.T @ vec if virial else None)


def _periodic(cell, pbc):
    return cell is not None and pbc is not None and np.any(pbc)


def _window_of(element, neighbour, window, scaled):
    if not scaled:
        return float(window[0]), float(window[1])
    rsum = (covalent_radii[atomic_numbers[element]]
            + covalent_radii[atomic_numbers[neighbour]]) / BOHR
    return float(window[0]) * rsum, float(window[1]) * rsum


@dataclass(frozen=True)
class CoordinationNumber:
    """L1: the coordination number CN_A = sum_B f(r_AB) of the atoms of ``element`` with M5a's
    C2 counting function (1 below r_on, 0 above r_off, smoothstep between).

    neighbours : the elements counted (None: all).
    window     : (r_on, r_off), factors of R_cov,A + R_cov,B (``scaled``, ASE's radii) or Bohr.
    split      : one CN per neighbour element, in the order of ``neighbours`` (which is then
                 required): e.g. the O coordination of Li next to its Li coordination.
    """
    element: str
    neighbours: tuple | None = None
    window: tuple = (0.9, 1.3)
    scaled: bool = True
    split: bool = False
    has_derivatives: ClassVar[bool] = True
    kind: ClassVar[str] = "cn"

    def __post_init__(self):
        if self.element not in atomic_numbers:
            raise ValueError(f"CoordinationNumber: {self.element!r} is not a chemical symbol")
        if self.neighbours is not None:
            object.__setattr__(self, "neighbours", tuple(str(e) for e in self.neighbours))
        if self.split and not self.neighbours:
            raise ValueError("CoordinationNumber(split=True) needs the neighbour elements")
        r_on, r_off = (float(x) for x in self.window)
        if not 0.0 < r_on < r_off:
            raise ValueError("CoordinationNumber window needs 0 < r_on < r_off")
        object.__setattr__(self, "window", (r_on, r_off))

    @property
    def size(self) -> int:
        return len(self.neighbours) if self.split else 1

    @property
    def names(self) -> list:
        if self.split:
            return [f"{self.element}.cn[{e}]" for e in self.neighbours]
        return [f"{self.element}.cn"]

    def windows(self, elements) -> dict:
        """(r_on, r_off) in Bohr for every neighbour element counted."""
        out = {}
        for el in sorted(set(elements)):
            if self.neighbours is not None and el not in self.neighbours:
                continue
            out[el] = _window_of(self.element, el, self.window, self.scaled)
        return out

    def compute(self, elements, pos, cell=None, pbc=None, derivatives: bool = False) -> Features:
        elements = list(elements)
        pos = np.asarray(pos, float)
        atoms = np.array([a for a, e in enumerate(elements) if e == self.element], int)
        values = np.zeros((len(atoms), self.size))
        win = self.windows(elements)
        empty = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3)),
                 np.zeros((0, self.size, 3)))
        if not win or len(atoms) == 0:
            return Features(atoms, values, empty if derivatives else None, self.names)
        rc = max(w[1] for w in win.values())
        periodic = _periodic(cell, pbc)
        i, j, _, vec, d = neighbor_list(pos, cell if periodic else None,
                                        pbc if periodic else None, rc)
        el = np.array(elements)
        r_on = np.array([win.get(e, (0.0, 0.0))[0] for e in el])
        r_off = np.array([win.get(e, (0.0, 0.0))[1] for e in el])
        keep = (el[i] == self.element) & (d < r_off[j])
        i, j, vec, d = i[keep], j[keep], vec[keep], d[keep]
        span = r_off[j] - r_on[j]
        s, ds = smoothstep((d - r_on[j]) / span)
        f, fr = 1.0 - s, -ds / span / d
        col = (np.array([self.neighbours.index(e) for e in el[j]], int) if self.split
               else np.zeros(len(j), int))
        row = np.full(len(elements), -1, int)
        row[atoms] = np.arange(len(atoms))
        np.add.at(values, (row[i], col), f)
        if not derivatives:
            return Features(atoms, values, None, self.names)
        grad = np.zeros((len(i), self.size, 3))
        grad[np.arange(len(i)), col] = fr[:, None] * vec
        return Features(atoms, values, (i, j, vec, grad), self.names)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "element": self.element,
                "neighbours": None if self.neighbours is None else list(self.neighbours),
                "window": list(self.window), "scaled": self.scaled, "split": self.split}


@dataclass(frozen=True)
class Stack:
    """Several descriptors of one element side by side (values and gradients concatenated)."""
    parts: tuple
    kind: ClassVar[str] = "stack"

    def __post_init__(self):
        parts = tuple(self.parts)
        if not parts:
            raise ValueError("Stack needs at least one descriptor")
        if len({p.element for p in parts}) != 1:
            raise ValueError("Stack: all parts must describe the same element")
        object.__setattr__(self, "parts", parts)

    @property
    def element(self) -> str:
        return self.parts[0].element

    @property
    def size(self) -> int:
        return sum(p.size for p in self.parts)

    @property
    def names(self) -> list:
        return [n for p in self.parts for n in p.names]

    @property
    def has_derivatives(self) -> bool:
        return all(p.has_derivatives for p in self.parts)

    def compute(self, elements, pos, cell=None, pbc=None, derivatives: bool = False) -> Features:
        outs = [p.compute(elements, pos, cell, pbc, derivatives) for p in self.parts]
        atoms = outs[0].atoms
        values = np.hstack([o.values for o in outs])
        if not derivatives:
            return Features(atoms, values, None, self.names)
        i, j, vec, grad = [], [], [], []
        k0 = 0
        for o in outs:
            oi, oj, ov, og = o.pairs
            g = np.zeros((len(oi), self.size, 3))
            g[:, k0:k0 + o.size] = og
            i.append(oi), j.append(oj), vec.append(ov), grad.append(g)
            k0 += o.size
        return Features(atoms, values, (np.concatenate(i).astype(int), np.concatenate(j).astype(int),
                                        np.vstack(vec), np.concatenate(grad)), self.names,
                        all(o.virial_ok for o in outs))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "parts": [p.to_dict() for p in self.parts]}


# ---- L2: radial and angular functions --------------------------------------------------------------
def _cutoff(r, rc: float, r_on: float = 0.0):
    """C2 cut-off f_c: 1 below r_on, 1 - S((r - r_on) / (rc - r_on)) up to rc, 0 beyond (S the
    smoothstep), and df_c/dr."""
    span = rc - r_on
    s, ds = smoothstep((np.asarray(r, float) - r_on) / span)
    return 1.0 - s, -ds / span


def _centres(element, elements) -> np.ndarray:
    return np.array([a for a, e in enumerate(elements) if e == element], int)


def _row_of(natom: int, atoms) -> np.ndarray:
    out = np.full(natom, -1, int)
    out[atoms] = np.arange(len(atoms))
    return out


def _entries(element, elements, pos, cell, pbc, rc, allowed=None):
    """Neighbour entries (i, j, vec, d) of the atoms of ``element`` closer than rc (Bohr), the
    neighbours restricted to the ``allowed`` elements (None: all)."""
    periodic = _periodic(cell, pbc)
    i, j, _, vec, d = neighbor_list(np.asarray(pos, float), cell if periodic else None,
                                    pbc if periodic else None, rc)
    el = np.array(elements)
    keep = el[i] == element
    if allowed is not None:
        keep &= np.isin(el[j], list(allowed))
    return i[keep], j[keep], vec[keep], d[keep]


def _check_element(name, element):
    if element not in atomic_numbers:
        raise ValueError(f"{name}: {element!r} is not a chemical symbol")


def _tuple_or_none(v):
    return None if v is None else tuple(v)


@dataclass(frozen=True)
class Radial:
    """L2: Gaussian radial functions of the atoms of ``element``,
    G_k = sum_B exp(-(r_AB - mu_k)^2 / (2 w_k^2)) f_c(r_AB), with the C2 cut-off f_c (1 below
    ``r_on``, smoothstep to 0 at ``cutoff``).  Bohr throughout.

    neighbours : the elements counted (None: all); ``split``: one block of functions per
                 neighbour element, in the order of ``neighbours`` (then required).
    """
    element: str
    centres: tuple
    width: object = 1.0
    cutoff: float = 10.0
    r_on: float = 0.0
    neighbours: tuple | None = None
    split: bool = False
    has_derivatives: ClassVar[bool] = True
    kind: ClassVar[str] = "radial"

    def __post_init__(self):
        _check_element("Radial", self.element)
        c = tuple(float(x) for x in np.ravel(self.centres))
        if not c:
            raise ValueError("Radial needs at least one centre")
        w = tuple(float(x) for x in np.ravel(self.width))
        if len(w) == 1:
            w = w * len(c)
        if len(w) != len(c) or any(x <= 0.0 for x in w):
            raise ValueError("Radial width: one positive value, or one per centre")
        if not 0.0 <= self.r_on < self.cutoff:
            raise ValueError("Radial needs 0 <= r_on < cutoff")
        object.__setattr__(self, "centres", c)
        object.__setattr__(self, "width", w if len(set(w)) > 1 else w[0])
        object.__setattr__(self, "neighbours", _tuple_or_none(self.neighbours))
        if self.split and not self.neighbours:
            raise ValueError("Radial(split=True) needs the neighbour elements")

    @property
    def widths(self) -> np.ndarray:
        return np.broadcast_to(np.asarray(self.width, float), (len(self.centres),)).copy()

    @property
    def blocks(self) -> tuple:
        if self.split:
            return self.neighbours
        return ("all" if self.neighbours is None else "+".join(self.neighbours),)

    @property
    def size(self) -> int:
        return len(self.blocks) * len(self.centres)

    @property
    def names(self) -> list:
        return [f"{self.element}.rad[{b}]({mu:.4g})" for b in self.blocks for mu in self.centres]

    def compute(self, elements, pos, cell=None, pbc=None, derivatives: bool = False) -> Features:
        elements = list(elements)
        natom = len(elements)
        atoms = _centres(self.element, elements)
        values = np.zeros((len(atoms), self.size))
        i, j, vec, d = _entries(self.element, elements, pos, cell, pbc, self.cutoff,
                                self.neighbours)
        n_mu = len(self.centres)
        mu, w = np.asarray(self.centres), self.widths
        x = (d[:, None] - mu[None, :]) / w[None, :]
        g = np.exp(-0.5 * x * x)
        dg = -g * x / w[None, :]
        fc, dfc = _cutoff(d, self.cutoff, self.r_on)
        val = g * fc[:, None]
        dval = dg * fc[:, None] + g * dfc[:, None]
        el = np.array(elements)
        block = (np.array([self.neighbours.index(e) for e in el[j]], int) if self.split
                 else np.zeros(len(j), int))
        cols = block[:, None] * n_mu + np.arange(n_mu)[None, :]
        row = _row_of(natom, atoms)[i]
        np.add.at(values, (np.broadcast_to(row[:, None], cols.shape), cols), val)
        if not derivatives:
            return Features(atoms, values, None, self.names)
        grad = np.zeros((len(i), self.size, 3))
        if len(i):
            u = vec / d[:, None]
            grad[np.arange(len(i))[:, None], cols] = dval[:, :, None] * u[:, None, :]
        return Features(atoms, values, (i, j, vec, grad), self.names)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "element": self.element, "centres": list(self.centres),
                "width": list(self.widths) if np.ndim(self.width) else self.width,
                "cutoff": self.cutoff, "r_on": self.r_on,
                "neighbours": None if self.neighbours is None else list(self.neighbours),
                "split": self.split}

    @classmethod
    def from_dict(cls, d: dict) -> "Radial":
        d = dict(d)
        d["centres"] = tuple(d["centres"])
        d["neighbours"] = _tuple_or_none(d.get("neighbours"))
        return cls(**d)


@dataclass(frozen=True)
class Angular:
    """L2: Behler-type angular functions of the atoms of ``element``, one per (zeta, lambda,
    eta) and neighbour-pair block,

        G5 = 2^(1 - zeta) sum_{B != C} (1 + lambda cos theta_BAC)^zeta
             exp(-eta (r_AB^2 + r_AC^2)) f_c(r_AB) f_c(r_AC)

    over ordered pairs of neighbour entries; ``form="G4"`` also has exp(-eta r_BC^2) f_c(r_BC).
    zeta >= 1, lambda = +-1, eta in Bohr^-2.  ``split``: one block per unordered pair of
    neighbour elements (from ``neighbours``, then required), e.g. Li-centred (Li, Li),
    (Li, O), (O, O).  M8b's q (pydftb.adaptive) is a normalised Legendre-P3 relative.
    """
    element: str
    zetas: tuple = (1.0, 2.0, 4.0)
    lambdas: tuple = (-1.0, 1.0)
    etas: tuple = (0.0,)
    cutoff: float = 9.0
    r_on: float = 0.0
    neighbours: tuple | None = None
    split: bool = False
    form: str = "G5"
    has_derivatives: ClassVar[bool] = True
    kind: ClassVar[str] = "angular"

    def __post_init__(self):
        _check_element("Angular", self.element)
        z = tuple(float(v) for v in np.ravel(self.zetas))
        lam = tuple(float(v) for v in np.ravel(self.lambdas))
        eta = tuple(float(v) for v in np.ravel(self.etas))
        if not z or any(v < 1.0 for v in z):
            raise ValueError("Angular zetas must be >= 1")
        if not lam or any(v not in (-1.0, 1.0) for v in lam):
            raise ValueError("Angular lambdas must be +1 or -1")
        if not eta or any(v < 0.0 for v in eta):
            raise ValueError("Angular etas must be >= 0 (Bohr^-2)")
        if self.form not in ("G4", "G5"):
            raise ValueError("Angular form is 'G4' or 'G5'")
        if not 0.0 <= self.r_on < self.cutoff:
            raise ValueError("Angular needs 0 <= r_on < cutoff")
        object.__setattr__(self, "zetas", z)
        object.__setattr__(self, "lambdas", lam)
        object.__setattr__(self, "etas", eta)
        object.__setattr__(self, "neighbours", _tuple_or_none(self.neighbours))
        if self.split and not self.neighbours:
            raise ValueError("Angular(split=True) needs the neighbour elements")

    @property
    def combos(self) -> list:
        return [(z, lam, eta) for z in self.zetas for lam in self.lambdas for eta in self.etas]

    @property
    def blocks(self) -> list:
        if not self.split:
            return [("all",)]
        nb = self.neighbours
        return [(nb[a], nb[b]) for a in range(len(nb)) for b in range(a, len(nb))]

    @property
    def size(self) -> int:
        return len(self.blocks) * len(self.combos)

    @property
    def names(self) -> list:
        return [f"{self.element}.{self.form}[{'-'.join(b)}](z{z:g},l{lam:+g},e{eta:g})"
                for b in self.blocks for z, lam, eta in self.combos]

    def compute(self, elements, pos, cell=None, pbc=None, derivatives: bool = False) -> Features:
        elements = list(elements)
        natom = len(elements)
        atoms = _centres(self.element, elements)
        values = np.zeros((len(atoms), self.size))
        i, j, vec, d = _entries(self.element, elements, pos, cell, pbc, self.cutoff,
                                self.neighbours)
        grad = np.zeros((len(i), self.size, 3)) if derivatives else None
        el = np.array(elements)
        blocks = self.blocks
        table = None
        if self.split:                    # block of an (ordered by neighbours) element pair
            nbs = self.neighbours
            table = np.full((len(nbs), len(nbs)), -1, int)
            for k, (x, y) in enumerate(blocks):
                table[nbs.index(x), nbs.index(y)] = k
        ncombo = len(self.combos)
        g4 = self.form == "G4"
        row = _row_of(natom, atoms)
        for a in np.unique(i):
            e = np.flatnonzero(i == a)
            n = len(e)
            if n < 2:
                continue
            v, r = vec[e], d[e]
            u = v / r[:, None]
            c = np.clip(u @ u.T, -1.0, 1.0)
            f, df = _cutoff(r, self.cutoff, self.r_on)
            off = ~np.eye(n, dtype=bool)
            if self.split:
                nb = np.array([self.neighbours.index(x) for x in el[j[e]]], int)
                bt = table[np.minimum(nb[:, None], nb[None, :]),
                           np.maximum(nb[:, None], nb[None, :])]
            else:
                bt = np.zeros((n, n), int)
            dv = v[:, None, :] - v[None, :, :]               # vec_e - vec_e'
            if g4:
                rjk = np.linalg.norm(dv, axis=2)
                rsafe = np.where(off, rjk, 1.0)
                f3, df3 = _cutoff(rsafe, self.cutoff, self.r_on)
                f3 = np.where(off, f3, 0.0)
                df3 = np.where(off, df3, 0.0)
            else:
                f3, df3, rsafe = np.ones((n, n)), np.zeros((n, n)), np.ones((n, n))
            ff = f[:, None] * f[None, :]
            fval = ff * f3
            r2 = r[:, None] ** 2 + r[None, :] ** 2 + (rjk ** 2 if g4 else 0.0)
            if derivatives:
                dc = (u[None, :, :] - c[:, :, None] * u[:, None, :]) / r[:, None, None]
                dfv = ((df[:, None] * f[None, :] * f3)[:, :, None] * u[:, None, :]
                       + (ff * df3 / rsafe)[:, :, None] * dv)
                dr2 = 2.0 * (v[:, None, :] + (dv if g4 else 0.0))
            for k, (z, lam, eta) in enumerate(self.combos):
                base = 1.0 + lam * c
                amp = base ** z
                norm = 2.0 ** (1.0 - z)
                bexp = np.exp(-eta * r2)
                t = np.where(off, norm * amp * bexp * fval, 0.0)
                for b in range(len(blocks)):
                    m = bt == b
                    values[row[a], b * ncombo + k] += t[m].sum()
                if derivatives:
                    damp = z * lam * base ** (z - 1.0)
                    dt = norm * ((damp * bexp * fval)[:, :, None] * dc
                                 - (eta * amp * bexp * fval)[:, :, None] * dr2
                                 + (amp * bexp)[:, :, None] * dfv)
                    dt = np.where(off[:, :, None], dt, 0.0)
                    for b in range(len(blocks)):
                        m = (bt == b)[:, :, None]
                        grad[e, b * ncombo + k] = 2.0 * np.where(m, dt, 0.0).sum(axis=1)
        pairs = (i, j, vec, grad) if derivatives else None
        return Features(atoms, values, pairs, self.names)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "element": self.element, "zetas": list(self.zetas),
                "lambdas": list(self.lambdas), "etas": list(self.etas), "cutoff": self.cutoff,
                "r_on": self.r_on,
                "neighbours": None if self.neighbours is None else list(self.neighbours),
                "split": self.split, "form": self.form}

    @classmethod
    def from_dict(cls, d: dict) -> "Angular":
        d = dict(d)
        for k in ("zetas", "lambdas", "etas"):
            d[k] = tuple(d[k])
        d["neighbours"] = _tuple_or_none(d.get("neighbours"))
        return cls(**d)


# ---- transforms of another descriptor ---------------------------------------------------------
def _smooth_clamp(x, lo: float, hi: float, d: float):
    """The identity on [lo + d, hi - d], quadratic over 2d at each end (value and slope
    continuous), constant beyond (adaptive.RBF's clamp), and its derivative."""
    x = np.asarray(x, float)
    y, dy = x.copy(), np.ones_like(x)
    a = x <= lo - d
    y[a], dy[a] = lo, 0.0
    b = (x > lo - d) & (x < lo + d)
    y[b] = lo + (x[b] - (lo - d)) ** 2 / (4 * d)
    dy[b] = (x[b] - (lo - d)) / (2 * d)
    a = x >= hi + d
    y[a], dy[a] = hi, 0.0
    b = (x > hi - d) & (x < hi + d)
    y[b] = hi - (hi + d - x[b]) ** 2 / (4 * d)
    dy[b] = (hi + d - x[b]) / (2 * d)
    return y, dy


def _transform_pairs(f: Features, jac):
    """Gradient pairs of a transform whose values y (n_c, m) depend on the inner values x by
    ``jac`` (n_c, m, n_in) = dy/dx."""
    if f.pairs is None:
        return None
    i, j, vec, grad = f.pairs
    if len(i) == 0:
        return i, j, vec, np.zeros((0, jac.shape[1], 3))
    row = f.rows(int(max(i.max(), j.max(), f.atoms.max(initial=-1))) + 1)[i]
    return i, j, vec, np.einsum("pmf,pfx->pmx", jac[row], grad)


def _inner_of(d):
    return descriptor_from_dict(d)


@dataclass(frozen=True)
class Expand:
    """Gaussian expansion of every feature of another descriptor (``inner``): feature x_f gives
    exp(-(y_f - c)^2 / (2 w^2)) for each of its centres c, with y_f the smooth clamp of x_f to
    ``clamp`` = (lo, hi[, d]) (as :class:`pydftb.adaptive.RBF`; constant beyond the training
    range instead of extrapolating).  ``centres``, ``width`` and ``clamp`` are one value for all
    features or one per feature; ``include_input`` keeps the inner values in front."""
    inner: object
    centres: tuple
    width: object = 1.0
    clamp: object = None
    include_input: bool = False
    kind: ClassVar[str] = "expand"

    def __post_init__(self):
        n = self.inner.size
        c = self.centres
        if len(c) and np.ndim(c[0]) == 0:
            c = (tuple(float(x) for x in c),) * n
        c = tuple(tuple(float(x) for x in cf) for cf in c)
        if len(c) != n:
            raise ValueError(f"Expand: centres for {n} features, got {len(c)}")
        w = np.ravel(np.asarray(self.width, float))
        w = tuple(float(x) for x in (np.full(n, w[0]) if len(w) == 1 else w))
        if len(w) != n or any(x <= 0.0 for x in w):
            raise ValueError("Expand width: one positive value or one per feature")
        cl = self.clamp
        if cl is not None:
            if all(x is not None and np.ndim(x) == 0 for x in cl):        # one for all features
                cl = (tuple(cl),) * n
            cl = tuple(None if x is None else (tuple(float(v) for v in x) + ((0.5,) if len(x) == 2 else ()))
                       for x in cl)
            if len(cl) != n:
                raise ValueError("Expand clamp: one (lo, hi[, d]) or one per feature")
            for x in cl:
                if x is not None and not (x[2] > 0.0 and x[0] + 2 * x[2] <= x[1]):
                    raise ValueError("Expand clamp is (lo, hi[, d]) with d > 0 and lo + 2d <= hi")
        object.__setattr__(self, "centres", c)
        object.__setattr__(self, "width", w)
        object.__setattr__(self, "clamp", cl)

    @property
    def element(self) -> str:
        return self.inner.element

    @property
    def has_derivatives(self) -> bool:
        return self.inner.has_derivatives

    @property
    def size(self) -> int:
        return (self.inner.size if self.include_input else 0) + sum(len(c) for c in self.centres)

    @property
    def names(self) -> list:
        inner = self.inner.names
        out = list(inner) if self.include_input else []
        for nm, cf in zip(inner, self.centres):
            out += [f"{nm}~g({c:.4g})" for c in cf]
        return out

    def _map(self, x):
        n_c = len(x)
        cols, jac = [], np.zeros((n_c, self.size, self.inner.size))
        k = 0
        if self.include_input:
            cols.append(x)
            jac[:, np.arange(self.inner.size), np.arange(self.inner.size)] = 1.0
            k = self.inner.size
        for fi, (cf, w) in enumerate(zip(self.centres, self.width)):
            xf = x[:, fi]
            cl = None if self.clamp is None else self.clamp[fi]
            y, dy = (xf, np.ones_like(xf)) if cl is None else _smooth_clamp(xf, *cl)
            cc = np.asarray(cf)[None, :]
            g = np.exp(-0.5 * ((y[:, None] - cc) / w) ** 2)
            dg = -g * (y[:, None] - cc) / w ** 2 * dy[:, None]
            cols.append(g)
            jac[:, k:k + len(cf), fi] = dg
            k += len(cf)
        return np.hstack(cols) if cols else np.zeros((n_c, 0)), jac

    def compute(self, elements, pos, cell=None, pbc=None, derivatives: bool = False) -> Features:
        f = self.inner.compute(elements, pos, cell, pbc, derivatives)
        y, jac = self._map(f.values)
        pairs = _transform_pairs(f, jac) if derivatives else None
        return Features(f.atoms, y, pairs, self.names, f.virial_ok)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "inner": self.inner.to_dict(),
                "centres": [list(c) for c in self.centres], "width": list(self.width),
                "clamp": None if self.clamp is None else [None if c is None else list(c)
                                                          for c in self.clamp],
                "include_input": self.include_input}

    @classmethod
    def from_dict(cls, d: dict) -> "Expand":
        return cls(_inner_of(d["inner"]), tuple(tuple(c) for c in d["centres"]),
                   tuple(d["width"]),
                   None if d.get("clamp") is None else tuple(None if c is None else tuple(c)
                                                             for c in d["clamp"]),
                   d.get("include_input", False))


def compute_atoms(desc, atoms, derivatives: bool = False) -> Features:
    """``desc.compute`` for ASE Atoms (Angstrom; the descriptor works in Bohr)."""
    periodic = bool(np.any(atoms.pbc))
    return desc.compute(atoms.get_chemical_symbols(), atoms.positions / BOHR,
                        atoms.cell.array / BOHR if periodic else None,
                        np.asarray(atoms.pbc, bool) if periodic else None, derivatives)


def training_features(desc, frames) -> np.ndarray:
    """(n, size) values of the centre atoms of all ``frames`` (ASE Atoms) stacked."""
    vals = [compute_atoms(desc, a).values for a in frames]
    vals = [v for v in vals if len(v)]
    if not vals:
        raise ValueError(f"no {desc.element} atoms in the frames")
    return np.vstack(vals)


@dataclass(frozen=True)
class Affine:
    """A fixed linear map of another descriptor, y = (x - mean) @ matrix.T: standardisation
    (:meth:`standardize`) and principal components (:meth:`pca`), fitted on training frames."""
    inner: object
    mean: tuple
    matrix: tuple
    label: str = "affine"
    kind: ClassVar[str] = "affine"

    def __post_init__(self):
        mean = tuple(float(x) for x in np.ravel(self.mean))
        mat = np.atleast_2d(np.asarray(self.matrix, float))
        if len(mean) != self.inner.size or mat.shape[1] != self.inner.size:
            raise ValueError(f"Affine: mean and matrix need {self.inner.size} input columns")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "matrix", tuple(tuple(r) for r in mat))

    @property
    def element(self) -> str:
        return self.inner.element

    @property
    def has_derivatives(self) -> bool:
        return self.inner.has_derivatives

    @property
    def size(self) -> int:
        return len(self.matrix)

    @property
    def names(self) -> list:
        if self.label == "standardize":
            return [f"{n}~std" for n in self.inner.names]
        return [f"{self.element}.{self.label}{k}" for k in range(self.size)]

    def compute(self, elements, pos, cell=None, pbc=None, derivatives: bool = False) -> Features:
        f = self.inner.compute(elements, pos, cell, pbc, derivatives)
        m = np.asarray(self.matrix)
        y = (f.values - np.asarray(self.mean)[None, :]) @ m.T
        pairs = None
        if derivatives:
            jac = np.broadcast_to(m, (len(f.values),) + m.shape)
            pairs = _transform_pairs(f, jac)
        return Features(f.atoms, y, pairs, self.names, f.virial_ok)

    @classmethod
    def standardize(cls, inner, frames, floor: float = 1e-8) -> "Affine":
        """(x - mean) / std over the centre atoms of the training ``frames`` (a constant
        feature keeps scale 1)."""
        x = training_features(inner, frames)
        sd = x.std(axis=0)
        sd = np.where(sd > floor, sd, 1.0)
        return cls(inner, tuple(x.mean(axis=0)), tuple(map(tuple, np.diag(1.0 / sd))),
                   "standardize")

    @classmethod
    def pca(cls, inner, frames, n_components: int | None = None, variance: float = 0.99,
            standardize: bool = True, floor: float = 1e-8) -> "Affine":
        """The leading principal components of the (standardised) training features: the
        ``n_components`` first, or as many as explain ``variance`` of the total."""
        x = training_features(inner, frames)
        mean = x.mean(axis=0)
        sd = x.std(axis=0)
        sd = np.where(sd > floor, sd, 1.0) if standardize else np.ones_like(sd)
        z = (x - mean) / sd
        _, s, vt = np.linalg.svd(z, full_matrices=False)
        var = s ** 2
        if n_components is None:
            frac = np.cumsum(var) / max(var.sum(), 1e-300)
            n_components = int(np.searchsorted(frac, variance) + 1)
        n_components = max(1, min(int(n_components), len(vt)))
        mat = vt[:n_components] / sd[None, :]
        return cls(inner, tuple(mean), tuple(map(tuple, mat)), "pc")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "inner": self.inner.to_dict(), "mean": list(self.mean),
                "matrix": [list(r) for r in self.matrix], "label": self.label}

    @classmethod
    def from_dict(cls, d: dict) -> "Affine":
        return cls(_inner_of(d["inner"]), tuple(d["mean"]), tuple(tuple(r) for r in d["matrix"]),
                   d.get("label", "affine"))


def Standardize(inner, frames) -> Affine:                                    # noqa: N802
    """(x - mean) / std of ``inner`` over the training ``frames`` (:meth:`Affine.standardize`)."""
    return Affine.standardize(inner, frames)


def PCA(inner, frames, n_components=None, variance=0.99, standardize=True) -> Affine:  # noqa: N802
    """Principal components of ``inner`` over the training ``frames`` (:meth:`Affine.pca`)."""
    return Affine.pca(inner, frames, n_components, variance, standardize)


# ---- L3, L4: SOAP and MACE features (optional dependencies) -------------------------------------
def _ase_atoms(elements, pos, cell, pbc):
    from ase import Atoms
    periodic = _periodic(cell, pbc)
    return Atoms(symbols=list(elements), positions=np.asarray(pos, float) * BOHR,
                 cell=np.asarray(cell, float) * BOHR if periodic else None,
                 pbc=np.asarray(pbc, bool) if periodic else False)


@dataclass(frozen=True)
class SOAP:
    """L3: SOAP power spectrum of the atoms of ``element`` (dscribe; ``pip install dscribe``).

    species : every element the structures may contain (dscribe's feature layout depends on it).
    r_cut, sigma in Bohr (converted to dscribe's Angstrom).  Without ``derivatives`` the
    features drive band-only fits (``has_derivatives`` False); with it dscribe's analytical
    derivatives give forces, but not the stress of periodic cells (images summed per atom).
    Reduce the (large) feature count with :class:`Affine` (``PCA``) before a regressor.
    """
    element: str
    species: tuple
    r_cut: float = 9.4486
    n_max: int = 6
    l_max: int = 4
    sigma: float = 0.94486
    rbf: str = "gto"
    derivatives: bool = False
    kind: ClassVar[str] = "soap"

    def __post_init__(self):
        _check_element("SOAP", self.element)
        object.__setattr__(self, "species", tuple(sorted(set(self.species),
                                                         key=lambda e: atomic_numbers[e])))
        if self.element not in self.species:
            raise ValueError("SOAP: species must include the centre element")

    @property
    def has_derivatives(self) -> bool:
        return bool(self.derivatives)

    def _soap(self, periodic: bool):
        try:
            from dscribe.descriptors import SOAP as _DSOAP
        except ImportError as exc:                       # pragma: no cover - optional
            raise ImportError("the SOAP descriptor needs dscribe (pip install dscribe)") from exc
        return _DSOAP(species=list(self.species), r_cut=self.r_cut * BOHR, n_max=self.n_max,
                      l_max=self.l_max, sigma=self.sigma * BOHR, rbf=self.rbf,
                      periodic=periodic, sparse=False)

    @property
    def size(self) -> int:
        return int(self._soap(False).get_number_of_features())

    @property
    def names(self) -> list:
        return [f"{self.element}.soap{k}" for k in range(self.size)]

    def compute(self, elements, pos, cell=None, pbc=None, derivatives: bool = False) -> Features:
        elements = list(elements)
        atoms = _centres(self.element, elements)
        if derivatives and not self.derivatives:
            raise ValueError("this SOAP descriptor was made without derivatives "
                             "(SOAP(..., derivatives=True))")
        periodic = _periodic(cell, pbc)
        soap = self._soap(periodic)
        at = _ase_atoms(elements, pos, cell, pbc)
        if len(atoms) == 0:
            return Features(atoms, np.zeros((0, self.size)), None, self.names)
        if not derivatives:
            v = np.atleast_2d(soap.create(at, centers=atoms.tolist()))
            return Features(atoms, v, None, self.names)
        dd, v = soap.derivatives(at, centers=atoms.tolist(), method="auto",
                                 return_descriptor=True)
        v = np.atleast_2d(v)
        dd = np.asarray(dd).reshape(len(atoms), len(elements), 3, -1) * BOHR    # per Bohr
        i, j, vec, grad = [], [], [], []
        p = np.asarray(pos, float)
        for r, a in enumerate(atoms):
            for b in range(len(elements)):
                if b == a:
                    continue
                i.append(a), j.append(b), vec.append(p[b] - p[a])
                grad.append(dd[r, b].T)                                         # (size, 3)
        pairs = (np.array(i, int), np.array(j, int), np.array(vec).reshape(-1, 3),
                 np.array(grad).reshape(-1, v.shape[1], 3))
        return Features(atoms, v, pairs, self.names, virial_ok=not periodic)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "element": self.element, "species": list(self.species),
                "r_cut": self.r_cut, "n_max": self.n_max, "l_max": self.l_max,
                "sigma": self.sigma, "rbf": self.rbf, "derivatives": self.derivatives}

    @classmethod
    def from_dict(cls, d: dict) -> "SOAP":
        d = dict(d)
        d["species"] = tuple(d["species"])
        return cls(**d)


_MACE_CACHE = {}


@dataclass(frozen=True)
class MACEFeatures:
    """L4: MACE invariant node features of the atoms of ``element`` (mace-torch).

    model : a mace_mp size ("small", "medium", ...; ``family="mp"``), a mace_off size
            (``family="off"``), or a model file (``family="path"``).
    size  : the feature count (``MACEFeatures(...).with_size(atoms)`` fills it in).
    Band-only fits: ``has_derivatives`` is False (torch autograd through the features is not
    wired up), so no forces with these features.
    """
    element: str
    model: str = "small"
    family: str = "mp"
    num_layers: int = -1
    device: str = "cpu"
    size: int | None = None
    has_derivatives: ClassVar[bool] = False
    kind: ClassVar[str] = "mace"

    def __post_init__(self):
        _check_element("MACEFeatures", self.element)
        if self.family not in ("mp", "off", "path"):
            raise ValueError("MACEFeatures family is 'mp', 'off' or 'path'")

    def _calc(self):
        key = (self.family, self.model, self.device)
        if key not in _MACE_CACHE:
            try:
                if self.family == "mp":
                    from mace.calculators import mace_mp
                    calc = mace_mp(model=self.model, device=self.device, default_dtype="float64")
                elif self.family == "off":
                    from mace.calculators import mace_off
                    calc = mace_off(model=self.model, device=self.device, default_dtype="float64")
                else:
                    from mace.calculators import MACECalculator
                    calc = MACECalculator(model_paths=self.model, device=self.device,
                                          default_dtype="float64")
            except ImportError as exc:                   # pragma: no cover - optional
                raise ImportError("MACEFeatures needs mace-torch (pip install mace-torch)") from exc
            _MACE_CACHE[key] = calc
        return _MACE_CACHE[key]

    @property
    def names(self) -> list:
        return [f"{self.element}.mace{k}" for k in range(self.size or 0)]

    def with_size(self, atoms) -> "MACEFeatures":
        """A copy with ``size`` from the features of ``atoms`` (ASE, must contain the element)."""
        from dataclasses import replace
        v = self._values(atoms)
        return replace(self, size=int(v.shape[1]))

    def _values(self, at):
        v = np.asarray(self._calc().get_descriptors(at, invariants_only=True,
                                                     num_layers=self.num_layers), float)
        return v.reshape(len(at), -1)

    def compute(self, elements, pos, cell=None, pbc=None, derivatives: bool = False) -> Features:
        if derivatives:
            raise ValueError("MACEFeatures has no position derivatives (band-only fits)")
        elements = list(elements)
        atoms = _centres(self.element, elements)
        at = _ase_atoms(elements, pos, cell, pbc)
        v = self._values(at)[atoms]
        if self.size is not None and v.shape[1] != self.size:
            raise ValueError(f"MACEFeatures: {v.shape[1]} features, size says {self.size}")
        return Features(atoms, v, None, self.names)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "element": self.element, "model": self.model,
                "family": self.family, "num_layers": self.num_layers, "device": self.device,
                "size": self.size}


_KINDS = {"cn": CoordinationNumber, "stack": Stack, "radial": Radial, "angular": Angular,
          "expand": Expand, "affine": Affine, "soap": SOAP, "mace": MACEFeatures}


def register(kind: str, cls) -> None:
    """Make ``descriptor_from_dict`` know another descriptor class (``cls.from_dict(d)`` if it
    has one, else ``cls(**fields)``)."""
    _KINDS[kind] = cls


def descriptor_from_dict(d: dict):
    """The descriptor of a ``to_dict()`` result."""
    d = dict(d)
    kind = d.pop("kind")
    if kind not in _KINDS:
        raise ValueError(f"unknown descriptor kind {kind!r} (known: {sorted(_KINDS)})")
    cls = _KINDS[kind]
    if hasattr(cls, "from_dict"):
        return cls.from_dict(d)
    if kind == "stack":
        return Stack(tuple(descriptor_from_dict(p) for p in d["parts"]))
    if "neighbours" in d and d["neighbours"] is not None:
        d["neighbours"] = tuple(d["neighbours"])
    if "window" in d:
        d["window"] = tuple(d["window"])
    return cls(**d)
