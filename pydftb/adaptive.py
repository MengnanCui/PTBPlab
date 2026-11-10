"""Environment-adaptive parameters: DFTB parameters that change continuously with the
environment of every atom (milestone M8a).

DFTB+ (and pydftb's type labels) give every atom one fixed parameter set; M5a mixes a few
discrete sets with coordination-number weights.  Here the parameters themselves are smooth
functions of a descriptor of each atom's environment, with exact forces and stress, so that
relaxations, MD and periodic systems work, and with the derivatives of the free energy with
respect to the model's coefficients for fitting.

Descriptor: the coordination number CN_A = sum_B f(r_AB) (periodic images included), with M5a's
counting function f = 1 - S((r - r_on) / (r_off - r_on)) between r_on and r_off (S the C2
smoothstep).  The default window (0.9, 1.3) x (R_A + R_B) (ASE's covalent radii) starts inside
the first-neighbour shell of close-packed metals, so CN changes with the volume as well as with
the structure.

Basis: Phi(CN) = [1, exp(-(CN - c_k)^2 / (2 w^2)) for the centres c_k] (:class:`RBF`); with no
centres only the constant is left, which is the same model without environment dependence.

Channels of an element (every one linear in its coefficients theta, p_A = Phi(CN_A) . theta):

* ``onsite[l]``: a shift of the on-site energies of shell l, eps_A,l + p_A (Hartree);
* ``hopping[l]``: a scale of the off-site Hamiltonian, H0_mu,nu -> c_mu H0_mu,nu c_nu with
  c_mu = exp(p_A,l / 2) for the orbitals mu of shell l of atom A (symmetric, and S is not
  scaled, so it stays positive definite);
* ``repulsive``: a scale of the pair repulsive, V_AB -> V_AB exp((p_A + p_B) / 2);
* ``pair_repulsive[B]``: a pair repulsive added between this element and element B,
  V(r) = sum_n c_n (r_c - r)^n for r < r_c (n = 2, 3, ...; :class:`PairRepulsive`), also scaled
  by the repulsive channel.

Forces: at fixed parameters the scaled H0 and its derivatives and the scaled repulsive go
through the usual code; the parameters' dependence on the geometry adds
sum_A dE/dCN_A dCN_A/dR, with the variational dE/dp at fixed density: the on-site density
D_mu,mu for the on-site shifts, 1/2 sum_e P_e H_e over the pair elements of the scaled H0 that
touch the shell for the hopping scales, and the repulsive's own terms.  The same dE/dp give the
derivatives of the Mermin free energy with respect to every coefficient (:class:`AdaptiveResult`),
which is what a fit needs.  With zero coefficients the calculation is the plain one, bit for bit.

Descriptor models (A series, A0 section 4): ``AdaptiveParams(element, descriptor=...,
regressor=..., channels=("onsite.s", "hopping.p", "repulsive"), theta=...)`` replaces the CN
basis by any :mod:`pydftb.descriptors` descriptor (radial, angular, SOAP, MACE features) and
any :mod:`pydftb.regressors` regressor (linear, kernel, MLP) whose outputs are the channels'
p_A; forces and stress go through the descriptor's gradient pairs
(:meth:`pydftb.descriptors.Features.chain`), and dF/dtheta = sum_A dF/dp_A dp_A/dtheta.  A
descriptor without position derivatives allows energies and bands only (no forces).

NumPy engine, dense solvers, no spin, two-component spin, hybrids, Casida, type switching or
autodiff (refused by :class:`pydftb.calculator.DFTB`).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

import numpy as np
from ase.data import atomic_numbers, covalent_radii

from .constants import BOHR
from .neighbors import neighbor_list
from .system import element_of_label
from .typeswitch import smoothstep

DEFAULT_WINDOW = (0.9, 1.3)       # (r_on, r_off) / (R_cov,A + R_cov,B)
ANGULAR_EPS = 1e-2                # q = S / (CN^2 + eps): an isolated atom has q = 0
SHELL_LETTERS = "spdf"


# ---- basis --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RBF:
    """Phi(CN) = [1 (if ``constant``), exp(-(CN - c)^2 / (2 width^2)) for c in ``centres``].

    ``clamp`` (lo, hi[, d]): CN enters through a smooth clamp to [lo, hi], so Phi is constant
    outside the range of the fitting data instead of extrapolating (M8d: a fit to the VASP Cu
    set overbound compressed bcc, CN above every training structure).  The clamp is the
    identity on [lo + d, hi - d], quadratic over 2d at each end (value and slope continuous),
    constant beyond; d defaults to 0.5."""
    centres: tuple = ()
    width: float = 1.5
    constant: bool = True
    clamp: tuple | None = None

    def __post_init__(self):
        object.__setattr__(self, "centres", tuple(float(c) for c in self.centres))
        if not self.width > 0.0:
            raise ValueError("RBF width must be > 0")
        if not self.constant and not self.centres:
            raise ValueError("an RBF basis needs the constant or at least one centre")
        if self.clamp is not None:
            c = tuple(float(v) for v in self.clamp)
            c = c + (0.5,) if len(c) == 2 else c
            if len(c) != 3 or not (c[2] > 0.0 and c[0] + 2 * c[2] <= c[1] - 0.0):
                raise ValueError("RBF clamp is (lo, hi[, d]) with d > 0 and lo + 2d <= hi")
            object.__setattr__(self, "clamp", c)

    def _clamped(self, x):
        """The smooth clamp of x and its derivative."""
        if self.clamp is None:
            return x, np.ones_like(x)
        lo, hi, d = self.clamp
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

    @property
    def size(self) -> int:
        return int(self.constant) + len(self.centres)

    def __call__(self, cn):
        """Phi (n, size) and dPhi/dCN (n, size)."""
        cn, dcl = self._clamped(np.asarray(cn, float))
        cn, dcl = cn[:, None], dcl[:, None]
        c = np.asarray(self.centres)[None, :]
        g = np.exp(-0.5 * ((cn - c) / self.width) ** 2)
        dg = -g * (cn - c) / self.width ** 2 * dcl
        if self.constant:
            n = len(cn)
            return (np.hstack([np.ones((n, 1)), g]), np.hstack([np.zeros((n, 1)), dg]))
        return g, dg


@dataclass(frozen=True)
class PairRepulsive:
    """V(r) = sum_n coefficients[n - 2] (cutoff - r)^n for r < cutoff (Bohr, Hartree),
    n = 2, 3, ...; zero and smooth at the cut-off.  Optionally a fixed core wall
    ``wall_strength`` (wall - r)^3 for r < ``wall`` (not a coefficient, never fitted): put below
    the shortest bond of the fitting data, it changes none of their energies and keeps the
    pair repulsive at short distances where the data say nothing (M8a: without it a fitted
    Cu13 cluster collapsed to 1.4 A in MD)."""
    cutoff: float
    coefficients: tuple = (0.0, 0.0, 0.0, 0.0)
    wall: float = 0.0
    wall_strength: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "coefficients", tuple(float(c) for c in self.coefficients))
        if not self.cutoff > 0.0:
            raise ValueError("PairRepulsive cutoff must be > 0 (Bohr)")
        if self.wall < 0.0 or self.wall_strength < 0.0 or self.wall > self.cutoff:
            raise ValueError("PairRepulsive needs 0 <= wall <= cutoff and wall_strength >= 0")

    def powers(self, r):
        """(cutoff - r)^n for n = 2, 3, ... (len(r), n_coef) and d/dr, zero beyond the cut-off."""
        x = np.clip(self.cutoff - np.asarray(r, float), 0.0, None)[:, None]
        n = np.arange(2, 2 + len(self.coefficients))[None, :]
        return x ** n, -n * x ** (n - 1)

    def core(self, r):
        """The fixed wall and its derivative (zero for r >= wall)."""
        x = np.clip(self.wall - np.asarray(r, float), 0.0, None)
        return self.wall_strength * x ** 3, -3.0 * self.wall_strength * x ** 2

    def __call__(self, r):
        p, dp = self.powers(r)
        c = np.asarray(self.coefficients)
        w, dw = self.core(r)
        return p @ c + w, dp @ c + dw


# ---- the model of one element -----------------------------------------------------------------------
@dataclass(frozen=True)
class AdaptiveParams:
    """Environment-adaptive parameters of the atoms of ``element``.

    basis     : :class:`RBF` of the coordination number.
    onsite    : {shell letter: coefficients} on-site shifts (Hartree).
    hopping   : {shell letter: coefficients} log scales of the off-site H0 of that shell.
    repulsive : coefficients of the log scale of the pair repulsive (None: no scaling).
    pair_repulsive : {neighbour element: :class:`PairRepulsive`} pair repulsives added.
    window    : (r_on, r_off) of the counting function: Bohr, or factors of R_A + R_B with
                ``window_scaled`` (default: DEFAULT_WINDOW, scaled).
    neighbours: the elements counted in CN (None: all).
    angular   : :class:`RBF` (without the constant) of the angular descriptor
                q_A = sum_BC f_AB f_AC P3(cos theta_BAC) / (CN_A^2 + eps) (M8b), added to the
                CN basis; None: CN only.  q is 0 in centrosymmetric environments (fcc, bcc, sc),
                5/9 in diamond, 5/8 in graphene-like planes, about 1 for a dimer.

    Coefficients have ``size`` = basis.size (+ angular.size) values; missing channels are zero.
    """
    element: str
    basis: RBF = field(default_factory=RBF)
    onsite: dict = field(default_factory=dict)
    hopping: dict = field(default_factory=dict)
    repulsive: tuple | None = None
    pair_repulsive: dict = field(default_factory=dict)
    window: tuple = DEFAULT_WINDOW
    window_scaled: bool = True
    neighbours: tuple | None = None
    angular: RBF | None = None
    descriptor: object | None = None       # pydftb.descriptors: replaces the CN basis
    regressor: object | None = None        # pydftb.regressors (default: Linear)
    channels: tuple = ()                   # descriptor mode: "onsite.<l>", "hopping.<l>", "repulsive"
    theta: tuple | None = None             # descriptor mode: the regressor's coefficients
    engine_keyword: ClassVar[str] = "adaptive"        # pydftb.envmodel protocol

    @property
    def size(self) -> int:
        return self.basis.size + (self.angular.size if self.angular is not None else 0)

    def gradient_from(self, result):
        """dF/d(:meth:`vector`) in Hartree from a calculation with this model
        (:mod:`pydftb.envmodel` protocol)."""
        ad = getattr(result, "adaptive", None)
        if ad is None or self.element not in ad.gradients:
            return None
        return np.asarray(ad.gradients[self.element], float)

    def to_dict(self) -> dict:
        """JSON form (:func:`pydftb.envmodel.model_from_dict` reads it back)."""
        def rbf(b):
            return None if b is None else {"centres": list(b.centres), "width": b.width,
                                           "constant": b.constant,
                                           "clamp": None if b.clamp is None else list(b.clamp)}
        return {"kind": "adaptive", "element": self.element, "basis": rbf(self.basis),
                "onsite": {k: list(v) for k, v in self.onsite.items()},
                "hopping": {k: list(v) for k, v in self.hopping.items()},
                "repulsive": None if self.repulsive is None else list(self.repulsive),
                "pair_repulsive": {el: {"cutoff": r.cutoff, "coefficients": list(r.coefficients),
                                        "wall": r.wall, "wall_strength": r.wall_strength}
                                   for el, r in self.pair_repulsive.items()},
                "window": list(self.window), "window_scaled": self.window_scaled,
                "neighbours": None if self.neighbours is None else list(self.neighbours),
                "angular": rbf(self.angular),
                "descriptor": None if self.descriptor is None else self.descriptor.to_dict(),
                "regressor": None if self.regressor is None else self.regressor.to_dict(),
                "channels": list(self.channels),
                "theta": None if self.theta is None else list(self.theta)}

    def features(self, cn, q=None):
        """Phi (n, size), dPhi/dCN and dPhi/dq (n, size): the CN basis, then the angular one."""
        ph, dph = self.basis(cn)
        if self.angular is None:
            return ph, dph, np.zeros_like(ph)
        pq, dpq = self.angular(q)
        z = np.zeros_like(pq)
        return np.hstack([ph, pq]), np.hstack([dph, z]), np.hstack([np.zeros_like(ph), dpq])

    @property
    def descriptor_mode(self) -> bool:
        """The parameters come from ``descriptor`` and ``regressor`` (not the CN basis)."""
        return self.descriptor is not None

    def channel_keys(self) -> list:
        """The channel of every parameter p_A, in the order of the regressor outputs
        (descriptor mode) or of :meth:`vector` (CN mode): ("onsite"|"hopping", element, shell)
        or ("repulsive", element)."""
        if self.descriptor_mode:
            out = []
            for c in self.channels:
                kind, _, shell = c.partition(".")
                out.append(("repulsive", self.element) if kind == "repulsive"
                           else (kind, self.element, shell))
            return out
        return [key for key, _ in _channels(self)]

    def onsite_shells(self) -> set:
        if self.descriptor_mode:
            return {k[2] for k in self.channel_keys() if k[0] == "onsite"}
        return set(self.onsite)

    def hopping_shells(self) -> set:
        if self.descriptor_mode:
            return {k[2] for k in self.channel_keys() if k[0] == "hopping"}
        return set(self.hopping)

    @property
    def has_repulsive(self) -> bool:
        if self.descriptor_mode:
            return "repulsive" in self.channels
        return self.repulsive is not None

    def _check_descriptor_mode(self):
        from .regressors import Linear
        if self.onsite or self.hopping or self.repulsive is not None or self.angular is not None:
            raise ValueError(f"AdaptiveParams({self.element}) with a descriptor: give the "
                             "channels in 'channels' (not onsite/hopping/repulsive/angular)")
        if getattr(self.descriptor, "element", None) != self.element:
            raise ValueError(f"AdaptiveParams({self.element}): the descriptor describes "
                             f"{getattr(self.descriptor, 'element', None)!r} atoms")
        chans = tuple(str(c) for c in self.channels)
        if not chans:
            raise ValueError(f"AdaptiveParams({self.element}) with a descriptor needs channels "
                             "(e.g. ('onsite.s', 'hopping.s'))")
        if len(set(chans)) != len(chans):
            raise ValueError(f"AdaptiveParams({self.element}): repeated channels {chans}")
        for c in chans:
            kind, dot, shell = c.partition(".")
            ok = (c == "repulsive" or (kind in ("onsite", "hopping") and dot
                                       and shell in SHELL_LETTERS and len(shell) == 1))
            if not ok:
                raise ValueError(f"channel {c!r}: use 'onsite.<s|p|d|f>', 'hopping.<s|p|d|f>' "
                                 "or 'repulsive'")
        object.__setattr__(self, "channels", chans)
        reg = self.regressor
        if reg is None:
            reg = Linear(int(self.descriptor.size), n_out=len(chans))
            object.__setattr__(self, "regressor", reg)
        if reg.n_out != len(chans):
            raise ValueError(f"AdaptiveParams({self.element}): the regressor has {reg.n_out} "
                             f"outputs for {len(chans)} channels")
        if reg.n_in != self.descriptor.size:
            raise ValueError(f"AdaptiveParams({self.element}): the regressor takes {reg.n_in} "
                             f"inputs, the descriptor gives {self.descriptor.size}")
        theta = reg.initial() if self.theta is None else np.ravel(np.asarray(self.theta, float))
        if len(theta) != reg.size:
            raise ValueError(f"AdaptiveParams({self.element}): theta needs {reg.size} "
                             f"coefficients, not {len(theta)}")
        object.__setattr__(self, "theta", tuple(float(v) for v in theta))

    def __post_init__(self):
        if self.descriptor is not None:
            self._check_descriptor_mode()
        elif self.regressor is not None or self.channels or self.theta is not None:
            raise ValueError("regressor, channels and theta need a descriptor")
        if self.angular is not None and self.angular.constant:
            raise ValueError("the angular basis has no constant (the CN basis has it): "
                             "RBF(centres, width, constant=False)")
        n = self.size
        for name in ("onsite", "hopping"):
            d = {}
            for shell, coef in dict(getattr(self, name)).items():
                if shell not in SHELL_LETTERS:
                    raise ValueError(f"{name}: shells are letters s, p, d, f, not {shell!r}")
                coef = tuple(float(c) for c in np.ravel(coef))
                if len(coef) != n:
                    raise ValueError(f"{name}[{shell!r}] needs {n} coefficients (the basis size)")
                d[shell] = coef
            object.__setattr__(self, name, d)
        if self.repulsive is not None:
            coef = tuple(float(c) for c in np.ravel(self.repulsive))
            if len(coef) != n:
                raise ValueError(f"repulsive needs {n} coefficients (the basis size)")
            object.__setattr__(self, "repulsive", coef)
        pr = {}
        for el, rep in dict(self.pair_repulsive).items():
            if not isinstance(rep, PairRepulsive):
                rep = PairRepulsive(**rep) if isinstance(rep, dict) else PairRepulsive(*rep)
            pr[str(el)] = rep
        object.__setattr__(self, "pair_repulsive", pr)
        r_on, r_off = (float(x) for x in self.window)
        if not 0.0 < r_on < r_off:
            raise ValueError("window needs 0 < r_on < r_off")
        object.__setattr__(self, "window", (r_on, r_off))
        if self.neighbours is not None:
            object.__setattr__(self, "neighbours", tuple(self.neighbours))

    # -- the coefficients as one vector (fitting) ----------------------------------------------
    def parameter_names(self) -> list[str]:
        """Names of the entries of :meth:`vector`, in order."""
        if self.descriptor_mode:
            out = []
            for name in self.regressor.names:
                head, dot, rest = name.partition(".")
                if dot and head.startswith("out") and head[3:].isdigit():
                    out.append(f"{self.element}.{self.channels[int(head[3:])]}.{rest}")
                else:
                    out.append(f"{self.element}.reg.{name}")
            for el in sorted(self.pair_repulsive):
                out += [f"{self.element}-{el}.pair[{k}]"
                        for k in range(len(self.pair_repulsive[el].coefficients))]
            return out
        n = self.size
        out = [f"{self.element}.onsite.{s}[{k}]" for s in sorted(self.onsite) for k in range(n)]
        out += [f"{self.element}.hopping.{s}[{k}]" for s in sorted(self.hopping) for k in range(n)]
        if self.repulsive is not None:
            out += [f"{self.element}.repulsive[{k}]" for k in range(n)]
        for el in sorted(self.pair_repulsive):
            out += [f"{self.element}-{el}.pair[{k}]"
                    for k in range(len(self.pair_repulsive[el].coefficients))]
        return out

    def vector(self) -> np.ndarray:
        if self.descriptor_mode:
            parts = [self.theta]
            parts += [self.pair_repulsive[el].coefficients for el in sorted(self.pair_repulsive)]
            return np.concatenate([np.asarray(p, float) for p in parts])
        parts = [self.onsite[s] for s in sorted(self.onsite)]
        parts += [self.hopping[s] for s in sorted(self.hopping)]
        if self.repulsive is not None:
            parts.append(self.repulsive)
        parts += [self.pair_repulsive[el].coefficients for el in sorted(self.pair_repulsive)]
        return np.concatenate([np.asarray(p, float) for p in parts]) if parts else np.zeros(0)

    def with_vector(self, x) -> AdaptiveParams:
        x = np.asarray(x, float)
        if len(x) != len(self.vector()):
            raise ValueError(f"needs {len(self.vector())} values, not {len(x)}")
        if self.descriptor_mode:
            k = self.regressor.size
            theta = tuple(x[:k])
            pair = {}
            for el in sorted(self.pair_repulsive):
                r = self.pair_repulsive[el]
                m = len(r.coefficients)
                pair[el], k = replace(r, coefficients=tuple(x[k:k + m])), k + m
            return replace(self, theta=theta, pair_repulsive=pair)
        n, k = self.size, 0
        onsite, hopping = {}, {}
        for s in sorted(self.onsite):
            onsite[s], k = tuple(x[k:k + n]), k + n
        for s in sorted(self.hopping):
            hopping[s], k = tuple(x[k:k + n]), k + n
        rep = None
        if self.repulsive is not None:
            rep, k = tuple(x[k:k + n]), k + n
        pair = {}
        for el in sorted(self.pair_repulsive):
            r = self.pair_repulsive[el]
            m = len(r.coefficients)
            pair[el], k = replace(r, coefficients=tuple(x[k:k + m])), k + m   # wall kept
        return replace(self, onsite=onsite, hopping=hopping, repulsive=rep, pair_repulsive=pair)

    # -- descriptor -------------------------------------------------------------------------------
    def windows(self, elements) -> dict:
        """(r_on, r_off) in Bohr for every neighbour element counted."""
        out = {}
        ra = covalent_radii[atomic_numbers[self.element]] / BOHR
        for el in set(elements):
            if self.neighbours is not None and el not in self.neighbours:
                continue
            if self.window_scaled:
                rb = covalent_radii[atomic_numbers[el]] / BOHR
                out[el] = (self.window[0] * (ra + rb), self.window[1] * (ra + rb))
            else:
                out[el] = self.window
        return out

    def _neighbours(self, elements, pos, cell, pbc):
        """The counted neighbour entries of the element's atoms: i, j, vec, d, f(r), f'(r)/r."""
        win = self.windows(elements)
        if not win or self.element not in elements:
            z = np.zeros(0)
            return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3)), z, z, z
        rc = max(w[1] for w in win.values())
        periodic = cell is not None and pbc is not None and np.any(pbc)
        i, j, _, vec, d = neighbor_list(np.asarray(pos, float), cell if periodic else None,
                                        pbc if periodic else None, rc)
        el = np.array(elements)
        r_on = np.array([win.get(e, (0.0, 0.0))[0] for e in el])
        r_off = np.array([win.get(e, (0.0, 0.0))[1] for e in el])
        keep = (el[i] == self.element) & (d < r_off[j])
        i, j, vec, d = i[keep], j[keep], vec[keep], d[keep]
        span = r_off[j] - r_on[j]
        s, ds = smoothstep((d - r_on[j]) / span)
        return i, j, vec, d, 1.0 - s, -ds / span / d

    def coordination_numbers(self, elements, pos, cell=None, pbc=None):
        """CN of every atom of the element (0 for the others) and its gradient pairs
        (i, j, vec, f'(r)/r), i the atom whose CN it is."""
        elements = list(elements)
        i, j, vec, _, f, fr = self._neighbours(elements, pos, cell, pbc)
        return np.bincount(i, f, minlength=len(elements)), (i, j, vec, fr)

    def descriptors(self, elements, pos, cell=None, pbc=None):
        """CN and the angular q of every atom of the element (0 for the others), the CN
        gradient pairs (i, j, vec, f'(r)/r) and the q gradient pairs (i, j, vec, dq_i/dvec)."""
        elements = list(elements)
        natom = len(elements)
        i, j, vec, d, f, fr = self._neighbours(elements, pos, cell, pbc)
        cn = np.bincount(i, f, minlength=natom)
        q = np.zeros(natom)
        dq = np.zeros((len(i), 3))
        if self.angular is not None:
            for a in np.unique(i):
                e = np.flatnonzero(i == a)
                q[a], dq[e] = _angular(vec[e], d[e], f[e], fr[e], cn[a])
        return cn, q, (i, j, vec, fr), (i, j, vec, dq)


def _angular(vec, d, f, fr, cn):
    """q = S / (CN^2 + eps) with S = sum_BC f_B f_C P3(u_B . u_C) over one atom's neighbour
    entries, and dq/dvec_B (n, 3)."""
    u = vec / d[:, None]
    c = np.clip(u @ u.T, -1.0, 1.0)
    p3 = 0.5 * (5.0 * c ** 3 - 3.0 * c)
    dp3 = 0.5 * (15.0 * c ** 2 - 3.0)
    s = f @ p3 @ f
    pf = p3 @ f
    m = dp3 * f[None, :]
    ds = 2.0 * (pf * fr)[:, None] * vec + \
        2.0 * (f / d)[:, None] * (m @ u - ((m * c).sum(1))[:, None] * u)
    den = cn * cn + ANGULAR_EPS
    q = s / den
    dq = ds / den - (2.0 * cn * s / den ** 2) * fr[:, None] * vec
    return q, dq


def make_models(spec) -> tuple:
    """A tuple of AdaptiveParams, one per element, from one or a list of them."""
    if spec is None:
        return ()
    models = (spec,) if isinstance(spec, AdaptiveParams) else tuple(spec)
    for m in models:
        if not isinstance(m, AdaptiveParams):
            raise TypeError("adaptive takes AdaptiveParams (or a list of them)")
    els = [m.element for m in models]
    if len(set(els)) != len(els):
        raise ValueError("one AdaptiveParams per element")
    return models


# ---- the terms of one geometry ---------------------------------------------------------------------
@dataclass
class AdaptiveTerms:
    """What the calculator keeps of the models for one geometry."""
    models: tuple
    elements: list
    cn: np.ndarray                      # (n_atom,)
    cn_pairs: tuple                     # (i, j, vec, f'(r)/r) of all models' atoms
    phi: dict                           # element -> (atoms, Phi, dPhi/dCN, dPhi/dq)
    onsite: dict                        # (element, shell) -> p per atom of the element
    hopping: dict
    rep_scale: np.ndarray               # (n_atom,) p of the repulsive channel (0: none)
    orb_scale: np.ndarray               # c_mu (n_orb,)
    onsite_shift: np.ndarray            # (n_orb,)
    shell_of: dict                      # (element, shell) -> list of orbital index arrays per atom
    rep_energy: float = 0.0
    rep_gradient: np.ndarray | None = None
    rep_virial: np.ndarray | None = None
    rep_dedp: np.ndarray | None = None              # dE_rep/dp_repulsive per atom
    rep_dedc: dict | None = None                    # (element, neighbour) -> dE/dc_n
    q: np.ndarray | None = None                     # the angular descriptor (M8b)
    q_pairs: tuple | None = None                    # (i, j, vec, dq_i/dvec)
    feat: dict = field(default_factory=dict)        # descriptor mode: element -> (atoms,
                                                    # Features, P, dP/dx or None, dP/dtheta)


def setup(models, species, pos, cell, pbc, basis, sk, mode: str, derivatives: bool):
    """The coordination numbers, the parameters of every atom, the orbital scales and on-site
    shifts, and the (scaled and added) repulsive with its gradient."""
    elements = [element_of_label(s) for s in species]
    natom = len(elements)
    pos = np.asarray(pos, float)
    cn = np.zeros(natom)
    q = np.zeros(natom)
    pi, pj, pv, pf = [], [], [], []
    qi, qj, qv, qd = [], [], [], []
    phi, onsite, hopping = {}, {}, {}
    rep_scale = np.zeros(natom)
    orb_scale = np.ones(basis.norb)
    onsite_shift = np.zeros(basis.norb)
    shell_of = {}
    feat = {}
    for m in models:
        atoms = np.array([a for a in range(natom) if elements[a] == m.element], int)
        if m.descriptor_mode:                    # descriptor + regressor (A series)
            if derivatives and not m.descriptor.has_derivatives:
                raise ValueError(f"the {m.element} descriptor ({type(m.descriptor).__name__}) "
                                 "has no position derivatives: no forces or stress with it "
                                 "(energies and bands only: forces=False)")
            c, _ = m.coordination_numbers(elements, pos, cell, pbc)
            cn[atoms] = c[atoms]
            fx = m.descriptor.compute(elements, pos, cell, pbc, derivatives)
            if not np.array_equal(fx.atoms, atoms):
                raise ValueError(f"the {m.element} descriptor does not describe all "
                                 f"{m.element} atoms")
            theta = np.asarray(m.theta, float)
            pv_ = m.regressor.value(fx.values, theta)
            dpx = m.regressor.d_input(fx.values, theta) if derivatives else None
            dpt = m.regressor.d_theta(fx.values, theta)
            feat[m.element] = (atoms, fx, pv_, dpx, dpt)
            phi[m.element] = (atoms, None, None, None)
            for col, key in enumerate(m.channel_keys()):
                if key[0] == "onsite":
                    onsite[(m.element, key[2])] = pv_[:, col]
                elif key[0] == "hopping":
                    hopping[(m.element, key[2])] = pv_[:, col]
                else:
                    rep_scale[atoms] = pv_[:, col]
        else:
            c, qa, (i, j, v, f), (_, _, _, dqv) = m.descriptors(elements, pos, cell, pbc)
            cn[atoms] = c[atoms]
            q[atoms] = qa[atoms]
            pi.append(i), pj.append(j), pv.append(v), pf.append(f)
            if m.angular is not None:
                qi.append(i), qj.append(j), qv.append(v), qd.append(dqv)
            ph, dph, dphq = m.features(cn[atoms], q[atoms])
            phi[m.element] = (atoms, ph, dph, dphq)
            for chan, store in (("onsite", onsite), ("hopping", hopping)):
                for shell, coef in getattr(m, chan).items():
                    store[(m.element, shell)] = ph @ np.asarray(coef)
            if m.repulsive is not None:
                rep_scale[atoms] = ph @ np.asarray(m.repulsive)
        on_shells, hop_shells = m.onsite_shells(), m.hopping_shells()
        for shell in sorted(on_shells | hop_shells):
            l = SHELL_LETTERS.index(shell)
            orbs = []
            for a in atoms:
                o0 = basis.offsets[a]
                sel = [o0 + k for k, ll in enumerate(_orbital_l(basis, a)) if ll == l]
                if not sel:
                    raise ValueError(f"{m.element} has no {shell} shell in this basis")
                orbs.append(np.array(sel, int))
            shell_of[(m.element, shell)] = orbs
            for a_idx, o in enumerate(orbs):
                if shell in on_shells:
                    onsite_shift[o] += onsite[(m.element, shell)][a_idx]
                if shell in hop_shells:
                    orb_scale[o] = np.exp(0.5 * hopping[(m.element, shell)][a_idx])
    cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape)
    cn_pairs = (cat(pi, 0).astype(int), cat(pj, 0).astype(int), cat(pv, (0, 3)), cat(pf, 0))
    q_pairs = (cat(qi, 0).astype(int), cat(qj, 0).astype(int), cat(qv, (0, 3)), cat(qd, (0, 3)))
    terms = AdaptiveTerms(models, elements, cn, cn_pairs, phi, onsite, hopping, rep_scale,
                          orb_scale, onsite_shift, shell_of, q=q, q_pairs=q_pairs, feat=feat)
    _repulsive(terms, species, pos, cell, pbc, sk, mode, derivatives)
    return terms


def _orbital_l(basis, a):
    """The angular momentum of every orbital of atom a."""
    out = []
    for l in basis.shells[a]:
        out += [l] * (2 * l + 1)
    return out


def scale_pairs(terms: AdaptiveTerms, pb):
    """H0 (and dH0) of the pair blocks scaled by c_row c_col, in place (exactly 1: unchanged)."""
    c = terms.orb_scale[pb.rows] * terms.orb_scale[pb.cols]
    pb.h *= c
    if pb.dh is not None:
        pb.dh *= c[:, None]


def _repulsive(terms, species, pos, cell, pbc, sk, mode, grad):
    """E_rep = 1/2 sum over ordered pairs (V^SK + V^model)(r) exp((p_i + p_j)/2), its gradient
    at fixed parameters, virial, dE/dp per atom and dE/dc of the pair repulsives."""
    natom = len(species)
    labels = sorted(set(species))
    elements = terms.elements
    added = {}                         # ordered (element a, element b) -> (owner key, PairRepulsive)
    for m in terms.models:
        for el, rep in m.pair_repulsive.items():
            key = (m.element, el)
            for pair in {key, (el, m.element)}:
                if pair in added:
                    raise ValueError(f"the pair repulsive {m.element}-{el} is given twice")
                added[pair] = (key, rep)
    rc = max([r.cutoff for a in labels for b in labels
              if (r := sk.repulsive(a, b, mode)) is not None]
             + [r.cutoff for _, r in added.values()], default=0.0)
    terms.rep_gradient = np.zeros((natom, 3)) if grad else None
    terms.rep_virial = np.zeros((3, 3)) if grad else None
    terms.rep_dedp = np.zeros(natom)
    terms.rep_dedc = {}
    if rc == 0.0:
        return
    periodic = cell is not None and pbc is not None and np.any(pbc)
    i, j, _, vec, d = neighbor_list(pos, cell if periodic else None, pbc if periodic else None, rc)
    spec = np.array(species)
    el = np.array(elements)
    s = np.exp(0.5 * (terms.rep_scale[i] + terms.rep_scale[j]))
    e_tot = 0.0
    for a in labels:                              # the SK repulsive, label by label
        for b in labels:
            rep = sk.repulsive(a, b, mode)
            m = (spec[i] == a) & (spec[j] == b)
            if rep is None or not m.any():
                continue
            v, dv = rep(d[m])
            e_tot += _add_pair_term(terms, i[m], j[m], vec[m], d[m], s[m], v, dv, grad)
    for (ea, eb), (key, rep) in sorted(added.items()):   # the model's pair repulsives
        m = (el[i] == ea) & (el[j] == eb) & (d < rep.cutoff)
        if not m.any():
            continue
        p, dp = rep.powers(d[m])
        c = np.asarray(rep.coefficients)
        w, dw = rep.core(d[m])
        e_tot += _add_pair_term(terms, i[m], j[m], vec[m], d[m], s[m], p @ c + w, dp @ c + dw,
                                grad)
        terms.rep_dedc[key] = terms.rep_dedc.get(key, 0.0) + 0.5 * (s[m][:, None] * p).sum(0)
    terms.rep_energy = e_tot


def _add_pair_term(terms, i, j, vec, d, s, v, dv, grad):
    """Adds 1/2 sum s V over these ordered pairs: energy (returned), gradient and virial at
    fixed parameters, dE/dp."""
    vs = v * s
    np.add.at(terms.rep_dedp, i, 0.25 * vs)
    np.add.at(terms.rep_dedp, j, 0.25 * vs)
    if grad:
        g = (0.5 * s * dv / d)[:, None] * vec
        np.add.at(terms.rep_gradient, i, -g)
        np.add.at(terms.rep_gradient, j, g)
        terms.rep_virial += g.T @ vec
    return 0.5 * vs.sum()


# ---- dE/dp, forces and the derivatives of the coefficients ------------------------------------------
def parameter_derivatives(terms: AdaptiveTerms, pb, p_e, d_onsite) -> dict:
    """dE/dp per channel: {("onsite"|"hopping", element, shell): (n_atom_el,),
    ("repulsive", element): (n_atom_el,)} at fixed density (``p_e`` the density at the pair
    elements, ``d_onsite`` the on-site density D_mu,mu)."""
    out = {}
    if terms.hopping:
        he = p_e * pb.h                           # scaled H0: d(c_r c_c H0)/dp = 1/2 (...) H
        per_orb = np.bincount(pb.rows, he, minlength=len(terms.orb_scale)) + \
            np.bincount(pb.cols, he, minlength=len(terms.orb_scale))
    for (el, shell), orbs in terms.shell_of.items():
        if (el, shell) in terms.onsite:
            out[("onsite", el, shell)] = np.array([d_onsite[o].sum() for o in orbs])
        if (el, shell) in terms.hopping:
            out[("hopping", el, shell)] = np.array([0.5 * per_orb[o].sum() for o in orbs])
    for m in terms.models:
        if m.has_repulsive:
            atoms = terms.phi[m.element][0]
            out[("repulsive", m.element)] = terms.rep_dedp[atoms]
    return out


def coordinate_gradient(terms: AdaptiveTerms, dedp: dict, virial: bool):
    """The part of dE/dR and of the virial from the parameters' dependence on CN (and on the
    angular descriptor q)."""
    natom = len(terms.cn)
    dedcn = np.zeros(natom)
    dedq = np.zeros(natom)
    for m in terms.models:
        if m.descriptor_mode:
            continue
        atoms, _, dph, dphq = terms.phi[m.element]
        for key, coef in _channels(m):
            d = dedp.get(key)
            if d is not None:
                dedcn[atoms] += d * (dph @ np.asarray(coef))
                if m.angular is not None:
                    dedq[atoms] += d * (dphq @ np.asarray(coef))
    i, j, vec, fr = terms.cn_pairs
    g = (dedcn[i] * fr)[:, None] * vec
    if terms.q_pairs is not None and len(terms.q_pairs[0]):
        qi, qj, qv, dq = terms.q_pairs
        i, j, vec = np.concatenate([i, qi]), np.concatenate([j, qj]), np.vstack([vec, qv])
        g = np.vstack([g, dedq[qi][:, None] * dq])
    grad = np.zeros((natom, 3))
    np.add.at(grad, j, g)
    np.add.at(grad, i, -g)
    vir = g.T @ vec if virial else None
    for m in terms.models:                       # descriptor models: through their pairs
        if not m.descriptor_mode:
            continue
        _, fx, _, dpx, _ = terms.feat[m.element]
        if dpx is None:
            raise ValueError(f"the {m.element} descriptor was computed without derivatives")
        dedx = np.einsum("ac,acf->af", _channel_dedp(m, dedp, len(fx.atoms)), dpx)
        g2, v2 = fx.chain(dedx, natom, virial)
        grad = grad + g2
        if virial:
            vir = vir + v2
    return grad, vir


def _channel_dedp(m, dedp: dict, n: int) -> np.ndarray:
    """dF/dp (n_atom_el, n_channels) of a descriptor model, in the regressor's output order."""
    out = np.zeros((n, len(m.channels)))
    for col, key in enumerate(m.channel_keys()):
        d = dedp.get(key)
        if d is not None:
            out[:, col] = d
    return out


def channel_jacobians(m: AdaptiveParams, terms: AdaptiveTerms) -> dict:
    """dp_A/d(:meth:`AdaptiveParams.vector`) of every channel of a model at one geometry:
    {channel key: (atoms of the element, (n_atom_el, len(vector)))} (pydftb.banddiff)."""
    n_x = len(m.vector())
    out = {}
    if m.descriptor_mode:
        atoms, _, _, _, dpt = terms.feat[m.element]
        for col, key in enumerate(m.channel_keys()):
            jm = np.zeros((len(atoms), n_x))
            jm[:, :dpt.shape[2]] = dpt[:, col, :]
            out[key] = (atoms, jm)
        return out
    atoms, ph, _, _ = terms.phi[m.element]
    k = 0
    for key, coef in _channels(m):
        jm = np.zeros((len(atoms), n_x))
        jm[:, k:k + len(coef)] = ph
        out[key] = (atoms, jm)
        k += len(coef)
    return out


def _channels(m: AdaptiveParams):
    for s in sorted(m.onsite):
        yield ("onsite", m.element, s), m.onsite[s]
    for s in sorted(m.hopping):
        yield ("hopping", m.element, s), m.hopping[s]
    if m.repulsive is not None:
        yield ("repulsive", m.element), m.repulsive


@dataclass
class AdaptiveResult:
    """``DFTBResult.adaptive``: the coordination numbers, the parameters of every atom and the
    derivatives of the free energy with respect to every model coefficient (at fixed density,
    i.e. exact for the Mermin free energy), in the order of ``AdaptiveParams.vector``."""
    cn: np.ndarray
    parameters: dict                    # channel key -> p per atom of the element
    gradients: dict                     # element -> dF/d vector (same order as vector())
    q: np.ndarray | None = None         # the angular descriptor (M8b; zeros without it)
    dedp: dict | None = None            # channel key -> dF/dp per atom of the element
    features: dict | None = None        # descriptor models: element -> (atoms, values)

    def gradient(self, models) -> np.ndarray:
        """dF/d(the concatenated vectors of ``models``)."""
        return np.concatenate([self.gradients[m.element] for m in models])


def result(terms: AdaptiveTerms, dedp: dict) -> AdaptiveResult:
    grads, params, feats = {}, {}, {}
    for m in terms.models:
        if m.descriptor_mode:
            atoms, fx, pv_, _, dpt = terms.feat[m.element]
            d = _channel_dedp(m, dedp, len(atoms))
            parts = [np.einsum("ac,acs->s", d, dpt)]
            for col, key in enumerate(m.channel_keys()):
                params[key] = pv_[:, col].copy()
            for el in sorted(m.pair_repulsive):
                parts.append(np.asarray(terms.rep_dedc.get(
                    (m.element, el), np.zeros(len(m.pair_repulsive[el].coefficients)))))
            grads[m.element] = np.concatenate(parts)
            feats[m.element] = (atoms.copy(), fx.values.copy())
            continue
        atoms, ph, *_ = terms.phi[m.element]
        parts = []
        for key, _ in _channels(m):
            d = dedp.get(key)
            parts.append(ph.T @ d if d is not None else np.zeros(m.size))
            params[key] = (terms.onsite.get(key[1:]) if key[0] == "onsite"
                           else terms.hopping.get(key[1:]) if key[0] == "hopping"
                           else terms.rep_scale[atoms])
        for el in sorted(m.pair_repulsive):
            parts.append(np.asarray(terms.rep_dedc.get((m.element, el),
                                                       np.zeros(len(m.pair_repulsive[el].coefficients)))))
        grads[m.element] = np.concatenate(parts) if parts else np.zeros(0)
    return AdaptiveResult(terms.cn.copy(), params, grads,
                          None if terms.q is None else terms.q.copy(),
                          {k: np.asarray(v, float).copy() for k, v in dedp.items()},
                          feats or None)


def adaptive_from_dict(d: dict) -> AdaptiveParams:
    """The AdaptiveParams of :meth:`AdaptiveParams.to_dict`."""
    from .descriptors import descriptor_from_dict
    from .regressors import regressor_from_dict
    def rbf(b):
        if b is None:
            return None
        return RBF(tuple(b["centres"]), b["width"], b["constant"],
                   None if b.get("clamp") is None else tuple(b["clamp"]))
    return AdaptiveParams(
        d["element"], basis=rbf(d["basis"]),
        onsite={k: tuple(v) for k, v in d.get("onsite", {}).items()},
        hopping={k: tuple(v) for k, v in d.get("hopping", {}).items()},
        repulsive=None if d.get("repulsive") is None else tuple(d["repulsive"]),
        pair_repulsive={el: PairRepulsive(r["cutoff"], tuple(r["coefficients"]), r.get("wall", 0.0),
                                          r.get("wall_strength", 0.0))
                        for el, r in d.get("pair_repulsive", {}).items()},
        window=tuple(d.get("window", DEFAULT_WINDOW)), window_scaled=d.get("window_scaled", True),
        neighbours=None if d.get("neighbours") is None else tuple(d["neighbours"]),
        angular=rbf(d.get("angular")),
        descriptor=None if d.get("descriptor") is None else descriptor_from_dict(d["descriptor"]),
        regressor=None if d.get("regressor") is None else regressor_from_dict(d["regressor"]),
        channels=tuple(d.get("channels", ())),
        theta=None if d.get("theta") is None else tuple(d["theta"]))


def _register():
    from . import envmodel
    envmodel.register("adaptive", adaptive_from_dict)


_register()
