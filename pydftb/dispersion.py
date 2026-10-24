"""Dispersion corrections, as DFTB+ ``Hamiltonian { Dispersion = ... }``.

==============  ==========================  ===========================================
DFTB+ keyword   pydftb                      implementation
==============  ==========================  ===========================================
LennardJones    :class:`LennardJones`       native (UFF or own parameters)
SlaterKirkwood  :class:`SlaterKirkwood`     native
DftD3           :class:`DFTD3`              ``dftd3`` library (s-dftd3), as DFTB+
SimpleDftD3     :class:`SimpleDFTD3`        ``dftd3`` library, BJ two-body terms only
DftD4           :class:`DFTD4`              ``dftd4`` library (the reference implementation)
==============  ==========================  ===========================================

Every model returns :class:`DispersionTerms`: the energy (Ha), the gradient dE/dR
(Ha/Bohr) and the virial dE/d(strain) (Ha), the convention of
``calculator.DFTB._gradient``.  The energy depends on the geometry only, so it is
evaluated once per calculation, outside the SCC loop.  Parameters are in DFTB+'s
default units (Bohr, Hartree).  Formulas, the DFTB+ source they follow and where
DFTB+ itself goes wrong: ``docs/design/M2d-dispersion.html``.
"""
from __future__ import annotations

import copy
import importlib
import math
from typing import NamedTuple

import numpy as np
from ase.data import atomic_numbers as _NUMBERS
from ase.data import chemical_symbols
from scipy.special import erfc

from .constants import BOHR, KCAL_MOL, N_SEARCH_ITER, TOL_DISP_DAMP, TOL_DISPERSION
from .neighbors import neighbor_list
from .system import element_of_label


class DispersionTerms(NamedTuple):
    energy: float
    gradient: np.ndarray | None      # (n_atom, 3) dE/dR
    virial: np.ndarray | None        # (3, 3) dE/d(strain)
    atoms: np.ndarray | None = None  # (n_atom,) energy of every atom, on request (M6f-4a)


def _sum_terms(*terms: DispersionTerms) -> DispersionTerms:
    energy = sum(t.energy for t in terms)
    atoms = None if any(t.atoms is None for t in terms) else sum(t.atoms for t in terms)
    if any(t.gradient is None for t in terms):
        return DispersionTerms(energy, None, None, atoms)
    return DispersionTerms(energy, sum(t.gradient for t in terms), sum(t.virial for t in terms),
                           atoms)


def _pair_terms(i, j, vec, d, e, de, natom, grad, atoms=False) -> DispersionTerms:
    """Pair energy e(d) and de/dd over ordered pairs (each pair is listed twice).
    ``atoms``: also the energy of every atom, half of each of its pairs (as DFTB+)."""
    energy = 0.5 * float(np.sum(e))
    per_atom = None
    if atoms:
        per_atom = np.zeros(natom)
        np.add.at(per_atom, i, 0.5 * e)
    if not grad:
        return DispersionTerms(energy, None, None, per_atom)
    g = (0.5 * de / d)[:, None] * vec            # dE/d(vec) of each ordered pair
    gradient = np.zeros((natom, 3))
    np.add.at(gradient, i, -g)
    np.add.at(gradient, j, g)
    return DispersionTerms(energy, gradient, g.T @ vec, per_atom)


def _zero_terms(natom, grad, atoms) -> DispersionTerms:
    per_atom = np.zeros(natom) if atoms else None
    if grad:
        return DispersionTerms(0.0, np.zeros((natom, 3)), np.zeros((3, 3)), per_atom)
    return DispersionTerms(0.0, None, None, per_atom)


def _periodicity(cell, pbc):
    """None for a cluster, else the (3,) periodicity mask."""
    if cell is None:
        return None
    pbc = np.ones(3, bool) if pbc is None else np.asarray(pbc, bool)
    return pbc if pbc.any() else None


def _native_pbc(cell, pbc, model):
    pbc = _periodicity(cell, pbc)
    if pbc is not None and not pbc.all():
        raise NotImplementedError(f"{model}: clusters or 3D periodic systems only (as in DFTB+)")
    return pbc


class Dispersion:
    """Base class.  ``atomic_numbers`` maps type labels to atomic numbers (DFTB+
    ``AtomicNumbers``); without an entry the element is read from the label."""

    def __init__(self, atomic_numbers: dict | None = None):
        self.atomic_numbers = dict(atomic_numbers or {})

    def numbers(self, labels) -> np.ndarray:
        return np.array([self.atomic_numbers[lab] if lab in self.atomic_numbers
                         else _NUMBERS[element_of_label(lab)] for lab in labels])

    def with_elements(self, mapping: dict) -> "Dispersion":
        """Copy whose label -> atomic number map is completed from ``mapping``;
        entries given explicitly take precedence."""
        new = copy.copy(self)
        new.atomic_numbers = {**mapping, **self.atomic_numbers}
        return new

    def evaluate(self, labels, pos, cell=None, pbc=None, grad: bool = True,
                 charge: float = 0.0, atoms: bool | str = False) -> DispersionTerms:
        """Energy, gradient and virial; ``atoms`` adds the energy of every atom (True: pairs
        half to each atom, three-body terms a third to each; "dftbplus": as DFTB+, which
        spreads the D3 library's energy evenly, E/N)."""
        raise NotImplementedError

    def atom_energies(self, labels, pos, cell=None, pbc=None, charge: float = 0.0,
                      dftbplus: bool = False) -> np.ndarray:
        """The energy of every atom (milestone M6f-4a), summing to the energy."""
        return self.evaluate(labels, pos, cell, pbc, grad=False, charge=charge,
                             atoms="dftbplus" if dftbplus else True).atoms


# ----------------------------------------------------------------------------
# r^-6 lattice sum (dftb/dispcommon.F90)
# ----------------------------------------------------------------------------
def _search_cutoff(error, tol: float = TOL_DISPERSION, x0: float = 1e-8) -> float:
    """Cutoff where ``error`` drops below ``tol``, found as DFTB+ does: doubling,
    then at most N_SEARCH_ITER bisection steps; returns the last point tried."""
    x, y = x0, error(x0)
    while y > tol:
        x *= 2.0
        if not math.isfinite(x):
            raise RuntimeError("dispersion cutoff search diverged")
        y = error(x)
    if x == x0:
        raise RuntimeError("dispersion cutoff search: error already below tolerance at x0")
    xl, yl, xr, yr = 0.5 * x, error(0.5 * x), x, y
    it = 1
    while yl - yr > tol and it <= N_SEARCH_ITER:
        x = 0.5 * (xl + xr)
        y = error(x)
        if y >= tol:
            xl, yl = x, y
        else:
            xr, yr = x, y
        it += 1
    if it > N_SEARCH_ITER:
        raise RuntimeError("dispersion cutoff search did not converge")
    return x


def _reciprocal_vectors(cell: np.ndarray, gcut: float) -> np.ndarray:
    """All G != 0 with |G| <= gcut, (nG, 3) Cartesian."""
    recip = 2.0 * np.pi * np.linalg.inv(cell).T          # rows b_i, a_i . b_j = 2 pi delta_ij
    nmax = np.floor(gcut * np.linalg.norm(cell, axis=1) / (2.0 * np.pi)).astype(int)
    m = np.array(np.meshgrid(*[np.arange(-n, n + 1) for n in nmax], indexing="ij")).reshape(3, -1).T
    g = m @ recip
    g2 = (g * g).sum(1)
    return g[(g2 > 0.0) & (g2 <= gcut * gcut)]


def ewald_r6(pos, cell, c6, grad: bool = True, rcut_min: float = 0.0,
             tol: float = TOL_DISPERSION, atoms: bool = False) -> DispersionTerms:
    """E = -1/2 sum_ij sum_L' C6_ij / |r_j + L - r_i|^6 of a 3D periodic system.

    Ewald splitting with DFTB+'s parameters (``dftb/dispcommon.F90``): eta is the
    optimal Ewald eta / sqrt(2), the real- and reciprocal-space cutoffs make the
    error estimates drop below ``tol``.  ``rcut_min`` enlarges the real-space
    cutoff (Slater-Kirkwood does).  ``c6`` is the (n_atom, n_atom) matrix.  ``atoms``: also
    the energy of every atom, as DFTB+'s addDispEGr_per_atom: half of each real-space pair,
    the reciprocal sum and the self terms of its row.
    """
    natom = len(pos)
    vol = abs(np.linalg.det(cell))
    lens = np.linalg.norm(cell, axis=1)
    a, b = np.argsort(lens, kind="stable")[:2]
    eta = math.sqrt(lens[a] * vol / (math.pi * np.linalg.norm(np.cross(cell[a], cell[b]))))
    eta /= math.sqrt(2.0)
    c6sum = float(np.abs(c6).sum())
    pi15 = math.pi ** 1.5
    if c6sum == 0.0:
        return _zero_terms(natom, grad, atoms)

    def real_error(r):
        return pi15 * c6sum / vol * eta * (r ** -4 + 1.0 / (r * r * eta * eta)
                                           + 0.5 / eta ** 4) * erfc(r / eta)

    def recip_error(g):
        return c6sum / (6.0 * math.sqrt(math.pi)) / eta ** 6 * (
            g * eta * math.exp(-(0.5 * g * eta) ** 2) + math.sqrt(math.pi) * erfc(0.5 * g * eta))

    rcut = max(_search_cutoff(real_error, tol), rcut_min)
    gcut = _search_cutoff(recip_error, tol)

    # real space: -C6 exp(-x^2) (1 + x^2 + x^4/2) / r^6, x = r / eta
    i, j, _, vec, d = neighbor_list(pos, cell, np.ones(3, bool), rcut)
    x2 = (d / eta) ** 2
    ex = np.exp(-x2)
    cij = c6[i, j]
    e = -cij * ex * (1.0 + x2 + 0.5 * x2 * x2) / d ** 6
    de = cij * ex * (((x2 + 3.0) * x2 + 6.0) * x2 + 6.0) / d ** 7
    real = _pair_terms(i, j, vec, d, e, de, natom, grad, atoms)

    # reciprocal space: -pi^1.5/(24V) sum_G |G|^3 h(b) S(G), b = |G| eta / 2
    gvec = _reciprocal_vectors(cell, gcut)
    gl = np.linalg.norm(gvec, axis=1)
    bb = 0.5 * gl * eta
    ebb = np.exp(-bb * bb)
    h = math.sqrt(math.pi) * erfc(bb) + (0.5 / (bb * bb) - 1.0) / bb * ebb
    phase = pos @ gvec.T
    cs, sn = np.cos(phase), np.sin(phase)
    c6c, c6s = c6 @ cs, c6 @ sn
    sg = (cs * c6c + sn * c6s).sum(0)                    # sum_ij C6_ij cos(G.(r_i - r_j))
    wg = gl ** 3 * h
    pref = pi15 / (24.0 * vol)
    e_rec = -pref * float(wg @ sg)
    e_self = float(np.trace(c6)) / (12.0 * eta ** 6)
    e_g0 = -pi15 * float(c6.sum()) / (6.0 * vol * eta ** 3)
    energy = real.energy + e_rec + e_self + e_g0
    per_atom = None
    if atoms:
        per_atom = (real.atoms - pref * ((cs * c6c + sn * c6s) @ wg)
                    + np.diag(c6) / (12.0 * eta ** 6) - pi15 * c6.sum(1) / (6.0 * vol * eta ** 3))
    if not grad:
        return DispersionTerms(energy, None, None, per_atom)
    g_rec = 2.0 * pref * (((sn * c6c - cs * c6s) * wg) @ gvec)
    t = 3.0 * gl * (math.sqrt(math.pi) * erfc(bb) - ebb / bb)
    w_rec = -e_rec * np.eye(3) + pref * (gvec * (sg * t)[:, None]).T @ gvec
    return DispersionTerms(energy, real.gradient + g_rec, real.virial + w_rec - e_g0 * np.eye(3),
                           per_atom)


# ----------------------------------------------------------------------------
# Lennard-Jones / UFF (dftb/dispuff.F90)
# ----------------------------------------------------------------------------
# UFF van der Waals distance x_I (Angstrom) and well depth D_I (kcal/mol),
# Rappe et al., J. Am. Chem. Soc. 114, 10024 (1992), Table 1.
_UFF_TABLE = """
    H 2.886 0.044   He 2.362 0.056   Li 2.451 0.025   Be 2.745 0.085   B 4.083 0.180   C 3.851 0.105
    N 3.660 0.069   O 3.500 0.060   F 3.364 0.050   Ne 3.243 0.042   Na 2.983 0.030   Mg 3.021 0.111
    Al 4.499 0.505   Si 4.295 0.402   P 4.147 0.305   S 4.035 0.274   Cl 3.947 0.227   Ar 3.868 0.185
    K 3.812 0.035   Ca 3.999 0.238   Sc 3.295 0.019   Ti 3.175 0.017   V 3.144 0.016   Cr 3.023 0.015
    Mn 2.961 0.013   Fe 2.912 0.013   Co 2.872 0.014   Ni 2.834 0.015   Cu 3.495 0.005   Zn 2.763 0.124
    Ga 4.383 0.415   Ge 4.280 0.379   As 4.230 0.309   Se 4.205 0.291   Br 4.189 0.251   Kr 4.141 0.220
    Rb 4.114 0.040   Sr 3.641 0.235   Y 3.345 0.072   Zr 3.124 0.069   Nb 3.165 0.059   Mo 3.052 0.056
    Tc 2.998 0.048   Ru 2.963 0.056   Rh 2.929 0.053   Pd 2.899 0.048   Ag 3.148 0.036   Cd 2.848 0.228
    In 4.463 0.599   Sn 4.392 0.567   Sb 4.420 0.449   Te 4.470 0.398   I 4.500 0.339   Xe 4.404 0.332
    Cs 4.517 0.045   Ba 3.703 0.364   La 3.522 0.017   Ce 3.556 0.013   Pr 3.606 0.010   Nd 3.575 0.010
    Pm 3.547 0.009   Sm 3.520 0.008   Eu 3.493 0.008   Gd 3.368 0.009   Tb 3.451 0.007   Dy 3.428 0.007
    Ho 3.409 0.007   Er 3.391 0.007   Tm 3.374 0.006   Yb 3.355 0.228   Lu 3.640 0.041   Hf 3.141 0.072
    Ta 3.170 0.081   W 3.069 0.067   Re 2.954 0.066   Os 3.120 0.037   Ir 2.840 0.073   Pt 2.754 0.080
    Au 3.293 0.039   Hg 2.705 0.385   Tl 4.347 0.680   Pb 4.297 0.663   Bi 4.370 0.518   Po 4.709 0.325
    At 4.750 0.284   Rn 4.765 0.248   Fr 4.900 0.050   Ra 3.677 0.404   Ac 3.478 0.033   Th 3.396 0.026
    Pa 3.424 0.022   U 3.395 0.022   Np 3.424 0.019   Pu 3.424 0.016   Am 3.381 0.014   Cm 3.326 0.013
    Bk 3.339 0.013   Cf 3.313 0.013   Es 3.299 0.012   Fm 3.286 0.012   Md 3.274 0.011   No 3.248 0.011
    Lr 3.236 0.011
"""
UFF = {s: (float(x), float(d)) for s, x, d in zip(*[iter(_UFF_TABLE.split())] * 3)}


class LennardJones(Dispersion):
    """Lennard-Jones dispersion, DFTB+ ``LennardJones``.

    ``parameters`` is ``"uff"`` (the UFF table above, DFTB+ ``UFFParameters {}``)
    or ``{label: (distance, energy)}`` in Bohr and Hartree.  Pairs combine as
    d = sqrt(d_i d_j), r = sqrt(r_i r_j): E = d r^12 / R^12 - 2 d r^6 / R^6; below
    r0 = 2^(-1/6) r (where E = 0) a polynomial wall U0 - U5 R^5 - U10 R^10 joins it
    smoothly.  Periodic systems sum the -c6/R^6 part by Ewald (:func:`ewald_r6`).
    """

    def __init__(self, parameters="uff", atomic_numbers: dict | None = None):
        super().__init__(atomic_numbers)
        if isinstance(parameters, str) and parameters.lower() != "uff":
            raise ValueError(f"parameters must be 'uff' or {{label: (distance, energy)}}, "
                             f"not {parameters!r}")
        self.parameters = "uff" if isinstance(parameters, str) else dict(parameters)

    def site_parameters(self, labels):
        """Distance (Bohr) and well depth (Hartree) of every atom."""
        if self.parameters == "uff":
            elements = [chemical_symbols[z] for z in self.numbers(labels)]
            missing = sorted(set(elements) - set(UFF))
            if missing:
                raise ValueError(f"no UFF parameters for {missing}")
            return (np.array([UFF[e][0] for e in elements]) / BOHR,
                    np.array([UFF[e][1] for e in elements]) * KCAL_MOL)
        missing = sorted(set(labels) - set(self.parameters))
        if missing:
            raise ValueError(f"no Lennard-Jones parameters for labels {missing}")
        vals = np.array([self.parameters[lab] for lab in labels], float).reshape(-1, 2)
        return vals[:, 0], vals[:, 1]

    def evaluate(self, labels, pos, cell=None, pbc=None, grad=True, charge=0.0, atoms=False):
        pbc = _native_pbc(cell, pbc, "LennardJones")
        pos = np.asarray(pos, float)
        dist, depth = self.site_parameters(labels)
        dij = np.sqrt(np.outer(depth, depth))
        rij = np.sqrt(np.outer(dist, dist))
        c6 = 2.0 * dij * rij ** 6
        c12 = dij * rij ** 12
        if pbc is None:
            rcut = (c6.max() / TOL_DISPERSION) ** (1.0 / 6.0)
            i, j, _, vec, d = neighbor_list(pos, None, None, rcut)
            e, de = self._pair(i, j, d, dij, rij, c6, c12, with_r6=True)
            return _pair_terms(i, j, vec, d, e, de, len(pos), grad, atoms)
        cell = np.asarray(cell, float)
        # r^-12 and the wall in real space (cutoff with a 100x tighter tolerance, as
        # DFTB+); every -c6/R^6, also inside the wall, comes from the Ewald sum
        rcut = (c12.max() / (TOL_DISPERSION * 1e-2)) ** (1.0 / 12.0)
        i, j, _, vec, d = neighbor_list(pos, cell, pbc, rcut)
        e, de = self._pair(i, j, d, dij, rij, c6, c12, with_r6=False)
        return _sum_terms(_pair_terms(i, j, vec, d, e, de, len(pos), grad, atoms),
                          ewald_r6(pos, cell, c6, grad, atoms=bool(atoms)))

    @staticmethod
    def _pair(i, j, d, dij, rij, c6, c12, with_r6):
        dd, rr, k6, k12 = dij[i, j], rij[i, j], c6[i, j], c12[i, j]
        f6 = 1.0 if with_r6 else 0.0
        d6 = d ** 6
        e_lj = k12 / (d6 * d6) - f6 * k6 / d6
        de_lj = (-12.0 * k12 / (d6 * d6) + 6.0 * f6 * k6 / d6) / d
        u0 = 396.0 / 25.0 * dd
        u5 = 2.0 ** (5.0 / 6.0) * 672.0 / 25.0 * dd / rr ** 5
        u10 = -(2.0 ** (2.0 / 3.0)) * 552.0 / 25.0 * dd / rr ** 10
        d5 = d ** 5
        e_wall = u0 - u5 * d5 - u10 * d5 * d5 + (1.0 - f6) * k6 / d6
        de_wall = (-5.0 * u5 * d5 - 10.0 * u10 * d5 * d5 - 6.0 * (1.0 - f6) * k6 / d6) / d
        wall = d <= 2.0 ** (-1.0 / 6.0) * rr
        return np.where(wall, e_wall, e_lj), np.where(wall, de_wall, de_lj)


# ----------------------------------------------------------------------------
# Slater-Kirkwood (dftb/dispslaterkirkw.F90)
# ----------------------------------------------------------------------------
class SlaterKirkwood(Dispersion):
    """Slater-Kirkwood dispersion with Elstner's damping, DFTB+ ``SlaterKirkwood``.

    Give either ``polar_radius_charge``, (n_atom, 3): polarisability (Bohr^3),
    van der Waals radius (Bohr) and effective electron number of every atom
    (DFTB+ ``PolarRadiusCharge = {...}``), or ``hybrid_dependent_pol``
    {label: {"covalent_radius": Bohr, "hybrid_polarisations": 13 values}}
    (DFTB+ ``HybridDependentPol``): polarisabilities and radii for 0, 1, 2, 3, 4,
    >= 5 neighbours, then the electron number.  Atoms closer than the sum of their
    covalent radii are neighbours.  DFTB+ counts them once, in the input
    geometry; here they are counted in every geometry.

    C6 = 1.5 a_i a_j / (sqrt(a_i/N_i) + sqrt(a_j/N_j)),
    R0 = (R_i^3 + R_j^3) / (R_i^2 + R_j^2),
    E = -C6 [1 - exp(-3 (R/R0)^7)]^4 / R^6.
    """

    def __init__(self, polar_radius_charge=None, hybrid_dependent_pol: dict | None = None,
                 atomic_numbers: dict | None = None):
        super().__init__(atomic_numbers)
        if (polar_radius_charge is None) == (hybrid_dependent_pol is None):
            raise ValueError("give either polar_radius_charge or hybrid_dependent_pol")
        self.polar_radius_charge = (None if polar_radius_charge is None
                                    else np.asarray(polar_radius_charge, float).reshape(-1, 3))
        self.hybrid_dependent_pol = None
        if hybrid_dependent_pol is not None:
            self.hybrid_dependent_pol = {}
            for lab, entry in hybrid_dependent_pol.items():
                vals = np.asarray(entry["hybrid_polarisations"], float).ravel()
                if vals.size != 13:
                    raise ValueError(f"hybrid_polarisations of {lab!r}: 13 values expected "
                                     f"(6 polarisabilities, 6 radii, 1 charge), got {vals.size}")
                self.hybrid_dependent_pol[lab] = (float(entry["covalent_radius"]), vals)

    def atom_parameters(self, labels, pos, cell=None, pbc=None):
        """Polarisability, van der Waals radius and electron number of every atom."""
        if self.polar_radius_charge is not None:
            p = self.polar_radius_charge
            if len(p) != len(labels):
                raise ValueError(f"polar_radius_charge has {len(p)} rows for {len(labels)} atoms")
            return p[:, 0], p[:, 1], p[:, 2]
        missing = sorted(set(labels) - set(self.hybrid_dependent_pol))
        if missing:
            raise ValueError(f"no hybrid_dependent_pol entry for labels {missing}")
        rcov = np.array([self.hybrid_dependent_pol[lab][0] for lab in labels])
        table = np.array([self.hybrid_dependent_pol[lab][1] for lab in labels])
        i, j, _, _, d = neighbor_list(pos, cell, pbc, 2.0 * rcov.max())
        nn = np.bincount(i[d <= rcov[i] + rcov[j]], minlength=len(labels))
        k = np.minimum(nn, 5)
        rows = np.arange(len(labels))
        return table[rows, k], table[rows, 6 + k], table[:, 12]

    def evaluate(self, labels, pos, cell=None, pbc=None, grad=True, charge=0.0, atoms=False):
        pbc = _native_pbc(cell, pbc, "SlaterKirkwood")
        pos = np.asarray(pos, float)
        cell = None if pbc is None else np.asarray(cell, float)
        alpha, rvdw, nel = self.atom_parameters(labels, pos, cell, pbc)
        eps = np.finfo(float).eps
        on = (alpha >= eps) & (rvdw >= eps)          # DFTB+ skips the other atoms
        pair_on = np.outer(on, on)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.sqrt(alpha / nel)
            c6 = np.where(pair_on, 1.5 * np.outer(alpha, alpha) / (s[:, None] + s[None, :]), 0.0)
            r0 = np.where(pair_on, (rvdw[:, None] ** 3 + rvdw[None, :] ** 3)
                          / (rvdw[:, None] ** 2 + rvdw[None, :] ** 2), 1.0)
        h0 = np.where(pair_on, 3.0 / r0 ** 7, 0.0)
        natom = len(pos)
        if not pair_on.any():
            return _zero_terms(natom, grad, atoms)
        if pbc is None:
            rcut = (c6.max() / TOL_DISPERSION) ** (1.0 / 6.0)
            i, j, _, vec, d = neighbor_list(pos, None, None, rcut)
            e, de = self._pair(c6[i, j], h0[i, j], d, corr=0.0)
            return _pair_terms(i, j, vec, d, e, de, natom, grad, atoms)
        # pure -C6/R^6 by Ewald, then the damping where 1 - f^4 >= TOL_DISP_DAMP
        rmax = r0[pair_on].max()
        damp_cut = rmax * (-math.log(1.0 - (1.0 - TOL_DISP_DAMP) ** 0.25) / 3.0) ** (1.0 / 7.0)
        i, j, _, vec, d = neighbor_list(pos, cell, pbc, damp_cut)
        e, de = self._pair(c6[i, j], h0[i, j], d, corr=-1.0)
        return _sum_terms(ewald_r6(pos, cell, c6, grad, rcut_min=damp_cut, atoms=bool(atoms)),
                          _pair_terms(i, j, vec, d, e, de, natom, grad, atoms))

    @staticmethod
    def _pair(c6, h0, d, corr):
        """-C6 (f^4 + corr) / R^6 and its derivative; f = 1 - exp(-h0 R^7)."""
        h1 = np.exp(-h0 * d ** 7)
        h2 = 1.0 - h1
        e = -c6 * (h2 ** 4 + corr) / d ** 6
        de = -c6 * (28.0 * h0 * h1 * h2 ** 3 - 6.0 * (h2 ** 4 + corr) / d ** 7)  # f' = 7 h0 R^6 h1
        return e, de


# ----------------------------------------------------------------------------
# D3 and D4 through the reference libraries
# ----------------------------------------------------------------------------
def _interface(package: str, model: str):
    try:
        return importlib.import_module(f"{package}.interface")
    except ImportError as err:
        raise ImportError(
            f"{model} needs the optional package {package!r}: conda install -c conda-forge "
            f"{package}-python (the version DFTB+ links is best for comparisons), "
            f"or pip install {package}") from err


def library_versions() -> dict:
    """Versions of the optional dispersion libraries (None if not installed)."""
    out = {}
    for package in ("dftd3", "dftd4"):
        try:
            out[package] = importlib.import_module(package).__version__
        except ImportError:
            out[package] = None
    return out


def _library_terms(result: dict, grad: bool, model=None, param=None) -> DispersionTerms:
    """``model``, ``param``: also the energy of every atom, the rows of the library's pairwise
    energies (two-body pairs half to each atom, three-body terms a third to each)."""
    per_atom = None
    if model is not None:
        pw = model.get_pairwise_dispersion(param)
        per_atom = (np.asarray(pw["additive pairwise energy"]).sum(1)
                    + np.asarray(pw["non-additive pairwise energy"]).sum(1))
    if not grad:
        return DispersionTerms(float(result["energy"]), None, None, per_atom)
    return DispersionTerms(float(result["energy"]), np.array(result["gradient"]),
                           np.array(result["virial"]), per_atom)


_DAMPING = {"beckejohnson": "bj", "bj": "bj", "rational": "bj",
            "zerodamping": "zero", "zero": "zero",
            "modifiedzerodamping": "mzero", "mzero": "mzero"}

# D3H5 H-H repulsion (Rezac and Hobza, JCTC 8, 141 (2012)), as DFTB+
HH_K = 0.30 * KCAL_MOL
HH_E = 14.31
HH_R0 = 2.35 / BOHR
HH_CUTOFF = 10.0


def hh_repulsion(numbers, pos, cell=None, pbc=None, grad=True, atoms=False) -> DispersionTerms:
    """E = sum over H-H pairs within 10 Bohr of k [1 - 1/(1 + exp(-e (R/r0 - 1)))]."""
    natom = len(pos)
    i, j, _, vec, d = neighbor_list(np.asarray(pos, float), cell,
                                    np.zeros(3, bool) if pbc is None else pbc, HH_CUTOFF)
    m = (numbers[i] == 1) & (numbers[j] == 1)
    i, j, vec, d = i[m], j[m], vec[m], d[m]
    ex = np.exp(-HH_E * (d / HH_R0 - 1.0))
    e = HH_K * ex / (1.0 + ex)
    de = -HH_K * HH_E * ex / (HH_R0 * (1.0 + ex) ** 2)
    return _pair_terms(i, j, vec, d, e, de, natom, grad, atoms)


class DFTD3(Dispersion):
    """DFT-D3 through the s-dftd3 library, DFTB+ ``DftD3``.

    ``damping``: "beckejohnson" (a1, a2), "zerodamping" (sr6, alpha6) or
    "modifiedzerodamping" (sr6, beta, alpha6).  ``threebody`` adds the
    Axilrod-Teller-Muto term (s9 = 1).  Cutoffs (Bohr) as DFTB+: pairs within
    ``cutoff``, coordination numbers and triples within ``cutoff_cn``.
    ``hhrepulsion`` adds the D3H5 H-H repulsion (:func:`hh_repulsion`).
    """

    def __init__(self, s6, s8, damping="beckejohnson", a1=None, a2=None, sr6=None,
                 alpha6=14.0, beta=None, threebody=False, hhrepulsion=False,
                 cutoff=math.sqrt(9000.0), cutoff_cn=40.0, atomic_numbers: dict | None = None):
        super().__init__(atomic_numbers)
        kind = _DAMPING.get(str(damping).lower().replace("_", "").replace(" ", ""))
        if kind is None:
            raise ValueError(f"unknown D3 damping {damping!r}; use 'beckejohnson', "
                             "'zerodamping' or 'modifiedzerodamping'")
        given = {"a1": a1, "a2": a2, "sr6": sr6, "beta": beta}
        need = {"bj": ("a1", "a2"), "zero": ("sr6",), "mzero": ("sr6", "beta")}[kind]
        absent = [k for k in need if given[k] is None]
        if absent:
            raise ValueError(f"{damping} damping needs {absent}")
        self.kind, self.s6, self.s8 = kind, float(s6), float(s8)
        self.a1, self.a2, self.sr6, self.alpha6, self.beta = a1, a2, sr6, float(alpha6), beta
        self.threebody, self.hhrepulsion = bool(threebody), bool(hhrepulsion)
        self.cutoff, self.cutoff_cn = float(cutoff), float(cutoff_cn)

    def _param(self, lib):
        s9 = 1.0 if self.threebody else 0.0
        if self.kind == "bj":       # DFTB+ passes alp = 14 (the ATM damping) for BJ
            return lib.RationalDampingParam(s6=self.s6, s8=self.s8, s9=s9, a1=self.a1,
                                            a2=self.a2, alp=14.0)
        if self.kind == "zero":
            return lib.ZeroDampingParam(s6=self.s6, s8=self.s8, s9=s9, rs6=self.sr6, rs8=1.0,
                                        alp=self.alpha6)
        return lib.ModifiedZeroDampingParam(s6=self.s6, s8=self.s8, s9=s9, rs6=self.sr6,
                                            rs8=1.0, alp=self.alpha6, bet=self.beta)

    # DFTB+ spreads the energy of its D3 library evenly over the atoms (getEnergies, E/N);
    # its native SimpleDftD3 is resolved by atom pairs
    _dftbplus_even = True

    def evaluate(self, labels, pos, cell=None, pbc=None, grad=True, charge=0.0, atoms=False):
        lib = _interface("dftd3", "DFT-D3")
        pbc = _periodicity(cell, pbc)
        pos = np.asarray(pos, float)
        numbers = self.numbers(labels)
        model = lib.DispersionModel(numbers, pos, None if pbc is None else np.asarray(cell, float),
                                    pbc)
        model.set_realspace_cutoff(self.cutoff, self.cutoff_cn, self.cutoff_cn)
        param = self._param(lib)
        terms = _library_terms(model.get_dispersion(param, grad=grad), grad,
                               model if atoms else None, param)
        if self.hhrepulsion:
            terms = _sum_terms(terms, hh_repulsion(numbers, pos, None if pbc is None else cell,
                                                   pbc, grad, bool(atoms)))
        if atoms == "dftbplus" and self._dftbplus_even:
            terms = terms._replace(atoms=np.full(len(pos), terms.energy / len(pos)))
        return terms


class SimpleDFTD3(DFTD3):
    """DFTB+ ``SimpleDftD3``: D3 with Becke-Johnson damping, two-body terms only.

    DFTB+ implements it natively; the same model is evaluated here by the D3
    library (pairs within ``cutoff_inter``, coordination numbers within
    ``cutoff_cn``).  Settings the library cannot reproduce (s10 != 0,
    WeightingFactor != 4) are refused.
    """

    _dftbplus_even = False

    def __init__(self, s8, a1, a2, s6=1.0, s10=0.0, cutoff_inter=64.0, cutoff_cn=40.0,
                 weighting_factor=4.0, atomic_numbers: dict | None = None):
        if s10 != 0.0:
            raise ValueError("SimpleDftD3: s10 != 0 is not supported (the D3 library has no C10 term)")
        if weighting_factor != 4.0:
            raise ValueError("SimpleDftD3: the D3 library uses WeightingFactor = 4")
        super().__init__(s6=s6, s8=s8, damping="beckejohnson", a1=a1, a2=a2, threebody=False,
                         cutoff=cutoff_inter, cutoff_cn=cutoff_cn, atomic_numbers=atomic_numbers)
        self.s10, self.weighting_factor = float(s10), float(weighting_factor)

    @property
    def cutoff_inter(self) -> float:
        return self.cutoff


class DFTD4(Dispersion):
    """DFT-D4 with EEQ charges through the dftd4 library, DFTB+ ``DftD4``.

    ``alpha`` is the three-body damping exponent; ``weighting_factor``,
    ``charge_steepness``, ``charge_scale`` are the library's wf, gc, ga.  The
    library's real-space cutoffs are fixed (pairs 60, triples 40, coordination
    numbers 30 Bohr; DFTB+ defaults to 64, 40, 40): other values are refused.
    DFTB+ evaluates D4 natively, and its periodic EEQ charges and three-body
    term are wrong; see ``docs/design/M2d-dispersion.html``.
    """

    LIBRARY_CUTOFFS = {"cutoff_inter": 60.0, "cutoff_three": 40.0, "cutoff_cn": 30.0}

    def __init__(self, s8, s9, a1, a2, s6=1.0, alpha=16.0, s10=0.0, weighting_factor=6.0,
                 charge_steepness=2.0, charge_scale=3.0, cutoff_inter=None, cutoff_three=None,
                 cutoff_cn=None, charge_model="eeq", atomic_numbers: dict | None = None):
        super().__init__(atomic_numbers)
        if s10 != 0.0:
            raise ValueError("DftD4: s10 != 0 is not supported by the dftd4 library")
        if str(charge_model).lower() != "eeq":
            raise NotImplementedError("DftD4: only ChargeModel = EEQ is implemented")
        for key, value in (("cutoff_inter", cutoff_inter), ("cutoff_three", cutoff_three),
                           ("cutoff_cn", cutoff_cn)):
            if value is not None and float(value) != self.LIBRARY_CUTOFFS[key]:
                raise ValueError(f"DftD4: {key} is fixed at {self.LIBRARY_CUTOFFS[key]} Bohr "
                                 "in the dftd4 library")
        self.s6, self.s8, self.s9, self.a1, self.a2 = map(float, (s6, s8, s9, a1, a2))
        self.alpha, self.s10 = float(alpha), float(s10)
        self.weighting_factor = float(weighting_factor)
        self.charge_steepness, self.charge_scale = float(charge_steepness), float(charge_scale)

    def evaluate(self, labels, pos, cell=None, pbc=None, grad=True, charge=0.0, atoms=False):
        lib = _interface("dftd4", "DFT-D4")
        pbc = _periodicity(cell, pbc)
        model_opts = {}
        if (self.charge_scale, self.charge_steepness, self.weighting_factor) != (3.0, 2.0, 6.0):
            model_opts = dict(ga=self.charge_scale, gc=self.charge_steepness,
                              wf=self.weighting_factor)
        model = lib.DispersionModel(self.numbers(labels), np.asarray(pos, float), float(charge),
                                    None if pbc is None else np.asarray(cell, float), pbc,
                                    **model_opts)
        param = lib.DampingParam(s6=self.s6, s8=self.s8, s9=self.s9, a1=self.a1, a2=self.a2,
                                 alp=self.alpha)
        return _library_terms(model.get_dispersion(param, grad=grad), grad,
                              model if atoms else None, param)


# ----------------------------------------------------------------------------
MODELS = {"lennardjones": LennardJones, "slaterkirkwood": SlaterKirkwood, "dftd3": DFTD3,
          "simpledftd3": SimpleDFTD3, "dftd4": DFTD4}


def make_dispersion(spec) -> Dispersion | None:
    """``None``, a :class:`Dispersion`, or ``{"model": <DFTB+ keyword>, **options}``,
    e.g. ``{"model": "dftd3", "damping": "beckejohnson", "a1": 0.5719,
    "a2": 3.6017, "s6": 1.0, "s8": 0.5883}``."""
    if spec is None or isinstance(spec, Dispersion):
        return spec
    if not isinstance(spec, dict) or "model" not in spec:
        raise TypeError("dispersion must be None, a Dispersion or a dict with a 'model' key")
    opts = dict(spec)
    name = str(opts.pop("model")).lower()
    if name not in MODELS:
        raise ValueError(f"unknown dispersion model {name!r}; choose from {sorted(MODELS)}")
    try:
        return MODELS[name](**opts)
    except TypeError as err:
        raise ValueError(f"dispersion model {name!r}: {err}") from None
