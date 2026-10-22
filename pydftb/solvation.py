"""Implicit solvation (milestone M6c), as DFTB+ 25.1's ``Solvation`` block for molecules:
the generalised Born model (``GeneralisedBorn``: Still or P16 kernel, the ALPB shape
term, CM5 charges, the hydrogen-bond correction), the solvent accessible surface area
model (``SASA``) and xtb-format GBSA parameter files.  Ports ``solvation/born.F90``,
``sasa.F90``, ``cm5.F90``, ``solvparser.F90`` and ``gbsafile.F90``.

Energy (DFTB+ "Solvation energy")::

    E = 1/2 sum_AB q_A B_AB q_B + sum_A SASA_A (gamma_A + h_A q_A^2) + shift

with q = dq - c (dq: Mulliken charge deviations in electrons as everywhere in pydftb;
c: the CM5 correction, zero without CM5), B_AB = k_eps f(R_AB, a_A, a_B) (diagonal
k_eps / a_A, plus k_eps alpha beta / A_det for ALPB), k_eps = (1/eps - 1) / (1 + alpha
beta), a the Born radii (GBOBC-II), SASA the surface areas, gamma = 4 pi 1e-5 times the
surface tension input, h the hydrogen-bond strengths over (probe + vdW radius)^2.  The
potential on the charges is B q + 2 SASA h q, like the third-order shift (input charges
during the SCC, output charges for energies and forces).

Units: lengths in Bohr, energies in Hartree, temperatures in K, molecular masses in amu,
densities in kg/L; ``SASA.surface_tension`` and the descreening values are DFTB+'s
numbers.  Per-species parameters are dicts keyed by type label or by element symbol.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
from ase.data import chemical_symbols

from . import solvation_data as data
from .constants import BOHR, KB_HARTREE, KCAL_MOL

AA_BOHR = 1.0 / BOHR                       # DFTB+ AA__Bohr
AMBIENT_TEMPERATURE = 298.15               # K
IDEAL_GAS_MOL_VOLUME = 24.79               # L/mol at 298.15 K, 1 bar (DFTB+)
ZETA_P16 = 1.028
GRID_SIZES = (6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590,
              770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802,
              5294, 5810)
_LEBEDEV_ORDER = dict(zip(GRID_SIZES, (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                                       35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107,
                                       113, 119, 125, 131)))
RADII_TABLES = {"d3": data.VDW_RADII_D3, "cosmo": data.VDW_RADII_COSMO,
                "bondi": data.VDW_RADII_BONDI}


# ---- solvents ------------------------------------------------------------------------
@dataclass(frozen=True)
class Solvent:
    """Dielectric constant, molecular mass (amu) and density (kg/L); the last two only
    matter for the "reference" state."""
    epsilon: float
    molecular_mass: float = 0.0
    density: float = 0.0


# DFTB+ solventdata.F90 (the names it accepts; "methanol" added)
SOLVENTS = {
    "acetone": Solvent(20.7, 58.08, 0.79), "acetonitrile": Solvent(37.5, 41.05, 0.786),
    "toluene": Solvent(7.0, 92.14, 0.867), "benzene": Solvent(7.0, 78.11, 0.867),
    "chloroform": Solvent(7.0, 119.38, 1.49), "dichloromethane": Solvent(7.0, 84.93, 1.33),
    "cs2": Solvent(2.6, 76.13, 1.266), "dmf": Solvent(37.0, 73.1, 0.95),
    "dmso": Solvent(47.2, 78.13, 1.1), "ether": Solvent(7.3, 74.12, 0.713),
    "water": Solvent(78.5, 18.0, 0.998), "methanole": Solvent(33.6, 32.04, 0.792),
    "nhexane": Solvent(1.88, 86.18, 0.66), "thf": Solvent(10.0, 72.1061, 0.883),
}
for _alias, _name in (("c6h6", "benzene"), ("trichloromethane", "chloroform"),
                      ("chcl3", "chloroform"), ("ch2cl2", "dichloromethane"), ("h2o", "water"),
                      ("methanol", "methanole"), ("n-hexane", "nhexane")):
    SOLVENTS[_alias] = SOLVENTS[_name]


def solvent_of(spec) -> Solvent:
    if isinstance(spec, Solvent):
        return spec
    if isinstance(spec, str):
        try:
            return SOLVENTS[spec.lower()]
        except KeyError:
            raise ValueError(f"unknown solvent {spec!r}; known: {sorted(SOLVENTS)} or "
                             "Solvent(epsilon, molecular_mass, density)") from None
    raise ValueError(f"solvent must be a name or a Solvent, not {spec!r}")


# ---- model inputs --------------------------------------------------------------------
@dataclass
class CM5:
    """Charge model 5 correction of the charges seen by the Born term (DFTB+ ``CM5``).

    ``double_count=True`` reproduces DFTB+ 25.1, whose Born model passes CM5 a symmetric
    neighbour list: every atom pair is counted twice, the correction doubled (its CM5
    analysis output, on the half list, is not).  The default follows the CM5 model."""
    alpha: float = 2.474 * BOHR              # 1/Bohr (2.474 1/Angstrom)
    radii: dict | None = None                # Bohr, overrides of the atomic radii
    cutoff: float = 30.0                     # Bohr
    double_count: bool = False


@dataclass
class SASA:
    """Solvent accessible surface area model (DFTB+ ``SASA``)."""
    probe_radius: float                      # Bohr
    surface_tension: dict                    # {label or element: DFTB+ value}
    radii: dict | None = None                # Bohr, overrides of the van der Waals radii
    radii_table: str = "d3"                  # "d3", "cosmo" or "bondi"
    smoothing: float = 0.3 * AA_BOHR         # Bohr
    tolerance: float = 1e-6
    angular_grid: int = 230
    offset: float = 2.0 * AA_BOHR            # Bohr

    def field_scaling(self) -> float:
        return 1.0                           # no electrostatics: DFTB+ turns rescaling off

    def setup(self, labels, numbers, pos) -> "SolvationTerms":
        return SolvationTerms(None, self, labels, numbers, pos)


@dataclass
class GeneralizedBorn:
    """Generalised Born model (DFTB+ ``GeneralisedBorn``); with ``sasa`` it is GBSA."""
    solvent: "Solvent | str"
    free_energy_shift: float                 # Hartree
    born_scale: float
    born_offset: float                       # Bohr
    descreening: "dict | str"                # {label or element: s} or "unity"
    radii: dict | None = None                # Bohr, overrides of the van der Waals radii
    radii_table: str = "d3"
    kernel: str = "still"                    # "still" or "p16"
    obc: tuple = (1.00, 0.80, 4.85)          # GBOBC-II
    cutoff: float = 35.0 * AA_BOHR           # Bohr, for the Born radii
    alpb: bool = False
    alpha: float = 0.571412                  # ALPB
    temperature: float = AMBIENT_TEMPERATURE  # K, for the reference states
    state: str = "gsolv"                     # "gsolv", "mol1bar" or "reference"
    cm5: CM5 | None = None
    sasa: SASA | None = None
    hbond: dict | None = None                # {label or element: strength (Hartree)}

    def __post_init__(self):
        self.solvent = solvent_of(self.solvent)
        if str(self.kernel).lower() not in ("still", "p16"):
            raise ValueError(f"kernel must be 'still' or 'p16', not {self.kernel!r}")
        self.kernel = str(self.kernel).lower()
        if str(self.state).lower() not in ("gsolv", "mol1bar", "reference"):
            raise ValueError(f"state must be 'gsolv', 'mol1bar' or 'reference', not {self.state!r}")
        self.state = str(self.state).lower()
        if self.hbond is not None and self.sasa is None:
            raise ValueError("the hydrogen-bond correction needs a SASA model (as in DFTB+)")
        if self.radii_table not in RADII_TABLES:
            raise ValueError(f"radii_table must be one of {sorted(RADII_TABLES)}")

    # DFTB+: k_eps and the ALPB beta
    @property
    def alpbet(self) -> float:
        return self.alpha / self.solvent.epsilon if self.alpb else 0.0

    @property
    def keps(self) -> float:
        return (1.0 / self.solvent.epsilon - 1.0) / (1.0 + self.alpbet)

    def field_scaling(self) -> float:
        """3 eps / (2 eps + 1), DFTB+'s RescaleSolvatedFields (3/2 for eps = inf)."""
        eps = self.solvent.epsilon
        return 1.5 if math.isinf(eps) else 3.0 * eps / (2.0 * eps + 1.0)

    def total_shift(self) -> float:
        """FreeEnergyShift plus the reference-state term (DFTB+ readReferenceState)."""
        kt = self.temperature * KB_HARTREE
        out = self.free_energy_shift
        if self.state in ("mol1bar", "reference"):
            out += kt * math.log(IDEAL_GAS_MOL_VOLUME * self.temperature / AMBIENT_TEMPERATURE)
        if self.state == "reference":
            s = self.solvent
            if s.molecular_mass <= 0 or s.density <= 0:
                raise ValueError("the 'reference' state needs the solvent's molecular mass "
                                 "and density")
            out += kt * math.log(1000.0 * s.density / s.molecular_mass)
        return out

    def setup(self, labels, numbers, pos) -> "SolvationTerms":
        return SolvationTerms(self, self.sasa, labels, numbers, pos)

    @classmethod
    def from_param_file(cls, path, **overrides) -> "GeneralizedBorn":
        """A GBSA model from an xtb-format parameter file (DFTB+ ``ParamFile``): 8 header
        lines (epsilon, molecular mass in amu, density in kg/L, Born scale, probe radius in
        Angstrom, free energy shift in kcal/mol, Born offset in 0.1 Angstrom, unused) and
        94 lines of (surface tension, descreening, hydrogen-bond parameter) per element.
        ``overrides`` replace any field (e.g. ``alpb=True``, ``state="mol1bar"``)."""
        with open(path) as f:
            rows = [line.split() for line in f if line.strip()]
        head = [float(r[0]) for r in rows[:8]]
        table = [[float(x) for x in r[:3]] for r in rows[8:8 + 94]]
        if len(head) < 8 or len(table) < 94:
            raise ValueError(f"{path}: need 8 header lines and 94 element lines")
        elems = chemical_symbols[1:95]
        tension = {e: row[0] for e, row in zip(elems, table)}
        descreening = {e: row[1] for e, row in zip(elems, table)}
        hbond = {e: -row[2] ** 2 * KCAL_MOL for e, row in zip(elems, table)}
        if all(v > -1e-14 for v in hbond.values()):
            hbond = None
        sasa = SASA(probe_radius=head[4] * AA_BOHR, surface_tension=tension)
        model = cls(solvent=Solvent(head[0], head[1], head[2]),
                    free_energy_shift=head[5] * KCAL_MOL, born_scale=head[3],
                    born_offset=head[6] * AA_BOHR * 0.1, descreening=descreening, sasa=sasa,
                    hbond=hbond)
        return replace(model, **overrides) if overrides else model


# ---- per-geometry terms --------------------------------------------------------------
def _per_atom(param, labels, numbers, what):
    """Values of a {label or element: value} dict for the atoms."""
    out = np.empty(len(labels))
    for k, (lab, z) in enumerate(zip(labels, numbers)):
        if lab in param:
            out[k] = param[lab]
        elif chemical_symbols[z] in param:
            out[k] = param[chemical_symbols[z]]
        else:
            raise ValueError(f"no {what} for {lab}")
    return out


def _radii(table, overrides, labels, numbers, what="van der Waals radius"):
    """Radii (Bohr) from a table of Angstrom values, overridden per label or element."""
    out = np.empty(len(labels))
    for k, (lab, z) in enumerate(zip(labels, numbers)):
        sym = chemical_symbols[z]
        if overrides and lab in overrides:
            out[k] = overrides[lab]
        elif overrides and sym in overrides:
            out[k] = overrides[sym]
        else:
            r = table[z - 1] if 0 < z <= len(table) else None
            if r is None:
                raise ValueError(f"no tabulated {what} for {sym}: give it explicitly")
            out[k] = r * AA_BOHR
    return out


def lebedev_grid(npoints: int):
    """Lebedev points (n, 3) and weights summing to 1 (DFTB+ ``getAngGrid``)."""
    if npoints not in _LEBEDEV_ORDER:
        raise ValueError(f"no Lebedev grid with {npoints} points; available: {GRID_SIZES}")
    try:
        from scipy.integrate import lebedev_rule
    except ImportError:                       # SciPy < 1.15
        raise ImportError("the SASA model needs scipy >= 1.15 (scipy.integrate.lebedev_rule)") from None
    x, w = lebedev_rule(_LEBEDEV_ORDER[npoints])
    if x.shape[1] != npoints:
        raise RuntimeError(f"scipy's Lebedev rule of order {_LEBEDEV_ORDER[npoints]} has "
                           f"{x.shape[1]} points, expected {npoints}")
    return x.T.copy(), w / (4.0 * np.pi)


@dataclass
class SolvationTerms:
    """Geometry-dependent parts of a solvation model for one molecule."""
    gb: GeneralizedBorn | None
    sasa_model: SASA | None
    labels: list
    numbers: np.ndarray
    pos: np.ndarray
    bmat: np.ndarray | None = field(init=False, default=None)    # Born matrix
    born: np.ndarray | None = field(init=False, default=None)    # Born radii
    dborn: np.ndarray | None = field(init=False, default=None)   # (i, j, 3): d a_i / d r_j
    sasa: np.ndarray | None = field(init=False, default=None)    # SASA_A
    dsasa: np.ndarray | None = field(init=False, default=None)   # (i, j, 3): d SASA_i / d r_j
    gamma: np.ndarray | None = field(init=False, default=None)   # surface tension per atom
    hb: np.ndarray | None = field(init=False, default=None)      # H-bond strength per atom
    cm5: np.ndarray | None = field(init=False, default=None)     # CM5 correction c_A
    dcm5: np.ndarray | None = field(init=False, default=None)    # (i, j, 3): d c_i / d r_j
    adet: float = field(init=False, default=0.0)                 # ALPB shape descriptor

    def __post_init__(self):
        self.pos = np.asarray(self.pos, float)
        self.numbers = np.asarray(self.numbers, int)
        if self.sasa_model is not None:
            self._setup_sasa(self.sasa_model)
        gb = self.gb
        if gb is None:
            return
        vdw = _radii(RADII_TABLES[gb.radii_table], gb.radii, self.labels, self.numbers)
        if isinstance(gb.descreening, str):
            if gb.descreening.lower() != "unity":
                raise ValueError("descreening must be 'unity' or {label or element: value}")
            desc = np.ones(len(self.labels))
        else:
            desc = _per_atom(gb.descreening, self.labels, self.numbers, "descreening value")
        self.born, self.dborn = born_radii(self.pos, vdw, vdw * desc, gb.born_offset,
                                           gb.born_scale, gb.obc, gb.cutoff)
        self.bmat = born_matrix(self.pos, self.born, gb.keps, gb.kernel)
        if gb.alpb:
            self.adet = _adet(self.pos, vdw)
            self.bmat = self.bmat + gb.keps * gb.alpbet / self.adet
        self._vdw = vdw
        if gb.hbond is not None:
            hb = _per_atom(gb.hbond, self.labels, self.numbers, "hydrogen-bond strength")
            self.hb = hb / self._sasa_radius ** 2 if np.any(hb != 0.0) else None
        if gb.cm5 is not None:
            self.cm5, self.dcm5 = cm5_correction(self.pos, self.labels, self.numbers, gb.cm5)

    def _setup_sasa(self, m: SASA):
        vdw = _radii(RADII_TABLES[m.radii_table], m.radii, self.labels, self.numbers)
        self._sasa_radius = m.probe_radius + vdw
        tension = _per_atom(m.surface_tension, self.labels, self.numbers, "surface tension")
        self.gamma = tension * 4.0e-5 * np.pi
        grid, weights = lebedev_grid(m.angular_grid)
        self.sasa, self.dsasa = surface_areas(self.pos, self._sasa_radius, m.smoothing,
                                              m.tolerance, m.offset, grid, weights)

    # ---- charges, potential, energy, gradient ------------------------------------------
    def _charges(self, dq_atom):
        return dq_atom if self.cm5 is None else dq_atom - self.cm5

    def shift(self, dq_atom) -> np.ndarray:
        """dE/d(dq_A): B q (+ 2 SASA h q)."""
        if self.gb is None:
            return np.zeros(len(self.pos))
        q = self._charges(dq_atom)
        v = self.bmat @ q
        if self.hb is not None:
            v = v + 2.0 * self.sasa * self.hb * q
        return v

    def energy(self, dq_atom) -> float:
        e = 0.0
        if self.gamma is not None:
            e += float(self.sasa @ self.gamma)
        if self.gb is not None:
            q = self._charges(dq_atom)
            e += 0.5 * float(q @ (self.bmat @ q))
            if self.hb is not None:
                e += float(self.sasa @ (self.hb * q * q))
            e += self.gb.total_shift()
        return e

    def gradient(self, dq_atom) -> np.ndarray:
        """dE/dr_A at fixed charges (n_atom, 3)."""
        n = len(self.pos)
        grad = np.zeros((n, 3))
        if self.gamma is not None:
            w = self.gamma.copy()
            if self.gb is not None and self.hb is not None:
                q = self._charges(dq_atom)
                w = w + self.hb * q * q
            grad += np.einsum("i,ijx->jx", w, self.dsasa)
        if self.gb is None:
            return grad
        gb = self.gb
        q = self._charges(dq_atom)
        g_pair, de_dborn = born_energy_gradient(self.pos, q, self.born, gb.keps, gb.kernel)
        grad += g_pair + np.einsum("i,ijx->jx", de_dborn, self.dborn)
        if gb.alpb:
            grad += _adet_gradient(self.pos, self._vdw, gb.keps * gb.alpbet, q)
        if self.cm5 is not None:
            # q = dq - c: dE/dc_i = -dE/dq_i (the whole potential, Born and H-bond)
            grad -= np.einsum("i,ijx->jx", self.shift(dq_atom), self.dcm5)
        return grad


# ---- Born radii (GBOBC-II, DFTB+ getPsi / getBornRadii) -----------------------------------
def _g_apart(d, rho):
    """Volume integral of a sphere of radius rho at distance d, no overlap; and
    (dg/dd)/d."""
    r1 = 1.0 / d
    ap, am = d + rho, d - rho
    ab = ap * am
    rhab = rho / ab
    lnab = 0.5 * np.log(am / ap) * r1
    return rhab + lnab, -2.0 * rhab / ab + (rhab - lnab) * r1 * r1


def _g_overlap(d, rho, rvdw):
    """The same when the sphere overlaps the van der Waals sphere rvdw of the atom."""
    r1 = 1.0 / d
    r12 = 0.5 * r1
    r24 = r12 * r12
    ap, am = d + rho, d - rho
    rh1 = 1.0 / rvdw
    rhr1 = 1.0 / ap
    aprh1 = ap * rh1
    lnab = np.log(aprh1)
    g = rh1 - rhr1 + r12 * (0.5 * am * (rhr1 - rh1 * aprh1) - lnab)
    dg = (rhr1 * rhr1 * (1.0 - 0.25 * am * r1 * (1.0 + aprh1 * aprh1))
          + rho * r24 * (rhr1 - rh1 * aprh1) + r12 * (r1 * lnab - rhr1))
    return g, dg * r1


def born_radii(pos, vdw, rho, offset, scale, obc, cutoff):
    """Born radii a (n,) and d a_i / d r_j (n, n, 3)."""
    n = len(pos)
    psi = np.zeros(n)
    dpsi = np.zeros((n, n, 3))
    i, j = np.triu_indices(n, 1)
    i, j = j, i                                      # pairs i > j, as DFTB+
    vec = pos[i] - pos[j]
    d = np.sqrt((vec * vec).sum(1))
    keep = d < cutoff
    i, j, vec, d = i[keep], j[keep], vec[keep], d[keep]
    rhoi, rhoj, rvi, rvj = rho[i], rho[j], vdw[i], vdw[j]
    ov_ij = d < rvi + rhoj
    ov_ji = d < rhoi + rvj
    # term on psi_i (from atom j's descreened sphere) and on psi_j (from atom i's)
    gi = np.zeros(len(d))
    dgi = np.zeros(len(d))
    gj = np.zeros(len(d))
    dgj = np.zeros(len(d))
    a_i = ~ov_ij                                     # psi_i: j's sphere does not overlap i
    g, dg = _g_apart(d[a_i], rhoj[a_i])
    gi[a_i], dgi[a_i] = g, dg
    o_i = ov_ij & (d + rhoj > rvi)
    g, dg = _g_overlap(d[o_i], rhoj[o_i], rvi[o_i])
    gi[o_i], dgi[o_i] = g, dg
    a_j = ~ov_ji
    g, dg = _g_apart(d[a_j], rhoi[a_j])
    gj[a_j], dgj[a_j] = g, dg
    o_j = ov_ji & (d + rhoi > rvj)
    g, dg = _g_overlap(d[o_j], rhoi[o_j], rvj[o_j])
    gj[o_j], dgj[o_j] = g, dg
    np.add.at(psi, i, gi)
    np.add.at(psi, j, gj)
    gri = dgi[:, None] * vec                         # d psi_i / d r_i (pair part)
    grj = dgj[:, None] * vec                         # -d psi_j / d r_j
    np.add.at(dpsi, (i, i), gri)
    np.add.at(dpsi, (i, j), -gri)
    np.add.at(dpsi, (j, j), -grj)
    np.add.at(dpsi, (j, i), grj)
    svdw = vdw - offset
    s1, v1, s2 = 1.0 / svdw, 1.0 / vdw, 0.5 * svdw
    br = psi * s2
    arg2 = br * (obc[2] * br - obc[1])
    arg = br * (obc[0] + arg2)
    arg2 = 2.0 * arg2 + obc[0] + obc[2] * br * br
    th, ch = np.tanh(arg), np.cosh(arg)
    radii = scale / (s1 - v1 * th)
    dfac = ch * (s1 - v1 * th)
    dfac = scale * (s2 * v1 * arg2 / (dfac * dfac))
    return radii, dpsi * dfac[:, None, None]


# ---- Born matrix and its gradient ----------------------------------------------------------
def _pairs(pos):
    i, j = np.triu_indices(len(pos), 1)
    i, j = j, i
    vec = pos[i] - pos[j]
    return i, j, vec, (vec * vec).sum(1)


def born_matrix(pos, born, keps, kernel):
    n = len(pos)
    b = np.zeros((n, n))
    i, j, _, d2 = _pairs(pos)
    if kernel == "still":
        aa = born[i] * born[j]
        f = keps * np.sqrt(1.0 / (d2 + aa * np.exp(-0.25 * d2 / aa)))
    else:
        r1 = np.sqrt(d2)
        ab = np.sqrt(born[i] * born[j])
        arg = (ab / (ab + ZETA_P16 / 16.0 * r1)) ** 16
        f = keps / (r1 + ab * arg)
    b[i, j] = f
    b[j, i] = f
    b[np.diag_indices(n)] = keps / born
    return b


def born_energy_gradient(pos, q, born, keps, kernel):
    """Pair part of dE_GB/dr (n, 3) and dE_GB/da (n,) at fixed charges and radii."""
    n = len(pos)
    i, j, vec, d2 = _pairs(pos)
    qq = q[i] * q[j]
    if kernel == "still":
        aa = born[i] * born[j]
        dd = 0.25 * d2 / aa
        expd = np.exp(-dd)
        dfgb2 = 1.0 / (d2 + aa * expd)
        dfgb3 = dfgb2 * np.sqrt(dfgb2) * keps
        g = ((1.0 - 0.25 * expd) * dfgb3 * qq)[:, None] * vec
        bp = -0.5 * expd * (1.0 + dd) * dfgb3 * qq
        dbi, dbj = born[j] * bp, born[i] * bp
    else:
        r1 = np.sqrt(d2)
        ab = np.sqrt(born[i] * born[j])
        arg1 = ab / (ab + ZETA_P16 / 16.0 * r1)
        arg16 = arg1 ** 16
        dfgb = 1.0 / (r1 + ab * arg16)
        dfgb2 = dfgb * dfgb
        g = ((1.0 - ZETA_P16 * arg1 * arg16) * dfgb2 * keps / r1 * qq)[:, None] * vec
        bp = -0.5 * (r1 * ZETA_P16 / ab * arg1 + 1.0) / ab * arg16 * dfgb2 * keps * qq
        dbi, dbj = born[j] * bp, born[i] * bp
    grad = np.zeros((n, 3))
    np.add.at(grad, i, -g)
    np.add.at(grad, j, g)
    de_db = np.zeros(n)
    np.add.at(de_db, i, dbi)
    np.add.at(de_db, j, dbj)
    de_db += -0.5 * keps * q * q / (born * born)
    return grad, de_db


# ---- ALPB shape descriptor (DFTB+ getADet / getADetDeriv) -----------------------------------
def _inertia(pos, rad):
    r3 = rad ** 3
    center = (pos * r3[:, None]).sum(0) / r3.sum()
    vec = pos - center
    r2 = (vec * vec).sum(1)
    inertia = (np.einsum("a,a->", r3, r2 + 0.4 * rad * rad) * np.eye(3)
               - np.einsum("a,ax,ay->xy", r3, vec, vec))
    return inertia, vec, r3


def _adet(pos, rad) -> float:
    inertia, _, r3 = _inertia(pos, rad)
    return math.sqrt(np.linalg.det(inertia) ** (1.0 / 3.0) / (2.0 * r3.sum()) * 5.0)


def _adet_gradient(pos, rad, keps_ab, q):
    inertia, vec, r3 = _inertia(pos, rad)
    tot = r3.sum()
    adet = math.sqrt(np.linalg.det(inertia) ** (1.0 / 3.0) / (2.0 * tot) * 5.0)
    i_ = inertia
    a = np.array([
        [i_[0, 0] * (i_[1, 1] + i_[2, 2]) - i_[0, 1] ** 2 - i_[0, 2] ** 2,
         i_[0, 1] * i_[2, 2] - i_[0, 2] * i_[1, 2],
         i_[0, 2] * i_[1, 1] - i_[0, 1] * i_[2, 1]],
        [i_[0, 1] * i_[2, 2] - i_[0, 2] * i_[1, 2],
         i_[1, 1] * (i_[0, 0] + i_[2, 2]) - i_[0, 1] ** 2 - i_[1, 2] ** 2,
         i_[0, 0] * i_[1, 2] - i_[0, 1] * i_[0, 2]],
        [i_[0, 2] * i_[1, 1] - i_[0, 1] * i_[2, 1],
         i_[0, 0] * i_[1, 2] - i_[0, 1] * i_[0, 2],
         i_[2, 2] * (i_[0, 0] + i_[1, 1]) - i_[0, 2] ** 2 - i_[1, 2] ** 2]])
    a = a * (250.0 / (48.0 * tot ** 3 * adet ** 5)) * (-0.5 * keps_ab * q.sum() ** 2 / adet ** 2)
    return r3[:, None] * (vec @ a.T)


# ---- surface areas (DFTB+ getSASA) --------------------------------------------------------
def surface_areas(pos, radius, w, tolerance, offset, grid, weights):
    """SASA (n,) and d SASA_i / d r_j (n, n, 3) with DFTB+'s smoothed step."""
    n = len(pos)
    ah1, ah2, ah3 = 0.5, 3.0 / (4.0 * w), -1.0 / (4.0 * w ** 3)
    rm, rp = radius - w, radius + w
    thr1, thr2 = rm ** 2, rp ** 2
    wrp = ((0.25 / w + 3.0 * ah3 * (0.2 * rp * rp - 0.5 * rp * radius + radius * radius / 3.0)) * rp ** 3
           - (0.25 / w + 3.0 * ah3 * (0.2 * rm * rm - 0.5 * rm * radius + radius * radius / 3.0)) * rm ** 3)
    cutoff = 2.0 * (radius.max() + w) + offset
    sasa = np.zeros(n)
    dsasa = np.zeros((n, n, 3))
    diff = pos[None, :, :] - pos[:, None, :]
    dist = np.sqrt((diff * diff).sum(-1))
    for a in range(n):
        nb = np.nonzero((dist[a] < cutoff) & (np.arange(n) != a))[0]
        points = pos[a] + radius[a] * grid                      # (np, 3)
        vec = points[:, None, :] - pos[nb][None, :, :]          # (np, nn, 3)
        d2 = (vec * vec).sum(-1)
        inside = d2 < thr1[nb][None, :]
        smooth = (d2 < thr2[nb][None, :]) & ~inside
        dd = np.sqrt(np.where(smooth, d2, 1.0))
        uj = dd - radius[nb][None, :]
        ah3uj2 = ah3 * uj * uj
        f = np.where(smooth, ah1 + (ah2 + ah3uj2) * uj, 1.0)
        f = np.where(inside, 0.0, f)
        sasap = f.prod(axis=1)
        keep = sasap > tolerance
        wsa = np.where(keep, weights * wrp[a] * sasap, 0.0)
        sasa[a] = wsa.sum()
        df = np.where(smooth, (ah2 + 3.0 * ah3uj2) / (np.where(smooth, f, 1.0) * dd), 0.0)
        # ... (d f / d dist) / (f dist) of the smooth region
        dgr = (wsa[:, None] * df)[:, :, None] * vec                 # (np, nn, 3)
        per_nb = dgr.sum(0)                                        # (nn, 3)
        dsasa[a, a] += per_nb.sum(0)
        dsasa[a, nb] -= per_nb
    return sasa, dsasa


# ---- CM5 (DFTB+ cm5.F90) ------------------------------------------------------------------
_CM5_PAIRS = {(1, 6): 0.0502, (1, 7): 0.1747, (1, 8): 0.1671, (6, 7): 0.0556, (6, 8): 0.0234,
              (7, 8): -0.0346}


def cm5_pair_parameter(z1: int, z2: int) -> float:
    if not (0 < z1 <= len(data.CM5_PARAMETERS) and 0 < z2 <= len(data.CM5_PARAMETERS)):
        return 0.0
    if (z1, z2) in _CM5_PAIRS:
        return _CM5_PAIRS[(z1, z2)]
    if (z2, z1) in _CM5_PAIRS:
        return -_CM5_PAIRS[(z2, z1)]
    return data.CM5_PARAMETERS[z1 - 1] - data.CM5_PARAMETERS[z2 - 1]


def cm5_correction(pos, labels, numbers, cm5: CM5):
    """c_i = sum_j D_ij exp(-alpha (R_ij - R_i - R_j)) over atoms of other types, and
    d c_i / d r_j (n, n, 3)."""
    n = len(pos)
    rad = _radii(data.ATOMIC_RADII, cm5.radii, labels, numbers, "atomic radius")
    i, j, vec, d2 = _pairs(pos)                        # i > j
    d = np.sqrt(d2)
    lab = np.asarray(labels, dtype=object)
    keep = (lab[i] != lab[j]) & (d < cm5.cutoff)       # DFTB+: other species, within cut-off
    i, j, vec, d = i[keep], j[keep], vec[keep], d[keep]
    z = np.asarray(numbers)
    pab = np.array([cm5_pair_parameter(a, b) for a, b in zip(z[i], z[j])])
    pba = np.array([cm5_pair_parameter(b, a) for a, b in zip(z[i], z[j])])
    e = np.exp(-cm5.alpha * (d - rad[i] - rad[j]))
    c = np.zeros(n)
    np.add.at(c, i, e * pab)
    np.add.at(c, j, e * pba)
    dg = (e * cm5.alpha / d)[:, None] * vec            # d e / d r_j = -d e / d r_i
    dc = np.zeros((n, n, 3))
    np.add.at(dc, (i, i), -dg * pab[:, None])
    np.add.at(dc, (j, j), dg * pba[:, None])
    np.add.at(dc, (j, i), -dg * pba[:, None])
    np.add.at(dc, (i, j), dg * pab[:, None])
    if cm5.double_count:
        return 2.0 * c, 2.0 * dc
    return c, dc


def make_solvation(spec):
    """A solvation model from a GeneralizedBorn / SASA object, or None."""
    if spec is None or isinstance(spec, (GeneralizedBorn, SASA)):
        return spec
    raise ValueError("solvation must be a pydftb.solvation.GeneralizedBorn or SASA object")


def check_rescaling(model, field, rescale: bool):
    """DFTB+ warns and turns the field rescaling off for a model without electrostatics."""
    if model is not None and field is not None and rescale and isinstance(model, SASA):
        warnings.warn("field rescaling requested for a solvent model with no electrostatic "
                      "effects (SASA): turning scaling off, as DFTB+", RuntimeWarning,
                      stacklevel=3)
