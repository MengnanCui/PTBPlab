"""Validation cases shared by the reference generator and the offline tests.

A case = structure + settings + parameter set.  Builders are only used when
(re)generating references; the frozen JSON in ``pydftb/tests/data/ref`` stores
the exact coordinates, so tests never depend on ASE builder details.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule

from ..blockpot import ONSITE_PBE
from ..constants import BOHR, HARTREE, KCAL_MOL
from ..external import fold_into_cell
from ..kpoints import monkhorst_pack, reduce_time_reversal

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "data")
SKF_SETS = {
    "ptbp_hcocu": os.path.join(DATA, "skf", "ptbp_hcocu"),
    "orbit": os.path.join(DATA, "skf", "orbit"),
    "orbit_poly": os.path.join(DATA, "skf", "orbit_poly"),   # make_poly_set.py
    "hotcent_gdo": os.path.join(DATA, "skf", "hotcent_gdo"), # make_f_set.py (f orbitals)
    "cu_types": os.path.join(DATA, "skf", "cu_types"),       # make_cu_types_set.py (M5a)
}
REF_DIR = os.path.join(DATA, "ref")
MAXL = {"H": 0, "C": 1, "O": 1, "Cu": 2, "Li": 0, "Gd": 3}
# DFTB+'s finite-difference H0/S derivative error at this water geometry is 1.2e-6 Ha/Bohr:
# the H-H distance lies 4e-6 Bohr from SK grid node 153, inside DFTB+'s step of 1.22e-4
# (pydftb's exact forces agree with its own energy finite differences to 2e-9 at h = 1e-6)
TOL_DFTBP_FD = {"energy": 1e-6, "forces": 2e-6, "charges": 1e-6}
# DFTB3: 3ob Hubbard derivatives (Ha/e) and its standard settings; Cu values are for testing only
UD_3OB = {"H": -0.1857, "C": -0.1492, "O": -0.1575}
DFTB3_3OB = {"third_order": "full", "hubbard_derivs": UD_3OB, "h_damping": 4.0}
# spin constants W (Ha): typical PBE values for H and O; Cu values are for testing only
W_SPIN = {"H": -0.072, "O": [[-0.035, -0.030], [-0.030, -0.028]],
          "Cu": [[-0.016, -0.012, -0.004], [-0.012, -0.040, -0.002], [-0.004, -0.002, -0.017]],
          # free-atom values of the hotcent_gdo Gd (make_f_set.py), shells s p d f
          "Gd": [[-0.010552, -0.008687, -0.006723, -0.000704],
                 [-0.008687, -0.013022, -0.004158, -0.000290],
                 [-0.006723, -0.004158, -0.008679, -0.001912],
                 [-0.000704, -0.000290, -0.001912, -0.013626]]}
# f orbitals (milestone M2g): the hotcent Gd/O set has no repulsive (zero polynomial), and the
# half-filled 4f shell needs a smeared Fermi level; Ud per shell for testing only
GDO = {"skf_set": "hotcent_gdo", "filling": ("fermi", 1000.0), "repulsive": "polynomial"}
UD_GDO = {"Gd": [-0.05, -0.03, -0.08, -0.20], "O": [-0.1575, -0.12]}
# dispersion (milestone M2d).  D3(BJ) and D4 parameters of DFTB3/3ob from the parameter tables of
# s-dftd3 ("dftb3") and dftd4 ("dftb_3ob", with and without the three-body term)
D3BJ_3OB = {"model": "dftd3", "damping": "beckejohnson", "a1": 0.5719, "a2": 3.6017,
            "s6": 1.0, "s8": 0.5883}
D4_3OB = {"model": "dftd4", "s6": 1.0, "s8": 0.6635015, "s9": 1.0, "a1": 0.5523240,
          "a2": 4.3537076}
# periodic D4 against DFTB+: two-body only and no charge dependence, because DFTB+'s periodic
# EEQ charges and three-body sum are wrong (docs/design/M2d-dispersion.html)
D4_3OB_2BODY_Q0 = {"model": "dftd4", "s6": 1.0, "s8": 0.4727337, "s9": 0.0, "a1": 0.5467502,
                   "a2": 4.4955068, "charge_scale": 0.0}
# Slater-Kirkwood test values in atomic units (Bohr^3, Bohr, e), patterned on Elstner's table,
# with a different value for every neighbour count (0 ... >= 5) so that the hybridisation rule
# is tested (H radius 0.5 A instead of 0.4 A, so that every bond of the rattled test molecules
# counts); O with 4 neighbours has no polarisability (such atoms are skipped)
SK_HYBRID = {
    "H": {"covalent_radius": 0.5 / BOHR,
          "hybrid_polarisations": [2.60, 2.67, 2.70, 2.74, 2.78, 2.82, 6.60, 6.62, 6.64, 6.66, 6.68,
                                   6.70, 0.80]},
    "C": {"covalent_radius": 0.8 / BOHR,
          "hybrid_polarisations": [9.33, 9.30, 9.27, 7.21, 7.18, 7.15, 7.18, 7.20, 7.22, 7.24, 7.26,
                                   7.28, 2.50]},
    "O": {"covalent_radius": 0.8 / BOHR,
          "hybrid_polarisations": [3.78, 3.70, 3.62, 3.55, 0.00, 3.40, 7.18, 7.16, 7.14, 7.12, 7.10,
                                   7.08, 3.15]},
}
SK_ATOM = {"H": [2.67, 6.62, 0.80], "O": [3.62, 7.14, 3.15]}
# band structures and PDOS (milestone M2e): eigenvalues are compared to 1e-8 Ha, so DFTB+ runs
# with a tight Ewald sum (its default 1e-9 alone moves water-box eigenvalues by ~1e-7 Ha);
# region weights are printed with 6 decimals; dp_dos curves are compared relative to their maximum
TOL_BANDS = {"energy": 1e-6, "forces": 1e-6, "charges": 1e-6, "eigenvalues": 1e-8,
             "occupations": 1e-8, "fermi": 1e-8, "regions": 1e-6, "dp_dos": 1e-9}
# special points (fractional, ASE's convention for these cells)
FCC = {"G": (0, 0, 0), "X": (0.5, 0, 0.5), "W": (0.5, 0.25, 0.75), "L": (0.5, 0.5, 0.5),
       "K": (0.375, 0.375, 0.75)}
BCC = {"G": (0, 0, 0), "H": (0.5, -0.5, 0.5), "N": (0, 0, 0.5), "P": (0.25, 0.25, 0.25)}
ORC = {"G": (0, 0, 0), "X": (0.5, 0, 0), "S": (0.5, 0.5, 0), "Z": (0, 0, 0.5), "U": (0.5, 0, 0.5)}


def path(points, names, steps):
    """Klines lines [(n, k), ...]: the first special point once, then ``steps[i]`` points
    up to ``names[i + 1]`` (n = 0: jump there without a point)."""
    return [(1, points[names[0]])] + [(n, points[name]) for n, name in zip(steps, names[1:])]


def sk_per_atom(labels, skip=()):
    """Slater-Kirkwood ``polar_radius_charge`` for these atoms; atoms in ``skip`` get alpha = 0
    (and keep their electron number: with N = 0 too, DFTB+ divides 0 by 0 and returns NaN)."""
    rows = [[0.0] + SK_ATOM[lab][1:] if k in skip else SK_ATOM[lab]
            for k, lab in enumerate(labels)]
    return {"model": "slaterkirkwood", "polar_radius_charge": rows}


@dataclass
class Case:
    name: str
    build: Callable[[], Atoms]
    skf_set: str = "ptbp_hcocu"
    kgrid: tuple | None = None            # Monkhorst-Pack grid, time-reversal reduced
    filling: tuple = ("fermi", 0.0)       # ("fermi", T) | ("mp", T, order) | ("gaussian", T)
    scc: bool = True
    shell_resolved: bool = False
    repulsive: str = "skf"
    labels: Callable[[Atoms], list] | None = None   # type labels (default: elements)
    typed: dict | None = None             # label -> base label for write_typed_copies
    dftb3: dict | None = None             # third_order / hubbard_derivs / h_damping options
    spin: dict | None = None              # unpaired_electrons / relax_total_spin / spin_constants
    dispersion: dict | Callable[[list], dict] | None = None   # spec, or labels -> spec
    noncolinear: dict | None = None       # initial_spins (per atom) / spin_constants (M6a)
    spin_orbit: dict | None = None        # xi {label: per shell, Hartree} / dual (M6a)
    time_reversal: bool = True            # reduce the Monkhorst-Pack grid by k -> -k
    plus_u: dict | None = None            # DFTB+U {"groups": {label: [(shells, U-J)]},
                                          # "functional": "fll" | "psic"} (M6b)
    onsite: dict | None = None            # onsite corrections {label: {"uu", "ud"}} (M6b)
    efield: dict | None = None            # electric field {"strength" (V/A), "direction"} (M6c)
    point_charges: dict | None = None     # {"positions" (A), "charges" (e), "blur" (A)}
    site_potentials: dict | None = None   # {"gross"/"net": {atom (0-based): V (eV)}}
    solvation: dict | None = None         # solvation spec (atomic units), see solvation_model
    rescale_fields: bool = True           # DFTB+ RescaleSolvatedFields
    charge: float = 0.0                   # net charge (e)
    casida: dict | None = None            # Casida excitations (M6d): {"n", "symmetry" (None with
                                          # spin), "solver" (DFTB+), "state", "energy_window" (Ha),
                                          # "oscillator_window", "spin_constants" (no spin)}
    torch: bool = True                    # also checked on the torch backend (test_reference)
    klines: list | None = None            # band run on DFTB+ Klines [(n, (k1, k2, k3)), ...]
    regions: list | None = None           # ProjectStates [{"label", "atoms" (0-based), ...}]
    dp_dos: list | None = None            # dp_dos curves to freeze: [(file, [arguments])]
    ewald_tolerance: float | None = None  # DFTB+ EwaldTolerance (default 1e-9)
    scc_tol: float = 1e-10                # SCC tolerance of both codes
    tol: dict = field(default_factory=lambda: {"energy": 1e-6, "forces": 1e-6, "charges": 1e-6})
    note: str = ""
    # milestone M6f-1: the start of the SCC and its mixer (spin dicts may hold "initial_spins")
    mixer: str | dict | None = None       # pydftb's mixer option (DFTB+ Mixer)
    initial_charges: list | dict | None = None   # e per atom, or {atom (0-based): e}
    read_charges: dict | None = None      # two runs (DFTB+ ReadInitialCharges): {"text": bool,
                                          # "displace" (A), "seed"}: run 1 at the case geometry
                                          # displaced by that much writes the file, run 2 (the
                                          # one compared) at the case geometry reads it
    # milestone M6f-2: occupations and overrides of DFTB+'s Hamiltonian block
    fermi: dict | None = None             # {"fixed": E_f or (up, down) (Ha)} (FixedFermiLevel)
                                          # or {"independent_k": True} (IndependentKFilling)
    hubbard: dict | None = None           # CustomisedHubbards {label: U or [U per shell]} (Ha)
    occupations: dict | None = None       # CustomisedOccupations {atom (0-based): {shell: e}}
    truncate_sk: dict | None = None       # TruncateSKRange {"max_distance" (Bohr), "hard_cutoff"}
    stress_fd: bool | tuple = False       # reference stress from strain differences of DFTB+'s
                                          # force-related energy (its analytic stress of a
                                          # charged cell is not that derivative); a tuple of
                                          # (a, b): those components only (M6f-3), the others
                                          # DFTB+'s analytic stress
    # milestone M6f-3: corrections (HCorrection = H5, HalogenXCorr, DipoleCorrection)
    h5: bool | dict | None = None         # True or {"r_scaling", "w_scaling", "scaling": {el: k}}
    halogen: dict | None = None           # HalogenXCorr: {"compat": compare DFTB+'s forces with
                                          # pydftb's DFTB+ sign of the switching terms,
                                          # "forces_fd": reference forces from differences of
                                          # DFTB+'s energy}
    dipole_correction: bool | dict | None = None   # True or {"layer_position": z0 (A)}
    dftbplus: str = "25.1"                # "dev": references from the development build
                                          # ($DFTBPLUS_DEV), for what 25.1 does not have
    # milestone M5a: smooth type switching in the discrete limit, the fields of a TypeSwitch
    # ({"element", "types", "centre", "width", "cutoff"}): the labels are the fixed types the
    # switch gives (weights 0 or 1), DFTB+ runs them; pydftb also runs the switch, which must
    # give bitwise the explicit labels' results ("switch" deviation)
    type_switching: dict | None = None

    @property
    def corrections(self) -> bool:
        """M6f-3 settings: H5, the halogen correction, the dipole correction."""
        return (self.h5 is not None or self.halogen is not None
                or self.dipole_correction is not None)

    @property
    def overrides(self) -> bool:
        """M6f-2 settings: a fixed Fermi level, independent k-point filling, custom Hubbard U,
        custom reference occupations, a truncated SK table."""
        return (self.fermi is not None or self.hubbard is not None
                or self.occupations is not None or self.truncate_sk is not None)

    @property
    def track_scc(self) -> bool:
        """References keep the SCC error of every iteration and pydftb must take the same
        number of iterations (M6f cases)."""
        return self.scc_start or self.overrides or self.corrections

    @property
    def scc_start(self) -> bool:
        """M6f-1 settings: mixer, initial charges or spins, a charge file; references keep the
        SCC error of every iteration."""
        return (self.mixer is not None or self.initial_charges is not None
                or self.read_charges is not None
                or (self.spin or {}).get("initial_spins") is not None)

    def run1_positions(self, atoms: Atoms) -> np.ndarray:
        """Run 1's positions (A) of a ``read_charges`` case."""
        rc = self.read_charges
        rng = np.random.default_rng(rc.get("seed", 0))
        return atoms.positions + rng.normal(scale=rc.get("displace", 0.03), size=atoms.positions.shape)

    @property
    def pauli(self) -> bool:
        """Two-component (non-collinear spin or spin-orbit coupling)."""
        return self.noncolinear is not None or self.spin_orbit is not None

    @property
    def block(self) -> bool:
        """Block potentials: DFTB+U or onsite corrections (M6b)."""
        return self.plus_u is not None or self.onsite is not None

    @property
    def environment(self) -> bool:
        """Field, point charges, site potentials or implicit solvation (M6c)."""
        return (self.efield is not None or self.point_charges is not None
                or self.site_potentials is not None or self.solvation is not None)

    def kpoints(self):
        if self.kgrid is None:
            return None
        k = monkhorst_pack(self.kgrid)
        return reduce_time_reversal(*k) if self.time_reversal else k


def _water():
    a = molecule("H2O")
    a.rattle(0.05, seed=1)
    return a


def _meoh():
    a = molecule("CH3OH")
    a.rattle(0.05, seed=2)
    return a


def _cu4co():
    a = Atoms("Cu4CO", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.1, 0], [1.25, 0.7, 2.0],
                                   [1.25, 0.7, 3.9], [1.25, 0.75, 5.05]])
    a.rattle(0.03, seed=3)
    return a


def _cu_fcc():
    return bulk("Cu", "fcc", a=3.61)


def _cu8():
    a = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((2, 1, 1))
    a.rattle(0.05, seed=4)
    a.set_cell(a.cell.array @ np.array([[1, 0.03, 0], [0, 1, 0.02], [0.01, 0, 1]]), scale_atoms=True)
    return a


def _water_box():
    rng = np.random.default_rng(0)
    w = molecule("H2O")
    box = Atoms(cell=[6.2, 6.2, 6.2], pbc=True)
    for _ in range(8):
        m = w.copy()
        m.rotate(rng.uniform(0, 360), rng.normal(size=3))
        m.translate(rng.uniform(0, 6.2, 3))
        box += m
    return box


def _cuo_chain():
    a = Atoms("CuOCuO", positions=[[0, 0, 0], [1.9, 0.1, 0], [3.8, 0, 0.1], [5.7, 0, 0]],
              cell=[7.6, 4, 4], pbc=True)
    a.rattle(0.02, seed=5)
    return a


def _water_dimer():
    a = molecule("H2O")
    b = molecule("H2O")
    b.translate([0.3, 0.1, 2.9])
    a += b
    a.rattle(0.04, seed=7)
    return a


def _li2h2():
    a = Atoms("Li2H2", positions=[[0, 0, 0], [2.9, 0.2, 0.1], [1.4, 1.2, 0], [1.5, -1.1, 0.2]])
    a.rattle(0.02, seed=8)
    return a


def _li2_tail():
    # 14.8 Bohr: inside the 1-Bohr tail DFTB+ appends after the last table row it uses
    d = 14.8 * 0.529177249
    return Atoms("Li2", positions=[[0, 0, 0], [0.3 * d, 0.2 * d, np.sqrt(0.87) * d]])


def _water_box_folded():
    """The water box with every atom folded into the cell: DFTB+ takes the field's stress
    from the unfolded coordinates (M6c design, section 2)."""
    a = _water_box()
    a.positions = fold_into_cell(a.positions, a.cell.array)
    return a


def _cuo_chain_cubic():
    """The CuO chain in a 7.6 A cube, centred in y and z: DFTB+'s field stress is right for
    cubic cells only (M6c report)."""
    a = _cuo_chain()
    a.set_cell([7.6, 7.6, 7.6])
    a.positions += [0.0, 3.8, 3.8]
    a.positions = fold_into_cell(a.positions, a.cell.array)
    return a


def _o2():
    return Atoms("O2", positions=[[0, 0, 0], [0.05, 0.1, 1.21]])


def _cu4co_box():
    """Cu4CO in a 10 x 10 x 12 A box: a charged cell with a fixed Fermi level whose SCC
    converges like the molecule's (M6f-2)."""
    a = _cu4co()
    a.set_cell([10.0, 10.0, 12.0])
    a.pbc = True
    a.positions += [3.5, 3.5, 3.0]
    return a


def _oh():
    return Atoms("OH", positions=[[0, 0, 0], [0.1, 0.05, 0.97]])


def _li_bcc():
    return bulk("Li", "bcc", a=3.50)     # every shell >= 4e-3 Bohr from a grid node


def _gdo2():
    # bent and off every symmetry plane, so that all elements of the f rotations matter
    return Atoms("GdO2", positions=[[0, 0, 0], [1.85, 0.12, -0.07], [-0.55, 1.75, 0.20]])


def _gdo():
    return Atoms("GdO", positions=[[0, 0, 0], [0.3, -0.2, 1.80]])


def _gd2o2():
    return Atoms("Gd2O2", positions=[[0, 0, 0], [3.4, 0.1, 0.2], [1.7, 1.3, -0.1],
                                     [1.6, -1.2, 0.15]])


def _gdo_rocksalt():
    return bulk("GdO", "rocksalt", a=4.9)    # hypothetical rock-salt GdO


def _gdo_rocksalt_displaced():
    a = _gdo_rocksalt()
    a.positions[1] += [0.03, 0.01, -0.02]    # off the lattice site: non-zero forces
    return a


def _li_bcc_cubic():
    return bulk("Li", "bcc", a=3.50, cubic=True)


def _li_bcc_node():
    return bulk("Li", "bcc", a=3.45)     # the a*sqrt(2) shell is 4.3e-5 Bohr from grid node 461


# non-collinear spin and spin-orbit coupling (milestone M6a): the xi (Hartree) are larger than
# real ones so that every term is clearly non-zero; spins start along a tilted direction where
# DFTB+ converges (its SCC diverges for periodic non-collinear moments off the axes, M6a design)
TILT = [0.48, 0.6, 0.64]
XI_CU4CO = {"Cu": [0.0, 0.0, 0.01], "C": [0.0, 0.004], "O": [0.0, 0.006]}
XI_CUO = {"Cu": [0.0, 0.0, 0.01], "O": [0.0, 0.006]}
XI_GDO = {"Gd": [0.0, 0.005, 0.005, 0.01], "O": [0.0, 0.006]}
NC_O2 = {"initial_spins": [TILT, TILT], "spin_constants": {"O": -0.028}}
NC_OH = {"initial_spins": [TILT, [0.0, 0.0, 0.0]], "spin_constants": W_SPIN}
NC_GDO = {"initial_spins": [[7 * x for x in TILT], TILT], "spin_constants": W_SPIN}
# magnetisation vectors to 1e-6 (results.tag), angular momenta to 1e-8 (detailed.out, 8 decimals)
TOL_PAULI = {"energy": 1e-6, "forces": 1e-6, "charges": 1e-6, "spins": 1e-6, "moments": 1e-8}

# DFTB+U and onsite corrections (milestone M6b): U - J of the Cu d shell 0.15 Ha (4.1 eV, a
# typical literature value), O p 0.1 Ha; onsite constants from the DFTB+ manual (PBE), Cu with
# Au's s/p/d values only to test the code.  Orbital populations (results.tag) are compared too
U_CU = {"groups": {"Cu": [("d", 0.15)]}}
U_O = {"groups": {"O": [("p", 0.1)]}}
ONS = {lab: ONSITE_PBE[lab] for lab in ("H", "C", "O")} | {"Cu": {**ONSITE_PBE["Au"]}}
TOL_BLOCK = {"energy": 1e-6, "forces": 1e-6, "charges": 1e-6, "orbital": 1e-6,
             "eigenvalues": 1e-7, "spins": 1e-6, "moments": 1e-8}

# External potentials and implicit solvation (milestone M6c): a field of 0.5 V/A in a general
# direction, three point charges about 3 A away (or two at 1.2 A with a 0.8 A Gaussian blur),
# gross and net site potentials (eV).  Solvation: the GBSA(water) example of the DFTB+ manual
# (fitted for GFN2-xTB; here only to test the code) and a simple GB model (atomic units)
FIELD = {"strength": 0.5, "direction": [1.0, 0.3, -0.2]}
PC3 = {"positions": [[3.0, 0.2, 0.1], [-2.5, 1.0, 0.5], [0.3, -2.8, 1.2]],
       "charges": [0.8, -0.6, 0.4]}
PC_CLOSE = {"positions": [[1.2, 0.2, 0.1], [-1.1, 0.6, 0.5]], "charges": [0.8, -0.6],
            "blur": 0.8}
SITES = {"gross": {0: 0.5}, "net": {1: -0.3, 2: 0.2}}
GB_WATER = {"model": "gb", "solvent": "water", "free_energy_shift": 0.0, "born_scale": 1.5,
            "born_offset": 0.0, "descreening": "unity"}
GBSA_WATER = {
    "model": "gb", "solvent": {"epsilon": 80.2, "molecular_mass": 18.0, "density": 1.0},
    "free_energy_shift": 1.16556316 * KCAL_MOL, "born_scale": 1.55243817,
    "born_offset": 2.462811043694508e-02,
    "descreening": {"H": 0.71893869, "C": 0.74298311, "N": 0.90261230, "O": 0.75369019},
    "sasa": {"probe_radius": 1.843075777670416,
             "surface_tension": {"H": -0.33498306, "C": -0.74769065, "N": -2.31291292,
                                 "O": 0.91797911}},
    "hbond": {"H": -7.172800544988973e-02, "C": -2.548469535762511e-03,
              "N": -1.976849501504001e-02, "O": -8.462476828587280e-03}}
TOL_ENV = {"energy": 1e-6, "forces": 1e-6, "charges": 1e-6, "pc_forces": 1e-6, "dipole": 1e-6}


def _h2co():
    a = molecule("H2CO")
    a.rattle(0.02, seed=3)
    return a


def _c2h4():
    a = molecule("C2H4")
    a.rattle(0.02, seed=12)        # every distance >= 3e-3 Bohr from an SK grid node
    return a


def _c6h6():
    return molecule("C6H6")                  # D6h: degenerate excitations


def _cuco_linear():
    return Atoms("CuCO", positions=[[0, 0, 0], [0, 0, 1.85], [0, 0, 3.0]])


def _h2co_box():
    a = _h2co()
    a.positions += 4.0
    a.cell = np.eye(3) * 8.0
    a.pbc = True
    return a


def _meoh_water(r):
    """CH3OH and one water (M6f-3): the water's O on the methanol's C->O axis at r (A) from the
    methanol's O, the water's H atoms pointing away; the methanol's O (atom 1) and the water's
    O (atom 6) are the ones relabelled as halogen and acceptor."""
    m = molecule("CH3OH")
    c, o = m.positions[0], m.positions[1]
    u = (o - c) / np.linalg.norm(o - c)
    w = molecule("H2O")
    w.positions -= w.positions[0]
    w.rotate((0, 0, -1), u)
    w.rotate(37.0, u)
    w.positions += o + r * u
    return m + w


def _relabel(**at):
    """Type labels: the chemical symbols with atom k (0-based) renamed, ``at`` = {"a1": "Cl"}."""
    def labels(a):
        out = a.get_chemical_symbols()
        for key, lab in at.items():
            out[int(key[1:])] = lab
        return out
    return labels


def _water_layer(z=(0.0, 0.0, 0.0, 0.0), seed=11):
    """Four waters in a layer, every dipole along +z, in a 6 x 6 x 20 A cell (M6f-3); ``z``
    shifts each molecule (a whole period puts it into another image)."""
    w = molecule("H2O")
    w.rotate(180, "x")
    a = Atoms(cell=[6.0, 6.0, 20.0], pbc=True)
    for (x, y), dz in zip([(0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (3.0, 3.0)], z):
        m = w.copy()
        m.translate([x + 0.5, y + 0.7, dz])
        a += m
    a.rattle(0.05, seed=seed)
    return a


def _water_layer_folded():
    return _water_layer((15.0, -5.0, 15.0, -5.0), seed=16)


W_HCO = {"H": -0.072, "C": -0.023, "O": -0.028}       # spin constants of the Casida cases
TOL_CASIDA = {"energy": 1e-6, "forces": 1e-6, "charges": 1e-6, "exc": 1e-8, "exc_osc": 1e-7,
              "exc_tdip": 1e-7, "exc_table": 0, "spx": 0, "exc_state": 1e-9}
# M6e: the state of interest's forces; XCH.DAT (8 decimals), XREST.DAT (8 decimals, Debye)
EXC = {"n": 6, "symmetry": "singlet", "solver": "Arpack", "state": 2, "excited_forces": True}
EXC_FD = {**EXC, "solver": "Stratmann", "forces_fd": True}
TOL_EXC_FD = {"forces": 1e-9}          # against differences of DFTB+'s full-precision E0 + w
TOL_EXC = {**TOL_CASIDA, "exc_charges": 1e-7, "exc_dipole": 1e-7, "forces_dftbp": 1e-6}
# SCC start and mixers (milestone M6f-1): the SCC iterations exactly, the SCC error of every
# iteration relative to DFTB+'s 8 printed digits (where above 1e-7; round-off differences grow
# to 3e-5 in the longest runs), a charge file's populations against pydftb's own run 1
TOL_SCC = {"energy": 1e-6, "forces": 1e-6, "charges": 1e-6, "spins": 1e-6, "n_scc": 0,
           "scc_errors": 1e-4, "charges_file": 1e-8}
BROYDEN_WEIGHTS = {"kind": "broyden", "mixing": 0.3, "inverse_jacobi_weight": 0.02,
                   "minimal_weight": 2.0, "maximal_weight": 1e4, "weight_factor": 0.05}
# occupations and overrides (milestone M6f-2): as TOL_SCC, with the force-related energy, the
# electron count and the Fermi level(s) (results.tag); the stress of a charged cell against
# stress_fd
TOL_FILL = {"energy": 1e-6, "forces": 1e-6, "charges": 1e-6, "spins": 1e-6, "n_scc": 0,
            "scc_errors": 1e-4, "e_fr": 1e-6, "n_electrons": 1e-6, "fermi_level": 1e-8}
# M6f-3: corrections; "e_halogen" against detailed.out's "Halogen correction energy" (10 decimals)
TOL_CORR = {"energy": 1e-6, "forces": 1e-6, "charges": 1e-6, "n_scc": 0, "scc_errors": 1e-4,
            "e_halogen": 1e-10}
UD_HALOGEN = {**UD_3OB, "Cl": UD_3OB["O"], "Br": UD_3OB["O"], "N": UD_3OB["O"]}   # type copies
                                                                               # of O: test only
# M5a: smooth type switching, discrete limit (the cu_types set: Cu = PTBP's Cu, Cu2 = looser
# orbitals; make_cu_types_set.py).  Default counting window 2.904-3.432 A: the first fcc shell
# counts 1, the second 0, so the coordination numbers are integers; "switch" is the deviation of
# pydftb with the switch from pydftb with the explicit labels (bitwise: 0)
SWITCH_SLAB = {"element": "Cu", "types": ("Cu2", "Cu"), "centre": 10.5, "width": 1.0}
SWITCH_CLUSTER = {"element": "Cu", "types": ("Cu2", "Cu"), "centre": 8.5, "width": 3.0}
TOL_TYPES = {"energy": 1e-6, "forces": 1e-6, "charges": 1e-6, "switch": 0.0}


def _cu_slab111():
    """Cu(111), 2 x 2 x 4 layers, 10 A of vacuum: surface atoms CN 9, inner atoms CN 12."""
    from ase.build import fcc111
    a = fcc111("Cu", size=(2, 2, 4), a=3.61, vacuum=5.0, periodic=True)
    a.rattle(0.03, seed=11)
    return a


def _cu13():
    """The 13-atom cuboctahedron: the 12 outer atoms CN 5, the centre CN 12."""
    from ase.cluster import Octahedron
    a = Octahedron("Cu", 3, cutoff=1, latticeconstant=3.61)
    a.rattle(0.04, seed=12)
    return a


def _switch_labels(spec):
    """Type labels of the discrete limit of a switch: each atom's type of weight 1 (an error if
    a weight lies strictly between 0 and 1)."""
    def labels(atoms):
        from ..typeswitch import TypeSwitch
        sw = TypeSwitch.make(spec)
        periodic = bool(atoms.pbc.any())
        elements = atoms.get_chemical_symbols()
        cn = sw.coordination_numbers(elements, atoms.positions / BOHR,
                                     atoms.cell.array / BOHR if periodic else None,
                                     atoms.pbc if periodic else None)
        w, _ = sw.weights(cn)
        out = []
        for el, ww in zip(elements, w):
            if el != sw.element:
                out.append(el)
                continue
            if not set(np.unique(ww)) <= {0.0, 1.0}:
                raise ValueError("a weight between 0 and 1: not the discrete limit")
            out.append(sw.types[int(np.argmax(ww))])
        return out
    return labels


def solvation_model(spec):
    """The pydftb model (:mod:`pydftb.solvation`) of a case's solvation spec."""
    from .. import solvation as sv
    if spec is None:
        return None

    def sasa(d):
        keys = ("probe_radius", "surface_tension", "radii", "radii_table", "smoothing",
                "tolerance", "angular_grid", "offset")
        return sv.SASA(**{k: d[k] for k in keys if k in d})

    if spec["model"] == "sasa":
        return sasa(spec)
    opts = {k: spec[k] for k in ("free_energy_shift", "born_scale", "born_offset", "descreening",
                                 "kernel", "alpb", "alpha", "state", "temperature", "hbond")
            if k in spec}
    if "solvent" in spec:
        sol = spec["solvent"]
        opts["solvent"] = sol if isinstance(sol, str) else sv.Solvent(**sol)
    if "sasa" in spec:
        opts["sasa"] = sasa(spec["sasa"])
    if spec.get("cm5") is not None:
        opts["cm5"] = sv.CM5(**spec["cm5"])
    if "param_file" in spec:
        return sv.GeneralizedBorn.from_param_file(os.path.join(DATA, spec["param_file"]), **opts)
    return sv.GeneralizedBorn(**opts)


CASES = [
    Case("h2o", _water, tol=TOL_DFTBP_FD, note="force tolerance 2e-6, see TOL_DFTBP_FD"),
    Case("h2o_nonscc", _water, scc=False),
    Case("ch3oh", _meoh),
    Case("cu4co_300K", _cu4co, filling=("fermi", 300.0)),
    Case("cu4co_shell_resolved", _cu4co, filling=("fermi", 300.0), shell_resolved=True),
    Case("cu_fcc_k6", _cu_fcc, kgrid=(6, 6, 6), filling=("fermi", 1000.0)),
    Case("cu8_distorted_k355", _cu8, kgrid=(3, 5, 5), filling=("fermi", 1000.0)),
    Case("water_box_gamma", _water_box),
    Case("cuo_chain_k422", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0)),
    Case("water_dimer_typed", _water_dimer, typed={"O": "O", "O2": "O", "H": "H"},
         labels=lambda a: ["O", "H", "H", "O2", "H", "H"],
         note="second O uses its own type label O2 (copy of O): multi-type machinery vs DFTB+"),
    Case("li2h2_orbit", _li2h2, skf_set="orbit", filling=("fermi", 300.0),
         note="second parameter set (test/orbit), Li with s-only basis (ASE/PTBP max-l rule)"),
    Case("cu_fcc_k6_T0", _cu_fcc, kgrid=(6, 6, 6), filling=("fermi", 0.0),
         note="metal at T=0: DFTB+ uses Fermi filling at kT=1e-8 Ha"),
    # --- smearing (milestone M1) ---
    Case("cu_fcc_k6_mp1_3000K", _cu_fcc, kgrid=(6, 6, 6), filling=("mp", 3000.0, 1)),
    Case("cu8_distorted_k355_mp2_2000K", _cu8, kgrid=(3, 5, 5), filling=("mp", 2000.0, 2),
         note="the smearing PTBP intended (MP order 2, 2000 K)"),
    Case("cu8_distorted_k355_gauss_1000K", _cu8, kgrid=(3, 5, 5), filling=("gaussian", 1000.0)),
    Case("cu4co_mp2_1000K", _cu4co, filling=("mp", 1000.0, 2)),
    Case("water_box_mp1_1000K", _water_box, filling=("mp", 1000.0, 1),
         note="gapped system with MP smearing: Fermi level from DFTB+'s mid-gap search"),
    Case("h2o_gauss_2000K", _water, filling=("gaussian", 2000.0), tol=TOL_DFTBP_FD,
         note="force tolerance 2e-6, see TOL_DFTBP_FD"),
    # --- repulsive variants (milestone M1) ---
    Case("water_dimer_polyrep", _water_dimer, skf_set="orbit_poly", repulsive="polynomial",
         note="non-zero polynomial repulsive, DFTB+ PolynomialRepulsive = SetForAll {Yes}"),
    # --- SK table end and stress (milestone M2a) ---
    Case("li2_sk_tail", _li2_tail, skf_set="orbit",
         tol={"energy": 1e-9, "forces": 1e-9, "charges": 1e-9},
         note="Li-Li at 14.8 Bohr, in the 5th-order tail after the table: DFTB+ uses ngrid - 1 rows"),
    Case("li_bcc_k666_1000K", _li_bcc, skf_set="orbit", kgrid=(6, 6, 6), filling=("fermi", 1000.0),
         tol={"energy": 1e-8, "forces": 1e-9, "charges": 1e-9, "stress": 1e-10},
         note="s-only metal; two neighbour shells lie in the SK table tail"),
    Case("li_bcc_k666_node", _li_bcc_node, skf_set="orbit", kgrid=(6, 6, 6), filling=("fermi", 1000.0),
         tol={"energy": 1e-8, "forces": 1e-9, "charges": 1e-9, "stress": 2e-8},
         note="a neighbour shell is closer to a grid node than DFTB+'s finite-difference step, "
              "so DFTB+'s stress is off by 1.2e-8 (pydftb with fd_delta agrees to 1e-12)"),
    # --- DFTB3 (milestone M2b) ---
    Case("h2o_dftb3", _water, dftb3=DFTB3_3OB, tol=TOL_DFTBP_FD,
         note="full third order + H-X damping (3ob settings); force tolerance see TOL_DFTBP_FD"),
    Case("ch3oh_dftb3", _meoh, dftb3=DFTB3_3OB),
    Case("water_dimer_dftb3_nodamp", _water_dimer,
         dftb3={"third_order": "full", "hubbard_derivs": UD_3OB}),
    Case("water_dimer_hdamp", _water_dimer, dftb3={"h_damping": 4.0},
         note="DFTB2 with the H-X damping of gamma only"),
    Case("h2o_dftb3_onsite", _water, dftb3={"third_order": "onsite", "hubbard_derivs": UD_3OB},
         tol=TOL_DFTBP_FD,
         note="DFTB+ ThirdOrder (R = 0 terms only), which DFTB+ books under 'Energy SCC'"),
    Case("cu4co_dftb3_shell", _cu4co, filling=("fermi", 300.0), shell_resolved=True,
         dftb3={"third_order": "full",
                "hubbard_derivs": {"Cu": [-0.05, -0.03, -0.08], "C": [-0.1492, -0.12],
                                   "O": [-0.1575, -0.14]}},
         note="orbital-resolved SCC, one Ud per shell: on-site Gamma between shells of different U"),
    Case("water_box_dftb3", _water_box, dftb3=DFTB3_3OB),
    Case("cuo_chain_dftb3_k422", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         dftb3={"third_order": "full", "hubbard_derivs": {"Cu": -0.05, "O": -0.1575}}),
    # --- collinear spin (milestone M2c) ---
    Case("o2_triplet", _o2, spin={"unpaired_electrons": 2, "spin_constants": {"O": -0.028}},
         note="one spin constant per type (DFTB+ ShellResolvedSpin = No)"),
    Case("o2_triplet_wmatrix", _o2, spin={"unpaired_electrons": 2, "spin_constants": W_SPIN}),
    Case("o2_triplet_shell", _o2, shell_resolved=True,
         spin={"unpaired_electrons": 2, "spin_constants": W_SPIN}),
    # 300 K: at T = 0 the half-filled, degenerate pi level of the linear radical makes the
    # last SCC steps wander at the 1e-9 level in DFTB+ (identical paths down to 4e-9)
    Case("oh_radical", _oh, filling=("fermi", 300.0),
         spin={"unpaired_electrons": 1, "spin_constants": W_SPIN}),
    Case("o2_relaxed_spin", _o2, filling=("fermi", 300.0),
         spin={"unpaired_electrons": 2, "relax_total_spin": True, "spin_constants": W_SPIN},
         note="one Fermi level for both spins (DFTB+ RelaxTotalSpin)"),
    Case("oh_radical_dftb3", _oh, filling=("fermi", 300.0), dftb3=DFTB3_3OB,
         spin={"unpaired_electrons": 1, "spin_constants": W_SPIN}),
    Case("cuo_chain_spin_k422", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         spin={"unpaired_electrons": 2, "spin_constants": W_SPIN}),
    # --- dispersion (milestone M2d) ---
    Case("water_dimer_uff", _water_dimer, dispersion={"model": "lennardjones"}),
    Case("water_box_uff", _water_box, dispersion={"model": "lennardjones"},
         note="r^-6 by Ewald, r^-12 and the short-range wall in real space"),
    Case("cuo_chain_uff_k422", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         dispersion={"model": "lennardjones"}),
    Case("water_dimer_sk", _water_dimer, dispersion=lambda labels: sk_per_atom(labels, skip=(3,)),
         note="per-atom PolarRadiusCharge; the second O has no polarisability and is skipped"),
    Case("ch3oh_sk_hybrid", _meoh, dispersion={"model": "slaterkirkwood",
                                               "hybrid_dependent_pol": SK_HYBRID}),
    Case("water_box_sk_hybrid", _water_box, dispersion={"model": "slaterkirkwood",
                                                        "hybrid_dependent_pol": SK_HYBRID}),
    Case("ch3oh_d3bj", _meoh, dispersion=D3BJ_3OB),
    Case("water_box_d3bj_3body", _water_box, dispersion={**D3BJ_3OB, "threebody": True}),
    Case("cuo_chain_d3bj_k422", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         dispersion=D3BJ_3OB),
    Case("water_dimer_d3zero_3body", _water_dimer,
         dispersion={"model": "dftd3", "damping": "zerodamping", "sr6": 1.25, "alpha6": 29.61,
                     "s6": 1.0, "s8": 0.49, "threebody": True}),
    Case("water_dimer_d3_hhrep", _water_dimer,
         dispersion={"model": "dftd3", "damping": "zerodamping", "sr6": 1.25, "alpha6": 29.61,
                     "s6": 1.0, "s8": 0.49, "hhrepulsion": True},
         note="D3H5-style zero damping with the H-H repulsion (without the H5 correction)"),
    Case("water_dimer_typed_d3", _water_dimer, typed={"O": "O", "O2": "O", "H": "H"},
         labels=lambda a: ["O", "H", "H", "O2", "H", "H"],
         dispersion={"model": "dftd3", "damping": "modifiedzerodamping", "sr6": 1.338153,
                     "beta": 0.013988, "s6": 1.0, "s8": 1.532981},
         note="type label O2 mapped to oxygen (DFTB+ AtomicNumbers); modified zero damping "
              "with the B3LYP-D3M(0) values (only to test the input path)"),
    Case("ch3oh_simple_d3", _meoh,
         dispersion={"model": "simpledftd3", "s8": 0.5883, "a1": 0.5719, "a2": 3.6017}),
    Case("water_box_simple_d3", _water_box,
         dispersion={"model": "simpledftd3", "s8": 0.5883, "a1": 0.5719, "a2": 3.6017}),
    Case("water_dimer_d4", _water_dimer, dispersion=D4_3OB),
    Case("ch3oh_d4", _meoh, dispersion=D4_3OB),
    Case("water_box_d4_2body", _water_box, dispersion=D4_3OB_2BODY_Q0,
         note="D4 two-body terms without charge dependence (see D4_3OB_2BODY_Q0)"),
    Case("h2o_dftb3_d3bj", _water, dftb3=DFTB3_3OB, dispersion=D3BJ_3OB, tol=TOL_DFTBP_FD,
         note="DFTB3/3ob + D3(BJ); force tolerance see TOL_DFTBP_FD"),
    Case("o2_triplet_d3bj", _o2, spin={"unpaired_electrons": 2, "spin_constants": W_SPIN},
         dispersion=D3BJ_3OB),
    # --- band structures and projected DOS (milestone M2e) ---
    Case("cu_fcc_k6_bands", _cu_fcc, kgrid=(6, 6, 6), filling=("fermi", 1000.0),
         klines=path(FCC, "GXWLGK", [10, 5, 7, 9, 10]),
         regions=[{"label": "cu", "atoms": [0], "shell_resolved": True}],
         ewald_tolerance=1e-12, tol=TOL_BANDS,
         note="SCC on the grid with s/p/d projections, then DFTB+'s band run on G-X-W-L-G-K"),
    Case("li_bcc_bands", _li_bcc, skf_set="orbit", kgrid=(6, 6, 6), filling=("fermi", 1000.0),
         klines=path(BCC, "GHNGPH", [10, 7, 7, 9, 5]), ewald_tolerance=1e-12, tol=TOL_BANDS),
    Case("cuo_chain_spin_bands", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         spin={"unpaired_electrons": 2, "spin_constants": W_SPIN},
         klines=path(ORC, "GXSGZU", [10, 5, 10, 0, 6]),
         regions=[{"label": "cu", "atoms": [0, 2]},
                  {"label": "o", "atoms": [1, 3], "shell_resolved": True}],
         dp_dos=[("band.out", ["-g", "0.05"])], ewald_tolerance=1e-12, tol=TOL_BANDS,
         note="two spin channels and Fermi levels; the path jumps (n = 0) from G to Z"),
    Case("cuo_chain_dftb3_bands", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         dftb3={"third_order": "full", "hubbard_derivs": {"Cu": -0.05, "O": -0.1575}},
         klines=path(ORC, "GX", [12]), ewald_tolerance=1e-12, tol=TOL_BANDS),
    Case("water_box_bands", _water_box, klines=path({"G": (0, 0, 0), "X": (0.5, 0, 0),
                                                     "M": (0.5, 0.5, 0)}, "GXM", [6, 6]),
         ewald_tolerance=1e-12, tol=TOL_BANDS, note="SCC at Gamma, band run on G-X-M"),
    Case("cu4co_pdos", _cu4co, filling=("fermi", 300.0),
         regions=[{"label": "cu", "atoms": [0, 1, 2, 3], "shell_resolved": True},
                  {"label": "c", "atoms": [4], "orbital_resolved": True},
                  {"label": "o", "atoms": [5]}],
         dp_dos=[("band.out", ["-g", "0.05"]), ("cu.3.out", ["-w", "-g", "0.05"]),
                 ("c.2.0.out", ["-w", "-f", "fermi", "-b", "0.2", "-g", "0.05"]),
                 ("o.out", ["-w", "-f", "mp", "-o", "2", "-g", "0.05"])],
         tol=TOL_BANDS, note="molecular PDOS from the SCC run's projections; dp_dos curves"),
    # --- geometry optimisation and MD (milestone M2f; the driver cases are in dynamics.py) ---
    Case("li_bcc_cubic_k444_1000K", _li_bcc_cubic, skf_set="orbit", kgrid=(4, 4, 4),
         filling=("fermi", 1000.0), tol={"energy": 1e-8, "forces": 1e-9, "charges": 1e-9},
         note="conventional cubic cell (2 atoms), base of the isotropic lattice optimisation"),
    # --- loose SCC tolerance, as PTBP uses (milestone M2f): forces from the output charges ---
    Case("ch3oh_loose_scc", _meoh, scc_tol=1e-5,
         note="SCC tolerance 1e-5: DFTB+ takes the force charges from the last output charges "
              "(RecomputeAfterDensity); input charges would differ by 1.4e-7 Ha/Bohr"),
    Case("water_dimer_dftb3_loose_scc", _water_dimer, scc_tol=1e-5,
         dftb3={"third_order": "full", "hubbard_derivs": UD_3OB},
         note="as ch3oh_loose_scc, with DFTB3 (input charges: 8.1e-7 Ha/Bohr)"),
    Case("cuo_chain_spin_loose_scc", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         spin={"unpaired_electrons": 2, "spin_constants": W_SPIN}, scc_tol=1e-5,
         ewald_tolerance=1e-12, tol={"energy": 1e-6, "forces": 1e-6, "charges": 1e-6, "stress": 1e-9},
         note="as ch3oh_loose_scc, periodic with spin; also the stress"),
    # --- f orbitals (milestone M2g), extended SKF format ---
    Case("gdo2_nonscc", _gdo2, scc=False, **GDO, note="H0 and S only: f integrals and rotations"),
    Case("gdo2_atom", _gdo2, **GDO, note="f orbitals in the Mulliken charges and the SCC"),
    Case("gdo2_shell", _gdo2, shell_resolved=True, **GDO, note="U of the f shell"),
    Case("gdo2_dftb3_shell", _gdo2, shell_resolved=True, **GDO,
         dftb3={"third_order": "full", "hubbard_derivs": UD_GDO}, note="Ud of the f shell"),
    Case("gdo_spin", _gdo, shell_resolved=True, **GDO,
         spin={"unpaired_electrons": 8, "spin_constants": W_SPIN}, note="4x4 spin constants"),
    Case("gdo_rocksalt_k444", _gdo_rocksalt_displaced, kgrid=(4, 4, 4), ewald_tolerance=1e-12,
         **GDO, note="k-points, smearing, forces and stress; Gd-Gd uses all 20 integrals"),
    Case("gd2o2_typed", _gd2o2, typed={"Gd": "Gd", "Gd2": "Gd", "O": "O"},
         labels=lambda a: ["Gd", "Gd2", "O", "O"], **GDO,
         note="type copies of extended-format files (write_typed_copies)"),
    Case("gdo_rocksalt_bands", _gdo_rocksalt, kgrid=(4, 4, 4), ewald_tolerance=1e-12, **GDO,
         klines=path(FCC, "GXWLGK", [10, 5, 7, 9, 10]),
         regions=[{"label": "gd", "atoms": [0], "shell_resolved": True}], tol=TOL_BANDS,
         note="band run with f bands; s/p/d/f projections of Gd"),
    # --- non-collinear spin and spin-orbit coupling (milestone M6a), NumPy only ---
    Case("o2_noncolinear_tilted", _o2, noncolinear=NC_O2, torch=False, tol=TOL_PAULI,
         note="moments along (0.48, 0.6, 0.64): the energy of o2_triplet (rotation invariance)"),
    Case("oh_noncolinear_tilted", _oh, filling=("fermi", 300.0), noncolinear=NC_OH,
         torch=False, tol=TOL_PAULI),
    Case("oh_noncolinear_soc", _oh, filling=("fermi", 300.0), noncolinear=NC_OH,
         spin_orbit={"xi": {"O": [0.0, 0.01]}}, torch=False, tol=TOL_PAULI,
         note="dual spin-orbit model (DFTB+ default); the moment turns to the bond axis"),
    Case("oh_noncolinear_soc_onsite", _oh, filling=("fermi", 300.0), noncolinear=NC_OH,
         spin_orbit={"xi": {"O": [0.0, 0.01]}, "dual": False}, torch=False, tol=TOL_PAULI),
    Case("oh_noncolinear_soc_dftb3", _oh, filling=("fermi", 300.0), dftb3=DFTB3_3OB,
         noncolinear=NC_OH, spin_orbit={"xi": {"O": [0.0, 0.01]}}, torch=False, tol=TOL_PAULI),
    Case("o2_noncolinear_soc", _o2,
         noncolinear={"initial_spins": [[0.0, 0.0, 1.0]] * 2, "spin_constants": {"O": -0.028}},
         spin_orbit={"xi": {"O": [0.0, 0.01]}}, torch=False, tol=TOL_PAULI,
         note="T = 0; the moments turn from z to the molecular axis"),
    Case("o2_noncolinear_soc_onsite", _o2,
         noncolinear={"initial_spins": [[0.0, 0.0, 1.0]] * 2, "spin_constants": {"O": -0.028}},
         spin_orbit={"xi": {"O": [0.0, 0.01]}, "dual": False}, torch=False, tol=TOL_PAULI),
    Case("cu4co_soc", _cu4co, filling=("fermi", 300.0), spin_orbit={"xi": XI_CU4CO},
         torch=False, tol=TOL_PAULI, note="spin-orbit coupling without spin polarisation"),
    Case("cu4co_soc_onsite", _cu4co, filling=("fermi", 300.0),
         spin_orbit={"xi": XI_CU4CO, "dual": False}, torch=False, tol=TOL_PAULI),
    Case("cu4co_soc_dftb3_shell", _cu4co, filling=("fermi", 300.0), shell_resolved=True,
         dftb3={"third_order": "full",
                "hubbard_derivs": {"Cu": [-0.05, -0.03, -0.08], "C": [-0.1492, -0.12],
                                   "O": [-0.1575, -0.14]}},
         spin_orbit={"xi": XI_CU4CO}, torch=False, tol=TOL_PAULI),
    Case("cu4co_soc_nonscc", _cu4co, scc=False, filling=("fermi", 300.0),
         spin_orbit={"xi": XI_CU4CO}, torch=False, tol=TOL_PAULI),
    Case("gdo_noncolinear_tilted", _gdo, shell_resolved=True, **GDO, noncolinear=NC_GDO,
         torch=False, tol=TOL_PAULI, note="f orbitals, 4x4 spin constants"),
    Case("gdo_noncolinear_soc", _gdo, shell_resolved=True, **GDO, noncolinear=NC_GDO,
         spin_orbit={"xi": XI_GDO}, torch=False, tol=TOL_PAULI,
         note="spin-orbit coupling of p, d and f shells (l = 3 operators); a long SCC"),
    Case("cuo_chain_soc_k422", _cuo_chain, kgrid=(4, 2, 2), time_reversal=False,
         filling=("fermi", 500.0), spin_orbit={"xi": XI_CUO}, ewald_tolerance=1e-12,
         torch=False, tol={**TOL_PAULI, "stress": 1e-9},
         note="full k-point grid; forces and stress of the dual model"),
    Case("cuo_chain_noncolinear_k422", _cuo_chain, kgrid=(4, 2, 2), time_reversal=False,
         filling=("fermi", 500.0),
         noncolinear={"initial_spins": [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]] * 2,
                      "spin_constants": W_SPIN},
         ewald_tolerance=1e-12, torch=False, tol={**TOL_PAULI, "stress": 1e-9},
         note="periodic two-component path (moments along z: DFTB+ converges)"),
    Case("cu_fcc_soc_bands", _cu_fcc, kgrid=(6, 6, 6), time_reversal=False,
         filling=("fermi", 1000.0), spin_orbit={"xi": {"Cu": [0.0, 0.0, 0.01]}},
         klines=path(FCC, "GXWLGK", [10, 5, 7, 9, 10]),
         regions=[{"label": "cu", "atoms": [0], "shell_resolved": True}],
         dp_dos=[("band.out", ["-g", "0.05"])], ewald_tolerance=1e-12, torch=False,
         tol={**TOL_BANDS, "spins": 1e-6, "moments": 1e-8},
         note="spinor bands (d bands split at Gamma), projections and DOS"),
    # --- DFTB+U and onsite corrections (milestone M6b), NumPy only ---
    Case("cu4co_plusu", _cu4co, filling=("fermi", 300.0), plus_u=U_CU, torch=False,
         tol=TOL_BLOCK, note="FLL on the Cu d shells"),
    Case("cu4co_plusu_psic", _cu4co, filling=("fermi", 300.0),
         plus_u={**U_CU, "functional": "psic"}, torch=False, tol=TOL_BLOCK),
    Case("cu4co_plusu_groups_shell", _cu4co, filling=("fermi", 300.0), shell_resolved=True,
         plus_u={"groups": {"Cu": [("pd", 0.15)], "O": [("p", 0.1)]}}, torch=False,
         tol=TOL_BLOCK, note="a p-d group (cross-shell blocks), two types, shell charges"),
    Case("cuo_chain_spin_plusu_k422", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         spin={"unpaired_electrons": 2, "spin_constants": W_SPIN}, plus_u=U_CU,
         ewald_tolerance=1e-12, torch=False, tol={**TOL_BLOCK, "stress": 1e-9},
         note="collinear spin (q, m blocks, up/down mixing), k-points, forces and stress"),
    Case("gdo_spin_plusu", _gdo, shell_resolved=True, **GDO,
         spin={"unpaired_electrons": 8, "spin_constants": W_SPIN},
         plus_u={"groups": {"Gd": [("f", 0.2)]}}, torch=False, tol=TOL_BLOCK,
         note="f shell (7 x 7 blocks)"),
    Case("oh_noncolinear_plusu", _oh, filling=("fermi", 300.0), noncolinear=NC_OH, plus_u=U_O,
         torch=False, tol=TOL_BLOCK, note="the four Pauli components"),
    Case("oh_noncolinear_soc_plusu", _oh, filling=("fermi", 300.0), noncolinear=NC_OH,
         spin_orbit={"xi": {"O": [0.0, 0.01]}}, plus_u=U_O, torch=False, tol=TOL_BLOCK,
         note="dual spin-orbit coupling: imaginary +U potentials and skew blocks"),
    Case("cu4co_soc_plusu", _cu4co, filling=("fermi", 300.0), spin_orbit={"xi": XI_CU4CO},
         plus_u=U_CU, torch=False, tol=TOL_BLOCK, note="two-component without spin"),
    Case("cuo_chain_soc_plusu_k422", _cuo_chain, kgrid=(4, 2, 2), time_reversal=False,
         filling=("fermi", 1000.0), spin_orbit={"xi": XI_CUO}, plus_u=U_CU,
         ewald_tolerance=1e-12, torch=False, tol={**TOL_BLOCK, "stress": 1e-9},
         note="forces and stress of imaginary block potentials at k-points; 1000 K (at 500 K "
              "both codes need ~140 SCC iterations, with U - J = 0.1 DFTB+ diverges)"),
    Case("h2o_ons", _water, onsite=ONS, torch=False, tol={**TOL_BLOCK, **TOL_DFTBP_FD},
         note="onsite corrections; force tolerance see TOL_DFTBP_FD"),
    Case("ch3oh_ons", _meoh, onsite=ONS, torch=False, tol=TOL_BLOCK),
    Case("ch3oh_ons_dftb3", _meoh, dftb3=DFTB3_3OB, onsite=ONS, torch=False, tol=TOL_BLOCK),
    Case("oh_radical_ons", _oh, filling=("fermi", 300.0),
         spin={"unpaired_electrons": 1, "spin_constants": W_SPIN}, onsite=ONS, torch=False,
         tol=TOL_BLOCK, note="the magnetisation uses f_uu - f_ud"),
    Case("water_box_ons", _water_box, onsite=ONS, ewald_tolerance=1e-12, torch=False,
         tol={**TOL_BLOCK, "stress": 1e-9}, note="periodic images in the block populations"),
    Case("cuo_chain_spin_ons_k422", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         spin={"unpaired_electrons": 2, "spin_constants": W_SPIN}, onsite=ONS,
         ewald_tolerance=1e-12, torch=False, tol={**TOL_BLOCK, "stress": 1e-9},
         note="s/p/d onsite blocks at k-points (Cu with Au's constants)"),
    Case("oh_noncolinear_soc_ons", _oh, filling=("fermi", 300.0), noncolinear=NC_OH,
         spin_orbit={"xi": {"O": [0.0, 0.01]}}, onsite=ONS, torch=False, tol=TOL_BLOCK,
         note="imaginary onsite potentials (dual spin-orbit coupling)"),
    Case("cu4co_plusu_ons", _cu4co, filling=("fermi", 300.0), plus_u=U_CU, onsite=ONS,
         torch=False, tol=TOL_BLOCK, note="both block potentials"),
    Case("cuo_chain_spin_plusu_bands", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         spin={"unpaired_electrons": 2, "spin_constants": W_SPIN}, plus_u=U_CU,
         klines=path(ORC, "GXSGZU", [10, 5, 10, 0, 6]),
         regions=[{"label": "cu", "atoms": [0, 2], "shell_resolved": True}],
         ewald_tolerance=1e-12, torch=False, tol={**TOL_BANDS, "orbital": 1e-6, "spins": 1e-6},
         note="band run with +U (the block vector read back), d projections"),
    # --- fields, point charges, site potentials, implicit solvation (milestone M6c) ---
    Case("h2o_field", _water, efield=FIELD, torch=False, tol={**TOL_ENV, **TOL_DFTBP_FD},
         note="force tolerance see TOL_DFTBP_FD"),
    Case("h2o_pc", _water, point_charges=PC3, torch=False, tol={**TOL_ENV, **TOL_DFTBP_FD}),
    Case("h2o_pc_blur", _water, point_charges=PC_CLOSE, torch=False,
         tol={**TOL_ENV, **TOL_DFTBP_FD}, note="charges at 1.2 A with a 0.8 A Gaussian blur"),
    Case("h2o_site", _water, site_potentials=SITES, torch=False,
         tol={**TOL_ENV, **TOL_DFTBP_FD}, note="gross (O) and net (H) site potentials"),
    Case("ch3oh_dftb3_field_site", _meoh, dftb3=DFTB3_3OB, efield=FIELD,
         site_potentials={"gross": {0: 0.3}, "net": {1: -0.2}}, torch=False, tol=TOL_ENV),
    Case("o2_triplet_field", _o2, spin={"unpaired_electrons": 2, "spin_constants": {"O": -0.028}},
         efield=FIELD, torch=False, tol=TOL_ENV),
    Case("oh_noncolinear_soc_field_pc", _oh, filling=("fermi", 300.0), noncolinear=NC_OH,
         spin_orbit={"xi": {"O": [0.0, 0.01]}}, efield=FIELD, point_charges=PC3, torch=False,
         tol={**TOL_PAULI, **TOL_ENV}),
    Case("cu4co_plusu_field_net", _cu4co, filling=("fermi", 300.0), plus_u=U_CU, efield=FIELD,
         site_potentials={"net": {0: 0.3}}, torch=False, tol={**TOL_BLOCK, **TOL_ENV},
         note="the block path with a net site potential on a Cu atom"),
    Case("ch3oh_ons_pc", _meoh, onsite=ONS, point_charges=PC3, torch=False,
         tol={**TOL_BLOCK, **TOL_ENV}),
    Case("water_box_field", _water_box_folded, efield=FIELD, ewald_tolerance=1e-12, torch=False,
         tol={**TOL_ENV, "stress": 1e-9},
         note="saw-tooth field; DFTB+'s stress is symmetrised before the comparison"),
    Case("cuo_chain_field_k411", _cuo_chain_cubic, kgrid=(4, 1, 1), filling=("fermi", 500.0),
         efield={**FIELD, "strength": 0.1}, ewald_tolerance=1e-12, torch=False,
         tol={**TOL_ENV, "stress": 1e-9},
         note="k-points; a cubic cell (DFTB+'s field stress is wrong off the diagonal of others)"),
    Case("water_box_pc", _water_box,
         point_charges={"positions": (np.array(PC3["positions"]) + 2.0).tolist(),
                        "charges": PC3["charges"], "blur": 0.3},
         ewald_tolerance=1e-12, torch=False, tol=TOL_ENV,
         note="Ewald sum over the point charges' images; no stress (as DFTB+)"),
    Case("h2o_gbsa", _water, solvation=GBSA_WATER, torch=False, tol={**TOL_ENV, **TOL_DFTBP_FD},
         note="GBSA: Born term, SASA and hydrogen-bond correction"),
    Case("ch3oh_gb", _meoh, solvation=GB_WATER, torch=False, tol=TOL_ENV),
    Case("ch3oh_alpb_p16", _meoh, solvation={**GB_WATER, "alpb": True, "kernel": "p16"},
         torch=False, tol=TOL_ENV),
    Case("oh_minus_alpb", _oh, filling=("fermi", 300.0), charge=-1.0,
         solvation={**GB_WATER, "alpb": True}, torch=False, tol=TOL_ENV,
         note="the ALPB shape term of a charged molecule"),
    Case("ch3oh_gb_cm5", _meoh, solvation={**GB_WATER, "cm5": {"double_count": True}},
         torch=False, tol=TOL_ENV, note="CM5 as DFTB+ 25.1 counts it (every pair twice)"),
    Case("ch3oh_sasa", _meoh, solvation={"model": "sasa", **GBSA_WATER["sasa"]}, torch=False,
         tol=TOL_ENV),
    Case("ch3oh_gbsa_paramfile", _meoh,
         solvation={"model": "gb", "param_file": "solvation/param_gbsa_test.txt",
                    "state": "mol1bar"}, torch=False, tol=TOL_ENV,
         note="an xtb-format parameter file and the mol1bar reference state"),
    Case("h2o_gb_field_pc", _water, solvation=GB_WATER, efield=FIELD, point_charges=PC3,
         torch=False, tol={**TOL_ENV, **TOL_DFTBP_FD}, note="the field scaled by 3 eps/(2 eps + 1)"),
    Case("h2o_gb_field_norescale", _water, solvation=GB_WATER, efield=FIELD, rescale_fields=False,
         torch=False, tol={**TOL_ENV, **TOL_DFTBP_FD}),
    Case("oh_noncolinear_soc_gb", _oh, filling=("fermi", 300.0), noncolinear=NC_OH,
         spin_orbit={"xi": {"O": [0.0, 0.01]}}, solvation=GB_WATER, torch=False,
         tol={**TOL_PAULI, **TOL_ENV}),
    # ---- M6d: Casida excitations (DFTB+ ExcitedState { Casida }) ----------------------------
    Case("h2o_casida", _water, casida={"n": 6, "symmetry": "singlet", "solver": "Stratmann"},
         torch=False, tol={**TOL_CASIDA, **TOL_DFTBP_FD},
         note="singlets; the HOMO -> LUMO transitions do not couple (zero transition charges)"),
    Case("h2co_casida", _h2co, casida={"n": 8, "symmetry": "singlet", "solver": "Arpack"},
         torch=False, tol=TOL_CASIDA),
    Case("h2co_casida_triplet", _h2co,
         casida={"n": 6, "symmetry": "triplet", "solver": "Stratmann", "spin_constants": W_HCO},
         torch=False, tol=TOL_CASIDA),
    Case("c2h4_casida_both", _c2h4,
         casida={"n": 4, "symmetry": "both", "solver": "Stratmann", "spin_constants": W_HCO},
         torch=False, tol=TOL_CASIDA, note="triplets then singlets, as DFTB+ lists them"),
    Case("ch3oh_casida", _meoh, casida={"n": 12, "symmetry": "singlet", "solver": "Arpack"},
         torch=False, tol=TOL_CASIDA),
    Case("c6h6_casida", _c6h6, casida={"n": 8, "symmetry": "singlet", "solver": "Arpack"},
         torch=False, tol=TOL_CASIDA, note="degenerate excited states (ARPACK: DFTB+'s Stratmann "
                                           "gets their oscillator-strength sums to ~5e-7 only)"),
    Case("h2co_cation_casida", _h2co, charge=1.0,
         spin={"unpaired_electrons": 1, "spin_constants": W_HCO},
         casida={"n": 6, "symmetry": None, "solver": "Stratmann"}, torch=False, tol=TOL_CASIDA,
         note="spin polarised (doublet): both spin channels, D<S^2>"),
    Case("o2_triplet_casida", _o2, spin={"unpaired_electrons": 2, "spin_constants": {"O": -0.028}},
         casida={"n": 6, "symmetry": None, "solver": "Stratmann"}, torch=False, tol=TOL_CASIDA,
         note="spin polarised triplet ground state, degenerate excitations"),
    Case("h2co_casida_3000K", _h2co, filling=("fermi", 3000.0),
         casida={"n": 6, "symmetry": "singlet", "solver": "Stratmann"}, torch=False,
         tol=TOL_CASIDA, note="fractional occupations: transitions between partly filled levels"),
    Case("h2co_casida_onsite", _h2co, onsite=ONS,
         casida={"n": 6, "symmetry": "singlet", "solver": "Arpack"}, torch=False,
         tol={**TOL_BLOCK, **TOL_CASIDA},
         note="onsite corrections in the response (DFTB+: ARPACK only)"),
    Case("h2co_cation_casida_onsite", _h2co, charge=1.0, onsite=ONS,
         spin={"unpaired_electrons": 1, "spin_constants": W_HCO},
         casida={"n": 6, "symmetry": None, "solver": "Arpack"}, torch=False,
         tol={**TOL_BLOCK, **TOL_CASIDA},
         note="spin polarised with onsite corrections (f_uu for equal, f_ud for opposite spins)"),
    Case("h2co_casida_ewindow", _h2co,
         casida={"n": 4, "symmetry": "singlet", "solver": "Arpack", "energy_window": 2.0 / HARTREE},
         torch=False, tol=TOL_CASIDA, note="EnergyWindow 2 eV"),
    Case("h2co_casida_owindow", _h2co,
         casida={"n": 4, "symmetry": "singlet", "solver": "Arpack", "oscillator_window": 0.01},
         torch=False, tol=TOL_CASIDA, note="OscillatorWindow 0.01: only bright transitions"),
    Case("h2co_box_casida", _h2co_box, ewald_tolerance=1e-12,
         casida={"n": 6, "symmetry": "singlet", "solver": "Stratmann"}, torch=False,
         tol=TOL_CASIDA, note="Gamma point of an 8 A cell (Ewald gamma); DFTB+ gives no forces"),
    Case("cuco_cation_casida", _cuco_linear, charge=1.0,
         casida={"n": 6, "symmetry": "singlet", "solver": "Stratmann"}, torch=False,
         tol=TOL_CASIDA, note="Cu d levels; a linear molecule with degenerate states"),
    Case("h2co_casida_field_pc", _h2co, efield=FIELD, point_charges=PC3,
         casida={"n": 6, "symmetry": "singlet", "solver": "Arpack"}, torch=False,
         tol={**TOL_CASIDA, **TOL_ENV}),
    Case("h2co_casida_bright", _h2co,
         casida={"n": 6, "symmetry": "singlet", "solver": "Stratmann", "state": -1},
         torch=False, tol=TOL_CASIDA, note="StateOfInterest = -1: the brightest singlet"),
    Case("h2co_casida_state", _h2co,
         casida={"n": 6, "symmetry": "singlet", "solver": "Arpack", "state": 4}, torch=False,
         tol=TOL_CASIDA, note="StateOfInterest = 4 (ground-state forces, ExcitedStateForces = No)"),
    # ---- M6e: excited-state forces (Z vector), charges and dipole (WriteMulliken) -----------
    Case("h2co_exc_forces", _h2co, casida=EXC, torch=False, tol=TOL_EXC,
         note="second singlet: forces -d(E0 + w)/dR, XCH.DAT, XREST.DAT"),
    Case("h2co_exc_forces_triplet", _h2co,
         casida={**EXC, "symmetry": "triplet", "spin_constants": W_HCO}, torch=False, tol=TOL_EXC,
         note="triplet (W in the kernel; the Z vector uses the singlet A+B)"),
    Case("h2co_cation_exc_forces", _h2co, charge=1.0,
         spin={"unpaired_electrons": 1, "spin_constants": W_HCO},
         casida={**EXC, "symmetry": None}, torch=False, tol=TOL_EXC,
         note="spin polarised doublet"),
    Case("h2co_exc_forces_bright", _h2co, casida={**EXC, "solver": "Stratmann", "state": -1},
         torch=False, tol=TOL_EXC, note="the brightest singlet"),
    Case("ch3oh_exc_forces", _meoh, casida={**EXC, "n": 8, "state": 3}, torch=False, tol=TOL_EXC,
         note="lower symmetry, more transitions"),
    Case("h2co_exc_forces_lj", _h2co, dispersion={"model": "lennardjones"}, casida=EXC,
         torch=False, tol=TOL_EXC, note="dispersion (ground-state part only)"),
    Case("h2co_exc_forces_s1", _h2co, casida={**EXC, "n": 4, "state": 1}, torch=False,
         tol=TOL_EXC, note="first singlet (n -> pi*); the base of the S1 optimisation"),
    Case("h2co_exc_charges", _h2co,
         casida={**EXC, "state": 4, "excited_forces": False, "excited_charges": True},
         torch=False, tol=TOL_EXC,
         note="excited charges and dipole with ground-state forces (WriteMulliken only)"),
    # DFTB+'s excited forces are wrong here: forces_fd = DFTB+'s own E0 + w differentiated
    # (4-point, 1e-4 Bohr; Stratmann, whose energies are smoother than ARPACK's single-precision
    # ones); compat = what DFTB+ leaves out, to reproduce its forces
    Case("h2co_exc_forces_field_pc", _h2co, efield=FIELD, point_charges=PC3,
         casida={**EXC_FD, "compat": ["external"]}, scc_tol=1e-12, torch=False,
         tol={**TOL_EXC, **TOL_ENV, **TOL_EXC_FD},
         note="field and point charges: DFTB+ misses dq_ex grad V_ext and the field in its shift"),
    Case("h2co_exc_forces_hdamp", _h2co, dftb3={"h_damping": 4.0},
         casida={**EXC_FD, "compat": ["damping"]}, scc_tol=1e-12, torch=False,
         tol={**TOL_EXC, **TOL_EXC_FD},
         note="H-X damping: DFTB+'s gamma' in the excited gradient is undamped"),
    Case("h2co_exc_forces_ewindow", _h2co,
         casida={**EXC_FD, "n": 4, "energy_window": 2.0 / HARTREE, "compat": ["window"]},
         scc_tol=1e-12, torch=False, tol={**TOL_EXC, **TOL_EXC_FD},
         note="energy window: DFTB+ solves the Z vector in the window only (its excited "
              "charges too)"),
    # ---- M6f-1: SCC start and mixers (DFTB+ Mixer, InitialCharges, InitialSpins,
    # ReadInitialCharges), NumPy only.  No long mixer histories on systems with few SCC
    # sites: the history matrices of DIIS and Broyden become singular there (charge
    # conservation) and both codes follow round-off (OH + U: 30 % after 8 iterations) ------
    Case("h2o_mixer_simple", _water, mixer={"kind": "simple", "mixing": 0.3}, torch=False,
         tol={**TOL_SCC, **TOL_DFTBP_FD}, note="simple mixing, beta = 0.3"),
    Case("ch3oh_mixer_anderson", _meoh, mixer="anderson", torch=False, tol=TOL_SCC,
         note="DFTB+'s Anderson mixer with its defaults"),
    Case("cu4co_300K_mixer_anderson_dyn", _cu4co, filling=("fermi", 300.0),
         mixer={"kind": "anderson", "generations": 6,
                "dynamic_mixing": [(1e-2, 0.3), (1e-4, 0.6)]},
         torch=False, tol=TOL_SCC, note="Generations 6, mixing parameter chosen by |F|"),
    Case("water_box_mixer_diis", _water_box, mixer="diis", ewald_tolerance=1e-12, torch=False,
         tol=TOL_SCC, note="DIIS defaults; with DFTB+'s Ewald tolerance 1e-9 the SCC errors "
                           "differ by 1e-5"),
    Case("o2_triplet_shell_mixer_diis_late", _o2, shell_resolved=True,
         spin={"unpaired_electrons": 2, "spin_constants": W_SPIN},
         mixer={"kind": "diis", "use_from_start": False}, torch=False, tol=TOL_SCC,
         note="collinear spin; DIIS once its history is full (UseFromStart = No)"),
    Case("oh_noncolinear_mixer_anderson", _oh, filling=("fermi", 300.0), noncolinear=NC_OH,
         mixer="anderson", torch=False, tol={**TOL_SCC, **TOL_PAULI},
         note="the two-component (q, m) vector"),
    Case("cu4co_plusu_mixer_diis", _cu4co, filling=("fermi", 300.0), plus_u=U_CU, mixer="diis",
         torch=False, tol={**TOL_SCC, **TOL_BLOCK}, note="the block vector of DFTB+U"),
    Case("water_dimer_hdamp_mixer_anderson", _water_dimer, dftb3={"h_damping": 4.0},
         mixer={"kind": "anderson", "generations": 6, "mixing": 0.3}, torch=False, tol=TOL_SCC,
         note="the start of the MD case water_dimer_anderson_md (dynamics.py)"),
    Case("cuo_chain_k422_broyden_weights", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         mixer=BROYDEN_WEIGHTS, ewald_tolerance=1e-12, torch=False,
         tol={**TOL_SCC, "stress": 1e-9}, note="every Broyden option off its default"),
    Case("h2co_initial_charges", _h2co, initial_charges=[0.4, -0.4, 0.1, -0.1], torch=False,
         tol=TOL_SCC, note="AllAtomCharges: aufbau shell populations"),
    Case("ch3oh_atom_charge_blocks", _meoh, initial_charges={0: 0.3, 1: -0.1, 2: -0.1},
         torch=False, tol=TOL_SCC,
         note="AtomCharge blocks summing to 0.1, not the total charge: DFTB+ warns and scales "
              "the populations to the electron count"),
    Case("cu4co_shell_initial_charges", _cu4co, filling=("fermi", 300.0), shell_resolved=True,
         initial_charges=[0.2, -0.2, 0.1, -0.1, 0.3, -0.3], torch=False, tol=TOL_SCC,
         note="aufbau over the SKF shells s, p, d: Cu with 11.2 electrons starts as 4s2 4p6 3d3.2"),
    Case("o2_triplet_initial_spins", _o2,
         spin={"unpaired_electrons": 2, "spin_constants": {"O": -0.028}, "initial_spins": {0: 1.5}},
         torch=False, tol=TOL_SCC, note="AtomSpin on the first atom only"),
    Case("o2_triplet_shell_initial_spins", _o2, shell_resolved=True,
         spin={"unpaired_electrons": 2, "spin_constants": W_SPIN, "initial_spins": [0.5, 1.5]},
         torch=False, tol=TOL_SCC, note="spins shared over the shells by their populations"),
    Case("cuo_chain_spin_k422_afm_start", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         spin={"unpaired_electrons": 0, "spin_constants": W_SPIN,
               "initial_spins": [1.0, 0.0, -1.0, 0.0]},
         ewald_tolerance=1e-12, torch=False, tol={**TOL_SCC, "stress": 1e-9},
         note="antiferromagnetic start without net spin: the Cu moments stay at +-0.14"),
    Case("oh_radical_initial_charges_spins", _oh, filling=("fermi", 300.0),
         spin={"unpaired_electrons": 1, "spin_constants": W_SPIN, "initial_spins": [0.8, 0.2]},
         initial_charges=[-0.3, 0.3], torch=False, tol=TOL_SCC),
    # two runs: DFTB+ writes charges.bin (charges.dat) at displaced positions, then starts from it
    Case("h2o_read_charges", _water, read_charges={"text": False}, torch=False,
         tol={**TOL_SCC, **TOL_DFTBP_FD}),
    Case("h2o_read_charges_text", _water, read_charges={"text": True}, torch=False,
         tol={**TOL_SCC, **TOL_DFTBP_FD}, note="charges.dat (list-directed text)"),
    Case("cu4co_shell_read_charges", _cu4co, filling=("fermi", 300.0), shell_resolved=True,
         read_charges={"text": False}, torch=False, tol=TOL_SCC,
         note="shell populations, each on the first orbital of its shell"),
    Case("o2_triplet_read_charges", _o2,
         spin={"unpaired_electrons": 2, "spin_constants": {"O": -0.028}},
         read_charges={"text": False}, torch=False, tol=TOL_SCC, note="(q, m) records"),
    Case("oh_noncolinear_read_charges", _oh, filling=("fermi", 300.0), noncolinear=NC_OH,
         read_charges={"text": False}, torch=False, tol={**TOL_SCC, **TOL_PAULI},
         note="(q, mx, my, mz) records; run 2 has no InitialSpins (DFTB+ stops on them)"),
    Case("cu4co_plusu_read_charges", _cu4co, filling=("fermi", 300.0), plus_u=U_CU,
         read_charges={"text": False}, torch=False, tol={**TOL_SCC, **TOL_BLOCK},
         note="orbital populations and the population blocks"),
    Case("oh_noncolinear_soc_plusu_read_charges", _oh, filling=("fermi", 300.0),
         noncolinear=NC_OH, spin_orbit={"xi": {"O": [0.0, 0.01]}}, plus_u=U_O,
         read_charges={"text": False}, torch=False, tol={**TOL_SCC, **TOL_BLOCK},
         note="imaginary blocks (dual spin-orbit coupling with +U)"),
    Case("o2_triplet_plusu_read_charges_text", _o2,
         spin={"unpaired_electrons": 2, "spin_constants": W_SPIN}, plus_u=U_O,
         read_charges={"text": True}, torch=False, tol={**TOL_SCC, **TOL_BLOCK},
         note="(q, m) blocks in charges.dat"),
    Case("cuo_chain_spin_k422_read_charges", _cuo_chain, kgrid=(4, 2, 2),
         filling=("fermi", 500.0), spin={"unpaired_electrons": 2, "spin_constants": W_SPIN},
         read_charges={"text": False}, ewald_tolerance=1e-12, torch=False,
         tol={**TOL_SCC, "stress": 1e-9}, note="periodic with spin"),
    # --- occupations and overrides of the Hamiltonian block (milestone M6f-2), NumPy only ---
    # SCC tolerances such that DFTB+'s last two errors lie at least about 2x either side of
    # the threshold (exact numbers of iterations; round-off decides below about 1e-10).
    # Fixed Fermi levels: only systems whose SCC converges smoothly (M6f-2 design, section 2:
    # a half-filled level pinned at E_f or a dense periodic cell does not)
    Case("cu_fcc_k6_fixed_ef", _cu_fcc, kgrid=(6, 6, 6), filling=("fermi", 1000.0),
         fermi={"fixed": -0.10}, ewald_tolerance=1e-12, stress_fd=True, torch=False,
         tol={**TOL_FILL, "stress": 1e-9},
         note="the cell loses electrons (N < 11); stress against stress_fd (DFTB+'s is 3e-7 off)"),
    Case("cu4co_300K_fixed_ef", _cu4co, filling=("fermi", 300.0), fermi={"fixed": -0.140},
         scc_tol=1e-9, torch=False, tol={**TOL_FILL, "scc_errors": 1e-3},
         note="N = 54.03; SCC tolerance 1e-9: at 1e-10 the last iterations are round-off "
              "(DFTB+ 1.1e-10, pydftb 8e-11); the error 2e-7 of iteration 35 agrees to 4e-4 "
              "(8e-11 absolute) after a 38-iteration path"),
    Case("cu4co_box_fixed_ef", _cu4co_box, kgrid=(1, 1, 1), filling=("fermi", 300.0),
         fermi={"fixed": -0.140}, scc_tol=1e-9, ewald_tolerance=1e-12, stress_fd=True,
         torch=False,
         tol={**TOL_FILL, "stress": 1e-9}, note="a charged cell (N = 54.08): forces and stress"),
    Case("o2_triplet_fixed_ef_1000K", _o2, filling=("fermi", 1000.0),
         spin={"unpaired_electrons": 2, "spin_constants": W_SPIN}, fermi={"fixed": (-0.13, -0.24)},
         scc_tol=1e-7, torch=False, tol=TOL_FILL,
         note="two Fermi levels; the down one 0.0125 Ha below the empty pi*: N = (7, 5.01); "
              "below 1e-9 both codes wander in round-off"),
    Case("o2_noncolinear_soc_fixed_ef_1000K", _o2, filling=("fermi", 1000.0),
         noncolinear={"initial_spins": [[0.0, 0.0, 1.0]] * 2, "spin_constants": {"O": -0.028}},
         spin_orbit={"xi": {"O": [0.0, 0.01]}}, fermi={"fixed": -0.25}, torch=False,
         tol={**TOL_PAULI, **TOL_FILL}, note="two-component: one Fermi level for the spinors"),
    # every k-point filled on its own (IndependentKFilling; collinear spin crashes DFTB+ 25.1)
    Case("cuo_chain_k422_indep_k", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         fermi={"independent_k": True}, ewald_tolerance=1e-12, torch=False,
         tol={**TOL_FILL, "stress": 1e-9},
         note="the Fermi level is the k-weighted mean of the k-points' own"),
    Case("cu8_distorted_k355_mp2_2000K_indep_k", _cu8, kgrid=(3, 5, 5), filling=("mp", 2000.0, 2),
         fermi={"independent_k": True}, ewald_tolerance=1e-12, torch=False,
         tol={**TOL_FILL, "stress": 1e-8},
         note="Methfessel-Paxton: forces 5e-7 and stress 3e-9 from DFTB+, as in "
              "cu8_distorted_k355_mp2_2000K"),
    Case("cuo_chain_noncolinear_k422_indep_k", _cuo_chain, kgrid=(4, 2, 2), time_reversal=False,
         filling=("fermi", 500.0),
         noncolinear={"initial_spins": [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]] * 2,
                      "spin_constants": W_SPIN},
         fermi={"independent_k": True}, scc_tol=1e-9, ewald_tolerance=1e-12, torch=False,
         tol={**TOL_PAULI, **TOL_FILL, "stress": 1e-9}),
    Case("cuo_chain_soc_k422_indep_k", _cuo_chain, kgrid=(4, 2, 2), time_reversal=False,
         filling=("fermi", 500.0), spin_orbit={"xi": XI_CUO}, fermi={"independent_k": True},
         ewald_tolerance=1e-12, torch=False, tol={**TOL_PAULI, **TOL_FILL, "stress": 1e-9}),
    # custom Hubbard U (CustomisedHubbards; <= 0 keeps the SKF's): no two shells of an atom
    # with the same U, which DFTB+ would mix as one class
    Case("ch3oh_custom_hubbards", _meoh, hubbard={"O": 0.5, "H": -1.0}, torch=False,
         tol=TOL_FILL, note="H's -1 keeps the SKF value"),
    Case("cu4co_shell_custom_hubbards", _cu4co, filling=("fermi", 300.0), shell_resolved=True,
         hubbard={"Cu": [0.3, 0.0, 0.4], "C": [0.40, 0.0], "O": [0.0, 0.62]}, scc_tol=1e-9,
         torch=False, tol=TOL_FILL, note="per shell; the zeros keep the SKF values"),
    Case("ch3oh_dftb3_custom_hubbards", _meoh, dftb3=DFTB3_3OB, hubbard={"O": 0.5, "C": 0.35},
         torch=False, tol=TOL_FILL, note="the third order (and the H-X damping) use the new U"),
    Case("cuo_chain_k422_custom_hubbards", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         hubbard={"Cu": 0.3}, ewald_tolerance=1e-12, torch=False,
         tol={**TOL_FILL, "stress": 1e-9}),
    # custom reference occupations (CustomisedOccupations, "virtual doping")
    Case("ch3oh_custom_occupations_300K", _meoh, filling=("fermi", 300.0),
         occupations={1: {"p": 4.2}}, torch=False, tol=TOL_FILL,
         note="O p = 4.2: 14.2 electrons, charges relative to the custom q0 (at T = 0 the "
              "fractional HOMO leaves the SCC in round-off)"),
    Case("ch3oh_custom_occupations_c", _meoh, occupations={0: {"s": 1.5, "p": 2.5}}, torch=False,
         tol=TOL_FILL, note="C 2s1.5 2p2.5: the same electron count, another reference"),
    Case("cu4co_plusu_custom_occ", _cu4co, filling=("fermi", 300.0), plus_u=U_CU,
         occupations={0: {"d": 9.8}}, torch=False, tol={**TOL_BLOCK, **TOL_FILL},
         note="the start of the block vector"),
    Case("ch3oh_ons_custom_occ", _meoh, onsite=ONS, occupations={1: {"s": 1.8, "p": 4.2}},
         scc_tol=1e-9, torch=False, tol={**TOL_BLOCK, **TOL_FILL},
         note="onsite corrections act on q - q0"),
    Case("cuo_chain_spin_k422_custom_occ", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         spin={"unpaired_electrons": 2, "spin_constants": W_SPIN}, occupations={0: {"d": 9.9}},
         scc_tol=5e-9, ewald_tolerance=1e-12, torch=False, tol={**TOL_FILL, "stress": 1e-9}),
    # truncated SK tables (TruncateSKRange; the repulsives are not cut)
    Case("water_dimer_truncated_sk", _water_dimer, truncate_sk={"max_distance": 6.0},
         scc_tol=1e-9, torch=False, tol=TOL_FILL,
         note="the intermolecular pairs lie in the new tail"),
    Case("water_box_truncated_sk_soft", _water_box,
         truncate_sk={"max_distance": 6.0, "hard_cutoff": False}, scc_tol=1e-9,
         ewald_tolerance=1e-12, torch=False, tol={**TOL_FILL, "stress": 1e-9}),
    # ---- M6f-3: corrections (DFTB+ HCorrection = H5, HalogenXCorr, DipoleCorrection), NumPy
    # only.  Acceptors other than O and the halogens are type copies of O (DFTB+ identifies them
    # by species name, so its code runs exactly as for the real elements: test values only).
    # The periodic halogen correction (25.1 refuses it) and the dipole correction (after 25.1)
    # have references from the development build 041e723 (dftbplus="dev") ---------------------
    Case("water_dimer_h5", _water_dimer, h5=True, torch=False, tol=TOL_CORR,
         note="H5 with the paper's values (DFTB2): gamma of the O-H pairs times 1 + g(r)"),
    Case("water_dimer_h5_typed_n", _water_dimer, typed={"O": "O", "N": "O", "H": "H"},
         labels=_relabel(a3="N"), h5=True, torch=False, tol=TOL_CORR,
         note="second water's O typed N (a copy of O): N's default scaling 0.18 and radius"),
    Case("water_dimer_h5_typed_s_custom", _water_dimer, typed={"O": "O", "S": "O", "H": "H"},
         labels=_relabel(a3="S"), torch=False, tol=TOL_CORR,
         h5={"r_scaling": 0.8, "w_scaling": 0.3, "scaling": {"O": 0.1, "S": 0.3}},
         note="second water's O typed S; RScaling, WScaling and H5Scaling set"),
    Case("water_dimer_dftb3_d3h5", _water_dimer, h5=True, torch=False, tol=TOL_CORR,
         dftb3={"third_order": "full", "hubbard_derivs": UD_3OB},
         dispersion={"model": "dftd3", "damping": "zerodamping", "sr6": 1.25, "alpha6": 29.61,
                     "s6": 1.0, "s8": 0.49, "hhrepulsion": True},
         note="DFTB3 + H5 + D3 with the H-H repulsion: the published DFTB3-D3H5 model"),
    Case("ch3oh_h5_cscale", _meoh, h5={"scaling": {"C": 0.1}}, torch=False, tol=TOL_CORR,
         note="H5Scaling {C = 0.1}: the C-H pairs too (and O-H inside the molecule)"),
    Case("water_box_h5", _water_box, h5=True, ewald_tolerance=1e-12, scc_tol=4e-10, torch=False,
         tol={**TOL_CORR, "stress": 1e-9}, note="H5 in the short-range part of Ewald; stress"),
    Case("water_dimer_h5_exc_forces", _water_dimer, h5=True,
         casida={**EXC_FD, "n": 6, "state": 1, "compat": ["damping"]}, scc_tol=1e-12,
         torch=False, tol={**TOL_CORR, **TOL_EXC, **TOL_EXC_FD},
         note="S1 forces: DFTB+'s gamma' in the excited gradient has no H5 (the excitation "
              "energies do)"),
    Case("meoh_water_halogen", lambda: _meoh_water(3.0), labels=_relabel(a1="Cl"),
         typed={"C": "C", "H": "H", "O": "O", "Cl": "O"}, halogen={},
         dftb3={"third_order": "full", "hubbard_derivs": UD_HALOGEN, "h_damping": 4.0},
         scc_tol=1e-9, torch=False, tol=TOL_CORR,
         note="CH3-Cl-H (Cl a copy of O) and a water, r(O-Cl) = 3.0 A: sigma = 1"),
    Case("meoh_water_halogen_switch", lambda: _meoh_water(2.45), labels=_relabel(a1="Cl"),
         typed={"C": "C", "H": "H", "O": "O", "Cl": "O"},
         halogen={"compat": True, "forces_fd": True},
         dftb3={"third_order": "full", "hubbard_derivs": UD_HALOGEN, "h_damping": 4.0},
         scc_tol=1e-12, torch=False, tol={**TOL_CORR, "forces": 1e-8, "forces_dftbp": 1e-6},
         note="r(O-Cl) = 2.45 A, in the switching region (2.289-2.616 A): DFTB+'s forces are "
              "wrong there; forces against differences of its energy"),
    Case("meoh_water_halogen_nbr_h5", lambda: _meoh_water(2.9), labels=_relabel(a1="Br", a6="N"),
         typed={"C": "C", "H": "H", "Br": "O", "N": "O"}, halogen={}, h5=True,
         dftb3={"third_order": "onsite", "hubbard_derivs": UD_HALOGEN},
         scc_tol=2e-10, torch=False, tol=TOL_CORR,
         note="N-Br pair (copies of O), on-site third order, H5 on N-H"),
    Case("water_box_halogen", _water_box, labels=_relabel(a0="Cl", a12="Cl", a3="N"),
         typed={"O": "O", "Cl": "O", "N": "O", "H": "H"}, halogen={}, ewald_tolerance=1e-12,
         dftb3={"third_order": "full", "hubbard_derivs": UD_HALOGEN, "h_damping": 4.0},
         dftbplus="dev", torch=False, tol={**TOL_CORR, "stress": 1e-9},
         note="periodic halogen correction (development DFTB+); no pair in a switching region"),
    Case("water_slab_dipole", _water_layer, kgrid=(2, 2, 1), dipole_correction=True,
         ewald_tolerance=1e-12, scc_tol=1e-9, dftbplus="dev", torch=False,
         stress_fd=((0, 0), (1, 1), (2, 2), (0, 1)), tol={**TOL_CORR, "stress": 1e-9},
         note="dipole correction (development DFTB+), layer at -h/2; DFTB+ has no stress term "
              "(xx, yy, zz, xy against its energy differences; it refuses z shears)"),
    Case("water_slab_dipole_layer", _water_layer_folded, kgrid=(2, 2, 1),
         dipole_correction={"layer_position": 5.0}, ewald_tolerance=1e-12, dftbplus="dev",
         stress_fd=((0, 0), (1, 1), (2, 2), (0, 1)), torch=False,
         tol={**TOL_CORR, "stress": 1e-9},
         note="layer at 5 A, two molecules given one period lower (folded)"),
    # ---- M5a: smooth type switching in the discrete limit (every weight exactly 0 or 1): DFTB+
    # runs the fixed type labels the switch gives; pydftb also runs the switch with the element
    # labels, bitwise the same (NumPy only) -----------------------------------------------------
    Case("cu_slab_types_k441", _cu_slab111, skf_set="cu_types", kgrid=(4, 4, 1),
         filling=("fermi", 500.0), labels=_switch_labels(SWITCH_SLAB),
         type_switching=SWITCH_SLAB, torch=False, tol=TOL_TYPES,
         note="Cu(111) slab: surface atoms (CN 9) Cu2, inner atoms (CN 12) Cu; stress"),
    Case("cu13_types", _cu13, skf_set="cu_types", filling=("fermi", 300.0),
         labels=_switch_labels(SWITCH_CLUSTER), type_switching=SWITCH_CLUSTER, torch=False,
         tol=TOL_TYPES, note="cuboctahedron: the 12 outer atoms (CN 5) Cu2, the centre (CN 12) Cu"),
]
CASES_BY_NAME = {c.name: c for c in CASES}
