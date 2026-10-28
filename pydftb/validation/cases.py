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
from ..kpoints import monkhorst_pack, reduce_time_reversal, supercell_folding

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "data")
SKF_SETS = {
    "ptbp_hcocu": os.path.join(DATA, "skf", "ptbp_hcocu"),
    "orbit": os.path.join(DATA, "skf", "orbit"),
    "orbit_poly": os.path.join(DATA, "skf", "orbit_poly"),   # make_poly_set.py
    "hotcent_gdo": os.path.join(DATA, "skf", "hotcent_gdo"), # make_f_set.py (f orbitals)
    "cu_types": os.path.join(DATA, "skf", "cu_types"),       # make_cu_types_set.py (M5a)
    # make_hybrid_sets.py (M6h-1): ptbp_hcocu with a RangeSep section
    "ptbp_hcocu_lc": os.path.join(DATA, "skf", "ptbp_hcocu_lc"),
    "ptbp_hcocu_cam": os.path.join(DATA, "skf", "ptbp_hcocu_cam"),
    "ptbp_hcocu_global": os.path.join(DATA, "skf", "ptbp_hcocu_global"),
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
    # milestone M6f-4a: analysis outputs; {"atom_energies", "net_populations", "cm5",
    # "esp": {"points" (A)} | {"grid": {"spacing" (A), "origin" (A), "points"}},
    # "eigenvectors", "band_out" (compare band.out), "compat" (DFTB+'s E/N for D3)}
    analysis: dict | None = None
    # milestone M5a: smooth type switching in the discrete limit, the fields of a TypeSwitch
    # ({"element", "types", "centre", "width", "cutoff"}): the labels are the fixed types the
    # switch gives (weights 0 or 1), DFTB+ runs them; pydftb also runs the switch, which must
    # give bitwise the explicit labels' results ("switch" deviation)
    type_switching: dict | None = None
    # milestone M6f-5: DFTB+ SelectedShells {label: [(file label, shell letters), ...]}, a label's
    # shells taken from several SK files; the file labels' files come from ``typed``
    # (write_typed_copies), DFTB+ gets them listed pair by pair (SlaterKosterFiles)
    selected_shells: dict | None = None
    # milestone M6g-2: ``fd_delta``: pydftb's H0/S derivatives as DFTB+'s central differences
    # with this step (Bohr; DFTB+'s default FiniteDiff is 2^-13) for the coordinate response
    # and the Hessian; ``hessian``: a second DFTB+ run with Driver = SecondDerivatives
    # {"delta", "atoms", "moved_atoms" (0-based), "polarisability": bornderiv.out,
    # "modes": DFTB+'s modes on its Hessian ({"remove": RemoveTranslation/Rotation})}, compared
    # with pydftb's second_derivatives (and normal_modes); with the analysis
    # "coord_derivatives" also pydftb's analytic Born charges against DFTB+'s differences
    fd_delta: float | None = None
    hessian: dict | None = None
    # milestone M6h-1: a hybrid functional, pydftb's ``hybrid`` option ("lc", "cam" or {"type",
    # "screening", "threshold", "cutoff_reduction" (Bohr)}; M6h-3a, periodic: "coulomb",
    # "coulomb_cutoff", "gsum_cutoff" (Bohr)); the SK set must have RangeSep
    hybrid: str | dict | None = None
    # milestone M6h-1b: the reference forces are 4-point differences of DFTB+'s energy (where its
    # analytic forces are not that derivative); pydftb's DFTB+-compatible forces are compared with
    # DFTB+'s own (forces_dftbp)
    forces_fd: bool = False
    # milestone M6h-3a: pydftb takes DFTB+ 25.1's input populations of a periodic Gamma-point
    # hybrid (the lower-triangular DeltaP_in's dense Mulliken, DFTB._dftbp_gamma_q_in), which in
    # a low-symmetry cell are not DeltaP_in's Mulliken populations
    gamma_q_in_compat: bool = False
    # milestone M6h-3b: k-points of the supercell folding diag(folding) without shift: pydftb
    # takes DFTB+'s set (kpoints.supercell_folding), DFTB+ the SupercellFolding (and with the
    # neighbour screening Threshold = dftbplus.KPOINT_NEIGHBOUR_THRESHOLD)
    folding: tuple | None = None
    # milestone M6h-3b: pydftb keeps the starting block populations of +U and onsite corrections
    # as DFTB+ 25.1 at k-points (DFTB._dftbp_stale_blocks, B20)
    stale_blocks_compat: bool = False
    # milestone M6h-3b: both codes stop after this many SCC iterations unconverged (DFTB+
    # ConvergentSCCOnly = No; the untruncated kernel, whose SCC does not settle)
    fixed_scc: int | None = None

    @property
    def dftbp_forces(self) -> bool:
        """DFTB+ computes forces: not for k-point hybrids with another Coulomb matrix than
        Truncated or with spin, which DFTB+ 25.1 refuses (M6h-3b; pydftb's are tested against
        differences of its energy)."""
        if self.folding is None or self.hybrid is None:
            return True
        coulomb = (self.hybrid.get("coulomb", "truncated") if isinstance(self.hybrid, dict)
                   else "truncated")
        return coulomb == "truncated" and not self.spin

    @property
    def pydftb_forces(self) -> bool:
        """pydftb computes forces: not for the NeighbourBased MinimumImage at k-points, whose
        exchange matrix is not the derivative of its energy (M6h-3c, B22)."""
        h = self.hybrid if isinstance(self.hybrid, dict) else {}
        return not (self.folding is not None and h.get("coulomb") == "minimum_image"
                    and h.get("screening") == "neighbour")

    @property
    def screened_history(self) -> bool:
        """DFTB+'s Threshold of the neighbour screening at k-points (M6h-3c): the SCC's exchange
        accumulates screened increments, so a build from q_in alone (pydftb's eigenstates,
        DFTB+'s band run) is not the SCC's own."""
        h = self.hybrid if isinstance(self.hybrid, dict) else {}
        return self.folding is not None and h.get("neighbour_threshold") is not None

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
        return (self.scc_start or self.overrides or self.corrections or self.analysis is not None
                or self.hybrid is not None)

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
        if self.folding is not None:                       # M6h-3b
            return supercell_folding(self.folding)
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
PC_BOX = {"positions": (np.array(PC3["positions"]) + 2.0).tolist(), "charges": PC3["charges"],
          "blur": 0.3}
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


def _co2():
    return molecule("CO2")


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
# M6f-4a: analysis outputs.  detailed.out prints the atom-resolved energies and net populations
# with 8 decimals (1e-8); CM5 charges (results.tag) and the potentials (autotest.tag) have full
# precision; eigenvec.out's Mulliken weights 6 decimals (compared summed over degenerate
# levels, per level); band.out 4 decimals (eV)
TOL_ANALYSIS = {"energy": 1e-6, "forces": 1e-6, "charges": 1e-6, "spins": 1e-6, "n_scc": 0,
                "scc_errors": 1e-4}
TOL_ANALYSIS_ONLY = {"atom_energies": 1e-8, "net_populations": 1e-8,
                "cm5": 1e-9, "esp": 1e-9, "esp_file": 1e-9, "eigvec": 6e-7,
                "band_out": 1.01e-4, "band_occ": 1.01e-5, "detailed": 0}
TOL_ANALYSIS = {**TOL_ANALYSIS, **TOL_ANALYSIS_ONLY}
ESP3 = [[0.0, 0.0, 2.5], [1.0, -1.5, 0.3], [-2.0, 0.5, -1.0]]           # Angstrom
ATOM_E = {"atom_energies": True}
# M6f-4b: the Pipek-Mezey functional (autotest.tag and the final values printed with 12
# digits; the initial ones with 8, E15.8, compared relative), the localised orbitals' populations
# (the acos angles fix the orbitals to ~1e-8 only), the bond matrices (results.tag; the
# files with 8 digits) and the transition charges (q^2 in autotest.tag, ATQ.DAT 9 decimals)
TOL_LOC = {"pm_value": 1e-9, "pm_final": 1e-9, "pm_initial": 1e-7, "pm_populations": 1e-7,
           "pm_file": 0}
TOL_BONDS = {"bonds": 1e-9, "bonds_file": 5e-8}
TOL_TRANSQ = {"transq": 1e-9, "atq_file": 1e-9}
PM_TIGHT = {"tolerance": 1e-10, "max_iterations": 10000}
BONDS = {"bond_populations": True, "bond_energies": True, "bond_orders": True}
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
    Case("water_box_pc", _water_box, point_charges=PC_BOX,
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
# ---- M6f-5: shells from several SK files (DFTB+ SelectedShells), NumPy only ----------------------
# Cu {"sd" "p"}: the p group from Cu2, a copy of Cu (the same basis reordered: s d p); an O with an
# extra s shell taken from H's files (Ox, a copy of H) whose reference occupation is set to 0 (s2);
# H with a second s shell from Cu's files (Hy).  All files of the set share one grid (DFTB+'s rule).
SEL_CU = {"Cu": [("Cu", "sd"), ("Cu2", "p")]}
SEL_O = {"O": [("O", "sp"), ("Ox", "s")]}
SEL_OH = {**SEL_O, "H": [("H", "s"), ("Hy", "s")]}
TYPED_CU = {"Cu": "Cu", "Cu2": "Cu", "C": "C", "O": "O"}
TYPED_OX = {"O": "O", "Ox": "H", "H": "H"}
W_O3 = [[-0.035, -0.030, -0.010], [-0.030, -0.028, -0.020], [-0.010, -0.020, -0.015]]
# water's forces: DFTB+ differentiates H0 and S numerically (up to 2.7e-6 Ha/Bohr here, as h2o's
# 1.2e-6); with its step pydftb reproduces them to 1e-9 (test_dftbplus_finite_difference_emulation)
TOL_SEL_FD = {"energy": 1e-6, "forces": 5e-6, "charges": 1e-6}
TOL_SEL_OCC = {**TOL_FILL, **TOL_SEL_FD}        # with CustomisedOccupations (as the M6f-2 cases)
CASES += [
    Case("cu4co_selected_sdp", _cu4co, filling=("fermi", 300.0), typed=TYPED_CU,
         selected_shells=SEL_CU, torch=False,
         note="Cu {sd p}: the p group from Cu2 (a copy), the same as cu4co_300K reordered"),
    Case("cu_fcc_k6_selected_sdp", _cu_fcc, kgrid=(6, 6, 6), filling=("fermi", 1000.0),
         typed={"Cu": "Cu", "Cu2": "Cu"}, selected_shells=SEL_CU, torch=False,
         note="periodic, k-points and stress with Cu {sd p}"),
    Case("h2o_selected_s2", _water, typed=TYPED_OX, selected_shells=SEL_O,
         occupations={0: {"s2": 0.0}}, analysis={"eigenvectors": True}, torch=False,
         tol={**TOL_ANALYSIS, **TOL_SEL_OCC},
         note="O {sp s}: s* from H's files, reference occupation 0 (s2); eigenvec.out labels s2"),
    Case("h2o_selected_s2_shell_resolved", _water, typed=TYPED_OX, selected_shells=SEL_O,
         occupations={0: {"s2": 0.0}}, shell_resolved=True, torch=False, tol=TOL_SEL_OCC,
         note="shell-resolved SCC: the s* shell has H's U"),
    Case("h2o_selected_s2_noocc", _water, typed=TYPED_OX, selected_shells=SEL_O, torch=False,
         filling=("fermi", 300.0), tol=TOL_SEL_FD,
         note="the s* shell keeps H's occupation (1 electron) from its group's file: 9 electrons, "
              "300 K (at 0 K the half-filled level makes the 3-site Broyden history singular, "
              "and the two codes leave it differently; M6f-1)"),
    Case("h2o_selected_both", _water, typed={**TYPED_OX, "Hy": "Cu"}, selected_shells=SEL_OH,
         occupations={0: {"s2": 0.0}, 1: {"s2": 0.0}, 2: {"s2": 0.0}}, shell_resolved=True,
         torch=False, tol=TOL_SEL_OCC,
         note="O {sp s} and H {s s}: 2 x 2 files per O-H pair (second label fastest)"),
    Case("o2_triplet_selected", _o2, typed={"O": "O", "Ox": "H"}, selected_shells=SEL_O,
         occupations={0: {"s2": 0.0}, 1: {"s2": 0.0}},
         spin={"unpaired_electrons": 2.0, "spin_constants": {"O": W_O3}}, torch=False,
         tol=TOL_FILL,
         note="collinear spin with a 3 x 3 spin-constant matrix (shells s p s2)"),
    Case("h2o_selected_dftb3", _water, typed=TYPED_OX, selected_shells=SEL_O,
         occupations={0: {"s2": 0.0}}, shell_resolved=True,
         dftb3={"third_order": "full", "hubbard_derivs": {"O": [-0.1575, -0.1575, -0.05],
                                                          "H": [-0.1857]}, "h_damping": 4.05},
         torch=False, tol=TOL_SEL_OCC, note="shell-resolved DFTB3: one Hubbard derivative per shell"),
]
CASES += [
    # --- analysis outputs (milestone M6f-4a), NumPy only ---
    Case("ch3oh_analysis", _meoh, torch=False, tol=TOL_ANALYSIS,
         analysis={"atom_energies": True, "net_populations": True, "cm5": True,
                   "esp": {"points": ESP3}, "eigenvectors": True, "band_out": True},
         note="every M6f-4a output of a molecule"),
    Case("water_dimer_dftb3_d4_h5_atom", _water_dimer, h5=True, dispersion=D4_3OB,
         dftb3={"third_order": "full", "hubbard_derivs": UD_3OB}, torch=False,
         tol=TOL_ANALYSIS, analysis=ATOM_E,
         note="full third order, D4 (three-body terms a third to each atom), H5"),
    Case("ch3oh_d3_atom", _meoh, dispersion=D3BJ_3OB, torch=False, tol=TOL_ANALYSIS,
         analysis={"atom_energies": True, "compat": True},
         note="DFTB+ spreads the D3 library's energy evenly (E/N): compared with the compat "
              "switch; pydftb resolves it by atom pairs"),
    Case("water_dimer_sk_atom", _water_dimer,
         dispersion=lambda labels: sk_per_atom(labels, skip=(3,)), torch=False,
         tol=TOL_ANALYSIS, analysis=ATOM_E),
    Case("ch3oh_simple_d3_atom", _meoh,
         dispersion={"model": "simpledftd3", "s8": 0.5883, "a1": 0.5719, "a2": 3.6017},
         torch=False, tol=TOL_ANALYSIS, analysis=ATOM_E),
    Case("water_box_uff_atom", _water_box, dispersion={"model": "lennardjones"},
         ewald_tolerance=1e-12, scc_tol=1.7e-10, torch=False, tol=TOL_ANALYSIS, analysis=ATOM_E,
         note="the r^-6 Ewald sum per atom (real-space pairs, reciprocal and self terms)"),
    Case("water_box_d4_atom", _water_box, dispersion=D4_3OB_2BODY_Q0, ewald_tolerance=1e-12,
         scc_tol=1.7e-10, torch=False, tol=TOL_ANALYSIS, analysis=ATOM_E),
    Case("o2_triplet_field_atom", _o2,
         spin={"unpaired_electrons": 2, "spin_constants": {"O": -0.028}}, efield=FIELD,
         torch=False, tol={**TOL_ANALYSIS, **TOL_ENV},
         analysis={"atom_energies": True, "eigenvectors": True, "band_out": True}),
    Case("h2o_pc_esp", _water, point_charges=PC3, torch=False,
         tol={**TOL_ANALYSIS, **TOL_ENV, **TOL_DFTBP_FD},
         analysis={"atom_energies": True, "esp": {"points": ESP3}},
         note="the potential's external column (point charges)"),
    Case("h2o_site_atom", _water, site_potentials=SITES, torch=False,
         tol={**TOL_ANALYSIS, **TOL_ENV, **TOL_DFTBP_FD},
         analysis={"atom_energies": True, "net_populations": True}),
    Case("h2o_onsite3rd_atom", _water,
         dftb3={"third_order": "onsite", "hubbard_derivs": UD_3OB}, torch=False,
         tol={**TOL_ANALYSIS, **TOL_DFTBP_FD}, analysis=ATOM_E,
         note="on-site third order (DFTB+ books it in E_SCC, pydftb in e_3rd)"),
    Case("cu4co_plusu_atom", _cu4co, filling=("fermi", 300.0), plus_u=U_CU, torch=False,
         tol={**TOL_ANALYSIS, **TOL_BLOCK}, analysis={"atom_energies": True, "cm5": True}),
    Case("ch3oh_ons_atom", _meoh, onsite=ONS, scc_tol=2.5e-10, torch=False,
         tol={**TOL_ANALYSIS, **TOL_BLOCK}, analysis=ATOM_E),
    Case("oh_noncolinear_soc_atom", _oh, filling=("fermi", 300.0), noncolinear=NC_OH,
         spin_orbit={"xi": {"O": [0.0, 0.01]}}, scc_tol=3e-10, torch=False,
         tol={**TOL_ANALYSIS, **TOL_PAULI},
         analysis={"atom_energies": True, "eigenvectors": True, "band_out": True},
         note="dual spin-orbit energy per atom; two-component eigenvec.out"),
    Case("h2o_gbsa_atom", _water, solvation=GBSA_WATER, torch=False,
         tol={**TOL_ANALYSIS, **TOL_ENV, **TOL_DFTBP_FD},
         analysis={"atom_energies": True, "cm5": True}),
    Case("meoh_water_halogen_atom", lambda: _meoh_water(3.0), labels=_relabel(a1="Cl"),
         typed={"C": "C", "H": "H", "O": "O", "Cl": "O"}, halogen={},
         dftb3={"third_order": "full", "hubbard_derivs": UD_HALOGEN, "h_damping": 4.0},
         scc_tol=1e-9, torch=False, tol={**TOL_ANALYSIS, **TOL_CORR}, analysis=ATOM_E,
         note="the halogen correction, half of each pair, only in the total energies"),
    Case("water_box_analysis", _water_box, ewald_tolerance=1e-12, scc_tol=1.7e-10, torch=False,
         tol=TOL_ANALYSIS,
         analysis={"atom_energies": True, "net_populations": True, "cm5": True,
                   "esp": {"grid": {"spacing": [1.5, 1.5, 1.5], "origin": [0.0, 0.0, 0.0],
                                    "points": [3, 2, 2]}}, "eigenvectors": True},
         note="periodic: Ewald SCC per atom, CM5 with images, a potential grid"),
    Case("cuo_chain_k422_analysis", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         ewald_tolerance=1e-12, torch=False, tol=TOL_ANALYSIS,
         analysis={"atom_energies": True, "net_populations": True, "cm5": True,
                   "eigenvectors": True, "band_out": True},
         note="k-points: complex eigenvec.out, band.out"),
    Case("water_slab_dipole_analysis", _water_layer, kgrid=(2, 2, 1), dipole_correction=True,
         ewald_tolerance=1e-12, scc_tol=1e-9, dftbplus="dev", torch=False,
         stress_fd=((0, 0), (1, 1), (2, 2), (0, 1)), tol={**TOL_ANALYSIS, **TOL_CORR,
                                                          "stress": 1e-9},
         analysis={"atom_energies": True, "esp": {"points": [[1.0, 2.0, 3.0], [2.0, 1.0, 12.0],
                                                             [0.5, 0.5, 18.0]]}},
         note="the dipole correction in the SCC energies and the potential (development "
              "DFTB+)"),
    # --- Pipek-Mezey, bond analysis, transition charges (milestone M6f-4b), NumPy only ---
    Case("ch3oh_pm", _meoh, torch=False, tol={**TOL_ANALYSIS, **TOL_LOC},
         analysis={"localise": PM_TIGHT, "eigenvectors": True},
         note="Pipek-Mezey (DFTB+'s dense algorithm), localOrbs.out"),
    Case("h2co_cation_pm", _h2co, charge=1.0, spin={"unpaired_electrons": 1,
                                                    "spin_constants": W_HCO},
         torch=False, tol={**TOL_ANALYSIS, **TOL_LOC}, analysis={"localise": PM_TIGHT},
         note="spin polarised: each channel localised (the tag holds the second); O2's "
              "degenerate pi orbitals make the sweep count a matter of rounding"),
    Case("water_box_pm", _water_box, ewald_tolerance=1e-12, scc_tol=1.7e-10, torch=False,
         tol={**TOL_ANALYSIS, **TOL_LOC}, analysis={"localise": PM_TIGHT},
         note="periodic, Gamma point (DFTB+ always uses the dense algorithm there)"),
    Case("ch3oh_bonds", _meoh, dftbplus="dev", torch=False, tol={**TOL_ANALYSIS, **TOL_BONDS},
         analysis=BONDS, note="bond populations, energies, Mayer orders (development DFTB+)"),
    Case("o2_triplet_bonds", _o2, spin={"unpaired_electrons": 2, "spin_constants": {"O": -0.028}},
         dftbplus="dev", torch=False, tol={**TOL_ANALYSIS, **TOL_BONDS}, analysis=BONDS,
         note="spin polarised: the spin-summed density (development DFTB+)"),
    Case("cuo_chain_k422_bonds", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         ewald_tolerance=1e-12, dftbplus="dev", torch=False, tol={**TOL_ANALYSIS, **TOL_BONDS},
         analysis={"bond_populations": True, "bond_energies": True},
         note="k-points (no Mayer orders for periodic systems, as DFTB+)"),
    Case("cu_fcc_k6_bonds", _cu_fcc, kgrid=(6, 6, 6), filling=("fermi", 1000.0),
         ewald_tolerance=1e-12, dftbplus="dev", torch=False, tol={**TOL_ANALYSIS, **TOL_BONDS},
         analysis={"bond_populations": True, "bond_energies": True},
         note="one atom per cell: every bond is with its own images (B = 11 electrons)"),
    Case("h2co_exc_transq", _h2co,
         casida={"n": 6, "symmetry": "singlet", "solver": "Arpack", "state": 2,
                 "transition_charges": True}, torch=False, tol={**TOL_CASIDA, **TOL_TRANSQ},
         note="transition charges of the second singlet (ATQ.DAT, transq_sqr)"),
    Case("h2co_cation_exc_transq", _h2co, charge=1.0,
         spin={"unpaired_electrons": 1, "spin_constants": W_HCO},
         casida={"n": 6, "symmetry": None, "solver": "Stratmann", "state": 2,
                 "transition_charges": True}, torch=False, tol={**TOL_CASIDA, **TOL_TRANSQ},
         note="spin polarised: kappa = 1, both channels' transitions"),
    Case("h2co_casida_3000K_transq", _h2co, filling=("fermi", 3000.0),
         casida={"n": 6, "symmetry": "singlet", "solver": "Stratmann", "state": 2,
                 "transition_charges": True}, torch=False, tol={**TOL_CASIDA, **TOL_TRANSQ},
         note="fractional occupations: the charges carry no sqrt(f_i - f_a), unlike the dipole"),
]
# ---- M6g-1: polarisability and response kernel (DFTB+ Analysis { Polarisability,
# ResponseKernel }, the response converged to PerturbSccTol = 1e-11), NumPy only.  The tags of
# results.tag: alpha of every frequency, the charge and eigenvalue derivatives d/dE of the last
# one, N_F and dE_F/dE of metals, the kernel's dq/dV, net populations and eigenvalue
# derivatives; "response_text": the detailed.out blocks and dE_band.out / dV_band.out
TOL_RESPONSE = {"pol": 1e-8, "dqdE": 1e-8, "dedE": 1e-8, "dEf": 1e-9, "dos": 1e-8,
                "dqdV": 1e-9, "dqnet": 1e-9, "dedV": 1e-9, "response_text": 1e-4,
                "dqdE_default": 1e-4}
TOL_RESP = {"energy": 1e-6, "forces": 2e-6, "charges": 1e-6, "spins": 1e-6, "dipole": 1e-6,
            "pc_forces": 1e-6, "n_scc": 0, "scc_errors": 1e-4, "e_fr": 1e-6, "n_electrons": 1e-6,
            "fermi_level": 1e-8, **TOL_RESPONSE}
POL = {"polarisability": True}
KERNEL = {"response_kernel": True}
O2_SPIN = {"unpaired_electrons": 2, "spin_constants": {"O": -0.028}}


def _freq(*ev):
    """Frequencies in Hartree from eV (DFTB+'s Hartree__eV)."""
    return [x / HARTREE for x in ev]


CASES += [
    Case("h2o_pol", _water, torch=False, tol=TOL_RESP,
         analysis={"polarisability": {"frequencies": _freq(1.0, 3.0, 5.0)}},
         note="static and three frequencies below the first excitation"),
    Case("h2o_pol_nonscc", _water, scc=False, torch=False, tol=TOL_RESP, analysis=POL,
         note="non-SCC: one pass, the bare response"),
    Case("h2o_pol_shell", _water, shell_resolved=True, torch=False, tol=TOL_RESP, analysis=POL),
    Case("ch3oh_pol_dftb3", _meoh, dftb3=DFTB3_3OB, torch=False, tol=TOL_RESP, analysis=POL,
         note="the full third order linearised at the ground state's charges, H-X damping"),
    Case("h2o_pol_field", _water, efield=FIELD, torch=False, tol=TOL_RESP, analysis=POL,
         note="the ground state in an external field (the response operator is unchanged)"),
    Case("o2_triplet_pol", _o2, spin=O2_SPIN, torch=False, tol=TOL_RESP,
         analysis={"polarisability": {"frequencies": _freq(2.0)}},
         note="collinear spin: the response of charge and magnetisation (W)"),
    Case("h2o_anion_pol_300K", _water, charge=-1.0, filling=("fermi", 300.0), torch=False,
         tol={**TOL_RESP, "forces": 5e-6}, analysis=POL,
         note="metallic: a half-filled level, the Fermi level moves (N_F, dE_F/dE)"),
    Case("cu4co_pol_3000K", _cu4co, filling=("fermi", 3000.0), torch=False, tol=TOL_RESP,
         analysis=POL, note="metallic Cu cluster, many partly filled levels"),
    Case("cu4co_pol_fixef", _cu4co, filling=("fermi", 3000.0), fermi={"fixed": -0.17},
         torch=False, tol=TOL_RESP, analysis=POL,
         note="fixed Fermi level: no Fermi-level move, the electron count changes"),
    Case("c6h6_pol", _c6h6, torch=False, tol=TOL_RESP,
         analysis={"polarisability": {"frequencies": _freq(2.0)}},
         note="degenerate levels: the eigenvalue derivatives of a degenerate pair"),
    Case("cuco_pol_300K", _cuco_linear, filling=("fermi", 300.0), torch=False, tol=TOL_RESP,
         analysis=POL, note="degenerate levels and one half-filled sigma level (metallic)"),
    Case("o2_pol_300K", _o2, filling=("fermi", 300.0), torch=False, tol=TOL_RESP,
         analysis={"polarisability": {"frequencies": _freq(2.0)}},
         note="without spin polarisation: the half-filled degenerate pi* pair is a degenerate "
              "run inside W (DFTB+ rotates it, pydftb weights the whole run)"),
    Case("water_box_pol", _water_box, ewald_tolerance=1e-12, torch=False, tol=TOL_RESP,
         analysis=POL, note="periodic, Gamma point: charge response only (no tensor)"),
    Case("cuo_chain_pol_k", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 1000.0),
         ewald_tolerance=1e-12, torch=False, tol={**TOL_RESP, "dqdE": 5e-8},
         analysis={"polarisability": {"frequencies": _freq(1.0)}, "eta_compat": True},
         note="k-points (complex eigenvectors), metallic; at the finite frequency DFTB+ keeps "
              "an eta term that cancels between k and -k (reduced set: compared with "
              "DFTB._response_eta_compat; 'dqdE_default' is pydftb's full-set value)"),
    Case("h2o_kernel", _water, torch=False, tol=TOL_RESP, analysis=KERNEL,
         note="SCC-screened response to a potential on each atom"),
    Case("h2o_kernel_rpa", _water, torch=False, tol=TOL_RESP,
         analysis={"response_kernel": {"rpa": True}}, note="the bare (unscreened) response"),
    Case("h2o_kernel_freq", _water, torch=False, tol=TOL_RESP,
         analysis={"response_kernel": {"static": False, "frequencies": _freq(2.0, 4.0)}}),
    Case("o2_triplet_kernel", _o2, spin=O2_SPIN, torch=False, tol=TOL_RESP, analysis=KERNEL),
    Case("cu4co_kernel_3000K", _cu4co, filling=("fermi", 3000.0), torch=False, tol=TOL_RESP,
         analysis=KERNEL, note="metallic: the Fermi-level move keeps every column's sum zero"),
]


# ---- milestone M6g-2: coordinate derivatives (Born charges), SecondDerivatives, modes -------
# pydftb with DFTB+'s difference step for dH0, dS (FD_DFTBP) so that the matrices and forces are
# DFTB+'s; "dqdx": dN/dx per orbital (e/Bohr, DFTB+'s dqdx), "born": the Born charges (DFTB+'s
# borncharges = -born), "coord_text": the detailed.out blocks and dBand_dx.out (d eps/dx in
# eV/Bohr, 6 significant digits); "born_fd": pydftb's analytic Born
# charges against DFTB+'s finite differences of the dipole (delta 1e-4: an O(delta^2) error);
# "hessian" (Ha/Bohr^2), "hess_born" (d mu/dx), "hess_dpol" (d alpha/dx): pydftb's
# second_derivatives against DFTB+'s; "hess_text": hessian.out, born.out, bornderiv.out and the
# detailed.out block (whose 6 decimals may round either way: 2e-6); "modes_freq" (Hartree),
# "modes_int" (relative, vibrational groups only),
# "modes_vec" (normalised displacements of the non-degenerate vibrational modes, up to sign).
# Both codes converge the SCC to 1e-12 (SCC_DX): central differences with delta = 1e-4 multiply
# the noise of forces and dipoles by 5000
FD_DFTBP = 2.0 ** -13
SCC_DX = 1e-12
TOL_COORD = {"dqdx": 1e-10, "born": 1e-10, "coord_text": 1e-4, "born_fd": 1e-7,
             "hessian": 1e-7, "hess_born": 5e-8, "hess_dpol": 1e-6, "hess_text": 2e-6,
             "modes_freq": 1e-7, "modes_int": 1e-5, "modes_vec": 1e-4}
TOL_DX = {**TOL_RESP, **TOL_COORD, "forces": 1e-9}
COORD = {"coord_derivatives": True}
COORD_FD = {"coord_derivatives": True, "dftbplus_coord": False}   # DFTB+ refuses or errs
DX = dict(scc_tol=SCC_DX, fd_delta=FD_DFTBP, torch=False)
CASES += [
    # analytic coordinate derivatives against DFTB+ 25.1's CoordDerivatives
    Case("h2o_dx", _water, **DX, tol=TOL_DX, analysis=COORD, hessian={},
         note="Born charges and dq/dx; the hessian run's differences check them too"),
    Case("h2o_dx_nonscc", _water, scc=False, **DX, tol=TOL_DX, analysis=COORD,
         note="non-SCC: the bare response"),
    Case("h2o_dx_shell", _water, shell_resolved=True, **DX, tol=TOL_DX, analysis=COORD),
    Case("ch3oh_dx", _meoh, **DX, tol=TOL_DX, analysis=COORD),
    Case("h2co_dx", _h2co, **DX, tol=TOL_DX, analysis=COORD),
    Case("h2o_dx_pol", _water, **DX, tol=TOL_DX, analysis={**COORD, **POL},
         note="both analyses: DFTB+ 25.1 writes the polarisability blocks first"),
    Case("ch3oh_dx_dftb3", _meoh, dftb3=DFTB3_3OB, dftbplus="dev", **DX,
         tol={**TOL_DX, "n_scc": 10, "scc_errors": 1.0}, analysis=COORD,
         note="full third order: 25.1 leaves its terms out (1.4e-2), the development build has "
              "them; the development build's SCC takes another path (fewer iterations)"),
    # DFTB+ refuses or gets them wrong: its finite-difference Born charges as the reference
    Case("h2o_dx_field", _water, efield=FIELD, **DX, tol=TOL_DX, analysis=COORD_FD,
         hessian={},
         note="DFTB+ leaves the field's position derivative out (Born charges off by 1.7e-2)"),
    Case("h2o_dx_pc", _water, point_charges=PC3, **DX, tol=TOL_DX, analysis=COORD_FD,
         hessian={}, note="DFTB+ leaves the point charges' position derivative out"),
    Case("o2_triplet_dx", _o2, spin=O2_SPIN, **DX, tol=TOL_DX, analysis=COORD_FD, hessian={},
         note="collinear spin: DFTB+ refuses (O2's Born charges vanish by symmetry; the Hessian "
              "is checked)"),
    Case("h2o_cation_dx", _water, charge=1.0, spin={"unpaired_electrons": 1,
                                                    "spin_constants": W_SPIN},
         **DX, tol=TOL_DX, analysis=COORD_FD, hessian={},
         note="spin polarised and polar: DFTB+ refuses; Born charges sum to +1"),
    Case("cu4co_dx_3000K", _cu4co, filling=("fermi", 3000.0), **DX, tol=TOL_DX,
         analysis={**COORD_FD, "perturbation": {"filling_tol": 1e-15}}, hessian={},
         note="metallic (Fermi-level move): DFTB+ refuses; every partly filled level (DFTB+'s "
              "window of M6g-1 leaves out 4.7e-7 of the Born charges)"),
    Case("co2_dx", _co2, **DX, tol=TOL_DX, analysis=COORD_FD, hessian={},
         note="degenerate levels: DFTB+ refuses"),
    # SecondDerivatives (and modes)
    Case("h2o_hess_pol", _water, **DX, tol=TOL_DX,
         hessian={"polarisability": True, "modes": {"remove": False}},
         note="Hessian, Born charges and polarisability derivatives; modes without "
              "projection (the H-H distance is 4.6e-6 Bohr from an SK grid point: the "
              "differences see the interpolation's derivative jump, the same in both codes)"),
    Case("h2o_hess_projected", _water, **DX, tol=TOL_DX,
         hessian={"polarisability": True, "modes": {"remove": True}},
         note="modes with translations and rotations projected out"),
    Case("h2o_hess_partial", _water, **DX, tol=TOL_DX,
         hessian={"atoms": [0, 1, 2], "moved_atoms": [1, 2]},
         note="partial Hessian: hessian.out.2-3 (DFTB+ also leaves an empty hessian.out)"),
    Case("h2o_hess_nonscc", _water, scc=False, **DX, tol=TOL_DX, hessian={}),
    Case("ch3oh_hess_dftb3", _meoh, dftb3=DFTB3_3OB, **DX, tol=TOL_DX,
         hessian={"modes": {"remove": True}}),
    Case("water_box_hess", _water_box, ewald_tolerance=1e-12, **DX, tol=TOL_DX, hessian={},
         note="periodic: no dipole derivatives (DFTB+ writes meaningless ones)"),
    Case("co2_hess_pol", _co2, **DX, tol=TOL_DX,
         hessian={"polarisability": True, "modes": {"remove": True}},
         note="linear molecule: five rigid motions, degenerate bending modes"),
]


# ---- milestone M6g-3a: responses with block potentials (DFTB+U, onsite corrections) and the
# derivatives with respect to the point charges' positions.  The polarisability and the kernel
# against DFTB+ 25.1 (it refuses +U together with onsite); the coordinate derivatives against
# DFTB+'s finite-difference Born charges (it refuses them with block potentials); "dqdxext":
# dN/dR of the point charges (e/Bohr) against the development build's ExternalChargeDerivatives
# (its tag; reversed sign for periodic systems, whose Ewald derivative it takes with respect to
# the atom), "extq_text": its detailed.out block (8 decimals in e/Angstrom)
TOL_BRESP = {**TOL_RESP, "orbital": 1e-6, "eigenvalues": 1e-7, "moments": 1e-8}
TOL_BDX = {**TOL_DX, "orbital": 1e-6, "eigenvalues": 1e-7, "moments": 1e-8}
# metals with block potentials converge the SCC more slowly: the dipoles of the displaced
# geometries carry ~1e-11 of SCC error in each code, which the central differences (2 delta =
# 2e-4 Bohr) turn into ~5e-8 per code ("hess_born"); the last SCC step may land on either side
# of the tolerance (one iteration, as M6b's block cases allow)
TOL_BDX_METAL = {**TOL_BDX, "n_scc": 1, "hess_born": 1e-7}
TOL_EXTQ = {**TOL_BRESP, "dqdxext": 1e-9, "extq_text": 2e-8}
EXTQ = {"external_charge_derivatives": True}
CASES += [
    # polarisability and response kernel against DFTB+ 25.1
    Case("cu4co_plusu_pol", _cu4co, filling=("fermi", 300.0), plus_u=U_CU, torch=False,
         tol=TOL_BRESP, analysis=POL,
         note="FLL +U on the Cu d shells, metallic: the d blocks are response channels"),
    Case("cu4co_plusu_psic_pol", _cu4co, filling=("fermi", 300.0),
         plus_u={**U_CU, "functional": "psic"}, torch=False, tol=TOL_BRESP, analysis=POL),
    Case("gdo_spin_plusu_pol", _gdo, shell_resolved=True, **GDO,
         spin={"unpaired_electrons": 8, "spin_constants": W_SPIN},
         plus_u={"groups": {"Gd": [("f", 0.2)]}}, torch=False, tol={**TOL_BRESP, "dEf": 5e-9},
         analysis=POL,
         note="collinear spin and an f-shell +U: up and down blocks as (q, m) channels; the "
              "down channel's N_F = 1.6e-4 amplifies DFTB+'s response convergence "
              "(PerturbSccTol 1e-11) in its dE_F/dE about 6000-fold (up channel: 4e-15)"),
    Case("h2o_ons_pol", _water, onsite=ONS, torch=False, tol=TOL_BRESP,
         analysis={"polarisability": {"frequencies": _freq(1.0, 3.0)}},
         note="onsite corrections: every element of the atom blocks is a channel"),
    Case("ch3oh_ons_dftb3_pol", _meoh, dftb3=DFTB3_3OB, onsite=ONS, torch=False, tol=TOL_BRESP,
         analysis=POL),
    Case("cuo_chain_spin_plusu_pol_k422", _cuo_chain, kgrid=(4, 2, 2), filling=("fermi", 500.0),
         spin={"unpaired_electrons": 2, "spin_constants": W_SPIN}, plus_u=U_CU,
         ewald_tolerance=1e-12, torch=False, tol=TOL_BRESP, analysis=POL,
         note="periodic (charge response only), k-points (complex block elements), spin, +U"),
    Case("cu4co_plusu_kernel", _cu4co, filling=("fermi", 300.0), plus_u=U_CU, torch=False,
         tol=TOL_BRESP, analysis=KERNEL),
    Case("h2o_ons_kernel", _water, onsite=ONS, torch=False, tol=TOL_BRESP, analysis=KERNEL),
    # coordinate derivatives: DFTB+ refuses block potentials, its differences are the reference
    Case("h2o_ons_dx", _water, onsite=ONS, **DX, tol=TOL_BDX, analysis=COORD_FD, hessian={},
         note="the ground state's block potential enters through S' and the block populations"),
    Case("ch3oh_ons_dx", _meoh, onsite=ONS, **DX, tol=TOL_BDX, analysis=COORD_FD, hessian={}),
    Case("h2o_cation_plusu_dx", _water, charge=1.0,
         spin={"unpaired_electrons": 1, "spin_constants": W_SPIN}, plus_u=U_O, **DX,
         tol=TOL_BDX, analysis=COORD_FD, hessian={},
         note="spin polarised, +U on the O p shell; Born charges sum to +1"),
    Case("cu4co_plusu_dx_3000K", _cu4co, filling=("fermi", 3000.0), plus_u=U_CU, **DX,
         tol=TOL_BDX_METAL, analysis={**COORD_FD, "perturbation": {"filling_tol": 1e-15}},
         hessian={}, note="metallic (every partly filled level, as cu4co_dx_3000K)"),
    Case("cu4co_plusu_ons_dx_3000K", _cu4co, filling=("fermi", 3000.0), plus_u=U_CU,
         onsite=ONS, **DX, tol=TOL_BDX_METAL,
         analysis={**COORD_FD, "perturbation": {"filling_tol": 1e-15}}, hessian={},
         note="+U and onsite corrections together (DFTB+ refuses any response with both)"),
    # point charges' positions against the development build's ExternalChargeDerivatives
    Case("h2o_pc_extq", _water, point_charges=PC3, dftbplus="dev", torch=False, tol=TOL_EXTQ,
         analysis=EXTQ),
    Case("h2o_pc_blur_extq", _water, point_charges=PC_CLOSE, dftbplus="dev", torch=False,
         tol=TOL_EXTQ, analysis={"external_charge_derivatives": {"charges": [1]}},
         note="Gaussian blur, only the second charge (WrtCharges = 2)"),
    Case("h2o_cation_pc_extq", _water, charge=1.0, point_charges=PC3,
         spin={"unpaired_electrons": 1, "spin_constants": W_SPIN}, dftbplus="dev",
         torch=False, tol=TOL_EXTQ, analysis=EXTQ, note="collinear spin"),
    Case("ch3oh_ons_pc_extq", _meoh, onsite=ONS, point_charges=PC3, dftbplus="dev",
         torch=False, tol=TOL_EXTQ, analysis=EXTQ, note="with onsite corrections"),
    Case("water_box_pc_extq", _water_box, point_charges=PC_BOX, ewald_tolerance=1e-12,
         dftbplus="dev", torch=False, tol=TOL_EXTQ, analysis=EXTQ,
         note="periodic (Ewald): the development build's sign is reversed"),
]


# ---- milestone M6g-3b: responses of two-component calculations (non-collinear spin, onsite and
# dual spin-orbit coupling, also with +U and onsite corrections).  The polarisability and the
# kernel against DFTB+ 25.1: the four components (q, m_x, m_y, m_z) of dq/dE and dq/dV as charge
# ("dqdE", "dqdV", absolute) and magnetisation ("dmdE", "dmdV", relative to max(1, |DFTB+|):
# soft spin modes make them large, without moments they are noise).  Dual spin-orbit coupling
# with block potentials: DFTB+ 25.1 leaves the skew block response out; "skew_compat" compares
# pydftb with DFTB+'s own finite fields ("pol_fd", relative; "field_fd": the runs, h in a.u.)
# and with DFTB+'s analytic value ("pol_dftbp": its error, ~1e-4), then repeats the comparison
# of the tags and texts with DFTB._response_skew_compat.  The coordinate derivatives against
# DFTB+'s finite-difference Born charges (DFTB+ refuses two-component ones), the point charges'
# positions against the development build
TOL_PRESP = {**TOL_RESP, "moments": 1e-8, "orbital": 1e-6, "eigenvalues": 1e-7, "dmdE": 1e-8,
             "dmdV": 1e-8}
TOL_SKEW = {**TOL_PRESP, "pol_fd": 1e-8, "pol_dftbp": 1e-3}
SOC_O = {"xi": {"O": [0.0, 0.01]}}
SKEW = {"polarisability": True, "skew_compat": True, "field_fd": True}
_OHNC = {"filling": ("fermi", 300.0), "noncolinear": NC_OH}
_O2Z = {"noncolinear": {"initial_spins": [[0.0, 0.0, 1.0]] * 2, "spin_constants": {"O": -0.028}},
        "spin_orbit": SOC_O}
CASES += [
    # polarisability and response kernel against DFTB+ 25.1
    Case("oh_noncolinear_pol", _oh, **_OHNC, torch=False, tol=TOL_PRESP, analysis=POL,
         note="no spin-orbit coupling: the global rotations of the tilted moments are exact zero "
              "modes (bordered solve); DFTB+'s magnetisation response is longitudinal"),
    Case("oh_noncolinear_soc_pol", _oh, **_OHNC, spin_orbit=SOC_O, torch=False, tol=TOL_PRESP,
         analysis={"polarisability": {"frequencies": _freq(1.0, 3.0)}},
         note="dual spin-orbit coupling, static and two frequencies"),
    Case("oh_noncolinear_soc_onsite_pol", _oh, **_OHNC, spin_orbit={**SOC_O, "dual": False},
         torch=False, tol=TOL_PRESP, analysis=POL),
    Case("o2_noncolinear_soc_pol", _o2, **_O2Z, torch=False, tol=TOL_PRESP, analysis=POL,
         note="T = 0 (insulating)"),
    Case("cu4co_soc_pol", _cu4co, filling=("fermi", 300.0), spin_orbit={"xi": XI_CU4CO},
         torch=False, tol=TOL_PRESP, analysis=POL,
         note="no spin polarisation: Kramers pairs (degenerate runs), metallic"),
    Case("gdo_noncolinear_soc_pol", _gdo, shell_resolved=True, **GDO, noncolinear=NC_GDO,
         spin_orbit={"xi": XI_GDO}, torch=False, tol={**TOL_PRESP, "scc_errors": 0.5},
         analysis=POL,
         note="7 muB and weak spin-orbit coupling: a soft spin mode (|dm/dE| ~ 240) and an exact "
              "zero mode (left out with a warning); along the 166 SCC iterations the soft mode "
              "lets the Broyden steps drift apart (errors from 2.5e-3 on, up to 22 %), same "
              "iterations and final state"),
    Case("cuo_chain_soc_pol_k422", _cuo_chain, kgrid=(4, 2, 2), time_reversal=False,
         filling=("fermi", 500.0), spin_orbit={"xi": XI_CUO}, ewald_tolerance=1e-12,
         torch=False, tol=TOL_PRESP, analysis=POL, note="k-points (charge response only)"),
    Case("oh_noncolinear_plusu_pol", _oh, **_OHNC, plus_u=U_O, torch=False,
         tol={**TOL_PRESP, "scc_errors": 5.0}, analysis=POL,
         note="+U: the zero modes rotate the block magnetisations too; the SCC errors agree to "
              "1e-8 down to 4e-7, then the free rotation lets the last two Broyden steps "
              "(errors ~1e-6) differ by a factor of 3 (same iterations and final state)"),
    Case("oh_noncolinear_kernel", _oh, **_OHNC, torch=False, tol=TOL_PRESP, analysis=KERNEL),
    Case("oh_noncolinear_soc_kernel", _oh, **_OHNC, spin_orbit=SOC_O, torch=False,
         tol=TOL_PRESP, analysis=KERNEL),
    Case("cu4co_soc_kernel", _cu4co, filling=("fermi", 300.0), spin_orbit={"xi": XI_CU4CO},
         torch=False, tol=TOL_PRESP, analysis=KERNEL),
    # dual spin-orbit coupling with block potentials: DFTB+ 25.1 leaves the skew response out
    Case("oh_noncolinear_soc_plusu_pol", _oh, **_OHNC, spin_orbit=SOC_O, plus_u=U_O,
         torch=False, tol=TOL_SKEW, analysis=SKEW),
    Case("oh_noncolinear_soc_ons_pol", _oh, **_OHNC, spin_orbit=SOC_O, onsite=ONS, torch=False,
         tol={**TOL_SKEW, "scc_errors": 0.5}, analysis=SKEW,
         note="the SCC errors agree to 7e-5 down to 1e-6, the last Broyden steps differ by up "
              "to 34 % (same iterations and final state)"),
    Case("cu4co_soc_plusu_pol", _cu4co, filling=("fermi", 300.0), spin_orbit={"xi": XI_CU4CO},
         plus_u=U_CU, torch=False, tol=TOL_SKEW, analysis={**SKEW, "field_fd": {"h": 2.5e-4}},
         note="metallic: field step 2.5e-4 a.u. (5e-4 leaves 3e-8 of h^4 truncation)"),
    Case("cuo_chain_soc_plusu_kernel_k422", _cuo_chain, kgrid=(4, 2, 2), time_reversal=False,
         filling=("fermi", 1000.0), spin_orbit={"xi": XI_CUO}, plus_u=U_CU,
         ewald_tolerance=1e-12, torch=False,
         tol={**TOL_SKEW, "kernel_dftbp": 1.0, "dmdV": 1e-6, "dedV": 5e-9, "n_scc": 1,
              "scc_errors": 1.0},
         analysis={"response_kernel": True, "skew_compat": True},
         note="periodic, k-points: no finite fields; pydftb's physical kernel is checked by "
              "differences in site potentials (test_response_pauli); 'kernel_dftbp' records "
              "DFTB+'s error (75 %), the rest is compared with DFTB._response_skew_compat.  "
              "Without moments (1e-11) the magnetisation response is noise, ~2.5e-7 in both "
              "codes and in each environment (dmdV 1e-6); the 63-iteration metallic SCC ends "
              "one iteration apart and the ground states agree to ~1e-9 (N_F 4e-9, the level "
              "derivatives 2e-9)"),
    # coordinate derivatives: DFTB+ refuses them, its differences are the reference
    Case("oh_noncolinear_dx", _oh, **_OHNC, **DX, tol=TOL_BDX, analysis=COORD_FD, hessian={}),
    Case("oh_noncolinear_soc_dx", _oh, **_OHNC, spin_orbit=SOC_O, **DX, tol=TOL_BDX,
         analysis=COORD_FD, hessian={},
         note="dual model: H_SO = i/2 (S Lambda + Lambda S) x sigma moves with S (1.2e-4 of the "
              "Born charges)"),
    Case("oh_noncolinear_soc_onsite_dx", _oh, **_OHNC, spin_orbit={**SOC_O, "dual": False},
         **DX, tol=TOL_BDX, analysis=COORD_FD, hessian={}),
    Case("oh_noncolinear_soc_plusu_dx", _oh, **_OHNC, spin_orbit=SOC_O, plus_u=U_O, **DX,
         tol=TOL_BDX, analysis=COORD_FD, hessian={},
         note="skew block potentials and populations in S' and the response"),
    Case("cu4co_soc_dx_3000K", _cu4co, filling=("fermi", 3000.0), spin_orbit={"xi": XI_CU4CO},
         **DX, tol=TOL_BDX_METAL, analysis={**COORD_FD, "perturbation": {"filling_tol": 1e-15}},
         hessian={}, note="metallic, Kramers pairs"),
    # point charges' positions against the development build
    Case("oh_noncolinear_soc_pc_extq", _oh, **_OHNC, spin_orbit=SOC_O, point_charges=PC3,
         dftbplus="dev", torch=False, tol={**TOL_EXTQ, "moments": 1e-8}, analysis=EXTQ,
         note="the development build prints the charge part; pydftb gives all four components"),
]

# Range-separated hybrid functionals (milestone M6h-1): LC-DFTB, CAM and a global hybrid for
# molecules, on ptbp_hcocu with a RangeSep section (make_hybrid_sets.py; not a physical
# parametrisation).  The references keep every SCC error: the mixed vector is DeltaP, and the
# iteration counts must match.  fd_delta: DFTB+'s difference step for the H0/S derivatives.
HYB_FD = 2.0 ** -13
TOL_HYB = {"energy": 1e-8, "forces": 1e-8, "charges": 1e-8, "spins": 1e-8, "n_scc": 0,
           "scc_errors": 1e-4, "orbital": 1e-8, "eigenvalues": 1e-8, "moments": 1e-8,
           "pc_forces": 1e-8, "dipole": 1e-8, "e_fr": 1e-8, "n_electrons": 1e-8,
           "fermi_level": 1e-8, "charges_file": 1e-8}
W_C = [[-0.031, -0.025], [-0.025, -0.023]]        # spin constants of C (Ha), mio's values
W_HCO = {"H": W_SPIN["H"], "C": W_C, "O": W_SPIN["O"]}
LC = {"skf_set": "ptbp_hcocu_lc", "hybrid": "lc", "fd_delta": HYB_FD, "torch": False,
      "tol": TOL_HYB}


def _ch3():
    a = molecule("CH3")
    a.rattle(0.02, seed=21)
    return a


def _c6h6_rattled():
    a = molecule("C6H6")
    a.rattle(0.02, seed=22)
    return a


def _hco():
    """The formyl radical (bent): no degenerate level.  The OH radical is no test for a hybrid
    with spin: its half-filled pi pair in the minority channel breaks the symmetry in a direction
    set by rounding, and the SCC path (not the energy) depends on it."""
    return Atoms("HCO", positions=[[-0.637, 0.909, 0.0], [0, 0, 0], [1.18, 0, 0]])


def _cuh():
    """CuH, closed shell.  CuCO (an odd electron count) with LC and +U at 300 K is no test: the
    SCC diverges in both codes, and DFTB+ converges after 387 iterations by chance."""
    return Atoms("CuH", positions=[[0, 0, 0], [0.3, 0.2, 1.45]])


CASES += [
    Case("h2o_lc", _water, **LC),
    Case("ch3oh_lc", _meoh, **LC),
    Case("c6h6_lc", _c6h6_rattled, **LC),
    Case("h2co_cam", _h2co, **{**LC, "skf_set": "ptbp_hcocu_cam", "hybrid": "cam"},
         note="CAM 0.3 0.19 0.46 (DFTB+'s CAM test values)"),
    Case("c2h4_global", _c2h4, **{**LC, "skf_set": "ptbp_hcocu_global", "hybrid": "cam"},
         note="a global hybrid: CAM with beta = 0 (DFTB+ has no Hybrid = Global)"),
    Case("ch3oh_lc_cam_keyword", _meoh, **{**LC, "hybrid": "cam"},
         note="Hybrid = CAM takes LC files"),
    # screening
    Case("c6h6_lc_neighbour_10", _c6h6_rattled,
         **{**LC, "hybrid": {"type": "lc", "screening": "neighbour", "cutoff_reduction": 10.0}},
         note="S blocks beyond the largest SK cutoff minus 10 Bohr (5.4 Bohr: the meta and para "
              "C-H and H-H pairs) left out of the exchange; the forces stay exact (as DFTB+): "
              "not the energy's derivative"),
    Case("c6h6_lc_threshold", _c6h6_rattled,
         **{**LC, "hybrid": {"type": "lc", "screening": "threshold"}}, scc_tol=1e-6,
         note="DFTB+'s incremental Thresholded build, threshold 1e-6.  scc_tol 1e-6: below it "
              "the screening decisions flip with rounding and the tail of the SCC (not its "
              "result) differs between the codes (74 and 52 iterations to 1e-10)"),
    Case("c6h6_lc_threshold_1e-4_red4", _c6h6_rattled,
         **{**LC, "hybrid": {"type": "lc", "screening": "threshold", "threshold": 1e-4,
                             "cutoff_reduction": 4.0}}),
    # collinear spin
    Case("ch3_lc_spin", _ch3, **LC, spin={"unpaired_electrons": 1.0, "spin_constants": W_HCO}),
    Case("hco_lc_spin_neighbour", _hco,
         **{**LC, "hybrid": {"type": "lc", "screening": "neighbour"}},
         spin={"unpaired_electrons": 1.0, "spin_constants": W_HCO}),
    Case("o2_cam_triplet", _o2, **{**LC, "skf_set": "ptbp_hcocu_cam", "hybrid": "cam"},
         spin={"unpaired_electrons": 2.0, "spin_constants": W_HCO}),
    Case("h2o_lc_spin_closed", _water, **LC,
         spin={"unpaired_electrons": 0.0, "spin_constants": W_HCO},
         note="a closed shell run spin-polarised: the restricted result (test_hybrid)"),
    Case("h2o_cation_lc_spin", _water, **LC, charge=1.0,
         spin={"unpaired_electrons": 1.0, "spin_constants": W_HCO}),
    Case("ch3_lc_relax_spin", _ch3, **LC, filling=("fermi", 300.0),
         spin={"unpaired_electrons": 1.0, "relax_total_spin": True, "spin_constants": W_HCO}),
    # charged, smearing, d shells
    Case("oh_anion_lc", _oh, **LC, charge=-1.0),
    Case("cuco_lc_300K", _cuco_linear, **LC, filling=("fermi", 300.0)),
    Case("c6h6_lc_mp", _c6h6_rattled, **LC, filling=("mp", 1000.0, 2)),
    # with the rest of the Hamiltonian
    Case("ch3oh_lc_dftb3_d3", _meoh, **LC, dftb3=DFTB3_3OB, dispersion=D3BJ_3OB,
         note="the full third order (DFTB+ refuses only the on-site ThirdOrder)"),
    Case("ch3oh_lc_field_pc", _meoh, **LC, efield=FIELD, point_charges=PC3),
    Case("h2o_lc_gbsa", _water, **LC, solvation=GBSA_WATER),
    Case("h2co_lc_plusu", _h2co, **LC, plus_u=U_O),
    Case("cuh_lc_plusu_300K", _cuh, **LC, filling=("fermi", 300.0),
         plus_u={"groups": {"Cu": [("d", 0.05)]}},
         note="+U on Cu d: the block populations come from DeltaP_in, mixed from the first "
              "iteration (no unmixed first output without spin).  U - J = 0.05 Ha: with LC on "
              "this semi-local set, 0.15 Ha (U_CU) makes the SCC of Cu molecules diverge"),
    Case("hco_lc_plusu_spin", _hco, **LC, plus_u=U_O,
         spin={"unpaired_electrons": 1.0, "spin_constants": W_HCO}),
    Case("h2o_lc_onsite", _water, **LC, onsite=ONS),
    Case("ch3oh_lc_anderson", _meoh, **LC, mixer="anderson"),
    Case("h2o_lc_custom_hubbards", _water, **LC, hubbard={"O": 0.5, "H": 0.45},
         note="tau = 3.2 U of the custom Hubbard values"),
    # restart files with the density matrix (charges.bin / charges.dat): run 2 starts from run 1's
    # populations and DeltaP_in
    Case("h2o_lc_read_charges", _water, **LC, read_charges={"text": False}),
    Case("ch3_lc_spin_read_charges_text", _ch3, **LC, read_charges={"text": True},
         spin={"unpaired_electrons": 1.0, "spin_constants": W_HCO},
         note="DeltaP_in of both spin channels (up, down) after the (q, m) populations"),
    Case("h2co_lc_plusu_read_charges", _h2co, **LC, plus_u=U_O, read_charges={"text": False},
         note="blocks, then the density matrix"),
]

# M6h-1b: non-collinear spin with a hybrid functional (DFTB+: the matrix screening, molecules).
# The spins start along TILT (out of every axis): a complex DeltaP whose magnetisation keeps one
# direction, so the results are those of the collinear calculation turned
NC_HCO = {"initial_spins": [[0.0, 0.0, 0.0], TILT, [0.0, 0.0, 0.0]], "spin_constants": W_HCO}
NC_CH3 = {"initial_spins": [TILT, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          "spin_constants": W_HCO}


def _h3():
    """An H3 triangle (sides 2.5 A): from spins at 120 degrees in its plane the SCC ends in a
    non-collinear state (DeltaP's spin-diagonal blocks have imaginary parts)."""
    r = 2.5 / np.sqrt(3.0)
    ang = np.deg2rad([90.0, 210.0, 330.0])
    return Atoms("H3", positions=np.c_[r * np.cos(ang), r * np.sin(ang), np.zeros(3)])


CASES += [
    Case("hco_lc_noncolinear", _hco, **LC, noncolinear=NC_HCO,
         note="complex DeltaP; the energy and forces of the collinear calculation"),
    Case("ch3_lc_noncolinear", _ch3, **LC, noncolinear=NC_CH3),
    Case("hco_lc_noncolinear_300K", _hco, **LC, noncolinear=NC_HCO, filling=("fermi", 300.0)),
    Case("ch3_lc_noncolinear_anderson", _ch3, **LC, noncolinear=NC_CH3, mixer="anderson",
         note="DFTB+'s complex Anderson mixer is the real one on [Re | Im]"),
    Case("h3_lc_noncolinear_120", _h3,
         **{**LC, "tol": {**TOL_HYB, "forces_dftbp": 1e-10}}, filling=("fermi", 100.0),
         noncolinear={"initial_spins": [[0.0, 1.0, 0.0], [-0.8660254, -0.5, 0.0],
                                        [0.8660254, -0.5, 0.0]],
                      "spin_constants": W_HCO},
         scc_tol=1e-12, forces_fd=True,
         note="a non-collinear SCC solution: the spins stay at 120 degrees, and DeltaP's Pauli "
              "components have imaginary (antisymmetric) parts.  DFTB+'s forces are not the "
              "energy's derivative (1.1e-5 Ha/Bohr off): addCamGradientsMatrix_pauli takes the "
              "gamma term from the real parts of the components only (pauli2components); "
              "compared with differences of DFTB+'s energy, and with that term as DFTB+'s "
              "(HybridTerms._dftbp_pauli_gamma) with DFTB+'s forces"),
    Case("o2_cam_noncolinear_90", _o2,
         **{**LC, "skf_set": "ptbp_hcocu_cam", "hybrid": "cam",
            "tol": {**TOL_HYB, "n_scc": 4, "scc_errors": 1e3, "spins": 1e-2}},
         noncolinear={"initial_spins": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                      "spin_constants": W_HCO},
         note="the O spins start at 90 degrees and end parallel.  Not the same SCC path: DFTB+'s "
              "input populations after mixing (denseMullikenPauli) take only the lower left spin "
              "block of every atom pair, which is not the Mulliken population while that block "
              "is not symmetric (in this SCC's first iterations); pydftb's are the Mulliken "
              "populations.  The same energy and forces, more iterations in DFTB+; the direction the "
              "parallel spins end in depends on the path (without spin-orbit coupling every "
              "direction is a solution): 7e-3 between the codes"),
    Case("hco_lc_noncolinear_field_pc", _hco, **LC, noncolinear=NC_HCO, efield=FIELD,
         point_charges=PC3),
    Case("hco_lc_noncolinear_dftb3_d3", _hco, **LC, noncolinear=NC_HCO, dftb3=DFTB3_3OB,
         dispersion=D3BJ_3OB),
    Case("hco_lc_onsite_spin", _hco, **LC, onsite=ONS,
         spin={"unpaired_electrons": 1.0, "spin_constants": W_HCO},
         note="collinear: the result of pydftb's non-collinear calculation with onsite "
              "corrections (test_hybrid); DFTB+'s non-collinear one keeps the first output's "
              "blocks and ends 1.3e-3 Ha above (M6h-1b report)"),
    Case("hco_lc_noncolinear_read_charges", _hco, **LC, noncolinear=NC_HCO,
         read_charges={"text": True, "run1_skf_set": "ptbp_hcocu"},
         note="DFTB+ writes no charge file for a non-collinear hybrid: run 1 without the hybrid "
              "(the SK files without RangeSep) writes the populations, run 2 starts from them "
              "with DeltaP_in = 0"),
]


# M6h-2: TD-LC, Casida with the long-range exchange (DFTB+ needs its Stratmann solver).  One spin
# constant per type, the HOAO values of W_HCO's shell matrices: a restricted triplet's HSD has one W
# per type (dftbplus.spin_constants_block), and excited-state forces need one per type (M6e).
# DFTB+'s Stratmann solver converges the squared residuals to 1.2e-8: exact where its subspace
# (20 x the states) holds every transition, ~1e-9 in Omega^2 beyond (benzene, naphthalene).
W_HOAO = {"H": -0.072, "C": -0.023, "O": -0.028}
TDLC = {"n": 6, "symmetry": "singlet", "solver": "Stratmann"}
TDLC_EXC = {**TDLC, "state": 2, "excited_forces": True}
TOL_TDLC = {**TOL_HYB, "exc": 1e-8, "exc_osc": 1e-7, "exc_tdip": 1e-7, "exc_table": 0, "spx": 0,
            "exc_state": 1e-9, "exc_charges": 1e-7, "exc_dipole": 1e-7, "forces_dftbp": 1e-10}
TDLC_CASE = {**LC, "tol": TOL_TDLC}
TOL_TDLC_STRATMANN = {**TOL_TDLC, "exc": 3e-8, "exc_osc": 3e-6, "exc_tdip": 3e-4,
                      "exc_state": 3e-8, "exc_table": 10, "forces": 3e-6, "exc_charges": 1e-6,
                      "exc_dipole": 1e-5}


def _naphthalene():
    """Naphthalene (rattled): 576 transitions, so pydftb's Stratmann iteration ("auto" beyond
    500) and the Z vector's conjugate gradients.  Seed 37: every atom pair at least 3e-4 Bohr
    from a point of the SK grid (0.02 Bohr), so DFTB+'s difference step (2^-13 Bohr) and the
    tests' differences stay on one interpolation polynomial (seed 5 put a C-C pair 7.5e-5 from
    one: DFTB+'s forces 3e-5 off there)."""
    c = [(0, 0.70), (0, -0.70), (1.2124, 1.40), (2.4249, 0.70), (2.4249, -0.70),
         (1.2124, -1.40), (-1.2124, -1.40), (-2.4249, -0.70), (-2.4249, 0.70), (-1.2124, 1.40)]
    h = [(1.2124, 2.49), (3.369, 1.245), (3.369, -1.245), (1.2124, -2.49), (-1.2124, -2.49),
         (-3.369, -1.245), (-3.369, 1.245), (-1.2124, 2.49)]
    a = Atoms("C10H8", positions=[(x, y, 0.0) for x, y in c + h])
    a.rattle(0.02, seed=37)
    return a


CASES += [
    # excitations
    Case("h2o_tdlc", _water, **TDLC_CASE, casida=TDLC),
    Case("h2co_tdlc", _h2co, **TDLC_CASE, casida={**TDLC, "n": 8}),
    Case("h2co_tdlc_triplet", _h2co, **TDLC_CASE,
         casida={**TDLC, "symmetry": "triplet", "spin_constants": W_HOAO},
         note="the same exchange in A+B and A-B as the singlets (DFTB+ warns: not well tested)"),
    Case("h2co_tdlc_both", _h2co, **TDLC_CASE,
         casida={**TDLC, "n": 4, "symmetry": "both", "spin_constants": W_HOAO}),
    Case("ch3oh_tdlc", _meoh, **TDLC_CASE, casida={**TDLC, "n": 12}),
    Case("h2co_cation_tdlc", _h2co, **TDLC_CASE, charge=1.0,
         spin={"unpaired_electrons": 1.0, "spin_constants": W_HCO},
         casida={**TDLC, "symmetry": None},
         note="spin polarised: the exchange within each spin channel only"),
    Case("o2_tdlc", _o2, **TDLC_CASE, spin={"unpaired_electrons": 2.0, "spin_constants": W_HCO},
         casida={**TDLC, "symmetry": None}, note="spin-polarised triplet, degenerate states"),
    Case("cuh_tdlc", _cuh, **TDLC_CASE, casida=TDLC, note="Cu d orbitals"),
    Case("c2h4_tdlc_bright", _c2h4, **TDLC_CASE, casida={**TDLC, "state": -1}),
    Case("ch3oh_tdlc_cam_keyword", _meoh, **{**TDLC_CASE, "hybrid": "cam"}, casida=TDLC,
         note="Hybrid = CAM with LC files: DFTB+ takes the type from the files, TD-LC runs"),
    Case("h2co_tdlc_transq", _h2co, **{**TDLC_CASE, "tol": {**TOL_TDLC, **TOL_TRANSQ}},
         casida={**TDLC, "state": 2, "transition_charges": True}),
    Case("c6h6_tdlc", _c6h6_rattled, **{**TDLC_CASE, "tol": TOL_TDLC_STRATMANN},
         casida={**TDLC, "n": 8},
         note="225 transitions: DFTB+'s Stratmann subspace (160) does not hold them all, its "
              "Omega^2 are 1.8e-9 and its oscillator strengths 2.3e-7 off (EXC.DAT's 8 "
              "decimals: some printed values differ)"),
    # excited-state forces and charges (Z vector with the exchange)
    Case("h2co_tdlc_exc_forces", _h2co, **TDLC_CASE, casida=TDLC_EXC),
    Case("h2co_tdlc_exc_forces_triplet", _h2co, **TDLC_CASE,
         casida={**TDLC_EXC, "symmetry": "triplet", "spin_constants": W_HOAO}),
    Case("h2co_cation_tdlc_exc_forces", _h2co, **TDLC_CASE, charge=1.0,
         spin={"unpaired_electrons": 1.0, "spin_constants": W_HOAO},
         casida={**TDLC_EXC, "symmetry": None},
         note="one W per type: with W_HCO's shell matrices DFTB+'s forces are 2.6e-4 off "
              "(pydftb refuses, M6e)"),
    Case("c2h4_tdlc_exc_forces_s1", _c2h4, **TDLC_CASE, casida={**TDLC_EXC, "state": 1}),
    Case("h2co_tdlc_exc_forces_s1", _h2co, **TDLC_CASE, casida={**TDLC_EXC, "n": 4, "state": 1},
         note="first singlet (n -> pi*); the base of the TD-LC S1 optimisation"),
    Case("ch3oh_tdlc_exc_forces", _meoh, **TDLC_CASE, casida={**TDLC_EXC, "n": 8, "state": 3}),
    Case("h2co_tdlc_exc_charges", _h2co, **TDLC_CASE,
         casida={**TDLC_EXC, "state": 4, "excited_forces": False, "excited_charges": True}),
    Case("h2co_tdlc_exc_forces_field_pc", _h2co,
         **{**TDLC_CASE, "tol": {**TOL_TDLC, **TOL_ENV, "forces": 1e-8, "forces_dftbp": 1e-10,
                                 "n_scc": 1}},
         efield=FIELD, point_charges=PC3, scc_tol=1e-12,
         casida={**TDLC_EXC, "forces_fd": True, "compat": ["external"]},
         note="DFTB+'s TD-LC forces are 3.0e-3 off with a field and point charges (the terms "
              "M6e found missing): compared with differences of DFTB+'s E0 + w, and pydftb "
              "leaving out what DFTB+ leaves out with DFTB+'s forces.  SCC tolerance 1e-12 (the "
              "differences): the codes stop one iteration apart (30 and 31) at round-off"),
    Case("h2co_tdlc_exc_forces_threshold", _h2co,
         **{**TDLC_CASE, "hybrid": {"type": "lc", "screening": "threshold"},
            "tol": {**TOL_TDLC, "scc_errors": 1.0}}, casida=TDLC_EXC,
         note="as DFTB+: the TD part unscreened; neither code's forces are the derivative of "
              "E0 + w (the thresholded ground state's own forces are 1.4e-3 off its energy's "
              "differences).  The SCC tail's errors differ (screening decisions flip with "
              "rounding, as c6h6_lc_threshold), not the iterations or the result"),
    Case("c6h6_tdlc_exc_forces_neighbour", _c6h6_rattled,
         **{**TDLC_CASE, "hybrid": {"type": "lc", "screening": "neighbour",
                                    "cutoff_reduction": 10.0},
            "tol": TOL_TDLC_STRATMANN}, casida=TDLC_EXC,
         note="the meta and para C-H and H-H pairs out of the ground state's exchange; as "
              "DFTB+ (both 4.8e-5 off the differences of E0 + w); DFTB+'s Stratmann accuracy "
              "on 225 transitions limits the comparison to 1e-7"),
    Case("naphthalene_tdlc_exc_forces", _naphthalene,
         **{**TDLC_CASE, "tol": TOL_TDLC_STRATMANN}, casida={**TDLC_EXC, "n": 8, "state": 1},
         note="576 transitions: pydftb's Stratmann iteration and conjugate gradients; DFTB+'s "
              "Stratmann leaves its Omega 1e-8 and its forces 6.6e-7 off (its forces against "
              "the differences of its own E0 + w; pydftb's 1.1e-9)"),
]

# M6h-3a: hybrid functionals of periodic systems at the Gamma point (DFTB+: one k-point, Gamma;
# the references give it as an identity SupercellFolding, which DFTB+ needs for a restart).
# EwaldTolerance 1e-12: DFTB+'s default (1e-9) moves the energies by 8.6e-8
LC_GAMMA = {**LC, "ewald_tolerance": 1e-12}
CAM_GAMMA = {**LC_GAMMA, "skf_set": "ptbp_hcocu_cam", "hybrid": "cam"}
RC8 = {"coulomb_cutoff": 8.0, "gsum_cutoff": 16.0}            # Bohr


def _co2_crystal():
    """Dry ice (Pa-3, a = 5.624 A), rattled: DFTB+'s default Coulomb cutoff is a / 2 = 5.31 Bohr,
    within the 5.8 Bohr between neighbouring molecules' atoms."""
    a, x = 5.624, 0.1185
    c = [(0, 0, 0), (.5, 0, .5), (0, .5, .5), (.5, .5, 0)]
    o = [(x, x, x), (-x, -x, -x), (.5 - x, -x, .5 + x), (.5 + x, x, .5 - x),
         (-x, .5 + x, .5 - x), (x, .5 - x, .5 + x), (.5 + x, .5 - x, -x), (.5 - x, .5 + x, x)]
    at = Atoms("C4O8", scaled_positions=c + o, cell=[a, a, a], pbc=True)
    at.rattle(0.02, seed=7)
    return at


def _diamond8():
    at = bulk("C", "diamond", a=3.567, cubic=True)
    at.rattle(0.02, seed=8)
    return at


def _cu4_cubic():
    at = bulk("Cu", "fcc", a=3.61, cubic=True)
    at.rattle(0.02, seed=9)
    return at


def _o2_box():
    return Atoms("O2", positions=[(0, 0, 0), (1.21, 0.05, 0.03)], cell=[5.0, 5.2, 5.4], pbc=True)


def _co_triclinic():
    """CO in a small triclinic cell: S(Gamma) has intra-atomic off-diagonal elements (5.2e-3)
    from the atoms' overlaps with their own images, which cubic, orthorhombic and hexagonal cells
    cancel by symmetry."""
    cell = [[3.0, 0.0, 0.0], [1.2, 2.8, 0.0], [0.5, 0.7, 3.1]]
    return Atoms("CO", positions=[(0.2, 0.3, 0.1), (1.3, 0.5, 0.4)], cell=cell, pbc=True)


def _h2o_monoclinic():
    cell = [[3.2, 0.0, 0.0], [0.0, 3.0, 0.0], [1.1, 0.0, 2.9]]
    return Atoms("OH2", positions=[(0.1, 0.2, 0.3), (1.05, 0.25, 0.35), (-0.15, 1.1, 0.4)],
                 cell=cell, pbc=True)


B1_NOTE = ("DFTB+'s forces are not the energy's derivative: in the damping region it takes the "
           "HF part's derivative from the long-range polynomial (getdHfTruncatedAndDampedGamma"
           "Value); compared with differences of DFTB+'s energy (SCC tolerance 1e-12; good to "
           "about 1e-8 here: pydftb's exact derivative is 1.2e-8 and 7e-9 off them), and with "
           "that derivative as DFTB+'s (HybridTerms._dftbp_hf_damped_derivative) with DFTB+'s "
           "forces")
TOL_B1 = {**TOL_HYB, "forces": 5e-8, "forces_dftbp": 1e-10, "n_scc": 1}
B6_NOTE = ("DFTB+'s input populations are denseMullikenReal of the lower-triangular DeltaP_in: "
           "of an atom's own block only the column sums, which drops half of the intra-atomic "
           "off-diagonal terms of S(Gamma); compared with that (DFTB._dftbp_gamma_q_in).  "
           "pydftb's Mulliken populations differ from DFTB+'s by {}")
CASES += [
    Case("co2_crystal_lc", _co2_crystal, **LC_GAMMA,
         note="DFTB+'s defaults: Truncated, cutoff a / 2 = 5.31 Bohr, summation radius 10.6 Bohr"),
    Case("co2_crystal_lc_rc8", _co2_crystal, **{**LC_GAMMA, "hybrid": {"type": "lc", **RC8}},
         note="cutoff 8 Bohr: the neighbouring molecules' exchange (65 SCC iterations)"),
    Case("co2_crystal_lc_damped_rc8", _co2_crystal,
         **{**LC_GAMMA, "hybrid": {"type": "lc", "coulomb": "damped", **RC8}},
         note="TruncatedAndDamped: a quintic over (7, 8) Bohr"),
    Case("co2_crystal_lc_neighbour", _co2_crystal,
         **{**LC_GAMMA, "hybrid": {"type": "lc", "screening": "neighbour"}}),
    Case("co2_crystal_lc_neighbour_red10_rc8", _co2_crystal,
         **{**LC_GAMMA, "hybrid": {"type": "lc", "screening": "neighbour",
                                   "cutoff_reduction": 10.0, **RC8}},
         note="the CAM cutoff 5.36 Bohr leaves the neighbouring molecules out of S(Gamma); the "
              "forces with the reduced overlap (DFTB+'s are the energy's derivative here)"),
    Case("co2_crystal_cam", _co2_crystal, **CAM_GAMMA),
    Case("co2_crystal_cam_damped_rc8", _co2_crystal,
         **{**CAM_GAMMA, "hybrid": {"type": "cam", "coulomb": "damped", **RC8},
            "tol": TOL_B1}, scc_tol=1e-12, forces_fd=True,
         note="CAM: " + B1_NOTE + " (7.0e-7 Ha/Bohr).  At 1e-12 the codes stop one iteration "
              "apart (31 and 32) at round-off"),
    Case("co2_crystal_global_damped_rc8", _co2_crystal,
         **{**CAM_GAMMA, "skf_set": "ptbp_hcocu_global",
            "hybrid": {"type": "cam", "coulomb": "damped", **RC8},
            "tol": TOL_B1}, scc_tol=1e-12, forces_fd=True,
         note="a global hybrid: " + B1_NOTE + " (5.6e-6 Ha/Bohr)"),
    Case("diamond8_lc", _diamond8, **LC_GAMMA, note="the default cutoff 3.37 Bohr"),
    Case("diamond8_lc_rc6", _diamond8, **{**LC_GAMMA, "hybrid": {"type": "lc",
                                                                 "coulomb_cutoff": 6.0}},
         note="Rc 6 Bohr, the summation radius twice that (DFTB+'s default)"),
    Case("cu4_lc_1000K", _cu4_cubic, **LC_GAMMA, filling=("fermi", 1000.0),
         note="d shells, a metal"),
    Case("o2_box_lc_spin", _o2_box, **LC_GAMMA,
         spin={"unpaired_electrons": 2.0, "spin_constants": W_HCO}),
    Case("co_triclinic_lc", _co_triclinic, **LC_GAMMA, gamma_q_in_compat=True,
         note=B6_NOTE.format("1.3e-5 e (forces 7.1e-6 Ha/Bohr)")),
    Case("h2o_monoclinic_lc", _h2o_monoclinic, **LC_GAMMA, gamma_q_in_compat=True,
         note=B6_NOTE.format("4.6e-6 e (forces 3.0e-7 Ha/Bohr)")),
    # with the rest of the Hamiltonian
    Case("co2_crystal_lc_plusu", _co2_crystal, **LC_GAMMA, plus_u=U_O),
    Case("co2_crystal_lc_onsite", _co2_crystal, **LC_GAMMA, onsite=ONS),
    Case("co2_crystal_lc_dftb3_d3", _co2_crystal, **LC_GAMMA, dftb3=DFTB3_3OB,
         dispersion=D3BJ_3OB),
    Case("co2_crystal_lc_field_pc", _co2_crystal, **LC_GAMMA, efield=FIELD, point_charges=PC3),
    Case("co2_crystal_lc_mp", _co2_crystal, **LC_GAMMA, filling=("mp", 1000.0, 2)),
    Case("co2_crystal_lc_anderson", _co2_crystal, **LC_GAMMA, mixer="anderson"),
    Case("co2_crystal_lc_custom_hubbards", _co2_crystal, **LC_GAMMA,
         hubbard={"C": 0.4, "O": 0.5}),
    # restarts: the folding record before DeltaP_in
    Case("co2_crystal_lc_read_charges", _co2_crystal, **LC_GAMMA, read_charges={"text": False}),
    Case("o2_box_lc_spin_read_charges_text", _o2_box, **LC_GAMMA, read_charges={"text": True},
         spin={"unpaired_electrons": 2.0, "spin_constants": W_HCO}),
]


# M6h-3b: hybrid functionals of periodic systems at the k-points of a supercell folding (DFTB+
# SupercellFolding without shift; pydftb DFTB+'s set, kpoints.supercell_folding).  DFTB+ gives no
# forces for another Coulomb matrix than Truncated or with spin (Case.dftbp_forces): energies
# only there, pydftb's forces against differences of its energy in test_hybrid_kpoints.py
def _diamond2():
    """Diamond's primitive cell, rattled: atom 0 at a negative fractional coordinate, which
    DFTB+ folds into the cell (the gauge of its DeltaP(k), its lattice sums: B19)."""
    at = bulk("C", "diamond", a=3.567)
    at.rattle(0.03, seed=3)
    return at


def _centred(build, f):
    """``build``'s structure moved by f times the cell's diagonal."""
    def centred():
        at = build()
        at.translate(f * at.cell.sum(0))
        return at
    centred.__name__ = build.__name__ + "_centred"
    return centred


O2_SPIN = {"unpaired_electrons": 2.0, "spin_constants": W_HCO}
TOL_HYB_BANDS = {**TOL_BANDS, **TOL_HYB}
NEIGHBOUR = {"type": "lc", "screening": "neighbour"}
STALE_NOTE = ("DFTB+ 25.1 never updates the block populations at k-points (B20: "
              "getNextInputDensityCplx updates only qInput): compared with pydftb keeping the "
              "starting blocks (DFTB._dftbp_stale_blocks); pydftb's own energy is {}")
CASES += [
    Case("diamond2_lc_k222", _diamond2, **LC_GAMMA, folding=(2, 2, 2),
         note="the default cutoff of the folding (3.89 Bohr); 8 k-points, all their own inverses"),
    Case("diamond2_lc_k333", _diamond2, **LC_GAMMA, folding=(3, 3, 3),
         note="14 of the 27 k-points (k and -k merged)"),
    Case("diamond2_lc_k123", _diamond2, **LC_GAMMA, folding=(1, 2, 3),
         note="an anisotropic folding: DFTB+'s default cutoff scales the Cartesian components "
              "of the lattice vectors (B2), as pydftb's"),
    Case("diamond2_lc_k222_rc8", _diamond2, **{**LC_GAMMA, "hybrid": {"type": "lc", **RC8}},
         folding=(2, 2, 2), note="Rc 8 Bohr beyond the BvK supercell's half spacing"),
    Case("diamond2_lc_k222_damped", _diamond2,
         **{**LC_GAMMA, "hybrid": {"type": "lc", "coulomb": "damped"}}, folding=(2, 2, 2),
         note="TruncatedAndDamped; DFTB+ has no forces for it at k-points"),
    Case("diamond2_lc_k222_full", _diamond2,
         **{**LC_GAMMA, "hybrid": {"type": "lc", "coulomb": "full", "gsum_cutoff": 12.0}},
         folding=(2, 2, 2), fixed_scc=5,
         note="the untruncated kernel over a 12 Bohr parallelepiped about the folded atoms (B19; "
              "about the unfolded ones the energy is 0.1-0.3 Ha off); the SCC does not settle "
              "(the sum grows with the radius): 5 iterations, no forces from DFTB+"),
    Case("diamond2_cam_k222", _diamond2, **CAM_GAMMA, folding=(2, 2, 2)),
    Case("diamond2_global_k222", _diamond2, **{**CAM_GAMMA, "skf_set": "ptbp_hcocu_global"},
         folding=(2, 2, 2), note="a global hybrid (CAM, beta = 0)"),
    Case("diamond2_lc_k312_neighbour", _diamond2, **{**LC_GAMMA, "hybrid": NEIGHBOUR},
         folding=(3, 1, 2),
         note="NeighbourBased mixing the BvK array; DFTB+'s Threshold 1e-100 (pydftb does not "
              "screen; DFTB+'s default 1e-6 moves the energy by up to 1.6e-6 Ha)"),
    Case("diamond2_lc_k222_neighbour_red4", _diamond2,
         **{**LC_GAMMA, "hybrid": {**NEIGHBOUR, "cutoff_reduction": 4.0}}, folding=(2, 2, 2),
         note="the pairs beyond the SK cutoff minus 4 Bohr out of the exchange's overlap"),
    Case("co2_crystal_lc_k211", _co2_crystal, **LC_GAMMA, folding=(2, 1, 1),
         note="most atoms folded into the cell by DFTB+"),
    Case("co2_crystal_lc_k211_neighbour", _co2_crystal, **{**LC_GAMMA, "hybrid": NEIGHBOUR},
         folding=(2, 1, 1)),
    Case("cu_fcc_lc_k333_1000K", _cu_fcc, **LC_GAMMA, folding=(3, 3, 3),
         filling=("fermi", 1000.0), note="d shells, a metal"),
    Case("o2_box_lc_spin_k222", _o2_box, **LC_GAMMA, folding=(2, 2, 2), spin=O2_SPIN,
         note="no forces from DFTB+ with spin at k-points"),
    Case("o2_box_lc_spin_k222_neighbour", _o2_box, **{**LC_GAMMA, "hybrid": NEIGHBOUR},
         folding=(2, 2, 2), spin=O2_SPIN),
    Case("co_triclinic_lc_k222", _co_triclinic, **LC_GAMMA, folding=(2, 2, 2),
         note="the low-symmetry cell of the Gamma point's B6: at k-points DFTB+'s input "
              "populations are DeltaP_in's Mulliken populations"),
    # with the rest of the Hamiltonian
    Case("co2_crystal_lc_k211_plusu_b20", _co2_crystal, **LC_GAMMA, folding=(2, 1, 1),
         plus_u=U_O, stale_blocks_compat=True, note=STALE_NOTE.format("2.2e-3 Ha lower")),
    Case("co2_crystal_lc_k211_onsite_b20", _co2_crystal, **LC_GAMMA, folding=(2, 1, 1),
         onsite=ONS, stale_blocks_compat=True, note=STALE_NOTE.format("3.6e-2 Ha lower")),
    Case("co2_crystal_lc_k211_dftb3_d3", _co2_crystal, **LC_GAMMA, folding=(2, 1, 1),
         dftb3=DFTB3_3OB, dispersion=D3BJ_3OB),
    Case("co2_crystal_lc_k211_field_pc", _co2_crystal, **LC_GAMMA, folding=(2, 1, 1),
         efield=FIELD, point_charges=PC3),
    Case("co2_crystal_lc_k211_mp", _co2_crystal, **LC_GAMMA, folding=(2, 1, 1),
         filling=("mp", 1000.0, 2)),
    Case("co2_crystal_lc_k211_diis", _co2_crystal, **LC_GAMMA, folding=(2, 1, 1), mixer="diis",
         note="DFTB+'s complex mixers on DeltaP(k) are the real ones on [Re | Im]"),
    # restarts: DeltaP_in(k) or the BvK array after the folding record, of the folded atoms
    Case("diamond2_lc_k312_read_charges", _diamond2, **LC_GAMMA, folding=(3, 1, 2),
         read_charges={"text": False, "seed": 4},
         note="run 1 folds atom 0 by another lattice vector than run 2: DFTB+ takes the file's "
              "DeltaP(k) in run 2's gauge, as pydftb's populations_from_file at run 2's "
              "positions"),
    Case("diamond2_lc_k312_neighbour_read_charges_text", _diamond2,
         **{**LC_GAMMA, "hybrid": NEIGHBOUR}, folding=(3, 1, 2),
         read_charges={"text": True, "seed": 5}),
    Case("o2_box_lc_spin_k211_read_charges_text", _o2_box, **LC_GAMMA, folding=(2, 1, 1),
         spin=O2_SPIN, read_charges={"text": True, "seed": 1}),
    # band runs from the SCC's DeltaP(k): H_x at any k
    Case("diamond2_lc_k312_bands", _diamond2, **{**LC_GAMMA, "tol": TOL_HYB_BANDS},
         folding=(3, 1, 2), klines=path(FCC, "GXL", [6, 6]),
         note="DFTB+'s band run (ReadInitialCharges, one iteration) on G-X-L"),
    Case("diamond2_lc_k312_neighbour_bands", _diamond2,
         **{**LC_GAMMA, "hybrid": NEIGHBOUR, "tol": TOL_HYB_BANDS}, folding=(3, 1, 2),
         klines=path(FCC, "GXL", [6, 6])),
    # the bases of the dynamics (dynamics.py): no atom near a face of the cell, so none crosses
    # one along the trajectory (DFTB+ carries DeltaP_in(k) of the atoms as folded at the step
    # before: B21)
    Case("diamond2_centred_lc_k222", _centred(_diamond2, 1 / 8), **LC_GAMMA, folding=(2, 2, 2)),
    Case("co2_crystal_centred_lc_k211", _centred(_co2_crystal, 1 / 4), **LC_GAMMA,
         folding=(2, 1, 1)),
]

# ---- milestone M6h-3c: MinimumImage and the NeighbourBased Threshold at k-points -----------
# MinimumImage: the untruncated kernel over the Wigner-Seitz cell of the supercell (DFTB+ has no
# forces for it, pydftb's MatrixBased ones are tested against differences of its energy); the
# Threshold: DFTB+'s screened increments of DeltaP_in (dftbplus writes the spec's threshold)
MIC = {"type": "lc", "coulomb": "minimum_image"}
MIC_NEIGHBOUR = {**MIC, "screening": "neighbour"}
CAM_MIC = {"type": "cam", "coulomb": "minimum_image"}
T6 = {**NEIGHBOUR, "neighbour_threshold": 1e-6}
# the cases stopped before the frozen exchange's tail (below) end at SCC errors of 2-4e-7, where
# DFTB+'s k-point neighbour gradient, which takes one DeltaP from DeltaP_out(k) and the other
# from the last DeltaP_in(R) (B5), moves the forces by up to 1.7e-8 Ha/Bohr (pydftb: the
# derivative, DeltaP_out twice; with DFTB+'s pair of densities 4e-13 for diamond)
TOL_HYB_T6 = {**TOL_HYB, "forces": 5e-8}
CASES += [
    Case("diamond2_lc_k312_mic", _diamond2, **{**LC_GAMMA, "hybrid": MIC}, folding=(3, 1, 2),
         note="MatrixBased: G(k - q) over the 11 Wigner-Seitz vectors of the 3 x 1 x 2 supercell "
              "(6 classes: the ties on its surface each count once, B9)"),
    Case("diamond2_lc_k222_mic_ws1", _diamond2,
         **{**LC_GAMMA, "hybrid": {**MIC, "wigner_seitz_reduction": 1}}, folding=(2, 2, 2),
         note="WignerSeitzReduction 1: the Wigner-Seitz cell of the 1 x 1 x 1 cell, g = 0 only"),
    Case("diamond2_cam_k222_mic", _diamond2, **{**CAM_GAMMA, "hybrid": CAM_MIC},
         folding=(2, 2, 2)),
    Case("diamond2_global_k222_mic", _diamond2,
         **{**CAM_GAMMA, "skf_set": "ptbp_hcocu_global", "hybrid": CAM_MIC}, folding=(2, 2, 2),
         note="a global hybrid (CAM, beta = 0)"),
    Case("co2_crystal_lc_k211_mic", _co2_crystal, **{**LC_GAMMA, "hybrid": MIC},
         folding=(2, 1, 1)),
    Case("co_triclinic_lc_k222_mic", _co_triclinic, **{**LC_GAMMA, "hybrid": MIC},
         folding=(2, 2, 2)),
    Case("diamond2_lc_k222_mic_neighbour_ws1", _diamond2,
         **{**LC_GAMMA, "hybrid": {**MIC_NEIGHBOUR, "wigner_seitz_reduction": 1}},
         folding=(2, 2, 2),
         note="NeighbourBased: the paths M -> A -> B -> N with A-B in the Wigner-Seitz cell, gamma "
              "at the paths' distances (no forces: not the derivative of its energy, B22)"),
    Case("diamond2_lc_k222_mic_neighbour", _diamond2,
         **{**LC_GAMMA, "hybrid": MIC_NEIGHBOUR, "scc_tol": 3e-6}, folding=(2, 2, 2),
         note="the 19 Wigner-Seitz vectors; DFTB+ about 22 s per iteration.  Stopped at 3e-6, "
              "where the two codes' errors still agree to their printed digits: the slow SCC (59 "
              "iterations to 1e-10) parts at ~1e-6 (rounding amplified by Broyden), the "
              "converged energies agree to 5.7e-12 Ha.  Matrix and Neighbour MinimumImage differ "
              "by 1.1e-3 Ha here (B10)"),
    Case("diamond2_lc_k312_mic_bands", _diamond2, **{**LC_GAMMA, "hybrid": MIC,
                                                     "tol": TOL_HYB_BANDS},
         folding=(3, 1, 2), klines=path(FCC, "GXL", [6, 6])),
    Case("diamond2_lc_k222_mic_neighbour_ws1_bands", _diamond2,
         **{**LC_GAMMA, "hybrid": {**MIC_NEIGHBOUR, "wigner_seitz_reduction": 1},
            "tol": TOL_HYB_BANDS}, folding=(2, 2, 2), klines=path(FCC, "GXL", [6, 6]),
         note="the explicit cells' blocks with their own phases at the band's k-points"),
    # DFTB+'s Threshold of the neighbour screening (default 1e-6 at k-points).  Once every
    # increment of DeltaP_in falls below it (errors of about 1e-6) the exchange stops changing,
    # and Broyden's next step can be nearly zero (|dDeltaP| 1e-12 to 2e-8); its update then
    # divides differences of nearly equal residuals, and the two codes' errors part from there
    # (by 1e-4 to 1 relative, both reaching the same energy to 3e-11 Ha).  scc_tol stops the
    # cases just before that step
    Case("diamond2_lc_k312_neighbour_t6", _diamond2,
         **{**LC_GAMMA, "hybrid": T6, "scc_tol": 5e-7, "tol": TOL_HYB_T6}, folding=(3, 1, 2),
         note="DFTB+'s default Threshold (at 1e-10: 4.8e-6 Ha from the unscreened build, 28 SCC "
              "iterations instead of 17); the 14th iteration's build keeps no increment"),
    Case("diamond2_lc_k312_neighbour_t4", _diamond2,
         **{**LC_GAMMA, "hybrid": {**NEIGHBOUR, "neighbour_threshold": 1e-4}}, folding=(3, 1, 2),
         note="1.7e-5 Ha from the unscreened build"),
    Case("diamond2_lc_k312_neighbour_t6_damped", _diamond2,
         **{**LC_GAMMA, "hybrid": {**T6, "coulomb": "damped"}}, folding=(3, 1, 2)),
    Case("co2_crystal_lc_k211_neighbour_t6", _co2_crystal,
         **{**LC_GAMMA, "hybrid": T6, "scc_tol": 4e-7, "tol": TOL_HYB_T6}, folding=(2, 1, 1),
         note="21 iterations, the last one with no increment kept (the nearly zero step is the "
              "25th)"),
    Case("diamond2_lc_k222_mic_neighbour_t6", _diamond2,
         **{**LC_GAMMA, "hybrid": {**MIC_NEIGHBOUR, "neighbour_threshold": 1e-6}},
         folding=(2, 2, 2), fixed_scc=20,
         note="the screened MinimumImage stalls at errors of 1-2e-5 (DFTB+'s default): 20 "
              "iterations, 7.4e-5 Ha from the unscreened build"),
    Case("diamond2_lc_k312_neighbour_t6_bands", _diamond2,
         **{**LC_GAMMA, "hybrid": T6, "scc_tol": 5e-7, "tol": {**TOL_HYB_BANDS, **TOL_HYB_T6}},
         folding=(3, 1, 2), klines=path(FCC, "GXL", [6, 6]),
         note="DFTB+'s band run: one screened build of the file's DeltaP, not the SCC's "
              "accumulated exchange (the SCC's own levels are compared on its grid)"),
    Case("diamond2_lc_k312_neighbour_t6_read_charges", _diamond2,
         **{**LC_GAMMA, "hybrid": T6, "scc_tol": 5e-7, "tol": TOL_HYB_T6}, folding=(3, 1, 2),
         read_charges={"text": False, "seed": 4},
         note="the first build of a restart is a whole one of the file's DeltaP, as of every "
              "geometry step (updateCoords)"),
    Case("diamond2_centred_lc_k222_neighbour_t6", _centred(_diamond2, 1 / 8),
         **{**LC_GAMMA, "hybrid": T6, "scc_tol": 3e-7, "tol": TOL_HYB_T6}, folding=(2, 2, 2),
         note="the nearly zero step is the 14th"),
]
CASES_BY_NAME = {c.name: c for c in CASES}
