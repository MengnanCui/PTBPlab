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

from ..constants import BOHR
from ..kpoints import monkhorst_pack, reduce_time_reversal

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "data")
SKF_SETS = {
    "ptbp_hcocu": os.path.join(DATA, "skf", "ptbp_hcocu"),
    "orbit": os.path.join(DATA, "skf", "orbit"),
    "orbit_poly": os.path.join(DATA, "skf", "orbit_poly"),   # make_poly_set.py
}
REF_DIR = os.path.join(DATA, "ref")
MAXL = {"H": 0, "C": 1, "O": 1, "Cu": 2, "Li": 0}
# DFTB+'s finite-difference H0/S derivative error at this water geometry is 1.2e-6 Ha/Bohr:
# the H-H distance lies 4e-6 Bohr from SK grid node 153, inside DFTB+'s step of 1.22e-4
# (pydftb's exact forces agree with its own energy finite differences to 2e-9 at h = 1e-6)
TOL_DFTBP_FD = {"energy": 1e-6, "forces": 2e-6, "charges": 1e-6}
# DFTB3: 3ob Hubbard derivatives (Ha/e) and its standard settings; Cu values are for testing only
UD_3OB = {"H": -0.1857, "C": -0.1492, "O": -0.1575}
DFTB3_3OB = {"third_order": "full", "hubbard_derivs": UD_3OB, "h_damping": 4.0}
# spin constants W (Ha): typical PBE values for H and O; Cu values are for testing only
W_SPIN = {"H": -0.072, "O": [[-0.035, -0.030], [-0.030, -0.028]],
          "Cu": [[-0.016, -0.012, -0.004], [-0.012, -0.040, -0.002], [-0.004, -0.002, -0.017]]}
# dispersion (milestone M2d).  D3(BJ) and D4 parameters of DFTB3/3ob from the parameter tables of
# s-dftd3 ("dftb3") and dftd4 ("dftb_3ob", with and without the three-body term)
D3BJ_3OB = {"model": "dftd3", "damping": "beckejohnson", "a1": 0.5719, "a2": 3.6017,
            "s6": 1.0, "s8": 0.5883}
D4_3OB = {"model": "dftd4", "s6": 1.0, "s8": 0.6635015, "s9": 1.0, "a1": 0.5523240,
          "a2": 4.3537076}
# periodic D4 against DFTB+: two-body only and no charge dependence, because DFTB+'s periodic
# EEQ charges and three-body sum are wrong (docs/design/M2d-dispersion.md)
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
    tol: dict = field(default_factory=lambda: {"energy": 1e-6, "forces": 1e-6, "charges": 1e-6})
    note: str = ""

    def kpoints(self):
        if self.kgrid is None:
            return None
        return reduce_time_reversal(*monkhorst_pack(self.kgrid))


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


def _o2():
    return Atoms("O2", positions=[[0, 0, 0], [0.05, 0.1, 1.21]])


def _oh():
    return Atoms("OH", positions=[[0, 0, 0], [0.1, 0.05, 0.97]])


def _li_bcc():
    return bulk("Li", "bcc", a=3.50)     # every shell >= 4e-3 Bohr from a grid node


def _li_bcc_node():
    return bulk("Li", "bcc", a=3.45)     # the a*sqrt(2) shell is 4.3e-5 Bohr from grid node 461


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
]
CASES_BY_NAME = {c.name: c for c in CASES}
