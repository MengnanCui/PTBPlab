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

from ..kpoints import monkhorst_pack, reduce_time_reversal

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "data")
SKF_SETS = {
    "ptbp_hcocu": os.path.join(DATA, "skf", "ptbp_hcocu"),
    "orbit": os.path.join(DATA, "skf", "orbit"),
    "orbit_poly": os.path.join(DATA, "skf", "orbit_poly"),   # make_poly_set.py
}
REF_DIR = os.path.join(DATA, "ref")
MAXL = {"H": 0, "C": 1, "O": 1, "Cu": 2, "Li": 0}
# DFTB+'s finite-difference H0/S derivative error at this water geometry is 1.2e-6 Ha/Bohr
# (pydftb agrees with its own energy finite differences to 3e-9)
TOL_DFTBP_FD = {"energy": 1e-6, "forces": 2e-6, "charges": 1e-6}


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
]
CASES_BY_NAME = {c.name: c for c in CASES}
