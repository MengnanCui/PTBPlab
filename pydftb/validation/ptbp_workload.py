"""PTBP's Cu DFTB stage without the dataset file (milestone M7e), for the GPU self-check.

The six phases of ``dft/dft.xyz`` (PTBP's Cu dataset: dimer, AA-graphite, diamond, beta-Sn,
bcc, fcc) at its 11 EOS volumes V/V0 = 0.80 + 0.04 i, the cell scaled by the cube root and
the fractional positions of the centre structure (within 1e-14 A and 3e-9 of the dataset);
PTBP's k-point grids (Gamma only for the three phases whose Boltzmann weight at 3000 K is
below 0.1, otherwise ``int(5 * 2 pi |b_i|)`` per axis, ``utils.dftb_stage.kpoint_grid``); and
the arguments of PTBP's pydftb calculator (``utils.engines._pydftb_calculator``), with the
Cu-Cu table of the bundled ``ptbp_hcocu`` set instead of the tables of a BO step.

    from pydftb.validation.ptbp_workload import structures, calculator
    for s in structures(quick=True):
        s.atoms.calc = calculator(s)
        s.atoms.get_forces()            # energy, forces and stress, as PTBP's DFTB stage
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from ase import Atoms

# centre cells (Angstrom, rows) and fractional positions of the EOS groups of dft/dft.xyz
PHASES = {
    "dimer": ([[9.9, 0.0, 0.0], [0.0, 7.92, 0.0], [0.0, 0.0, 7.92]],
              [[-0.0120238323232, 0.0, 0.0], [0.212023832323, 0.0, 0.0]]),
    "AA-graphite": ([[1.9794010365823, -3.42842315600249, 0.0],
                     [1.9794010365823, 3.42842315600249, 0.0],
                     [0.0, 0.0, 5.62329839938153]],
                    [[0.0, 0.0, 0.479357143896], [0.333333331476, 0.666666665198, 0.479357143896]]),
    "diamond": ([[0.0, 2.67332480617239, 2.67332480617239],
                 [2.67332480617239, 0.0, 2.67332480617239],
                 [2.67332480617239, 2.67332480617239, 0.0]],
                [[0.124999999711, 0.124999999711, 0.124999999711],
                 [0.87499999985, 0.87499999985, 0.87499999985]]),
    "beta-Sn": ([[0.0, 4.59935716491857, 0.0],
                 [2.29967858245928, -2.29967858245928, 1.25473220644178],
                 [2.29967858245928, 2.29967858245928, -1.25473220644178]],
                [[0.875000004715, 0.125000003343, -0.125000003343],
                 [0.125000000153, 0.875000001525, 0.124999997406]]),
    "bcc": ([[-1.44403203735787, 1.44403203735787, 1.44403203735787],
             [1.44403203735787, -1.44403203735787, 1.44403203735787],
             [1.44403203735787, 1.44403203735787, -1.44403203735787]],
            [[0.0, 0.0, 0.0]]),
    "fcc": ([[0.0, 1.81827300633046, 1.81827300633046],
             [1.81827300633046, 0.0, 1.81827300633046],
             [1.81827300633046, 1.81827300633046, 0.0]],
            [[0.0, 0.0, 0.0]]),
}
GAMMA_ONLY = ("dimer", "AA-graphite", "diamond")   # Boltzmann weight < 0.1 at 3000 K
EOS_POINTS = 11
CENTRE = 5
QUICK_EOS = (0, 5, 10)
KPT_DENSITY = 5.0
ORBITALS = {"Cu": 9}
# the pydftb calculator of PTBP's DFTB stage (utils.engines._pydftb_calculator, default
# settings: T = 0 Fermi filling, polynomial repulsive with zero coefficients, shell-resolved
# SCC to 1e-5, DFTB+'s 0.01 Bohr finite-difference forces, one BLAS thread, no k threads)
PTBP_OPTIONS = {"time_reversal": True, "smearing": ("fermi", 0.0), "scc": True, "scc_tol": 1e-5,
                "max_scc": 1000, "repulsive": "polynomial", "shell_resolved": True,
                "fd_delta": 0.01, "threads": 1, "k_threads": 1}


@dataclass
class Structure:
    name: str                        # e.g. "fcc 07"
    phase: str
    eos: int                         # EOS point 0..10 (5 is the centre)
    atoms: Atoms
    kpts: tuple                      # Monkhorst-Pack grid ((1, 1, 1): Gamma only)
    norb: int

    @property
    def gamma(self) -> bool:
        return self.kpts == (1, 1, 1)


def volume_ratio(eos: int) -> float:
    return 0.80 + 0.04 * eos


def kpoint_grid(atoms: Atoms, density: float = KPT_DENSITY) -> tuple:
    """PTBP's grid (``utils.calculator.monkhorstpack2kptdensity``) for a periodic cell."""
    rec = atoms.cell.reciprocal()
    return tuple(max(1, int(density * (2 * math.pi * math.sqrt((rec[i] ** 2).sum()))))
                 for i in range(3))


def structures(quick: bool = False, phases=None) -> list:
    """The stage's structures in the dataset's order (phase by phase, EOS point 0..10);
    ``quick``: EOS points 0, 5 and 10 only."""
    out = []
    for phase, (cell, frac) in PHASES.items():
        if phases is not None and phase not in phases:
            continue
        cell, frac = np.array(cell, float), np.array(frac, float)
        for i in (QUICK_EOS if quick else range(EOS_POINTS)):
            c = cell * volume_ratio(i) ** (1.0 / 3.0) / volume_ratio(CENTRE) ** (1.0 / 3.0)
            atoms = Atoms(["Cu"] * len(frac), positions=frac @ c, cell=c, pbc=True)
            grid = (1, 1, 1) if phase in GAMMA_ONLY else kpoint_grid(atoms)
            out.append(Structure(f"{phase} {i:02d}", phase, i, atoms, grid,
                                 sum(ORBITALS[s] for s in atoms.get_chemical_symbols())))
    return out


def skf_dir() -> str:
    """The bundled SK set whose Cu-Cu table stands in for a BO step's tables."""
    from .cases import SKF_SETS
    return SKF_SETS["ptbp_hcocu"]


def calculator(structure: Structure, skdir: str | None = None, **overrides):
    """PTBP's pydftb ASE calculator for ``structure`` (``overrides``: e.g. ``backend``,
    ``device``, ``dtype``, ``k_threads``)."""
    from ..ase_calc import PyDFTB
    opts = dict(PTBP_OPTIONS, **overrides)
    return PyDFTB(skdir or skf_dir(), kpts=structure.kpts, **opts)
