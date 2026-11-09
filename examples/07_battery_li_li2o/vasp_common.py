"""Shared VASP settings and input writer for the Li / Li2O battery demo (milestone M5c)."""
from __future__ import annotations

import os

import ase.io
import numpy as np

BASE = {
    "PREC": "Accurate", "ENCUT": 800, "EDIFF": 1e-7, "LREAL": ".FALSE.",
    "ISMEAR": 0, "SIGMA": 0.05, "KSPACING": 0.15, "KGAMMA": ".TRUE.",
    "ISPIN": 1, "NELM": 200, "ALGO": "Normal", "IBRION": -1, "NSW": 0, "ISIF": 2,
    "LWAVE": ".FALSE.", "LCHARG": ".FALSE.",
}
POTCARS = {"Li": "Li_sv", "O": "O"}


def antifluorite(a: float):
    """Li2O primitive cell: O on the fcc site, the two Li at (1/4, 1/4, 1/4) and (3/4, 3/4, 3/4)."""
    from ase import Atoms
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    return Atoms("OLi2", scaled_positions=[[0, 0, 0], [0.25] * 3, [0.75] * 3], cell=cell, pbc=True)


def li2o2(a: float = 3.14, c: float = 7.65, z: float = 0.649):
    """Li2O2, P6_3/mmc (Foeppl): Li 2a, Li 2c, O 4f (1/3, 2/3, z); 8 atoms, O-O 1.55 A."""
    from ase.spacegroup import crystal
    return crystal(["Li", "Li", "O"], basis=[(0, 0, 0), (1 / 3, 2 / 3, 0.25), (1 / 3, 2 / 3, z)],
                   spacegroup=194, cellpar=[a, a, c, 90, 90, 120])


def write_case(directory: str, atoms, gamma_only: bool = False, **overrides) -> None:
    """Write POSCAR (elements sorted, symbols line) and INCAR; POTCAR is built on the cluster
    from POSCAR line 6 by run_all.sh."""
    os.makedirs(directory, exist_ok=True)
    order = sorted(set(atoms.get_chemical_symbols()), key=lambda s: ["Li", "O"].index(s))
    idx = [i for el in order for i, s in enumerate(atoms.get_chemical_symbols()) if s == el]
    atoms = atoms[idx]
    ase.io.write(os.path.join(directory, "POSCAR"), atoms, format="vasp", direct=True,
                 sort=False, vasp5=True)
    incar = dict(BASE)
    incar.update(overrides)
    if gamma_only:                     # vasp_gam, one k-point (isolated atoms)
        incar.pop("KSPACING"), incar.pop("KGAMMA")
        with open(os.path.join(directory, "KPOINTS"), "w") as fh:
            fh.write("Gamma\n0\nGamma\n1 1 1\n0 0 0\n")
        open(os.path.join(directory, "USE_GAM"), "w").close()
    with open(os.path.join(directory, "INCAR"), "w") as fh:
        for k, v in incar.items():
            if v is not None:
                fh.write(f"{k} = {v}\n")
    with open(os.path.join(directory, "POTCARS"), "w") as fh:
        fh.write(" ".join(POTCARS[s] for s in order) + "\n")
