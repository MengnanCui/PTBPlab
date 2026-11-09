"""Collect the M5c VASP runs into PTBP's input: train.xyz, test.xyz (extxyz: energy = VASP's
energy(sigma->0) in eV, forces in eV/A, structure_name, group), E0s_dft.json (the
spin-unpolarized isolated atoms, keyed by atomic number: the reference of DFTB's atoms) and
E0s_spin_polarized.json (for the record).

    python collect.py RUNS_DIR [RUNS_DIR ...] --out DIR

A directory counts when its OUTCAR has "General timing"; a relaxation contributes its last
ionic step only if it is listed in --keep-relax (they are not used by default).
"""
import argparse
import glob
import json
import os
import re

import ase.io
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator


def sigma0_energies(outcar):
    return [float(m) for m in re.findall(r"energy\(sigma->0\) =\s+(-?\d+\.\d+)", open(outcar).read())]


def load(d):
    outcar = os.path.join(d, "OUTCAR")
    if not (os.path.exists(outcar) and "General timing" in open(outcar).read()):
        return None
    at = ase.io.read(outcar, index=-1, format="vasp-out")
    e0 = sigma0_energies(outcar)[-1]
    forces = at.get_forces()
    at.calc = SinglePointCalculator(at, energy=e0, free_energy=e0, forces=forces)
    return at


def group_of(name):
    for g in ("eos_li_bcc", "eos_li_fcc", "eos_li2o2", "eos_li2o", "rat_li_bcc16", "rat_li2o12",
              "rat_li16o", "rat_li8o3", "iface",
              # stage 4 (M5d): dimers and compressed bulk
              "dimer_Li2", "dimer_LiO", "dimer_O2", "cmp_li_bcc", "cmp_li2o2", "cmp_li2o"):
        if name.startswith(g):
            return g
    return "other"


ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
train, test, e0s, e0s_spin = [], [], {}, {}
for root in a.runs:
    for d in sorted(glob.glob(os.path.join(root, "*/"))):
        name = os.path.basename(d.rstrip("/"))
        if name.startswith("relax_") or name.startswith("p141"):
            continue
        at = load(d)
        if at is None:
            print("not finished:", d)
            continue
        if name.startswith("atom_"):             # E0: the spin-unpolarized atoms (as DFTB's)
            key = str(int(at.numbers[0]))
            (e0s if name.endswith("_nospin") else e0s_spin)[key] = at.get_potential_energy()
            continue
        at.info.update(structure_name=name, group=group_of(name))
        (test if "test" in name else train).append(at)
for fn, data in (("train.xyz", train), ("test.xyz", test)):
    ase.io.write(os.path.join(a.out, fn), data, format="extxyz")
json.dump(e0s, open(os.path.join(a.out, "E0s_dft.json"), "w"), indent=2)
json.dump(e0s_spin, open(os.path.join(a.out, "E0s_spin_polarized.json"), "w"), indent=2)
fmax = max(np.abs(x.get_forces()).max() for x in train + test)
print(f"{len(train)} train, {len(test)} test, E0s {e0s}, max |F| {fmax:.3f} eV/A")
