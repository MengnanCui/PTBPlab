"""Evaluate one DFTB parameter set (a directory of full SKFs) on a DFT dataset of the M5c
battery demo, with PTBP's own DFTB stage (utils.dftb_stage.run_dataset, k-point density 5.0,
T = 0, the pydftb engine) and PTBP's atomization energies (isolated atoms in a 20 A box).

    PYTHONPATH=<repo> python evaluate.py DATA.xyz --e0s E0s_dft.json --skf DIR [--types JSON]
        [--json out.json]

Prints, per group (the ``group`` key of every structure) and overall: the MAE of the
atomization energy per atom, the MAE of the energy per atom relative to the group's mean
(shift-free), and the MAE / RMSE of the force components.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil
import tempfile

import ase
import ase.io
import numpy as np


def atomization_per_atom(energies, structures, e0):
    return np.array([(e - sum(e0[int(z)] for z in at.numbers)) / len(at)
                     for e, at in zip(energies, structures)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--e0s", required=True)
    ap.add_argument("--skf", required=True, help="directory with the full SKFs (band + rep)")
    ap.add_argument("--types", default=None, help="type switches as JSON (PTBP's --types)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-l", default="occupied", choices=["occupied", "highest"],
                    help="MaxAngularMomentum rule (default: the occupied shells, PTBP's "
                         "basis, Li s; 'highest': every parametrised shell, Li s and p)")
    a = ap.parse_args(argv)

    from utils.dftb_stage import run_dataset
    from utils.engines import configure, make_calculator, set_type_switching
    from utils.typemodel import switch_from_declaration

    structures = ase.io.read(a.data, index=":")
    e0_dft = {int(k): v for k, v in json.load(open(a.e0s)).items()}
    configure(engine="pydftb", max_l=a.max_l)
    switches = [switch_from_declaration(d) for d in json.loads(a.types)] if a.types else []
    set_type_switching(switches)

    work = tempfile.mkdtemp(prefix="m5c_eval_")
    for f in glob.glob(os.path.join(a.skf, "*.skf")):
        shutil.copy(f, work)
    here = os.getcwd()
    os.chdir(work)
    try:
        dftb, kept_dft, kept = run_dataset(structures, [1.0] * len(structures), 1, 5.0,
                                           poly_rep="No", SCC="Yes")
        assert len(kept) == len(structures), "some structures had no SKF"
        e0_dftb = {}
        for z in sorted({int(z) for at in structures for z in at.numbers}):
            atom = ase.Atoms(numbers=[z], positions=[[0, 0, 0]], cell=np.eye(3) * 20.0, pbc=True)
            atom.calc = make_calculator(atom, {"kpts": [1, 1, 1], "poly_rep": "No", "SCC": "Yes",
                                               "slater_p": "./", "calc_type": "static"})
            e0_dftb[z] = atom.get_potential_energy()
    finally:
        os.chdir(here)
        shutil.rmtree(work, ignore_errors=True)

    e_dft = np.array([at.get_potential_energy() for at in structures])
    e_dftb = np.array([at.get_potential_energy() for at in dftb])
    ea_dft = atomization_per_atom(e_dft, structures, e0_dft)
    ea_dftb = atomization_per_atom(e_dftb, structures, e0_dftb)
    groups = [at.info.get("group", "all") for at in structures]
    rows = {}
    for g in sorted(set(groups)) + ["ALL"]:
        idx = [i for i, x in enumerate(groups) if g == "ALL" or x == g]
        d = ea_dftb[idx] - ea_dft[idx]
        rel = d - d.mean() if g != "ALL" else None
        if g == "ALL":                      # shift-free per group, pooled
            rel = np.concatenate([(ea_dftb[j] - ea_dft[j]) - np.mean(ea_dftb[jj] - ea_dft[jj])
                                  for gg in sorted(set(groups))
                                  for jj in [[i for i, x in enumerate(groups) if x == gg]]
                                  for j in [jj]])
        df = np.concatenate([(dftb[i].get_forces() - structures[i].get_forces()).ravel()
                             for i in idx])
        rows[g] = {"n": len(idx), "mae_ea": float(np.abs(d).mean()), "mean_ea": float(d.mean()),
                   "mae_rel": float(np.abs(rel).mean()), "mae_f": float(np.abs(df).mean()),
                   "rmse_f": float(np.sqrt(np.mean(df ** 2)))}
    print(f"{'group':16s} {'n':>3s} {'MAE Ea':>9s} {'mean dEa':>9s} {'MAE rel':>9s} "
          f"{'MAE F':>8s} {'RMSE F':>8s}   (eV/atom, eV/A)")
    for g, r in rows.items():
        print(f"{g:16s} {r['n']:3d} {r['mae_ea']:9.4f} {r['mean_ea']:9.4f} {r['mae_rel']:9.4f} "
              f"{r['mae_f']:8.4f} {r['rmse_f']:8.4f}")
    print("E0 DFTB:", e0_dftb)
    if a.json:
        out = {"label": a.label, "skf": os.path.abspath(a.skf), "types": a.types, "rows": rows,
               "per_structure": [{"name": at.info.get("structure_name"), "group": g,
                                  "ea_dft": float(x), "ea_dftb": float(y)}
                                 for at, g, x, y in zip(structures, groups, ea_dft, ea_dftb)]}
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
