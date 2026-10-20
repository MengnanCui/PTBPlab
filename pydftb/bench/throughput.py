"""PTBP's DFTB stage, timed per engine: every structure of a dataset through
``utils.calculator.dftbplus_calculator`` with PTBP's real settings (PTBP
k-grid from ``monkhorstpack2kptdensity``, ``PolynomialRepulsive = Yes``,
SCC tolerance 1e-5, effective T = 0, shell-resolved SCC).

Run inside the PTBP environment (needs ase + dftb+ for the DFTB+ engine)::

    python pydftb/bench/throughput.py dft/dft.xyz /path/to/skf_dir

The SKF directory must contain the files for the dataset's elements.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import warnings

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


def run_all(images, skdir, engine):
    import utils.engines as eng
    from utils.calculator import dftbplus_calculator, monkhorstpack2kptdensity
    eng.configure(engine=engine)
    out = []
    with tempfile.TemporaryDirectory(prefix=f"tp_{engine}_") as wd:
        for f in os.listdir(skdir):
            if f.endswith(".skf"):
                shutil.copy(os.path.join(skdir, f), wd)
        cwd = os.getcwd()
        os.chdir(wd)
        try:
            t0 = time.perf_counter()
            for at in images:
                _, kpts = monkhorstpack2kptdensity(at, 5.0)
                a = at.copy()
                a.calc = dftbplus_calculator(a, dict(calc_type="static", poly_rep="Yes", SCC="Yes",
                                                     kpts=kpts, slater_p="./"))
                out.append((a.get_potential_energy(), a.get_forces()))
            wall = time.perf_counter() - t0
        finally:
            os.chdir(cwd)
    return wall, out


def main():
    from ase.io import read
    warnings.filterwarnings("ignore")
    data = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "dft", "dft.xyz")
    skdir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/claude-0/skf2"
    images = read(data, ":")
    res = {}
    for engine in ("dftbplus", "pydftb"):
        res[engine] = run_all(images, skdir, engine)
        print(f"{engine:9s} {len(images)} structures: {res[engine][0]:.1f} s", flush=True)
    de = max(abs(a[0] - b[0]) for a, b in zip(res["dftbplus"][1], res["pydftb"][1]))
    df = max(np.abs(a[1] - b[1]).max() for a, b in zip(res["dftbplus"][1], res["pydftb"][1]))
    summary = {"structures": len(images), "dftbplus_s": res["dftbplus"][0], "pydftb_s": res["pydftb"][0],
               "max_dE_eV": de, "max_dF_eV_per_A": float(df),
               "settings": "PTBP path: PTBP k-grid (density 5.0), PolynomialRepulsive=Yes, "
                           "SCCTolerance 1e-5, T=0, shell-resolved SCC, pydftb_forces=dftbplus"}
    print(json.dumps(summary, indent=1))
    with open(os.path.join(os.path.dirname(__file__), "results_throughput.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
