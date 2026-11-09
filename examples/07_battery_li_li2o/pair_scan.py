"""M5c: physrep with a separate sigma for the Li-Li pair (physrep pair_scale), on top of a fixed
electronic part, and/or a short-range Li-Li wall (physrep pair_wall). Builds one SKF directory
per spec and prints the Li2 / LiO dimer curves (eV relative to 3.0 A).

    PYTHONPATH=<repo> python pair_scan.py OUT_DIR NAME=SKF_DIR:SIGMA_LI:SIGMA_LILI[:E,R_E,R_C[,N]] [...]

SIGMA_LILI "-" keeps the element sigma for Li-Li; the wall is E (eV) at R_E, zero at R_C (A),
with an optional fourth number, the exponent (default 4).
"""
import os
import shutil
import sys
import warnings

import numpy as np
from ase import Atoms

from pydftb.ase_calc import PyDFTB
from tools.physrep import physrep

warnings.filterwarnings("ignore")
SIGMA_O = 0.55538096                                     # PTBP's published O


def rebuild(src, dst, s_li, s_lili, wall=None):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for f in ("Li-Li.skf", "Li-O.skf", "O-Li.skf", "O-O.skf"):
        shutil.copy(os.path.join(src, f), dst)
    cwd = os.getcwd()
    os.chdir(dst)                                        # physrep splits the file name at '-'
    try:
        physrep(elements=[3, 8], rmin=0.5, rmax=20.0, npoints=700, scale_cov=[s_li, SIGMA_O],
                k_xc=[0.0, 0.0], filename="",
                pair_scale=None if s_lili is None else {(3, 3): s_lili},
                pair_wall=None if wall is None else {(3, 3): wall}).write_skf("ptbp")
    finally:
        os.chdir(cwd)


def energy(skf, at):
    at.calc = PyDFTB(skf + "/", kpts=None, smearing=("fermi", 300.0), scc=True, scc_tol=1e-6,
                     max_scc=500, repulsive="skf", shell_resolved=True)
    return at.get_potential_energy()


if __name__ == "__main__":
    out = sys.argv[1]
    rs = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0]
    for spec in sys.argv[2:]:
        name, rest = spec.split("=", 1)
        src, s_li, s_lili, *w = rest.split(":")
        d = os.path.join(out, name)
        wall = tuple(float(v) for v in w[0].split(",")) if w else None
        rebuild(src, d, float(s_li), None if s_lili == "-" else float(s_lili), wall)
        for pair in ("Li2", "LiO"):
            e = np.array([energy(d, Atoms(pair, positions=[[0, 0, 0], [0, 0, r]], cell=[20] * 3,
                                          pbc=False)) for r in rs])
            e -= e[-1]
            print(f"{name} {pair} " + " ".join(f"{r}:{v:+.2f}" for r, v in zip(rs, e)), flush=True)
        print(f"{name} dir {d}", flush=True)
