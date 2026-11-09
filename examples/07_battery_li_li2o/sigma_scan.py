"""M5c: rebuild the physrep repulsion of an SKF set for several sigma_rep(Li) (O kept at PTBP's
0.555) and print the Li2 / LiO dimer curves (eV, isolated, Fermi 300 K, relative to 3.0 A).

    PYTHONPATH=<repo> python sigma_scan.py OUT_DIR SKF_A SKF_B
"""
import os, shutil, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ase import Atoms
from tools.physrep import physrep
from pydftb.ase_calc import PyDFTB

def rebuild(src, dst, s_li, s_o=0.55538096):
    if os.path.exists(dst): shutil.rmtree(dst)
    os.makedirs(dst)
    for f in ("Li-Li.skf", "Li-O.skf", "O-Li.skf", "O-O.skf"):
        shutil.copy(os.path.join(src, f), dst)
    cwd = os.getcwd(); os.chdir(dst)
    try:
        pr = physrep(elements=[3, 8], rmin=0.5, rmax=20.0, npoints=700,
                     scale_cov=[s_li, s_o], k_xc=[0.0, 0.0], filename="")
        pr.write_skf("ptbp")
    finally:
        os.chdir(cwd)

def E(skf, at):
    at.calc = PyDFTB(skf + "/", kpts=None, smearing=("fermi", 300.0), scc=True, scc_tol=1e-6,
                     max_scc=500, repulsive="skf", shell_resolved=True)
    return at.get_potential_energy()

if __name__ == "__main__":
    out = sys.argv[1]
    sets = dict(a=sys.argv[2], b30=sys.argv[3])
    rs = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0]
    for L, src in sets.items():
        for s in [0.5, 0.65, 0.8, 1.0, 1.2]:
            d = os.path.join(out, f"{L}_s{s:.2f}")
            rebuild(src, d, s)
            for pair in ("Li2", "LiO"):
                e = [E(d, Atoms(pair, positions=[[0, 0, 0], [0, 0, r]], cell=[20] * 3, pbc=False)) for r in rs]
                e = np.array(e) - e[-1]
                print(f"{L} sLi={s:.2f} {pair} " + " ".join(f"{r}:{v:+.2f}" for r, v in zip(rs, e)), flush=True)
