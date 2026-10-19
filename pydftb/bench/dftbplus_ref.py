"""Run DFTB+ on the same input as pydftb, for validation and timing."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time

import numpy as np

from pydftb.constants import BOHR

DFTBPLUS = os.environ.get("DFTBPLUS", "/opt/mm/dftb/bin/dftb+")


def write_gen(path, symbols, pos_ang, cell_ang=None):
    elems = list(dict.fromkeys(symbols))
    with open(path, "w") as f:
        f.write(f"{len(symbols)} {'S' if cell_ang is not None else 'C'}\n")
        f.write(" ".join(elems) + "\n")
        for k, (s, p) in enumerate(zip(symbols, pos_ang)):
            f.write(f"{k+1} {elems.index(s)+1} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}\n")
        if cell_ang is not None:
            f.write("0.0 0.0 0.0\n")
            for v in cell_ang:
                f.write(f"{v[0]:.12f} {v[1]:.12f} {v[2]:.12f}\n")


def run(symbols, pos_ang, cell_ang=None, skdir=".", max_l=None, scc=True, kpts=None,
        temperature=0.0, shell_resolved=False, threads=1, scc_tol=1e-10, workdir=None,
        extra_ham="", timing=False):
    workdir = workdir or tempfile.mkdtemp(prefix="dftbp_")
    write_gen(os.path.join(workdir, "geo.gen"), symbols, pos_ang, cell_ang)
    maxl = "\n".join(f'    {e} = "{"spdf"[l]}"' for e, l in max_l.items())
    kblock = ""
    if cell_ang is not None:
        if kpts is None:
            kblock = "  KPointsAndWeights = {\n 0.0 0.0 0.0 1.0\n }\n"
        else:
            k, w = kpts
            lines = "\n".join(f"  {a:.16f} {b:.16f} {c:.16f} {x:.16f}" for (a, b, c), x in zip(k, w))
            kblock = f"  KPointsAndWeights = {{\n{lines}\n  }}\n"
    hsd = f"""Geometry = GenFormat {{
  <<< "geo.gen"
}}
Hamiltonian = DFTB {{
  SCC = {"Yes" if scc else "No"}
  SCCTolerance = {scc_tol}
  MaxSCCIterations = 500
  ShellResolvedSCC = {"Yes" if shell_resolved else "No"}
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{os.path.abspath(skdir)}/"
    Separator = "-"
    Suffix = ".skf"
  }}
  MaxAngularMomentum {{
{maxl}
  }}
  Filling = Fermi {{
    Temperature [K] = {temperature}
  }}
{kblock}{extra_ham}
}}
Analysis {{
  CalculateForces = Yes
  MullikenAnalysis = Yes
}}
Options {{
  WriteDetailedOut = Yes
  TimingVerbosity = {2 if timing else 0}
}}
ParserOptions {{
  ParserVersion = 12
}}
Parallel {{
  UseOmpThreads = Yes
}}
"""
    with open(os.path.join(workdir, "dftb_in.hsd"), "w") as f:
        f.write(hsd)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    t0 = time.perf_counter()
    p = subprocess.run([DFTBPLUS], cwd=workdir, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(p.stdout[-3000:] + p.stderr[-3000:])
    out = open(os.path.join(workdir, "detailed.out")).read()
    res = {"wall": wall, "stdout": p.stdout, "dir": workdir}
    for key, name in [("energy", "Total energy"), ("free_energy", "Total Mermin free energy"),
                      ("e_rep", "Repulsive energy"), ("e_scc", "Energy SCC"),
                      ("e_band", "Energy H0"), ("fermi", "Fermi level"),
                      ("ts", "-TS")]:
        m = re.search(rf"^{re.escape(name)}:\s+(\S+)\s+H", out, re.M)
        if m:
            res[key] = float(m.group(1))
    m = re.search(r"Total Forces\s*\n((?:\s+\d+\s+\S+\s+\S+\s+\S+\s*\n)+)", out)
    if m:
        res["forces"] = np.array([[float(x) for x in ln.split()[1:4]]
                                  for ln in m.group(1).strip().splitlines()])
    m = re.search(r"Atom\s+Population\s*\n((?:\s+\d+\s+\S+\s*\n)+)", out)
    if m:
        res["charges"] = np.array([float(ln.split()[1]) for ln in m.group(1).strip().splitlines()])
    return res
