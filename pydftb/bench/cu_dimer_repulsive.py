"""Milestone M8d: a Cu-Cu repulsive with first-shell repulsion, from the VASP Cu2 dimer curve.

The classic DFTB construction on one pair: V_rep(r) = E_DFT(r) - E_el(r) - C, with E_el the
table's electronic free energy of Cu2 (``repulsive="none"``, 300 K, no spin), E_DFT a cubic
spline through the dimer group of ``cu_m8b.extxyz`` and C such that V_rep(cutoff) = 0.  It is
written as the Spline block of a copy of ptbp_hcocu's Cu-Cu.skf (cubic intervals with a clamped,
zero slope at the cut-off; below the shortest dimer an exp(-a1 r + a2) head that matches value
and slope), every other file copied unchanged.

    python -m pydftb.bench.cu_dimer_repulsive cu_m8b.extxyz OUT_DIR [--cutoff 3.5]

ptbp_hcocu's own spline ends at 2.63 A and is far too weak there: at 1.7 A the table's Cu2 is
35 eV below DFT.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil

import numpy as np
from ase.io import read

from pydftb.bench import adaptive_cu as ac
from pydftb.constants import BOHR, HARTREE

SRC = os.path.join(ac.REPO, "pydftb", "tests", "data", "skf", "ptbp_hcocu")


def electronic_dimer(r_angstrom, sk):
    """Free energy (eV) of Cu2 at these distances without any repulsive."""
    from pydftb.calculator import DFTB
    calc = DFTB(sk, repulsive="none", temperature=300.0, scc_tol=1e-9, max_scc=1000,
                mixing=0.05)
    out = []
    for r in r_angstrom:
        res = calc.calculate(["Cu", "Cu"], np.array([[0, 0, 0], [0, 0, r]]) / BOHR, None, None)
        if not res.converged:
            raise RuntimeError(f"Cu2 SCC not converged at {r:.3f} A")
        out.append(res.free_energy * HARTREE)
    return np.array(out)


def dimer_data(path):
    r, e = [], []
    for at in read(path, ":"):
        if at.info.get("group") != "dimer":
            continue
        ef = at.info.get("free_energy")
        if ef is None:
            ef = at.calc.get_potential_energy(force_consistent=True)
        r.append(at.get_distance(0, 1))
        e.append(float(ef))
    k = np.argsort(r)
    return np.array(r)[k], np.array(e)[k]


def repulsive(r_dft, e_dft, sk, cutoff, step=0.05):
    """Grid (A) and V_rep (eV) from the shortest dimer to ``cutoff``, and the spline."""
    from scipy.interpolate import CubicSpline
    edft = CubicSpline(r_dft, e_dft)
    grid = np.arange(r_dft[0], cutoff + 1e-9, step)
    grid[-1] = cutoff
    v = edft(grid) - electronic_dimer(grid, sk)
    v -= v[-1]
    spl = CubicSpline(grid / BOHR, v / HARTREE, bc_type=("not-a-knot", (1, 0.0)))
    return grid, v, spl


def spline_block(spl, cutoff_bohr):
    """The SKF Spline block (Hartree, Bohr) of a scipy CubicSpline with knots x."""
    x, c = spl.x, spl.c                     # c[k, i]: coefficient of (r - x_i)^(3 - k)
    v0, d1 = float(spl(x[0])), float(spl(x[0], 1))
    if not (v0 > 0.0 and d1 < 0.0):
        raise ValueError(f"no repulsive exponential head: value {v0}, slope {d1}")
    a1 = -d1 / v0                           # exp(-a1 r + a2): value and slope, a3 = 0
    a2 = np.log(v0) + a1 * x[0]
    a3 = 0.0
    f = repr
    lines = ["Spline", f"{len(x) - 1} {f(float(cutoff_bohr))}",
             f"{f(float(a1))} {f(float(a2))} {f(float(a3))}"]
    for i in range(len(x) - 1):
        cc = [c[3, i], c[2, i], c[1, i], c[0, i]]
        tail = " 0.0 0.0" if i == len(x) - 2 else ""
        lines.append(f"{f(float(x[i]))} {f(float(x[i + 1]))} "
                     + " ".join(f(float(q)) for q in cc) + tail)
    return lines


def write_table(out_dir, block):
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(SRC):
        if f != "Cu-Cu.skf.gz":
            shutil.copy(os.path.join(SRC, f), os.path.join(out_dir, f))
    with gzip.open(os.path.join(SRC, "Cu-Cu.skf.gz"), "rt") as fh:
        lines = fh.read().split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.strip().startswith("Spline"))
    j = next((k for k, ln in enumerate(lines) if ln.strip().startswith("<Documentation>")),
             len(lines))
    new = lines[:i] + block + lines[j:]
    with gzip.open(os.path.join(out_dir, "Cu-Cu.skf.gz"), "wt") as fh:
        fh.write("\n".join(new))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", help="cu_m8b.extxyz (its dimer group)")
    ap.add_argument("out_dir")
    ap.add_argument("--cutoff", type=float, default=3.5, help="A")
    args = ap.parse_args(argv)
    from pydftb.params import ParamSet
    sk = ParamSet(SRC, max_l={"Cu": 2}, shell_resolved=True).skset(["Cu"])
    r, e = dimer_data(args.data)
    grid, v, spl = repulsive(r, e, sk, args.cutoff)
    write_table(args.out_dir, spline_block(spl, args.cutoff / BOHR))
    print("V_rep (eV): " + " ".join(f"{a:.2f}:{b:.3f}" for a, b in zip(grid[::4], v[::4])))
    with open(os.path.join(args.out_dir, "dimer_repulsive.json"), "w") as fh:
        json.dump({"r_dft": r.tolist(), "e_dft": e.tolist(), "grid": grid.tolist(),
                   "v_rep": v.tolist(), "cutoff": args.cutoff}, fh, indent=1)
    print(f"wrote {args.out_dir} (Cu-Cu spline from {r[0]:.2f} to {args.cutoff:.2f} A)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
