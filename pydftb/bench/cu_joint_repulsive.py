"""Milestone M8d: a Cu-Cu repulsive fitted to dimers, clusters and bulk together.

The dimer-only repulsive (``cu_dimer_repulsive``) reproduces Cu2 but counts every bond in bulk
(fcc about 1.5 eV/atom too high).  Here V_rep(r) is a cubic B-spline on [r_lo, cutoff] (value
and slope zero at the cut-off) fitted by linear least squares to the training structures of
``cu_dft_fit`` (every group):

    (E_DFT - E_el) / N = sum_pairs V(r_ij) / N + c

with E_el the table's electronic free energy (``repulsive="none"``, 300 K) and a penalty on the
second differences of the spline coefficients.  It is written, with an exponential head below
r_lo, as the Spline block of a copy of ptbp_hcocu (``cu_dimer_repulsive.write_table``), and the
CN range of the training set is reported for ``cu_dft_fit --cn-clamp``.

    python -m pydftb.bench.cu_joint_repulsive cu_m8b.extxyz OUT_DIR [--smooth 1e-3]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.interpolate import BSpline, CubicSpline

from pydftb.bench import adaptive_cu as ac
from pydftb.bench import cu_dft_fit as cf
from pydftb.bench import cu_dimer_repulsive as cdr
from pydftb.constants import BOHR, HARTREE


def pairs(atoms, cutoff):
    """Every pair distance below ``cutoff`` (A) once, periodic images included."""
    from ase.neighborlist import neighbor_list
    d = neighbor_list("d", atoms, cutoff)
    return d, len(atoms)          # each pair twice (i->j and j->i): halved in the design row


def electronic(samples, sk, processes):
    """E_el per atom (eV) of every sample: the table without its repulsive."""
    from pydftb import adaptive_fit as af
    opts = {**ac.OPTIONS, "repulsive": "none"}
    af._WORK.update(models=(), samples=list(samples), sk=sk, options=opts)
    out = af._pool_map(_el_work, list(range(len(samples))), processes)
    return np.array(out)


def _el_work(k):
    from pydftb import adaptive_fit as af
    w = af._WORK
    # a zero-coefficient model: the plain table (its 1.6 A wall is zero for all the data)
    return af.evaluate_sample(cf.models("fixed", 1.6), w["samples"][k], w["sk"],
                              w["options"])[0]


def basis(r_lo, cutoff, n_knots):
    """Cubic B-splines on [r_lo, cutoff] whose value and slope vanish at the cut-off."""
    inner = np.linspace(r_lo, cutoff, n_knots)
    t = np.concatenate([[r_lo] * 3, inner, [cutoff] * 3])
    n = len(t) - 4
    keep = list(range(n - 2))     # the last two vanish in value and slope only together: drop
    return t, n, keep


def design(samples, t, n, keep, cutoff):
    rows = []
    for s in samples:
        d, nat = pairs(s.atoms, cutoff)
        row = np.zeros(len(keep))
        if len(d):
            for c, k in enumerate(keep):
                coef = np.zeros(n)
                coef[k] = 1.0
                row[c] = 0.5 * BSpline(t, coef, 3, extrapolate=False)(d).sum() / nat
        rows.append(np.nan_to_num(row))
    return np.array(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", help="cu_m8b.extxyz")
    ap.add_argument("out_dir")
    ap.add_argument("--cutoff", type=float, default=3.5, help="A")
    ap.add_argument("--r-lo", type=float, default=1.6, help="A, below the shortest bond")
    ap.add_argument("--knots", type=int, default=14)
    ap.add_argument("--smooth", type=float, default=1e-3, help="second-difference penalty")
    ap.add_argument("--test-every", type=int, default=5)
    ap.add_argument("--processes", type=int, default=6)
    args = ap.parse_args(argv)
    from pydftb.params import ParamSet
    sk = ParamSet(cdr.SRC, max_l={"Cu": 2}, shell_resolved=True).skset(["Cu"])
    train, _ = cf.split(cf.load(args.data), args.test_every)       # the test set stays out
    e_el = electronic(train, sk, args.processes)
    y = np.array([s.energy for s in train]) - e_el
    t, n, keep = basis(args.r_lo, args.cutoff, args.knots)
    a = design(train, t, n, keep, args.cutoff)
    a1 = np.hstack([a, np.ones((len(train), 1))])
    m = len(keep)
    dd = np.diff(np.eye(m), 2, axis=0)
    reg = np.hstack([np.sqrt(args.smooth) * dd, np.zeros((m - 2, 1))])
    sol = np.linalg.lstsq(np.vstack([a1, reg]), np.concatenate([y, np.zeros(m - 2)]),
                          rcond=None)[0]
    r = a1 @ sol - y
    groups = [cf.group_of(s) for s in train]
    by = {g: float(np.sqrt(np.mean(r[[k for k, q in enumerate(groups) if q == g]] ** 2)))
          for g in dict.fromkeys(groups)}
    print("pair-only fit (train), meV/atom: " + ", ".join(f"{g} {v * 1000:.0f}"
                                                           for g, v in by.items()), flush=True)
    coef = np.zeros(n)
    coef[keep] = sol[:-1]
    spl_b = BSpline(t, coef, 3, extrapolate=False)
    grid = np.linspace(args.r_lo, args.cutoff, 120)
    v = np.nan_to_num(spl_b(grid))
    v[-1] = 0.0
    spl = CubicSpline(grid / BOHR, v / HARTREE, bc_type=("not-a-knot", (1, 0.0)))
    cdr.write_table(args.out_dir, cdr.spline_block(spl, args.cutoff / BOHR))
    probe = cf.models("adaptive", 1.6)[0]
    cns = np.concatenate([probe.coordination_numbers(s.atoms.get_chemical_symbols(),
                                                     s.atoms.positions / BOHR,
                                                     s.atoms.cell.array / BOHR if any(s.atoms.pbc)
                                                     else None,
                                                     s.atoms.pbc if any(s.atoms.pbc) else None)[0]
                          for s in train])
    print("V_rep (eV): " + " ".join(f"{q:.2f}:{w:.3f}" for q, w in zip(grid[::10], v[::10])))
    print(f"training CN range {cns.min():.2f} .. {cns.max():.2f}", flush=True)
    with open(os.path.join(args.out_dir, "joint_repulsive.json"), "w") as fh:
        json.dump({"grid": grid.tolist(), "v_rep": v.tolist(), "by_group": by,
                   "cn_range": [float(cns.min()), float(cns.max())], "smooth": args.smooth,
                   "knots": args.knots, "r_lo": args.r_lo, "cutoff": args.cutoff}, fh, indent=1)
    print(f"wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
