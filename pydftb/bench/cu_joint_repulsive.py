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

With ``--force-weight W`` the structures' DFT forces enter the same linear fit (the repulsive
force is linear in the spline coefficients; target F_DFT - F_el, each structure's rows scaled by
W / sqrt(3N)); ``--monotonic`` keeps V_rep non-negative and non-increasing.  Energy alone gives a
wiggly spline and, after the adaptive fit, a spurious Cu2 well at 1.8 A.

    python -m pydftb.bench.cu_joint_repulsive cu_m8b.extxyz OUT_DIR [--smooth 1e-3]
        [--force-weight 0.1 --monotonic]
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
    """E_el per atom (eV) and the electronic forces (eV/A; None where the sample has no DFT
    forces) of every sample: the table without its repulsive."""
    from pydftb import adaptive_fit as af
    opts = {**ac.OPTIONS, "repulsive": "none"}
    af._WORK.update(models=(), samples=list(samples), sk=sk, options=opts)
    out = af._pool_map(_el_work, list(range(len(samples))), processes)
    return np.array([o[0] for o in out]), [o[2] for o in out]


def _el_work(k):
    from pydftb import adaptive_fit as af
    w = af._WORK
    s = w["samples"][k]
    # a zero-coefficient model: the plain table (its 1.6 A wall is zero for all the data)
    return af.evaluate_sample(cf.models("fixed", 1.6), s, w["sk"], w["options"],
                              forces=s.forces is not None)


def basis(r_lo, cutoff, n_knots):
    """Cubic B-splines on [r_lo, cutoff] whose value and slope vanish at the cut-off."""
    inner = np.linspace(r_lo, cutoff, n_knots)
    t = np.concatenate([[r_lo] * 3, inner, [cutoff] * 3])
    n = len(t) - 4
    keep = list(range(n - 2))     # the last two vanish in value and slope only together: drop
    return t, n, keep


def _unit(t, n, k):
    coef = np.zeros(n)
    coef[k] = 1.0
    return BSpline(t, coef, 3, extrapolate=False)


def design(samples, t, n, keep, cutoff):
    """Energy rows: E_rep / N per basis function."""
    rows = []
    for s in samples:
        d, nat = pairs(s.atoms, cutoff)
        row = np.zeros(len(keep))
        if len(d):
            for c, k in enumerate(keep):
                row[c] = 0.5 * np.nan_to_num(_unit(t, n, k)(d)).sum() / nat
        rows.append(row)
    return np.array(rows)


def force_design(atoms, t, n, keep, cutoff):
    """(3N, n_basis): the repulsive force on every atom per basis function,
    F_i = sum_j V'(r_ij) (r_j - r_i) / r_ij (the neighbour list has each pair both ways)."""
    from ase.neighborlist import neighbor_list
    i, d, vec = neighbor_list("idD", atoms, cutoff)
    out = np.zeros((len(atoms), 3, len(keep)))
    for c, k in enumerate(keep):
        dv = np.nan_to_num(_unit(t, n, k).derivative()(d))
        np.add.at(out[:, :, c], i, (dv / d)[:, None] * vec)
    return out.reshape(-1, len(keep))


def solve(a_e, y_e, a_f, y_f, m, smooth, monotonic):
    """Least squares for (spline coefficients, constant).  ``monotonic``: the coefficients
    non-increasing and non-negative (c_k = sum_{j >= k} u_j, u >= 0), which makes V_rep >= 0 and
    non-increasing (a B-spline's slope is a positive combination of its coefficient
    differences)."""
    from scipy.optimize import lsq_linear
    u = np.triu(np.ones((m, m))) if monotonic else np.eye(m)    # c = u @ p
    dd = np.diff(np.eye(m), 2, axis=0) @ u
    rows = [np.hstack([a_e @ u, np.ones((len(y_e), 1))]),
            np.hstack([np.sqrt(smooth) * dd, np.zeros((m - 2, 1))])]
    rhs = [y_e, np.zeros(m - 2)]
    if a_f is not None and len(y_f):
        rows.insert(1, np.hstack([a_f @ u, np.zeros((len(y_f), 1))]))
        rhs.insert(1, y_f)
    lo = np.concatenate([np.zeros(m) if monotonic else np.full(m, -np.inf), [-np.inf]])
    p = lsq_linear(np.vstack(rows), np.concatenate(rhs), bounds=(lo, np.inf),
                   lsmr_tol="auto", max_iter=5000).x
    return u @ p[:-1], p[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", help="cu_m8b.extxyz")
    ap.add_argument("out_dir")
    ap.add_argument("--cutoff", type=float, default=3.5, help="A")
    ap.add_argument("--r-lo", type=float, default=1.6, help="A, below the shortest bond")
    ap.add_argument("--knots", type=int, default=14)
    ap.add_argument("--smooth", type=float, default=1e-3, help="second-difference penalty")
    ap.add_argument("--force-weight", type=float, default=0.0,
                    help="weight of a structure's force rows (eV/A) relative to its energy row "
                         "(eV/atom); each structure's rows are scaled by 1/sqrt(3N)")
    ap.add_argument("--monotonic", action="store_true",
                    help="V_rep non-negative and non-increasing")
    ap.add_argument("--test-every", type=int, default=5)
    ap.add_argument("--processes", type=int, default=6)
    args = ap.parse_args(argv)
    from pydftb.params import ParamSet
    sk = ParamSet(cdr.SRC, max_l={"Cu": 2}, shell_resolved=True).skset(["Cu"])
    train, _ = cf.split(cf.load(args.data), args.test_every)       # the test set stays out
    e_el, f_el = electronic(train, sk, args.processes)
    y = np.array([s.energy for s in train]) - e_el
    t, n, keep = basis(args.r_lo, args.cutoff, args.knots)
    a = design(train, t, n, keep, args.cutoff)
    m = len(keep)
    a_f, y_f, fidx = None, np.zeros(0), []
    if args.force_weight > 0:
        blocks, rhs = [], []
        for k, s in enumerate(train):
            if s.forces is None:
                continue
            w = args.force_weight / np.sqrt(s.forces.size)
            blocks.append(w * force_design(s.atoms, t, n, keep, args.cutoff))
            rhs.append(w * (s.forces - f_el[k]).ravel())
            fidx.append((k, w, s.forces.size))
        a_f, y_f = np.vstack(blocks), np.concatenate(rhs)
    coef_k, const = solve(a, y, a_f, y_f, m, args.smooth, args.monotonic)
    r = a @ coef_k + const - y
    if fidx:
        rf, off = a_f @ coef_k - y_f, 0
        frms = {}
        for k, w, sz in fidx:
            g = cf.group_of(train[k])
            frms.setdefault(g, []).append(rf[off:off + sz] / w)
            off += sz
        print("pair-only fit (train), forces eV/A: " + ", ".join(
            f"{g} {np.sqrt(np.mean(np.concatenate(v) ** 2)):.2f}" for g, v in frms.items()),
            flush=True)
    groups = [cf.group_of(s) for s in train]
    by = {g: float(np.sqrt(np.mean(r[[k for k, q in enumerate(groups) if q == g]] ** 2)))
          for g in dict.fromkeys(groups)}
    print("pair-only fit (train), meV/atom: " + ", ".join(f"{g} {v * 1000:.0f}"
                                                           for g, v in by.items()), flush=True)
    coef = np.zeros(n)
    coef[keep] = coef_k
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
                   "force_weight": args.force_weight, "monotonic": args.monotonic,
                   "knots": args.knots, "r_lo": args.r_lo, "cutoff": args.cutoff}, fh, indent=1)
    print(f"wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
