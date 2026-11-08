"""Milestones M8b/M8c on the VASP Cu set: fit the adaptive models to energies and forces.

Reads ``cu_m8b.extxyz`` of the set's ``collect.py`` (``pydftb.bench.cu_dft_dataset``; PBE, Mermin
free energy at 300 K, forces), splits it into a training and a test set (every ``--test-every``-th
structure of each group), fits the chosen kinds of ``pydftb.bench.adaptive_cu.models`` with the
force residuals, and reports the energy (meV/atom, relative: one common offset) and force (eV/A)
errors by group on both sets.  The JSON has the layout of ``adaptive_cu`` (``models[kind]["x"]``),
so ``adaptive_md`` and ``active_cu`` read it as they are.

    python -m pydftb.bench.cu_dft_fit cu_m8b.extxyz --out cu_dft_fit.json \
        [--kinds fixed,adaptive,angular] [--force-weight 0.01] [--processes 8]

The VASP clusters and dimers are spin polarised; the pydftb models are not, so those groups carry
a spin energy that the fit has to absorb (reported separately by group).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import replace

import numpy as np
from ase.io import read

from pydftb import adaptive_fit as af
from pydftb.adaptive_fit import Sample, fit_adaptive
from pydftb.bench import adaptive_cu as ac
from pydftb.constants import BOHR
from pydftb.validation import ptbp_workload as pw

MOLECULAR = ("dimer", "cluster", "md", "active", "round2")
ZERO_FORCE = 1e-3                 # eV/A


def load(path: str, groups=None):
    out = []
    for at in read(path, ":"):
        g = at.info.get("group", "")
        if groups and g not in groups:
            continue
        n = len(at)
        mol = g in MOLECULAR or not all(at.pbc)
        f = at.arrays.get("dft_forces")
        e = at.info.get("free_energy")
        if e is None:                 # ase's extxyz reader moves it to a SinglePointCalculator
            e = at.calc.get_potential_energy(force_consistent=True)
            if f is None and "forces" in at.calc.results:
                f = at.calc.results["forces"]
        if f is not None and np.abs(f).max() < ZERO_FORCE:
            f = None                  # zero by symmetry (ideal phases, one-atom cells): no signal
        at = at.copy()
        if mol:
            at.pbc = False
        out.append(Sample(at, float(e) / n,
                          kpts=None if mol else pw.kpoint_grid(at),
                          name=f"{g} {at.info.get('name', '')}",
                          forces=None if f is None else np.array(f)))
    return out


def models(kind: str, wall: float, cn_clamp=None):
    """``adaptive_cu.models(kind)`` with the fixed core wall at ``wall`` (Angstrom).  The VASP set
    has bonds down to 1.7 A (the dimer, the restructured Cu13 frames), inside M8a's 2.0 A wall.
    ``cn_clamp`` (lo, hi): the CN basis is constant outside it (``RBF.clamp``)."""
    m = ac.models(kind)[0]
    pr = {k: replace(v, wall=wall / BOHR) for k, v in m.pair_repulsive.items()}
    m = replace(m, pair_repulsive=pr)
    if cn_clamp is not None:
        m = replace(m, basis=replace(m.basis, clamp=tuple(cn_clamp)))
    return [m]


def init_pair(models, samples, sk, processes, ridge=1e-6):
    """Start vector with the pair-repulsive coefficients solved by linear least squares on the
    energies (the pair term is linear in them; everything else stays at zero), with a free
    offset.  The bare table has no repulsion inside the first shell: from zero the nonlinear
    fit has to build the wall in many small trust-region steps."""
    x = np.concatenate([m.vector() for m in models])
    names = [n for m in models for n in m.parameter_names()]
    cols = [i for i, n in enumerate(names) if ".pair[" in n]
    e, jac = af.evaluate(models, samples, sk, ac.OPTIONS, x, processes)
    w = np.sqrt(np.array([s.weight for s in samples], float))
    a = np.hstack([jac[:, cols], np.ones((len(samples), 1))]) * w[:, None]
    b = (np.array([s.energy for s in samples]) - e) * w
    reg = np.sqrt(ridge) * np.eye(a.shape[1])
    reg[-1, -1] = 0.0                                  # the offset is free
    c = np.linalg.lstsq(np.vstack([a, reg]), np.concatenate([b, np.zeros(a.shape[1])]),
                        rcond=None)[0]
    x = x.copy()
    x[cols] = c[:-1]
    r = (a @ c - b) / w
    return x, float(np.sqrt(np.mean((r - r.mean()) ** 2)))


def group_of(s):
    return s.name.split(" ")[0]


def split(data, every: int):
    seen, train, test = defaultdict(int), [], []
    for s in data:
        g = group_of(s)
        (test if every and seen[g] % every == every - 1 else train).append(s)
        seen[g] += 1
    return train, test


def _safe_work(job):
    try:
        return af._work(job)
    except (RuntimeError, np.linalg.LinAlgError):
        return None                                   # SCC not converged: reported, not fatal


def errors(models, data, sk, offset, processes):
    """Per-sample energy residuals (eV/atom, minus the fit's offset; nan where the SCC does not
    converge) and force errors (eV/A)."""
    x = np.concatenate([m.vector() for m in models])
    af._WORK.update(models=tuple(models), samples=list(data), sk=sk, options=dict(ac.OPTIONS))
    jobs = [(k, x, s.forces is not None) for k, s in enumerate(data)]
    out = (af._pool_map(_safe_work, jobs, processes) if processes > 1
           else [_safe_work(j) for j in jobs])
    e = np.array([np.nan if o is None else o[0] for o in out])
    de = e - offset - np.array([s.energy for s in data])
    df = [None if (s.forces is None or out[k] is None) else out[k][2] - s.forces
          for k, s in enumerate(data)]
    return de, df


def by_group(data, de, df):
    out = {}
    for g in dict.fromkeys(group_of(s) for s in data):
        k = [i for i, s in enumerate(data) if group_of(s) == g]
        fs = [df[i].ravel() for i in k if df[i] is not None]
        ok = [i for i in k if np.isfinite(de[i])]
        out[g] = {"n": len(k), "failed": len(k) - len(ok),
                  "energy_rmse": float(np.sqrt(np.mean(de[ok] ** 2))) if ok else None,
                  "force_rmse": float(np.sqrt(np.mean(np.concatenate(fs) ** 2))) if fs else None}
    return out


def show(tag, table):
    for g, r in table.items():
        f = "" if r["force_rmse"] is None else f"  forces {r['force_rmse']:6.3f} eV/A"
        e = "    -" if r["energy_rmse"] is None else f"{r['energy_rmse'] * 1000:7.1f}"
        bad = f"  ({r['failed']} SCC failed)" if r["failed"] else ""
        print(f"  {tag:5s} {g:8s} n {r['n']:3d}  energy {e} meV/atom{f}{bad}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", help="cu_m8b.extxyz (collect.py)")
    ap.add_argument("--out", default="cu_dft_fit.json")
    ap.add_argument("--kinds", default="fixed,adaptive,angular")
    ap.add_argument("--groups", default=None, help="only these groups (comma separated)")
    ap.add_argument("--test-every", type=int, default=5, help="0: no test set")
    ap.add_argument("--force-weight", type=float, default=0.01)
    ap.add_argument("--force-groups", default=None,
                    help="only these groups' forces enter the fit (the Jacobian costs 2 n_x "
                         "calculations per structure); all are still reported")
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--max-evaluations", type=int, default=150)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--force-max-atoms", type=int, default=None,
                    help="only structures up to this size carry forces into the fit")
    ap.add_argument("--wall", type=float, default=1.6, help="fixed core wall (A), below the data")
    ap.add_argument("--skf", default=None,
                    help="the SK directory (default ptbp_hcocu; e.g. cu_dimer_repulsive's table)")
    ap.add_argument("--cn-clamp", type=float, nargs="+", default=None,
                    help="LO HI [D]: the CN basis constant outside [LO, HI] (the training "
                         "range), D the half-width of the smooth transition (RBF.clamp)")
    ap.add_argument("--x0", default=None,
                    help="start from models[kind].x of this JSON (with --max-evaluations 0: "
                         "only report its errors)")
    ap.add_argument("--no-init", action="store_true",
                    help="start from zero instead of the linear pair-repulsive solution")
    args = ap.parse_args(argv)
    from pydftb.params import ParamSet
    skdir = args.skf or f"{ac.REPO}/pydftb/tests/data/skf/ptbp_hcocu"
    sk = ParamSet(skdir, max_l={"Cu": 2}, shell_resolved=True).skset(["Cu"])
    data = load(args.data, args.groups.split(",") if args.groups else None)
    train, test = split(data, args.test_every)
    keep = args.force_groups.split(",") if args.force_groups else None
    nmax = args.force_max_atoms

    def with_f(s):
        return ((keep is None or group_of(s) in keep)
                and (nmax is None or len(s.atoms) <= nmax))

    fit_set = [s if with_f(s) else Sample(s.atoms, s.energy, s.kpts, s.weight, s.name)
               for s in train]
    print(f"forces in the fit: {sum(s.forces is not None for s in fit_set)} structures",
          flush=True)
    print(f"{len(data)} structures: {len(train)} train, {len(test)} test", flush=True)
    report = {"options": ac.OPTIONS, "data": args.data, "force_weight": args.force_weight,
              "force_groups": args.force_groups,
              "force_max_atoms": args.force_max_atoms, "skf": args.skf, "cn_clamp": args.cn_clamp, "wall": args.wall,
              "l2": args.l2, "train": [s.name for s in train], "test": [s.name for s in test],
              "models": {}}
    for kind in args.kinds.split(","):
        t0 = time.perf_counter()
        mod = models(kind, args.wall, args.cn_clamp)
        if args.x0:
            with open(args.x0) as fh:
                x0, init_rmse = np.array(json.load(fh)["models"][kind]["x"]), None
        else:
            x0, init_rmse = (None, None) if args.no_init else init_pair(mod, train, sk,
                                                                         args.processes)
        if init_rmse is not None:
            print(f"{kind}: linear pair-repulsive start, energy rmse {init_rmse * 1000:.1f} "
                  "meV/atom", flush=True)
        if args.max_evaluations > 0:
            res = fit_adaptive(mod, fit_set, sk, ac.OPTIONS, l2=args.l2, x0=x0,
                               processes=args.processes, max_evaluations=args.max_evaluations,
                               force_weight=args.force_weight, verbose=True)
        else:                                          # report only: one evaluation at x0
            res = fit_adaptive(mod, fit_set, sk, ac.OPTIONS, l2=0.0, x0=x0,
                               processes=args.processes, max_evaluations=1)
        entry = {"init_rmse": init_rmse, "wall": args.wall, "rmse": res.rmse, "force_rmse": res.force_rmse, "x": res.x.tolist(),
                 "residuals": res.residuals.tolist(), "offset": res.offset,
                 "names": res.models[0].parameter_names(), "n_parameters": len(res.x),
                 "evaluations": res.n_evaluations, "history": res.history}
        report["models"][kind] = dict(entry)            # the fit is saved before the errors
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        de, df = errors(res.models, train, sk, res.offset, args.processes)
        entry["train"] = by_group(train, de, df)
        if test:
            de, df = errors(res.models, test, sk, res.offset, args.processes)
            entry["test"] = by_group(test, de, df)
        entry["dimer_curve"] = ac.dimer_curve(res.models, sk)
        entry["seconds"] = time.perf_counter() - t0
        print(f"{kind}: train rmse {res.rmse * 1000:.1f} meV/atom, forces "
              f"{res.force_rmse if res.force_rmse is not None else float('nan'):.3f} eV/A, "
              f"{len(res.x)} parameters, {res.n_evaluations} evaluations, "
              f"{entry['seconds']:.0f} s", flush=True)
        show("train", entry["train"])
        if test:
            show("test", entry["test"])
        report["models"][kind] = entry
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
