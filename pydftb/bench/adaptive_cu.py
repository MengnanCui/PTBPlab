"""Milestone M8a: environment-adaptive Cu parameters on PTBP's Cu set (needs a PTBPlab checkout).

The six phases of ``dft/fit.json`` (dimer, AA-graphite, diamond, beta-Sn, bcc, fcc: coordination
1 to 12) at the 11 EOS volumes of ``pydftb.validation.ptbp_workload``, the reference energies
from the FHI-aims Birch-Murnaghan curves of ``fit.json``.  Two models on the bundled ptbp_hcocu
Cu-Cu table, which carries no repulsive within the first shell:

* fixed:    constant on-site shifts and hopping scales (s, p, d) and a pair repulsive;
* adaptive: the same channels as functions of the coordination number (RBF basis), plus a
            CN-dependent scale of the repulsive.

Both are fitted to all phases, and in a leave-one-phase-out cross-validation.

    python -m pydftb.bench.adaptive_cu --out adaptive_cu.json [--processes 8] [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from pydftb.adaptive import RBF, AdaptiveParams, PairRepulsive
from pydftb.adaptive_fit import Sample, fit_adaptive, predict
from pydftb.constants import BOHR
from pydftb.params import ParamSet
from pydftb.validation import ptbp_workload as pw

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE_KEYS = {"dimer": "Cu_0", "AA-graphite": "Cu_1", "diamond": "Cu_2", "beta-Sn": "Cu_3",
              "bcc": "Cu_4", "fcc": "Cu_5"}
OPTIONS = {"temperature": 300.0, "scc_tol": 1e-9, "max_scc": 300}
PAIR_CUTOFF = 3.2 / BOHR                  # Angstrom -> Bohr: inside fcc's second shell
N_PAIR = 6                                # (r_c - r)^2 .. (r_c - r)^7
# a fixed core wall below the shortest bond of the data (2.059 A, the dimer at 0.8 V0): zero for
# every structure of the fit, so the fits do not see it (M8a: without it MD collapsed to 1.4 A)
WALL = 2.0 / BOHR
WALL_STRENGTH = 1000.0 / 27.2113845 * BOHR ** 3        # 1000 eV/A^3 in Hartree/Bohr^3


def birch_murnaghan(v, e0, v0, b, bp):
    """ase.eos.birchmurnaghan (eV, A^3)."""
    eta = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9.0 * b * v0 / 16.0 * (eta - 1.0) ** 2 * (6.0 + bp * (eta - 1.0) - 4.0 * eta)


def samples(quick: bool = False):
    with open(os.path.join(REPO, "dft", "fit.json")) as fh:
        fit = json.load(fh)
    out = []
    for s in pw.structures(quick=quick):
        f = fit[PHASE_KEYS[s.phase]]
        v = s.atoms.get_volume() / len(s.atoms)
        e = birch_murnaghan(v, f["aims_energy"], f["aims_volume"], f["aims_B"], f["aims_Bp"])
        kpts = None if s.phase == "dimer" else pw.kpoint_grid(s.atoms)
        out.append(Sample(s.atoms, e, kpts=kpts, name=s.name))
    return out


def models(kind: str):
    if kind == "fixed":
        b = RBF(())
        z = (0.0,)
        return [AdaptiveParams("Cu", b, onsite={"s": z, "p": z, "d": z},
                               hopping={"s": z, "p": z, "d": z},
                               pair_repulsive={"Cu": PairRepulsive(PAIR_CUTOFF, (0.0,) * N_PAIR, WALL, WALL_STRENGTH)})]
    b = RBF((2.0, 6.0, 10.0), width=2.5)
    z = (0.0,) * b.size
    return [AdaptiveParams("Cu", b, onsite={"s": z, "p": z, "d": z},
                           hopping={"s": z, "p": z, "d": z}, repulsive=z,
                           pair_repulsive={"Cu": PairRepulsive(PAIR_CUTOFF, (0.0,) * N_PAIR, WALL, WALL_STRENGTH)})]


def dimer_curve(models, sk, r=None):
    """E(r) - E(3 A) of an isolated Cu2 (eV): the short-range wall below the data's shortest
    bond (about 2.06 A)."""
    from ase import Atoms
    r = np.arange(1.2, 3.05, 0.1) if r is None else np.asarray(r)
    out = []
    for x in r:
        at = Atoms("Cu2", positions=[[0, 0, 0], [0, 0, x]])
        out.append(predict(models, [Sample(at, 0.0, name="dimer")], sk, OPTIONS,
                           processes=1)[0] * 2)
    e = np.array(out)
    return {"r": r.tolist(), "energy": (e - e[np.argmin(np.abs(r - 3.0))]).tolist()}


def phase_rmse(names, resid):
    out = {}
    for ph in PHASE_KEYS:
        r = np.array([x for n, x in zip(names, resid) if n.startswith(ph + " ")])
        if len(r):
            out[ph] = float(np.sqrt((r ** 2).mean()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="adaptive_cu.json")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="3 volumes per phase")
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--max-evaluations", type=int, default=150)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--kinds", default="fixed,adaptive")
    args = ap.parse_args(argv)
    sk = ParamSet(os.path.join(REPO, "pydftb", "tests", "data", "skf", "ptbp_hcocu"),
                  max_l={"Cu": 2}, shell_resolved=True).skset(["Cu"])
    data = samples(args.quick)
    names = [s.name for s in data]
    report = {"options": OPTIONS, "n_samples": len(data), "l2": args.l2, "models": {}}
    t_ref = np.array([s.energy for s in data])
    # the bare table (no model terms): one evaluation
    e0 = predict(models("fixed"), data, sk, OPTIONS, processes=args.processes)
    d0 = e0 - t_ref
    d0 -= d0.mean()
    report["bare"] = {"rmse": float(np.sqrt((d0 ** 2).mean())), "by_phase": phase_rmse(names, d0),
                      "residuals": d0.tolist()}
    print(f"bare table: rmse {report['bare']['rmse'] * 1000:.1f} meV/atom", flush=True)
    for kind in args.kinds.split(","):
        t0 = time.perf_counter()
        res = fit_adaptive(models(kind), data, sk, OPTIONS, l2=args.l2,
                           processes=args.processes, max_evaluations=args.max_evaluations,
                           verbose=True)
        entry = {"rmse": res.rmse, "by_phase": phase_rmse(names, res.residuals),
                 "residuals": res.residuals.tolist(), "x": res.x.tolist(),
                 "names": res.models[0].parameter_names(), "n_parameters": len(res.x),
                 "evaluations": res.n_evaluations, "seconds": time.perf_counter() - t0,
                 "history": res.history}
        print(f"{kind}: rmse {res.rmse * 1000:.1f} meV/atom, {len(res.x)} parameters, "
              f"{res.n_evaluations} evaluations, {entry['seconds']:.0f} s", flush=True)
        entry["dimer_curve"] = dimer_curve(res.models, sk)
        e = entry["dimer_curve"]["energy"]
        print(f"  dimer E(r) at r = {entry['dimer_curve']['r'][0]:.1f}..: "
              + " ".join(f"{x:.2f}" for x in e[:8]) + " eV", flush=True)
        report["models"][kind] = entry
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        if not args.no_cv:
            cv = {}
            for ph in PHASE_KEYS:
                train = [s for s in data if not s.name.startswith(ph + " ")]
                test = [s for s in data if s.name.startswith(ph + " ")]
                r = fit_adaptive(models(kind), train, sk, OPTIONS, l2=args.l2,
                                 processes=args.processes, max_evaluations=args.max_evaluations)
                e = predict(r.models, test, sk, OPTIONS, offset=r.offset, processes=args.processes)
                d = e - np.array([s.energy for s in test])
                cv[ph] = {"test_rmse": float(np.sqrt((d ** 2).mean())), "train_rmse": r.rmse,
                          "test_residuals": d.tolist(), "evaluations": r.n_evaluations}
                print(f"  {kind} without {ph:12s}: train {r.rmse * 1000:7.1f}  "
                      f"test {cv[ph]['test_rmse'] * 1000:8.1f} meV/atom", flush=True)
            entry["cv"] = cv
        report["models"][kind] = entry
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
