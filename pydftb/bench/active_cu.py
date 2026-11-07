"""Milestone M8c: a committee of adaptive Cu models, its uncertainty, and the structures it picks.

Two experiments on PTBP's Cu set (``pydftb.bench.adaptive_cu``), no new DFT:

* calibration: a committee fitted without diamond (base fit from zero, members refitted to
  bootstrap resamples of whole phases).  Does its spread flag the phase it has not seen, and
  does the spread follow the base model's real error phase by phase?
* screening: a committee around the M8a adaptive fit (``adaptive_cu.json``) on the frames of
  the Cu13 MD in which that model restructured (``adaptive_md --traj``): the spread along the
  trajectory, and the frames it picks, written as VASP inputs (only written, not submitted).

    python -m pydftb.bench.active_cu adaptive_cu.json cu13_adaptive.extxyz --out active_cu.json \
        [--write cu_dft_m8b_active] [--members 4] [--processes 8]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from ase.io import read

from pydftb.active import (
    committee_predict,
    fingerprints,
    fit_committee,
    select,
    write_candidates,
)
from pydftb.adaptive_fit import FitResult, predict
from pydftb.bench import adaptive_cu as ac


def phase_of(sample):
    return sample.name.split(" ")[0]


def _save(path, report):
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fit_json", help="adaptive_cu.json (the M8a fits)")
    ap.add_argument("traj", help="the Cu13 trajectory (extxyz) of adaptive_md --traj")
    ap.add_argument("--out", default="active_cu.json")
    ap.add_argument("--write", default=None, help="write the picked frames as VASP inputs here")
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--member-evaluations", type=int, default=30)
    ap.add_argument("--base-evaluations", type=int, default=100)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--pick", type=int, default=10)
    ap.add_argument("--held-out", default="diamond")
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument("--groups", choices=("phase", "none"), default="none",
                    help="bootstrap whole phases or single structures")
    ap.add_argument("--data", default=None,
                    help="cu_m8b.extxyz: screen around a cu_dft_fit model (fit_json) on the VASP "
                         "set instead of PTBP's phases (implies --skip-calibration)")
    ap.add_argument("--kind", default="adaptive", help="the model of fit_json")
    ap.add_argument("--perturb", type=float, default=0.1,
                    help="anchor perturbation (0: every member anchored at the base)")
    args = ap.parse_args(argv)
    from pydftb.params import ParamSet
    sk = ParamSet(f"{ac.REPO}/pydftb/tests/data/skf/ptbp_hcocu", max_l={"Cu": 2},
                  shell_resolved=True).skset(["Cu"])
    opts = ac.OPTIONS
    with open(args.fit_json) as fh:
        fit_rep = json.load(fh)
    if args.data:
        from pydftb.bench import cu_dft_fit as cf
        args.skip_calibration = True
        data = cf.load(args.data)
        names = set(fit_rep.get("train", []))
        data = [s for s in data if s.name in names] if names else data   # its training set
        model0 = cf.models(args.kind, fit_rep["wall"])[0]
    else:
        data = ac.samples()
        model0 = ac.models(args.kind)[0]
    phases = [phase_of(s) for s in data]
    report = {"options": opts, "members": args.members, "groups": args.groups,
              "perturb": args.perturb, "data": args.data, "kind": args.kind}

    def grp(ss):
        return [phase_of(s) for s in ss] if args.groups == "phase" else None

    # ---- calibration: a committee that has not seen one phase ----------------------------------
    if not args.skip_calibration:
        t0 = time.perf_counter()
        train = [s for s, p in zip(data, phases) if p != args.held_out]
        com = fit_committee(ac.models("adaptive"), train, sk, opts, n=args.members, seed=11,
                            groups=grp(train), perturb=args.perturb, processes=args.processes,
                            max_evaluations=args.member_evaluations,
                            base_evaluations=args.base_evaluations, verbose=True)
        u = committee_predict(com, [s.atoms for s in data], sk, opts,
                              kpts=[s.kpts for s in data], forces=False,
                              processes=args.processes)
        e_base = predict(com.base, data, sk, opts, offset=com.base_offset,
                         processes=args.processes)
        err = np.abs(e_base - np.array([s.energy for s in data]))
        by = {}
        for p in ac.PHASE_KEYS:
            k = [i for i, q in enumerate(phases) if q == p]
            by[p] = {"sigma_e": float(np.sqrt(np.mean(u.sigma_e[k] ** 2))),
                     "error": float(np.sqrt(np.mean(err[k] ** 2))), "seen": p != args.held_out}
            print(f"  {p:12s} {'seen' if by[p]['seen'] else 'HELD OUT':8s} sigma "
                  f"{by[p]['sigma_e'] * 1000:7.1f}  base error {by[p]['error'] * 1000:7.1f} "
                  "meV/atom", flush=True)
        rho = float(np.corrcoef(np.log(u.sigma_e + 1e-6), np.log(err + 1e-6))[0, 1])
        report["calibration"] = {"held_out": args.held_out, "by_phase": by,
                                 "log_correlation": rho, "sigma_e": u.sigma_e.tolist(),
                                 "error": err.tolist(), "names": [s.name for s in data],
                                 "seconds": time.perf_counter() - t0}
        print(f"calibration: corr(log sigma, log error) = {rho:.2f}, "
              f"{report['calibration']['seconds']:.0f} s", flush=True)
        _save(args.out, report)

    # ---- screening: the M8a trajectory ---------------------------------------------------------
    t0 = time.perf_counter()
    fit = fit_rep["models"][args.kind]
    base_models = (model0.with_vector(np.array(fit["x"])),)
    e_fit = predict(base_models, data, sk, opts, processes=args.processes)
    off = float(np.mean(e_fit - np.array([s.energy for s in data])))
    base = FitResult(base_models, np.array(fit["x"]), fit["rmse"], np.zeros(len(data)), off)
    com = fit_committee(None, data, sk, opts, n=args.members, seed=12, groups=grp(data),
                        base=base, perturb=args.perturb, processes=args.processes,
                        max_evaluations=args.member_evaluations, verbose=True)
    frames = read(args.traj, ":")
    u = committee_predict(com, frames, sk, opts, forces=True, processes=min(args.processes,
                                                                            4 * len(frames)))
    dmin = [float(np.min(a.get_all_distances()[np.triu_indices(len(a), 1)])) for a in frames]
    fp = fingerprints(com.base, frames)
    ref = fingerprints(com.base, [s.atoms for s in data])
    score = u.score()
    pick = select(score, fp, n=args.pick, reference=ref)
    for k in range(len(frames)):
        print(f"  frame {k:3d} ({k * 10:4d} fs)  dmin {dmin[k]:.2f} A  sigma_E "
              f"{u.sigma_e[k] * 1000:8.1f} meV/atom  sigma_F {u.sigma_f[k]:6.2f} eV/A"
              f"{'  failed' if u.failed[k] else ''}{'  PICK' if k in pick else ''}", flush=True)
    report["screening"] = {
        "frames": len(frames), "time_fs": [10 * k for k in range(len(frames))], "dmin": dmin,
        "sigma_e": u.sigma_e.tolist(), "sigma_f": u.sigma_f.tolist(),
        "failed": u.failed.tolist(), "mean_e": u.mean_e.tolist(), "picked": pick,
        "member_offsets": com.offsets.tolist(), "draws": com.draws, "perturb": com.perturb,
        "seconds": time.perf_counter() - t0,
        "train_sigma_e": None}
    # the spread on the training set itself, for scale
    ut = committee_predict(com, [s.atoms for s in data], sk, opts, kpts=[s.kpts for s in data],
                           forces=False, processes=args.processes)
    report["screening"]["train_sigma_e"] = float(np.sqrt(np.mean(ut.sigma_e ** 2)))
    print(f"screening: training-set sigma_E {report['screening']['train_sigma_e'] * 1000:.1f} "
          f"meV/atom; picked {pick}; {report['screening']['seconds']:.0f} s", flush=True)
    if args.write:
        names = [f"cu13_md_{10 * k:04d}fs" for k in pick]
        man = write_candidates(args.write, [frames[k] for k in pick], names, group="round1")
        report["screening"]["written"] = {"dir": args.write, "structures": len(man),
                                          "core_hours": float(sum(m["core_hours_estimate"]
                                                                  for m in man))}
        print(f"wrote {len(man)} VASP inputs to {args.write} (not submitted)", flush=True)
    _save(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
