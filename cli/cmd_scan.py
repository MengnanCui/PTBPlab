"""`ptbp scan`: the A-series accuracy scans (utils/accuracy.py, design A0 section 3).

    ptbp scan points CONFIG                  # parameter points of the config's space
    ptbp scan run CONFIG [--jobs N]          # SK sets + electronic energies/forces/bands
    ptbp scan run CONFIG --index I --stride S    # one Slurm array task: points I, I+S, ...
    ptbp scan losses CONFIG [--jobs N]       # every strategy, the R2b control and the oracle
    ptbp scan report CONFIG                  # best point per strategy, report.md / report.json
    ptbp scan refine CONFIG --strategy S1    # Nelder-Mead from the strategy's best point
    ptbp scan refine CONFIG --strategy S1 --method bo --seeds 3   # BO (scikit-optimize), 3 seeds
    ptbp scan export CONFIG --point I --strategy S1 --out DIR [--repulsive R2b]
    ptbp scan status CONFIG

Each command logs to ``<out>/logs/scan-<command>/<time>-<host>-<pid>/`` (utils/runlog.py):
run.log, meta.json (git commit, versions, host, arguments), events.jsonl, status.json.
Nothing is submitted to a queue: the Slurm templates in examples/08_accuracy/slurm/ are for the
user to submit.
"""
from __future__ import annotations

import argparse
import json
import os


def add_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "scan", help="A-series accuracy scans: one SK set per point, every strategy on the "
                     "same points (utils/accuracy.py).",
        description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="scan_cmd", metavar="STAGE", required=True)

    def stage(name, help_):
        q = sub.add_parser(name, help=help_)
        q.add_argument("config", help="scan config (YAML or JSON)")
        return q

    stage("points", "write points.json")
    q = stage("run", "SK sets and electronic part of the points")
    q.add_argument("--jobs", type=int, default=1, help="parallel processes (one point each)")
    q.add_argument("--index", type=int, default=None, help="first point (Slurm array task id)")
    q.add_argument("--stride", type=int, default=1, help="with --index: every stride-th point")
    q.add_argument("--points", default=None, help="explicit points, e.g. 0-31,40")
    q.add_argument("--force", action="store_true", help="recompute finished points")
    q = stage("losses", "score every finished point")
    q.add_argument("--jobs", type=int, default=1)
    q.add_argument("--force", action="store_true")
    stage("report", "best point per strategy")
    q = stage("refine", "local optimisation of one strategy")
    q.add_argument("--strategy", default="S1")
    q.add_argument("--start", type=int, default=None, help="start point (default: best)")
    q.add_argument("--maxfev", type=int, default=30, help="evaluations (per seed with bo)")
    q.add_argument("--method", default="nm", choices=["nm", "bo"],
                   help="nm: Nelder-Mead (default); bo: Bayesian optimisation (scikit-optimize)")
    q.add_argument("--seeds", type=int, default=3, help="bo: independent seeds 0..N-1")
    q.add_argument("--box", type=float, default=0.15,
                   help="bo: half-width of the local box, fraction of the space's range")
    q = stage("export", "full SK files of a point")
    q.add_argument("--point", type=int, required=True)
    q.add_argument("--strategy", default="S1")
    q.add_argument("--repulsive", default="physrep", choices=["physrep", "R2b"])
    q.add_argument("--out", required=True)
    stage("status", "progress of the scan")
    p.set_defaults(func=run)
    return p


def _indices(spec: str) -> list:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def run(args: argparse.Namespace) -> int:
    from utils import accuracy as acc
    from utils.runlog import session
    cfg = acc.load_config(args.config)
    os.makedirs(cfg["out"], exist_ok=True)
    root = os.path.join(cfg["out"], "logs")
    with session(f"scan-{args.scan_cmd}", vars(args), root=root) as log:
        log.event("config", path=cfg["_path"], out=cfg["out"], cache=cfg["sk"]["cache"])
        if args.scan_cmd == "points":
            pts = acc.write_points(cfg)
            log.event("points", n=len(pts))
        elif args.scan_cmd == "run":
            if not os.path.exists(os.path.join(cfg["out"], "points.json")):
                acc.write_points(cfg)
            n = len(acc.read_points(cfg))
            if args.points:
                idx = [i for i in _indices(args.points) if i < n]
            elif args.index is not None:
                idx = list(range(args.index, n, max(1, args.stride)))
            else:
                idx = list(range(n))
            st = acc.run_points(cfg, idx, jobs=args.jobs, force=args.force, log=print)
            bad = {i: s[2] for i, s in st.items() if s[2]}
            log.event("run", points=len(idx), done=len(st), errors=len(bad))
            log.save_json("run_status.json", {str(i): {"failed_structures": s[1], "error": s[2]}
                                              for i, s in st.items()})
            if bad:
                print(f"[scan] {len(bad)} point(s) failed; see points/pNNNNN/error.txt")
        elif args.scan_cmd == "losses":
            path = acc.score_points(cfg, jobs=args.jobs, force=args.force)
            log.event("losses", table=path)
        elif args.scan_cmd == "report":
            path = acc.report(cfg)
            log.event("report", report=path)
        elif args.scan_cmd == "refine":
            out = acc.refine(cfg, args.strategy, args.start, args.maxfev, method=args.method,
                             seeds=args.seeds, box=args.box)
            if args.method == "bo":
                log.event("refine", strategy=args.strategy, method="bo", loss=out["best"]["loss"],
                          seed=out["best"]["seed"], seeds=len(out["seeds"]),
                          nfev=sum(r["nfev"] for r in out["seeds"]))
            else:
                log.event("refine", strategy=args.strategy, method="nm", loss=out["loss"],
                          nfev=out["nfev"])
        elif args.scan_cmd == "export":
            out = acc.export_model(cfg, args.point, args.strategy, os.path.abspath(args.out),
                                   repulsive=args.repulsive)
            log.event("export", out=out)
        elif args.scan_cmd == "status":
            st = acc.status(cfg)
            print(json.dumps(st, indent=1))
            log.event("status", **st)
    return 0
