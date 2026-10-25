"""BO against the gradient optimiser at equal wall time (milestone M4c).

Runs, on the same dataset, loss mode and core budget:

* BO as ``Loss.Bayesian_optimization`` runs it (gp_minimize, EI, the PTBP seed first)
  with ``--n-calls`` evaluations, for several random states;
* the gradient optimiser (``--optimizer gradient``) from the same seed with
  ``--grad-maxiter`` iterations;
* BO again with as many evaluations as fit in the gradient run's wall time.

and prints, for every run, the best loss after each evaluation against the elapsed time
(also written as JSON, after every run; ``--resume`` skips the runs already in it).
Example (about an hour on 4 cores)::

    python tools/bo_vs_grad.py dft/dft.xyz --mode energygeometry --json bo_vs_grad.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _timed(loss, cwd):
    """Loss_EnergyGeometry that records (elapsed, x, mu, failed) and returns to ``cwd``."""
    from utils.gradient_opt import loss_failed
    trace, t0 = [], time.perf_counter()

    def f(x):
        try:
            mu = float(loss.Loss_EnergyGeometry([float(x[0]), float(x[1])]))
        finally:
            os.chdir(cwd)
        trace.append((time.perf_counter() - t0, [float(v) for v in x], mu, loss_failed(loss, mu)))
        return mu
    return f, trace


def run_bo(loss, cov_radius, n_calls, random_state, cwd):
    from skopt import gp_minimize

    from utils import gradient_opt as GO
    dims, _, _, _ = loss._bo_dimensions_and_cost(cov_radius)
    f, trace = _timed(loss, cwd)
    gp_minimize(func=f, dimensions=dims, n_calls=n_calls, acq_func="EI",
                x0=GO.seed(loss, cov_radius), random_state=random_state)
    return trace


def run_gradient(loss, cov_radius, maxiter, cwd):
    from utils import gradient_opt as GO
    t0 = time.perf_counter()
    ev = GO.Evaluator(loss, log_path=str(Path(cwd) / "grad.out"))
    trace = []

    def f(x):
        try:
            e = ev(x)
        finally:
            os.chdir(cwd)
        trace.append((time.perf_counter() - t0, [float(v) for v in x], float(e.f),
                      None if e.g is None else [float(v) for v in e.g], not e.valid))
        return e
    lo, hi = GO.bounds(loss, cov_radius)
    best, hist = GO.projected_bfgs(f, GO.seed(loss, cov_radius), lo, hi, maxiter=maxiter,
                                   history=ev.history)
    return trace, hist.message


def _failed(row) -> bool:
    """A failed step: the flag at the end of the row (M5b-2), else (rows of a JSON written
    before) the old failure values 97-99, which were below the real losses of the dataset mode."""
    from utils.gradient_opt import failed
    return row[-1] if isinstance(row[-1], bool) else failed(row[2])


def best_so_far(trace):
    """(elapsed, lowest loss so far) after each evaluation; failed steps do not count."""
    out, best = [], float("inf")
    for row in trace:
        if not _failed(row):
            best = min(best, row[2])
        out.append((row[0], best))
    return out


def summary(trace):
    """(lowest real loss, its x, number of failed steps) of a run."""
    ok = [r for r in trace if not _failed(r)]
    b = min(ok, key=lambda r: r[2]) if ok else (None, None, float("nan"))
    return b[2], b[1], len(trace) - len(ok)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--mode", default="energygeometry",
                    choices=["energygeometry", "dataset", "reaction"])
    ap.add_argument("--n-calls", type=int, default=11)
    ap.add_argument("--random-states", type=int, nargs="+", default=[123, 7, 2024])
    ap.add_argument("--grad-maxiter", type=int, default=8)
    ap.add_argument("--workdir", type=Path, default=Path("bo_vs_grad"))
    ap.add_argument("--runs", nargs="+", default=["gradient", "bo", "matched"],
                    choices=["gradient", "bo", "matched"],
                    help="which runs (matched needs the other two, in this call or the JSON)")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="keep the runs already in --json and do only the missing ones")
    ap.add_argument("--confinement", default="selfconsistent",
                    choices=["selfconsistent", "perturbative"],
                    help="hotcent's confinement scheme (M4e; perturbative: as before M4e)")
    a = ap.parse_args(argv)
    if a.resume and a.json is None:
        ap.error("--resume needs --json")
    from utils.hotpy import set_confinement
    set_confinement(a.confinement)

    import utils.engines as eng
    from tools.engine_parity import build_loss
    from utils.dftb_stage import torch_one_thread
    from utils.parallel import detect_cpu, set_plan
    from utils.recommender import make_plan, workload_from_dataset
    eng.configure(engine="pydftb")
    torch_one_thread()
    dataset = a.dataset.resolve()
    out = {"mode": a.mode, "runs": {}}
    if a.resume and a.json.exists():
        out = json.loads(a.json.read_text())
        if out.get("mode") != a.mode:
            raise SystemExit(f"{a.json} holds the {out.get('mode')} mode, not {a.mode}")
        print(f"[resume] kept: {', '.join(out['runs']) or 'nothing'}", flush=True)
    cwd = os.getcwd()

    def save():
        if a.json:
            a.json.write_text(json.dumps(out, indent=1))

    def fresh(tag):
        wd = (a.workdir / f"{a.mode}_{tag}").resolve()
        if wd.exists():                  # an interrupted run: start it again
            shutil.rmtree(wd)
        wd.mkdir(parents=True)
        loss, cov = build_loss(dataset, a.mode, wd, 11)
        work = workload_from_dataset(loss.dft_input, loss.bolt_f11,
                                     kpts_density=loss.args_input.kpt_density,
                                     eos_points=loss.args_input.eos_points, engine="pydftb",
                                     mode=a.mode)
        set_plan(make_plan(detect_cpu("auto"), work, n_calls=a.n_calls))
        return loss, cov, str(wd)

    runs = out["runs"]
    if "gradient" in a.runs and "gradient" not in runs:
        loss, cov, wd = fresh("gradient")
        trace, message = run_gradient(loss, cov, a.grad_maxiter, wd)
        runs["gradient"] = {"trace": trace, "message": message}
        save()
        print(f"[gradient] {len(trace)} evaluations, {trace[-1][0]:.0f} s, "
              f"best {summary(trace)[0]:.6g} ({message})", flush=True)
    for rs in (a.random_states if "bo" in a.runs else []):
        if f"bo_{rs}" in runs:
            continue
        loss, cov, wd = fresh(f"bo_{rs}")
        trace = run_bo(loss, cov, a.n_calls, rs, wd)
        runs[f"bo_{rs}"] = {"trace": trace}
        save()
        print(f"[bo rs={rs}] {len(trace)} evaluations, {trace[-1][0]:.0f} s, "
              f"best {summary(trace)[0]:.6g}", flush=True)
    bo_runs = [runs[f"bo_{rs}"] for rs in a.random_states if f"bo_{rs}" in runs]
    if ("matched" in a.runs and "bo_matched" not in runs and runs.get("gradient", {}).get("trace")
            and bo_runs):
        t_grad = runs["gradient"]["trace"][-1][0]
        per_eval = sum(r["trace"][-1][0] / len(r["trace"]) for r in bo_runs) / len(bo_runs)
        n_match = max(2, int(round(t_grad / per_eval)))
        loss, cov, wd = fresh("bo_matched")
        trace = run_bo(loss, cov, n_match, a.random_states[0], wd)
        runs["bo_matched"] = {"trace": trace, "n_calls": n_match}
        save()
        print(f"[bo matched] {n_match} evaluations, {trace[-1][0]:.0f} s, "
              f"best {summary(trace)[0]:.6g}", flush=True)
    set_plan(None)
    os.chdir(cwd)
    print(f"\n{'run':>14} {'evals':>6} {'failed':>6} {'wall s':>8} {'best mu':>14}  best x")
    for tag, run in out["runs"].items():
        tr = run["trace"]
        mu, x, nfail = summary(tr)
        print(f"{tag:>14} {len(tr):6d} {nfail:6d} {tr[-1][0]:8.0f} {mu:14.8g}  {x}")
        run["best_so_far"] = best_so_far(tr)
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
