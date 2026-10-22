"""Deterministic loss-function parity between the DFTB engines (DFTB+ vs pydftb).

Builds PTBP's ``Loss`` exactly like ``cli/run.py`` and evaluates the real
objective (``Loss.Loss_EnergyGeometry``: hotcent SKFs -> DFTB on every
structure -> physrep repulsion fit -> loss) at fixed (r0_w, r0_d) points
with each engine, so the comparison does not depend on the optimiser's path.

Example (run inside the conda environment with hotcent, pylibxc and dftb+)::

    python tools/engine_parity.py dft/dft.xyz --mode dataset \\
        --points 3.0,6.0 4.5,9.0 --workdir /tmp/parity

Prints, per point, the loss of each engine, the relative difference and the
time spent in the DFTB part of the pipeline.

``--ncores 1 4`` evaluates every point once per core budget (the parallel plan of
``cli/run.py``: the M3c recommender), and prints the measured stage times next to the
plan's estimates; ``--compare-serial`` (= ``--ncores 1 auto``) also checks that the
losses are bitwise equal and the files of the BO step byte-identical::

    python tools/engine_parity.py dft/dft.xyz --mode energygeometry \
        --points 2.749,12.036 --engines pydftb --compare-serial
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from math import floor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# files of a BO step that must not depend on the core budget
STEP_FILES = ("dft_final.xyz", "dftb_band.xyz", "search_index.txt", "dftb_full.xyz")


def step_digests(opt_dir: Path) -> dict:
    """sha256 of the BO step's output files and SKFs (``results/opt_<r0_w>_<r0_d>``)."""
    names = sorted(STEP_FILES) + sorted(p.name for p in opt_dir.glob("*.skf"))
    return {n: hashlib.sha256((opt_dir / n).read_bytes()).hexdigest()
            for n in names if (opt_dir / n).exists()}


def build_loss(dataset: Path, mode: str, workdir: Path, eos_points: int):
    import ase.io

    import utils.data as data
    from cli.arg_parser import build_default_arg_parser
    from utils.loss import Loss, penalty_radius

    argv = ["run.py", "--ref_dir", str(dataset.parent), "--dft_file", dataset.name,
            "--optimization_option", mode, "--parameters", "r0_w", "r0_d", "sigma_rep",
            "--superposition", "density", "--eos_points", str(eos_points),
            "--para_file", str(workdir / "par.out"), "--log_file", str(workdir / "log.out")]
    saved = sys.argv
    sys.argv = argv
    try:
        args = build_default_arg_parser()
    finally:
        sys.argv = saved
    if mode in ("dataset", "reaction"):
        args.eos_points = 1
        cache = dataset.parent / "E0s_dft.json"
        e0s = data.load_E0s_cache(str(cache))
        if e0s is None:
            e0s = data.fit_atomic_E0s(ase.io.read(dataset, ":"))
        args.E0s = e0s
    for f in (args.para_file, args.log_file):
        Path(f).write_text("#\n")
    dft_input = ase.io.read(dataset, ":")
    dft_center = dft_input[floor(args.eos_points / 2)::args.eos_points]
    if mode in ("dataset", "reaction"):
        bolt_f, bolt_f11 = data.get_bolt_factor(dft_center, "identical", 3000)
    else:
        bolt_f, bolt_f11 = data.get_bolt_factor(dft_center, "boltzmann", 3000)
    cov_radius = penalty_radius(dft_input, [])
    loss = Loss(args_input=args, dft_input=dft_input, bolt_f11=bolt_f11, opt_elem=[],
                cov_radius=cov_radius, path=workdir)
    return loss, cov_radius


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--mode", default="dataset", choices=["dataset", "energygeometry", "reaction"])
    ap.add_argument("--eos-points", type=int, default=11)
    ap.add_argument("--points", nargs="+", default=["3.0,6.0"], help="r0_w,r0_d pairs (Bohr)")
    ap.add_argument("--engines", nargs="+", default=["dftbplus", "pydftb"])
    ap.add_argument("--pydftb-forces", default="dftbplus", choices=["dftbplus", "exact"])
    ap.add_argument("--workdir", type=Path, default=Path("engine_parity"))
    ap.add_argument("--ncores", nargs="+", default=None, metavar="N",
                    help="core budgets (integers or 'auto') to evaluate each point with; "
                         "default: no parallel plan (serial, thread settings untouched)")
    ap.add_argument("--compare-serial", action="store_true",
                    help="--ncores 1 auto, and check that the results are identical")
    ap.add_argument("--json", type=Path, default=None, help="write results here")
    ap.add_argument("--confinement", default="selfconsistent",
                    choices=["selfconsistent", "perturbative"],
                    help="hotcent's confinement scheme (M4e; perturbative: as before M4e)")
    a = ap.parse_args(argv)
    from utils.hotpy import set_confinement
    set_confinement(a.confinement)

    import utils.engines as eng
    from utils.parallel import detect_cpu, set_plan
    from utils.recommender import make_plan, workload_from_dataset

    dataset = a.dataset.resolve()
    points = [tuple(float(x) for x in p.split(",")) for p in a.points]
    budgets = ["1", "auto"] if a.compare_serial else (a.ncores or [None])
    results = {}
    cwd = os.getcwd()
    for engine in a.engines:
        for nc in budgets:
            tag = engine if nc is None else f"{engine}_n{nc}"
            wd = (a.workdir / f"{a.mode}_{tag}").resolve()
            wd.mkdir(parents=True, exist_ok=True)
            eng.configure(engine=engine, pydftb_forces=a.pydftb_forces)
            loss, _ = build_loss(dataset, a.mode, wd, a.eos_points)
            plan = None
            if nc is not None:
                work = workload_from_dataset(loss.dft_input, loss.bolt_f11,
                                             kpts_density=loss.args_input.kpt_density,
                                             eos_points=loss.args_input.eos_points,
                                             engine=engine, mode=a.mode)
                plan = make_plan(detect_cpu(nc), work, n_calls=1)
                print(plan.render())
            set_plan(plan)
            for p in points:
                t0 = time.perf_counter()
                mu = float(loss.Loss_EnergyGeometry(list(p)))
                wall = time.perf_counter() - t0
                os.chdir(cwd)                              # EnergyGeometry changes directory
                opt_dir = wd / loss.args_input.results_dir / f"opt_{p[0]:.3f}_{p[1]:.3f}"
                results.setdefault(p, {})[tag] = {"mu": mu, "mu_hex": mu.hex(), "wall": wall,
                                                  "stages": getattr(loss, "step_timings", None),
                                                  "estimates": dict(plan.estimates) if plan else None,
                                                  "files": step_digests(opt_dir)}
                print(f"[{tag}] r0_w={p[0]:.3f} r0_d={p[1]:.3f}  mu={mu:.10g}  ({wall:.1f} s)",
                      flush=True)
            set_plan(None)
    tags = list(next(iter(results.values())))
    print(f"\n{'r0_w':>6} {'r0_d':>6}  " + "  ".join(f"{t:>16}" for t in tags) + "   rel.diff")
    for p, r in results.items():
        mus = [r[t]["mu"] for t in tags]
        rel = abs(mus[-1] - mus[0]) / max(abs(mus[0]), 1e-300) if len(mus) > 1 else 0.0
        print(f"{p[0]:6.3f} {p[1]:6.3f}  " + "  ".join(f"{m:16.10g}" for m in mus) + f"   {rel:.2e}")
    print(f"\n{'run':>18}  " + "".join(f"{st:>9}" for st in ("hotcent", "dftb", "physrep", "total")))
    for p, r in results.items():
        for tag, x in r.items():
            if x.get("stages"):
                print(f"{tag:>18}  " + "".join(f"{x['stages'][st]:8.1f}s"
                                               for st in ("hotcent", "dftb", "physrep", "total")))
            if x.get("estimates"):
                est = x["estimates"]
                print(f"{'(estimated)':>18}  " + "".join(f"{est[st]:8.1f}s"
                                                         for st in ("hotcent", "dftb", "physrep", "step")))
    same = True
    if len(budgets) > 1:
        for p, r in results.items():
            for engine in a.engines:
                runs = [r[f"{engine}_n{nc}"] for nc in budgets]
                ok = all(x["mu_hex"] == runs[0]["mu_hex"] and x["files"] == runs[0]["files"]
                         for x in runs[1:])
                same &= ok
                print(f"{engine} r0_w={p[0]:.3f} r0_d={p[1]:.3f}: --ncores {' / '.join(budgets)} "
                      f"{'bitwise identical' if ok else 'DIFFER'} (loss and "
                      f"{len(runs[0]['files'])} files)")
    if a.json:
        a.json.write_text(json.dumps({f"{p[0]},{p[1]}": r for p, r in results.items()}, indent=1))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
