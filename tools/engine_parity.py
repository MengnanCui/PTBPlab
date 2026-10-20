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
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from math import floor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_loss(dataset: Path, mode: str, workdir: Path, eos_points: int):
    import ase.io
    from ase.data import covalent_radii
    from ase.units import Bohr

    import utils.data as data
    from cli.arg_parser import build_default_arg_parser
    from utils.loss import Loss

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
    numbers, _, _, _ = data.get_elements_from_dataset(dft_input, [])
    cov_radius = covalent_radii[numbers[-1]] / Bohr
    loss = Loss(args_input=args, dft_input=dft_input, bolt_f11=bolt_f11, opt_elem=[], path=workdir)
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
    ap.add_argument("--json", type=Path, default=None, help="write results here")
    a = ap.parse_args(argv)

    import utils.engines as eng

    dataset = a.dataset.resolve()
    points = [tuple(float(x) for x in p.split(",")) for p in a.points]
    results = {}
    cwd = os.getcwd()
    for engine in a.engines:
        wd = (a.workdir / f"{a.mode}_{engine}").resolve()
        wd.mkdir(parents=True, exist_ok=True)
        eng.configure(engine=engine, pydftb_forces=a.pydftb_forces)
        loss, _ = build_loss(dataset, a.mode, wd, a.eos_points)
        for p in points:
            t0 = time.perf_counter()
            mu = float(loss.Loss_EnergyGeometry(list(p)))
            wall = time.perf_counter() - t0
            os.chdir(cwd)                                  # EnergyGeometry changes directory
            results.setdefault(p, {})[engine] = {"mu": mu, "wall": wall}
            print(f"[{engine}] r0_w={p[0]:.3f} r0_d={p[1]:.3f}  mu={mu:.10g}  ({wall:.1f} s)",
                  flush=True)
    print(f"\n{'r0_w':>6} {'r0_d':>6}  " + "  ".join(f"{e:>16}" for e in a.engines) + "   rel.diff")
    for p, r in results.items():
        mus = [r[e]["mu"] for e in a.engines]
        rel = abs(mus[-1] - mus[0]) / max(abs(mus[0]), 1e-300) if len(mus) > 1 else 0.0
        print(f"{p[0]:6.3f} {p[1]:6.3f}  " + "  ".join(f"{m:16.10g}" for m in mus) + f"   {rel:.2e}")
    if a.json:
        a.json.write_text(json.dumps({f"{p[0]},{p[1]}": r for p, r in results.items()}, indent=1))


if __name__ == "__main__":
    main()
