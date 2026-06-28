"""`ptbp postprocess RUN_DIR` — re-emit plots/packaging from a prior run.

Phase M.1 wires this as a thin alias of `ptbp optimize ... --postprocess-only`.
The path argument is interpreted as the run cwd; M.2 will turn it into a
proper run-folder pointer with all artifacts inside.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_RUN = REPO_ROOT / 'cli' / 'run.py'


def add_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        'postprocess',
        help="Re-emit plots/packaging from a prior run's result.pkl.",
        description="Skip the optimisation entirely. Loads <RUN_DIR>/result.pkl "
                    "and regenerates convergence.png + best opt_<r0_w>_<r0_d>/ "
                    "+ results.tgz. Useful when a long PSO/BO run completed but "
                    "post-processing failed.",
    )
    p.add_argument('run_dir', metavar='RUN_DIR',
                   help="Directory containing a prior run's result.pkl "
                        "(and dft/ symlink, par.out, log.out etc.).")
    p.set_defaults(func=run)
    return p


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"error: run_dir not found or not a directory: {run_dir}", file=sys.stderr)
        return 2
    result_pkl = run_dir / 'result.pkl'
    if not result_pkl.exists():
        print(f"error: {result_pkl} not found — run optimize once first.", file=sys.stderr)
        return 2

    os.chdir(run_dir)

    # Prefer ptbp_run.yaml (written by `ptbp optimize` since M.3) for
    # mode and dataset. Fall back to glob discovery for runs that
    # predate that snapshot.
    mode = 'dataset'
    dft_file: str | None = None

    yaml_path = run_dir / 'ptbp_run.yaml'
    if yaml_path.exists():
        from cli import config as _config
        snap = _config.load_yaml(yaml_path)
        mode = snap.get('mode', 'dataset')
        ds_path = snap.get('dataset')
        if ds_path:
            dft_file = Path(ds_path).name

    if dft_file is None:
        dft_dir = run_dir / 'dft'
        candidates = sorted(dft_dir.glob('*.xyz')) if dft_dir.exists() else []
        if not candidates:
            print(f"error: cannot locate the original dataset. Looked at "
                  f"{yaml_path} and {dft_dir}.", file=sys.stderr)
            return 2
        dft_file = candidates[0].name

    legacy_argv = [
        'run.py',
        '--ref_dir', 'dft',
        '--dft_file', dft_file,
        '--optimization_option', mode,
        '--parameters', 'r0_w', 'r0_d', 'sigma_rep',
        '--superposition', 'density',
        '--postprocess_only',
    ]
    sys.argv = legacy_argv
    runpy.run_path(str(LEGACY_RUN), run_name='__main__')
    return 0
