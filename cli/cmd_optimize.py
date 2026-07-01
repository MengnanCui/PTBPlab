"""`ptbp optimize` — Parameter optimisation.

Translates the new subcommand surface into the legacy flat CLI flags so
the existing `cli/run.py` optimization branch keeps doing the actual
work.

Phase M.1 is a thin translation layer. M.2 will add the `--output`
run-folder semantics; M.3 the YAML config; M.4 auto-mode.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_RUN = REPO_ROOT / 'cli' / 'run.py'

# Module-global so cli/config.py can recover the parser's defaults to
# distinguish "user passed this on the CLI" from "argparse filled in the
# default value" — needed for the CLI-overrides-YAML merge.
_PARSER: argparse.ArgumentParser | None = None


def add_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        'optimize',
        help="Optimise DFTB parameters against a DFT reference dataset.",
        description="Run the outer-loop optimiser (BO / parallel_BO / PSO) over "
                    "(r0_w, r0_d) with the inner-loop sigma_rep brute search.",
    )
    p.add_argument('dataset', metavar='DATASET',
                   help="Path to the DFT reference (xyz/traj/extxyz/json).")
    p.add_argument('--mode', default='auto',
                   choices=['auto', 'energygeometry', 'dataset', 'reaction', 'bandstructure'],
                   help="Which loss to optimise. Default: auto — picks "
                        "bandstructure for .json input, energygeometry when a "
                        "fit.json sits next to the dataset, reaction when the "
                        "dataset spans multiple chemical formulas, and dataset "
                        "otherwise. Override explicitly to skip detection.")
    p.add_argument('--optimizer', default='bo', choices=['bo', 'parallel_bo', 'pso'],
                   help="Outer-loop optimiser. Default: bo.")
    p.add_argument('--n-calls', type=int, default=None, metavar='N',
                   dest='n_calls',
                   help="Total BO/PSO evaluations. Default: 11 for energygeometry/"
                        "dataset/reaction, 100 for bandstructure.")
    p.add_argument('--n-particles', type=int, default=4, metavar='N',
                   dest='n_particles',
                   help="Population/batch width for PSO and parallel_bo. Default: 4.")
    p.add_argument('--E0s', default=None, metavar='JSON',
                   help='Atomic reference energies as JSON, e.g. \'{"6": -37.8, "1": -0.5}\'. '
                        'Omit to auto-fit from the dataset and cache to <ref_dir>/E0s_dft.json.')
    p.add_argument('--eos-points', type=int, default=11, metavar='N',
                   dest='eos_points',
                   help="EOS points per phase. Default: 11; ignored / set to 1 for "
                        "non-EOS modes by run.py.")
    p.add_argument('--xc', default='GGA_X_PBE+GGA_C_PBE',
                   help="Exchange-correlation functional. Default: %(default)s.")
    p.add_argument('--kpts', type=float, default=5.0, metavar='F',
                   dest='kpt_density',
                   help="DFTB+ k-point density. Default: 5.0.")
    p.add_argument('--params', nargs='+', default=['r0_w', 'r0_d', 'sigma_rep'],
                   metavar='P', dest='parameters',
                   help="Parameters to optimise. Default: r0_w r0_d sigma_rep.")
    p.add_argument('--multi-element', nargs='+', default=None, metavar='E',
                   dest='multi_element',
                   help="Element symbol(s) whose params are jointly optimised; "
                        "others are held at PTBP defaults.")
    p.add_argument('--superposition', default='density',
                   choices=['density', 'potential'],
                   help="Density vs potential superposition. Default: density.")
    p.add_argument('--seed', type=int, default=None, metavar='N',
                   help="Random seed for reproducibility (numpy + optimiser "
                        "random_state). Default: 123 in run.py if omitted.")
    p.add_argument('--checkpoint', default='checkpoint.pkl',
                   help="Checkpoint filename (for skopt resume). Default: %(default)s.")
    p.add_argument('--postprocess-only', action='store_true',
                   dest='postprocess_only',
                   help="Skip optimisation; reuse ./result.pkl to redo plots/packaging.")
    p.add_argument('--output', default=None, metavar='DIR',
                   help="Run folder for all outputs (par.out, log.out, "
                        "results/, result.pkl, ptbp_run.yaml, summary.json, "
                        "convergence.png). Default: ./run_<UTC-timestamp>/.")
    p.add_argument('--config', default=None, metavar='FILE',
                   help="Load defaults from a YAML file. CLI flags override "
                        "individual fields. After a successful run, the "
                        "resolved (effective) settings are written back to "
                        "<output>/ptbp_run.yaml.")
    p.set_defaults(func=run)
    global _PARSER
    _PARSER = p
    return p


def _default_output_dir() -> Path:
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')
    return Path.cwd() / f"run_{ts}"


def _write_summary(run_dir: Path, args: argparse.Namespace,
                   dataset: Path, started_at: float) -> None:
    """Best-effort summary.json next to result.pkl. Failures here should
    NOT propagate — postprocess can always be re-run later."""
    import json
    import time
    try:
        result_pkl = run_dir / 'result.pkl'
        if not result_pkl.exists():
            return
        import pickle
        with open(result_pkl, 'rb') as fh:
            res = pickle.load(fh)
        runtime_s = round(time.time() - started_at, 2)
        summary = {
            'optimizer': getattr(res, 'optimizer', args.optimizer),
            'mode': args.mode,
            'dataset': str(dataset),
            'x_best': list(map(float, res.x)),
            'fun_best': float(res.fun),
            'n_evaluations': len(getattr(res, 'x_iters', [])),
            'runtime_s': runtime_s,
            'n_calls_requested': args.n_calls,
            'n_particles': args.n_particles,
        }
        # If reaction mode produced a per-formula CSV, point at it.
        per_formula = list(run_dir.glob('results/opt_*/reaction_per_formula.csv'))
        if per_formula:
            summary['reaction_per_formula_csv'] = [str(p.relative_to(run_dir))
                                                    for p in per_formula]
        with open(run_dir / 'summary.json', 'w') as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        print(f"[run-folder] summary.json written to {run_dir / 'summary.json'}")
    except Exception as e:
        print(f"[run-folder] summary.json write failed (non-fatal): {e}",
              file=sys.stderr)


def run(args: argparse.Namespace) -> int:
    """Translate to legacy --optimization_option ... and runpy cli/run.py.

    All relative paths the legacy code uses (par.out, log.out, results/,
    result.pkl, etc.) get redirected into `--output DIR` by chdir'ing
    before invoking the legacy entry. Dataset is converted to an absolute
    path so the chdir doesn't break the read.
    """
    import time

    # Layer YAML config underneath CLI flags. CLI values that match the
    # parser's default are treated as "not specified" and fall through to
    # the YAML value if present.
    if getattr(args, 'config', None):
        from cli import config as _config
        yaml_dict = _config.load_yaml(args.config)
        defaults = _config.get_parser_defaults(_PARSER) if _PARSER else {}
        _config.merge_into_namespace(args, yaml_dict, defaults)

    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        print(f"error: dataset not found: {dataset}", file=sys.stderr)
        return 2

    if args.mode == 'auto':
        from cli import auto_mode
        detected, rationale = auto_mode.detect_mode(dataset)
        print(f"[auto] mode = {detected} ({rationale})")
        args.mode = detected

    # Resolve output dir; create if needed.
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = _default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run-folder] writing all outputs into {output_dir}")
    os.chdir(output_dir)

    # Symlink dataset into <output_dir>/dft/ so postprocess can find it
    # later via run_dir/dft/dft.xyz (M.3 will generalise this via
    # ptbp_run.yaml).
    dft_link = output_dir / 'dft'
    if not dft_link.exists():
        dft_link.symlink_to(dataset.parent)

    legacy_argv = [
        'run.py',
        '--ref_dir', str(dataset.parent),
        '--dft_file', dataset.name,
        '--optimization_option', args.mode,
        '--parameters', *args.parameters,
        '--superposition', args.superposition,
        '--xc', args.xc,
        '--kpt_density', str(args.kpt_density),
        '--eos_points', str(args.eos_points),
        '--optimizer', args.optimizer,
        '--n_particles', str(args.n_particles),
        '--checkpoint', args.checkpoint,
    ]
    if args.n_calls is not None:
        legacy_argv += ['--n_calls', str(args.n_calls)]
    if getattr(args, 'seed', None) is not None:
        legacy_argv += ['--seed', str(args.seed)]
    if args.E0s is not None:
        legacy_argv += ['--E0s', args.E0s]
    if args.multi_element:
        legacy_argv += ['--multi_element', *args.multi_element]
    if args.postprocess_only:
        legacy_argv += ['--postprocess_only']

    sys.argv = legacy_argv
    started_at = time.time()
    try:
        runpy.run_path(str(LEGACY_RUN), run_name='__main__')
    finally:
        _write_summary(output_dir, args, dataset, started_at)
        # Snapshot effective settings so the user can re-run with
        # `--config <run_dir>/ptbp_run.yaml` or `ptbp postprocess <run_dir>`.
        try:
            from cli import config as _config
            _config.write_effective(args, output_dir)
        except Exception as e:
            print(f"[run-folder] ptbp_run.yaml write failed (non-fatal): {e}",
                  file=sys.stderr)
    return 0
