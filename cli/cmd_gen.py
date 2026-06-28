"""`ptbp gen` — Slater-Koster file generator.

Translates the new subcommand surface into the legacy flat CLI flags
(`--skf_generator full --symbols ... --known_parameters PTBP`) so the
existing `cli/run.py` body keeps doing the actual work.

Phase M.1 is intentionally a thin translation layer; the SKF generation
itself is unchanged.
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
        'gen',
        help="Generate Slater-Koster files from published or user parameters.",
        description="Generate SKF files for a list of element symbols. "
                    "Uses hotcent + physrep — no DFTB+ required.",
    )
    p.add_argument('symbols', nargs='+', metavar='SYMBOL',
                   help="Element symbols, e.g. H O Cu.")
    p.add_argument('--params', default='PTBP',
                   choices=['PTBP', 'QNplusRep', 'QUASINANO2013', 'Prior', 'Personal',
                            'ptbp', 'qnplusrep', 'quasinano2013', 'prior', 'personal'],
                   help="Which published parameter set to seed from. Default: PTBP.")
    p.add_argument('--mode', default='full', choices=['full', 'band', 'rep'],
                   help="Whole SKF (band+rep), band-only, or rep-only. Default: full.")
    p.add_argument('--r0', nargs='+', type=float, default=None, metavar='F',
                   help="Confinement parameters r0_w r0_d p (per element, "
                        "3*N_symbols values). Required for --params Personal.")
    p.add_argument('--rep', nargs='+', type=float, default=None, metavar='F',
                   help="Repulsive sigma_rep kxc (per element, 2*N_symbols values). "
                        "Optional for --params Personal.")
    p.add_argument('--xc', default='GGA_X_PBE+GGA_C_PBE',
                   help="Exchange-correlation functional. Default: %(default)s.")
    p.add_argument('--out', default=None, metavar='DIR',
                   help="Output directory for the .skf files. Default: cwd.")
    p.set_defaults(func=run)
    return p


def run(args: argparse.Namespace) -> int:
    """Translate to legacy --skf_generator invocation and runpy cli/run.py."""
    if args.out:
        out_dir = Path(args.out).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(out_dir)

    legacy_argv = [
        'run.py',
        '--skf_generator', args.mode,
        '--symbols', *args.symbols,
        '--known_parameters', args.params,
        '--xc', args.xc,
    ]
    if args.r0 is not None:
        legacy_argv += ['--confinement_parameters', *map(str, args.r0)]
    if args.rep is not None:
        legacy_argv += ['--repulsive_parameters', *map(str, args.rep)]

    sys.argv = legacy_argv
    runpy.run_path(str(LEGACY_RUN), run_name='__main__')
    return 0
