"""Top-level dispatcher for the new subcommand CLI.

Layout:
    ptbp gen [SYMBOLS...]               # SKF generation
    ptbp optimize DATASET [...]          # Parameter optimization
    ptbp postprocess RUN_DIR             # Re-emit plots / packaging from a prior run

Backwards-compat: when `parameterize.main()` sees a sys.argv[1] that is NOT
one of {gen, optimize, postprocess}, it falls through to the legacy flat
CLI (`cli/run.py`) with a deprecation warning. So this dispatcher only
handles the new path; legacy stays untouched.

In Phase M.1 each subcommand translates its parsed args into the legacy
sys.argv then calls cli/run.py via runpy. Subsequent phases (run-folder,
YAML config, auto mode) layer on top of this scaffold.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_RUN = REPO_ROOT / 'cli' / 'run.py'

SUBCOMMANDS = {'gen', 'optimize', 'postprocess', 'setup'}


def _build_top_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='ptbp',
        description="Density Functional Tight-Binding Parameterization toolkit.",
    )
    sub = parser.add_subparsers(dest='cmd', metavar='COMMAND', required=True)

    from cli import cmd_gen, cmd_optimize, cmd_postprocess, cmd_setup
    cmd_gen.add_parser(sub)
    cmd_optimize.add_parser(sub)
    cmd_postprocess.add_parser(sub)
    cmd_setup.add_parser(sub)

    return parser


def dispatch(argv: list[str] | None = None) -> int:
    """Entry point for new subcommand CLI. Returns process exit code."""
    parser = _build_top_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(dispatch())
