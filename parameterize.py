"""User-facing entry point for the PTBP toolkit.

After `pip install -e .` this is wired up as the `ptbp` console script.

Two CLI surfaces coexist:

1. **New subcommand CLI** (preferred):

    ptbp gen H O --params PTBP
    ptbp optimize my_data.xyz --mode dataset --n-calls 30
    ptbp postprocess run_2026-05-05_22-30/

2. **Legacy flat CLI** (deprecated; kept for backwards compat with
   existing examples/ scripts and external usage):

    ptbp --skf_generator full --symbols H O --known_parameters PTBP
    ptbp --optimization_option dataset --dft_file my_data.xyz ...

When `sys.argv[1]` is one of {gen, optimize, postprocess}, the new
dispatcher handles it. Otherwise the legacy `cli/run.py` runs (with a
deprecation note printed to stderr).
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

NEW_SUBCOMMANDS = {'gen', 'optimize', 'postprocess', 'setup'}


def main() -> int:
    cli_run = Path(__file__).resolve().parent / 'cli' / 'run.py'

    first = sys.argv[1] if len(sys.argv) >= 2 else None

    # `ptbp` (no args) or `ptbp --help` / `ptbp -h` → show the new
    # subcommand-style help so users discover gen / optimize / postprocess.
    if first in (None, '--help', '-h', 'help'):
        from cli.main import dispatch
        return dispatch(['--help'] if first in (None, 'help') else [first])

    if first in NEW_SUBCOMMANDS:
        from cli.main import dispatch
        return dispatch(sys.argv[1:])

    # Legacy path: keep working unchanged for examples/0[1-5]/run.sh and any
    # existing user scripts. Print a one-line note so the user knows there's
    # a friendlier surface.
    print(
        "[ptbp] Using the legacy flat CLI. The new subcommand surface "
        "(`ptbp gen` / `ptbp optimize` / `ptbp postprocess`) is "
        "preferred — see `ptbp --help`.",
        file=sys.stderr,
    )
    runpy.run_path(str(cli_run), run_name='__main__')
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
