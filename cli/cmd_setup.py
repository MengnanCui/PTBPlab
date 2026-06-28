"""`ptbp setup` — Textual TUI wizard.

Setup-only: the wizard walks the user through dataset selection, shallow
validation, mode/optimizer/E0s choice, and writes a `ptbp_run.yaml` +
prints the equivalent CLI command. Long-running jobs are NOT launched
from the TUI — keeps responsibilities clean.

Usage:
    ptbp setup                      # welcome menu (optimize or gen)
    ptbp setup optimize             # straight into the optimize wizard
    ptbp setup gen                  # straight into the gen wizard
    ptbp setup --output FILE.yaml   # where to save (default ./ptbp_run.yaml)
"""
from __future__ import annotations

import argparse
from pathlib import Path


def add_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        'setup',
        help="Interactive TUI wizard to assemble an optimize/gen invocation.",
        description="Multi-step terminal wizard. Picks a dataset, validates "
                    "it, lets you choose mode / optimizer / E0s / advanced "
                    "knobs, then writes a ptbp_run.yaml + prints the "
                    "equivalent `ptbp ...` command for you to run.",
    )
    p.add_argument('flow', nargs='?', default=None,
                   choices=['optimize', 'gen'],
                   help="Skip the welcome menu and go straight into a flow.")
    p.add_argument('--output', default=None, metavar='FILE',
                   help="Where to save the generated YAML. "
                        "Default: ./ptbp_run.yaml.")
    p.set_defaults(func=run)
    return p


def run(args: argparse.Namespace) -> int:
    import os
    import shutil
    import sys
    from cli.tui.app import PtbpSetupApp

    output_yaml = Path(args.output).resolve() if args.output else None
    app = PtbpSetupApp(initial_flow=args.flow, output_yaml=output_yaml)
    result = app.run()
    if not isinstance(result, dict):
        return 0

    if 'yaml_path' in result:
        print()
        print(f"Saved: {result['yaml_path']}")
    if 'command' in result:
        print(f"Cmd:   {result['command']}")

    # "Save & Run" / "Run now" — replace the current process with the
    # assembled ptbp command via os.execvp so output streams natively to
    # the user's terminal (Textual has already torn down by the time
    # app.run() returns, so the terminal is back in cooked mode).
    if result.get('launch_now') and result.get('argv'):
        argv = list(result['argv'])
        # Resolve `ptbp` to its absolute path so execvp works even on the
        # off chance PATH was perturbed. argv[0] is `ptbp`.
        ptbp_bin = shutil.which(argv[0])
        if ptbp_bin is None:
            print(f"error: '{argv[0]}' not on PATH; can't launch directly. "
                  f"Run the command above manually.", file=sys.stderr)
            return 0
        print(f"Launching {argv[0]} (replacing the wizard process)...\n")
        os.execvp(ptbp_bin, argv)
        # os.execvp doesn't return on success.
    return 0
