"""`ptbp gui` — launch the customtkinter desktop app.

Kept import-light: `add_parser` uses only argparse so building the top-level
parser never pulls in customtkinter/Tk. The heavy import happens inside
`run()`, guarded so a missing dependency produces a friendly message rather
than a traceback.
"""
from __future__ import annotations

import argparse
import sys


def add_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        'gui',
        help="Launch the desktop GUI (customtkinter).",
        description="A modern desktop front end: structure/EOS builder, SKF "
                    "generation, optimisation setup + live monitoring, and a "
                    "dependency doctor. Needs the 'gui' extra: pip install "
                    "'ptbp[gui]'.",
    )
    p.add_argument('--project', default=None, metavar='DIR',
                   help="Working directory the GUI opens in. Default: cwd.")
    p.set_defaults(func=run)
    return p


def run(args: argparse.Namespace) -> int:
    try:
        import customtkinter  # noqa: F401  (probe before importing the app)
    except Exception as e:
        print(
            "error: the GUI needs customtkinter (and a working Tk).\n"
            f"  import failed: {type(e).__name__}: {e}\n"
            "  install it with:  pip install 'ptbp[gui]'\n"
            "  (on Linux you may also need the system Tk: apt install python3-tk)",
            file=sys.stderr,
        )
        return 1

    from gui.app import main
    return main(project_dir=args.project)
