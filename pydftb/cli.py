"""The ``pydftb`` command: pydftb's tools behind one entry point (P1).

    pydftb advisor RUN_DIR            check a DFTB+ input and run for known pitfalls
    pydftb refgen --check             pydftb against the frozen DFTB+ references
    pydftb dynamics --check           optimisations and MD against DFTB+'s drivers
    pydftb golden --base COMMIT       NumPy path bitwise unchanged since COMMIT (git checkout)
    pydftb gpu-check --quick          GPU self-check (needs torch)
    pydftb gradcheck --check          pydftb.autodiff against finite differences (needs torch)

``pydftb COMMAND ARGS...`` imports the tool only then and hands ARGS unchanged to its
``main(argv)``; ``pydftb COMMAND --help`` shows the tool's own options.  ``python -m pydftb``
is the same command, and the module forms (``python -m pydftb.validation.refgen --check``,
``python -m pydftb.advisor RUN_DIR``, ...) keep working.
"""
from __future__ import annotations

import argparse
import importlib
import sys

from . import __version__

#: command -> (module with ``main(argv=None, prog=None)``, summary for ``pydftb --help``)
COMMANDS = {
    "advisor": ("pydftb.advisor.__main__",
                "check a DFTB+ input and run for known pitfalls"),
    "refgen": ("pydftb.validation.refgen",
               "frozen DFTB+ references: --check, --live, --regenerate"),
    "dynamics": ("pydftb.validation.dynamics",
                 "optimisations and MD vs DFTB+: --check, --regenerate"),
    "golden": ("pydftb.validation.golden",
               "bitwise regression vs an earlier commit (git checkout)"),
    "gpu-check": ("pydftb.validation.gpu_check",
                  "GPU self-check: correctness and timings (needs torch)"),
    "gradcheck": ("pydftb.validation.gradcheck",
                  "autodiff vs finite differences (needs torch)"),
}
ALIASES = {"gpu_check": "gpu-check"}

EPILOG = """\
pydftb COMMAND --help shows the options of a command.  The module forms keep working,
e.g. python -m pydftb.validation.refgen --check.  The --regenerate modes write into the
package's tests/data (pydftb/tests/data): run them in a git checkout, not an installed copy."""


def build_parser() -> argparse.ArgumentParser:
    """The top-level parser: ``--help``, ``--version`` and the list of commands (each
    command parses its own arguments)."""
    ap = argparse.ArgumentParser(prog="pydftb", epilog=EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description="pydftb's command-line tools.")
    ap.add_argument("--version", action="version", version=f"pydftb {__version__}")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND", title="commands")
    for name, (_, summary) in COMMANDS.items():
        aliases = [alias for alias, target in ALIASES.items() if target == name]
        sub.add_parser(name, aliases=aliases, help=summary, add_help=False)
    return ap


def main(argv=None) -> int:
    """``pydftb COMMAND ARGS...``: the exit status of the command's ``main``.

    ``--help``, ``--version`` and usage errors exit through argparse (``SystemExit``), as
    do the commands' own parsers and ``sys.exit`` calls.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["help"]:                       # pydftb help [COMMAND]
        argv = argv[1:2] + ["--help"]
    name = ALIASES.get(argv[0], argv[0]) if argv else None
    if name not in COMMANDS:
        # nothing, --help, --version, an unknown option or command: argparse's handling
        ap = build_parser()
        ap.parse_args(argv or ["--help"])
        ap.error("expected a command first: pydftb COMMAND [ARGS...]")    # e.g. "-- refgen"
    module, _ = COMMANDS[name]
    try:
        rc = importlib.import_module(module).main(argv[1:], prog=f"pydftb {name}")
    except ModuleNotFoundError as err:
        if err.name != "torch":
            raise
        print(f"pydftb {name}: needs PyTorch, which is not installed "
              f"(pip install 'pydftb[torch]'; GPU builds: https://pytorch.org)", file=sys.stderr)
        return 2
    return int(rc or 0)
