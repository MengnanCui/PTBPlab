"""``pydftb advisor PATH`` (or ``python -m pydftb.advisor PATH``): check a DFTB+ run
directory (or a dftb_in.hsd).

Exit status: 0 no finding at or above ``--fail-on`` (default: error), 1 there is one,
2 usage error or no input, 3 the input could not be read.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import advise, hsd
from .report import render_json, render_text
from .rules import RULES, SEVERITIES


def main(argv=None, prog=None) -> int:
    ap = argparse.ArgumentParser(prog=prog or "python -m pydftb.advisor",
                                 description="Check a DFTB+ input and run for known pitfalls "
                                             "(report only; nothing is changed).")
    ap.add_argument("path", nargs="?", help="run directory or dftb_in.hsd")
    ap.add_argument("--stdout", help="DFTB+ screen output (default: the *.out file with "
                                     "DFTB+'s banner in the directory)")
    ap.add_argument("--json", action="store_true", help="print JSON")
    ap.add_argument("--fail-on", default="error", choices=SEVERITIES + ("never",),
                    help="exit with 1 when a finding of this severity or worse is found "
                         "(default: error)")
    ap.add_argument("--cores", type=int, default=None, help="cores of the machine DFTB+ runs on")
    ap.add_argument("--mpi-procs", type=int, default=None, help="MPI processes DFTB+ runs with")
    ap.add_argument("--list-rules", action="store_true", help="list the rules and exit")
    ap.add_argument("--explain", metavar="RULE", help="explain one rule and exit")
    a = ap.parse_args(argv)

    if a.list_rules:
        for r in RULES.values():
            print(f"{r.id}  {r.severity:<7} {r.title}  [{r.source}]")
        return 0
    if a.explain:
        r = RULES.get(a.explain.upper())
        if r is None:
            print(f"unknown rule {a.explain!r}; see --list-rules", file=sys.stderr)
            return 2
        print(f"{r.id}  {r.title}\nseverity: {r.severity}\nsource: {r.source}\n\n{r.explain}")
        return 0
    if not a.path:
        ap.print_usage(sys.stderr)
        return 2
    path = Path(a.path)
    if not (path / "dftb_in.hsd" if path.is_dir() else path).is_file():
        print(f"no dftb_in.hsd at {path}", file=sys.stderr)
        return 2
    try:
        run, findings = advise(path, stdout=a.stdout, cores=a.cores, mpi_procs=a.mpi_procs)
    except (hsd.HSDError, OSError, UnicodeDecodeError) as err:
        print(f"cannot read the input: {err}", file=sys.stderr)
        return 3
    print(render_json(run, findings) if a.json else render_text(run, findings))
    if a.fail_on == "never":
        return 0
    worst = SEVERITIES.index(a.fail_on)
    return 1 if any(SEVERITIES.index(f.severity) <= worst for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
