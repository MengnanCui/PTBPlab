"""DFTB+ configuration advisor (milestone M3d).

Reads a DFTB+ run -- ``dftb_in.hsd``, and if present ``dftb_pin.hsd``, the screen
output and the SKF files -- and reports the pitfalls found while validating pydftb
against DFTB+ (``docs/dftbplus_map.md``): input DFTB+ silently rewrites or ignores,
parameters that hit numerical weak spots, parallel set-ups that oversubscribe the
machine, results returned from unconverged SCC loops.  It only reports; nothing is
changed and DFTB+ is not run.

    pydftb advisor RUN_DIR [--json] [--fail-on warning]     # or python -m pydftb.advisor

    from pydftb.advisor import advise
    run, findings = advise("run_dir")
"""
from __future__ import annotations

from .inputs import RunFiles, collect
from .rules import RULES, SEVERITIES, Finding, run_rules


def advise(path, *, stdout=None, environ=None, cores=None, mpi_procs=None, concurrent: int = 1,
           context: str | None = None, only=None):
    """(RunFiles, findings) for the run at ``path`` (a directory or a dftb_in.hsd).

    ``context="ptbp-legacy"`` marks the findings that PTBP's historical input expects.
    """
    run = collect(path, stdout=stdout, environ=environ, cores=cores, mpi_procs=mpi_procs,
                  concurrent=concurrent, context=context)
    return run, run_rules(run, only=only)


__all__ = ["advise", "collect", "run_rules", "RunFiles", "Finding", "RULES", "SEVERITIES"]
