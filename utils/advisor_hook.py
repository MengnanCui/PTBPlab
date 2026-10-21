"""The DFTB+ configuration advisor inside PTBP runs (milestone M3d): it only reports.

``cli/run.py`` calls :func:`configure` for ``ptbp optimize`` runs with the DFTB+ engine.
The DFTB stage then shows the advisor the first DFTB+ run directory of the run
(:func:`check_once`) and, when a DFTB+ run fails, the directory it kept
(:func:`explain_failure`, the findings are also attached to the exception).  Findings
are printed with an ``[advisor]`` prefix and collected in ``advisor.json`` in the run
directory.  Nothing here raises, and no setting is changed; ``PTBP_ADVISOR=0`` turns it off.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PREFIX = "[advisor] "
_STATE = {"run_dir": None, "context": None, "checked": False}


def configure(run_dir, legacy: bool = True) -> None:
    """Check DFTB+ runs of this PTBP run; ``legacy``: PTBP's historical DFTB+ input
    (``--smearing none``), whose findings P01 and P03 are expected."""
    if os.environ.get("PTBP_ADVISOR", "").strip() == "0":
        _STATE.update(run_dir=None, checked=False)
        return
    _STATE.update(run_dir=Path(run_dir), context="ptbp-legacy" if legacy else "ptbp",
                  checked=False)


def reset() -> None:
    _STATE.update(run_dir=None, context=None, checked=False)


def wants_check() -> bool:
    """True until the first DFTB+ run of this PTBP run has been checked."""
    return _STATE["run_dir"] is not None and not _STATE["checked"]


def _advise(directory, concurrent: int = 1):
    from pydftb.advisor import advise, report
    run, findings = advise(directory, concurrent=concurrent, context=_STATE["context"])
    return run, findings, report


def _save(entry: dict) -> None:
    path = _STATE["run_dir"] / "advisor.json"
    try:
        data = json.loads(path.read_text()) if path.is_file() else {"checks": []}
    except (OSError, ValueError):
        data = {"checks": []}
    data["checks"].append(entry)
    try:
        path.write_text(json.dumps(data, indent=1) + "\n")
    except OSError:
        pass


def check_once(directory, concurrent: int = 1):
    """Advise on ``directory`` (a finished DFTB+ run) once per PTBP run; the findings or None."""
    if not wants_check():
        return None
    _STATE["checked"] = True
    try:
        run, findings, report = _advise(directory, concurrent)
        print(report.render_text(run, findings, prefix=PREFIX), flush=True)
        _save(dict(report.to_json(run, findings), when="first DFTB+ run"))
        return findings
    except Exception as err:                       # noqa: BLE001  (never stop PTBP)
        print(f"{PREFIX}not checked ({type(err).__name__}: {err})", flush=True)
        return None


def explain_failure(directory, error: BaseException) -> None:
    """Advise on a failed DFTB+ run; the report is printed and attached to ``error``."""
    if _STATE["run_dir"] is None or directory is None:
        return
    try:
        run, findings, report = _advise(directory)
        text = report.render_text(run, findings, prefix=PREFIX)
        print(text, flush=True)
        if hasattr(error, "add_note"):
            error.add_note(text)
        _save(dict(report.to_json(run, findings), when="failed DFTB+ run"))
    except Exception as err:                       # noqa: BLE001
        print(f"{PREFIX}failed run not checked ({type(err).__name__}: {err})", flush=True)
