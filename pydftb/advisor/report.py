"""Text and JSON output of the advisor."""
from __future__ import annotations

import json
import textwrap

from .inputs import RunFiles
from .rules import SEVERITIES


def summary(findings) -> dict:
    return {s: sum(1 for f in findings if f.severity == s) for s in SEVERITIES}


def describe(run: RunFiles) -> str:
    parts = [f"input {run.dir / 'dftb_in.hsd'}", f"ParserVersion {run.parser_version}",
             "pin " + ("yes" if run.pin is not None else "no"),
             "screen output " + (run.stdout.path.rsplit('/', 1)[-1] if run.stdout is not None else "none")]
    if run.stdout is not None and run.stdout.version:
        parts.append(f"DFTB+ {run.stdout.version}")
    if run.geometry is not None:
        g = run.geometry
        parts.append(f"{g.natoms} atom{'s' * (g.natoms != 1)} ({' '.join(g.species)}), "
                     + ("periodic" if g.periodic else "cluster"))
    return "; ".join(parts)


def render_text(run: RunFiles, findings, prefix: str = "") -> str:
    lines = [f"{prefix}DFTB+ advisor: {describe(run)}"]
    for note in run.notes:
        lines.append(f"{prefix}  note: {note}")
    for f in findings:
        head = f"{prefix}[{f.rule}] {f.severity:<7} {f.title}" + (f"  ({f.where})" if f.where else "")
        lines.append(head)
        wrap = textwrap.TextWrapper(width=96, initial_indent=f"{prefix}    ",
                                    subsequent_indent=f"{prefix}    ")
        lines.extend(wrap.wrap(f.message))
        if f.advice:
            lines.extend(wrap.wrap(f"advice: {f.advice}"))
        if f.expected:
            lines.extend(wrap.wrap(f"expected here: {f.expected}"))
    n = summary(findings)
    lines.append(f"{prefix}{n['error']} error(s), {n['warning']} warning(s), {n['info']} info")
    return "\n".join(lines)


def to_json(run: RunFiles, findings) -> dict:
    return {"run": describe(run), "dir": str(run.dir), "notes": list(run.notes),
            "summary": summary(findings), "findings": [f.to_json() for f in findings]}


def render_json(run: RunFiles, findings) -> str:
    return json.dumps(to_json(run, findings), indent=2)
