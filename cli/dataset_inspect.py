"""Shallow + actionable dataset inspection.

Used by:
- `ptbp setup` (TUI) — drives the validation step before the user picks a
  mode / optimizer.
- `ptbp optimize` (CLI) — replaces the previous verbose Boltzmann-factor
  dump with a structured pre-flight summary so the user sees at startup
  exactly what the optimiser is going to chew on.

By design this is fast (sub-second on a few hundred structures) and
read-only: just `ase.io.read` once, walk over the atoms list, count
features. No DFTB+ / hotcent / fit_atomic_E0s involved.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DatasetReport:
    """Outcome of a shallow validation pass over a DFT reference dataset."""

    path: Path
    readable: bool = False
    n_structures: int = 0
    formulas: Dict[str, int] = field(default_factory=dict)
    energies_present: int = 0
    forces_present: int = 0
    detected_mode: str = ""
    detection_rationale: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def n_formulas(self) -> int:
        return len(self.formulas)

    def render(self) -> str:
        """Render as a multi-line plain-text block. The TUI's Markdown
        widget consumes a slightly different formatting (built in
        cli/tui), so this method is the canonical CLI string only."""
        if self.error:
            return f"Dataset: {self.path}\n  ✗ Read failed: {self.error}\n"

        ok = "✓"
        bad = "✗"
        rows = [f"Dataset: {self.path}"]
        rows.append(f"  {ok} Read OK — {self.n_structures} structures")

        nE = self.energies_present
        if nE == self.n_structures:
            rows.append(f"  {ok} Energies present on all {nE}")
        elif nE > 0:
            rows.append(f"  {bad} Energies present on {nE}/{self.n_structures}")
        else:
            rows.append(f"  {bad} Energies missing on all {self.n_structures}")

        nF = self.forces_present
        if nF == self.n_structures:
            rows.append(f"  {ok} Forces present on all {nF}")
        elif nF > 0:
            rows.append(f"  {bad} Forces present on {nF}/{self.n_structures}")
        else:
            rows.append(f"  {bad} Forces missing on all {self.n_structures}")

        if self.formulas:
            top = ", ".join(f"{f} ({n})" for f, n in
                            sorted(self.formulas.items(), key=lambda kv: -kv[1])[:6])
            extra = "" if self.n_formulas <= 6 else f" + {self.n_formulas - 6} more"
            rows.append(f"  Formulas ({self.n_formulas} unique): {top}{extra}")

        if self.detected_mode:
            rows.append(f"  → Auto-detected mode: {self.detected_mode} "
                        f"({self.detection_rationale})")

        for w in self.warnings:
            rows.append(f"  ! {w}")
        return "\n".join(rows)


def inspect(path: str | Path) -> DatasetReport:
    """Run the shallow validation pass over `path`. Always returns a
    DatasetReport — read failures populate the `error` field rather than
    raising, so a TUI screen can render both success and failure
    uniformly."""
    p = Path(path).resolve()
    report = DatasetReport(path=p)

    if not p.exists():
        report.error = f"file not found"
        return report

    try:
        import ase.io
        atoms_list = ase.io.read(str(p), ':')
    except Exception as e:
        report.error = f"ase.io.read raised {type(e).__name__}: {e}"
        return report

    report.readable = True
    report.n_structures = len(atoms_list)

    # Formulas, energies, forces. Forces can live on calc.results (modern
    # ASE) or on atoms.arrays (legacy mirror). Either counts.
    formulas: Counter[str] = Counter()
    for atoms in atoms_list:
        formulas[atoms.get_chemical_formula()] += 1

        has_energy = ('energy' in atoms.info or
                      (atoms.calc is not None
                       and 'energy' in getattr(atoms.calc, 'results', {})))
        if has_energy:
            report.energies_present += 1

        has_forces = (
            'forces' in atoms.arrays
            or (atoms.calc is not None
                and 'forces' in getattr(atoms.calc, 'results', {}))
        )
        if has_forces:
            report.forces_present += 1

    report.formulas = dict(formulas)

    # Re-use cli.auto_mode for the mode call so we never drift out of sync.
    try:
        from cli.auto_mode import detect_mode
        mode, rationale = detect_mode(p)
        report.detected_mode = mode
        report.detection_rationale = rationale
    except Exception as e:
        report.warnings.append(
            f"could not auto-detect mode ({type(e).__name__}: {e}); fall back to dataset")
        report.detected_mode = 'dataset'
        report.detection_rationale = 'fallback'

    if report.energies_present < report.n_structures:
        report.warnings.append(
            f"{report.n_structures - report.energies_present} structure(s) "
            "have no DFT energy — they will be skipped or cause errors at "
            "loss time. Either fix the dataset or pass --E0s explicitly.")

    if report.forces_present == 0 and report.detected_mode in ('dataset', 'reaction'):
        report.warnings.append(
            "no forces in any structure — μ_F (force loss) will be 0; "
            "loss is energy-only.")

    return report
