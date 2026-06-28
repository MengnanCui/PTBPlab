"""Auto-detect the optimisation mode from the dataset's shape.

Detection priority:

  1. Dataset extension is `.json` → bandstructure (k-path data).
  2. A `fit.json` exists alongside the dataset (i.e. the user has DFT EOS
     reference data) → energygeometry.
  3. Read the dataset; if it spans more than one chemical formula → reaction.
  4. Otherwise → dataset.

Returns the mode string and a human-readable rationale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple


def detect_mode(dataset_path: str | Path) -> Tuple[str, str]:
    """Return (mode, rationale)."""
    dataset = Path(dataset_path).resolve()

    if dataset.suffix.lower() == '.json':
        return 'bandstructure', f"{dataset.name} is .json (k-path)"

    fit_json = dataset.parent / 'fit.json'
    if fit_json.exists():
        return 'energygeometry', f"found {fit_json.name} alongside (EOS DFT reference)"

    try:
        import ase.io
        atoms_list = ase.io.read(str(dataset), ':')
    except Exception as e:
        # Cannot read — fall back to dataset and let cli/run.py error out
        # with a more specific message.
        return 'dataset', f"could not parse {dataset.name} ({e}); defaulting"

    formulas = {a.get_chemical_formula() for a in atoms_list}
    if len(formulas) > 1:
        return 'reaction', f"{len(atoms_list)} structures span {len(formulas)} formulas"
    return 'dataset', f"{len(atoms_list)} structures, single formula {next(iter(formulas))}"
