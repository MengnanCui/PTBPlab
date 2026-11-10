"""ASE-only structure manipulation for the GUI's Structures view.

The PTBP optimiser reads *pre-expanded* datasets (e.g. an EOS scan of N
volumes per phase) and never generates them itself. This module fills that
gap so the GUI can turn a single relaxed structure into a training set:

- `scale_volume`      — isotropically rescale the cell (材料体积调整)
- `make_eos_series`   — volume scan → list of structures (生成 EOS 训练结构)
- `perturb_positions` — rattle atomic positions (原子位置调整)
- `combine_and_write` — write phases contiguously so cli/run.py's
                        `dft_input[floor(eos_points/2)::eos_points]` centring
                        still selects the equilibrium structure of each phase.
- `summarize`         — quick counts/formulas for the UI.

No DFTB+/hotcent needed — pure ASE + numpy, so it runs and tests anywhere.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def _require_cell(atoms) -> float:
    """Return the cell volume, raising a clear error for non-periodic input."""
    vol = float(abs(atoms.get_volume())) if atoms.cell.rank == 3 else 0.0
    if vol <= 0.0:
        raise ValueError(
            "structure has no 3D cell (volume is zero) — volume scaling needs a "
            "periodic cell. Load a bulk structure, not an isolated molecule."
        )
    return vol


def scale_volume(atoms, scale: float):
    """Return a copy of `atoms` whose cell volume is `scale`× the original.

    `scale` is a *volume* ratio (0.9 → 90 % of the original volume). The
    linear cell factor is `scale ** (1/3)`; `scale_atoms=True` keeps the
    fractional coordinates fixed so the structure stays self-similar.
    """
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    _require_cell(atoms)
    out = atoms.copy()
    factor = scale ** (1.0 / 3.0)
    out.set_cell(atoms.get_cell() * factor, scale_atoms=True)
    return out


def make_eos_series(
    atoms,
    vmin_pct: float = 90.0,
    vmax_pct: float = 110.0,
    n_points: int = 11,
    name: Optional[str] = None,
) -> List:
    """Generate an equation-of-state volume scan from a single structure.

    Returns `n_points` copies spanning `vmin_pct`..`vmax_pct` percent of the
    input volume, ascending, so index `n_points // 2` is the (near-)
    equilibrium structure — matching the centring convention in cli/run.py.

    Each returned structure carries `info['structure_name']` and
    `info['eos_scale']` (the volume ratio) for downstream bookkeeping.
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    if vmin_pct > vmax_pct:
        raise ValueError(f"vmin_pct ({vmin_pct}) must be <= vmax_pct ({vmax_pct})")
    _require_cell(atoms)

    label = name or atoms.info.get("structure_name") or atoms.get_chemical_formula()
    if n_points == 1:
        scales = np.array([1.0])
    else:
        scales = np.linspace(vmin_pct / 100.0, vmax_pct / 100.0, n_points)

    series = []
    for s in scales:
        scaled = scale_volume(atoms, float(s))
        scaled.info["structure_name"] = label
        scaled.info["eos_scale"] = float(s)
        series.append(scaled)
    return series


def perturb_positions(atoms, sigma: float = 0.02, seed: Optional[int] = None):
    """Return a copy with Gaussian noise (stddev `sigma` Å) on every position.

    Deterministic for a given `seed` (uses ase's `rattle`, which seeds numpy
    internally), so the same seed reproduces the same perturbation.
    """
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    out = atoms.copy()
    out.rattle(stdev=sigma, seed=seed)
    return out


def combine_and_write(phases: Sequence[Sequence], path: str | Path) -> Path:
    """Write one or more phases to a single extxyz dataset.

    `phases` is a sequence of series (each series a list of Atoms, e.g. from
    `make_eos_series`). Structures are written **phase-by-phase, contiguous**,
    which is exactly what `energygeometry` mode expects: with `eos_points = N`
    per phase, cli/run.py slices `dft_input[floor(N/2)::N]` to grab each
    phase's equilibrium structure.
    """
    import ase.io

    flat = [atoms for series in phases for atoms in series]
    if not flat:
        raise ValueError("nothing to write — `phases` is empty")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(str(out_path), flat, format="extxyz")
    return out_path


def summarize(atoms_list: Sequence) -> Dict:
    """Small structured summary of a structure list for the UI header."""
    from collections import Counter

    formulas: Counter = Counter()
    n_periodic = 0
    for atoms in atoms_list:
        formulas[atoms.get_chemical_formula()] += 1
        if atoms.cell.rank == 3 and abs(atoms.get_volume()) > 0.0:
            n_periodic += 1
    return {
        "n_structures": len(atoms_list),
        "n_formulas": len(formulas),
        "formulas": dict(formulas),
        "n_periodic": n_periodic,
    }
