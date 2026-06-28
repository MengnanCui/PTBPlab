"""Shallow validation report tests."""
from __future__ import annotations

from pathlib import Path

import ase
import ase.io
import numpy as np
import pytest


def _h2_dataset(path: Path, n: int = 5, with_forces: bool = True,
                with_energy: bool = True) -> Path:
    """Write n H2 structures to `path`."""
    atoms_list = []
    for i in range(n):
        a = ase.Atoms('H2', positions=[[0, 0, 0], [0.74 + 0.01 * i, 0, 0]],
                      cell=np.eye(3) * 10.0, pbc=False)
        if with_energy:
            a.info['energy'] = -1.0 - 0.05 * i
        if with_forces:
            a.set_array('forces', np.array([[0, 0, 0.01 * (i + 1)],
                                            [0, 0, -0.01 * (i + 1)]]))
        atoms_list.append(a)
    ase.io.write(str(path), atoms_list)
    return path


def _mixed_formula_dataset(path: Path) -> Path:
    """5 H2O + 5 OH structures, energies but no forces."""
    atoms_list = []
    for i in range(5):
        h2o = ase.Atoms('H2O',
                        positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]],
                        cell=np.eye(3) * 10.0, pbc=False)
        h2o.info['energy'] = -76.0 - 0.01 * i
        atoms_list.append(h2o)
    for i in range(5):
        oh = ase.Atoms('OH', positions=[[0, 0, 0], [0.96, 0, 0]],
                       cell=np.eye(3) * 10.0, pbc=False)
        oh.info['energy'] = -75.0 - 0.01 * i
        atoms_list.append(oh)
    ase.io.write(str(path), atoms_list)
    return path


def test_inspect_missing_file_returns_error_report(tmp_path):
    from cli.dataset_inspect import inspect
    r = inspect(tmp_path / "does_not_exist.xyz")
    assert r.readable is False
    assert r.error == "file not found"
    assert r.n_structures == 0


def test_inspect_unreadable_file_returns_error_report(tmp_path):
    from cli.dataset_inspect import inspect
    bad = tmp_path / "garbage.xyz"
    bad.write_text("this is not a valid xyz")
    r = inspect(bad)
    assert r.readable is False
    assert r.error is not None and "ase.io.read" in r.error


def test_inspect_single_formula_with_energies_and_forces(tmp_path):
    from cli.dataset_inspect import inspect
    p = _h2_dataset(tmp_path / "h2.xyz", n=5, with_forces=True, with_energy=True)
    r = inspect(p)
    assert r.readable
    assert r.n_structures == 5
    assert r.formulas == {'H2': 5}
    assert r.energies_present == 5
    assert r.forces_present == 5
    assert r.detected_mode == 'dataset'  # 1 formula, no fit.json sibling
    # No warnings expected
    assert not [w for w in r.warnings if 'forces missing' in w.lower()]


def test_inspect_multi_formula_routes_to_reaction_mode(tmp_path):
    from cli.dataset_inspect import inspect
    p = _mixed_formula_dataset(tmp_path / "mixed.xyz")
    r = inspect(p)
    assert r.readable
    assert r.n_structures == 10
    assert r.formulas == {'H2O': 5, 'HO': 5}
    assert r.detected_mode == 'reaction'


def test_inspect_warns_when_forces_missing_in_dataset_mode(tmp_path):
    from cli.dataset_inspect import inspect
    p = _h2_dataset(tmp_path / "noforces.xyz", n=4, with_forces=False)
    r = inspect(p)
    assert r.forces_present == 0
    assert any('forces' in w.lower() for w in r.warnings), r.warnings


def test_inspect_warns_when_energy_missing(tmp_path):
    from cli.dataset_inspect import inspect
    p = _h2_dataset(tmp_path / "noE.xyz", n=4, with_energy=False)
    r = inspect(p)
    assert r.energies_present == 0
    assert any('energy' in w.lower() or 'no DFT energy' in w for w in r.warnings)


def test_inspect_eos_dataset_via_fit_json_sibling(tmp_path):
    """If fit.json sits alongside the dataset, mode is energygeometry
    even though formulas == 1."""
    from cli.dataset_inspect import inspect
    p = _h2_dataset(tmp_path / "eos.xyz", n=11)
    (tmp_path / "fit.json").write_text("{}")
    r = inspect(p)
    assert r.detected_mode == 'energygeometry'


def test_render_includes_formulas_and_mode(tmp_path):
    from cli.dataset_inspect import inspect
    p = _mixed_formula_dataset(tmp_path / "mixed.xyz")
    r = inspect(p)
    text = r.render()
    assert 'H2O' in text
    assert 'HO' in text
    assert 'reaction' in text
    assert '10 structures' in text
