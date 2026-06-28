"""Auto mode detection tests."""
from __future__ import annotations

from pathlib import Path

import ase
import ase.io
import numpy as np
import pytest


def _write_h2_dataset(path: Path, n: int = 3) -> Path:
    """Write `n` H2 structures with a fake DFT energy in info."""
    atoms_list = []
    for i in range(n):
        a = ase.Atoms('H2', positions=[[0, 0, 0], [0.74 + 0.01 * i, 0, 0]],
                      cell=np.eye(3) * 10.0, pbc=False)
        a.info['energy'] = -1.0 - 0.05 * i
        atoms_list.append(a)
    ase.io.write(str(path), atoms_list)
    return path


def _write_mixed_formula_dataset(path: Path) -> Path:
    """5 H2O + 5 OH structures."""
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


def test_detect_bandstructure_for_json(tmp_path):
    from cli.auto_mode import detect_mode
    p = tmp_path / "kpath.json"
    p.write_text("{}")
    mode, _ = detect_mode(p)
    assert mode == 'bandstructure'


def test_detect_energygeometry_when_fit_json_present(tmp_path):
    from cli.auto_mode import detect_mode
    ds = _write_h2_dataset(tmp_path / "dft.xyz", n=11)
    (tmp_path / "fit.json").write_text("{}")
    mode, rationale = detect_mode(ds)
    assert mode == 'energygeometry'
    assert 'fit.json' in rationale


def test_detect_dataset_for_single_formula(tmp_path):
    from cli.auto_mode import detect_mode
    ds = _write_h2_dataset(tmp_path / "dft.xyz", n=5)
    mode, rationale = detect_mode(ds)
    assert mode == 'dataset'
    assert 'single formula' in rationale


def test_detect_reaction_for_multi_formula(tmp_path):
    from cli.auto_mode import detect_mode
    ds = _write_mixed_formula_dataset(tmp_path / "mixed.xyz")
    mode, rationale = detect_mode(ds)
    assert mode == 'reaction'
    assert '2 formulas' in rationale


def test_explicit_mode_skips_detection(tmp_path, monkeypatch):
    """If user passes --mode explicitly, auto_mode.detect_mode is not called."""
    ds = _write_h2_dataset(tmp_path / "dft.xyz", n=5)
    output = tmp_path / "run"
    monkeypatch.chdir(tmp_path)

    called = {'detect': False}

    def fake_detect(path):
        called['detect'] = True
        return ('reaction', 'fake')

    captured = {}

    def fake_run_path(path, run_name=None):
        import sys
        captured['argv'] = list(sys.argv)

    monkeypatch.setattr('cli.auto_mode.detect_mode', fake_detect)
    monkeypatch.setattr('cli.cmd_optimize.runpy.run_path', fake_run_path)

    from cli.main import dispatch
    rc = dispatch(['optimize', str(ds),
                   '--mode', 'energygeometry',
                   '--output', str(output)])

    assert rc == 0
    assert not called['detect'], "explicit --mode should bypass auto-detection"
    assert captured['argv'][captured['argv'].index('--optimization_option') + 1] \
        == 'energygeometry'


def test_auto_mode_routes_to_correct_legacy_flag(tmp_path, monkeypatch):
    """Default --mode auto on a multi-formula dataset → reaction passed to legacy."""
    ds = _write_mixed_formula_dataset(tmp_path / "mixed.xyz")
    output = tmp_path / "run"
    monkeypatch.chdir(tmp_path)

    captured = {}

    def fake_run_path(path, run_name=None):
        import sys
        captured['argv'] = list(sys.argv)

    monkeypatch.setattr('cli.cmd_optimize.runpy.run_path', fake_run_path)

    from cli.main import dispatch
    rc = dispatch(['optimize', str(ds), '--output', str(output)])

    assert rc == 0
    argv = captured['argv']
    assert argv[argv.index('--optimization_option') + 1] == 'reaction'
