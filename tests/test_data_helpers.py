"""Pure-Python helpers in utils.data (no DFTB+ / hotcent involved)."""
import json
from pathlib import Path

import ase
import numpy as np
import pytest

import utils.data as data


@pytest.fixture
def synthetic_HC_dataset():
    """Two structures: H2 + CH4, with energies that are exactly
    `Σ_Z N_Z · E0[Z]` for known E0[H]=-1.0 and E0[C]=-37.0. The lstsq fit
    should recover those E0s to numerical precision."""
    E0_true = {1: -1.0, 6: -37.0}

    h2 = ase.Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]],
                   cell=np.eye(3) * 10.0, pbc=False)
    h2.info['energy'] = 2 * E0_true[1]

    ch4 = ase.Atoms('CH4',
                    positions=[[0, 0, 0], [0.6, 0.6, 0.6],
                               [-0.6, -0.6, 0.6], [0.6, -0.6, -0.6],
                               [-0.6, 0.6, -0.6]],
                    cell=np.eye(3) * 10.0, pbc=False)
    ch4.info['energy'] = 1 * E0_true[6] + 4 * E0_true[1]

    return [h2, ch4], E0_true


def test_fit_atomic_E0s_recovers_truth(synthetic_HC_dataset):
    atoms_list, E0_true = synthetic_HC_dataset
    fitted = data.fit_atomic_E0s(atoms_list)
    assert fitted.keys() == E0_true.keys()
    for z in E0_true:
        assert fitted[z] == pytest.approx(E0_true[z], abs=1e-9)


def test_fit_atomic_E0s_warns_on_rank_deficient(capsys):
    """A dataset with only one element (one column) and a single composition
    is full-rank for one unknown — but adding two structures of the SAME
    composition keeps the design matrix at rank 1 with one unknown, which is
    still full rank. To trigger rank-deficiency we need >= 2 unknowns and
    < 2 distinct compositions; build that."""
    h2 = ase.Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]],
                   cell=np.eye(3) * 10.0, pbc=False)
    h2.info['energy'] = -2.0
    ch_h = ase.Atoms('CH', positions=[[0, 0, 0], [1, 0, 0]],
                     cell=np.eye(3) * 10.0, pbc=False)
    ch_h.info['energy'] = -38.0
    # Both rows have rank-2 design matrix (2 unknowns, 2 distinct rows) — OK,
    # this won't actually trigger the warning. Use a degenerate case instead:
    # two structures with proportional compositions (e.g. C2H4 vs C4H8) — they
    # span the same direction in (n_C, n_H) space, dropping the rank to 1.
    c2h4 = ase.Atoms('C2H4', positions=[[i * 0.6, 0, 0] for i in range(6)],
                     cell=np.eye(3) * 10.0, pbc=False)
    c2h4.info['energy'] = 2 * -37.0 + 4 * -1.0
    c4h8 = ase.Atoms('C4H8', positions=[[i * 0.6, 0, 0] for i in range(12)],
                     cell=np.eye(3) * 10.0, pbc=False)
    c4h8.info['energy'] = 4 * -37.0 + 8 * -1.0

    data.fit_atomic_E0s([c2h4, c4h8])
    captured = capsys.readouterr()
    assert "rank-deficient" in captured.out


def test_E0s_cache_roundtrip(tmp_path):
    src = {1: -0.5, 6: -37.8, 8: -75.1}
    dst = tmp_path / "E0s_dft.json"
    data.save_E0s_cache(src, str(dst))
    assert dst.exists()
    loaded = data.load_E0s_cache(str(dst))
    assert loaded == src


def test_E0s_cache_load_missing_returns_none(tmp_path):
    assert data.load_E0s_cache(str(tmp_path / "does_not_exist.json")) is None


def test_E0s_cache_load_corrupt_returns_none(tmp_path):
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("not valid json {{{")
    assert data.load_E0s_cache(str(corrupt)) is None
