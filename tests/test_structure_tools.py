"""Tests for gui.core.structure_tools (ASE-only, runs anywhere)."""
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from gui.core import structure_tools as st


def _cu():
    return bulk("Cu", "fcc", a=3.6, cubic=True)


def test_scale_volume_ratio_and_fractional_coords():
    atoms = _cu()
    before_frac = atoms.get_scaled_positions()
    scaled = st.scale_volume(atoms, 0.8)
    assert scaled.get_volume() / atoms.get_volume() == pytest.approx(0.8, rel=1e-6)
    # fractional coordinates are preserved (self-similar rescale)
    assert np.allclose(before_frac, scaled.get_scaled_positions())
    # original untouched
    assert atoms.get_volume() != pytest.approx(scaled.get_volume())


def test_scale_volume_rejects_nonperiodic():
    mol = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    with pytest.raises(ValueError):
        st.scale_volume(mol, 0.9)


def test_scale_volume_rejects_bad_scale():
    with pytest.raises(ValueError):
        st.scale_volume(_cu(), 0.0)


def test_make_eos_series_count_center_and_monotonic():
    atoms = _cu()
    series = st.make_eos_series(atoms, 90, 110, 11)
    assert len(series) == 11
    scales = [a.info["eos_scale"] for a in series]
    # ascending
    assert scales == sorted(scales)
    # center is equilibrium
    assert series[5].info["eos_scale"] == pytest.approx(1.0)
    vols = [a.get_volume() for a in series]
    assert vols == sorted(vols)
    # every structure carries a name
    assert all("structure_name" in a.info for a in series)


def test_make_eos_series_single_point():
    series = st.make_eos_series(_cu(), 90, 110, 1)
    assert len(series) == 1
    assert series[0].info["eos_scale"] == pytest.approx(1.0)


def test_perturb_positions_deterministic():
    atoms = _cu()
    a = st.perturb_positions(atoms, 0.05, seed=1)
    b = st.perturb_positions(atoms, 0.05, seed=1)
    c = st.perturb_positions(atoms, 0.05, seed=2)
    assert np.allclose(a.positions, b.positions)
    assert not np.allclose(a.positions, c.positions)
    # original untouched
    assert np.allclose(atoms.positions, _cu().positions)


def test_combine_and_write_contiguous_roundtrip(tmp_path):
    import ase.io

    phase_a = st.make_eos_series(_cu(), 95, 105, 3, name="A")
    phase_b = st.make_eos_series(_cu(), 95, 105, 3, name="B")
    out = st.combine_and_write([phase_a, phase_b], tmp_path / "eos.xyz")
    back = ase.io.read(str(out), ":")
    assert len(back) == 6
    names = [a.info.get("structure_name") for a in back]
    # phases stay contiguous (A block then B block) so eos_points slicing works
    assert names == ["A", "A", "A", "B", "B", "B"]


def test_combine_and_write_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        st.combine_and_write([], tmp_path / "x.xyz")


def test_summarize():
    s = st.summarize(st.make_eos_series(_cu(), 90, 110, 5))
    assert s["n_structures"] == 5
    assert s["n_periodic"] == 5
    assert s["n_formulas"] == 1
