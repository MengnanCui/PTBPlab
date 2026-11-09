"""Band loss (A series, A0 section 3): window, shift, matching, gap term, DOVE score, and the
DFTB side on a metal and a molecule."""
import numpy as np
import pytest
from ase.build import bulk, molecule

from pydftb.ase_calc import PyDFTB
from pydftb.bandloss import (BandSettings, DFTBBands, band_loss, compare, dftb_bands,
                             window_weight)
from pydftb.bandref import BandRef, special_kpoints

from .conftest import SKF_HCOCU


def _ref(e, occ=None, kind="ibz", **kw):
    e = np.asarray(e, float)
    nk = e.shape[1]
    return BandRef(name="r", kind=kind, kpts=np.arange(nk)[:, None] * np.array([[0.1, 0, 0]]),
                   kweights=np.ones(nk), energies=e, occupations=occ, **kw)


def test_window_weight_is_one_inside_and_smooth():
    w = window_weight(np.array([-5.0, 0.0, 5.0]), -3.0, 1.0, 0.1)
    assert w[1] == pytest.approx(1.0, abs=1e-4) and w[0] < 1e-8 and w[2] < 1e-8   # 10 widths in
    assert window_weight(np.array([-3.0]), -3.0, 1.0, 0.1)[0] == pytest.approx(0.5)


def test_identical_bands_up_to_a_shift_give_zero_loss():
    e = np.array([[[-4.0, -1.0, 2.0, 3.0], [-3.5, -0.8, 2.2, 3.1]]])
    occ = np.array([[[1, 1, 0, 0]] * 2], float)
    ref = _ref(e, occ)
    out = compare(ref, DFTBBands(e - 1.7, n_valence=2, fermi=0.0))
    assert out["loss"] == pytest.approx(0.0, abs=1e-10)
    assert out["shift"] == pytest.approx(-1.7, abs=1e-10)
    assert out["gap_err"] == pytest.approx(0.0, abs=1e-12)


def test_missing_dft_states_are_free_spurious_dftb_states_are_not():
    e_dft = np.array([[[-2.0, -1.0, 2.0, 2.5]]])        # DFT has two empty states in the window
    occ = np.array([[[1, 1, 0, 0]]], float)
    ref = _ref(e_dft, occ, kind="molecule")
    # DFTB lacks the second empty state: no penalty (one-way matching)
    a = compare(ref, DFTBBands(np.array([[[-2.0, -1.0, 2.0]]]), 2, 0.0))
    assert a["rms"] == pytest.approx(0.0, abs=1e-10)
    # DFTB has a spurious empty state at 0.5 eV inside the window: penalised
    b = compare(ref, DFTBBands(np.array([[[-2.0, -1.0, 0.5, 2.0]]]), 2, 0.0))
    assert b["rms"] > 0.1 and b["gap_err"] == pytest.approx(-1.5)


def test_gap_term_and_metal():
    e = np.array([[[-2.0, -1.0, 2.0]]])
    occ = np.array([[[1, 1, 0]]], float)
    ref = _ref(e, occ, kind="molecule")
    s = BandSettings(gap_weight=1.0)
    out = compare(ref, DFTBBands(np.array([[[-2.0, -1.0, 2.6]]]), 2, 0.0), s)
    assert out["gap_err"] == pytest.approx(0.6)
    assert out["loss"] == pytest.approx(out["rms"] + 0.6)
    occ_m = np.array([[[1, 0.5, 0]]], float)
    met = compare(_ref(e, occ_m, kind="ibz", fermi=-1.0), DFTBBands(e, 2, -1.0), s)
    assert met["metal"] and met["gap_err"] == 0.0 and met["loss"] == pytest.approx(0.0, abs=1e-10)


def test_band_loss_averages_groups_first():
    res = [{"name": "a", "group": "g1", "loss": 1.0}, {"name": "b", "group": "g1", "loss": 3.0},
           {"name": "c", "group": "g2", "loss": 0.0}]
    assert band_loss(res) == pytest.approx(1.0)        # (2 + 0) / 2


def test_dftb_self_comparison_on_a_metal_path():
    atoms = bulk("Cu", "fcc", a=3.61)
    calc = PyDFTB(SKF_HCOCU, max_l={"Cu": 2}, kpts=(4, 4, 4), smearing=("gaussian", 580.0),
                  scc_tol=1e-8)
    atoms.calc = calc
    k, labels, level, segments = special_kpoints(atoms, points_per_segment=7)
    probe = BandRef(name="cu", kind="path", kpts=k, kweights=np.ones(len(k)),
                    energies=np.zeros((1, len(k), 9)), labels=labels, level=level,
                    segments=segments)
    b = dftb_bands(calc, atoms, probe)
    assert b.energies.shape == (1, len(k), 9) and b.n_valence in (5, 6)
    ref = BandRef(name="cu", kind="path", kpts=k, kweights=np.ones(len(k)),
                  energies=b.energies + 3.21, fermi=b.fermi + 3.21, labels=labels, level=level,
                  segments=segments,
                  occupations=(b.energies < b.fermi).astype(float))
    out = compare(ref, b, BandSettings(levels=(1, 2)))
    assert out["metal"]
    assert out["rms"] == pytest.approx(0.0, abs=1e-9)
    assert out["shift"] == pytest.approx(-3.21, abs=1e-9)
    assert out["dove_A"] == pytest.approx(1.0, abs=1e-12)
    assert set(out["segment_mae"]) == {f"{a}-{b_}" for a, b_, _, _ in segments}


def test_dftb_self_comparison_on_a_molecule():
    atoms = molecule("H2O")
    atoms.center(vacuum=6.0)
    calc = PyDFTB(SKF_HCOCU, scc_tol=1e-8)
    atoms.calc = calc
    probe = BandRef(name="h2o", kind="molecule", kpts=[[0, 0, 0]], kweights=[1.0],
                    energies=np.zeros((1, 1, 6)))
    b = dftb_bands(calc, atoms, probe)
    assert b.n_valence == 4
    ref = BandRef(name="h2o", kind="molecule", kpts=[[0, 0, 0]], kweights=[1.0],
                  energies=b.energies - 0.4,
                  occupations=np.array([[[1, 1, 1, 1, 0, 0]]], float))
    out = compare(ref, b)
    assert not out["metal"] and out["gap_err"] == pytest.approx(0.0, abs=1e-10)
    assert out["loss"] == pytest.approx(0.0, abs=1e-9)
