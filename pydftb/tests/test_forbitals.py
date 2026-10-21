"""f orbitals (milestone M2g): extended SKF format, l = 3 rotations, type copies.

Comparisons with DFTB+ for whole calculations are the gdo*/gd2o2* cases in test_reference.py.
"""
import gzip
import json
import os

import numpy as np
import pytest

from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.params import ParamSet
from pydftb.skf import N_INTEGRALS, SIMPLE_COLUMNS, SK_COLUMN, SKFile, _expand_fortran, \
    max_l_from_skf, write_typed_copies
from pydftb.slako import bond_frames, rotation_blocks, sk_block

from .conftest import DATA, SKF_GDO, SKF_HCOCU

# bent GdO2, off every symmetry plane (Bohr); as validation/cases.py:_gdo2
GDO2 = np.array([[0.0, 0.0, 0.0], [1.85, 0.12, -0.07], [-0.55, 1.75, 0.20]]) / BOHR


def test_f_rotations_are_orthogonal_and_compose():
    rng = np.random.default_rng(1)
    r1 = bond_frames(rng.normal(size=(20, 3)))
    r2 = bond_frames(rng.normal(size=(20, 3)))
    d1 = rotation_blocks(r1, 3)[3]
    d2 = rotation_blocks(r2, 3)[3]
    assert np.abs(d1 @ d1.transpose(0, 2, 1) - np.eye(7)).max() < 1e-14
    assert np.abs(rotation_blocks(r1 @ r2, 3)[3] - d1 @ d2).max() < 1e-14


def test_f_blocks_match_dftbplus_formulas():
    """sf, pf, df and ff blocks against DFTB+'s sk.F90 expressions evaluated at three
    directions (tests/data/sk_f_blocks.json)."""
    ref = json.load(open(os.path.join(DATA, "sk_f_blocks.json")))
    vec = np.array(ref["directions"]) * 2.7
    dm = rotation_blocks(bond_frames(vec), 3)
    for pair, d in ref["pairs"].items():
        l1 = "spdf".index(pair[0])
        integ = np.zeros((len(vec), 2 * N_INTEGRALS))
        for m, value in enumerate(d["integrals"]):
            integ[:, SK_COLUMN[(l1, 3, m)]] = value
        blk = sk_block(vec, integ, integ, [l1], [3], 0, dm)
        assert np.abs(blk - np.array(d["blocks"])).max() < 1e-13, pair


def test_extended_and_simple_formats():
    gd = SKFile.read(os.path.join(SKF_GDO, "Gd-Gd.skf.gz"))
    assert gd.table.shape == (718, 40) and np.all(gd.columns == np.arange(40))
    # on-site line "Ef Ed Ep Es SPE Uf Ud Up Us ff fd fp fs", stored as s p d f
    assert gd.onsite == pytest.approx([-0.151522, -0.049059, -0.093254, -0.320071])
    assert gd.hubbard == pytest.approx([0.207404, 0.138484, 0.309801, 1.067014])
    assert np.all(gd.occupation == [2, 0, 1, 7])
    # at a grid node the interpolation returns the table row: the 20 H then 20 S of the file
    with gzip.open(os.path.join(SKF_GDO, "Gd-Gd.skf.gz"), "rt") as fh:
        row = np.array(_expand_fortran(fh.read().splitlines()[3 + 200]))      # '@', dr, on-site, mass
    assert np.allclose(gd.evaluate(np.array([200 * gd.dr]))[0], row, rtol=1e-12, atol=0)
    # a simple-format file: the spd integrals land in DFTB+'s places, the f ones are zero
    oo = SKFile.read(os.path.join(SKF_GDO, "O-O.skf.gz"))
    full = oo.evaluate(np.array([2.3, 4.1]))
    assert full.shape == (2, 40) and oo.table.shape[1] == 20
    f_cols = [c for c in range(N_INTEGRALS) if c not in SIMPLE_COLUMNS]
    assert np.all(full[:, f_cols] == 0) and np.all(full[:, np.array(f_cols) + N_INTEGRALS] == 0)
    compact = oo.evaluate(np.array([2.3, 4.1]), full=False)
    assert np.all(full[:, oo.columns] == compact)
    assert np.all(oo.colmap[SIMPLE_COLUMNS] == np.arange(10)) and oo.colmap[0] == -1
    assert oo.onsite[3] == 0 and oo.occupation == pytest.approx([2, 4, 0, 0])


def test_max_l_of_extended_files():
    assert max_l_from_skf(os.path.join(SKF_GDO, "Gd-Gd.skf.gz")) == 3
    assert max_l_from_skf(os.path.join(SKF_GDO, "O-O.skf.gz")) == 1
    assert ParamSet(SKF_GDO).skset(["Gd", "O"]).norb("Gd") == 16


def test_spd_files_unchanged_by_the_new_layout():
    """The simple-format integrals go through the 40-column layout unchanged."""
    f = SKFile.read(os.path.join(SKF_HCOCU, "Cu-O.skf.gz"))
    r = np.array([2.1, 3.7, 6.2])
    with gzip.open(os.path.join(SKF_HCOCU, "Cu-O.skf.gz"), "rt") as fh:
        rows = fh.read().splitlines()
    k = 150
    row = np.array(_expand_fortran(rows[1 + k])[:20])                       # dr line, mass line
    assert np.allclose(f.evaluate(np.array([k * f.dr]))[0][f.columns], row, rtol=1e-12, atol=0)
    assert np.all(f.evaluate(r)[:, f.columns] == f.evaluate(r, full=False))


def test_typed_copies_of_extended_files(tmp_path):
    """A second Gd type copied from an extended-format file gives identical results
    (the copy Gd-Gd2 must lose the on-site line, which is line 3 after '@')."""
    write_typed_copies(SKF_GDO, str(tmp_path), {"Gd": "Gd", "Gd2": "Gd", "O": "O"},
                       template="{a}-{b}.skf.gz")
    with open(tmp_path / "Gd-Gd2.skf") as fh:
        head = fh.read().splitlines()[:3]
    assert head[0].startswith("@") and head[1].startswith("0.02") and "*0.0" in head[2]
    pos = np.array([[0, 0, 0], [3.4, 0.1, 0.2], [1.7, 1.3, -0.1], [1.6, -1.2, 0.15]]) / BOHR
    plain = DFTB(ParamSet(SKF_GDO).skset(["Gd", "O"]), temperature=1000.0, scc_tol=1e-11)
    typed = DFTB(ParamSet(str(tmp_path)).skset(["Gd", "Gd2", "O"]), temperature=1000.0,
                 scc_tol=1e-11)
    a = plain.calculate(["Gd", "Gd", "O", "O"], pos)
    b = typed.calculate(["Gd", "Gd2", "O", "O"], pos)
    assert abs(a.free_energy - b.free_energy) < 1e-12
    assert np.abs(a.forces - b.forces).max() < 1e-12


@pytest.mark.parametrize("opts", [{}, {"shell_resolved": True}])
def test_f_forces_match_energy_finite_differences(opts):
    labels = ["Gd", "O", "O"]
    sk = ParamSet(SKF_GDO, shell_resolved=opts.get("shell_resolved", False)).skset(labels)
    calc = DFTB(sk, temperature=1000.0, scc_tol=1e-12, max_scc=500)
    res = calc.calculate(labels, GDO2)
    h = 1e-4
    for a, k in [(0, 2), (1, 0), (2, 1)]:
        p, m = GDO2.copy(), GDO2.copy()
        p[a, k] += h
        m[a, k] -= h
        ep = calc.calculate(labels, p, forces=False).free_energy
        em = calc.calculate(labels, m, forces=False).free_energy
        assert abs(-(ep - em) / (2 * h) - res.forces[a, k]) < 1e-7
