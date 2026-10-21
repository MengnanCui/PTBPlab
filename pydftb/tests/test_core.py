"""Unit tests for the core engine (SK tables, rotations, SCC, forces, solvers).

Self-contained: uses the compressed copy of test/orbit in pydftb/tests/data.
Comparisons with DFTB+ are in test_reference.py (frozen JSON, no dftb+ needed).
"""
import os

import numpy as np
import pytest

from pydftb.calculator import DFTB
from pydftb.kpoints import monkhorst_pack, reduce_time_reversal
from pydftb.skf import SIMPLE_COLUMNS, SKFile, SKSet
from pydftb.slako import bond_frames, rotation_blocks, sk_block

SKDIR = os.path.join(os.path.dirname(__file__), "data", "skf", "orbit")
WATER = np.array([[0.0, 0.0, 0.2], [1.44, 0.0, -0.85], [-1.32, 0.19, -0.94]])  # Bohr


@pytest.fixture(scope="module")
def sk():
    return SKSet(SKDIR, ["O", "H"], {"O": 1, "H": 0}, template="{a}-{b}.skf.gz")


def test_skf_interpolation_derivative():
    f = SKFile.read(os.path.join(SKDIR, "O-O.skf.gz"))
    # off-node points: the piecewise stencil switches exactly at grid nodes
    r = np.array([1.3037, 2.7113, 5.0071, f.ngrid * f.dr + 0.3])
    v, dv = f.evaluate(r, deriv=True)
    h = 1e-6
    fd = (f.evaluate(r + h) - f.evaluate(r - h)) / (2 * h)
    assert np.allclose(dv, fd, atol=1e-7)
    assert np.all(f.evaluate(np.array([f.cutoff + 0.1])) == 0.0)


def test_skf_table_rows_like_dftbplus():
    """Header ``0.02, 719`` with 719 rows: DFTB+ (type/oldskdata.F90) uses only 718,
    which fixes where the table ends and its 1-Bohr tail starts."""
    f = SKFile.read(os.path.join(SKDIR, "Li-Li.skf.gz"))
    assert f.table.shape[0] == 718
    assert f.cutoff == pytest.approx(718 * 0.02 + 1.0)


def test_slater_koster_table_entries():
    """Rotation-based blocks reproduce textbook Slater-Koster expressions."""
    rng = np.random.default_rng(0)
    vec = rng.normal(size=(5, 3))
    l, m, n = (vec / np.linalg.norm(vec, axis=1)[:, None]).T
    integ = np.zeros((5, 40))
    integ[:, SIMPLE_COLUMNS] = rng.normal(size=10)    # spd H integrals, same for every pair
    dm = rotation_blocks(bond_frames(vec), 2)
    blk = sk_block(vec, integ, integ, [0, 1, 2], [0, 1, 2], 0, dm)
    hdd0, hdd1, hdd2, hpd0, hpd1, hpp0, hpp1, hsd0, hsp0, hss0 = integ[0, SIMPLE_COLUMNS]
    # s-p_x ; p_x-p_x ; p_x-p_y ; s-d_xy ; p_x-d_xy ; d_xy-d_xy (orbital order s|y z x|xy yz z2 xz x2-y2)
    assert np.allclose(blk[:, 0, 3], l * hsp0)
    assert np.allclose(blk[:, 3, 3], l * l * hpp0 + (1 - l * l) * hpp1)
    assert np.allclose(blk[:, 3, 1], l * m * (hpp0 - hpp1))
    assert np.allclose(blk[:, 0, 4], np.sqrt(3) * l * m * hsd0)
    assert np.allclose(blk[:, 3, 4], np.sqrt(3) * l * l * m * hpd0 + m * (1 - 2 * l * l) * hpd1)
    assert np.allclose(blk[:, 4, 4], 3 * l * l * m * m * hdd0 + (l * l + m * m - 4 * l * l * m * m) * hdd1
                       + (n * n + l * l * m * m) * hdd2)
    z2 = n * n - 0.5 * (l * l + m * m)
    assert np.allclose(blk[:, 6, 6], z2 ** 2 * hdd0 + 3 * n * n * (l * l + m * m) * hdd1
                       + 0.75 * (l * l + m * m) ** 2 * hdd2)


def test_scc_forces_match_finite_differences(sk):
    calc = DFTB(sk, scc=True, scc_tol=1e-11)
    sym = ["O", "H", "H"]
    res = calc.calculate(sym, WATER)
    assert res.converged
    assert abs(res.charges.sum() - 8.0) < 1e-8
    h = 1e-4
    for a, k in [(0, 2), (1, 0), (2, 1)]:
        p, m = WATER.copy(), WATER.copy()
        p[a, k] += h
        m[a, k] -= h
        fd = -(calc.calculate(sym, p, forces=False).energy - calc.calculate(sym, m, forces=False).energy) / (2 * h)
        assert abs(fd - res.forces[a, k]) < 1e-7


def test_periodic_kpoints_and_solvers_agree(sk):
    cell = np.eye(3) * 9.0
    sym = ["O", "H", "H"]
    k = reduce_time_reversal(*monkhorst_pack((2, 2, 2)))
    assert len(k[0]) == 4 and np.isclose(k[1].sum(), 1.0)
    ref = DFTB(sk, kpts=k, temperature=300, scc_tol=1e-10, solver="eigh").calculate(sym, WATER, cell)
    bat = DFTB(sk, kpts=k, temperature=300, scc_tol=1e-10, solver="batched").calculate(sym, WATER, cell)
    assert abs(ref.free_energy - bat.free_energy) < 1e-9
    assert np.abs(ref.forces - bat.forces).max() < 1e-8
    # translation invariance of the total force
    assert np.abs(ref.forces.sum(0)).max() < 1e-8


def test_sp2_matches_diagonalisation(sk):
    cell = np.eye(3) * 9.0
    sym = ["O", "H", "H"]
    a = DFTB(sk, scc_tol=1e-10, solver="eigh").calculate(sym, WATER, cell)
    b = DFTB(sk, scc_tol=1e-10, solver="sp2").calculate(sym, WATER, cell)
    assert abs(a.energy - b.energy) < 1e-9
    assert np.abs(a.forces - b.forces).max() < 1e-7
