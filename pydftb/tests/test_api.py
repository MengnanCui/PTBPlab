"""High-level API, type labels, parameter sets and SKF utilities."""
import gzip
import os
import shutil

import numpy as np
import pytest
from ase.build import molecule

import pydftb
from pydftb.backend import ops
from pydftb.skf import SKFile, max_l_from_skf, write_typed_copies
from pydftb.system import System, element_of_label

from .conftest import SKF_HCOCU, SKF_ORBIT


def _dimer():
    a = molecule("H2O")
    b = molecule("H2O")
    b.translate([0.3, 0.1, 2.9])
    a += b
    a.rattle(0.04, seed=7)
    return a


def test_element_of_label():
    assert [element_of_label(x) for x in ["O", "O2", "Li1", "Li_el", "Li@cat", "Co", "Cl3"]] == \
        ["O", "O", "Li", "Li", "Li", "Co", "Cl"]
    for bad in ["Cab", "Xx"]:
        with pytest.raises(ValueError):
            element_of_label(bad)
    with pytest.raises(ValueError):
        System(["O-1"], np.zeros((1, 3)))


def test_max_l_rule_matches_ase():
    from ase.calculators.dftb import read_max_angular_momentum
    tmp = os.path.join(os.path.dirname(__file__), "_tmp_maxl")
    os.makedirs(tmp, exist_ok=True)
    try:
        for name in ["H-H", "O-O", "Li-Li"]:
            plain = os.path.join(tmp, name + ".skf")
            with gzip.open(os.path.join(SKF_ORBIT, name + ".skf.gz"), "rb") as fi, open(plain, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            assert max_l_from_skf(plain) == read_max_angular_momentum(plain)
        assert max_l_from_skf(os.path.join(SKF_HCOCU, "Cu-Cu.skf.gz")) == 2
    finally:
        shutil.rmtree(tmp)


def test_polynomial_repulsive_columns():
    """SKF line: mass c2..c9 rcut -> polynomial c2..c9 are fields 1..8."""
    f = SKFile.read(os.path.join(SKF_HCOCU, "O-H.skf.gz"))
    assert f.poly_repulsive is None or np.all(f.poly_repulsive.poly == 0.0)
    assert f.repulsive is not None and f.repulsive.knots is not None    # spline present


def test_scc_api_matches_low_level():
    atoms = molecule("CH3OH")
    atoms.rattle(0.05, seed=2)
    params = pydftb.ParamSet(SKF_HCOCU)
    mf = pydftb.SCC(System.from_ase(atoms), params, scc_tol=1e-10)
    e = mf.kernel()
    f = mf.forces()
    low = pydftb.DFTB(params.skset(["C", "H", "O"]), scc_tol=1e-10).calculate(
        atoms.get_chemical_symbols(), atoms.positions / pydftb.constants.BOHR)
    assert abs(e - low.energy) < 1e-10
    assert np.abs(f - low.forces).max() < 1e-8
    assert abs(mf.mulliken.sum()) < 1e-9          # neutral molecule
    assert mf.converged and mf.e_free == pytest.approx(mf.e_tot)


def test_typed_labels_reproduce_plain_parameters(tmp_path):
    """A second oxygen type 'O2' that copies O must give identical results."""
    write_typed_copies(SKF_HCOCU, str(tmp_path), {"O": "O", "O2": "O", "H": "H"},
                       template="{a}-{b}.skf.gz")
    params = pydftb.ParamSet(str(tmp_path))
    assert params.labels_available() == ["H", "O", "O2"]
    a = _dimer()
    plain = pydftb.SCC(System.from_ase(a), params, scc_tol=1e-10)
    labels = a.get_chemical_symbols()
    labels[3] = "O2"
    typed = pydftb.SCC(System.from_ase(a, labels=labels), params, scc_tol=1e-10)
    assert typed.system.elements == a.get_chemical_symbols()
    assert abs(plain.kernel() - typed.kernel()) < 1e-12
    assert np.abs(plain.forces() - typed.forces()).max() < 1e-12


def test_ops_are_functional():
    t = np.zeros(3)
    out = ops.scatter_add(t, np.array([0, 0, 2]), np.array([1.0, 2.0, 3.0]))
    assert np.all(t == 0) and np.allclose(out, [3, 0, 3])
    v = np.arange(12.0).reshape(6, 2)
    assert np.allclose(ops.segment_sum(v, np.array([0, 1, 0, 1, 2, 2]), 3),
                       [[4, 6], [8, 10], [18, 20]])


def test_periodic_api_kgrid():
    from ase.build import bulk
    cu = bulk("Cu", "fcc", a=3.61)
    mf = pydftb.SCC(System.from_ase(cu), pydftb.ParamSet(SKF_HCOCU), kpts=(4, 4, 4),
                    temperature=1000.0)
    e = mf.kernel()
    assert np.isfinite(e) and len(mf.kpts[0]) == 32     # 64 -> 32 by time reversal
