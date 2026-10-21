"""physrep.evaluate_pbc (vectorised, M3b) against the original double loop: bit for bit."""
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

import tools.physrep as P

ROOT = Path(__file__).resolve().parents[1]


def _bits(x):
    return type(x), np.asarray(x).dtype.str, np.asarray(x).tobytes()


def _same(a, b):
    (e1, f1), (e2, f2) = a, b
    return _bits(e1) == _bits(e2) and _bits(f1) == _bits(f2)


@pytest.fixture(scope="module")
def cu_dataset():
    return ase.io.read(ROOT / "dft" / "dft.xyz", ":")


@pytest.mark.parametrize("sigma", [0.1, 0.24, 0.33058594, 0.38, 0.52, 0.8])
def test_cu_dataset_bitwise(cu_dataset, sigma):
    pr = P.physrep(elements=[29], rmin=0.5, rmax=20.0, npoints=700, scale_cov=[sigma], k_xc=[0.0])
    for atoms in cu_dataset:
        ref = pr.evaluate_pbc_reference(atoms)
        assert _same(pr.evaluate_pbc(atoms), ref)
        assert _same(pr.evaluate_pbc(atoms), ref)            # from the pair cache


def test_random_triclinic_multi_element_cells():
    rng = np.random.default_rng(3)
    for _ in range(30):
        cell = np.eye(3) * rng.uniform(2.6, 4.5) + rng.uniform(-0.8, 0.8, (3, 3))
        nat = int(rng.integers(1, 5))
        atoms = Atoms(list(rng.choice(["Cu", "O", "H"], size=nat)),
                      scaled_positions=rng.random((nat, 3)), cell=cell, pbc=True)
        els = sorted(set(atoms.numbers.tolist()))
        pr = P.physrep(elements=els, rmin=0.5, rmax=20.0, npoints=700,
                       scale_cov=list(rng.uniform(0.2, 0.6, len(els))), k_xc=[0.0] * len(els))
        assert _same(pr.evaluate_pbc(atoms), pr.evaluate_pbc_reference(atoms))


def test_no_pair_within_the_cutoff():
    atoms = bulk("Cu", "fcc", a=12.0)                     # neighbours far beyond the cutoff
    pr = P.physrep(elements=[29], rmin=0.5, rmax=20.0, npoints=700, scale_cov=[0.1], k_xc=[0.0])
    e, f = pr.evaluate_pbc(atoms)
    assert _same((e, f), pr.evaluate_pbc_reference(atoms))
    assert e == 0.0 and type(e) is float


def test_fallback_without_vecdot(monkeypatch, cu_dataset):
    monkeypatch.setattr(P, "_VECDOT_MATCHES_NORM", False)
    monkeypatch.setattr(P, "_PAIR_CACHE", type(P._PAIR_CACHE)())
    pr = P.physrep(elements=[29], rmin=0.5, rmax=20.0, npoints=700, scale_cov=[0.33], k_xc=[0.0])
    for atoms in cu_dataset[::7]:
        assert _same(pr.evaluate_pbc(atoms), pr.evaluate_pbc_reference(atoms))


def test_reference_switch(monkeypatch, cu_dataset):
    pr = P.physrep(elements=[29], rmin=0.5, rmax=20.0, npoints=700, scale_cov=[0.33], k_xc=[0.0])
    calls = []
    monkeypatch.setattr(P.physrep, "evaluate_pbc_reference",
                        lambda self, mol: calls.append(1) or (0.0, None))
    monkeypatch.setenv("PTBP_PHYSREP_REFERENCE", "1")
    pr.evaluate_pbc(cu_dataset[0])
    assert calls == [1]


def test_vecdot_self_check_agrees_with_norm():
    if not hasattr(np, "vecdot"):
        pytest.skip("NumPy < 2.0")
    v = np.random.default_rng(0).normal(size=(2000, 3)) * 5
    assert P._vecdot_matches_norm()
    assert np.array_equal(P._norms(v), np.array([np.linalg.norm(x) for x in v]))
