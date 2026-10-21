"""Cost model of a single point (pydftb.perfmodel, M3c): features, estimates, probe."""
import json

import numpy as np
import pytest
from ase.build import bulk, molecule

from pydftb import perfmodel as pm
from pydftb.coulomb import Coulomb

CU = pm.Species(((2, 0.509501), (1, 0.079980), (0, 0.240327)))      # PTBP's Cu shells
C = pm.Species(((1, 0.356481), (0, 0.445693)))
H = pm.Species(((0, 0.391066),))


@pytest.mark.parametrize("grid, nk", [((1, 1, 1), 1), ((2, 2, 2), 4), ((3, 3, 3), 14),
                                      ((4, 4, 8), 64), ((15, 15, 15), 1688), ((2, 1, 1), 1)])
def test_irreducible_kpoints(grid, nk):
    assert pm.irreducible_kpoints(grid) == nk


def test_block_shells_and_basis():
    assert [pm.basis_size(pm.block_shells(z)) for z in (1, 6, 29, 64)] == [1, 4, 9, 16]


def _exact_counts(atoms, species):
    """Atom pairs and shell pairs of pydftb's own Coulomb set-up (the reference)."""
    from pydftb.constants import BOHR
    sym = atoms.get_chemical_symbols()
    site_atom = [i for i, s in enumerate(sym) for _ in species[s].shells]
    hub = [u for s in sym for _, u in species[s].shells]
    cell = np.array(atoms.cell) / BOHR if atoms.pbc.any() else None
    c = Coulomb(atoms.positions / BOHR, site_atom, hub, cell, atoms.pbc if cell is not None else None)
    return len(c.nl[0]), len(c.sp_a)


@pytest.mark.parametrize("atoms", [bulk("Cu", "fcc", a=3.61, cubic=True),
                                   bulk("C", "diamond", a=3.567, cubic=True) * (2, 1, 1)])
def test_pair_counts_of_periodic_cells_match_the_coulomb_setup(atoms):
    species = {"Cu": CU, "C": C}
    s = pm.shape(atoms.get_chemical_symbols(), species, cell=atoms.cell, pbc=atoms.pbc)
    pairs, site_pairs = _exact_counts(atoms, species)
    assert s.coulomb_pairs == pytest.approx(pairs, rel=0.02)
    assert s.site_pairs == pytest.approx(site_pairs, rel=0.02)
    assert s.norb == sum(species[x].norb for x in atoms.get_chemical_symbols())


def test_cluster_pairs_are_counted():
    mol = molecule("CH4")
    s = pm.shape(mol.get_chemical_symbols(), {"C": C, "H": H}, positions=mol.positions)
    assert not s.periodic and s.nk == 1 and s.gamma
    assert s.coulomb_pairs == 5 * 4
    assert s.site_pairs == _exact_counts(mol, {"C": C, "H": H})[1]
    # every pair is within the SK cut-off: sum of norb_i * norb_j over ordered pairs
    assert s.sk_pairs == 4 * 1 * 4 * 2 + 1 * 1 * 4 * 3


def test_the_smallest_hubbard_u_sets_the_reach():
    cell = bulk("Cu", "fcc", a=3.61, cubic=True)
    far = pm.shape(["Cu"] * 4, {"Cu": CU}, cell=cell.cell, pbc=cell.pbc)
    near = pm.shape(["Cu"] * 4, {"Cu": pm.Species(((2, 0.51), (1, 0.4), (0, 0.4)))},
                    cell=cell.cell, pbc=cell.pbc)
    assert far.coulomb_pairs > 20 * near.coulomb_pairs      # 96 vs ~20 Bohr


def _cu(reps, grid):
    a = bulk("Cu", "fcc", a=3.61, cubic=True) * reps
    return pm.shape(a.get_chemical_symbols(), {"Cu": CU}, cell=a.cell, pbc=a.pbc, grid=grid)


@pytest.mark.parametrize("reps, grid, n_scc, wall", [            # calibration runs (M3c)
    ((1, 1, 1), (8, 8, 8), 7, 1.49), ((2, 2, 2), (4, 4, 4), 10, 29.4),
    ((2, 2, 4), (1, 1, 1), 22, 26.1), ((3, 3, 3), (2, 2, 2), 14, 152.9)])
def test_pydftb_time_within_the_calibration_error(reps, grid, n_scc, wall):
    t = pm.CostModel().pydftb_time(_cu(reps, grid), n_scc=n_scc)
    assert t == pytest.approx(wall, rel=0.4)


def test_memory_follows_the_coulomb_pairs():
    model = pm.CostModel()
    big = model.memory("pydftb", _cu((2, 2, 4), (1, 1, 1)))          # measured 2231 MB
    assert big / 2 ** 20 == pytest.approx(2231, rel=0.3)
    assert model.memory("dftbplus", _cu((2, 2, 4), (1, 1, 1))) / 2 ** 20 == pytest.approx(318, rel=0.3)
    assert model.memory("pydftb", _cu((1, 1, 1), (8, 8, 8))) < big


def test_threads():
    model = pm.CostModel()
    kpts = _cu((1, 1, 1), (12, 12, 12))
    gamma = _cu((2, 2, 4), (1, 1, 1))
    assert pm.effective_kthreads(kpts, 4) == 4
    assert pm.effective_kthreads(gamma, 4) == 1                       # nothing to split
    assert model.kthread_speedup(gamma, 4) == 1.0
    assert 1.0 < model.kthread_speedup(kpts, 2) < model.kthread_speedup(kpts, 4) < 2.0
    small = _cu((2, 2, 2), (4, 4, 4))                                 # 288 orbitals
    assert model.omp_speedup(small, 4) == 1.0                         # not planned below 500
    large = _cu((3, 3, 3), (2, 2, 2))
    assert 1.2 < model.omp_speedup(large, 4) < 2.0                   # measured 1.74


def test_speed_scales_every_estimate():
    s = _cu((2, 2, 2), (2, 2, 2))
    assert pm.CostModel(speed=2.0).time("pydftb", s) == pytest.approx(pm.CostModel().time("pydftb", s) / 2)


def test_probe_and_its_cache(tmp_path, monkeypatch):
    out = pm.probe(repeat=1)
    assert set(out["timings"]) == set(pm.PROBE_REFERENCE) and out["speed"] > 0
    assert pm.speed_from_timings(pm.PROBE_REFERENCE) == pytest.approx(1.0)
    assert pm.speed_from_timings({k: 2 * v for k, v in pm.PROBE_REFERENCE.items()}) == pytest.approx(0.5)
    path = tmp_path / "cache" / "probe.json"
    assert pm.load_probe(path) is None                                # nothing cached, not run
    monkeypatch.setattr(pm, "probe", lambda repeat=5: {"timings": dict(pm.PROBE_REFERENCE), "speed": 1.0})
    assert pm.load_probe(path, run=True) == pytest.approx(1.0)
    cache = json.loads(path.read_text())
    assert list(cache) == [pm.machine_fingerprint()]

    def fail(repeat=5):
        raise AssertionError("cached: must not probe again")

    monkeypatch.setattr(pm, "probe", fail)
    assert pm.load_probe(path, run=True) == pytest.approx(1.0)
