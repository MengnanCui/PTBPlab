"""Active learning (M8c): bootstrap committees, their uncertainty, selection, VASP inputs."""
from __future__ import annotations

import os

import numpy as np
import pytest
from ase.build import bulk, molecule
from ase.cluster import Icosahedron

from pydftb import DFTB, ParamSet
from pydftb.active import (
    Committee,
    bootstrap_weights,
    committee_predict,
    fingerprints,
    fit_committee,
    select,
    write_candidates,
)
from pydftb.adaptive import RBF, AdaptiveParams
from pydftb.adaptive_fit import Sample, evaluate_sample
from pydftb.constants import BOHR
from pydftb.validation.cases import SKF_SETS

HA = 27.2113845


@pytest.fixture(scope="module")
def sk():
    return ParamSet(SKF_SETS["ptbp_hcocu"], max_l={"H": 0, "O": 1}).skset(["O", "H"])


def _water(d, ang, k=0):
    w = molecule("H2O")
    w.set_distance(0, 1, d, fix=0)
    w.set_angle(1, 0, 2, ang)
    w.positions[2] += [0.01 * k, -0.01, 0.015]
    return w


TRUTH = AdaptiveParams("O", RBF((1.0, 2.0), 0.4), onsite={"p": (0.02, -0.03, 0.01)},
                       hopping={"p": (0.05, 0.04, -0.03)})
MODEL = AdaptiveParams("O", RBF((1.5,), 0.6), onsite={"p": (0.0, 0.0)},
                       hopping={"p": (0.0, 0.0)})
GEOMS = [(0.92, 100), (0.95, 104), (0.98, 108), (1.00, 104), (1.02, 112), (1.05, 109)]


@pytest.fixture(scope="module")
def data(sk):
    """Waters near equilibrium, energies of a richer model plus 1 meV/atom of noise; groups
    of two."""
    rng = np.random.default_rng(5)
    out = []
    for k, (d, a) in enumerate(GEOMS):
        w = _water(d, a, k)
        r = DFTB(sk, scc_tol=1e-11, adaptive=TRUTH).calculate(w.get_chemical_symbols(),
                                                              w.positions / BOHR)
        out.append(Sample(w, r.free_energy * HA / 3 + 1e-3 * rng.normal(), name=str(k)))
    return out


@pytest.fixture(scope="module")
def committee(sk, data):
    return fit_committee([MODEL], data, sk, {"scc_tol": 1e-11}, n=3, seed=1,
                         groups=[k // 2 for k in range(len(data))], processes=1,
                         max_evaluations=15, base_evaluations=30)


def test_bootstrap_weights():
    rng = np.random.default_rng(0)
    w = bootstrap_weights(10, rng)
    assert w.sum() == 10 and w.dtype.kind == "i"
    g = np.array([0, 0, 1, 1, 1, 2])
    for _ in range(20):
        w = bootstrap_weights(6, rng, g)
        for lab in range(3):
            assert len(set(w[g == lab])) == 1                     # a group comes whole
        assert sum(w[g == lab][0] for lab in range(3)) == 3       # 3 groups drawn
    a = bootstrap_weights(6, np.random.default_rng(3), g)
    b = bootstrap_weights(6, np.random.default_rng(3), g)
    assert (a == b).all()


def test_committee_predictions(sk, data, committee):
    """Members differ; predictions are each member's own energy minus its offset; parallel
    equals serial bit for bit; the spread grows away from the data."""
    assert len(committee.members) == 3 and len(committee.draws) == 3
    xs = [np.concatenate([m.vector() for m in mem]) for mem in committee.members]
    assert max(np.abs(x - xs[0]).max() for x in xs) > 1e-6
    frames = [_water(1.0, 106, 2), _water(1.35, 140, 3)]          # inside, far outside
    opts = {"scc_tol": 1e-11}
    u1 = committee_predict(committee, frames, sk, opts, processes=1)
    u2 = committee_predict(committee, frames, sk, opts, processes=2)
    assert np.array_equal(u1.energy, u2.energy) and np.array_equal(u1.sigma_f, u2.sigma_f)
    e, _, f = evaluate_sample(committee.members[1], Sample(frames[1], 0.0), sk, opts, True)
    assert u1.energy[1, 1] == e - committee.offsets[1]
    assert np.array_equal(u1.forces[1][1], f)
    assert np.allclose(u1.sigma_e, u1.energy.std(axis=0))
    dev = u1.forces[0] - u1.forces[0].mean(axis=0)
    assert np.isclose(u1.sigma_f[0], np.sqrt((dev ** 2).sum(2).mean(0)).max())
    assert not u1.failed.any()
    assert u1.sigma_e[1] > 3 * u1.sigma_e[0]
    assert u1.sigma_f[1] > 3 * u1.sigma_f[0]


def test_identical_members_and_failures(sk, committee):
    same = Committee(committee.base, (committee.base,) * 3, np.zeros(3))
    frames = [_water(1.0, 106), _water(1.2, 130)]
    u = committee_predict(same, frames, sk, {"scc_tol": 1e-11}, processes=1)
    assert (u.sigma_e == 0).all() and u.sigma_f.max() < 1e-12      # the mean's rounding
    bad = committee_predict(same, frames[:1], sk, {"scc_tol": 1e-14, "max_scc": 1},
                            processes=1)
    assert bad.failed.all() and np.isinf(bad.score()).all() and bad.forces == [None]


def test_fingerprints_and_select():
    cu = AdaptiveParams("Cu", RBF(()))
    fcc = bulk("Cu", "fcc", a=3.61)
    sup = fcc.repeat(2)
    sup.positions += [0.3, 0.1, 0.2]
    ico = Icosahedron("Cu", 2, latticeconstant=3.61)
    fp = fingerprints(cu, [fcc, sup, bulk("Cu", "bcc", a=2.87), ico])
    assert np.abs(fp[0] - fp[1]).max() < 1e-12                    # per atom, translation
    assert np.linalg.norm(fp[0] - fp[2]) > 0.1 and np.linalg.norm(fp[0] - fp[3]) > 0.1
    # select: most uncertain first, the threshold, no near duplicates, away from the reference
    score = np.array([0.5, 0.9, 0.85, 0.1, 0.7])
    f = np.array([[0.0], [1.0], [1.01], [2.0], [3.0]])
    assert select(score, f, n=10, min_distance=0.0) == [1, 2, 4, 0, 3]
    assert select(score, f, n=10, min_distance=0.1) == [1, 4, 0, 3]
    assert select(score, f, n=2, min_distance=0.1) == [1, 4]
    assert select(score, f, n=10, threshold=0.6, min_distance=0.1) == [1, 4]
    assert select(score, f, n=10, min_distance=0.1, reference=[[3.0]]) == [1, 0, 3]
    assert select(np.array([np.inf, 1.0]), f[:2], n=1) == [0]


def test_write_candidates(tmp_path):
    ico = Icosahedron("Cu", 2, latticeconstant=3.61)
    cell = bulk("Cu", "fcc", a=3.61, cubic=True)
    man = write_candidates(str(tmp_path), [ico, cell], ["cu13_md_0100fs", "fcc4_hot"],
                           group="round1")
    assert [m["path"] for m in man] == ["round1/cu13_md_0100fs", "round1/fcc4_hot"]
    d0, d1 = (os.path.join(tmp_path, m["path"]) for m in man)
    assert os.path.exists(os.path.join(d0, "KPOINTS"))
    with open(os.path.join(d0, "INCAR")) as fh:
        inc0 = fh.read()
    with open(os.path.join(d1, "INCAR")) as fh:
        inc1 = fh.read()
    assert "ISPIN = 2" in inc0 and "KSPACING" not in inc0
    assert "KSPACING" in inc1 and not os.path.exists(os.path.join(d1, "KPOINTS"))
    assert man[0]["gamma_only"] and not man[1]["gamma_only"]
    assert os.path.exists(os.path.join(tmp_path, "run_19.sh"))
    from ase.io import read
    box = read(os.path.join(d0, "POSCAR"))
    assert box.cell.lengths().min() > 12.0
