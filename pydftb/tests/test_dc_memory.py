"""Memory of the divide and conquer and of the SK pair blocks (milestone M7d3).

The compact subsystem indices (int32, the entries with p > 0 first, no copies), the workers that
build their own indices and get only the pattern entries they use, the contiguous assignment,
and build_pairs in chunks of pairs: all give the same bits as before.
"""
import dataclasses
import os

import numpy as np
import pytest
import scipy.sparse as sp
from ase.build import bulk, molecule

from pydftb import hamiltonian as ham
from pydftb import subsystems as dc
from pydftb.constants import BOHR
from pydftb.params import ParamSet
from pydftb.subsystems import Subsystems
from pydftb.validation.cases import SKF_SETS

from .test_subsystems import _calc, _water_box


@pytest.fixture(scope="module")
def sk_water():
    return ParamSet(SKF_SETS["ptbp_hcocu"], max_l={"H": 0, "O": 1}).skset(["O", "H"])


def _old_index(idx, offsets, norb_atom, core, atoms):
    """M7d's index (40 bytes per entry), kept here as the reference."""
    orbs = np.concatenate([np.arange(offsets[a], offsets[a] + norb_atom[a]) for a in atoms])
    is_core = np.repeat(np.isin(atoms, core), norb_atom[atoms])
    blk = idx[orbs][:, orbs].tocoo()
    r, c = blk.row.astype(np.int32), blk.col.astype(np.int32)
    g = blk.data.astype(np.int64) - 1
    p = 0.5 * (is_core[r].astype(float) + is_core[c].astype(float))
    keep = p > 0
    return {"orbs": orbs, "core": is_core, "r": r, "c": c, "g": g, "kr": r[keep],
            "kc": c[keep], "kg": g[keep], "kp": p[keep]}


def _pattern(n, density, seed):
    rng = np.random.default_rng(seed)
    a = sp.random(n, n, density=density, random_state=rng, format="csr")
    a = (a + a.T + sp.eye(n)).tocsr()
    a.sort_indices()
    return a.indptr, a.indices


def test_compact_index_is_the_old_one():
    """The same entries, partition weights, dense blocks and density contributions."""
    norb_atom = np.array([4, 1, 1] * 10)
    offsets = np.concatenate([[0], np.cumsum(norb_atom)[:-1]])
    n = int(norb_atom.sum())
    indptr, indices = _pattern(n, 0.15, 0)
    idx = dc._position_matrix(indptr, indices, n)
    rng = np.random.default_rng(1)
    data = rng.normal(size=len(indices))
    for core, atoms in [(np.array([0, 1, 2]), np.arange(12)),
                        (np.array([3, 4, 5, 9]), np.array([0, 3, 4, 5, 9, 20, 21, 22])),
                        (np.arange(30), np.arange(30))]:
        new, old = dc._index(idx, offsets, norb_atom, core, atoms), \
            _old_index(idx, offsets, norb_atom, core, atoms)
        m = len(new["orbs"])
        assert new["lin"].dtype == np.int32 and new["g"].dtype == np.int32
        assert np.array_equal(new["orbs"], old["orbs"])
        # all entries: the same set of (local position, pattern position)
        got = sorted(zip(new["lin"].tolist(), new["g"].tolist()))
        ref = sorted(zip((old["r"].astype(int) * m + old["c"]).tolist(), old["g"].tolist()))
        assert got == ref
        nk = new["nk"]
        kept = sorted(zip(new["lin"][:nk].tolist(), new["g"][:nk].tolist(),
                          dc._partition_weights(new).tolist()))
        ref = sorted(zip((old["kr"].astype(int) * m + old["kc"]).tolist(), old["kg"].tolist(),
                         old["kp"].tolist()))
        assert kept == ref
        dense = np.zeros((m, m))
        dense[old["r"], old["c"]] = data[old["g"]]
        assert np.array_equal(dc._dense(data, new, float), dense)
        # a density contribution, bit for bit
        e = np.sort(rng.normal(size=m))
        c = rng.normal(size=(m, m))
        f = (e < 0).astype(float) * 0.7
        rho, wdm = dc.density_subsystems([new], [(e, None, c)], [f], True, len(data), float)
        r_ref, w_ref = np.zeros(len(data)), np.zeros(len(data))
        occ = f > 0
        co, cf = c[:, occ], c[:, occ] * f[occ]
        r_ref[old["kg"]] += old["kp"] * (cf @ co.T)[old["kr"], old["kc"]]
        w_ref[old["kg"]] += old["kp"] * ((cf * e[occ]) @ co.T)[old["kr"], old["kc"]]
        assert np.array_equal(rho.view(np.uint64), r_ref.view(np.uint64))
        assert np.array_equal(wdm.view(np.uint64), w_ref.view(np.uint64))


def test_localised_entries():
    norb_atom = np.array([4, 1, 1] * 8)
    offsets = np.concatenate([[0], np.cumsum(norb_atom)[:-1]])
    n = int(norb_atom.sum())
    indptr, indices = _pattern(n, 0.2, 2)
    data = np.random.default_rng(3).normal(size=len(indices))
    group = [(np.array([0, 1, 2]), np.arange(9)), (np.array([6, 7]), np.arange(3, 12))]
    idx = dc._position_matrix(indptr, indices, n)
    full = [dc._index(idx, offsets, norb_atom, c, a) for c, a in group]
    used, local = dc._build_group(group, (indptr, indices, n, offsets, norb_atom))
    assert np.all(np.diff(used) > 0) and len(used) < len(indices)
    for s_full, s_loc in zip(full, local):
        assert np.array_equal(data[used][s_loc["g"]], data[s_full["g"]])
    assert dc._build_group([], (indptr, indices, n, offsets, norb_atom))[0].size == 0


def test_contiguous_assignment():
    s = Subsystems()
    s.sizes = np.array([10, 12, 11, 30, 9, 10, 10, 12] * 5)
    for nw in (1, 2, 3, 5):
        owner = s._assign(nw)
        assert sorted(a for o in owner for a in o) == list(range(40))
        assert all(o == list(range(o[0], o[-1] + 1)) for o in owner if o)   # runs
        load = [float((s.sizes[o].astype(float) ** 3).sum()) for o in owner]
        assert max(load) - min(load) <= 2 * 30.0 ** 3
    owner = s._assign(8)                                   # 40 < 8 * 8: largest first
    assert sorted(a for o in owner for a in o) == list(range(40))


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_workers_use_part_of_the_pattern(sk_water):
    """216 waters, 4 workers: every worker gets fewer entries than the pattern, and the results
    are the serial ones, bit for bit."""
    box = _water_box(216)
    out, used = [], None
    for p in (None, 4):
        s = Subsystems(buffer=3.0, core=3.0, processes=p)
        out.append(_calc(sk_water, box, s))
        if p:
            used = [len(u) for u in s._exec.used]
            nnz = s.nnz
        s.close()
    assert all(u < 0.8 * nnz for u in used), (used, nnz)
    a, b = out
    assert a.energy == b.energy and np.array_equal(a.charges, b.charges)
    assert np.array_equal(a.forces, b.forces) and np.array_equal(a.stress, b.stress)


def test_worker_errors_are_reported():
    bad = (np.zeros(2, int), np.zeros(0, int), 1, np.zeros(1, int), np.ones(1, int))
    with pytest.raises(RuntimeError, match="worker failed"):
        dc._Workers([[(np.array([0]), np.array([5]))]], bad)     # atom 5 does not exist


# ---- build_pairs in chunks ------------------------------------------------------------------------
def _pairs(sk, atoms, **kw):
    lab = atoms.get_chemical_symbols()
    per = bool(atoms.pbc.any())
    cell = atoms.cell.array / BOHR if per else None
    return ham.build_pairs(lab, atoms.positions / BOHR, cell, np.ones(3, bool) if per else None,
                           sk, ham.Basis.build(lab, sk), **kw)


FIELDS = ("i", "j", "shift", "vec", "dist", "rows", "cols", "pair_of", "h", "s", "dh", "ds")


@pytest.mark.parametrize("kw", [{}, {"derivatives": True},
                                {"derivatives": True, "fd_delta": 1e-4}])
@pytest.mark.parametrize("system", ["water", "cu"])
def test_pair_chunks_give_the_same_bits(monkeypatch, system, kw):
    if system == "water":
        sk = ParamSet(SKF_SETS["ptbp_hcocu"], max_l={"H": 0, "O": 1}).skset(["O", "H"])
        atoms = _water_box(8)
    else:
        sk = ParamSet(SKF_SETS["ptbp_hcocu"], max_l={"Cu": 2}).skset(["Cu"])
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    ref = _pairs(sk, atoms, **kw)
    # (pairs, elements) per chunk; 1 and 7 elements, 1 pair: two pairs per chunk at the least
    for pairs, elements in ((1 << 14, 1), (1 << 14, 7), (1, 1 << 18), (3, 1 << 18), (5, 33)):
        monkeypatch.setattr(ham, "PAIR_CHUNK_PAIRS", pairs)
        monkeypatch.setattr(ham, "PAIR_CHUNK_ELEMENTS", elements)
        got = _pairs(sk, atoms, **kw)
        for f in FIELDS:
            x, y = getattr(got, f), getattr(ref, f)
            if y is None:
                assert x is None
                continue
            assert x.dtype == y.dtype and x.shape == y.shape
            assert np.array_equal(x.view(np.uint8), y.view(np.uint8)), (pairs, elements, f)
        assert [(a, b, s.tolist()) for a, b, s in got.groups] == \
            [(a, b, s.tolist()) for a, b, s in ref.groups]


def test_pairs_of_a_molecule_and_of_nothing():
    sk = ParamSet(SKF_SETS["ptbp_hcocu"], max_l={"C": 1, "O": 1}).skset(["C", "O"])
    pb = _pairs(sk, molecule("CO"), derivatives=True)
    assert len(pb.h) == 2 * 16 and pb.dh.shape == (32, 3)
    one = dataclasses.replace(pb)                 # a dataclass: the fields as before
    assert one.rows.dtype == np.int64 and one.h.dtype == np.float64
    far = molecule("CO")
    far.positions[1] += [0.0, 0.0, 50.0]
    pb = _pairs(sk, far, derivatives=True)
    assert len(pb.h) == 0 and pb.dh.shape == (0, 3) and pb.rows.dtype == np.int64
