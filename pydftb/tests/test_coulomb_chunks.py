"""The periodic Coulomb set-up without the pair table (milestone M7d2, :mod:`pydftb.coulomb`):
the block-by-block neighbour search, gamma, the gradient and the virial over blocks of central
atoms and of G vectors against the stored-table path, one block against many, the third-order
term on such a Coulomb, whole calculations (NumPy and the torch backend) and two MD steps
through ASE, the autodiff refusal, molecules unchanged, the memory of the set-up and the cost
model's memory cap.
"""
import dataclasses
import importlib.util
import tracemalloc

import numpy as np
import pytest

from pydftb import coulomb
from pydftb.constants import BOHR
from pydftb.coulomb import Coulomb
from pydftb.neighbors import neighbor_list, pair_blocks
from pydftb.thirdorder import ThirdOrder
from pydftb.validation import refgen

TORCH = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")


def _box(n=40, seed=3):
    """A triclinic cell with some atoms outside it, every third atom with two sites (unequal U),
    some atoms H-damped."""
    rng = np.random.default_rng(seed)
    cell = np.array([[14.0, 0.5, 0.3], [0.2, 13.0, 0.4], [0.1, 0.6, 15.0]])
    pos = rng.random((n, 3)) @ cell + rng.normal(scale=2.0, size=(n, 3))
    site_atom = np.repeat(np.arange(n), [2 if k % 3 == 0 else 1 for k in range(n)])
    hub = rng.uniform(0.25, 0.5, size=len(site_atom))
    damped = rng.random(n) < 0.3
    return pos, cell, site_atom, hub, damped, rng


@pytest.fixture
def blocks(monkeypatch):
    """Force the blocks: no pair table at any size, small blocks of pairs, G vectors and rows."""
    def force(block_pairs=2 ** 9, g_chunk=16, row_block=7):
        monkeypatch.setattr(coulomb, "PAIR_TABLE_LIMIT", 0)
        monkeypatch.setattr(coulomb, "BLOCK_PAIRS", block_pairs)
        monkeypatch.setattr(coulomb, "G_CHUNK", g_chunk)
        monkeypatch.setattr(coulomb, "ROW_BLOCK", row_block)
    return force


# ---- the neighbour search -----------------------------------------------------------------------
@pytest.mark.parametrize("step", [1, 7, 1000])
def test_pair_blocks_are_the_neighbor_list(step):
    pos, cell, *_ = _box()
    ref = neighbor_list(pos, cell, np.ones(3, bool), 18.0)
    parts = list(pair_blocks(pos, cell, np.ones(3, bool), 18.0, step))
    assert parts[0][0] == 0 and parts[-1][1] == len(pos)
    for t in range(5):
        assert np.array_equal(np.concatenate([p[2 + t] for p in parts]), ref[t])


# ---- gamma, gradient and virial -----------------------------------------------------------------
def test_blocks_match_the_table(blocks):
    pos, cell, site_atom, hub, damped, rng = _box()
    a = Coulomb(pos, site_atom, hub, cell, np.ones(3, bool), damped=damped, damp_exp=4.0)
    blocks()
    b = Coulomb(pos, site_atom, hub, cell, np.ones(3, bool), damped=damped, damp_exp=4.0)
    assert a.nl is not None and b.nl is None and b.sp_a is None
    assert np.abs(a.gamma - b.gamma).max() <= 1e-13 * np.abs(a.gamma).max()
    dq = rng.normal(size=len(site_atom)) * 0.1
    other = rng.normal(size=len(site_atom)) * 0.1
    for kw in ({"virial": True}, {"other": other, "virial": True}):
        (ga, wa), (gb, wb) = a.gradient(dq, **kw), b.gradient(dq, **kw)
        assert np.abs(ga - gb).max() <= 1e-12 * np.abs(ga).max()
        assert np.abs(wa - wb).max() <= 1e-12 * np.abs(wa).max()


def test_one_block_is_many_blocks(blocks):
    """Every entry of gamma and every atom's force comes from one block: the blocking does not
    change a bit."""
    pos, cell, site_atom, hub, damped, rng = _box()
    dq = rng.normal(size=len(site_atom)) * 0.1
    out = []
    for block_pairs in (2 ** 30, 2 ** 9):
        blocks(block_pairs=block_pairs)
        c = Coulomb(pos, site_atom, hub, cell, np.ones(3, bool), damped=damped, damp_exp=4.0)
        out.append((c._step, c.gamma, c.gradient(dq)[0]))
    assert out[0][0] >= len(pos) > out[1][0]
    assert np.array_equal(out[0][1], out[1][1]) and np.array_equal(out[0][2], out[1][2])


def test_molecules_keep_the_table(blocks):
    pos, _, site_atom, hub, _, _ = _box(n=12)
    blocks()
    c = Coulomb(pos, site_atom, hub)
    assert c.nl is not None and c._step is None


def test_third_order_on_blocks(blocks):
    pos, cell, site_atom, hub, damped, rng = _box()
    ud = rng.uniform(-0.2, -0.05, size=len(site_atom))
    a = ThirdOrder(Coulomb(pos, site_atom, hub, cell, np.ones(3, bool), damped=damped,
                           damp_exp=4.0), ud)
    blocks()
    b = ThirdOrder(Coulomb(pos, site_atom, hub, cell, np.ones(3, bool), damped=damped,
                           damp_exp=4.0), ud)
    for x, y in zip(a.nl, b.nl):
        assert np.array_equal(x, y)
    assert np.array_equal(a.gamma3, b.gamma3)


# ---- whole calculations -------------------------------------------------------------------------
# the frozen periodic cases; shell-resolved charges (several sites per atom, Cu's soft 4p shell)
# on the CuO chain, which no frozen periodic case has
SHELL = {"shell_resolved": True}
SHELL_DFTB3 = {"shell_resolved": True, "dftb3": {
    "third_order": "full", "hubbard_derivs": {"Cu": [-0.05, -0.03, -0.08], "O": [-0.1575, -0.14]}}}
CALCULATIONS = [
    ("water_box_gamma", {}), ("cuo_chain_k422", {}), ("gdo_rocksalt_k444", {}),   # Gamma, k, f
    ("water_box_dftb3", {}), ("water_box_h5", {}), ("water_box_pc", {}),       # DFTB3, H5, charges
    ("cuo_chain_k422", SHELL), ("cuo_chain_dftb3_k422", SHELL_DFTB3)]           # shell-resolved


@pytest.mark.parametrize("name, over", CALCULATIONS,
                         ids=[n + ("_shell" if o else "") for n, o in CALCULATIONS])
def test_calculations_match_the_table(name, over, blocks, monkeypatch):
    why = refgen.missing_requirement(refgen.load(name))
    if why:
        pytest.skip(why)
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], **over)
    a = refgen.run_pydftb(case, refgen._inputs(case))
    calls = []
    matrix_blocks = Coulomb._matrix_blocks
    monkeypatch.setattr(Coulomb, "_matrix_blocks",
                        lambda self: calls.append(self._step) or matrix_blocks(self))
    blocks(block_pairs=2 ** 11, g_chunk=64, row_block=5)
    b = refgen.run_pydftb(case, refgen._inputs(case))
    assert calls                                     # the block path ran
    assert a.n_scc == b.n_scc
    assert abs(a.energy - b.energy) <= 1e-11
    assert np.abs(a.charges - b.charges).max() <= 1e-11
    assert np.abs(a.forces - b.forces).max() <= 1e-11
    if a.stress is not None:
        assert np.abs(a.stress - b.stress).max() <= 1e-13


def test_ase_md_two_steps(blocks, monkeypatch):
    """Two velocity-Verlet steps of a periodic water box through the ASE calculator (a new
    Coulomb at every geometry): blocks and table give the same trajectory to rounding."""
    from ase import units
    from ase.md.verlet import VelocityVerlet

    from pydftb.ase_calc import PyDFTB
    from pydftb.validation.cases import SKF_SETS

    from .test_subsystems import _water_box
    calls = []
    matrix_blocks = Coulomb._matrix_blocks
    monkeypatch.setattr(Coulomb, "_matrix_blocks",
                        lambda self: calls.append(self._step) or matrix_blocks(self))
    out = []
    for force in (False, True):
        if force:
            blocks(block_pairs=2 ** 13, g_chunk=64, row_block=17)
        atoms = _water_box(8)
        atoms.set_velocities(np.random.default_rng(5).normal(scale=0.02, size=(len(atoms), 3)))
        atoms.calc = PyDFTB(SKF_SETS["ptbp_hcocu"], max_l={"H": 0, "O": 1}, scc_tol=1e-10)
        VelocityVerlet(atoms, timestep=0.5 * units.fs).run(2)
        out.append((atoms.positions.copy(), atoms.get_potential_energy(), atoms.get_forces(),
                    len(calls)))
    (pa, ea, fa, na), (pb, eb, fb, nb) = out
    assert na == 0 and nb >= 3                       # three geometries on the block path
    assert np.abs(pa - pb).max() <= 1e-12
    assert abs(ea - eb) <= 1e-9 and np.abs(fa - fb).max() <= 1e-9


def test_set_up_memory(blocks):
    """216 waters: without the table the set-up allocates a few times gamma, not the 1.8M pairs
    (130 MB) and their temporaries."""
    from .test_subsystems import _water_box
    box = _water_box(216)
    pos, cell = box.positions / BOHR, box.cell.array / BOHR
    hub = np.where(np.array(box.get_chemical_symbols()) == "O", 0.4, 0.42)
    peaks = []
    for force in (False, True):
        if force:
            blocks(block_pairs=2 ** 15, g_chunk=256, row_block=256)
        tracemalloc.start()
        c = Coulomb(pos, np.arange(len(pos)), hub, cell, np.ones(3, bool))
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    assert peaks[0] > 100e6 and peaks[1] < 3 * c.gamma.nbytes + 40e6


@TORCH
@pytest.mark.parametrize("name", ["water_box_gamma", "water_box_dftb3"])
def test_torch_backend_on_blocks(name, blocks):
    """The torch backend's forward calculation takes gamma and its gradient from the NumPy
    Coulomb: on blocks it gives the table path's results to rounding."""
    case = refgen.CASES_BY_NAME[name]
    a = refgen.run_pydftb(case, refgen._inputs(case), backend="torch")
    blocks(block_pairs=2 ** 11, g_chunk=64, row_block=5)
    b = refgen.run_pydftb(case, refgen._inputs(case), backend="torch")
    assert a.n_scc == b.n_scc
    assert abs(a.energy - b.energy) <= 1e-11
    assert np.abs(np.asarray(a.charges) - np.asarray(b.charges)).max() <= 1e-11
    assert np.abs(np.asarray(a.forces) - np.asarray(b.forces)).max() <= 1e-11


@TORCH
def test_autodiff_refuses_blocks(blocks):
    import torch

    from pydftb.autodiff.model import gamma_matrix
    pos, cell, site_atom, hub, _, _ = _box()
    blocks()
    c = Coulomb(pos, site_atom, hub, cell, np.ones(3, bool))
    with pytest.raises(NotImplementedError, match="PAIR_TABLE_LIMIT"):
        gamma_matrix(c, torch.as_tensor(pos), torch.as_tensor(cell), torch.as_tensor(3.2 * hub))


def test_cost_model_memory_cap():
    """pydftb's Coulomb memory (atom and site pairs) stops growing at the pair-table limit for
    periodic systems; DFTB+'s and that of molecules (always a table) do not."""
    from pydftb.perfmodel import CostModel, Shape
    limit = float(coulomb.PAIR_TABLE_LIMIT)
    base = Shape(natoms=1000, norb=6000, sk_pairs=1e6)
    at_limit = dataclasses.replace(base, coulomb_pairs=limit, site_pairs=0.2 * limit)
    beyond = dataclasses.replace(base, coulomb_pairs=1e9, site_pairs=0.2e9)
    model = CostModel()
    assert model.memory("pydftb", beyond) == pytest.approx(model.memory("pydftb", at_limit),
                                                           rel=1e-12)
    assert model.memory("pydftb", beyond) < model.memory("pydftb", dataclasses.replace(
        beyond, periodic=False))
    assert model.memory("dftbplus", beyond) > model.memory("dftbplus", at_limit)
