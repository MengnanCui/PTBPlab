"""Divide-and-conquer densities (milestone M7d, :mod:`pydftb.subsystems`): the partition (bonded
fragments, grid cores, buffers through periodic images), the chemical potential of weighted
levels, the exact limit (every subsystem holds all atoms) against the eigensolver for the
Hamiltonians the pattern of M7c takes, finite buffers on a water box, the same results for any
number of worker processes and MPI ranks, the ASE calculator and the refusals.
"""
import dataclasses
import importlib.util
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule

from pydftb import subsystems as dc
from pydftb.ase_calc import PyDFTB
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.params import ParamSet
from pydftb.subsystems import Subsystems
from pydftb.validation import refgen
from pydftb.validation.cases import SKF_SETS

from .conftest import SKF_HCOCU

EXACT = dict(buffer=float("inf"), core=3.0)      # several subsystems, each of all the atoms
TORCH = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")


def _run(name, solver="eigh", **over):
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], **over)
    return refgen.run_pydftb(case, refgen._inputs(case), solver=solver)


def _water_box(n, seed=0):
    """n waters at about 1 g/cm3 on a jittered cubic grid (as the M7c and M7d probes)."""
    rng = np.random.default_rng(seed)
    side = (n * 29.9) ** (1 / 3)
    m = int(np.ceil(n ** (1 / 3)))
    box = Atoms(cell=[side] * 3, pbc=True)
    k = 0
    for i in range(m):
        for j in range(m):
            for l in range(m):
                if k == n:
                    break
                w = molecule("H2O")
                w.rotate(rng.uniform(0, 360), rng.normal(size=3))
                w.translate((np.array([i, j, l]) + 0.5) * side / m + rng.uniform(-0.3, 0.3, 3))
                box += w
                k += 1
    return box


@pytest.fixture(scope="module")
def sk_water():
    return ParamSet(SKF_SETS["ptbp_hcocu"], max_l={"H": 0, "O": 1}).skset(["O", "H"])


def _calc(sk, atoms, solver, **kw):
    calc = DFTB(sk, scc=True, solver=solver, scc_tol=1e-8, **kw)
    return calc.calculate(atoms.get_chemical_symbols(), atoms.positions / BOHR,
                          atoms.cell.array / BOHR, np.ones(3, bool), forces=True, stress=True)


# ---- partition ----------------------------------------------------------------------------------
def test_fragments_through_periodic_images():
    """Bonded fragments across the cell boundary stay whole; cores and buffers keep them."""
    box = _water_box(27)
    box.positions += 0.45 * box.cell[0]           # cut molecules at the boundary...
    box.wrap()                                    # ...and fold their atoms back into the cell
    sym, pos, cell = box.get_chemical_symbols(), box.positions / BOHR, box.cell.array / BOHR
    frags = dc.bonded_fragments(sym, pos, cell, np.ones(3, bool))
    assert len(frags) == 27 and all(len(f) == 3 and sym[f[0]] == "O" for f in frags)
    cores, subs, units = dc.partition(sym, pos, cell, np.ones(3, bool), buffer=3.0, core=3.0)
    whole = {tuple(f) for f in frags}
    for c, s in zip(cores, subs):
        assert set(c) <= set(s)
        for f in frags:
            assert set(f) <= set(c) or not set(f) & set(c)
            assert set(f) <= set(s) or not set(f) & set(s)
    assert sorted(np.concatenate(cores)) == list(range(len(sym)))
    assert {tuple(u) for u in units} == whole


def test_buffer_through_images():
    """Two atoms at opposite faces of the cell are in each other's buffer (minimum image)."""
    atoms = Atoms("H2", positions=[[0.2, 5.0, 5.0], [9.8, 5.0, 5.0]], cell=[10.0] * 3, pbc=True)
    cores, subs, _ = dc.partition(["H", "H"], atoms.positions / BOHR, atoms.cell.array / BOHR,
                                  np.ones(3, bool), buffer=1.0, core=5.0, units="atoms")
    assert len(cores) == 2 and all(len(s) == 2 for s in subs)


def test_networks_go_by_atoms():
    """A metal is one bonded fragment: "auto" cuts it by atoms, "fragments" keeps it whole."""
    cu = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((3, 3, 3))
    sym, pos, cell = cu.get_chemical_symbols(), cu.positions / BOHR, cu.cell.array / BOHR
    _, _, units = dc.partition(sym, pos, cell, np.ones(3, bool), units="auto")
    assert len(units) == len(sym)
    _, subs, units = dc.partition(sym, pos, cell, np.ones(3, bool), units="fragments")
    assert len(units) == 1 and all(len(s) == len(sym) for s in subs)


# ---- the chemical potential of weighted levels ----------------------------------------------------
def test_chemical_potential_of_weighted_levels():
    """Copies of the levels in two subsystems (weights summing to 1) give the levels' own mid-gap
    and occupations; a fractional filling goes to the bisection and keeps the count."""
    e = np.array([-0.5, -0.3, 0.1, 0.4])
    blocks = [(e, np.array([0.25, 0.6, 0.5, 0.7])), (e, np.array([0.75, 0.4, 0.5, 0.3]))]
    mu, fs, ent = dc.chemical_potential(blocks, 2.0, 1e-8)
    assert mu == pytest.approx(-0.1) and ent == 0.0
    assert all(np.array_equal(f, [1.0, 1.0, 0.0, 0.0]) for f in fs)
    mu, fs, ent = dc.chemical_potential(blocks, 2.5, 0.01)
    assert sum(float((b[1] * f).sum()) for b, f in zip(blocks, fs)) == pytest.approx(2.5, abs=1e-12)
    assert -0.3 < mu < 0.4 and ent > 0.0


# ---- the exact limit ------------------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    "h2o", "h2o_nonscc", "ch3oh",                       # molecules, non-SCC
    "water_box_gamma", "water_box_field",               # periodic Gamma point, field (stress)
    "water_slab_dipole",                                # k-points, dipole correction
    "o2_triplet", "o2_relaxed_spin",                    # collinear spin, a shared Fermi level
    "cu4co_300K", "cu4co_300K_fixed_ef",                # T > 0, a fixed Fermi level
    "cuo_chain_k422_indep_k",                           # k-points filled one by one
    "water_box_dftb3", "h2o_gb_field_pc", "h2o_site"])  # DFTB3, solvation, site potentials
def test_exact_limit_is_the_eigensolver(name):
    """Subsystems that all hold every atom give the eigensolver's SCC: energies, charges, forces,
    stress and Fermi levels to rounding, in the same SCC iterations."""
    a = _run(name)
    b = _run(name, Subsystems(**EXACT))
    assert abs(a.energy - b.energy) <= 1e-11
    assert np.abs(a.charges - b.charges).max() <= 1e-11
    assert np.abs(a.forces - b.forces).max() <= 1e-11
    if a.stress is not None:
        assert np.abs(a.stress - b.stress).max() <= 1e-13
    fa = np.atleast_1d(a.fermi_spin if a.fermi_spin is not None else a.fermi)
    fb = np.atleast_1d(b.fermi_spin if b.fermi_spin is not None else b.fermi)
    assert np.abs(fa - fb).max() <= 1e-10
    assert a.n_scc == b.n_scc and b.eigenvalues is None


def test_refgen_skips():
    assert refgen.subsystems_skip(refgen.CASES_BY_NAME["h2o"]) is None
    assert "smearing" in refgen.subsystems_skip(refgen.CASES_BY_NAME["cu4co_mp2_1000K"])
    assert refgen.subsystems_skip(refgen.CASES_BY_NAME["h2o_ntpoly"]) == "DFTB+'s Solver case (M7a)"


# ---- finite buffers and processes -----------------------------------------------------------------
def test_finite_buffers(sk_water):
    """64 waters: the error falls with the buffer (whole molecules), the SCC iterations stay."""
    box = _water_box(64)
    ref = _calc(sk_water, box, "eigh")
    dev = {}
    for rb in (4.0, 6.0):
        s = Subsystems(buffer=rb, core=4.0)
        r = _calc(sk_water, box, s)
        assert s.n_subsystems == 27 and r.n_scc == ref.n_scc
        assert abs(s.last_count_error) < 1e-9
        dev[rb] = (np.abs(r.charges - ref.charges).max(), np.abs(r.forces - ref.forces).max())
    assert dev[4.0][0] < 2e-4 and dev[4.0][1] < 2e-4
    assert dev[6.0][0] < 1e-6 and dev[6.0][1] < 1e-6


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_processes_give_the_same_bits(sk_water):
    """The sums over worker processes do not depend on their number (two contributions per
    entry of the pattern at most)."""
    box = _water_box(27)
    out = []
    for p in (None, 1, 2):
        s = Subsystems(buffer=3.0, core=3.0, processes=p)
        out.append(_calc(sk_water, box, s))
        s.close()
    for r in out[1:]:
        assert r.energy == out[0].energy and np.array_equal(r.charges, out[0].charges)
        assert np.array_equal(r.forces, out[0].forces) and np.array_equal(r.stress, out[0].stress)


MPI_SCRIPT = """
import sys
import numpy as np
from ase import Atoms
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.params import ParamSet
from pydftb.subsystems import Subsystems
from pydftb.validation.cases import SKF_SETS
data = np.load(sys.argv[1])
atoms = Atoms(data["symbols"].tolist(), positions=data["positions"], cell=data["cell"], pbc=True)
sk = ParamSet(SKF_SETS["ptbp_hcocu"], max_l={"H": 0, "O": 1}).skset(["O", "H"])
r = DFTB(sk, scc=True, solver=Subsystems(buffer=3.0, core=3.0, processes="mpi"),
         scc_tol=1e-8).calculate(atoms.get_chemical_symbols(), atoms.positions / BOHR,
                                 atoms.cell.array / BOHR, np.ones(3, bool), forces=True,
                                 stress=True)
np.savez(sys.argv[2], energy=r.energy, charges=r.charges, forces=r.forces, stress=r.stress)
"""


@pytest.mark.skipif(importlib.util.find_spec("mpi4py") is None or shutil.which("mpiexec") is None,
                    reason="needs mpi4py and mpiexec")
def test_mpi_ranks_give_the_same_bits(sk_water, tmp_path):
    """``mpiexec -n 3 python -m pydftb.subsystems script.py``: the in-process result."""
    box = _water_box(27)
    np.savez(tmp_path / "in.npz", symbols=np.array(box.get_chemical_symbols()),
             positions=box.positions, cell=box.cell.array)
    (tmp_path / "run.py").write_text(MPI_SCRIPT)
    env = dict(os.environ, OMP_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([os.path.dirname(os.path.dirname(dc.__file__))]
                                          + sys.path))
    subprocess.run(["mpiexec", "-n", "3", sys.executable, "-m", "pydftb.subsystems",
                    str(tmp_path / "run.py"), str(tmp_path / "in.npz"), str(tmp_path / "out.npz")],
                   check=True, env=env, timeout=600)
    got = np.load(tmp_path / "out.npz")
    ref = _calc(sk_water, box, Subsystems(buffer=3.0, core=3.0))
    assert float(got["energy"]) == ref.energy and np.array_equal(got["charges"], ref.charges)
    assert np.array_equal(got["forces"], ref.forces) and np.array_equal(got["stress"], ref.stress)


def test_ase_calculator():
    """PyDFTB with Subsystems: a new partition at every geometry (two MD-like steps)."""
    box = _water_box(8)
    out = []
    for solver in ("eigh", Subsystems(**EXACT)):
        atoms = box.copy()
        atoms.calc = PyDFTB(SKF_SETS["ptbp_hcocu"], max_l={"H": 0, "O": 1}, scc_tol=1e-10,
                            solver=solver)
        e1, f1 = atoms.get_potential_energy(), atoms.get_forces()
        atoms.positions += 0.01 * f1
        out.append((e1, f1, atoms.get_potential_energy(), atoms.get_forces()))
    for x, y in zip(out[0], out[1]):
        assert np.abs(np.asarray(x) - np.asarray(y)).max() < 1e-9


# ---- refusals -------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sk():
    return ParamSet(SKF_HCOCU).skset(["C", "O", "H"])


def test_refusals(sk):
    w = {"O": -0.028, "H": -0.072, "C": -0.023}
    for opts, msg in (
            (dict(spin_orbit={"O": [0.0, 0.01]}), "two-component spin"),
            (dict(spin="noncolinear", spin_constants=w), "two-component spin"),
            (dict(plus_u={"O": [("p", 0.1)]}), "DFTB\\+U and onsite corrections"),
            (dict(analysis={"cm5": True}), "analysis outputs"),
            (dict(force_evaluation="dynamics"), "force_evaluation='dynamics'"),
            (dict(smearing=("mp", 300.0, 1)), "mp smearing")):
        with pytest.raises(ValueError, match=r"the subsystems \(Subsystems, milestone M7d\) do "
                                             r"not support .*" + msg):
            DFTB(sk, solver=Subsystems(), **opts)
    for kw, msg in ((dict(buffer=-1.0), "buffer"), (dict(units="molecules"), "units"),
                    (dict(processes=-2), "processes")):
        with pytest.raises(ValueError, match=msg):
            Subsystems(**kw)


@TORCH
def test_torch_is_refused(sk):
    with pytest.raises(ValueError, match="the torch backend"):
        DFTB(sk, solver=Subsystems(), backend="torch")
