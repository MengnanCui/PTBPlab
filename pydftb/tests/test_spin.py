"""Collinear spin: limits, magnetic moments, forces and stress, ASE interface.

Comparisons with DFTB+ are the frozen references o2_*, oh_radical*, cuo_chain_spin_k422
in test_reference.py; here only self-consistency (no dftb+ needed).
"""
import numpy as np
import pytest

from pydftb.ase_calc import PyDFTB
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.params import ParamSet
from pydftb.validation import refgen

from .conftest import SKF_HCOCU


def _system(name):
    case = refgen.CASES_BY_NAME[name]
    inp = refgen._inputs(case)
    params = ParamSet(refgen.SKF_SETS[case.skf_set], max_l=inp["max_l"],
                      shell_resolved=case.shell_resolved)
    kpts = None if inp["kpoints"] is None else (np.array(inp["kpoints"]), np.array(inp["kweights"]))
    cell = None if inp["cell_angstrom"] is None else np.array(inp["cell_angstrom"]) / BOHR
    opts = dict(case.dftb3 or {}, spin=True, **case.spin) if case.spin else dict(case.dftb3 or {})
    return (case, params.skset(inp["labels"]), inp["labels"],
            np.array(inp["positions_angstrom"]) / BOHR, cell, kpts, opts)


def test_unpolarised_limit_equals_spin_restricted_run():
    _, sk, labels, pos, _, _, _ = _system("h2o")
    ref = DFTB(sk, scc_tol=1e-12).calculate(labels, pos)
    r = DFTB(sk, scc_tol=1e-12, spin=True, spin_constants={"O": 0.0, "H": 0.0}).calculate(labels, pos)
    assert r.energy == pytest.approx(ref.energy, abs=1e-11)
    assert np.allclose(r.forces, ref.forces, atol=1e-10)
    assert np.allclose(r.charges, ref.charges, atol=1e-10)
    assert np.allclose(r.magmoms, 0.0, atol=1e-10)


def test_magnetic_moments_and_spin_energy():
    _, sk, labels, pos, _, _, opts = _system("o2_triplet")       # one W per type
    r = DFTB(sk, scc_tol=1e-12, **opts).calculate(labels, pos)
    assert r.magmoms.sum() == pytest.approx(2.0, abs=1e-10)
    assert r.q_spin.sum(axis=1) == pytest.approx([7.0, 5.0], abs=1e-10)
    assert r.e_spin == pytest.approx(0.5 * -0.028 * (r.magmoms ** 2).sum(), rel=1e-10)
    assert r.e_spin < 0


def test_relaxed_total_spin_shares_one_fermi_level():
    _, sk, labels, pos, _, _, opts = _system("o2_relaxed_spin")
    r = DFTB(sk, scc_tol=1e-12, **opts).calculate(labels, pos)
    assert r.fermi_spin[0] == r.fermi_spin[1]
    assert r.q_spin.sum() == pytest.approx(12.0, abs=1e-10)


@pytest.mark.parametrize("name", ["o2_triplet_shell", "oh_radical_dftb3"])
def test_spin_forces_match_energy_finite_differences(name):
    case, sk, labels, pos, cell, kpts, opts = _system(name)
    calc = DFTB(sk, smearing=tuple(case.filling), scc_tol=1e-12, max_scc=500, **opts)
    f = calc.calculate(labels, pos).forces
    h = 1e-5
    for atom in range(len(labels)):
        for ax in range(3):
            e = []
            for sgn in (1, -1):
                p = pos.copy()
                p[atom, ax] += sgn * h
                e.append(calc.calculate(labels, p, forces=False).free_energy)
            assert -(e[0] - e[1]) / (2 * h) == pytest.approx(f[atom, ax], abs=1e-8)


def test_spin_stress_matches_strain_finite_differences():
    case, sk, labels, pos, cell, kpts, opts = _system("cuo_chain_spin_k422")
    calc = DFTB(sk, kpts=kpts, smearing=tuple(case.filling), scc_tol=1e-12, max_scc=500, **opts)
    pbc = np.ones(3, bool)
    w = -calc.calculate(labels, pos, cell, pbc).stress * abs(np.linalg.det(cell))
    h = 1e-5
    for a, b in [(0, 0), (0, 1)]:
        eps = np.zeros((3, 3))
        eps[a, b] += h / 2
        eps[b, a] += h / 2
        e = [calc.calculate(labels, pos @ (np.eye(3) + s * eps).T, cell @ (np.eye(3) + s * eps).T,
                            pbc, forces=False).free_energy for s in (1, -1)]
        assert (e[0] - e[1]) / (2 * h) == pytest.approx(w[a, b], abs=5e-8)


def test_ase_interface_reports_spin():
    from ase import Atoms
    o2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.21]])
    o2.calc = PyDFTB(SKF_HCOCU, spin=True, unpaired_electrons=2,
                     spin_constants={"O": [[-0.035, -0.030], [-0.030, -0.028]]})
    assert o2.get_magnetic_moment() == pytest.approx(2.0, abs=1e-8)
    assert o2.get_magnetic_moments().shape == (2,)
    assert o2.calc.get_number_of_spins() == 2
    up, down = o2.calc.get_eigenvalues(spin=0), o2.calc.get_eigenvalues(spin=1)
    assert up.shape == down.shape and not np.allclose(up, down)
    assert o2.calc.get_occupation_numbers(spin=0).sum() == pytest.approx(7.0)
    assert len(o2.calc.get_fermi_levels()) == 2


def test_spin_options_are_checked():
    _, sk, labels, pos, _, _, _ = _system("o2_triplet")
    with pytest.raises(ValueError):
        DFTB(sk, spin=True)                                  # no spin constants
    with pytest.raises(ValueError):
        DFTB(sk, spin=True, spin_constants={"O": -0.03}, solver="sp2")
    with pytest.raises(ValueError, match="needs 1 or 2x2"):
        DFTB(sk, spin=True, spin_constants={"O": [-0.03, -0.02, -0.01]}).calculate(labels, pos)
