"""The start of the SCC (milestone M6f-1): DFTB+'s InitialCharges (aufbau over the shells)
and collinear InitialSpins, and the ASE calculator's initial magnetic moments.

The SCC iterations and errors against DFTB+ are the frozen references *_initial_* in
test_reference.py; here the rules themselves and that the start does not change the solution.
"""
import numpy as np
import pytest
from ase import Atoms

from pydftb.ase_calc import PyDFTB
from pydftb.calculator import DFTB, SCCControl, _aufbau_shell_populations
from pydftb.hamiltonian import Basis

from .conftest import SKF_HCOCU
from .test_spin import _system


def _start(calc, labels, pos):
    """The input populations of the first SCC iteration (the start vector): one iteration
    and no mixing after it."""
    return calc.calculate(labels, pos, forces=False, scc_control=SCCControl(1, 1, 1e-10)).q_in


def test_aufbau_fills_the_shells_in_order():
    case, sk, labels, _, _, _, _ = _system("cu4co_shell_resolved")       # Cu Cu Cu Cu C O
    b = Basis.build(labels, sk)
    occ = np.array([sk.occupation[b.species[a]][l] for a, l in zip(b.shell_atom, b.shell_l)])
    q = _aufbau_shell_populations(b, occ, [0.2, -0.2, 0.1, -0.1, 0.3, -0.3])
    by_atom = [q[b.shell_atom == a] for a in range(6)]
    assert np.allclose(by_atom[0], [2.0, 6.0, 2.8])      # Cu (s1 d10) with 10.8: 4s2 4p6 3d2.8
    assert np.allclose(by_atom[1], [2.0, 6.0, 3.2])
    assert np.allclose(by_atom[4], [2.0, 1.7])           # C with 3.7
    assert np.allclose(by_atom[5], [2.0, 4.3])           # O with 6.3
    _, sk_w, labels_w, pos_w, _, _, _ = _system("h2o")
    with pytest.raises(ValueError, match="not enough orbitals on 'H'"):
        DFTB(sk_w, initial_charges=[1.5, 0.0, -1.5]).calculate(labels_w, pos_w)


def test_initial_charges_start_vector():
    """Atom populations N_A = Z_A - q_A (aufbau, here within the shells), scaled to the
    electron count with a warning when the charges do not add up to the total charge."""
    _, sk, labels, pos, _, _, _ = _system("ch3oh")              # C O H H H H
    z = np.array([4.0, 6.0, 1.0, 1.0, 1.0, 1.0])
    q = _start(DFTB(sk, initial_charges=[0.1, -0.3, 0.05, 0.05, 0.05, 0.05]), labels, pos)
    assert np.allclose(q, z - [0.1, -0.3, 0.05, 0.05, 0.05, 0.05], atol=1e-14)
    with pytest.warns(RuntimeWarning, match="sum to 0.1, not the total charge 0"):
        q = _start(DFTB(sk, initial_charges={0: 0.3, 1: -0.1, 2: -0.1}), labels, pos)
    raw = z - [0.3, -0.1, -0.1, 0.0, 0.0, 0.0]
    assert np.allclose(q, raw * z.sum() / raw.sum(), atol=1e-14)


def test_collinear_initial_spins_follow_the_shell_populations():
    case, sk, labels, pos, _, _, opts = _system("o2_triplet_shell")
    q = _start(DFTB(sk, initial_spins=[0.5, 1.5], **opts), labels, pos).reshape(2, -1)
    m, n = q[0] - q[1], q[0] + q[1]
    assert np.allclose(n, [2.0, 4.0, 2.0, 4.0])                  # O s2 p4
    assert np.allclose(m, [0.5 * 2 / 6, 0.5 * 4 / 6, 1.5 * 2 / 6, 1.5 * 4 / 6])
    # without initial spins: m = q (N_up - N_down) / N on every shell, as before M6f
    q = _start(DFTB(sk, **opts), labels, pos).reshape(2, -1)
    assert np.allclose(q[0] - q[1], np.array([2.0, 4.0, 2.0, 4.0]) * 2 / 12)


@pytest.mark.parametrize("name, start", [
    ("h2co_casida", {"initial_charges": [0.4, -0.4, 0.1, -0.1]}),
    ("o2_triplet", {"initial_spins": {0: 1.5}}),
    ("oh_radical", {"initial_charges": [-0.3, 0.3], "initial_spins": [0.8, 0.2]}),
    ("cu4co_plusu", {"initial_charges": [0.2, -0.2, 0.1, -0.1, 0.3, -0.3]}),
    ("oh_noncolinear_tilted", {"initial_charges": [-0.2, 0.2]})])
def test_the_start_does_not_change_the_solution(name, start):
    from pydftb.validation import refgen
    case, sk, labels, pos, cell, kpts, opts = _system(name)
    opts = {**opts, **refgen.pauli_options(case, labels), **refgen.block_options(case, labels)}
    base = dict(smearing=tuple(case.filling), scc_tol=1e-12, max_scc=500, **opts)
    ref = DFTB(sk, **base).calculate(labels, pos)
    r = DFTB(sk, **base, **start).calculate(labels, pos)
    assert r.n_scc != ref.n_scc
    assert r.energy == pytest.approx(ref.energy, abs=1e-10)
    assert np.allclose(r.forces, ref.forces, atol=1e-8)
    if ref.magmoms is not None:
        assert np.allclose(r.magmoms, ref.magmoms, atol=1e-8)


def test_option_errors():
    _, sk, labels, pos, _, _, opts = _system("o2_triplet")
    with pytest.raises(ValueError, match="initial_spins needs spin polarisation"):
        DFTB(sk, initial_spins=[1.0, 1.0])
    with pytest.raises(ValueError, match="initial_charges needs SCC"):
        DFTB(sk, scc=False, initial_charges=[0.0, 0.0])
    with pytest.raises(ValueError, match="one value per atom"):
        DFTB(sk, initial_charges=[0.1, -0.1, 0.0]).calculate(labels, pos)
    with pytest.raises(ValueError, match="no atom 2"):
        DFTB(sk, initial_spins={2: 1.0}, **opts).calculate(labels, pos)


def test_ase_initial_magnetic_moments():
    o2 = Atoms("O2", positions=[[0, 0, 0], [0.05, 0.1, 1.21]])
    w = {"O": -0.028}
    o2.set_initial_magnetic_moments([1.5, 0.0])
    o2.calc = PyDFTB(SKF_HCOCU, spin=True, unpaired_electrons=2, spin_constants=w,
                     scc_tol=1e-10)
    o2.get_potential_energy()
    n_moments = o2.calc.raw.n_scc
    o2.calc = PyDFTB(SKF_HCOCU, spin=True, unpaired_electrons=2, spin_constants=w,
                     scc_tol=1e-10, initial_spins={0: 1.5})
    o2.get_potential_energy()
    assert o2.calc.raw.n_scc == n_moments
    o2.set_initial_magnetic_moments([0.0, 0.0])                # all zero: the default start
    o2.calc = PyDFTB(SKF_HCOCU, spin=True, unpaired_electrons=2, spin_constants=w,
                     scc_tol=1e-10)
    o2.get_potential_energy()
    assert o2.calc.raw.n_scc != n_moments
    assert o2.get_magnetic_moments() == pytest.approx([1.0, 1.0], abs=1e-8)


def test_torch_and_autodiff_refuse():
    pytest.importorskip("torch")
    from pydftb.autodiff import evaluate
    _, sk, labels, pos, _, _, opts = _system("o2_triplet")
    with pytest.raises(ValueError, match="initial_charges: implemented for the NumPy"):
        DFTB(sk, initial_charges=[0.1, -0.1], backend="torch")
    with pytest.raises(ValueError, match="collinear initial_spins: implemented for the NumPy"):
        DFTB(sk, initial_spins=[1.0, 1.0], backend="torch", **opts)
    with pytest.raises(NotImplementedError, match="initial_charges: NumPy engine only"):
        evaluate(DFTB(sk, initial_charges=[0.1, -0.1]), labels, pos)
    o2 = Atoms("O2", positions=[[0, 0, 0], [0.05, 0.1, 1.21]])
    o2.set_initial_magnetic_moments([1.5, 0.0])                # ignored by torch, as before
    o2.calc = PyDFTB(SKF_HCOCU, backend="torch", **opts)
    assert o2.get_magnetic_moments() == pytest.approx([1.0, 1.0], abs=1e-6)
