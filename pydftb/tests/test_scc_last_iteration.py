"""An SCC stopped at ``max_scc`` (DFTB+'s MaxSCCIterations with ConvergentSCCOnly = No).

DFTB+ mixes only while the SCC goes on (main.F90 getNextInputCharges: not after the last
iteration, converged or not), so its charges.bin, and the charges its own drivers carry to the
next geometry, are the input populations of the last Hamiltonian.  pydftb's ``DFTBResult.q_in``
is that vector too, and ``PyDFTB(reuse_charges=True)`` carries it.  Against DFTB+ itself: the
frozen references *_scc3_read_charges (run 1 stops after 3 iterations; its charges.bin against
pydftb's q_in in test_reference.py and test_charges_file.py) and, with dftb+, an MD of capped
SCCs on both engines (test_dual_engine.py).
"""
import numpy as np
import pytest
from ase.build import molecule

from pydftb import calculator
from pydftb.ase_calc import PyDFTB
from pydftb.calculator import DFTB
from pydftb.validation import refgen

from .conftest import SKF_HCOCU
from .test_spin import _system

# (case, mixer calls in 3 iterations): with spin, +U and two-component spin DFTB+ takes the
# first output unmixed (main.F90), which leaves one mixing step
SYSTEMS = [("h2o", 2), ("oh_radical", 1), ("cu4co_plusu", 1), ("oh_noncolinear_tilted", 1)]
NAMES = [name for name, _ in SYSTEMS]


def _engine(name, **options):
    """The case's DFTB engine, labels, positions and cell (Bohr) and pbc."""
    case, sk, labels, pos, cell, kpts, opts = _system(name)
    opts = {**opts, **refgen.pauli_options(case, labels), **refgen.block_options(case, labels)}
    calc = DFTB(sk, smearing=tuple(case.filling), kpts=kpts, scc_tol=1e-10, **opts, **options)
    return calc, (labels, pos, cell, None if cell is None else np.ones(3, bool))


def _record(monkeypatch):
    """The populations every Hamiltonian of the SCC is built from (DFTB._potentials, called once
    per iteration when no forces are asked for) and the number of mixing steps."""
    inputs, mixes = [], []
    potentials, make_mixer = DFTB._potentials, calculator.make_mixer

    def recorded(self, st, q, *args, **kwargs):
        inputs.append(np.array(q, dtype=float, copy=True))
        return potentials(self, st, q, *args, **kwargs)

    def counted(*args, **kwargs):
        mixer = make_mixer(*args, **kwargs)
        mix = mixer.mix

        def counted_mix(q_in, q_out):
            mixes.append(len(inputs))
            return mix(q_in, q_out)

        mixer.mix = counted_mix
        return mixer

    monkeypatch.setattr(DFTB, "_potentials", recorded)
    monkeypatch.setattr(calculator, "make_mixer", counted)
    return inputs, mixes


@pytest.mark.parametrize("name, n_mix", SYSTEMS)
def test_q_in_built_the_last_hamiltonian(name, n_mix, monkeypatch):
    """Three iterations, unconverged: q_in is the vector the third Hamiltonian was built from,
    not mixed again (DFTB+'s charges.bin); no mixing step after the third iteration."""
    calc, geo = _engine(name, max_scc=3, warn_unconverged=False)
    inputs, mixes = _record(monkeypatch)
    r = calc.calculate(*geo, forces=False)
    assert r.n_scc == 3 and not r.converged and len(inputs) == 3
    assert np.array_equal(r.q_in, inputs[2])
    assert len(mixes) == n_mix and max(mixes) < 3          # after iterations 1-2 only
    assert np.abs(r.q_out - r.q_in).max() == r.scc_errors[-1] > 1e-4     # it would matter


@pytest.mark.parametrize("name", NAMES)
def test_one_iteration_keeps_the_start(name, monkeypatch):
    """max_scc = 1: q_in is the start vector; the first output is not taken unmixed either
    (DFTB+ does that only when another iteration follows)."""
    calc, geo = _engine(name, max_scc=1, warn_unconverged=False)
    inputs, mixes = _record(monkeypatch)
    r = calc.calculate(*geo, forces=False)
    assert r.n_scc == 1 and len(inputs) == 1 and not mixes
    assert np.array_equal(r.q_in, inputs[0])
    assert not np.allclose(r.q_in, r.q_out)


@pytest.mark.parametrize("name", NAMES)
def test_restart_from_q_in_repeats_the_last_iteration(name):
    """Started from the q_in of a stopped SCC (reuse_charges, DFTB+'s charges.bin), the first
    iteration is that SCC's last one again: the same Hamiltonian, output, error and energy."""
    calc, geo = _engine(name, max_scc=3, warn_unconverged=False)
    r = calc.calculate(*geo, forces=False)
    once, _ = _engine(name, max_scc=1, warn_unconverged=False)
    again = once.calculate(*geo, forces=False, q_init=r.q_in)
    assert again.scc_errors[0] == r.scc_errors[-1]
    assert np.array_equal(again.q_out, r.q_out) and np.array_equal(again.q_in, r.q_in)
    assert again.energy == r.energy
    assert all(np.array_equal(a, b) for a, b in zip(again.eigenvalues, r.eigenvalues))


def test_unconverged_warning_reports_the_last_error():
    calc, geo = _engine("h2o", max_scc=3, convergent_scc_only=False)
    with pytest.warns(RuntimeWarning, match="SCC not converged after 3 iterations") as rec:
        r = calc.calculate(*geo, forces=False)
    assert r.scc_error == r.scc_errors[-1]
    assert f"(max |dq| = {r.scc_errors[-1]:.2e} > 1.0e-10)" in str(rec[0].message)


@pytest.mark.filterwarnings("ignore:SCC not converged")
def test_reused_charges_continue_from_the_last_hamiltonian(monkeypatch):
    """PyDFTB(reuse_charges=True) with a capped SCC: every geometry step starts from the
    populations of the step before's last Hamiltonian, as DFTB+'s drivers (they keep qInput)."""
    w = molecule("H2O")
    w.calc = PyDFTB(SKF_HCOCU, max_scc=3, scc_tol=1e-10, reuse_charges=True,
                    convergent_scc_only=False)
    inputs, _ = _record(monkeypatch)
    w.get_potential_energy()
    for step in range(1, 3):
        w.positions[0, 2] += 0.02
        w.get_potential_energy()
        assert len(inputs) == 3 * (step + 1) and w.calc.raw.n_scc == 3
        assert np.array_equal(inputs[3 * step], inputs[3 * step - 1])
