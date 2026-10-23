"""Casida linear-response excitations (milestone M6d).

The comparisons with DFTB+ are the frozen references *_casida* in test_reference.py; here
the solvers against each other, the Casida matrix (symmetric, its matrix-free product, the
normalisation of X + Y), the uncoupled limit, invariance under rotations and translations,
the printed tables (DFTB+'s EXC.DAT and SPX.DAT), the ASE and SCC interfaces and the errors.
Excited-state forces and charges (M6e) are tested in test_casida_forces.py.
"""
import os

import numpy as np
import pytest
from ase import units
from ase.build import molecule

import pydftb
from pydftb import casida as cs
from pydftb.ase_calc import PyDFTB
from pydftb.constants import BOHR
from pydftb.validation import dftbplus, refgen
from pydftb.validation.cases import CASES_BY_NAME, GB_WATER, UD_3OB, solvation_model

from .conftest import SKF_HCOCU


def _engine(name, **opts):
    case = CASES_BY_NAME[name]
    inp = refgen._inputs(case)
    res, calc = refgen.run_pydftb(case, inp, engine=True, scc_tol=1e-10, **opts)
    return res, calc, inp["labels"], np.array(inp["positions_angstrom"]) / BOHR


def _ground_state(name, **opts):
    _, calc, labels, pos = _engine(name, **opts)
    _, st, sc = calc._run(labels, pos, None, None, False, None, False)
    return calc._casida_ground_state(st, sc).prepare()


def _groups(omega2, tol=1e-7):
    """[start, stop) of the runs of degenerate Omega^2 (DFTB+'s tolerance)."""
    cut = np.flatnonzero(np.abs(np.diff(omega2)) > tol) + 1
    return list(zip(np.r_[0, cut], np.r_[cut, len(omega2)]))


@pytest.mark.parametrize("name,n", [("ch3oh_casida", 12), ("c6h6_casida", 8),
                                    ("h2co_cation_casida", 6)])
def test_solvers_agree(name, n):
    """Dense, Davidson and ARPACK: the same states, and per group of degenerate states the
    same summed oscillator strength (benzene: degenerate pairs; the cation spin polarised)."""
    xs = [_engine(name, casida=cs.Casida(n, solver=s))[0].excitations
          for s in ("exact", "davidson", "arpack")]
    groups = [x.degenerate_groups() for x in xs]
    for x, g in zip(xs[1:], groups[1:]):
        assert np.abs(x.omega2 - xs[0].omega2).max() < 1e-12
        assert np.abs(g["oscillator"] - groups[0]["oscillator"]).max() < 1e-9
        assert np.abs(g["transdip"] - groups[0]["transdip"]).max() < 1e-7


def test_davidson_eigenvectors():
    """Davidson's lowest 20 eigenpairs of benzene's Omega: the dense eigenvalues and, per
    group of degenerate states, the same subspace."""
    gs = _ground_state("c6h6_casida")
    pairs, w, sqrt_n = cs.single_particle_transitions(gs)
    resp = cs.Response(gs, pairs, w, sqrt_n, cs.transition_charges(gs, pairs), "S")
    ev, f = resp.solve(20, "exact")
    ev_d, f_d = resp.davidson(20)
    assert np.abs(ev_d - ev).max() < 1e-13
    for a, b in _groups(ev):
        cosines = np.linalg.svd(f[:, a:b].T @ f_d[:, a:b], compute_uv=False)
        assert np.abs(cosines - 1.0).max() < 1e-8


@pytest.mark.parametrize("name,sym", [("h2co_casida", "S"), ("h2co_casida_triplet", "T"),
                                      ("h2co_cation_casida", " "), ("h2co_casida_onsite", "S"),
                                      ("h2co_cation_casida_onsite", " ")])
def test_casida_matrix(name, sym):
    """Omega is symmetric and its matrix-free product is the matrix (singlets, triplets,
    spin polarisation, onsite corrections with and without it); the eigenvectors
    F = (X+Y) sqrt(Omega / w) are normalised."""
    gs = _ground_state(name)
    pairs, w, sqrt_n = cs.single_particle_transitions(gs)
    resp = cs.Response(gs, pairs, w, sqrt_n, cs.transition_charges(gs, pairs), sym)
    m = resp.matrix()
    assert np.array_equal(m, m.T)
    assert np.abs(resp.product(np.eye(len(w))) - m).max() < 1e-12 * np.abs(m).max()
    assert np.abs(resp.product(m[:, 0]) - m @ m[:, 0]).max() < 1e-12 * np.abs(m).max() ** 2
    x = _engine(name)[0].excitations
    norm = (x.xpy ** 2 * x.energies[None, :] / x.single_particle.energies[:, None]).sum(axis=0)
    assert np.abs(norm - 1.0).max() < 1e-12


def test_uncoupled_limit():
    """Without coupling (gamma = 0, no spin constants or onsite terms) the excitations are the
    single-particle transitions, and a closed shell's oscillator strengths twice theirs."""
    gs = _ground_state("h2co_casida")
    gs.gamma = np.zeros_like(gs.gamma)
    x = cs.excitations(cs.Casida(6, solver="exact"), gs)
    sp = x.single_particle
    assert np.abs(x.energies - sp.energies[:6]).max() < 1e-14
    assert np.abs(x.oscillator_strengths - 2.0 * sp.oscillator_strengths[:6]).max() < 1e-14
    assert np.array_equal(x.transitions, sp.pairs[:6]) and np.allclose(x.weights, 1.0)


def test_spin_polarised_closed_shell():
    """A closed shell run spin-polarised (no unpaired electrons): the excitations are the
    singlets and triplets together, D<S^2> = 0 for the singlets and 2 for the triplets."""
    w = {"H": -0.072, "C": -0.023, "O": -0.028}
    t, s = _engine("h2co_casida", casida=cs.Casida(12, symmetry="both"),
                   spin_constants=w)[0].excitations
    x = _engine("h2co_casida", casida=cs.Casida(12), spin=True, unpaired_electrons=0.0,
                spin_constants=w)[0].excitations
    both = np.concatenate([t.omega2, s.omega2])
    order = np.argsort(both)[:12]
    assert np.abs(x.omega2 - both[order]).max() < 1e-12
    assert np.abs(x.spin_contamination - np.where(order < 12, 2.0, 0.0)).max() < 1e-8


def test_rotation_and_translation_invariance():
    _, calc, labels, pos = _engine("h2co_casida")
    rot, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    x0 = calc.calculate(labels, pos, forces=False).excitations
    x1 = calc.calculate(labels, pos @ rot.T + [1.0, -2.0, 0.5], forces=False).excitations
    assert np.abs(x1.omega2 - x0.omega2).max() < 1e-10
    assert np.abs(x1.oscillator_strengths - x0.oscillator_strengths).max() < 1e-10
    # transition dipoles rotate with the molecule (each up to the sign of its state)
    proj = np.abs((x1.transition_dipoles * (x0.transition_dipoles @ rot.T)).sum(axis=1))
    assert np.abs(proj - (x0.transition_dipoles ** 2).sum(axis=1)).max() < 1e-9


@pytest.mark.parametrize("name", ["h2co_casida", "h2co_cation_casida", "c2h4_casida_both",
                                  "h2co_casida_owindow"])
def test_tables_as_dftbplus_prints_them(name, tmp_path):
    """EXC.DAT and SPX.DAT in DFTB+'s layout, read back by the parser of the references: the
    printed values of DFTB+ (molecules without degenerate levels; identical files)."""
    xs = _engine(name)[0].excitations
    xs = xs if isinstance(xs, list) else [xs]
    (tmp_path / "EXC.DAT").write_text("".join(x.table(header=k == 0) for k, x in enumerate(xs)))
    (tmp_path / "SPX.DAT").write_text(xs[-1].single_particle.table())
    out = dftbplus.parse_casida(os.fspath(tmp_path), "")
    ref = refgen.load(name)["reference"]
    assert out["exc_rows"] == ref["exc_rows"]
    assert out["spx_rows"] == ref["spx_rows"]


def test_states_of_interest_and_excited_forces():
    ground = _engine("h2co_casida")[0]
    exc = _engine("h2co_casida", casida=cs.Casida(6, state_of_interest=2))[0]   # M6e
    assert exc.energy == ground.energy                          # the ground state's
    assert exc.excited_energy == exc.energy + exc.excitation_energy
    assert np.abs(exc.forces - ground.forces).max() > 1e-2      # the excited state's
    res, *_ = _engine("h2co_casida", casida=cs.Casida(6, state_of_interest=2,
                                                       excited_forces=False))
    assert res.excitation_energy == res.excitations.energies[1]
    assert res.energy == ground.energy
    assert np.array_equal(res.forces, ground.forces) and res.excited_charges is None
    bright = _engine("h2co_casida", casida=cs.Casida(6, state_of_interest=-1,
                                                      excited_forces=False))[0]
    x = bright.excitations
    assert x.state == np.argmax(x.oscillator_strengths) + 1
    assert bright.excitation_energy == x.energies[x.state - 1]


def test_ase_and_scc_interfaces():
    a = molecule("H2CO")
    a.calc = PyDFTB(SKF_HCOCU, scc_tol=1e-10, casida={"n_excitations": 4})
    a.get_forces()                                   # ground-state forces
    x = a.calc.excitations
    assert len(x.energies) == 4 and np.all(x.energies > 0) and x.symmetries == ["S"] * 4
    scc = pydftb.SCC(pydftb.System.from_ase(a), pydftb.ParamSet(SKF_HCOCU), scc_tol=1e-10,
                     casida=cs.Casida(4))
    scc.kernel()
    assert np.abs(scc.result.excitations.omega2 - x.omega2).max() < 1e-12
    e0 = a.get_potential_energy()
    a.calc = PyDFTB(SKF_HCOCU, scc_tol=1e-10,
                    casida={"n_excitations": 4, "state_of_interest": 1})
    a.get_forces()                                   # the excited state's (M6e)
    w = a.calc.raw.excitation_energy * units.Hartree
    assert abs(a.get_potential_energy() - (e0 + w)) < 1e-8


def test_option_errors():
    with pytest.raises(ValueError, match="at least 1"):
        cs.Casida(0)
    with pytest.raises(ValueError, match="symmetry must be"):
        cs.Casida(2, symmetry="quintet")
    with pytest.raises(ValueError, match="one symmetry"):
        cs.Casida(2, symmetry="both", state_of_interest=1)
    with pytest.raises(ValueError, match="only for singlets"):
        cs.Casida(2, symmetry="triplet", state_of_interest=-1)
    with pytest.raises(ValueError, match="beyond"):
        cs.Casida(2, state_of_interest=3)
    with pytest.raises(ValueError, match="positive"):
        cs.Casida(2, energy_window=-0.1)
    with pytest.raises(ValueError, match="solver"):
        cs.Casida(2, solver="stratmann")
    with pytest.raises(TypeError):
        cs.Casida.make(4)


def test_unsupported_calculations():
    """DFTB+'s conditions for linear response (initprogram), and pydftb's own."""
    cas = cs.Casida(4)
    for opts, msg in ((dict(scc=False), "needs SCC"),
                      (dict(third_order="full", hubbard_derivs=UD_3OB), "third order"),
                      (dict(spin_orbit={"O": [0.0, 0.01]}), "spin-orbit"),
                      (dict(plus_u={"O": [("p", 0.1)]}), r"DFTB\+U"),
                      (dict(solvation=solvation_model(GB_WATER)), "implicit solvation"),
                      (dict(solver="sp2"), "SP2"),
                      (dict(casida=cs.Casida(4, symmetry="triplet")), "spin_constants")):
        with pytest.raises(ValueError, match=msg):
            _engine("h2co_casida", **{"casida": cas, **opts})
    sk = pydftb.ParamSet(SKF_HCOCU, shell_resolved=True).skset(["C", "O", "H"])
    with pytest.raises(ValueError, match="shell-resolved"):
        pydftb.DFTB(sk, casida=cas)
    with pytest.raises(ValueError, match="Gamma point"):
        _engine("h2co_box_casida", kpts=(np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
                                         np.array([0.5, 0.5])))
    with pytest.raises(ValueError, match="insufficient"):
        _engine("h2o_casida", casida=cs.Casida(30))


def test_torch_and_autodiff_refuse():
    pytest.importorskip("torch")
    from pydftb.autodiff import evaluate
    with pytest.raises(ValueError, match="NumPy backend only"):
        _engine("h2co_casida", backend="torch")
    _, calc, labels, pos = _engine("h2co_casida")
    with pytest.raises(NotImplementedError, match="milestone M6d"):
        evaluate(calc, labels, pos)
