"""Excited-state forces, charges and dipole (Z-vector method, milestone M6e).

The comparisons with DFTB+ are the frozen references *_exc_* in test_reference.py (forces,
XCH.DAT, XREST.DAT; where DFTB+'s forces are not -d(E0 + Omega)/dR, the differences of its
energies).  Here: the forces against differences of pydftb's own E0 + Omega (exact SK
derivatives), the excited dipole and point-charge forces as derivatives of the same energy,
the two Z-vector solvers, DFTB+'s gradient reproduced by leaving out what it leaves out, the
ASE interface (energy E0 + Omega for optimisers and MD) and the refusals.
"""
from dataclasses import replace

import numpy as np
import pytest
from ase import units
from ase.build import molecule
from ase.optimize import BFGS

import pydftb
from pydftb import casida_grad as cg
from pydftb.ase_calc import PyDFTB
from pydftb.calculator import DFTB
from pydftb.casida import Casida
from pydftb.constants import BOHR, HARTREE
from pydftb.validation import dftbplus, refgen
from pydftb.validation.cases import CASES_BY_NAME, EXC, FIELD, ONS

from .conftest import SKF_HCOCU

BASE = CASES_BY_NAME["h2co_exc_forces"]
W_SHELL = {"H": [[-0.072]], "C": [[-0.031, -0.025], [-0.025, -0.023]],
           "O": [[-0.035, -0.030], [-0.030, -0.028]]}
VARIANTS = {
    "singlet": BASE,
    "triplet": CASES_BY_NAME["h2co_exc_forces_triplet"],
    "spin": CASES_BY_NAME["h2co_cation_exc_forces"],
    "bright": CASES_BY_NAME["h2co_exc_forces_bright"],
    "field_pc": CASES_BY_NAME["h2co_exc_forces_field_pc"],
    "hdamp": CASES_BY_NAME["h2co_exc_forces_hdamp"],
    "ewindow": CASES_BY_NAME["h2co_exc_forces_ewindow"],
    "spin_window_field": replace(CASES_BY_NAME["h2co_cation_exc_forces"], efield=FIELD,
                                 casida={**EXC, "symmetry": None, "n": 4,
                                         "energy_window": 1.0 / HARTREE}),
    "sites": replace(BASE, site_potentials={"gross": {0: 0.5, 2: -0.3}}),
}


def _run(case, positions=None, **opts):
    inp = refgen._inputs(case)
    if positions is not None:
        inp = dict(inp, positions_angstrom=np.asarray(positions).tolist())
    return refgen.run_pydftb(case, inp, scc_tol=1e-13, **opts), inp


def _fd(energy, x0, idx, h):
    """-dE/dx at x0 for the flat components idx (4-point central differences)."""
    out = []
    for i in idx:
        e = {}
        for s in (-2, -1, 1, 2):
            x = x0.copy()
            x.flat[i] += s * h
            e[s] = energy(x)
        out.append(-(8.0 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12.0 * h))
    return np.array(out)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_forces_are_the_derivative_of_e0_plus_omega(name):
    """-d(E0 + Omega)/dR to ~1e-10 Ha/Bohr: singlets, triplets, spin polarisation, the
    brightest state, external field and point charges, H-X damping, an energy window (with
    the canonical-orbital terms DFTB+ leaves out), site potentials."""
    case = VARIANTS[name]
    res, inp = _run(case)
    pos = np.array(inp["positions_angstrom"])
    idx = [2, 3, 7, 11]                                  # O z, C x, H1 y, H2 z
    fd = _fd(lambda p: _run(case, p)[0].excited_free_energy, pos, idx, 1e-4 * BOHR) * BOHR
    assert np.abs(res.forces.flat[idx] - fd).max() < 2e-10


def _field_run(field):
    inp = refgen._inputs(BASE)
    labels, pos = inp["labels"], np.array(inp["positions_angstrom"]) / BOHR
    sk = pydftb.ParamSet(refgen.SKF_SETS[BASE.skf_set], max_l=inp["max_l"]).skset(labels)
    calc = DFTB(sk, scc_tol=1e-13, max_scc=500, electric_field=np.asarray(field, float),
                casida=Casida(6, state_of_interest=2))
    return calc.calculate(labels, pos)


def test_excited_dipole_is_the_field_derivative():
    """The relaxed density's dipole is -d(E0 + Omega)/dF (at a finite field F)."""
    f0 = np.array([0.002, 0.001, -0.003])
    res = _field_run(f0)
    fd = _fd(lambda f: _field_run(f).excited_free_energy, f0, range(3), 1e-4)
    assert np.abs(res.excited_dipole - fd).max() < 1e-8


def test_point_charge_forces_are_derivatives():
    """Forces on the point charges: -d(E0 + Omega)/dR_charge (DFTB+ prints the ground
    state's)."""
    case = CASES_BY_NAME["h2co_exc_forces_field_pc"]
    res, _ = _run(case)
    pcpos = np.array(case.point_charges["positions"])

    def energy(p):
        c = replace(case, point_charges={**case.point_charges, "positions": p.tolist()})
        return _run(c)[0].excited_free_energy

    fd = _fd(energy, pcpos, range(pcpos.size), 1e-4 * BOHR) * BOHR
    assert np.abs(res.point_charge_forces.ravel() - fd).max() < 2e-10


def test_dense_and_iterative_z_vector_agree(monkeypatch):
    """The Z vector by the dense solve and by preconditioned conjugate gradients (used above
    EXACT_LIMIT pairs), closed shell and spin polarised with a window."""
    for case in (BASE, VARIANTS["spin_window_field"]):
        a, _ = _run(case)
        monkeypatch.setattr(cg, "EXACT_LIMIT", 0)
        b, _ = _run(case)
        monkeypatch.undo()
        assert np.abs(a.forces - b.forces).max() < 1e-13
        assert np.abs(a.excited_charges - b.excited_charges).max() < 1e-13


def test_charges_dipole_and_invariance():
    """The relaxed density moves charge without changing the electron count; the forces sum
    to zero without a field and rotate with the molecule, as does the excited dipole."""
    res, inp = _run(BASE)
    assert abs(res.excited_charges.sum() - res.charges.sum()) < 1e-12
    assert np.abs(res.forces.sum(axis=0)).max() < 1e-12
    pos = np.array(inp["positions_angstrom"])
    rot, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1.0
    r2, _ = _run(BASE, pos @ rot.T + [0.3, -1.0, 2.0])
    assert np.abs(r2.forces - res.forces @ rot.T).max() < 1e-9
    assert np.abs(r2.excited_charges - res.excited_charges).max() < 1e-9
    assert np.abs(r2.excited_dipole - res.excited_dipole @ rot.T).max() < 1e-8


@pytest.mark.parametrize("name", ["h2co_exc_forces_field_pc", "h2co_exc_forces_hdamp",
                                  "h2co_exc_forces_ewindow"])
def test_dftbplus_gradient_reproduced(name):
    """Leaving out what DFTB+ leaves out (the external potentials, the damping of gamma', the
    pairs outside the window) and with its finite-difference SK derivatives, pydftb gives
    DFTB+'s forces: the differences from DFTB+ are exactly those terms."""
    doc = refgen.load(name)
    case = CASES_BY_NAME[name]
    DFTB._excited_compat = frozenset(case.casida["compat"])
    try:
        res = refgen.run_pydftb(case, doc["input"], fd_delta=2.0 ** -13)
    finally:
        DFTB._excited_compat = frozenset()
    assert np.abs(res.forces - np.array(doc["reference"]["forces"])).max() < 1e-9


def test_excited_charges_without_forces():
    """Casida(excited_charges=True): the relaxed density's charges and dipole (DFTB+
    WriteMulliken) with the ground state's forces."""
    exc, _ = _run(BASE)
    res, _ = _run(replace(BASE, casida={**EXC, "excited_forces": False,
                                        "excited_charges": True}))
    ground, _ = _run(replace(BASE, casida={**EXC, "excited_forces": False}))
    assert np.array_equal(res.forces, ground.forces) and ground.excited_charges is None
    assert np.abs(res.excited_charges - exc.excited_charges).max() < 1e-12
    assert np.abs(res.excited_dipole - exc.excited_dipole).max() < 1e-12


def test_ase_runs_in_the_excited_state():
    """PyDFTB with a state of interest: energy E0 + Omega with consistent forces, so an ASE
    optimiser lowers it (the state DFTB+'s drivers follow)."""
    a = molecule("H2CO")
    a.rattle(0.02, seed=3)
    a.calc = PyDFTB(SKF_HCOCU, scc_tol=1e-10, casida={"n_excitations": 4, "state_of_interest": 1})
    e = a.get_potential_energy()
    assert a.calc.excited
    assert abs(e - a.calc.raw.excited_energy * units.Hartree) < 1e-10
    f = a.get_forces()
    p0 = a.positions.copy()
    h = 1e-4
    for i, k in ((0, 2), (2, 1)):
        es = []
        for s in (1, -1):
            p = p0.copy()
            p[i, k] += s * h
            a.positions = p
            es.append(a.get_potential_energy())
        assert abs(f[i, k] + (es[0] - es[1]) / (2 * h)) < 1e-6
    a.positions = p0
    BFGS(a, logfile=None).run(fmax=0.05, steps=30)
    assert a.get_potential_energy() < e


@pytest.mark.parametrize("name", ["h2co_exc_forces", "h2co_cation_exc_forces"])
def test_mulliken_tables_as_dftbplus_prints_them(name, tmp_path):
    """XCH.DAT and XREST.DAT in DFTB+'s layout (h2co_exc_forces: byte-identical to DFTB+'s
    files), read back by the parser of the references."""
    case = CASES_BY_NAME[name]
    doc = refgen.load(name)
    res = refgen.run_pydftb(case, doc["input"])
    q0 = refgen._neutral_populations(case, doc["input"])
    sym = " " if case.spin else "S"
    xch, xrest = cg.mulliken_tables(q0 - res.excited_charges, res.excited_dipole, sym,
                                    res.excitations.state)
    (tmp_path / "XCH.DAT").write_text(xch)
    (tmp_path / "XREST.DAT").write_text(xrest)
    out = dftbplus.parse_casida(str(tmp_path), "")
    ref = doc["reference"]
    assert np.abs(out["excited_charges_net"] - ref["excited_charges_net"]).max() < 1.5e-8
    assert np.abs(out["excited_dipole_debye"] - ref["excited_dipole_debye"]).max() < 1.5e-8
    assert xch.startswith(f"# MULLIKEN CHARGES of excited state {sym} 2\n#  Natoms =   4\n")


def test_unsupported():
    """DFTB+'s conditions (periodic, fractional occupations, oscillator window, onsite
    corrections) and pydftb's (spin constants per shell)."""
    exc = {**EXC, "n": 4}
    for case, msg in (
            (replace(CASES_BY_NAME["h2co_box_casida"], casida=exc), "periodic"),
            (replace(BASE, casida={**exc, "oscillator_window": 0.01}), "oscillator window"),
            (replace(BASE, onsite=ONS, casida=exc), "onsite"),
            (replace(BASE, filling=("fermi", 3000.0), casida=exc), "integer occupations"),
            (replace(BASE, charge=1.0, spin={"unpaired_electrons": 1, "spin_constants": W_SHELL},
                     casida={**exc, "symmetry": None}), "one spin constant per type")):
        with pytest.raises(ValueError, match=msg):
            _run(case)
