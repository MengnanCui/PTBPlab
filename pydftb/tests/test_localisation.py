"""Analysis outputs of milestone M6f-4b: Pipek-Mezey localised orbitals, bond populations,
bond energies and Mayer bond orders, and Casida transition charges.

The results against DFTB+ are the frozen references *_pm, *_bonds and *_transq in
test_reference.py; here the rules themselves: the localised orbitals span the occupied space
and stay S-orthonormal, the functional does not decrease, the sum rules of the bond matrices,
the transition charges' sum and dipole, the files read back, and the refusals.
"""
import warnings

import numpy as np
import pytest

from pydftb import Analysis, dftbplus_io as io
from pydftb.analysis import pipek_mezey, pm_populations
from pydftb.calculator import DFTB
from pydftb.casida import Casida
from pydftb.constants import BOHR
from pydftb.validation import refgen

PM = {"localise": {"tolerance": 1e-10, "max_iterations": 10000}}
BONDS = {"bond_populations": True, "bond_energies": True, "bond_orders": True}
EXC2 = {"n_excitations": 6, "symmetry": "singlet", "state_of_interest": 2, "solver": "exact",
        "excited_forces": False, "transition_charges": True}
Q0 = {"C": 4.0, "O": 6.0, "H": 1.0}


def _case(name, analysis=None, **options):
    case = refgen.CASES_BY_NAME[name]
    inp = refgen._inputs(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res, calc = refgen.run_pydftb(case, inp, engine=True, analysis=analysis, **options)
    pos = np.array(inp["positions_angstrom"]) / BOHR
    cell = None if inp["cell_angstrom"] is None else np.array(inp["cell_angstrom"]) / BOHR
    return res, calc, inp["labels"], pos, cell


def _overlap(calc, labels, pos, cell=None):
    st = calc._setup(labels, pos, cell, None if cell is None else np.ones(3, bool), False, {})
    starts = np.searchsorted(np.asarray(st.basis.orb_atom), np.arange(len(labels)))
    return np.asarray(st.smat_precise)[0].real, starts


@pytest.mark.parametrize("name", ["ch3oh", "h2co_cation_casida", "water_box_gamma"])
def test_localised_orbitals_span_the_occupied_space(name):
    res, calc, labels, pos, cell = _case(name, analysis={**PM, "eigenvectors": True},
                                         casida=None)
    loc = res.analysis.localisation
    s, starts = _overlap(calc, labels, pos, cell)
    for c, c0, n, q, w in zip(loc.orbitals, res.analysis.eigenvectors, loc.n_localised,
                              loc.populations, loc.weights):
        c0 = np.asarray(c0).real
        np.testing.assert_allclose(c.T @ s @ c, np.eye(len(c)), atol=1e-12)
        np.testing.assert_allclose(c[:, :n] @ c[:, :n].T, c0[:, :n] @ c0[:, :n].T, atol=1e-12)
        np.testing.assert_array_equal(c[:, n:], c0[:, n:])           # virtual ones unchanged
        np.testing.assert_allclose(q, pm_populations(c[:, :n], s, starts), atol=1e-14)
        np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(w, c * (s @ c), atol=1e-14)
    assert all(f >= i for i, f in zip(loc.initial, loc.final))
    assert all(loc.converged)
    assert loc.value == loc.final[-1]
    nel = [round(float(np.sum(f))) for f in res.occupations]
    assert loc.n_localised == ([nel[0] // 2] if len(nel) == 1 else nel)


def test_pipek_mezey_is_sign_equivariant():
    """Flipping the sign of an input orbital flips that of its output only: A is even and B
    odd in each orbital's sign (to the precision of the acos angles, ~1e-8)."""
    res, calc, labels, pos, _ = _case("ch3oh", analysis={"eigenvectors": True})
    s, starts = _overlap(calc, labels, pos)
    c = np.asarray(res.analysis.eigenvectors[0]).real[:, :7]
    flip = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0])
    a, _, conv_a = pipek_mezey(c, s, starts, 1e-10, 10000)
    b, _, conv_b = pipek_mezey(c * flip, s, starts, 1e-10, 10000)
    assert conv_a and conv_b
    np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-7)
    np.testing.assert_allclose(pm_populations(a, s, starts), pm_populations(b, s, starts),
                               atol=1e-7)


def test_localisation_defaults_warnings_and_refusals():
    res, *_ = _case("ch3oh", analysis={"localise": True})
    assert res.analysis.eigenvectors is None              # kept only when asked for
    assert 3 <= res.analysis.localisation.sweeps[0] <= 100
    case = refgen.CASES_BY_NAME["h2co_casida_3000K"]
    with pytest.warns(RuntimeWarning, match="Fractional occupations"):
        refgen.run_pydftb(case, refgen._inputs(case), analysis=PM, casida=None)
    for name, match in (("cuo_chain_k422", "real eigenvectors"),
                        ("oh_noncolinear_soc", "non-collinear")):
        with pytest.raises(ValueError, match=match):
            _case(name, analysis=PM)
    for spec, err in (({"localise": {"sweeps": 3}}, ValueError),
                      ({"localise": {"max_iterations": 0}}, ValueError),
                      ({"localise": {"tolerance": -1.0}}, ValueError),
                      ({"localise": "yes"}, TypeError)):
        with pytest.raises(err):
            Analysis.make(spec)


def test_localised_orbital_files(tmp_path):
    res, *_ = _case("ch3oh", analysis=PM)
    loc = res.analysis.localisation
    io.write_eigenvectors(tmp_path / "localOrbs.out", res, localised=True)
    kind, vecs = io.read_eigenvec_out(tmp_path / "localOrbs.out")
    assert kind == "real" and len(vecs) == loc.orbitals[0].shape[1]
    np.testing.assert_allclose([w for _, w in vecs], loc.weights[0].T, atol=5e-7)
    io.write_eigenvectors(tmp_path / "localOrbs.bin", res, localised=True, run_id=7)
    raw = (tmp_path / "localOrbs.bin").read_bytes()
    assert np.frombuffer(raw[:4], "<i4")[0] == 7
    c = np.frombuffer(raw[4:], "<f8").reshape(-1, loc.orbitals[0].shape[0]).T
    np.testing.assert_array_equal(c, loc.orbitals[0])
    plain, *_ = _case("ch3oh", analysis={"eigenvectors": True})
    with pytest.raises(ValueError, match="localisation"):
        io.write_eigenvectors(tmp_path / "x.out", plain, localised=True)


@pytest.mark.parametrize("name", ["ch3oh", "o2_triplet", "oh_noncolinear_soc"])
def test_bond_matrices_sum_rules_molecules(name):
    res, *_ = _case(name, analysis=BONDS if name == "ch3oh" else
                    {"bond_populations": True, "bond_energies": True})
    a = res.analysis
    populations = res.charges                         # Mulliken populations (electrons)
    np.testing.assert_allclose(a.bond_populations, a.bond_populations.T, atol=1e-14)
    np.testing.assert_allclose(a.bond_populations.sum(axis=1), populations, atol=1e-10)
    np.testing.assert_allclose(a.bond_energies, a.bond_energies.T, atol=1e-14)
    assert a.bond_energies.sum() == pytest.approx(res.e_band, abs=1e-11)
    if name == "ch3oh":           # closed shell, integer occupations: PSPS = 2 PS
        np.testing.assert_allclose(a.bond_orders.sum(axis=1), 2.0 * populations, atol=1e-10)
        assert 0.9 < a.bond_orders[0, 1] < 1.0                     # C-O
    else:
        assert a.bond_orders is None


def test_bond_populations_periodic_supercell():
    """A cell with the k-points 0 and 1/2 along a equals the doubled cell at Gamma: the bond
    populations of an atom with the two copies of another sum to the cell's."""
    res, calc, labels, pos, cell = _case("water_box_gamma")
    n = len(labels)
    opts = dict(scc_tol=1e-12, max_scc=500,
                analysis={"bond_populations": True, "bond_energies": True})
    one = DFTB(calc.sk, kpts=(np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]), np.ones(2)), **opts)
    r1 = one.calculate(labels, pos, cell, np.ones(3, bool))
    cell2 = cell.copy()
    cell2[0] *= 2.0
    r2 = DFTB(calc.sk, **opts).calculate(list(labels) * 2, np.vstack([pos, pos + cell[0]]),
                                         cell2, np.ones(3, bool))
    for key in ("bond_populations", "bond_energies"):
        b1, b2 = getattr(r1.analysis, key), getattr(r2.analysis, key)
        np.testing.assert_allclose(b2[:n, :n] + b2[:n, n:], b1, atol=1e-9, err_msg=key)
    assert r1.analysis.bond_populations.sum() == pytest.approx(
        sum(Q0[x] for x in labels), abs=1e-9)
    with pytest.raises(ValueError, match="Mayer"):
        _case("water_box_gamma", analysis={"bond_orders": True})


def test_bond_files(tmp_path):
    res, *_ = _case("ch3oh", analysis=BONDS)
    names = io.write_bond_files(res, tmp_path)
    assert names == ["bond_pop.dat", "bond_energy.dat", "bond_order.dat"]
    lines = (tmp_path / "bond_pop.dat").read_text().splitlines()
    assert lines[0].startswith("# Mulliken bond populations") and lines[1] == "# nAtom = 6"
    assert len(lines[2]) == 6 * 17 - 1                     # ES16.8 and a blank between
    np.testing.assert_allclose(io.read_bond_matrix(tmp_path / "bond_pop.dat"),
                               res.analysis.bond_populations, rtol=1e-8, atol=0)
    with pytest.raises(ValueError, match="bond analysis"):
        io.write_bond_files(_case("ch3oh")[0], tmp_path)


def test_transition_charges():
    res, _, _, pos, _ = _case("h2co_exc_forces", casida=EXC2)
    exc = res.excitations
    q = exc.transition_charges
    assert abs(q.sum()) < 1e-12
    np.testing.assert_allclose(q @ pos, exc.transition_dipoles[1], atol=1e-12)
    bright, *_ = _case("h2co_exc_forces", casida={**EXC2, "state_of_interest": -1})
    x = bright.excitations
    assert x.state == int(np.argmax(x.oscillator_strengths)) + 1
    assert np.all(np.isfinite(x.transition_charges))          # DFTB+ writes NaN here
    plain, *_ = _case("h2co_exc_forces", casida={**EXC2, "transition_charges": False})
    assert plain.excitations.transition_charges is None
    for bad in ({"state_of_interest": 0}, {"state_of_interest": 1, "symmetry": "triplet"},
                {"state_of_interest": 1, "symmetry": "both"}):
        with pytest.raises(ValueError, match="transition charges"):
            Casida(6, transition_charges=True, **{"symmetry": "singlet", **bad})


def test_transition_charge_file(tmp_path):
    res, *_ = _case("h2co_exc_forces", casida=EXC2)
    q = res.excitations.transition_charges
    io.write_transition_charges(tmp_path / "ATQ.DAT", res)
    text = (tmp_path / "ATQ.DAT").read_text().splitlines()
    assert text[:3] == ["#", "#  atom     transition charge", "#"]
    assert text[3] == f"  {1:5d}     {q[0]:12.9f}"
    np.testing.assert_allclose(io.read_transition_charges(tmp_path / "ATQ.DAT"), q, atol=5e-10)
    with pytest.raises(ValueError, match="transition charges"):
        io.write_transition_charges(tmp_path / "x", _case("ch3oh")[0])
