"""Analysis outputs of milestone M6f-4a: atom-resolved energies, net populations, CM5 charges,
the electrostatic potential, eigenvectors and DFTB+'s files.

The results against DFTB+ are the frozen references *_atom, *_analysis, h2o_pc_esp in
test_reference.py; here the rules themselves: every term per atom sums to its result field,
the split does not depend on where the molecule is, the potential's internal consistency, the
files read back, and the refusals.
"""
import importlib.util
import os
import struct
import warnings

import numpy as np
import pytest

from pydftb import Analysis, dftbplus_io as io
from pydftb.analysis import ENERGY_TERMS, electrostatic_potential, esp_points, orbital_labels
from pydftb.ase_calc import PyDFTB
from pydftb.calculator import DFTB
from pydftb.constants import BOHR, HARTREE
from pydftb.validation import refgen

from .conftest import SKF_HCOCU

FIELDS = {"h0": "e_band", "scc": "e_scc", "third_order": "e_3rd", "spin": "e_spin",
          "spin_orbit": "e_ls", "plus_u": "e_plus_u", "onsite": "e_onsite", "external": "e_ext",
          "solvation": "e_solv", "repulsive": "e_rep", "dispersion": "e_disp",
          "halogen": "e_halogen", "total": "energy"}
ALL = {"atom_energies": True, "net_populations": True, "cm5": True, "eigenvectors": True}
needs_d3 = pytest.mark.skipif(importlib.util.find_spec("dftd3") is None,
                              reason="optional package dftd3 not installed")


def _case(name, analysis=ALL, **options):
    case = refgen.CASES_BY_NAME[name]
    inp = refgen._inputs(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # periodic halogen, saw-tooth
        res, calc = refgen.run_pydftb(case, inp, engine=True, analysis=analysis, **options)
    pos = np.array(inp["positions_angstrom"]) / BOHR
    cell = None if inp["cell_angstrom"] is None else np.array(inp["cell_angstrom"]) / BOHR
    return res, calc, inp["labels"], pos, cell


@pytest.mark.parametrize("name", ["ch3oh_dftb3", "o2_triplet", "oh_noncolinear_soc",
                                  "oh_noncolinear_soc_onsite", "cu4co_plusu_ons",
                                  "cuo_chain_spin_plusu_k422", "ch3oh_dftb3_field_site",
                                  "h2o_pc", "h2o_gbsa", "meoh_water_halogen",
                                  "water_slab_dipole", "water_box_uff",
                                  pytest.param("water_dimer_d3_hhrep", marks=needs_d3),
                                  "h2o_dftb3_onsite", "water_box_sk_hybrid"])
def test_terms_sum_to_the_result_fields(name):
    res, *_ = _case(name, analysis={"atom_energies": True})
    a = res.analysis.atom_energies
    for term, field in FIELDS.items():
        value = getattr(res, field)
        assert a[term].sum() == pytest.approx(0.0 if value is None else value, abs=1e-11), term
    electronic = sum(a[k] for k in ENERGY_TERMS)
    np.testing.assert_allclose(a["electronic"], electronic, atol=1e-14)
    np.testing.assert_allclose(a["total"], electronic + a["repulsive"] + a["dispersion"]
                               + a["halogen"], atol=1e-14)


def test_split_is_invariant_under_rigid_motion():
    res, calc, labels, pos, _ = _case("ch3oh", analysis={"atom_energies": True, "cm5": True,
                                                         "net_populations": True})
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    moved = calc.calculate(labels, pos @ q.T + [1.0, -2.0, 0.5])
    for k, v in res.analysis.atom_energies.items():
        np.testing.assert_allclose(moved.analysis.atom_energies[k], v, atol=1e-9, err_msg=k)
    np.testing.assert_allclose(moved.analysis.cm5_charges, res.analysis.cm5_charges, atol=1e-9)
    np.testing.assert_allclose(moved.analysis.net_populations, res.analysis.net_populations,
                               atol=1e-9)


@needs_d3
def test_dftbplus_d3_compat_spreads_the_energy_evenly(monkeypatch):
    res, *_ = _case("ch3oh_d3bj", analysis={"atom_energies": True})
    pair = res.analysis.atom_energies["dispersion"]
    assert np.ptp(pair) > 1e-5                   # resolved by atom pairs
    monkeypatch.setattr(DFTB, "_atom_energy_compat", True)
    res2, *_ = _case("ch3oh_d3bj", analysis={"atom_energies": True})
    even = res2.analysis.atom_energies["dispersion"]
    np.testing.assert_allclose(even, res.e_disp / len(even), rtol=0, atol=1e-15)
    assert even.sum() == pytest.approx(pair.sum(), abs=1e-14)


def test_net_populations_and_hybridisation():
    res, *_ = _case("ch3oh")
    a = res.analysis
    q0 = np.array([4.0, 6.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(a.hybridisation, 1.0 - a.net_populations / q0, atol=1e-15)
    assert np.all(a.net_populations > 0.0)


def test_cm5_skips_pairs_of_one_type_and_counts_images():
    """CM5 = q0 - q plus pair terms: the corrections of all atoms sum to zero for every pair
    of types (D_AB = -D_BA), with periodic images too."""
    for name in ("ch3oh", "water_box_gamma"):
        res, *_ = _case(name)
        a = res.analysis
        mulliken = res.charges
        q0 = np.array([{"C": 4.0, "O": 6.0, "H": 1.0}[x] for x in
                       refgen._inputs(refgen.CASES_BY_NAME[name])["labels"]])
        corr = a.cm5_charges - (q0 - mulliken)
        assert abs(corr.sum()) < 1e-12
        assert np.abs(corr).max() > 1e-3


def test_esp_molecule_is_the_direct_sum():
    res, calc, labels, pos, _ = _case("ch3oh", analysis={"esp": {"points": [[0.0, 0.0, 5.0],
                                                                           [3.0, 1.0, 0.0]]}})
    q = (np.array([4.0, 6.0, 1.0, 1.0, 1.0, 1.0]) - res.charges)
    pts = res.analysis.esp.points
    direct = (q[None, :] / np.sqrt(((pts[:, None] - pos[None]) ** 2).sum(-1) + 1e-12)).sum(1)
    np.testing.assert_allclose(res.analysis.esp.internal, direct, atol=1e-15)
    assert res.analysis.esp.external is None


def test_esp_periodic_supercell_and_grid():
    """The Ewald potential at a point is the same for the cell and a doubled cell; a grid
    equals its points given one by one, and a fractional grid the equivalent Cartesian one."""
    res, calc, labels, pos, cell = _case("water_box_gamma", analysis={})
    st_res, st, sc = calc._run(labels, pos, cell, np.ones(3, bool), False, None, False)
    dq = calc._atom_dq(st, sc.q_out)
    pts = np.array([[1.0, 2.0, 3.0], [7.5, 0.3, 11.0], [0.0, 0.0, 0.0]])
    one = electrostatic_potential(pts, -dq, pos, st.coulomb)
    big = np.vstack([pos, pos + cell[0]])
    cell2 = cell.copy()
    cell2[0] *= 2.0
    from pydftb.coulomb import Coulomb
    coul2 = Coulomb(big, np.arange(len(big)), np.ones(len(big)), cell2, np.ones(3, bool))
    two = electrostatic_potential(pts, np.concatenate([-dq, -dq]), big, coul2)
    np.testing.assert_allclose(one, two, atol=1e-9)
    spec = {"grid": {"spacing": [1.0, 2.0, 0.5], "origin": [0.1, 0.2, 0.3], "points": [2, 3, 2]}}
    gpts, (shape, origin, axes) = esp_points(spec, cell, True)
    assert gpts.shape == (12, 3) and shape == (2, 3, 2)
    np.testing.assert_allclose(gpts[1], origin + axes[2])          # the last index fastest
    np.testing.assert_allclose(gpts[2], origin + axes[1])
    frac = {"fractional": True, "grid": {"spacing": [0.25, 0.5, 0.5], "origin": [0.0, 0.0, 0.0],
                                         "points": [2, 2, 2]}}
    fpts, (_, _, faxes) = esp_points(frac, cell, True)
    np.testing.assert_allclose(fpts[-1], 0.25 * cell[0] + 0.5 * cell[1] + 0.5 * cell[2])
    np.testing.assert_allclose(faxes, cell * np.array([0.25, 0.5, 0.5])[:, None])
    rot = {"grid": {"spacing": [1.0, 1.0, 1.0], "origin": [1.0, 0.0, 0.0], "points": [2, 1, 1],
                    "directions": [[0, 1, 0], [-1, 0, 0], [0, 0, 2]]}}
    rpts, (_, rorigin, raxes) = esp_points(rot, None, False)
    np.testing.assert_allclose(rorigin, [0.0, 1.0, 0.0])          # the origin rotated too
    np.testing.assert_allclose(raxes[0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(rpts[1], [0.0, 2.0, 0.0])


def test_esp_point_on_an_atom_leaves_out_its_own_term():
    _, calc, labels, pos, cell = _case("water_box_gamma", analysis={})
    _, st, sc = calc._run(labels, pos, cell, np.ones(3, bool), False, None, False)
    dq = calc._atom_dq(st, sc.q_out)
    on = electrostatic_potential(pos[:1], -dq, pos, st.coulomb)
    image = electrostatic_potential(pos[:1] + cell[1], -dq, pos, st.coulomb)
    near = electrostatic_potential(pos[:1] + [1e-3, 0, 0], -dq, pos, st.coulomb)
    np.testing.assert_allclose(image, on, atol=1e-10)              # images treated alike
    # the regular part: near the atom the potential is its own q/r plus the rest
    assert near[0] - (-dq[0]) / 1e-3 == pytest.approx(on[0], abs=1e-2)


def test_esp_external_column_and_dipole():
    res, *_ = _case("h2o_pc", analysis={"esp": {"points": [[0.0, 0.0, 4.0]]}})
    esp = res.analysis.esp
    case = refgen.CASES_BY_NAME["h2o_pc"]
    pc = np.array(case.point_charges["positions"]) / BOHR
    qpc = np.array(case.point_charges["charges"])
    expected = (qpc / np.sqrt(((esp.points[0] - pc) ** 2).sum(1) + 1e-12)).sum()
    assert esp.external[0] == pytest.approx(expected, abs=1e-14)
    res, *_ = _case("h2o_field", analysis={"esp": {"points": [[1.0, 2.0, 3.0]]}})
    field = res.analysis.esp
    case = refgen.CASES_BY_NAME["h2o_field"]
    assert field.external is not None and np.isfinite(field.external[0])
    # dipole correction: the internal column changes by -V_dc(point)
    res, calc, labels, pos, cell = _case("water_slab_dipole",
                                         analysis={"esp": {"points": [[1.0, 1.0, 9.0]]}})
    _, st, sc = calc._run(labels, pos, cell, np.ones(3, bool), False, None, False)
    dq = calc._atom_dq(st, sc.q_out)
    plain = electrostatic_potential(res.analysis.esp.points, -dq, pos, st.coulomb)
    vdc = st.dipole_corr.point_potential(res.analysis.esp.points, dq)
    np.testing.assert_allclose(res.analysis.esp.internal, plain - vdc, atol=1e-14)
    assert abs(vdc[0]) > 1e-4


def test_files_read_back(tmp_path):
    res, *_ = _case("cuo_chain_k422", analysis={**ALL, "esp": {
        "grid": {"spacing": [1.0, 1.0, 1.0], "origin": [0.0, 0.0, 0.0], "points": [2, 2, 1]}}})
    a = res.analysis
    io.write_band_out(tmp_path / "band.out", res)
    blocks = io.read_band_out(tmp_path / "band.out")
    assert len(blocks) == len(res.eigenvalues)
    for b, e, f in zip(blocks, res.eigenvalues, res.occupations):
        np.testing.assert_allclose(b["energies"], np.asarray(e) * HARTREE, atol=5e-5)
        np.testing.assert_allclose(b["occupations"], np.ravel(f), atol=5e-6)
    io.write_eigenvectors(tmp_path / "eigenvec.out", res)
    kind, vecs = io.read_eigenvec_out(tmp_path / "eigenvec.out")
    assert kind == "complex" and len(vecs) == sum(len(e) for e in res.eigenvalues)
    np.testing.assert_allclose(vecs[0][1], a.mulliken_weights[0][:, 0], atol=5e-7)
    io.write_eigenvectors(tmp_path / "eigenvec.bin", res, run_id=7)
    data = (tmp_path / "eigenvec.bin").read_bytes()
    assert struct.unpack("<i", data[:4]) == (7,)
    first = np.frombuffer(data[4:4 + 16 * a.eigenvectors[0].shape[0]], "<c16")
    np.testing.assert_array_equal(first, a.eigenvectors[0][:, 0])
    io.write_esp(tmp_path / "ESP.dat", a.esp)
    back = io.read_esp(tmp_path / "ESP.dat")
    assert back["shape"] == (2, 2, 1) and back["n"] == 4 and not back["external"]
    np.testing.assert_allclose(back["internal"], a.esp.internal * HARTREE, rtol=1e-11)
    np.testing.assert_allclose(back["axes"], a.esp.axes * BOHR, rtol=1e-11)
    io.write_esp(tmp_path / "ESP.dat", a.esp, geo_step=1, append=True)
    text = (tmp_path / "ESP.dat").read_text()
    assert text.count("#  F") == 1 and "Geo 1" in text
    tables = io.detailed_tables(res)
    assert "Atom resolved total energies" in tables and "CM5 corrected" in tables
    eigen_sums = [w.sum(0) for w in a.mulliken_weights]       # each state holds one electron
    np.testing.assert_allclose(np.concatenate(eigen_sums), 1.0, atol=1e-12)


def test_eigenvectors_are_those_of_the_last_hamiltonian():
    for name in ("o2_triplet", "oh_noncolinear_soc", "water_box_gamma"):
        res, calc, *_ = _case(name, analysis={"eigenvectors": True})
        a = res.analysis
        assert a.eigenvector_kind == {"o2_triplet": "real", "oh_noncolinear_soc": "pauli",
                                      "water_box_gamma": "real"}[name]
        weights = a.mulliken_weights
        total = [np.asarray(w)[0].sum(0) if a.eigenvector_kind == "pauli" else
                 np.asarray(w).sum(0) for w in weights]
        np.testing.assert_allclose(np.concatenate(total), 1.0, atol=1e-12)
        occ_weighted = sum((np.asarray(w)[0] if a.eigenvector_kind == "pauli" else np.asarray(w))
                           @ np.ravel(f) for w, f in zip(weights, res.occupations))
        # the Mulliken populations of the last diagonalisation's density (the output charges)
        np.testing.assert_allclose(occ_weighted, res.q_orbital, atol=1e-10)


def test_orbital_labels():
    class B:
        species = ["Cu", "O"]
        shells = [[0, 1, 2], [0, 1, 0]]
    labels = orbital_labels(B)
    names = [x[3] for x in labels]
    assert names[:4] == ["s", "p_y", "p_z", "p_x"]
    assert names[4:9] == ["d_xy", "d_yz", "d_z2", "d_xz", "d_x2-y2"]
    assert names[-1] == "s2" and labels[-1][2] == 2


def test_default_has_no_analysis_and_options():
    res, *_ = _case("ch3oh", analysis=None)
    assert res.analysis is None
    assert Analysis.make(True) == Analysis(atom_energies=True, net_populations=True, cm5=True)
    assert Analysis.make({"cm5": {"alpha": 1.3}}).cm5_model.alpha == 1.3
    with pytest.raises(ValueError, match="either 'points' or 'grid'"):
        Analysis(esp={"points": [[0, 0, 0]], "grid": {}})
    with pytest.raises(ValueError, match="unknown keys"):
        Analysis.make({"atom_energy": True})
    with pytest.raises(ValueError, match="directions are not read"):
        Analysis(esp={"fractional": True, "grid": {"spacing": [1, 1, 1], "origin": [0, 0, 0],
                                                   "points": [1, 1, 1],
                                                   "directions": np.eye(3)}})
    with pytest.raises(ValueError, match="spacings must be non-zero"):
        Analysis(esp={"grid": {"spacing": [0, 1, 1], "origin": [0, 0, 0], "points": [2, 1, 1]}})
    with pytest.raises(ValueError, match="fractional coordinates need a periodic"):
        _case("ch3oh", analysis={"esp": {"fractional": True, "points": [[0, 0, 0]]}})


def test_refusals():
    with pytest.raises(ValueError, match="needs SCC"):
        _case("h2o_nonscc", analysis={"esp": {"points": [[0, 0, 3.0]]}})
    with pytest.raises(ValueError, match="Generalized Born"):
        _case("h2o_gbsa", analysis={"esp": {"points": [[0, 0, 3.0]]}})
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="analysis"):
        _case("ch3oh", analysis={"atom_energies": True}, backend="torch")


def test_xlbomd_refuses_atom_energies():
    from pydftb.validation.cases import CASES_BY_NAME
    case = CASES_BY_NAME["ch3oh"]
    inp = refgen._inputs(case)
    calc = DFTB(refgen.ParamSet(refgen.SKF_SETS[case.skf_set], max_l=inp["max_l"]).skset(
        inp["labels"]), scc_tol=1e-8, force_evaluation="dynamics_t0",
        analysis={"atom_energies": True})
    with pytest.raises(ValueError, match="XL-BOMD shadow energy"):
        calc._run(inp["labels"], np.array(inp["positions_angstrom"]) / BOHR, None, None, True,
                  None, False, xl=True)


def test_ase_potential_energies_and_units():
    from ase.build import molecule
    atoms = molecule("H2O")
    atoms.calc = PyDFTB(SKF_HCOCU, scc_tol=1e-10,
                        analysis={"atom_energies": True,
                                  "esp": {"points": [[0.0, 0.0, 3.0]], "softening": 1e-4}})
    e = atoms.get_potential_energy()
    assert atoms.get_potential_energies().sum() == pytest.approx(e, abs=1e-10)
    esp = atoms.calc.raw.analysis.esp
    np.testing.assert_allclose(esp.points, [[0.0, 0.0, 3.0 / BOHR]])
    atoms.calc = PyDFTB(SKF_HCOCU)
    from ase.calculators.calculator import PropertyNotImplementedError
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_potential_energies()
    assert os.path.isdir(SKF_HCOCU)
