"""Implicit solvation (milestone M6c): generalised Born (Still, P16, ALPB), SASA, the
hydrogen-bond correction, CM5, parameter files.

Comparisons with DFTB+ are the frozen references h2o_gbsa, ch3oh_gb*, ch3oh_alpb_p16,
oh_minus_alpb, ch3oh_sasa, ch3oh_gbsa_paramfile, h2o_gb_field_* and oh_noncolinear_soc_gb
in test_reference.py and h2o_gbsa_md in test_md_opt.py; here the derivatives of the
geometric parts, forces, limits and errors (no dftb+).
"""
import math
import os
import warnings
from dataclasses import replace

import numpy as np
import pytest
from ase.build import molecule

from pydftb import solvation as sv
from pydftb.ase_calc import PyDFTB
from pydftb.constants import BOHR, KB_HARTREE
from pydftb.validation import refgen
from pydftb.validation.cases import CASES_BY_NAME, GB_WATER, GBSA_WATER, solvation_model

from .conftest import DATA, SKF_HCOCU

GB = solvation_model(GB_WATER)
GBSA = solvation_model(GBSA_WATER)


def _methanol():
    inp = refgen._inputs(CASES_BY_NAME["ch3oh"])
    pos = np.array(inp["positions_angstrom"]) / BOHR
    labels = inp["labels"]
    numbers = [{"C": 6, "O": 8, "H": 1}[lab] for lab in labels]
    return labels, numbers, pos


def _engine(name, **opts):
    case = CASES_BY_NAME[name]
    inp = refgen._inputs(case)
    _, calc = refgen.run_pydftb(case, inp, engine=True, scc_tol=1e-12, **opts)
    return calc, inp["labels"], np.array(inp["positions_angstrom"]) / BOHR


def _numeric_derivative(f, pos, h=1e-5):
    """d f / d r_j (n_out, n_atom, 3) by central differences."""
    out = []
    for j in range(len(pos)):
        for k in range(3):
            p, m = pos.copy(), pos.copy()
            p[j, k] += h
            m[j, k] -= h
            out.append((f(p) - f(m)) / (2 * h))
    return np.array(out).T.reshape(-1, len(pos), 3)


def test_lebedev_grids():
    for n in (6, 26, 230, 5810):
        x, w = sv.lebedev_grid(n)
        assert x.shape == (n, 3) and abs(w.sum() - 1.0) < 1e-13
        assert np.allclose((x * x).sum(1), 1.0)
    with pytest.raises(ValueError, match="no Lebedev grid"):
        sv.lebedev_grid(100)


def test_born_radii_derivatives():
    labels, numbers, pos = _methanol()
    vdw = sv._radii(sv.RADII_TABLES["d3"], None, labels, numbers)
    rho = vdw * 0.8

    def radii(p):
        return sv.born_radii(p, vdw, rho, 0.02, 1.5, (1.0, 0.8, 4.85), 66.0)[0]
    _, d = sv.born_radii(pos, vdw, rho, 0.02, 1.5, (1.0, 0.8, 4.85), 66.0)
    assert np.abs(_numeric_derivative(radii, pos) - d).max() < 1e-8


def test_surface_area_derivatives():
    labels, numbers, pos = _methanol()
    radius = sv._radii(sv.RADII_TABLES["d3"], None, labels, numbers) + 2.0
    grid, weights = sv.lebedev_grid(230)

    def areas(p):
        return sv.surface_areas(p, radius, 0.3 / BOHR, 1e-6, 2.0 / BOHR, grid, weights)[0]
    s, d = sv.surface_areas(pos, radius, 0.3 / BOHR, 1e-6, 2.0 / BOHR, grid, weights)
    assert np.all(s > 0)
    assert np.abs(_numeric_derivative(areas, pos) - d).max() < 1e-7


def test_cm5_correction():
    labels, numbers, pos = _methanol()
    c, d = sv.cm5_correction(pos, labels, numbers, sv.CM5())
    assert np.abs(_numeric_derivative(lambda p: sv.cm5_correction(p, labels, numbers,
                                                                  sv.CM5())[0], pos)
                  - d).max() < 1e-9
    c2, _ = sv.cm5_correction(pos, labels, numbers, sv.CM5(double_count=True))
    assert np.allclose(c2, 2 * c)
    assert sv.cm5_pair_parameter(1, 8) == 0.1671 and sv.cm5_pair_parameter(8, 1) == -0.1671
    assert abs(sv.cm5_pair_parameter(6, 6)) == 0.0


@pytest.mark.parametrize("model", [
    GB, replace(GB, kernel="p16", alpb=True), replace(GB, cm5=sv.CM5()), GBSA,
    replace(GBSA, cm5=sv.CM5()), GBSA.sasa], ids=["still", "alpb_p16", "cm5", "gbsa",
                                                   "gbsa_cm5", "sasa"])
def test_forces(model):
    """Forces of the SCC energy with each model match energy differences."""
    calc, labels, pos = _engine("ch3oh", solvation=model)
    r = calc.calculate(labels, pos)
    h = 1e-4
    for atom, k in ((0, 0), (1, 2), (3, 1)):
        e = []
        for s in (1, -1):
            p = pos.copy()
            p[atom, k] += s * h
            e.append(calc.calculate(labels, p, forces=False).free_energy)
        assert abs(-(e[0] - e[1]) / (2 * h) - r.forces[atom, k]) < 1e-7


def test_charged_alpb_forces():
    calc, labels, pos = _engine("oh_minus_alpb")
    r = calc.calculate(labels, pos)
    assert calc.charge == -1.0 and r.e_solv < -0.1
    h = 1e-4
    e = []
    for s in (1, -1):
        p = pos.copy()
        p[0, 2] += s * h
        e.append(calc.calculate(labels, p, forces=False).free_energy)
    assert abs(-(e[0] - e[1]) / (2 * h) - r.forces[0, 2]) < 1e-7


def test_vacuum_dielectric_gives_no_born_term():
    """eps = 1: k_eps = 0, bit for bit the plain calculation."""
    calc, labels, pos = _engine("ch3oh")
    ref = calc.calculate(labels, pos)
    vac = sv.GeneralizedBorn(sv.Solvent(1.0), 0.0, 1.5, 0.0, "unity")
    c, *_ = _engine("ch3oh", solvation=vac)
    r = c.calculate(labels, pos)
    assert r.energy == ref.energy and np.array_equal(r.forces, ref.forces) and r.e_solv == 0.0


def test_reference_states_and_field_scaling():
    kt = 298.15 * KB_HARTREE
    assert GB.total_shift() == 0.0
    assert math.isclose(replace(GB, state="mol1bar").total_shift(), kt * math.log(24.79))
    water = sv.SOLVENTS["water"]
    assert math.isclose(replace(GB, state="reference").total_shift(),
                        kt * (math.log(24.79) + math.log(1000 * water.density / water.molecular_mass)))
    assert math.isclose(GB.field_scaling(), 3 * 78.5 / (2 * 78.5 + 1))
    assert replace(GB, solvent=sv.Solvent(math.inf)).field_scaling() == 1.5
    with pytest.warns(RuntimeWarning, match="no electrostatic"):
        _engine("ch3oh", solvation=GBSA.sasa, electric_field=(0.01, 0, 0))


def test_param_file():
    m = sv.GeneralizedBorn.from_param_file(os.path.join(DATA, "solvation", "param_gbsa_test.txt"),
                                           state="mol1bar")
    assert m.solvent.epsilon == 80.2 and m.state == "mol1bar"
    assert math.isclose(m.sasa.probe_radius, 1.843075777670416 / BOHR)
    assert m.descreening["O"] == 0.75369019 and m.sasa.surface_tension["H"] == -0.33498306
    assert m.hbond["N"] < 0 and m.hbond["C"] < 0


def test_ase_and_errors():
    water = molecule("H2O")
    water.calc = PyDFTB(SKF_HCOCU, scc_tol=1e-8, solvation=GBSA)
    e = water.get_potential_energy()
    assert water.calc.raw.e_solv < 0 and np.isfinite(e)
    with pytest.raises(ValueError, match="molecules only"):
        _engine("water_box_gamma", solvation=GB)
    with pytest.raises(ValueError, match="needs SCC"):
        _engine("h2o_nonscc", solvation=GB)
    with pytest.raises(ValueError, match="hydrogen-bond correction needs a SASA"):
        replace(GB, hbond={"H": -0.01})
    with pytest.raises(ValueError, match="unknown solvent"):
        sv.GeneralizedBorn("tea", 0.0, 1.5, 0.0, "unity")
    with pytest.raises(ValueError, match="kernel"):
        replace(GB, kernel="born")
    with pytest.raises(ValueError, match="no descreening value for Cu"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _engine("cu4co_300K", solvation=replace(GB, descreening={"C": 1.0, "O": 1.0}))
    with pytest.raises(ValueError, match="GeneralizedBorn or SASA"):
        _engine("ch3oh", solvation={"model": "gb"})


def test_torch_and_autodiff_refuse():
    pytest.importorskip("torch")
    from pydftb.autodiff import evaluate
    with pytest.raises(ValueError, match="NumPy backend only"):
        _engine("ch3oh", solvation=GB, backend="torch")
    calc, labels, pos = _engine("ch3oh", solvation=GB)
    with pytest.raises(NotImplementedError, match="milestone M6c"):
        evaluate(calc, labels, pos)
