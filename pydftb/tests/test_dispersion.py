"""Dispersion: forces and stress vs finite differences, lattice-sum invariances, option checks.

Comparisons with DFTB+ are the frozen references *_uff, *_sk*, *_d3*, *_d4* in
test_reference.py; here only self-consistency (no dftb+ needed).  Tests of the D3/D4
library models are skipped when the optional packages are missing.
"""
import importlib.util

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from pydftb.ase_calc import PyDFTB
from pydftb.calculator import DFTB
from pydftb.constants import BOHR, KCAL_MOL
from pydftb.dispersion import (UFF, DFTD4, LennardJones, SlaterKirkwood, ewald_r6, hh_repulsion,
                               make_dispersion)
from pydftb.params import ParamSet
from pydftb.validation.cases import (D3BJ_3OB, D4_3OB, SK_HYBRID, _cuo_chain, _meoh, _water_box,
                                     _water_dimer, sk_per_atom)

from .conftest import SKF_HCOCU

needs_d3 = pytest.mark.skipif(importlib.util.find_spec("dftd3") is None,
                              reason="optional package dftd3 not installed")
needs_d4 = pytest.mark.skipif(importlib.util.find_spec("dftd4") is None,
                              reason="optional package dftd4 not installed")


def _geometry(build):
    a = build()
    periodic = bool(a.pbc.any())
    return (a.get_chemical_symbols(), a.positions / BOHR,
            a.cell.array / BOHR if periodic else None, np.ones(3, bool) if periodic else None)


def _check_derivatives(model, build, tol_grad, tol_virial, h=1e-5):
    labels, pos, cell, pbc = _geometry(build)
    t = model.evaluate(labels, pos, cell, pbc)
    for atom, ax in ((0, 0), (1, 2), (3, 1)):
        e = []
        for sgn in (1, -1):
            p = pos.copy()
            p[atom, ax] += sgn * h
            e.append(model.evaluate(labels, p, cell, pbc, grad=False).energy)
        assert (e[0] - e[1]) / (2 * h) == pytest.approx(t.gradient[atom, ax], abs=tol_grad)
    if cell is None:
        return
    for a, b in ((0, 0), (0, 1), (2, 1)):
        eps = np.zeros((3, 3))
        eps[a, b] += h / 2
        eps[b, a] += h / 2
        e = [model.evaluate(labels, pos @ (np.eye(3) + s * eps).T, cell @ (np.eye(3) + s * eps).T,
                            pbc, grad=False).energy for s in (1, -1)]
        assert (e[0] - e[1]) / (2 * h) == pytest.approx(t.virial[a, b], abs=tol_virial)


@pytest.mark.parametrize("build", [_water_dimer, _water_box, _cuo_chain])
def test_lennard_jones_derivatives(build):
    _check_derivatives(LennardJones(), build, 1e-9, 1e-7)


@pytest.mark.parametrize("build", [_meoh, _water_box])
def test_slater_kirkwood_derivatives(build):
    _check_derivatives(SlaterKirkwood(hybrid_dependent_pol=SK_HYBRID), build, 1e-9, 1e-7)


def test_hh_repulsion_derivatives():
    class HH:
        def evaluate(self, labels, pos, cell=None, pbc=None, grad=True, charge=0.0):
            return hh_repulsion(np.array([1 if s == "H" else 8 for s in labels]), pos, cell, pbc,
                                grad)
    _check_derivatives(HH(), _water_dimer, 1e-12, None)
    _check_derivatives(HH(), _water_box, 1e-12, 1e-10)


def test_lennard_jones_wall_joins_smoothly():
    """At r0 = 2^(-1/6) r_ij both forms are 0 with slope -24 d / r0 (UFF O-O)."""
    lj = LennardJones()
    x, d = UFF["O"][0] / BOHR, UFF["O"][1] * KCAL_MOL
    r0 = 2.0 ** (-1.0 / 6.0) * x
    dij, rij = np.full((1, 1), d), np.full((1, 1), x)
    i = j = np.zeros(1, int)
    for r in (r0 * (1 - 1e-9), r0 * (1 + 1e-9)):
        e, de = lj._pair(i, j, np.array([r]), dij, rij, 2 * dij * rij ** 6, dij * rij ** 12, True)
        assert e[0] == pytest.approx(-24.0 * d / r0 * (r - r0), abs=1e-18)
        assert de[0] == pytest.approx(-24.0 * d / r0, rel=1e-6)


def test_uff_table():
    assert len(UFF) == 103
    assert UFF["H"] == (2.886, 0.044) and UFF["C"] == (3.851, 0.105)
    assert UFF["O"] == (3.500, 0.060) and UFF["Cu"] == (3.495, 0.005)
    assert UFF["Lr"] == (3.236, 0.011)          # DFTB+ writes the old symbol "Lw"


def test_ewald_r6_is_a_property_of_the_crystal():
    """Same energy for another basis of the same lattice (another eta and G set) and
    per cell for a supercell; the r^-6 tail of a direct sum agrees too."""
    a = _cuo_chain()
    labels, pos, cell, _ = _geometry(lambda: a)
    c6 = np.full((4, 4), 30.0) + np.diag([10.0, 5.0, 10.0, 5.0])
    e0 = ewald_r6(pos, cell, c6).energy
    skew = np.array([cell[0], cell[1] + cell[0], cell[2] - 2 * cell[1]])
    assert ewald_r6(pos, skew, c6).energy == pytest.approx(e0, abs=1e-9)
    sup = a.repeat((1, 2, 1))
    c6s = np.tile(c6, (2, 2))
    assert ewald_r6(sup.positions / BOHR, sup.cell.array / BOHR, c6s).energy / 2 == \
        pytest.approx(e0, abs=1e-9)


def test_hybrid_dependent_pol_counts_neighbours():
    labels, pos, cell, pbc = _geometry(_meoh)        # C, O, H (on O), H, H, H (on C)
    alpha, rvdw, nel = SlaterKirkwood(hybrid_dependent_pol=SK_HYBRID).atom_parameters(labels, pos)
    pol = {k: np.array(v["hybrid_polarisations"]) for k, v in SK_HYBRID.items()}
    expect_n = {"C": 4, "O": 2, "H": 1}
    for lab, a, r, n in zip(labels, alpha, rvdw, nel):
        k = expect_n[lab]
        assert (a, r, n) == (pol[lab][k], pol[lab][6 + k], pol[lab][12])


def test_slater_kirkwood_skips_unpolarisable_atoms():
    """An atom with alpha = 0 adds nothing: same energy as without that atom."""
    labels, pos, _, _ = _geometry(_water_dimer)
    skip = SlaterKirkwood(**{k: v for k, v in sk_per_atom(labels, skip=(3,)).items()
                             if k != "model"})
    keep = [k for k in range(len(labels)) if k != 3]
    rows = np.array(sk_per_atom(labels)["polar_radius_charge"])[keep]
    ref = SlaterKirkwood(polar_radius_charge=rows)
    assert skip.evaluate(labels, pos).energy == pytest.approx(
        ref.evaluate([labels[k] for k in keep], pos[keep]).energy, abs=1e-15)


def test_dispersion_enters_energy_forces_and_stress():
    sk = ParamSet(SKF_HCOCU, max_l={"Cu": 2, "O": 1}).skset(["Cu", "O"])
    labels, pos, cell, pbc = _geometry(_cuo_chain)
    kpts = (np.zeros((1, 3)), np.ones(1))
    base = DFTB(sk, kpts=kpts, temperature=500.0, scc_tol=1e-10).calculate(labels, pos, cell, pbc)
    calc = DFTB(sk, kpts=kpts, temperature=500.0, scc_tol=1e-10,
                dispersion={"model": "lennardjones"})
    r = calc.calculate(labels, pos, cell, pbc)
    t = LennardJones().evaluate(labels, pos, cell, pbc)
    assert r.e_disp == pytest.approx(t.energy, abs=1e-14)
    assert r.terms["dispersion"] == r.e_disp
    assert r.energy - base.energy == pytest.approx(t.energy, abs=1e-11)
    assert np.allclose(r.forces - base.forces, -t.gradient, atol=1e-10)
    vol = abs(np.linalg.det(cell))
    assert np.allclose(r.stress - base.stress, -0.5 * (t.virial + t.virial.T) / vol, atol=1e-13)
    assert calc.calculate(labels, pos, cell, pbc, forces=False).energy == pytest.approx(r.energy,
                                                                                        abs=1e-12)


def test_ase_interface_with_dispersion():
    w = molecule("H2O")
    w.calc = PyDFTB(SKF_HCOCU, dispersion={"model": "lennardjones"})
    e = w.get_potential_energy()
    w.calc = PyDFTB(SKF_HCOCU)
    e0 = w.get_potential_energy()
    labels, pos, _, _ = _geometry(lambda: molecule("H2O"))
    assert e - e0 == pytest.approx(LennardJones().evaluate(labels, pos).energy * 27.211386, rel=1e-6)


def test_labels_map_to_elements():
    m = make_dispersion({"model": "lennardjones", "atomic_numbers": {"Ow": 8}})
    assert list(m.numbers(["Ow", "H1", "Li@el"])) == [8, 1, 3]
    assert list(m.with_elements({"Ow": 1, "X": 6}).numbers(["Ow", "X"])) == [8, 6]   # explicit wins


@pytest.mark.parametrize("spec, err", [
    ({"model": "nope"}, ValueError),
    ({"model": "dftd3", "s6": 1.0, "s8": 0.5}, ValueError),                     # BJ without a1/a2
    ({"model": "dftd3", "damping": "nope", "s6": 1.0, "s8": 0.5}, ValueError),
    ({"model": "simpledftd3", "s8": 0.5, "a1": 0.5, "a2": 3.6, "s10": 1.0}, ValueError),
    ({"model": "simpledftd3", "s8": 0.5, "a1": 0.5, "a2": 3.6, "weighting_factor": 6.0}, ValueError),
    ({**D4_3OB, "s10": 0.1}, ValueError),
    ({**D4_3OB, "cutoff_inter": 64.0}, ValueError),                              # fixed at 60
    ({**D4_3OB, "charge_model": "selfconsistent"}, NotImplementedError),
    ({"model": "slaterkirkwood"}, ValueError),
    ({"model": "slaterkirkwood", "hybrid_dependent_pol": {"H": {"covalent_radius": 0.7,
                                                                "hybrid_polarisations": [1.0]}}},
     ValueError),
    ({"model": "lennardjones", "parameters": "mm3"}, ValueError),
    ({"model": "lennardjones", "colour": "red"}, ValueError),                     # unknown key
])
def test_options_are_checked(spec, err):
    with pytest.raises(err):
        make_dispersion(spec)


def test_missing_parameters_are_reported():
    labels, pos, _, _ = _geometry(_water_dimer)
    with pytest.raises(ValueError, match="no Lennard-Jones parameters"):
        LennardJones(parameters={"O": (6.6, 1e-4)}).evaluate(labels, pos)
    with pytest.raises(NotImplementedError, match="3D periodic"):
        LennardJones().evaluate(labels, pos, np.eye(3) * 20, np.array([True, True, False]))


@needs_d3
def test_d3_forces_and_stress():
    _check_derivatives(make_dispersion({**D3BJ_3OB, "threebody": True}), _water_dimer, 1e-11, None)
    # the library's hard cutoffs (pairs at 95 Bohr, coordination numbers at 40 Bohr) make the
    # energy jump when strain moves pairs across them: only sign and scale are tested here
    # (the stress agrees with DFTB+, which calls the same library, to 1e-18)
    _check_derivatives(make_dispersion(D3BJ_3OB), _cuo_chain, 1e-7, 1e-4)


@needs_d4
def test_d4_forces_and_periodic_limits():
    d4 = make_dispersion(D4_3OB)
    _check_derivatives(d4, _meoh, 1e-11, None)
    labels, pos, _, _ = _geometry(_water_dimer)
    mol = d4.evaluate(labels, pos).energy
    box = 40.0 / BOHR
    shift = pos - pos.mean(0) + box / 2
    per = d4.evaluate(labels, shift, np.eye(3) * box, np.ones(3, bool)).energy
    assert per == pytest.approx(mol, abs=1e-8)           # periodic EEQ -> molecular limit
    # supercell invariance; DFTB+'s own D4 fails it (docs/design/M2d-dispersion.html).  The
    # library's periodic EEQ charges converge to ~1e-10 here, but only to ~1e-6 Ha in elongated
    # cells such as the 2x1x1 CuO chain; without charge dependence the sums are exact
    a = _water_box()
    cell_e = d4.evaluate(*_geometry(lambda: a)).energy
    assert d4.evaluate(*_geometry(lambda: a.repeat((1, 2, 1)))).energy / 2 == \
        pytest.approx(cell_e, abs=1e-9)
    d4q0 = make_dispersion({**D4_3OB, "charge_scale": 0.0})
    a = _cuo_chain()
    cell_e = d4q0.evaluate(*_geometry(lambda: a)).energy
    assert d4q0.evaluate(*_geometry(lambda: a.repeat((2, 1, 1)))).energy / 2 == \
        pytest.approx(cell_e, abs=1e-13)


@needs_d4
def test_d4_model_options_reach_the_library():
    labels, pos, _, _ = _geometry(_meoh)
    e = DFTD4(**{k: v for k, v in D4_3OB.items() if k != "model"}).evaluate(labels, pos).energy
    e0 = DFTD4(**{k: v for k, v in D4_3OB.items() if k != "model"},
               charge_scale=0.0).evaluate(labels, pos).energy
    assert e != pytest.approx(e0, abs=1e-8)
    e_cation = DFTD4(**{k: v for k, v in D4_3OB.items() if k != "model"}).evaluate(
        labels, pos, charge=1.0).energy
    assert e_cation != pytest.approx(e, abs=1e-8)
