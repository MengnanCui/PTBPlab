"""External fields, point charges and site potentials (milestone M6c).

Comparisons with DFTB+ are the frozen references h2o_field, h2o_pc*, h2o_site,
*_field*, *_pc and water_box_* in test_reference.py; here the consistency of forces,
forces on the point charges and stress with the energy, limits, the ASE interface
(QM/MM embedding) and errors (no dftb+).
"""
import warnings

import numpy as np
import pytest
from ase.build import molecule

from pydftb.ase_calc import PyDFTB
from pydftb.bands import eigenstates
from pydftb.constants import BOHR, VOLT_PER_ANGSTROM_AU
from pydftb.external import PointCharges, fold_into_cell
from pydftb.validation import refgen
from pydftb.validation.cases import CASES_BY_NAME

from .conftest import SKF_HCOCU

FIELD = 0.5 * VOLT_PER_ANGSTROM_AU * np.array([1.0, 0.3, -0.2]) / np.linalg.norm([1.0, 0.3, -0.2])
PC = (np.array([[3.0, 0.2, 0.1], [-2.5, 1.0, 0.5], [0.3, -2.8, 1.2]]) / BOHR,
      np.array([0.8, -0.6, 0.4]))


def _engine(name, **opts):
    case = CASES_BY_NAME[name]
    inp = refgen._inputs(case)
    _, calc = refgen.run_pydftb(case, inp, engine=True, scc_tol=1e-12, **opts)
    cell = None if inp["cell_angstrom"] is None else np.array(inp["cell_angstrom"]) / BOHR
    return calc, inp["labels"], np.array(inp["positions_angstrom"]) / BOHR, cell


def _free_energy(calc, labels, pos, cell):
    pbc = None if cell is None else np.ones(3, bool)
    return calc.calculate(labels, pos, cell, pbc, forces=False).free_energy


def _fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_zero_potentials_change_nothing():
    """A zero field, uncharged point charges and zero site potentials: bit for bit the
    plain calculation."""
    calc, labels, pos, _ = _engine("ch3oh")
    ref = calc.calculate(labels, pos)
    for opts in (dict(electric_field=(0.0, 0.0, 0.0)),
                 dict(point_charges=(PC[0], np.zeros(3))),
                 dict(site_potentials={"gross": {0: 0.0}, "net": {1: 0.0}})):
        c, *_ = _engine("ch3oh", **opts)
        r = c.calculate(labels, pos)
        assert r.energy == ref.energy and np.array_equal(r.forces, ref.forces)
        assert r.e_ext == 0.0


def test_molecule_forces_and_charge_forces():
    """Forces and the forces on the point charges are minus the energy's derivatives."""
    calc, labels, pos, _ = _engine("ch3oh", electric_field=FIELD,
                                   point_charges=(PC[0], PC[1], 0.4 / BOHR),
                                   site_potentials={"gross": {0: 0.01}, "net": {1: -0.01}})
    r = calc.calculate(labels, pos)
    h = 1e-4
    for atom, k in ((0, 0), (1, 1), (4, 2)):
        def e_at(x, atom=atom, k=k):
            p = pos.copy()
            p[atom, k] = x
            return _free_energy(calc, labels, p, None)
        assert abs(-_fd(e_at, pos[atom, k], h) - r.forces[atom, k]) < 1e-7
    pc = calc.point_charges
    for j, k in ((0, 0), (1, 2), (2, 1)):
        def e_pc(x, j=j, k=k):
            q = pc.positions.copy()
            q[j, k] = x
            calc.point_charges = PointCharges(q, pc.charges, pc.blur)
            return _free_energy(calc, labels, pos, None)
        fd = -_fd(e_pc, pc.positions[j, k], h)
        calc.point_charges = pc
        assert abs(fd - r.point_charge_forces[j, k]) < 1e-9


def test_dipole_is_the_field_derivative():
    """Hellmann-Feynman: dE/dE_field = sum_A dq_A r_A = -dipole."""
    calc, labels, pos, _ = _engine("h2o", electric_field=FIELD)
    d = calc.calculate(labels, pos, forces=False).dipole
    h = 1e-4
    for k in range(3):
        def e_of(x, k=k):
            f = FIELD.copy()
            f[k] = x
            calc.electric_field = f
            return calc.calculate(labels, pos, forces=False).energy
        assert abs(_fd(e_of, FIELD[k], h) + d[k]) < 1e-8
    calc.electric_field = FIELD


def test_periodic_field_stress_in_a_rectangular_cell():
    """The saw-tooth field's stress matches strain differences also where DFTB+ 25.1's is
    wrong (off the diagonal of a non-cubic cell, M6c report)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # saw-tooth warnings
        calc, labels, pos, cell = _engine("cuo_chain_k422", electric_field=0.2 * FIELD)
        pos = fold_into_cell(pos, cell)
        r = calc.calculate(labels, pos, cell, np.ones(3, bool), stress=True)
        vol = abs(np.linalg.det(cell))
        h = 1e-5
        for i, j in ((0, 0), (0, 1), (0, 2), (1, 2)):
            def e_eps(x, i=i, j=j):
                eps = np.zeros((3, 3))
                eps[i, j] += 0.5 * x
                eps[j, i] += 0.5 * x
                m = np.eye(3) + eps
                return _free_energy(calc, labels, pos @ m, cell @ m)
            assert abs(-_fd(e_eps, 0.0, h) / vol - r.stress[i, j]) < 1e-8


def test_periodic_point_charges():
    """Ewald sums over the point charges' images: forces on atoms and charges."""
    calc, labels, pos, cell = _engine("water_box_gamma",
                                      point_charges=(PC[0] + 2.0 / BOHR, PC[1], 0.3 / BOHR))
    r = calc.calculate(labels, pos, cell, np.ones(3, bool))
    assert r.stress is None                                  # as DFTB+
    h = 1e-4

    def e_at(x):
        p = pos.copy()
        p[3, 1] = x
        return _free_energy(calc, labels, p, cell)
    assert abs(-_fd(e_at, pos[3, 1], h) - r.forces[3, 1]) < 1e-7
    pc = calc.point_charges

    def e_pc(x):
        q = pc.positions.copy()
        q[1, 0] = x
        calc.point_charges = PointCharges(q, pc.charges, pc.blur)
        return _free_energy(calc, labels, pos, cell)
    fd = -_fd(e_pc, pc.positions[1, 0], h)
    calc.point_charges = pc
    assert abs(fd - r.point_charge_forces[1, 0]) < 1e-8
    with pytest.raises(ValueError, match="no stress with external point charges"):
        calc.calculate(labels, pos, cell, np.ones(3, bool), stress=True)


def test_field_without_scc_and_sawtooth_warnings():
    calc, labels, pos, _ = _engine("h2o_nonscc", electric_field=FIELD)
    r = calc.calculate(labels, pos)
    assert r.e_ext != 0.0 and r.n_scc == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        calc, labels, pos, cell = _engine("water_box_gamma", electric_field=FIELD)
    with pytest.warns(RuntimeWarning, match="saw-tooth"):
        calc.calculate(labels, pos, cell, np.ones(3, bool), forces=False)


def test_eigenstates_include_the_potentials():
    """The band-structure path builds the SCC's H: field, charges, net site potential."""
    calc, labels, pos, _ = _engine("ch3oh", electric_field=FIELD, point_charges=PC,
                                   site_potentials={"net": {1: -0.02}})
    r = calc.calculate(labels, pos)
    st = eigenstates(calc, labels, pos, populations=r.q_in)
    assert np.abs(st.eigenvalues[0, 0] - np.asarray(r.eigenvalues[0])).max() < 1e-9


def test_ase_units_and_qmmm_embedding():
    """PyDFTB takes V/A, A and eV; ASE's EIQMMM drives the point charges through embed()."""
    from ase.calculators.qmmm import EIQMMM, LJInteractions
    from ase.calculators.tip3p import TIP3P, epsilon0, sigma0

    water = molecule("H2O")
    water.rattle(0.03, seed=1)
    calc = PyDFTB(SKF_HCOCU, scc_tol=1e-10, electric_field=(0.5, 0.0, 0.0))
    water.calc = calc
    e_field = water.get_potential_energy()
    engine, labels, pos, _ = _engine("h2o", electric_field=(0.5 * VOLT_PER_ANGSTROM_AU, 0, 0))
    raw = engine.calculate(water.get_chemical_symbols(), water.positions / BOHR)
    assert abs(calc.raw.energy - raw.energy) < 1e-9
    assert water.calc.get_property("dipole").shape == (3,)
    assert e_field != PyDFTB(SKF_HCOCU, scc_tol=1e-10).get_potential_energy(water.copy())

    dimer = molecule("H2O") + molecule("H2O")
    dimer.positions[3:] += [0.2, 0.3, 2.9]
    dimer.rattle(0.02, seed=2)
    lj = LJInteractions({("O", "O"): (epsilon0, sigma0)})
    qmmm = EIQMMM([0, 1, 2], PyDFTB(SKF_HCOCU, scc_tol=1e-10), TIP3P(), lj)
    dimer.calc = qmmm
    f = dimer.get_forces()
    h = 1e-4
    for atom, k in ((0, 2), (3, 2), (4, 0)):          # a QM atom and two MM atoms
        e = []
        for s in (1, -1):
            d = dimer.copy()
            d.positions[atom, k] += s * h
            d.calc = EIQMMM([0, 1, 2], PyDFTB(SKF_HCOCU, scc_tol=1e-10), TIP3P(), lj)
            e.append(d.get_potential_energy())
        assert abs(-(e[0] - e[1]) / (2 * h) - f[atom, k]) < 1e-5     # eV/A


def test_errors():
    calc, labels, pos, _ = _engine("h2o")
    with pytest.raises(ValueError, match="sits on an atom"):
        _engine("h2o", point_charges=(pos[:1], [1.0]))[0].calculate(labels, pos)
    with pytest.raises(ValueError, match="need SCC"):
        _engine("h2o_nonscc", point_charges=PC)
    with pytest.raises(ValueError, match="unknown kinds"):
        _engine("h2o", site_potentials={"total": {0: 0.1}})
    with pytest.raises(ValueError, match="outside"):
        _engine("h2o", site_potentials={"gross": {7: 0.1}})[0].calculate(labels, pos)
    with pytest.raises(ValueError, match="vector"):
        _engine("h2o", electric_field=(1.0, 0.0))
    with pytest.raises(ValueError, match="may not be negative"):
        PointCharges.make((PC[0], PC[1], -1.0))


def test_torch_and_autodiff_refuse():
    torch = pytest.importorskip("torch")      # noqa: F841
    from pydftb.autodiff import evaluate
    with pytest.raises(ValueError, match="NumPy backend only"):
        _engine("h2o", electric_field=FIELD, backend="torch")
    calc, labels, pos, _ = _engine("h2o", electric_field=FIELD)
    with pytest.raises(NotImplementedError, match="milestone M6c"):
        evaluate(calc, labels, pos)
