"""Corrections of milestone M6f-3: H5, the halogen correction and the dipole correction.

The results against DFTB+ are the frozen references *_h5*, *_halogen* and *_dipole* in
test_reference.py (and water_dimer_h5_md in test_md_opt.py; the XL-BOMD forces with H5 in
test_xlbomd.py); here the rules themselves: forces and stress as derivatives of the energy
(also where DFTB+'s are not), the limits in which nothing changes, the switch that reproduces
DFTB+'s halogen forces, and the refusals.
"""
import warnings

import numpy as np
import pytest

from pydftb import corrections
from pydftb.ase_calc import PyDFTB
from pydftb.calculator import DFTB
from pydftb.constants import BOHR, KCAL_MOL
from pydftb.corrections import (GAUSS_WIDTH, H5, DipoleCorrection, h5_scale, halogen_cutoff,
                                halogen_terms, vdw_radius)
from pydftb.params import ParamSet
from pydftb.validation import refgen

from .conftest import SKF_HCOCU


def _case(name, **options):
    """pydftb's result and engine for a reference case (its M6f-3 options included)."""
    case = refgen.CASES_BY_NAME[name]
    inp = refgen._inputs(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # the periodic halogen warning
        res, calc = refgen.run_pydftb(case, inp, engine=True, **options)
    pos = np.array(inp["positions_angstrom"]) / BOHR
    cell = None if inp["cell_angstrom"] is None else np.array(inp["cell_angstrom"]) / BOHR
    return res, calc, inp["labels"], pos, cell


def _fd(energy, h):
    """dE/dx of ``energy(x)`` at 0 by 4-point central differences."""
    return (8.0 * (energy(h) - energy(-h)) - (energy(2 * h) - energy(-2 * h))) / (12.0 * h)


def _energy_at(calc, labels, pos, cell):
    pbc = None if cell is None else np.ones(3, bool)

    def at_coordinate(i, x):
        p = pos.copy()
        p.flat[i] += x
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return calc.calculate(labels, p, cell, pbc, forces=False).free_energy

    def at_strain(a, b, x):
        e = np.zeros((3, 3))
        e[a, b] += 0.5 * x
        e[b, a] += 0.5 * x
        m = np.eye(3) + e
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return calc.calculate(labels, pos @ m.T, cell @ m.T, pbc, forces=False).free_energy

    return at_coordinate, at_strain


def _check_derivatives(name, coords, strains=(), tol_f=2e-9, tol_w=5e-8, h_f=2e-4, **options):
    """Forces (4-point differences, a step well below the SK grid spacing, so no stencil
    crosses a grid node) and the virial W = -V sigma (central differences of a 1e-5 strain,
    as test_stress.py: larger strains move the long distances of a cell across nodes)."""
    res, calc, labels, pos, cell = _case(name, scc_tol=1e-12, **options)
    at_coordinate, at_strain = _energy_at(calc, labels, pos, cell)
    for i in coords:
        fd = _fd(lambda x: at_coordinate(i, x), h_f)
        assert -res.forces.flat[i] == pytest.approx(fd, abs=tol_f), (name, i)
    vol = None if cell is None else abs(np.linalg.det(cell))
    for a, b in strains:
        h = 1e-5
        fd = (at_strain(a, b, h) - at_strain(a, b, -h)) / (2.0 * h)
        assert -res.stress[a, b] * vol == pytest.approx(fd, abs=tol_w), (name, a, b)
    return res


# ---- H5 -------------------------------------------------------------------------------------
def test_h5_scale_is_gamma_times_one_plus_g():
    """S' = S (1 + g) - g / r is gamma' = gamma (1 + g), and dS'/dr is its derivative (a smooth
    stand-in S = e^-r)."""
    r = np.linspace(1.5, 7.0, 12)
    k, svdw = 0.06, (vdw_radius("O") + vdw_radius("H"))
    s, ds = h5_scale(np.exp(-r), -np.exp(-r), r, k, svdw, 0.714, 0.25)
    c = 0.25 * svdw * GAUSS_WIDTH
    g = k * np.exp(-0.5 * (r - 0.714 * svdw) ** 2 / c ** 2)
    np.testing.assert_allclose(1.0 / r - s, (1.0 / r - np.exp(-r)) * (1.0 + g), rtol=1e-13, atol=0)

    def sp(x):
        return h5_scale(np.exp(-x), -np.exp(-x), x, k, svdw, 0.714, 0.25)[0]
    np.testing.assert_allclose(ds, _fd(lambda h: sp(r + h), 1e-5), rtol=0, atol=1e-10)


def test_h5_forces_and_stress_are_derivatives():
    """H5 enters gamma, its derivative and the virial (the molecule and the Ewald sum)."""
    _check_derivatives("water_dimer_h5", coords=[2, 7, 12, 16])
    _check_derivatives("water_box_h5", coords=[5, 40], strains=[(0, 0), (1, 2)])


def test_h5_without_acceptors_is_the_plain_calculation():
    """Every scaling <= 0: no pair is corrected, bitwise the calculation without H5."""
    plain, *_ = _case("water_dimer_h5", h5=None)
    off, *_ = _case("water_dimer_h5", h5={"scaling": {"O": -1.0, "N": 0.0, "S": -0.1}})
    assert off.energy == plain.energy and off.n_scc == plain.n_scc
    assert np.array_equal(off.forces, plain.forces)
    on, *_ = _case("water_dimer_h5")
    assert on.energy - plain.energy == pytest.approx(-1.06e-3, abs=1e-5)    # DFTB+: -1.060e-3


def test_h5_by_element():
    """pydftb identifies acceptors by element: the type label O2 (a copy of O) is oxygen, so
    the results are those of plain labels (DFTB+ leaves O2-H pairs out: its species name)."""
    plain, *_ = _case("water_dimer_h5")
    typed, *_ = _case("water_dimer_typed", h5=True)
    assert typed.energy == pytest.approx(plain.energy, abs=1e-11)
    assert np.allclose(typed.forces, plain.forces, atol=1e-10)


def test_h5_parameters():
    assert H5.make(None) is None and H5.make(False) is None
    h5 = H5.make(True)
    assert (h5.r_scaling, h5.w_scaling, h5.scaling) == (0.714, 0.25, {})
    custom = H5.make({"r_scaling": 0.8, "scaling": {"S": 0.3}})
    is_h, k, radius, rs, ws = custom.atom_parameters(["O", "H", "S", "C"])
    assert list(is_h) == [False, True, False, False] and (rs, ws) == (0.8, 0.25)
    assert list(k) == [0.06, -1.0, 0.3, -1.0]              # defaults kept, C not corrected
    assert radius[1] == pytest.approx(1.20 / BOHR, rel=1e-15)
    with pytest.raises(ValueError, match="unknown keys"):
        H5.make({"rscaling": 0.8})
    with pytest.raises(ValueError, match="w_scaling must be positive"):
        H5.make({"w_scaling": 0.0})
    with pytest.raises(ValueError, match="without H atoms"):
        h5.atom_parameters(["O", "C"])
    with pytest.warns(RuntimeWarning, match="radius for Ti"):
        _, k, _, _, _ = H5.make({"scaling": {"Ti": 0.1}}).atom_parameters(["Ti", "H"])
    assert k[0] == -1.0                                       # no radius: left out, as DFTB+


def test_h5_refusals():
    sk = ParamSet(SKF_HCOCU).skset(["O", "H"])
    with pytest.raises(ValueError, match="needs SCC"):
        DFTB(sk, scc=False, h5=True)
    with pytest.raises(ValueError, match="exclusive"):
        DFTB(sk, h5=True, h_damping=4.0)
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="h5"):
        DFTB(sk, h5=True, backend="torch")
    from pydftb.autodiff import api
    with pytest.raises(NotImplementedError, match="h5"):
        api.evaluate(DFTB(sk, h5=True), ["O", "H", "H"],
                     np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 1.8, 0.0]]))


# ---- halogen correction ----------------------------------------------------------------------
def _pair(r_ang, acceptor="O", halogen="Cl", grad=True, compat=False):
    pos = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 1.0]])
    pos[1] *= r_ang / np.linalg.norm(pos[1]) / BOHR
    return halogen_terms([acceptor, halogen], pos, grad=grad, compat=compat), pos


def _formula(r, d, rv):
    c1, c2, c3 = 7.761, 0.050, 4.518
    x = (r - 0.7 * rv) / (0.1 * rv)
    sig = 0.0 if r < 0.7 * rv else 1.0 if r > 0.8 * rv else (-20 * x**7 + 70 * x**6 - 84 * x**5
                                                             + 35 * x**4)
    f = c1 * np.exp(-c2 * (r - d) ** c3) if sig > 0 else 0.0
    return (sig * f + (1 - sig) * c1 * np.exp(-c2 * (0.7 * rv - d) ** c3)) * KCAL_MOL


@pytest.mark.parametrize("acceptor, halogen", [("O", "Cl"), ("N", "Br"), ("O", "I")])
def test_halogen_energy_and_gradient(acceptor, halogen):
    """The pair energy of halogenx.F90 (kcal/mol, A) on both sides of the switching region;
    the gradient is its derivative everywhere, continuous at 0.7 R and 0.8 R; each atom gets
    half of the pair."""
    d = corrections.HALOGEN_D[(acceptor, halogen)]
    rv = 0.01 * (corrections.VDW_PM[acceptor] + corrections.VDW_PM[halogen])
    for r in (0.6 * rv, 0.72 * rv, 0.75 * rv, 0.79 * rv, 0.85 * rv, 1.2 * rv):
        t, pos = _pair(r, acceptor, halogen)
        assert t.energy == pytest.approx(_formula(r, d, rv), rel=1e-13, abs=1e-18)
        assert t.atom_energies == pytest.approx([0.5 * t.energy] * 2, rel=1e-15)
        u = (pos[1] - pos[0]) / np.linalg.norm(pos[1] - pos[0])
        dedr = t.gradient[1] @ u
        fd = _fd(lambda h: _pair(r + h * BOHR, acceptor, halogen, grad=False)[0].energy, 1e-4)
        assert dedr == pytest.approx(fd, abs=1e-11)
        assert np.array_equal(t.gradient[0], -t.gradient[1])
    for edge in (0.7 * rv, 0.8 * rv):                     # no jump in the energy or the gradient
        lo = _pair(edge - 1e-12, acceptor, halogen)[0]    # (the slope gives ~1e-14 here)
        hi = _pair(edge + 1e-12, acceptor, halogen)[0]
        assert abs(lo.energy - hi.energy) < 5e-14
        assert np.abs(lo.gradient - hi.gradient).max() < 1e-10


def test_halogen_compat_flips_the_switching_terms():
    """DFTB+'s getEnergyDeriv: +sigma'(f(r) - f(0.7 R)) instead of -sigma'(...); the same
    gradient outside the switching region."""
    rv = 0.01 * (152 + 175)
    for r, inside in ((0.6 * rv, False), (0.75 * rv, True), (0.9 * rv, False)):
        good = _pair(r)[0].gradient
        bad = _pair(r, compat=True)[0].gradient
        assert (np.abs(good - bad).max() > 1e-4) == inside
        if not inside:
            assert np.array_equal(good, bad)


def test_halogen_cutoff_and_short_distances():
    """DFTB+'s cut-off: the largest d present plus where a pair falls to 2e-14 kcal/mol; below
    0.7 R the term is constant, also below d where DFTB+'s (r - d)^c3 is NaN."""
    c1, c2, c3 = corrections.HALOGEN_C
    tail = (-np.log(2e-14 / c1) / c2) ** (1.0 / c3)
    assert tail == pytest.approx(4.2245, abs=5e-5)
    assert halogen_cutoff({("O", "Cl")}) * BOHR == pytest.approx(1.237 + tail, rel=1e-14)
    assert halogen_cutoff({("O", "Cl"), ("N", "I")}) * BOHR == pytest.approx(1.521 + tail,
                                                                             rel=1e-14)
    short, _ = _pair(1.0)                                     # r < d = 1.237 A
    far, _ = _pair(2.0)                                       # both below 0.7 R = 2.289 A
    assert np.isfinite(short.energy) and short.energy == far.energy
    assert not short.gradient.any()
    beyond, _ = _pair(halogen_cutoff({("O", "Cl")}) * BOHR + 0.01)
    assert beyond.energy == 0.0


def test_halogen_forces_and_stress_are_derivatives():
    """In the calculation: the switching region of the molecule, the periodic box (stress)."""
    _check_derivatives("meoh_water_halogen_switch", coords=[3, 18, 20, 25])
    _check_derivatives("water_box_halogen", coords=[0, 37], strains=[(0, 0), (0, 2)])


def test_halogen_refusals_and_warning():
    sk = ParamSet(SKF_HCOCU).skset(["O", "H", "C"])
    with pytest.raises(ValueError, match="needs third_order"):
        DFTB(sk, halogen_correction=True)
    calc = DFTB(sk, third_order="full", hubbard_derivs={"O": -0.1575, "H": -0.1857,
                                                       "C": -0.1492},
                halogen_correction=True)
    with pytest.raises(ValueError, match="No suitable"):
        calc.calculate(["O", "H", "H"], np.array([[0.0, 0, 0], [1.8, 0, 0], [0, 1.8, 0]]))
    case = refgen.CASES_BY_NAME["water_box_halogen"]
    with pytest.warns(RuntimeWarning, match="not fitted for periodic systems"):
        refgen.run_pydftb(case, refgen._inputs(case))


# ---- dipole correction -----------------------------------------------------------------------
def _dipole_check(name, layer=None):
    res, calc, labels, pos, cell = _case(name, scc_tol=1e-12)
    case = refgen.CASES_BY_NAME[name]
    dq = res.charges - refgen._neutral_populations(case, refgen._inputs(case))
    h, vol = cell[2, 2], abs(np.linalg.det(cell))
    z0 = -0.5 * h if layer is None else layer
    zf = np.mod(pos[:, 2] - z0, h) + z0
    dip = float(dq @ (zf - (z0 + 0.5 * h)))
    return res, dip, vol


def test_dipole_energy():
    """E = 2 pi D^2 / V with D = sum_A dq_A (z_f,A - z_c), booked in E_SCC as in DFTB+."""
    res, dip, vol = _dipole_check("water_slab_dipole")
    assert res.e_dipole_corr == pytest.approx(2.0 * np.pi * dip ** 2 / vol, rel=1e-12)
    assert res.e_dipole_corr == pytest.approx(5.896e-3, abs=1e-6)        # DFTB+ probe P11
    res, dip, vol = _dipole_check("water_slab_dipole_layer", 5.0 / BOHR)
    assert res.e_dipole_corr == pytest.approx(2.0 * np.pi * dip ** 2 / vol, rel=1e-12)


def test_dipole_forces_and_stress_are_derivatives(monkeypatch):
    """Forces and the whole stress tensor (DFTB+ has no stress term): the shears out of the slab
    plane with the geometry check off (the correction along the normal of the slab plane, which
    such a strain tilts)."""
    setup = DipoleCorrection.setup
    monkeypatch.setattr(DipoleCorrection, "setup",
                        lambda self, pos, cell, pbc, check=True: setup(self, pos, cell, pbc,
                                                                       check=False))
    # atom 0 has a distance 2e-4 Bohr from an SK grid node: a smaller step for the forces
    res = _check_derivatives("water_slab_dipole", coords=[2, 13, 29], h_f=5e-5,
                             strains=[(0, 0), (2, 2), (0, 2), (1, 2), (0, 1)])
    w = DipoleCorrection().setup(np.zeros((1, 3)), np.diag([6.0, 6.0, 20.0]) / BOHR,
                                 np.ones(3, bool))
    assert np.allclose(w.normal, [0.0, 0.0, 1.0])
    assert res.e_dipole_corr > 1e-3


def test_dipole_layer_and_folding():
    """Molecules given one period lower fold onto the same slab; the layer at z0 and at z0 + h
    is the same layer, and with no atom crossing it its position does not matter."""
    folded, calc, labels, pos, cell = _case("water_slab_dipole_layer")
    h = cell[2, 2]
    unfolded = pos.copy()
    unfolded[3:6, 2] += h                                    # molecules 2 and 4 given at -5 A
    unfolded[9:12, 2] += h
    pbc = np.ones(3, bool)
    same = calc.calculate(labels, unfolded, cell, pbc)
    assert same.energy == pytest.approx(folded.energy, abs=1e-11)
    assert np.allclose(same.forces, folded.forces, atol=1e-10)
    for z0 in (5.0 / BOHR + h, None, 0.5 * h):
        calc.dipole_correction = DipoleCorrection(z0)
        r = calc.calculate(labels, pos, cell, pbc)
        assert r.energy == pytest.approx(folded.energy, abs=1e-11), z0
    shifted = calc.calculate(labels, pos + [0.0, 0.0, 0.4], cell, pbc)      # no atom crosses
    assert shifted.energy == pytest.approx(folded.energy, abs=1e-11)


def test_dipole_ase_units():
    case = refgen.CASES_BY_NAME["water_slab_dipole_layer"]
    res = refgen.run_pydftb(case, refgen._inputs(case), scc_tol=1e-10)
    atoms = case.build()
    atoms.calc = PyDFTB(refgen.SKF_SETS[case.skf_set], kpts=case.kgrid, scc_tol=1e-10,
                        solver="eigh", dipole_correction={"layer_position": 5.0})
    atoms.get_potential_energy()
    assert atoms.calc.raw.energy == pytest.approx(res.energy, abs=1e-10)
    assert atoms.calc.raw.e_dipole_corr == pytest.approx(res.e_dipole_corr, abs=1e-12)


def test_dipole_refusals():
    sk = ParamSet(SKF_HCOCU).skset(["O", "H"])
    water = (["O", "H", "H"], np.array([[0.0, 0, 0], [1.8, 0, 0], [0, 1.8, 0]]))
    with pytest.raises(ValueError, match="periodic"):
        DFTB(sk, dipole_correction=True).calculate(*water)
    with pytest.raises(ValueError, match="charge neutral"):
        DFTB(sk, dipole_correction=True, charge=0.1)
    with pytest.raises(ValueError, match="fixed Fermi level"):
        DFTB(sk, dipole_correction=True, fixed_fermi_level=-0.2)
    with pytest.raises(ValueError, match="point charges or fields"):
        DFTB(sk, dipole_correction=True, electric_field=[0.0, 0.0, 0.01])
    tilted = np.diag([8.0, 8.0, 30.0])
    tilted[0, 2] = 0.5                                     # a1 with a z component
    with pytest.raises(ValueError, match="only one lattice vector"):
        DFTB(sk, dipole_correction=True).calculate(*water, tilted, np.ones(3, bool))
    leaning = np.diag([8.0, 8.0, 30.0])
    leaning[2, 0] = 0.5                                    # the normal vector not along z
    with pytest.raises(ValueError, match="apart from z"):
        DFTB(sk, dipole_correction=True).calculate(*water, leaning, np.ones(3, bool))
    calc = DFTB(sk, dipole_correction=True, force_evaluation="dynamics_t0")
    with pytest.raises(ValueError, match="XL-BOMD does not support the dipole correction"):
        calc.calculate(*water, np.diag([8.0, 8.0, 30.0]), np.ones(3, bool), xl=True)
    with pytest.raises(ValueError, match="dipole_correction: unknown keys"):
        DipoleCorrection.make({"z0": 1.0})
