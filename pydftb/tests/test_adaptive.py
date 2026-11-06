"""Environment-adaptive parameters (milestone M8a): zero models change no bit, forces, stress and
the coefficient derivatives are the derivatives of the free energy, refusals, the fit."""
import numpy as np
import pytest
from ase.build import bulk, molecule

from pydftb import DFTB, ParamSet, make_kpoints
from pydftb.adaptive import RBF, AdaptiveParams, PairRepulsive
from pydftb.adaptive_fit import Sample, fit_adaptive
from pydftb.ase_calc import PyDFTB
from pydftb.constants import BOHR
from pydftb.validation.cases import SKF_SETS

SK = SKF_SETS["ptbp_hcocu"]


@pytest.fixture(scope="module")
def params():
    return ParamSet(SK, max_l={"H": 0, "C": 1, "O": 1, "Cu": 2})


def _water(params, **kw):
    w = molecule("H2O")
    w.positions[1] += [0.05, -0.03, 0.02]                     # no symmetry
    lab = w.get_chemical_symbols()
    return lab, w.positions / BOHR, DFTB(params.skset(lab), scc_tol=1e-12, **kw)


def _models():
    o = AdaptiveParams("O", RBF((1.0, 2.0), 0.7),
                       onsite={"s": (0.01, -0.02, 0.03), "p": (-0.01, 0.02, 0.015)},
                       hopping={"s": (0.05, -0.1, 0.08), "p": (0.03, 0.06, -0.05)},
                       repulsive=(0.1, -0.2, 0.15),
                       pair_repulsive={"H": PairRepulsive(3.0, (0.02, -0.01, 0.005),
                                                        wall=2.2, wall_strength=0.5)})
    h = AdaptiveParams("H", RBF((1.0,), 0.5), onsite={"s": (0.01, 0.02)},
                       hopping={"s": (0.04, -0.03)})
    return [o, h]


def _no_walls(m):
    """The model with zero coefficients and without the fixed walls (not coefficients)."""
    from dataclasses import replace
    return replace(m, pair_repulsive={el: replace(r, wall=0.0, wall_strength=0.0)
                                      for el, r in m.pair_repulsive.items()})


def _cu3():
    at = bulk("Cu", "fcc", a=3.7, cubic=True)
    at.rattle(0.08, seed=1)
    at.cell[0, 1] += 0.15
    return at[:3]


def _cu_model():
    return AdaptiveParams(
        "Cu", RBF((4.0, 8.0), 2.0),
        onsite={"s": (0.01, -0.02, 0.01), "d": (-0.01, 0.02, 0.005)},
        hopping={"s": (0.05, -0.1, 0.04), "p": (0.02, 0.03, -0.05), "d": (0.03, -0.02, 0.06)},
        repulsive=(0.1, -0.1, 0.05), pair_repulsive={"Cu": PairRepulsive(6.0, (0.01, -0.002))})


def _fd(f, x, h):
    g = np.zeros_like(x)
    for k in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[k] += h
        xm[k] -= h
        g[k] = (f(xp) - f(xm)) / (2 * h)
    return g


# ---- zero models ----------------------------------------------------------------------------------
def test_zero_model_is_the_plain_calculation(params):
    lab, pos, plain = _water(params)
    zero = [_no_walls(m.with_vector(np.zeros(len(m.vector())))) for m in _models()]
    _, _, calc = _water(params, adaptive=zero)
    a, b = plain.calculate(lab, pos, forces=True), calc.calculate(lab, pos, forces=True)
    assert a.energy == b.energy and a.free_energy == b.free_energy
    assert np.array_equal(a.forces, b.forces) and np.array_equal(a.charges, b.charges)
    at = _cu3()
    sk = ParamSet(SK, max_l={"Cu": 2}).skset(["Cu"])
    args = (at.get_chemical_symbols(), at.positions / BOHR, at.cell.array / BOHR, np.ones(3, bool))
    kw = {"kpts": make_kpoints((2, 2, 2)), "temperature": 1000.0}
    zero = _cu_model().with_vector(np.zeros(len(_cu_model().vector())))
    a = DFTB(sk, **kw).calculate(*args, forces=True, stress=True)
    b = DFTB(sk, adaptive=zero, **kw).calculate(*args, forces=True, stress=True)
    assert a.energy == b.energy and np.array_equal(a.forces, b.forces)
    assert np.array_equal(a.stress, b.stress)


# ---- derivatives ------------------------------------------------------------------------------------
def test_forces_of_a_molecule(params):
    lab, pos, calc = _water(params, adaptive=_models())
    r = calc.calculate(lab, pos, forces=True)
    fd = -_fd(lambda p: calc.calculate(lab, p, forces=False).free_energy, pos, 1e-5)
    assert np.abs(r.forces - fd).max() < 2e-9
    cn = r.adaptive.cn                     # H-H is outside the window: CN_O = CN_H1 + CN_H2
    assert abs(cn[0] - cn[1] - cn[2]) < 1e-14 and 0.5 < cn[1] < 1.0 and 0.5 < cn[2] < 1.0


@pytest.mark.parametrize("kw", [
    {"solver": "ntpoly_trs2"}, {"solver": "foe", "temperature": 1000.0},
    {"third_order": "full", "hubbard_derivs": {"O": -0.1575, "H": -0.1857}, "h_damping": 4.0},
    {"dispersion": {"model": "lennardjones"}}], ids=["ntpoly", "foe", "dftb3", "lj"])
def test_forces_with_other_solvers_and_terms(params, kw):
    from pydftb.purification import NTPoly
    kw = dict(kw)
    if kw.get("solver") == "ntpoly_trs2":
        kw["solver"] = NTPoly("trs2", tolerance=1e-13, truncation=1e-16)
    lab, pos, calc = _water(params, adaptive=_models(), **kw)
    r = calc.calculate(lab, pos, forces=True)
    fd = -_fd(lambda p: calc.calculate(lab, p, forces=False).free_energy, pos, 1e-5)
    assert np.abs(r.forces - fd).max() < 2e-9


def test_forces_with_collinear_spin(params):
    lab = ["O", "O"]
    pos = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 2.3]])
    m = AdaptiveParams("O", RBF((1.0,), 0.5), onsite={"p": (0.01, 0.03)},
                       hopping={"p": (0.02, -0.05)})
    calc = DFTB(params.skset(["O"]), spin=True, unpaired_electrons=2,
                spin_constants={"O": -0.028}, scc_tol=1e-12, adaptive=m)
    r = calc.calculate(lab, pos, forces=True)
    fd = -_fd(lambda p: calc.calculate(lab, p, forces=False).free_energy, pos, 1e-5)
    assert np.abs(r.forces - fd).max() < 2e-9


@pytest.fixture(scope="module")
def cu_case():
    at = _cu3()
    sk = ParamSet(SK, max_l={"Cu": 2}, shell_resolved=True).skset(["Cu"])
    lab, pos, cell = at.get_chemical_symbols(), at.positions / BOHR, at.cell.array / BOHR

    def calc(m):
        return DFTB(sk, kpts=make_kpoints((2, 2, 2)), temperature=1000.0, scc_tol=1e-12,
                    adaptive=m)

    def free(m, p=pos, c=cell):
        return calc(m).calculate(lab, p, c, np.ones(3, bool), forces=False).free_energy

    r = calc(_cu_model()).calculate(lab, pos, cell, np.ones(3, bool), forces=True, stress=True)
    return lab, pos, cell, free, r


def test_forces_and_stress_of_a_metal(cu_case):
    _, pos, cell, free, r = cu_case
    m = _cu_model()
    fd = -_fd(lambda p: free(m, p), pos, 1e-5)
    assert np.abs(r.forces - fd).max() < 1e-8
    vol = abs(np.linalg.det(cell))

    def strained(eps):
        f = np.eye(3) + eps
        return free(m, pos @ f.T, cell @ f.T)

    sfd = -_fd(strained, np.zeros((3, 3)), 1e-6) / vol
    assert np.abs(r.stress - 0.5 * (sfd + sfd.T)).max() < 1e-9


def test_coefficient_derivatives(cu_case):
    *_, free, r = cu_case
    m = _cu_model()
    x0 = m.vector()
    g = r.adaptive.gradient([m])
    fd = _fd(lambda x: free(m.with_vector(x)), x0, 1e-6)
    assert len(g) == len(m.parameter_names())
    assert np.abs(g - fd).max() < 1e-7 * max(1.0, np.abs(fd).max())


def test_ase_calculator(params):
    w = molecule("H2O")
    w.calc = PyDFTB(SK, max_l={"H": 0, "O": 1}, scc_tol=1e-12, solver="eigh", adaptive=_models())
    e = w.get_potential_energy()
    f = w.get_forces()
    lab, _, calc = _water(params, adaptive=_models())
    r = calc.calculate(lab, molecule("H2O").positions / BOHR, forces=True)
    from ase import units
    assert abs(e - r.energy * units.Hartree) < 1e-7 and np.abs(f).max() > 0


# ---- the model --------------------------------------------------------------------------------------
def test_basis_and_vectors():
    b = RBF((2.0, 5.0), 1.3)
    cn = np.array([0.0, 2.1, 4.7, 9.0])
    phi, dphi = b(cn)
    num = (b(cn + 1e-6)[0] - b(cn - 1e-6)[0]) / 2e-6
    assert phi.shape == (4, 3) and np.abs(dphi - num).max() < 1e-8 and np.all(phi[:, 0] == 1)
    m = _cu_model()
    x = np.arange(len(m.vector()), dtype=float)
    assert np.array_equal(m.with_vector(x).vector(), x)
    assert len(m.parameter_names()) == len(x)
    rep = PairRepulsive(4.0, (1.0, 2.0))
    v, dv = rep(np.array([1.0, 3.9, 4.0, 5.0]))
    assert v[2] == v[3] == 0.0 and abs(v[0] - (9.0 + 2 * 27.0)) < 1e-12
    assert abs(dv[0] - (-(2 * 3.0) - 2 * 3 * 9.0)) < 1e-12
    walled = PairRepulsive(4.0, (1.0, 2.0), wall=2.0, wall_strength=3.0)
    r = np.array([1.0, 1.9, 2.0, 2.5])
    v2, dv2 = walled(r)
    assert np.array_equal(v2[2:], rep(r[2:])[0]) and abs(v2[0] - v[0] - 3.0) < 1e-12
    num = (walled(r + 1e-6)[0] - walled(r - 1e-6)[0]) / 2e-6
    assert np.abs(dv2 - num).max() < 1e-6
    assert walled.coefficients == rep.coefficients          # the wall is not a coefficient
    m = AdaptiveParams("Cu", RBF(()), pair_repulsive={"Cu": walled})
    assert m.with_vector(m.vector() * 2).pair_repulsive["Cu"].wall == 2.0
    with pytest.raises(ValueError, match="wall"):
        PairRepulsive(4.0, (1.0,), wall=5.0, wall_strength=1.0)


def test_refusals(params):
    sk = params.skset(["O", "H"])
    m = _models()
    with pytest.raises(ValueError, match="two-component"):
        DFTB(sk, spin="noncolinear", spin_constants={"O": -0.028, "H": -0.072}, adaptive=m)
    with pytest.raises(ValueError, match="analysis"):
        DFTB(sk, analysis=True, adaptive=m)
    with pytest.raises(ValueError, match="one AdaptiveParams per element"):
        DFTB(sk, adaptive=[m[0], m[0]])
    with pytest.raises(ValueError, match="coefficients"):
        AdaptiveParams("O", RBF((1.0,)), onsite={"s": (0.1,)})
    o2 = AdaptiveParams("H", RBF(()), pair_repulsive={"O": PairRepulsive(3.0, (0.1,))})
    lab, pos, _ = _water(params)
    with pytest.raises(ValueError, match="given twice"):
        DFTB(sk, adaptive=[m[0], o2]).calculate(lab, pos)


# ---- the fit ----------------------------------------------------------------------------------------
def test_fit_recovers_a_model(params):
    """Energies from a known model; a fit from zero finds them again."""
    truth = AdaptiveParams("O", RBF((1.0,), 0.6), onsite={"p": (0.02, -0.03)},
                           hopping={"p": (0.05, 0.04)})
    sk = params.skset(["O", "H"])
    data = []
    for k, (d, ang) in enumerate([(0.92, 100), (0.97, 104), (1.03, 109), (1.10, 115),
                                  (0.96, 140), (1.3, 104)]):
        w = molecule("H2O")
        w.set_distance(0, 1, d, fix=0)
        w.set_angle(1, 0, 2, ang)
        lab = w.get_chemical_symbols()
        r = DFTB(sk, scc_tol=1e-11, adaptive=truth).calculate(lab, w.positions / BOHR)
        data.append(Sample(w, r.free_energy * 27.2113845 / 3, name=str(k)))
    start = truth.with_vector(np.zeros(4))
    fit = fit_adaptive([start], data, sk, {"scc_tol": 1e-11}, l2=0.0, processes=1,
                       max_evaluations=60)
    assert fit.rmse < 1e-6
