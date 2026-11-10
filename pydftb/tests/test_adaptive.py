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


def _cu_model(angular=False):
    """Three CN basis functions; with ``angular`` two more of the angular descriptor (M8b)."""
    ang = RBF((0.02, 0.05), 0.02, constant=False) if angular else None
    x = (0.04, -0.03) if angular else ()
    return AdaptiveParams(
        "Cu", RBF((4.0, 8.0), 2.0),
        onsite={"s": (0.01, -0.02, 0.01) + x, "d": (-0.01, 0.02, 0.005) + x},
        hopping={"s": (0.05, -0.1, 0.04) + x, "p": (0.02, 0.03, -0.05) + x,
                 "d": (0.03, -0.02, 0.06) + x},
        repulsive=(0.1, -0.1, 0.05) + x, pair_repulsive={"Cu": PairRepulsive(6.0, (0.01, -0.002))},
        angular=ang)


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


@pytest.fixture(scope="module", params=[False, True], ids=["cn", "cn+angular"])
def cu_case(request):
    angular = request.param
    at = _cu3()
    sk = ParamSet(SK, max_l={"Cu": 2}, shell_resolved=True).skset(["Cu"])
    lab, pos, cell = at.get_chemical_symbols(), at.positions / BOHR, at.cell.array / BOHR

    def calc(m):
        return DFTB(sk, kpts=make_kpoints((2, 2, 2)), temperature=1000.0, scc_tol=1e-12,
                    adaptive=m)

    def free(m, p=pos, c=cell):
        return calc(m).calculate(lab, p, c, np.ones(3, bool), forces=False).free_energy

    r = calc(_cu_model(angular)).calculate(lab, pos, cell, np.ones(3, bool), forces=True,
                                           stress=True)
    return angular, pos, cell, free, r


def test_forces_and_stress_of_a_metal(cu_case):
    angular, pos, cell, free, r = cu_case
    m = _cu_model(angular)
    if angular:
        assert np.all(r.adaptive.q > 0.005)          # a distorted cell: no inversion centre
    fd = -_fd(lambda p: free(m, p), pos, 1e-5)
    assert np.abs(r.forces - fd).max() < 1e-8
    vol = abs(np.linalg.det(cell))

    def strained(eps):
        f = np.eye(3) + eps
        return free(m, pos @ f.T, cell @ f.T)

    sfd = -_fd(strained, np.zeros((3, 3)), 1e-6) / vol
    assert np.abs(r.stress - 0.5 * (sfd + sfd.T)).max() < 1e-9


def test_coefficient_derivatives(cu_case):
    angular, _, _, free, r = cu_case
    m = _cu_model(angular)
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


def test_angular_descriptor():
    """q: about 1 for a dimer, 5/8 for graphite, 5/9 for diamond, 0 with inversion symmetry;
    its gradient against central differences."""
    from pydftb.validation import ptbp_workload as pw
    m = AdaptiveParams("Cu", RBF(()), angular=RBF((0.5,), 0.2, constant=False))
    want = {"dimer": 0.990, "AA-graphite": 0.624, "diamond": 0.555, "bcc": 0.0, "fcc": 0.0}
    for s in pw.structures(quick=True, phases=list(want)):
        if s.eos != 5:
            continue
        a = s.atoms
        _, q, _, _ = m.descriptors(a.get_chemical_symbols(), a.positions / BOHR,
                                   a.cell.array / BOHR, np.ones(3, bool))
        assert np.abs(q - want[s.phase]).max() < 2e-3, (s.phase, q)
    at = _cu3()
    lab, pos, cell = at.get_chemical_symbols(), at.positions / BOHR, at.cell.array / BOHR
    _, q, _, (i, j, _, dq) = m.descriptors(lab, pos, cell, np.ones(3, bool))
    g = np.zeros((3, 3, 3))
    np.add.at(g, (i, j), dq)
    np.add.at(g, (i, i), -dq)
    num = _fd(lambda p: np.array(m.descriptors(lab, p.reshape(3, 3), cell, np.ones(3, bool))[1]
                                 [0]), pos.ravel(), 1e-6)
    assert np.abs(num.reshape(3, 3) - g[0]).max() < 1e-8
    with pytest.raises(ValueError, match="constant"):
        AdaptiveParams("Cu", angular=RBF((0.5,), 0.2))


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


# ---- forces in the fit (M8b) -------------------------------------------------------------------------
def _distorted_waters():
    out = []
    for k, (d, ang) in enumerate([(0.92, 100), (1.03, 109), (1.10, 115), (0.97, 125)]):
        w = molecule("H2O")
        w.set_distance(0, 1, d, fix=0)
        w.set_angle(1, 0, 2, ang)
        w.positions[2] += [0.02 * k, -0.01, 0.015]
        out.append(w)
    return out


def test_force_jacobian_is_the_gradient_of_dF_dtheta(params):
    """d(forces)/dtheta by central differences in theta equals -grad_R (dF/dtheta), the
    position derivative of the analytic coefficient derivatives (symmetric second derivatives)."""
    from pydftb.adaptive_fit import evaluate_all
    sk = params.skset(["O", "H"])
    m = AdaptiveParams("O", RBF((1.0,), 0.6), onsite={"p": (0.02, -0.03)},
                       hopping={"p": (0.05, 0.04)})
    w = _distorted_waters()[1]
    s = Sample(w, 0.0, forces=np.zeros((3, 3)))
    opts = {"scc_tol": 1e-12}
    x = m.vector()
    _, _, _, fjac = evaluate_all([m], [s], sk, opts, x, processes=1, step=1e-5)
    lab = w.get_chemical_symbols()
    pos = w.positions / BOHR

    def dfdx(p):                                   # dF/dtheta (eV) at positions p (Bohr)
        r = DFTB(sk, adaptive=[m], **opts).calculate(lab, p, forces=False)
        return r.adaptive.gradient([m]) * 27.2113845

    ref = np.zeros((9, len(x)))
    h = 1e-5
    for a in range(3):
        for c in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[a, c] += h
            pm[a, c] -= h
            ref[3 * a + c] = -(dfdx(pp) - dfdx(pm)) / (2 * h * BOHR)    # eV/A per unit theta
    assert np.abs(fjac[0] - ref).max() < 1e-5 * max(1.0, np.abs(ref).max())


def test_fit_with_forces_recovers_a_model(params):
    truth = AdaptiveParams("O", RBF((1.0,), 0.6), onsite={"p": (0.02, -0.03)},
                           hopping={"p": (0.05, 0.04)})
    sk = params.skset(["O", "H"])
    data = []
    for k, w in enumerate(_distorted_waters()):
        lab = w.get_chemical_symbols()
        r = DFTB(sk, scc_tol=1e-11, adaptive=truth).calculate(lab, w.positions / BOHR,
                                                              forces=True)
        data.append(Sample(w, r.free_energy * 27.2113845 / 3, name=str(k),
                           forces=r.forces * 27.2113845 / BOHR))
    assert max(np.abs(s.forces).max() for s in data) > 0.5            # far from equilibrium
    start = truth.with_vector(np.zeros(4))
    fit = fit_adaptive([start], data, sk, {"scc_tol": 1e-11}, l2=0.0, processes=1,
                       max_evaluations=40, force_weight=1.0, force_step=1e-5)
    assert fit.force_rmse < 1e-5 and fit.rmse < 1e-6
    assert np.abs(fit.x - truth.vector()).max() < 1e-4


def test_fit_rejects_a_step_where_the_scc_fails(params, monkeypatch):
    """A trial point whose SCC does not converge is a rejected step, not the end of the fit."""
    import pydftb.adaptive_fit as af
    m = AdaptiveParams("O", RBF(()), onsite={"p": (0.0,)})
    calls = {"n": 0, "failed": 0}

    def fake(models, samples, sk, options, x, processes=1, with_forces=True, step=1e-4):
        calls["n"] += 1
        if x[0] > 0.6:                       # beyond here the SCC "does not converge"
            calls["failed"] += 1
            raise RuntimeError("SCC not converged for a sample")
        e = np.array([x[0], -x[0]])          # optimum (without the wall) at x = 1
        return e, np.array([[1.0], [-1.0]]), {}, {}

    monkeypatch.setattr(af, "evaluate_all", fake)
    samples = [Sample(None, 1.0), Sample(None, -1.0)]
    r = af.fit_adaptive([m], samples, None, {}, l2=0.0, max_evaluations=30)
    assert calls["failed"] >= 1
    assert r.x[0] <= 0.6 and r.x[0] > 0.3
    calls["n"] = 0
    with pytest.raises(RuntimeError):          # a failure at the start still raises
        af.fit_adaptive([m.with_vector(np.array([0.9]))], samples, None, {}, l2=0.0,
                        max_evaluations=5)


def test_a_failed_scc_is_retried_with_gentler_mixing(params):
    """A sample whose SCC does not converge at the default mixing is retried (gentler mixing,
    more iterations) before it counts as failed; an explicit mixing turns the retry off."""
    from pydftb.adaptive_fit import evaluate_sample
    m = AdaptiveParams("O", RBF(()), onsite={"p": (0.0,)})
    s = Sample(molecule("H2O"), 0.0, name="water")
    sk = params.skset(["O", "H"])
    e, _, _ = evaluate_sample([m], s, sk, {"scc_tol": 1e-10, "max_scc": 2})
    ref, _, _ = evaluate_sample([m], s, sk, {"scc_tol": 1e-10})
    assert abs(e - ref) < 1e-8
    with pytest.raises(RuntimeError):
        evaluate_sample([m], s, sk, {"scc_tol": 1e-10, "max_scc": 2, "mixing": 0.2})


def test_rbf_clamp():
    """A clamped basis is the plain one inside the range, constant outside, smooth between."""
    plain = RBF((2.0, 6.0), 2.0)
    cl = RBF((2.0, 6.0), 2.0, clamp=(1.0, 8.0))
    x = np.array([2.0, 4.0, 7.0])
    assert np.allclose(cl(x)[0], plain(x)[0]) and np.allclose(cl(x)[1], plain(x)[1])
    out = np.array([-3.0, 0.2, 9.0, 20.0])
    assert np.allclose(cl(out)[0], plain(np.array([1.0, 1.0, 8.0, 8.0]))[0])   # the bounds
    assert np.allclose(cl(out)[1], 0.0)
    xs = np.linspace(0.0, 9.0, 901)                       # value and slope continuous
    ph, dph = cl(xs)
    fd = np.gradient(ph, xs, axis=0)
    assert np.abs(fd[1:-1] - dph[1:-1]).max() < 1e-3
    with pytest.raises(ValueError):
        RBF((2.0,), 1.0, clamp=(5.0, 5.5))




def test_joint_repulsive_force_rows_match_the_energy_rows():
    """cu_joint_repulsive: the force design is minus the gradient of the energy design."""
    from ase import Atoms
    from pydftb.bench import cu_joint_repulsive as cj
    rng = np.random.default_rng(3)
    at = Atoms("Cu4", positions=[[0, 0, 0], [2.3, 0, 0], [0, 2.4, 0.2], [1.2, 1.1, 2.0]])
    at.positions += rng.normal(0, 0.05, (4, 3))
    t, n, keep = cj.basis(1.6, 3.5, 8)
    f = cj.force_design(at, t, n, keep, 3.5)
    h = 1e-5
    for a in range(4):
        for x in range(3):
            p, q = at.copy(), at.copy()
            p.positions[a, x] += h
            q.positions[a, x] -= h
            de = (cj.design([type("S", (), {"atoms": p})], t, n, keep, 3.5)
                  - cj.design([type("S", (), {"atoms": q})], t, n, keep, 3.5))[0] * 4 / (2 * h)
            assert np.allclose(f[3 * a + x], -de, atol=1e-6)


def test_joint_repulsive_monotonic_solve():
    from pydftb.bench import cu_joint_repulsive as cj
    rng = np.random.default_rng(0)
    a = rng.random((40, 6))
    y = a @ np.array([3.0, 1.0, 2.0, 0.5, -0.2, 0.1]) + 0.3       # not monotone
    c, _ = cj.solve(a, y, None, None, 6, 1e-6, True)
    assert np.all(np.diff(c) <= 1e-9) and np.all(c >= -1e-9)
    c2, k = cj.solve(a, y, None, None, 6, 1e-12, False)
    assert np.allclose(c2, [3.0, 1.0, 2.0, 0.5, -0.2, 0.1], atol=1e-4) and abs(k - 0.3) < 1e-4


# ---- descriptor models (A series, A0 section 4) ----------------------------------------------------
def _desc_models(regressor="linear"):
    """O: radial + angular functions, H: its coordination number, with a linear, kernel or MLP
    regressor driving on-site, hopping and repulsive channels."""
    from pydftb.descriptors import Angular, CoordinationNumber, Radial, Stack
    from pydftb.regressors import MLP, Kernel, Linear
    d_o = Stack((Radial("O", centres=(1.6, 2.4), width=0.6, cutoff=6.0),
                 Angular("O", zetas=(1.0, 2.0), lambdas=(1.0,), cutoff=6.0)))
    d_h = CoordinationNumber("H", window=(0.6, 1.6))
    ch_o = ("onsite.s", "onsite.p", "hopping.p", "repulsive")
    ch_h = ("onsite.s", "hopping.s")
    rng = np.random.default_rng(7)
    if regressor == "linear":
        r_o, r_h = Linear(d_o.size, len(ch_o)), Linear(1, len(ch_h))
    elif regressor == "kernel":
        r_o = Kernel(tuple(map(tuple, rng.normal(size=(3, d_o.size)))), 1.5, len(ch_o))
        r_h = Kernel(((0.5,), (1.0,)), 0.5, len(ch_h))
    else:
        r_o, r_h = MLP(d_o.size, len(ch_o), hidden=(3,), seed=1), MLP(1, len(ch_h), (2,), seed=2)
    t_o = 0.02 * rng.normal(size=r_o.size)
    t_h = 0.02 * rng.normal(size=r_h.size)
    if regressor == "mlp":                       # keep the hidden layers of initial()
        t_o = r_o.initial() + t_o
        t_h = r_h.initial() + t_h
    o = AdaptiveParams("O", descriptor=d_o, regressor=r_o, channels=ch_o, theta=tuple(t_o),
                       pair_repulsive={"H": PairRepulsive(3.0, (0.02, -0.01))})
    h = AdaptiveParams("H", descriptor=d_h, regressor=r_h, channels=ch_h, theta=tuple(t_h))
    return [o, h]


def test_zero_descriptor_model_is_the_plain_calculation(params):
    lab, pos, plain = _water(params)
    zero = [m.with_vector(np.zeros(len(m.vector()))) for m in _desc_models()]
    _, _, calc = _water(params, adaptive=zero)
    a, b = plain.calculate(lab, pos, forces=True), calc.calculate(lab, pos, forces=True)
    assert a.energy == b.energy and np.array_equal(a.forces, b.forces)


@pytest.mark.parametrize("regressor", ["linear", "kernel", "mlp"])
def test_descriptor_model_forces_and_coefficient_derivatives(params, regressor):
    models = _desc_models(regressor)
    lab, pos, calc = _water(params, adaptive=models)
    r = calc.calculate(lab, pos, forces=True)
    fd = -_fd(lambda p: calc.calculate(lab, p, forces=False).free_energy, pos, 1e-5)
    assert np.abs(r.forces - fd).max() < 2e-9 and np.abs(r.forces).max() > 1e-3
    g = r.adaptive.gradient(models)
    x0 = np.concatenate([m.vector() for m in models])

    def free(x):
        ms = [models[0].with_vector(x[:len(models[0].vector())]),
              models[1].with_vector(x[len(models[0].vector()):])]
        return _water(params, adaptive=ms)[2].calculate(lab, pos, forces=False).free_energy

    num = _fd(free, x0, 1e-6)
    assert np.abs(g - num).max() < 1e-7 * max(1.0, np.abs(num).max())
    assert set(r.adaptive.dedp) >= {("onsite", "O", "s"), ("hopping", "H", "s"), ("repulsive", "O")}


def test_descriptor_model_forces_and_stress_of_a_metal():
    from pydftb.descriptors import Angular, Radial, Stack
    from pydftb.regressors import Kernel
    at = _cu3()
    sk = ParamSet(SK, max_l={"Cu": 2}, shell_resolved=True).skset(["Cu"])
    lab, pos, cell = at.get_chemical_symbols(), at.positions / BOHR, at.cell.array / BOHR
    d = Stack((Radial("Cu", centres=(4.5, 6.0), width=1.0, cutoff=9.0),
               Angular("Cu", zetas=(1.0,), lambdas=(1.0, -1.0), cutoff=8.0)))
    reg = Kernel(tuple(map(tuple, np.random.default_rng(3).normal(1.0, 0.5, (3, d.size)))), 2.0, 3)
    m = AdaptiveParams("Cu", descriptor=d, regressor=reg, channels=("onsite.d", "hopping.s",
                                                                    "repulsive"),
                       theta=tuple(0.03 * np.random.default_rng(4).normal(size=reg.size)))

    def calc():
        return DFTB(sk, kpts=make_kpoints((2, 2, 2)), temperature=1000.0, scc_tol=1e-12,
                    adaptive=m)

    def free(p=pos, c=cell):
        return calc().calculate(lab, p, c, np.ones(3, bool), forces=False).free_energy

    r = calc().calculate(lab, pos, cell, np.ones(3, bool), forces=True, stress=True)
    fd = -_fd(lambda p: free(p), pos, 1e-5)
    assert np.abs(r.forces - fd).max() < 1e-8
    vol = abs(np.linalg.det(cell))

    def strained(eps):
        f = np.eye(3) + eps
        return free(pos @ f.T, cell @ f.T)

    sfd = -_fd(strained, np.zeros((3, 3)), 1e-6) / vol
    assert np.abs(r.stress - 0.5 * (sfd + sfd.T)).max() < 1e-9


def test_descriptor_model_vectors_json_and_refusals(params):
    import json
    from dataclasses import dataclass
    from pydftb import envmodel
    from pydftb.adaptive import channel_jacobians
    from pydftb.descriptors import CoordinationNumber
    from pydftb.regressors import Linear
    models = _desc_models("mlp")
    for m in models:
        x = np.arange(len(m.vector()), dtype=float) * 0.01
        assert np.array_equal(m.with_vector(x).vector(), x)
        assert len(m.parameter_names()) == len(x)
        back = envmodel.model_from_dict(json.loads(json.dumps(m.to_dict())))
        assert np.array_equal(back.vector(), m.vector()) and back.channels == m.channels
    # dF/dtheta = sum over channels of dF/dp_A dp_A/dtheta (channel_jacobians)
    lab, pos, calc = _water(params, adaptive=models)
    r = calc.calculate(lab, pos, forces=False)
    st = calc._setup(lab, pos, None, None, False, {}, matrices=False)
    for m in models:
        jac = channel_jacobians(m, st.adaptive)
        g = sum(r.adaptive.dedp[key] @ jm for key, (_, jm) in jac.items())
        n = len(m.regressor.names)
        assert np.allclose(g[:n], r.adaptive.gradients[m.element][:n], rtol=1e-10, atol=1e-14)

    @dataclass(frozen=True)
    class NoDerivatives(CoordinationNumber):
        has_derivatives = False

    m = AdaptiveParams("O", descriptor=NoDerivatives("O"), channels=("onsite.p",),
                       theta=(0.01, 0.02))
    lab, pos, calc = _water(params, adaptive=[m])
    calc.calculate(lab, pos, forces=False)                     # energies are fine
    with pytest.raises(ValueError, match="no position derivatives"):
        calc.calculate(lab, pos, forces=True)
    with pytest.raises(ValueError, match="channels"):
        AdaptiveParams("O", descriptor=CoordinationNumber("O"))
    with pytest.raises(ValueError, match="channel"):
        AdaptiveParams("O", descriptor=CoordinationNumber("O"), channels=("onsite.x",))
    with pytest.raises(ValueError, match="inputs"):
        AdaptiveParams("O", descriptor=CoordinationNumber("O"), regressor=Linear(2),
                       channels=("onsite.p",))
    with pytest.raises(ValueError, match="describes"):
        AdaptiveParams("H", descriptor=CoordinationNumber("O"), channels=("onsite.s",))
    with pytest.raises(ValueError, match="not onsite"):
        AdaptiveParams("O", descriptor=CoordinationNumber("O"), channels=("onsite.p",),
                       onsite={"p": (0.0,)})
