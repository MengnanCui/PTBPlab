"""DFTB3: Gamma functions, third-order energy/potential, forces, stress, H-X damping.

Comparisons with DFTB+ are the frozen references *_dftb3*, water_dimer_hdamp in
test_reference.py; here only self-consistency (no dftb+, no mpmath needed).
"""
import numpy as np
import pytest

from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.coulomb import _gamma0, _short, _short_near
from pydftb.params import ParamSet
from pydftb.thirdorder import _short_terms, dgamma_du, dgamma_du_onsite
from pydftb.validation import refgen
from pydftb.validation.cases import UD_3OB

from .conftest import SKF_HCOCU


def test_near_equal_tau_form_matches_closed_form_and_its_limit(monkeypatch):
    import pydftb.thirdorder as third
    r = np.array([0.7, 3.0, 9.0])
    ta = np.full(3, 1.3)
    with monkeypatch.context() as m:
        m.setattr(third, "NEAR_TAU", 0.0)             # force DFTB+'s closed form
        for rel in (0.03, 0.05, 0.1):                 # where the closed form is still accurate
            tb = ta * (1 + rel)
            s, st, sr, _ = _short_terms(ta, tb, r, np.zeros(3, bool))
            s_n, sr_n, st_n, _ = _short_near(r, ta, tb, tau_derivs=True)
            assert np.allclose(s_n, s, rtol=1e-8)
            assert np.allclose(sr_n, sr, rtol=1e-8)
            assert np.allclose(st_n, st, rtol=1e-8)
    # tau_b -> tau_a: S tends to DFTB+'s equal-U formula; dS/dtau_a to half its total derivative
    tb = ta * (1 + 1e-7)
    s_near, _, st_near, _ = _short_near(r, ta, tb, tau_derivs=True)
    s_eq, st_eq, _, _ = _short_terms(ta, ta, r, np.ones(3, bool))
    assert np.allclose(s_near, s_eq, rtol=1e-6)
    assert np.allclose(st_near, 0.5 * st_eq, rtol=1e-6)


def test_nearly_equal_hubbard_u_is_stable():
    """U differing by 4e-6 (just above DFTB+'s 'equal' threshold): DFTB+'s closed form
    gives a dgamma/dU that is wrong by orders of magnitude; ours stays at half the
    equal-U (total derivative) value."""
    g_eq, _ = dgamma_du(0.41, 0.41, 5.5)
    g_near, _ = dgamma_du(0.41, 0.41 + 4e-6, 5.5)
    assert g_near == pytest.approx(0.5 * g_eq, rel=1e-4)
    s_eq, _ = _short(np.array([5.5]), np.array([1.312]), np.array([1.312]))
    s_near, _ = _short(np.array([5.5]), np.array([1.312]), np.array([1.312 * (1 + 1e-5)]))
    assert s_near[0] == pytest.approx(s_eq[0], rel=1e-4)


@pytest.mark.parametrize("damped", [False, True])
@pytest.mark.parametrize("ub", [0.41, 0.41 + 2e-3, 0.62])
def test_gamma3_r_derivative(ub, damped):
    r = np.array([1.5, 3.0, 6.0])
    _, gr = dgamma_du(0.41, ub, r, damped, 4.0)
    h = 1e-6
    fd = (dgamma_du(0.41, ub, r + h, damped, 4.0)[0] - dgamma_du(0.41, ub, r - h, damped, 4.0)[0]) / (2 * h)
    assert np.allclose(gr, fd, rtol=1e-6, atol=1e-12)


def test_gamma3_onsite_is_partial_u_derivative_of_gamma0():
    ua, ub, h = 0.36, 0.51, 1e-6
    fd = (_gamma0(3.2 * (ua + h), 3.2 * ub) - _gamma0(3.2 * (ua - h), 3.2 * ub)) / (2 * h)
    assert float(dgamma_du_onsite(ua, ub)) == pytest.approx(fd, rel=1e-8)
    assert float(dgamma_du_onsite(ua, ua)) == 0.5


def _system(name):
    case = refgen.CASES_BY_NAME[name]
    inp = refgen._inputs(case)
    params = ParamSet(refgen.SKF_SETS[case.skf_set], max_l=inp["max_l"],
                      shell_resolved=case.shell_resolved)
    kpts = None if inp["kpoints"] is None else (np.array(inp["kpoints"]), np.array(inp["kweights"]))
    cell = None if inp["cell_angstrom"] is None else np.array(inp["cell_angstrom"]) / BOHR
    return (case, params.skset(inp["labels"]), inp["labels"],
            np.array(inp["positions_angstrom"]) / BOHR, cell, kpts)


@pytest.mark.parametrize("name", ["water_dimer_dftb3_nodamp", "cu4co_dftb3_shell"])
def test_third_order_potential_is_energy_derivative(name):
    case, sk, labels, pos, cell, kpts = _system(name)
    calc = DFTB(sk, smearing=tuple(case.filling), **case.dftb3)
    st = calc._setup(labels, pos, cell, None, False, {})
    rng = np.random.default_rng(0)
    dq = rng.normal(scale=0.2, size=len(st.q0))
    _, v = st.third.energy_shift(dq)
    h = 1e-6
    for c in range(len(dq)):
        e = np.zeros_like(dq)
        e[c] = h
        fd = (st.third.energy_shift(dq + e)[0] - st.third.energy_shift(dq - e)[0]) / (2 * h)
        assert v[c] == pytest.approx(fd, rel=1e-7, abs=1e-11)


def test_onsite_third_order_energy():
    case, sk, labels, pos, cell, kpts = _system("h2o_dftb3_onsite")
    r = DFTB(sk, scc_tol=1e-10, **case.dftb3).calculate(labels, pos)
    q0 = np.array([sk.occupation[lab].sum() for lab in labels])
    dq = r.charges - q0
    ud = np.array([UD_3OB[lab] for lab in labels])
    assert r.e_3rd == pytest.approx((ud / 6 * dq ** 3).sum(), rel=1e-9)


# In this water the H-H distance is 4e-6 Bohr from an SK grid node: a difference across
# the node mixes two interpolation polynomials (6e-7 error at h = 1e-5), hence h = 1e-6.
@pytest.mark.parametrize("name,h", [("h2o_dftb3", 1e-6), ("water_dimer_hdamp", 1e-5),
                                    ("cu4co_dftb3_shell", 1e-5)])
def test_dftb3_forces_match_energy_finite_differences(name, h):
    case, sk, labels, pos, cell, kpts = _system(name)
    calc = DFTB(sk, smearing=tuple(case.filling), scc_tol=1e-12, max_scc=500, **case.dftb3)
    f = calc.calculate(labels, pos).forces
    for atom, ax in [(0, 0), (1, 2), (len(labels) - 1, 1)]:
        e = []
        for sgn in (1, -1):
            p = pos.copy()
            p[atom, ax] += sgn * h
            e.append(calc.calculate(labels, p, forces=False).free_energy)
        assert -(e[0] - e[1]) / (2 * h) == pytest.approx(f[atom, ax], abs=1e-8)


def test_dftb3_stress_matches_strain_finite_differences():
    case, sk, labels, pos, cell, kpts = _system("cuo_chain_dftb3_k422")
    calc = DFTB(sk, kpts=kpts, smearing=tuple(case.filling), scc_tol=1e-12, max_scc=500,
                **case.dftb3)
    pbc = np.ones(3, bool)
    w = -calc.calculate(labels, pos, cell, pbc).stress * abs(np.linalg.det(cell))
    h = 1e-5
    for a, b in [(0, 0), (1, 2)]:
        eps = np.zeros((3, 3))
        eps[a, b] += h / 2
        eps[b, a] += h / 2
        e = [calc.calculate(labels, pos @ (np.eye(3) + s * eps).T, cell @ (np.eye(3) + s * eps).T,
                            pbc, forces=False).free_energy for s in (1, -1)]
        assert (e[0] - e[1]) / (2 * h) == pytest.approx(w[a, b], abs=5e-8)


def test_dftb3_options_are_checked():
    _, sk, labels, pos, _, _ = _system("h2o_dftb3")
    with pytest.raises(ValueError):
        DFTB(sk, third_order="full")                       # no hubbard_derivs
    with pytest.raises(ValueError):
        DFTB(sk, third_order="2nd", hubbard_derivs=UD_3OB)
    with pytest.raises(ValueError, match="no entry"):
        DFTB(sk, third_order="full", hubbard_derivs={"O": -0.16}).calculate(labels, pos)
    sk_shell = ParamSet(SKF_HCOCU, shell_resolved=True).skset(labels)
    with pytest.raises(ValueError):
        DFTB(sk_shell, third_order="onsite", hubbard_derivs=UD_3OB)
