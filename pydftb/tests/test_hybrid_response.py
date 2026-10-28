"""Responses with a hybrid functional (milestone M6h-4): the polarisability, the response kernel
and the point charges' derivatives of molecules and the Gamma point.

Comparisons with DFTB+ are the frozen references *_pol, *_kernel and *_extq of M6h-4 in
test_reference.py; here the solver, the limits that M6g's code also gives, pydftb's own finite
differences, the TD-LC sum over states and the switches (no dftb+).
"""
import dataclasses
import math

import numpy as np
import pytest

from pydftb import casida as C
from pydftb import response as R
from pydftb import response_hybrid as H
from pydftb.calculator import DFTB
from pydftb.constants import BOHR, HARTREE
from pydftb.validation import refgen
from pydftb.validation.cases import W_HCO, W_SPIN
from pydftb.validation.refgen import VOLT_PER_ANGSTROM_AU

POL_ONLY = {"polarisability": True}
CUCO_SPIN = {"spin": {"unpaired_electrons": 1.0, "spin_constants": {**W_HCO, "Cu": W_SPIN["Cu"]}}}


class _Grabbed(Exception):
    pass


def _grab(calc, st, sc, spec, opts):
    raise _Grabbed(calc, st, sc, opts)


def _ground(name, monkeypatch, **over):
    """(calc, st, sc, opts) of a reference case's ground state: the analysis' arguments."""
    monkeypatch.setattr(R, "polarisability", _grab)
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], analysis=POL_ONLY, **over)
    try:
        refgen.run_pydftb(case, refgen._inputs(case))
    except _Grabbed as g:
        return g.args
    raise AssertionError("the analysis was not called")


def _run(name, **over):
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], **over)
    return refgen.run_pydftb(case, refgen._inputs(case))


def _support(hr):
    """The pair unknowns (block, i <= j) of a response and their weights in the Frobenius
    product (1, or 2 off the diagonal)."""
    return [(b, i, j) for b in range(len(hr.gs.blocks)) for i in range(hr.n)
            for j in range(i, hr.n) if hr.d[b][i, j] > 0.0]


def _to_z(hr, idx, v):
    z = [np.zeros((v.shape[1], hr.n, hr.n)) for _ in hr.gs.blocks]
    for k, (b, i, j) in enumerate(idx):
        f = 1.0 if i == j else math.sqrt(0.5)
        z[b][:, i, j] = z[b][:, j, i] = v[k] * f
    return z


def _to_v(idx, z):
    return np.array([z[b][:, i, j] * (1.0 if i == j else math.sqrt(2.0)) for b, i, j in idx])


def _fermi_projector(hr, idx):
    """I - u u^T of the Fermi groups in the coordinates of :func:`_support`."""
    p = np.eye(len(idx))
    for parts, u in hr.groups:
        uu = np.zeros(len(idx))
        o = 0
        for b, win in parts:
            for t, i in enumerate(win):
                uu[idx.index((b, i, i))] = u[o + t]
            o += len(win)
        p -= np.outer(uu, uu)
    return p


def _dense(hr):
    """The dense I + T^T K T on the support (the Fermi directions projected out)."""
    idx = _support(hr)
    eye = np.eye(len(idx))
    a = np.hstack([_to_v(idx, hr._operator(_to_z(hr, idx, eye[:, lo:lo + 128])))
                   for lo in range(0, len(idx), 128)])
    p = _fermi_projector(hr, idx)
    return idx, p @ a @ p, p


# ---- the equation ----------------------------------------------------------------------------
@pytest.mark.parametrize("name, over", [("h2o_lc", {}), ("ch3_lc_spin", {}),
                                        ("h2co_lc_plusu", {}), ("cuco_lc_300K", {}),
                                        ("co2_crystal_lc_onsite", {})])
def test_the_equation_is_symmetric(name, over, monkeypatch):
    """(I + T^T K T) is symmetric: K's parts (gamma, spin, blocks, exchange) are second
    derivatives of the energy; these ground states are minima, so it is positive definite."""
    hr = H.HybridResponse(*_ground(name, monkeypatch, **over))
    _, a, p = _dense(hr)
    assert np.abs(a - a.T).max() <= 1e-13 * np.abs(a).max()
    ev = np.linalg.eigvalsh(0.5 * (a + a.T))
    assert ev[np.abs(ev) > 1e-10].min() > 1e-3


@pytest.mark.parametrize("name, over, indefinite", [("h2o_lc", {}, False),
                                                    ("cuco_lc_300K", {}, False),
                                                    ("cuco_lc_300K", CUCO_SPIN, True)])
def test_minres_is_the_direct_solve(name, over, indefinite, monkeypatch):
    """MINRES against the dense solve on the support, a definite and an indefinite equation
    (CuCO with spin: three negative eigenvalues, a saddle point of the SCC; the warning)."""
    calc, st, sc, opts = _ground(name, monkeypatch, **over)
    hr = H.HybridResponse(calc, st, sc, opts)
    idx, a, p = _dense(hr)
    ev = np.linalg.eigvalsh(0.5 * (a + a.T))
    assert (ev.min() < -0.1) == indefinite
    v = np.asarray(st.pos, float)[hr.gs.shell_atom]
    b = _to_v(idx, hr.t_adjoint([hr._h_shell(v[hr.gs.orb_shell].T)] * hr.ns))
    direct = np.linalg.solve(a + (np.eye(len(idx)) - p), b)
    z = _to_v(idx, hr._minres(_to_z(hr, idx, b)))
    assert np.abs(z - direct).max() <= 1e-11 * np.abs(direct).max()
    assert hr.indefinite == indefinite
    if indefinite:
        with pytest.warns(RuntimeWarning, match="saddle point of the SCC"):
            hr.solve(v)


def test_right_hand_sides_in_batches(monkeypatch):
    """The kernel of CO2's crystal in batches of one right-hand side equals all at once."""
    args = _ground("co2_crystal_lc", monkeypatch)
    v = np.eye(len(args[1].species))[np.asarray(args[1].basis.shell_atom)]
    whole = H.HybridResponse(*args).solve(v)
    monkeypatch.setattr(H, "RHS_ELEMENTS", 1)
    single = H.HybridResponse(*args).solve(v)
    for x, y in zip(whole[:3], single[:3]):
        assert np.abs(x - y).max() <= 1e-12


# ---- limits that M6g's code gives ------------------------------------------------------------
def _m6g(calc, st, sc, opts, v, scc):
    """Populations (n_rhs, n_qm, n_orb) of M6g's direct solve on the same ground state."""
    gs = R.GroundState(calc, st, sc, opts["degeneracy_tol"], opts["filling_tol"])
    ch = R.block_channels(st)
    x = R.chi0(gs, [0.0], channels=ch)[0]
    resp = R.Response(gs, R.scc_kernel(calc, st, sc), x, scc, ch)
    return np.transpose(resp.populations(resp.potentials(v)), (2, 0, 1))


@pytest.mark.parametrize("name", ["h2o_lc", "cuco_lc_300K", "co2_crystal_lc_onsite"])
def test_rpa_is_the_bare_response(name, monkeypatch):
    calc, st, sc, opts = _ground(name, monkeypatch)
    hr = H.HybridResponse(calc, st, sc, opts, rpa=True)
    v = np.eye(len(st.species))[np.asarray(st.basis.shell_atom)]
    assert np.abs(hr.solve(v)[0] - _m6g(calc, st, sc, opts, v, False)).max() <= 1e-12


@pytest.mark.parametrize("name, over", [("ch3_lc_spin", {}), ("h2co_lc_plusu", {}),
                                        ("ch3oh_lc_dftb3_d3", {"dispersion": None}),
                                        ("cuco_lc_300K", {}), ("co2_crystal_lc_onsite", {}),
                                        ("cuh_lc_plusu_300K", {})])
def test_without_exchange_is_m6g(name, over, monkeypatch):
    """With the exchange left out the equation is M6g's (gamma, the third order's Hessian,
    the spin constants, the block kernel, the Fermi level's move), solved directly there."""
    calc, st, sc, opts = _ground(name, monkeypatch, **over)
    monkeypatch.setattr(type(st.hybrid), "response_fock",
                        lambda self, ps: [np.zeros_like(p) for p in ps])
    v = np.asarray(st.pos, float)[np.asarray(st.basis.shell_atom)]
    mine = H.HybridResponse(calc, st, sc, opts).solve(v)[0]
    assert np.abs(mine - _m6g(calc, st, sc, opts, v, True)).max() <= 1e-11


# ---- pydftb's own finite differences ---------------------------------------------------------
def _finite_field(name, h=1e-3, **over):
    """d mu / d E (j, i) from 4-point central differences of pydftb's dipole."""
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], scc_tol=1e-13, analysis=None, **over)
    inp = refgen._inputs(case)
    base = np.zeros(3)
    if case.efield:
        d = np.asarray(case.efield["direction"], float)
        base = case.efield["strength"] * d / np.linalg.norm(d)
    alpha = np.zeros((3, 3))
    for i in range(3):
        mu = []
        for s in (2, 1, -1, -2):
            e = base + s * h / VOLT_PER_ANGSTROM_AU * np.eye(3)[i]
            c = dataclasses.replace(case, efield={"strength": float(np.linalg.norm(e)),
                                                  "direction": (e / np.linalg.norm(e)).tolist()})
            mu.append(np.asarray(refgen.run_pydftb(c, inp).dipole))
        alpha[:, i] = (8.0 * (mu[1] - mu[2]) - (mu[0] - mu[3])) / (12.0 * h)
    return alpha


@pytest.mark.parametrize("name, over", [("h2o_lc", {}), ("ch3_lc_spin", {}),
                                        ("cuco_lc_300K", {}), ("h2co_lc_plusu", {}),
                                        ("hco_lc_onsite_spin", {}), ("ch3oh_lc_field_pc", {})])
def test_polarisability_is_the_finite_field_one(name, over):
    """pydftb's analytic polarisability against its own finite fields: +U and onsite with their
    block response (B26) and point charges without their potential (B27)."""
    alpha = _run(name, analysis=POL_ONLY, scc_tol=1e-13, **over).analysis.polarisability
    fd = _finite_field(name, **over)
    assert np.abs(alpha.tensor[0] - fd).max() <= 2e-8 * np.abs(fd).max()


@pytest.mark.parametrize("name, atoms", [("co2_crystal_lc_onsite", [0, 3]),
                                         ("h2co_lc_plusu", [0, 2])])
def test_kernel_is_the_site_potential_one(name, atoms):
    """dN_B / dV_A against differences in a gross site potential on atom A."""
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], scc_tol=1e-13,
                               analysis={"response_kernel": True})
    inp = refgen._inputs(case)
    ker = refgen.run_pydftb(case, inp).analysis.response_kernel.atom[0, :, 0, :]
    h = 1e-3
    for a in atoms:
        q = {}
        for s in (2, 1, -1, -2):
            c = dataclasses.replace(case, analysis=None,
                                    site_potentials={"gross": {a: s * h * HARTREE}})
            q[s] = np.asarray(refgen.run_pydftb(c, inp).charges)
        fd = (8.0 * (q[1] - q[-1]) - (q[2] - q[-2])) / (12.0 * h)
        assert np.abs(ker[a] - fd).max() <= 1e-9


def test_point_charge_derivatives_are_the_finite_difference_ones():
    """dN / dR_j of the point charges against moving them (pydftb's SCC)."""
    case = refgen.CASES_BY_NAME["h2o_lc_pc_extq"]
    case = dataclasses.replace(case, scc_tol=1e-13)
    inp = refgen._inputs(case)
    ext = refgen.run_pydftb(case, inp).analysis.external_charge_derivatives.atom[:, :, 0, :]
    h = 1e-3
    pos = np.array(case.point_charges["positions"], float)
    for j, c in ((0, 0), (2, 2)):
        q = {}
        for s in (2, 1, -1, -2):
            p = pos.copy()
            p[j, c] += s * h * BOHR                              # Angstrom
            cc = dataclasses.replace(case, analysis=None,
                                     point_charges={**case.point_charges, "positions": p.tolist()})
            q[s] = np.asarray(refgen.run_pydftb(cc, inp).charges)
        fd = (8.0 * (q[1] - q[-1]) - (q[2] - q[-2])) / (12.0 * h)
        assert np.abs(ext[j, c] - fd).max() <= 1e-7


def test_casida_sum_over_states(monkeypatch):
    """The static polarisability is TD-LC's sum over all singlet states,
    alpha_ij = 2 sum_n mu_n,i mu_n,j / omega_n: the same exchange kernel (H2O, CH3OH)."""
    for name in ("h2o_lc", "ch3oh_lc"):
        alpha = _run(name, analysis=POL_ONLY, scc_tol=1e-12).analysis.polarisability.tensor[0]
        got = {}
        orig = C.excitations

        def grab(cas, gs):
            got["gs"] = gs
            return orig(cas, gs)
        monkeypatch.setattr(C, "excitations", grab)
        case = dataclasses.replace(refgen.CASES_BY_NAME[name], scc_tol=1e-12)
        refgen.run_pydftb(case, refgen._inputs(case),
                          casida={"n_excitations": 1, "solver": "exact"})
        monkeypatch.undo()
        gs = got["gs"]
        pairs, w, sqrt_n = C.single_particle_transitions(gs)
        q = C.transition_charges(gs, pairs)
        resp = C.HybridResponse(gs, pairs, w, sqrt_n, q, "S")
        omega2, xpy, _ = resp.solve(len(w), "exact")
        mu = math.sqrt(2.0) * ((q @ gs.positions * resp.sqrt_n[:, None]).T @ xpy).T
        sos = 2.0 * np.einsum("n,ni,nj->ij", 1.0 / np.sqrt(omega2), mu, mu)
        assert np.abs(alpha - sos).max() <= 1e-10


# ---- the exchange, the screening and the switches --------------------------------------------
def test_threshold_response_is_the_exact_operator(monkeypatch):
    """The Thresholded screening's response uses its limit (MatrixBased without a cutoff
    reduction) and leaves the screening's history alone: the same answer twice."""
    calc, st, sc, opts = _ground("h2o_lc", monkeypatch,
                                 hybrid={"type": "lc", "screening": "threshold"})
    hyb = st.hybrid
    rng = np.random.default_rng(3)
    p = rng.normal(size=(2, hyb.n, hyb.n))
    p = p + np.transpose(p, (0, 2, 1))
    exact = [-0.125 * hyb._four_terms(hyb.S, x) for x in p]
    assert np.abs(hyb.response_fock([p])[0] - np.array(exact)).max() <= 1e-14
    before = (hyb._h_prev.copy(), hyb._dp_prev.copy())
    v = np.asarray(st.pos, float)[np.asarray(st.basis.shell_atom)]
    first = H.HybridResponse(calc, st, sc, opts).solve(v)[0]
    second = H.HybridResponse(calc, st, sc, opts).solve(v)[0]
    assert np.array_equal(first, second)
    assert np.array_equal(before[0], hyb._h_prev) and np.array_equal(before[1], hyb._dp_prev)


def test_blocks_switch_leaves_the_block_response_out(monkeypatch):
    """B26: DFTB._response_blocks_compat is the response without the block potentials'."""
    calc, st, sc, opts = _ground("h2co_lc_plusu", monkeypatch)
    v = np.asarray(st.pos, float)[np.asarray(st.basis.shell_atom)]
    hr = H.HybridResponse(calc, st, sc, opts)
    own = hr.solve(v)[0]
    hr.channels = None
    without = hr.solve(v)[0]
    monkeypatch.setattr(DFTB, "_response_blocks_compat", True)
    compat = H.HybridResponse(calc, st, sc, opts).solve(v)[0]
    assert np.array_equal(compat, without)
    assert np.abs(own - compat).max() > 1e-2


@pytest.mark.parametrize("name", ["ch3oh_lc_field_pc", "ch3oh_field_pc_pol"])
def test_point_charge_switch_adds_their_potential(name, monkeypatch):
    """B27: with DFTB._response_pc_compat dq/dE_i gains the response to the point charges'
    potential, sum_A v_pc,A dq/dV_A (with and without a hybrid)."""
    pol = _run(name, analysis=POL_ONLY).analysis.polarisability
    ker = _run(name, analysis={"response_kernel": True}).analysis.response_kernel
    with monkeypatch.context() as m:
        m.setattr(DFTB, "_response_pc_compat", True)
        compat = _run(name, analysis=POL_ONLY).analysis.polarisability
    st = _ground(name, monkeypatch)[1]
    shift = np.einsum("acb,a->cb", ker.atom[0], np.asarray(st.ext.v_pc, float))
    expect = pol.charge_derivatives[0] + shift[None]
    assert np.abs(compat.charge_derivatives[0] - expect).max() <= 1e-10


# ---- refusals --------------------------------------------------------------------------------
def test_refusals():
    """DFTB+'s refusals of a hybrid's responses (k-points and two-component spin: M6h-4b,
    tests/test_hybrid_response_complex.py)."""
    with pytest.raises(ValueError, match="Finite frequency hybrid"):
        _run("h2o_lc", analysis={"polarisability": {"static": False, "frequencies": [0.1]}})
    with pytest.raises(ValueError, match="Coordinate derivative perturbations"):
        _run("h2o_lc", analysis={"coord_derivatives": True})
