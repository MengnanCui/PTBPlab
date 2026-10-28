"""Finite-frequency responses with a hybrid functional (milestone M6h-4c): DFTB+ refuses them, so
the checks are pydftb's own -- TD-LC Casida's sum over all states, M6g's and M6g-3b's
finite-frequency responses without the exchange, the supercell's Gamma point, the collinear
limit, the static limit and the dense solve near an excitation.
"""
import dataclasses
import math

import numpy as np
import pytest

from pydftb import casida as C
from pydftb import response as R
from pydftb import response_hybrid as H
from pydftb import response_pauli as RP
from pydftb.calculator import DFTB
from pydftb.hybrid import HybridTerms
from pydftb.kpoints import supercell_folding
from pydftb.tests import test_hybrid_kpoints as TK
from pydftb.tests import test_hybrid_response_complex as TC
from pydftb.validation import refgen

W_CH3 = {"unpaired_electrons": 1.0, "spin_constants": {"H": -0.072, "C": -0.023}}


def _pol(omegas, static=False):
    return {"polarisability": {"static": static, "frequencies": list(omegas)}}


def _run(name, analysis, **over):
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], analysis=analysis, scc_tol=1e-12,
                               **over)
    return refgen.run_pydftb(case, refgen._inputs(case))


def _ground(name, what="polarisability", **over):
    return TC._ground(name, what=what, scc_tol=1e-12, **over)


def _casida_sum(name, omegas, monkeypatch, **over):
    """TD-LC Casida's alpha(w) = 2 sum_n w_n mu_n,i mu_n,j / (w_n^2 - w^2) over all states
    (exact solver): the singlets of a closed shell (mu with sqrt 2) or every spin-polarised
    state; and the excitation energies."""
    got = {}
    orig = C.excitations

    def grab(cas, gs):
        got["gs"] = gs
        return orig(cas, gs)
    monkeypatch.setattr(C, "excitations", grab)
    case = dataclasses.replace(refgen.CASES_BY_NAME[name], scc_tol=1e-12, **over)
    refgen.run_pydftb(case, refgen._inputs(case), casida={"n_excitations": 1, "solver": "exact"})
    monkeypatch.undo()
    gs = got["gs"]
    pairs, w, sqrt_n = C.single_particle_transitions(gs)
    q = C.transition_charges(gs, pairs)
    resp = C.HybridResponse(gs, pairs, w, sqrt_n, q, " " if gs.spin_polarised else "S")
    omega2, xpy, _ = resp.solve(len(w), "exact")
    mu = (1.0 if gs.spin_polarised else math.sqrt(2.0)) * (
        (q @ gs.positions * resp.sqrt_n[:, None]).T @ xpy).T
    wn = np.sqrt(omega2)
    return [2.0 * np.einsum("n,ni,nj->ij", wn / (omega2 - om * om), mu, mu) for om in omegas], wn


def _zero_exchange(m, cls):
    m.setattr(cls, "response_fock", lambda self, ps, antihermitian=False:
              [np.zeros_like(p) for p in ps])


# ---- molecules and the Gamma point -----------------------------------------------------------
@pytest.mark.parametrize("name, omegas, over", [
    ("h2o_lc", [0.1, 0.25, 0.5], {}),                      # 0.25, 0.5 above the first excitation
    ("ch3_lc_spin", [0.05, 0.2], {"spin": W_CH3})])
def test_polarisability_is_tdlc_casidas_sum_over_states(name, omegas, over, monkeypatch):
    """The exchange of both parts of P1 against TD-LC's A - B and A + B: alpha(w) of all states,
    also above excitations (the equation is indefinite there) and spin-polarised (one spin
    constant per element: Casida's coupling takes the HOAO shell's, the CPKS the shells')."""
    sos, _ = _casida_sum(name, omegas, monkeypatch, **over)
    pol = _run(name, _pol(omegas), **over).analysis.polarisability
    for k in range(len(omegas)):
        assert np.abs(pol.tensor[k] - sos[k]).max() <= 1e-10 * np.abs(sos[k]).max()


def test_near_an_excitation_the_dense_solve_takes_over(monkeypatch):
    """C6H6 1e-5 Ha below an excitation in a dense cluster: MINRES stalls (its Lanczos vectors
    lose orthogonality), the dense solve on the physical space is accurate to cond * eps."""
    _, wn = _casida_sum("c6h6_lc", [0.0], monkeypatch)
    om = wn[np.argmin(np.abs(wn - 0.15))] - 1e-5
    sos, _ = _casida_sum("c6h6_lc", [om], monkeypatch)
    calls = []
    dense = H.HybridResponse._dense_solve
    monkeypatch.setattr(H.HybridResponse, "_dense_solve",
                        lambda self, *a: calls.append(1) or dense(self, *a))
    pol = _run("c6h6_lc", _pol([om])).analysis.polarisability
    assert calls
    assert np.abs(pol.tensor[0] - sos[0]).max() <= 1e-8 * np.abs(sos[0]).max()


@pytest.mark.parametrize("name", ["cuco_lc_300K", "h2co_lc_plusu_pol", "co2_crystal_lc_kernel"])
def test_without_exchange_is_m6g(name, monkeypatch):
    """Without the exchange: M6g's finite-frequency response (DFTB+'s weights, eta 1e-8) on the
    same ground state -- a metal (no Fermi-level move at w != 0), +U blocks, the Gamma point --
    populations and level derivatives."""
    calc, st, sc, opts = _ground(name)
    spec = {"static": False, "frequencies": [0.05, 0.15]}
    with pytest.MonkeyPatch.context() as m:
        _zero_exchange(m, type(st.hybrid))
        mine = H.polarisability(calc, st, sc, spec, opts)
    ref = R.polarisability(calc, dataclasses.replace(st, hybrid=None), sc, spec, opts)
    assert np.abs(mine.charge_derivatives - ref.charge_derivatives).max() <= 1e-11
    for a, b in zip(mine.eigenvalue_derivatives, ref.eigenvalue_derivatives):
        assert np.abs(a - b).max() <= 1e-11


def test_rpa_is_the_bare_chi0():
    calc, st, sc, opts = _ground("h2o_lc", what="response_kernel")
    spec = {"static": False, "frequencies": [0.3], "rpa": True}
    mine = H.response_kernel(calc, st, sc, spec, opts)
    ref = R.response_kernel(calc, dataclasses.replace(st, hybrid=None), sc, spec, opts)
    assert np.abs(mine.atom - ref.atom).max() <= 1e-12


def test_static_limit_and_the_static_entry():
    """alpha(w) -> alpha(0) as w^2; the static entry of a call with frequencies is the static
    call's, bit for bit."""
    both = _run("h2o_lc", {"polarisability": {"frequencies": [1e-4, 1e-2]}}).analysis.polarisability
    static = _run("h2o_lc", {"polarisability": True}).analysis.polarisability
    assert np.array_equal(both.tensor[0], static.tensor[0])
    d4, d2 = (np.abs(both.tensor[k] - both.tensor[0]).max() for k in (1, 2))
    assert d4 <= 1e-6 and d4 / d2 <= 2e-4                        # (1e-4 / 1e-2)^2 = 1e-4


def test_minres_plateau_rule(monkeypatch):
    """MINRES with ``plateau`` (finite frequency without the dense solve): a relative residual of
    at most OMEGA_PLATEAU that has not halved in PLATEAU_STEPS (here 20) iterations ends the
    iteration as converged -- an inconsistent part of 1e-12 along a null space, where MINRES's
    residual recursion would drift below the true residual and the iterate degrade; without the
    rule the iteration runs to its limit."""
    monkeypatch.setattr(H, "PLATEAU_STEPS", 20)
    lam = np.random.default_rng(0).uniform(0.5, 2.0, size=(1, 40, 40))
    lam[0, :2] = 0.0                                     # a null space of 80 unknowns
    b = np.where(lam > 0.0, 1.0, 4.4e-12)                # relative 1e-12 along it
    hr = H.HybridResponse.__new__(H.HybridResponse)
    hr.iterations = 0
    op = lambda v: [lam * v[0]]                          # noqa: E731
    x = hr._minres([b], op, maxit=400, strict=False, curvature=False, plateau=H.OMEGA_PLATEAU)
    assert hr._converged and hr.iterations < 100
    assert 5e-13 <= hr._residual <= 2e-12
    assert np.abs(lam * x[0] - np.where(lam > 0.0, b, 0.0)).max() <= 1e-13
    hr.iterations = 0
    hr._minres([b], op, maxit=400, strict=False, curvature=False)
    assert not hr._converged and hr.iterations == 400


def test_solver_fallbacks(monkeypatch):
    """Not converged in as many iterations as unknowns: the dense solve (small systems), else a
    warning with the last iterate."""
    ref = _run("h2o_lc", _pol([0.3])).analysis.polarisability.tensor[0]
    monkeypatch.setattr(H, "SOLVE_MAX", 3)
    dense = _run("h2o_lc", _pol([0.3])).analysis.polarisability.tensor[0]
    assert np.abs(dense - ref).max() <= 1e-10 * np.abs(ref).max()
    monkeypatch.setattr(H, "DENSE_LIMIT", 0)
    with pytest.warns(RuntimeWarning, match="did not converge in 3 MINRES iterations"):
        _run("h2o_lc", _pol([0.3]))


# ---- k-points ----------------------------------------------------------------------------------
@pytest.mark.filterwarnings("error:SCC not converged")
@pytest.mark.parametrize("spin", [False, True])
def test_kpoint_kernel_is_the_supercells_gamma_point(spin):
    """CO (triplet O2) in a box without self-image overlaps at w = 0.2: the reduced k-point
    set's convolution also holds for the antisymmetric part (P^a(-k) = conj P^a(k), the
    supercell's P^a being real)."""
    labels, pos, cell = TC._o2_in_co_box() if spin else TK._co_box()
    sk = TK._sk(labels)
    extra = {"spin": True, "unpaired_electrons": 2.0, "spin_constants": {"O": -0.028}} if spin \
        else {}
    kw = dict(hybrid=TC.CO_BOX_HYBRID, scc_tol=1e-12,
              analysis={"response_kernel": {"static": False, "frequencies": [0.2]}}, **extra)
    r = DFTB(sk, kpts=supercell_folding((2, 1, 1)), **kw).calculate(labels, pos, cell, TK.PBC,
                                                                    forces=False)
    if spin:
        kw["unpaired_electrons"] = 4.0
    sc_pos, sc_cell, _ = TK._supercell(pos, cell, (2, 1, 1))
    r_sc = DFTB(sk, **kw).calculate(labels * 2, sc_pos, sc_cell, TK.PBC, forces=False)
    ker = r.analysis.response_kernel.atom[0, :, 0, :]
    ksc = r_sc.analysis.response_kernel.atom[0, :, 0, :].reshape(2, 2, 2, 2).sum(0)[:, 0, :]
    assert np.abs(ker - ksc).max() <= 1e-11


def test_kpoints_without_exchange_are_m6g():
    calc, st, sc, opts = TC._ground("diamond2_lc_k222", filling=("fermi", 30000.0))
    spec = {"static": False, "frequencies": [0.2]}
    with pytest.MonkeyPatch.context() as m:
        _zero_exchange(m, type(st.hybrid))
        mine = H.polarisability(calc, st, sc, spec, opts)
    ref = R.polarisability(calc, dataclasses.replace(st, hybrid=None), sc, spec, opts)
    assert np.abs(mine.charge_derivatives - ref.charge_derivatives).max() <= 1e-11


# ---- two-component spin ------------------------------------------------------------------------
def test_two_component_without_exchange_is_m6g3b():
    calc, st, sc, opts = _ground("h3_lc_noncolinear_120")
    spec = {"static": False, "frequencies": [0.05, 0.2]}
    with pytest.MonkeyPatch.context() as m:
        _zero_exchange(m, HybridTerms)
        mine = H.polarisability(calc, st, sc, spec, opts)
    ref = RP.polarisability(calc, dataclasses.replace(st, hybrid=None), sc, spec, opts)
    assert np.abs(mine.charge_derivatives - ref.charge_derivatives).max() <= 1e-12


def test_two_component_collinear_limit():
    """Moments along z at w = 0.2: the collinear finite-frequency response (charge, m_z; the
    transverse magnetisations zero)."""
    nc = refgen.CASES_BY_NAME["hco_lc_noncolinear"].noncolinear
    z = {**nc, "initial_spins": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]}
    res = _run("hco_lc_noncolinear", _pol([0.2]), noncolinear=z).analysis.polarisability
    col = _run("hco_lc_spin_neighbour", _pol([0.2]), hybrid="lc").analysis.polarisability
    assert np.abs(res.tensor[0] - col.tensor[0]).max() <= 1e-8
    dq, dc = res.charge_derivatives[0], col.charge_derivatives[0]
    assert np.abs(dq[:, 0] - dc[:, 0]).max() <= 1e-8
    assert np.abs(dq[:, 3] - dc[:, 1]).max() <= 1e-8
    assert np.abs(dq[:, 1:3]).max() <= 1e-8


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_two_component_solver(monkeypatch):
    """The rotations near the w -> 0 pole (eigenvalues ~ w^2; rounding along them can stall
    MINRES just above its tolerance, H3's ground state at scc_tol 1e-12 in some builds): no
    warning, and alpha(w) - alpha(0) follows w^2 (H3's spin excitation near 0.01 Ha).  When
    MINRES fails (forced), the dense solve is the collinear limit's response."""
    dq = [_run("h3_lc_noncolinear_120", _pol([w], static=True)).analysis.polarisability
          .charge_derivatives for w in (1e-5, 2e-5)]
    d1, d2 = (np.abs(d[1] - d[0]).max() for d in dq)
    assert 3.95 <= d2 / d1 <= 4.05
    nc = refgen.CASES_BY_NAME["hco_lc_noncolinear"].noncolinear
    z = {**nc, "initial_spins": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]}
    col = _run("hco_lc_spin_neighbour", _pol([0.5]), hybrid="lc").analysis.polarisability
    calls = []
    dense = H.HybridResponse._dense_solve
    monkeypatch.setattr(H.HybridResponse, "_dense_solve",
                        lambda self, *a: calls.append(1) or dense(self, *a))
    monkeypatch.setattr(H, "SOLVE_MAX", 3)
    res = _run("hco_lc_noncolinear", _pol([0.5]), noncolinear=z).analysis.polarisability
    assert calls
    assert np.abs(res.tensor[0] - col.tensor[0]).max() <= 1e-10 * np.abs(col.tensor[0]).max()


def test_pauli_exchange_of_antihermitian_matrices():
    """The spinor exchange of an anti-Hermitian P is anti-Hermitian: every N x N block's four-term
    operator, the upper-right block the negative adjoint of the lower-left one."""
    st = _ground("h3_lc_noncolinear_120")[1]
    hyb, n = st.hybrid, st.hybrid.n
    x = np.random.default_rng(1).normal(size=(2, 2 * n, 2 * n)) * (1.0 + 0.3j)
    x = x - np.conj(np.swapaxes(x, -1, -2))
    h = hyb.response_fock([x], antihermitian=True)[0]
    full = np.empty_like(x)
    for r in (0, 1):
        for c in (0, 1):
            full[:, r * n:(r + 1) * n, c * n:(c + 1) * n] = -0.25 * hyb._four_terms(
                hyb.S, x[:, r * n:(r + 1) * n, c * n:(c + 1) * n])
    assert np.abs(h - full).max() <= 1e-14 * np.abs(full).max()
    assert np.abs(h + np.conj(np.swapaxes(h, -1, -2))).max() <= 1e-14 * np.abs(h).max()
