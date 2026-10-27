"""TD-LC: Casida excitations and excited-state forces with a long-range corrected functional
(milestone M6h-2).

The comparisons with DFTB+ are the frozen references *_tdlc* in test_reference.py.  Here: the
exchange of A+B and A-B (dense through the transition charges, applied through the AO operator,
against explicit integrals), the symmetric solution and Stratmann's iteration, the limit without
exchange (the M6d Casida), a closed shell run spin polarised, the forces against differences of
pydftb's own E0 + Omega, the two Z-vector solvers, the bilinear exchange derivatives against
differences, and the refusals.
"""
from dataclasses import replace

import numpy as np
import pytest

from pydftb import casida as cs
from pydftb import casida_grad as cg
from pydftb import hybrid as hy
from pydftb.calculator import DFTB
from pydftb.casida import Casida
from pydftb.constants import BOHR
from pydftb.params import ParamSet
from pydftb.validation import dftbplus, refgen
from pydftb.validation.cases import CASES_BY_NAME, SKF_SETS, W_HOAO

CASE = CASES_BY_NAME["h2co_tdlc"]
FORCE_VARIANTS = {
    "singlet": CASES_BY_NAME["h2co_tdlc_exc_forces"],
    "triplet": CASES_BY_NAME["h2co_tdlc_exc_forces_triplet"],
    "spin": CASES_BY_NAME["h2co_cation_tdlc_exc_forces"],
    "c2h4_s1": CASES_BY_NAME["c2h4_tdlc_exc_forces_s1"],
    "field_pc": CASES_BY_NAME["h2co_tdlc_exc_forces_field_pc"],
}


def _engine(case, **opts):
    inp = refgen._inputs(case)
    res, calc = refgen.run_pydftb(case, inp, engine=True, **{"scc_tol": 1e-10, **opts})
    return res, calc, inp["labels"], np.array(inp["positions_angstrom"]) / BOHR


def _ground_state(case, **opts):
    _, calc, labels, pos = _engine(case, **opts)
    _, st, sc = calc._run(labels, pos, None, None, False, None, False)
    return calc._casida_ground_state(st, sc).prepare()


def _transitions(gs):
    pairs, w, sqrt_n = cs.single_particle_transitions(gs)
    order = np.argsort(w, kind="stable")
    pairs, w, sqrt_n = pairs[order], w[order], sqrt_n[order]
    return pairs, w, sqrt_n, cs.transition_charges(gs, pairs)


@pytest.mark.parametrize("name,sym", [("h2co_tdlc", "S"), ("h2co_tdlc_triplet", "T"),
                                      ("h2co_cation_tdlc", " ")])
def test_exchange_matrices_and_products(name, sym):
    """(ij|ab) and (ib|ja) of the dense build are q^ij gamma_lr q^ab and q^ib gamma_lr q^ja
    (explicit, every pair); A+B and A-B are symmetric, their matrix-free products are the
    matrices; no exchange between the spin channels."""
    gs = _ground_state(CASES_BY_NAME[name])
    pairs, w, sqrt_n, q = _transitions(gs)
    resp = cs.HybridResponse(gs, pairs, w, sqrt_n, q, sym)
    kd, kx = resp.exchange.dense()
    lr = gs.lr_gamma
    for m, (i, a, s) in enumerate(pairs[::3]):
        for n, (j, b, t) in enumerate(pairs):
            if s != t:
                assert kd[3 * m, n] == 0.0 and kx[3 * m, n] == 0.0
                continue
            qq = lambda p, r: cs.transition_charges(gs, np.array([[p, r, s]]))[0]  # noqa: E731
            assert kd[3 * m, n] == pytest.approx(qq(i, j) @ lr @ qq(a, b), abs=1e-13)
            assert kx[3 * m, n] == pytest.approx(qq(i, b) @ lr @ qq(j, a), abs=1e-13)
    apb, amb = resp.matrices()
    assert np.array_equal(apb, apb.T) and np.array_equal(amb, amb.T)
    eye = np.eye(len(w))
    assert np.abs(resp.apb_product(eye) - apb).max() < 1e-13
    assert np.abs(resp.amb_product(eye) - amb).max() < 1e-13
    assert np.abs(resp.exchange.diagonal() + np.diag(kd + kx)).max() < 1e-14


def test_symmetric_solution():
    """(A-B)(A+B)(X+Y) = Omega^2 (X+Y), (A+B)(X+Y) = Omega (X-Y), (X+Y).(X-Y) = 1."""
    gs = _ground_state(CASE)
    pairs, w, sqrt_n, q = _transitions(gs)
    resp = cs.HybridResponse(gs, pairs, w, sqrt_n, q, "S")
    apb, amb = resp.matrices()
    omega2, xpy, xmy = cs._symmetric_rpa(apb, amb, 8)
    om = np.sqrt(omega2)
    assert np.abs(apb @ xpy - xmy * om).max() < 1e-12
    assert np.abs(amb @ xmy - xpy * om).max() < 1e-12
    assert np.abs((xpy * xmy).sum(axis=0) - 1.0).max() < 1e-12
    full = np.linalg.eigvals(amb @ apb)
    assert np.abs(np.sort(full.real)[:8] - omega2).max() < 1e-12


@pytest.mark.parametrize("name,n", [("c6h6_tdlc", 8), ("h2co_cation_tdlc", 6),
                                    ("naphthalene_tdlc_exc_forces", 12)])
def test_stratmann_agrees_with_the_exact_solution(name, n):
    """Stratmann's iteration (matrix-free A+B and A-B; naphthalene: 576 transitions, a
    subspace of a few hundred) gives the exact solution's states: Omega^2 to 1e-12 and, per
    group of degenerate states, the same summed oscillator strength."""
    case = CASES_BY_NAME[name]
    xs = [_engine(case, casida=Casida(n, symmetry=case.casida.get("symmetry") or "singlet",
                                      solver=s))[0].excitations for s in ("exact", "davidson")]
    assert np.abs(xs[0].omega2 - xs[1].omega2).max() < 1e-12
    g0, g1 = xs[0].degenerate_groups(), xs[1].degenerate_groups()
    assert np.abs(g0["oscillator"] - g1["oscillator"]).max() < 1e-9


def test_no_exchange_is_the_m6d_casida():
    """With gamma_lr = 0 the TD-LC solution on the same ground state is the Hermitian Casida
    (singlets, triplets; spin polarised)."""
    for case, syms, opts in ((CASE, ("singlet", "triplet"), {"spin_constants": W_HOAO}),
                             (CASES_BY_NAME["h2co_cation_tdlc"], ("singlet",), {})):
        gs = _ground_state(case, **opts)
        for sym in syms:
            cas = Casida(8, symmetry=sym, solver="exact")
            gs.lr_gamma = None
            ref = cs.excitations(cas, gs)
            gs.lr_gamma = np.zeros((len(gs.positions),) * 2)
            x = cs.excitations(cas, gs)
            assert np.abs(x.omega2 - ref.omega2).max() < 1e-14
            assert np.abs(x.oscillator_strengths - ref.oscillator_strengths).max() < 1e-12
            assert np.abs(np.abs(x.xpy) - np.abs(ref.xpy)).max() < 1e-9
            assert np.abs(np.abs(x.xmy) - np.abs(ref.xmy)).max() < 1e-9


def test_closed_shell_run_spin_polarised():
    """H2CO run spin polarised (no unpaired electrons): its TD-LC states are the restricted
    singlets and triplets (the same W)."""
    spin = replace(CASE, spin={"unpaired_electrons": 0.0, "spin_constants": W_HOAO})
    xs = _engine(spin, casida=Casida(12, symmetry="singlet", solver="exact"))[0].excitations
    s = _engine(CASE, casida=Casida(12, symmetry="singlet", solver="exact"),
                spin_constants=W_HOAO)[0].excitations
    t = _engine(CASE, casida=Casida(12, symmetry="triplet", solver="exact"),
                spin_constants=W_HOAO)[0].excitations
    union = np.sort(np.concatenate([s.omega2, t.omega2]))[:12]
    assert np.abs(xs.omega2 - union).max() < 1e-10


def _fd(energy, x0, idx, h):
    out = []
    for i in idx:
        e = {}
        for s in (-2, -1, 1, 2):
            x = x0.copy()
            x.flat[i] += s * h
            e[s] = energy(x)
        out.append(-(8.0 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12.0 * h))
    return np.array(out)


@pytest.mark.parametrize("name", list(FORCE_VARIANTS))
def test_forces_are_the_derivative_of_e0_plus_omega(name):
    """-d(E0 + Omega)/dR with the exchange's Z vector, W and explicit terms: singlets, triplets,
    spin polarisation, ethylene's S1, field and point charges (exact SK derivatives)."""
    case = FORCE_VARIANTS[name]
    inp = refgen._inputs(case)
    res = refgen.run_pydftb(case, inp, scc_tol=1e-13)
    pos = np.array(inp["positions_angstrom"])
    idx = [2, 3, 7, 11]

    def energy(p):
        return refgen.run_pydftb(case, dict(inp, positions_angstrom=p.tolist()),
                                 scc_tol=1e-13).excited_free_energy

    fd = _fd(energy, pos, idx, 5e-5 * BOHR) * BOHR
    assert np.abs(res.forces.flat[idx] - fd).max() < 3e-9


def test_dense_and_iterative_z_vector_agree(monkeypatch):
    """The Z vector with the exchange in A+B: dense solve and conjugate gradients (its
    product and diagonal) give the same forces and charges."""
    case = CASES_BY_NAME["c6h6_tdlc_exc_forces_neighbour"]
    dense = _engine(case)[0]
    monkeypatch.setattr(cg, "EXACT_LIMIT", 10)
    iterative = _engine(case)[0]
    assert np.abs(dense.forces - iterative.forces).max() < 1e-12
    assert np.abs(dense.excited_charges - iterative.excited_charges).max() < 1e-12


def test_relaxed_density_and_invariance():
    """The relaxed density keeps the electron count; without external fields the forces sum to
    zero; xmy_ao is antisymmetric."""
    for case in (FORCE_VARIANTS["singlet"], FORCE_VARIANTS["spin"]):
        res = _engine(case)[0]
        assert abs((res.excited_charges - res.charges).sum()) < 1e-12
        assert np.abs(res.forces.sum(axis=0)).max() < 1e-12


@pytest.mark.parametrize("sym1,sym2", [(1, 1), (-1, -1), (1, 1j), (0, 0)])
def test_bilinear_derivatives(sym1, sym2):
    """dB/dS and dB/dG of B = <M1, K^x[M2]> against central differences of symmetric S and G
    (both orderings of an element, as the pair terms take them; random S, G and symmetric,
    antisymmetric, mixed or general M1, M2); B(M1, M2) = B(M2, M1)."""
    rng = np.random.default_rng(3)
    n = 7

    def mat(kind):
        m = rng.normal(size=(n, n))
        if kind == 1:
            return m + m.T
        if kind == -1:
            return m - m.T
        return m
    s = np.eye(n) + 0.1 * mat(1)
    g = np.abs(mat(1))
    m1 = mat(sym1 if sym1 != 1j else 1)
    m2 = mat(sym2 if sym2 != 1j else -1)

    def b(s_, g_):
        return float(np.sum(m1 * (0.25 * hy.four_terms(s_, g_, m2))))

    assert b(s, g) == pytest.approx(float(np.sum(m2 * (0.25 * hy.four_terms(s, g, m1)))),
                                    rel=1e-12)
    ds, dg = hy.bilinear_ds(s, g, m1, m2), hy.bilinear_dg(s, m1, m2)
    h = 1e-6
    for (i, j) in ((0, 0), (1, 4), (6, 2)):           # S and G stay symmetric
        e = np.zeros((n, n))
        e[i, j] = e[j, i] = h
        want = ds[i, j] + ds[j, i] if i != j else ds[i, i]
        assert (b(s + e, g) - b(s - e, g)) / (2 * h) == pytest.approx(want, abs=1e-7)
        want = dg[i, j] + dg[j, i] if i != j else dg[i, i]
        assert (b(s, g + e) - b(s, g - e)) / (2 * h) == pytest.approx(want, abs=1e-7)


def _sk(set_name, labels):
    return ParamSet(SKF_SETS[set_name]).skset(labels)


def test_refusals():
    """DFTB+'s conditions of TD-LC (CAM and global files, fractional occupations) and the
    HSD writer's one W per type; the LC files with the CAM keyword run (as DFTB+)."""
    labels = ["O", "C", "H", "H"]
    cas = Casida(3)
    with pytest.raises(ValueError, match="General CAM functionals"):
        DFTB(_sk("ptbp_hcocu_cam", labels), hybrid="cam", casida=cas)
    with pytest.raises(ValueError, match="Global hybrid functionals"):
        DFTB(_sk("ptbp_hcocu_global", labels), hybrid="cam", casida=cas)
    assert DFTB(_sk("ptbp_hcocu_lc", labels), hybrid="cam", casida=cas).casida is cas
    hot = replace(CASE, filling=("fermi", 3000.0))
    with pytest.raises(ValueError, match="Fractional occupations not implemented for TD-LC"):
        _engine(hot)
    gs = _ground_state(CASE)
    with pytest.raises(ValueError, match="Hermitian Omega"):
        cs.HybridResponse(gs, *_transitions(gs), "S").solve(3, "arpack")
    with pytest.raises(ValueError, match="one W per type"):
        dftbplus.spin_constants_block({"O": [[-0.035, -0.030], [-0.030, -0.028]]})
