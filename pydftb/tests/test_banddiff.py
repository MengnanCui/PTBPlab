"""Eigenvalue derivatives at frozen charges (A series, A0 section 4): Hellmann-Feynman (generic
central differences of H and S, and the analytic path of the adaptive channels) against
eigenvalue differences at the same frozen populations; degenerate levels; the size of the
charge response that the frozen-charge Jacobian leaves out."""
import numpy as np
import pytest
from ase.build import bulk, molecule
from scipy.linalg import eigh

from pydftb import DFTB, ParamSet, make_kpoints
from pydftb.adaptive import RBF, AdaptiveParams
from pydftb.banddiff import (adaptive_eigenvalue_derivatives, eigenvalue_derivatives,
                             frozen_matrices, hellmann_feynman)
from pydftb.bands import eigenstates
from pydftb.constants import BOHR
from pydftb.validation.cases import SKF_SETS

SK = SKF_SETS["ptbp_hcocu"]


def _water_case():
    p = ParamSet(SK, max_l={"H": 0, "O": 1})
    w = molecule("H2O")
    w.positions[1] += [0.05, -0.03, 0.02]
    lab = w.get_chemical_symbols()
    model = AdaptiveParams("O", RBF((1.0, 2.0), 0.7), onsite={"s": (0.01, -0.02, 0.03),
                                                                "p": (-0.01, 0.02, 0.015)},
                           hopping={"p": (0.03, 0.06, -0.05)})
    sk = p.skset(lab)

    def make(x):
        return DFTB(sk, scc_tol=1e-12, adaptive=[model.with_vector(x)])

    return model, make, (lab, w.positions / BOHR, None, None), None


def _cu_case():
    at = bulk("Cu", "fcc", a=3.7, cubic=True)
    at.rattle(0.08, seed=1)
    at.cell[0, 1] += 0.15
    at = at[:3]
    sk = ParamSet(SK, max_l={"Cu": 2}).skset(["Cu"])
    model = AdaptiveParams("Cu", RBF((4.0, 8.0), 2.0),
                           onsite={"s": (0.01, -0.02, 0.01), "d": (-0.01, 0.02, 0.005)},
                           hopping={"s": (0.05, -0.1, 0.04), "d": (0.03, -0.02, 0.06)})

    def make(x):
        return DFTB(sk, kpts=make_kpoints((2, 2, 2)), temperature=1000.0, scc_tol=1e-12,
                    adaptive=[model.with_vector(x)])

    path = (np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.05], [0.5, 0.0, 0.25]]), np.ones(3))
    geo = (at.get_chemical_symbols(), at.positions / BOHR, at.cell.array / BOHR,
           np.ones(3, bool))
    return model, make, geo, path


@pytest.fixture(scope="module", params=["water", "cu"])
def case(request):
    model, make, geo, kpts = _water_case() if request.param == "water" else _cu_case()
    x0 = model.vector()
    r = make(x0).calculate(*geo, forces=False)
    st = eigenstates(make(x0), *geo, r.q_in, kpts=kpts, vectors=True)
    return model, make, geo, kpts, x0, r, st


def _frozen_fd(make, geo, q, kpts, x0, h=1e-5):
    """Central differences of the eigenvalues themselves at the frozen populations q."""
    out = np.zeros(eigenstates(make(x0), *geo, q, kpts=kpts).eigenvalues.shape + (len(x0),))
    for j in range(len(x0)):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h
        xm[j] -= h
        out[..., j] = (eigenstates(make(xp), *geo, q, kpts=kpts).eigenvalues
                       - eigenstates(make(xm), *geo, q, kpts=kpts).eigenvalues) / (2 * h)
    return out


def test_generic_and_analytic_derivatives_are_the_frozen_differences(case):
    model, make, geo, kpts, x0, r, st = case
    ref = _frozen_fd(make, geo, r.q_in, kpts, x0)
    gen = eigenvalue_derivatives(make, x0, *geo, r.q_in, kpts=kpts, states=st)
    ana = adaptive_eigenvalue_derivatives(make(x0), [model], *geo, r.q_in, kpts=kpts, states=st)
    assert np.abs(ref).max() > 1e-3
    assert np.abs(gen - ref).max() < 1e-7
    assert np.abs(ana - ref).max() < 1e-7
    assert np.abs(ana - gen).max() < 1e-7


def test_charge_response_left_out(case):
    """The full-SCC derivative differs from the frozen-charge one by the charge response.  Most
    of it is a common shift of all levels (screening), which the band loss's free shift removes:
    measured (2026-11-10) water 64 % of the largest derivative with the shift, 21 % without it
    (cosine 0.975); the Cu metal 0.5 %."""
    model, make, geo, kpts, x0, r, st = case
    hf = adaptive_eigenvalue_derivatives(make(x0), [model], *geo, r.q_in, kpts=kpts, states=st)
    h = 1e-5
    scc = np.zeros_like(hf)
    for j in range(len(x0)):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h
        xm[j] -= h
        ep = eigenstates(make(xp), *geo, make(xp).calculate(*geo, forces=False).q_in,
                         kpts=kpts).eigenvalues
        em = eigenstates(make(xm), *geo, make(xm).calculate(*geo, forces=False).q_in,
                         kpts=kpts).eigenvalues
        scc[..., j] = (ep - em) / (2 * h)
    a0 = hf - hf.mean(axis=2, keepdims=True)               # without the common shift
    b0 = scc - scc.mean(axis=2, keepdims=True)
    rel = np.abs(a0 - b0).max() / np.abs(b0).max()
    cos = (a0 * b0).sum() / np.linalg.norm(a0) / np.linalg.norm(b0)
    assert 1e-6 < rel < 0.4 and cos > 0.95, (rel, cos)


def test_degenerate_levels_get_the_sorted_split():
    rng = np.random.default_rng(0)
    n = 6
    a = rng.normal(size=(n, n))
    s = a @ a.T + n * np.eye(n)                        # SPD overlap
    # H with a threefold level: H = S U diag(e) U^T S in the S-orthonormal basis
    e0 = np.array([-1.0, 0.2, 0.2, 0.2, 0.9, 1.5])
    l = np.linalg.cholesky(s)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    h0 = l @ q @ np.diag(e0) @ q.T @ l.T
    dh = rng.normal(size=(n, n))
    dh = dh + dh.T
    ds = rng.normal(size=(n, n)) * 0.1
    ds = ds + ds.T
    e, c = eigh(h0, s)
    d = hellmann_feynman(c, e, dh, ds, degeneracy=1e-8)
    step = 1e-7
    ep = eigh(h0 + step * dh, s + step * ds, eigvals_only=True)
    fwd = (ep - e) / step                              # the right derivative, sorted
    assert np.abs(d - fwd).max() < 1e-5
    assert np.ptp(d[1:4]) > 1e-2                       # the level really splits


def test_frozen_matrices_reproduce_the_eigenvalues():
    p = ParamSet(SK, max_l={"O": 1})
    lab = ["O", "O"]
    pos = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 2.3]])
    eng = DFTB(p.skset(["O"]), spin=True, unpaired_electrons=2, spin_constants={"O": -0.028},
               scc_tol=1e-10)
    r = eng.calculate(lab, pos, forces=False)
    fm = frozen_matrices(eng, lab, pos, None, None, r.q_in)
    st = eigenstates(eng, lab, pos, None, None, r.q_in)
    assert fm.h.shape == (2, 1, 8, 8) and fm.nspin == 2
    for sp in range(2):
        e = eigh(fm.h[sp, 0], fm.s[sp, 0], eigvals_only=True)
        assert np.abs(e - st.eigenvalues[sp, 0]).max() < 1e-12
    assert not np.allclose(st.eigenvalues[0], st.eigenvalues[1])      # spin split


def test_wanted_states_only(case):
    model, make, geo, kpts, x0, r, st = case
    full = adaptive_eigenvalue_derivatives(make(x0), [model], *geo, r.q_in, kpts=kpts, states=st)
    part = adaptive_eigenvalue_derivatives(make(x0), [model], *geo, r.q_in, kpts=kpts, states=st,
                                           wanted=[0, 1])
    assert np.allclose(part[:, :, :2], full[:, :, :2], atol=1e-13)
    assert np.isnan(part[:, :, -1]).all() or full.shape[2] <= 2
