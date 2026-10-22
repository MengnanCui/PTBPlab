"""pydftb.autodiff mode "implicit" (milestone M4b): the DensityMatrix Function against
finite differences (torch.autograd.gradcheck), the implicit SCC derivative against an
unrolled SCC and against finite differences of the engine, and its agreement with mode
"hf" where both apply."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pydftb.autodiff import Params, evaluate  # noqa: E402
from pydftb.autodiff import api, scc  # noqa: E402
from pydftb.autodiff.api import _engine_for, _implicit_energies  # noqa: E402
from pydftb.autodiff.density import DensityMatrix, Occupation, _level_mean  # noqa: E402
from pydftb.occupations import Smearing  # noqa: E402
from pydftb.validation import gradcheck, refgen  # noqa: E402

KT = 3.166811563e-6        # Hartree per kelvin


def _hermitian(n, size, complex_, levels, rng):
    """Matrices with prescribed eigenvalues (degeneracies allowed) in random bases."""
    out = []
    for k in range(n):
        a = rng.normal(size=(size, size)) + (1j * rng.normal(size=(size, size)) if complex_ else 0)
        q, _ = np.linalg.qr(a)
        lv = np.asarray(levels[k] if np.ndim(levels) == 2 else levels, float)
        out.append(q @ np.diag(lv) @ q.conj().T)
    return torch.tensor(np.array(out))


LV = [-0.5, -0.3, -0.1, 0.0, 0.2, 0.4]
LV2 = [[-0.5, -0.3, -0.12, 0.01, 0.2, 0.4], [-0.45, -0.2, -0.05, 0.03, 0.3, 0.5]]
GAP = [[-0.5, -0.3, -0.05, 0.05, 0.2, 0.4], [-0.4, -0.2, -0.06, 0.07, 0.1, 0.4]]
DM_CASES = {
    # name: (n, complex, levels, groups, smearing, k weights, use the Fermi level)
    "real_fermi": (1, False, LV, [(np.arange(1), 5.0, 2.0)], Smearing("fermi", 1e-2 / KT), None, True),
    "complex_kpoints": (2, True, LV2, [(np.arange(2), 5.3, 2.0)], Smearing("fermi", 1e-2 / KT),
                        np.array([0.7, 0.3]), True),
    "kT_1e-3": (2, True, LV2, [(np.arange(2), 5.3, 2.0)], Smearing("fermi", 1e-3 / KT), None, True),
    "degenerate_at_mu": (2, True, [[-0.5, -0.3, -0.3, 0.0, 0.0, 0.3], [-0.4, -0.2, -0.2, 0.05, 0.05, 0.4]],
                         [(np.arange(2), 7.0, 2.0)], Smearing("fermi", 1e-2 / KT), None, True),
    "spin_groups": (4, True, LV, [(np.array([0, 1]), 3.2, 1.0), (np.array([2, 3]), 2.4, 1.0)],
                    Smearing("fermi", 1e-2 / KT), np.array([0.6, 0.4, 0.6, 0.4]), True),
    "shared_fermi_level": (4, False, LV, [(np.arange(4), 5.2, 1.0)], Smearing("fermi", 1e-2 / KT),
                           np.array([0.3, 0.7, 0.3, 0.7]), True),
    "gaussian": (2, True, LV2, [(np.arange(2), 5.4, 2.0)], Smearing("gaussian", 1e-2 / KT), None, True),
    "methfessel_paxton": (2, True, LV2, [(np.arange(2), 5.4, 2.0)], Smearing("mp", 2e-2 / KT, 2),
                          None, True),
    "midgap_rule": (2, True, GAP, [(np.arange(2), 6.0, 2.0)], Smearing("fermi", 1e-3 / KT), None, True),
    "T0_gapped": (2, True, GAP, [(np.arange(2), 6.0, 2.0)], Smearing("fermi", 0.0), None, True),
    "T0_exact_count": (2, True, GAP, [(np.arange(2), 6.0, 2.0)],
                       Smearing("fermi", 0.0, fermi_level="exact"), None, False),
}


@pytest.mark.parametrize("name", DM_CASES)
def test_density_matrix_gradcheck(name):
    """rho, W, eigenvalues and Fermi levels against finite differences of the whole
    forward pass (eigh and the host Fermi search).  Degenerate levels are weighted by the
    eigenvalue sum only (single degenerate eigenvalues have no derivative); with an exact
    count at T ~ 0 mu lies anywhere in the gap and is left out."""
    n, cplx, levels, groups, sm, kw, use_mu = DM_CASES[name]
    rng = np.random.default_rng(0)
    hb = _hermitian(n, 6, cplx, levels, rng)
    occ = Occupation(np.full(n, 1.0 / n) if kw is None else kw, groups, sm)
    w_rho, w_w = (torch.tensor(rng.normal(size=hb.shape)).to(hb.dtype) for _ in range(2))
    w_e = torch.tensor(rng.normal(size=(n, 6))) if "degenerate" not in name else torch.ones(n, 6)
    w_mu = torch.tensor(rng.normal(size=len(groups))) * float(use_mu)

    def fn(a):
        rho, w, e, mu = DensityMatrix.apply(hb + 0.5 * (a + a.mH), occ)
        return ((w_rho * rho).real.sum() + (w_w * w).real.sum() + (w_e * e).sum()
                + (w_mu * mu).sum())

    a0 = torch.zeros_like(hb, requires_grad=True)
    eps = 1e-7 if "1e-3" in name else 1e-6
    assert torch.autograd.gradcheck(fn, (a0,), eps=eps, atol=1e-6, rtol=1e-5)


def test_degenerate_level_gradient_is_the_level_mean():
    """A perturbation that splits a 3-fold level: single eigenvalues of it have no linear
    derivative, so the gradient of a weighted eigenvalue sum is that of the sum with the
    weights averaged over the level -- exact for any symmetric function of the level."""
    rng = np.random.default_rng(5)
    hb = _hermitian(1, 6, False, [-0.4, -0.2, -0.2, -0.2, 0.1, 0.3], rng)
    occ = Occupation(np.ones(1), [(np.arange(1), 4.0, 2.0)], Smearing("fermi", 0.0))
    w = torch.tensor(rng.normal(size=(1, 6)))
    w_sym = w.clone()
    w_sym[0, 1:4] = w[0, 1:4].mean()
    d = torch.tensor(rng.normal(size=hb.shape))
    d = 0.5 * (d + d.mT)
    a = torch.zeros_like(hb, requires_grad=True)
    (g,) = torch.autograd.grad((w * DensityMatrix.apply(hb + a, occ)[2]).sum(), a)
    h = 1e-6
    fd = ((w_sym * torch.linalg.eigvalsh(hb + h * d)).sum()
          - (w_sym * torch.linalg.eigvalsh(hb - h * d)).sum()) / (2 * h)
    assert abs(float((g * d).sum()) - float(fd)) < 1e-8
    e = torch.tensor([[-1.0, 0.5, 0.5 + 1e-12, 0.5 + 2e-12, 2.0]])
    gm = _level_mean(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]), e)
    assert torch.allclose(gm, torch.tensor([[1.0, 3.0, 3.0, 3.0, 5.0]]))


IMPLICIT_FD = ["h2o", "ch3oh_dftb3", "cu4co_shell_resolved", "o2_triplet_wmatrix",
               "cuo_chain_spin_k422", "li_bcc_k666_1000K"]


@pytest.mark.parametrize("name", IMPLICIT_FD)
def test_implicit_matches_finite_differences(name):
    """E, F, charges, eigenvalues, Fermi levels and forces (and the stress) along random
    parameter directions = Richardson-extrapolated central differences of the engine with
    DFTB+'s conventions (tolerance 1e-6 plus the noise floor of the differences)."""
    if refgen.missing_requirement(refgen.load(name)):
        pytest.skip("reference needs an optional library")
    rows = gradcheck.check_case(name, mode="implicit")
    assert {r[3] for r in rows} >= {"energy", "charges", "eigenvalues", "fermi", "forces"}
    bad = [r for r in rows if not gradcheck.passes(r)]
    assert not bad, bad


@pytest.mark.parametrize("name", ["h2o", "cu4co_shell_resolved", "cuo_chain_spin_k422",
                                  "gdo_rocksalt_k444"])
def test_hf_and_implicit_gradients_agree(name):
    """dF/dtheta is the same in both modes, entry by entry (F is stationary), with the
    fixed-N variational F of mode "hf"."""
    calc, (species, pos, cell, pbc) = gradcheck.reference_engine(name)
    grads = {}
    for mode in ("hf", "implicit"):
        params = Params.from_skset(calc.sk, calc.repulsive, calc.hubbard_derivs,
                                   calc.spin_constants)
        params.requires_grad_(*[k for k in gradcheck.available(calc)
                                if gradcheck.entries(params, k)])
        res = evaluate(calc, species, pos, cell, pbc, params, mode=mode, forces=False)
        leaves = params.leaves()
        g = torch.autograd.grad(res.free_energy, leaves, allow_unused=True)
        grads[mode] = [torch.zeros_like(x) if y is None else y for x, y in zip(leaves, g)]
    for a, b in zip(grads["hf"], grads["implicit"]):
        assert (a - b).abs().max() <= 1e-9 * max(a.abs().max().item(), 1e-12)


def _unrolled_case(name, mixing, iterations):
    calc, (species, pos, cell, pbc) = gradcheck.reference_engine(name, variational=False)
    params = Params.from_skset(calc.sk, calc.repulsive, calc.hubbard_derivs,
                               calc.spin_constants, requires_grad=("tables", "hubbard"))
    res = evaluate(calc, species, pos, cell, pbc, params, mode="implicit", forces=False)
    w = torch.tensor(np.random.default_rng(0).normal(size=res.charges.shape))
    g_impl = torch.autograd.grad((w * res.charges).sum() + res.energy, params.leaves(),
                                 allow_unused=True)
    eng = _engine_for(calc, params)
    _, st, _ = eng._run(species, pos, cell, pbc, False, None, False)
    phi = scc.SCCMap(eng, st, params, torch.tensor(pos), None)
    out = phi.solve(scc.unrolled(phi, st.q0, mixing, iterations))
    qa = out["q_orb"].new_zeros(len(species)).index_add(
        0, torch.as_tensor(st.basis.orb_atom), out["q_orb"])
    e, _ = _implicit_energies(eng, st, phi, out, params, torch.tensor(pos), None)
    g_unr = torch.autograd.grad((w * qa).sum() + e, params.leaves(), allow_unused=True)
    return g_impl, g_unr


@pytest.mark.parametrize("name, mixing, iterations", [("h2o", 0.3, 200),
                                                      ("ch3oh_dftb3", 0.2, 300),
                                                      ("cu4co_shell_resolved", 0.05, 1500)])
def test_implicit_equals_unrolled_scc(name, mixing, iterations):
    """The implicit derivative = backpropagation through an SCC of plain linear mixing run
    to machine precision (measured 1e-14, 4e-12, 6e-13)."""
    g_impl, g_unr = _unrolled_case(name, mixing, iterations)
    for a, b in zip(g_impl, g_unr):
        if a is not None and b is not None:
            assert (a - b).abs().max() <= 1e-8 * a.abs().max()


def test_adjoint_dense_and_gmres(monkeypatch):
    """The adjoint SCC equation, by the dense solve and by GMRES, to 1e-10 residual."""
    calc, (species, pos, cell, pbc) = gradcheck.reference_engine("cuo_chain_spin_k422",
                                                                variational=False)
    out = {}
    for dense_max in (64, 0):
        monkeypatch.setattr(scc, "DENSE_MAX", dense_max)
        params = Params.from_skset(calc.sk, calc.repulsive, calc.hubbard_derivs,
                                   calc.spin_constants, requires_grad=("hubbard",))
        res = evaluate(calc, species, pos, cell, pbc, params, mode="implicit", forces=False)
        out[dense_max] = torch.autograd.grad(res.charges.sum() + res.energy, params.leaves())
        assert res.adjoint.residual < 1e-10
    for a, b in zip(out[64], out[0]):
        assert (a - b).abs().max() <= 1e-9 * a.abs().max()


@pytest.mark.parametrize("fd_delta", [1.22e-4, 1e-2])
def test_implicit_forces_with_dftbplus_finite_differences(fd_delta):
    """With fd_delta (DFTB+'s finite-difference H0/S derivatives; PTBP runs 0.01 Bohr) the
    forces of mode "implicit" are the engine's and their parameter derivative is that of
    the engine's forces."""
    doc = refgen.load("h2o")
    case = refgen.CASES_BY_NAME[doc["case"]]
    _, calc = refgen.run_pydftb(case, doc["input"], engine=True, scc_tol=1e-13, fd_delta=fd_delta)
    species, pos, cell, pbc = gradcheck._structure(doc["input"])
    ref = calc.calculate(species, pos, cell, pbc)
    params = Params.from_skset(calc.sk, calc.repulsive, requires_grad=("tables",))
    res = evaluate(calc, species, pos, cell, pbc, params, mode="implicit")
    assert np.array_equal(res.forces.detach().numpy(), ref.forces)
    eng = _engine_for(calc, params)
    _, st, sc = eng._run(species, pos, cell, pbc, True, None, False)
    phi = scc.SCCMap(eng, st, params, torch.tensor(pos), None, need_w=True)
    q_hat, _ = scc.attach(phi, sc.q_in)
    dens = api._torch_densities(phi, phi.solve(q_hat))
    _, f_t, _ = api._explicit(eng, st, params, torch.tensor(pos), None, dens, True, False, True)
    assert np.abs(f_t.detach().numpy() - ref.forces).max() < 1e-11
    rng = np.random.default_rng(1)
    w = rng.normal(size=ref.forces.shape)
    t = gradcheck.tangent(params, "tables", rng)
    g = torch.autograd.grad((torch.as_tensor(w) * res.forces).sum(), list(params.tables.values()),
                            allow_unused=True)
    ad = sum(float((x * torch.as_tensor(t[k])).sum()) for k, x in zip(params.tables, g)
             if x is not None)
    fd, spread = gradcheck._richardson(
        lambda h: {"f": (w * _engine_for(calc, gradcheck.shifted(params, "tables", t, h))
                         .calculate(species, pos, cell, pbc).forces).sum()}, 2e-5)
    assert abs(ad - fd["f"]) <= 1e-6 * abs(fd["f"]) + spread["f"]


def test_implicit_values_are_the_engines():
    calc, (species, pos, cell, pbc) = gradcheck.reference_engine("cuo_chain_spin_k422",
                                                                variational=False)
    ref = calc.calculate(species, pos, cell, pbc)
    res = evaluate(calc, species, pos, cell, pbc, mode="implicit")
    assert res.energy.item() == ref.energy and res.free_energy.item() == ref.free_energy
    assert np.array_equal(res.charges.detach().numpy(), ref.charges)
    assert np.array_equal(res.forces.detach().numpy(), ref.forces)
    assert np.array_equal(res.stress.detach().numpy(), ref.stress)
    assert np.array_equal(res.eigenvalues.detach().numpy(), np.array(ref.eigenvalues))
    assert np.array_equal(res.fermi.detach().numpy(), np.array(ref.fermi_spin))


def test_engine_backend_does_not_matter():
    """The densities may come from the NumPy or the torch engine (M4a): same gradients."""
    doc = refgen.load("h2o")
    case = refgen.CASES_BY_NAME[doc["case"]]
    species, pos, cell, pbc = gradcheck._structure(doc["input"])
    grads = []
    for backend in ("numpy", "torch"):
        _, calc = refgen.run_pydftb(case, doc["input"], engine=True, scc_tol=1e-12, backend=backend)
        params = Params.from_skset(calc.sk, calc.repulsive, requires_grad=("tables", "hubbard"))
        res = evaluate(calc, species, pos, cell, pbc, params, mode="implicit")
        grads.append(torch.autograd.grad(res.free_energy + res.forces.sum(), params.leaves(),
                                         allow_unused=True))
    for a, b in zip(*grads):
        if a is not None:
            assert (a - b).abs().max() <= 1e-12 * a.abs().max()
