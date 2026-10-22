"""pydftb.autodiff (milestone M4b): the torch kernels against the NumPy engine, and the
gradients of mode "hf" against the engine's analytic forces and stress and against finite
differences of its free energy (``pydftb.validation.gradcheck``)."""
import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pydftb import ParamSet  # noqa: E402
from pydftb.autodiff import GradientUnavailable, Params, evaluate  # noqa: E402
from pydftb.autodiff import api  # noqa: E402
from pydftb.autodiff.api import _engine_for  # noqa: E402
from pydftb.coulomb import _short, _short_near  # noqa: E402
from pydftb.hamiltonian import Basis, build_pairs, pair_values  # noqa: E402
from pydftb.occupations import Smearing  # noqa: E402
from pydftb.thirdorder import dgamma_du, dgamma_du_onsite  # noqa: E402
from pydftb.validation import gradcheck, refgen  # noqa: E402

from .conftest import SKF_GDO, SKF_HCOCU  # noqa: E402


def _engine(name, **options):
    doc = refgen.load(name)
    why = refgen.missing_requirement(doc)
    if why:
        pytest.skip(why)
    _, calc = refgen.run_pydftb(refgen.CASES_BY_NAME[doc["case"]], doc["input"], engine=True,
                                scc_tol=1e-12, **options)
    return calc, gradcheck._structure(doc["input"])


def _t(a, grad=False):
    return torch.tensor(np.asarray(a, dtype=float), dtype=torch.float64, requires_grad=grad)


# ---- kernels --------------------------------------------------------------------------------
@pytest.mark.parametrize("skdir, labels", [(SKF_HCOCU, ["Cu", "O"]), (SKF_GDO, ["Gd", "O"])])
def test_sk_table_interpolation(skdir, labels):
    """The table as weights times rows (differentiable in the table) = the engine's
    interpolation; autograd dI/dr = its derivative.  Beyond the last grid point DFTB+'s
    tail uses a second difference with step 1e-5, ill-conditioned to ~1e-12 absolute."""
    sk = ParamSet(skdir).skset(labels)
    for f in sk.files.values():
        n_in = f.ngrid * f.dr
        r = np.concatenate([np.linspace(0.4, n_in - 1e-3, 1500), [f.dr * 7, f.dr * 100],
                            np.linspace(n_in + 1e-3, f.cutoff + 0.2, 200)])
        rt = _t(r, grad=True)
        out = f.evaluate_table(rt, _t(f.table))
        ref, dref = f.evaluate(r, deriv=True, full=False)
        inside = r < n_in
        scale = np.abs(f.table).max()
        assert np.abs(out.detach().numpy() - ref)[inside].max() < 1e-13 * scale
        assert np.abs(out.detach().numpy() - ref)[~inside].max() < 1e-11
        cols = np.nonzero(np.abs(f.table).max(0) > 0)[0][:3]
        for c in cols:
            (d,) = torch.autograd.grad(out[:, c].sum(), rt, retain_graph=True)
            assert np.abs(d.numpy() - dref[:, c])[inside].max() < 1e-11 * scale
            assert np.abs(d.numpy() - dref[:, c])[~inside].max() < 1e-6


@pytest.mark.parametrize("fd_delta", [None, 1.22e-4])
def test_pair_values_equal_build_pairs(fd_delta):
    """H0/S of every pair element (and DFTB+'s finite differences) from torch tables."""
    sk = ParamSet(SKF_GDO).skset(["Gd", "O"])
    pos = np.array([[0.0, 0.0, 0.0], [4.45, 0.1, -0.05], [0.2, 4.4, 0.1]])
    cell = np.diag([8.9, 8.9, 8.9])
    species = ["Gd", "O", "O"]
    basis = Basis.build(species, sk)
    pb = build_pairs(species, pos, cell, np.ones(3, bool), sk, basis, derivatives=True,
                     fd_delta=fd_delta)
    out = pair_values(pb, sk, _t(pb.vec), tables={k: _t(f.table) for k, f in sk.files.items()},
                      fd_delta=fd_delta)
    assert np.abs(out[0].numpy() - pb.h).max() < 1e-11
    assert np.abs(out[1].numpy() - pb.s).max() < 1e-11
    if fd_delta is not None:
        assert np.abs(out[2].numpy() - pb.dh).max() < 1e-6
        assert np.abs(out[3].numpy() - pb.ds).max() < 1e-6


def test_short_gamma_all_branches():
    """S(r) of the second-order gamma for equal, near (Hermite-Genocchi) and distinct U."""
    rng = np.random.default_rng(3)
    r = rng.uniform(0.5, 15.0, 300)
    ta = rng.uniform(0.8, 2.0, 300)
    tb = np.concatenate([ta[:100], ta[100:200] * (1 + rng.uniform(-0.03, 0.03, 100)),
                         ta[200:] * rng.uniform(1.2, 2.0, 100)])
    s, ds = _short(r, ta, tb)
    rt, at, bt = _t(r, True), _t(ta, True), _t(tb)
    st, _ = _short(rt, at, bt)
    assert np.abs(st.detach().numpy() - s).max() < 1e-14
    gr, ga = torch.autograd.grad(st.sum(), (rt, at))
    assert np.abs(gr.numpy() - ds).max() < 1e-13
    near = slice(100, 200)
    _, _, dta, _ = _short_near(r[near], ta[near], tb[near], tau_derivs=True)
    assert np.abs(ga.numpy()[near] - dta).max() < 1e-12


def test_third_order_kernels():
    """dGamma/dU of DFTB3, torch = NumPy; DFTB+'s closed form cancels for close U values
    (both implementations then differ in the last ~3 digits)."""
    rng = np.random.default_rng(4)
    ua = rng.uniform(0.2, 0.6, 300)
    ub = np.concatenate([ua[:50], ua[50:250] * rng.uniform(1.3, 2.0, 200),
                         ua[250:] * rng.uniform(1.02, 1.1, 50)])
    r = rng.uniform(0.8, 12.0, 300)
    damped = rng.random(300) < 0.3
    g, gr = dgamma_du(ua, ub, r, damped, 4.0)
    gt, grt = dgamma_du(_t(ua), _t(ub), _t(r), damped, 4.0)
    for tol, sl in ((1e-13, slice(0, 250)), (1e-11, slice(250, 300))):
        assert np.abs(gt.numpy() - g)[sl].max() < tol * np.abs(g).max()
        assert np.abs(grt.numpy() - gr)[sl].max() < tol * np.abs(gr).max()
    assert np.abs(dgamma_du_onsite(_t(ua), _t(ub)).numpy() - dgamma_du_onsite(ua, ub)).max() < 1e-15


def test_repulsive_energy_and_autograd():
    sk = ParamSet(SKF_HCOCU).skset(["Cu", "O"])
    for mode in ("skf", "polynomial"):
        for key in sk.files:
            rep = sk.repulsive(*key, mode)
            if rep is None:
                continue
            r = np.linspace(0.6, rep.cutoff + 0.5, 700)
            e, de = rep(r)
            rt = _t(r, True)
            et = rep.energy(rt)
            assert np.abs(et.detach().numpy() - e).max() < 1e-13 * max(1.0, np.abs(e).max())
            (d,) = torch.autograd.grad(et.sum(), rt)
            assert np.abs(d.numpy() - de).max() < 1e-12 * max(1.0, np.abs(de).max())


# ---- mode "hf" ------------------------------------------------------------------------------
HF_CASES = ["h2o", "ch3oh_dftb3", "water_dimer_hdamp", "cu4co_shell_resolved",
            "o2_triplet_wmatrix", "cuo_chain_spin_k422", "gdo_rocksalt_k444",
            "water_box_d3bj_3body", "li_bcc_k666_1000K", "cu8_distorted_k355_mp2_2000K"]


@pytest.mark.parametrize("name", HF_CASES)
def test_hf_forces_and_stress_equal_engine(name):
    """Autograd of the surrogate (position leaf, strain) = the engine's analytic forces and
    stress (measured <= 4e-15 Ha/Bohr and 7e-15 Ha/Bohr^3); the values returned are the
    engine's own."""
    calc, (species, pos, cell, pbc) = _engine(name)
    ref = calc.calculate(species, pos, cell, pbc)
    res = evaluate(calc, species, pos, cell, pbc)
    raw = res.raw                      # the engine run inside evaluate (D3 threads may
    assert res.free_energy.item() == raw.free_energy and res.energy.item() == raw.energy
    assert np.array_equal(res.forces.numpy(), raw.forces)       # reorder its sums)
    assert np.array_equal(res.charges.numpy(), raw.charges)
    params = Params.from_skset(calc.sk, calc.repulsive, calc.hubbard_derivs, calc.spin_constants)
    eng = _engine_for(calc, params)
    _, st, sc = eng._run(species, pos, cell, pbc, True, None, False)
    _, f_t, s_t = api._explicit(eng, st, params, _t(pos), None if cell is None else _t(cell),
                                api._host_densities(eng, st, sc), True, cell is not None, False)
    assert np.abs(f_t.numpy() - ref.forces).max() < 1e-12
    if ref.stress is not None:
        assert np.abs(s_t.numpy() - ref.stress).max() < 1e-12


@pytest.mark.parametrize("name", ["h2o", "ch3oh_dftb3", "o2_triplet_wmatrix",
                                  "cu4co_shell_resolved", "cuo_chain_spin_k422",
                                  "li_bcc_k666_1000K", "gdo_rocksalt_k444"])
def test_hf_parameter_gradients(name):
    """dF/dtheta along random directions of every parameter kind = the Richardson-
    extrapolated central difference of the engine (fixed-N, variational F)."""
    rows = gradcheck.check_case(name)
    assert {r[2] for r in rows} >= {"tables", "onsite"}
    bad = [r for r in rows if not gradcheck.passes(r)]
    assert not bad, bad


def test_position_and_cell_leaves():
    """The caller's tensors receive dF/dR = -forces and dF/dcell (at fixed Cartesian
    positions); with them the virial g^T R + C^T H gives the stress."""
    calc, (species, pos, cell, pbc) = _engine("gdo_rocksalt_k444")
    pt, ct = _t(pos, True), _t(cell, True)
    res = evaluate(calc, species, pt, ct, pbc)
    res.free_energy.backward()
    assert np.abs(pt.grad.numpy() + res.forces.numpy()).max() < 1e-14
    w = pt.grad.numpy().T @ pos + ct.grad.numpy().T @ cell
    vol = abs(np.linalg.det(cell))
    assert np.abs(-0.5 * (w + w.T) / vol - res.stress.numpy()).max() < 1e-14
    h = 1e-5
    for i, k in [(0, 0), (1, 2)]:
        f = []
        for sign in (1, -1):
            c = cell.copy()
            c[i, k] += sign * h
            f.append(calc.calculate(species, pos, c, pbc, forces=False).free_energy)
        assert abs((f[0] - f[1]) / (2 * h) - ct.grad[i, k].item()) < 1e-8


def test_gradient_unavailable():
    # E with smearing: mode "hf" has only dF
    calc, (species, pos, cell, pbc) = _engine("li_bcc_k666_1000K")
    res = evaluate(calc, species, _t(pos, True), cell, pbc, forces=False)
    with pytest.raises(GradientUnavailable):
        res.energy.backward()
    # Methfessel-Paxton with DFTB+'s TS: F is not variational
    calc, (species, pos, cell, pbc) = _engine("cu8_distorted_k355_mp2_2000K")
    params = Params.from_skset(calc.sk, requires_grad=("onsite",))
    res = evaluate(calc, species, pos, cell, pbc, params, forces=False)
    with pytest.raises(GradientUnavailable, match="variational"):
        res.free_energy.backward()
    # an SCC that did not converge
    calc, (species, pos, cell, pbc) = _engine("h2o", max_scc=2, warn_unconverged=False)
    with pytest.raises(GradientUnavailable):
        evaluate(calc, species, pos, cell, pbc)


def test_params_round_trip_and_files_untouched():
    calc, (species, pos, cell, pbc) = _engine("ch3oh_dftb3")
    params = Params.from_skset(calc.sk, calc.repulsive, calc.hubbard_derivs,
                               requires_grad=("tables", "hubbard"))
    a = calc.calculate(species, pos, cell, pbc)
    b = _engine_for(calc, params).calculate(species, pos, cell, pbc)
    assert a.free_energy == b.free_energy and np.array_equal(a.forces, b.forces)
    before = {k: f.table.copy() for k, f in calc.sk.files.items()}
    with torch.no_grad():
        for v in params.tables.values():
            v.mul_(1.01)
    moved = _engine_for(calc, params).calculate(species, pos, cell, pbc)
    assert moved.free_energy != a.free_energy
    assert all(np.array_equal(f.table, before[k]) for k, f in calc.sk.files.items())
    with pytest.raises(ValueError):
        params.requires_grad_("nonsense")


def test_hf_needs_fixed_n_for_exact_parameter_gradients():
    """DFTB+'s mid-gap Fermi level (integer electron count, count there within 1e-7) lets N
    drift, which the fixed-N gradient does not see; fermi_level='exact' removes it."""
    calc, (species, pos, cell, pbc) = _engine("cu4co_shell_resolved")
    params = Params.from_skset(calc.sk, calc.repulsive)
    t = gradcheck.tangent(params, "tables", np.random.default_rng(0))
    params.requires_grad_("tables")
    errs = {}
    for level in ("dftb+", "exact"):
        eng = copy_with_smearing(calc, fermi_level=level)
        res = evaluate(eng, species, pos, cell, pbc, params, forces=False)
        grads = torch.autograd.grad(res.free_energy, list(params.tables.values()),
                                    allow_unused=True)
        ad = sum(float((g * torch.as_tensor(t[k])).sum())
                 for k, g in zip(params.tables, grads) if g is not None)
        fd = []
        for h in (1e-5, 5e-6):
            f = [_engine_for(eng, gradcheck.shifted(params, "tables", t, s * h))
                 .calculate(species, pos, cell, pbc, forces=False).free_energy for s in (1, -1)]
            fd.append((f[0] - f[1]) / (2 * h))
        errs[level] = abs((4 * fd[1] - fd[0]) / 3 - ad) / abs(ad)
    assert errs["exact"] < 1e-7 and errs["dftb+"] > 1e-6


def copy_with_smearing(calc, **changes):
    import copy
    eng = copy.copy(calc)
    eng.smearing = dataclasses.replace(calc.smearing, **changes)
    return eng


def test_smearing_options_validated():
    with pytest.raises(ValueError):
        Smearing("fermi", 300.0, fermi_level="middle")
    assert Smearing("mp", 300.0, 2).variational is False
    assert Smearing("mp", 300.0, 2, "variational").variational
    assert Smearing("fermi", 300.0).variational
