"""The gradient of PTBP's loss (utils.loss_grad, milestone M4c): the pure loss equals
Loss bit for bit, the residual terms and their sigma derivatives, the EOS sensitivity, the
inner minimum (smooth, kink, bound) and the kink-corrected gradient against exact
differences, and a BO step with the gradient hook (same loss and files)."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import ase.io
import numpy as np
import pytest

import utils.data as data
import utils.engines as eng
import utils.loss as L
from utils import dftb_stage
from utils import loss_grad as LG

ROOT = Path(__file__).resolve().parents[1]
CU_SKF_GZ = ROOT / "pydftb" / "tests" / "data" / "skf" / "ptbp_hcocu" / "Cu-Cu.skf.gz"
R0 = (2.749, 12.036)
linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork backend")


@pytest.fixture(autouse=True)
def _restore_engine():
    saved = eng.settings()
    yield
    eng._SETTINGS.clear()
    eng._SETTINGS.update(saved)


def _skf(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    with gzip.open(CU_SKF_GZ, "rb") as fi, open(path / "Cu-Cu.skf", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    return path


def _args(tmp_path, mode, eos):
    return argparse.Namespace(eos_points=eos, kpt_density=5.0, multi_element=None,
                              optimization_option=mode, log_file=str(tmp_path / "log.out"),
                              E0s={29: -1500.0}, xc="GGA_X_PBE+GGA_C_PBE",
                              superposition="density", results_dir="results")


@pytest.fixture(scope="module")
def images():
    return ase.io.read(ROOT / "dft" / "dft.xyz", ":")


@pytest.fixture
def eg_step(tmp_path, monkeypatch, images):
    """A two-group energygeometry BO step (22 structures) after its DFTB stage."""
    monkeypatch.chdir(_skf(tmp_path / "eg"))
    eng.configure(engine="pydftb")
    sub = images[:22]
    _, bolt = data.get_bolt_factor(sub[5::11], "boltzmann", 3000)
    loss = L.Loss(args_input=_args(tmp_path, "energygeometry", 11), dft_input=sub,
                  bolt_f11=bolt, opt_elem=[])
    dftb_stage.write_outputs(*dftb_stage.run_dataset(sub, bolt, 11, 5.0))
    return loss, LG.StepData.from_files(loss, *R0)


@pytest.fixture
def ds_step(tmp_path, monkeypatch, images):
    """A dataset-mode BO step on seven cheap structures (Gamma-only and small grids)."""
    monkeypatch.chdir(_skf(tmp_path / "ds"))
    eng.configure(engine="pydftb")
    pick = [0, 1, 5, 6, 7, 60, 65]
    sub = [images[i] for i in pick]
    _, bolt_all = data.get_bolt_factor(images[5::11], "boltzmann", 3000)
    bolt = [bolt_all[i] for i in pick]
    loss = L.Loss(args_input=_args(tmp_path, "dataset", 1), dft_input=sub, bolt_f11=bolt,
                  opt_elem=[])
    dftb_stage.write_outputs(*dftb_stage.run_dataset(sub, bolt, 1, 5.0))
    return loss, LG.StepData.from_files(loss, *R0)


# ---- the loss, bit for bit ----------------------------------------------------------
@linux
def test_mu_is_the_loss_of_parallel_repulsion(eg_step, capsys):
    loss, step = eg_step
    for sigma in (0.15, 0.3305859375, 0.47, 0.8):
        ref = loss.parallel_repulsion(np.array([sigma]), *R0, False)
        assert float(LG.mu(step, sigma)).hex() == float(ref).hex()


@linux
def test_mu_is_the_loss_of_unparallel_repulsion(ds_step):
    loss, step = ds_step
    for sigma in (0.1, 0.33, 0.4673979, 0.7):
        ref = loss.unparallel_repulsion(np.array([sigma]), *R0, False)
        assert float(LG.mu(step, sigma)).hex() == float(ref).hex()


@linux
@pytest.mark.parametrize("which", ["eg", "ds"])
def test_terms_give_the_loss_and_its_sigma_derivative(which, eg_step, ds_step):
    """P sum w|r| is the loss; dr/dsigma against differences of every residual: energies and
    forces at a small step, the volume ratios (through the EOS fits, noisy at ~1e-8) at a
    larger one.  (Where the repulsion is tiny, sigma < 0.3 here, differences of physrep's
    splines are noise: the spline solve leaves ~1e-13 Ha in the tail.)"""
    loss, step = eg_step if which == "eg" else ds_step
    for sigma in (0.42, 0.55):
        t = LG.terms(step, sigma)
        m = LG.mu(step, sigma)
        assert abs(t.value() - m) <= 1e-12 * abs(m)
        left, right = t.slopes()
        assert left == right                           # no kink at these sigma
        for kinds, h, tol in ((("E", "F"), 1e-5, 1e-4), (("V",), 1e-3, 1e-3)):
            sel = [j for j, k in enumerate(t.kind) if k in kinds]
            if not sel:
                continue
            fd = (LG.terms(step, sigma + h).r - LG.terms(step, sigma - h).r)[sel] / (2 * h)
            scale = max(np.abs(fd).max(), 1e-6)
            assert np.abs(t.drds[sel] - fd).max() <= tol * scale


# ---- EOS sensitivity ----------------------------------------------------------------
def _bm_data(seed=0):
    rng = np.random.default_rng(seed)
    vols = np.linspace(10.0, 14.0, 11)
    p = (-3.7, 0.86, 4.6, 11.95)
    return vols, LG.birch_murnaghan(vols, *p) + rng.normal(scale=2e-3, size=vols.size), p


def _exact_fit(vols, ens, p):
    """The least-squares Birch-Murnaghan parameters by Newton steps on grad S = 0 with exact
    derivatives (curve_fit stops ~1e-8 short in the parameters)."""
    import torch
    v, e = torch.as_tensor(vols), torch.as_tensor(ens)
    q = torch.as_tensor(np.array(p, dtype=float))

    def s(x):
        return ((LG.birch_murnaghan(v, x[0], x[1], x[2], x[3]) - e) ** 2).sum()
    for _ in range(50):
        dq = torch.linalg.solve(torch.autograd.functional.hessian(s, q),
                                torch.autograd.functional.jacobian(s, q))
        q = q - dq
        if dq.abs().max() < 1e-15:
            break
    return q.numpy()


def test_eos_sensitivity_is_the_implicit_derivative_of_the_fit():
    pytest.importorskip("torch")
    vols, ens, p0 = _bm_data()
    p = _exact_fit(vols, ens, p0)
    dv0 = LG.eos_v0_sensitivity(vols, ens, p)
    for j in (0, 4, 10):
        de = np.zeros_like(ens)
        de[j] = 1e-5
        fd = (_exact_fit(vols, ens + de, p)[3] - _exact_fit(vols, ens - de, p)[3]) / 2e-5
        assert abs(dv0[j] - fd) <= 1e-8 * max(np.abs(dv0))
    assert abs(dv0.sum()) <= 1e-8 * np.abs(dv0).max()   # a constant shift moves only E0


def test_eos_sensitivity_matches_ptbp_fits():
    pytest.importorskip("torch")
    vols, ens, _ = _bm_data(1)

    class A:                                            # Atoms stand-in for L._eos_fit
        def __init__(self, v, e):
            self.v, self.e = v, e

        def get_volume(self):
            return self.v

        def get_potential_energy(self):
            return self.e

        def __len__(self):
            return 1
    e0, v0, b, bp = L._eos_fit([A(v, e) for v, e in zip(vols, ens)])
    dv0 = LG.eos_v0_sensitivity(vols, ens, (e0, b, bp, v0))
    j = 6
    fd = (L._eos_fit([A(v, e + (1e-4 if k == j else 0)) for k, (v, e) in enumerate(zip(vols, ens))])[1]
          - L._eos_fit([A(v, e - (1e-4 if k == j else 0)) for k, (v, e) in enumerate(zip(vols, ens))])[1]) / 2e-4
    assert abs(dv0[j] - fd) <= 1e-4 * np.abs(dv0).max()


# ---- inner minimum and the kink-corrected gradient (synthetic losses) ---------------
class Model:
    """mu(x, sigma) = sum_j w_j |a_j + b_j sigma + c_j sigma^2 + g_j x|."""

    def __init__(self, a, b, c, g, w):
        self.a, self.b, self.c, self.g, self.w = (np.asarray(v, float) for v in (a, b, c, g, w))
        self.x = 0.0

    def r(self, s):
        return self.a + self.b * s + self.c * s * s + self.g * self.x

    def mu(self, s):
        return float(np.sum(self.w * np.abs(self.r(s))))

    def terms(self, s):
        n = len(self.a)
        return LG.Terms(float(s), 1.0, self.w, self.r(s), self.b + 2 * self.c * s, ["E"] * n,
                        np.eye(n), np.zeros((n, 0)), [], [None] * n, [1] * n)

    def mu_star(self, lo=0.1, hi=0.8):
        """The exact inner minimum: mu at the bounds, at every kink and at the stationary
        point of every piece between kinks (residuals at most quadratic in sigma)."""
        cand = [lo, hi]
        for a, b, c in zip(self.a + self.g * self.x, self.b, self.c):
            roots = np.roots([c, b, a]) if c != 0 else ([-a / b] if b != 0 else [])
            cand += [float(np.real(r)) for r in roots if abs(np.imag(r)) < 1e-14]
        cand = sorted(x for x in cand if lo <= x <= hi)
        for x0, x1 in zip(cand[:-1], cand[1:]):
            sgn = np.sign(self.r(0.5 * (x0 + x1)))
            lin, quad = np.sum(self.w * sgn * self.b), np.sum(self.w * sgn * self.c)
            if quad != 0 and x0 < -lin / (2 * quad) < x1:
                cand.append(-lin / (2 * quad))
        return min(self.mu(x) for x in cand)


def _fake_step():
    loss = SimpleNamespace(cov_radius=1.0)
    return SimpleNamespace(loss=loss, r0_w=2.0, r0_d=4.0)       # dlnP = 0 here


@pytest.mark.parametrize("case, model, sigma_hat, expect", [
    ("kink", Model([-0.3, 0.5, 0.2], [1, -1, 0.5], [0, 0, 0], [0.7, 0.2, -0.4], [1, 1, 0.2]),
     0.31, 0.3),
    ("smooth", Model([1.0], [-0.8], [1.0], [0.3], [1.0]), 0.35, 0.4),
    ("bound", Model([1.0, 2.0], [1.0, 0.5], [0, 0], [0.2, 0.1], [1, 1]), 0.12, 0.1),
    ("kink", Model([-0.45, 0.6, 0.1, 0.3], [1, -1.5, 2, -0.2], [0.3, 0.1, 0, 0],
                   [0.5, -0.3, 0.2, 0.9], [1, 0.8, 0.3, 0.5]), 0.5, None),
])
def test_inner_minimum_and_gradient(monkeypatch, case, model, sigma_hat, expect):
    monkeypatch.setattr(LG, "mu", lambda step, s: model.mu(s))
    monkeypatch.setattr(LG, "terms", lambda step, s: model.terms(s))
    inner = LG.refine_sigma(None, sigma_hat)
    assert inner.case == case, inner
    if expect is not None:
        assert abs(inner.sigma - expect) < 1e-7
    assert inner.mu <= model.mu(sigma_hat) + 1e-12
    g = LG.assemble(_fake_step(), inner, {"r0_w": model.g})["r0_w"]
    h = 1e-5
    mus = []
    for x in (h, -h):
        model.x = x
        mus.append(model.mu_star())
    model.x = 0.0
    fd = (mus[0] - mus[1]) / (2 * h)
    assert abs(g - fd) <= 1e-6 * max(abs(fd), 1.0)


def test_kink_correction_matters(monkeypatch):
    """Without the correction the derivative of a kink minimum is wrong."""
    model = Model([-0.3, 0.5, 0.2], [1, -1, 0.5], [0, 0, 0], [0.7, 0.2, -0.4], [1, 1, 0.2])
    monkeypatch.setattr(LG, "mu", lambda step, s: model.mu(s))
    monkeypatch.setattr(LG, "terms", lambda step, s: model.terms(s))
    inner = LG.refine_sigma(None, 0.31)
    parts = {}
    g = LG.assemble(_fake_step(), inner, {"r0_w": model.g}, parts=parts)["r0_w"]
    assert abs(parts["r0_w"]["kink"]) > 0.1 * abs(g)


def test_penalty_derivative():
    step = SimpleNamespace(loss=SimpleNamespace(cov_radius=2.4944), r0_w=2.749, r0_d=12.036)
    d = LG.dlnp(step)
    for name, dx in (("r0_w", (1e-6, 0)), ("r0_d", (0, 1e-6))):
        p = [L.r0_penalty(step.r0_w + s * dx[0], step.r0_d + s * dx[1], 2.4944) for s in (1, -1)]
        fd = (np.log(p[0]) - np.log(p[1])) / 2e-6
        assert abs(d[name] - fd) < 1e-8


def test_force_weights_skip_kinks_and_zeros():
    t = LG.Terms(0.3, 2.0, np.array([0.9, 0.1, 0.1, 0.1]), np.array([0.5, -1.0, 2.0, 0.0]),
                 np.array([1.0, 1.0, 1.0, 0.0]), ["E", "F", "F", "F"], np.zeros((4, 1)),
                 np.zeros((4, 0)), [], [None, (0, 0, 0), (0, 0, 1), (0, 1, 2)], [2])
    w = LG.force_weights(t, exclude=(2,))
    assert np.allclose(w[0], [[-0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])


# ---- a BO step with the gradient hook -------------------------------------------------
@linux
def test_grad_request_keeps_the_loss_and_files(tmp_path, monkeypatch, images):
    """Loss_EnergyGeometry with GradRequest (a synthetic tangent instead of hotcent): the
    same loss bit for bit and the same files as without; a finite gradient whose
    r0_d part is the penalty's (zero tangent)."""
    pytest.importorskip("torch")
    from utils import hotcent_tangent as HT
    from utils.hotcent_tangent import Tangent
    sub = images[:22]
    _, bolt = data.get_bolt_factor(sub[5::11], "boltzmann", 3000)
    eng.configure(engine="pydftb")

    class Step(L.Loss):
        pass

    def fake_hotcent(self, superposition, opt_elem=(), label="ptbp"):
        _skf(Path("."))

    monkeypatch.setattr(L.hotpy, "full_hotcent", fake_hotcent)

    def fake_tangent(confinement, hp, base_dir, labels, r0_w, r0_d, **kw):
        from pydftb.skf import SKFile
        table = SKFile.read(str(Path(base_dir) / "Cu-Cu.skf"), homo=True).table
        return {"r0_w": Tangent("r0_w", 0.04, tables={("Cu", "Cu"): 0.01 * table}),
                "r0_d": Tangent("r0_d", 0.0, zero=True, probed=True)}

    monkeypatch.setattr(HT, "compute", fake_tangent)
    out = {}
    for g in (False, True):
        wd = tmp_path / f"run_{g}"
        wd.mkdir()
        monkeypatch.chdir(wd)
        loss = Step(args_input=_args(wd, "energygeometry", 11), dft_input=sub, bolt_f11=bolt,
                    opt_elem=[], cov_radius=2.4944384861047455, path=wd)
        loss.args_input.para_file = str(wd / "par.out")
        req = LG.GradRequest() if g else None
        loss.grad_request = req
        mu = float(loss.Loss_EnergyGeometry(list(R0)))
        d = wd / "results" / ("opt_2.749000_12.036000" if g else "opt_2.749_12.036")
        files = {n: hashlib.sha256((d / n).read_bytes()).hexdigest()
                 for n in ("dft_final.xyz", "dftb_band.xyz", "dftb_full.xyz", "search_index.txt",
                           "Cu-Cu.skf")}
        out[g] = (mu, files, req)
    assert out[False][0].hex() == out[True][0].hex()
    assert out[False][1] == out[True][1]
    res = out[True][2].result
    assert res.grad is not None, res.reason
    assert np.isfinite(res.grad["r0_w"]) and res.case in ("smooth", "kink", "bound")
    dl = LG.dlnp(SimpleNamespace(loss=SimpleNamespace(cov_radius=2.4944384861047455),
                                 r0_w=R0[0], r0_d=R0[1]))
    assert res.parts["r0_d"]["energies"] == 0.0
    assert abs(res.grad["r0_d"] - res.parts["r0_d"]["penalty"]) <= 1e-12
    mu_star = res.mu - res.gap                       # mu at sigma*, the gradient's point
    assert abs(res.parts["r0_d"]["penalty"] - mu_star * dl["r0_d"]) <= 1e-12 * res.mu
