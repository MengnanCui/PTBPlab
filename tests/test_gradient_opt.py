"""Gradient optimisers (utils.gradient_opt, milestone M4c): projected BFGS on analytic
functions with bounds and failed evaluations, reproducibility, the optimiser entry points
on a stand-in loss, and the command-line options."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from utils import gradient_opt as GO
from utils.gradient_opt import Evaluation, projected_bfgs


def _fun(f, g, fail=lambda x: False, calls=None):
    def fun(x):
        if calls is not None:
            calls.append(np.array(x))
        if fail(x):
            return Evaluation(np.array(x), 98.0, None, failed=True)    # no gradient
        return Evaluation(np.array(x), float(f(x)), np.asarray(g(x), float))
    return fun


ROSEN = (lambda x: (1 - x[0]) ** 2 + 10 * (x[1] - x[0] ** 2) ** 2 + 0.5,
         lambda x: [-2 * (1 - x[0]) - 40 * x[0] * (x[1] - x[0] ** 2), 20 * (x[1] - x[0] ** 2)])
BOWL = (lambda x: (x[0] - 5) ** 2 + (x[1] - 1) ** 2 + 1,
        lambda x: [2 * (x[0] - 5), 2 * (x[1] - 1)])


def test_unconstrained_minimum():
    best, hist = projected_bfgs(_fun(*ROSEN), [2.5, 3.0], [0.0, 0.0], [3.0, 4.0], maxiter=80,
                                gtol=1e-7, xtol=1e-10)
    assert np.allclose(best.x, [1.0, 1.0], atol=1e-5) and abs(best.f - 0.5) < 1e-9
    assert hist.message == "projected gradient below gtol"


def test_minimum_on_a_bound():
    best, hist = projected_bfgs(_fun(*BOWL), [2.0, 3.0], [0.0, 0.0], [3.0, 4.0], maxiter=30,
                                gtol=1e-9, xtol=1e-12)
    assert np.allclose(best.x, [3.0, 1.0], atol=1e-8)


def test_failed_evaluations_are_rejected_steps():
    """The minimum (5, 1) sits in a region where the loss fails: steps into it are
    rejected, the best point is a valid one next to it."""
    calls = []
    fail = lambda x: 4.6 < x[0] < 5.4                       # noqa: E731
    best, hist = projected_bfgs(_fun(*BOWL, fail=fail, calls=calls), [2.0, 3.0], [0.0, 0.0],
                                [6.0, 4.0], maxiter=30, gtol=1e-9, xtol=1e-6)
    assert any(fail(x) for x in calls)                      # it met the failure region
    assert best.ok and not fail(best.x)                     # and never accepted a failure
    assert best.f < BOWL[0](np.array([2.0, 3.0]))
    assert hist.message in ("no acceptable step", "step below xtol", "maxiter reached")


def test_a_failed_start_returns_at_once():
    best, hist = projected_bfgs(_fun(*BOWL, fail=lambda x: True), [2.0, 3.0], [0, 0], [6, 4])
    assert not best.ok and hist.message == "the starting point has no gradient"


def test_reproducible():
    runs = [projected_bfgs(_fun(*ROSEN), [2.5, 3.0], [0.0, 0.0], [3.0, 4.0], maxiter=15)[0]
            for _ in range(2)]
    assert runs[0].x.tobytes() == runs[1].x.tobytes() and runs[0].f == runs[1].f


def test_values_above_97_are_losses_not_failures():
    """Real losses can be larger than energygeometry's failure values 97-99 (dataset mode)."""
    f = lambda x: 400 + (x[0] - 3) ** 2 + (x[1] - 2) ** 2          # noqa: E731
    g = lambda x: [2 * (x[0] - 3), 2 * (x[1] - 2)]                 # noqa: E731
    best, _ = projected_bfgs(_fun(f, g), [1.0, 1.0], [0, 0], [5, 5], gtol=1e-8, xtol=1e-12)
    assert np.allclose(best.x, [3, 2], atol=1e-6)


def test_the_result_is_the_lowest_real_loss():
    """As BO: the lowest loss evaluated, also one without a gradient (an SCC that did not
    converge), never a failed step (marked by the loss, whatever its value)."""
    ev = [Evaluation(np.array([1.0, 1.0]), 446.2, np.ones(2)),
          Evaluation(np.array([2.0, 1.0]), 98.0, None, failed=True),
          Evaluation(np.array([3.0, 1.0]), 211.3, None),             # no gradient
          Evaluation(np.array([4.0, 1.0]), 216.9, np.ones(2))]
    assert GO.lowest(ev) is ev[2]
    assert GO.lowest(ev[1:2]) is None and GO.lowest(ev[1:2], ev[0]) is ev[0]
    # the mark decides, not the value
    assert Evaluation(np.array([2.0, 1.0]), 98.0, None).valid
    assert not Evaluation(np.array([2.0, 1.0]), 1.0, None, failed=True).valid


class StubLoss:
    """Loss_EnergyGeometry of a smooth bowl, with the gradient set as GradRequest does."""

    failed = False                  # as Loss marks a failed step: none here

    def __init__(self, tmp_path):
        self.grad_request = None
        self.calls = []
        self.dft_input = []
        self.path = tmp_path

    def _bo_dimensions_and_cost(self, cov_radius):
        return None, [2.1, 3.4], [7.0 * cov_radius, 9.0 * cov_radius], None

    def Loss_EnergyGeometry(self, x):
        from utils.loss_grad import GradResult
        self.calls.append(list(x))
        mu = 1.9 + 0.8 * (x[0] - 3.1) ** 2 + 0.004 * (x[1] - 9.98) ** 2
        if self.grad_request is not None:
            self.grad_request.result = GradResult(
                x[0], x[1], mu, {"r0_w": 1.6 * (x[0] - 3.1), "r0_d": 0.008 * (x[1] - 9.98)},
                case="smooth")
        return mu


def test_optimize_gradient_on_a_stand_in(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(GO, "seed", lambda loss, r: [2.749, 12.036])
    loss = StubLoss(tmp_path)
    res = GO.optimize_gradient(loss, 2.4944, maxiter=20)
    assert abs(res.x[0] - 3.1) < 1e-3 and abs(res.x[1] - 9.98) < 0.05   # r0_d: flat
    assert res.fun == min(res.func_vals) and len(res.x_iters) == len(res.func_vals)
    assert res.best_dir == f"opt_{res.x[0]:.6f}_{res.x[1]:.6f}"
    lines = (tmp_path / "grad.out").read_text().splitlines()
    assert lines[0].startswith("#") and len(lines) == len(loss.calls) + 1
    assert loss.grad_request is None                        # the hook is removed again


def test_optimize_bo_grad_keeps_the_better_point(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loss = StubLoss(tmp_path)
    bo = SimpleNamespace(x=[3.3, 10.5], fun=1.9 + 0.8 * 0.04 + 0.004 * 0.27,
                         x_iters=[[3.3, 10.5], [4.0, 12.0]], func_vals=[1.93, 2.6])
    loss.Bayesian_optimization = lambda r, saver, n: bo
    res = GO.optimize_bo_grad(loss, 2.4944, None, 2, maxiter=5)
    assert res.fun < bo.fun and res.optimizer == "bo_grad"
    assert res.x_iters[:2] == bo.x_iters and len(res.x_iters) == 2 + len(loss.calls)


class DatasetStub(StubLoss):
    """Losses near 400, as in the dataset mode: above energygeometry's failure values."""

    @staticmethod
    def mu(x):
        return 400.0 + 80.0 * (x[0] - 3.1) ** 2 + 0.4 * (x[1] - 9.98) ** 2

    def Loss_EnergyGeometry(self, x):
        from utils.loss_grad import GradResult
        self.calls.append(list(x))
        if self.grad_request is not None:
            self.grad_request.result = GradResult(
                x[0], x[1], self.mu(x), {"r0_w": 160.0 * (x[0] - 3.1), "r0_d": 0.8 * (x[1] - 9.98)},
                case="smooth")
        return self.mu(x)


def test_bo_grad_starts_from_a_real_loss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loss = DatasetStub(tmp_path)
    f_bo = DatasetStub.mu([3.3, 10.5])
    bo = SimpleNamespace(x=[4.0, 12.0], fun=98.0, x_iters=[[3.3, 10.5], [4.0, 12.0]],
                         func_vals=[f_bo, 98.0], failed=[False, True])  # BO's minimum failed
    loss.Bayesian_optimization = lambda r, saver, n: bo
    res = GO.optimize_bo_grad(loss, 2.4944, None, 2, maxiter=5)
    assert loss.calls[0] == [3.3, 10.5]
    assert 400.0 < res.fun < f_bo


class FailingStub(DatasetStub):
    """Fails beyond r0_w = 3.2 Bohr as Loss.EnergyGeometry does: marked, with the gradient
    hook told, at a failure value above the real losses (the dataset mode's worst so far)."""

    def Loss_EnergyGeometry(self, x):
        self.failed = x[0] > 3.2
        if not self.failed:
            return super().Loss_EnergyGeometry(x)
        self.calls.append(list(x))
        if self.grad_request is not None:
            self.grad_request.fail([x[0], x[1], 2.0], 7000.0, "RuntimeError: no SKF")
        return 7000.0


def test_the_evaluator_takes_failures_from_the_loss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ev = GO.Evaluator(FailingStub(tmp_path))
    bad, good = ev([3.5, 10.0]), ev([3.1, 10.0])
    assert bad.failed and not bad.valid and not bad.ok and bad.f == 7000.0
    assert not good.failed and good.valid and good.ok
    assert GO.lowest(ev.history.evaluations) is good
    assert "the loss failed: RuntimeError: no SKF" in (tmp_path / "grad.out").read_text()


def test_requirements():
    with pytest.raises(SystemExit, match="--engine pydftb"):
        GO.check_requirements("dftbplus", "energygeometry")
    with pytest.raises(SystemExit, match="bandstructure"):
        GO.check_requirements("pydftb", "bandstructure")
    with pytest.raises(SystemExit, match="--pydftb_backend numpy"):
        GO.check_requirements("pydftb", "dataset", "torch")
    pytest.importorskip("torch")
    GO.check_requirements("pydftb", "dataset")


def test_command_line_options(tmp_path, monkeypatch):
    from unittest.mock import patch

    from cli.arg_parser import build_default_arg_parser
    monkeypatch.setattr(sys, "argv", ["run.py", "--optimizer", "gradient", "--grad_maxiter", "7",
                                      "--hotcent_fd_step", "0.02", "0.1"])
    args = build_default_arg_parser()
    assert args.optimizer == "gradient" and args.grad_maxiter == 7
    assert args.hotcent_fd_step == [0.02, 0.1]
    # ptbp optimize forwards them to run.py
    dataset = tmp_path / "data.xyz"
    dataset.write_text("0\n\n")
    monkeypatch.chdir(tmp_path)
    captured = {}
    with patch('cli.cmd_optimize.runpy.run_path',
               side_effect=lambda path, run_name=None: captured.setdefault('argv', list(sys.argv))):
        from cli.main import dispatch
        assert dispatch(['optimize', str(dataset), '--output', str(tmp_path / "run"),
                         '--mode', 'energygeometry', '--engine', 'pydftb',
                         '--optimizer', 'bo_grad', '--grad-maxiter', '3',
                         '--hotcent-fd-step', '0.02']) == 0
    argv = captured['argv']
    assert argv[argv.index('--optimizer') + 1] == 'bo_grad'
    assert argv[argv.index('--grad_maxiter') + 1] == '3'
    assert argv[argv.index('--hotcent_fd_step') + 1] == '0.02'


def test_a_step_clipped_by_a_bound_is_not_evaluated_twice():
    """Backtracking halves the step projected on the box: no point twice (M4c benchmark:
    a BFGS step beyond r0_w = 2.1 used to be retried at the same clipped point)."""
    calls = []
    f = lambda x: (x[0] - 2.5) ** 2 * 10 + (x[1] - 3.0) ** 2 + 1      # noqa: E731
    g = lambda x: [20 * (x[0] - 2.5), 2 * (x[1] - 3.0)]               # noqa: E731
    fail = lambda x: x[0] < 2.2                                       # noqa: E731
    projected_bfgs(_fun(f, g, fail=fail, calls=calls), [3.5, 3.5], [2.1, 0.0], [6.0, 6.0],
                   maxiter=10, init_step=5.0)
    pts = [tuple(np.round(x, 12)) for x in calls]
    assert len(pts) == len(set(pts))
