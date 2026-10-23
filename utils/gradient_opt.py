"""Gradient-based optimisers of the confinement radii (milestone M4c).

``--optimizer gradient``: projected BFGS with Armijo backtracking in the box of the BO
bounds, from BO's seed.  ``--optimizer bo_grad``: BO with ``--n_calls``, then at most
``--grad_maxiter`` (default 5) gradient steps from its best point.  Every evaluation is
the loss PTBP computes (written to par.out/log.out as for BO) plus its gradient
(:mod:`utils.loss_grad`), recorded in ``grad.out``.  An evaluation without a gradient
(a failed step, which the loss marks as such, or no derivative there) counts as a
rejected step.  The result is the lowest loss evaluated, as for BO, failed steps excluded.

The results have the attributes cli/run.py uses (``x``, ``fun``, ``x_iters``,
``func_vals``), plus ``best_dir`` (the directory of the best evaluation) and ``grads``.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

GRAD_FILE = "grad.out"


@dataclass
class Evaluation:
    x: np.ndarray
    f: float
    g: np.ndarray | None          # None: no gradient (failure)
    result: object = None         # utils.loss_grad.GradResult
    directory: str = ""
    failed: bool = False          # a failed step (Loss.failed): f is its failure value

    @property
    def ok(self) -> bool:
        """A value with a gradient (a failed step has none)."""
        return self.g is not None and np.isfinite(self.f) and bool(np.all(np.isfinite(self.g)))

    @property
    def valid(self) -> bool:
        """A loss value, with or without a gradient (not a failed step)."""
        return bool(np.isfinite(self.f)) and not self.failed


@dataclass
class History:
    evaluations: list = field(default_factory=list)
    iterations: list = field(default_factory=list)      # (iteration, x, f, |pg|, step)
    message: str = ""


def check_requirements(engine: str, mode: str, backend: str = "numpy") -> None:
    """The gradient needs pydftb (autodiff) with torch, its NumPy engine (on which the
    autodiff was validated, M4b) and the r0 loss modes."""
    if engine != "pydftb":
        raise SystemExit("--optimizer gradient/bo_grad needs --engine pydftb (the gradient comes "
                         "from pydftb.autodiff)")
    if backend != "numpy":
        raise SystemExit("--optimizer gradient/bo_grad needs --pydftb_backend numpy (the "
                         "gradient is validated with the NumPy engine)")
    if mode not in ("energygeometry", "dataset", "reaction"):
        raise SystemExit(f"--optimizer gradient/bo_grad: no gradient for the {mode} mode")
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("--optimizer gradient/bo_grad needs torch (pip install ptbp[torch])")


class Evaluator:
    """Loss value and gradient at x = (r0_w, r0_d), written to ``grad.out``."""

    def __init__(self, loss, steps=None, log_path: str | None = None):
        self.loss = loss
        self.steps = steps
        self.log_path = log_path or os.path.abspath(GRAD_FILE)
        self.history = History()
        if not os.path.exists(self.log_path):
            from utils.loss_grad import GRAD_HEADER
            with open(self.log_path, "w") as fh:
                fh.write(GRAD_HEADER + "\n")

    def __call__(self, x) -> Evaluation:
        from utils.loss_grad import GradRequest
        x = np.asarray(x, dtype=float)
        cwd = os.getcwd()
        req = GradRequest(self.steps)
        self.loss.grad_request = req
        try:
            f = float(self.loss.Loss_EnergyGeometry([float(x[0]), float(x[1])]))
        finally:
            self.loss.grad_request = None
            os.chdir(cwd)
        res = req.result
        g = None
        if res is not None and res.grad is not None:
            g = np.array([res.grad["r0_w"], res.grad["r0_d"]], dtype=float)
        ev = Evaluation(x, f, g, res, req.directory_name(*x), failed=bool(self.loss.failed))
        self.history.evaluations.append(ev)
        if res is not None:
            with open(self.log_path, "a") as fh:
                fh.write(res.line() + "\n")
        return ev


def _projected(g, x, lo, hi):
    """The gradient with the components that push against an active bound removed."""
    pg = g.copy()
    pg[(x <= lo) & (g > 0)] = 0.0
    pg[(x >= hi) & (g < 0)] = 0.0
    return pg


def projected_bfgs(fun, x0, lo, hi, maxiter: int = 20, gtol: float = 1e-3, xtol: float = 1e-3,
                   c1: float = 1e-4, init_step: float = 0.2, max_backtrack: int = 6,
                   history: History | None = None):
    """Minimise ``fun`` (x -> Evaluation) in the box [lo, hi].

    BFGS on the inverse Hessian, directions projected on the box (components against an
    active bound dropped), Armijo backtracking (halving) on the projected path; a failed
    evaluation is a rejected step.  The first step has length ``init_step`` (Bohr).  Stops
    when the projected gradient is below ``gtol`` |f| per Bohr, when the step is shorter
    than ``xtol``, when no step is accepted, or after ``maxiter`` iterations.
    Returns (best Evaluation, History)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    hist = history or History()
    cur = fun(np.clip(np.asarray(x0, float), lo, hi))
    if not cur.ok:
        hist.message = "the starting point has no gradient"
        return cur, hist
    best = cur
    n = len(cur.x)
    h_inv = np.eye(n) * (init_step / max(np.linalg.norm(cur.g), 1e-300))
    for it in range(maxiter):
        pg = _projected(cur.g, cur.x, lo, hi)
        hist.iterations.append((it, cur.x.copy(), cur.f, float(np.max(np.abs(pg))), None))
        if np.max(np.abs(pg)) <= gtol * max(abs(cur.f), 1.0):
            hist.message = "projected gradient below gtol"
            break
        free = pg != 0
        d = np.zeros(n)
        d[free] = -(h_inv[np.ix_(free, free)] @ cur.g[free])
        p = np.clip(cur.x + d, lo, hi) - cur.x             # the step, projected on the box
        if cur.g @ p >= 0:                                 # not a descent step: reset
            h_inv = np.eye(n) * (init_step / max(np.linalg.norm(pg), 1e-300))
            p = np.clip(cur.x - h_inv @ pg, lo, hi) - cur.x
        alpha, nxt = 1.0, None
        for _ in range(max_backtrack + 1):                 # halve the projected step: inside
            s = alpha * p                                  # the box, every trial a new point
            if np.max(np.abs(s)) < xtol:
                break
            trial = fun(cur.x + s)
            if trial.ok and trial.f <= cur.f + c1 * float(cur.g @ s):
                nxt = trial
                break
            alpha *= 0.5
        if nxt is None:
            hist.message = ("step below xtol" if np.max(np.abs(alpha * p)) < xtol * 2
                            else "no acceptable step")
            break
        s, y = nxt.x - cur.x, nxt.g - cur.g
        sy = float(s @ y)
        if sy > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
            rho = 1.0 / sy
            v = np.eye(n) - rho * np.outer(s, y)
            h_inv = v @ h_inv @ v.T + rho * np.outer(s, s)
        hist.iterations[-1] = (it, cur.x.copy(), cur.f, float(np.max(np.abs(pg))),
                               float(np.max(np.abs(s))))
        cur = nxt
        if cur.f < best.f:
            best = cur
    else:
        hist.message = "maxiter reached"
    return best, hist


def lowest(evaluations, default=None):
    """The valid evaluation with the lowest loss (the first of equals), else ``default``."""
    return min((e for e in evaluations if e.valid), key=lambda e: e.f, default=default)


def _result(best, evaluations, optimizer, history, extra_x=(), extra_f=(), best_dir=None):
    return SimpleNamespace(
        x=[float(v) for v in best.x], fun=float(best.f),
        x_iters=[list(map(float, x)) for x in extra_x] + [list(map(float, e.x)) for e in evaluations],
        func_vals=list(map(float, extra_f)) + [float(e.f) for e in evaluations],
        grads=[None if e.g is None else list(map(float, e.g)) for e in evaluations],
        best_dir=best_dir or best.directory, optimizer=optimizer, message=history.message,
        iterations=history.iterations)


def bounds(loss, cov_radius):
    """The BO box (Loss._bo_dimensions_and_cost)."""
    _, lo, hi, _ = loss._bo_dimensions_and_cost(cov_radius)
    return np.array(lo, float), np.array(hi, float)


def seed(loss, cov_radius):
    """BO's starting point: the PTBP table for a single element that has an entry, else
    r_cov x (3, 5)."""
    from ase.data import chemical_symbols

    from utils.parameters_set import get_PTBP
    symbols = sorted({chemical_symbols[int(z)] for a in loss.dft_input for z in a.numbers})
    if len(symbols) == 1 and symbols[0] in get_PTBP():
        p = get_PTBP()[symbols[0]]
        return [p['r0_w'], p['r0_d']]
    return [cov_radius * 3, cov_radius * 5]


def optimize_gradient(loss, cov_radius, maxiter: int = 20, steps=None, x0=None):
    """``--optimizer gradient``."""
    from utils.dftb_stage import torch_one_thread
    torch_one_thread()
    lo, hi = bounds(loss, cov_radius)
    x0 = seed(loss, cov_radius) if x0 is None else x0
    print(f"[gradient] projected BFGS from r0_w={x0[0]}, r0_d={x0[1]} in "
          f"[{lo[0]:.3f}, {hi[0]:.3f}] x [{lo[1]:.3f}, {hi[1]:.3f}], maxiter {maxiter}")
    ev = Evaluator(loss, steps)
    t0 = time.perf_counter()
    best, hist = projected_bfgs(ev, x0, lo, hi, maxiter=maxiter, history=ev.history)
    best = lowest(ev.history.evaluations, best)    # also one without a gradient, as BO
    print(f"[gradient] {hist.message}: best r0_w={best.x[0]:.6f}, r0_d={best.x[1]:.6f}, "
          f"mu={best.f:.10g} after {len(ev.history.evaluations)} evaluations "
          f"({time.perf_counter() - t0:.0f} s)")
    return _result(best, ev.history.evaluations, "gradient", hist)


def optimize_bo_grad(loss, cov_radius, checkpoint_saver, n_calls: int, maxiter: int = 5,
                     steps=None):
    """``--optimizer bo_grad``: BO, then gradient steps from its best point."""
    from utils.dftb_stage import torch_one_thread
    res = loss.Bayesian_optimization(cov_radius, checkpoint_saver, n_calls)
    x_bo, f_bo = list(map(float, res.x)), float(res.fun)
    # BO's minimum is a failed step (marked by the loss) only when no step before it
    # succeeded, or in the energygeometry mode when every real loss is above 97-99:
    # then start from the best real one
    failed = list(getattr(res, "failed", None) or [False] * len(res.func_vals))
    ok = [(f, x) for x, f, bad in zip(res.x_iters, res.func_vals, failed) if not bad]
    if ok and failed[int(np.argmin(res.func_vals))]:
        f_bo, x_bo = min(ok, key=lambda v: v[0])
        x_bo, f_bo = list(map(float, x_bo)), float(f_bo)
    print(f"[bo_grad] BO best r0_w={x_bo[0]:.6f}, r0_d={x_bo[1]:.6f}, mu={f_bo:.10g}; "
          f"{maxiter} gradient step(s) from there")
    torch_one_thread()
    lo, hi = bounds(loss, cov_radius)
    ev = Evaluator(loss, steps)
    _, hist = projected_bfgs(ev, x_bo, lo, hi, maxiter=maxiter, history=ev.history)
    best = lowest(ev.history.evaluations)
    bo_dir = f"opt_{x_bo[0]:.3f}_{x_bo[1]:.3f}"
    if best is None or best.f >= f_bo:
        best_x = SimpleNamespace(x=np.array(x_bo), f=f_bo, directory=bo_dir)
    else:
        best_x = best
    return _result(best_x, ev.history.evaluations, "bo_grad", hist, extra_x=res.x_iters,
                   extra_f=res.func_vals, best_dir=best_x.directory)
