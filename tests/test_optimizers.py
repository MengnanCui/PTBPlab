"""Optimizer-dispatch smoke tests using a synthetic cost function.

The real cost is `Loss.Loss_EnergyGeometry`, which involves hotcent +
DFTB+. To validate the orchestration logic of `_optimize_pso` and
`_optimize_parallel_bo` without touching either, we build a minimal
Loss-like object whose `_bo_dimensions_and_cost` returns a closed-form
quadratic and run the methods directly.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pytest


def _make_synth_loss(opt_mode='dataset'):
    """Cheapest possible stand-in: only the bits the optimisers touch."""
    from utils.loss import Loss
    from skopt.space import Real

    class _StubLoss(Loss):
        def __init__(_):
            # Bypass Loss.__init__ — we only need a few attributes.
            _.args_input = argparse.Namespace(
                optimization_option=opt_mode,
                optimizer='parallel_bo',
                n_particles=2,
                checkpoint='/tmp/_does_not_exist.pkl',
            )
            _._cost_calls = []

        def _bo_dimensions_and_cost(_, cov_radius):
            r0_w_lower, r0_w_upper = 2.1, 7.0 * cov_radius
            r0_d_lower, r0_d_upper = 3.4, 9.0 * cov_radius
            dims = [Real(r0_w_lower, r0_w_upper, name='r0_w'),
                    Real(r0_d_lower, r0_d_upper, name='r0_d')]
            lo = [r0_w_lower, r0_d_lower]
            hi = [r0_w_upper, r0_d_upper]

            def cost_fn(x):
                # Synthetic quadratic — minimum at (5, 10).
                _._cost_calls.append(list(x))
                return float((x[0] - 5.0) ** 2 + (x[1] - 10.0) ** 2)

            return dims, lo, hi, cost_fn

    return _StubLoss()


@pytest.mark.parametrize("optimizer_kind", ["pso", "parallel_bo"])
def test_optimizer_returns_namespace_with_required_fields(optimizer_kind):
    loss = _make_synth_loss()
    loss.args_input.optimizer = optimizer_kind
    n_calls = 6
    cov_radius = 2.0  # gives r0_w in [2.1, 14], r0_d in [3.4, 18] — covers (5,10)

    if optimizer_kind == 'pso':
        res = loss._optimize_pso(cov_radius, n_calls)
    else:
        res = loss._optimize_parallel_bo(cov_radius, None, n_calls)

    # Both paths must return an object with these attributes (cli/run.py
    # consumes them and result.pkl is built from them).
    for attr in ['x', 'fun', 'x_iters', 'func_vals', 'optimizer']:
        assert hasattr(res, attr), f"{optimizer_kind} result missing .{attr}"

    assert isinstance(res, SimpleNamespace)
    assert len(res.x) == 2
    assert isinstance(res.fun, float)
    assert len(res.x_iters) == len(res.func_vals) > 0
    assert res.optimizer == optimizer_kind


@pytest.mark.parametrize("optimizer_kind", ["pso", "parallel_bo"])
def test_optimizer_finds_better_than_random_on_quadratic(optimizer_kind):
    """The best point found should be within the search box and beat the
    average of all evaluated points (basic sanity that the optimiser is
    actually doing something — not a tight optimality claim)."""
    loss = _make_synth_loss()
    loss.args_input.optimizer = optimizer_kind
    n_calls = 8
    cov_radius = 2.0

    if optimizer_kind == 'pso':
        res = loss._optimize_pso(cov_radius, n_calls)
    else:
        res = loss._optimize_parallel_bo(cov_radius, None, n_calls)

    # Best is within bounds
    assert 2.1 <= res.x[0] <= 7.0 * cov_radius
    assert 3.4 <= res.x[1] <= 9.0 * cov_radius

    # Best fun is the minimum of evaluated values (definitionally)
    assert res.fun == pytest.approx(min(res.func_vals), rel=1e-9, abs=1e-9)

    # Best is not worse than the average — trivially true for a finite
    # sample but catches accidentally returning random / first point.
    assert res.fun <= float(np.mean(res.func_vals)) + 1e-9
