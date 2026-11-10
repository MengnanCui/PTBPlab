"""BO post-processing plots (utils/bo_plots.py, called from cli/run.py).

skopt 0.9's plot_evaluations / plot_objective return a 2-D array of Axes,
skopt 0.10's a single Axes. eval.png / obj.png must be saved from either,
otherwise a BO run stops before the best-folder copy and results.tgz.
"""
from __future__ import annotations

import matplotlib
import pytest

matplotlib.use('Agg')


@pytest.mark.parametrize("nrows, ncols", [(1, 1), (2, 2)])
def test_figure_of_single_axes_and_axes_array(nrows, ncols):
    """(1, 1) gives a single Axes like skopt 0.10, (2, 2) a 2-D array like skopt 0.9."""
    import matplotlib.pyplot as plt
    from utils.bo_plots import _figure_of

    fig, ax = plt.subplots(nrows, ncols)
    assert _figure_of(ax) is fig
    plt.close(fig)


def test_save_bo_plots_on_small_gp_minimize_result(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from skopt import gp_minimize
    from skopt.space import Real
    from utils.bo_plots import save_bo_plots

    # Same call shape as Loss.Bayesian_optimization: (r0_w, r0_d) dimensions,
    # an x0 seed and the CLI default n_calls=11, with a cheap quadratic in
    # place of the DFTB+ loss.
    dims = [Real(name='r0_w', low=2.1, high=14.0, prior='uniform'),
            Real(name='r0_d', low=3.4, high=18.0, prior='uniform')]
    res = gp_minimize(func=lambda x: (x[0] - 5.0) ** 2 + (x[1] - 10.0) ** 2,
                      dimensions=dims,
                      n_calls=11,
                      acq_func="EI",
                      x0=[6.0, 10.0],
                      random_state=123)

    monkeypatch.chdir(tmp_path)
    save_bo_plots(res)
    plt.close('all')

    for name in ('convergence.png', 'eval.png', 'obj.png'):
        path = tmp_path / name
        assert path.is_file() and path.stat().st_size > 0, f"{name} not written"
