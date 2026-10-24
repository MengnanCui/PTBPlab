"""skopt diagnostic plots written at the end of a sequential-BO run.

Used by both optimisation branches of cli/run.py. Kept out of cli/run.py
(a script with import-time side effects) and utils/plot.py (pulls in
hotcent via utils.calculator) so it can be imported on its own.
"""
import matplotlib.pyplot as plt
import numpy as np
from skopt.plots import plot_convergence, plot_evaluations, plot_objective


def _figure_of(ax):
    """Return the Figure behind what a skopt plot helper returned.

    plot_evaluations / plot_objective return a 2-D array of Axes in
    skopt 0.9 and a single Axes in skopt 0.10.
    """
    return np.asarray(ax, dtype=object).flat[0].figure


def save_bo_plots(res):
    """Write convergence.png, eval.png and obj.png for a skopt result to cwd."""
    plot_convergence(res)
    plt.savefig('convergence.png', dpi=300)
    plt.clf()

    eval = plot_evaluations(res, bins=20)
    _figure_of(eval).savefig('eval.png', dpi=300, bbox_inches='tight')

    obje = plot_objective(res, sample_source='result')
    _figure_of(obje).savefig('obj.png', dpi=300, bbox_inches='tight')
