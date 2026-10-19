"""Figures for pydftb/RESEARCH.md (Mengnan plotting style)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = os.path.dirname(__file__)
COLORS = ["#2470a0", "#ca3e47", "#f29c2b", "#1f640a", "#2ca02c", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#005555"]
MARKERS = ['o', 'v', 's', '*', 'p', 'P', ',']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 5, 1, 5, 1, 5)), (0, (5, 5))]
plt.rcParams.update({
    "font.size": 15, "axes.linewidth": 2, "axes.labelsize": 15, "axes.titlesize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "xtick.major.width": 2, "ytick.major.width": 2,
    "xtick.minor.width": 1.5, "ytick.minor.width": 1.5, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.minor.size": 3, "ytick.minor.size": 3, "lines.linewidth": 2,
    "lines.markersize": 13, "lines.markeredgewidth": 2, "legend.fontsize": 10,
    "legend.markerscale": 1.5, "legend.framealpha": 0.5, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def line(ax, x, y, i, label, ls=0):
    ax.plot(x, y, color=COLORS[i], linestyle=LINESTYLES[ls], linewidth=2, alpha=0.7, zorder=2)
    ax.scatter(x, y, facecolors="none", edgecolors=COLORS[i], marker=MARKERS[i % 6], s=80,
               linewidths=2, zorder=3, label=label)


def style(ax, xl, yl):
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.tick_params(axis="both", which="major", length=6, width=2, direction="in", labelsize=13)
    ax.tick_params(axis="both", which="minor", length=3, width=1.5, direction="in")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(framealpha=0.5, fontsize=10, loc="best", markerscale=1.5)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
sc = json.load(open(os.path.join(HERE, "results_scaling.json")))
n = [r["natom"] for r in sc]
line(a1, n, [r["dftb1"] for r in sc], 0, "DFTB+ 1 thread")
line(a1, n, [r["dftb4"] for r in sc], 2, "DFTB+ 4 threads", ls=1)
line(a1, n, [r["py1"]["total"] for r in sc], 1, "pydftb (NumPy) 1 thread")
line(a1, n, [r["py4"]["total"] for r in sc], 5, "pydftb (NumPy) 4 threads", ls=1)
a1.set_xscale("log"); a1.set_yscale("log")
style(a1, "Atoms (water, SCC, E+F)", "Wall time (s)")
a1.set_title("(a) Reference engine vs DFTB+")

small = json.load(open(os.path.join(HERE, "results_sparse_small.json")))
large = json.load(open(os.path.join(HERE, "results_sparse_1e-05.json")))
sp = small + [r for r in large if r["natom"] > small[-1]["natom"]]
sv = json.load(open(os.path.join(HERE, "results_solver.json")))
N = [r["N"] for r in sv]
line(a2, N, [r["t_eigh_gen"] for r in sv], 0, "dense eigh (LAPACK, 4 thr)")
line(a2, N, [r["t_sp2_fp64"] + r["t_loewdin"] for r in sv], 2, "dense SP2 FP64 (GEMM, 4 thr)")
line(a2, N, [r["t_sp2_fp32+2xMcWeeny64"] + r["t_loewdin"] for r in sv], 5, "dense SP2 FP32+FP64 polish", ls=1)
line(a2, [r["N"] for r in sp], [r["t_sp2"] + r["t_inv_sqrt"] for r in sp], 3, "sparse SP2 O(N) (SciPy, 1 thr)")
xx = np.array([6000, 35000])
a2.plot(xx, sv[-1]["t_eigh_gen"] * (xx / 6000) ** 3, color=COLORS[0], linestyle=LINESTYLES[3],
        linewidth=2, alpha=0.7, label="eigh extrapolated $N^3$")
a2.set_xscale("log"); a2.set_yscale("log")
style(a2, "Basis functions N", "Time per solve (s)")
a2.set_title("(b) Density-matrix solvers")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "..", "figures_scaling.png"))
print("saved")
