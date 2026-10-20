"""Backward-compatible shim: the DFTB+ runner moved to pydftb.validation.dftbplus.

Kept so the benchmark scripts in this directory keep working; new code should
use :func:`pydftb.validation.dftbplus.run` (explicit ``filling=`` argument).
"""
from pydftb.validation.dftbplus import write_gen  # noqa: F401
from pydftb.validation import dftbplus as _d

DFTBPLUS = _d.DFTBPLUS


def run(symbols, pos_ang, cell_ang=None, skdir=".", max_l=None, scc=True, kpts=None,
        temperature=0.0, shell_resolved=False, threads=1, scc_tol=1e-10, workdir=None,
        extra_ham="", timing=False):
    res = _d.run(symbols, pos_ang, cell_ang, skdir=skdir, max_l=max_l, kpts=kpts, scc=scc,
                 filling=("fermi", temperature), shell_resolved=shell_resolved, scc_tol=scc_tol,
                 extra_ham=extra_ham, threads=threads, timing=timing, workdir=workdir)
    return res
