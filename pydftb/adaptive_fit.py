"""Fitting environment-adaptive parameters to reference energies (milestone M8a).

The coefficients of :class:`pydftb.adaptive.AdaptiveParams` enter the free energy smoothly and
its derivatives with respect to every coefficient come with each calculation
(``DFTBResult.adaptive``, exact for the Mermin free energy), so a least-squares fit has its
Jacobian at the cost of the energies.  The residuals are relative energies per atom: the
model's F/N minus the reference, minus their weighted mean (the free reference of isolated
atoms drops out), plus a ridge term sqrt(l2) (theta - theta_0).

    samples = [Sample(atoms, e_ref, kpts=(8, 8, 8)), ...]     # eV per atom, any common zero
    fit = fit_adaptive([model], samples, sk, options={"temperature": 300.0}, processes=8)
    fit.models, fit.rmse, fit.history

Calculations of one evaluation run in forked worker processes (one BLAS thread each).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from .adaptive import make_models
from .constants import BOHR, HARTREE

_WORK = {}                                 # fork-inherited: the samples, sk and options


@dataclass
class Sample:
    """A structure and its reference energy per atom (eV); ``kpts`` a Monkhorst-Pack grid (None:
    Gamma or a molecule); ``weight`` of its residual."""
    atoms: object
    energy: float
    kpts: tuple | None = None
    weight: float = 1.0
    name: str = ""


@dataclass
class FitResult:
    models: tuple
    x: np.ndarray
    rmse: float                            # eV/atom, weighted, relative energies
    residuals: np.ndarray                  # eV/atom per sample (without the ridge)
    offset: float                          # eV/atom: mean of model - reference
    history: list = field(default_factory=list)
    n_evaluations: int = 0
    seconds: float = 0.0


def _split(models, x):
    out, k = [], 0
    for m in models:
        n = len(m.vector())
        out.append(m.with_vector(x[k:k + n]))
        k += n
    return tuple(out)


def evaluate_sample(models, sample: Sample, sk, options: dict):
    """(F/N in eV, dF/N/dx in eV) of one sample with these models."""
    from .api import make_kpoints
    from .calculator import DFTB
    at = sample.atoms
    periodic = bool(np.any(at.pbc))
    opts = dict(options)
    if periodic and sample.kpts is not None:
        opts["kpts"] = make_kpoints(tuple(sample.kpts))
    calc = DFTB(sk, adaptive=list(models), **opts)
    r = calc.calculate(at.get_chemical_symbols(), at.positions / BOHR,
                       at.cell.array / BOHR if periodic else None,
                       np.ones(3, bool) if periodic else None, forces=False, stress=False)
    if not r.converged:
        raise RuntimeError(f"SCC not converged for {sample.name or 'a sample'}")
    n = len(at)
    return r.free_energy * HARTREE / n, r.adaptive.gradient(models) * HARTREE / n


def _work(args):
    from threadpoolctl import threadpool_limits
    k, x = args
    w = _WORK
    with threadpool_limits(1, user_api="blas"):
        return evaluate_sample(_split(w["models"], x), w["samples"][k], w["sk"], w["options"])


def evaluate(models, samples, sk, options, x, processes: int = 1):
    """F/N (n,) and its Jacobian (n, n_x) for every sample, in eV."""
    _WORK.update(models=tuple(models), samples=list(samples), sk=sk, options=dict(options))
    jobs = [(k, np.asarray(x, float)) for k in range(len(samples))]
    if processes > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(processes) as pool:
            out = pool.map(_work, jobs, chunksize=1)
    else:
        out = [_work(j) for j in jobs]
    e = np.array([o[0] for o in out])
    jac = np.array([o[1] for o in out])
    return e, jac


def fit_adaptive(models, samples, sk, options: dict | None = None, l2: float = 1e-4,
                 processes: int | None = None, max_evaluations: int = 200, x0=None,
                 verbose: bool = False) -> FitResult:
    """Least-squares fit of the coefficients of ``models`` to the relative energies of
    ``samples`` (scipy's trust-region reflective method with the analytic Jacobian).  ``l2``
    (eV/atom per unit coefficient, squared) pulls the coefficients towards their start."""
    from scipy.optimize import least_squares
    models = make_models(models)
    options = dict(options or {})
    if processes is None:
        processes = max(1, min(len(samples), (os.cpu_count() or 1)))
    x_start = np.concatenate([m.vector() for m in models]) if x0 is None else np.asarray(x0)
    w = np.sqrt(np.array([s.weight for s in samples], float))
    wn = w ** 2 / (w ** 2).sum()
    t_ref = np.array([s.energy for s in samples], float)
    cache, history = {}, []
    t0 = time.perf_counter()

    def compute(x):
        key = x.tobytes()
        if key not in cache:
            e, jac = evaluate(models, samples, sk, options, x, processes)
            d = e - t_ref
            off = wn @ d
            res = w * (d - off)
            jres = w[:, None] * (jac - wn @ jac)
            reg = np.sqrt(l2) * (x - x_start)
            cache.clear()
            cache[key] = (np.concatenate([res, reg]),
                          np.vstack([jres, np.sqrt(l2) * np.eye(len(x))]), d - off, off)
            rmse = float(np.sqrt(wn @ (d - off) ** 2))
            history.append((time.perf_counter() - t0, rmse))
            if verbose:
                print(f"  eval {len(history):3d}  rmse {rmse * 1000:8.2f} meV/atom", flush=True)
        return cache[key]

    sol = least_squares(lambda x: compute(x)[0], x_start, jac=lambda x: compute(x)[1],
                        method="trf", max_nfev=max_evaluations, x_scale="jac")
    _, _, resid, off = compute(sol.x)
    fitted = _split(models, sol.x)
    return FitResult(fitted, sol.x, float(np.sqrt(wn @ resid ** 2)), resid, float(off),
                     history, len(history), time.perf_counter() - t0)


def predict(models, samples, sk, options=None, offset: float = 0.0, processes: int | None = None):
    """Model energies per atom (eV) minus ``offset``, for samples not in the fit."""
    models = make_models(models)
    if processes is None:
        processes = max(1, min(len(samples), (os.cpu_count() or 1)))
    x = np.concatenate([m.vector() for m in models])
    e, _ = evaluate(models, samples, sk, dict(options or {}), x, processes)
    return e - offset
