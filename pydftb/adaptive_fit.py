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

Forces (M8b): samples with ``forces`` (eV/A) add residuals sqrt(force_weight) w (F - F_ref)
per component.  dF/dtheta is analytic, but d(forces)/dtheta is a second derivative of the
energy: its Jacobian block comes from central differences in theta (2 n_theta calculations per
sample with forces, in the same pool), which agree with -grad_R (dF/dtheta) by the symmetry of
second derivatives (tests/test_adaptive.py).
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
    forces: np.ndarray | None = None       # reference forces (eV/A), (n_atom, 3) (M8b)


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
    force_rmse: float | None = None        # eV/A over the force components (M8b)


def _split(models, x):
    out, k = [], 0
    for m in models:
        n = len(m.vector())
        out.append(m.with_vector(x[k:k + n]))
        k += n
    return tuple(out)


def evaluate_sample(models, sample: Sample, sk, options: dict, forces: bool = False):
    """(F/N in eV, dF/N/dx in eV, forces in eV/A or None) of one sample with these models."""
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
                       np.ones(3, bool) if periodic else None, forces=forces, stress=False)
    if not r.converged:
        raise RuntimeError(f"SCC not converged for {sample.name or 'a sample'}")
    n = len(at)
    f = r.forces * (HARTREE / BOHR) if forces else None
    return r.free_energy * HARTREE / n, r.adaptive.gradient(models) * HARTREE / n, f


def _work(args):
    from threadpoolctl import threadpool_limits
    k, x, forces = args
    w = _WORK
    with threadpool_limits(1, user_api="blas"):
        return evaluate_sample(_split(w["models"], x), w["samples"][k], w["sk"], w["options"],
                               forces)


def _run(jobs, processes):
    if processes > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(processes) as pool:
            return pool.map(_work, jobs, chunksize=1)
    return [_work(j) for j in jobs]


def evaluate(models, samples, sk, options, x, processes: int = 1):
    """F/N (n,) and its Jacobian (n, n_x) for every sample, in eV."""
    e, jac, _, _ = evaluate_all(models, samples, sk, options, x, processes, with_forces=False)
    return e, jac


def evaluate_all(models, samples, sk, options, x, processes: int = 1, with_forces: bool = True,
                 step: float = 1e-4):
    """F/N and its Jacobian, and for the samples with reference forces the model forces
    {k: (n_atom, 3)} and their Jacobian {k: (n_atom * 3, n_x)} (central differences, ``step``
    in every coefficient), in eV and eV/A."""
    _WORK.update(models=tuple(models), samples=list(samples), sk=sk, options=dict(options))
    x = np.asarray(x, float)
    with_f = [k for k, s in enumerate(samples) if with_forces and s.forces is not None]
    jobs = [(k, x, k in with_f) for k in range(len(samples))]
    for k in with_f:
        for i in range(len(x)):
            for sgn in (1.0, -1.0):
                xi = x.copy()
                xi[i] += sgn * step
                jobs.append((k, xi, True))
    out = _run(jobs, processes)
    n = len(samples)
    e = np.array([o[0] for o in out[:n]])
    jac = np.array([o[1] for o in out[:n]])
    forces = {k: out[k][2] for k in with_f}
    fjac, pos = {}, n
    for k in with_f:
        cols = []
        for _ in range(len(x)):
            cols.append((out[pos][2] - out[pos + 1][2]).ravel() / (2 * step))
            pos += 2
        fjac[k] = np.array(cols).T
    return e, jac, forces, fjac


def fit_adaptive(models, samples, sk, options: dict | None = None, l2: float = 1e-4,
                 processes: int | None = None, max_evaluations: int = 200, x0=None,
                 verbose: bool = False, force_weight: float = 0.0,
                 force_step: float = 1e-4) -> FitResult:
    """Least-squares fit of the coefficients of ``models`` to the relative energies of
    ``samples`` (scipy's trust-region reflective method with the analytic Jacobian).  ``l2``
    (eV/atom per unit coefficient, squared) pulls the coefficients towards their start.
    ``force_weight`` ((eV/atom)^2 per (eV/A)^2): the weight of the force residuals of the
    samples that have reference forces (M8b; their Jacobian by central differences of
    ``force_step`` in every coefficient)."""
    from scipy.optimize import least_squares
    models = make_models(models)
    options = dict(options or {})
    if processes is None:
        processes = max(1, min(len(samples), (os.cpu_count() or 1)))
    x_start = np.concatenate([m.vector() for m in models]) if x0 is None else np.asarray(x0)
    w = np.sqrt(np.array([s.weight for s in samples], float))
    wn = w ** 2 / (w ** 2).sum()
    t_ref = np.array([s.energy for s in samples], float)
    use_f = force_weight > 0.0 and any(s.forces is not None for s in samples)
    cache, history = {}, []
    t0 = time.perf_counter()

    def compute(x):
        key = x.tobytes()
        if key not in cache:
            e, jac, forces, fjac = evaluate_all(models, samples, sk, options, x, processes,
                                                with_forces=use_f, step=force_step)
            d = e - t_ref
            off = wn @ d
            res = [w * (d - off)]
            jres = [w[:, None] * (jac - wn @ jac)]
            f_err = []
            for k in sorted(forces):
                df = (forces[k] - np.asarray(samples[k].forces, float)).ravel()
                f_err.append(df)
                res.append(np.sqrt(force_weight) * w[k] * df)
                jres.append(np.sqrt(force_weight) * w[k] * fjac[k])
            res.append(np.sqrt(l2) * (x - x_start))
            jres.append(np.sqrt(l2) * np.eye(len(x)))
            f_rmse = float(np.sqrt(np.mean(np.concatenate(f_err) ** 2))) if f_err else None
            cache.clear()
            cache[key] = (np.concatenate(res), np.vstack(jres), d - off, off, f_rmse)
            rmse = float(np.sqrt(wn @ (d - off) ** 2))
            history.append((time.perf_counter() - t0, rmse, f_rmse))
            if verbose:
                fr = "" if f_rmse is None else f"  forces {f_rmse * 1000:8.2f} meV/A"
                print(f"  eval {len(history):3d}  rmse {rmse * 1000:8.2f} meV/atom{fr}",
                      flush=True)
        return cache[key]

    sol = least_squares(lambda x: compute(x)[0], x_start, jac=lambda x: compute(x)[1],
                        method="trf", max_nfev=max_evaluations, x_scale="jac")
    _, _, resid, off, f_rmse = compute(sol.x)
    fitted = _split(models, sol.x)
    return FitResult(fitted, sol.x, float(np.sqrt(wn @ resid ** 2)), resid, float(off),
                     history, len(history), time.perf_counter() - t0, f_rmse)


def predict(models, samples, sk, options=None, offset: float = 0.0, processes: int | None = None):
    """Model energies per atom (eV) minus ``offset``, for samples not in the fit."""
    models = make_models(models)
    if processes is None:
        processes = max(1, min(len(samples), (os.cpu_count() or 1)))
    x = np.concatenate([m.vector() for m in models])
    e, _ = evaluate(models, samples, sk, dict(options or {}), x, processes)
    return e - offset
