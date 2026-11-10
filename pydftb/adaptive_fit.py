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

A series (A0 sections 4 and 5):

* **bands** (``band_samples``, ``band_weight``): every :class:`BandSample` (a
  :class:`pydftb.bandref.BandRef` with its SCC k-points) adds the residuals of
  :func:`pydftb.bandloss.residuals` (window, free shift, valence by index, conduction one-way
  nearest, gap term; full SCC for the values) times sqrt(band_weight x weight / (n_groups x
  n_in_group)), so their squares are band_weight x the group mean of rms^2.  Jacobian:
  Hellmann-Feynman at frozen charges (:mod:`pydftb.banddiff`, ``band_jacobian="frozen"``,
  analytic for AdaptiveParams) or central differences with full SCC (``"scc"``, 2 n_x SCC
  runs per structure).  With no ``samples`` the fit is to the bands alone (A2: electronic part
  first, no repulsive).
* **offsets**: ``"global"`` (default, as before) subtracts the weighted mean of model -
  reference; ``"none"`` fits absolute energies (sample energies on the model's zero, e.g.
  atomization energy + the model's isolated atoms); ``"per_element"`` eliminates one offset per
  element (eV/atom, by linear least squares at every evaluation, the Jacobian projected).
* **models**: any :mod:`pydftb.envmodel` models (AdaptiveParams, r0 mixing, ...); a model whose
  ``gradient_from`` gives None gets its energy Jacobian by central differences in x.
* **k-points** of samples: None, an (n1, n2, n3) Monkhorst-Pack grid (as before), ASE's dict
  ({"size": (n1, n2, n3), "gamma": True} for the Gamma-centred grids of the A1 scans, as
  :class:`pydftb.ase_calc.PyDFTB`), or (kpts, weights).
"""
from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .adaptive import AdaptiveParams, make_models
from .constants import BOHR, HARTREE
from .envmodel import ENGINE_KEYWORDS, engine_kwargs, model_gradient, split_vector, vector_of

_WORK = {}                                 # fork-inherited: the samples, sk and options
SLOW_MIXING, SLOW_MAX_SCC = 0.05, 1000      # the retry of a sample whose SCC did not converge
ENERGY_STEP = 1e-5                          # x step of the energy Jacobian of models without dF/dx
OFFSETS = ("global", "none", "per_element")


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
class BandSample:
    """A band reference (:class:`pydftb.bandref.BandRef` with its structure) for the band
    residuals: ``kpts`` the SCC k-points (as :class:`Sample`), ``weight``, ``group`` (default
    the reference's meta group or name; groups are averaged first, as
    :func:`pydftb.bandloss.band_loss`), ``options`` extra engine options for this structure."""
    ref: object
    kpts: object = None
    weight: float = 1.0
    group: str = ""
    options: dict | None = None

    @property
    def name(self) -> str:
        return self.ref.name

    @property
    def group_name(self) -> str:
        return self.group or self.ref.meta.get("group") or self.ref.name


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
    band_loss: float | None = None         # pydftb.bandloss.band_loss of the final models
    band_results: list = field(default_factory=list)    # bandloss.compare dicts (final models)
    offsets: dict | None = None            # offsets="per_element": {element: eV/atom}


def _split(models, x):
    return split_vector(models, x)


def _as_models(models) -> tuple:
    """The models as a tuple: AdaptiveParams checked as before (one per element), other
    :mod:`pydftb.envmodel` models by their engine keyword."""
    ms = tuple(models) if isinstance(models, (list, tuple)) else (models,)
    if all(isinstance(m, AdaptiveParams) for m in ms):
        return make_models(ms)
    for m in ms:
        if getattr(m, "engine_keyword", None) not in ENGINE_KEYWORDS:
            raise TypeError(f"{type(m).__name__} is not an environment model")
    return ms


def _kpoints(atoms, spec):
    """The engine's (kpts, weights) of a sample's k-point spec (see the module docstring)."""
    from .api import make_kpoints
    if spec is None:
        return None
    if isinstance(spec, dict):
        from .ase_calc import kpoints_of
        return kpoints_of(atoms, spec)
    if isinstance(spec, np.ndarray) and spec.ndim == 2:
        return make_kpoints(spec)
    if len(spec) == 3 and np.ndim(spec[0]) == 0:
        return make_kpoints(tuple(spec))
    return make_kpoints(tuple(spec))


def _blas_one():
    """One BLAS thread per worker (threadpoolctl if installed)."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:                       # optional: the env variables then decide
        return contextlib.nullcontext()
    return threadpool_limits(1, user_api="blas")


def _calculate(models, sk, at, opts: dict, forces: bool, name: str = ""):
    """(engine, result) of a converged SCC of ``at`` with these models; a failed SCC is retried
    with gentler mixing (unless the options set one), then raises RuntimeError."""
    from .calculator import DFTB
    periodic = bool(np.any(at.pbc))
    kw = engine_kwargs(models)

    def run(o):
        import warnings
        # the convergence is checked here (retry with gentler mixing, then RuntimeError), so
        # the engine returns an unconverged result instead of raising SCCNotConvergedError
        o = {"convergent_scc_only": False, **o}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # "SCC not converged ..."
            eng = DFTB(sk, **kw, **o)
            return eng, eng.calculate(
                at.get_chemical_symbols(), at.positions / BOHR,
                at.cell.array / BOHR if periodic else None,
                np.ones(3, bool) if periodic else None, forces=forces, stress=False)
    eng, r = run(opts)
    if not r.converged and "mixing" not in opts:
        # Broyden at its default mixing gives up on some distorted clusters (the VASP Cu set's
        # rattled and restructured Cu13) that converge with gentler mixing; the converged
        # result does not depend on the mixer
        eng, r = run({**opts, "mixing": SLOW_MIXING, "max_scc": max(SLOW_MAX_SCC,
                                                                     opts.get("max_scc", 0))})
    if not r.converged:
        raise RuntimeError(f"SCC not converged for {name or 'a sample'}")
    return eng, r


def evaluate_sample(models, sample: Sample, sk, options: dict, forces: bool = False):
    """(F/N in eV, dF/N/dx in eV or None, forces in eV/A or None) of one sample with these
    models (None: a model without analytic dF/dx; the caller differentiates)."""
    at = sample.atoms
    periodic = bool(np.any(at.pbc))
    opts = dict(options)
    if periodic and sample.kpts is not None:
        opts["kpts"] = _kpoints(at, sample.kpts)
    _, r = _calculate(models, sk, at, opts, forces, sample.name)
    n = len(at)
    f = r.forces * (HARTREE / BOHR) if forces else None
    g = model_gradient(r, models)
    return r.free_energy * HARTREE / n, (None if g is None else g * HARTREE / n), f


def _work(args):
    k, x, forces = args
    w = _WORK
    with _blas_one():
        return evaluate_sample(_split(w["models"], x), w["samples"][k], w["sk"], w["options"],
                               forces)


def _pool_map(func, jobs, processes):
    """``map`` over a fresh fork pool of ``processes`` workers.  A worker that dies (a segfault)
    raises ``BrokenProcessPool`` instead of hanging the fit forever as ``multiprocessing.Pool``
    does.  On macOS run with ``VECLIB_MAXIMUM_THREADS=1``: Accelerate's threads (GCD) are not
    fork-safe, and a child forked after the parent used them crashes in the first LAPACK call."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(processes, mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(func, jobs, chunksize=1))


def _work_tolerant(args):
    try:
        return _work(args)
    except (RuntimeError, np.linalg.LinAlgError):
        return None


def _run(jobs, processes):
    if processes > 1 and len(jobs) > 1:
        return _pool_map(_work, jobs, processes)
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
    if not len(samples):
        return np.zeros(0), np.zeros((0, len(x))), {}, {}
    with_f = [k for k, s in enumerate(samples) if with_forces and s.forces is not None]
    jobs = [(k, x, k in with_f) for k in range(len(samples))]
    for k in with_f:
        for i in range(len(x)):
            for sgn in (1.0, -1.0):
                xi = x.copy()
                xi[i] += sgn * step
                jobs.append((k, xi, True))
    if not with_f:
        out = _run(jobs, processes)
    else:
        out = (_pool_map(_work_tolerant, jobs, processes) if processes > 1 and len(jobs) > 1
               else [_work_tolerant(j) for j in jobs])
        n_bad = next((k for k in range(len(samples)) if out[k] is None), None)
        if n_bad is not None:
            raise RuntimeError(f"SCC not converged for {samples[n_bad].name or 'a sample'}")
    n = len(samples)
    e = np.array([o[0] for o in out[:n]])
    if any(o[1] is None for o in out[:n]):
        jac = _energy_jacobian_fd(samples, x, processes)
    else:
        jac = np.array([o[1] for o in out[:n]])
    forces = {k: out[k][2] for k in with_f}
    fjac, pos = {}, n
    for k in with_f:
        # a displaced point whose SCC does not converge (a fragile cluster): the one-sided
        # difference with the other side, or a zero column if both fail
        cols = []
        f0 = out[k][2].ravel()
        for _ in range(len(x)):
            fp, fm = out[pos], out[pos + 1]
            if fp is not None and fm is not None:
                cols.append((fp[2] - fm[2]).ravel() / (2 * step))
            elif fp is not None:
                cols.append((fp[2].ravel() - f0) / step)
            elif fm is not None:
                cols.append((f0 - fm[2].ravel()) / step)
            else:
                cols.append(np.zeros_like(f0))
            pos += 2
        fjac[k] = np.array(cols).T
    return e, jac, forces, fjac


def _energy_jacobian_fd(samples, x, processes, step: float = ENERGY_STEP) -> np.ndarray:
    """d(F/N)/dx (n, n_x) by central differences (models without analytic dF/dx)."""
    jobs = []
    for k in range(len(samples)):
        for i in range(len(x)):
            for sgn in (1.0, -1.0):
                xi = x.copy()
                xi[i] += sgn * step
                jobs.append((k, xi, False))
    out = _run(jobs, processes)
    jac = np.zeros((len(samples), len(x)))
    p = 0
    for k in range(len(samples)):
        for i in range(len(x)):
            jac[k, i] = (out[p][0] - out[p + 1][0]) / (2 * step)
            p += 2
    return jac


# ---- band samples (A series) -------------------------------------------------------------------
def evaluate_band_sample(models, bs: BandSample, sk, options: dict, settings=None,
                         jacobian: bool = False, method: str = "frozen", step: float = 1e-5):
    """(compare dict, residuals r (eV), Jacobian dr/dx (eV per unit x) or None) of one band
    sample: full SCC at the reference's structure, eigenstates at its k-points,
    :func:`pydftb.bandloss.residuals`; Jacobian ``method`` "frozen" (Hellmann-Feynman at the
    SCC's charges) or "scc" (central differences of ``step`` with full SCC)."""
    from . import banddiff
    from . import bandloss
    from .bands import eigenstates
    from .calculator import DFTB
    ref = bs.ref
    at = ref.atoms()
    if at is None:
        raise ValueError(f"band reference {ref.name} has no structure")
    periodic = bool(np.any(at.pbc))
    opts = {**dict(options), **dict(bs.options or {})}
    if periodic and bs.kpts is not None:
        opts["kpts"] = _kpoints(at, bs.kpts)
    labels = at.get_chemical_symbols()
    pos = at.positions / BOHR
    cell = at.cell.array / BOHR if periodic else None
    pbc = np.ones(3, bool) if periodic else None
    kk = None
    if ref.kind != "molecule" and periodic:
        kk = (np.asarray(ref.kpts, float),
              ref.kweights if ref.kind == "ibz" else np.ones(len(ref.kpts)))
    settings = bandloss.BandSettings.from_dict(settings) if isinstance(settings, dict) else settings

    def bands_at(ms, want_vectors=False):
        eng, r = _calculate(ms, sk, at, opts, False, ref.name)
        st = eigenstates(eng, labels, pos, cell, pbc, r.q_in, kpts=kk, vectors=want_vectors)
        if st.eigenvalues.shape[0] != 1 or st.two_component:
            raise NotImplementedError("band fits support spin-unpolarized DFTB only")
        return eng, r, st

    eng, r, st = bands_at(models, jacobian and method == "frozen")
    n_el = float(r.n_electrons)
    nv = int(np.ceil(n_el / 2.0 - 1e-6))
    fermi = float(np.atleast_1d(r.fermi)[0]) * HARTREE
    db = bandloss.DFTBBands(np.asarray(st.eigenvalues, float) * HARTREE, nv, fermi, n_el)
    res = bandloss.residuals(ref, db, settings)
    if not jacobian:
        return res["compare"], res["r"], None
    x = vector_of(models)
    wanted = bandloss.residual_states(res)
    if method == "frozen":
        if all(isinstance(m, AdaptiveParams) for m in models):
            deps = banddiff.adaptive_eigenvalue_derivatives(eng, models, labels, pos, cell, pbc,
                                                            r.q_in, kk, states=st, wanted=wanted)
        else:
            frozen_opts = {"convergent_scc_only": False, **opts}

            def make(xx):
                return DFTB(sk, **engine_kwargs(split_vector(models, xx)), **frozen_opts)

            deps = banddiff.eigenvalue_derivatives(make, x, labels, pos, cell, pbc, r.q_in, kk,
                                                   states=st, step=step, wanted=wanted)
    elif method == "scc":
        deps = np.zeros(st.eigenvalues.shape + (len(x),))
        for j in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[j] += step
            xm[j] -= step
            ep = bands_at(split_vector(models, xp))[2].eigenvalues
            em = bands_at(split_vector(models, xm))[2].eigenvalues
            deps[..., j] = (ep - em) / (2 * step)
    else:
        raise ValueError(f"band_jacobian is 'frozen' or 'scc', not {method!r}")
    return res["compare"], res["r"], bandloss.residual_jacobian(res, deps * HARTREE)


def _work_band(args):
    k, x, jacobian = args
    w = _WORK
    with _blas_one():
        return evaluate_band_sample(_split(w["models"], x), w["band_samples"][k], w["sk"],
                                    w["options"], w.get("band_settings"), jacobian,
                                    w.get("band_method", "frozen"), w.get("band_step", 1e-5))


def _work_band_tolerant(args):
    try:
        return _work_band(args)
    except (RuntimeError, np.linalg.LinAlgError) as exc:
        return exc


def evaluate_band_all(models, band_samples, sk, options, x, processes: int = 1, settings=None,
                      jacobian: bool = True, method: str = "frozen", step: float = 1e-5,
                      tolerant: bool = False) -> list:
    """[(compare dict, r, J or None)] of every band sample at x (``tolerant``: a failed SCC
    gives the exception in place of the tuple instead of raising)."""
    _WORK.update(models=tuple(models), band_samples=list(band_samples), sk=sk,
                 options=dict(options), band_settings=settings, band_method=method,
                 band_step=step)
    x = np.asarray(x, float)
    jobs = [(k, x, jacobian) for k in range(len(band_samples))]
    func = _work_band_tolerant if tolerant else _work_band
    if processes > 1 and len(jobs) > 1:
        return _pool_map(func, jobs, processes)
    return [func(j) for j in jobs]


def evaluate_bands(models, band_samples, sk, options=None, settings=None,
                   processes: int = 1) -> list:
    """The :func:`pydftb.bandloss.compare` dict of every band sample with these models (a
    failed SCC gives ``loss`` NaN and its error text, as :func:`pydftb.bandloss.evaluate_refs`)."""
    models = _as_models(models)
    out = evaluate_band_all(models, band_samples, sk, dict(options or {}), vector_of(models),
                            processes, settings, jacobian=False, tolerant=True)
    res = []
    for bs, o in zip(band_samples, out):
        if isinstance(o, Exception):
            res.append({"name": bs.ref.name, "group": bs.group_name, "kind": bs.ref.kind,
                        "loss": np.nan, "error": f"{type(o).__name__}: {o}"})
        else:
            res.append({**o[0], "group": bs.group_name})
    return res


def _band_multipliers(band_samples, band_weight: float) -> np.ndarray:
    """sqrt(band_weight x weight / (n_groups x n_in_group)) per band sample."""
    groups = {}
    for bs in band_samples:
        groups[bs.group_name] = groups.get(bs.group_name, 0) + 1
    return np.array([np.sqrt(band_weight * bs.weight / (len(groups) * groups[bs.group_name]))
                     for bs in band_samples])


def _composition(samples):
    """(elements, (n, n_el) fractions n_el / N) of the samples."""
    els = sorted({e for s in samples for e in s.atoms.get_chemical_symbols()})
    comp = np.zeros((len(samples), len(els)))
    for k, s in enumerate(samples):
        sym = s.atoms.get_chemical_symbols()
        for c, e in enumerate(els):
            comp[k, c] = sym.count(e) / len(sym)
    return els, comp


def _energy_residuals(d, jac, w, wn, mode, comp=None):
    """(residuals, Jacobian, d - offsets, mean offset, per-element offsets or None)."""
    if mode == "global":
        off = wn @ d
        return w * (d - off), w[:, None] * (jac - wn @ jac), d - off, off, None
    if mode == "none":
        return w * d, w[:, None] * jac, d, 0.0, None
    a = w[:, None] * comp
    pinv = np.linalg.pinv(a, rcond=1e-10)
    c = pinv @ (w * d)
    resid = d - comp @ c
    pj = jac - comp @ (pinv @ (w[:, None] * jac))
    return w * resid, w[:, None] * pj, resid, float(wn @ (comp @ c)), c


def fit_adaptive(models, samples, sk, options: dict | None = None, l2: float = 1e-4,
                 processes: int | None = None, max_evaluations: int = 200, x0=None,
                 verbose: bool = False, force_weight: float = 0.0,
                 force_step: float = 1e-4, band_samples=None, band_weight: float = 0.0,
                 band_settings=None, offsets: str = "global", band_jacobian: str = "frozen",
                 band_step: float = 1e-5) -> FitResult:
    """Least-squares fit of the coefficients of ``models`` to the relative energies of
    ``samples`` (scipy's trust-region reflective method with the analytic Jacobian).  ``l2``
    (eV/atom per unit coefficient, squared) pulls the coefficients towards their start.
    ``force_weight`` ((eV/atom)^2 per (eV/A)^2): the weight of the force residuals of the
    samples that have reference forces (M8b; their Jacobian by central differences of
    ``force_step`` in every coefficient).  A trial step at which some SCC does not converge is
    rejected (scored far above the last good point), not fatal; only a failure at the start
    raises.

    A series: ``band_samples`` with ``band_weight`` > 0 add band residuals (``band_settings``:
    :class:`pydftb.bandloss.BandSettings` or its dict; ``band_jacobian`` "frozen" or "scc",
    ``band_step`` its x step); ``samples`` may then be empty (a fit to the bands alone);
    ``offsets`` "global", "none" or "per_element" (module docstring).  Without these
    arguments the fit is exactly the one before them."""
    from scipy.optimize import least_squares
    models = _as_models(models)
    options = dict(options or {})
    samples = list(samples)
    band_samples = list(band_samples or [])
    use_b = band_weight > 0.0 and len(band_samples) > 0
    if offsets not in OFFSETS:
        raise ValueError(f"offsets must be one of {OFFSETS}, not {offsets!r}")
    if not samples and not use_b:
        raise ValueError("nothing to fit: no samples, and no band samples with band_weight > 0")
    if processes is None:
        processes = max(1, min(len(samples) + (len(band_samples) if use_b else 0),
                               (os.cpu_count() or 1)))
    x_start = np.concatenate([m.vector() for m in models]) if x0 is None else np.asarray(x0)
    w = np.sqrt(np.array([s.weight for s in samples], float))
    wn = w ** 2 / (w ** 2).sum() if samples else np.zeros(0)
    t_ref = np.array([s.energy for s in samples], float)
    use_f = force_weight > 0.0 and any(s.forces is not None for s in samples)
    elements, comp = (_composition(samples) if (offsets == "per_element" and samples)
                      else ([], None))
    mult = _band_multipliers(band_samples, band_weight) if use_b else np.zeros(0)
    cache, history = {}, []
    t0 = time.perf_counter()

    last = {}

    def compute(x):
        key = x.tobytes()
        if key not in cache:
            try:
                if samples:
                    if use_f and last:
                        # a trial step: the energies alone (no force Jacobian, about 2 n_x
                        # times cheaper) show first whether every SCC converges there
                        evaluate_all(models, samples, sk, options, x, processes,
                                     with_forces=False)
                    e, jac, forces, fjac = evaluate_all(models, samples, sk, options, x,
                                                        processes, with_forces=use_f,
                                                        step=force_step)
                else:
                    e, jac, forces, fjac = np.zeros(0), np.zeros((0, len(x))), {}, {}
                bands = (evaluate_band_all(models, band_samples, sk, options, x, processes,
                                           band_settings, True, band_jacobian, band_step)
                         if use_b else [])
            except (RuntimeError, np.linalg.LinAlgError) as exc:
                # a trial step where some SCC does not converge: a residual far above the last
                # good one, so the trust-region method rejects the step and shrinks its radius
                if not last:
                    raise
                if verbose:
                    print(f"  trial step rejected: {exc}", flush=True)
                cache.clear()
                cache[key] = (last["res"] * 1e3, last["jac"]) + last["rest"]
                return cache[key]
            d = e - t_ref
            if samples:
                r_e, j_e, resid_e, off, c_el = _energy_residuals(d, jac, w, wn, offsets, comp)
            else:
                r_e, j_e, resid_e, off, c_el = np.zeros(0), np.zeros((0, len(x))), d, 0.0, None
            res = [r_e]
            jres = [j_e]
            f_err = []
            for k in sorted(forces):
                df = (forces[k] - np.asarray(samples[k].forces, float)).ravel()
                f_err.append(df)
                res.append(np.sqrt(force_weight) * w[k] * df)
                jres.append(np.sqrt(force_weight) * w[k] * fjac[k])
            res.append(np.sqrt(l2) * (x - x_start))
            jres.append(np.sqrt(l2) * np.eye(len(x)))
            b_loss, b_res = None, []
            if use_b:
                from .bandloss import band_loss
                for (comp_b, r_b, j_b), mb, bs in zip(bands, mult, band_samples):
                    res.append(mb * r_b)
                    jres.append(mb * j_b)
                    b_res.append({**comp_b, "group": bs.group_name})
                b_loss = band_loss(b_res)
            f_rmse = float(np.sqrt(np.mean(np.concatenate(f_err) ** 2))) if f_err else None
            extras = {"offsets": (None if c_el is None
                                  else {el: float(v) for el, v in zip(elements, c_el)}),
                      "band_loss": b_loss, "band_results": b_res}
            cache.clear()
            cache[key] = (np.concatenate(res), np.vstack(jres), resid_e, off, f_rmse, extras)
            last.update(res=cache[key][0], jac=cache[key][1], rest=cache[key][2:])
            rmse = float(np.sqrt(wn @ resid_e ** 2)) if samples else float("nan")
            history.append((time.perf_counter() - t0, rmse, f_rmse) if not use_b
                           else (time.perf_counter() - t0, rmse, f_rmse, b_loss))
            if verbose:
                fr = "" if f_rmse is None else f"  forces {f_rmse * 1000:8.2f} meV/A"
                br = "" if b_loss is None else f"  bands {b_loss * 1000:8.2f} meV"
                er = f"  rmse {rmse * 1000:8.2f} meV/atom" if samples else ""   # band-only fits
                print(f"  eval {len(history):3d}{er}{fr}{br}", flush=True)
        return cache[key]

    sol = least_squares(lambda x: compute(x)[0], x_start, jac=lambda x: compute(x)[1],
                        method="trf", max_nfev=max_evaluations, x_scale="jac")
    _, _, resid, off, f_rmse, *more = compute(sol.x)
    extras = more[0] if more else {}
    fitted = _split(models, sol.x)
    rmse = float(np.sqrt(wn @ resid ** 2)) if samples else float("nan")
    return FitResult(fitted, sol.x, rmse, resid, float(off),
                     history, len(history), time.perf_counter() - t0, f_rmse,
                     extras.get("band_loss"), extras.get("band_results", []),
                     extras.get("offsets"))


def predict(models, samples, sk, options=None, offset: float = 0.0, processes: int | None = None):
    """Model energies per atom (eV) minus ``offset``, for samples not in the fit."""
    models = _as_models(models)
    if processes is None:
        processes = max(1, min(len(samples), (os.cpu_count() or 1)))
    x = np.concatenate([m.vector() for m in models])
    e, _ = evaluate(models, samples, sk, dict(options or {}), x, processes)
    return e - offset
