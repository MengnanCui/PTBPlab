"""Active learning for environment-adaptive parameters (milestone M8c).

A committee of fits of the same :class:`pydftb.adaptive.AdaptiveParams` measures where the
model is unsure: every member is refitted to a bootstrap resample of the training samples
(``groups=`` resamples whole groups, e.g. phases, which spreads the members more where whole
kinds of structures are missing), starting from the base fit and pulled towards it by the
ridge term.  On a new structure the spread of the members' relative energies per atom and of
their forces is the uncertainty; the most uncertain structures of an MD trajectory, unlike each
other and unlike the training set (a fingerprint of the CN and angular descriptors), are the
ones to compute with DFT next (:func:`select`, written as VASP inputs by
:mod:`pydftb.bench.cu_dft_dataset`).

    com = fit_committee(models, samples, sk, options, n=4, groups=phases, base=fit)
    unc = committee_predict(com, frames, sk, options)        # unc.sigma_e, unc.sigma_f
    pick = select(unc.score, fingerprints(com.base, frames), n=10,
                  reference=fingerprints(com.base, [s.atoms for s in samples]))

Committee energies are relative: each member's prediction minus its own offset (the weighted
mean of model minus reference over its training samples), so the free zero of the energy drops
out of the spread.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np

from .adaptive import RBF, AdaptiveParams, make_models
from .adaptive_fit import Sample, evaluate_sample, fit_adaptive
from .constants import BOHR

_WORK = {}                                 # fork-inherited: members, frames, sk, options


@dataclass
class Committee:
    base: tuple                            # the base fit's models
    members: tuple                         # tuple of model tuples
    offsets: np.ndarray                    # eV/atom: each member's model - reference
    base_offset: float = 0.0
    draws: tuple = ()                      # the bootstrap counts of every member
    perturb: tuple = ()                    # the anchor perturbation each member got


@dataclass
class Uncertainty:
    energy: np.ndarray                     # (members, frames) relative eV/atom (nan: failed)
    forces: list                           # per frame: (members, atoms, 3) eV/A, or None
    mean_e: np.ndarray                     # (frames,)
    sigma_e: np.ndarray                    # (frames,) eV/atom, std over the members
    sigma_f: np.ndarray                    # (frames,) eV/A, the largest atom's force spread
    failed: np.ndarray                     # (frames,) bool: a member's SCC did not converge

    def score(self, force_scale: float = 0.1) -> np.ndarray:
        """sigma_e + sigma_f * force_scale (eV/atom per eV/A); a failed frame scores inf."""
        s = self.sigma_e + force_scale * np.nan_to_num(self.sigma_f)
        return np.where(self.failed, np.inf, s)


def bootstrap_weights(n_samples: int, rng, groups=None) -> np.ndarray:
    """Multiplicities of the samples in one bootstrap draw: of the samples, or (``groups``: one
    label per sample) of whole groups, each drawn group bringing all its samples."""
    if groups is None:
        return np.bincount(rng.integers(0, n_samples, n_samples), minlength=n_samples)
    groups = np.asarray(groups)
    labels = np.unique(groups)
    counts = np.bincount(rng.integers(0, len(labels), len(labels)), minlength=len(labels))
    out = np.zeros(n_samples, int)
    for lab, c in zip(labels, counts):
        out[groups == lab] = c
    return out


def fit_committee(models, samples, sk, options=None, n: int = 4, seed: int = 0, groups=None,
                  base=None, l2: float = 1e-4, max_evaluations: int = 30,
                  base_evaluations: int = 150, processes: int | None = None,
                  perturb: float = 0.0, perturb_skip: tuple = (), verbose: bool = False,
                  **fit_kw) -> Committee:
    """A base fit (``base``: a FitResult already made on ``samples``, else fitted here from
    ``models``) and ``n`` members refitted from it to bootstrap resamples of ``samples``
    (``groups``: resample whole groups).  Extra keywords go to every
    :func:`pydftb.adaptive_fit.fit_adaptive`.

    ``perturb``: every member starts from, and its ridge pulls it towards, its own anchor
    base + perturb * N(0, 1) per coefficient (an anchored ensemble, Pearce et al. 2020).
    Without it the ridge pulls every member to the base, so directions no data constrains keep
    the base's values in all members and the committee agrees exactly where it should not
    (M8c report).  ``perturb`` near sqrt(noise^2 / l2), the prior width the ridge implies.
    ``perturb_skip``: coefficients whose names contain one of these strings are not perturbed
    (e.g. ``("pair",)``: the pair-repulsive polynomial, whose high powers move short-range
    energies by eV for a perturbation of 0.003 and break the SCC of short dimers)."""
    rng = np.random.default_rng(seed)
    if base is None:
        base = fit_adaptive(models, samples, sk, options, l2=l2, processes=processes,
                            max_evaluations=base_evaluations, verbose=verbose, **fit_kw)
    members, offsets, draws, scales = [], [], [], []
    for k in range(n):
        w = bootstrap_weights(len(samples), rng, groups)
        while not w.any():                                     # pragma: no cover (n >= 1)
            w = bootstrap_weights(len(samples), rng, groups)
        sub = [replace(s, weight=s.weight * c) for s, c in zip(samples, w) if c > 0]
        z = rng.normal(size=len(base.x))
        if perturb_skip:
            names = [nm for m in base.models for nm in m.parameter_names()]
            z[[i for i, nm in enumerate(names) if any(t in nm for t in perturb_skip)]] = 0.0
        scale = perturb
        for attempt in range(4):              # an anchor too far for the SCC: halve it
            x0 = np.asarray(base.x, float) + scale * z if scale else None
            try:
                r = fit_adaptive(base.models, sub, sk, options, l2=l2, processes=processes,
                                 max_evaluations=max_evaluations, x0=x0, verbose=verbose,
                                 **fit_kw)
                break
            except RuntimeError:
                if not scale or attempt == 3:
                    raise
                scale *= 0.5
                if verbose:
                    print(f"member {k}: SCC failed, anchor perturbation -> {scale:g}",
                          flush=True)
        scales.append(scale)
        if verbose:
            print(f"member {k}: {int((w > 0).sum())} distinct samples, rmse "
                  f"{r.rmse * 1000:.2f} meV/atom", flush=True)
        members.append(r.models)
        offsets.append(r.offset)
        draws.append(tuple(int(c) for c in w))
    return Committee(tuple(base.models), tuple(members), np.array(offsets), float(base.offset),
                     tuple(draws), tuple(scales))


def _work(job):
    from threadpoolctl import threadpool_limits
    m, k = job
    w = _WORK
    sample = Sample(w["frames"][k], 0.0, kpts=w["kpts"][k], name=f"frame {k}")
    with threadpool_limits(1, user_api="blas"):
        try:
            e, _, f = evaluate_sample(w["members"][m], sample, w["sk"], w["options"],
                                      forces=w["forces"])
        except (RuntimeError, np.linalg.LinAlgError):
            return np.nan, None
    return e, f


def committee_predict(committee: Committee, frames, sk, options=None, kpts=None,
                      forces: bool = True, processes: int | None = None) -> Uncertainty:
    """Every member on every frame (ASE Atoms; ``kpts`` one grid or None per frame): relative
    energies per atom, forces, and their spread.  A frame where a member's SCC fails is
    marked ``failed`` (its sigma is nan)."""
    frames = list(frames)
    kpts = [None] * len(frames) if kpts is None else list(kpts)
    jobs = [(m, k) for m in range(len(committee.members)) for k in range(len(frames))]
    if processes is None:
        processes = max(1, min(len(jobs), os.cpu_count() or 1))
    _WORK.update(members=committee.members, frames=frames, kpts=kpts, sk=sk,
                 options=dict(options or {}), forces=forces)
    try:
        if processes > 1 and len(jobs) > 1:
            from .adaptive_fit import _pool_map
            out = _pool_map(_work, jobs, processes)
        else:
            out = [_work(j) for j in jobs]
    finally:
        _WORK.clear()
    nm, nf = len(committee.members), len(frames)
    e = np.array([o[0] for o in out], float).reshape(nm, nf) - committee.offsets[:, None]
    failed = np.isnan(e).any(axis=0)
    fr = []
    sig_f = np.full(nf, np.nan)
    for k in range(nf):
        if not forces or failed[k]:
            fr.append(None)
            continue
        f = np.stack([out[m * nf + k][1] for m in range(nm)])
        fr.append(f)
        dev = f - f.mean(axis=0)
        sig_f[k] = float(np.sqrt((dev ** 2).sum(axis=2).mean(axis=0)).max())
    return Uncertainty(e, fr, e.mean(axis=0), e.std(axis=0), sig_f, failed)


CN_BINS = np.arange(0.0, 14.5, 1.0)
Q_BINS = np.arange(-0.4, 1.05, 0.1)


def fingerprints(models, frames, width: float = 0.5, q_width: float = 0.05) -> np.ndarray:
    """One vector per frame: Gaussian-smeared histograms of every atom's CN and angular q
    (per atom, so frames of different sizes compare), from the models' own descriptors."""
    models = make_models(models)
    probe = [AdaptiveParams(m.element, window=m.window, window_scaled=m.window_scaled,
                            neighbours=m.neighbours,
                            angular=RBF((0.5,), 0.2, constant=False)) for m in models]
    out = []
    for at in frames:
        lab = at.get_chemical_symbols()
        periodic = bool(np.any(at.pbc))
        cell = at.cell.array / BOHR if periodic else None
        pbc = np.asarray(at.pbc, bool) if periodic else None
        cn = np.zeros(len(at))
        q = np.zeros(len(at))
        for m in probe:
            c, qq, _, _ = m.descriptors(lab, at.positions / BOHR, cell, pbc)
            cn += c
            q += qq
        h1 = np.exp(-0.5 * ((cn[:, None] - CN_BINS[None, :]) / width) ** 2).sum(0)
        h2 = np.exp(-0.5 * ((q[:, None] - Q_BINS[None, :]) / q_width) ** 2).sum(0)
        out.append(np.concatenate([h1 / h1.sum(), h2 / max(h2.sum(), 1e-300)]))
    return np.array(out)


def select(score, fp, n: int, threshold: float = 0.0, reference=None,
           min_distance: float = 0.05) -> list:
    """Indices of at most ``n`` frames, most uncertain first: ``score`` above ``threshold``,
    and each at least ``min_distance`` (fingerprint, Euclidean) from the frames already taken
    and from ``reference`` (the training set's fingerprints)."""
    score = np.asarray(score, float)
    fp = np.asarray(fp, float)
    taken = [] if reference is None else [np.asarray(r, float) for r in reference]
    out = []
    for k in np.argsort(-score, kind="stable"):
        if len(out) >= n or not score[k] > threshold:
            break
        if taken and min(np.linalg.norm(fp[k] - t) for t in taken) < min_distance:
            continue
        out.append(int(k))
        taken.append(fp[k])
    return out


def write_candidates(out_dir, frames, names, group: str = "active", time_limit=None):
    """The chosen frames as VASP single points (:func:`pydftb.bench.cu_dft_dataset.write_set`):
    a frame without periodicity is boxed in vacuum, Gamma only and spin-polarised.  Writes only;
    nothing is submitted."""
    from .bench.cu_dft_dataset import _boxed, write_set
    items = []
    for at, name in zip(frames, names):
        if np.any(at.pbc):
            items.append((group, name, at.copy(), False, False))
        else:
            items.append((group, name, _boxed(at), True, True))
    return write_set(out_dir, items, time_limit=time_limit)
