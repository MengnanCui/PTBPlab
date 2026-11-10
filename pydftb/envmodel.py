"""The fitting protocol of environment-dependent models (A series, A0 plan sections 4 and 5),
so that one fitter (:func:`pydftb.adaptive_fit.fit_adaptive`) and one model file format serve
every kind of model:

* M8a's environment-adaptive parameters (:class:`pydftb.adaptive.AdaptiveParams`; channel E3
  of A2 and the repulsive channels of A3), engine keyword ``adaptive``;
* the r0 mixing of :mod:`pydftb.r0mix` (channel E2) and per-type on-site/U mixing (E1), engine
  keyword ``type_switching``.

A model provides

* ``element``                    the element whose atoms it describes;
* ``engine_keyword``             the :class:`pydftb.DFTB` keyword that takes a list of such models;
* ``vector()`` / ``with_vector(x)`` / ``parameter_names()``   its fitted coefficients;
* ``gradient_from(result)``      dF/d(vector) in Hartree from a :class:`DFTBResult` of a
                                 calculation with this model, or None when the engine does not
                                 provide it (the fitter then uses finite differences);
* ``to_dict()``                  JSON; :func:`model_from_dict` reads it back.

Model files (:func:`save_models`, :func:`load_models`): JSON ``{"models": [...], "meta": {...}}``.
"""
from __future__ import annotations

import json

import numpy as np

ENGINE_KEYWORDS = ("adaptive", "type_switching")


def _as_tuple(models) -> tuple:
    if models is None:
        return ()
    if isinstance(models, (list, tuple)):
        return tuple(models)
    return (models,)


def engine_kwargs(models) -> dict:
    """{engine keyword: [models]} for :class:`pydftb.DFTB` (``DFTB(sk, **engine_kwargs(m))``)."""
    out = {}
    for m in _as_tuple(models):
        key = getattr(m, "engine_keyword", None)
        if key not in ENGINE_KEYWORDS:
            raise TypeError(f"{type(m).__name__} is not an environment model (engine_keyword "
                            f"must be one of {ENGINE_KEYWORDS}, not {key!r})")
        out.setdefault(key, []).append(m)
    return out


def vector_of(models) -> np.ndarray:
    parts = [np.asarray(m.vector(), float) for m in _as_tuple(models)]
    return np.concatenate(parts) if parts else np.zeros(0)


def split_vector(models, x) -> tuple:
    """The models with their coefficients replaced by consecutive slices of x."""
    x = np.asarray(x, float)
    out, k = [], 0
    for m in _as_tuple(models):
        n = len(m.vector())
        out.append(m.with_vector(x[k:k + n]))
        k += n
    if k != len(x):
        raise ValueError(f"the models have {k} coefficients, not {len(x)}")
    return tuple(out)


def parameter_names(models) -> list:
    return [n for m in _as_tuple(models) for n in m.parameter_names()]


def model_gradient(result, models):
    """dF/d(concatenated vectors) in Hartree, or None if some model cannot give it."""
    parts = []
    for m in _as_tuple(models):
        g = m.gradient_from(result)
        if g is None:
            return None
        parts.append(np.asarray(g, float))
    return np.concatenate(parts) if parts else np.zeros(0)


# ---- serialisation ------------------------------------------------------------------------------
_READERS = {}


def register(kind: str, reader) -> None:
    """``reader(d) -> model`` for ``to_dict()`` results with ``d["kind"] == kind``."""
    _READERS[kind] = reader


def model_to_dict(model) -> dict:
    return model.to_dict()


def model_from_dict(d: dict):
    kind = d.get("kind")
    if kind not in _READERS:
        _load_builtin_readers()
    if kind not in _READERS:
        raise ValueError(f"unknown model kind {kind!r} (known: {sorted(_READERS)})")
    return _READERS[kind](d)


def _load_builtin_readers():
    # importing the model modules registers their readers
    from . import adaptive  # noqa: F401
    try:
        from . import r0mix  # noqa: F401
    except ImportError:
        pass


def save_models(path: str, models, meta: dict | None = None) -> None:
    with open(path, "w") as fh:
        json.dump({"models": [model_to_dict(m) for m in _as_tuple(models)],
                   "meta": dict(meta or {})}, fh, indent=1)


def load_models(path: str):
    """(models tuple, meta dict) of a :func:`save_models` file."""
    with open(path) as fh:
        d = json.load(fh)
    return tuple(model_from_dict(m) for m in d["models"]), d.get("meta", {})
