"""Model directories of the A series: one directory is one ready-to-use model, so that the MD
checks, the baselines, the gates, A3 and A4 take any model by its path (A0 plan, sections 3-6).

model.json, format "a-series-model/1"::

    {"format": "a-series-model/1",
     "skdir": "/abs/path/or/relative/to/the/model/dir",   # SK files
     "repulsive": "skf",               # "skf": the files' Spline blocks; "none": electronic only
     "dftb": {...},                    # the A1 scan's dftb section: max_l, smearing, kspacing, ...
     "env_models": "env_models.json",  # pydftb.envmodel.save_models file (relative), or null
     "delta": {"kind": "mace", "path": "delta.model", "dtype": "float64"} | null,
     "dispersion": {"kind": "d3bj", "method": "PBE"} | null,
     "meta": {...}}                    # free: stage, variant, parent, fit statistics, ...

A directory written by ``ptbp scan export`` (:func:`utils.accuracy.export_model`) has a
model.json of its own (scan, index, strategy, repulsive, radii, sigma, max_l, skdir), which is
read as it is: the full SK files are the directory itself, ``repulsive = "skf"``, and the DFTB
settings are the A-series defaults with that ``max_l`` (pass the scan config's ``dftb`` section
to :func:`load_model` to use exactly the scan's).

Energies: the A-series DFT energies are VASP's energy(sigma->0), so :meth:`Model.calculator`
reports DFTB's ``E - TS/2`` (``energy="e_zero"``, as the scans) by default; MD uses the
force-consistent free energy (``energy="free"``), whose drift is the stability measure.  Forces
are always the gradient of the free energy (VASP's forces are too).

Dispersion: the ink references are plain PBE, and every fit compares plain PBE with plain DFTB.
A model's ``dispersion`` (PBE-D3(BJ) by simple-dftd3) is added only when a caller asks for it
(``dispersion=True``, e.g. the ink MD); a comparison against DFT data with D3 on must add the
same D3 to the reference (:func:`add_d3_to_reference`, ``score(..., dispersion=True)``).  3ob's
D3 parameters are never used.

Helpers for code that works with :class:`pydftb.DFTB` directly (the fitters of A2/A3):
:func:`engine_options`, :func:`engine_skset` and :func:`engine_kpoints` give exactly the A1 scan's
settings (utils.accuracy.calculator): Gaussian smearing, Gamma-centred k-points by the KSPACING
rule, the batched eigensolver, the SCC tolerance.  Never pass a bare (n1, n2, n3) tuple as
k-points for the A-series data: pydftb reads it as a Monkhorst-Pack grid, shifted for even n.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass, field

import numpy as np

FORMAT = "a-series-model/1"
MODEL_FILE = "model.json"
KB_EV = 8.617333262e-5
HARTREE_EV = 27.211386245988
# the A-series DFTB settings (A0 section 1; utils.accuracy.calculator's defaults)
DEFAULT_DFTB = {"smearing": ["gaussian", 0.05], "kspacing": 0.15, "scc_tol": 1.0e-6,
                "max_scc": 300, "mixer": "broyden", "mixing": None}
D3_DEFAULT = {"kind": "d3bj", "method": "PBE"}


def _setup_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    acc_dir = os.path.dirname(here)
    if acc_dir not in sys.path:
        sys.path.insert(0, acc_dir)
    from common.paths import setup
    setup()


_setup_paths()


# ---- DFTB settings -----------------------------------------------------------------------------
def smearing_of(dcfg: dict) -> tuple:
    """(kind, T in K) of a dftb section's ``smearing: [kind, width eV]`` ("fermi_K": T in K)."""
    kind, width = (dcfg or {}).get("smearing", DEFAULT_DFTB["smearing"])
    temperature = float(width) / KB_EV if kind != "fermi_K" else float(width)
    return {"fermi_K": "fermi"}.get(kind, kind), temperature


def _retry(opts: dict, attempt: int) -> dict:
    """utils.accuracy.calculator's retries of a failed SCC: slower Broyden, then Anderson."""
    if attempt == 1:
        opts.update(mixing=0.05, max_scc=1000)
    elif attempt >= 2:
        opts.update(mixer="anderson", mixing=0.02, max_scc=2000)
    return opts


def engine_options(dcfg: dict, attempt: int = 0, repulsive: str = "none",
                   for_fit: bool = False) -> dict:
    """Keyword options of :class:`pydftb.DFTB` equal to what utils.accuracy.calculator gives
    PyDFTB (smearing in K, SCC settings, the batched solver, DFTB3/U if set); ``max_l`` and
    ``shell_resolved`` belong to the SK set (:func:`engine_skset`), k-points to the structure
    (:func:`engine_kpoints`).

    ``for_fit``: for :func:`pydftb.adaptive_fit.fit_adaptive`, whose own retry (gentler mixing
    when an SCC does not converge) needs ``convergent_scc_only`` and an unset ``mixing`` left
    out; the other settings are the same."""
    d = dict(DEFAULT_DFTB, **(dcfg or {}))
    opts = dict(smearing=smearing_of(d), repulsive=repulsive, scc_tol=float(d["scc_tol"]),
                max_scc=int(d["max_scc"]), mixer=d.get("mixer", "broyden"),
                mixing=d.get("mixing"), solver="batched", convergent_scc_only=True)
    _retry(opts, attempt)
    if d.get("third_order"):
        opts["third_order"] = d["third_order"]
        opts["hubbard_derivs"] = d.get("hubbard_derivs")
    if d.get("hubbard"):
        opts["hubbard"] = d["hubbard"]
    if for_fit:
        opts.pop("convergent_scc_only")
        if opts.get("mixing") is None:
            opts.pop("mixing")
    return opts


def sk_options(dcfg: dict) -> dict:
    """ParamSet keywords of a dftb section (``max_l``, ``shell_resolved``)."""
    out = {"max_l": (dcfg or {}).get("max_l")}
    if (dcfg or {}).get("shell_resolved") is not None:
        out["shell_resolved"] = bool(dcfg["shell_resolved"])
    return out


def engine_skset(skdir: str, dcfg: dict, labels):
    """The SK set (pydftb.params.ParamSet(...).skset) of these labels with the section's basis."""
    from pydftb.params import ParamSet
    return ParamSet(skdir, **sk_options(dcfg)).skset(list(dict.fromkeys(labels)))


def kspacing_of(dcfg: dict):
    return (dcfg or {}).get("kspacing", DEFAULT_DFTB["kspacing"])


def engine_kpoints(frame, dcfg: dict):
    """(kpts (n_k, 3) fractional, weights) of utils.accuracy.kpoints_for (Gamma-centred,
    KSPACING rule or the frame's ``info['kpts']``), reduced by k -> -k as PyDFTB does; None
    for a molecule or the Gamma point."""
    from pydftb.ase_calc import kpoints_of
    from utils.accuracy import kpoints_for
    spec = kpoints_for(frame, kspacing_of(dcfg))
    if spec is None:
        return None
    out = kpoints_of(frame, spec)
    if out is None:
        return None
    k, w = out
    return np.asarray(k, float), np.asarray(w, float)


def pydftb_options(dcfg: dict, frame, attempt: int = 0) -> dict:
    """PyDFTB keywords equal to utils.accuracy.calculator's (without ``repulsive``)."""
    from utils.accuracy import _threads, kpoints_for
    d = dict(DEFAULT_DFTB, **(dcfg or {}))
    opts = dict(max_l=d.get("max_l"), kpts=kpoints_for(frame, kspacing_of(d)),
                smearing=smearing_of(d), scc_tol=float(d["scc_tol"]), max_scc=int(d["max_scc"]),
                mixer=d.get("mixer", "broyden"), mixing=d.get("mixing"), threads=_threads(d),
                convergent_scc_only=True)
    _retry(opts, attempt)
    if d.get("third_order"):
        opts["third_order"] = d["third_order"]
        opts["hubbard_derivs"] = d.get("hubbard_derivs")
    if d.get("hubbard"):
        opts["hubbard"] = d["hubbard"]
    if d.get("shell_resolved") is not None:
        opts["shell_resolved"] = bool(d["shell_resolved"])
    return opts


def _model_pydftb_class():
    from pydftb.ase_calc import PyDFTB

    class ModelPyDFTB(PyDFTB):
        """PyDFTB whose ``energy`` is E - TS/2 (``energy_kind="e_zero"``, VASP's
        energy(sigma->0)), the free energy (``"free"``, for MD) or E (``"total"``)."""

        def __init__(self, *args, energy_kind: str = "e_zero", **kw):
            if energy_kind not in ("e_zero", "free", "total"):
                raise ValueError(f"energy_kind must be e_zero, free or total, not {energy_kind!r}")
            self.energy_kind = energy_kind
            super().__init__(*args, **kw)

        def _store(self, r, sk, labels):
            super()._store(r, sk, labels)
            if self.excited:
                return
            from ase import units
            if self.energy_kind == "e_zero":
                self.results["energy"] = r.e_zero * units.Hartree
            elif self.energy_kind == "free":
                self.results["energy"] = r.e_force_related * units.Hartree

    return ModelPyDFTB


_CLASS = {}


def model_pydftb(*args, **kw):
    if "cls" not in _CLASS:
        _CLASS["cls"] = _model_pydftb_class()
    return _CLASS["cls"](*args, **kw)


# ---- dispersion ----------------------------------------------------------------------------------
def d3_calculator(spec: dict | None = None):
    """simple-dftd3's ASE calculator for ``spec`` ({"kind": "d3bj", "method": "PBE"})."""
    spec = dict(D3_DEFAULT, **(spec or {}))
    if spec["kind"] not in ("d3bj", "d3zero"):
        raise ValueError(f"dispersion kind must be d3bj or d3zero, not {spec['kind']!r}")
    try:
        from dftd3.ase import DFTD3
    except ImportError as exc:
        raise RuntimeError("dispersion needs simple-dftd3 with its Python API "
                           "(conda install -c conda-forge simple-dftd3 dftd3-python); "
                           "or switch dispersion off") from exc
    return DFTD3(method=spec["method"], damping={"d3bj": "d3bj", "d3zero": "d3zero"}[spec["kind"]])


def add_d3_to_reference(frames, spec: dict | None = None) -> list:
    """Copies of the frames whose reference energy and forces include the D3 of ``spec``, for
    comparing models with D3 on (the DFT data are plain PBE).  Isolated atoms have no D3."""
    from ase.calculators.singlepoint import SinglePointCalculator
    calc = d3_calculator(spec)
    out = []
    for at in frames:
        e = float(at.get_potential_energy())
        try:
            f = np.array(at.get_forces())
        except Exception:                 # noqa: BLE001
            f = None
        b = at.copy()
        b.calc = calc
        ed = float(b.get_potential_energy())
        fd = np.array(b.get_forces())
        c = at.copy()
        c.info = dict(at.info, d3_energy=ed)
        c.calc = SinglePointCalculator(c, energy=e + ed, free_energy=e + ed,
                                       forces=None if f is None else f + fd)
        out.append(c)
    return out


# ---- the model ------------------------------------------------------------------------------------
def _abs(base: str, p):
    if p is None:
        return None
    p = os.path.expandvars(os.path.expanduser(str(p)))
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


@dataclass
class Model:
    """A model directory (see the module docstring)."""
    path: str
    skdir: str
    repulsive: str = "skf"
    dftb: dict = field(default_factory=dict)
    env_models: tuple = ()
    env_file: str | None = None
    delta: dict | None = None
    dispersion: dict | None = None
    meta: dict = field(default_factory=dict)
    source: str = FORMAT                  # FORMAT, or "scan-export"
    _cache: dict = field(default_factory=dict, repr=False)

    # -- calculators -----------------------------------------------------------------------------
    def dftb_calculator(self, frame, energy: str = "e_zero", attempt: int = 0, **extra):
        """The DFTB part (PyDFTB with the model's SK files, env models, the A-series settings)."""
        from pydftb.envmodel import engine_kwargs
        opts = pydftb_options(self.dftb, frame, attempt)
        opts.update(extra)
        return model_pydftb(self.skdir, repulsive=self.repulsive, energy_kind=energy,
                            **engine_kwargs(self.env_models), **opts)

    def delta_calculator(self, device: str | None = None):
        """The Delta model's ASE calculator (cached per process), or None."""
        if not self.delta:
            return None
        key = ("delta", device)
        if key not in self._cache:
            kind = self.delta.get("kind", "mace")
            if kind != "mace":
                raise ValueError(f"unknown delta kind {kind!r} (mace)")
            from mace.calculators import MACECalculator
            self._cache[key] = MACECalculator(model_paths=_abs(self.path, self.delta["path"]),
                                              device=device or self.delta.get("device", "cpu"),
                                              default_dtype=self.delta.get("dtype", "float64"))
        return self._cache[key]

    def d3(self, spec: dict | None = None):
        key = ("d3", json.dumps(spec or self.dispersion or D3_DEFAULT, sort_keys=True))
        if key not in self._cache:
            self._cache[key] = d3_calculator(spec or self.dispersion)
        return self._cache[key]

    def calculator(self, frame, energy: str = "e_zero", dispersion: bool = False,
                   delta: bool = True, attempt: int = 0, device: str | None = None, **extra):
        """ASE calculator of the whole model for one structure: DFTB (+ Delta) (+ D3 when
        ``dispersion``; the model's own spec, PBE-D3(BJ) if it has none)."""
        calcs = [self.dftb_calculator(frame, energy, attempt, **extra)]
        if delta and self.delta:
            calcs.append(self.delta_calculator(device))
        if dispersion:
            calcs.append(self.d3())
        if len(calcs) == 1:
            return calcs[0]
        from ase.calculators.mixing import SumCalculator
        return SumCalculator(calcs)

    def atom_energies(self, elements, dispersion: bool = False, delta: bool = True,
                      energy: str = "e_zero") -> dict:
        """Isolated, neutral, spin-unpolarized atoms (20 A box), eV: the model's own zero of the
        atomization energies (D3 adds nothing for one atom)."""
        from ase import Atoms
        out = {}
        for el in elements:
            at = Atoms(el, positions=[[10.0, 10.0, 10.0]], cell=[20.0] * 3, pbc=False)
            err = None
            for attempt in range(3):
                try:
                    at.calc = self.calculator(at, energy, dispersion, delta, attempt)
                    out[el] = float(at.get_potential_energy())
                    err = None
                    break
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {exc}"
            if err:
                raise RuntimeError(f"isolated {el} failed: {err}")
        return out

    def fingerprint(self) -> str:
        """Hash of what determines the predictions (model.json, env models, Delta file, the SK
        file names and sizes): the key of prediction caches."""
        h = hashlib.sha1()
        mj = os.path.join(self.path, MODEL_FILE)
        if os.path.exists(mj):
            h.update(open(mj, "rb").read())
        h.update(json.dumps({"skdir": self.skdir, "repulsive": self.repulsive, "dftb": self.dftb,
                             "delta": self.delta}, sort_keys=True, default=str).encode())
        if self.env_file and os.path.exists(self.env_file):
            h.update(open(self.env_file, "rb").read())
        if self.delta:
            p = _abs(self.path, self.delta["path"])
            if os.path.exists(p):
                st = os.stat(p)
                h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
        if os.path.isdir(self.skdir):
            for fn in sorted(os.listdir(self.skdir)):
                if fn.endswith((".skf", ".skf.gz")):
                    h.update(f"{fn}:{os.path.getsize(os.path.join(self.skdir, fn))}".encode())
        return h.hexdigest()[:16]

    def to_json(self) -> dict:
        return {"format": FORMAT, "skdir": self.skdir, "repulsive": self.repulsive,
                "dftb": self.dftb,
                "env_models": (os.path.relpath(self.env_file, self.path) if self.env_file else None),
                "delta": self.delta, "dispersion": self.dispersion, "meta": self.meta}


def load_model(path: str, dftb: dict | None = None) -> Model:
    """The model of a directory (a-series-model/1, or a ``ptbp scan export`` directory).
    ``dftb`` overrides the DFTB settings key by key (e.g. the scan config's dftb section)."""
    from pydftb.envmodel import load_models
    path = os.path.abspath(path)
    mj = os.path.join(path, MODEL_FILE)
    if not os.path.exists(mj):
        raise FileNotFoundError(f"{mj} not found: not a model directory")
    with open(mj) as fh:
        d = json.load(fh)
    if d.get("format") == FORMAT:
        env_file = _abs(path, d.get("env_models"))
        env = load_models(env_file)[0] if env_file else ()
        settings = dict(DEFAULT_DFTB, **(d.get("dftb") or {}))
        settings.update(dftb or {})
        return Model(path, _abs(path, d["skdir"]), d.get("repulsive", "skf"), settings, tuple(env),
                     env_file, d.get("delta"), d.get("dispersion"), d.get("meta") or {}, FORMAT)
    if "strategy" in d and "scan" in d:           # utils.accuracy.export_model
        settings = dict(DEFAULT_DFTB, max_l=d.get("max_l"))
        settings.update(dftb or {})
        meta = {k: d.get(k) for k in ("scan", "index", "strategy", "repulsive", "radii", "sigma")}
        meta["electronic_skdir"] = d.get("skdir")
        return Model(path, path, "skf", settings, (), None, None, None, meta, "scan-export")
    raise ValueError(f"{mj}: neither {FORMAT} nor a `ptbp scan export` model.json")


def write_model(path: str, skdir: str, dftb: dict, repulsive: str = "skf", env_models=None,
                delta: dict | None = None, dispersion: dict | None = None,
                meta: dict | None = None) -> Model:
    """Write a model directory (model.json, env_models.json if any) and return its Model."""
    from pydftb.envmodel import save_models
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    env_models = tuple(env_models or ())
    env_file = None
    if env_models:
        env_file = os.path.join(path, "env_models.json")
        save_models(env_file, env_models, meta={"written_by": "common/models.py"})
    m = Model(path, os.path.abspath(skdir), repulsive, dict(dftb or {}), env_models, env_file,
              delta, dispersion, dict(meta or {}), FORMAT)
    with open(os.path.join(path, MODEL_FILE) + ".tmp", "w") as fh:
        json.dump(_clean(m.to_json()), fh, indent=1)
    os.replace(os.path.join(path, MODEL_FILE) + ".tmp", os.path.join(path, MODEL_FILE))
    return m


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


# ---- predictions (cached, resumable) -------------------------------------------------------------
_WORK: dict = {}


def _dftb_raw(calc):
    """The DFTBResult behind a model calculator (PyDFTB or a SumCalculator around one)."""
    if hasattr(calc, "raw"):
        return calc.raw
    for c in getattr(getattr(calc, "mixer", None), "calcs", []) or []:
        if hasattr(c, "raw"):
            return c.raw
    return None


def predict_one(model: Model, frame, forces: bool = True, energy: str = "e_zero",
                dispersion: bool = False, delta: bool = True, device: str | None = None) -> dict:
    """{"name", "e", "f", "e_zero", "free", "attempts", "error"} for one structure (eV, eV/A);
    a failed SCC is retried twice with slower mixing, then recorded."""
    err = None
    for attempt in range(3):
        a = frame.copy()
        try:
            a.calc = model.calculator(frame, energy, dispersion, delta, attempt, device)
            e = float(a.get_potential_energy())
            f = np.array(a.get_forces()) if forces else None
            raw = _dftb_raw(a.calc)
            out = {"name": frame.info.get("name"), "e": e,
                   "f": None if f is None else f.tolist(), "attempts": attempt + 1, "error": None}
            if raw is not None:
                out["e_zero_dftb"] = float(raw.e_zero) * HARTREE_EV
                out["free_dftb"] = float(raw.e_force_related) * HARTREE_EV
            return out
        except Exception as exc:          # noqa: BLE001  (retried, then recorded)
            err = f"{type(exc).__name__}: {exc}"
    return {"name": frame.info.get("name"), "e": None, "f": None, "attempts": 3, "error": err}


def _worker(k):
    w = _WORK
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:                   # the BLAS variables of predict() still apply
        return predict_one(w["model"], w["frames"][k], **w["kw"])
    with threadpool_limits(1):
        return predict_one(w["model"], w["frames"][k], **w["kw"])


def predict(model: Model, frames, cache: str | None = None, jobs: int = 1, log=print,
            force: bool = False, **kw) -> list:
    """Predictions of every frame (see :func:`predict_one`), in order.  ``cache`` (a JSONL
    file) keeps finished structures: a rerun computes only the missing ones, and a different
    model (fingerprint) or ``force`` starts it again.  ``jobs`` > 1 forks worker processes on
    Linux (one BLAS thread each); elsewhere the structures run one after the other."""
    names = [f.info.get("name") for f in frames]
    if len(set(names)) != len(names) or None in names:
        raise ValueError("predict needs unique info['name'] on every frame")
    done = {}
    key = {"model": model.fingerprint(), "kw": _clean(kw)}
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)) or ".", exist_ok=True)
        head = cache + ".key.json"
        if os.path.exists(cache) and not force:
            ok = os.path.exists(head) and json.load(open(head)) == key
            if ok:
                with open(cache) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            r = json.loads(line)
                            done[r["name"]] = r
            else:
                log(f"[models] {cache}: other model or settings, recomputed")
                os.replace(cache, cache + ".old")
        elif force and os.path.exists(cache):
            os.replace(cache, cache + ".old")
        with open(head, "w") as fh:
            json.dump(key, fh)
    todo = [k for k, n in enumerate(names) if n not in done]
    if todo:
        log(f"[models] {len(todo)} of {len(frames)} structures to compute ({jobs} process(es))")
    fh = open(cache, "a", buffering=1) if cache else None
    try:
        if jobs > 1 and len(todo) > 1 and sys.platform.startswith("linux"):
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            _WORK.update(model=model, frames=list(frames), kw=kw)
            with cf.ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
                for n, r in enumerate(ex.map(_worker, todo, chunksize=1), 1):
                    done[r["name"]] = r
                    if fh:
                        fh.write(json.dumps(_clean(r)) + "\n")
                    if n % 10 == 0 or n == len(todo):
                        log(f"[models] {n}/{len(todo)} done")
        else:
            for n, k in enumerate(todo, 1):
                r = predict_one(model, frames[k], **kw)
                done[r["name"]] = r
                if fh:
                    fh.write(json.dumps(_clean(r)) + "\n")
                if n % 10 == 0 or n == len(todo):
                    log(f"[models] {n}/{len(todo)} done")
    finally:
        if fh:
            fh.close()
    return [done[n] for n in names]


def records(frames, preds) -> list:
    """pydftb.assess records from frames and :func:`predict` results."""
    from pydftb import assess
    e = [np.nan if p["e"] is None else p["e"] for p in preds]
    f = [None if p["f"] is None else np.asarray(p["f"], float) for p in preds]
    return assess.records_from_arrays(frames, e, f, [p["error"] for p in preds])


def score(model: Model, frames, e0_ref: dict, reactions=None, out_dir: str | None = None,
          title: str = "assessment", dispersion: bool = False, jobs: int = 1, log=print,
          cache: bool = True, force: bool = False, device: str | None = None) -> dict:
    """pydftb.assess summary of a model on DFT frames (tiers on test and ood), with the model's
    own isolated atoms as the zero of its atomization energies.  ``dispersion``: D3 on the model
    and on the reference (the DFT data are plain PBE).  Writes summary.md/summary.json and the
    prediction cache to ``out_dir``."""
    from pydftb import assess
    if dispersion:
        frames = add_d3_to_reference(frames, model.dispersion)
    elements = sorted({s for a in frames for s in a.get_chemical_symbols()})
    e0_pred = model.atom_energies(elements, dispersion=dispersion)
    cache_path = (os.path.join(out_dir, "predictions.jsonl") if (out_dir and cache) else None)
    preds = predict(model, frames, cache_path, jobs, log, force=force, dispersion=dispersion,
                    device=device)
    summ = assess.summarize(records(frames, preds), e0_ref, e0_pred, reactions)
    summ["e0_pred"] = e0_pred
    summ["model"] = {"path": model.path, "fingerprint": model.fingerprint(),
                     "dispersion": bool(dispersion)}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        assess.write_markdown(summ, os.path.join(out_dir, "summary.md"), title=title)
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump(_clean(summ), fh, indent=1)
    return summ


def _fmt1(v, scale: float = 1e3) -> str:
    """eV/atom -> 'x.x' meV/atom (or a dash)."""
    return "—" if v is None or not np.isfinite(v) else f"{v * scale:.1f}"


def brief(summary: dict) -> dict:
    """The parts of an assessment kept in result files (no per-structure lists)."""
    en = summary["energies"]
    return _clean({"by_role": en["by_role"], "by_family": en["by_family"],
                   "offsets": en.get("offsets"), "eos": summary.get("eos"),
                   "reactions": summary.get("reactions"), "tier": summary.get("tier"),
                   "failed": summary.get("failed"), "e0_pred": summary.get("e0_pred")})


# ---- MD ------------------------------------------------------------------------------------------
LI_O_COLLAPSE = {"Li-Li": 2.0, "Li-O": 1.45, "O-O": 1.2}      # A (li_o/md_check.py)


def converging_calculator(model: Model, atoms, energy: str = "free", dispersion: bool = False,
                          device: str | None = None, **extra):
    """(calculator, attempt): the first of the SCC settings 0, 1, 2 (default, slower Broyden,
    Anderson; :func:`engine_options`) whose single point at ``atoms`` converges; MD then keeps
    it.  Raises the last error if none does."""
    err = None
    for attempt in range(3):
        calc = model.calculator(atoms, energy=energy, dispersion=dispersion, attempt=attempt,
                                device=device, **extra)
        a = atoms.copy()
        a.calc = calc
        try:
            a.get_forces()
            return calc, attempt
        except Exception as exc:          # noqa: BLE001
            err = exc
    raise err


def run_nve(model: Model, atoms, temperature: float, langevin: int = 300, nve: int = 2000,
            dt_fs: float = 1.0, trajectory: str | None = None, collapse: dict | None = None,
            dispersion: bool = False, device: str | None = None, seed: int = 1) -> dict:
    """pydftb.assess.nve_check with the model (force-consistent energy, charges reused between
    steps), plus a collapse flag (shortest distance per element pair below ``collapse``)."""
    from pydftb import assess
    try:
        calc, attempt = converging_calculator(model, atoms, "free", dispersion, device,
                                              reuse_charges=True)
    except Exception as exc:              # noqa: BLE001  (recorded as a failed run)
        return {"temperature": temperature, "failed": f"{type(exc).__name__}: {exc}",
                "shortest": {}, "collapse": False, "stable": False, "scc_attempt": None}
    r = assess.nve_check(atoms, calc, temperature=temperature, langevin_steps=langevin,
                         nve_steps=nve, dt_fs=dt_fs, trajectory=trajectory, seed=seed)
    r["scc_attempt"] = attempt
    lim = collapse or {}
    r["collapse"] = bool(any(r["shortest"].get(p, 99.0) < d for p, d in lim.items()))
    r["stable"] = bool(r["failed"] is None and not r["collapse"])
    return r
