"""Confinement-parameter spaces and a cache of SK sets for the A-series scans (A0, section 3).

A1 compares several fitting strategies.  To keep the optimiser from being one of the
differences between them, every strategy's loss is computed on one shared set of points in
(r0_w, r0_d) space: a grid, or a Sobol sequence.  The SK files of each point are generated once
(hotcent, PTBP's settings, :func:`utils.hotcent_tangent.run_hotcent`) and kept in a cache,
keyed by everything that changes their content.  Parallel workers (processes, Slurm array
tasks) share the cache through a lock file per entry.

:class:`Space` holds the elements of the model.  The fitted ones vary (r0_w, r0_d) within
bounds; the others keep fixed radii, PTBP's published values by default.  The confinement
power p is PTBP's (2).  Bounds are ``"ptbp"``, PTBP's optimisation box (r0_w in [2.1, 7 r_cov],
r0_d in [3.4, 9 r_cov] Bohr), ``{"around": [f_lo, f_hi]}`` (a factor box around the published
radii), or explicit ``{"r0_w": [lo, hi], "r0_d": [lo, hi]}``.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import socket
import time
from dataclasses import dataclass, field

import numpy as np

PTBP_BOX = {"r0_w": (2.1, 7.0), "r0_d": (3.4, 9.0)}   # lower bound (Bohr), upper (x r_cov)


def published(element: str) -> dict:
    """PTBP's published parameters of an element (r0_w, r0_d, p, sigma_rep, cov_radius in Bohr)."""
    from utils.parameters_set import get_PTBP
    d = get_PTBP()[element]
    return {k: d[k] for k in ("r0_w", "r0_d", "p", "sigma_rep", "cov_radius")}


def bounds_of(element: str, spec) -> tuple:
    """((r0_w lo, hi), (r0_d lo, hi)) in Bohr for an element and a bounds spec."""
    pub = published(element)
    rc = pub["cov_radius"]
    if spec in (None, "ptbp"):
        return ((PTBP_BOX["r0_w"][0], PTBP_BOX["r0_w"][1] * rc),
                (PTBP_BOX["r0_d"][0], PTBP_BOX["r0_d"][1] * rc))
    if isinstance(spec, dict) and "around" in spec:
        lo, hi = (float(x) for x in spec["around"])
        return ((lo * pub["r0_w"], hi * pub["r0_w"]), (lo * pub["r0_d"], hi * pub["r0_d"]))
    if isinstance(spec, dict) and "r0_w" in spec and "r0_d" in spec:
        return (tuple(float(x) for x in spec["r0_w"]), tuple(float(x) for x in spec["r0_d"]))
    raise ValueError(f"bounds of {element}: 'ptbp', {{'around': [lo, hi]}} or explicit "
                     f"{{'r0_w': [lo, hi], 'r0_d': [lo, hi]}}, got {spec!r}")


@dataclass
class Space:
    """The confinement radii of a model: fitted elements vary, the others are fixed."""
    elements: list                          # every element of the model (SK files of all pairs)
    fit: list                               # elements whose (r0_w, r0_d) vary
    bounds: dict = field(default_factory=dict)    # element -> ((lo, hi), (lo, hi)) Bohr
    fixed: dict = field(default_factory=dict)     # element -> (r0_w, r0_d) Bohr
    p: dict = field(default_factory=dict)         # element -> confinement power

    def __post_init__(self):
        for el in self.elements:
            pub = published(el)
            self.p.setdefault(el, pub["p"])
            if el in self.fit:
                self.bounds.setdefault(el, bounds_of(el, "ptbp"))
            else:
                self.fixed.setdefault(el, (pub["r0_w"], pub["r0_d"]))
        unknown = set(self.fit) - set(self.elements)
        if unknown:
            raise ValueError(f"fitted elements {sorted(unknown)} are not in the model")

    @classmethod
    def from_config(cls, cfg: dict) -> "Space":
        """``cfg``: {elements: [...], fit: [...], bounds: spec or {el: spec}, fixed: {el: [w, d]}}."""
        elements = list(cfg["elements"])
        fit = list(cfg.get("fit", elements))
        spec = cfg.get("bounds", "ptbp")
        per = spec if isinstance(spec, dict) and set(spec) <= set(elements) and spec else None
        bounds = {el: bounds_of(el, (per or {}).get(el, "ptbp") if per else spec) for el in fit}
        fixed = {el: tuple(float(x) for x in v) for el, v in (cfg.get("fixed") or {}).items()}
        p = {el: float(v) for el, v in (cfg.get("p") or {}).items()}
        return cls(elements, fit, bounds, fixed, p)

    @property
    def names(self) -> list:
        return [f"{el}.{k}" for el in self.fit for k in ("r0_w", "r0_d")]

    @property
    def dim(self) -> int:
        return 2 * len(self.fit)

    def lower_upper(self):
        lo = np.array([b[i][0] for el in self.fit for b in [self.bounds[el]] for i in (0, 1)])
        hi = np.array([b[i][1] for el in self.fit for b in [self.bounds[el]] for i in (0, 1)])
        return lo, hi

    def radii(self, x) -> dict:
        """{element: [r0_w, r0_d]} for a parameter vector (fitted elements in order)."""
        x = np.asarray(x, float).reshape(self.dim)
        out = {el: [float(v) for v in self.fixed[el]] for el in self.elements if el not in self.fit}
        for i, el in enumerate(self.fit):
            out[el] = [float(x[2 * i]), float(x[2 * i + 1])]
        return {el: out[el] for el in self.elements}

    def vector(self, radii: dict) -> np.ndarray:
        return np.array([radii[el][i] for el in self.fit for i in (0, 1)], float)

    def published_vector(self) -> np.ndarray:
        return np.array([published(el)[k] for el in self.fit for k in ("r0_w", "r0_d")], float)

    def points(self, kind: str = "sobol", n: int = 64, seed: int = 0,
               include_published: bool = True) -> np.ndarray:
        """Points of the space: ``"grid"`` (``n`` per dimension, n^dim points), ``"sobol"``
        (scrambled, ``n`` points; a power of 2 keeps its balance), ``"random"`` (uniform).
        ``include_published`` puts PTBP's published radii first when they are in the box."""
        lo, hi = self.lower_upper()
        if kind == "grid":
            axes = [np.linspace(a, b, int(n)) for a, b in zip(lo, hi)]
            pts = np.array(np.meshgrid(*axes, indexing="ij")).reshape(self.dim, -1).T
        elif kind == "sobol":
            from scipy.stats import qmc
            u = qmc.Sobol(self.dim, scramble=True, seed=seed).random(int(n))
            pts = qmc.scale(u, lo, hi)
        elif kind == "random":
            pts = np.random.default_rng(seed).uniform(lo, hi, size=(int(n), self.dim))
        else:
            raise ValueError(f"kind must be grid, sobol or random, got {kind!r}")
        if include_published:
            pub = self.published_vector()
            if np.all(pub >= lo) and np.all(pub <= hi):
                pts = np.vstack([pub, pts])
        return pts


# ---- the SK cache ---------------------------------------------------------------------------
def _hotcent_version() -> str:
    try:
        import hotcent
        return getattr(hotcent, "__version__", "unknown")
    except ImportError:
        return "missing"


@dataclass
class SKSettings:
    """What besides the radii changes the SK files."""
    xc: str = "GGA_X_PBE+GGA_C_PBE"
    superposition: str = "density"
    label: str = "ptbp"
    confinement: str = "selfconsistent"
    grid: dict = field(default_factory=dict)   # hotcent integration grid {"nr", "ntheta"}
    workers: int = 1                           # hotcent processes per SK set

    @classmethod
    def from_config(cls, cfg: dict | None) -> "SKSettings":
        cfg = dict(cfg or {})
        cfg.pop("cache", None)
        return cls(**cfg)


class SKCache:
    """SK sets on disk, one directory per (radii, powers, settings), generated on demand.

    ``root/<key>/`` holds the ``*.skf`` files and ``meta.json`` (radii, settings, hotcent
    version, host, seconds).  A worker that finds ``<key>.lock`` waits for the owner to finish.
    A lock older than ``stale`` seconds is taken over (a killed job)."""

    def __init__(self, root: str, settings: SKSettings, stale: float = 4 * 3600.0):
        self.root = os.path.abspath(root)
        self.settings = settings
        self.stale = stale
        os.makedirs(self.root, exist_ok=True)

    def key(self, radii: dict, p: dict) -> str:
        s = self.settings
        payload = {"radii": {el: [round(float(v), 6) for v in radii[el]] for el in sorted(radii)},
                   "p": {el: round(float(p[el]), 6) for el in sorted(p)}, "xc": s.xc,
                   "superposition": s.superposition, "label": s.label,
                   "confinement": s.confinement, "grid": dict(sorted(s.grid.items())),
                   "hotcent": _hotcent_version()}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def path(self, radii: dict, p: dict) -> str:
        return os.path.join(self.root, self.key(radii, p))

    def has(self, radii: dict, p: dict) -> bool:
        return os.path.exists(os.path.join(self.path(radii, p), "meta.json"))

    def get(self, radii: dict, p: dict, log=print) -> str:
        """The directory of this SK set; generated (and cached) if missing."""
        final = self.path(radii, p)
        if os.path.exists(os.path.join(final, "meta.json")):
            return final
        lock = final + ".lock"
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.time()}\n".encode())
                os.close(fd)
                break
            except FileExistsError:
                if os.path.exists(os.path.join(final, "meta.json")):
                    return final
                try:
                    age = time.time() - os.path.getmtime(lock)
                except FileNotFoundError:
                    continue
                if age > self.stale:
                    log(f"[skcache] taking over a stale lock ({age:.0f} s): {lock}")
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(lock)
                    continue
                time.sleep(2.0)
        try:
            if os.path.exists(os.path.join(final, "meta.json")):
                return final
            tmp = f"{final}.tmp-{socket.gethostname()}-{os.getpid()}"
            shutil.rmtree(tmp, ignore_errors=True)
            seconds = generate(radii, p, self.settings, tmp)
            missing = [f"{a}-{b}.skf" for a in radii for b in radii
                       if not os.path.exists(os.path.join(tmp, f"{a}-{b}.skf"))]
            if missing:
                raise RuntimeError(f"hotcent wrote no {missing} for radii {radii}")
            meta = {"radii": radii, "p": p, "settings": self.settings.__dict__,
                    "hotcent": _hotcent_version(), "host": socket.gethostname(),
                    "seconds": round(seconds, 2), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(os.path.join(tmp, "meta.json"), "w") as fh:
                json.dump(meta, fh, indent=1)
            shutil.rmtree(final, ignore_errors=True)
            os.replace(tmp, final)
            log(f"[skcache] generated {os.path.basename(final)} in {seconds:.1f} s")
            return final
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(lock)


def generate(radii: dict, p: dict, settings: SKSettings, outdir: str) -> float:
    """Every SK file of the elements of ``radii`` into ``outdir`` (hotcent, PTBP's driver);
    returns the wall time.  The integration grid applies to this call only."""
    from utils import hotpy as hp_mod
    from utils.hotcent_tangent import run_hotcent
    try:
        from utils import hotcent_parallel
    except ImportError:                   # hotcent < 2: the reference driver only
        hotcent_parallel = None
    old_conf = hp_mod.confinement()
    old_grid = hotcent_parallel.grid() if hotcent_parallel else {}
    hp_mod.set_confinement(settings.confinement)
    if hotcent_parallel:
        hotcent_parallel.set_grid(**settings.grid)
    elif settings.grid:
        raise RuntimeError("a hotcent integration grid needs utils.hotcent_parallel (hotcent >= 2)")
    try:
        conf = ({el: [float(v) for v in radii[el]] for el in radii},
                {el: float(p[el]) for el in radii})
        hp_settings = {"xc": settings.xc, "superposition": settings.superposition,
                       "label": settings.label, "opt_elem": []}
        with contextlib.redirect_stdout(io.StringIO()):
            return run_hotcent(hp_settings, conf, outdir, workers=max(1, int(settings.workers)))
    finally:
        hp_mod.set_confinement(old_conf)
        if hotcent_parallel:
            hotcent_parallel.set_grid(**old_grid)
