"""Tangent of the SKF parameters with respect to the confinement radii (milestone M4c).

hotcent cannot be differentiated, so the tangent t = d theta / d r0 of the parameters a
pydftb calculation reads from the SKF files (SK tables, onsite energies, Hubbard U) is a
central difference of the files hotcent writes at r0 +- h, with exactly the settings of
the BO step (:meth:`utils.loss.Loss.set_parameters`, the run's hotcent workers).  Its
arrays have the keys and shapes of :class:`pydftb.autodiff.Params`, so that a gradient
from ``pydftb.autodiff`` contracts with it directly.

hotcent 2.1 solves the density-confined atom perturbatively and keeps, for every subshell
that has a wave-function confinement (hotpy confines them all), the wave-function-confined
orbital: r0_d then changes no byte of the SKF files.  :func:`compute` probes r0_d once per
run (one hotcent run at r0_d + h_d compared byte by byte with the base files) and uses a
zero tangent while that holds; otherwise it takes central differences as for r0_w.

Step study (M4c design, section 5): the r0_w tangent of Cu changes by 0.3-1 % (rms)
between neighbouring steps of 0.005-0.32 Bohr, the default 0.04 Bohr is in the flat part.
"""
from __future__ import annotations

import contextlib
import filecmp
import glob
import io
import os
import shutil
import time
from dataclasses import dataclass, field

import numpy as np

DEFAULT_STEPS = (0.04, 0.2)          # Bohr: r0_w, r0_d
KINDS = ("tables", "onsite", "hubbard")
_PROBES: dict = {}                    # r0_d probe outcome per run setting (see compute)


@dataclass
class Tangent:
    """d theta / d r0 along one confinement radius."""
    direction: str                    # "r0_w" | "r0_d"
    step: float                       # Bohr (0 for a tangent known to vanish)
    tables: dict = field(default_factory=dict)    # (a, b) -> d table (stored columns)
    onsite: dict = field(default_factory=dict)    # label -> d onsite (per shell)
    hubbard: dict = field(default_factory=dict)   # label -> d U (per shell)
    zero: bool = False                # every SKF byte-identical: exactly zero
    probed: bool = False              # zero from the r0_d probe (or its cached outcome)
    seconds: float = 0.0              # hotcent time spent on this direction

    def kinds(self) -> tuple:
        """Parameter kinds with a non-zero tangent."""
        if self.zero:
            return ()
        return tuple(k for k in KINDS if any(np.any(v != 0) for v in getattr(self, k).values()))

    def arrays(self, kind: str) -> dict:
        return getattr(self, kind)


def skf_parameters(skdir: str, labels) -> dict:
    """{kind: {key: array}} of the SKF files in ``skdir`` as pydftb reads them
    (``ParamSet(..., shell_resolved=True)``, PTBP's setting; the tables are the stored
    columns, as :class:`pydftb.autodiff.Params` holds them)."""
    from pydftb.params import ParamSet
    sk = ParamSet(skdir, shell_resolved=True).skset(list(labels))
    return {"tables": {k: np.array(f.table, dtype=float) for k, f in sk.files.items()},
            "onsite": {a: np.array(v, dtype=float) for a, v in sk.onsite.items()},
            "hubbard": {a: np.array(v, dtype=float) for a, v in sk.hubbard.items()},
            "repulsive": {k: (_repulsive_state(f.repulsive), _repulsive_state(f.poly_repulsive))
                          for k, f in sk.files.items()}}


def _repulsive_state(rep):
    """The numbers of a :class:`pydftb.skf.Repulsive` (None stays None), for comparison."""
    if rep is None:
        return None
    return tuple(None if v is None else np.asarray(v, dtype=float).tobytes()
                 for v in (rep.cutoff, rep.exp_coeffs, rep.knots, rep.coeffs, rep.poly))


def same_files(a: str, b: str) -> bool:
    """Every SKF file of directory ``a`` is byte-identical in ``b`` (and vice versa)."""
    na = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a, "*.skf")))
    nb = sorted(os.path.basename(p) for p in glob.glob(os.path.join(b, "*.skf")))
    if not na or na != nb:
        return False
    return all(filecmp.cmp(os.path.join(a, n), os.path.join(b, n), shallow=False) for n in na)


def run_hotcent(hp_settings: dict, confinement, outdir: str, workers: int = 1) -> float:
    """hotcent for ``confinement = (confinement_r0, confinement_p)`` into ``outdir`` as the
    BO step runs it (sharded rows, or the reference implementation where the step uses
    it); returns the wall time."""
    from utils import hotcent_parallel
    from utils.hotpy import hotpy
    os.makedirs(outdir, exist_ok=True)
    hp = hotpy()
    hp.set_variables(confinement[0], confinement[1], xc=hp_settings["xc"],
                     superposition=hp_settings["superposition"])
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        if os.environ.get("PTBP_HOTCENT_REFERENCE") or not hotcent_parallel.sharding_supported():
            cwd = os.getcwd()
            os.chdir(outdir)
            try:
                hp.full_hotcent_reference(superposition=hp_settings["superposition"],
                                          opt_elem=hp_settings.get("opt_elem", []),
                                          label=hp_settings.get("label", "ptbp"))
            finally:
                os.chdir(cwd)
        else:
            hotcent_parallel.generate_all(hp, hp_settings["superposition"],
                                          hp_settings.get("opt_elem", []),
                                          hp_settings.get("label", "ptbp"), workers=workers,
                                          outdir=outdir)
    return time.perf_counter() - t0


def _difference(plus: dict, minus: dict, h: float, direction: str) -> dict:
    out = {}
    for kind in KINDS:
        out[kind] = {}
        for key, p in plus[kind].items():
            m = minus[kind][key]
            if p.shape != m.shape:
                raise ValueError(f"{direction}: the SKF grid of {key} changed with r0 "
                                 f"({m.shape} -> {p.shape}); no tangent")
            out[kind][key] = (p - m) / (2.0 * h)
    for key, rep in plus["repulsive"].items():
        if rep != minus["repulsive"][key]:
            raise ValueError(f"{direction}: the repulsive part of the SKF file {key} depends "
                             "on r0; only the tables, onsite energies and U have a tangent")
    return out


def compute(confinement, hp_settings: dict, base_dir: str, labels, r0_w: float, r0_d: float,
            steps=DEFAULT_STEPS, workers: int = 1, work_dir: str | None = None,
            keep: bool = False, probe_key=None) -> dict:
    """Tangents {"r0_w": Tangent, "r0_d": Tangent} at (r0_w, r0_d).

    ``confinement(r0_w, r0_d)`` -> (confinement_r0, confinement_p) as the BO step sets
    hotpy; ``hp_settings``: xc, superposition, opt_elem, label; ``base_dir``: the SKF
    files of the step itself (written at (r0_w, r0_d)); ``labels``: the elements of the
    dataset.  The perturbed files go to ``work_dir`` (default ``<base_dir>/tangent``),
    removed afterwards unless ``keep``.  ``probe_key``: identifies the run setting for the
    r0_d probe (None: probe every time).
    """
    h_w, h_d = (float(s) for s in steps)
    work = work_dir or os.path.join(base_dir, "tangent")
    os.makedirs(work, exist_ok=True)
    out = {}
    try:
        # r0_w: central differences
        t0 = time.perf_counter()
        pw, mw = os.path.join(work, "r0_w_plus"), os.path.join(work, "r0_w_minus")
        run_hotcent(hp_settings, confinement(r0_w + h_w, r0_d), pw, workers)
        run_hotcent(hp_settings, confinement(r0_w - h_w, r0_d), mw, workers)
        if same_files(pw, mw):
            out["r0_w"] = Tangent("r0_w", h_w, zero=True)
        else:
            diff = _difference(skf_parameters(pw, labels), skf_parameters(mw, labels), h_w, "r0_w")
            out["r0_w"] = Tangent("r0_w", h_w, **diff)
        out["r0_w"].seconds = time.perf_counter() - t0

        # r0_d: probe once per run setting, then central differences if it matters
        t0 = time.perf_counter()
        if probe_key is not None and _PROBES.get(probe_key) is True:
            out["r0_d"] = Tangent("r0_d", 0.0, zero=True, probed=True)
        else:
            pd = os.path.join(work, "r0_d_plus")
            run_hotcent(hp_settings, confinement(r0_w, r0_d + h_d), pd, workers)
            if same_files(base_dir, pd):
                out["r0_d"] = Tangent("r0_d", 0.0, zero=True, probed=True)
                if probe_key is not None:
                    _PROBES[probe_key] = True
            else:
                if probe_key is not None:
                    _PROBES[probe_key] = False
                md = os.path.join(work, "r0_d_minus")
                run_hotcent(hp_settings, confinement(r0_w, r0_d - h_d), md, workers)
                diff = _difference(skf_parameters(pd, labels), skf_parameters(md, labels), h_d,
                                   "r0_d")
                out["r0_d"] = Tangent("r0_d", h_d, **diff)
        out["r0_d"].seconds = time.perf_counter() - t0
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
    return out


def contract(grads: dict, tangent: Tangent) -> float:
    """<gradient, tangent> for ``grads`` = {kind: {key: array}} (missing keys: zero)."""
    if tangent.zero:
        return 0.0
    total = 0.0
    for kind in KINDS:
        tk = tangent.arrays(kind)
        for key, g in grads.get(kind, {}).items():
            t = tk.get(key)
            if t is not None and g is not None:
                total += float(np.sum(np.asarray(g) * t))
    return total
