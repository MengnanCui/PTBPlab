"""A-series accuracy scans: one SK set per point, every fitting strategy on the same points
(design ``pydftb/docs/design/A0-accuracy-programme.html``, section 3).

Stages (``ptbp scan``, :mod:`cli.cmd_scan`):

1. ``points`` writes the parameter points of the config's space (grid or Sobol, PTBP's
   published radii first);
2. ``run`` evaluates the electronic part at each point.  It generates the SK set (cached),
   then computes with ``repulsive="none"`` the DFTB energies and forces of every structure,
   the isolated atoms and the band references; results go to ``points/pNNNNN/``.  Points run
   in parallel processes, or one per Slurm array task (``--index``/``--stride``), and a
   finished point is skipped on a rerun;
3. ``losses`` scores every point.  physrep's σ (one per fitted element) is searched for each
   strategy, and the R₂b control and the oracle (:mod:`pydftb.repfit`) are fitted.  Output:
   ``losses.csv`` and ``points/pNNNNN/losses.json``;
4. ``report`` picks the best point of each strategy and writes ``report.md`` and
   ``report.json`` (metrics on train/test/ood, EOS, dimers, reactions, tier, Spearman of band
   loss vs R₂b), plus loss maps for a 2-D space;
5. ``refine`` locally optimises one strategy (Nelder–Mead on the radii from its best point);
6. ``export`` writes a model's full SK files (electronic tables + physrep or R₂b spline).

Strategies (A0 section 3).  ``L_EF`` is the mean over groups of
[mean |ΔE_atomization|/eps_e + lambda_f · mean RMS(ΔF)/eps_f] (eps 10 meV/atom and 0.1 eV/Å,
the acceptance line, so L_EF = 2 means "on the line" for both); the groups are the families:

* ``S1``: min over σ of L_EF on train (electronic and repulsive together);
* ``S1o``: S1 with per-element energy offsets fitted on train (control of the absolute reference);
* ``S1eos``: S1 on the EOS families only (separates the data from the loss in S2);
* ``S2``: PTBP's energygeometry μ = 0.5 μ_V + 0.5 μ_E on the EOS families (3000 K Boltzmann
  weights within and between groups), min over σ;
* ``S3``: the band loss decides the electronic part, then σ by L_EF.  At a given point its
  model is S1's; only the choice of point differs;
* ``S4``: L_EF + lambda_band · L_band.

PTBP's r0 penalty is computed and reported; it enters the losses only with ``loss.penalty: true``.
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import asdict

import numpy as np

from utils.skgrid import SKCache, SKSettings, Space, published

KB_EV = 8.617333262e-5
HARTREE_EV = 27.211386245988
STRATEGIES = ("S1", "S1o", "S1eos", "S2", "S3", "S4")


# ---- configuration ------------------------------------------------------------------------
def load_config(path: str) -> dict:
    """The scan config (YAML or JSON), with paths made absolute against its directory and the
    defaults filled in."""
    path = os.path.abspath(path)
    with open(path) as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    base = os.path.dirname(path)

    def absolute(p):
        p = os.path.expandvars(os.path.expanduser(str(p)))
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    cfg["_path"] = path
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    cfg["out"] = absolute(cfg.get("out", os.path.join("runs", cfg["name"])))
    sk = cfg.setdefault("sk", {})
    sk["cache"] = absolute(os.environ.get("PTBP_SKCACHE") or sk.get("cache", "skcache"))
    data = cfg.setdefault("data", {})
    for role in ("train", "val", "test", "ood"):
        data[role] = [absolute(p) for p in data.get(role, []) or []]
    data["e0s"] = absolute(data["e0s"]) if data.get("e0s") else None
    bands = cfg.setdefault("bands", {})
    bands["refs"] = [absolute(p) for p in bands.get("refs", []) or []]
    if cfg.get("reactions"):
        cfg["reactions"] = absolute(cfg["reactions"])
    cfg.setdefault("dftb", {})
    cfg.setdefault("physrep", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("repfit", {})
    cfg.setdefault("strategies", list(STRATEGIES))
    return cfg


def space_of(cfg) -> Space:
    return Space.from_config(cfg["space"])


def cache_of(cfg) -> SKCache:
    return SKCache(cfg["sk"]["cache"], SKSettings.from_config(cfg["sk"]))


def point_dir(cfg, index: int) -> str:
    return os.path.join(cfg["out"], "points", f"p{int(index):05d}")


def read_points(cfg) -> np.ndarray:
    with open(os.path.join(cfg["out"], "points.json")) as fh:
        return np.array(json.load(fh)["points"], float)


def write_points(cfg, log=print) -> np.ndarray:
    """points.json from the config's space ({kind, n, seed, include_published})."""
    sp = space_of(cfg)
    s = cfg["space"]
    pts = sp.points(s.get("kind", "sobol"), int(s.get("n", 64)), int(s.get("seed", 0)),
                    bool(s.get("include_published", True)))
    os.makedirs(cfg["out"], exist_ok=True)
    out = os.path.join(cfg["out"], "points.json")
    with open(out, "w") as fh:
        json.dump({"names": sp.names, "points": pts.tolist(), "space": cfg["space"]}, fh, indent=1)
    log(f"[scan] {len(pts)} points ({sp.dim}-D: {', '.join(sp.names)}) -> {out}")
    return pts


# ---- data -------------------------------------------------------------------------------
def load_data(cfg) -> list:
    """The frames of every role (``info['role']`` set from the config lists)."""
    import ase.io
    frames = []
    for role in ("train", "val", "test", "ood"):
        for path in cfg["data"][role]:
            for at in ase.io.read(path, ":"):
                at.info["role"] = role
                at.info.setdefault("name", at.info.get("structure_name"))
                at.info.setdefault("family", at.info.get("group", at.info["name"]))
                frames.append(at)
    names = [a.info["name"] for a in frames]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"structure names must be unique, repeated: {sorted(dup)[:10]}")
    return frames


def load_e0s(cfg) -> dict:
    """DFT isolated-atom energies {symbol: eV} (keys may be atomic numbers)."""
    from ase.data import chemical_symbols
    if not cfg["data"]["e0s"]:
        raise ValueError("data.e0s (DFT isolated atoms) is required: the atomization energies "
                         "compare DFT and DFTB atoms of the same kind (spin-unpolarized)")
    with open(cfg["data"]["e0s"]) as fh:
        raw = json.load(fh)
    return {(chemical_symbols[int(k)] if str(k).isdigit() else k): float(v) for k, v in raw.items()}


def load_refs(cfg) -> list:
    from pydftb.bandref import load_refs as _load
    return _load(cfg["bands"]["refs"]) if cfg["bands"]["refs"] else []


# ---- the DFTB electronic part ------------------------------------------------------------
def kpoints_for(frame, default_spacing):
    """The DFTB k-points of a structure, Gamma-centred like the DFT's:

    * ``info['kpts']`` (e.g. [4, 4, 1] or "4x4x1"), the DFT's own grid when it is known;
    * otherwise VASP's KSPACING rule N_i = max(1, ceil(2 pi |b_i| / KSPACING)), with
      ``info['kspacing']`` in 1/A or else the config's default;
    * ``"gamma"``, 0 or None: the Gamma point only.

    Check the grids against the DFT runs (OUTCAR "generate k-points for") in K1: a cell with
    2 pi |b| / KSPACING just above an integer is where codes may round differently."""
    if not np.any(frame.pbc):
        return None
    kp = frame.info.get("kpts")
    if kp is not None and not (isinstance(kp, str) and kp.lower() == "gamma"):
        n = [int(v) for v in (kp.lower().split("x") if isinstance(kp, str) else kp)]
        return {"size": n, "gamma": True}
    ks = frame.info.get("kspacing", default_spacing)
    if ks in (None, "gamma", "Gamma", 0, 0.0):
        return None
    rec = frame.cell.reciprocal()
    n = [max(1, int(np.ceil(2 * np.pi * np.linalg.norm(b) / float(ks) - 1e-9))) for b in rec]
    return {"size": n, "gamma": True}


def _threads(dcfg: dict):
    """BLAS threads per calculation (pydftb's threadpoolctl limit), None without threadpoolctl."""
    import importlib.util
    n = dcfg.get("threads", 1)
    if n is None or importlib.util.find_spec("threadpoolctl") is None:
        return None
    return int(n)


def calculator(skdir, dcfg: dict, frame, attempt: int = 0):
    """PyDFTB for one structure, electronic part only (``repulsive="none"``).
    ``attempt`` 1 and 2 are the slower-mixing retries of a failed SCC."""
    from pydftb.ase_calc import PyDFTB
    kind, width = dcfg.get("smearing", ["gaussian", 0.05])
    temperature = float(width) / KB_EV if kind != "fermi_K" else float(width)
    kind = {"fermi_K": "fermi"}.get(kind, kind)
    opts = dict(max_l=dcfg.get("max_l"), kpts=kpoints_for(frame, dcfg.get("kspacing", 0.15)),
                smearing=(kind, temperature), repulsive="none",
                scc_tol=float(dcfg.get("scc_tol", 1e-6)), max_scc=int(dcfg.get("max_scc", 300)),
                mixer=dcfg.get("mixer", "broyden"), mixing=dcfg.get("mixing"),
                threads=_threads(dcfg), convergent_scc_only=True)
    if attempt == 1:
        opts.update(mixing=0.05, max_scc=1000)
    elif attempt >= 2:
        opts.update(mixer="anderson", mixing=0.02, max_scc=2000)
    if dcfg.get("third_order"):
        opts["third_order"] = dcfg["third_order"]
        opts["hubbard_derivs"] = dcfg.get("hubbard_derivs")
    if dcfg.get("hubbard"):
        opts["hubbard"] = dcfg["hubbard"]
    if dcfg.get("shell_resolved") is not None:
        opts["shell_resolved"] = bool(dcfg["shell_resolved"])
    return PyDFTB(skdir, **opts)


def electronic(skdir, dcfg: dict, frame, forces: bool = True):
    """(E(sigma->0) eV, forces eV/A, attempts, error) of one structure; SCC failures are
    retried twice with slower mixing, then recorded."""
    err = None
    for attempt in range(3):
        a = frame.copy()
        try:
            a.calc = calculator(skdir, dcfg, frame, attempt)
            a.get_potential_energy()
            e = float(a.calc.raw.e_zero) * HARTREE_EV
            f = np.array(a.get_forces()) if forces else None
            return e, f, attempt + 1, None
        except Exception as exc:             # noqa: BLE001  (retried, then recorded)
            err = f"{type(exc).__name__}: {exc}"
    return np.nan, None, 3, err


def atom_energies(skdir, dcfg: dict, elements) -> dict:
    """DFTB energies (eV) of the neutral, spin-unpolarized isolated atoms (20 A box)."""
    from ase import Atoms
    out = {}
    for el in elements:
        at = Atoms(el, positions=[[10.0, 10.0, 10.0]], cell=[20.0] * 3, pbc=False)
        e, _, _, err = electronic(skdir, dcfg, at, forces=False)
        if err:
            raise RuntimeError(f"isolated {el} failed: {err}")
        out[el] = e
    return out


def evaluate_point(cfg, index: int, x=None, force: bool = False, log=print) -> dict:
    """Stage ``run`` for one point: SK set, electronic energies and forces, atoms, bands."""
    from pydftb.bandloss import BandSettings, compare, dftb_bands
    sp = space_of(cfg)
    if x is None:
        x = read_points(cfg)[index]
    out_dir = point_dir(cfg, index)
    done = os.path.join(out_dir, "result.json")
    if os.path.exists(done) and not force:
        with open(done) as fh:
            cached = json.load(fh)
        if np.allclose(cached["x"], np.asarray(x, float), rtol=0, atol=1e-9):
            return cached
        log(f"[scan] point {index}: cached at other radii, recomputed")
        force = True
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    radii = sp.radii(x)
    cache = cache_of(cfg)
    skdir = cache.get(radii, sp.p, log=log)
    t_sk = time.perf_counter() - t0
    frames = _data_cache(cfg)
    dcfg = cfg["dftb"]
    elements = sorted({s for a in frames for s in a.get_chemical_symbols()} |
                      {s for r in _refs_cache(cfg) for s in (r.atoms().get_chemical_symbols()
                                                             if r.numbers is not None else [])})
    e0 = atom_energies(skdir, dcfg, elements)
    energies, forces, attempts, errors = [], [], [], []
    for a in frames:
        e, f, n, err = electronic(skdir, dcfg, a)
        energies.append(e)
        forces.append(np.zeros((len(a), 3)) if f is None else f)
        attempts.append(n)
        errors.append(err)
    t_el = time.perf_counter() - t0 - t_sk
    band_results = []
    bset = BandSettings.from_dict(cfg["bands"].get("settings"))
    for ref in _refs_cache(cfg):
        at = ref.atoms()
        at.info["kspacing"] = ref.meta.get("kspacing", dcfg.get("kspacing", 0.15))
        try:
            calc = calculator(skdir, dcfg, at)
            at.calc = calc
            res = compare(ref, dftb_bands(calc, at, ref), bset,
                          group=ref.meta.get("group") or ref.name)
        except Exception as exc:              # noqa: BLE001
            res = {"name": ref.name, "group": ref.meta.get("group") or ref.name, "loss": np.nan,
                   "error": f"{type(exc).__name__}: {exc}"}
        band_results.append(res)
    t_b = time.perf_counter() - t0 - t_sk - t_el
    np.savez_compressed(os.path.join(out_dir, "electronic.npz"), energies=np.array(energies),
                        forces=np.concatenate([f.ravel() for f in forces]) if forces else np.zeros(0),
                        natoms=np.array([len(a) for a in frames]))
    result = {"index": int(index), "x": [float(v) for v in x], "radii": radii,
              "sk_key": os.path.basename(skdir), "skdir": skdir, "e0_dftb": e0,
              "names": [a.info["name"] for a in frames], "attempts": attempts,
              "errors": errors, "n_failed": int(sum(e is not None for e in errors)),
              "bands": band_results, "seconds": {"sk": round(t_sk, 2), "electronic": round(t_el, 2),
                                                 "bands": round(t_b, 2)}}
    with open(os.path.join(out_dir, "result.json.tmp"), "w") as fh:
        json.dump(_clean(result), fh, indent=1)
    os.replace(os.path.join(out_dir, "result.json.tmp"), done)
    log(f"[scan] point {index}: sk {t_sk:.0f} s, electronic {t_el:.0f} s, bands {t_b:.0f} s, "
        f"{result['n_failed']} failed")
    return result


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


_DATA: dict = {}


def _data_cache(cfg):
    key = cfg["_path"]
    if key not in _DATA:
        _DATA[key] = load_data(cfg)
    return _DATA[key]


_REFS: dict = {}


def _refs_cache(cfg):
    key = cfg["_path"]
    if key not in _REFS:
        _REFS[key] = load_refs(cfg)
    return _REFS[key]


def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:                         # noqa: BLE001
        pass


def _run_one(args):
    cfg_path, index, force = args
    cfg = load_config(cfg_path)
    try:
        r = evaluate_point(cfg, index, force=force)
        return index, r.get("n_failed", 0), None
    except Exception as exc:                  # noqa: BLE001
        tb = traceback.format_exc()
        os.makedirs(point_dir(cfg, index), exist_ok=True)
        with open(os.path.join(point_dir(cfg, index), "error.txt"), "w") as fh:
            fh.write(tb)
        return index, None, f"{type(exc).__name__}: {exc}"


def run_points(cfg, indices, jobs: int = 1, force: bool = False, log=print) -> dict:
    """Stage ``run`` for these points, ``jobs`` processes (fork on Linux, spawn elsewhere)."""
    todo = [int(i) for i in indices
            if force or not os.path.exists(os.path.join(point_dir(cfg, i), "result.json"))]
    log(f"[scan] run: {len(todo)} of {len(indices)} points to do, {jobs} process(es)")
    status = {}
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        if jobs > 1:                          # inherited by the workers before numpy loads
            os.environ.setdefault(var, "1")
    if jobs <= 1:
        for i in todo:
            status[i] = _run_one((cfg["_path"], i, force))
            log(f"[scan] point {i}: {'error ' + status[i][2] if status[i][2] else 'ok'}")
        return status
    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=_worker_init) as ex:
        futs = {ex.submit(_run_one, (cfg["_path"], i, force)): i for i in todo}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            i = futs[fut]
            status[i] = fut.result()
            log(f"[scan] {n}/{len(todo)} point {i}: "
                f"{'ERROR ' + status[i][2] if status[i][2] else 'ok, failed structures ' + str(status[i][1])}")
    return status


# ---- losses ------------------------------------------------------------------------------
def _point_records(cfg, index):
    """(frames, electronic energies, forces, result) of a finished point."""
    res = json.load(open(os.path.join(point_dir(cfg, index), "result.json")))
    z = np.load(os.path.join(point_dir(cfg, index), "electronic.npz"))
    frames = _data_cache(cfg)
    if res["names"] != [a.info["name"] for a in frames]:
        raise RuntimeError(f"point {index}: the data changed since it was run (rerun with --force)")
    e = np.array([np.nan if v is None else v for v in z["energies"]], float)
    f, off = [], 0
    for n in z["natoms"]:
        f.append(z["forces"][off:off + 3 * n].reshape(n, 3))
        off += 3 * n
    return frames, e, f, res


class _Physrep:
    """physrep energies and forces of the frames for a sigma vector (cached by sigma)."""

    def __init__(self, frames, fit_elements, fixed_sigma: dict):
        from ase.data import atomic_numbers
        self.frames = frames
        self.elements = sorted({s for a in frames for s in a.get_chemical_symbols()},
                               key=lambda s: atomic_numbers[s])
        self.z = [atomic_numbers[s] for s in self.elements]
        self.fit = [e for e in fit_elements if e in self.elements]
        self.fixed = {e: float(fixed_sigma.get(e, published(e)["sigma_rep"]))
                      for e in self.elements if e not in self.fit}
        self._cache = {}

    def sigma_dict(self, s):
        d = dict(self.fixed)
        d.update({e: float(v) for e, v in zip(self.fit, np.atleast_1d(s))})
        return d

    def model(self, s):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tools.physrep import physrep
        sd = self.sigma_dict(s)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return physrep(elements=self.z, scale_cov=[sd[e] for e in self.elements],
                           k_xc=[0.0] * len(self.z))

    def evaluate(self, s, idx):
        key = tuple(np.round(np.atleast_1d(s), 10))
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            pr = self.model(s)
            self._cache[key] = (pr, {})
        pr, done = self._cache[key]
        e, f = np.zeros(len(idx)), []
        for k, i in enumerate(idx):
            if i not in done:
                done[i] = pr.evaluate_pbc(self.frames[i])
            e[k] = done[i][0]
            f.append(done[i][1])
        return e, f


def _groups(frames, idx):
    g = {}
    for i in idx:
        g.setdefault(frames[i].info["family"], []).append(i)
    return g


def _atomization(frames, e_tot, e0):
    return np.array([(e - sum(e0[s] for s in a.get_chemical_symbols())) / len(a)
                     for a, e in zip(frames, e_tot)])


def loss_ef(frames, idx, e_pred, f_pred, e_ref, f_ref, e0_ref, e0_pred, lcfg, offsets=None):
    """L_EF on the structures ``idx`` (see the module docstring); NaN if one failed."""
    eps_e, eps_f = float(lcfg.get("eps_e", 0.010)), float(lcfg.get("eps_f", 0.10))
    lam = float(lcfg.get("lambda_f", 1.0))
    sub = [frames[i] for i in idx]
    da = (_atomization(sub, [e_pred[i] for i in idx], e0_pred)
          - _atomization(sub, [e_ref[i] for i in idx], e0_ref))
    if offsets is not None:
        da = da - np.array([sum(offsets.get(s, 0.0) for s in a.get_chemical_symbols()) / len(a)
                            for a in sub])
    if not np.all(np.isfinite(da)):
        return np.nan
    pos = {i: k for k, i in enumerate(idx)}
    terms = []
    for fam, members in _groups(frames, idx).items():
        te = np.mean([abs(da[pos[i]]) for i in members]) / eps_e
        fr = [np.sqrt(np.mean((f_pred[i] - f_ref[i]) ** 2)) for i in members
              if f_ref[i] is not None and len(frames[i]) > 1]
        tf = lam * np.mean(fr) / eps_f if fr else 0.0
        terms.append(te + tf)
    return float(np.mean(terms))


def per_element_offsets(frames, idx, e_pred, e_ref, e0_ref, e0_pred):
    sub = [frames[i] for i in idx]
    da = (_atomization(sub, [e_pred[i] for i in idx], e0_pred)
          - _atomization(sub, [e_ref[i] for i in idx], e0_ref))
    els = sorted({s for a in sub for s in a.get_chemical_symbols()})
    a_mat = np.array([[a.get_chemical_symbols().count(e) / len(a) for e in els] for a in sub])
    mu, *_ = np.linalg.lstsq(a_mat, da, rcond=None)
    return {e: float(v) for e, v in zip(els, mu)}


def loss_energygeometry(frames, idx, e_pred, e_ref, temperature=3000.0):
    """PTBP's energygeometry mu = 0.5 mu_V + 0.5 mu_E on the EOS groups of ``idx``, without
    the r0 penalty (utils/loss.py: get_energy_per_atom, eos_cal, collectAndcalculate_muEV;
    utils/data.py: get_bolt_factor with "boltzmann").

    Group weight b_g: exp(-(E_c - E_c,min)/kT) of the group's centre structure (energy per atom),
    normalised within its composition (Hill empirical formula); groups with b_g < 0.1 are
    dropped.  mu_E,g = sum |dE_rel| / (n b_g), the relative energies taken against the
    structure with the lowest DFT energy of the group, and mu_E = sum b_g mu_E,g / sum b_g.
    mu_V,g = |V0_DFTB / V0_DFT - 1| from Birch-Murnaghan fits (100 if a fit fails), and
    mu_V = sum b_g mu_V,g / sum b_g."""
    from pydftb.assess import birch_murnaghan
    kt = KB_EV * temperature
    groups = {f: m for f, m in _groups(frames, idx).items() if f.startswith("eos_")}
    if not groups:
        return np.nan
    stats = {}
    for fam, m in groups.items():
        n = np.array([len(frames[i]) for i in m])
        er = np.array([e_ref[i] for i in m]) / n
        ep = np.array([e_pred[i] for i in m]) / n
        if not np.all(np.isfinite(ep)):
            return np.nan
        jmin = int(np.argmin(er))
        abs_sum = float(np.sum(np.abs((ep - ep[jmin]) - (er - er[jmin]))))
        v = np.array([frames[i].get_volume() / len(frames[i]) for i in m])
        fr, fp = birch_murnaghan(v, er), birch_murnaghan(v, ep)
        mu_v = abs(fp[1] / fr[1] - 1.0) if fr and fp else 100.0
        if not np.isfinite(mu_v):
            mu_v = 100.0
        centre = len(m) // 2
        comp = frames[m[centre]].get_chemical_formula(mode="hill", empirical=True)
        stats[fam] = (abs_sum, len(m), mu_v, float(er[centre]), comp)
    b = {}
    for comp in {st[4] for st in stats.values()}:
        fams = [f for f, st in stats.items() if st[4] == comp]
        ec = np.array([stats[f][3] for f in fams])
        w = np.exp(-(ec - ec.min()) / kt)
        w /= w.sum()
        b.update(dict(zip(fams, w)))
    kept = [f for f in stats if b[f] >= 0.1]
    tot = sum(b[f] for f in kept)
    mu_e = sum(stats[f][0] / stats[f][1] for f in kept) / tot
    mu_v = sum(b[f] * stats[f][2] for f in kept) / tot
    return float(0.5 * mu_v + 0.5 * mu_e)


def sigma_search(objective, n, lo, hi, grid=11, maxiter=25):
    """min over sigma (n values in [lo, hi]) of ``objective``: a shared 1-D grid, one sweep of
    per-element grids, then bounded L-BFGS-B (always within the bounds)."""
    from scipy.optimize import minimize
    g = np.linspace(lo, hi, int(grid))
    seen = {}

    def f(s):
        s = np.clip(np.atleast_1d(np.asarray(s, float)), lo, hi)
        key = tuple(np.round(s, 10))
        if key not in seen:
            v = objective(s)
            seen[key] = v if np.isfinite(v) else 1e12
        return seen[key]

    best = min((f(np.full(n, v)), tuple(np.full(n, v))) for v in g)
    s = np.array(best[1])
    if n > 1:
        for i in range(n):
            for v in g:
                t = s.copy()
                t[i] = v
                if f(t) < f(s):
                    s = t
    if n:
        res = minimize(f, s, method="L-BFGS-B", bounds=[(lo, hi)] * n,
                       options={"maxiter": int(maxiter)})
        if f(res.x) < f(s):
            s = np.clip(res.x, lo, hi)
    return s, f(s), len(seen)


def r0_penalty_of(space: Space, radii: dict) -> float:
    """PTBP's r0 penalty, geometric mean over the fitted elements (M5b)."""
    def r0_penalty(r0_w, r0_d, rc):           # = utils.loss.r0_penalty (no skopt import here)
        return ((r0_w / (2 * rc) - 1) ** 2 + 1) ** 2 * ((r0_d / (4 * rc) - 1) ** 2 + 2) ** 2
    vals = [r0_penalty(radii[el][0], radii[el][1], published(el)["cov_radius"]) for el in space.fit]
    return float(np.exp(np.mean(np.log(vals)))) if vals else 1.0


def score_point(cfg, index: int, force: bool = False) -> dict:
    """Stage ``losses`` for one point (see the module docstring); cached in losses.json."""
    from pydftb import assess
    from pydftb.bandloss import band_loss
    out_path = os.path.join(point_dir(cfg, index), "losses.json")
    if os.path.exists(out_path) and not force:
        with open(out_path) as fh:
            return json.load(fh)
    frames, e_el, f_el, res = _point_records(cfg, index)
    e_ref = np.array([a.get_potential_energy() for a in frames])
    f_ref = []
    for a in frames:
        try:
            f_ref.append(np.array(a.get_forces()))
        except Exception:                     # noqa: BLE001
            f_ref.append(None)
    e0_ref = load_e0s(cfg)
    e0_pred = res["e0_dftb"]
    sp = space_of(cfg)
    pcfg, lcfg = cfg["physrep"], cfg["loss"]
    lo, hi = (float(v) for v in pcfg.get("sigma_bounds", [0.1, 0.8]))
    phys = _Physrep(frames, pcfg.get("fit", sp.fit), pcfg.get("fixed", {}))
    n_s = len(phys.fit)
    roles = {r: [i for i, a in enumerate(frames) if a.info["role"] == r]
             for r in ("train", "val", "test", "ood")}
    train = roles["train"]
    eos_train = [i for i in train if frames[i].info["family"].startswith("eos_")]
    all_idx = list(range(len(frames)))

    def predicted(s, idx):
        er, fr = phys.evaluate(s, idx)
        e = np.full(len(frames), np.nan)
        f = [None] * len(frames)
        for k, i in enumerate(idx):
            e[i] = e_el[i] + er[k]
            f[i] = f_el[i] + fr[k]
        return e, f

    def obj_ef(idx, offsets_mode=None):
        def obj(s):
            e, f = predicted(s, idx)
            off = (per_element_offsets(frames, idx, e, e_ref, e0_ref, e0_pred)
                   if offsets_mode == "per_element" else None)
            return loss_ef(frames, idx, e, f, e_ref, f_ref, e0_ref, e0_pred, lcfg, off)
        return obj

    def obj_eg(s):
        e, _ = predicted(s, eos_train)
        return loss_energygeometry(frames, eos_train, e, e_ref)

    radii = res["radii"]
    penalty = r0_penalty_of(sp, radii)
    use_pen = bool(lcfg.get("penalty", False))
    wanted = cfg["strategies"]
    out = {"index": int(index), "radii": radii, "penalty": penalty, "n_failed": res["n_failed"],
           "sigma_fit": phys.fit, "sigma_fixed": phys.fixed, "strategies": {}}
    searches = {"S1": (obj_ef(train), train)}
    if "S1o" in wanted:
        searches["S1o"] = (obj_ef(train, "per_element"), train)
    if "S1eos" in wanted and eos_train:
        searches["S1eos"] = (obj_ef(eos_train), eos_train)
    if "S2" in wanted and eos_train:
        searches["S2"] = (obj_eg, eos_train)
    lband = (band_loss([{**b, "loss": np.nan if b.get("loss") is None else b["loss"]}
                        for b in res["bands"]]) if res["bands"] else np.nan)
    reactions = assess.load_reactions(cfg["reactions"]) if cfg.get("reactions") else None
    for name, (obj, _) in searches.items():
        t0 = time.perf_counter()
        s, val, nev = sigma_search(obj, n_s, lo, hi, int(pcfg.get("grid", 11)),
                                   int(pcfg.get("maxiter", 25)))
        e, f = predicted(s, all_idx)
        rec = assess.records_from_arrays(frames, e, f, [res["errors"][i] for i in all_idx])
        summ = assess.summarize(rec, e0_ref, e0_pred, reactions,
                                offsets="per_element" if name == "S1o" else "none")
        out["strategies"][name] = {"loss": float(val) * (penalty if use_pen else 1.0),
                                   "loss_raw": float(val), "sigma": phys.sigma_dict(s),
                                   "n_eval": nev, "seconds": round(time.perf_counter() - t0, 2),
                                   "summary": _brief(summ)}
    s1 = out["strategies"]["S1"]
    if "S3" in wanted:
        out["strategies"]["S3"] = {"loss": lband, "sigma": s1["sigma"], "summary": s1["summary"],
                                   "note": "point chosen by the band loss, sigma by L_EF (S1)"}
    if "S4" in wanted:
        lam = float(lcfg.get("lambda_band", 1.0))
        out["strategies"]["S4"] = {"loss": s1["loss"] + lam * lband if np.isfinite(lband) else np.nan,
                                   "sigma": s1["sigma"], "summary": s1["summary"]}
    out["bands"] = {"loss": lband, "per_structure": res["bands"]}
    rcfg = cfg["repfit"]
    if rcfg.get("pairs"):
        out["R2b"] = _repfit_control(frames, e_el, f_el, e_ref, f_ref, e0_ref, e0_pred, rcfg,
                                     train, all_idx, reactions)
        out["oracle"] = _repfit_control(frames, e_el, f_el, e_ref, f_ref, e0_ref, e0_pred, rcfg,
                                        all_idx, all_idx, reactions)
    with open(out_path + ".tmp", "w") as fh:
        json.dump(_clean(out), fh, indent=1)
    os.replace(out_path + ".tmp", out_path)
    return _clean(out)


def _brief(summary: dict) -> dict:
    """The parts of an assessment kept per point (no per-structure lists)."""
    en = summary["energies"]
    return {"by_role": en["by_role"], "by_family": en["by_family"], "offsets": en["offsets"],
            "eos": summary["eos"], "dimers": {k: {kk: v for kk, v in d.items()
                                                 if kk not in ("r", "e_ref", "e_pred")}
                                             for k, d in summary["dimers"].items()},
            "reactions": summary["reactions"], "tier": summary["tier"],
            "failed": summary["failed"]}


def _repfit_control(frames, e_el, f_el, e_ref, f_ref, e0_ref, e0_pred, rcfg, fit_idx, all_idx,
                    reactions):
    """R2b (fit on train) or the oracle (fit on all): a flexible two-body spline repulsive on
    this electronic part; its assessment on every role."""
    from pydftb import assess
    from pydftb.repfit import RepSample, fit_pair_splines
    pairs = {tuple(k.split("-")): tuple(v) for k, v in rcfg["pairs"].items()}
    samples = []
    for i in fit_idx:
        if not np.isfinite(e_el[i]):
            continue
        a = frames[i]
        target = ((e_ref[i] - sum(e0_ref[s] for s in a.get_chemical_symbols()))
                  - (e_el[i] - sum(e0_pred[s] for s in a.get_chemical_symbols())))
        tf = None if f_ref[i] is None else f_ref[i] - f_el[i]
        samples.append(RepSample(a, target, tf, float(rcfg.get("weights", {}).get(
            a.info["family"], 1.0)), a.info["family"], a.info["name"]))
    try:
        fit = fit_pair_splines(samples, pairs, int(rcfg.get("n_knots", 12)),
                               float(rcfg.get("smooth", 1e-3)), float(rcfg.get("force_weight", 1.0)),
                               bool(rcfg.get("monotonic", True)), rcfg.get("offsets", "none"))
    except Exception as exc:                  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    e = np.full(len(frames), np.nan)
    f = [None] * len(frames)
    for i in all_idx:
        er, fr = fit.predict(frames[i])
        e[i] = e_el[i] + er + fit.offset_energy(frames[i])
        f[i] = f_el[i] + fr
    rec = assess.records_from_arrays(frames, e, f)
    summ = assess.summarize(rec, e0_ref, e0_pred, reactions)
    return {"stats": fit.stats, "summary": _brief(summ), "model": json.loads(json.dumps(
        {"splines": [s.to_dict() for s in fit.splines.values()], "offsets": fit.offsets}))}


def _score_one(args):
    cfg_path, index, force = args
    cfg = load_config(cfg_path)
    try:
        score_point(cfg, index, force)
        return index, None
    except Exception as exc:                  # noqa: BLE001
        with open(os.path.join(point_dir(cfg, index), "losses_error.txt"), "w") as fh:
            fh.write(traceback.format_exc())
        return index, f"{type(exc).__name__}: {exc}"


def score_points(cfg, jobs: int = 1, force: bool = False, log=print) -> str:
    """Stage ``losses`` for every finished point; writes losses.csv."""
    done = sorted(int(os.path.basename(os.path.dirname(p))[1:]) for p in
                  glob.glob(os.path.join(cfg["out"], "points", "p*", "result.json")))
    log(f"[scan] losses: {len(done)} finished points, {jobs} process(es)")
    if jobs > 1:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(var, "1")
    errors = {}
    if jobs <= 1:
        for i in done:
            _, err = _score_one((cfg["_path"], i, force))
            if err:
                errors[i] = err
    else:
        ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx,
                                    initializer=_worker_init) as ex:
            for i, err in ex.map(_score_one, [(cfg["_path"], i, force) for i in done]):
                if err:
                    errors[i] = err
    for i, err in errors.items():
        log(f"[scan] losses of point {i} failed: {err}")
    return write_table(cfg, log)


def _row(cfg, sc):
    row = {"index": sc["index"], "penalty": sc["penalty"], "n_failed": sc["n_failed"]}
    for el, (w, d) in sc["radii"].items():
        row[f"{el}.r0_w"], row[f"{el}.r0_d"] = w, d
    for name, st in sc["strategies"].items():
        row[f"{name}.loss"] = st.get("loss")
        for el, v in (st.get("sigma") or {}).items():
            row[f"{name}.sigma.{el}"] = v
        br = (st.get("summary") or {}).get("by_role", {})
        for role in ("train", "test", "ood"):
            b = br.get(role) or {}
            row[f"{name}.{role}.rel_meV"] = None if b.get("relative_mae") is None else 1e3 * b["relative_mae"]
            row[f"{name}.{role}.atom_meV"] = None if b.get("atomization_mae") is None else 1e3 * b["atomization_mae"]
            row[f"{name}.{role}.force"] = b.get("force_mae")
        row[f"{name}.tier"] = (st.get("summary") or {}).get("tier")
    row["band.loss"] = sc.get("bands", {}).get("loss")
    for ctl in ("R2b", "oracle"):
        br = ((sc.get(ctl) or {}).get("summary") or {}).get("by_role", {})
        for role in ("train", "test", "ood"):
            b = br.get(role) or {}
            row[f"{ctl}.{role}.rel_meV"] = None if b.get("relative_mae") is None else 1e3 * b["relative_mae"]
            row[f"{ctl}.{role}.atom_meV"] = None if b.get("atomization_mae") is None else 1e3 * b["atomization_mae"]
            row[f"{ctl}.{role}.force"] = b.get("force_mae")
    return row


def write_table(cfg, log=print) -> str:
    rows = []
    for p in sorted(glob.glob(os.path.join(cfg["out"], "points", "p*", "losses.json"))):
        with open(p) as fh:
            rows.append(_row(cfg, json.load(fh)))
    path = os.path.join(cfg["out"], "losses.csv")
    if not rows:
        log("[scan] no scored points yet")
        return path
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"[scan] {len(rows)} points -> {path}")
    return path


# ---- report -----------------------------------------------------------------------------
def _scored(cfg):
    out = []
    for p in sorted(glob.glob(os.path.join(cfg["out"], "points", "p*", "losses.json"))):
        with open(p) as fh:
            out.append(json.load(fh))
    return out


def best_points(scored, strategies) -> dict:
    """{strategy: the scored point with the smallest finite loss}."""
    out = {}
    for name in strategies:
        cands = [s for s in scored if name in s["strategies"]
                 and s["strategies"][name].get("loss") is not None
                 and np.isfinite(s["strategies"][name]["loss"])]
        if cands:
            out[name] = min(cands, key=lambda s: s["strategies"][name]["loss"])
    return out


def _get(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


def report(cfg, log=print) -> str:
    """report.md and report.json: best point per strategy with its metrics, controls, Spearman."""
    from scipy.stats import spearmanr
    scored = _scored(cfg)
    if not scored:
        raise RuntimeError("no scored points: run `ptbp scan losses` first")
    best = best_points(scored, cfg["strategies"])
    rows = []
    for name, sc in best.items():
        st = sc["strategies"][name]
        summ = st.get("summary") or {}
        r = {"strategy": name, "index": sc["index"], "radii": sc["radii"], "sigma": st.get("sigma"),
             "loss": st.get("loss"), "band_loss": _get(sc, "bands", "loss"), "tier": summ.get("tier")}
        for role in ("train", "test", "ood"):
            b = _get(summ, "by_role", role) or {}
            r[role] = {k: b.get(k) for k in ("relative_mae", "atomization_mae", "offset_mae",
                                             "force_mae", "n", "failed")}
        r["eos"] = {k: {kk: v.get(kk) for kk in ("v0_rel_err", "b_rel_err", "edge_pred")}
                    for k, v in (summ.get("eos") or {}).items()}
        r["reactions"] = [{k: x.get(k) for k in ("name", "de_ref", "de_pred", "error", "sign_ok",
                                                  "decisive", "unit")} for x in summ.get("reactions", [])]
        r["R2b_test"] = _get(sc, "R2b", "summary", "by_role", "test")
        r["oracle_all"] = {role: _get(sc, "oracle", "summary", "by_role", role)
                           for role in ("train", "test", "ood")}
        rows.append(r)
    # does a better band loss give a better repulsive-limited energy?
    pairs = [(_get(s, "bands", "loss"), _get(s, "R2b", "summary", "by_role", "test", "relative_mae"),
              _get(s, "strategies", "S1", "summary", "by_role", "test", "relative_mae"))
             for s in scored]
    corr = {}
    for k, label in ((1, "band_vs_R2b_test"), (2, "band_vs_S1_test")):
        xs = [(p[0], p[k]) for p in pairs if p[0] is not None and p[k] is not None
              and np.isfinite(p[0]) and np.isfinite(p[k])]
        if len(xs) >= 5:
            rho, pval = spearmanr([a for a, _ in xs], [b for _, b in xs])
            corr[label] = {"spearman": float(rho), "p": float(pval), "n": len(xs)}
    out = {"name": cfg["name"], "n_points": len(scored), "best": rows, "correlations": corr,
           "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}
    path_json = os.path.join(cfg["out"], "report.json")
    with open(path_json, "w") as fh:
        json.dump(_clean(out), fh, indent=1)
    md = _report_markdown(out)
    path_md = os.path.join(cfg["out"], "report.md")
    with open(path_md, "w") as fh:
        fh.write(md)
    try:
        _maps(cfg, scored)
    except Exception as exc:                  # noqa: BLE001  (plots are optional)
        log(f"[scan] maps skipped: {exc}")
    log(f"[scan] report -> {path_md}")
    return path_md


def _f(v, scale=1.0, digits=1):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v * scale:.{digits}f}"


def _report_markdown(out) -> str:
    L = [f"# scan {out['name']}: best point per strategy", "", f"{out['n_points']} scored points", "",
         "| strategy | point | loss | band loss | tier | test rel | test atom | test F | ood rel | ood atom | ood F | R2b test rel | oracle test rel |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["best"]:
        L.append(f"| {r['strategy']} | {r['index']} | {_f(r['loss'], 1, 3)} | {_f(r['band_loss'], 1, 3)} | "
                 f"{r['tier'] or '—'} | {_f(r['test'].get('relative_mae'), 1e3)} | "
                 f"{_f(r['test'].get('atomization_mae'), 1e3)} | {_f(r['test'].get('force_mae'), 1, 3)} | "
                 f"{_f(r['ood'].get('relative_mae'), 1e3)} | {_f(r['ood'].get('atomization_mae'), 1e3)} | "
                 f"{_f(r['ood'].get('force_mae'), 1, 3)} | "
                 f"{_f((r['R2b_test'] or {}).get('relative_mae'), 1e3)} | "
                 f"{_f(((r['oracle_all'] or {}).get('test') or {}).get('relative_mae'), 1e3)} |")
    L += ["", "energies meV/atom (rel = within-family relative, atom = absolute atomization), "
          "forces eV/A", "", "## radii and sigma", ""]
    for r in out["best"]:
        rad = ", ".join(f"{el} ({w:.3f}, {d:.3f})" for el, (w, d) in r["radii"].items())
        sig = ", ".join(f"{el} {v:.3f}" for el, v in (r["sigma"] or {}).items())
        L.append(f"- **{r['strategy']}**: r0 (w, d) Bohr: {rad}; sigma: {sig}")
    L += ["", "## reactions at each strategy's best point", ""]
    for r in out["best"]:
        for x in r["reactions"]:
            if x.get("de_ref") is None:
                continue
            L.append(f"- {r['strategy']}: {x['name']}: ref {x['de_ref']:.4f}, pred {x['de_pred']:.4f} "
                     f"{x['unit']} ({'ok' if x['sign_ok'] else 'WRONG SIGN'}{', decisive' if x['decisive'] else ''})")
    L += ["", "## correlations", ""]
    for k, v in out["correlations"].items():
        L.append(f"- {k}: Spearman {v['spearman']:.2f} (p {v['p']:.2g}, n {v['n']})")
    return "\n".join(L) + "\n"


def _maps(cfg, scored):
    """Loss maps on (r0_w, r0_d) for a one-element space (PNG next to the report)."""
    sp = space_of(cfg)
    if sp.dim != 2:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    el = sp.fit[0]
    x = np.array([s["radii"][el][0] for s in scored])
    y = np.array([s["radii"][el][1] for s in scored])
    names = [n for n in cfg["strategies"] if any(n in s["strategies"] for s in scored)]
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.0), squeeze=False)
    for ax, name in zip(axes[0], names):
        z = np.array([s["strategies"].get(name, {}).get("loss") or np.nan for s in scored], float)
        sc = ax.scatter(x, y, c=np.log10(z), s=18, cmap="viridis")
        k = int(np.nanargmin(z)) if np.any(np.isfinite(z)) else None
        if k is not None:
            ax.plot(x[k], y[k], "r*", ms=10)
        ax.set_title(name)
        ax.set_xlabel(f"{el} r0_w (Bohr)")
        ax.set_ylabel(f"{el} r0_d (Bohr)")
        fig.colorbar(sc, ax=ax, label="log10 loss")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["out"], "maps.png"), dpi=150)
    plt.close(fig)


# ---- refine and export ----------------------------------------------------------------------
def refine(cfg, strategy: str = "S1", start: int | None = None, maxfev: int = 30, log=print) -> dict:
    """Nelder–Mead on the radii of ``strategy`` from its best scored point (or ``start``);
    each evaluation is a new point (index 100000 + n) with its own SK set, run and score."""
    from scipy.optimize import minimize
    sp = space_of(cfg)
    lo, hi = sp.lower_upper()
    scored = _scored(cfg)
    if start is None:
        b = best_points(scored, [strategy]).get(strategy)
        if b is None:
            raise RuntimeError(f"no scored point for {strategy}")
        x0 = sp.vector(b["radii"])
    else:
        x0 = read_points(cfg)[start]
    used = [int(os.path.basename(p)[1:]) for p in glob.glob(os.path.join(cfg["out"], "points", "p*"))]
    first = max([100000] + [u + 1 for u in used if u >= 100000])    # refinements: 100000 +
    counter = {"n": 0}
    history = []

    def f(x):
        x = np.clip(x, lo, hi)
        idx = first + counter["n"]
        counter["n"] += 1
        evaluate_point(cfg, idx, x=x, log=log)
        sc = score_point(cfg, idx, force=True)
        v = sc["strategies"].get(strategy, {}).get("loss")
        v = 1e12 if v is None or not np.isfinite(v) else float(v)
        history.append({"index": idx, "x": x.tolist(), "loss": v})
        log(f"[refine] {strategy} eval {counter['n']}: loss {v:.5g} at {np.round(x, 4).tolist()}")
        return v

    res = minimize(f, x0, method="Nelder-Mead", bounds=list(zip(lo, hi)),
                   options={"maxfev": int(maxfev), "xatol": 1e-3, "fatol": 1e-4})
    out = {"strategy": strategy, "x0": x0.tolist(), "x": np.clip(res.x, lo, hi).tolist(),
           "loss": float(res.fun), "nfev": int(res.nfev), "history": history,
           "radii": sp.radii(np.clip(res.x, lo, hi))}
    with open(os.path.join(cfg["out"], f"refine_{strategy}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def export_model(cfg, index: int, strategy: str, out_dir: str, repulsive: str = "physrep",
                 r_head: float = 0.6, n_grid: int = 200) -> str:
    """Full SK files of a point: its electronic tables plus the repulsive of ``strategy``
    (physrep at the strategy's sigma) or of the R2b control (``repulsive="R2b"``), as Spline
    blocks.  Use them with ``PyDFTB(out_dir, repulsive="skf")`` (MD checks) or DFTB+."""
    from ase.data import atomic_numbers
    from pydftb.constants import BOHR, HARTREE
    from pydftb.repfit import PairSpline, skf_block_from_grid, write_spline_blocks
    with open(os.path.join(point_dir(cfg, index), "losses.json")) as fh:
        sc = json.load(fh)
    with open(os.path.join(point_dir(cfg, index), "result.json")) as fh:
        res = json.load(fh)
    blocks = {}
    if repulsive == "R2b":
        for d in sc["R2b"]["model"]["splines"]:
            sp = PairSpline.from_dict(d)
            if np.any(sp.coef):
                blocks[tuple(sp.pair)] = sp.skf_block()
    else:
        sigma = sc["strategies"][strategy]["sigma"]
        els = sorted(sigma, key=lambda s: atomic_numbers[s])
        frames = _data_cache(cfg)
        phys = _Physrep(frames, [], sigma)
        pr = phys.model([])
        for i, a in enumerate(els):
            for b in els[: i + 1]:
                key = f"{atomic_numbers[a]}_{atomic_numbers[b]}"
                rcut_a = float(pr.rcuts[key]) * BOHR
                r = np.linspace(r_head, rcut_a, n_grid)
                v = np.array(pr.Vreps[key](r / BOHR)) * HARTREE
                blocks[(a, b)] = skf_block_from_grid(r, v)
    write_spline_blocks(res["skdir"], out_dir, blocks)
    with open(os.path.join(out_dir, "model.json"), "w") as fh:
        json.dump({"scan": cfg["name"], "index": index, "strategy": strategy,
                   "repulsive": repulsive, "radii": sc["radii"],
                   "sigma": sc["strategies"].get(strategy, {}).get("sigma"),
                   "max_l": cfg["dftb"].get("max_l"), "skdir": res["skdir"]}, fh, indent=1)
    return out_dir


def status(cfg) -> dict:
    """How many points exist, ran, failed and were scored."""
    n = len(read_points(cfg)) if os.path.exists(os.path.join(cfg["out"], "points.json")) else 0
    ran = glob.glob(os.path.join(cfg["out"], "points", "p*", "result.json"))
    err = glob.glob(os.path.join(cfg["out"], "points", "p*", "error.txt"))
    sc = glob.glob(os.path.join(cfg["out"], "points", "p*", "losses.json"))
    return {"points": n, "ran": len(ran), "errors": len(err), "scored": len(sc)}
