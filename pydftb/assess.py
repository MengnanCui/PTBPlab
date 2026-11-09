"""Evaluation of a DFTB model against frozen DFT data sets (A series, A0 section 1).

Every structure becomes a :class:`Record`: name, family, role, reference and predicted
energies and forces.  Records come from :func:`predict`, which runs any ASE calculator, or
from :func:`records_from_arrays`, used by PTBP's scan tool, which assembles electronic and
repulsive parts itself.  :func:`summarize` then reports:

* **energies by role** (train / val / test / ood) and **by family**:
  * the absolute atomization energy error, for diagnosis only (user decision 2026-11-09);
  * the error of relative energies within a family (each side minus its family mean, so the
    differences of one composition).  It is undefined (NaN) for a family with one structure,
    and a role without any relative energy cannot meet a tier;
  * optionally the errors left after per-element offsets fitted on the training records;
* **forces:** MAE and RMSE per component;
* **EOS families** (name prefix ``eos_``): Birch–Murnaghan V0, B and E0 per atom on both sides;
* **dimer families** (prefix ``dimer_``): E(r) − E(r_max) curves, the position of the minimum
  and the well depth;
* **reactions** (:func:`load_reactions`, YAML or JSON): balanced energy differences between
  records or EOS minima (``family:<name>``), with the element balance checked, the error and
  whether the sign agrees; ``decisive`` ones decide the tier;
* **tier:** A (≤ 10 meV/atom, ≤ 0.1 eV/Å, user decision 2026-11-09), B (30, 0.2), C (100,
  signs only), on the roles given.

:func:`nve_check` runs a short Langevin-then-NVE MD and reports the drift of the conserved
energy and the shortest distance per element pair.  It is the stability test of the
acceptance line.  The module needs NumPy, SciPy and ASE only.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

TIERS = {"A": (0.010, 0.10), "B": (0.030, 0.20), "C": (0.100, None)}   # eV/atom, eV/A


@dataclass
class Record:
    name: str
    family: str
    role: str
    symbols: list
    volume: float | None
    e_ref: float                       # eV, total
    e_pred: float                      # eV, total (NaN when the calculation failed)
    f_ref: np.ndarray | None = None    # eV/A
    f_pred: np.ndarray | None = None
    error: str | None = None
    r_dimer: float | None = None       # A, for two-atom structures

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def ok(self) -> bool:
        return self.error is None and np.isfinite(self.e_pred)


def tags(atoms) -> tuple:
    """(name, family, role) from ``atoms.info`` (``structure_name``/``group`` are the older
    keys of PTBP's data)."""
    info = atoms.info
    name = info.get("name") or info.get("structure_name") or atoms.get_chemical_formula()
    family = info.get("family") or info.get("group") or name
    return str(name), str(family), str(info.get("role", "train"))


def load_frames(paths, role: str | None = None) -> list:
    """Frames of extxyz files (energies and forces in ``atoms.calc``); ``role`` sets
    ``info['role']`` where it is not given."""
    import ase.io
    out = []
    for p in ([paths] if isinstance(paths, str) else paths):
        for at in ase.io.read(p, ":"):
            if role is not None:
                at.info.setdefault("role", role)
            out.append(at)
    return out


def _reference(at):
    e = at.get_potential_energy()
    try:
        f = np.array(at.get_forces(), float)
    except Exception:                      # noqa: BLE001  (no forces stored)
        f = None
    return float(e), f


def _volume(at):
    return float(at.get_volume()) if at.pbc.all() and at.cell.rank == 3 else None


def _dimer_r(at):
    if len(at) != 2:
        return None
    return float(at.get_distance(0, 1, mic=bool(np.any(at.pbc))))


def records_from_arrays(frames, e_pred, f_pred=None, errors=None) -> list:
    """Records from frames (references in ``atoms.calc``) and predicted totals (eV) and
    forces (eV/A); NaN or an entry in ``errors`` marks a failed structure."""
    out = []
    for k, at in enumerate(frames):
        name, family, role = tags(at)
        e_ref, f_ref = _reference(at)
        err = None if errors is None else errors[k]
        ep = float(e_pred[k]) if e_pred[k] is not None else np.nan
        fp = None if f_pred is None or f_pred[k] is None else np.asarray(f_pred[k], float)
        out.append(Record(name, family, role, at.get_chemical_symbols(), _volume(at), e_ref, ep,
                          f_ref, fp, err if err is not None else (None if np.isfinite(ep) else
                                                                  "no prediction"),
                          _dimer_r(at)))
    return out


def predict(frames, calc_factory, forces: bool = True) -> list:
    """Records from an ASE calculator: ``calc_factory(atoms)`` returns the calculator for that
    structure (k-points, smearing).  A failed calculation is recorded, not raised."""
    e, f, err = [], [], []
    for at in frames:
        a = at.copy()
        try:
            a.calc = calc_factory(at)
            e.append(float(a.get_potential_energy()))
            f.append(np.array(a.get_forces()) if forces else None)
            err.append(None)
        except Exception as exc:          # noqa: BLE001
            e.append(np.nan)
            f.append(None)
            err.append(f"{type(exc).__name__}: {exc}")
    return records_from_arrays(frames, e, f, err)


# ---- energies and forces -------------------------------------------------------------------
def atomization(records, e0_ref: dict, e0_pred: dict):
    """Per-atom atomization energies (eV/atom) on both sides: (E - sum E0) / N."""
    ar, ap = [], []
    for r in records:
        sr = sum(e0_ref[s] for s in r.symbols)
        sp = sum(e0_pred[s] for s in r.symbols)
        ar.append((r.e_ref - sr) / r.n_atoms)
        ap.append((r.e_pred - sp) / r.n_atoms)
    return np.array(ar), np.array(ap)


def element_offsets(records, deltas, roles=("train",)) -> dict:
    """Least-squares per-element offsets mu_X (eV/atom of X) of the per-atom errors on ``roles``."""
    els = sorted({s for r in records for s in r.symbols})
    rows, y = [], []
    for r, d in zip(records, deltas):
        if r.role in roles and r.ok:
            c = Counter(r.symbols)
            rows.append([c[e] / r.n_atoms for e in els])
            y.append(d)
    if not rows:
        return {e: 0.0 for e in els}
    mu, *_ = np.linalg.lstsq(np.array(rows), np.array(y), rcond=None)
    return {e: float(v) for e, v in zip(els, mu)}


def _force_stats(records):
    diffs = [(r.f_pred - r.f_ref).ravel() for r in records
             if r.ok and r.f_ref is not None and r.f_pred is not None]
    if not diffs:
        return {"force_mae": np.nan, "force_rmse": np.nan, "force_n": 0}
    d = np.concatenate(diffs)
    return {"force_mae": float(np.abs(d).mean()), "force_rmse": float(np.sqrt((d ** 2).mean())),
            "force_n": int(d.size)}


def energy_metrics(records, e0_ref: dict, e0_pred: dict, offsets: str = "none",
                   fit_roles=("train",)) -> dict:
    """Atomization, within-family relative and (optionally) offset-corrected energy errors and
    force errors, by role and by family (eV/atom, eV/A)."""
    ar, ap = atomization(records, e0_ref, e0_pred)
    d = ap - ar
    mu = element_offsets(records, d, fit_roles) if offsets == "per_element" else {}
    d_off = np.array([dk - sum(mu.get(s, 0.0) for s in r.symbols) / r.n_atoms
                      for r, dk in zip(records, d)]) if mu else d
    fam_mean = {}
    for fam in {r.family for r in records}:
        idx = [k for k, r in enumerate(records) if r.family == fam and r.ok]
        if len(idx) >= 2:                 # a relative energy needs two structures
            fam_mean[fam] = float(np.mean(d[idx]))
    rel = np.array([d[k] - fam_mean.get(r.family, np.nan) for k, r in enumerate(records)])

    def block(sel):
        sel = [k for k in sel if records[k].ok]
        out = {"n": len(sel), "failed": 0}
        if sel:
            r_sel = np.abs(rel[sel])
            out.update(atomization_mae=float(np.abs(d[sel]).mean()),
                       atomization_mean=float(d[sel].mean()),
                       relative_mae=float(np.nanmean(r_sel)) if np.isfinite(r_sel).any() else np.nan,
                       relative_n=int(np.isfinite(r_sel).sum()),
                       offset_mae=float(np.abs(d_off[sel]).mean()))
        out.update(_force_stats([records[k] for k in sel]))
        return out

    by_role, by_family = {}, {}
    for role in sorted({r.role for r in records}):
        sel = [k for k, r in enumerate(records) if r.role == role]
        by_role[role] = block(sel)
        by_role[role]["failed"] = sum(not records[k].ok for k in sel)
    for fam in sorted({r.family for r in records}):
        sel = [k for k, r in enumerate(records) if r.family == fam]
        by_family[fam] = block(sel)
        by_family[fam]["role"] = records[sel[0]].role
        by_family[fam]["failed"] = sum(not records[k].ok for k in sel)
    return {"by_role": by_role, "by_family": by_family, "offsets": mu,
            "per_structure": [{"name": r.name, "family": r.family, "role": r.role,
                               "d_atomization": float(d[k]), "d_relative": float(rel[k]),
                               "ok": r.ok, "error": r.error} for k, r in enumerate(records)]}


# ---- EOS and dimers --------------------------------------------------------------------------
def birch_murnaghan(volumes, energies):
    """(E0, V0, B (eV/A^3), B') of a Birch–Murnaghan fit (ASE), or None if it fails."""
    from ase.eos import EquationOfState
    try:
        eos = EquationOfState(list(volumes), list(energies), "birchmurnaghan")
        eos.fit(warn=False)
        e0, b, bp, v0 = eos.eos_parameters
        return float(e0), float(v0), float(b), float(bp)
    except Exception:                      # noqa: BLE001
        return None


def eos_metrics(records, prefix: str = "eos_") -> dict:
    """Per EOS family: V0, B (GPa), E0 per atom of both sides and the relative V0 and B errors;
    ``edge`` marks a minimum at the end of the scanned volumes."""
    from ase.units import GPa
    out = {}
    fams = sorted({r.family for r in records if r.family.startswith(prefix)})
    for fam in fams:
        rs = [r for r in records if r.family == fam and r.ok and r.volume is not None]
        if len(rs) < 5:
            continue
        v = np.array([r.volume / r.n_atoms for r in rs])
        er = np.array([r.e_ref / r.n_atoms for r in rs])
        ep = np.array([r.e_pred / r.n_atoms for r in rs])
        fr, fp = birch_murnaghan(v, er), birch_murnaghan(v, ep)
        item = {"n": len(rs), "v_range": [float(v.min()), float(v.max())],
                "argmin_ref": float(v[np.argmin(er)]), "argmin_pred": float(v[np.argmin(ep)])}
        item["edge_pred"] = bool(np.argmin(ep) in (0, len(ep) - 1))
        if fr:
            item.update(v0_ref=fr[1], b_ref_GPa=fr[2] / GPa, e0_ref=fr[0])
        if fp:
            item.update(v0_pred=fp[1], b_pred_GPa=fp[2] / GPa, e0_pred=fp[0])
        if fr and fp:
            item.update(v0_rel_err=fp[1] / fr[1] - 1.0, b_rel_err=fp[2] / fr[2] - 1.0)
        out[fam] = item
    return out


def dimer_metrics(records, prefix: str = "dimer_") -> dict:
    """Per dimer family: the curves relative to their longest distance, MAE, minima."""
    out = {}
    for fam in sorted({r.family for r in records if r.family.startswith(prefix)}):
        rs = sorted([r for r in records if r.family == fam and r.ok and r.r_dimer is not None],
                    key=lambda r: r.r_dimer)
        if len(rs) < 3:
            continue
        r = np.array([x.r_dimer for x in rs])
        er = np.array([x.e_ref for x in rs])
        ep = np.array([x.e_pred for x in rs])
        er, ep = er - er[-1], ep - ep[-1]
        out[fam] = {"r": r.tolist(), "e_ref": er.tolist(), "e_pred": ep.tolist(),
                    "mae": float(np.abs(ep - er).mean()),
                    "r_min_ref": float(r[np.argmin(er)]), "r_min_pred": float(r[np.argmin(ep)]),
                    "depth_ref": float(er.min()), "depth_pred": float(ep.min())}
    return out


# ---- reactions -------------------------------------------------------------------------------
def load_reactions(path: str) -> list:
    """Reactions from YAML (needs PyYAML) or JSON: a list of {name, reactants: {key: coeff},
    products: {key: coeff}, normalize: "products_atoms" | "reaction" | number, decisive}.
    A key is a structure name, or ``family:<name>`` for the EOS minimum of a family (its cell)."""
    with open(path) as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return data["reactions"] if isinstance(data, dict) else data


def _term_energy(key, records, eos, side):
    if key.startswith("family:"):
        fam = key.split(":", 1)[1]
        rs = [r for r in records if r.family == fam]
        if not rs:
            raise KeyError(f"no family {fam}")
        n = rs[0].n_atoms
        e0 = eos.get(fam, {}).get("e0_ref" if side == "ref" else "e0_pred")
        if e0 is None:                    # no EOS fit: the lowest structure of the family
            ok = [r for r in rs if r.ok]
            e0 = min((r.e_ref if side == "ref" else r.e_pred) / r.n_atoms for r in ok)
        return e0 * n, Counter(rs[0].symbols)
    match = [r for r in records if r.name == key]
    if not match:
        raise KeyError(f"no structure {key}")
    r = match[0]
    if not r.ok:
        return np.nan, Counter(r.symbols)
    return (r.e_ref if side == "ref" else r.e_pred), Counter(r.symbols)


def reaction_metrics(records, reactions, eos=None) -> list:
    """Each reaction's energy (eV, normalised as asked) on both sides, error and sign check."""
    eos = eos if eos is not None else eos_metrics(records)
    out = []
    for rx in reactions:
        try:
            e = {"ref": 0.0, "pred": 0.0}
            bal = Counter()
            n_prod = 0.0
            for sign, terms in ((-1, rx.get("reactants", {})), (+1, rx.get("products", {}))):
                for key, c in terms.items():
                    for side in ("ref", "pred"):
                        en, comp = _term_energy(key, records, eos, side)
                        e[side] += sign * float(c) * en
                    for el, k in comp.items():
                        bal[el] += sign * float(c) * k
                    if sign > 0:
                        n_prod += float(c) * sum(comp.values())
            unbalanced = {el: v for el, v in bal.items() if abs(v) > 1e-6}
            if unbalanced:
                raise ValueError(f"not balanced: {unbalanced}")
            norm = rx.get("normalize", "products_atoms")
            div = n_prod if norm == "products_atoms" else (1.0 if norm == "reaction" else float(norm))
            de_r, de_p = e["ref"] / div, e["pred"] / div
            out.append({"name": rx["name"], "de_ref": de_r, "de_pred": de_p,
                        "error": de_p - de_r, "sign_ok": bool(np.sign(de_r) == np.sign(de_p)),
                        "decisive": bool(rx.get("decisive", False)), "normalize": norm,
                        "unit": "eV/atom" if norm == "products_atoms" else "eV"})
        except Exception as exc:          # noqa: BLE001
            out.append({"name": rx.get("name", "?"), "error_text": f"{type(exc).__name__}: {exc}",
                        "decisive": bool(rx.get("decisive", False)), "sign_ok": False})
    return out


# ---- summary ------------------------------------------------------------------------------
def tier(summary: dict, roles=("test", "ood"), energy_key: str = "relative_mae") -> str | None:
    """The best tier the given roles meet: energy (``energy_key``) and force MAE within the
    thresholds, every decisive reaction with the right sign, and every per-atom reaction energy
    within the energy threshold."""
    rx = summary.get("reactions", [])
    signs = all(r.get("sign_ok", False) for r in rx if r.get("decisive"))
    for name, (te, tf) in TIERS.items():
        ok = signs
        for role in roles:
            b = summary["energies"]["by_role"].get(role)
            if not b:
                continue
            if b.get("failed", 0):
                ok = False                    # a failed structure fails every tier
                continue
            if b.get("n", 0) == 0:
                continue
            ok &= b.get(energy_key, np.inf) <= te
            if tf is not None and b.get("force_n", 0):
                ok &= b.get("force_mae", np.inf) <= tf
        for r in rx:
            if r.get("unit") == "eV/atom" and "error" in r and isinstance(r["error"], float):
                ok &= abs(r["error"]) <= te
        if ok:
            return name
    return None


def summarize(records, e0_ref: dict, e0_pred: dict, reactions=None, offsets: str = "none",
              tier_roles=("test", "ood")) -> dict:
    """Everything above in one dictionary (JSON-ready)."""
    eos = eos_metrics(records)
    s = {"energies": energy_metrics(records, e0_ref, e0_pred, offsets),
         "eos": eos, "dimers": dimer_metrics(records),
         "reactions": reaction_metrics(records, reactions, eos) if reactions else [],
         "failed": [{"name": r.name, "error": r.error} for r in records if not r.ok]}
    s["tier"] = tier(s, tier_roles)
    return s


def _fmt(v, scale=1.0, digits=1):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v * scale:.{digits}f}"


def write_markdown(summary: dict, path: str, title: str = "assessment") -> str:
    """A readable report: energies (meV/atom), forces (eV/A), EOS, dimers, reactions."""
    L = [f"# {title}", "", f"tier: **{summary.get('tier') or 'none'}**", "",
         "| role | n | failed | atomization MAE | mean | relative MAE | offset MAE | force MAE |",
         "|---|---|---|---|---|---|---|---|"]
    for role, b in summary["energies"]["by_role"].items():
        L.append(f"| {role} | {b['n']} | {b['failed']} | {_fmt(b.get('atomization_mae'), 1e3)} | "
                 f"{_fmt(b.get('atomization_mean'), 1e3)} | {_fmt(b.get('relative_mae'), 1e3)} | "
                 f"{_fmt(b.get('offset_mae'), 1e3)} | {_fmt(b.get('force_mae'), 1, 3)} |")
    L += ["", "energies in meV/atom, forces in eV/A", "", "## families", "",
          "| family | role | n | atomization MAE | relative MAE | force MAE |", "|---|---|---|---|---|---|"]
    for fam, b in summary["energies"]["by_family"].items():
        L.append(f"| {fam} | {b['role']} | {b['n']} | {_fmt(b.get('atomization_mae'), 1e3)} | "
                 f"{_fmt(b.get('relative_mae'), 1e3)} | {_fmt(b.get('force_mae'), 1, 3)} |")
    if summary["eos"]:
        L += ["", "## EOS", "", "| family | V0 ref | V0 pred | ΔV0 | B ref (GPa) | B pred | edge |",
              "|---|---|---|---|---|---|---|"]
        for fam, e in summary["eos"].items():
            L.append(f"| {fam} | {_fmt(e.get('v0_ref'), 1, 3)} | {_fmt(e.get('v0_pred'), 1, 3)} | "
                     f"{_fmt(e.get('v0_rel_err'), 100, 1)}% | {_fmt(e.get('b_ref_GPa'))} | "
                     f"{_fmt(e.get('b_pred_GPa'))} | {e.get('edge_pred')} |")
    if summary["dimers"]:
        L += ["", "## dimers", "", "| family | MAE (eV) | r_min ref | r_min pred | depth ref | depth pred |",
              "|---|---|---|---|---|---|"]
        for fam, d in summary["dimers"].items():
            L.append(f"| {fam} | {_fmt(d['mae'], 1, 3)} | {d['r_min_ref']:.2f} | {d['r_min_pred']:.2f} | "
                     f"{d['depth_ref']:.2f} | {d['depth_pred']:.2f} |")
    if summary["reactions"]:
        L += ["", "## reactions", "", "| reaction | ref | pred | error | unit | sign | decisive |",
              "|---|---|---|---|---|---|---|"]
        for r in summary["reactions"]:
            if "error_text" in r:
                L.append(f"| {r['name']} | {r['error_text']} | | | | | {r['decisive']} |")
                continue
            L.append(f"| {r['name']} | {r['de_ref']:.4f} | {r['de_pred']:.4f} | {r['error']:+.4f} | "
                     f"{r['unit']} | {'ok' if r['sign_ok'] else 'WRONG'} | {r['decisive']} |")
    if summary["failed"]:
        L += ["", "## failed", ""] + [f"- {f['name']}: {f['error']}" for f in summary["failed"]]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


# ---- MD stability ------------------------------------------------------------------------
def nve_check(atoms, calc, temperature: float = 300.0, langevin_steps: int = 200,
              nve_steps: int = 1000, dt_fs: float = 1.0, seed: int = 1, friction: float = 0.02,
              trajectory: str | None = None, log_every: int = 10) -> dict:
    """Langevin at ``temperature`` then NVE: drift of the total energy (meV/atom/ps, a linear
    fit), its range, the shortest distance per element pair over the run, and the failure if
    the calculation stopped (e.g. SCC not converged)."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    a = atoms.copy()
    a.calc = calc
    MaxwellBoltzmannDistribution(a, temperature_K=temperature, rng=np.random.default_rng(seed))
    sym = np.array(a.get_chemical_symbols())
    shortest = defaultdict(lambda: np.inf)
    rec = {"t_fs": [], "etot": [], "temperature": []}

    def watch(t):
        d = a.get_all_distances(mic=bool(np.any(a.pbc)))
        iu = np.triu_indices(len(a), 1)
        for x, y, dist in zip(sym[iu[0]], sym[iu[1]], d[iu]):
            key = "-".join(sorted((x, y)))
            shortest[key] = min(shortest[key], float(dist))
        rec["t_fs"].append(t)
        rec["etot"].append(float(a.get_total_energy()))
        rec["temperature"].append(float(a.get_temperature()))

    out = {"temperature": temperature, "langevin_steps": langevin_steps, "nve_steps": nve_steps,
           "dt_fs": dt_fs, "failed": None}
    try:
        dyn = Langevin(a, dt_fs * units.fs, temperature_K=temperature, friction=friction,
                       rng=np.random.default_rng(seed + 1))
        if trajectory:
            from ase.io.trajectory import Trajectory
            traj = Trajectory(trajectory, "w", a)
            dyn.attach(traj.write, interval=log_every)
        dyn.run(langevin_steps)
        nve = VelocityVerlet(a, dt_fs * units.fs)
        if trajectory:
            nve.attach(traj.write, interval=log_every)
        nve.attach(lambda: watch(nve.nsteps * dt_fs), interval=log_every)
        watch(0.0)
        nve.run(nve_steps)
    except Exception as exc:              # noqa: BLE001
        out["failed"] = f"{type(exc).__name__}: {exc}"
    t = np.array(rec["t_fs"])
    e = np.array(rec["etot"])
    if len(t) > 2:
        slope = np.polyfit(t / 1000.0, e, 1)[0]            # eV / ps
        out.update(drift_meV_per_atom_per_ps=float(slope * 1000 / len(a)),
                   etot_range_meV_per_atom=float((e.max() - e.min()) * 1000 / len(a)),
                   t_final_fs=float(t[-1]))
    out["shortest"] = {k: v for k, v in shortest.items()}
    out["temperature_trace"] = rec["temperature"][:: max(1, len(rec["temperature"]) // 50)]
    return out
