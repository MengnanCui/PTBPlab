"""Baselines of the A series (A0 plan, section 1): other methods on the same data, with the same
assessment as the scans (pydftb.assess: energies by role and family, forces, EOS, dimers,
balanced reactions, tiers on test and ood).

Methods (``--methods``; an optional engine that is not installed is skipped with the reason):

* ``ptbp_published``  PTBP's published radii and sigma with the A-series numerics: the scan's
  point at the published radii (point 0, ``include_published``) plus physrep at the published
  sigma of every element.  Needs that point run (``ptbp scan run CONFIG --points 0``);
  ``--run-missing`` runs it here (SK set + DFTB on every structure).
* ``ptbp_skf``  a directory of full SK files (``--ptbp-skf DIR``, e.g. PTBP's published files)
  with PyDFTB(repulsive="skf") and the A-series numerics.
* ``gfn1_xtb``  GFN1-xTB (tblite's ASE calculator).  tblite has no k-points: a periodic cell runs
  as the Gamma point of its N1 x N2 x N3 supercell, which equals the Gamma-centred N1 x N2 x N3
  grid of utils.accuracy.kpoints_for (energy / N1N2N3, forces of the first copy); cells whose
  supercell exceeds ``--xtb-max-atoms`` are recorded as failed.
* ``mace_mp``  MACE-MP-0 zero-shot (``--mace-model``, default medium).
* ``mace_off``  MACE-OFF23 (organic elements only: skipped when the data have others).
* ``mace_omol``  MACE-OMOL (mace-torch with ``mace.calculators.mace_omol``).

The zero of every method's atomization energies is its own isolated, neutral atoms (DFTB: the
scan's ``e0_dftb``; the others computed here in a 20 A box).

Dispersion (``--reference-d3``): the DFT data are plain PBE.  GFN1-xTB, MACE-OFF and MACE-OMOL
contain dispersion, so ``auto`` (default) compares those with PBE + D3(BJ) (simple-dftd3; plain
PBE with a note if it is missing) and the others plain against plain; ``on`` adds PBE-D3(BJ) to
the reference and to every method without its own dispersion; ``off`` keeps everything plain.

    python common/baselines.py --config li_o/scan_li_o.yaml --out li_o/runs/baselines \\
        [--methods ptbp_published gfn1_xtb mace_mp] [--ptbp-skf DIR] [--device cpu]

Writes ``<out>/<method>/{summary.md, summary.json, predictions.jsonl}``, ``<out>/baselines.json``
and ``<out>/baselines_summary.md``; logs in ``<out>/logs/baselines/``.  Predictions are cached
per structure, so a rerun continues where it stopped (``--force`` recomputes).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.paths import setup  # noqa: E402

setup()

METHODS = ("ptbp_published", "ptbp_skf", "gfn1_xtb", "mace_mp", "mace_off", "mace_omol")
INTRINSIC_DISPERSION = {"gfn1_xtb", "mace_off", "mace_omol"}
MACE_OFF_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


class Skip(Exception):
    """A method that cannot run here (missing engine, missing input): recorded, not fatal."""


# ---- predictions of an ASE calculator, cached --------------------------------------------------
def _cached_predictions(name, frames, calc_for, cache, log, force=False, key=None):
    """{"e", "f", "error"} per frame (eV, eV/A) from ``calc_for(frame) -> (energy, forces)``;
    JSONL cache per structure name (a rerun continues; another ``key`` starts again)."""
    from common.models import _clean
    done = {}
    head = cache + ".key.json"
    if os.path.exists(cache) and not force and os.path.exists(head) and json.load(open(head)) == key:
        for line in open(cache):
            if line.strip():
                r = json.loads(line)
                done[r["name"]] = r
    elif os.path.exists(cache):
        os.replace(cache, cache + ".old")
    with open(head, "w") as fh:
        json.dump(key, fh)
    todo = [f for f in frames if f.info["name"] not in done]
    if todo:
        log(f"[baselines] {name}: {len(todo)} of {len(frames)} structures to compute")
    with open(cache, "a", buffering=1) as fh:
        for n, fr in enumerate(todo, 1):
            try:
                e, f = calc_for(fr)
                r = {"name": fr.info["name"], "e": float(e),
                     "f": None if f is None else np.asarray(f, float).tolist(), "error": None}
            except Exception as exc:          # noqa: BLE001  (recorded)
                r = {"name": fr.info["name"], "e": None, "f": None,
                     "error": f"{type(exc).__name__}: {exc}"}
            done[r["name"]] = r
            fh.write(json.dumps(_clean(r)) + "\n")
            if n % 10 == 0 or n == len(todo):
                log(f"[baselines] {name}: {n}/{len(todo)}")
    return [done[f.info["name"]] for f in frames]


def _single_atom(el):
    from ase import Atoms
    return Atoms(el, positions=[[10.0, 10.0, 10.0]], cell=[20.0] * 3, pbc=False)


def _ase_energy_forces(calc, at):
    a = at.copy()
    a.calc = calc
    return float(a.get_potential_energy()), np.array(a.get_forces())


def _with_d3(calc, use):
    if not use:
        return calc
    from ase.calculators.mixing import SumCalculator
    from common.models import d3_calculator
    return SumCalculator([calc, d3_calculator()])


# ---- the methods ------------------------------------------------------------------------------
def _published_point(cfg, acc, run_missing, log):
    sp = acc.space_of(cfg)
    pub = sp.published_vector()
    if not os.path.exists(os.path.join(cfg["out"], "points.json")):
        if not run_missing:
            raise Skip("no points.json: run `ptbp scan points CONFIG` and `ptbp scan run CONFIG "
                       "--points 0` first (or --run-missing)")
        acc.write_points(cfg, log=log)
    pts = acc.read_points(cfg)
    idx = [i for i, x in enumerate(pts) if np.allclose(x, pub, rtol=0, atol=1e-9)]
    if not idx:
        raise Skip("the scan has no point at PTBP's published radii (include_published: false, "
                   "or the published radii are outside the box)")
    i = idx[0]
    if not os.path.exists(os.path.join(acc.point_dir(cfg, i), "result.json")):
        if not run_missing:
            raise Skip(f"point {i} (published radii) not run: `ptbp scan run CONFIG --points {i}` "
                       "(or --run-missing)")
        acc.evaluate_point(cfg, i, log=log)
    return i


def ptbp_published(cfg, frames, args, log, d3_pred):
    """Records and e0 of PTBP's published radii + published sigma, from the scan's point."""
    from pydftb import assess
    from utils import accuracy as acc
    i = _published_point(cfg, acc, args.run_missing, log)
    fr, e_el, f_el, res = acc._point_records(cfg, i)
    phys = acc._Physrep(fr, [], {})                    # every element at its published sigma
    all_idx = list(range(len(fr)))
    er, frr = phys.evaluate([], all_idx)
    e = np.array([e_el[k] + er[k] for k in all_idx])
    f = [f_el[k] + frr[k] for k in all_idx]
    if d3_pred:
        from common.models import d3_calculator
        d3 = d3_calculator()
        for k, a in enumerate(fr):
            ed, fd = _ase_energy_forces(d3, a)
            e[k] += ed
            f[k] = f[k] + fd
    by_name = {a.info["name"]: k for k, a in enumerate(fr)}
    order = [by_name[a.info["name"]] for a in frames]
    rec = assess.records_from_arrays(frames, [e[k] for k in order], [f[k] for k in order],
                                     [res["errors"][k] for k in order])
    return rec, dict(res["e0_dftb"]), {"point": i, "sigma": phys.sigma_dict([]),
                                       "radii": res["radii"]}


def ptbp_skf(cfg, frames, args, log, d3_pred):
    from common.models import Model, predict, records
    if not args.ptbp_skf:
        raise Skip("needs --ptbp-skf DIR (a directory of full SK files)")
    d = os.path.abspath(args.ptbp_skf)
    m = Model(d, d, "skf", dict(cfg["dftb"]), meta={"note": "baseline ptbp_skf"})
    elements = sorted({s for a in frames for s in a.get_chemical_symbols()})
    e0 = m.atom_energies(elements, dispersion=d3_pred)
    preds = predict(m, frames, os.path.join(args.out, "ptbp_skf", "predictions.jsonl"),
                    args.jobs, log, force=args.force, dispersion=d3_pred)
    return records(frames, preds), e0, {"skdir": d}


def _tblite():
    try:
        from tblite.ase import TBLite
    except ImportError as exc:
        raise Skip("tblite is not installed (conda install -c conda-forge tblite-python)") from exc
    return TBLite


def gfn1_xtb(cfg, frames, args, log, d3_pred):
    from pydftb import assess
    from utils.accuracy import kpoints_for
    TBLite = _tblite()

    def make():
        return TBLite(method="GFN1-xTB", verbosity=0)

    def run(at):
        if not np.any(at.pbc):
            return _ase_energy_forces(make(), at)
        spec = kpoints_for(at, cfg["dftb"].get("kspacing", 0.15))
        rep = (1, 1, 1) if spec is None else tuple(int(n) for n in spec["size"])
        sup = at.repeat(rep)
        if len(sup) > args.xtb_max_atoms:
            raise RuntimeError(f"Gamma supercell {rep} has {len(sup)} atoms > --xtb-max-atoms "
                               f"{args.xtb_max_atoms}")
        e, f = _ase_energy_forces(make(), sup)
        return e / int(np.prod(rep)), f[:len(at)]

    elements = sorted({s for a in frames for s in a.get_chemical_symbols()})
    e0 = {el: _ase_energy_forces(make(), _single_atom(el))[0] for el in elements}
    key = {"method": "gfn1_xtb", "max_atoms": args.xtb_max_atoms,
           "kspacing": cfg["dftb"].get("kspacing", 0.15)}
    preds = _cached_predictions("gfn1_xtb", frames, run,
                                os.path.join(args.out, "gfn1_xtb", "predictions.jsonl"), log,
                                args.force, key)
    return _records(assess, frames, preds), e0, {"engine": "tblite GFN1-xTB",
                                                 "periodic": "Gamma point of the k-grid supercell"}


def _records(assess, frames, preds):
    e = [np.nan if p["e"] is None else p["e"] for p in preds]
    f = [None if p["f"] is None else np.asarray(p["f"], float) for p in preds]
    return assess.records_from_arrays(frames, e, f, [p["error"] for p in preds])


def _mace(kind, args):
    try:
        import mace.calculators as mc
    except ImportError as exc:
        raise Skip("mace-torch is not installed (pip install mace-torch)") from exc
    if kind == "mace_mp":
        return mc.mace_mp(model=args.mace_model, device=args.device, default_dtype="float64")
    if kind == "mace_off":
        return mc.mace_off(model="medium", device=args.device, default_dtype="float64")
    fn = getattr(mc, "mace_omol", None)
    if fn is None:
        raise Skip("this mace-torch has no mace.calculators.mace_omol (needs a newer version)")
    return fn(device=args.device, default_dtype="float64")


def mace_method(kind):
    def run_method(cfg, frames, args, log, d3_pred):
        from pydftb import assess
        elements = sorted({s for a in frames for s in a.get_chemical_symbols()})
        if kind == "mace_off" and not set(elements) <= MACE_OFF_ELEMENTS:
            raise Skip(f"MACE-OFF covers {sorted(MACE_OFF_ELEMENTS)}, the data have "
                       f"{sorted(set(elements) - MACE_OFF_ELEMENTS)}")
        calc = _with_d3(_mace(kind, args), d3_pred)
        e0 = {el: _ase_energy_forces(calc, _single_atom(el))[0] for el in elements}
        key = {"method": kind, "model": args.mace_model if kind == "mace_mp" else None,
               "d3": d3_pred}
        preds = _cached_predictions(kind, frames, lambda a: _ase_energy_forces(calc, a),
                                    os.path.join(args.out, kind, "predictions.jsonl"), log,
                                    args.force, key)
        return _records(assess, frames, preds), e0, {"engine": kind}
    return run_method


RUNNERS = {"ptbp_published": ptbp_published, "ptbp_skf": ptbp_skf, "gfn1_xtb": gfn1_xtb,
           "mace_mp": mace_method("mace_mp"), "mace_off": mace_method("mace_off"),
           "mace_omol": mace_method("mace_omol")}


def _dispersion_plan(method, mode, have_d3):
    """(reference gets D3, prediction gets D3, note)."""
    intrinsic = method in INTRINSIC_DISPERSION
    if mode == "off":
        return False, False, "plain PBE reference" + (" (the method has dispersion)" if intrinsic else "")
    want = intrinsic if mode == "auto" else True
    if want and not have_d3:
        return False, False, "simple-dftd3 missing: plain PBE reference" + (
            " (the method has dispersion)" if intrinsic else "")
    if not want:
        return False, False, "plain against plain"
    return True, (not intrinsic), "PBE-D3(BJ) reference" + ("" if intrinsic else ", method + D3(BJ)")


def _row(name, entry):
    s = entry.get("summary") or {}
    br = s.get("by_role", {})

    def g(role, key, scale=1.0, digits=1):
        v = (br.get(role) or {}).get(key)
        return "—" if v is None else f"{v * scale:.{digits}f}"
    signs = [r.get("sign_ok") for r in s.get("reactions") or [] if r.get("decisive")]
    sign_txt = "—" if not signs else f"{sum(bool(x) for x in signs)}/{len(signs)}"
    return (f"| {name} | {entry['status']} | {s.get('tier') or '—'} | {g('test', 'relative_mae', 1e3)} | "
            f"{g('test', 'atomization_mae', 1e3)} | {g('test', 'force_mae', 1, 3)} | "
            f"{g('ood', 'relative_mae', 1e3)} | {g('ood', 'atomization_mae', 1e3)} | "
            f"{g('ood', 'force_mae', 1, 3)} | {sign_txt} | {entry.get('reason') or entry.get('note') or ''} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="a scan config (data, e0s, reactions, dftb)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=["ptbp_published", "gfn1_xtb", "mace_mp"],
                    choices=list(METHODS))
    ap.add_argument("--ptbp-skf", default=None, help="ptbp_skf: directory of full SK files")
    ap.add_argument("--run-missing", action="store_true",
                    help="ptbp_published: run the scan's published point if it is missing")
    ap.add_argument("--reference-d3", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--mace-model", default="medium", help="MACE-MP-0 size")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--xtb-max-atoms", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=1, help="processes for ptbp_skf (Linux)")
    ap.add_argument("--force", action="store_true", help="recompute cached predictions")
    args = ap.parse_args(argv)
    from pydftb import assess
    from utils import accuracy as acc
    from utils.runlog import session
    from common.models import _clean, add_d3_to_reference, brief
    cfg = acc.load_config(args.config)
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    with session("baselines", vars(args), root=os.path.join(args.out, "logs")) as log:
        frames = acc.load_data(cfg)
        e0_ref = acc.load_e0s(cfg)
        reactions = assess.load_reactions(cfg["reactions"]) if cfg.get("reactions") else None
        import importlib.util
        have_d3 = importlib.util.find_spec("dftd3") is not None
        ref_d3_frames = None
        out_path = os.path.join(args.out, "baselines.json")
        results = json.load(open(out_path)).get("methods", {}) if os.path.exists(out_path) else {}
        for name in args.methods:
            ref_d3, pred_d3, note = _dispersion_plan(name, args.reference_d3, have_d3)
            os.makedirs(os.path.join(args.out, name), exist_ok=True)
            entry = {"status": "failed", "reason": None, "note": note,
                     "dispersion": {"reference_d3": ref_d3, "prediction_d3": pred_d3,
                                    "intrinsic": name in INTRINSIC_DISPERSION}}
            try:
                with log.step("baseline", method=name):
                    fr = frames
                    if ref_d3:
                        if ref_d3_frames is None:
                            ref_d3_frames = add_d3_to_reference(frames)
                        fr = ref_d3_frames
                    rec, e0_pred, info = RUNNERS[name](cfg, fr, args, print, pred_d3)
                    summ = assess.summarize(rec, e0_ref, e0_pred, reactions)
                    summ["e0_pred"] = e0_pred
                    assess.write_markdown(summ, os.path.join(args.out, name, "summary.md"),
                                          title=f"baseline {name} ({cfg['name']})")
                    with open(os.path.join(args.out, name, "summary.json"), "w") as fh:
                        json.dump(_clean(summ), fh, indent=1)
                entry.update(status="ok", summary=brief(summ), info=info)
                log.event("baseline_done", method=name, tier=summ.get("tier"))
            except Skip as exc:
                entry.update(status="skipped", reason=str(exc))
                log.event("baseline_skipped", method=name, reason=str(exc))
            except Exception as exc:              # noqa: BLE001  (recorded, the others go on)
                entry.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                log.event("baseline_failed", method=name, error=entry["reason"])
            results[name] = entry
            with open(out_path + ".tmp", "w") as fh:
                json.dump(_clean({"config": cfg["_path"], "methods": results}), fh, indent=1)
            os.replace(out_path + ".tmp", out_path)
        lines = [f"# Baselines on {cfg['name']}", "",
                 "| method | status | tier | test rel | test atom | test F | ood rel | ood atom | ood F | decisive signs | note |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        lines += [_row(n, e) for n, e in results.items()]
        lines += ["", "energies meV/atom (rel = within-family relative, atom = absolute atomization, "
                  "each method's own isolated atoms), forces eV/A; tiers on test and ood "
                  "(pydftb.assess.TIERS)"]
        with open(os.path.join(args.out, "baselines_summary.md"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
