"""A3, the repulsive dimension (A0 plan, section 5): R3 and R4 on top of an A1 model, and the
scoring and MD stages that give the G3 gate its inputs (common/gates.py, schema a3-result/1).

* ``r3``  environment-dependent repulsive.  M8a's adaptive parameters (pydftb.adaptive) with
  only the ``repulsive`` channel, V_AB -> V_AB exp((p_A + p_B) / 2), p = Phi(CN) . theta
  (an RBF basis of the coordination number, clamped to the training range, M8d), plus optional
  pair polynomials (``--pair A-B:cutoff_A[:n[:wall_A[:strength_eV_A3]]]``).  Fitted by
  pydftb.adaptive_fit.fit_adaptive on energies and forces of the training structures; the
  electronic part is the parent's and stays fixed.
* ``r4``  joint fine-tune: the electronic channels (on-site shifts, hopping scales per shell) and
  the repulsive channel together, on energies and forces, with the band references of the
  config as a regulariser (``--band-weight``; needs the band API of pydftb.adaptive_fit).
* ``score``  assess any model directory on the config's data (R1 = an A1 export with physrep,
  R2 = an export with ``--repulsive R2b``) and write its result.json.
* ``md``  Langevin + NVE stability runs of a model (as li_o/md_check.py, for any model dir) and
  the MD field of its result.json.

Energies (user decision 2026-11-09): absolute atomization energies by default.  Every target is
put on the model's zero: E_target = E_DFT - sum E0_DFT + sum E0_model (the parent's isolated
atoms), and since fit_adaptive fits the Mermin free energy F while the A-series energies are
E(sigma->0), each target is moved by the parent's own (F - E(sigma->0)) of that structure
(exact for R3, whose electronic part is fixed).  ``--offsets none`` (default) needs the offsets
option of fit_adaptive; ``global`` is today's behaviour (one shift for all atoms) and
``per_element`` the control.  R4 changes the isolated atoms too: ``--atom-iterations`` refits
with the targets moved to the new atoms.

    python a3/a3_repulsive.py r3 --config li_o/scan_li_o.yaml --parent li_o/models/S1 \\
        --out a3/runs/r3_S1 [--jobs 64] [--pair Li-O:3.2:4]
    python a3/a3_repulsive.py r4 --config li_o/scan_li_o.yaml --parent li_o/models/S1 \\
        --out a3/runs/r4_S1 [--band-weight 1.0] [--jobs 64]
    python a3/a3_repulsive.py score --config li_o/scan_li_o.yaml --model li_o/models/S1 \\
        --out a3/runs/r1_S1 --variant R1
    python a3/a3_repulsive.py md --config li_o/scan_li_o.yaml --model a3/runs/r3_S1/model \\
        --out a3/runs/r3_S1/md [--temps 300 600]

Each stage logs to ``<out>/logs/a3-<stage>/``.  Predictions are cached per structure
(predictions.jsonl), a finished fit is kept (``--force`` refits), so a rerun continues.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import setup  # noqa: E402

setup()

from common import models as M  # noqa: E402

SYSTEMS = {"iface": "iface_init", "li16": "rat_li_bcc16_s05_train1",
           "li2o12": "rat_li2o12_s05_train1"}
EPS_E, EPS_F = 0.010, 0.10
SHELLS = "spdf"


# ---- data ---------------------------------------------------------------------------------------
def _setup(args):
    from pydftb import assess
    from utils import accuracy as acc
    cfg = acc.load_config(args.config)
    frames = acc.load_data(cfg)
    e0_ref = acc.load_e0s(cfg)
    reactions = assess.load_reactions(cfg["reactions"]) if cfg.get("reactions") else None
    return cfg, frames, e0_ref, reactions


def _family_weights(frames, idx):
    """1 / (structures in the family), scaled to a mean of 1: every family counts the same (L_EF)."""
    fam = [frames[i].info["family"] for i in idx]
    n = {f: fam.count(f) for f in set(fam)}
    w = np.array([1.0 / n[f] for f in fam])
    return w * len(w) / w.sum()


def _targets(frames, idx, e0_ref, e0_model, preds):
    """Free-energy targets per atom on the model's zero (see the module docstring)."""
    out = []
    for i in idx:
        a = frames[i]
        n = len(a)
        sym = a.get_chemical_symbols()
        e_at = a.get_potential_energy() - sum(e0_ref[s] for s in sym)
        e = e_at + sum(e0_model[s] for s in sym)
        p = preds[i]
        delta = 0.0
        if p.get("free_dftb") is not None and p.get("e_zero_dftb") is not None:
            delta = p["free_dftb"] - p["e_zero_dftb"]
        out.append((e + delta) / n)
    return np.array(out)


def _samples(frames, idx, targets, weights, dcfg):
    from pydftb.adaptive_fit import Sample
    out = []
    for i, t, w in zip(idx, targets, weights):
        a = frames[i]
        try:
            f = np.array(a.get_forces())
        except Exception:                     # noqa: BLE001
            f = None
        out.append(Sample(a, float(t), M.engine_kpoints(a, dcfg), float(w), a.info["name"], f))
    return out


def _cn_range(element, frames, idx):
    from pydftb.adaptive import AdaptiveParams
    from pydftb.constants import BOHR
    probe = AdaptiveParams(element)
    vals = []
    for i in idx:
        a = frames[i]
        if element not in a.get_chemical_symbols():
            continue
        periodic = bool(np.any(a.pbc))
        cn, _ = probe.coordination_numbers(a.get_chemical_symbols(), a.positions / BOHR,
                                           a.cell.array / BOHR if periodic else None,
                                           np.ones(3, bool) if periodic else None)
        sym = np.array(a.get_chemical_symbols())
        vals += list(cn[sym == element])
    return (min(vals), max(vals)) if vals else (0.0, 0.0)


def _basis(element, frames, idx, n_centres, width):
    """RBF of CN with centres over the training range and the M8d clamp to that range."""
    from pydftb.adaptive import RBF
    lo, hi = _cn_range(element, frames, idx)
    if n_centres <= 0 or hi - lo < 1e-6:
        return RBF((), 1.5, True), (lo, hi)
    centres = np.linspace(lo, hi, n_centres) if n_centres > 1 else np.array([0.5 * (lo + hi)])
    w = width or (float(centres[1] - centres[0]) if n_centres > 1 else max(1.0, hi - lo))
    d = min(0.5, (hi - lo) / 4.0)
    clamp = (lo, hi, d) if hi - lo >= 2 * d and d > 0 else None
    return RBF(tuple(float(c) for c in centres), float(w), True, clamp), (lo, hi)


def _pairs(specs, element):
    """--pair A-B:cutoff_A[:n[:wall_A[:strength eV/A^3]]] owned by ``element`` (= A)."""
    from pydftb.adaptive import PairRepulsive
    from pydftb.constants import BOHR, HARTREE
    out = {}
    for spec in specs or []:
        pair, *vals = spec.split(":")
        a, b = pair.split("-")
        if a != element:
            continue
        cut = float(vals[0]) / BOHR
        n = int(vals[1]) if len(vals) > 1 else 4
        wall = float(vals[2]) / BOHR if len(vals) > 2 else 0.0
        strength = float(vals[3]) / HARTREE * BOHR ** 3 if len(vals) > 3 else 0.0
        out[b] = PairRepulsive(cut, (0.0,) * n, wall, strength)
    return out


def _shells(element, dcfg, skdir):
    """The shells of an element in the model's basis (the SK set with the section's max_l)."""
    sk = M.engine_skset(skdir, dcfg, [element])
    lmax = int(np.max(np.atleast_1d(sk.max_l[element])))
    return SHELLS[:lmax + 1]


def _models(variant, elements, frames, idx, args, dcfg, skdir):
    from pydftb.adaptive import AdaptiveParams
    out, info = [], {}
    for el in elements:
        basis, rng = _basis(el, frames, idx, args.centres, args.width)
        n = basis.size
        kw = dict(basis=basis, repulsive=(0.0,) * n, pair_repulsive=_pairs(args.pair, el))
        if variant == "r4":
            shells = _shells(el, dcfg, skdir)
            if "onsite" in args.channels:
                kw["onsite"] = {s: (0.0,) * n for s in shells}
            if "hopping" in args.channels:
                kw["hopping"] = {s: (0.0,) * n for s in shells}
            if "repulsive" not in args.channels:
                kw["repulsive"] = None
        out.append(AdaptiveParams(el, **kw))
        info[el] = {"cn_range": rng, "basis": {"centres": list(basis.centres), "width": basis.width,
                                               "clamp": basis.clamp}}
    return out, info


# ---- the fit -------------------------------------------------------------------------------------
def _fit_api():
    from pydftb import adaptive_fit as af
    params = inspect.signature(af.fit_adaptive).parameters
    return af, {"offsets": "offsets" in params, "bands": "band_samples" in params}


def _band_samples(cfg, dcfg):
    from pydftb.adaptive_fit import BandSample
    from utils import accuracy as acc
    out = []
    for ref in acc.load_refs(cfg):
        at = ref.atoms()
        at.info["kspacing"] = ref.meta.get("kspacing", dcfg.get("kspacing", 0.15))
        out.append(BandSample(ref, kpts=M.engine_kpoints(at, dcfg), weight=1.0,
                              group=ref.meta.get("group") or ref.name))
    return out


def fit_stage(variant, args):
    from pydftb import assess
    from utils.runlog import session
    cfg, frames, e0_ref, reactions = _setup(args)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    with session(f"a3-{variant}", vars(args), root=os.path.join(out, "logs")) as log:
        parent = M.load_model(args.parent, dftb=cfg["dftb"])
        if parent.env_models:
            raise SystemExit(f"{args.parent} already has environment models; {variant} starts from "
                             "a model without them (an A1 export)")
        if parent.delta:
            raise SystemExit("the parent has a Delta model; A3 works on DFTB models")
        dcfg = parent.dftb
        af, api = _fit_api()
        if args.offsets != "global" and not api["offsets"]:
            raise SystemExit("--offsets none/per_element needs fit_adaptive(offsets=...) "
                             "(pydftb.adaptive_fit of the A2 round); use --offsets global with "
                             "this pydftb")
        if variant == "r4" and args.band_weight > 0 and not api["bands"]:
            raise SystemExit("r4 with --band-weight needs fit_adaptive(band_samples=..., "
                             "band_weight=...) and BandSample (pydftb.adaptive_fit of the A2 "
                             "round); or run r4 with --band-weight 0")
        elements = args.elements or sorted({s for a in frames for s in a.get_chemical_symbols()})
        fit_idx = [i for i, a in enumerate(frames) if a.info["role"] in args.roles]
        log.event("data", structures=len(frames), fit=len(fit_idx), elements=elements)
        # the parent on every structure (cached): baseline row and the F - E(sigma->0) shifts
        with log.step("parent"):
            preds = M.predict(parent, frames, os.path.join(out, "parent", "predictions.jsonl"),
                              args.jobs, print)
            e0_parent = parent.atom_energies(elements)
            ps = assess.summarize(M.records(frames, preds), e0_ref, e0_parent, reactions)
            ps["e0_pred"] = e0_parent
            assess.write_markdown(ps, os.path.join(out, "parent", "summary.md"),
                                  title=f"parent {parent.path}")
        bad = [frames[i].info["name"] for i in fit_idx if preds[i]["error"]]
        if bad:
            log.event("parent_failed", names=bad[:20])
            fit_idx = [i for i in fit_idx if not preds[i]["error"]]
        model_dir = os.path.join(out, "model")
        if os.path.exists(os.path.join(model_dir, M.MODEL_FILE)) and not args.force:
            log.event("fit_skipped", reason="model exists (use --force to refit)")
            model = M.load_model(model_dir)
            fit_info = model.meta.get("fit", {})
        else:
            weights = _family_weights(frames, fit_idx)
            models, basis_info = _models(variant, elements, frames, fit_idx, args, dcfg,
                                         parent.skdir)
            n_comp = [3 * len(frames[i]) for i in fit_idx]
            fw = ((EPS_E / EPS_F) ** 2 / float(np.mean(n_comp)) if args.force_weight == "auto"
                  else float(args.force_weight))
            sk = M.engine_skset(parent.skdir, dcfg, elements)
            options = M.engine_options(dcfg, repulsive=parent.repulsive, for_fit=True)
            kw = dict(l2=args.l2, processes=args.jobs, max_evaluations=args.max_evaluations,
                      verbose=True, force_weight=fw)
            if api["offsets"]:
                kw["offsets"] = args.offsets
            if variant == "r4" and args.band_weight > 0:
                kw.update(band_samples=_band_samples(cfg, dcfg), band_weight=args.band_weight,
                          band_settings=(cfg.get("bands") or {}).get("settings"))
            e0_model = dict(e0_parent)
            x0 = None
            history = []
            iterations = args.atom_iterations if variant == "r4" else 1
            fit = None
            for it in range(max(1, iterations)):
                targets = _targets(frames, fit_idx, e0_ref, e0_model, preds)
                samples = _samples(frames, fit_idx, targets, weights, dcfg)
                with log.step("fit", variant=variant, iteration=it, n_samples=len(samples),
                              n_params=int(sum(len(m.vector()) for m in models)), force_weight=fw):
                    fit = af.fit_adaptive(models, samples, sk, options, x0=x0, **kw)
                history.append({"iteration": it, "rmse": fit.rmse, "force_rmse": fit.force_rmse,
                                "n_evaluations": fit.n_evaluations, "seconds": fit.seconds})
                log.event("fit_done", iteration=it, rmse_meV=1e3 * fit.rmse,
                          force_rmse=fit.force_rmse, evaluations=fit.n_evaluations)
                models, x0 = list(fit.models), fit.x
                if it + 1 < iterations:              # R4: the isolated atoms moved with the model
                    tmp = M.Model(out, parent.skdir, parent.repulsive, dcfg, tuple(models))
                    e0_model = tmp.atom_energies(elements)
                    log.event("atoms_updated", e0=e0_model)
            fit_info = {"variant": variant, "elements": elements, "roles": list(args.roles),
                        "force_weight": fw, "l2": args.l2, "offsets": args.offsets,
                        "band_weight": args.band_weight if variant == "r4" else 0.0,
                        "channels": list(args.channels) if variant == "r4" else ["repulsive"],
                        "pairs": list(args.pair or []), "basis": basis_info,
                        "rmse": fit.rmse, "force_rmse": fit.force_rmse, "offset": fit.offset,
                        "band_loss": getattr(fit, "band_loss", None),
                        "offsets_fitted": getattr(fit, "offsets", None),
                        "x": list(np.asarray(fit.x, float)), "iterations": history,
                        "trace": [list(h) for h in fit.history[-50:]]}
            model = M.write_model(model_dir, parent.skdir, dcfg, parent.repulsive, models,
                                  meta={"stage": "A3", "variant": variant.upper(),
                                        "parent": parent.path, "fit": M._clean(fit_info)})
        result = _score_and_write(model, frames, e0_ref, reactions, out, variant.upper(),
                                  args.label or f"{variant.upper()} on {os.path.basename(parent.path)}",
                                  parent.path, args, log, fit=fit_info, parent_summary=ps)
        log.event("result", tier=result["tier"])
    return 0


def _score_and_write(model, frames, e0_ref, reactions, out, variant, label, parent, args, log,
                     fit=None, parent_summary=None):
    with log.step("score", model=model.path):
        summ = M.score(model, frames, e0_ref, reactions, out_dir=out, title=f"A3 {label}",
                       dispersion=getattr(args, "dispersion", False), jobs=args.jobs, log=print,
                       force=getattr(args, "force_predict", False))
    result = {"schema": "a3-result/1", "system": args.system, "variant": variant, "label": label,
              "parent": parent, "model_dir": model.path, "summary": M.brief(summ),
              "tier": summ.get("tier"), "md": None, "fit": fit,
              "parent_summary": None if parent_summary is None else M.brief(parent_summary)}
    path = os.path.join(out, "result.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    if old.get("md") and old.get("model_dir") == model.path:
        result["md"] = old["md"]                    # keep an MD check of the same model
    with open(path + ".tmp", "w") as fh:
        json.dump(M._clean(result), fh, indent=1)
    os.replace(path + ".tmp", path)
    br = summ["energies"]["by_role"]
    lines = [f"# A3 {label}", "", f"model `{model.path}`; tier **{summ.get('tier') or 'none'}**", "",
             "| role | relative MAE (meV/atom) | atomization MAE | force MAE (eV/A) | failed |",
             "|---|---|---|---|---|"]
    for role, b in br.items():
        lines.append(f"| {role} | {M._fmt1(b.get('relative_mae'))} | {M._fmt1(b.get('atomization_mae'))} | "
                     f"{b.get('force_mae', float('nan')):.3f} | {b.get('failed', 0)} |")
    if parent_summary is not None:
        lines += ["", "parent (same scoring):", ""]
        for role, b in parent_summary["energies"]["by_role"].items():
            lines.append(f"- {role}: relative {M._fmt1(b.get('relative_mae'))} meV/atom, "
                         f"atomization {M._fmt1(b.get('atomization_mae'))}, force "
                         f"{b.get('force_mae', float('nan')):.3f}")
    with open(os.path.join(out, "result.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return result


def score_stage(args):
    from utils.runlog import session
    cfg, frames, e0_ref, reactions = _setup(args)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    with session("a3-score", vars(args), root=os.path.join(out, "logs")) as log:
        model = M.load_model(args.model, dftb=cfg["dftb"])
        res = _score_and_write(model, frames, e0_ref, reactions, out, args.variant,
                               args.label or f"{args.variant} {os.path.basename(model.path)}",
                               model.meta.get("parent"), args, log)
        log.event("result", tier=res["tier"])
    return 0


def md_stage(args):
    from utils import accuracy as acc
    from utils.runlog import session
    cfg = acc.load_config(args.config)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    with session("a3-md", vars(args), root=os.path.join(out, "logs")) as log:
        model = M.load_model(args.model, dftb=cfg["dftb"])
        frames = {f.info["name"]: f for f in acc.load_data(cfg)}
        path = os.path.join(out, "md_results.json")
        results = json.load(open(path)) if os.path.exists(path) else {}
        for sysname in args.systems:
            name = SYSTEMS.get(sysname, sysname)
            if name not in frames:
                raise SystemExit(f"{name} is not in the config's data")
            for t in args.temps:
                key = f"{sysname}_{int(t)}K"
                if key in results and not args.force:
                    continue
                with log.step("md", system=sysname, temperature=t):
                    r = M.run_nve(model, frames[name], t, args.langevin, args.nve, args.dt,
                                  os.path.join(out, f"{key}.traj"), M.LI_O_COLLAPSE)
                results[key] = M._clean(r)
                with open(path + ".tmp", "w") as fh:
                    json.dump(results, fh, indent=1)
                os.replace(path + ".tmp", path)
                log.event("md_result", run=key, drift=r.get("drift_meV_per_atom_per_ps"),
                          collapse=r["collapse"], failed=r["failed"])
        stable = all(r.get("stable") for r in results.values())
        result_file = args.result or os.path.join(os.path.dirname(model.path), "result.json")
        if os.path.exists(result_file):
            res = json.load(open(result_file))
            res["md"] = {"stable": stable, "runs": results, "dir": out}
            with open(result_file + ".tmp", "w") as fh:
                json.dump(res, fh, indent=1)
            os.replace(result_file + ".tmp", result_file)
            log.event("result_updated", file=result_file, stable=stable)
        lines = [f"# MD check of {model.path}", "",
                 "| run | drift (meV/atom/ps) | shortest Li-Li | Li-O | O-O | collapse | failed |",
                 "|---|---|---|---|---|---|---|"]
        for k, r in results.items():
            s = r.get("shortest", {})
            lines.append(f"| {k} | {r.get('drift_meV_per_atom_per_ps') or float('nan'):.3f} | "
                         f"{s.get('Li-Li') or float('nan'):.2f} | {s.get('Li-O') or float('nan'):.2f} | "
                         f"{s.get('O-O') or float('nan'):.2f} | {r.get('collapse')} | {r.get('failed') or ''} |")
        with open(os.path.join(out, "md_summary.md"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(p):
        p.add_argument("--config", required=True, help="scan config (data, e0s, reactions, dftb)")
        p.add_argument("--out", required=True)
        p.add_argument("--system", default="li_o", help="label for the gates (a3-result/1)")
        p.add_argument("--jobs", type=int, default=1, help="processes (fit and predictions)")
        p.add_argument("--label", default=None)
        p.add_argument("--dispersion", action="store_true",
                       help="score with D3 on both sides (the ink); default plain vs plain")

    for variant in ("r3", "r4"):
        p = sub.add_parser(variant, help=f"fit {variant.upper()} on a parent model")
        common(p)
        p.add_argument("--parent", required=True, help="an A1 export (ptbp scan export)")
        p.add_argument("--elements", nargs="*", default=None, help="default: all of the data")
        p.add_argument("--roles", nargs="+", default=["train"], help="roles fitted on")
        p.add_argument("--centres", type=int, default=3, help="RBF centres over the CN range")
        p.add_argument("--width", type=float, default=None, help="RBF width (default: spacing)")
        p.add_argument("--pair", nargs="*", default=None,
                       help="pair polynomials A-B:cutoff_A[:n[:wall_A[:strength_eV_A3]]]")
        p.add_argument("--force-weight", default="auto",
                       help="(eV/atom)^2 per (eV/A)^2 per component; auto = (0.01/0.1)^2 / <3N>")
        p.add_argument("--l2", type=float, default=1e-4)
        p.add_argument("--offsets", default="none", choices=["none", "global", "per_element"])
        p.add_argument("--max-evaluations", type=int, default=60)
        p.add_argument("--force", action="store_true", help="refit even if the model exists")
        p.add_argument("--force-predict", action="store_true", help="recompute cached predictions")
        if variant == "r4":
            p.add_argument("--channels", nargs="+", default=["onsite", "hopping", "repulsive"],
                           choices=["onsite", "hopping", "repulsive"])
            p.add_argument("--band-weight", type=float, default=1.0)
            p.add_argument("--atom-iterations", type=int, default=2)
        p.set_defaults(func=lambda a, v=variant: fit_stage(v, a))
    p = sub.add_parser("score", help="assess a model directory (R1/R2 rows)")
    common(p)
    p.add_argument("--model", required=True)
    p.add_argument("--variant", default="R1", choices=["R1", "R2", "R3", "R4"])
    p.add_argument("--force-predict", action="store_true")
    p.set_defaults(func=score_stage)
    p = sub.add_parser("md", help="MD stability of a model (Langevin + NVE)")
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    p.add_argument("--temps", nargs="+", type=float, default=[300.0, 600.0])
    p.add_argument("--langevin", type=int, default=300)
    p.add_argument("--nve", type=int, default=2000)
    p.add_argument("--dt", type=float, default=1.0, help="fs")
    p.add_argument("--result", default=None, help="result.json to update (default: next to the model)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=md_stage)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
