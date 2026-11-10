"""A2: the environment-dependent electronic part (A0 plan, section 4), on top of a finished A1
scan.  The repulsive is absent while fitting (bands only, as the plan says: "先从能带做起，忽略
rep"); every fitted electronic part is then given the A1 repulsives again (physrep sigma by
L_EF and the R2b spline control, exactly the scan's scoring), which is the transfer check:
do better bands give better energies and forces?

Channels (pydftb):

* **E3** :class:`pydftb.adaptive.AdaptiveParams` with a descriptor and a regressor: on-site
  shifts and hopping scales per shell as functions of every atom's environment (M8a);
* **E2** :class:`pydftb.r0mix.R0Switch`: the confinement radii r0 follow the environment; the SK
  tables of a line of r0 nodes around the A1 point are mixed with C2 B-spline weights (a typed
  SK set: hotcent, run ``prepare`` on the server);
* **E1** :class:`pydftb.typeswitch.OnsiteSwitch`: M5a's CN switching between two types (copies
  of the A1 tables) whose on-site energies (and optionally U) are shifted per type.

Descriptor ladder L1-L4 (:mod:`pydftb.descriptors`), regressors linear / kernel / mlp
(:mod:`pydftb.regressors`).  Configs: ``a2_li.yaml``, ``a2_li_o.yaml``, ``a2_ink.yaml``.

Stages (each logs to ``<out>/logs/a2-<stage>/`` and resumes; ``--force`` recomputes)::

    python a2/a2.py plan     a2/a2_li_o.yaml                    # runs, band groups, SK costs
    python a2/a2.py prepare  a2/a2_li_o.yaml [--run R] [--dry-run]   # SK sets (E2: hotcent)
    python a2/a2.py fit      a2/a2_li_o.yaml --run R [--holdout G|all|full]
    python a2/a2.py transfer a2/a2_li_o.yaml --run R [--jobs 8]
    python a2/a2.py summary  a2/a2_li_o.yaml                    # a2-result/1 + summary.md
    python a2/a2.py status   a2/a2_li_o.yaml

Outputs under ``out``: ``runs/<run>/full/`` (the fit on every non-test band group:
``fit.json``, ``env_models.json``, ``bands.json``, the model directory ``model/``),
``runs/<run>/loeo-<group>/`` (leave-one-environment-out fits), ``runs/<run>/transfer/``
(``predict.jsonl``, a pseudo scan point ``points/p00000/`` and its ``losses.json``,
``transfer.json``), ``result_<channel>.json`` (schema ``a2-result/1`` of common/gates.py),
``summary.md``.
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import setup  # noqa: E402

setup()

CHANNELS = ("E1", "E2", "E3")


# ---- configuration ---------------------------------------------------------------------------------
def load_config(path: str) -> dict:
    import yaml
    from utils import accuracy as acc
    path = os.path.abspath(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    base = os.path.dirname(path)

    def absolute(p):
        p = os.path.expandvars(os.path.expanduser(str(p)))
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    cfg["_path"] = path
    cfg["scan"] = absolute(cfg["scan"])
    cfg["out"] = absolute(cfg.get("out", os.path.join("runs", cfg["name"])))
    if cfg.get("k6"):
        cfg["k6"] = absolute(cfg["k6"])
    typed = cfg.setdefault("typed_sk", {})
    typed["cache"] = absolute(os.environ.get("PTBP_SKCACHE_TYPED")
                              or typed.get("cache", os.path.join(cfg["out"], "typed_sk")))
    cfg["_scan"] = acc.load_config(cfg["scan"])
    cfg.setdefault("bands", {})
    cfg.setdefault("fit", {})
    names = [run_name(r) for r in cfg.get("runs", [])]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: run names must be unique: {names}")
    for r in cfg.get("runs", []):
        if r["channel"] not in CHANNELS:
            raise ValueError(f"run {run_name(r)}: channel must be one of {CHANNELS}")
    return cfg


def run_name(spec: dict) -> str:
    if spec.get("name"):
        return str(spec["name"])
    if spec["channel"] == "E1":
        return "E1"
    return f"{spec['channel']}-{spec['level']}-{spec.get('regressor', 'linear')}"


def run_spec(cfg, name: str) -> dict:
    for r in cfg.get("runs", []):
        if run_name(r) == name:
            return r
    raise SystemExit(f"no run {name!r} in {cfg['_path']} (have: "
                     f"{[run_name(r) for r in cfg.get('runs', [])]})")


def spec_hash(cfg, spec: dict) -> str:
    """What a fit depends on (the run, its channel/descriptor/regressor sections, the fit and
    band settings, the base point): a changed config refits instead of reusing."""
    keys = {"run": spec, "channel": cfg.get("channels", {}).get(spec["channel"]),
            "level": cfg.get("descriptors", {}).get(spec.get("level")),
            "regressor": cfg.get("regressors", {}).get(spec.get("regressor", "linear")),
            "fit": cfg["fit"], "bands": cfg["bands"], "base": cfg.get("base")}
    return hashlib.sha1(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_dir(cfg, name: str, part: str = "full") -> str:
    return os.path.join(cfg["out"], "runs", name, part)


# ---- the A1 point the models start from ----------------------------------------------------------------
def base_point(cfg) -> dict:
    """{"index", "label", "radii", "skdir", "result", "losses"} of the A1 point (``base``:
    ``best:<strategy>`` or a point index)."""
    from utils import accuracy as acc
    scan = cfg["_scan"]
    spec = cfg.get("base", "best:S3")
    if isinstance(spec, str) and spec.startswith("best:"):
        strategy = spec.split(":", 1)[1]
        best = acc.best_points(acc._scored(scan), [strategy])
        if strategy not in best:
            raise SystemExit(f"no scored point for {strategy} in {scan['out']}: run "
                             "`ptbp scan run/losses` first")
        index, label = int(best[strategy]["index"]), f"A1 {strategy} point"
    else:
        index, label = int(spec), "A1 point"
    with open(os.path.join(acc.point_dir(scan, index), "result.json")) as fh:
        res = json.load(fh)
    losses = None
    lp = os.path.join(acc.point_dir(scan, index), "losses.json")
    if os.path.exists(lp):
        with open(lp) as fh:
            losses = json.load(fh)
    return {"index": index, "label": f"{label} {index}", "radii": res["radii"],
            "skdir": res["skdir"], "result": res, "losses": losses}


# ---- band references and their environment groups ----------------------------------------------------
def band_samples(cfg, refs=None):
    """[BandSample] of every band reference of the scan (k-points of its SCC as the scan's)."""
    from pydftb.adaptive_fit import BandSample
    from utils import accuracy as acc
    from common import models as M
    scan = cfg["_scan"]
    dcfg = scan["dftb"]
    out = []
    for ref in (refs if refs is not None else acc.load_refs(scan)):
        at = ref.atoms()
        at.info["kspacing"] = ref.meta.get("kspacing", dcfg.get("kspacing", 0.15))
        group = ref.meta.get("group") or ref.name
        out.append(BandSample(ref, kpts=M.engine_kpoints(at, dcfg), weight=1.0, group=group))
    if not out:
        raise SystemExit(f"{scan['_path']}: no band references (bands.refs); run D1 first")
    return out


def groups_of(samples) -> list:
    return sorted({s.group for s in samples})


def test_groups(cfg, samples) -> set:
    """Band groups never fitted: ``bands.test_groups``, plus every group with a reference
    whose ``meta["role"]`` is in ``bands.test_roles`` (the ink's orbital references carry the
    data's roles)."""
    out = set(cfg["bands"].get("test_groups") or [])
    roles = set(cfg["bands"].get("test_roles") or [])
    out |= {s.group for s in samples if s.ref.meta.get("role") in roles}
    return out


def split(cfg, samples, holdout: str | None):
    """(fit samples, held-out samples, test samples): test groups are never fitted; holdout =
    one more group left out (leave-one-environment-out)."""
    test = test_groups(cfg, samples)
    fit = [s for s in samples if s.group not in test and s.group != holdout]
    held = [s for s in samples if s.group == holdout]
    tst = [s for s in samples if s.group in test]
    return fit, held, tst


def band_settings(cfg):
    from pydftb.bandloss import BandSettings
    d = dict((cfg["_scan"].get("bands") or {}).get("settings") or {})
    d.update(cfg["bands"].get("settings") or {})
    return BandSettings.from_dict(d)


def loss_of(results, groups=None) -> float | None:
    """pydftb.bandloss.band_loss over the results of these groups (None if none is finite)."""
    from pydftb.bandloss import band_loss
    rs = [r for r in results if groups is None or r.get("group") in groups]
    rs = [{**r, "loss": np.nan if r.get("loss") is None else r["loss"]} for r in rs]
    if not rs or not any(np.isfinite(r["loss"]) for r in rs):
        return None
    return float(band_loss(rs))


# ---- models of a run ------------------------------------------------------------------------------------
def training_structures(cfg, fit_samples) -> list:
    """Structures the descriptor transforms and kernel centres are fitted on: the fitted band
    references and the scan's training data."""
    from utils import accuracy as acc
    frames = [s.ref.atoms() for s in fit_samples]
    frames += [a for a in acc._data_cache(cfg["_scan"]) if a.info.get("role") == "train"]
    return frames


def build_descriptor(cfg, level: str, element: str, frames):
    from pydftb import descriptors as D
    lcfg = cfg.get("descriptors", {}).get(level)
    if not lcfg or element not in lcfg:
        raise SystemExit(f"descriptors.{level}.{element} is not in {cfg['_path']}")
    parts = []
    for spec in lcfg[element]:
        d = D.descriptor_from_dict({**spec, "element": element})
        if getattr(d, "kind", None) == "mace" and getattr(d, "size", None) is None:
            d = d.with_size(next(a for a in frames if element in a.get_chemical_symbols()))
        parts.append(d)
    desc = parts[0] if len(parts) == 1 else D.Stack(tuple(parts))
    mine = [a for a in frames if element in a.get_chemical_symbols()]
    transform = lcfg.get("transform", "standardize")
    if transform == "standardize":
        desc = D.Affine.standardize(desc, mine)
    elif transform == "pca":
        desc = D.Affine.pca(desc, mine, lcfg.get("n_components"), float(lcfg.get("variance", 0.99)))
    elif transform not in (None, "none"):
        raise SystemExit(f"descriptors.{level}.transform must be standardize, pca or none")
    return desc, mine


def build_regressor(cfg, name: str, desc, frames, n_out: int):
    from pydftb import descriptors as D
    from pydftb import regressors as R
    spec = dict(cfg.get("regressors", {}).get(name) or {"kind": name})
    kind = spec.pop("kind", name)
    if kind == "linear":
        return R.Linear(desc.size, n_out)
    if kind == "kernel":
        x = D.training_features(desc, frames)
        return R.Kernel.from_features(x, int(spec.get("n_inducing", 24)), n_out,
                                      spec.get("length_scale", "median"), seed=int(spec.get("seed", 0)))
    if kind == "mlp":
        return R.MLP(desc.size, n_out, tuple(int(h) for h in spec.get("hidden", [8])),
                     seed=int(spec.get("seed", 0)))
    raise SystemExit(f"regressor kind {kind!r}: use linear, kernel or mlp")


def build_models(cfg, spec, base, fit_samples):
    """(models, labels of the SK set, SK directory spec, dftb settings) of a run."""
    from pydftb.r0mix import R0Switch, labels_max_l, node_radii
    from pydftb.typeswitch import OnsiteSwitch
    scan = cfg["_scan"]
    dcfg = dict(scan["dftb"])
    elements = sorted({s for a in training_structures(cfg, fit_samples)
                       for s in a.get_chemical_symbols()})
    ch = spec["channel"]
    ccfg = cfg.get("channels", {}).get(ch) or {}
    if not ccfg:
        raise SystemExit(f"channels.{ch} is not in {cfg['_path']}")
    frames = training_structures(cfg, fit_samples)
    models = []
    if ch == "E3":
        from pydftb.adaptive import AdaptiveParams
        for el, chans in ccfg.items():
            desc, mine = build_descriptor(cfg, spec["level"], el, frames)
            reg = build_regressor(cfg, spec.get("regressor", "linear"), desc, mine, len(chans))
            models.append(AdaptiveParams(el, descriptor=desc, regressor=reg, channels=tuple(chans)))
        return models, elements, {"kind": "base", "skdir": base["skdir"]}, dcfg
    if ch == "E2":
        for el, ecfg in ccfg.items():
            desc, mine = build_descriptor(cfg, spec["level"], el, frames)
            reg = build_regressor(cfg, spec.get("regressor", "linear"), desc, mine, 1)
            models.append(R0Switch.around(el, tuple(base["radii"][el][:2]), ecfg["spacing"],
                                          int(ecfg.get("n", 7)), descriptor=desc, regressor=reg))
        fixed = {el: (el, *base["radii"][el][:2]) for el in elements
                 if el not in {m.element for m in models}}
        from pydftb.system import element_of_label
        from utils import accuracy as acc
        power = acc.space_of(scan).p              # the scan's confinement powers (published by default)
        types = {lab: (spec_[0], float(spec_[1]), float(spec_[2]), float(power[element_of_label(lab)]))
                 for lab, spec_ in node_radii(models, fixed).items()}
        dcfg["max_l"] = labels_max_l(models, dcfg.get("max_l") or {})
        return models, sorted(types), {"kind": "typed", "types": types}, dcfg
    # E1: two types per switched element (copies of the A1 tables), shifts fitted
    base_of = {el: el for el in elements}
    for el, ecfg in ccfg.items():
        types = (f"{el}1", f"{el}2")             # low CN first
        for t in types:
            base_of[t] = el
        base_of.pop(el, None)
        shells = list(ecfg.get("shells", ["s"]))
        shift = {t: {s: 0.0 for s in shells} for t in types}
        hub = {t: {s: 0.0 for s in shells} for t in types} if ecfg.get("hubbard") else ()
        models.append(OnsiteSwitch(el, types, ecfg["centre"], ecfg["width"], ecfg.get("cutoff"),
                                   onsite_shift=shift, hubbard_shift=hub))
    dcfg["max_l"] = labels_max_l(models, dcfg.get("max_l") or {})
    return models, sorted(base_of), {"kind": "copies", "base_of": base_of,
                                     "skdir": base["skdir"]}, dcfg


def sk_directory(cfg, skspec, create: bool = True, log=print):
    """The SK directory of a run (E3: the A1 point's; E1: type copies; E2: the typed cache)."""
    if skspec["kind"] == "base":
        return skspec["skdir"]
    if skspec["kind"] == "copies":
        from pydftb.skf import write_typed_copies
        key = hashlib.sha1(json.dumps(skspec, sort_keys=True).encode()).hexdigest()[:12]
        d = os.path.join(cfg["out"], "sk_copies", key)
        if not os.path.exists(os.path.join(d, "done")) and create:
            write_typed_copies(skspec["skdir"], d, skspec["base_of"])
            with open(os.path.join(d, "done"), "w") as fh:
                json.dump(skspec, fh, indent=1)
        return d
    cache = _typed_cache(cfg)
    if not create:
        return cache.path(skspec["types"]) if cache.has(skspec["types"]) else None
    return cache.get(skspec["types"], log=log)


def typed_plan(cfg, skspec) -> dict | None:
    if skspec["kind"] != "typed":
        return None
    return _typed_cache(cfg).plan(skspec["types"])


def _typed_cache(cfg):
    """The typed SK cache with the scan's hotcent settings (xc, superposition, confinement,
    integration grid); ``typed_sk.workers`` overrides the hotcent processes per set."""
    from utils.skgrid import SKSettings, TypedSKCache
    sk = {k: v for k, v in cfg["_scan"]["sk"].items() if k != "cache"}
    if cfg["typed_sk"].get("workers"):
        sk["workers"] = int(cfg["typed_sk"]["workers"])
    return TypedSKCache(cfg["typed_sk"]["cache"], SKSettings.from_config(sk))


# ---- stages ---------------------------------------------------------------------------------------------
def stage_plan(cfg, args, log):
    from utils import accuracy as acc
    base = base_point(cfg)
    samples = band_samples(cfg)
    test = test_groups(cfg, samples)
    groups = groups_of(samples)
    print(f"base: {base['label']}  radii {base['radii']}")
    print(f"band references: {len(samples)} in {len(groups)} groups "
          f"({', '.join(g + (' [test]' if g in test else '') for g in groups)})")
    rows = []
    for spec in cfg.get("runs", []):
        name = run_name(spec)
        fit_s, _, _ = split(cfg, samples, None)
        models, labels, skspec, _ = build_models(cfg, spec, base, fit_s)
        from pydftb.envmodel import vector_of
        n = len(vector_of(models))
        plan = typed_plan(cfg, skspec)
        done = os.path.exists(os.path.join(run_dir(cfg, name), "fit.json"))
        rows.append({"run": name, "channel": spec["channel"], "n_params": n, "labels": labels,
                     "typed_sk": plan, "fitted": done})
        cost = "" if plan is None else (f"; typed SK set: {plan.get('tables')} tables, about "
                                        f"{plan.get('core_hours', float('nan')):.2f} core-hours"
                                        f"{' (cached)' if plan.get('cached') else ''}")
        print(f"  {name:22s} {n:4d} parameters, labels {','.join(labels)}"
              f"{' [fitted]' if done else ''}{cost}")
    n_fit = len([g for g in groups if g not in test])
    print(f"fits per run: 1 full + {n_fit} leave-one-environment-out; per fit "
          f"<= {cfg['fit'].get('max_evaluations', 40)} evaluations x {len(samples)} band SCCs")
    os.makedirs(cfg["out"], exist_ok=True)
    with open(os.path.join(cfg["out"], "plan.json"), "w") as fh:
        json.dump(acc._clean({"base": {k: base[k] for k in ("index", "label", "radii", "skdir")},
                              "groups": groups, "test_groups": sorted(test), "runs": rows}), fh,
                  indent=1)
    log.event("plan", runs=len(rows), groups=len(groups))


def stage_prepare(cfg, args, log):
    base = base_point(cfg)
    samples = band_samples(cfg)
    fit_s, _, _ = split(cfg, samples, None)
    for spec in cfg.get("runs", []):
        name = run_name(spec)
        if args.run and name != args.run:
            continue
        _, labels, skspec, _ = build_models(cfg, spec, base, fit_s)
        plan = typed_plan(cfg, skspec)
        if args.dry_run:
            print(f"{name}: {skspec['kind']} SK set" + (f", plan {json.dumps(plan)}" if plan else ""))
            continue
        with log.step("prepare", run=name, sk=skspec["kind"]):
            d = sk_directory(cfg, skspec, create=True, log=print)
        print(f"{name}: SK set {d}")


def fit_one(cfg, spec, holdout, args, log):
    """One fit: the full fit (holdout None) or one leave-one-environment-out fit."""
    from pydftb.adaptive_fit import evaluate_bands, fit_adaptive
    from pydftb.envmodel import parameter_names, save_models, vector_of
    from utils import accuracy as acc
    from common import models as M
    name = run_name(spec)
    part = "full" if holdout is None else f"loeo-{holdout}"
    out = run_dir(cfg, name, part)
    h = spec_hash(cfg, spec)
    done = os.path.join(out, "fit.json")
    if os.path.exists(done) and not args.force:
        with open(done) as fh:
            prev = json.load(fh)
        if prev.get("hash") == h and prev.get("status") == "ok":
            print(f"{name}/{part}: done (fit.json), skipped; --force refits")
            return prev
    os.makedirs(out, exist_ok=True)
    base = base_point(cfg)
    samples = band_samples(cfg)
    fit_s, held_s, test_s = split(cfg, samples, holdout)
    if not fit_s:
        raise SystemExit(f"{name}/{part}: no band groups left to fit")
    models, labels, skspec, dcfg = build_models(cfg, spec, base, fit_s)
    skdir = sk_directory(cfg, skspec, create=skspec["kind"] != "typed")   # copies are cheap
    if skdir is None:
        raise SystemExit(f"{name}: its typed SK set is not there yet; run `a2.py prepare "
                         f"--run {name}` first (hotcent, on the server)")
    sk = M.engine_skset(skdir, dcfg, labels)
    options = M.engine_options(dcfg, for_fit=True)
    fcfg = cfg["fit"]
    settings = band_settings(cfg)
    t0 = time.perf_counter()
    with log.step("fit", run=name, part=part, n_params=len(vector_of(models)),
                  n_bands=len(fit_s)):
        fit = fit_adaptive(models, [], sk, options, l2=float(fcfg.get("l2", 1e-3)),
                           processes=int(args.processes or fcfg.get("processes", 1)),
                           max_evaluations=int(fcfg.get("max_evaluations", 40)), verbose=True,
                           band_samples=fit_s, band_weight=1.0, band_settings=settings,
                           offsets=fcfg.get("offsets", "global"),
                           band_jacobian=fcfg.get("band_jacobian", "frozen"))
    fitted = tuple(fit.models)
    evaluated = fit_s + held_s + test_s
    with log.step("evaluate", run=name, part=part, n=len(evaluated)):
        results = evaluate_bands(fitted, evaluated, sk, options, settings,
                                 processes=int(args.processes or fcfg.get("processes", 1)))
    save_models(os.path.join(out, "env_models.json"), fitted,
                meta={"run": name, "part": part, "hash": h})
    roles = {s.ref.name: ("fit" if s in fit_s else "holdout" if s in held_s else "test")
             for s in evaluated}
    for r in results:
        r["role"] = roles.get(r["name"])
    with open(os.path.join(out, "bands.json"), "w") as fh:
        json.dump(acc._clean(results), fh, indent=1)
    rec = {"run": name, "part": part, "holdout": holdout, "hash": h, "status": "ok",
           "channel": spec["channel"], "level": spec.get("level"),
           "regressor": spec.get("regressor", "linear"), "labels": labels, "skdir": skdir,
           "n_params": int(len(fit.x)), "parameter_names": parameter_names(fitted),
           "x": [float(v) for v in fit.x], "n_evaluations": fit.n_evaluations,
           "seconds": round(time.perf_counter() - t0, 1), "history": fit.history,
           "band": {"fit": loss_of([r for r in results if r["role"] == "fit"]),
                    "holdout": loss_of([r for r in results if r["role"] == "holdout"]),
                    "test": loss_of([r for r in results if r["role"] == "test"])},
           "base": {k: base[k] for k in ("index", "label", "radii", "skdir")}}
    if holdout is None:
        M.write_model(os.path.join(out, "model"), skdir, dcfg, repulsive="none",
                      env_models=fitted, meta={"stage": "A2", "run": name, "channel": spec["channel"],
                                               "base": rec["base"], "electronic_only": True})
        rec["model_dir"] = os.path.join(out, "model")
    with open(done + ".tmp", "w") as fh:
        json.dump(acc._clean(rec), fh, indent=1)
    os.replace(done + ".tmp", done)
    print(f"{name}/{part}: band loss fit {rec['band']['fit']}, holdout {rec['band']['holdout']}, "
          f"test {rec['band']['test']} ({rec['n_evaluations']} evaluations)")
    return rec


def stage_fit(cfg, args, log):
    if not args.run:
        raise SystemExit("fit needs --run (see `a2.py plan`)")
    spec = run_spec(cfg, args.run)
    samples = band_samples(cfg)
    test = test_groups(cfg, samples)
    groups = [g for g in groups_of(samples) if g not in test]
    if args.holdout in (None, "full"):
        todo = [None]
    elif args.holdout == "all":
        todo = [None] + groups
    elif args.holdout.isdigit():                 # a Slurm array index: 0 = full, k = group k-1
        k = int(args.holdout)
        todo = [None] if k == 0 else [groups[k - 1]]
    else:
        if args.holdout not in groups:
            raise SystemExit(f"--holdout {args.holdout}: not one of {groups}")
        todo = [args.holdout]
    for g in todo:
        fit_one(cfg, spec, g, args, log)


def baseline_bands(cfg, base, args, log):
    """The band results of the A1 point itself with this config's band settings and k-points
    (cached in <out>/baseline_bands.json)."""
    from pydftb.adaptive_fit import evaluate_bands
    from utils import accuracy as acc
    from common import models as M
    path = os.path.join(cfg["out"], "baseline_bands.json")
    if os.path.exists(path) and not getattr(args, "force", False):
        with open(path) as fh:
            d = json.load(fh)
        if d.get("index") == base["index"]:
            return d["results"]
    samples = band_samples(cfg)
    dcfg = cfg["_scan"]["dftb"]
    labels = sorted({s for smp in samples for s in smp.ref.atoms().get_chemical_symbols()})
    sk = M.engine_skset(base["skdir"], dcfg, labels)
    with log.step("baseline_bands", n=len(samples)):
        res = evaluate_bands((), samples, sk, M.engine_options(dcfg, for_fit=True),
                             band_settings(cfg), processes=int(getattr(args, "processes", 0)
                                                               or cfg["fit"].get("processes", 1)))
    os.makedirs(cfg["out"], exist_ok=True)
    with open(path, "w") as fh:
        json.dump(acc._clean({"index": base["index"], "results": res}), fh, indent=1)
    return res


def stage_transfer(cfg, args, log):
    """Electronic energies and forces of the fitted model on every structure of the scan, then
    the scan's own scoring on them (physrep sigma by L_EF, R2b, oracle): a pseudo point."""
    from utils import accuracy as acc
    from common import models as M
    if not args.run:
        raise SystemExit("transfer needs --run")
    name = args.run
    full = run_dir(cfg, name)
    if not os.path.exists(os.path.join(full, "fit.json")):
        raise SystemExit(f"{name}: no full fit yet (a2.py fit --run {name})")
    with open(os.path.join(full, "fit.json")) as fh:
        rec = json.load(fh)
    out = run_dir(cfg, name, "transfer")
    os.makedirs(out, exist_ok=True)
    losses_path = os.path.join(out, "points", "p00000", "losses.json")
    if os.path.exists(losses_path) and not args.force:
        print(f"{name}: transfer done ({losses_path}); --force recomputes")
        return
    scan = cfg["_scan"]
    model = M.load_model(rec["model_dir"])
    frames = acc._data_cache(scan)
    t0 = time.perf_counter()
    with log.step("predict", run=name, n=len(frames), jobs=args.jobs):
        preds = M.predict(model, frames, cache=os.path.join(out, "predict.jsonl"), jobs=args.jobs,
                          log=print, forces=True, energy="e_zero", dispersion=False, delta=False)
    elements = sorted({s for a in frames for s in a.get_chemical_symbols()})
    with log.step("atoms", elements=elements):
        e0 = model.atom_energies(elements, dispersion=False, delta=False)
    with open(os.path.join(full, "bands.json")) as fh:
        bands = json.load(fh)
    energies = np.array([np.nan if p["e"] is None else p["e"] for p in preds], float)
    forces = [np.zeros((len(a), 3)) if p["f"] is None else np.asarray(p["f"], float)
              for a, p in zip(frames, preds)]
    errors = [p["error"] for p in preds]
    pdir = os.path.join(out, "points", "p00000")
    os.makedirs(pdir, exist_ok=True)
    np.savez_compressed(os.path.join(pdir, "electronic.npz"), energies=energies,
                        forces=np.concatenate([f.ravel() for f in forces]),
                        natoms=np.array([len(a) for a in frames]))
    base = rec["base"]
    result = {"index": 0, "x": [], "radii": base["radii"], "sk_key": os.path.basename(rec["skdir"]),
              "skdir": rec["skdir"], "e0_dftb": e0, "names": [a.info["name"] for a in frames],
              "attempts": [p["attempts"] for p in preds], "errors": errors,
              "n_failed": int(sum(e is not None for e in errors)), "bands": bands,
              "seconds": {"electronic": round(time.perf_counter() - t0, 1)},
              "a2": {"run": name, "model_dir": rec["model_dir"]}}
    with open(os.path.join(pdir, "result.json"), "w") as fh:
        json.dump(acc._clean(result), fh, indent=1)
    cfg2 = copy.deepcopy(scan)
    cfg2["out"] = out
    with log.step("score", run=name):
        losses = acc.score_point(cfg2, 0, force=True)
    bp = base_point(cfg)
    summary = {"run": name, "n_failed": result["n_failed"],
               "a2": _transfer_numbers(losses), "base": _transfer_numbers(bp["losses"]),
               "base_label": bp["label"]}
    with open(os.path.join(out, "transfer.json"), "w") as fh:
        json.dump(acc._clean(summary), fh, indent=1)
    print(json.dumps(acc._clean(summary), indent=1))


def _role_block(summary, role):
    b = (((summary or {}).get("by_role") or {}).get(role)) or {}
    return {k: b.get(k) for k in ("relative_mae", "atomization_mae", "offset_mae", "force_mae",
                                  "n", "failed")}


def _transfer_numbers(losses) -> dict | None:
    if not losses:
        return None
    s1 = (losses.get("strategies") or {}).get("S1") or {}
    r2b = (losses.get("R2b") or {}).get("summary")
    orc = (losses.get("oracle") or {}).get("summary")
    return {"S1_loss": s1.get("loss"), "S1_sigma": s1.get("sigma"),
            "S1": {r: _role_block(s1.get("summary"), r) for r in ("train", "test", "ood")},
            "R2b": {r: _role_block(r2b, r) for r in ("train", "test", "ood")},
            "oracle": {r: _role_block(orc, r) for r in ("train", "test", "ood")},
            "band_loss": (losses.get("bands") or {}).get("loss")}


def _per_structure_optimum(cfg, fit_groups):
    """K6: the mean band loss of the fitted groups with every structure at its own best scan
    point (a lower bound for any environment model on these r0 radii)."""
    if not cfg.get("k6") or not os.path.exists(cfg["k6"]):
        return None
    with open(cfg["k6"]) as fh:
        k6 = json.load(fh)
    samples = band_samples(cfg)
    group = {s.ref.name: s.group for s in samples}
    res = [{"name": n, "group": group.get(n), "loss": v.get("own"), "weight": 1.0}
           for n, v in (k6.get("bands") or {}).items() if group.get(n) in fit_groups]
    return loss_of(res)


def stage_summary(cfg, args, log):
    """result_<channel>.json (a2-result/1) and summary.md from the finished fits and transfers."""
    from utils import accuracy as acc
    base = base_point(cfg)
    samples = band_samples(cfg)
    test = test_groups(cfg, samples)
    groups = groups_of(samples)
    fit_groups = [g for g in groups if g not in test]
    try:
        bres = baseline_bands(cfg, base, args, log)
    except Exception as exc:                  # noqa: BLE001
        print(f"baseline bands: {exc}; using the scan's own band results")
        bres = base["result"].get("bands") or []
    base_num = _transfer_numbers(base["losses"])
    baseline = {"label": base["label"],
                "band": {"train": loss_of(bres, fit_groups), "loeo": loss_of(bres, fit_groups),
                         "test": loss_of(bres, test) if test else None},
                "r2b": {r: (base_num or {}).get("R2b", {}).get(r) for r in ("test", "ood")}}
    pso = _per_structure_optimum(cfg, fit_groups)
    by_channel = {}
    lines = [f"# A2 {cfg['name']}", "", f"base: {base['label']}, radii {base['radii']}", "",
             f"baseline band loss train {_f(baseline['band']['train'])}, test "
             f"{_f(baseline['band']['test'])}; per-structure optimum (K6) {_f(pso)}", "",
             "| run | params | band fit | band LOEO | band test | R2b test rel (meV/atom) | "
             "R2b test F (eV/A) | S1 test rel | failed |", "|---|---|---|---|---|---|---|---|---|"]
    for spec in cfg.get("runs", []):
        name = run_name(spec)
        fp = os.path.join(run_dir(cfg, name), "fit.json")
        if not os.path.exists(fp):
            continue
        with open(fp) as fh:
            rec = json.load(fh)
        loeo_vals = []
        for g in fit_groups:
            lp = os.path.join(run_dir(cfg, name, f"loeo-{g}"), "fit.json")
            if os.path.exists(lp):
                with open(lp) as fh:
                    v = json.load(fh)["band"]["holdout"]
                if v is not None:
                    loeo_vals.append(v)
        loeo = float(np.mean(loeo_vals)) if len(loeo_vals) == len(fit_groups) and loeo_vals else None
        tr = None
        tp = os.path.join(run_dir(cfg, name, "transfer"), "transfer.json")
        if os.path.exists(tp):
            with open(tp) as fh:
                tr = json.load(fh)
        a2n = (tr or {}).get("a2") or {}
        level = {"level": spec.get("level", "L1"), "run": name,
                 "descriptor": _descriptor_label(cfg, spec), "regressor": spec.get("regressor", "linear"),
                 "n_params": rec["n_params"],
                 "band": {"train": rec["band"]["fit"], "loeo": loeo, "test": rec["band"]["test"]},
                 "r2b": {r: (a2n.get("R2b") or {}).get(r) for r in ("test", "ood")} if tr else None,
                 "s1": {r: (a2n.get("S1") or {}).get(r) for r in ("test", "ood")} if tr else None,
                 "model_dir": rec.get("model_dir"),
                 "notes": "" if len(loeo_vals) == len(fit_groups) else
                 f"LOEO {len(loeo_vals)}/{len(fit_groups)} groups"}
        by_channel.setdefault(spec["channel"], []).append(level)
        r2 = (level["r2b"] or {}).get("test") or {}
        s1 = (level["s1"] or {}).get("test") or {}
        lines.append(f"| {name} | {rec['n_params']} | {_f(level['band']['train'])} | {_f(loeo)} | "
                     f"{_f(level['band']['test'])} | {_f(r2.get('relative_mae'), 1e3, 1)} | "
                     f"{_f(r2.get('force_mae'), 1, 3)} | {_f(s1.get('relative_mae'), 1e3, 1)} | "
                     f"{(tr or {}).get('n_failed', '—')} |")
    b2 = baseline["r2b"].get("test") or {}
    lines += ["", f"A1 point R2b test: rel {_f(b2.get('relative_mae'), 1e3, 1)} meV/atom, "
              f"F {_f(b2.get('force_mae'), 1, 3)} eV/A", "",
              "Gate G2: `python common/gates.py --a2 '<out>/result_*.json' ...` (rules in "
              "common/gates.py)."]
    for ch, levels in by_channel.items():
        doc = {"schema": "a2-result/1", "system": cfg.get("system", cfg["name"]), "channel": ch,
               "config": cfg["_path"], "baseline": baseline,
               "per_structure_optimum": {"train": pso}, "levels": levels}
        with open(os.path.join(cfg["out"], f"result_{ch}.json"), "w") as fh:
            json.dump(acc._clean(doc), fh, indent=1)
    with open(os.path.join(cfg["out"], "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    log.event("summary", channels=sorted(by_channel))


def _descriptor_label(cfg, spec):
    if spec["channel"] == "E1":
        return "cn-switch"
    lcfg = cfg.get("descriptors", {}).get(spec.get("level"), {})
    kinds = sorted({d.get("kind") for el, v in lcfg.items() if isinstance(v, list) for d in v})
    return "+".join(k for k in kinds if k)


def _f(v, scale=1.0, digits=4):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else \
        f"{v * scale:.{digits}g}" if digits > 3 else f"{v * scale:.{digits}f}"


def stage_status(cfg, args, log):
    try:
        samples = band_samples(cfg)
        test = test_groups(cfg, samples)
        groups = [g for g in groups_of(samples) if g not in test]
    except SystemExit as exc:
        print(exc)
        groups = []
    for spec in cfg.get("runs", []):
        name = run_name(spec)
        full = os.path.exists(os.path.join(run_dir(cfg, name), "fit.json"))
        lo = sum(os.path.exists(os.path.join(run_dir(cfg, name, f"loeo-{g}"), "fit.json"))
                 for g in groups)
        tr = os.path.exists(os.path.join(run_dir(cfg, name, "transfer"), "transfer.json"))
        print(f"{name:22s} full {'yes' if full else 'no ':3s}  LOEO {lo}/{len(groups)}  "
              f"transfer {'yes' if tr else 'no'}")


STAGES = {"plan": stage_plan, "prepare": stage_prepare, "fit": stage_fit,
          "transfer": stage_transfer, "summary": stage_summary, "status": stage_status}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("config")
    ap.add_argument("--run", default=None, help="a run name (see `plan`)")
    ap.add_argument("--holdout", default=None,
                    help="fit: full (default), all (full + every LOEO fit), a band group, or a "
                         "Slurm array index (0 = full, k = group k-1)")
    ap.add_argument("--processes", type=int, default=0, help="fit: worker processes (0: config)")
    ap.add_argument("--jobs", type=int, default=1, help="transfer: worker processes (Linux)")
    ap.add_argument("--dry-run", action="store_true", help="prepare: only print what it would build")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    from utils.runlog import session
    cfg = load_config(args.config)
    os.makedirs(cfg["out"], exist_ok=True)
    tag = f"a2-{args.stage}" + (f"-{args.run}" if args.run else "")
    with session(tag, vars(args), root=os.path.join(cfg["out"], "logs")) as log:
        STAGES[args.stage](cfg, args, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
