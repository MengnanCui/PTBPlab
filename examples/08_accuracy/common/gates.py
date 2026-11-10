"""The pre-registered gates of the A series (A0 plan, sections 3-7), evaluated from the files the
runs write.  Each gate lists its rules with the numbers and the outcome; a rule whose inputs
are missing says "not yet".  Nothing here computes a model.

    python common/gates.py --root li_o/runs --out li_o/runs/gates [--system li_o]
        [--scan li_o/runs/li_o_s_sobol] [--a2 'a2/runs/*/result.json'] ...

Inputs (defaults under ``--root``; every one can be given explicitly):

* G0  ``k1/k1_results.json`` (li_o/k1_numerics.py)
* G1  ``<scan>/report.json`` (``ptbp scan report``; every ``*/report.json`` under the root by
      default), the scan's ``points/p*/losses.json`` (signs over the whole grid), ``k6/k6_results.json``,
      ``k7/k7_results.json``, ``md*/md_results.json`` (li_o/md_check.py, or the ``md`` stage of
      a3/a4), ``baselines/baselines.json`` (common/baselines.py, context only)
* G2  A2 result files (schema ``a2-result/1`` below)
* G3  A3 result files (``a3-result/1``, written by a3/a3_repulsive.py)
* G4  A4 result files (``a4-result/1``, written by a4/delta_ml.py)

Rules (thresholds fixed in A0 before any result; ``--much-worse`` and ``--band-better`` are the
operationalisations of "much worse" and "better", stated in gates.md):

* **G0** k-grid and smearing (A-series vs converged) <= 2 meV/atom and <= 0.01 eV/A for every
  structure; SK-table force noise <= 0.02 eV/A on the scans' hotcent grid (``--sk-grid``);
  exact vs finite-difference forces <= 1e-4 eV/A.  Fail: fix the numerics first.
* **G1** on the test and ood sets (family-out):
  1. a strategy reaches tier A and its MD is stable: one parameter set suffices for this system;
  2. the spline oracle is within tier B while the best physrep strategy's energy error is
     >= ``--much-worse`` (2) x the oracle's: the repulsive form is the bottleneck, A3 before A2
     (the user decides);
  3. the oracle is worse than tier C, or a decisive interface sign is wrong at every point of
     the grid: one set + a 2-body repulsive cannot break bonds; A2 if K6 shows room (mean band
     gain >= 15 %), else an environment-dependent repulsive (A3) directly;
  4. Spearman(band loss, R2b test) < 0.3: the bands are only a regulariser, and A2 is judged by
     R2b, not by the bands;
  5. (K7) MACE on the same data misses tier A: the data are the limit, add data (D2, D3) first.
* **G2** per descriptor level, against the A1 electronic part it starts from:
  keep the model if the leave-one-environment-out band loss drops >= 30 % and the R2b error
  drops >= 20 %; bands better (>= ``--band-better``, 10 %) but R2b drop < 10 %: the environment
  dependence belongs in the repulsive (A3); climb to the next level only if the training band
  loss is >= 20 % above the per-structure optimum (K6) and the train-test gap is < 30 %.
* **G3** an A3 variant reaches tier A (test and ood) with stable MD: done; else A4.
* **G4** Delta-ML must beat a pure MLIP of the same size on the same data by >= 30 % at some
  N <= 100 (test) or on the family-out set (ood, largest N); else the DFTB base adds nothing to
  the energy accuracy (it helps only the electronic structure).  Error = relative MAE / 10 meV
  + force MAE / 0.1 eV/A (the L_EF scale).

Result schemas (JSON; energies eV/atom, forces eV/A, band losses eV as pydftb.bandloss):

``a2-result/1`` (one file per channel and ladder run; the A2 driver writes it)::

    {"schema": "a2-result/1", "system": "li_o", "channel": "E1" | "E2" | "E3",
     "baseline": {"label": "A1 S3 point 17",
                  "band": {"train": 0.21, "loeo": 0.25, "test": 0.27},
                  "r2b": {"test": {"relative_mae": 0.012, "force_mae": 0.11, "loss_ef": 2.3},
                          "ood": {...}}},
     "per_structure_optimum": {"train": 0.15},       # K6: each structure at its own best r0
     "levels": [{"level": "L1", "descriptor": "cn", "regressor": "linear", "n_params": 6,
                 "band": {"train": ..., "loeo": ..., "test": ...},
                 "r2b": {"test": {...}, "ood": {...}},      # R2b refitted on this electronic model
                 "model_dir": "...", "notes": "..."}, ...]}

  ``loeo``: every environment family left out of the fit in turn and scored (missing: ``test``
  is used).  ``loss_ef`` (L_EF on that role) is preferred for the R2b drop when both sides have
  it, else ``relative_mae``.

``a3-result/1``::

    {"schema": "a3-result/1", "system": "li_o", "variant": "R1" | "R2" | "R3" | "R4",
     "label": "...", "parent": "<model dir>", "model_dir": "<model dir>",
     "summary": {...},               # common.models.brief(pydftb.assess summary)
     "tier": "A" | "B" | "C" | null,
     "md": {"stable": true, "runs": {...}} | null,
     "fit": {...}}

``a4-result/1``::

    {"schema": "a4-result/1", "system": "li_o", "base_model": "<model dir>", "label": "...",
     "curve": [{"n": 25, "kind": "delta" | "pure", "model": "<file>",
                "test": {"relative_mae": ..., "atomization_mae": ..., "force_mae": ...},
                "ood": {...}, "tier": ...}, ...]}

Writes ``gates.json`` and ``gates.md`` to ``--out``; logs in ``<out>/logs/gates/``.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.paths import ACCURACY, setup  # noqa: E402

setup()

TIERS = {"A": (0.010, 0.10), "B": (0.030, 0.20), "C": (0.100, None)}     # = pydftb.assess.TIERS
EPS_E, EPS_F = 0.010, 0.10
NOT_YET = "not yet"


def _load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _num(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _fmt(v, scale=1.0, digits=3):
    return "—" if not _num(v) else f"{v * scale:.{digits}g}"


def rule(name, outcome, numbers=None, note=""):
    """One rule: outcome True / False / None ("not yet")."""
    return {"rule": name, "outcome": outcome, "numbers": numbers or {}, "note": note}


def tier_of(blocks: dict, roles=("test", "ood"), energy_key="relative_mae"):
    """The best tier the role blocks meet (energy and force MAE only, no reactions)."""
    present = [blocks.get(r) for r in roles if blocks.get(r)]
    if not present:
        return None
    for name, (te, tf) in TIERS.items():
        ok = True
        for b in present:
            if b.get("failed"):
                ok = False
                continue
            e = b.get(energy_key)
            if _num(e):
                ok &= e <= te
            if tf is not None and _num(b.get("force_mae")):
                ok &= b["force_mae"] <= tf
        if ok:
            return name
    return None


def energy_error(blocks: dict, roles=("test", "ood"), key="relative_mae"):
    vals = [blocks[r][key] for r in roles if blocks.get(r) and _num(blocks[r].get(key))]
    return sum(vals) / len(vals) if vals else None


# ---- G0 -----------------------------------------------------------------------------------------
def gate_g0(k1, sk_grid: str):
    if k1 is None:
        return {"gate": "G0", "decision": NOT_YET, "rules": [rule("K1 results", None)]}
    rules = []
    a = k1.get("a_kpoints_smearing")
    if a:
        worst_e = max(abs(r["a_series"]["dE_meV_per_atom"]) for r in a.values())
        worst_f = max(r["a_series"]["dF_max"] for r in a.values())
        rules.append(rule("k-grid and smearing <= 2 meV/atom, <= 0.01 eV/A",
                          all(r.get("pass") for r in a.values()),
                          {"worst_dE_meV_per_atom": worst_e, "worst_dF_eV_per_A": worst_f}))
    else:
        rules.append(rule("k-grid and smearing <= 2 meV/atom, <= 0.01 eV/A", None))
    b = k1.get("b_sk_noise")
    grid = (b or {}).get(sk_grid)
    if grid:
        noise = max(v["force_noise"] for v in grid.values())
        rules.append(rule(f"SK-table force noise <= 0.02 eV/A ({sk_grid})", noise <= 0.02,
                          {"max_force_noise": noise}))
    else:
        rules.append(rule(f"SK-table force noise <= 0.02 eV/A ({sk_grid})", None,
                          note="part (b) not run" if not b else f"grid {sk_grid} not in K1 (b)"))
    c = k1.get("c_forces")
    rules.append(rule("exact vs finite-difference forces <= 1e-4 eV/A",
                      None if c is None else bool(c.get("pass")),
                      {} if c is None else {"max_abs": c.get("max_abs_exact_minus_fd")}))
    outs = [r["outcome"] for r in rules]
    if any(o is False for o in outs):
        decision = "fail: fix the numerical settings before any fit"
    elif all(o is True for o in outs):
        decision = "pass: the numerics are not the limit"
    else:
        decision = NOT_YET
    return {"gate": "G0", "decision": decision, "rules": rules}


# ---- G1 -----------------------------------------------------------------------------------------
def _grid_signs(scan_dir):
    """Over every scored point: fraction of points where all decisive interface reactions (the
    names containing 'interface' or 'iface') have the right sign under S1, and the same for every
    decisive reaction; None if there are none."""
    files = sorted(glob.glob(os.path.join(scan_dir, "points", "p*", "losses.json")))
    iface, decisive, n = 0, 0, 0
    have_iface = have_dec = False
    for p in files:
        sc = _load(p)
        rx = (((sc.get("strategies") or {}).get("S1") or {}).get("summary") or {}).get("reactions") or []
        dec = [r for r in rx if r.get("decisive")]
        if not dec:
            continue
        n += 1
        ifc = [r for r in dec if "interface" in r.get("name", "") or "iface" in r.get("name", "")]
        have_dec = True
        decisive += all(r.get("sign_ok") for r in dec)
        if ifc:
            have_iface = True
            iface += all(r.get("sign_ok") for r in ifc)
    return {"points": n,
            "iface_right_fraction": (iface / n) if (n and have_iface) else None,
            "decisive_right_fraction": (decisive / n) if (n and have_dec) else None}


def _md_for(strategy, md_runs):
    """MD results that belong to a strategy: a directory name with _<strategy> (or the only one)."""
    pat = re.compile(rf"_{re.escape(strategy)}(_|$)")
    own = {d: r for d, r in md_runs.items() if pat.search(os.path.basename(d))}
    if not own and len(md_runs) == 1:
        own = dict(md_runs)
    if not own:
        return None
    stable = all(not v.get("collapse") and not v.get("failed") for r in own.values()
                 for v in r.values() if isinstance(v, dict))
    return {"dirs": sorted(own), "stable": stable}


def gate_g1(report, scan_dir, k6, k7, md_runs, much_worse, baselines=None):
    name = (report or {}).get("name", os.path.basename(scan_dir or "") or "scan")
    if report is None:
        return {"gate": "G1", "scan": name, "decision": NOT_YET,
                "rules": [rule("scan report.json", None)]}
    rows = {r["strategy"]: r for r in report.get("best", [])}
    rules = []
    # 1. tier A + MD
    tier_a = [s for s, r in rows.items() if r.get("tier") == "A"]
    md_state = {s: _md_for(s, md_runs) for s in tier_a}
    if tier_a:
        stable = [s for s in tier_a if md_state[s] and md_state[s]["stable"]]
        pending = [s for s in tier_a if md_state[s] is None]
        out1 = True if stable else (None if pending else False)
        note = ("MD stable for " + ", ".join(stable)) if stable else (
            "tier A, MD not run yet for " + ", ".join(pending) if pending else "tier A but MD unstable")
    else:
        out1, note = False, "no strategy at tier A on test + ood"
    rules.append(rule("1. a strategy at tier A (test, ood) with stable MD", out1,
                      {"tiers": {s: r.get("tier") for s, r in rows.items()}}, note))
    # 2. oracle within B and physrep much worse
    orc = None
    for r in rows.values():
        if r.get("oracle_all"):
            orc = r["oracle_all"]
            break
    phys_err = [energy_error({k: r.get(k) for k in ("test", "ood")}) for r in rows.values()]
    phys_err = [v for v in phys_err if _num(v)]
    best_phys = min(phys_err) if phys_err else None
    if orc:
        o_tier = tier_of(orc)
        o_err = energy_error(orc)
        ratio = best_phys / o_err if (_num(best_phys) and _num(o_err) and o_err > 0) else None
        out2 = (o_tier in ("A", "B")) and (ratio is not None and ratio >= much_worse)
        rules.append(rule(f"2. oracle within tier B and best physrep >= {much_worse} x the oracle",
                          bool(out2), {"oracle_tier": o_tier, "oracle_err": o_err,
                                       "best_physrep_err": best_phys, "ratio": ratio}))
    else:
        o_tier = o_err = None
        rules.append(rule("2. oracle within tier B and best physrep much worse", None,
                          note="no oracle (repfit.pairs) in the report"))
    # 3. oracle worse than C, or interface signs wrong everywhere
    signs = _grid_signs(scan_dir) if scan_dir else {"points": 0}
    if orc or signs.get("iface_right_fraction") is not None:
        worse_c = orc is not None and o_tier is None
        never = signs.get("iface_right_fraction") == 0.0
        rules.append(rule("3. oracle worse than tier C, or a decisive interface sign wrong at every "
                          "grid point", bool(worse_c or never),
                          {"oracle_tier": o_tier, **signs}))
    else:
        rules.append(rule("3. oracle worse than tier C, or interface signs wrong everywhere", None))
    # 4. Spearman
    sp = (report.get("correlations") or {}).get("band_vs_R2b_test")
    rules.append(rule("4. Spearman(band loss, R2b test) < 0.3",
                      None if sp is None else bool(sp["spearman"] < 0.3),
                      {} if sp is None else sp,
                      "" if sp is not None else "needs >= 5 points with band losses and R2b"))
    # K6 and K7 flags
    gain = None if not k6 else (k6.get("band_summary") or {}).get("mean_gain")
    rules.append(rule("K6: mean band gain of a per-structure r0 >= 15 % (room for E2)",
                      None if not _num(gain) else bool(gain >= 0.15), {"mean_gain": gain}))
    k7_tier = None
    if k7:
        trained = {k: v for k, v in k7.items() if not k.startswith("mace_mp")}
        k7_tier = next((v.get("tier") for v in trained.values()), None) if trained else None
    rules.append(rule("5. K7: MACE on the same data misses tier A (data limit)",
                      None if not k7 else bool(k7_tier != "A"), {"mace_tier": k7_tier}))
    # decision, in the pre-registered order
    outs = {i: r["outcome"] for i, r in enumerate(rules)}
    if outs[0] is True:
        decision = "one parameter set suffices for this system: move to the next system"
    elif outs[0] is None and tier_a:
        decision = "tier A reached: run the MD check before deciding"
    elif outs[1] is True:
        decision = "the repulsive form is the bottleneck: A3 before A2 (the user decides)"
    elif outs[2] is True:
        decision = ("one set + 2-body repulsive cannot break bonds: "
                    + ("A2 (K6 shows room)" if outs[4] is True else
                       "environment-dependent repulsive (A3) directly" if outs[4] is False else
                       "A2 if K6 shows room (K6 not run yet)"))
    elif all(o is None for o in outs.values()):
        decision = NOT_YET
    else:
        decision = "no pre-registered rule fired: A2 by default (the plan's next phase)"
    flags = []
    if outs[3] is True:
        flags.append("bands only as a regulariser; judge A2 by R2b")
    if outs[5] is True:
        flags.append("data limit: add D2/D3 data before more method work")
    out = {"gate": "G1", "scan": name, "decision": decision, "flags": flags, "rules": rules,
           "md": {s: v for s, v in md_state.items()}}
    if baselines:
        out["baselines"] = {m: {"status": e.get("status"),
                                "tier": (e.get("summary") or {}).get("tier")}
                            for m, e in (baselines.get("methods") or {}).items()}
    return out


# ---- G2 -----------------------------------------------------------------------------------------
def _r2b_metric(r2b: dict):
    t = (r2b or {}).get("test") or {}
    if _num(t.get("loss_ef")):
        return "loss_ef", t["loss_ef"]
    return "relative_mae", t.get("relative_mae")


def gate_g2(results, band_better):
    if not results:
        return {"gate": "G2", "decision": NOT_YET, "runs": []}
    runs = []
    for res in results:
        base = res.get("baseline") or {}
        b0 = (base.get("band") or {})
        b0_loeo = b0.get("loeo") if _num(b0.get("loeo")) else b0.get("test")
        key0, m0 = _r2b_metric(base.get("r2b"))
        opt = (res.get("per_structure_optimum") or {}).get("train")
        levels = []
        for lv in res.get("levels", []):
            b = lv.get("band") or {}
            loeo = b.get("loeo") if _num(b.get("loeo")) else b.get("test")
            key, m = _r2b_metric(lv.get("r2b"))
            if key != key0:              # compare like with like
                key, m = "relative_mae", ((lv.get("r2b") or {}).get("test") or {}).get("relative_mae")
                m0c = ((base.get("r2b") or {}).get("test") or {}).get("relative_mae")
            else:
                m0c = m0
            band_drop = 1 - loeo / b0_loeo if (_num(loeo) and _num(b0_loeo) and b0_loeo > 0) else None
            r2b_drop = 1 - m / m0c if (_num(m) and _num(m0c) and m0c > 0) else None
            keep = None if band_drop is None or r2b_drop is None else bool(band_drop >= 0.30 and r2b_drop >= 0.20)
            to_a3 = None if band_drop is None or r2b_drop is None else bool(band_drop >= band_better and r2b_drop < 0.10)
            tr, te = b.get("train"), b.get("test")
            climb = None
            if _num(tr) and _num(opt) and _num(te) and tr > 0 and opt > 0:
                climb = bool(tr >= 1.2 * opt and (te - tr) / tr < 0.30)
            levels.append({"level": lv.get("level"), "descriptor": lv.get("descriptor"),
                           "regressor": lv.get("regressor"), "band_drop": band_drop,
                           "r2b_metric": key, "r2b_drop": r2b_drop, "keep": keep, "to_A3": to_a3,
                           "climb": climb, "model_dir": lv.get("model_dir")})
        kept = [lv for lv in levels if lv["keep"]]
        if kept:
            decision = f"keep the {kept[0]['level']} model ({kept[0]['descriptor']}, {kept[0]['regressor']})"
        elif any(lv["to_A3"] for lv in levels):
            decision = "bands better but R2b not: the environment dependence belongs in the repulsive (A3)"
        elif levels and levels[-1]["climb"]:
            decision = f"climb to the level after {levels[-1]['level']}"
        elif levels and all(lv["keep"] is None for lv in levels):
            decision = NOT_YET
        elif levels:
            decision = "stop the ladder here (no gain, or the gain does not transfer)"
        else:
            decision = NOT_YET
        runs.append({"system": res.get("system"), "channel": res.get("channel"),
                     "baseline": {"band_loeo": b0_loeo, "r2b": m0, "r2b_metric": key0,
                                  "per_structure_optimum_train": opt},
                     "levels": levels, "decision": decision, "file": res.get("_file")})
    return {"gate": "G2", "runs": runs,
            "decision": "; ".join(f"{r['channel']}: {r['decision']}" for r in runs)}


# ---- G3 -----------------------------------------------------------------------------------------
def gate_g3(results):
    if not results:
        return {"gate": "G3", "decision": NOT_YET, "variants": []}
    rows = []
    for res in results:
        md = res.get("md")
        rows.append({"variant": res.get("variant"), "label": res.get("label"),
                     "tier": res.get("tier"), "md_stable": None if md is None else md.get("stable"),
                     "test_rel": (((res.get("summary") or {}).get("by_role") or {}).get("test") or {}).get("relative_mae"),
                     "ood_rel": (((res.get("summary") or {}).get("by_role") or {}).get("ood") or {}).get("relative_mae"),
                     "model_dir": res.get("model_dir"), "file": res.get("_file")})
    done = [r for r in rows if r["tier"] == "A" and r["md_stable"] is True]
    pending = [r for r in rows if r["tier"] == "A" and r["md_stable"] is None]
    if done:
        decision = f"done: {done[0]['variant']} ({done[0]['label']}) reaches the acceptance line"
    elif pending:
        decision = f"tier A for {pending[0]['variant']}: run its MD check before deciding"
    else:
        decision = "no A3 variant reaches tier A: go to A4 (Delta-ML)"
    return {"gate": "G3", "variants": rows, "decision": decision}


# ---- G4 -----------------------------------------------------------------------------------------
def _err(block):
    if not block or not _num(block.get("relative_mae")):
        return None
    f = block.get("force_mae")
    return block["relative_mae"] / EPS_E + (f / EPS_F if _num(f) else 0.0)


def gate_g4(results):
    if not results:
        return {"gate": "G4", "decision": NOT_YET, "runs": []}
    runs = []
    for res in results:
        by = {}
        for c in res.get("curve", []):
            by.setdefault(c["n"], {})[c["kind"]] = c
        rows = []
        for n in sorted(by, key=lambda v: (v == "all", v if isinstance(v, int) else 10 ** 9)):
            d, p = by[n].get("delta"), by[n].get("pure")
            row = {"n": n}
            for role in ("test", "ood"):
                ed, ep = _err((d or {}).get(role)), _err((p or {}).get(role))
                row[f"{role}_delta"], row[f"{role}_pure"] = ed, ep
                row[f"{role}_gain"] = (1 - ed / ep) if (_num(ed) and _num(ep) and ep > 0) else None
            rows.append(row)
        small = [r for r in rows if isinstance(r["n"], int) and r["n"] <= 100 and _num(r["test_gain"])]
        last = rows[-1] if rows else None
        hit_small = any(r["test_gain"] >= 0.30 for r in small)
        hit_ood = bool(last and _num(last["ood_gain"]) and last["ood_gain"] >= 0.30)
        if not small and not (last and _num(last.get("ood_gain"))):
            decision = NOT_YET
        elif hit_small or hit_ood:
            decision = "Delta-ML on the DFTB base pays (>= 30 % better than the pure MLIP)"
        else:
            decision = ("the DFTB base adds nothing to the energy accuracy (it helps only the "
                        "electronic structure)")
        runs.append({"system": res.get("system"), "label": res.get("label"),
                     "base_model": res.get("base_model"), "rows": rows, "decision": decision,
                     "file": res.get("_file")})
    return {"gate": "G4", "runs": runs, "decision": "; ".join(f"{r['label']}: {r['decision']}" for r in runs)}


# ---- report -------------------------------------------------------------------------------------
def _md(g) -> list:
    L = [f"## {g['gate']}{(' — ' + g['scan']) if g.get('scan') else ''}", "",
         f"**decision:** {g['decision']}", ""]
    for f in g.get("flags", []):
        L.append(f"- flag: {f}")
    if g.get("rules"):
        L += ["| rule | outcome | numbers | note |", "|---|---|---|---|"]
        for r in g["rules"]:
            o = {True: "yes", False: "no", None: NOT_YET}[r["outcome"]]
            nums = ", ".join(f"{k} {_fmt(v) if not isinstance(v, (dict, list, str)) else v}"
                             for k, v in r["numbers"].items())
            L.append(f"| {r['rule']} | {o} | {nums} | {r['note']} |")
    for run in g.get("runs", []):
        if g["gate"] == "G2":
            L += ["", f"### {run['system']} {run['channel']}: {run['decision']}", "",
                  f"baseline: band (LOEO) {_fmt(run['baseline']['band_loeo'])} eV, R2b "
                  f"{run['baseline']['r2b_metric']} {_fmt(run['baseline']['r2b'])}, per-structure "
                  f"optimum {_fmt(run['baseline']['per_structure_optimum_train'])} eV", "",
                  "| level | descriptor | regressor | band drop | R2b drop | keep | to A3 | climb |",
                  "|---|---|---|---|---|---|---|---|"]
            for lv in run["levels"]:
                L.append(f"| {lv['level']} | {lv['descriptor']} | {lv['regressor']} | "
                         f"{_fmt(lv['band_drop'], 100, 3)}% | {_fmt(lv['r2b_drop'], 100, 3)}% | "
                         f"{lv['keep']} | {lv['to_A3']} | {lv['climb']} |")
        if g["gate"] == "G4":
            L += ["", f"### {run['label']}: {run['decision']}", "",
                  "| N | test Δ | test pure | test gain | ood Δ | ood pure | ood gain |",
                  "|---|---|---|---|---|---|---|"]
            for r in run["rows"]:
                L.append(f"| {r['n']} | {_fmt(r['test_delta'])} | {_fmt(r['test_pure'])} | "
                         f"{_fmt(r['test_gain'], 100, 3)}% | {_fmt(r['ood_delta'])} | "
                         f"{_fmt(r['ood_pure'])} | {_fmt(r['ood_gain'], 100, 3)}% |")
    if g["gate"] == "G3" and g.get("variants"):
        L += ["| variant | label | tier | MD stable | test rel (meV/atom) | ood rel |", "|---|---|---|---|---|---|"]
        for r in g["variants"]:
            L.append(f"| {r['variant']} | {r['label']} | {r['tier'] or '—'} | {r['md_stable']} | "
                     f"{_fmt(r['test_rel'], 1e3)} | {_fmt(r['ood_rel'], 1e3)} |")
    return L + [""]


def _results(pattern, schema, system):
    out = []
    for p in sorted(glob.glob(pattern, recursive=True)):
        d = _load(p)
        if not d or d.get("schema") != schema:
            continue
        if system and d.get("system") not in (None, system):
            continue
        d["_file"] = p
        out.append(d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="li_o/runs", help="where the K checks and scans write")
    ap.add_argument("--out", default=None, help="default: <root>/gates")
    ap.add_argument("--system", default=None, help="keep A2/A3/A4 results of this system only")
    ap.add_argument("--scan", nargs="*", default=None, help="scan output dirs (default: */report.json)")
    ap.add_argument("--k1", default=None)
    ap.add_argument("--k6", default=None)
    ap.add_argument("--k7", default=None)
    ap.add_argument("--md", nargs="*", default=None, help="md_results.json files or their dirs")
    ap.add_argument("--baselines", default=None)
    ap.add_argument("--a2", default=os.path.join(ACCURACY, "a2", "runs", "**", "result.json"))
    ap.add_argument("--a3", default=os.path.join(ACCURACY, "a3", "runs", "**", "result.json"))
    ap.add_argument("--a4", default=os.path.join(ACCURACY, "a4", "runs", "**", "result.json"))
    ap.add_argument("--sk-grid", default="nr100_nt300", help="K1 (b) grid the scans use")
    ap.add_argument("--much-worse", type=float, default=2.0)
    ap.add_argument("--band-better", type=float, default=0.10)
    args = ap.parse_args(argv)
    from utils.runlog import session
    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out or os.path.join(root, "gates"))
    os.makedirs(out_dir, exist_ok=True)
    with session("gates", vars(args), root=os.path.join(out_dir, "logs")) as log:
        k1 = _load(args.k1 or os.path.join(root, "k1", "k1_results.json"))
        k6 = _load(args.k6 or os.path.join(root, "k6", "k6_results.json"))
        k7 = _load(args.k7 or os.path.join(root, "k7", "k7_results.json"))
        md_paths = args.md if args.md is not None else glob.glob(os.path.join(root, "md*", "md_results.json"))
        md_runs = {}
        for p in md_paths:
            p = os.path.join(p, "md_results.json") if os.path.isdir(p) else p
            d = _load(p)
            if d is not None:
                md_runs[os.path.dirname(os.path.abspath(p))] = d
        base = _load(args.baselines or os.path.join(root, "baselines", "baselines.json"))
        scans = args.scan if args.scan is not None else sorted(
            os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "report.json")))
        gates = [gate_g0(k1, args.sk_grid)]
        for s in scans or [None]:
            gates.append(gate_g1(_load(os.path.join(s, "report.json")) if s else None, s, k6, k7,
                                 md_runs, args.much_worse, base))
        gates.append(gate_g2(_results(args.a2, "a2-result/1", args.system), args.band_better))
        gates.append(gate_g3(_results(args.a3, "a3-result/1", args.system)))
        gates.append(gate_g4(_results(args.a4, "a4-result/1", args.system)))
        for g in gates:
            log.event("gate", gate=g["gate"], decision=g["decision"])
        with open(os.path.join(out_dir, "gates.json"), "w") as fh:
            json.dump(gates, fh, indent=1, default=str)
        L = ["# A-series gates", "",
             f"root `{root}`; thresholds as pre-registered in A0 (sections 3-7); operationalised: "
             f"'much worse' = x{args.much_worse}, 'bands better' = >= {100 * args.band_better:.0f} %.", ""]
        for g in gates:
            L += _md(g)
        with open(os.path.join(out_dir, "gates.md"), "w") as fh:
            fh.write("\n".join(L) + "\n")
        print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
