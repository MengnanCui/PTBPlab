"""A4, Delta-ML (A0 plan, section 6): a MACE model on the residual DFT - DFTB of a model
directory (Delta1; on an A2 model directory it is Delta3), against a pure MACE of the same size
trained on the same structures, with learning curves (N = 25, 50, 100, all) and the family-out
(ood) set.  The G4 gate reads result.json (schema a4-result/1, common/gates.py).  Delta2 (E3
with a neural-network regressor) belongs to the A2 driver, not here.

Stages:

* ``prepare``  the base model on every structure (cached), the residual data sets
  (REF_energy = E_DFT - E_DFTB with E(sigma->0) on both sides, REF_forces = F_DFT - F_DFTB) and
  the pure ones (REF = DFT), one fixed validation set (10 % of train), nested training subsets
  of a fixed permutation, the E0s (residual: E0_DFT - E0_DFTB of the model's isolated atoms;
  pure: E0_DFT), and one mace_run_train argument file per (kind, N).  Same architecture for
  Delta and pure (li_o/k7_mace.py's: r_max 5, 64 channels, L 1, correlation 3).
* ``train``  runs mace-torch's trainer for every argument file whose model does not exist yet
  (GPU: slurm/a4_delta_gpu.sh).  Check the arguments for your mace-torch version.
* ``evaluate``  DFTB + Delta (the cached DFTB predictions plus MACE) and pure MACE, scored with
  pydftb.assess on every role (tiers on test and ood); result.json, learning_curve.md/.png, and
  a model directory per Delta model (models/delta_<N>: SK files + the MACE file) for MD checks
  (``python a3/a3_repulsive.py md --model .../models/delta_all ...``).

    python a4/delta_ml.py prepare  --config li_o/scan_li_o.yaml --model li_o/models/S1 --out a4/runs/d1_S1 [--jobs 20]
    python a4/delta_ml.py train    --out a4/runs/d1_S1 --device cuda
    python a4/delta_ml.py evaluate --config li_o/scan_li_o.yaml --out a4/runs/d1_S1 [--device cuda]

Logs: ``<out>/logs/a4-<stage>/``; trainer output ``<out>/train/<name>.log``.  Every stage
continues where it stopped (cached predictions, finished models skipped; ``--force`` redoes).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import setup  # noqa: E402

setup()

from common import models as M  # noqa: E402

KINDS = ("delta", "pure")


def _model_file(out, name):
    """The trained model of mace-torch (the SWA stage if it exists)."""
    for cand in (f"{name}_stagetwo.model", f"{name}.model"):
        p = os.path.join(out, "mace_models", cand)
        if os.path.exists(p):
            return p
    return None


def prepare(args):
    import ase.io
    from ase.data import atomic_numbers
    from utils import accuracy as acc
    from utils.runlog import session
    cfg = acc.load_config(args.config)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    with session("a4-prepare", vars(args), root=os.path.join(out, "logs")) as log:
        base = M.load_model(args.model, dftb=cfg["dftb"])
        if base.delta:
            raise SystemExit("the base model has a Delta model already")
        frames = acc.load_data(cfg)
        e0_dft = acc.load_e0s(cfg)
        elements = sorted({s for a in frames for s in a.get_chemical_symbols()})
        with log.step("base", model=base.path, structures=len(frames)):
            preds = M.predict(base, frames, os.path.join(out, "base", "predictions.jsonl"),
                              args.jobs, print, force=args.force)
            e0_base = base.atom_energies(elements)
        bad = [f.info["name"] for f, p in zip(frames, preds) if p["error"]]
        if bad:
            log.event("base_failed", n=len(bad), names=bad[:20])
        sets = {"delta": {}, "pure": {}}
        for a, p in zip(frames, preds):
            if p["error"]:
                continue
            f_dft = np.array(a.get_forces())
            for kind in KINDS:
                b = a.copy()
                b.calc = None
                if kind == "delta":
                    b.info["REF_energy"] = float(a.get_potential_energy() - p["e"])
                    b.arrays["REF_forces"] = f_dft - np.asarray(p["f"], float)
                else:
                    b.info["REF_energy"] = float(a.get_potential_energy())
                    b.arrays["REF_forces"] = f_dft
                b.info["config_type"] = a.info["family"]
                b.info["dft_energy"] = float(a.get_potential_energy())
                b.info["dftb_energy"] = float(p["e"])
                sets[kind].setdefault(a.info["role"], []).append(b)
        rng = np.random.default_rng(args.seed)
        train = sets["delta"].get("train", [])
        names = [a.info["name"] for a in train]
        if not names:
            raise SystemExit("no training structure with a base prediction")
        perm = list(rng.permutation(len(names)))
        n_val = max(1, int(round(args.valid_fraction * len(names))))
        valid_idx, pool_idx = perm[:n_val], perm[n_val:]
        order = [pool_idx[i] for i in np.random.default_rng(args.seed + 1).permutation(len(pool_idx))]
        subsets = {}
        for n in args.sizes:
            k = len(order) if n == "all" else min(int(n), len(order))
            lab = "all" if (n == "all" or k == len(order)) else str(k)
            if lab in subsets:
                continue
            subsets[lab] = [names[i] for i in order[:k]]
        e0 = {"delta": {int(atomic_numbers[el]): e0_dft[el] - e0_base[el] for el in elements},
              "pure": {int(atomic_numbers[el]): e0_dft[el] for el in elements}}
        os.makedirs(os.path.join(out, "mace_args"), exist_ok=True)
        files = {}
        for kind in KINDS:
            d = os.path.join(out, "data", kind)
            os.makedirs(d, exist_ok=True)
            by = {a.info["name"]: a for a in sets[kind].get("train", [])}
            paths = {"valid": os.path.join(d, "valid.xyz")}
            ase.io.write(paths["valid"], [by[names[i]] for i in valid_idx], format="extxyz")
            for role in ("test", "ood"):
                if sets[kind].get(role):
                    paths[role] = os.path.join(d, f"{role}.xyz")
                    ase.io.write(paths[role], sets[kind][role], format="extxyz")
            for lab, sub in subsets.items():
                p = os.path.join(d, f"train_{lab}.xyz")
                ase.io.write(p, [by[n] for n in sub], format="extxyz")
                name = f"{kind}_{lab}"
                margs = ["--name", name, "--train_file", p, "--valid_file", paths["valid"],
                         "--E0s", json.dumps(e0[kind]).replace('"', ""),
                         "--energy_key", "REF_energy", "--forces_key", "REF_forces",
                         "--model", "MACE", "--r_max", str(args.r_max),
                         "--num_channels", str(args.channels), "--max_L", str(args.max_L),
                         "--correlation", str(args.correlation),
                         "--batch_size", str(max(1, min(args.batch, len(sub)))),
                         "--valid_batch_size", str(max(1, min(args.batch, n_val))),
                         "--max_num_epochs", str(args.epochs), "--swa",
                         "--start_swa", str(int(0.8 * args.epochs)), "--energy_weight", "1.0",
                         "--forces_weight", "10.0", "--default_dtype", "float64",
                         "--seed", str(args.seed + 1),
                         "--results_dir", os.path.join(out, "mace_results"),
                         "--model_dir", os.path.join(out, "mace_models"),
                         "--checkpoints_dir", os.path.join(out, "mace_checkpoints", name),
                         "--log_dir", os.path.join(out, "mace_logs")]
                if "test" in paths:
                    margs[6:6] = ["--test_file", paths["test"]]
                files[name] = os.path.join(out, "mace_args", f"{name}.json")
                with open(files[name], "w") as fh:
                    json.dump(margs, fh, indent=1)
        manifest = {"config": cfg["_path"], "base_model": base.path, "elements": elements,
                    "e0_dft": e0_dft, "e0_base": e0_base, "e0_mace": {k: {str(z): v for z, v in d.items()}
                                                                      for k, d in e0.items()},
                    "valid": [names[i] for i in valid_idx], "subsets": subsets,
                    "failed_base": bad, "args": files, "seed": args.seed,
                    "architecture": {"r_max": args.r_max, "channels": args.channels,
                                     "max_L": args.max_L, "correlation": args.correlation,
                                     "epochs": args.epochs}}
        with open(os.path.join(out, "manifest.json"), "w") as fh:
            json.dump(M._clean(manifest), fh, indent=1)
        log.event("prepared", subsets={k: len(v) for k, v in subsets.items()}, valid=n_val,
                  failed=len(bad))
        print(f"subsets {[(k, len(v)) for k, v in subsets.items()]}, valid {n_val}; "
              f"{len(files)} trainings in mace_args/ (check them for your mace-torch version)")
    return 0


def train(args):
    from utils.runlog import session
    out = os.path.abspath(args.out)
    man = json.load(open(os.path.join(out, "manifest.json")))
    os.makedirs(os.path.join(out, "train"), exist_ok=True)
    failed = []
    with session("a4-train", vars(args), root=os.path.join(out, "logs")) as log:
        for name, path in man["args"].items():
            kind, lab = name.split("_", 1)
            if kind not in args.kinds or (args.sizes and lab not in args.sizes):
                continue
            if _model_file(out, name) and not args.force:
                log.event("skip", name=name, reason="model exists")
                continue
            margs = json.load(open(path))
            cmd = [sys.executable, "-m", "mace.cli.run_train", *margs, "--device", args.device]
            with log.step("train", name=name):
                with open(os.path.join(out, "train", f"{name}.log"), "w") as fh:
                    proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
            log.event("train_done", name=name, returncode=proc.returncode,
                      model=_model_file(out, name))
            if proc.returncode:
                failed.append(name)
        if failed:
            raise SystemExit(f"training failed for {failed}; see {out}/train/<name>.log")
    return 0


def evaluate(args):
    from pydftb import assess
    from utils import accuracy as acc
    from utils.runlog import session
    cfg = acc.load_config(args.config)
    out = os.path.abspath(args.out)
    man = json.load(open(os.path.join(out, "manifest.json")))
    with session("a4-evaluate", vars(args), root=os.path.join(out, "logs")) as log:
        base = M.load_model(man["base_model"], dftb=cfg["dftb"])
        frames = acc.load_data(cfg)
        e0_ref = acc.load_e0s(cfg)
        reactions = assess.load_reactions(cfg["reactions"]) if cfg.get("reactions") else None
        base_preds = M.predict(base, frames, os.path.join(out, "base", "predictions.jsonl"),
                               1, print)
        fr, bp = frames, base_preds            # a failed DFTB structure fails Delta's tiers
        try:
            from mace.calculators import MACECalculator
        except ImportError as exc:
            raise SystemExit("evaluate needs mace-torch") from exc
        from ase import Atoms
        curve = []
        for name in man["args"]:
            kind, lab = name.split("_", 1)
            path = _model_file(out, name)
            if path is None:
                log.event("missing", name=name)
                continue
            calc = MACECalculator(model_paths=path, device=args.device, default_dtype="float64")
            cache = os.path.join(out, "eval", name, "predictions.jsonl")
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            done = {}
            if os.path.exists(cache) and not args.force:
                for line in open(cache):
                    if line.strip():
                        r = json.loads(line)
                        done[r["name"]] = r
            with open(cache, "a", buffering=1) as fh, log.step("mace", name=name):
                for a in fr:
                    if a.info["name"] in done:
                        continue
                    b = a.copy()
                    b.calc = calc
                    r = {"name": a.info["name"], "e": float(b.get_potential_energy()),
                         "f": np.array(b.get_forces()).tolist()}
                    done[r["name"]] = r
                    fh.write(json.dumps(r) + "\n")
            e0_m = {}
            for el in man["elements"]:
                at = Atoms(el, positions=[[10.0, 10.0, 10.0]], cell=[20.0] * 3, pbc=False)
                at.calc = calc
                e0_m[el] = float(at.get_potential_energy())
            e, f, err = [], [], []
            for a, p in zip(fr, bp):
                m = done[a.info["name"]]
                if kind == "delta" and p["error"]:
                    e.append(np.nan)
                    f.append(None)
                    err.append(f"base DFTB failed: {p['error']}")
                elif kind == "delta":
                    e.append(p["e"] + m["e"])
                    f.append(np.asarray(p["f"], float) + np.asarray(m["f"], float))
                    err.append(None)
                else:
                    e.append(m["e"])
                    f.append(np.asarray(m["f"], float))
                    err.append(None)
            e0_pred = ({el: man["e0_base"][el] + e0_m[el] for el in man["elements"]}
                       if kind == "delta" else e0_m)
            summ = assess.summarize(assess.records_from_arrays(fr, e, f, err), e0_ref, e0_pred,
                                    reactions)
            summ["e0_pred"] = e0_pred
            d = os.path.join(out, "eval", name)
            assess.write_markdown(summ, os.path.join(d, "summary.md"), title=f"A4 {name}")
            with open(os.path.join(d, "summary.json"), "w") as fh:
                json.dump(M._clean(summ), fh, indent=1)
            br = summ["energies"]["by_role"]
            pick = lambda b: {k: (b or {}).get(k) for k in ("relative_mae", "atomization_mae",
                                                            "force_mae", "n", "failed")}
            n_train = len(man["subsets"][lab])
            curve.append({"n": n_train, "label": lab, "kind": kind, "model": path,
                          "train": pick(br.get("train")), "test": pick(br.get("test")),
                          "ood": pick(br.get("ood")), "tier": summ.get("tier")})
            log.event("scored", name=name, tier=summ.get("tier"))
            if kind == "delta":
                M.write_model(os.path.join(out, "models", name), base.skdir, base.dftb,
                              base.repulsive, base.env_models,
                              delta={"kind": "mace", "path": path, "dtype": "float64"},
                              dispersion=base.dispersion,
                              meta={"stage": "A4", "base": base.path, "n_train": n_train})
        result = {"schema": "a4-result/1", "system": args.system,
                  "label": args.label or f"Delta on {os.path.basename(base.path)}",
                  "base_model": base.path, "curve": curve,
                  "architecture": man.get("architecture"), "failed_base": man.get("failed_base")}
        with open(os.path.join(out, "result.json"), "w") as fh:
            json.dump(M._clean(result), fh, indent=1)
        _curve_outputs(out, curve, log)
    return 0


def _curve_outputs(out, curve, log):
    rows = sorted(curve, key=lambda c: (c["n"], c["kind"]))
    L = ["# A4 learning curves (Delta = DFTB + MACE on the residual, pure = MACE alone)", "",
         "| N | kind | test rel (meV/atom) | test F (eV/A) | ood rel | ood F | tier |",
         "|---|---|---|---|---|---|---|"]
    for c in rows:
        t, o = c["test"], c["ood"]
        L.append(f"| {c['n']} | {c['kind']} | {M._fmt1(t.get('relative_mae'))} | "
                 f"{(t.get('force_mae') or float('nan')):.3f} | {M._fmt1(o.get('relative_mae'))} | "
                 f"{(o.get('force_mae') or float('nan')):.3f} | {c.get('tier') or '—'} |")
    with open(os.path.join(out, "learning_curve.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
        for kind, mk in (("delta", "o-"), ("pure", "s--")):
            cs = [c for c in rows if c["kind"] == kind]
            for ax, role in zip(axes, ("test", "ood")):
                x = [c["n"] for c in cs if c[role].get("relative_mae") is not None]
                y = [1e3 * c[role]["relative_mae"] for c in cs if c[role].get("relative_mae") is not None]
                if x:
                    ax.plot(x, y, mk, label=kind)
        for ax, role in zip(axes, ("test", "ood")):
            ax.set_xscale("log")
            ax.set_xlabel("training structures N")
            ax.set_ylabel(f"{role} relative MAE (meV/atom)")
            ax.axhline(10.0, color="0.6", lw=0.8)
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "learning_curve.png"), dpi=150)
        plt.close(fig)
    except Exception as exc:                     # noqa: BLE001  (the plot is optional)
        log.event("plot_skipped", error=str(exc))


def _size(v):
    return "all" if v == "all" else int(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True, help="the base model directory (A1, A2 or A3)")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--sizes", nargs="+", type=_size, default=[25, 50, 100, "all"])
    p.add_argument("--valid-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--max-L", type=int, default=1)
    p.add_argument("--correlation", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--force", action="store_true", help="recompute the base predictions")
    p.set_defaults(func=prepare)
    p = sub.add_parser("train")
    p.add_argument("--out", required=True)
    p.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS))
    p.add_argument("--sizes", nargs="*", default=None, help="labels, e.g. 25 50 all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=train)
    p = sub.add_parser("evaluate")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--system", default="li_o")
    p.add_argument("--label", default=None)
    p.add_argument("--force", action="store_true", help="recompute the MACE predictions")
    p.set_defaults(func=evaluate)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
