"""K7: what the same data can support (A0, section 7).  A MACE model trained on exactly the
Li-O training set of the scans is scored with the same assessment.  If even MACE misses tier A
on the test and ood sets, the data are the limit: add data (D2, D3) before more method work.
MACE-MP-0, zero-shot, is reported alongside.

    python k7_mace.py prepare  --config scan_li_o.yaml --out runs/k7
    python k7_mace.py train    --out runs/k7 [--device cuda] [--epochs 400]   # remote (GPU: slurm/mace_gpu.sh)
    python k7_mace.py evaluate --config scan_li_o.yaml --out runs/k7 [--foundation medium]

``prepare`` writes train/valid/test files with REF_energy and REF_forces and the E0s (DFT
isolated atoms, as the scans use).  ``train`` calls mace-torch's ``mace_run_train``
(``python -m mace.cli.run_train``) with the arguments saved in mace_args.json; check them for
your mace-torch version.  ``evaluate`` scores the trained model and MACE-MP-0 with
pydftb.assess (tiers, reactions) and writes k7_summary.md.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.paths import setup  # noqa: E402

setup()


def prepare(args, cfg, log):
    import ase.io
    from ase.data import atomic_numbers
    from utils import accuracy as acc
    frames = acc.load_data(cfg)
    rng = np.random.default_rng(0)
    out = {}
    for role in ("train", "test", "ood"):
        sel = []
        for a in frames:
            if a.info["role"] != role:
                continue
            b = a.copy()
            b.info["REF_energy"] = a.get_potential_energy()
            b.arrays["REF_forces"] = np.array(a.get_forces())
            b.info["config_type"] = a.info["family"]
            sel.append(b)
        out[role] = sel
    train = out["train"]
    idx = rng.permutation(len(train))
    n_val = max(1, len(train) // 10)
    valid = [train[i] for i in idx[:n_val]]
    train = [train[i] for i in idx[n_val:]]
    paths = {}
    for name, fr in (("train", train), ("valid", valid), ("test", out["test"]), ("ood", out["ood"])):
        paths[name] = os.path.join(args.out, f"{name}.xyz")
        ase.io.write(paths[name], fr, format="extxyz")
        log.event("wrote", set=name, n=len(fr))
    e0 = acc.load_e0s(cfg)
    e0s = {int(atomic_numbers[k]): v for k, v in e0.items()}
    margs = ["--name", "mace_k7", "--train_file", paths["train"], "--valid_file", paths["valid"],
             "--test_file", paths["test"], "--E0s", json.dumps(e0s).replace('"', ""),
             "--energy_key", "REF_energy", "--forces_key", "REF_forces", "--model", "MACE",
             "--r_max", "5.0", "--num_channels", "64", "--max_L", "1", "--correlation", "3",
             "--batch_size", "4", "--valid_batch_size", "4", "--max_num_epochs", str(args.epochs),
             "--swa", "--start_swa", str(int(0.8 * args.epochs)), "--energy_weight", "1.0",
             "--forces_weight", "10.0", "--default_dtype", "float64", "--seed", "1",
             "--results_dir", os.path.join(args.out, "results"),
             "--model_dir", args.out, "--checkpoints_dir", os.path.join(args.out, "checkpoints"),
             "--log_dir", os.path.join(args.out, "mace_logs")]
    with open(os.path.join(args.out, "mace_args.json"), "w") as fh:
        json.dump(margs, fh, indent=1)
    print("mace arguments -> mace_args.json (check them for your mace-torch version)")


def train(args, log):
    with open(os.path.join(args.out, "mace_args.json")) as fh:
        margs = json.load(fh)
    cmd = [sys.executable, "-m", "mace.cli.run_train", *margs, "--device", args.device]
    log.event("train", cmd=" ".join(cmd))
    with open(os.path.join(args.out, "mace_train.log"), "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    log.event("train_done", returncode=proc.returncode)
    if proc.returncode:
        raise SystemExit(f"mace training failed ({proc.returncode}); see mace_train.log")


def evaluate(args, cfg, log):
    from pydftb import assess
    from utils import accuracy as acc
    frames = acc.load_data(cfg)
    e0 = acc.load_e0s(cfg)
    reactions = assess.load_reactions(cfg["reactions"]) if cfg.get("reactions") else None
    models = {}
    model_path = args.model or os.path.join(args.out, "mace_k7.model")
    if os.path.exists(model_path):
        from mace.calculators import MACECalculator
        models["mace_k7"] = lambda: MACECalculator(model_paths=model_path, device=args.device,
                                                   default_dtype="float64")
    if args.foundation:
        from mace.calculators import mace_mp
        models[f"mace_mp_{args.foundation}"] = lambda: mace_mp(model=args.foundation,
                                                               device=args.device,
                                                               default_dtype="float64")
    results = {}
    for name, make in models.items():
        calc = make()
        with log.step("evaluate", model=name):
            rec = assess.predict(frames, lambda a: calc)
            # MACE predicts total energies with its own E0s; MACE-MP-0's isolated atoms differ
            # from the DFT E0s, so its atomization reference is its own isolated atoms
            if name.startswith("mace_mp"):
                from ase import Atoms
                e0p = {}
                for el in e0:
                    at = Atoms(el, positions=[[10, 10, 10]], cell=[20] * 3, pbc=False)
                    at.calc = calc
                    e0p[el] = float(at.get_potential_energy())
            else:
                e0p = dict(e0)
            summ = assess.summarize(rec, e0, e0p, reactions)
        results[name] = summ
        assess.write_markdown(summ, os.path.join(args.out, f"k7_{name}.md"), title=f"K7 {name}")
    with open(os.path.join(args.out, "k7_results.json"), "w") as fh:
        json.dump(acc._clean(results), fh, indent=1)
    lines = ["# K7: MACE on the same data", ""]
    for name, s in results.items():
        br = s["energies"]["by_role"]
        lines.append(f"- {name}: tier {s['tier']}; " + "; ".join(
            f"{r} rel {1e3 * (b.get('relative_mae') or np.nan):.1f} meV/atom, F {b.get('force_mae', np.nan):.3f}"
            for r, b in br.items()))
    with open(os.path.join(args.out, "k7_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["prepare", "train", "evaluate"])
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "scan_li_o.yaml"))
    ap.add_argument("--out", default="runs/k7")
    ap.add_argument("--epochs", type=int, default=400, help="written into mace_args.json by prepare")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default=None)
    ap.add_argument("--foundation", default="medium", help="MACE-MP-0 size ('' to skip)")
    args = ap.parse_args(argv)
    from utils import accuracy as acc
    from utils.runlog import session
    os.makedirs(args.out, exist_ok=True)
    with session(f"k7_{args.stage}", vars(args), root=os.path.join(args.out, "logs")) as log:
        if args.stage == "prepare":
            prepare(args, acc.load_config(args.config), log)
        elif args.stage == "train":
            train(args, log)
        else:
            evaluate(args, acc.load_config(args.config), log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
