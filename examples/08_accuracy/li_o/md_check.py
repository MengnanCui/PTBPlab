"""MD stability of an exported model (part of the acceptance line: 300-600 K MD stable).

    ptbp scan export scan_li_o.yaml --point I --strategy S1 --out models/S1_pI
    python md_check.py --model models/S1_pI --config scan_li_o.yaml --out runs/md_S1_pI \\
        [--systems iface li16 li2o12] [--temps 300 600] [--langevin 300 --nve 2000]

For each system and temperature it runs Langevin and then NVE (pydftb.assess.nve_check) with
``PyDFTB(model, repulsive="skf")``.  It reports the energy drift (meV/atom/ps), the shortest
distance per element pair, a collapse flag, and a failed SCC if one happens.  The collapse
flag is set when Li-Li < 2.0 A or Li-O < 1.45 A (DFT: about 2.4 and 1.75 A at the shortest in
the interface data).  Trajectories (every 10 steps) are written next to the results.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.paths import setup  # noqa: E402

setup()

SYSTEMS = {"iface": "iface_init", "li16": "rat_li_bcc16_s05_train1", "li2o12": "rat_li2o12_s05_train1"}
COLLAPSE = {"Li-Li": 2.0, "Li-O": 1.45, "O-O": 1.2}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="ptbp scan export directory")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "scan_li_o.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    ap.add_argument("--temps", nargs="+", type=float, default=[300.0, 600.0])
    ap.add_argument("--langevin", type=int, default=300)
    ap.add_argument("--nve", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=1.0, help="fs")
    args = ap.parse_args(argv)
    from pydftb import assess
    from pydftb.ase_calc import PyDFTB
    from utils import accuracy as acc
    from utils.runlog import session
    cfg = acc.load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.model, "model.json")) as fh:
        model = json.load(fh)
    with session("md_check", vars(args), root=os.path.join(args.out, "logs")) as log:
        frames = {f.info["name"]: f for f in acc.load_data(cfg)}
        results = {}
        for sysname in args.systems:
            atoms = frames[SYSTEMS[sysname]]
            for t in args.temps:
                calc = PyDFTB(args.model, max_l=model.get("max_l"), kpts=acc.kpoints_for(atoms, 0.15),
                              smearing=("gaussian", 0.05 / acc.KB_EV), repulsive="skf",
                              scc_tol=1e-6, max_scc=500, reuse_charges=True)
                key = f"{sysname}_{int(t)}K"
                with log.step("md", system=sysname, temperature=t):
                    r = assess.nve_check(atoms, calc, temperature=t, langevin_steps=args.langevin,
                                         nve_steps=args.nve, dt_fs=args.dt,
                                         trajectory=os.path.join(args.out, f"{key}.traj"))
                r["collapse"] = any(r["shortest"].get(p, 9.9) < d for p, d in COLLAPSE.items())
                results[key] = r
                log.event("md_result", system=key, drift=r.get("drift_meV_per_atom_per_ps"),
                          collapse=r["collapse"], failed=r["failed"])
        with open(os.path.join(args.out, "md_results.json"), "w") as fh:
            json.dump(acc._clean(results), fh, indent=1)
        lines = [f"# MD check of {args.model}", "",
                 "| run | drift (meV/atom/ps) | shortest Li-Li | Li-O | O-O | collapse | failed |",
                 "|---|---|---|---|---|---|---|"]
        for k, r in results.items():
            s = r["shortest"]
            lines.append(f"| {k} | {r.get('drift_meV_per_atom_per_ps', float('nan')):.3f} | "
                         f"{s.get('Li-Li', float('nan')):.2f} | {s.get('Li-O', float('nan')):.2f} | "
                         f"{s.get('O-O', float('nan')):.2f} | {r['collapse']} | {r['failed'] or ''} |")
        with open(os.path.join(args.out, "md_summary.md"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
