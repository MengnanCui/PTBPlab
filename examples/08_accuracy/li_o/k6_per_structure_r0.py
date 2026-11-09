"""K6: how much could an environment-dependent r0 gain at most (A0, section 7)?  Post-processing
of a finished scan; nothing is computed.

* **Bands.** For every band reference, compare its band loss at its own best scan point (the
  per-structure optimum over the scan's points, a lower bound on the grid) with its loss at the
  single point that is best for all references (S3's point).  gain = 1 - own / shared.  A mean
  gain < 15 % means letting r0 follow the environment (channel E2 of A2) cannot improve the bands
  much; try the Hamiltonian corrections (E3) instead.
* **Energies** (the same idea, for the decision between A2 and A3).  For every family, compare
  its within-family relative MAE at its own best point (with that point's S1 sigma) with its MAE
  at the S1 point.  A large gain on some families means they want other radii than the rest.

    python k6_per_structure_r0.py --config scan_li_o.yaml --out runs/k6
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.paths import setup  # noqa: E402

setup()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "scan_li_o.yaml"))
    ap.add_argument("--out", default="runs/k6")
    ap.add_argument("--gate", type=float, default=0.15, help="mean band gain below which E2 is dropped")
    args = ap.parse_args(argv)
    from utils import accuracy as acc
    from utils.runlog import session
    cfg = acc.load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    with session("k6_per_structure_r0", vars(args), root=os.path.join(args.out, "logs")) as log:
        scored = []
        for p in sorted(glob.glob(os.path.join(cfg["out"], "points", "p*", "losses.json"))):
            with open(p) as fh:
                scored.append(json.load(fh))
        if not scored:
            raise SystemExit("no scored points: run ptbp scan run/losses first")
        # bands
        names = sorted({b["name"] for s in scored for b in s.get("bands", {}).get("per_structure", [])})
        loss = {n: np.full(len(scored), np.nan) for n in names}
        for k, s in enumerate(scored):
            for b in s.get("bands", {}).get("per_structure", []):
                if b.get("loss") is not None:
                    loss[b["name"]][k] = b["loss"]
        total = np.array([s.get("bands", {}).get("loss") or np.nan for s in scored], float)
        bands = {}
        if names and np.any(np.isfinite(total)):
            k_shared = int(np.nanargmin(total))
            for n in names:
                own = np.nanmin(loss[n]) if np.any(np.isfinite(loss[n])) else np.nan
                shared = loss[n][k_shared]
                bands[n] = {"own": own, "own_point": int(scored[int(np.nanargmin(loss[n]))]["index"])
                            if np.isfinite(own) else None,
                            "shared": shared, "gain": float(1 - own / shared) if shared else np.nan}
            gains = [v["gain"] for v in bands.values() if np.isfinite(v["gain"])]
            band_summary = {"shared_point": int(scored[k_shared]["index"]),
                            "mean_gain": float(np.mean(gains)) if gains else np.nan,
                            "E2_worth_it": bool(gains and np.mean(gains) >= args.gate)}
        else:
            band_summary = {"note": "no band losses in this scan"}
        # energies by family (S1's sigma at each point)
        fams = sorted({f for s in scored for f in (s["strategies"]["S1"]["summary"]["by_family"])})
        rel = {f: np.full(len(scored), np.nan) for f in fams}
        for k, s in enumerate(scored):
            for f, b in s["strategies"]["S1"]["summary"]["by_family"].items():
                if b.get("relative_mae") is not None:
                    rel[f][k] = b["relative_mae"]
        s1 = np.array([s["strategies"]["S1"]["loss"] or np.nan for s in scored], float)
        k1 = int(np.nanargmin(s1))
        energies = {}
        for f in fams:
            if not np.any(np.isfinite(rel[f])):
                continue
            own = float(np.nanmin(rel[f]))
            energies[f] = {"own_meV": 1e3 * own, "at_S1_meV": 1e3 * float(rel[f][k1]),
                           "own_point": int(scored[int(np.nanargmin(rel[f]))]["index"]),
                           "gain": float(1 - own / rel[f][k1]) if rel[f][k1] else np.nan}
        out = {"scan": cfg["name"], "n_points": len(scored), "bands": bands,
               "band_summary": band_summary, "S1_point": int(scored[k1]["index"]),
               "energies_by_family": energies}
        with open(os.path.join(args.out, "k6_results.json"), "w") as fh:
            json.dump(acc._clean(out), fh, indent=1)
        lines = [f"# K6 ({cfg['name']}, {len(scored)} points)", "",
                 f"bands: {json.dumps(acc._clean(band_summary))}", "",
                 "| reference | own best | at shared point | gain |", "|---|---|---|---|"]
        for n, v in bands.items():
            lines.append(f"| {n} | {v['own']:.4f} | {v['shared']:.4f} | {100 * v['gain']:.0f}% |")
        lines += ["", f"energies (S1 point {out['S1_point']}):", "",
                  "| family | own best (meV/atom) | at S1 | gain |", "|---|---|---|---|"]
        for f, v in energies.items():
            lines.append(f"| {f} | {v['own_meV']:.1f} | {v['at_S1_meV']:.1f} | {100 * v['gain']:.0f}% |")
        with open(os.path.join(args.out, "k6_summary.md"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        log.event("k6", **{k: v for k, v in band_summary.items() if not isinstance(v, dict)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
