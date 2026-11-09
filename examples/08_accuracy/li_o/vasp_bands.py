"""D1 with VASP (A series): the band references of bands_gpaw.py from the code of the training
data, written as inputs and never submitted.

These are the same structures as bands_gpaw.reference_structures:

* 15 crystals (bcc, fcc, hcp Li, Li2O, Li2O2 at 0.952 / 1.000 / 1.048 V0);
* 15 dimers (Li2, LiO, O2 at 5 distances).

Each crystal gets two directories:

* ``<name>/scf/``: the training settings (vasp_common.BASE: PBE, Li_sv/O, ENCUT 800, KSPACING
  0.15, ISMEAR 0, SIGMA 0.05, ISPIN 1) with LCHARG .TRUE.;
* ``<name>/band/``: ICHARG 11 on that charge density, explicit k-points (KPOINTS from
  pydftb.bandref.vasp_kpoints_text, the k-points of pydftb.bandref.special_kpoints), ISYM 0,
  LORBIT 11, NBANDS = NELECT/2 + max(12, 4 N); ``kpath.json`` keeps the labels, levels and
  segments for collect_bands.py.

A dimer gets only ``<name>/scf/`` (Gamma, ISYM 0, NBANDS = NELECT/2 + 8), whose eigenvalues
are the orbital reference.  ``run_dirs.sh`` copies scf/CHGCAR into band/ before the band run.

    python vasp_bands.py --out ../runs/d1_vasp [--ncores 32] [--budget 8] [--dry-run]
    # on the cluster: bash submit_all.sh (by hand); afterwards collect_bands.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import setup  # noqa: E402

setup()
from bands_gpaw import band_kpoints, reference_structures  # noqa: E402
from make_d2 import (KSPACING, core_hours, kgrid, nk_time_reversal, write_cost,  # noqa: E402
                     write_jobs, write_manifest)

ZVAL = {"Li": 3, "O": 6}                        # Li_sv, O (potpaw_PBE.54)


def nelect(atoms) -> int:
    return int(sum(ZVAL[s] for s in atoms.get_chemical_symbols()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", type=int, default=30, help="path points per segment")
    ap.add_argument("--ncores", type=int, default=32)
    ap.add_argument("--budget", type=float, default=8.0, help="yuan (list) for the wall-time cap")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-dimers", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list cases and costs, write nothing")
    args = ap.parse_args(argv)
    from pydftb.bandref import vasp_kpoints_text
    items = reference_structures(dimers=not args.no_dimers)
    plan, rows = [], []
    for s in items:
        at = s["atoms"]
        ne = nelect(at)
        if s["kind"] == "path":
            k, labels, level, segments = band_kpoints(at, args.points)
            nb = int(math.ceil(ne / 2)) + max(12, 4 * len(at))
            g = kgrid(at, KSPACING)
            nk = nk_time_reversal(g)
            est = core_hours(at, nk) + 0.5 * core_hours(at, len(k))   # non-SCF: no SCF cycles
            plan.append((s, k, labels, level, segments, nb))
            rows.append({"name": s["name"], "family": s["group"], "role": "bandref",
                         "natoms": len(at), "formula": at.get_chemical_formula(),
                         "kspacing": KSPACING, "kgrid": "x".join(map(str, g)),
                         "nk_tr": f"{nk}+{len(k)}", "est_core_h": round(est, 2),
                         "source": s["source"]})
        else:
            nb = int(math.ceil(ne / 2)) + 8
            plan.append((s, None, None, None, None, nb))
            rows.append({"name": s["name"], "family": s["group"], "role": "bandref",
                         "natoms": len(at), "formula": at.get_chemical_formula(),
                         "kspacing": "gamma", "kgrid": "1x1x1", "nk_tr": 1,
                         "est_core_h": round(core_hours(at, 1), 2), "source": s["source"]})
    tot = sum(r["est_core_h"] for r in rows)
    for r in rows:
        print(f"  {r['name']:22s} {r['natoms']:3d} atoms  scf k {r['kgrid']:>9s}  "
              f"(+band) {r['nk_tr']}  ~{r['est_core_h']:.2f} core-h")
    print(f"[d1-vasp] {len(rows)} structures, ~{tot:.1f} core-h = {0.1 * tot:.2f} yuan list")
    if args.dry_run:
        return 0
    from utils.runlog import session
    from vasp_common import write_case
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    with session("d1_vasp_bands", vars(args), root=os.path.join(out, "logs")) as log:
        for s, k, labels, level, segments, nb in plan:
            at = s["atoms"]
            base = os.path.join(out, s["name"])
            if s["kind"] == "path":
                write_case(os.path.join(base, "scf"), at, LCHARG=".TRUE.")
                band = os.path.join(base, "band")
                write_case(band, at, ICHARG=11, LORBIT=11, ISYM=0, NBANDS=nb, KSPACING=None,
                           KGAMMA=None, LCHARG=".FALSE.")
                with open(os.path.join(band, "KPOINTS"), "w") as fh:
                    fh.write(vasp_kpoints_text(k, comment=f"{s['name']} band reference (A0 D1)"))
                with open(os.path.join(band, "kpath.json"), "w") as fh:
                    json.dump({"name": s["name"], "group": s["group"], "kind": "path",
                               "kpts": np.asarray(k).tolist(), "labels": list(labels),
                               "level": np.asarray(level).tolist(), "segments": segments,
                               "scaling": s["scaling"], "source": s["source"],
                               "kspacing": KSPACING}, fh, indent=1)
            else:
                d = os.path.join(base, "scf")
                write_case(d, at, gamma_only=True, ISPIN=1, ISYM=0, NBANDS=nb)
                with open(os.path.join(d, "kpath.json"), "w") as fh:
                    json.dump({"name": s["name"], "group": s["group"], "kind": "molecule",
                               "source": s["source"], "kspacing": "gamma"}, fh, indent=1)
        write_manifest(out, rows)
        scripts = write_jobs(out, [s["name"] for s, *_ in plan], "a0_d1", args.ncores,
                             args.budget, args.jobs)
        cost = write_cost(out, rows, args.ncores, args.budget)
        log.event("written", cases=len(plan), jobs=scripts, est_core_h=cost["est_core_h"])
        print(f"[d1-vasp] wrote {len(plan)} structures, {scripts}; nothing submitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
