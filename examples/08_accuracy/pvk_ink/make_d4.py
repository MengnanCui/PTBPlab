"""D4 (ink): VASP inputs for the gas-phase data of the A1.3 fits.  Nothing is submitted.

    python make_d4.py [--structures data/structures.xyz] [--out d4] [--ncores 32]

It writes, in the conventions of vasp_ink.py:

* ``sp/<name>/``: a single point for every structure of structures.xyz (prepare_structures.py);
* ``relax/<name>/``: relaxations (IBRION 2, NSW 100, EDIFFG -0.02) of the families matching
  ``--relax-families`` (default the complexes and the ion clusters, whose geometries are MACE
  or hand-built).  Their ionic steps become extra training structures in collect_d4.py;
* ``atoms/atom_<El>/``: the isolated atoms of every element present, spin-unpolarized and in a
  20 x 20.5 x 21 Å box (orthorhombic, to break the symmetry of partly filled shells).  They
  give E0s_dft.json, the reference of the atomization energies, of the same kind as DFTB's
  neutral spin-free atoms;
* ``job_NN.sh``: Slurm jobs for dg83, each under ``--max-hours`` of estimated wall time;
* ``run_cases.sh``: the runner, whose VASP and POTCAR paths are set by environment variables;
* ``submit_all.sh``: for the user to run after checking ``cost_estimate.json``;
* ``README.txt``.

The cost model is rough: about 3 core-h per single point of 40 atoms, growing with N², and
about 8 single points per relaxation.  Calibrate it with one test case before submitting
everything (``bash run_cases.sh sp/<name>`` inside an interactive job).
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import setup  # noqa: E402

setup()
from vasp_ink import write_case, write_jobs  # noqa: E402

ATOM_BOX = [20.0, 20.5, 21.0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--structures", default=os.path.join(HERE, "data", "structures.xyz"))
    ap.add_argument("--out", default=os.path.join(HERE, "d4"))
    ap.add_argument("--relax-families", nargs="*", default=["cplx_*", "ion_*"],
                    help="family patterns that also get a relaxation (none: no relaxations)")
    ap.add_argument("--relax-rattled", action="store_true",
                    help="also relax the rattled copies (default: only the originals)")
    ap.add_argument("--only", nargs="*", default=None, help="name or family patterns")
    ap.add_argument("--ncores", type=int, default=32)
    ap.add_argument("--max-hours", type=float, default=12.0, help="estimated wall time per job")
    ap.add_argument("--nbands-extra", type=int, default=24)
    ap.add_argument("--dry-run", action="store_true", help="count and estimate, write nothing")
    args = ap.parse_args(argv)
    import ase.io
    from ase import Atoms
    from vasp_ink import core_hours
    frames = ase.io.read(args.structures, ":")
    if args.only:
        frames = [a for a in frames if any(fnmatch.fnmatch(a.info["name"], p)
                                           or fnmatch.fnmatch(a.info.get("family", ""), p)
                                           for p in args.only)]
    relax = [a for a in frames if args.relax_families
             and any(fnmatch.fnmatch(a.info.get("family", ""), p) for p in args.relax_families)
             and (args.relax_rattled or not re.search(r"_r\d+$", a.info["name"]))]
    elements = sorted({s for a in frames for s in a.get_chemical_symbols()})
    if args.dry_run:
        sp_h = sum(core_hours(len(a)) for a in frames)
        rx_h = sum(8 * core_hours(len(a)) for a in relax)
        print(f"{len(frames)} single points (~{sp_h:.0f} core-h), {len(relax)} relaxations "
              f"(~{rx_h:.0f} core-h), {len(elements)} atoms; ~{0.1 * (sp_h + rx_h):.0f} yuan list")
        return 0
    from utils.runlog import session
    os.makedirs(args.out, exist_ok=True)
    with session("make_d4", vars(args), root=os.path.join(args.out, "logs")) as log:
        cases = []
        for a in frames:
            tags = {k: a.info.get(k) for k in ("name", "family", "role", "source", "kspacing")}
            c = write_case(os.path.join(args.out, "sp", a.info["name"]), a,
                           nbands_extra=args.nbands_extra, tags={**tags, "kind": "sp"})
            c["path"] = f"sp/{a.info['name']}"
            cases.append(c)
        for a in relax:
            tags = {k: a.info.get(k) for k in ("name", "family", "role", "source", "kspacing")}
            c = write_case(os.path.join(args.out, "relax", a.info["name"]), a, relax=True,
                           nbands_extra=args.nbands_extra, tags={**tags, "kind": "relax"})
            c["path"] = f"relax/{a.info['name']}"
            cases.append(c)
        for el in elements:
            at = Atoms(el, positions=[[10.0, 10.25, 10.5]], cell=ATOM_BOX, pbc=False)
            c = write_case(os.path.join(args.out, "atoms", f"atom_{el}"), at, dipole=False,
                           nbands_extra=8, tags={"name": f"atom_{el}", "family": "atom",
                                                 "role": "e0", "kind": "atom"})
            c["path"] = f"atoms/atom_{el}"
            cases.append(c)
        est = write_jobs(args.out, cases, "pvk_d4", ncores=args.ncores, max_hours=args.max_hours)
        with open(os.path.join(args.out, "README.txt"), "w") as fh:
            fh.write(
                "D4 (ink): VASP inputs written by examples/08_accuracy/pvk_ink/make_d4.py.\n"
                "Nothing has been submitted.  Check cost_estimate.json, run one case in an\n"
                "interactive job to calibrate (bash run_cases.sh sp/<name>), then\n"
                "  bash submit_all.sh\n"
                "Set VASP_BIN_DIR / VASP_PP_PATH / VASP_MODULES if the defaults (the dghpc\n"
                "account) do not apply.  Collect with collect_d4.py when the jobs are done.\n")
        log.event("written", single_points=len(frames), relaxations=len(relax), atoms=len(elements),
                  jobs=est["jobs"], core_hours=est["core_hours"], yuan=est["yuan_list_price"])
        print(f"{len(cases)} cases in {est['jobs']} job(s): ~{est['core_hours']} core-h, "
              f"~{est['yuan_list_price']} yuan at list price -> {args.out} (NOT submitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
