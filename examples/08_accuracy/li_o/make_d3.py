"""D3 (A series, A0 section 2): Li | Li2O interface frames for DFT, sampled by MACE-MP-0, not by
DFTB (whose dynamics collapse the interface, M5c/M5d).

Stages:

* ``sample`` (remote; needs mace-torch): Langevin MD with MACE-MP-0 medium (float64, 1 fs) of
  the 80-atom Li64O16 superlattice of examples/07 (make_interface.interface at the relaxed
  lattice constants) and of 2 variants strained +-2 % in-plane, at 300 / 600 / 900 K.
  Trajectories (extxyz, one frame every ``--every`` fs) go to ``--traj-dir``, by default
  ``$PTBP_OFFLOAD/pydftb/a1_interface`` (~/Documents/MyGit/PProjects-offload/...), because
  files of 20 MB or more must not land in the Nextcloud-synced PProjects;
* ``select``: farthest-point sampling of ``--n`` frames (default 80) on a descriptor of
  element-resolved RDFs (Li-Li, Li-O, O-O, 1-6 A) and the histogram of the O coordination of
  Li.  The first ``--burn`` fs of every MACE trajectory are skipped.  ``--extra-traj`` adds
  frames from other trajectories (e.g. the DFTB collapse runs of md_interface.py) as
  negative examples;
* ``write``: VASP single points as examples/07 make_iface_dft.py (KSPACING 0.25, ISYM 0, NCORE
  4), job scripts with a wall-time cap from ``--budget``, manifest.csv and cost_estimate.json.
  The cost is about 1.5 yuan per 80-atom frame (M5c: 3 frames billed 4.49 yuan), plus the
  rough core-hour model of make_d2.py.  Nothing is submitted.

Roles: frames of the unstrained interface and of ``--extra-traj`` are ``ood`` (the test of
transfer); the strained variants are ``train`` (``--ood-variants`` changes this).

    python make_d3.py sample --out ../runs/d3 [--steps 5000] [--device cuda]      # remote
    python make_d3.py select --out ../runs/d3 [--n 80] [--extra-traj ../md_c60s4/relax.traj]
    python make_d3.py write  --out ../runs/d3 [--ncores 76] [--budget 150]
    python make_d3.py sample --out ../runs/d3 --dry-run     # what would run, no MACE import
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import OFFLOAD, setup  # noqa: E402

setup()
from make_d2 import (core_hours, equilibrium, kgrid, nk_time_reversal, write_cost,  # noqa: E402
                     write_jobs, write_manifest)

TEMPS = (300, 600, 900)
STRAINS = {"base": 1.0, "strain_p2": 1.02, "strain_m2": 0.98}
YUAN_PER_FRAME_M5C = 4.49 / 3            # billed per 80-atom frame (M5c interface check)
PAIRS = (("Li", "Li"), ("Li", "O"), ("O", "O"))
R_BINS = np.linspace(1.0, 6.0, 51)


def default_traj_dir() -> str:
    return os.path.join(OFFLOAD, "pydftb", "a1_interface")


def structures(path: str | None = None) -> dict:
    """{variant: Atoms}: the M5c interface and its strained variants, or one given structure."""
    import ase.io
    if path:
        return {"given": ase.io.read(path)}
    from make_interface import interface
    eq = equilibrium()
    base = interface(eq["a_li2o"], eq["a_bcc"])
    out = {}
    for name, f in STRAINS.items():
        a = base.copy()
        cell = a.cell.array.copy()
        cell[0] *= f
        cell[1] *= f
        a.set_cell(cell, scale_atoms=True)
        out[name] = a
    return out


def shortest(atoms, a="Li", b="Li") -> float:
    sym = np.array(atoms.get_chemical_symbols())
    d = atoms.get_all_distances(mic=True)
    m = np.outer(sym == a, sym == b) | np.outer(sym == b, sym == a)
    np.fill_diagonal(m, False)
    return float(d[m].min()) if m.any() else float("nan")


def stage_sample(args, log) -> None:
    import ase.io
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from mace.calculators import mace_mp
    traj_dir = os.path.abspath(args.traj_dir)
    os.makedirs(traj_dir, exist_ok=True)
    calc = mace_mp(model=args.model, default_dtype="float64", device=args.device)
    summary = {}
    for variant, at0 in structures(args.structure).items():
        for temp in args.temps:
            name = f"{variant}_T{temp}"
            path = os.path.join(traj_dir, f"{name}.extxyz")
            if os.path.exists(path) and not args.force:
                print(f"[d3] skip {name} (exists)")
                continue
            at = at0.copy()
            at.calc = calc
            rng = np.random.default_rng(args.seed + temp)
            MaxwellBoltzmannDistribution(at, temperature_K=temp, rng=rng)
            dyn = Langevin(at, 1.0 * units.fs, temperature_K=temp, friction=0.01 / units.fs,
                           rng=np.random.default_rng(args.seed + temp + 1))
            tmp = path + ".part"
            if os.path.exists(tmp):
                os.remove(tmp)
            t0 = time.perf_counter()
            mins = []

            def dump():
                step = dyn.nsteps
                a = at.copy()
                a.calc = None
                e = float(at.get_potential_energy())
                a.info.update(variant=variant, temperature=temp, time_fs=float(step),
                              step=int(step), e_mace=e, source=f"MACE-MP-0 {args.model} Langevin")
                ase.io.write(tmp, a, format="extxyz", append=True)
                mins.append(shortest(at))
            dyn.attach(dump, interval=args.every)
            dyn.run(args.steps)
            os.replace(tmp, path)
            summary[name] = {"steps": args.steps, "frames": len(mins),
                             "shortest_LiLi_min": float(np.min(mins)),
                             "seconds": round(time.perf_counter() - t0, 1), "path": path}
            log.event("trajectory", name=name, frames=len(mins),
                      shortest_LiLi=summary[name]["shortest_LiLi_min"])
    with open(os.path.join(args.out, "sample_summary.json"), "a") as fh:
        fh.write(json.dumps(summary) + "\n")


def descriptor(atoms) -> np.ndarray:
    """Element-resolved RDFs (Gaussian-smeared, per pair normalised) and the histogram of the
    O coordination (< 2.5 A) of the Li atoms: one vector per frame."""
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list("ijd", atoms, R_BINS[-1])
    sym = np.array(atoms.get_chemical_symbols())
    centres = 0.5 * (R_BINS[1:] + R_BINS[:-1])
    parts = []
    for a, b in PAIRS:
        m = (sym[i] == a) & (sym[j] == b)
        h = np.exp(-0.5 * ((d[m][:, None] - centres[None, :]) / 0.1) ** 2).sum(0)
        parts.append(h / max(h.sum(), 1e-12))
    li = sym == "Li"
    m = (sym[i] == "Li") & (sym[j] == "O") & (d < 2.5)
    cn = np.bincount(i[m], minlength=len(atoms))[li]
    hist = np.bincount(np.clip(cn, 0, 8), minlength=9).astype(float)
    parts.append(hist / max(hist.sum(), 1.0))
    return np.concatenate(parts)


def farthest_point(x: np.ndarray, n: int) -> list:
    x = np.asarray(x, float)
    if len(x) <= n:
        return list(range(len(x)))
    first = int(np.argmax(np.linalg.norm(x - x.mean(0), axis=1)))
    chosen = [first]
    dmin = np.linalg.norm(x - x[first], axis=1)
    while len(chosen) < n:
        k = int(np.argmax(dmin))
        chosen.append(k)
        dmin = np.minimum(dmin, np.linalg.norm(x - x[k], axis=1))
    return chosen


def stage_select(args, log) -> None:
    import ase.io
    frames, origin = [], []
    for path in sorted(glob.glob(os.path.join(os.path.abspath(args.traj_dir), "*.extxyz"))):
        for k, a in enumerate(ase.io.read(path, ":")):
            if a.info.get("time_fs", 0.0) < args.burn:
                continue
            frames.append(a)
            origin.append({"file": path, "frame": k, "variant": a.info.get("variant", "given"),
                           "kind": "mace"})
    for path in args.extra_traj or []:
        for k, a in enumerate(ase.io.read(path, f"::{max(1, args.extra_every)}")):
            a.info.update(variant="extra", source=f"extra:{path}")
            frames.append(a)
            origin.append({"file": path, "frame": k, "variant": "extra", "kind": "extra"})
    if not frames:
        raise SystemExit(f"no frames in {args.traj_dir} (run `sample` first) or --extra-traj")
    x = np.array([descriptor(a) for a in frames])
    pick = farthest_point(x, args.n)
    sel = []
    for n, k in enumerate(pick):
        a = frames[k].copy()
        a.calc = None
        a.info.update(name=f"iface_d3_{n:03d}", origin_file=origin[k]["file"],
                      origin_frame=origin[k]["frame"], variant=origin[k]["variant"])
        sel.append(a)
    out = os.path.join(args.out, "selected.extxyz")
    if not args.dry_run:
        ase.io.write(out, sel, format="extxyz")
        with open(os.path.join(args.out, "selection.json"), "w") as fh:
            json.dump({"n_candidates": len(frames), "picked": [origin[k] for k in pick],
                       "burn_fs": args.burn}, fh, indent=1)
    by = {}
    for k in pick:
        by[origin[k]["variant"]] = by.get(origin[k]["variant"], 0) + 1
    log.event("selected", candidates=len(frames), n=len(sel), by_variant=by)
    print(f"[d3] {len(sel)} of {len(frames)} frames selected {by} -> {out}")


def stage_write(args, log) -> None:
    import ase.io
    from vasp_common import write_case
    sel = ase.io.read(os.path.join(args.out, "selected.extxyz"), ":")
    ood = {v.strip() for v in args.ood_variants.split(",")}
    rows, dirs = [], []
    for a in sel:
        name = a.info["name"]
        variant = a.info.get("variant", "given")
        g = kgrid(a, 0.25)
        nk = nk_time_reversal(g)
        rows.append({"name": name, "family": f"iface_md_{variant}",
                     "role": "ood" if variant in ood else "train", "natoms": len(a),
                     "formula": a.get_chemical_formula(), "kspacing": 0.25,
                     "kgrid": "x".join(map(str, g)), "nk_tr": nk,
                     "est_core_h": round(core_hours(a, nk), 2),
                     "source": f"{a.info.get('origin_file')}#{a.info.get('origin_frame')}"})
        if not args.dry_run:
            write_case(os.path.join(args.out, "vasp", name), a, KSPACING=0.25, ISYM=0, NCORE=4,
                       LREAL=".FALSE.")
        dirs.append(name)
    m5c = YUAN_PER_FRAME_M5C * len(rows)
    tot = sum(r["est_core_h"] for r in rows)
    print(f"[d3] {len(rows)} frames: ~{m5c:.0f} yuan at the M5c price "
          f"({YUAN_PER_FRAME_M5C:.2f}/frame), model {tot:.0f} core-h = {0.1 * tot:.0f} yuan list")
    if args.dry_run:
        return
    vdir = os.path.join(args.out, "vasp")
    write_manifest(vdir, rows)
    scripts = write_jobs(vdir, dirs, "a0_d3", args.ncores, args.budget, args.jobs)
    write_cost(vdir, rows, args.ncores, args.budget,
               {"yuan_at_m5c_price": round(m5c, 2), "m5c_price_per_frame": YUAN_PER_FRAME_M5C})
    log.event("written", frames=len(rows), jobs=scripts, yuan_m5c=round(m5c, 2))
    print(f"[d3] wrote {vdir}: {scripts}, manifest.csv, cost_estimate.json; nothing submitted")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["sample", "select", "write"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj-dir", default=None, help="MACE trajectories (default under PTBP_OFFLOAD)")
    ap.add_argument("--structure", default=None, help="sample: one given structure instead")
    ap.add_argument("--temps", type=int, nargs="+", default=list(TEMPS))
    ap.add_argument("--steps", type=int, default=5000, help="MD steps of 1 fs")
    ap.add_argument("--every", type=int, default=10, help="fs between saved frames")
    ap.add_argument("--model", default="medium", help="MACE-MP-0 model size")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n", type=int, default=80, help="select: frames to keep")
    ap.add_argument("--burn", type=float, default=500.0, help="select: fs skipped per trajectory")
    ap.add_argument("--extra-traj", nargs="*", default=[], help="select: more trajectories")
    ap.add_argument("--extra-every", type=int, default=5)
    ap.add_argument("--ood-variants", default="base,extra,given")
    ap.add_argument("--ncores", type=int, default=76)
    ap.add_argument("--budget", type=float, default=150.0, help="write: yuan (list) for the caps")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    args.out = os.path.abspath(args.out)
    args.traj_dir = args.traj_dir or default_traj_dir()
    if "Documents/PProjects" in os.path.abspath(args.traj_dir):
        print("[d3] warning: the trajectories go into the Nextcloud-synced PProjects; "
              "use --traj-dir under PTBP_OFFLOAD for files of 20 MB or more")
    if args.stage == "sample" and args.dry_run:
        for variant, at in structures(args.structure).items():
            print(f"  {variant:10s} {at.get_chemical_formula():8s} {len(at)} atoms, cell "
                  f"{np.round(at.cell.lengths(), 3)}, shortest Li-Li {shortest(at):.2f} A, "
                  f"T {args.temps}, {args.steps} steps -> {args.traj_dir}")
        print(f"[d3] {len(args.temps) * len(structures(args.structure))} trajectories, "
              f"{args.steps // args.every} frames each (not run: --dry-run)")
        return 0
    from utils.runlog import session
    os.makedirs(args.out, exist_ok=True)
    with session(f"d3_{args.stage}", vars(args), root=os.path.join(args.out, "logs")) as log:
        {"sample": stage_sample, "select": stage_select, "write": stage_write}[args.stage](args, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
