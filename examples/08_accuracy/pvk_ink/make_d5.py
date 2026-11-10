"""D5 (ink): clusters cut from the MACE-MPA M409 trajectories plus whole M409 frames, and their
VASP inputs (not submitted).  A series, A1.3.

    python make_d5.py [--every 400] [--radius 6.0] [--full-frames 10] [--out d5]
    python make_d5.py --traj /path/traj.extxyz:train --traj /path/other.extxyz:test

The default trajectories are the 20 ps NVT runs of the perovskite ink project
($PTBP_PEROVSKITE or --perovskite; 5 fs between frames):

* ``runs/M409/T02/mace_mpa_B``: start B, role train;
* ``runs/M409/T02/mace_mpa_A2``: the clean start A, role test.

They are large (218 MB each) and must be copied to the machine.  If none of them can be read, the script
lists what is missing and exits with code 2.

1. **Frames.** Every ``--every``-th frame is used (400 = 2 ps).
2. **Fragments.** The organic atoms (H, C, N, O, S) are joined by covalent bonds (ASE natural
   cut-offs x 1.15, minimum image).  Pb, Sn, Cs and I are one-atom fragments.  The charges
   follow from the formulas (FA+ CH5N2 +1, DMSO and DMF 0, Pb/Sn +2, Cs +1, I -1, and the
   proton-transferred forms).  A frame with an unknown fragment is skipped and reported.
3. **Clusters.** Around each Pb and Sn, every fragment with an atom within ``--radius`` (6 Å,
   minimum image) is taken.  The nearest I− (when the charge is positive) or FA+/Cs+ (when it is
   negative) is then added until the cluster is neutral.  The fragments are unwrapped around the
   metal and centred in a cubic box with ``--vacuum`` (8 Å).  The cluster is non-periodic
   (VASP gets the dipole correction) and named ``clu_<traj>_f<frame>_<metal><index>``, family
   ``clu_<traj>``.
4. **Whole frames.** Up to ``--full-frames`` (10) whole M409 frames, evenly spaced, kept
   periodic (Γ point, no dipole correction), role ood, family ``m409_full``.

The outputs are ``clusters.xyz``, ``manifest.csv``, ``skipped.json``, the VASP cases under
``vasp/``, ``job_NN.sh``, ``run_cases.sh``, ``submit_all.sh`` and ``cost_estimate.json``.  The
cost is scaled from about 150 core-h for a 409-atom frame, times (N/409)².
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import PEROVSKITE, setup  # noqa: E402

setup()
from vasp_ink import boxed, write_case, write_jobs  # noqa: E402

ORGANIC = {"H", "C", "N", "O", "S"}
IONS = {"Pb": 2, "Sn": 2, "Cs": 1, "I": -1}
CHARGES = {"CH5N2": 1, "C2H6OS": 0, "C3H7NO": 0,                 # FA+, DMSO, DMF
           "CH4N2": 0, "C2H7OS": 1, "C3H8NO": 1}                 # after a proton transfer
CATIONS = {"CH5N2", "Cs"}
DEFAULT_TRAJ = [("runs/M409/T02/mace_mpa_B/traj.extxyz", "train"),
                ("runs/M409/T02/mace_mpa_A2/traj.extxyz", "test")]


def hill(symbols):
    c = Counter(symbols)
    keys = (["C", "H"] if "C" in c else []) + sorted(k for k in c if k not in ("C", "H") or "C" not in c)
    return "".join(f"{k}{c[k] if c[k] > 1 else ''}" for k in keys)


def fragments(atoms, mult: float = 1.15):
    """[(indices, formula, charge)] of a periodic frame (see the module docstring); raises
    ValueError on a fragment of unknown charge."""
    from ase.neighborlist import natural_cutoffs, neighbor_list
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    sym = np.array(atoms.get_chemical_symbols())
    org = np.isin(sym, list(ORGANIC))
    cut = [c if o else 0.0 for c, o in zip(natural_cutoffs(atoms, mult=mult), org)]
    i, j = neighbor_list("ij", atoms, cut)
    keep = org[i] & org[j]
    n = len(atoms)
    graph = coo_matrix((np.ones(int(keep.sum())), (i[keep], j[keep])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    out = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) == 1 and sym[idx[0]] in IONS:
            out.append((idx, sym[idx[0]], IONS[sym[idx[0]]]))
            continue
        f = hill(sym[idx])
        if f not in CHARGES:
            raise ValueError(f"unknown fragment {f} (atoms {idx[:6].tolist()}...)")
        out.append((idx, f, CHARGES[f]))
    return out


def mic_vectors(atoms, origin, idx):
    """Minimum-image vectors from point ``origin`` to the atoms ``idx`` (orthorhombic or not)."""
    from ase.geometry import find_mic
    d = atoms.positions[idx] - origin
    v, _ = find_mic(d, atoms.cell, atoms.pbc)
    return v


def cluster(atoms, metal: int, frags, radius: float):
    """The neutral cluster around atom ``metal`` (positions unwrapped around it)."""
    from ase import Atoms
    origin = atoms.positions[metal]
    owner = np.empty(len(atoms), int)
    for k, (idx, _, _) in enumerate(frags):
        owner[idx] = k
    d_all = np.linalg.norm(mic_vectors(atoms, origin, np.arange(len(atoms))), axis=1)
    dist = np.array([d_all[idx].min() for idx, _, _ in frags])
    chosen = [k for k in range(len(frags)) if dist[k] <= radius]
    q = sum(frags[k][2] for k in chosen)
    order = np.argsort(dist)
    for want, ok in ((1, lambda f: f[2] < 0), (-1, lambda f: f[1] in CATIONS)):
        while (q > 0 and want == 1) or (q < 0 and want == -1):
            add = next((k for k in order if k not in chosen and ok(frags[k])), None)
            if add is None:
                raise ValueError("cannot neutralise the cluster")
            chosen.append(int(add))
            q += frags[add][2]
    pos, syms = [], []
    sym = atoms.get_chemical_symbols()
    for k in chosen:
        idx = frags[k][0]
        first = mic_vectors(atoms, origin, idx[:1])[0]              # the fragment's anchor
        rel = mic_vectors(atoms, atoms.positions[idx[0]], idx)      # whole fragment around it
        pos.append(first + rel)
        syms += [sym[i] for i in idx]
    return Atoms(syms, positions=np.vstack(pos)), q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--perovskite", default=PEROVSKITE)
    ap.add_argument("--traj", action="append", default=None,
                    help="PATH:ROLE (repeatable; default the two T02 runs under $PTBP_PEROVSKITE)")
    ap.add_argument("--every", type=int, default=400, help="frame stride (5 fs per frame)")
    ap.add_argument("--radius", type=float, default=6.0, help="A around each metal")
    ap.add_argument("--vacuum", type=float, default=8.0)
    ap.add_argument("--full-frames", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "d5"))
    ap.add_argument("--ncores", type=int, default=64)
    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--dry-run", action="store_true", help="cut and report, write no VASP input")
    args = ap.parse_args(argv)
    import ase.io
    trajs = []
    for spec in (args.traj or [f"{os.path.join(args.perovskite, p)}:{r}" for p, r in DEFAULT_TRAJ]):
        path, role = spec.rsplit(":", 1)
        trajs.append((path, role))
    missing = [p for p, _ in trajs if not os.path.isfile(p)]
    readable = [(p, r) for p, r in trajs if os.path.isfile(p)]
    if not readable:
        print("no trajectory can be read:")
        for p in missing:
            print(f"  missing {p}" + ("  (a dangling symlink)" if os.path.islink(p) else ""))
        print("copy them from the machine that has them, set PTBP_PEROVSKITE, "
              "or pass --traj PATH:ROLE")
        return 2
    from utils.runlog import session
    os.makedirs(args.out, exist_ok=True)
    with session("make_d5", vars(args), root=os.path.join(args.out, "logs")) as log:
        for p in missing:
            log.event("missing", path=p)
        clusters, full, skipped, rows = [], [], [], []
        for path, role in readable:
            tag = os.path.basename(os.path.dirname(path))
            picked = []
            for k, fr in enumerate(ase.io.iread(path, index=f"::{args.every}")):
                frame = k * args.every
                picked.append((frame, fr))
                try:
                    frags = fragments(fr)
                except ValueError as exc:
                    skipped.append({"traj": tag, "frame": frame, "reason": str(exc)})
                    continue
                sym = np.array(fr.get_chemical_symbols())
                for m in np.where(np.isin(sym, ["Pb", "Sn"]))[0]:
                    try:
                        clu, q = cluster(fr, int(m), frags, args.radius)
                    except ValueError as exc:
                        skipped.append({"traj": tag, "frame": frame, "metal": int(m),
                                        "reason": str(exc)})
                        continue
                    a = boxed(clu, args.vacuum)
                    name = f"clu_{tag}_f{frame:05d}_{sym[m]}{int(m)}"
                    a.info = {"name": name, "family": f"clu_{tag}", "role": role,
                              "kspacing": "gamma", "charge": 0, "formula": a.get_chemical_formula(),
                              "source": f"{os.path.relpath(path, args.perovskite)} frame {frame} "
                                        f"around atom {int(m)}, R {args.radius} A"}
                    clusters.append(a)
            full += [(tag, f, fr) for f, fr in picked]
            log.event("trajectory", path=path, role=role, frames=len(picked))
        # whole frames, evenly spaced over all the trajectories, role ood
        sel = np.unique(np.linspace(0, len(full) - 1, min(args.full_frames, len(full))).astype(int)) \
            if full and args.full_frames > 0 else []
        frames_full = []
        for k in sel:
            tag, f, fr = full[int(k)]
            a = fr.copy()
            a.calc = None
            a.info = {"name": f"m409_{tag}_f{f:05d}_full", "family": "m409_full", "role": "ood",
                      "kspacing": "gamma", "charge": 0, "formula": a.get_chemical_formula(),
                      "source": f"{tag} frame {f} (periodic)"}
            frames_full.append(a)
        allf = clusters + frames_full
        ase.io.write(os.path.join(args.out, "clusters.xyz"), allf, format="extxyz")
        for a in allf:
            rows.append({"name": a.info["name"], "family": a.info["family"], "role": a.info["role"],
                         "natoms": len(a), "formula": a.info["formula"], "source": a.info["source"]})
        with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        with open(os.path.join(args.out, "skipped.json"), "w") as fh:
            json.dump(skipped, fh, indent=1)
        sizes = [len(a) for a in clusters]
        log.event("cut", clusters=len(clusters), full_frames=len(frames_full), skipped=len(skipped),
                  min_atoms=min(sizes) if sizes else 0, max_atoms=max(sizes) if sizes else 0)
        if args.dry_run:
            print(f"dry run: {len(clusters)} clusters ({min(sizes, default=0)}-"
                  f"{max(sizes, default=0)} atoms), {len(frames_full)} whole frames; no VASP input")
            return 0
        cases = []
        for a in allf:
            periodic = bool(np.all(a.pbc))
            hours = 150.0 * (len(a) / 409.0) ** 2
            c = write_case(os.path.join(args.out, "vasp", a.info["name"]), a, hours=hours,
                           tags={k: a.info[k] for k in ("name", "family", "role", "source",
                                                        "kspacing")} | {"kind": "sp"})
            c["path"] = f"vasp/{a.info['name']}"
            cases.append(c)
        est = write_jobs(args.out, cases, "pvk_d5", ncores=args.ncores, max_hours=args.max_hours)
        log.event("written", cases=len(cases), jobs=est["jobs"], core_hours=est["core_hours"],
                  yuan=est["yuan_list_price"])
        print(f"{len(cases)} cases, {est['jobs']} job(s), ~{est['core_hours']} core-h, "
              f"~{est['yuan_list_price']} yuan at list price -> {args.out} (NOT submitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
