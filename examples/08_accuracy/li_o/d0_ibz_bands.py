"""D0: band data from the existing VASP SCF runs, at no cost (A0 plan, section 2).  Every SCF run
of the Li-O data (M5c, M5d) has its eigenvalues on the IBZ k-points of its own grid; this
script turns them into BandRef JSON files (pydftb.bandref, kind "ibz", with the k-point
weights) named after the structures of the scan config, for the band loss of ``ptbp scan``
(``bands.refs``) and of A2.

Reads, per run directory (its name = the structure name, as examples/07's collect.py; a
vasp_bands.py layout ``<name>/scf/`` works too):

* ``vasprun.xml`` (preferred): k-points, weights, eigenvalues, occupations, Fermi level,
  structure, and the POTCAR valences (ZVAL) for the semicore count;
* else ``EIGENVAL`` + ``OUTCAR`` (Fermi level, structure), or ``OUTCAR`` alone (its last
  eigenvalue listing and the weights of its k-point list; ISPIN 1), with ZVAL from ``POTCAR`` or
  ``--zval``.

``n_semicore`` = (sum of ZVAL - DFTB valence) / 2 per structure (Li_sv: one 1s band per Li),
with the DFTB valence of PTBP's basis (Li 1, O 6; ``--dftb-valence``).  The structure of every
run is checked against the data (same atoms, positions within 1e-3 A).

    python d0_ibz_bands.py --runs ~/a1_d0 --config scan_li_o.yaml --out bands_ibz --dry-run
    python d0_ibz_bands.py --runs ~/a1_d0 --config scan_li_o.yaml --out bands_ibz

Getting the files (a download only, no job; ask the user first, A0 section 2 lists D0 as
"只下载、不算，先问用户").  On the workstation, with the remote run directories of M5c/M5d
(the jobs behind examples/07_battery_li_li2o/data, e.g. 981219 and 981265 for short_range.xyz)::

    rsync -avP --prune-empty-dirs --include='*/' --include='vasprun.xml' --include='OUTCAR' \\
        --include='EIGENVAL' --include='CONTCAR' --exclude='*' dghpc:<remote runs dir>/ ~/a1_d0/

Writes ``<out>/<name>.json`` and ``<out>/summary.json`` (matched, unmatched, per structure:
source, k-points, bands, semicore, gap or metal); logs in ``<out>/logs/d0_ibz_bands/``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import setup  # noqa: E402

setup()

DFTB_VALENCE = {"Li": 1.0, "O": 6.0}
DEFAULT_ZVAL = {"Li": 3.0, "O": 6.0}          # Li_sv, O (potpaw_PBE.54, the M5c runs)


def _pairs(items):
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k] = float(v)
    return out


def _run_dirs(roots):
    """{name: directory with the VASP output} under the roots (name = the directory's name, or
    its parent's for a ``scf`` subdirectory)."""
    out = {}
    for root in roots:
        root = os.path.abspath(os.path.expanduser(root))
        for f in sorted(glob.glob(os.path.join(root, "**", "OUTCAR"), recursive=True)
                        + glob.glob(os.path.join(root, "**", "vasprun.xml"), recursive=True)):
            d = os.path.dirname(f)
            name = os.path.basename(d)
            if name in ("scf", "band"):
                if name == "band":
                    continue
                name = os.path.basename(os.path.dirname(d))
            out.setdefault(name, d)
    return out


def _finished(d):
    p = os.path.join(d, "OUTCAR")
    if not os.path.exists(p):
        return os.path.exists(os.path.join(d, "vasprun.xml"))
    with open(p, errors="ignore") as fh:
        return "General timing" in fh.read()


def _same_structure(a, b, tol=1e-3):
    if len(a) != len(b) or list(a.numbers) != list(b.numbers):
        return False, "different atoms"
    if np.any(a.pbc):
        if not np.allclose(a.cell.array, b.cell.array, atol=tol):
            return False, "different cell"
        ds = a.get_scaled_positions(wrap=False) - b.get_scaled_positions(wrap=False)
        ds -= np.round(ds)
        dist = np.linalg.norm(ds @ a.cell.array, axis=1)
    else:
        dist = np.linalg.norm(a.positions - b.positions, axis=1)
    worst = float(dist.max()) if len(dist) else 0.0
    return worst <= tol, f"max displacement {worst:.2e} A"


def from_outcar(path, name, n_semicore=0, meta=None):
    """A BandRef (kind "ibz") from an OUTCAR alone: the last eigenvalue listing ("k-point n :"
    blocks followed by "band No."), the weights of "Following reciprocal coordinates", the last
    E-fermi and the last structure.  ISPIN 1 only (use vasprun.xml otherwise)."""
    import ase.io
    from pydftb.bandref import BandRef, outcar_fermi
    with open(path, errors="ignore") as fh:
        lines = fh.read().splitlines()
    head = [k for k, ln in enumerate(lines) if "Following reciprocal coordinates:" in ln]
    if not head:
        raise ValueError("no 'Following reciprocal coordinates' block")
    rows, j = [], head[0] + 2
    while j < len(lines) and len(lines[j].split()) == 4:
        rows.append([float(x) for x in lines[j].split()])
        j += 1
    rows = np.array(rows)
    kpts, w = rows[:, :3], rows[:, 3]
    starts = [k for k in range(len(lines) - 1)
              if lines[k].strip().startswith("k-point") and "band No." in lines[k + 1]]
    if len(starts) < len(kpts):
        raise ValueError(f"{len(starts)} eigenvalue blocks for {len(kpts)} k-points")
    starts = starts[-len(kpts):]                  # the last ionic step
    if any("spin component 2" in ln for ln in lines[starts[0] - 400:starts[-1]]):
        raise ValueError("spin-polarized OUTCAR: use vasprun.xml")
    e, o = [], []
    for s0 in starts:
        ek, ok = [], []
        k = s0 + 2
        while k < len(lines) and lines[k].split() and lines[k].split()[0].isdigit():
            c = lines[k].split()
            ek.append(float(c[1]))
            ok.append(float(c[2]))
            k += 1
        e.append(ek)
        o.append(ok)
    at = ase.io.read(path, index=-1, format="vasp-out")
    ref = BandRef(name, "ibz", kpts, w, np.array(e)[None], np.array(o)[None], outcar_fermi(path),
                  n_semicore, meta=dict(meta or {}, code="VASP", source=os.path.abspath(path),
                                        reader="OUTCAR"))
    return ref.set_atoms(at)


def convert(name, d, frame, dftb_valence, zval_default, log):
    import ase.io
    from pydftb import bandref as br
    meta = {"group": frame.info.get("family"), "role": frame.info.get("role"),
            "kspacing": frame.info.get("kspacing"), "kpts": frame.info.get("kpts"), "source_dir": d}
    xml = os.path.join(d, "vasprun.xml")
    if os.path.exists(xml):
        try:
            zval = br.zval_from_vasprun(xml)
        except Exception:                       # noqa: BLE001
            zval = dict(zval_default)
        at = ase.io.read(xml, index=-1, format="vasp-xml")
        nsc = br.count_semicore(at, zval, dftb_valence)
        ref = br.from_vasprun(xml, name=name, kind="ibz", n_semicore=nsc, meta=meta)
        src = "vasprun.xml"
    elif not os.path.exists(os.path.join(d, "EIGENVAL")) and os.path.exists(os.path.join(d, "OUTCAR")):
        out = os.path.join(d, "OUTCAR")
        at = ase.io.read(out, index=-1, format="vasp-out")
        pot = os.path.join(d, "POTCAR")
        zval = br.zval_from_potcar(pot) if os.path.exists(pot) else dict(zval_default)
        nsc = br.count_semicore(at, zval, dftb_valence)
        ref = from_outcar(out, name, nsc, meta)
        src = "OUTCAR"
    else:
        eig = os.path.join(d, "EIGENVAL")
        out = os.path.join(d, "OUTCAR")
        if not (os.path.exists(eig) and os.path.exists(out)):
            raise FileNotFoundError("neither vasprun.xml nor EIGENVAL + OUTCAR")
        at = ase.io.read(out, index=-1, format="vasp-out")
        pot = os.path.join(d, "POTCAR")
        zval = br.zval_from_potcar(pot) if os.path.exists(pot) else dict(zval_default)
        nsc = br.count_semicore(at, zval, dftb_valence)
        ref = br.from_eigenval(eig, atoms=at, name=name, kind="ibz", fermi=br.outcar_fermi(out),
                               n_semicore=nsc)
        ref.meta.update(meta)
        src = "EIGENVAL + OUTCAR"
    same, why = _same_structure(ref.atoms(), frame)
    if not same:
        log.event("structure_mismatch", name=name, detail=why)
    return ref, {"source": src, "dir": d, "nk": int(len(ref.kpts)), "nbands": int(ref.nbands),
                 "nspin": int(ref.nspin), "n_semicore": nsc, "fermi": ref.fermi,
                 "same_structure": same, "structure_check": why}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, help="directories with the VASP runs")
    ap.add_argument("--config", default=os.path.join(HERE, "scan_li_o.yaml"))
    ap.add_argument("--out", default=os.path.join(HERE, "bands_ibz"))
    ap.add_argument("--dftb-valence", nargs="*", default=None, help="e.g. Li=1 O=6")
    ap.add_argument("--zval", nargs="*", default=None, help="fallback ZVAL, e.g. Li=3 O=6")
    ap.add_argument("--dry-run", action="store_true", help="list the matches, write nothing")
    ap.add_argument("--force", action="store_true", help="rewrite existing JSON files")
    args = ap.parse_args(argv)
    from utils import accuracy as acc
    from utils.runlog import session
    cfg = acc.load_config(args.config)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    dv = dict(DFTB_VALENCE, **_pairs(args.dftb_valence))
    zv = dict(DEFAULT_ZVAL, **_pairs(args.zval))
    with session("d0_ibz_bands", vars(args), root=os.path.join(out, "logs")) as log:
        frames = {a.info["name"]: a for a in acc.load_data(cfg)}
        runs = _run_dirs(args.runs)
        matched = {n: d for n, d in runs.items() if n in frames}
        unmatched_runs = sorted(n for n in runs if n not in frames)
        no_run = sorted(n for n in frames if n not in runs)
        log.event("matches", runs=len(runs), matched=len(matched), runs_not_in_data=len(unmatched_runs),
                  data_without_run=len(no_run))
        print(f"{len(matched)} of {len(frames)} structures have a run; {len(unmatched_runs)} runs "
              f"are not in the data ({unmatched_runs[:8]}{' ...' if len(unmatched_runs) > 8 else ''})")
        if args.dry_run:
            for n, d in sorted(matched.items()):
                files = [f for f in ("vasprun.xml", "EIGENVAL", "OUTCAR") if os.path.exists(os.path.join(d, f))]
                print(f"  {n:32s} {d}  [{', '.join(files)}]{'' if _finished(d) else '  (not finished)'}")
            print(f"data without a run: {no_run[:20]}{' ...' if len(no_run) > 20 else ''}")
            return 0
        summary = {"config": cfg["_path"], "dftb_valence": dv, "structures": {},
                   "runs_not_in_data": unmatched_runs, "data_without_run": no_run, "failed": {}}
        for n, d in sorted(matched.items()):
            path = os.path.join(out, f"{n}.json")
            if os.path.exists(path) and not args.force:
                summary["structures"][n] = {"skipped": "exists"}
                continue
            if not _finished(d):
                summary["failed"][n] = "run not finished"
                continue
            try:
                with log.step("convert", name=n):
                    ref, info = convert(n, d, frames[n], dv, zv, log)
                    ref.to_json(path)
                    try:
                        info["edges"] = ref.edges()
                    except Exception as exc:    # noqa: BLE001
                        info["edges"] = f"{type(exc).__name__}: {exc}"
                summary["structures"][n] = info
            except Exception as exc:            # noqa: BLE001  (recorded, the others go on)
                summary["failed"][n] = f"{type(exc).__name__}: {exc}"
                log.event("convert_failed", name=n, error=summary["failed"][n])
        from common.models import _clean
        with open(os.path.join(out, "summary.json"), "w") as fh:
            json.dump(_clean(summary), fh, indent=1, default=str)
        print(f"{len([s for s in summary['structures'].values() if 'nk' in s])} written, "
              f"{len(summary['failed'])} failed -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
