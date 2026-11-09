"""Collect D1 band references into BandRef JSON (pydftb.bandref), the input of ``ptbp scan``
(config key ``bands.refs``).

* **VASP** (``--vasp ROOT``, the output of vasp_bands.py after the runs):
  * a crystal is read from ``<name>/band/vasprun.xml`` with the labels, levels and segments of
    its ``kpath.json``; the k-point order is checked against it;
  * the Fermi level and E(sigma->0) come from ``<name>/scf/`` (OUTCAR).  A non-SCF run's own
    Fermi level is set by the path's k-points, so it is not used;
  * occupations are recomputed from that Fermi level with the Gaussian of the runs (0.05 eV);
  * a dimer is read from ``<name>/scf/vasprun.xml`` (Gamma, kind "molecule");
  * ``n_semicore`` = (sum of ZVAL - DFTB valence) / 2, with ZVAL from vasprun.xml (Li_sv 3)
    and DFTB valence Li 1, O 6 (PTBP).
* **GPAW** (``--gpaw DIR``, bands_gpaw.py's ``bands/``): the JSON files are checked (they load,
  the semicore count is consistent, the edges) and copied.

``--out`` gets ``<name>.json`` per structure and ``summary.json``: code, k-points, bands,
semicore, gap or metal, VBM/CBM.  A structure whose run did not finish is listed and skipped.

    python collect_bands.py --vasp ../runs/d1_vasp --out ../data/bands_vasp
    python collect_bands.py --gpaw ../runs/d1_li_o/bands --out ../data/bands_gpaw
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import setup  # noqa: E402

setup()

DFTB_VALENCE = {"Li": 1.0, "O": 6.0}
SIGMA = 0.05


def finished(run_dir: str) -> bool:
    out = os.path.join(run_dir, "OUTCAR")
    return os.path.exists(out) and "General timing" in open(out, errors="ignore").read()


def sigma0_energy(outcar: str) -> float | None:
    vals = re.findall(r"energy\(sigma->0\) =\s+(-?\d+\.\d+)", open(outcar, errors="ignore").read())
    return float(vals[-1]) if vals else None


def gaussian_occupations(eps, fermi, sigma=SIGMA):
    from scipy.special import erfc
    return 0.5 * erfc((np.asarray(eps) - fermi) / sigma)


def from_vasp_dir(base: str, dftb_valence: dict):
    """BandRef of one vasp_bands.py structure directory, or (None, reason)."""
    from pydftb.bandref import count_semicore, from_vasprun, outcar_fermi, zval_from_vasprun
    name = os.path.basename(base.rstrip("/"))
    scf = os.path.join(base, "scf")
    band = os.path.join(base, "band")
    meta_path = os.path.join(band if os.path.isdir(band) else scf, "kpath.json")
    info = json.load(open(meta_path)) if os.path.exists(meta_path) else {"kind": "molecule"}
    if not finished(scf):
        return None, "scf not finished"
    fermi = outcar_fermi(os.path.join(scf, "OUTCAR"))
    e0 = sigma0_energy(os.path.join(scf, "OUTCAR"))
    if info.get("kind") == "path":
        if not finished(band):
            return None, "band run not finished"
        xml = os.path.join(band, "vasprun.xml")
        ref = from_vasprun(xml, name=name, kind="path", labels=info["labels"],
                           level=info["level"], segments=info["segments"])
        if ref.kpts.shape != np.asarray(info["kpts"]).shape or not np.allclose(
                ref.kpts, np.asarray(info["kpts"]), atol=1e-6):
            return None, "band k-points differ from kpath.json (order or symmetry reduction)"
    else:
        xml = os.path.join(scf, "vasprun.xml")
        ref = from_vasprun(xml, name=name, kind="molecule")
    atoms = ref.atoms()
    ref.n_semicore = count_semicore(atoms, zval_from_vasprun(xml), dftb_valence)
    ref.fermi = fermi
    ref.occupations = gaussian_occupations(ref.energies, fermi)
    ref.meta.update(code="VASP", group=info.get("group", name), source=info.get("source"),
                    scaling=info.get("scaling"), kspacing=info.get("kspacing", 0.15),
                    energy_sigma0_eV=e0, smearing=f"gaussian {SIGMA} eV (ISMEAR 0)",
                    run_dir=os.path.abspath(base))
    return ref, None


def describe(ref, dftb_valence: dict) -> dict:
    nval = sum(dftb_valence[s] for s in ref.atoms().get_chemical_symbols())
    nv = int(np.ceil(nval / 2 - 1e-6))
    try:
        ed = ref.edges(nv)
    except Exception as exc:                     # noqa: BLE001
        ed = {"error": str(exc)}
    return {"name": ref.name, "code": ref.meta.get("code"), "kind": ref.kind,
            "group": ref.meta.get("group"), "n_k": len(ref.kpts), "n_bands": ref.energies.shape[2],
            "n_semicore": ref.n_semicore, "n_valence_dftb": nv, **ed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vasp", default=None, help="vasp_bands.py output root (after the runs)")
    ap.add_argument("--gpaw", default=None, help="bands_gpaw.py bands/ directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dftb-valence", default=None,
                    help='JSON, default {"Li": 1, "O": 6} (PTBP valence_shell)')
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)
    if not (args.vasp or args.gpaw):
        ap.error("give --vasp and/or --gpaw")
    from pydftb.bandref import BandRef
    from utils.runlog import session
    dval = dict(DFTB_VALENCE)
    if args.dftb_valence:
        dval.update({k: float(v) for k, v in json.loads(args.dftb_valence).items()})
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    with session("d1_collect_bands", vars(args), root=os.path.join(out, "logs")) as log:
        summary, skipped = [], []
        if args.vasp:
            for base in sorted(glob.glob(os.path.join(args.vasp, "*/"))):
                if not os.path.isdir(os.path.join(base, "scf")):
                    continue
                ref, why = from_vasp_dir(base, dval)
                if ref is None:
                    skipped.append({"dir": base, "reason": why})
                    print(f"[collect] skip {base}: {why}")
                    continue
                if not args.dry_run:
                    ref.to_json(os.path.join(out, f"{ref.name}.json"))
                summary.append(describe(ref, dval))
        if args.gpaw:
            for path in sorted(glob.glob(os.path.join(args.gpaw, "*.json"))):
                if os.path.basename(path) in ("energies.json", "summary.json"):
                    continue
                try:
                    ref = BandRef.from_json(path)
                except Exception as exc:          # noqa: BLE001
                    skipped.append({"file": path, "reason": f"{type(exc).__name__}: {exc}"})
                    continue
                d = describe(ref, dval)
                summary.append(d)
                if not args.dry_run and os.path.abspath(os.path.dirname(path)) != out:
                    shutil.copy2(path, os.path.join(out, os.path.basename(path)))
        for d in summary:
            gap = "metal" if d.get("metal") else f"gap {d.get('gap', float('nan')):.3f} eV"
            print(f"  {d['name']:22s} {d['code']:5s} {d['kind']:8s} k {d['n_k']:4d}  "
                  f"bands {d['n_bands']:3d}  semicore {d['n_semicore']}  {gap}")
        if not args.dry_run:
            with open(os.path.join(out, "summary.json"), "w") as fh:
                json.dump({"structures": summary, "skipped": skipped}, fh, indent=1, default=float)
        log.event("collected", n=len(summary), skipped=len(skipped))
        print(f"[collect] {len(summary)} references, {len(skipped)} skipped -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
