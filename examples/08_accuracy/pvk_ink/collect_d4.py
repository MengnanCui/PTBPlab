"""Collect the finished D4 VASP runs into the data of scan_ink.yaml (A series, A1.3).

    python collect_d4.py [D4_DIR] [--out data] [--relax-steps 5] [--d3] [--bands]

A case counts when its OUTCAR contains "General timing".  The energy is VASP's
``energy(sigma->0)``, the forces are in eV/Å, and the tags come from the case's ``case.json``,
which make_d4.py writes.  The atoms are in VASP's (element-sorted) order.  Molecules and
clusters get pbc False back, with their box as the cell.  The outputs are:

* ``<name>_{train,test,ood}.xyz`` (``--name``, default structures_dft): one file per role,
  because utils.accuracy sets ``info['role']`` from the config list a file is in.
  ``<name>.xyz`` has all of them;
* ``sp/<name>`` gives the frame ``<name>``.  ``relax/<name>`` gives its last ionic step as
  ``<name>_relaxed`` and, with ``--relax-steps K``, every K-th step as ``<name>_relax<NNN>``,
  all in the family and role of the original;
* ``E0s_dft.json``: the isolated atoms (``atoms/atom_<El>``), keyed by atomic number as in
  examples/07;
* ``<name>_collect_report.json``: finished, unfinished and failed cases.

D5 (make_d5.py) is collected the same way, without isolated atoms:
``python collect_d4.py d5 --subdirs vasp --name clusters_dft``.

``--d3`` stores PBE-D3(BJ) energies and forces from simple-dftd3 (``dftd3.ase``) in
``info['d3_energy']`` and ``arrays['d3_forces']``.  They are not used by the fits, which compare
plain PBE with DFTB, but are kept for checks against PBE+D3.  ``--subtract-d3`` is for runs
that did include D3 (IVDW = 12): it removes the D3 part, so that the stored energy is plain PBE
again.

``--bands`` writes VASP BandRefs (kind "molecule") from each single point's ``vasprun.xml`` to
``bands_vasp/``.  These are orbital references from the same calculations as the energies.
n_semicore is the POTCAR ZVAL minus PTBP's DFTB valence, over two.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import setup  # noqa: E402

setup()


def finished(d):
    p = os.path.join(d, "OUTCAR")
    if not os.path.exists(p):
        return False
    with open(p, errors="ignore") as fh:
        return "General timing" in fh.read()


def sigma0(outcar):
    """energy(sigma->0) of every ionic step: the value in each "FREE ENERGIE OF THE
    ION-ELECTRON SYSTEM" block (VASP also prints it at electronic steps, which are skipped)."""
    with open(outcar, errors="ignore") as fh:
        text = fh.read()
    out = []
    for m in re.finditer(r"FREE ENERGIE OF THE ION-ELECTRON SYSTEM \(eV\)", text):
        e = re.search(r"energy\(sigma->0\)\s*=\s*(-?\d+\.\d+)", text[m.end():m.end() + 800])
        if e:
            out.append(float(e.group(1)))
    return out


def frames_of(d, steps: int | None):
    """(ionic frames with energy(sigma->0) and forces, case.json) of a finished case;
    ``steps``: None = the last frame only, K = every K-th frame and the last."""
    import ase.io
    from ase.calculators.singlepoint import SinglePointCalculator
    outcar = os.path.join(d, "OUTCAR")
    with open(os.path.join(d, "case.json")) as fh:
        case = json.load(fh)
    imgs = ase.io.read(outcar, index=":", format="vasp-out")
    e0 = sigma0(outcar)
    if len(e0) != len(imgs):
        raise RuntimeError(f"{d}: {len(imgs)} ionic steps but {len(e0)} energy(sigma->0) values")
    keep = [len(imgs) - 1] if steps is None else sorted(set(range(0, len(imgs), steps))
                                                        | {len(imgs) - 1})
    out = []
    for k in keep:
        at = imgs[k]
        f = at.get_forces()
        at.calc = SinglePointCalculator(at, energy=e0[k], free_energy=e0[k], forces=f)
        if not case.get("periodic", False):
            at.pbc = False
        out.append((k, len(imgs), at))
    return out, case


def dftb_valence(el):
    from utils.parameters_set import get_PTBP
    return float(sum(int(re.findall(r"\d+$", s)[0])
                     for s in get_PTBP()[el]["valence_shell"].split()))


def d3_terms(atoms):
    from dftd3.ase import DFTD3
    a = atoms.copy()
    a.calc = DFTD3(method="PBE", damping="d3bj")
    return float(a.get_potential_energy()), np.array(a.get_forces())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("d4", nargs="?", default=os.path.join(HERE, "d4"))
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--relax-steps", type=int, default=5,
                    help="every K-th ionic step of the relaxations (0: only the last)")
    ap.add_argument("--d3", action="store_true", help="store PBE-D3(BJ) energy/forces (dftd3)")
    ap.add_argument("--subtract-d3", action="store_true",
                    help="the runs included D3 (IVDW): subtract it to get plain PBE")
    ap.add_argument("--bands", action="store_true", help="VASP BandRefs from vasprun.xml")
    ap.add_argument("--bands-out", default=os.path.join(HERE, "bands_vasp"))
    ap.add_argument("--subdirs", nargs="+", default=["sp", "relax"],
                    help="case groups to read (D5: --subdirs vasp)")
    ap.add_argument("--name", default="structures_dft",
                    help="output stem: <name>_{train,test,ood}.xyz (D5: clusters_dft)")
    args = ap.parse_args(argv)
    import ase.io
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.data import atomic_numbers
    from utils.runlog import session
    os.makedirs(args.out, exist_ok=True)
    with session("collect_d4", vars(args), root=os.path.join(args.out, "logs")) as log:
        report = {"finished": [], "unfinished": [], "failed": {}}
        frames, e0s = [], {}
        for d in sorted(glob.glob(os.path.join(args.d4, "atoms", "atom_*"))):
            if not finished(d):
                report["unfinished"].append(os.path.relpath(d, args.d4))
                continue
            imgs, _ = frames_of(d, None)
            at = imgs[-1][2]
            e0s[str(atomic_numbers[at.get_chemical_symbols()[0]])] = float(at.get_potential_energy())
            report["finished"].append(os.path.relpath(d, args.d4))
        for kind in args.subdirs:
            for d in sorted(glob.glob(os.path.join(args.d4, kind, "*"))):
                rel = os.path.relpath(d, args.d4)
                if not finished(d):
                    report["unfinished"].append(rel)
                    continue
                try:
                    steps = (args.relax_steps or None) if kind == "relax" else None
                    imgs, case = frames_of(d, steps)
                except Exception as exc:          # noqa: BLE001
                    report["failed"][rel] = f"{type(exc).__name__}: {exc}"
                    continue
                for k, n, at in imgs:
                    name = case["name"]
                    if kind == "relax":
                        name = f"{name}_relaxed" if k == n - 1 else f"{name}_relax{k:03d}"
                    at.info = {"name": name, "family": case["family"], "role": case["role"],
                               "kspacing": "gamma", "charge": 0, "source": f"D4 {rel} step {k}",
                               "dft": "VASP PBE ENCUT 520 Gamma ISPIN 1 SIGMA 0.01 (no IVDW)"}
                    if args.d3 or args.subtract_d3:
                        try:
                            ed3, fd3 = d3_terms(at)
                        except Exception as exc:  # noqa: BLE001
                            report["failed"][f"{rel} d3"] = f"{type(exc).__name__}: {exc}"
                            ed3, fd3 = None, None
                        if ed3 is not None:
                            e, f = at.get_potential_energy(), at.get_forces()
                            if args.subtract_d3:
                                e, f = e - ed3, f - fd3
                                at.info["dft"] = at.info["dft"].replace("(no IVDW)", "(D3 removed)")
                            at.calc = SinglePointCalculator(at, energy=e, free_energy=e, forces=f)
                            at.info["d3_energy"] = ed3
                            at.arrays["d3_forces"] = fd3
                    frames.append(at)
                report["finished"].append(rel)
                if args.bands and kind != "relax" and os.path.exists(os.path.join(d, "vasprun.xml")):
                    try:
                        from pydftb.bandref import count_semicore, from_vasprun, zval_from_vasprun
                        xml = os.path.join(d, "vasprun.xml")
                        zv = zval_from_vasprun(xml)
                        ref = from_vasprun(xml, name=case["name"], kind="molecule",
                                           meta={"group": case["family"], "role": case["role"],
                                                 "kspacing": "gamma", "functional": "PBE",
                                                 "settings": case.get("incar", {})})
                        at0 = ref.atoms()
                        ref.n_semicore = count_semicore(at0, zv, {s: dftb_valence(s) for s in
                                                                  set(at0.get_chemical_symbols())})
                        mol = at0.copy()
                        if not case.get("periodic", False):
                            mol.pbc = False
                        ref.set_atoms(mol)
                        ref.to_json(os.path.join(args.bands_out, f"{case['name']}.json"))
                    except Exception as exc:      # noqa: BLE001
                        report["failed"][f"{rel} bands"] = f"{type(exc).__name__}: {exc}"
        names = [a.info["name"] for a in frames]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise RuntimeError(f"duplicate names: {dup[:10]}")
        for role in ("train", "test", "ood"):
            sub = [a for a in frames if a.info["role"] == role]
            ase.io.write(os.path.join(args.out, f"{args.name}_{role}.xyz"), sub, format="extxyz")
            log.event("role", role=role, structures=len(sub))
        ase.io.write(os.path.join(args.out, f"{args.name}.xyz"), frames, format="extxyz")
        if e0s:                                    # D5 has no isolated atoms: keep D4's file
            with open(os.path.join(args.out, "E0s_dft.json"), "w") as fh:
                json.dump(e0s, fh, indent=2)
        log.save_json("collect_report.json", report)
        with open(os.path.join(args.out, f"{args.name}_collect_report.json"), "w") as fh:
            json.dump(report, fh, indent=1)
        log.event("collected", frames=len(frames), atoms=len(e0s),
                  unfinished=len(report["unfinished"]), failed=len(report["failed"]))
        print(f"{len(frames)} frames, E0s {sorted(e0s)}, {len(report['unfinished'])} unfinished, "
              f"{len(report['failed'])} failed -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
