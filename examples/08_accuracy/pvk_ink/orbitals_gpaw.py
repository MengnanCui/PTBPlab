"""D1, ink part: GPAW PBE orbital energies of the ink structures, as BandRef references.

Part of the A series, A0 section 2.

    python orbitals_gpaw.py --dry-run                      # list what would run, no GPAW import
    mpirun -np 8 gpaw python orbitals_gpaw.py [--only 'ion_*' 'cplx_*DMSO1*'] [--ecut 600]

For every structure of ``data/structures.xyz`` (prepare_structures.py) the script runs:

* **Method:** a PBE calculation in GPAW's plane-wave mode (``--ecut``, default 600 eV), at the Γ
  point, closed shell (spin-unpolarized).  The box is periodic for GPAW's PW mode:
  orthorhombic with ``--vacuum`` (6 Å) per side (``--box ortho``, cheaper than the VASP box), or
  the cubic 8 Å box of structures.xyz (``--box keep``).
* **Smearing:** Methfessel–Paxton of order 0 (= Gaussian) with width ``--width`` (0.01 eV),
  or ``--occupations fd`` for Fermi–Dirac.
* **Bands:** the occupied bands plus ``--nbands-extra`` (20) empty ones (GPAW's ``nbands=-20``).
  The states up to ``--converge-bands`` (GPAW's ``"CBM+2.0"``: 2 eV above the LUMO) are
  converged, so LUMO + 1 eV is covered.

Outputs (``--out``, default ``bands/``):

* ``bands/<name>.json``: a :class:`pydftb.bandref.BandRef` of kind "molecule", with the Γ
  eigenvalues, occupations, Fermi level and the structure (non-periodic, as the DFTB side
  treats it).  ``n_semicore`` = (GPAW valence electrons − DFTB valence electrons) / 2, the
  DFTB valence being the occupations of PTBP's valence shells.  meta holds code, version,
  functional, settings, group (the family) and kspacing "gamma";
* ``energies/<name>.json`` and ``energies.json``: GPAW total energies.  They are for the GPAW
  data only and are never mixed with the VASP energies of D4 in one fit;
* ``gpaw_txt/<name>.txt``: GPAW's log of each structure.

A structure whose JSON exists is skipped (``--force`` recomputes).  Under MPI every rank takes
part in each calculation and rank 0 writes the files.  Cost: seconds to minutes per small
molecule on a laptop, about an hour for the 39-atom complexes; the scan points are
cheaper than complexes.
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import setup  # noqa: E402

setup()


def dftb_valence(element: str) -> float:
    """Valence electrons of PTBP's neutral atom (the occupations of its valence shells)."""
    from utils.parameters_set import get_PTBP
    shells = get_PTBP()[element]["valence_shell"].split()
    return float(sum(int(re.findall(r"\d+$", s)[0]) for s in shells))


def selected(frames, only, max_atoms):
    out = []
    for a in frames:
        name, fam = a.info["name"], a.info.get("family", "")
        if only and not any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(fam, p) for p in only):
            continue
        if max_atoms and len(a) > max_atoms:
            continue
        out.append(a)
    return out


def occupations_spec(kind, width):
    if kind == "mp":
        return {"name": "methfessel-paxton", "width": width, "order": 0}
    if kind == "fd":
        return {"name": "fermi-dirac", "width": width}
    raise ValueError(f"occupations must be mp or fd, got {kind!r}")


def run_one(atoms, args, txt):
    """GPAW for one structure -> (BandRef, energy eV, seconds)."""
    import gpaw
    from gpaw import GPAW, PW
    from pydftb.bandref import BandRef
    from vasp_ink import boxed
    # GPAW only: a tighter orthorhombic box (the eigenvalues get a free shift in the band loss,
    # so the vacuum level does not matter; the volume sets the plane-wave cost)
    a = boxed(atoms, args.vacuum, cubic=False) if args.box == "ortho" else atoms.copy()
    a.pbc = True                                   # GPAW's PW mode is periodic
    a.calc = GPAW(mode=PW(args.ecut), xc="PBE", kpts=(1, 1, 1), spinpol=False, charge=0,
                  nbands=-int(args.nbands_extra), symmetry="off",
                  occupations=occupations_spec(args.occupations, args.width),
                  convergence={"energy": 1e-6, "density": 1e-6, "eigenstates": 1e-9,
                               "bands": args.converge_bands},
                  maxiter=args.maxiter, txt=txt)
    t0 = time.perf_counter()
    energy = float(a.get_potential_energy())
    calc = a.calc
    eps = np.asarray(calc.get_eigenvalues(kpt=0, spin=0), float)
    occ = np.asarray(calc.get_occupation_numbers(kpt=0, spin=0), float)
    fermi = float(calc.get_fermi_level())
    n_gpaw = float(calc.get_number_of_electrons())
    n_dftb = sum(dftb_valence(s) for s in a.get_chemical_symbols())
    extra = n_gpaw - n_dftb
    if extra < -1e-6 or abs(extra / 2 - round(extra / 2)) > 1e-6:
        raise ValueError(f"{atoms.info['name']}: GPAW valence {n_gpaw} - DFTB valence {n_dftb} "
                         "is not a non-negative even number")
    meta = {"code": "GPAW", "version": getattr(gpaw, "__version__", "?"), "functional": "PBE",
            "settings": {"mode": "pw", "ecut_eV": args.ecut, "kpts": "gamma",
                         "occupations": occupations_spec(args.occupations, args.width),
                         "nbands_extra": args.nbands_extra,
                         "converge_bands": args.converge_bands, "box_A": a.cell.lengths().tolist()},
            "group": atoms.info.get("family"), "role": atoms.info.get("role"),
            "kspacing": "gamma", "energy_eV": energy, "n_valence_gpaw": n_gpaw,
            "n_valence_dftb": n_dftb, "source": atoms.info.get("source")}
    ref = BandRef(name=atoms.info["name"], kind="molecule", kpts=[[0.0, 0.0, 0.0]],
                  kweights=[1.0], energies=eps[None, None, :], occupations=occ[None, None, :],
                  fermi=fermi, n_semicore=int(round(extra / 2)), meta=meta)
    mol = atoms.copy()
    mol.pbc = False                                # the DFTB side: a molecule in its box
    ref.set_atoms(mol)
    return ref, energy, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--structures", default=os.path.join(HERE, "data", "structures.xyz"))
    ap.add_argument("--out", default=os.path.join(HERE, "bands"))
    ap.add_argument("--ecut", type=float, default=600.0, help="plane-wave cut-off (eV)")
    ap.add_argument("--occupations", default="mp", choices=["mp", "fd"])
    ap.add_argument("--width", type=float, default=0.01, help="smearing width (eV)")
    ap.add_argument("--nbands-extra", type=int, default=20)
    ap.add_argument("--converge-bands", default="CBM+2.0",
                    help="GPAW convergence['bands']: 'occupied', 'all', an integer or 'CBM+x' (eV)")
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--box", default="ortho", choices=["ortho", "keep"],
                    help="ortho: re-box with --vacuum per axis (cheaper); keep: the cubic box of "
                         "structures.xyz (the VASP box)")
    ap.add_argument("--vacuum", type=float, default=6.0, help="A, with --box ortho")
    ap.add_argument("--only", nargs="*", default=None, help="name or family patterns")
    ap.add_argument("--max-atoms", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list the structures, import no GPAW")
    args = ap.parse_args(argv)
    import ase.io
    frames = selected(ase.io.read(args.structures, ":"), args.only, args.max_atoms)
    if args.dry_run:
        print(f"{len(frames)} structures (dry run, nothing computed):")
        for a in frames:
            done = os.path.exists(os.path.join(args.out, f"{a.info['name']}.json"))
            print(f"  {a.info['name']:28s} {len(a):3d} atoms  box {a.cell[0, 0]:5.1f} A  "
                  f"{a.info.get('role', '?'):5s} {'done' if done else ''}")
        n = np.array([len(a) for a in frames])
        print(f"atoms: total {n.sum()}, max {n.max() if len(n) else 0}; "
              f"rough cost ~{(0.05 * (n / 10.0) ** 2).sum():.1f} h on 8 cores (PW 600 eV)")
        return 0
    try:
        from gpaw.mpi import world
        rank = world.rank
    except ImportError:
        print("GPAW is not importable in this Python (use the 'gpaw' environment)")
        return 2
    os.makedirs(os.path.join(args.out, "energies"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "gpaw_txt"), exist_ok=True)
    from utils.runlog import session
    ctx = session("orbitals_gpaw", vars(args), root=os.path.join(args.out, "logs")) \
        if rank == 0 else contextlib.nullcontext()
    with ctx as log:
        done, failed = 0, []
        for a in frames:
            name = a.info["name"]
            path = os.path.join(args.out, f"{name}.json")
            if os.path.exists(path) and not args.force:
                continue
            txt = os.path.join(args.out, "gpaw_txt", f"{name}.txt")
            try:
                ref, energy, sec = run_one(a, args, txt)
            except Exception as exc:              # noqa: BLE001  (recorded, next structure)
                failed.append(name)
                if rank == 0:
                    log.event("failed", name=name, error=f"{type(exc).__name__}: {exc}")
                continue
            if rank == 0:
                ref.to_json(path)
                with open(os.path.join(args.out, "energies", f"{name}.json"), "w") as fh:
                    json.dump({"name": name, "energy_eV": energy, "natoms": len(a),
                               "settings": ref.meta["settings"]}, fh, indent=1)
                edges = ref.edges(n_valence_bands=int(round(ref.meta["n_valence_dftb"] / 2))) \
                    if ref.meta["n_valence_dftb"] % 2 == 0 else ref.edges()
                log.event("done", name=name, seconds=round(sec, 1), energy=energy,
                          gap=edges["gap"], n_semicore=ref.n_semicore)
            done += 1
        if rank == 0:
            allen = {}
            for fn in sorted(os.listdir(os.path.join(args.out, "energies"))):
                with open(os.path.join(args.out, "energies", fn)) as fh:
                    d = json.load(fh)
                allen[d["name"]] = d["energy_eV"]
            with open(os.path.join(args.out, "energies.json"), "w") as fh:
                json.dump(allen, fh, indent=1)
            log.event("summary", computed=done, failed=len(failed), total=len(allen))
            if failed:
                print("failed:", ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
