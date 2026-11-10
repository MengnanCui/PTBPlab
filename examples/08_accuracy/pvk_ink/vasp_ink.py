"""VASP input writer for the Pb–Sn perovskite ink (A series, D4 and D5), not submitted.

The conventions come from the S1 plan of the perovskite ink project.

* **Functional and PAW:** PBE with Pb_d, Sn_d, I, Cs_sv, S, O, N, C, H.
* **Basis and convergence:** ENCUT 520 eV, PREC Accurate, EDIFF 1e-6, the Γ point only
  (vasp_gam), ISPIN 1 (closed shell), ISMEAR 0 with SIGMA 0.01.
* **No IVDW.** The fits compare plain PBE with DFTB; the same D3(BJ) is added to both sides at
  run time.
* **Molecules and clusters** sit at the centre of a cubic box with at least 8 Å of vacuum.  They
  get the dipole correction (LDIPOL, IDIPOL = 4, DIPOL = centre of mass), so the DFTB side
  treats them as non-periodic.
* **Periodic M409 frames** have no dipole correction.

``write_case`` writes these files:

* POSCAR, with the elements grouped in a fixed order (``ELEMENT_ORDER``);
* INCAR;
* KPOINTS (Γ only) and an empty ``USE_GAM``;
* ``POTCARS``, the POTCAR names in POSCAR order, from which the cluster builds POTCAR (as
  examples/07_battery_li_li2o/run_all.sh does);
* ``case.json``, the tags of the structure.

:func:`write_jobs` splits the cases into Slurm jobs (dg83) under a wall-time limit.  It also
writes ``run_cases.sh`` (the runner, with the paths settable by environment), ``submit_all.sh``
and ``cost_estimate.json`` at 0.1 yuan per core-hour.  Nothing is submitted.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

ELEMENT_ORDER = ["Pb", "Sn", "Cs", "I", "S", "O", "N", "C", "H"]
POTCARS = {"Pb": "Pb_d", "Sn": "Sn_d", "Cs": "Cs_sv", "I": "I", "S": "S", "O": "O", "N": "N",
           "C": "C", "H": "H"}
ZVAL = {"Pb_d": 14, "Sn_d": 14, "Cs_sv": 9, "I": 7, "S": 6, "O": 6, "N": 5, "C": 4, "H": 1}
BASE = {"PREC": "Accurate", "ENCUT": 520, "EDIFF": 1e-6, "LREAL": ".FALSE.", "ISMEAR": 0,
        "SIGMA": 0.01, "ISPIN": 1, "NELM": 200, "ALGO": "Normal", "IBRION": -1, "NSW": 0,
        "ISYM": 0, "LWAVE": ".FALSE.", "LCHARG": ".FALSE."}
RELAX = {"IBRION": 2, "NSW": 100, "EDIFFG": -0.02, "ISIF": 2, "POTIM": 0.3}
PRICE = 0.1                      # yuan per core-hour on dg83 (list price)


def n_electrons(atoms) -> int:
    """Valence electrons of the POTCARs (VASP's NELECT for a neutral structure)."""
    return int(sum(ZVAL[POTCARS[s]] for s in atoms.get_chemical_symbols()))


def core_hours(natoms: int, periodic: bool = False) -> float:
    """Rough cost of one SCF: about 3 core-h for a 40-atom molecule in its box, growing with N^2;
    a periodic 409-atom M409 frame about 150 core-h.  To be calibrated with one or two test runs."""
    if periodic:
        return 150.0 * (natoms / 409.0) ** 2
    return max(0.2, 3.0 * (natoms / 40.0) ** 2)


def write_case(directory: str, atoms, dipole: bool | None = None, nbands_extra: int = 24,
               relax: bool = False, tags: dict | None = None, hours: float | None = None,
               **overrides) -> dict:
    """Write one VASP case (see the module docstring); returns its case.json content.

    ``dipole`` defaults to True for non-periodic structures.  NBANDS is half of NELECT plus
    ``nbands_extra``, so that the empty states up to LUMO + 1 eV are there for the orbital
    references (VASP rounds NBANDS up to its parallel layout).  ``hours`` overrides the
    core-hour estimate of :func:`core_hours`."""
    import ase.io
    os.makedirs(directory, exist_ok=True)
    syms = atoms.get_chemical_symbols()
    unknown = sorted(set(syms) - set(ELEMENT_ORDER))
    if unknown:
        raise ValueError(f"{directory}: no POTCAR choice for {unknown}")
    order = [el for el in ELEMENT_ORDER if el in syms]
    idx = [i for el in order for i, s in enumerate(syms) if s == el]
    a = atoms[idx]
    periodic = bool(np.all(atoms.pbc))
    if dipole is None:
        dipole = not periodic
    a.pbc = True                                     # VASP is periodic; POSCAR needs the cell
    ase.io.write(os.path.join(directory, "POSCAR"), a, format="vasp", direct=True, sort=False,
                 vasp5=True)
    incar = dict(BASE)
    if relax:
        incar.update(RELAX)
    nel = n_electrons(a)
    incar["NBANDS"] = int(math.ceil(nel / 2)) + int(nbands_extra)
    if dipole:
        com = a.get_center_of_mass(scaled=True) % 1.0
        incar.update(LDIPOL=".TRUE.", IDIPOL=4, DIPOL=" ".join(f"{x:.4f}" for x in com))
    incar.update(overrides)
    with open(os.path.join(directory, "INCAR"), "w") as fh:
        for k, v in incar.items():
            if v is not None:
                fh.write(f"{k} = {v}\n")
    with open(os.path.join(directory, "KPOINTS"), "w") as fh:
        fh.write("Gamma\n0\nGamma\n1 1 1\n0 0 0\n")
    open(os.path.join(directory, "USE_GAM"), "w").close()
    with open(os.path.join(directory, "POTCARS"), "w") as fh:
        fh.write(" ".join(POTCARS[s] for s in order) + "\n")
    if hours is None:
        hours = core_hours(len(a), periodic) * (8.0 if relax else 1.0)   # ~8 SCF-equivalents
    case = {"dir": os.path.basename(directory.rstrip("/")), "natoms": len(a),
            "formula": a.get_chemical_formula(), "nelect": nel, "periodic": periodic,
            "relax": relax, "core_hours_estimate": round(hours, 2), "incar": incar}
    case.update(tags or {})
    with open(os.path.join(directory, "case.json"), "w") as fh:
        json.dump(case, fh, indent=1, default=str)
    return case


RUNNER = r"""#!/bin/bash
# Run VASP in every case directory given (default: every subdirectory with INCAR and POSCAR),
# one after another with $SLURM_NTASKS ranks.  POTCAR is concatenated from the names in POTCARS
# (POSCAR order).  Finished cases (OUTCAR with "General timing") are skipped, so a resubmitted
# job continues where the last one stopped.  Paths (defaults: the dghpc account of the M5c runs):
#   VASP_BIN_DIR   directory with vasp_std / vasp_gam
#   VASP_PP_PATH   potpaw_PBE.54 directory
#   VASP_MODULES   modules to load (space separated; empty: none)
VASP_BIN_DIR=${VASP_BIN_DIR:-/data/home/dg002434/software/vasp.6.6.0/bin}
VASP_PP_PATH=${VASP_PP_PATH:-/data/home/dg002434/software/potpaw_PBE.54}
VASP_MODULES=${VASP_MODULES-intel/2022.1}
if [ -n "$VASP_MODULES" ] && command -v module >/dev/null 2>&1; then
  module purge; for m in $VASP_MODULES; do module load "$m"; done
fi
export PATH=$VASP_BIN_DIR:$PATH
NP=${SLURM_NTASKS:-4}
dirs=("$@"); [ ${#dirs[@]} -eq 0 ] && dirs=(*/)
for d in "${dirs[@]}"; do
  d=${d%/}
  [ -f "$d/INCAR" ] && [ -f "$d/POSCAR" ] || continue
  if grep -qs "General timing" "$d/OUTCAR"; then echo "skip $d"; continue; fi
  ( cd "$d" || exit
    cat $(for p in $(cat POTCARS); do echo "$VASP_PP_PATH/$p/POTCAR"; done) > POTCAR
    exe=vasp_std; [ -f USE_GAM ] && exe=vasp_gam
    t0=$(date +%s)
    mpirun -np $NP $exe > vasp.log 2>&1
    echo "done $d $(( $(date +%s) - t0 )) s $(grep -c 'F=' OSZICAR 2>/dev/null) ionic $(date '+%F %T')"
  )
done
"""


def write_jobs(out_dir: str, cases: list, name: str, ncores: int = 32, max_hours: float = 24.0,
               partition: str = "dg83", safety: float = 1.5) -> dict:
    """Slurm jobs for the case directories (relative to ``out_dir``), each at most
    ``max_hours`` of estimated wall time (times ``safety``); plus run_cases.sh, submit_all.sh
    and cost_estimate.json.  Returns the estimate."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_cases.sh"), "w") as fh:
        fh.write(RUNNER)
    os.chmod(os.path.join(out_dir, "run_cases.sh"), 0o755)
    cases = sorted(cases, key=lambda c: -c["core_hours_estimate"])
    jobs, cur, cur_h = [], [], 0.0
    for c in cases:
        h = c["core_hours_estimate"] * safety / ncores
        if cur and cur_h + h > max_hours:
            jobs.append((cur, cur_h))
            cur, cur_h = [], 0.0
        cur.append(c)
        cur_h += h
    if cur:
        jobs.append((cur, cur_h))
    lines = ["#!/bin/bash", "# Submit every job of this batch (written by vasp_ink.write_jobs; "
             "NOT run automatically).", "# Check cost_estimate.json first.", "set -e",
             'cd "$(dirname "$0")"']
    for k, (members, hours) in enumerate(jobs):
        wall = max(1, int(math.ceil(min(hours * 1.2, max_hours * 1.5) * 60)))
        job = f"job_{k:02d}.sh"
        with open(os.path.join(out_dir, job), "w") as fh:
            fh.write(f"""#!/bin/bash
#SBATCH -J {name}_{k:02d}
#SBATCH -p {partition}
#SBATCH -N 1 -n {ncores}
#SBATCH -t {wall // 60:02d}:{wall % 60:02d}:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
# {len(members)} case(s), estimated {hours:.1f} h on {ncores} cores
cd "$SLURM_SUBMIT_DIR"
bash ./run_cases.sh {' '.join(c['path'] for c in members)}
""")
        lines.append(f"sbatch {job}")
    with open(os.path.join(out_dir, "submit_all.sh"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(os.path.join(out_dir, "submit_all.sh"), 0o755)
    total = sum(c["core_hours_estimate"] for c in cases)
    est = {"batch": name, "cases": len(cases), "jobs": len(jobs), "ncores": ncores,
           "core_hours": round(total, 1), "yuan_list_price": round(total * PRICE, 2),
           "price_yuan_per_core_hour": PRICE, "partition": partition,
           "note": "rough: 3 core-h per 40-atom single point, ~N^2; 150 core-h per 409-atom "
                   "frame; relaxations ~8 single points. Calibrate with one or two test runs.",
           "by_case": {c["path"]: c["core_hours_estimate"] for c in cases}}
    with open(os.path.join(out_dir, "cost_estimate.json"), "w") as fh:
        json.dump(est, fh, indent=1)
    return est


def boxed(atoms, vacuum: float = 8.0, cubic: bool = True):
    """A copy centred in a box with at least ``vacuum`` Å on every side (cubic by default),
    non-periodic (the DFTB side treats it as a molecule; VASP adds the dipole correction)."""
    a = atoms.copy()
    a.pbc = False
    pos = a.positions
    extent = pos.max(axis=0) - pos.min(axis=0)
    if cubic:
        side = float(np.max(extent) + 2.0 * vacuum)
        a.cell = [side, side, side]
    else:
        a.cell = np.diag(extent + 2.0 * vacuum)
    a.center()
    return a
