"""VASP inputs for the interface check of M5c (design section 5): one single point per MD
snapshot (snap_XX.vasp from md_interface.py), the settings of the training data except
KSPACING 0.25 (VASP generates 4 x 4 x 2 here -- OUTCAR, checked 2026-11-10 -- instead of 7 x 7 x 2; P141: 1.8 meV/atom on bcc Li, which
cancels in the energies relative to the first snapshot) and no symmetry (rattled frames).

    python make_iface_dft.py MD_DIR OUT_DIR [NCORES] [BUDGET_YUAN]

Writes OUT_DIR/iface_XX/ and OUT_DIR/job.sh (not submitted) and prints a cost estimate from
the measured cost of the rattled 16-17 atom cells of stage 2/3. The job's wall-time limit is
set from BUDGET_YUAN (default 6) at 0.1 yuan / core-h on dg83, so the scheduler stops the job
before the budget is passed; run_all.sh leaves the unfinished snapshots without OUTCAR timing.
"""
import glob
import os
import sys

import ase.io

from vasp_common import write_case

md, out = sys.argv[1], sys.argv[2]
ncores = int(sys.argv[3]) if len(sys.argv) > 3 else 76
budget = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
minutes = int(budget / (0.1 * ncores) * 60)  # hard cap at the list price 0.1 yuan/core-h (billed 0.075: conservative)
snaps = sorted(glob.glob(os.path.join(md, "snap_*.vasp")))
if not snaps:
    sys.exit(f"no snap_*.vasp in {md}")
for s in snaps:
    k = os.path.basename(s)[5:-5]                       # snap_<k>.vasp
    write_case(os.path.join(out, f"iface_{k}"), ase.io.read(s), KSPACING=0.25, ISYM=0,
               NCORE=4, LREAL=".FALSE.")
with open(os.path.join(out, "job.sh"), "w") as fh:
    fh.write(f"""#!/bin/bash
#SBATCH -J m5c_iface
#SBATCH -p dg83
#SBATCH -N 1 -n {ncores}
#SBATCH -t {minutes // 60:02d}:{minutes % 60:02d}:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
bash ~/m5c/run_all.sh iface_*
""")
# cost: rattled Li16 bcc (16 atoms, 7x7x7 -> ~172 k after time reversal) took ~4.3 core-h each
# at ENCUT 800; plane-wave cost ~ N_k * N_atoms^2.3 (FFT + subspace); interface: 80 atoms,
# KSPACING 0.25 -> 4x4x2 in the OUTCARs (20 k-points with ISYM 0); not 4x4x1 as first noted.
n_at = len(ase.io.read(snaps[0]))
per = 4.3 * (8 / 172) * (n_at / 16) ** 2.3
print(f"{len(snaps)} snapshots of {n_at} atoms; estimate {per:.1f} core-h each, "
      f"{per * len(snaps):.0f} core-h = {0.1 * per * len(snaps):.1f} yuan on dg83 "
      f"(rough: x0.5-2); wall-time cap {minutes} min = at most "
      f"{0.1 * ncores * minutes / 60:.2f} yuan")
