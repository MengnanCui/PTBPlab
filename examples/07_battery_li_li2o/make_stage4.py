"""Stage 4 of the M5c data (follow-up, report section 9): the short range the first fits never saw.

* dimers Li2, LiO, O2 in the box of the isolated atoms (Gamma only, spin-unpolarized like the
  E0 atoms, so the atomization energies have the same reference as the rest of the data);
* compressed bulk: bcc Li, Li2O and Li2O2 at 0.70-0.85 V0 (atoms scaled with the cell).

    python make_stage4.py STAGE1_DIR OUT_DIR [NCORES] [BUDGET_YUAN]

Writes OUT_DIR/<case>/ and OUT_DIR/job.sh (not submitted).  The wall-time limit is set from
BUDGET_YUAN (default 4) at the list price 0.1 yuan / core-h (billed 0.075: conservative).
"""
import os
import sys

import ase.io
import numpy as np
from ase import Atoms

from vasp_common import write_case

s1, out = sys.argv[1], sys.argv[2]
ncores = int(sys.argv[3]) if len(sys.argv) > 3 else 16
budget = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
minutes = int(budget / (0.1 * ncores) * 60)

box = [[15.0, 0, 0], [0, 15.5, 0], [0, 0, 16.0]]
centre = np.array([7.1, 7.3, 7.6])
grids = {
    "Li2": np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.5]),
    "LiO": np.array([1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5]),
    "O2": np.array([1.0, 1.1, 1.2, 1.3, 1.45, 1.6, 1.8, 2.1, 2.5, 3.0, 3.5]),
}
n = 0
for name, rs in grids.items():
    for r in rs:
        a = Atoms(name, positions=[centre - [0, 0, r / 2], centre + [0, 0, r / 2]], cell=box,
                  pbc=True)
        write_case(f"{out}/dimer_{name}_{r:.2f}", a, gamma_only=True, ISPIN=1, ISYM=0)
        n += 1
read = lambda d: ase.io.read(os.path.join(s1, d, "CONTCAR"), format="vasp")
for name in ("li_bcc", "li2o", "li2o2"):
    at = read(f"relax_{name}")
    for f in (0.70, 0.75, 0.80, 0.85):
        a = at.copy()
        a.set_cell(at.cell * f ** (1 / 3), scale_atoms=True)
        write_case(f"{out}/cmp_{name}_{int(round(f * 100))}", a)
        n += 1
with open(os.path.join(out, "job.sh"), "w") as fh:
    fh.write(f"""#!/bin/bash
#SBATCH -J m5c_s4
#SBATCH -p dg83
#SBATCH -N 1 -n {ncores}
#SBATCH -t {minutes // 60:02d}:{minutes % 60:02d}:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
bash ~/m5c/run_all.sh dimer_* cmp_*
""")
# measured on 16 cores: isolated atoms in this box 23-26 s, EOS points 10-46 s (stage 2)
est = 16 * (31 * 60 + 12 * 45) / 3600
print(f"{n} cases; estimate ~{est:.1f} core-h = {0.1 * est:.2f} yuan list ({0.075 * est:.2f} "
      f"billed); wall-time cap {minutes} min = at most {0.1 * ncores * minutes / 60:.2f} yuan list")
