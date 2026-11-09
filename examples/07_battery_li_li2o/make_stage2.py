"""Stage 2 of the M5c data, from the relaxed cells of stage 1 (CONTCARs):

* EOS: 11 volumes from 0.88 to 1.12 V0 for bcc Li, fcc Li, Li2O, Li2O2 (atoms scaled with
  the cell; the hexagonal Li2O2 keeps its c/a and internal z);
* rattled supercells (forces): bcc Li 2x2x2 conventional (16 atoms) and Li2O conventional (12
  atoms), sigma 0.05 and 0.10 A, train seeds 1-3, test seed 101;
* defects to relax at the host's lattice constant (ISIF 2): Li16O (O in an octahedral site of
  bcc Li 2x2x2) and Li8O3 (one O vacancy in conventional Li2O); stage 3 rattles them.

    python make_stage2.py STAGE1_DIR OUT_DIR
"""
import os
import sys

import ase.io
import numpy as np
from ase import Atom
from ase.build import bulk

from vasp_common import write_case

s1, out = sys.argv[1], sys.argv[2]
read = lambda name: ase.io.read(os.path.join(s1, name, "CONTCAR"), format="vasp")
phases = {"li_bcc": read("relax_li_bcc"), "li_fcc": read("relax_li_fcc"),
          "li2o": read("relax_li2o"), "li2o2": read("relax_li2o2")}
for name, at in phases.items():
    for k, f in enumerate(np.linspace(0.88, 1.12, 11)):
        a = at.copy()
        a.set_cell(at.cell * f ** (1 / 3), scale_atoms=True)
        write_case(f"{out}/eos_{name}_{k:02d}", a)

a_bcc = phases["li_bcc"].cell.cellpar()[0] * 2 / np.sqrt(3)       # primitive -> cubic a
a_li2o = phases["li2o"].cell.cellpar()[0] * np.sqrt(2)             # fcc primitive -> cubic a
li16 = bulk("Li", "bcc", a=a_bcc, cubic=True).repeat(2)
li2o_conv = ase.io.read(os.path.join(s1, "relax_li2o", "CONTCAR"))   # primitive; build cubic:
from vasp_common import antifluorite
from ase.build import make_supercell
li2o_cub = make_supercell(antifluorite(a_li2o), np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]))
assert len(li2o_cub) == 12 and abs(li2o_cub.get_volume() - a_li2o ** 3) < 1e-6
for name, host in (("li_bcc16", li16), ("li2o12", li2o_cub)):
    for sigma in (0.05, 0.10):
        for seed in (1, 2, 3, 101):
            a = host.copy()
            a.rattle(stdev=sigma, seed=seed)
            tag = "test" if seed > 100 else "train"
            write_case(f"{out}/rat_{name}_s{int(sigma * 100):02d}_{tag}{seed}", a)

li16o = li16.copy()
li16o.append(Atom("O", (0.5 * a_bcc, 0.5 * a_bcc, 0.0)))           # octahedral site
write_case(f"{out}/relax_li16o", li16o, IBRION=2, ISIF=2, NSW=100, EDIFFG=-2e-3)
li8o3 = li2o_cub.copy()
del li8o3[[i for i, s in enumerate(li8o3.get_chemical_symbols()) if s == "O"][0]]
write_case(f"{out}/relax_li8o3", li8o3, IBRION=2, ISIF=2, NSW=100, EDIFFG=-2e-3)
print(f"bcc a = {a_bcc:.4f} A, Li2O a = {a_li2o:.4f} A")
