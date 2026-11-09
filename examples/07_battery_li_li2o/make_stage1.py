"""Stage 1 of the M5c data: relax the four bulk phases (ISIF 3) and the isolated atoms (E0)."""
import sys

from ase import Atoms
from ase.build import bulk

from vasp_common import antifluorite, li2o2, write_case

out = sys.argv[1]
RELAX = dict(IBRION=2, ISIF=3, NSW=80, EDIFFG=-1e-3, EDIFF=1e-7)
write_case(f"{out}/relax_li_bcc", bulk("Li", "bcc", a=3.44), **RELAX)
write_case(f"{out}/relax_li_fcc", bulk("Li", "fcc", a=4.33), **RELAX)
write_case(f"{out}/relax_li2o", antifluorite(4.62), **RELAX)
write_case(f"{out}/relax_li2o2", li2o2(), **RELAX)
box = [[15.0, 0, 0], [0, 15.5, 0], [0, 0, 16.0]]
for el, nup in (("Li", 1), ("O", 2)):
    write_case(f"{out}/atom_{el}", Atoms(el, positions=[[7.1, 7.3, 7.6]], cell=box, pbc=True),
               gamma_only=True, ISPIN=2, NUPDOWN=nup, MAGMOM=f"{nup}", ISYM=0, ALGO="Normal")
# Spin-unpolarized, spherical atoms: the reference DFTB uses (its atoms have no spin), so the
# atomization energies of DFT and DFTB have the same reference.  These are PTBP's E0s.
for el in ("Li", "O"):
    write_case(f"{out}/atom_{el}_nospin", Atoms(el, positions=[[7.1, 7.3, 7.6]], cell=box, pbc=True),
               gamma_only=True, ISPIN=1, ISYM=0, ALGO="Normal")
