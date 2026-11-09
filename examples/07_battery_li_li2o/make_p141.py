"""Probe P141: ENCUT and KSPACING convergence on bcc Li and Li2O (antifluorite)."""
import sys

from ase.build import bulk

from vasp_common import antifluorite, write_case

out = sys.argv[1]
li = bulk("Li", "bcc", a=3.44)
li2o = antifluorite(4.62)
for name, at in (("li_bcc", li), ("li2o", li2o)):
    for encut in (500, 600, 700, 800):
        write_case(f"{out}/{name}_enc{encut}", at, ENCUT=encut, KSPACING=0.20)
    for ks in (0.30, 0.25, 0.20, 0.15):
        write_case(f"{out}/{name}_ks{ks:.2f}", at, ENCUT=800, KSPACING=ks)
