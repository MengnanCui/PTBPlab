"""Stage 3 of the M5c data: rattle the relaxed defects of stage 2 (Li16O: O in bcc Li, mixed
Li environments with 0-6 O neighbours; Li8O3: O vacancy in Li2O), sigma 0.05 and 0.10 A,
train seeds 1-3, test seed 101.

    python make_stage3.py STAGE2_DIR OUT_DIR
"""
import os
import sys

import ase.io

from vasp_common import write_case

s2, out = sys.argv[1], sys.argv[2]
for name, relax in (("li16o", "relax_li16o"), ("li8o3", "relax_li8o3")):
    host = ase.io.read(os.path.join(s2, relax, "CONTCAR"), format="vasp")
    for sigma in (0.05, 0.10):
        for seed in (1, 2, 3, 101):
            a = host.copy()
            a.rattle(stdev=sigma, seed=seed)
            tag = "test" if seed > 100 else "train"
            write_case(f"{out}/rat_{name}_s{int(sigma * 100):02d}_{tag}{seed}", a)
