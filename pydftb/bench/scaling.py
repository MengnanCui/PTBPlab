"""Where does the time go?  pydftb (reference) vs DFTB+ on growing water boxes."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from threadpoolctl import threadpool_limits
from pydftb.skf import SKSet
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
import dftbplus_ref as ref
from systems import water_box

SK = os.environ.get("SKDIR", "/tmp/claude-0/skf2")
out = []
for n in [int(x) for x in sys.argv[1:]] or [3, 4, 5, 6, 7, 8]:
    at = water_box(n)
    sym = at.get_chemical_symbols()
    sk = SKSet(SK, ["H", "O"], {"H": 0, "O": 1})
    row = {"natom": len(at), "norb": 2 * len(at)}
    for thr in (1, 4):
        with threadpool_limits(thr):
            r = DFTB(sk, scc=True, scc_tol=1e-6).calculate(sym, at.positions / BOHR, at.cell.array / BOHR, at.pbc)
        row[f"py{thr}"] = {k: round(v, 3) if isinstance(v, float) else v for k, v in r.timings.items()}
        d = ref.run(sym, at.positions, at.cell.array, skdir=SK, max_l={"H": 0, "O": 1},
                    threads=thr, scc_tol=1e-6, timing=True)
        row[f"dftb{thr}"] = round(d["wall"], 3)
        row["dE"] = r.free_energy - d["free_energy"]
    print(json.dumps(row), flush=True)
    out.append(row)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "results_scaling.json"), "w"), indent=1)
