"""PTBP-style workload: one BO step = evaluate E+F for every structure in a dataset.

DFTB+: one subprocess per structure (inputs written, SKFs re-read, outputs parsed),
as PTBP/ASE does today.  pydftb: in-process, SKFs parsed once per BO step.
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from ase.io import read
from ase.calculators.calculator import kptdensity2monkhorstpack
from threadpoolctl import threadpool_limits
import dftbplus_ref as ref
from pydftb.skf import SKSet
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
from pydftb.kpoints import monkhorst_pack, reduce_time_reversal

SK = os.environ.get("SKDIR", "/tmp/claude-0/skf2")
DATA = os.environ.get("DATA", os.path.join(os.path.dirname(__file__), "../../dft/dft.xyz"))
TEMP = 2000.0
images = read(DATA, ":")
jobs = []
for at in images:
    k = reduce_time_reversal(*monkhorst_pack(kptdensity2monkhorstpack(at, 5.0, even=False)))
    jobs.append((at, k))
print(f"{len(jobs)} structures, k-points per structure: {min(len(k[0]) for _, k in jobs)}-{max(len(k[0]) for _, k in jobs)}")


def run_dftbp(job):
    at, k = job
    d = ref.run(at.get_chemical_symbols(), at.positions, at.cell.array, skdir=SK,
                max_l={"Cu": 2}, kpts=k, temperature=TEMP, scc_tol=1e-5)
    return d["free_energy"]


_SK = None


def run_py(job):
    global _SK
    at, k = job
    with threadpool_limits(1):
        r = DFTB(_SK, kpts=k, temperature=TEMP, scc_tol=1e-5, solver="batched").calculate(
            at.get_chemical_symbols(), at.positions / BOHR, at.cell.array / BOHR, at.pbc)
    return r.free_energy


def init():
    global _SK
    _SK = SKSet(SK, ["Cu"], {"Cu": 2})


res = {}
t = time.perf_counter(); e_ref = [run_dftbp(j) for j in jobs]; res["dftb+_serial"] = time.perf_counter() - t
with ThreadPoolExecutor(4) as ex:
    t = time.perf_counter(); list(ex.map(run_dftbp, jobs)); res["dftb+_4proc"] = time.perf_counter() - t
t = time.perf_counter(); init(); e_py = [run_py(j) for j in jobs]; res["pydftb_serial"] = time.perf_counter() - t
with ProcessPoolExecutor(4, initializer=init) as ex:
    t = time.perf_counter(); list(ex.map(run_py, jobs, chunksize=4)); res["pydftb_4proc"] = time.perf_counter() - t
res["max_dE_Ha"] = float(np.abs(np.array(e_py) - np.array(e_ref)).max())
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(__file__), "results_throughput.json"), "w"), indent=1)
