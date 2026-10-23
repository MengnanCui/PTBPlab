"""Golden check: the NumPy path must stay bitwise identical to an earlier commit (M4).

The same calculations run in the working tree and in a git worktree of a base commit,
each in its own subprocess with the same interpreter and one BLAS thread.  The raw bytes
of every result field and of the intermediates are compared: pair blocks and their
derivatives, H0/S, the gamma and third-order matrices, the final Hamiltonian, the
density matrices, eigenvalues, occupations, band states, trajectories.

Calculations:

* every frozen reference case (``tests/data/ref``) with the ``eigh`` and ``batched``
  solvers, plus the batched solver with DFTB+'s finite-difference step;
* the first steps of every optimisation / MD reference (``tests/data/ref_dynamics``);
* PTBP's settings (``utils/engines.py``): Cu cells with Monkhorst-Pack grids, the
  0.01 Bohr finite-difference forces, shell-resolved SCC, one and two k-threads.

    python -m pydftb.validation.golden --base 65115d1           # all
    python -m pydftb.validation.golden --base 65115d1 --quick   # a subset (about 1/4)

Nothing platform-specific is stored: both sides are computed on the spot.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONE_THREAD = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                               "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                               "NUMEXPR_NUM_THREADS")}

# Runs inside the tree being checked (``import pydftb`` resolves to that tree), so it may
# only use what both the base commit and the working tree provide.
DUMP = r'''
import os, pickle, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
out_path, quick = sys.argv[1], sys.argv[2] == "1"

from pydftb import calculator as C
from pydftb.validation import refgen as rg, dynamics as dyn

CAPTURE = []
_setup, _scc = C.DFTB._setup, C.DFTB._scc

def setup(self, *a, **k):
    st = _setup(self, *a, **k)
    CAPTURE.append(("setup", st))
    return st

def scc(self, *a, **k):
    sc = _scc(self, *a, **k)
    CAPTURE.append(("scc", sc))
    return sc

C.DFTB._setup, C.DFTB._scc = setup, scc
STATES = []
_eig = rg.eigenstates

def eig(*a, **k):
    st = _eig(*a, **k)
    STATES.append(st)
    return st

rg.eigenstates = eig
SKIP = {"timings", "wall", "t_build", "t_diag"}

def flat(prefix, obj, out, depth=0):
    """Arrays, numbers and strings of obj (recursively) under dotted names."""
    if depth > 4:
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype != object:
            out[prefix] = ("a", obj.dtype.str, obj.shape, obj.tobytes())
        else:
            for i, v in enumerate(obj.ravel()):
                flat(f"{prefix}[{i}]", v, out, depth + 1)
    elif isinstance(obj, (bool, int, float, complex, np.generic)):
        a = np.asarray(obj)
        out[prefix] = ("a", a.dtype.str, a.shape, a.tobytes())
    elif obj is None or isinstance(obj, str):
        out[prefix] = ("v", obj)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            flat(f"{prefix}[{i}]", v, out, depth + 1)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k not in SKIP:
                flat(f"{prefix}.{k}", v, out, depth + 1)
    elif hasattr(obj, "__dict__") and not callable(obj):
        for k, v in vars(obj).items():
            if k not in SKIP and not k.startswith("_"):
                flat(f"{prefix}.{k}", v, out, depth + 1)

def record(result):
    out = {}
    flat("result", result, out)
    for i, (kind, obj) in enumerate(CAPTURE):
        flat(f"{kind}{i}", obj, out)
    for i, st in enumerate(STATES):
        flat(f"states{i}", st, out)
    CAPTURE.clear()
    STATES.clear()
    return out

data = {}
names = sorted(f[:-5] for f in os.listdir(rg.REF_DIR) if f.endswith(".json"))
if quick:
    names = names[::4]
for name in names:
    doc = rg.load(name)
    if rg.missing_requirement(doc):
        continue
    for solver, fd in (("eigh", None), ("batched", None), ("batched", 1.22e-4)):
        CAPTURE.clear(); STATES.clear()
        r = rg.compare(doc, solver, fd_delta=fd)
        data[f"ref/{name}/{solver}/{fd}"] = record(r["result"])

from ase.optimize import BFGS
dyn.FD_DELTA = 2.0 ** -13      # one step in both trees (M6e corrected the validation's)
for case in dyn.DYN_CASES[::4] if quick else dyn.DYN_CASES:
    path = os.path.join(dyn.REF_DIR, f"{case.name}.json")
    if not os.path.exists(path):
        continue
    doc = dyn.load(case.name)
    if rg.missing_requirement(doc):
        continue
    inp = doc["input"]
    CAPTURE.clear()
    if case.kind == "md":
        res = dyn.run_pydftb(case, inp, steps=2)
    else:
        atoms = dyn._atoms(inp)
        calc = dyn.pydftb_calculator(case, inp)
        dyn._with_masses(atoms, calc, inp["labels"])
        atoms.calc = calc
        BFGS(dyn._target(atoms, case), logfile=None).run(fmax=1e-12, steps=3)
        res = {"positions": atoms.positions, "cell": atoms.cell.array,
               "free_energy": calc.raw.free_energy, "forces": calc.raw.forces,
               "stress": calc.raw.stress}
    del CAPTURE[:-2]                      # the last calculation's setup and SCC state
    data[f"dyn/{case.name}"] = record(res)

from ase.build import bulk
from pydftb.ase_calc import PyDFTB
skdir = rg.SKF_SETS["ptbp_hcocu"]
rng = np.random.default_rng(20261022)
cu1 = bulk("Cu", "fcc", a=3.61)
cu2 = bulk("Cu", "fcc", a=3.63).repeat((2, 1, 1))
cu2.positions += rng.normal(scale=0.05, size=cu2.positions.shape)
cu2.set_cell(cu2.cell.array @ (np.eye(3) + rng.normal(scale=0.01, size=(3, 3))),
             scale_atoms=True)
runs = [("cu1", cu1, (8, 8, 8), 0.01, 1, ("fermi", 0.0)),
        ("cu2", cu2, (4, 6, 6), 0.01, 1, ("fermi", 0.0)),
        ("cu2_k2", cu2, (4, 6, 6), 0.01, 2, ("fermi", 0.0)),
        ("cu2_exact_mp", cu2, (3, 4, 4), None, 1, ("mp", 2000.0, 2))]
for label, atoms, kpts, fd, kth, smear in runs[::2] if quick else runs:
    a = atoms.copy()
    a.calc = PyDFTB(skdir, max_l={"Cu": 2}, kpts=kpts, smearing=smear, scc_tol=1e-5,
                    max_scc=1000, repulsive="polynomial", shell_resolved=True, fd_delta=fd,
                    threads=1, k_threads=kth)
    CAPTURE.clear()
    f = a.get_forces()
    s = a.get_stress()
    data[f"ptbp/{label}"] = record({"raw": a.calc.raw, "forces": f, "stress": s,
                                    "energy": a.get_potential_energy()})

with open(out_path, "wb") as fh:
    pickle.dump(data, fh, protocol=4)
print(len(data))
'''


def _git(*args, cwd=REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def worktree(base: str, root: str) -> str:
    """A detached worktree of ``base`` under ``root`` (reused when it already exists)."""
    sha = _git("rev-parse", "--short", base)
    path = os.path.join(root, f"golden_{sha}")
    if not os.path.isdir(path):
        _git("worktree", "add", "--detach", path, sha)
    return path


def dump(tree: str, out: str, quick: bool, python: str = sys.executable) -> float:
    """Run the calculations in ``tree``; returns the wall time."""
    with tempfile.NamedTemporaryFile("w", suffix="_golden_dump.py", delete=False) as fh:
        fh.write(DUMP)
        script = fh.name
    env = dict(os.environ, **ONE_THREAD, PYTHONPATH=tree, PYTHONHASHSEED="0")
    env.pop("PTBP_NCORES", None)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([python, script, out, "1" if quick else "0"], cwd=tree, env=env,
                              capture_output=True, text=True)
    finally:
        os.unlink(script)
    if proc.returncode != 0:
        raise RuntimeError(f"golden dump failed in {tree}:\n{proc.stderr[-4000:]}")
    return time.perf_counter() - t0


def _diff(a, b) -> str:
    if a[0] != b[0]:
        return "kind"
    if a[0] == "v":
        return "" if a[1] == b[1] else f"{a[1]!r} != {b[1]!r}"
    if a[1:3] != b[1:3]:
        return f"dtype/shape {a[1]}{a[2]} != {b[1]}{b[2]}"
    if a[3] == b[3]:
        return ""
    x = np.frombuffer(a[3], dtype=a[1]).reshape(a[2])
    y = np.frombuffer(b[3], dtype=b[1]).reshape(b[2])
    if x.dtype.kind in "fc":
        d = np.abs(x - y)
        return f"max |diff| {np.nanmax(d):.3e}" if d.size else "bytes differ"
    return "values differ"


def _components(field: str) -> set:
    return {p.rstrip("]") for p in re.split(r"[.\[]", field)}


def compare(base: dict, work: dict, allow_missing=()) -> tuple[int, list]:
    """(number of fields compared, problems).  A field present on one side only is a
    problem unless one of the components of its dotted name is in ``allow_missing``.
    Calculations of the work tree only (references added since the base, e.g. M6a's
    two-component cases) have nothing to compare with: see :func:`new_calculations`."""
    problems, n = [], 0
    allow = set(allow_missing)
    for key in sorted(set(base) | set(work)):
        if key not in base:
            continue
        if key not in work:
            problems.append(f"{key}: only in base")
            continue
        fb, fw = base[key], work[key]
        for field in sorted(set(fb) | set(fw)):
            if field not in fb or field not in fw:
                if not _components(field) & allow:
                    side = "base" if field in fb else "work tree"
                    problems.append(f"{key}: {field} only in {side}")
                continue
            n += 1
            why = _diff(fb[field], fw[field])
            if why:
                problems.append(f"{key}: {field}: {why}")
    return n, problems


def new_calculations(base: dict, work: dict) -> list:
    """Calculations of the work tree that the base commit does not have."""
    return sorted(set(work) - set(base))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="commit to compare with (e.g. 65115d1)")
    ap.add_argument("--quick", action="store_true", help="about a quarter of the cases")
    ap.add_argument("--python", default=sys.executable, help="interpreter for both sides")
    ap.add_argument("--root", default=tempfile.gettempdir(),
                    help="where the base worktree and the dumps go")
    ap.add_argument("--allow-new", nargs="*", default=(),
                    help="names of fields that may exist on one side only (new or renamed "
                         "internals, e.g. dev)")
    a = ap.parse_args(argv)
    base_tree = worktree(a.base, a.root)
    out_b = os.path.join(a.root, "golden_base.pkl")
    out_w = os.path.join(a.root, "golden_work.pkl")
    tb = dump(base_tree, out_b, a.quick, a.python)
    tw = dump(REPO, out_w, a.quick, a.python)
    with open(out_b, "rb") as fh:
        base = pickle.load(fh)
    with open(out_w, "rb") as fh:
        work = pickle.load(fh)
    n, problems = compare(base, work, a.allow_new)
    new = new_calculations(base, work)
    print(f"golden: {len(work) - len(new)} calculations, {n} fields compared "
          f"(base {a.base} {tb:.0f} s, work tree {tw:.0f} s, {os.path.basename(a.python)})")
    if new:
        print(f"  {len(new)} new calculations, not in the base (not compared): "
              f"{', '.join(sorted({k.split('/')[1] for k in new}))}")
    for p in problems[:40]:
        print("  " + p)
    if len(problems) > 40:
        print(f"  ... {len(problems) - 40} more")
    print("bitwise identical" if not problems else f"{len(problems)} differences")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
