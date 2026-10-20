"""Run DFTB+ on the same input as pydftb (reference generation and timing).

Only needed for validation; pydftb itself never calls DFTB+.

Notes on matching DFTB+ exactly:
* ``ParserVersion = 14`` is always written.  For older parser versions DFTB+
  silently rewrites ``MethfesselPaxton { Order = n }`` to ``n - 1``
  (``dftbplus/oldcompat.F90``), and ASE writes version 1.
* DFTB+ differentiates H0/S numerically.  Its default finite difference is
  used for the references: ``Differentiation = Richardson {}`` was tested and
  is *less* accurate for these tables (H2O force error 1.2e-5 vs 1.2e-6
  Ha/Bohr), presumably because its larger steps cross the grid nodes where
  the piecewise SK interpolation polynomial changes.  pydftb's complex-step
  derivatives agree with its own energy finite differences to ~3e-9.
* Energies come from ``detailed.out`` (10 decimals), everything else from
  ``results.tag`` (full precision).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

DFTBPLUS = os.environ.get("DFTBPLUS", shutil.which("dftb+") or "/opt/mm/dftb/bin/dftb+")
PARSER_VERSION = 14


def available() -> bool:
    return bool(DFTBPLUS) and os.path.exists(DFTBPLUS)


def _f(x) -> str:
    """Shortest exact (round-trip) text form of a float."""
    return repr(float(x))


def write_gen(path, labels, pos_ang, cell_ang=None):
    """DFTB+ gen file; species names are the type labels (Type2FileNames)."""
    species = list(dict.fromkeys(labels))
    with open(path, "w") as f:
        f.write(f"{len(labels)} {'S' if cell_ang is not None else 'C'}\n")
        f.write(" ".join(species) + "\n")
        for k, (s, p) in enumerate(zip(labels, pos_ang)):
            f.write(f"{k + 1} {species.index(s) + 1} {_f(p[0])} {_f(p[1])} {_f(p[2])}\n")
        if cell_ang is not None:
            f.write("0.0 0.0 0.0\n")
            for v in cell_ang:
                f.write(f"{_f(v[0])} {_f(v[1])} {_f(v[2])}\n")


def filling_block(filling) -> str:
    """("fermi", T_K) | ("mp", T_K, order) | ("gaussian", T_K) -> HSD."""
    kind = filling[0].lower()
    temp = _f(filling[1])
    if kind == "fermi":
        return f"Fermi {{\n    Temperature [K] = {temp}\n  }}"
    if kind in ("mp", "methfesselpaxton"):
        return (f"MethfesselPaxton {{\n    Order = {int(filling[2])}\n"
                f"    Temperature [K] = {temp}\n  }}")
    if kind == "gaussian":
        # DFTB+ has no separate Gaussian keyword: it is Methfessel-Paxton order 0
        return f"MethfesselPaxton {{\n    Order = 0\n    Temperature [K] = {temp}\n  }}"
    raise ValueError(f"unknown filling {filling!r}")


def make_hsd(skdir, max_l, periodic, kpts=None, scc=True, filling=("fermi", 0.0),
             shell_resolved=False, scc_tol=1e-10, max_scc=500, repulsive="skf",
             richardson=False, charge=0.0, extra_ham="", timing=False) -> str:
    maxl = "\n".join(f'    {e} = "{"spdf"[l]}"' for e, l in max_l.items())
    kblock = ""
    if periodic:
        if kpts is None:
            kpts = (np.zeros((1, 3)), np.ones(1))
        k, w = kpts
        lines = "\n".join(f"    {_f(a)} {_f(b)} {_f(c)} {_f(x)}" for (a, b, c), x in zip(np.atleast_2d(k), w))
        kblock = f"  KPointsAndWeights = {{\n{lines}\n  }}\n"
    poly = "  PolynomialRepulsive = SetForAll { Yes }\n" if repulsive == "polynomial" else ""
    diff = "  Differentiation = Richardson {}\n" if richardson else ""
    scc_block = ""
    if scc:   # DFTB+ rejects SCC-only keywords in non-SCC runs
        scc_block = (f"  SCCTolerance = {_f(scc_tol)}\n  MaxSCCIterations = {max_scc}\n"
                     f"  ShellResolvedSCC = {'Yes' if shell_resolved else 'No'}\n")
    return f"""Geometry = GenFormat {{
  <<< "geo.gen"
}}
Hamiltonian = DFTB {{
  SCC = {"Yes" if scc else "No"}
{scc_block}  Charge = {_f(charge)}
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{os.path.abspath(skdir)}/"
    Separator = "-"
    Suffix = ".skf"
  }}
  MaxAngularMomentum {{
{maxl}
  }}
  Filling = {filling_block(filling)}
{kblock}{poly}{diff}{extra_ham}
}}
Analysis {{
  PrintForces = Yes
  MullikenAnalysis = Yes
}}
Options {{
  WriteDetailedOut = Yes
  WriteResultsTag = Yes
  TimingVerbosity = {2 if timing else 0}
}}
ParserOptions {{
  ParserVersion = {PARSER_VERSION}
}}
Parallel {{
  UseOmpThreads = Yes
}}
"""


def parse_results_tag(path) -> dict:
    """Parse DFTB+ ``results.tag`` (name :type:rank:shape, then values)."""
    out = {}
    with open(path) as fh:
        text = fh.read()
    blocks = re.split(r"^(\w+)\s*:(\w+):(\d+):([\d,]*)\s*$", text, flags=re.M)
    for name, typ, rank, shape, body in zip(*[iter(blocks[1:])] * 5):
        vals = np.array([float(x) for x in body.split()]) if typ == "real" else body.split()
        rank = int(rank)
        if typ == "real" and rank == 0:
            out[name] = float(vals[0])
        elif typ == "real":
            dims = [int(x) for x in shape.split(",")]
            out[name] = vals.reshape(dims[::-1])        # Fortran column-major
        else:
            out[name] = vals
    return out


def run(labels, pos_ang, cell_ang=None, skdir=".", max_l=None, kpts=None, scc=True,
        filling=("fermi", 0.0), shell_resolved=False, scc_tol=1e-10, max_scc=500,
        repulsive="skf", richardson=False, charge=0.0, extra_ham="", threads=1,
        timing=False, workdir=None, keep=False) -> dict:
    """Run DFTB+ and return energies (Ha), forces (Ha/Bohr), charges, metadata."""
    if not available():
        raise FileNotFoundError("dftb+ binary not found; set $DFTBPLUS")
    own_dir = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="dftbp_")
    try:
        write_gen(os.path.join(workdir, "geo.gen"), labels, np.asarray(pos_ang),
                  None if cell_ang is None else np.asarray(cell_ang))
        hsd = make_hsd(skdir, max_l, cell_ang is not None, kpts, scc, filling, shell_resolved,
                       scc_tol, max_scc, repulsive, richardson, charge, extra_ham, timing)
        with open(os.path.join(workdir, "dftb_in.hsd"), "w") as f:
            f.write(hsd)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
        t0 = time.perf_counter()
        p = subprocess.run([DFTBPLUS], cwd=workdir, env=env, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        if p.returncode != 0:
            raise RuntimeError(p.stdout[-3000:] + p.stderr[-3000:])
        det = open(os.path.join(workdir, "detailed.out")).read()
        tag = parse_results_tag(os.path.join(workdir, "results.tag"))
        res = {"wall": wall, "hsd": hsd, "stdout": p.stdout, "dir": workdir}
        for key, name in [("energy", "Total energy"), ("free_energy", "Total Mermin free energy"),
                          ("e_zero", "Extrapolated to 0"), ("e_rep", "Repulsive energy"),
                          ("e_scc", "Energy SCC"), ("e_band", "Energy H0"), ("ts", "-TS")]:
            m = re.search(rf"^{re.escape(name)}:\s+(\S+)\s+H", det, re.M)
            if m:
                res[key] = float(m.group(1))
        if "ts" in res:
            res["ts"] = -res["ts"]          # detailed.out prints -TS
        res["fermi"] = float(np.atleast_1d(tag["fermi_level"])[0])
        res["forces"] = tag["forces"]
        res["charges_net"] = tag["gross_atomic_charges"]
        res["mermin_tag"] = tag.get("mermin_energy")
        res["extrapolated0_tag"] = tag.get("extrapolated0_energy")
        ver = re.search(r"\|\s+(DFTB\+[^|]*?)\s*$", p.stdout, re.M)
        res["version"] = ver.group(1) if ver else "unknown"
        iters = re.findall(r"^\s+(\d+)\s+[-\d.E+]+\s+[-\d.E+]+\s+[-\d.E+]+\s*$", p.stdout, re.M)
        res["n_scc"] = int(iters[-1]) if iters else (1 if not scc else None)
        return res
    finally:
        if own_dir and not keep:
            shutil.rmtree(workdir, ignore_errors=True)
