"""Frozen DFTB+ runs for the tests of the configuration advisor (milestone M3d).

Each case is a small DFTB+ input that shows one pitfall (or none); DFTB+ 25.1 is run on
it and the input, the processed input (``dftb_pin.hsd``), the screen output
(``dftb.out``), the geometry and the SKF headers are frozen in
``pydftb/tests/data/advisor/<case>/``, so the tests know exactly what DFTB+ printed.
The SKF copies keep the lines the advisor reads (format line, on-site line, the Spline
section) but not the tables.

    python -m pydftb.validation.advisor_cases --regenerate [CASE ...]   # needs dftb+

It writes into the package's ``tests/data/advisor``: run it in a git checkout, not in an
installed copy of pydftb.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .dftbplus import DFTBPLUS, available

HERE = Path(__file__).resolve().parent
SKF_DIR = HERE.parent / "tests" / "data" / "skf"
OUT_DIR = HERE.parent / "tests" / "data" / "advisor"

CU_PRIM = """1 S
Cu
1 1 0.0 0.0 0.0
0.0 0.0 0.0
0.0 1.805 1.805
1.805 0.0 1.805
1.805 1.805 0.0
"""
CU_CUBIC = """4 S
Cu
1 1 0.02 0.0 0.0
2 1 0.0 1.805 1.805
3 1 1.805 0.0 1.805
4 1 1.805 1.805 0.0
0.0 0.0 0.0
3.61 0.0 0.0
0.0 3.61 0.0
0.0 0.0 3.61
"""
GDO_PRIM = """2 S
Gd O
1 1 0.0 0.0 0.0
2 2 2.45 2.45 2.45
0.0 0.0 0.0
0.0 2.45 2.45
2.45 0.0 2.45
2.45 2.45 0.0
"""

SK_PTBP = """   SlaterKosterFiles = Type2FileNames {
      Prefix = ./
      Separator = "-"
      Suffix = ".skf"
   }"""


def _hsd(body: str, parser: int = 14, geometry: str = '<<< "geo.gen"', extra: str = "") -> str:
    return f"""Geometry = GenFormat {{
    {geometry}
}}
{body}
{extra}ParserOptions {{
   ParserVersion = {parser}
}}
"""


# PTBP's historical input, as ASE writes it (utils/calculator.py, --smearing none)
PTBP_LEGACY = """Geometry = GenFormat {
    <<< "geo_end.gen"
}

Analysis {
   CalculateForces = Yes
   }
Hamiltonian = DFTB{
   ConvergentSCCOnly = No
   Fermi = {
      Temperature = 2000
   }
   ForceEvaluation = dynamics
   KPointsAndWeights = SupercellFolding {
      4 0 0
      0 4 0
      0 0 4
      0.5 0.5 0.5
   }
   MaxAngularMomentum = {
      Cu = "d"
   }
   MaxSCCIterations = 1000
   MethfesselPaxton = {
      Order = 2
   }
   PolynomialRepulsive = SetForAll {Yes}
   ReadInitialCharges = No
   SCC = Yes
   SCCTolerance = 1e-05
   SlaterKosterFiles = Type2FileNames{
      Prefix = ./
      Separator = "-"
      Suffix = ".skf"
      }
   }
Options {
   WriteDetailedOut = Yes
   WriteResultsTag = Yes
   }
Parallel {
   Groups = 1
   }
ParserOptions {
   IgnoreUnprocessedNodes = Yes
   ParserVersion = 1
}
"""


def _dftb(lines: str, max_l: str = 'Cu = "d"', kpts: str = "SupercellFolding {\n      2 0 0\n      0 2 0\n      0 0 2\n      0.5 0.5 0.5\n   }") -> str:
    return f"""Hamiltonian = DFTB {{
   SCC = Yes
   SCCTolerance = 1e-6
   MaxAngularMomentum {{
      {max_l}
   }}
   KPointsAndWeights = {kpts}
{SK_PTBP}
{lines}
}}
"""


CASES = {
    # name: (input text, geometry file name and text, SKF set, species, OMP threads)
    "ptbp_legacy": (PTBP_LEGACY, ("geo_end.gen", CU_PRIM), "ptbp_hcocu", ["Cu"], 1),
    "clean": (_hsd(_dftb("""   PolynomialRepulsive = SetForAll { Yes }
   Filling = Fermi { Temperature [K] = 300 }
   EwaldTolerance = 1e-12""")), ("geo.gen", CU_PRIM), "ptbp_hcocu", ["Cu"], 1),
    "scc_unconverged": (_hsd(_dftb("""   PolynomialRepulsive = SetForAll { Yes }
   MaxSCCIterations = 3
   ConvergentSCCOnly = No""")), ("geo.gen", CU_CUBIC), "ptbp_hcocu", ["Cu"], 1),
    "no_spline": (_hsd(_dftb("   Filling = Fermi { Temperature [K] = 1000 }",
                             max_l='Gd = "f"\n      O = "p"')),
                  ("geo.gen", GDO_PRIM), "hotcent_gdo", ["Gd", "O"], 1),
    "klines": (_hsd(_dftb("   PolynomialRepulsive = SetForAll { Yes }",
                          kpts="Klines {\n      1 0.0 0.0 0.0\n      4 0.5 0.0 0.0\n   }")),
               ("geo.gen", CU_CUBIC), "ptbp_hcocu", ["Cu"], 1),
    "mp_order": (_hsd(_dftb("""   PolynomialRepulsive = SetForAll { Yes }
   Filling = MethfesselPaxton {
      Order = 2
      Temperature [K] = 1000
   }"""), parser=13),
                 ("geo.gen", CU_PRIM), "ptbp_hcocu", ["Cu"], 1),
    "temp_no_unit": (_hsd(_dftb("""   PolynomialRepulsive = SetForAll { Yes }
   Filling = Fermi { Temperature = 2000 }""")), ("geo.gen", CU_PRIM), "ptbp_hcocu", ["Cu"], 1),
    "groups": (_hsd(_dftb("   PolynomialRepulsive = SetForAll { Yes }"),
                    extra="Parallel {\n   Groups = 2\n}\n"), ("geo.gen", CU_PRIM), "ptbp_hcocu", ["Cu"], 1),
    "omp_small": (_hsd(_dftb("   PolynomialRepulsive = SetForAll { Yes }")),
                  ("geo.gen", CU_PRIM), "ptbp_hcocu", ["Cu"], 4),
    "d4_periodic": (_hsd(_dftb("""   PolynomialRepulsive = SetForAll { Yes }
   Dispersion = DftD4 {
      s6 = 1.0
      s8 = 0.6
      s9 = 1.0
      a1 = 0.5
      a2 = 4.0
   }""")), ("geo.gen", CU_PRIM), "ptbp_hcocu", ["Cu"], 1),
    "lattice_iso": (_hsd(_dftb("   PolynomialRepulsive = SetForAll { Yes }"),
                         extra="""Driver = GeometryOptimisation {
   Optimiser = LBFGS {}
   LatticeOpt = Yes
   Isotropic = Yes
   MaxSteps = 2
}
"""), ("geo.gen", CU_PRIM), "ptbp_hcocu", ["Cu"], 1),
}


def _skf_header(src: Path) -> str:
    """Format and on-site lines plus the Spline section: what the advisor reads."""
    opener = gzip.open if src.suffix == ".gz" else open
    with opener(src, "rt") as fh:
        lines = fh.read().splitlines()
    head = lines[:3] if lines[0].lstrip().startswith("@") else lines[:2]
    spline = next((i for i, ln in enumerate(lines) if ln.strip().lower().startswith("spline")), None)
    tail = lines[spline:] if spline is not None else []
    return "\n".join(head + ["# (tables removed: advisor test copy)"] + tail) + "\n"


def regenerate(name: str) -> dict:
    text, (gname, gtext), skset, species, omp = CASES[name]
    out = OUT_DIR / name
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "dftb_in.hsd").write_text(text)
        (tmp / gname).write_text(gtext)
        for a in species:
            for b in species:
                src = SKF_DIR / skset / f"{a}-{b}.skf.gz"
                with gzip.open(src, "rb") as fi, open(tmp / f"{a}-{b}.skf", "wb") as fo:
                    shutil.copyfileobj(fi, fo)
        env = dict(os.environ, OMP_NUM_THREADS=str(omp))
        proc = subprocess.run([DFTBPLUS], cwd=tmp, env=env, capture_output=True, text=True, timeout=600)
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        shutil.copy(tmp / "dftb_in.hsd", out / "dftb_in.hsd")
        shutil.copy(tmp / gname, out / gname)
        if (tmp / "dftb_pin.hsd").exists():
            shutil.copy(tmp / "dftb_pin.hsd", out / "dftb_pin.hsd")
        (out / "dftb.out").write_text(proc.stdout + proc.stderr)
        for a in species:
            for b in species:
                (out / f"{a}-{b}.skf").write_text(_skf_header(SKF_DIR / skset / f"{a}-{b}.skf.gz"))
    meta = {"returncode": proc.returncode, "omp_threads": omp, "skf_set": skset}
    (out / "case.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("cases", nargs="*")
    a = ap.parse_args(argv)
    if not a.regenerate:
        for name in CASES:
            meta = OUT_DIR / name / "case.json"
            print(f"{name:18s} {'frozen' if meta.exists() else 'missing'}")
        return 0
    if not available():
        print("dftb+ not found (set DFTBPLUS)", file=sys.stderr)
        return 2
    for name in a.cases or CASES:
        meta = regenerate(name)
        print(f"{name:18s} dftb+ exit {meta['returncode']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
