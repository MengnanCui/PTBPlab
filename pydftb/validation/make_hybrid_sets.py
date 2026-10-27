"""Create the hybrid-functional test sets (milestone M6h-1): copies of ``ptbp_hcocu`` with a
``RangeSep`` section in every file, inserted before ``<Documentation>`` as in DFTB+'s ob2 files.

* ``ptbp_hcocu_lc``: H, C, O, Cu; ``LC 0.3`` (LC-DFTB, omega = 0.3 / Bohr);
* ``ptbp_hcocu_cam``: H, C, O; ``CAM 0.3 0.19 0.46`` (DFTB+'s CAM test values);
* ``ptbp_hcocu_global``: H, C, O; ``CAM 0.0 0.2 0.0`` (a global hybrid, beta = 0).

The integrals stay those of the semi-local parametrisation: for testing the code against DFTB+
only, not a physical parametrisation.  Deterministic (gzip without a time stamp); run
``python -m pydftb.validation.make_hybrid_sets`` to rebuild.  It writes into the package's
``tests/data/skf``: run it in a git checkout, not in an installed copy.
"""
import gzip
import hashlib
import os

from .cases import DATA

SETS = {
    "ptbp_hcocu_lc": ("LC 0.3", ("H", "C", "O", "Cu")),
    "ptbp_hcocu_cam": ("CAM 0.3 0.19 0.46", ("H", "C", "O")),
    "ptbp_hcocu_global": ("CAM 0.0 0.2 0.0", ("H", "C", "O")),
}


def with_range_sep(text: str, line: str) -> str:
    """``text`` with ``RangeSep`` and ``line`` before ``<Documentation>`` (or at the end)."""
    section = f"RangeSep\n{line}\n"
    i = text.find("<Documentation>")
    if i < 0:
        return text.rstrip("\n") + "\n" + section
    return text[:i] + section + text[i:]


def main():
    src = os.path.join(DATA, "skf", "ptbp_hcocu")
    for name, (line, elements) in SETS.items():
        dst = os.path.join(DATA, "skf", name)
        os.makedirs(dst, exist_ok=True)
        for a in elements:
            for b in elements:
                with gzip.open(os.path.join(src, f"{a}-{b}.skf.gz"), "rt") as fh:
                    text = with_range_sep(fh.read(), line)
                with gzip.GzipFile(os.path.join(dst, f"{a}-{b}.skf.gz"), "wb", mtime=0) as gz:
                    gz.write(text.encode())
                print(f"{hashlib.sha256(text.encode()).hexdigest()}  {name}/{a}-{b}.skf")
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
