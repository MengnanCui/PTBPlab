"""Build a small multi-formula dataset for the reaction-mode example by
extracting the B2N2 and N3Na structures from the bundled nitrides.xyz.

22 structures (11 + 11), 3 elements (B, N, Na). Tractable test case for
hotcent SKF generation + DFTB+ on a multi-element system.
"""
from pathlib import Path

import ase.io


HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "dft" / "nitrides.xyz"
DST_DIR = HERE / "dft"
DST_DIR.mkdir(exist_ok=True)


def main() -> None:
    keep = {"B2N2", "N3Na"}
    nit = ase.io.read(str(SRC), ":")
    sub = [a for a in nit if a.get_chemical_formula() in keep]
    ase.io.write(str(DST_DIR / "dft.xyz"), sub)
    print(f"Wrote {len(sub)} structures to {DST_DIR / 'dft.xyz'}")
    for f in keep:
        n = sum(1 for a in sub if a.get_chemical_formula() == f)
        print(f"  {f}: {n} structures")


if __name__ == "__main__":
    main()
