"""Create the ``orbit_poly`` test set: test/orbit H/O files with non-zero
polynomial repulsive coefficients, to validate ``repulsive="polynomial"``
(DFTB+ ``PolynomialRepulsive = SetForAll { Yes }``).

Deterministic; run ``python -m pydftb.validation.make_poly_set`` to rebuild.  It writes into
the package's ``tests/data/skf``: run it in a git checkout, not in an installed copy.
"""
import gzip
import os

from .cases import DATA

# c2..c9 and rcut (Bohr) per pair; smooth, positive, decaying to zero at rcut
POLY = {
    ("H", "H"): ([0.12, -0.04, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0], 2.6),
    ("H", "O"): ([0.25, -0.08, 0.015, -0.002, 0.0, 0.0, 0.0, 0.0], 3.2),
    ("O", "H"): ([0.25, -0.08, 0.015, -0.002, 0.0, 0.0, 0.0, 0.0], 3.2),
    ("O", "O"): ([0.40, -0.10, 0.02, 0.0, 0.001, 0.0, 0.0, 0.0], 3.8),
}


def main():
    src = os.path.join(DATA, "skf", "orbit")
    dst = os.path.join(DATA, "skf", "orbit_poly")
    os.makedirs(dst, exist_ok=True)
    for (a, b), (c, rcut) in POLY.items():
        with gzip.open(os.path.join(src, f"{a}-{b}.skf.gz"), "rt") as fh:
            lines = fh.read().splitlines(keepends=True)
        k = 2 if a == b else 1                     # mass/polynomial line
        mass = lines[k].replace(",", " ").split()[0]
        coeffs = " ".join(repr(x) for x in c)
        lines[k] = f"{mass} {coeffs} {rcut!r} 10*0.0\n"
        with gzip.GzipFile(os.path.join(dst, f"{a}-{b}.skf.gz"), "wb", mtime=0) as gz:
            gz.write("".join(lines).encode())
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
