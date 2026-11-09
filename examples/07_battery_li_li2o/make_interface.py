"""The Li | Li2O interface of the M5c demo: a periodic superlattice (no vacuum, two interfaces)
of Li2O(100) in a sqrt(2) x sqrt(2) R45 cell and bcc Li(100) 2 x 2, 80 atoms.

* Li2O: a = the relaxed VASP value; the block is 2 conventional cells thick (planes O, Li, O,
  Li, ... from z = 0 to 7a/4: 16 O + 32 Li), so one interface is O-terminated and the other
  Li-terminated;
* Li: 8 (100) layers of 4 atoms; in-plane the Li is compressed to the Li2O cell
  (2 a_Li -> sqrt(2) a_Li2O, -4.7 %), its layer spacing stretched by 1 / 0.953^2 to keep the
  volume per atom;
* gaps: first Li layer 1.9 A above the Li plane that ends Li2O, the next Li2O block's O plane
  1.6 A above the last Li layer (Li over the O plane's hollow sites, as in the bulk).

    python make_interface.py A_LI2O A_LI OUT.vasp
"""
import sys

import numpy as np
from ase import Atoms


def interface(a_ox: float, a_li: float) -> Atoms:
    L = np.sqrt(2.0) * a_ox                           # in-plane edge of the R45 cell
    # Li2O conventional positions (fractional of a): O fcc, Li simple cubic at a/4 offsets
    o_frac = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    li_frac = np.array([[x, y, z] for x in (0.25, 0.75) for y in (0.25, 0.75) for z in (0.25, 0.75)])
    rot = np.array([[1, 1], [-1, 1]]) / 2.0          # conventional (x, y) -> R45 cell fractions
    sym, pos = [], []
    for nz in range(2):
        for el, frac in (("O", o_frac), ("Li", li_frac)):
            for f in frac:
                for sx in range(2):                   # the R45 cell holds two conventional cells
                    xy = (f[:2] + [sx, 0]) @ rot.T
                    xy = np.mod(xy, 1.0) * L
                    pos.append([xy[0], xy[1], (f[2] + nz) * a_ox])
                    sym.append(el)
    pos = np.array(pos)
    # dedupe (the R45 mapping folds equivalent sites together)
    keep, seen = [], set()
    for i, p in enumerate(pos):
        key = (sym[i], *np.round(np.mod(p[:2], L), 3), round(p[2], 3))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    sym = [sym[i] for i in keep]
    pos = pos[keep]
    top = pos[:, 2].max()                              # the last Li plane, 7a/4
    strain = L / (2 * a_li)
    dz = 0.5 * a_li / strain ** 2
    for k in range(8):
        z = top + 1.9 + k * dz
        off = 0.0 if k % 2 == 0 else 0.25 * L          # bcc(100) stacking: alternate layers
        for i in range(2):
            for j in range(2):
                x = (i * 0.5 + off / L) * L
                y = (j * 0.5 + off / L) * L
                pos = np.vstack([pos, [x, y, z]])
                sym.append("Li")
    c = pos[:, 2].max() + 1.6
    at = Atoms(sym, positions=pos, cell=[L, L, c], pbc=True)
    at.wrap()
    return at


if __name__ == "__main__":
    a_ox, a_li, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    at = interface(a_ox, a_li)
    print(at.get_chemical_formula(), len(at), "cell", np.round(at.cell.lengths(), 3),
          f"Li strain {np.sqrt(2) * a_ox / (2 * a_li) - 1:+.3%}")
    at.write(out, format="vasp", direct=True, sort=True)
