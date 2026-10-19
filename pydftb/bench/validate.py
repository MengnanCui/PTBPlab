"""Validate pydftb against DFTB+ on molecules, d-metals and periodic systems."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ase.build import bulk, molecule
from pydftb.skf import SKSet
from pydftb.calculator import DFTB
from pydftb.constants import BOHR
import dftbplus_ref as ref

SK = os.environ.get("SKDIR", "/tmp/claude-0/skf2")
MAXL = {"H": 0, "C": 1, "O": 1, "Cu": 2}


def mp_grid(n):
    """Gamma-centred-free Monkhorst-Pack grid (fractional), full BZ."""
    g = [(np.arange(k) + 0.5) / k - 0.5 for k in n]
    k = np.array(np.meshgrid(*g, indexing="ij")).reshape(3, -1).T
    return k, np.full(len(k), 1.0 / len(k))


def compare(name, atoms, kpts=None, temperature=0.0, shell_resolved=False):
    sym = atoms.get_chemical_symbols()
    el = sorted(set(sym))
    sk = SKSet(SK, el, {e: MAXL[e] for e in el}, shell_resolved=shell_resolved)
    periodic = atoms.pbc.any()
    calc = DFTB(sk, scc=True, scc_tol=1e-10, kpts=kpts, temperature=temperature)
    r = calc.calculate(sym, atoms.positions / BOHR,
                       atoms.cell.array / BOHR if periodic else None,
                       atoms.pbc if periodic else None)
    d = ref.run(sym, atoms.positions, atoms.cell.array if periodic else None, skdir=SK,
                max_l={e: MAXL[e] for e in el}, kpts=kpts, temperature=temperature,
                shell_resolved=shell_resolved)
    de = r.free_energy - d["free_energy"]
    df = np.abs(r.forces - d["forces"]).max()
    dq = np.abs(r.charges - d["charges"]).max()
    print(f"{name:28s} N={len(atoms):4d}  dE={de:+.2e} Ha  max|dF|={df:.2e} Ha/Bohr  "
          f"max|dq|={dq:.2e}  nSCC={r.n_scc}  t_py={r.timings['total']:.2f}s  t_dftb+={d['wall']:.2f}s")
    return de, df, dq


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows = []
    h2o = molecule("H2O"); h2o.rattle(0.05, seed=1)
    rows.append(compare("H2O", h2o))
    meoh = molecule("CH3OH"); meoh.rattle(0.05, seed=2)
    rows.append(compare("CH3OH", meoh))
    cu = bulk("Cu", "fcc", a=3.6, cubic=True)[:4]
    cu = cu.repeat((1, 1, 1)); cu.center(vacuum=0.0)
    from ase import Atoms
    clus = Atoms("Cu4CO", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.1, 0], [1.25, 0.7, 2.0],
                                      [1.25, 0.7, 3.9], [1.25, 0.75, 5.05]])
    clus.rattle(0.03, seed=3)
    rows.append(compare("Cu4-CO cluster (d)", clus, temperature=300))
    rows.append(compare("Cu4-CO shell-resolved", clus, temperature=300, shell_resolved=True))
    cub = bulk("Cu", "fcc", a=3.61) ; cub.rattle(0.0)
    rows.append(compare("Cu fcc prim, 6x6x6 k", cub, kpts=mp_grid((6, 6, 6)), temperature=1000))
    cus = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((2, 1, 1)); cus.rattle(0.05, seed=4)
    cus.set_cell(cus.cell.array @ np.array([[1, 0.03, 0], [0, 1, 0.02], [0.01, 0, 1]]), scale_atoms=True)
    rows.append(compare("Cu 8-atom distorted, 3x5x5 k", cus, kpts=mp_grid((3, 5, 5)), temperature=1000))
    w = molecule("H2O")
    from ase.build import make_supercell
    box = Atoms(cell=[6.2, 6.2, 6.2], pbc=True)
    for k in range(8):
        m = w.copy(); m.rotate(rng.uniform(0, 360), rng.normal(size=3)); m.translate(rng.uniform(0, 6.2, 3))
        box += m
    rows.append(compare("water box 24 atoms, Gamma", box))
    co2 = Atoms("CuOCuO", positions=[[0, 0, 0], [1.9, 0.1, 0], [3.8, 0, 0.1], [5.7, 0, 0]], cell=[7.6, 4, 4], pbc=True)
    co2.rattle(0.02, seed=5)
    rows.append(compare("CuO chain, 4x2x2 k", co2, kpts=mp_grid((4, 2, 2)), temperature=500))
    rows = np.array(rows)
    print("worst: dE=%.2e  dF=%.2e  dq=%.2e" % tuple(np.abs(rows).max(0)))
