"""Structures of the Pb–Sn perovskite ink data set (A series, A1.3): molecules, MI2(L)n complexes,
neutral ion clusters and bond / proton-transfer scans, all neutral and boxed.

    python prepare_structures.py [--out data] [--perovskite ../../../../Perovskite]

The inputs come from the ../Perovskite project (stage 1 of its tests):

* **Molecules.**  These are ``data/stage1/molecule_list.csv`` with their files.  The
  MACE-OMOL relaxed geometry (``molecules/mace_omol``) is used when it exists, else the starting
  one (``molecules/init``).
* **Complexes.**  These are ``data/stage1/complex_list.csv``.  The geometry is relaxed with
  MACE-OMOL (``mace_omol_complexes``), else with MACE-MPA, else the starting one.
* **Proton-transfer starts.**  ``molecules/proton_transfer/FA_{DMSO,DMF,I}_HB_mace_omol.xyz``
  are the H-bonded FA+.X pairs.

The script writes these files to ``--out``:

* ``structures.xyz``: extxyz, every structure in a cubic box with >= 8 Å of vacuum, pbc False.
  ``info`` holds name (unique), family, role, kspacing "gamma", charge 0, source and formula;
* ``manifest.csv``: one row per structure;
* ``exclusions.csv``: the inputs left out and why (charged species, elements outside the model).

The families do not clash with pydftb.assess's ``eos_``/``dimer_`` prefixes:

| family | contents | role |
|---|---|---|
| ``mol_<id>_<name>`` | the equilibrium molecule plus ``--rattle-molecules`` rattled copies | train; the optional solvents R15–R20 are test |
| ``cplx_<M>I2_<L><n>`` | three orientations of one complex | train for n = 1, 2; test for n = 3 |
| ``ion_<name>`` | FA+·I−, CsI, FA+·I−·DMSO, FA+·I−·DMF, PbI3−·FA+, SnI3−·FA+, PbI4²−·2FA+, SnI4²−·2FA+ | the last two are test |
| ``scan_<kind>`` | 9 points each: Pb–O and Sn–O (MI2(DMSO), the ligand moved rigidly), Pb–I and Sn–I (MI2), FA+ N–H → I− (in FA+·I−) and FA+ N–H → O(DMSO) (in FA+·I−·DMSO) | train, except the N–H → O(DMSO) scan, which is test |

The charged species (FA+, Cs+, I−, PbI3−, SnI3−) never appear alone: VASP and DFTB compare
neutral structures only.  The ion clusters are rough starting geometries (an N–H···I contact of
2.6 Å); make_d4.py also writes relaxation inputs for them.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import PEROVSKITE, setup  # noqa: E402

setup()
from vasp_ink import boxed  # noqa: E402

MODEL_ELEMENTS = ["H", "C", "N", "O", "S", "I", "Cs", "Sn", "Pb"]
OPTIONAL_TEST = {"R15", "R16", "R17", "R18", "R19", "R20"}    # alternative solvents: test
ION_PAIR_MOLECULES = {"R10": "ion_fa_i", "R11": "ion_csi"}   # neutral pairs, kept as ion_*
NH, OH, HI = 1.02, 0.98, 1.62                                   # bond lengths (Å) of the scans


# ---- geometry helpers ------------------------------------------------------------------------
def rot_to(a, b):
    """The rotation matrix that turns direction a into direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-10:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0.0, 0.0])                 # 180 degrees about any normal
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0.0, 1.0, 0.0])
        p /= np.linalg.norm(p)
        return 2.0 * np.outer(p, p) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def axis_rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def nh_pairs(atoms):
    """(N, H) index pairs of the N–H bonds (< 1.3 Å)."""
    s = np.array(atoms.get_chemical_symbols())
    out = []
    for n in np.where(s == "N")[0]:
        for h in np.where(s == "H")[0]:
            if atoms.get_distance(n, h) < 1.3:
                out.append((int(n), int(h)))
    return out


def attach_cation(anion, cation, iodide: int, contact: float = 2.6, avoid=None):
    """``cation`` (FA+) placed so that one of its N–H points at atom ``iodide`` of ``anion``,
    along the direction away from the anion's centre, with H···I = ``contact``.  The rotation
    about that line keeps the cation farthest from the other atoms."""
    from ase import Atoms
    centre = anion.positions.mean(axis=0) if avoid is None else avoid
    v = anion.positions[iodide] - centre
    if np.linalg.norm(v) < 1e-6:
        v = np.array([0.0, 0.0, 1.0])
    v /= np.linalg.norm(v)
    target = anion.positions[iodide] + contact * v
    best = None
    for n, h in nh_pairs(cation):
        base = (cation.positions - cation.positions[h]) @ rot_to(
            cation.positions[h] - cation.positions[n], -v).T
        for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            pos = base @ axis_rotation(v, ang).T + target
            others = np.delete(pos, h, axis=0)
            dmin = float(np.min(np.linalg.norm(others[:, None] - anion.positions[None], axis=-1)))
            if best is None or dmin > best[0]:
                best = (dmin, pos)
    if best is None:
        raise ValueError("the cation has no N-H bond")
    return anion + Atoms(cation.get_chemical_symbols(), positions=best[1])


def add_iodide_opposite(anion, metal: int, distance: float):
    """MI3- -> MI4(2-): a fourth I opposite the mean direction of the three M–I bonds."""
    from ase import Atoms
    s = np.array(anion.get_chemical_symbols())
    iod = np.where(s == "I")[0]
    u = sum((anion.positions[i] - anion.positions[metal]) /
            np.linalg.norm(anion.positions[i] - anion.positions[metal]) for i in iod)
    u = -u / np.linalg.norm(u)
    return anion + Atoms("I", positions=[anion.positions[metal] + distance * u])


def rattle(atoms, sigma: float, seed: int):
    a = atoms.copy()
    a.positions = a.positions + np.random.default_rng(seed).normal(0.0, sigma, a.positions.shape)
    return a


# ---- inputs ----------------------------------------------------------------------------------
def read_xyz(path):
    import ase.io
    a = ase.io.read(path)
    a.calc = None
    a.info = {}
    a.pbc = False
    return a


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def molecule_file(root, rel_file):
    base = os.path.basename(rel_file)
    return first_existing(os.path.join(root, "molecules", "mace_omol", base),
                          os.path.join(root, rel_file))


def complex_file(root, rel_file):
    base = os.path.basename(rel_file)
    return first_existing(os.path.join(root, "molecules", "mace_omol_complexes", base),
                          os.path.join(root, "molecules", "mace_mpa_complexes", base),
                          os.path.join(root, rel_file))


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---- the data set -----------------------------------------------------------------------------
class Collector:
    def __init__(self, vacuum):
        self.frames, self.rows, self.excluded, self.vacuum = [], [], [], vacuum

    def add(self, atoms, name, family, role, source, note=""):
        if any(f.info["name"] == name for f in self.frames):
            raise ValueError(f"duplicate structure name {name}")
        a = boxed(atoms, self.vacuum)
        a.info = {"name": name, "family": family, "role": role, "kspacing": "gamma", "charge": 0,
                  "source": source, "formula": a.get_chemical_formula()}
        self.frames.append(a)
        self.rows.append({"name": name, "family": family, "role": role, "natoms": len(a),
                          "formula": a.get_chemical_formula(), "box_A": round(a.cell[0, 0], 2),
                          "source": source, "note": note})
        return a

    def exclude(self, what, reason):
        self.excluded.append({"input": what, "reason": reason})


def molecules(col, root, args):
    rows = read_csv(os.path.join(root, "data", "stage1", "molecule_list.csv"))
    out = {}
    for r in rows:
        mid, name, q = r["id"], r["name"], int(r["charge"])
        path = molecule_file(root, r["file"])
        if path is None:
            col.exclude(f"{mid} {name}", f"no file {r['file']}")
            continue
        a = read_xyz(path)
        out[mid] = a                       # charged ones are building blocks of the ion clusters
        if q != 0:
            col.exclude(f"{mid} {name}", f"charge {q}: only in neutral ion clusters")
            continue
        bad = sorted(set(a.get_chemical_symbols()) - set(args.elements))
        if bad:
            col.exclude(f"{mid} {name}", f"elements {bad} outside the model {args.elements}")
            continue
        if mid in ION_PAIR_MOLECULES:
            continue                       # FA+.I- and CsI go to ion_* below
        fam = f"mol_{mid}_{_safe(name)}"
        role = "test" if mid in OPTIONAL_TEST else "train"
        src = os.path.relpath(path, root)
        col.add(a, fam, fam, role, src)
        for k in range(1, args.rattle_molecules + 1):
            col.add(rattle(a, args.rattle_sigma, 1000 * k + len(col.frames)), f"{fam}_r{k}", fam,
                    role, src, f"rattled {args.rattle_sigma} A")
    return out


def _safe(name):
    return name.replace("+", "p").replace("-", "m").replace("(", "").replace(")", "").replace(" ", "")


def complexes(col, root, args):
    rows = read_csv(os.path.join(root, "data", "stage1", "complex_list.csv"))
    out = {}
    for r in rows:
        path = complex_file(root, r["file"])
        if path is None:
            col.exclude(r["name"], f"no file {r['file']}")
            continue
        a = read_xyz(path)
        out[r["name"]] = a
        n = int(r["n"])
        fam = f"cplx_{r['metal']}I2_{r['ligand']}{n}"
        role = "test" if n == 3 else "train"
        src = os.path.relpath(path, root)
        col.add(a, f"cplx_{r['name']}", fam, role, src)
        for k in range(1, args.rattle_complexes + 1):
            col.add(rattle(a, args.rattle_sigma, 2000 * k + len(col.frames)),
                    f"cplx_{r['name']}_r{k}", fam, role, src, f"rattled {args.rattle_sigma} A")
    return out


def ion_clusters(col, root, mols, args):
    """Neutral ion pairs and clusters (see the module docstring); returns them by name."""
    pt = os.path.join(root, "molecules", "proton_transfer")
    fa = mols.get("R7")
    built = {}
    if "R10" in mols:
        built["ion_fa_i"] = (mols["R10"], "molecules/mace_omol/R10_FAI.xyz")
    if "R11" in mols:
        built["ion_csi"] = (mols["R11"], "molecules/mace_omol/R11_CsI.xyz")
    iodine = mols.get("R9")
    for solv in ("DMSO", "DMF"):
        p = first_existing(os.path.join(pt, f"FA_{solv}_HB_mace_omol.xyz"),
                           os.path.join(pt, f"FA_{solv}_HB_init.xyz"))
        if p is None or iodine is None:
            col.exclude(f"ion_fa_i_{solv.lower()}", "no FA+.solvent H-bonded geometry")
            continue
        pair = read_xyz(p)
        built[f"ion_fa_i_{solv.lower()}"] = (_add_iodide_to_free_nh(pair), os.path.relpath(p, root)
                                             + " + I- on the free N-H")
    for metal, mid in (("Pb", "R21a"), ("Sn", "R21b")):
        if mid not in mols or fa is None:
            col.exclude(f"ion_{metal.lower()}i3_fa", f"no {mid} or FA+ geometry")
            continue
        anion = mols[mid]
        s = np.array(anion.get_chemical_symbols())
        m = int(np.where(s == metal)[0][0])
        iod = [int(i) for i in np.where(s == "I")[0]]
        one = attach_cation(anion, fa, iod[0], avoid=anion.positions[m])
        built[f"ion_{metal.lower()}i3_fa"] = (one, f"molecules/mace_omol/{mid} + FA+ (built)")
        d_mi = float(np.mean([anion.get_distance(m, i) for i in iod])) + 0.15
        four = add_iodide_opposite(anion, m, d_mi)
        i4 = len(four) - 1
        two = attach_cation(four, fa, iod[0], avoid=four.positions[m])
        two = attach_cation(two, fa, i4, avoid=four.positions[m])
        built[f"ion_{metal.lower()}i4_2fa"] = (two, f"molecules/mace_omol/{mid} + I- + 2 FA+ (built)")
    for name, (a, src) in built.items():
        role = "test" if name.endswith("i4_2fa") else "train"
        col.add(a, name, name, role, src)
        for k in range(1, args.rattle_ions + 1):
            col.add(rattle(a, args.rattle_sigma, 3000 * k + len(col.frames)), f"{name}_r{k}", name,
                    role, src, f"rattled {args.rattle_sigma} A")
    return {k: v[0] for k, v in built.items()}


def _add_iodide_to_free_nh(pair, contact: float = 2.6):
    """FA+.solvent (+1) -> FA+.I-.solvent (0): the I- at the N–H farthest from the solvent."""
    from ase import Atoms
    # FA+ is the first 8 atoms in the stage-3 geometries (proton_transfer.py builds FA+ + solvent)
    fa_idx = set(range(8))
    solvent = [i for i in range(len(pair)) if i not in fa_idx]
    best = None
    for n, h in nh_pairs(pair):
        if n not in fa_idx:
            continue
        u = pair.positions[h] - pair.positions[n]
        u /= np.linalg.norm(u)
        x = pair.positions[h] + contact * u
        d = float(np.min(np.linalg.norm(pair.positions[solvent] - x, axis=1))) if solvent else 9.9
        if best is None or d > best[0]:
            best = (d, x)
    return pair + Atoms("I", positions=[best[1]])


def scans(col, root, mols, cplx, ions, args):
    """Six 9-point scans (see the module docstring)."""
    out = []

    def add_scan(kind, frames, role, src, what):
        fam = f"scan_{kind}"
        for k, (a, val) in enumerate(frames):
            col.add(a, f"{fam}_{k:02d}", fam, role, src, f"{what} = {val:.3f} A")  # scan coordinate
        out.append(fam)

    # M–O: the ligand of MI2(DMSO) moved rigidly along M -> O
    for metal in ("Pb", "Sn"):
        name = f"{metal}I2_DMSO1_o0"
        if name not in cplx:
            col.exclude(f"scan_{metal.lower()}o", f"no {name}")
            continue
        a = cplx[name]
        s = np.array(a.get_chemical_symbols())
        m = int(np.where(s == metal)[0][0])
        lig = [i for i in range(len(a)) if s[i] not in (metal, "I")]
        o = min((i for i in lig if s[i] == "O"), key=lambda i: a.get_distance(m, i))
        d0 = a.get_distance(m, o)
        u = (a.positions[o] - a.positions[m]) / d0
        frames = []
        for dd in np.linspace(-0.35, 2.0, 9):
            b = a.copy()
            b.positions[lig] += dd * u
            frames.append((b, d0 + dd))
        add_scan(f"{metal.lower()}o", frames, "train", f"cplx_{name}", f"d({metal}-O)")
    # M–I: one iodine of MI2 moved along M -> I
    for metal, mid, lo, hi in (("Pb", "R1", 2.55, 4.6), ("Sn", "R2", 2.45, 4.5)):
        if mid not in mols:
            col.exclude(f"scan_{metal.lower()}i", f"no {mid}")
            continue
        a = mols[mid]
        s = np.array(a.get_chemical_symbols())
        m = int(np.where(s == metal)[0][0])
        i = int(np.where(s == "I")[0][0])
        u = a.positions[i] - a.positions[m]
        u /= np.linalg.norm(u)
        frames = []
        for d in np.linspace(lo, hi, 9):
            b = a.copy()
            b.positions[i] = b.positions[m] + d * u
            frames.append((b, d))
        add_scan(f"{metal.lower()}i", frames, "train", f"mol_{mid}", f"d({metal}-I)")
    # proton transfer: H moved on the N...X line, N and X fixed
    for kind, cluster, acceptor, x_h, role in (("pt_fa_i", "ion_fa_i", "I", HI, "train"),
                                               ("pt_fa_o", "ion_fa_i_dmso", "O", OH, "test")):
        if cluster not in ions:
            col.exclude(f"scan_{kind}", f"no {cluster}")
            continue
        a = ions[cluster]
        s = np.array(a.get_chemical_symbols())
        acc = [int(i) for i in np.where(s == acceptor)[0]]
        best = None
        for n, h in nh_pairs(a):
            for x in acc:
                d = a.get_distance(h, x)
                if best is None or d < best[0]:
                    best = (d, n, h, x)
        _, n, h, x = best
        d_nx = a.get_distance(n, x)
        u = (a.positions[x] - a.positions[n]) / d_nx
        end = max(NH + 0.3, d_nx - x_h)
        frames = []
        for d in np.linspace(NH, end, 9):
            b = a.copy()
            b.positions[h] = b.positions[n] + d * u
            frames.append((b, d))
        add_scan(kind, frames, role, cluster, f"d(N...{acceptor}) = {d_nx:.2f} A fixed; d(N-H)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--perovskite", default=PEROVSKITE, help="the ../Perovskite project")
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--elements", nargs="+", default=MODEL_ELEMENTS)
    ap.add_argument("--vacuum", type=float, default=8.0, help="A on every side")
    ap.add_argument("--rattle-molecules", type=int, default=2)
    ap.add_argument("--rattle-complexes", type=int, default=0)
    ap.add_argument("--rattle-ions", type=int, default=1)
    ap.add_argument("--rattle-sigma", type=float, default=0.05, help="A")
    ap.add_argument("--only", nargs="*", default=None, help="keep families matching these patterns")
    args = ap.parse_args(argv)
    from utils.runlog import session
    os.makedirs(args.out, exist_ok=True)
    with session("prepare_structures", vars(args), root=os.path.join(args.out, "logs")) as log:
        root = os.path.abspath(args.perovskite)
        if not os.path.isdir(os.path.join(root, "molecules")):
            print(f"no Perovskite project at {root} (set --perovskite or PTBP_PEROVSKITE)")
            return 2
        col = Collector(args.vacuum)
        mols = molecules(col, root, args)
        cplx = complexes(col, root, args)
        ions = ion_clusters(col, root, mols, args)
        fams = scans(col, root, mols, cplx, ions, args)
        if args.only:
            keep = [k for k, f in enumerate(col.frames)
                    if any(fnmatch.fnmatch(f.info["family"], p) for p in args.only)]
            col.frames = [col.frames[k] for k in keep]
            col.rows = [col.rows[k] for k in keep]
        import ase.io
        path = os.path.join(args.out, "structures.xyz")
        ase.io.write(path, col.frames, format="extxyz")
        for fn, rows in (("manifest.csv", col.rows), ("exclusions.csv", col.excluded)):
            with open(os.path.join(args.out, fn), "w", newline="") as fh:
                if rows:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
        roles = {r: sum(1 for x in col.rows if x["role"] == r) for r in ("train", "test", "ood")}
        log.event("written", structures=len(col.frames), path=path, scans=len(fams), **roles,
                  excluded=len(col.excluded), max_atoms=max(x["natoms"] for x in col.rows))
        for e in col.excluded:
            print(f"  excluded {e['input']}: {e['reason']}")
        print(f"{len(col.frames)} structures -> {path} ({roles})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
