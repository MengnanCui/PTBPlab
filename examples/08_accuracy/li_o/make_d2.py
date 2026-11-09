"""D2 (A series, A0 section 2): Li and Li-O structures the M5c/M5d data lack, as VASP single
points (written, never submitted).

About 80 single points, every one unrelaxed or rattled (no relaxations: they would cost ~20
single points each).  The settings are those of the training data (examples/07 vasp_common.BASE:
PBE, Li_sv/O, ENCUT 800, KSPACING 0.15, ISMEAR 0, SIGMA 0.05, ISPIN 1).  Rattled and defect
cells get ISYM 0.  Lattice constants come from the relaxed cells, i.e. the 1.00 V0 EOS points
of data/train.xyz.

| family                   | role  | what                                                          |
|--------------------------|-------|---------------------------------------------------------------|
| eos_li_hcp               | ood   | 11 volumes 0.88-1.12 V0, hcp at the bcc V0 per atom, c/a 1.633 |
| rat_li_hcp16             | ood   | hcp 2x2x2 (16 atoms), sigma 0.10 A, seeds 1-3                 |
| surf_li_100/110/111      | ood   | bcc Li slabs (1x1), 2 thicknesses, 12 A vacuum, ideal + rattled |
| vac_li, bulk_li54        | ood   | bcc 3x3x3 (54 atoms) with one vacancy: ideal + 2 rattled; perfect: ideal + 1 rattled |
| surf_li2o_111/110        | ood   | non-polar Li2O slabs (1x1), 2 thicknesses, ideal + rattled    |
| def_li2o_vli/ili/vo      | ood   | 24-atom Li2O (2x2x2 primitive): Li vacancy, Li interstitial (octahedral), O vacancy; ideal + 2 rattled |
| rat_li2o2                | train/test | Li2O2 (8 atoms), sigma 0.05/0.10, seeds 1-3 (train), 101 (test) |
| rat_hot_li_bcc16/li2o12  | train/test | sigma 0.15/0.20, seeds 1-3 (train), sigma 0.15 seed 101 (test) |
| rat_hot_li2o2            | train | sigma 0.15, seeds 1-3                                         |
| rat_cmp_li_bcc16/li2o12  | train | 0.90 V0, sigma 0.10, seeds 1-3                                |

    python make_d2.py write --out ../runs/d2_vasp [--ncores 76] [--budget 40] [--jobs 2] [--dry-run]
    # after the runs, on the cluster or a copy of them:
    python ../../07_battery_li_li2o/collect.py ../runs/d2_vasp --out ../runs/d2_collected
    python make_d2.py retag --manifest ../runs/d2_vasp/manifest.csv \\
        --frames ../runs/d2_collected/train.xyz ../runs/d2_collected/test.xyz --out ../data/d2.xyz

collect.py names its groups by prefix, so the new families come out as "other", and it splits
train/test by name; ``retag`` sets name, family, role, kspacing and source from manifest.csv.
This module also holds the helpers that vasp_bands.py and make_d3.py import: the cost model,
the run script, the job scripts and ``retag_frames``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import EX07_DATA, setup  # noqa: E402

setup()

YUAN_PER_CORE_H = 0.1            # dg83 list price (billed about 0.075 in M5c/M5d)
KSPACING = 0.15

# ---- shared helpers (vasp_bands.py, make_d3.py) ----------------------------------------
RUN_DIRS_SH = r"""#!/bin/bash
# Run VASP in each directory given (default: every subdirectory with INCAR and POSCAR), one
# after another, with $SLURM_NTASKS ranks.  POTCAR is concatenated from the names in POTCARS
# (same order as POSCAR line 6).  Finished directories (OUTCAR with "General timing") are
# skipped, so a resubmission continues where the last one stopped.  A directory "band" next
# to "scf" (vasp_bands.py) gets the SCF's CHGCAR first.
# Paths for another machine: VASP_MODULE, VASP_BIN, VASP_PP (potpaw_PBE.54 directory).
module purge 2>/dev/null
module load ${VASP_MODULE:-intel/2022.1} 2>/dev/null
export PATH=${VASP_BIN:-/data/home/dg002434/software/vasp.6.6.0/bin}:$PATH
PP=${VASP_PP:-/data/home/dg002434/software/potpaw_PBE.54}
NP=${SLURM_NTASKS:-4}
run_one() {
  local d=$1
  [ -f "$d/INCAR" ] && [ -f "$d/POSCAR" ] || return 0
  if grep -qs "General timing" "$d/OUTCAR"; then echo "skip $d"; return 0; fi
  ( cd "$d" || exit
    cat $(for p in $(cat POTCARS); do echo $PP/$p/POTCAR; done) > POTCAR
    exe=vasp_std; [ -f USE_GAM ] && exe=vasp_gam
    t0=$(date +%s)
    mpirun -np $NP $exe > vasp.log 2>&1
    echo "done $d $(( $(date +%s) - t0 )) s $(grep -c 'F=' OSZICAR 2>/dev/null) ionic"
  )
}
dirs=("$@"); [ ${#dirs[@]} -eq 0 ] && dirs=(*/)
for d in "${dirs[@]}"; do
  d=${d%/}
  if [ -d "$d/scf" ]; then
    run_one "$d/scf"
    if [ -d "$d/band" ]; then
      [ -f "$d/scf/CHGCAR" ] && cp "$d/scf/CHGCAR" "$d/band/CHGCAR"
      run_one "$d/band"
    fi
  else
    run_one "$d"
  fi
done
"""


def kgrid(atoms, kspacing: float) -> list:
    """VASP's KSPACING rule, Gamma-centred: N_i = max(1, ceil(2 pi |b_i| / kspacing))."""
    if kspacing in (None, "gamma"):
        return [1, 1, 1]
    rec = atoms.cell.reciprocal()
    return [max(1, int(math.ceil(2 * math.pi * np.linalg.norm(b) / float(kspacing) - 1e-9)))
            for b in rec]


def nk_time_reversal(grid) -> int:
    n = int(np.prod(grid))
    return max(1, (n + 1) // 2)       # no point-group symmetry: conservative for ideal cells


def core_hours(atoms, nk: int) -> float:
    """Rough VASP cost at ENCUT 800 (core-h): 0.1 h overhead plus 4.3 core-h x (nk / 172) x
    (N / 16)^1.3 x (V / 325 A^3), calibrated on the rattled bcc Li16 of M5c (16 atoms, 325 A^3,
    172 k-points after time reversal, ~4.3 core-h).  Bulk cells scale as N^2.3, vacuum as V.
    Good to a factor of about 2."""
    n, v = len(atoms), float(atoms.get_volume())
    return 0.1 + 4.3 * (nk / 172.0) * (n / 16.0) ** 1.3 * (v / 325.0)


def write_jobs(out: str, dirs: list, name: str, ncores: int, budget: float, jobs: int = 1,
               est=None) -> list:
    """job_XX.sh (Slurm, dg83) over ``dirs`` split in ``jobs`` parts, each with the wall-time
    cap of its share of ``budget`` (yuan at the list price), run_dirs.sh and submit_all.sh.
    Nothing is submitted."""
    with open(os.path.join(out, "run_dirs.sh"), "w") as fh:
        fh.write(RUN_DIRS_SH)
    os.chmod(os.path.join(out, "run_dirs.sh"), 0o755)
    jobs = max(1, min(int(jobs), len(dirs)))
    parts = [dirs[i::jobs] for i in range(jobs)]
    scripts = []
    for k, part in enumerate(parts):
        minutes = max(10, int(budget / jobs / (YUAN_PER_CORE_H * ncores) * 60))
        path = os.path.join(out, f"job_{k:02d}.sh")
        with open(path, "w") as fh:
            fh.write(f"""#!/bin/bash
#SBATCH -J {name}_{k:02d}
#SBATCH -p dg83
#SBATCH -N 1 -n {ncores}
#SBATCH -t {minutes // 60:02d}:{minutes % 60:02d}:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
# written by examples/08_accuracy/li_o (A series); wall-time cap from a budget of
# {budget / jobs:.2f} yuan at {YUAN_PER_CORE_H} yuan/core-h; rerun to continue unfinished dirs
cd "$SLURM_SUBMIT_DIR"
bash run_dirs.sh {' '.join(part)}
""")
        scripts.append(os.path.basename(path))
    with open(os.path.join(out, "submit_all.sh"), "w") as fh:
        fh.write("#!/bin/bash\n# Not run by any script: submit by hand after checking "
                 "cost_estimate.json\ncd \"$(dirname \"$0\")\"\n"
                 + "".join(f"sbatch {s}\n" for s in scripts))
    os.chmod(os.path.join(out, "submit_all.sh"), 0o755)
    return scripts


def write_manifest(out: str, rows: list) -> str:
    path = os.path.join(out, "manifest.csv")
    keys = ["name", "family", "role", "natoms", "formula", "kspacing", "kgrid", "nk_tr",
            "est_core_h", "source"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    return path


def read_manifest(path: str) -> list:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def retag_frames(frames, rows) -> tuple:
    """Set info name, family, role, kspacing and source of collected frames from manifest rows
    (matched by ``structure_name`` or ``name``); returns (tagged, unmatched names)."""
    by = {r["name"]: r for r in rows}
    out, missing = [], []
    for a in frames:
        name = a.info.get("structure_name") or a.info.get("name")
        r = by.get(name)
        if r is None:
            missing.append(name)
            continue
        ks = r.get("kspacing")
        a.info.update(name=name, structure_name=name, family=r["family"], group=r["family"],
                      role=r["role"], kspacing=("gamma" if ks in ("gamma", "", None) else float(ks)),
                      source=r.get("source", ""))
        out.append(a)
    return out, missing


def write_cost(out: str, rows: list, ncores: int, budget: float, extra: dict | None = None) -> dict:
    tot = float(sum(r["est_core_h"] for r in rows))
    cost = {"n_cases": len(rows), "est_core_h": round(tot, 1),
            "est_yuan_list": round(YUAN_PER_CORE_H * tot, 2),
            "est_yuan_billed_0.075": round(0.075 * tot, 2), "ncores": ncores,
            "budget_yuan_cap": budget, "note": "rough, x0.5-2; never submitted by these scripts"}
    cost.update(extra or {})
    with open(os.path.join(out, "cost_estimate.json"), "w") as fh:
        json.dump(cost, fh, indent=1)
    return cost


# ---- D2 structures ------------------------------------------------------------------------
def equilibrium(data_dir: str = EX07_DATA) -> dict:
    """Relaxed cells and lattice constants from the 1.00 V0 EOS points of train.xyz."""
    import ase.io
    train = {a.info["structure_name"]: a for a in ase.io.read(os.path.join(data_dir, "train.xyz"), ":")}
    bcc = train["eos_li_bcc_05"]
    li2o = train["eos_li2o_05"]
    li2o2 = train["eos_li2o2_05"].copy()
    li2o2.calc = None
    return {"a_bcc": float(bcc.cell.cellpar()[0] * 2 / math.sqrt(3)),
            "a_li2o": float(li2o.cell.cellpar()[0] * math.sqrt(2)),
            "v_bcc_atom": float(bcc.get_volume() / len(bcc)), "li2o2": li2o2}


def hcp_cell(v_atom: float, coa: float = 1.633):
    from ase.build import bulk
    a = (4.0 * v_atom / (math.sqrt(3.0) * coa)) ** (1.0 / 3.0)
    return bulk("Li", "hcp", a=a, c=coa * a)


def li2o_conventional(a: float):
    from ase.build import make_supercell
    from vasp_common import antifluorite
    conv = make_supercell(antifluorite(a), np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]))
    assert len(conv) == 12
    return conv


def _planes(atoms, tol: float = 0.3) -> list:
    """Atom indices of each plane along z, bottom to top."""
    order = np.argsort(atoms.positions[:, 2])
    z = atoms.positions[order, 2]
    planes, cur = [], [order[0]]
    for i in range(1, len(order)):
        if z[i] - z[i - 1] > tol:
            planes.append(cur)
            cur = []
        cur.append(order[i])
    planes.append(cur)
    return planes


def _slab_from_planes(bulk_slab, chosen, vacuum: float):
    from ase import Atoms
    idx = [i for p in chosen for i in p]
    pos = bulk_slab.positions[idx].copy()
    pos[:, 2] -= pos[:, 2].min() - vacuum
    cell = np.array(bulk_slab.cell.array, float)
    thick = pos[:, 2].max() - vacuum
    cell[2] = [0.0, 0.0, thick + 2 * vacuum]
    s = Atoms([bulk_slab.get_chemical_symbols()[i] for i in idx], positions=pos, cell=cell,
              pbc=True)
    s.wrap()
    return s


def li2o_slab(a: float, miller: str, units: int, vacuum: float = 6.0):
    """A stoichiometric, non-polar 1x1 Li2O slab: (111) as ``units`` Li-O-Li trilayers ending in
    Li, (110) as ``units`` neutral Li2O planes; ``vacuum`` on each side."""
    from ase.build import surface
    from vasp_common import antifluorite
    prim = antifluorite(a)
    indices = {"111": (1, 1, 1), "110": (1, 1, 2)}[miller]   # in the primitive fcc basis
    thick = surface(prim, indices, layers=units + 4, vacuum=0.0, periodic=True)
    thick.wrap()
    planes = _planes(thick)
    kinds = ["".join(sorted(set(thick.get_chemical_symbols()[i] for i in p))) for p in planes]
    if miller == "111":
        start = next(i for i in range(1, len(planes) - 3)
                     if kinds[i:i + 3] == ["Li", "O", "Li"])
        chosen = planes[start:start + 3 * units]
        if [kinds[start + j] for j in range(3 * units)] != ["Li", "O", "Li"] * units:
            raise RuntimeError(f"(111) planes are not Li-O-Li trilayers: {kinds}")
    else:
        chosen = planes[1:1 + units]
    s = _slab_from_planes(thick, chosen, vacuum)
    sym = s.get_chemical_symbols()
    if sym.count("Li") != 2 * sym.count("O"):
        raise RuntimeError(f"Li2O({miller}) slab is not stoichiometric: {s.get_chemical_formula()}")
    q = np.array([1.0 if x == "Li" else -2.0 for x in sym])
    dip = float(q @ (s.positions[:, 2] - s.positions[:, 2].mean()))
    if abs(dip) > 1e-6 * len(s):
        raise RuntimeError(f"Li2O({miller}) slab has a dipole {dip:.3g} e A")
    return s


def li_slab(a: float, miller: str, layers: int, vacuum: float = 6.0):
    from ase.build import bcc100, bcc110, bcc111
    f = {"100": bcc100, "110": bcc110, "111": bcc111}[miller]
    s = f("Li", size=(1, 1, layers), a=a, vacuum=vacuum)
    s.pbc = True
    return s


def rattled(atoms, sigma: float, seed: int):
    a = atoms.copy()
    a.rattle(stdev=sigma, seed=seed)
    return a


def d2_cases(data_dir: str = EX07_DATA) -> list:
    """[(name, family, role, atoms, incar overrides, source)] of D2."""
    from ase import Atom
    from ase.build import bulk, make_supercell
    from vasp_common import antifluorite
    eq = equilibrium(data_dir)
    a_bcc, a_ox = eq["a_bcc"], eq["a_li2o"]
    cases = []

    def add(name, family, role, atoms, source, **inc):
        if any(c[0] == name for c in cases):
            raise ValueError(f"duplicate {name}")
        cases.append((name, family, role, atoms, inc, source))

    hcp = hcp_cell(eq["v_bcc_atom"])
    for k, f in enumerate(np.linspace(0.88, 1.12, 11)):
        a = hcp.copy()
        a.set_cell(hcp.cell * f ** (1 / 3), scale_atoms=True)
        add(f"eos_li_hcp_{k:02d}", "eos_li_hcp", "ood", a, f"hcp at bcc V0/atom x {f:.3f}")
    hcp16 = hcp.repeat((2, 2, 2))
    for seed in (1, 2, 3):
        add(f"rat_li_hcp16_s10_{seed}", "rat_li_hcp16", "ood", rattled(hcp16, 0.10, seed),
            "hcp 2x2x2 rattled", ISYM=0)
    for miller, thick in (("100", (7, 11)), ("110", (6, 10)), ("111", (12, 18))):
        for n in thick:
            s = li_slab(a_bcc, miller, n)
            fam = f"surf_li_{miller}"
            add(f"{fam}_n{n}_ideal", fam, "ood", s, f"bcc Li({miller}) {n} layers")
            add(f"{fam}_n{n}_s05_1", fam, "ood", rattled(s, 0.05, 1), f"bcc Li({miller}) rattled",
                ISYM=0)
    li54 = bulk("Li", "bcc", a=a_bcc, cubic=True).repeat(3)
    vac = li54.copy()
    del vac[0]
    add("vac_li_ideal", "vac_li", "ood", vac, "bcc 3x3x3 minus one atom, unrelaxed", ISYM=0)
    for seed in (1, 2):
        add(f"vac_li_s05_{seed}", "vac_li", "ood", rattled(vac, 0.05, seed), "vacancy rattled",
            ISYM=0)
    add("bulk_li54_ideal", "bulk_li54", "ood", li54, "bcc 3x3x3")
    add("bulk_li54_s05_1", "bulk_li54", "ood", rattled(li54, 0.05, 1), "bcc 3x3x3 rattled", ISYM=0)
    for miller, thick in (("111", (3, 5)), ("110", (6, 10))):
        for n in thick:
            s = li2o_slab(a_ox, miller, n)
            fam = f"surf_li2o_{miller}"
            add(f"{fam}_n{n}_ideal", fam, "ood", s, f"Li2O({miller}) {n} units, non-polar")
            add(f"{fam}_n{n}_s05_1", fam, "ood", rattled(s, 0.05, 1), f"Li2O({miller}) rattled",
                ISYM=0)
    sc = make_supercell(antifluorite(a_ox), 2 * np.eye(3, dtype=int))
    assert len(sc) == 24
    o0 = next(i for i, s in enumerate(sc.get_chemical_symbols()) if s == "O")
    li0 = next(i for i, s in enumerate(sc.get_chemical_symbols()) if s == "Li")
    vli = sc.copy()
    del vli[li0]
    ili = sc.copy()
    ili.append(Atom("Li", sc.positions[o0] + 0.5 * a_ox * np.ones(3)))      # octahedral site
    ili.wrap()
    vo = sc.copy()
    del vo[o0]
    for tag, at, what in (("vli", vli, "Li vacancy"), ("ili", ili, "Li interstitial (octahedral)"),
                          ("vo", vo, "O vacancy")):
        fam = f"def_li2o_{tag}"
        add(f"{fam}_ideal", fam, "ood", at, f"Li2O 2x2x2 primitive, {what}, unrelaxed", ISYM=0)
        for seed in (1, 2):
            add(f"{fam}_s05_{seed}", fam, "ood", rattled(at, 0.05, seed), f"{what} rattled",
                ISYM=0)
    li2o2 = eq["li2o2"]
    for sigma in (0.05, 0.10):
        for seed in (1, 2, 3, 101):
            role = "test" if seed > 100 else "train"
            add(f"rat_li2o2_s{int(sigma * 100):02d}_{role}{seed}", "rat_li2o2", role,
                rattled(li2o2, sigma, seed), "Li2O2 (8 atoms) rattled", ISYM=0)
    li16 = bulk("Li", "bcc", a=a_bcc, cubic=True).repeat(2)
    li2o12 = li2o_conventional(a_ox)
    for host_name, host in (("li_bcc16", li16), ("li2o12", li2o12)):
        for sigma, seeds in ((0.15, (1, 2, 3, 101)), (0.20, (1, 2, 3))):
            for seed in seeds:
                role = "test" if seed > 100 else "train"
                add(f"rat_hot_{host_name}_s{int(sigma * 100):02d}_{role}{seed}",
                    f"rat_hot_{host_name}", role, rattled(host, sigma, seed),
                    f"{host_name} hot rattle", ISYM=0)
    for seed in (1, 2, 3):
        add(f"rat_hot_li2o2_s15_train{seed}", "rat_hot_li2o2", "train",
            rattled(li2o2, 0.15, seed), "Li2O2 hot rattle", ISYM=0)
    for host_name, host in (("li_bcc16", li16), ("li2o12", li2o12)):
        cmp_ = host.copy()
        cmp_.set_cell(host.cell * 0.90 ** (1 / 3), scale_atoms=True)
        for seed in (1, 2, 3):
            add(f"rat_cmp_{host_name}_v090_s10_train{seed}", f"rat_cmp_{host_name}", "train",
                rattled(cmp_, 0.10, seed), f"{host_name} at 0.90 V0 rattled", ISYM=0)
    for name, _, _, at, _, _ in cases:
        d = at.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        if d.min() < 1.2:
            raise RuntimeError(f"{name}: shortest distance {d.min():.2f} A")
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", nargs="?", default="write", choices=["write", "retag"])
    ap.add_argument("--out", required=True, help="write: case directory; retag: output extxyz")
    ap.add_argument("--ncores", type=int, default=76)
    ap.add_argument("--budget", type=float, default=40.0, help="yuan (list price) for the wall-time caps")
    ap.add_argument("--jobs", type=int, default=1, help="split the cases over this many Slurm jobs")
    ap.add_argument("--manifest", help="retag: manifest.csv of the write stage")
    ap.add_argument("--frames", nargs="*", default=[], help="retag: collected extxyz files")
    ap.add_argument("--dry-run", action="store_true", help="list cases and costs, write nothing")
    args = ap.parse_args(argv)
    from utils.runlog import session
    if args.stage == "retag":
        import ase.io
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with session("d2_retag", vars(args), root=os.path.join(os.path.dirname(out), "logs")) as log:
            frames = [a for p in args.frames for a in ase.io.read(p, ":")]
            tagged, missing = retag_frames(frames, read_manifest(args.manifest))
            if missing:
                print(f"[d2] {len(missing)} frames not in the manifest (skipped): {missing[:10]}")
            ase.io.write(out, tagged, format="extxyz")
            log.event("retag", frames=len(frames), tagged=len(tagged), missing=len(missing))
            print(f"[d2] {len(tagged)} frames -> {out}")
        return 0
    out = os.path.abspath(args.out)
    cases = d2_cases()
    rows = []
    for name, fam, role, at, inc, src in cases:
        g = kgrid(at, KSPACING)
        nk = nk_time_reversal(g)
        rows.append({"name": name, "family": fam, "role": role, "natoms": len(at),
                     "formula": at.get_chemical_formula(), "kspacing": KSPACING, "kgrid": "x".join(map(str, g)),
                     "nk_tr": nk, "est_core_h": round(core_hours(at, nk), 2), "source": src})
    tot = sum(r["est_core_h"] for r in rows)
    print(f"[d2] {len(rows)} single points, ~{tot:.0f} core-h = {YUAN_PER_CORE_H * tot:.1f} yuan "
          f"list (rough x0.5-2)")
    by_fam = {}
    for r in rows:
        by_fam.setdefault(r["family"], []).append(r)
    for fam, rs in by_fam.items():
        print(f"  {fam:22s} {rs[0]['role']:5s} {len(rs):2d} cases, {rs[0]['natoms']:3d} atoms, "
              f"k {rs[0]['kgrid']}, {sum(r['est_core_h'] for r in rs):6.1f} core-h")
    if args.dry_run:
        return 0
    from vasp_common import write_case
    os.makedirs(out, exist_ok=True)
    with session("d2_write", vars(args), root=os.path.join(out, "logs")) as log:
        for name, fam, role, at, inc, src in cases:
            write_case(os.path.join(out, name), at, **inc)
        write_manifest(out, rows)
        scripts = write_jobs(out, [c[0] for c in cases], "a0_d2", args.ncores, args.budget,
                             args.jobs)
        cost = write_cost(out, rows, args.ncores, args.budget)
        log.event("written", cases=len(cases), jobs=scripts, **{k: v for k, v in cost.items()
                                                                if isinstance(v, (int, float))})
        print(f"[d2] wrote {len(cases)} cases, {scripts}, manifest.csv, cost_estimate.json in {out}; "
              f"nothing submitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
