"""Milestone M8b: the Cu DFT training set that M8a's adaptive model needs (forces and distorted
structures), as VASP inputs and Slurm array scripts for the 电光云 cluster (queue dg83).  It only
writes files: nothing is uploaded or submitted.

    python -m pydftb.bench.cu_dft_dataset OUT_DIR [--md-frames cu13_adaptive.extxyz]

Groups (every structure single point with forces and stress):

* ``dimer``      Cu2 at 1.7-3.6 A (the short-range region M8a's fit had no data for);
* ``cluster``    Cu3, Cu4, Cu6, Cu7, Cu13 (icosahedron, cuboctahedron, Ino decahedron), Cu38
                 (truncated octahedron), Cu55 (icosahedron), ideal and rattled;
* ``md``         frames of the Cu13 cluster that restructured in M8a's adaptive-model MD;
* ``surface``    fcc(111), (100), (110) slabs and an adatom on (111), ideal and rattled;
* ``vacancy``    a vacancy and a divacancy in a 32-atom fcc cell;
* ``rattled``    fcc, bcc, hcp, diamond and simple cubic supercells, rattled, several volumes;
* ``strain``     the fcc primitive cell under six strain patterns;
* ``ptbp``       the 66 structures of PTBP's Cu set (pydftb.validation.ptbp_workload), so that the
                 old FHI-aims energies can be replaced by VASP ones on the same footing.

Settings: PBE PAW (Cu, 11 electrons), ENCUT 450 eV, Fermi smearing at 300 K (ISMEAR = -1,
SIGMA = 0.0259 eV: the free energy is the Mermin F that pydftb fits), EDIFF 1e-6, KSPACING
0.2 1/A for periodic cells; clusters and dimers Gamma only (vasp_gam), spin-polarised, 12 A of
vacuum at least.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc100, fcc110, fcc111
from ase.cluster import Decahedron, Icosahedron, Octahedron
from ase.io import read, write

A_FCC = 3.63                       # PBE lattice constant of Cu (A)
VACUUM = 12.0                      # A between images of a cluster or slab
KSPACING = 0.2                     # 1/A (VASP's convention, 2 pi included)
INCAR_COMMON = {"PREC": "Accurate", "ENCUT": 450, "EDIFF": "1E-6", "ISMEAR": -1,
                "SIGMA": 0.0259, "NELM": 200, "ALGO": "Normal", "IBRION": -1, "NSW": 0,
                "ISIF": 2, "LWAVE": ".FALSE.", "LCHARG": ".FALSE.", "LORBIT": 0}
# rough cost model: core-hours of one k-point of a 32-atom bulk cell, scaled by atoms^2.2 and
# by the plane waves (box volume); Gamma-only vasp_gam is about half; spin doubles.  To be
# calibrated with the first jobs (README).
CORE_H_PER_KPOINT_32 = 1.0


def _rng(seed):
    return np.random.default_rng(seed)


def _rattled(at, sigma, seed):
    a = at.copy()
    a.positions += _rng(seed).normal(scale=sigma, size=a.positions.shape)
    return a


def _boxed(at):
    a = at.copy()
    a.center(vacuum=VACUUM / 2)
    a.pbc = True
    return a


def structures(md_frames=None):
    out = []                                         # (group, name, atoms, gamma_only, spin)

    def add(group, name, at, gamma=False, spin=False):
        out.append((group, name, at, gamma, spin))

    for r in np.round(np.arange(1.7, 3.61, 0.2), 2):
        add("dimer", f"dimer_{r:.2f}", _boxed(Atoms("Cu2", positions=[[0, 0, 0], [0, 0, r]])),
            True, True)
    d = A_FCC / math.sqrt(2)
    small = {
        "cu3_triangle": Atoms("Cu3", positions=[[0, 0, 0], [d, 0, 0], [d / 2, d * 0.866, 0]]),
        "cu4_tetrahedron": Atoms("Cu4", positions=[[0, 0, 0], [d, 0, 0], [d / 2, d * 0.866, 0],
                                                   [d / 2, d * 0.289, d * 0.816]]),
        "cu4_rhombus": Atoms("Cu4", positions=[[0, 0, 0], [d, 0, 0], [d / 2, d * 0.866, 0],
                                               [1.5 * d, d * 0.866, 0]]),
        "cu6_octahedron": Octahedron("Cu", 2, cutoff=0, latticeconstant=A_FCC),
        "cu7_pbp": Decahedron("Cu", 2, 1, 0, latticeconstant=A_FCC),
        "cu13_icosahedron": Icosahedron("Cu", 2, latticeconstant=A_FCC),
        "cu13_cuboctahedron": Octahedron("Cu", 3, cutoff=1, latticeconstant=A_FCC),
        "cu13_decahedron": Decahedron("Cu", 2, 2, 0, latticeconstant=A_FCC),
        "cu38_truncoct": Octahedron("Cu", 4, cutoff=1, latticeconstant=A_FCC),
        "cu55_icosahedron": Icosahedron("Cu", 3, latticeconstant=A_FCC),
    }
    for k, (name, at) in enumerate(small.items()):
        add("cluster", name, _boxed(at), True, True)
        for j, s in enumerate((0.05, 0.12)):
            if len(at) <= 38 or j == 0:
                add("cluster", f"{name}_rattle{s:.2f}", _boxed(_rattled(at, s, 100 * k + j)),
                    True, True)
    if md_frames and os.path.exists(md_frames):
        frames = read(md_frames, ":")
        for k in range(4, len(frames), 4):
            add("md", f"cu13_md_{k * 10:04d}fs", _boxed(frames[k]), True, True)
    slabs = {"fcc111": fcc111("Cu", (3, 3, 4), a=A_FCC, vacuum=VACUUM / 2),
             "fcc100": fcc100("Cu", (3, 3, 4), a=A_FCC, vacuum=VACUUM / 2),
             "fcc110": fcc110("Cu", (3, 2, 5), a=A_FCC, vacuum=VACUUM / 2)}
    ad = fcc111("Cu", (3, 3, 4), a=A_FCC, vacuum=VACUUM / 2)
    add_adsorbate(ad, "Cu", 2.1, "fcc")
    ad.center(vacuum=VACUUM / 2, axis=2)
    slabs["fcc111_adatom"] = ad
    for k, (name, at) in enumerate(slabs.items()):
        add("surface", name, at)
        add("surface", f"{name}_rattle0.08", _rattled(at, 0.08, 200 + k))
    cell = bulk("Cu", "fcc", a=A_FCC, cubic=True).repeat(2)              # 32 atoms
    vac = cell.copy()
    del vac[0]
    add("vacancy", "fcc32_vacancy", vac)
    add("vacancy", "fcc32_vacancy_rattle0.06", _rattled(vac, 0.06, 301))
    div = cell.copy()
    del div[[0, 1]]
    add("vacancy", "fcc32_divacancy", div)
    for k, (name, at) in enumerate({
            "fcc32": cell, "bcc16": bulk("Cu", "bcc", a=2.89, cubic=True).repeat(2),
            "hcp16": bulk("Cu", "hcp", a=2.57, c=4.19).repeat((2, 2, 2)),
            "diamond8": bulk("Cu", "diamond", a=5.36, cubic=True),
            "sc8": bulk("Cu", "sc", a=2.42).repeat(2)}.items()):
        for v in (0.90, 1.00, 1.10):
            scaled = at.copy()
            scaled.set_cell(at.cell * v ** (1 / 3), scale_atoms=True)
            for j, s in enumerate((0.05, 0.15)):
                if name in ("sc8", "diamond8") and j == 1 and v != 1.0:
                    continue
                add("rattled", f"{name}_v{v:.2f}_rattle{s:.2f}", _rattled(scaled, s,
                                                                         400 + 10 * k + j))
    prim = bulk("Cu", "fcc", a=A_FCC)
    patterns = {"xx": [[1, 0, 0], [0, 0, 0], [0, 0, 0]], "xy": [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
                "zz": [[0, 0, 0], [0, 0, 0], [0, 0, 1]], "xz": [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
                "biax": [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
                "tetra": [[1, 0, 0], [0, 1, 0], [0, 0, -2]]}
    for name, e in patterns.items():
        for m in (-0.06, -0.03, 0.03, 0.06):
            at = prim.copy()
            at.set_cell(prim.cell @ (np.eye(3) + 0.5 * m * np.array(e, float)).T,
                        scale_atoms=True)
            add("strain", f"fcc_{name}_{m:+.2f}", at)
    from pydftb.validation import ptbp_workload as pw
    for s in pw.structures():
        name = "ptbp_" + s.name.replace(" ", "_")
        if s.phase == "dimer":
            add("ptbp", name, s.atoms.copy(), True, True)
        else:
            add("ptbp", name, s.atoms.copy())
    return out


def kpoint_grid(at):
    rec = 2 * math.pi * at.cell.reciprocal()
    return tuple(max(1, math.ceil(np.linalg.norm(b) / KSPACING)) for b in rec)


def estimate_core_hours(at, gamma, spin):
    n = len(at)
    nk = 1 if gamma else max(1, math.prod(kpoint_grid(at)) // 2)   # time reversal, no symmetry
    pw = max(1.0, at.get_volume() / (11.9 * n)) ** 0.4               # vacuum costs plane waves
    t = CORE_H_PER_KPOINT_32 * nk * (n / 32.0) ** 2.2 * pw
    return t * (0.5 if gamma else 1.0) * (2.0 if spin else 1.0)


def cores_for(n_atoms):
    return 19 if n_atoms <= 8 else 38 if n_atoms <= 20 else 76


def write_set(out_dir, items):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    groups = {19: [], 38: [], 76: []}
    for group, name, at, gamma, spin in items:
        d = os.path.join(out_dir, group, name)
        os.makedirs(d, exist_ok=True)
        write(os.path.join(d, "POSCAR"), at, format="vasp", direct=True, sort=True)
        incar = dict(INCAR_COMMON)
        if spin:
            incar.update(ISPIN=2, MAGMOM=f"{len(at)}*1.0")
        if len(at) > 30:
            incar["LREAL"] = "Auto"
        if gamma:
            with open(os.path.join(d, "KPOINTS"), "w") as fh:
                fh.write("Gamma only\n0\nGamma\n1 1 1\n0 0 0\n")
        else:
            incar.update(KSPACING=KSPACING, KGAMMA=".TRUE.")
        with open(os.path.join(d, "INCAR"), "w") as fh:
            fh.write("".join(f"{k} = {v}\n" for k, v in incar.items()))
        cores = cores_for(len(at))
        groups[cores].append((f"{group}/{name}", "vasp_gam" if gamma else "vasp_std"))
        manifest.append({"group": group, "name": name, "path": f"{group}/{name}",
                         "atoms": len(at), "gamma_only": gamma, "spin": spin,
                         "kpoints": [1, 1, 1] if gamma else list(kpoint_grid(at)),
                         "cores": cores, "core_hours_estimate": estimate_core_hours(at, gamma,
                                                                                    spin)})
    for cores, jobs in groups.items():
        if not jobs:
            continue
        with open(os.path.join(out_dir, f"jobs_{cores}.txt"), "w") as fh:
            fh.write("".join(f"{p} {exe}\n" for p, exe in jobs))
        with open(os.path.join(out_dir, f"run_{cores}.sh"), "w") as fh:
            fh.write(SLURM.format(cores=cores, n=len(jobs) - 1))
    with open(os.path.join(out_dir, "make_potcar.sh"), "w") as fh:
        fh.write(POTCAR_SH)
    with open(os.path.join(out_dir, "collect.py"), "w") as fh:
        fh.write(COLLECT)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


SLURM = """#!/bin/bash
# array job over jobs_{cores}.txt ({cores} cores each); submit from this directory:
#   sbatch --array=0-{n}%20 run_{cores}.sh     (%20: at most 20 at a time)
#SBATCH -J cu_m8b_{cores}
#SBATCH -N 1
#SBATCH -n {cores}
#SBATCH -p dg83
#SBATCH -o logs/%x.%A_%a.out
#SBATCH -e logs/%x.%A_%a.err
module purge
module load intel/2022.1
export PATH=/data/home/dg002434/software/vasp.6.6.0/bin:$PATH
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" jobs_{cores}.txt)
dir=${{line%% *}}; exe=${{line##* }}
cd "$dir" || exit 1
if grep -q "General timing" OUTCAR 2>/dev/null; then echo "done already"; exit 0; fi
mpirun -np {cores} $exe > vasp.log 2>&1
"""

POTCAR_SH = """#!/bin/bash
# run on the cluster once: a POTCAR (PBE PAW Cu, 11 electrons) in every job directory
P=/data/home/dg002434/software/potpaw_PBE.54/Cu/POTCAR
mkdir -p logs
for f in $(find . -name POSCAR); do cp "$P" "$(dirname "$f")/POTCAR"; done
echo "POTCARs: $(find . -name POTCAR | wc -l)"
"""

COLLECT = '''"""Collect the finished jobs into cu_m8b.extxyz (free energy, forces, stress) and report the
unfinished or unconverged ones.  python collect.py   (needs ase)"""
import json, os
from ase.io import read, write
man = json.load(open("manifest.json"))
good, bad = [], []
for m in man:
    try:
        at = read(os.path.join(m["path"], "vasprun.xml"), index=-1)
        e = at.get_potential_energy(force_consistent=True)
        at.info.update(group=m["group"], name=m["name"], free_energy=e)
        at.arrays["dft_forces"] = at.get_forces()
        if all(at.pbc) and m["group"] not in ("dimer", "cluster", "md"):
            at.info["dft_stress"] = at.get_stress()
        at.calc = None
        good.append(at)
    except Exception as exc:                      # noqa: BLE001
        bad.append((m["path"], f"{type(exc).__name__}: {exc}"))
write("cu_m8b.extxyz", good)
print(f"{len(good)} collected, {len(bad)} missing or failed")
for p, why in bad:
    print("  ", p, why)
'''


README = """# Cu 训练集（M8b，VASP，电光云）

M8a 的自适应模型只用 6 个理想相的能量拟合，在 Cu13 团簇的 MD 里重构、蒸发（pydftb 报告
`docs/reports/M8a-adaptive-params.html` 第四节）。这个训练集补上它缺的：力、应力、畸变与混合配位的
结构、短程区。**只生成了输入，没有上传，没有提交。**

## 结构

{table}

合计 **{n} 个结构**，估计 **{core_h:.0f} 核时**，约 **{yuan:.0f} 元**（dg83，0.1 元/核时）。
估计很粗（一个 32 原子体相晶胞每个 k 点约 {c0} 核时，按原子数的 2.2 次方与真空的平面波数放大，
Gamma-only 减半、自旋加倍），**先算三个校准作业再定**（下面第 2 步）。

## 设置

PBE PAW（`potpaw_PBE.54/Cu`，11 个电子），ENCUT 450 eV，PREC Accurate，EDIFF 1e-6；Fermi 展宽
300 K（ISMEAR = -1，SIGMA = 0.0259 eV），自由能就是 pydftb 拟合的 Mermin F；周期体系 KSPACING
0.2 1/Å（KGAMMA），团簇与二聚体只用 Gamma 点（vasp_gam）并且自旋极化（MAGMOM 每个原子 1）；
真空至少 12 Å；单点，带力与应力；超过 30 个原子 LREAL = Auto。

队列：**dg83（CPU）**。dg50 的 RTX 5090 的双精度只有 1.7 TFLOPS，不适合 VASP。

## 怎么用（用户决定之后）

1. 上传：`rsync -avP {out}/ dghpc:/data/home/dg002434/cu_m8b/`，然后在集群上
   `cd ~/cu_m8b && bash make_potcar.sh`。
2. 校准：先只提交每组一个，例如 `sbatch --array=0 run_19.sh`、`sbatch --array=0 run_38.sh`、
   `sbatch --array=0 run_76.sh`，看 `sacct -j <ID> --format=Elapsed,NCPUS` 的实际核时，再按比例修正上面的总数。
3. 全部：`sbatch --array=0-{n19}%20 run_19.sh`、`sbatch --array=0-{n38}%20 run_38.sh`、
   `sbatch --array=0-{n76}%10 run_76.sh`（已经算完的目录会跳过，可以重复提交）。
4. 收集：`python collect.py`，得到 `cu_m8b.extxyz`（自由能、力、应力），下载回本地给 M8b 的拟合。

## 文件

`manifest.json`（每个结构的组、原子数、k 网格、核数、估计核时），`jobs_<核数>.txt` 与 `run_<核数>.sh`
（Slurm 数组作业，按原子数分为 19、38、76 核），`make_potcar.sh`，`collect.py`，每个结构一个目录
（POSCAR、INCAR，Gamma-only 的另有 KPOINTS）。由 `python -m pydftb.bench.cu_dft_dataset` 生成。
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--md-frames", default=None)
    args = ap.parse_args(argv)
    items = structures(args.md_frames)
    man = write_set(args.out_dir, items)
    by = {}
    for m in man:
        g = by.setdefault(m["group"], {"n": 0, "atoms": [], "core_h": 0.0})
        g["n"] += 1
        g["atoms"].append(m["atoms"])
        g["core_h"] += m["core_hours_estimate"]
    rows = ["| 组 | 结构数 | 原子数 | 估计核时 |", "|---|---|---|---|"]
    for k, g in by.items():
        rows.append(f"| {k} | {g['n']} | {min(g['atoms'])}–{max(g['atoms'])} | "
                    f"{g['core_h']:.0f} |")
    core_h = sum(m["core_hours_estimate"] for m in man)
    count = {c: sum(1 for m in man if m["cores"] == c) for c in (19, 38, 76)}
    with open(os.path.join(args.out_dir, "README.md"), "w") as fh:
        fh.write(README.format(table="\n".join(rows), n=len(man), core_h=core_h,
                               yuan=0.1 * core_h, c0=CORE_H_PER_KPOINT_32, out=args.out_dir,
                               n19=count[19] - 1, n38=count[38] - 1, n76=count[76] - 1))
    print(f"{len(man)} structures, ~{core_h:.0f} core-hours (~{0.1 * core_h:.0f} yuan) -> "
          f"{args.out_dir}")
    for k, g in by.items():
        print(f"  {k:8s} {g['n']:4d}  atoms {min(g['atoms'])}-{max(g['atoms'])}  "
              f"~{g['core_h']:.0f} core-h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
