"""K1, the numerical floor (gate G0 of A0): DFTB's settings must not add errors of the size the
acceptance line allows (10 meV/atom, 0.1 eV/A).  Three checks, on PTBP's published Li and O radii:

(a) k-points and smearing.  Each structure runs with
    (i) PTBP's default (k-point density 5 per 1/A, Fermi at T = 0),
    (ii) the A-series setting (VASP KSPACING 0.15, Gamma-centred, Gaussian 0.05 eV), and
    (iii) a converged reference (KSPACING 0.08, Gaussian 0.05 eV).
    Pass: (ii) is within 2 meV/atom and 0.01 eV/A of (iii).
(b) SK-table noise.  Dimers Li2, LiO and O2 are scanned in 0.002 A steps with the SK sets of
    hotcent's default integration grid (nr 50, ntheta 150) and of denser grids.  The force noise
    is the RMS of F(r) minus a smooth local fit (cubic over 21-point windows); the SK tables are
    also compared directly.  Pass: noise <= 0.02 eV/A for the grid the scans use.
(c) Exact against finite-difference forces on a rattled Li2O cell: the central difference of
    the Mermin free energy, 1e-4 A, on six atoms.  Pass: <= 1e-4 eV/A.

    python k1_numerics.py --config scan_li_o.yaml --out runs/k1 [--grids 50:150 100:300 150:450]

Writes k1_results.json and k1_summary.md; logs in <out>/logs/k1_numerics/.  SK sets go to the
config's cache (sk.cache, or $PTBP_SKCACHE).  Light: a few minutes on 8 cores.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.paths import setup  # noqa: E402

setup()

STRUCTURES = ["rat_li_bcc16_s05_train1", "rat_li2o12_s05_train1", "eos_li2o2_05", "eos_li_bcc_05"]


def sk_dir(cfg, grid, log):
    from utils import accuracy as acc
    from utils.skgrid import SKCache, SKSettings
    sp = acc.space_of(cfg)
    radii = sp.radii(sp.published_vector())
    s = SKSettings.from_config({**cfg["sk"], "grid": grid})
    return SKCache(cfg["sk"]["cache"], s).get(radii, sp.p, log=log)


def run_setting(skdir, cfg, frame, kpts, smearing):
    from pydftb.ase_calc import PyDFTB
    from utils.accuracy import HARTREE_EV
    a = frame.copy()
    a.calc = PyDFTB(skdir, max_l=cfg["dftb"].get("max_l"), kpts=kpts, smearing=smearing,
                    repulsive="none", scc_tol=1e-8, max_scc=500)
    a.get_potential_energy()
    return float(a.calc.raw.e_zero) * HARTREE_EV, np.array(a.get_forces())


def part_a(cfg, skdir, frames, log):
    from ase.calculators.calculator import kptdensity2monkhorstpack
    from utils.accuracy import KB_EV, kpoints_for
    out = {}
    gauss = ("gaussian", 0.05 / KB_EV)
    for f in frames:
        name = f.info["name"]
        dense = f.copy()
        dense.info["kspacing"] = 0.08
        settings = {"ptbp_default": (list(kptdensity2monkhorstpack(f, 5.0, even=False)),
                                     ("fermi", 0.0)),
                    "a_series": (kpoints_for(f, 0.15), gauss),
                    "converged": (kpoints_for(dense, 0.08), gauss)}
        res = {}
        for key, (k, sm) in settings.items():
            with log.step("k1a", structure=name, setting=key, kpts=str(k)):
                res[key] = run_setting(skdir, cfg, f, k, sm)
        e_ref, f_ref = res["converged"]
        n = len(f)
        out[name] = {key: {"dE_meV_per_atom": 1e3 * (e - e_ref) / n,
                           "dF_max": float(np.abs(fo - f_ref).max()),
                           "dF_mae": float(np.abs(fo - f_ref).mean()),
                           "kpts": str(settings[key][0])}
                     for key, (e, fo) in res.items() if key != "converged"}
        out[name]["pass"] = bool(abs(out[name]["a_series"]["dE_meV_per_atom"]) <= 2.0
                                 and out[name]["a_series"]["dF_max"] <= 0.01)
    return out


def dimer_scan(skdir, cfg, pair, r):
    from ase import Atoms
    from pydftb.ase_calc import PyDFTB
    from utils.accuracy import HARTREE_EV, KB_EV
    e, f = [], []
    calc = PyDFTB(skdir, max_l=cfg["dftb"].get("max_l"), smearing=("gaussian", 0.05 / KB_EV),
                  repulsive="none", scc_tol=1e-9, max_scc=500, reuse_charges=True)
    for x in r:
        a = Atoms(pair, positions=[[0, 0, 0], [0, 0, x]], cell=[15, 15.5, 16], pbc=False)
        a.center()
        a.calc = calc
        a.get_potential_energy()
        e.append(float(a.calc.raw.free_energy) * HARTREE_EV)
        f.append(float(a.get_forces()[1, 2]))
    return np.array(e), np.array(f)


def noise(r, f, window=21):
    """RMS of F minus a cubic fitted over a sliding window (the jitter of the tables)."""
    res = []
    h = window // 2
    for i in range(h, len(r) - h):
        x, y = r[i - h:i + h + 1], f[i - h:i + h + 1]
        c = np.polyfit(x - r[i], y, 3)
        res.append(y[h] - np.polyval(c, 0.0))
    return float(np.sqrt(np.mean(np.square(res))))


def part_b(cfg, grids, log):
    out = {}
    pairs = {"Li2": np.arange(1.6, 3.4, 0.002), "LiO": np.arange(1.4, 3.0, 0.002),
             "O2": np.arange(1.0, 2.0, 0.002)}
    tables = {}
    for g in grids:
        key = f"nr{g.get('nr', 50)}_nt{g.get('ntheta', 150)}"
        skdir = sk_dir(cfg, g, print)
        tables[key] = skdir
        out[key] = {}
        for pair, r in pairs.items():
            with log.step("k1b", grid=key, pair=pair):
                e, f = dimer_scan(skdir, cfg, pair, r)
            fd = -np.gradient(e, r)
            out[key][pair] = {"force_noise": noise(r, f), "fd_noise": noise(r, fd),
                              "max_abs_force_minus_fd": float(np.abs(f - fd)[2:-2].max())}
    # direct table differences against the densest grid
    from pydftb.skf import SKFile
    keys = list(tables)
    ref = keys[-1]
    diffs = {}
    for k in keys[:-1]:
        diffs[k] = {}
        for fn in sorted(os.listdir(tables[ref])):
            if not fn.endswith(".skf"):
                continue
            a = np.asarray(SKFile.read_cached(os.path.join(tables[k], fn), homo=fn.split("-")[0] ==
                                              fn.split("-")[1][:-4]).table, float)
            b = np.asarray(SKFile.read_cached(os.path.join(tables[ref], fn), homo=fn.split("-")[0] ==
                                              fn.split("-")[1][:-4]).table, float)
            m = min(len(a), len(b))
            diffs[k][fn] = float(np.abs(a[:m] - b[:m]).max())
    out["table_max_abs_diff_vs_" + ref] = diffs
    return out


def part_c(cfg, skdir, frame, log, h=1e-4):
    from pydftb.ase_calc import PyDFTB
    from utils.accuracy import HARTREE_EV, KB_EV, kpoints_for
    a = frame.copy()
    a.calc = PyDFTB(skdir, max_l=cfg["dftb"].get("max_l"), kpts=kpoints_for(a, 0.15),
                    smearing=("gaussian", 0.05 / KB_EV), repulsive="none", scc_tol=1e-10,
                    max_scc=500)
    f0 = a.get_forces()
    worst = 0.0
    rng = np.random.default_rng(0)
    for i in rng.choice(len(a), size=min(6, len(a)), replace=False):
        for c in range(3):
            e = []
            for s in (+1, -1):
                b = a.copy()
                p = b.positions.copy()
                p[i, c] += s * h
                b.positions = p
                b.calc = a.calc
                b.get_potential_energy()
                e.append(float(b.calc.raw.free_energy) * HARTREE_EV)
            fd = -(e[0] - e[1]) / (2 * h)
            worst = max(worst, abs(fd - f0[i, c]))
    log.event("k1c", worst=float(worst))
    return {"max_abs_exact_minus_fd": float(worst), "pass": bool(worst <= 1e-4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "scan_li_o.yaml"))
    ap.add_argument("--out", default="runs/k1")
    ap.add_argument("--grids", nargs="+", default=["50:150", "100:300", "150:450"],
                    help="hotcent integration grids nr:ntheta (the first is hotcent's default)")
    ap.add_argument("--skip", nargs="*", default=[], choices=["a", "b", "c"])
    args = ap.parse_args(argv)
    from utils import accuracy as acc
    from utils.runlog import session
    cfg = acc.load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    grids = [dict(zip(("nr", "ntheta"), (int(v) for v in g.split(":")))) for g in args.grids]
    with session("k1_numerics", vars(args), root=os.path.join(args.out, "logs")) as log:
        frames = {f.info["name"]: f for f in acc.load_data(cfg)}
        missing = [n for n in STRUCTURES if n not in frames]
        if missing:
            raise SystemExit(f"structures not in the config's data: {missing}")
        skdir = sk_dir(cfg, cfg["sk"].get("grid", {}), print)
        results = {"config": cfg["_path"], "skdir": skdir}
        if "a" not in args.skip:
            results["a_kpoints_smearing"] = part_a(cfg, skdir, [frames[n] for n in STRUCTURES], log)
        if "b" not in args.skip:
            results["b_sk_noise"] = part_b(cfg, grids, log)
        if "c" not in args.skip:
            results["c_forces"] = part_c(cfg, skdir, frames["rat_li2o12_s05_train1"], log)
        path = os.path.join(args.out, "k1_results.json")
        with open(path, "w") as fh:
            json.dump(results, fh, indent=1,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))
        lines = ["# K1 numerical floor", ""]
        for name, r in results.get("a_kpoints_smearing", {}).items():
            lines.append(f"- (a) {name}: A-series vs converged "
                         f"{r['a_series']['dE_meV_per_atom']:+.2f} meV/atom, max dF "
                         f"{r['a_series']['dF_max']:.4f} eV/A; PTBP default "
                         f"{r['ptbp_default']['dE_meV_per_atom']:+.2f} meV/atom, max dF "
                         f"{r['ptbp_default']['dF_max']:.4f}; {'PASS' if r['pass'] else 'FAIL'}")
        for g, d in results.get("b_sk_noise", {}).items():
            if g.startswith("table_"):
                continue
            lines.append(f"- (b) {g}: force noise " + ", ".join(
                f"{p} {v['force_noise']:.4f}" for p, v in d.items()) + " eV/A")
        if "c_forces" in results:
            c = results["c_forces"]
            lines.append(f"- (c) exact vs FD forces: {c['max_abs_exact_minus_fd']:.2e} eV/A, "
                         f"{'PASS' if c['pass'] else 'FAIL'}")
        with open(os.path.join(args.out, "k1_summary.md"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
