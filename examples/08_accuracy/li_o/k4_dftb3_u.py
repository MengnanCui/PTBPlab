"""K4: do knobs outside r0 change the sign of the interface collapse (A0, section 7)?

M5c/M5d: with every Li set so far, the collapsed snapshots md02 and md04 come out below the
initial Li|Li2O interface in DFTB (-0.6 to -3.4 eV/atom), while DFT puts them +1.29 and +1.44
eV/atom above.  K4 keeps the electronic tables (PTBP's published radii, or one scan point) and
the physrep repulsive (published sigma, or the point's S1 sigma), and varies only:

* DFTB2, the baseline;
* DFTB3 (``third_order="full"``) with O's 3ob Hubbard derivative -0.1575 Ha/e and Li's Ud in
  {-0.02, -0.05, -0.10};
* U_O and U_Li scaled by 0.8 and 1.2 (``hubbard``: DFTB+ CustomisedHubbards).

It reports the collapse energies per atom, their signs against DFT, and the Li2O and Li2O2 EOS
minima per atom.  The minima use the same seven volumes, so they change consistently.

    python k4_dftb3_u.py --config scan_li_o.yaml --out runs/k4 [--point INDEX]

Light: 3 interface single points (80 atoms, 4x4x2 as the DFT's) and 14 bulk points per variant.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.paths import setup  # noqa: E402

setup()

DFT_COLLAPSE = {"iface_md02": 1.286, "iface_md04": 1.437}       # eV/atom above iface_init


def variants(skdir):
    """(name, extra PyDFTB options) of every K4 variant."""
    from pydftb.params import ParamSet
    sk = ParamSet(skdir, max_l={"Li": "s", "O": "p"}).skset(["Li", "O"])
    u = {el: float(np.atleast_1d(sk.hubbard[el])[0]) for el in ("Li", "O")}
    out = [("DFTB2", {})]
    for ud_li in (-0.02, -0.05, -0.10):
        out.append((f"DFTB3 Ud(Li) {ud_li}", {"third_order": "full",
                                              "hubbard_derivs": {"Li": ud_li, "O": -0.1575}}))
    for el in ("O", "Li"):
        for f in (0.8, 1.2):
            h = {e: (u[e] * f if e == el else u[e]) for e in u}
            out.append((f"U_{el} x{f}", {"hubbard": h}))
    return out, u


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "scan_li_o.yaml"))
    ap.add_argument("--out", default="runs/k4")
    ap.add_argument("--point", type=int, default=None,
                    help="a scored scan point (default: PTBP's published radii and sigma)")
    args = ap.parse_args(argv)
    from pydftb.ase_calc import PyDFTB
    from utils import accuracy as acc
    from utils.runlog import session
    from utils.skgrid import published
    cfg = acc.load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    with session("k4_dftb3_u", vars(args), root=os.path.join(args.out, "logs")) as log:
        sp = acc.space_of(cfg)
        if args.point is None:
            radii = sp.radii(sp.published_vector())
            skdir = acc.cache_of(cfg).get(radii, sp.p)
            sigma = {el: published(el)["sigma_rep"] for el in sp.elements}
        else:
            res = json.load(open(os.path.join(acc.point_dir(cfg, args.point), "result.json")))
            sc = json.load(open(os.path.join(acc.point_dir(cfg, args.point), "losses.json")))
            skdir, radii, sigma = res["skdir"], res["radii"], sc["strategies"]["S1"]["sigma"]
        frames = {f.info["name"]: f for f in acc.load_data(cfg)}
        iface = [frames[n] for n in ("iface_init", "iface_md02", "iface_md04")]
        bulk = {fam: [frames[f"{fam}_{k:02d}"] for k in range(2, 9)] for fam in ("eos_li2o", "eos_li2o2")}
        phys = acc._Physrep(list(frames.values()), [], sigma)
        pr = phys.model([])

        def energy(a, opts):
            b = a.copy()
            dcfg = cfg["dftb"]
            b.calc = PyDFTB(skdir, max_l=dcfg.get("max_l"), kpts=acc.kpoints_for(a, 0.15),
                            smearing=("gaussian", 0.05 / acc.KB_EV), repulsive="none",
                            scc_tol=1e-6, max_scc=1000, mixing=0.1, **opts)
            b.get_potential_energy()
            return float(b.calc.raw.e_zero) * acc.HARTREE_EV + pr.evaluate_pbc(a)[0]

        table = {}
        vs, u = variants(skdir)
        for name, opts in vs:
            row = {}
            try:
                with log.step("k4", variant=name):
                    e = {a.info["name"]: energy(a, opts) for a in iface}
                    n = len(iface[0])
                    for k, dft in DFT_COLLAPSE.items():
                        de = (e[k] - e["iface_init"]) / n
                        row[k] = {"dftb_eV_per_atom": de, "dft_eV_per_atom": dft,
                                  "sign_ok": bool(np.sign(de) == np.sign(dft))}
                    for fam, fr in bulk.items():
                        row[f"{fam}_min_eV_per_atom"] = min(energy(a, opts) / len(a) for a in fr)
            except Exception as exc:          # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
            table[name] = row
            print(name, json.dumps(row))
        out = {"skdir": skdir, "radii": radii, "sigma": sigma, "hubbard_skf": u, "variants": table}
        with open(os.path.join(args.out, "k4_results.json"), "w") as fh:
            json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        lines = ["# K4: interface collapse sign under DFTB3 and U scaling", "",
                 "| variant | md02 (DFT +1.29) | md04 (DFT +1.44) | signs |", "|---|---|---|---|"]
        for name, row in table.items():
            if "error" in row:
                lines.append(f"| {name} | {row['error']} | | |")
                continue
            a, b = row["iface_md02"], row["iface_md04"]
            lines.append(f"| {name} | {a['dftb_eV_per_atom']:+.3f} | {b['dftb_eV_per_atom']:+.3f} | "
                         f"{'ok' if a['sign_ok'] and b['sign_ok'] else 'WRONG'} |")
        with open(os.path.join(args.out, "k4_summary.md"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
