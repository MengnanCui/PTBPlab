"""Kill test K5 (A series, A0 section 7): can the confinement radii of C, N or I open DFTB's
FA+ π* / I− 5p gap to the DFT value?

In the M409 tests, DFTB(PTBP) gave FA+.I− a gap of about 0.09 eV, against several eV in DFT.  The
HOMO is the free iodide's 5p and the LUMO the C–N π* of FA+, and about 0.3 e leaks from
I− to FA+.  The protons then move within 10 fs (../Perovskite data/stage4/pitfalls.md).  If no
choice of radii within a reasonable box opens this gap, a single SK set cannot fix the
problem, and the on-site energies or U must change (A2 channels E1/E3).

    python k5_fa_iodide_gap.py [--structure ion_fa_i] [--ref bands/ion_fa_i.json]
        [--elements C N I] [--factors 0.8 0.9 1.0 1.1 1.25] [--mode oat] [--cache skcache]

* ``--mode oat`` (one at a time): each of the six radii (r0_w and r0_d of C, N and I) is moved
  through ``--factors`` × its published value while the others stay published.  With five
  factors that is 6 × 5 − 5 = 25 points.
* ``--mode joint``: a grid over r0_w of the chosen elements (factors^n points), with r0_d
  published.

Each point:

1. generates the SK set of the cluster's elements (H, C, N, I) through
   :class:`utils.skgrid.SKCache`, which caches it;
2. runs one DFTB SCC of the cluster (pydftb ``PyDFTB``, non-periodic,
   ``repulsive="none"``, max_l S/I/Sn/Cs = d, Pb = p, Gaussian smearing 0.01 eV);
3. records HOMO, LUMO, the gap and the Mulliken charge of the iodide;
4. if ``--ref`` (a BandRef of D1 GPAW or D4 VASP) is given, adds
   :func:`pydftb.bandloss.compare` against it.

The output is ``k5_results.json`` (every point) and ``k5_summary.md``, which gives the DFT gap,
the DFTB gap at the published radii, the largest gap found, where it was found, and the verdict.
The verdict counts as "opened" when the gap reaches ``--target`` × the DFT gap (default 0.8).
Without a reference the target is ``--gap-target`` eV.

Cost: one hotcent run per point for the four elements (minutes) and a second-long DFTB.  Run it
on the server: ``python k5_fa_iodide_gap.py --jobs 8``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import setup  # noqa: E402

setup()

MAX_L = {"S": "d", "I": "d", "Sn": "d", "Cs": "d", "Pb": "p"}
KB_EV = 8.617333262e-5
HARTREE_EV = 27.211386245988


def points(elements_fit, all_elements, factors, mode):
    """[(label, {element: [r0_w, r0_d]})] around PTBP's published radii."""
    from utils.skgrid import published
    pub = {el: [published(el)["r0_w"], published(el)["r0_d"]] for el in all_elements}
    out = [("published", {el: list(v) for el, v in pub.items()})]
    if mode == "oat":
        for el in elements_fit:
            for k, rname in enumerate(("r0_w", "r0_d")):
                for f in factors:
                    if abs(f - 1.0) < 1e-12:
                        continue
                    r = {e: list(v) for e, v in pub.items()}
                    r[el][k] = pub[el][k] * f
                    out.append((f"{el}.{rname}x{f:g}", r))
    elif mode == "joint":
        for combo in itertools.product(factors, repeat=len(elements_fit)):
            if all(abs(f - 1.0) < 1e-12 for f in combo):
                continue
            r = {e: list(v) for e, v in pub.items()}
            for el, f in zip(elements_fit, combo):
                r[el][0] = pub[el][0] * f
            out.append(("joint " + " ".join(f"{el}x{f:g}" for el, f in zip(elements_fit, combo)), r))
    else:
        raise ValueError(f"mode must be oat or joint, got {mode!r}")
    return out


def evaluate(args_tuple):
    label, radii, cfg = args_tuple
    from pydftb.ase_calc import PyDFTB
    from pydftb.bandloss import BandSettings, compare, dftb_bands
    from pydftb.bandref import BandRef
    from utils.skgrid import SKCache, SKSettings, published
    import ase.io
    cache = SKCache(cfg["cache"], SKSettings(grid=cfg["grid"], workers=cfg["hotcent_workers"]))
    p = {el: published(el)["p"] for el in radii}
    out = {"label": label, "radii": radii}
    try:
        skdir = cache.get(radii, p)
        ref = BandRef.from_json(cfg["ref"]) if cfg["ref"] else None
        if ref is not None and ref.numbers is not None:
            atoms = ref.atoms()
        else:
            atoms = next(a for a in ase.io.read(cfg["structures"], ":")
                         if a.info.get("name") == cfg["structure"])
        atoms.pbc = False
        max_l = {k: v for k, v in MAX_L.items() if k in set(atoms.get_chemical_symbols())}
        calc = PyDFTB(skdir, max_l=max_l, repulsive="none", scc_tol=1e-8, max_scc=500,
                      smearing=("gaussian", cfg["width"] / KB_EV), convergent_scc_only=True)
        atoms.calc = calc
        atoms.get_potential_energy()
        st = calc.states(atoms=atoms)
        eps = np.asarray(st.eigenvalues[0, 0], float) * HARTREE_EV
        nv = int(np.ceil(float(calc.raw.n_electrons) / 2 - 1e-6))
        q = np.asarray(atoms.get_charges(), float)
        sym = np.array(atoms.get_chemical_symbols())
        out.update(homo=float(eps[nv - 1]), lumo=float(eps[nv]) if nv < len(eps) else None,
                   gap=float(eps[nv] - eps[nv - 1]) if nv < len(eps) else None,
                   iodide_charge=float(q[sym == "I"].sum()), skdir=skdir)
        if ref is not None:
            out["band"] = compare(ref, dftb_bands(calc, atoms, ref), BandSettings())
    except Exception as exc:                  # noqa: BLE001  (recorded per point)
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--structures", default=os.path.join(HERE, "data", "structures.xyz"))
    ap.add_argument("--structure", default="ion_fa_i", help="name in structures.xyz")
    ap.add_argument("--ref", default=None, help="BandRef JSON (bands/ion_fa_i.json if present)")
    ap.add_argument("--elements", nargs="+", default=["C", "N", "I"])
    ap.add_argument("--factors", nargs="+", type=float, default=[0.8, 0.9, 1.0, 1.1, 1.25])
    ap.add_argument("--mode", default="oat", choices=["oat", "joint"])
    ap.add_argument("--cache", default=os.environ.get("PTBP_SKCACHE", os.path.join(HERE, "skcache")))
    ap.add_argument("--grid", nargs=2, type=int, default=None, metavar=("NR", "NTHETA"),
                    help="hotcent integration grid (default hotcent's)")
    ap.add_argument("--hotcent-workers", type=int, default=1)
    ap.add_argument("--width", type=float, default=0.01, help="Gaussian smearing (eV)")
    ap.add_argument("--target", type=float, default=0.8, help="fraction of the DFT gap")
    ap.add_argument("--gap-target", type=float, default=2.0, help="eV, without a reference")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "k5"))
    args = ap.parse_args(argv)
    import ase.io
    from utils.runlog import session
    if args.ref is None:
        guess = os.path.join(HERE, "bands", f"{args.structure}.json")
        args.ref = guess if os.path.exists(guess) else None
    os.makedirs(args.out, exist_ok=True)
    with session("k5_fa_iodide_gap", vars(args), root=os.path.join(args.out, "logs")) as log:
        frame = next((a for a in ase.io.read(args.structures, ":")
                      if a.info.get("name") == args.structure), None)
        if frame is None:
            print(f"no structure {args.structure} in {args.structures}")
            return 2
        elements = sorted(set(frame.get_chemical_symbols()))
        pts = points(args.elements, elements, args.factors, args.mode)
        cfg = {"cache": args.cache, "grid": ({"nr": args.grid[0], "ntheta": args.grid[1]}
                                             if args.grid else {}),
               "hotcent_workers": args.hotcent_workers, "ref": args.ref,
               "structures": args.structures, "structure": args.structure, "width": args.width}
        log.event("points", n=len(pts), elements=elements, fit=args.elements, mode=args.mode)
        tasks = [(lab, r, cfg) for lab, r in pts]
        if args.jobs > 1:
            import concurrent.futures as cf
            import multiprocessing as mp
            ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
            with cf.ProcessPoolExecutor(args.jobs, mp_context=ctx) as ex:
                results = list(ex.map(evaluate, tasks))
        else:
            results = [evaluate(t) for t in tasks]
        for r in results:
            log.event("point", label=r["label"], gap=r.get("gap"), homo=r.get("homo"),
                      error=r.get("error"))
        dft_gap = None
        if args.ref:
            from pydftb.bandref import BandRef
            ref = BandRef.from_json(args.ref)
            nv = next((int(r["band"]["n_valence"]) for r in results if "band" in r), None)
            dft_gap = ref.edges(nv)["gap"] if nv else ref.edges()["gap"]
        target = args.target * dft_gap if dft_gap else args.gap_target
        ok = [r for r in results if r.get("gap") is not None]
        pub = next((r for r in ok if r["label"] == "published"), None)
        best = max(ok, key=lambda r: r["gap"]) if ok else None
        opened = [r for r in ok if r["gap"] >= target]
        verdict = ("OPENED: some radii reach the target gap" if opened else
                   "NOT OPENED: no radii in this box reach the target gap; the single SK set "
                   "cannot fix FA+/I- (go to on-site/U changes, A2 channels E1/E3)")
        summary = {"dft_gap": dft_gap, "target": target, "published_gap": pub and pub["gap"],
                   "best": best and {k: best[k] for k in ("label", "gap", "homo", "lumo",
                                                          "iodide_charge", "radii")},
                   "n_points": len(results), "n_failed": len(results) - len(ok),
                   "n_opened": len(opened), "verdict": verdict}
        with open(os.path.join(args.out, "k5_results.json"), "w") as fh:
            json.dump({"summary": summary, "points": results}, fh, indent=1, default=str)
        lines = [f"# K5: FA+/I- gap in {args.structure}", "",
                 f"- DFT gap: {dft_gap if dft_gap is not None else 'no reference'} eV; target {target:.3f} eV",
                 f"- DFTB gap at the published radii: {pub['gap'] if pub else 'failed'} eV",
                 f"- largest DFTB gap: {best['gap']:.3f} eV at {best['label']}" if best else "- no point converged",
                 f"- points: {len(results)}, failed {len(results) - len(ok)}, reaching the target {len(opened)}",
                 "", f"**{verdict}**", "", "| point | gap (eV) | HOMO | LUMO | q(I) | band loss |",
                 "|---|---|---|---|---|---|"]
        for r in results:
            if r.get("gap") is None:
                lines.append(f"| {r['label']} | failed: {r.get('error', '')[:60]} | | | | |")
                continue
            bl = r.get("band", {}).get("loss")
            lines.append(f"| {r['label']} | {r['gap']:.3f} | {r['homo']:.3f} | {r['lumo']:.3f} | "
                         f"{r['iodide_charge']:+.3f} | {'' if bl is None else f'{bl:.3f}'} |")
        with open(os.path.join(args.out, "k5_summary.md"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        log.event("verdict", **{k: v for k, v in summary.items() if k != "best"})
        print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
