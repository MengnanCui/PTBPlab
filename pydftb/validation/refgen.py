"""Frozen DFTB+ reference data: generate, check offline, or compare live.

    python -m pydftb.validation.refgen --regenerate   # needs dftb+; writes JSON
    python -m pydftb.validation.refgen --check        # pydftb vs frozen JSON (no dftb+)
    python -m pydftb.validation.refgen --live         # pydftb vs a fresh dftb+ run

Each JSON file stores the exact input (labels, Angstrom coordinates, cell,
k-points, settings, the HSD text), the DFTB+ version and the reference
results, so the offline check is reproducible without DFTB+.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

from ..bands import broaden, eigenstates, klines
from ..calculator import DFTB
from ..constants import BOHR
from ..dispersion import library_versions
from ..params import ParamSet
from ..skf import write_typed_copies
from . import dftbplus
from .cases import CASES, CASES_BY_NAME, MAXL, REF_DIR, SKF_SETS, Case

SCHEMA = 1
STATE_KEYS = ("eigenvalues", "occupations", "fermi", "regions", "dp_dos")   # band/PDOS cases


# ----------------------------------------------------------------------------
def _plain_skf_dir(case: Case, tmp: str) -> str:
    """Uncompressed SKF directory for this case (typed copies if requested)."""
    src = SKF_SETS[case.skf_set]
    plain = os.path.join(tmp, "plain")
    os.makedirs(plain, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith(".skf.gz"):
            with gzip.open(os.path.join(src, f), "rb") as fi, \
                    open(os.path.join(plain, f[:-3]), "wb") as fo:
                shutil.copyfileobj(fi, fo)
    if case.typed:
        typed = os.path.join(tmp, "typed")
        write_typed_copies(plain, typed, case.typed)
        return typed
    return plain


def _skf_hashes(case: Case) -> dict:
    src = SKF_SETS[case.skf_set]
    out = {}
    for f in sorted(os.listdir(src)):
        if f.endswith(".skf.gz"):
            with gzip.open(os.path.join(src, f), "rb") as fh:
                out[f[:-3]] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _inputs(case: Case) -> dict:
    atoms = case.build()
    labels = case.labels(atoms) if case.labels else atoms.get_chemical_symbols()
    periodic = bool(atoms.pbc.any())
    k = case.kpoints() if periodic else None
    base = case.typed or {}
    disp = case.dispersion(list(labels)) if callable(case.dispersion) else case.dispersion
    return {
        "labels": list(labels),
        "positions_angstrom": atoms.positions.tolist(),
        "cell_angstrom": atoms.cell.array.tolist() if periodic else None,
        "kpoints": None if k is None else k[0].tolist(),
        "kweights": None if k is None else k[1].tolist(),
        "max_l": {lab: MAXL[base.get(lab, lab).rstrip("0123456789")] for lab in dict.fromkeys(labels)},
        "dispersion": None if disp is None else json.loads(json.dumps(disp)),
        "klines": None if case.klines is None else [[n, list(k)] for n, k in case.klines],
        "band_kpoints": None if case.klines is None else klines(case.klines)[0].tolist(),
        "regions": case.regions,
    }


# dispersion models that call a library, and which one
_LIBRARY_OF = {"dftd3": "dftd3", "simpledftd3": "dftd3", "dftd4": "dftd4"}


def dispersion_library(inp: dict) -> str | None:
    disp = inp.get("dispersion")
    return None if disp is None else _LIBRARY_OF.get(str(disp["model"]).lower())


def missing_requirement(doc: dict) -> str | None:
    """Why a frozen case cannot be checked here (a D3/D4 library missing or of another
    version than the one that made the reference), or None."""
    lib = dispersion_library(doc["input"])
    if lib is None:
        return None
    have = library_versions()[lib]
    want = doc.get("libraries", {}).get(lib)
    if have is None:
        return f"needs the optional package {lib} (reference made with {lib} {want})"
    if have != want:
        return f"reference made with {lib} {want}, installed {have}"
    return None


def run_pydftb(case: Case, inp: dict, skdir: str | None = None, solver: str = "eigh",
               fd_delta: float | None = None, engine: bool = False, k_threads=1,
               backend=None, device=None, dtype=None, **options):
    """Run pydftb for a case.  ``skdir`` defaults to the (compressed) test set.
    ``engine``: also return the pydftb.DFTB object (for band structures).
    ``backend``/``device``/``dtype``: the array backend (milestone M4, default NumPy);
    ``options``: other DFTB settings that replace the case's (e.g. ``scc_tol``)."""
    tmp = None
    if skdir is None and case.typed:
        tmp = tempfile.mkdtemp(prefix="pydftb_case_")
        skdir = write_typed_copies(SKF_SETS[case.skf_set], tmp, case.typed,
                                   template="{a}-{b}.skf.gz") and tmp
    skdir = skdir or SKF_SETS[case.skf_set]
    try:
        params = ParamSet(skdir, max_l=inp["max_l"], shell_resolved=case.shell_resolved)
        labels = inp["labels"]
        sk = params.skset(labels)
        periodic = inp["cell_angstrom"] is not None
        kpts = None
        if periodic and inp["kpoints"] is not None:
            kpts = (np.array(inp["kpoints"]), np.array(inp["kweights"]))
        opts = dict(scc=case.scc, kpts=kpts, smearing=tuple(case.filling), scc_tol=case.scc_tol,
                    max_scc=500, solver=solver, repulsive=case.repulsive, fd_delta=fd_delta,
                    dispersion=inp.get("dispersion"), k_threads=k_threads,
                    backend=backend, device=device, dtype=dtype,
                    **(case.dftb3 or {}), **({"spin": True, **case.spin} if case.spin else {}))
        calc = DFTB(sk, **{**opts, **options})
        cell = np.array(inp["cell_angstrom"]) / BOHR if periodic else None
        res = calc.calculate(labels, np.array(inp["positions_angstrom"]) / BOHR, cell,
                             np.ones(3, bool) if periodic else None)
        return (res, calc) if engine else res
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def run_dftbplus(case: Case, inp: dict, threads: int = 1) -> dict:
    with tempfile.TemporaryDirectory(prefix="pydftb_ref_") as tmp:
        skdir = _plain_skf_dir(case, tmp)
        periodic = inp["cell_angstrom"] is not None
        kpts = None
        if periodic and inp["kpoints"] is not None:
            kpts = (np.array(inp["kpoints"]), np.array(inp["kweights"]))
        extra = (dftbplus.dftb3_block(case.dftb3, set(inp["labels"]))
                 + dftbplus.spin_block(case.spin, set(inp["labels"]), case.shell_resolved)
                 + dftbplus.dispersion_block(inp.get("dispersion"), inp["labels"]))
        if case.ewald_tolerance is not None:
            extra += f"  EwaldTolerance = {dftbplus._f(case.ewald_tolerance)}\n"
        args = (inp["labels"], np.array(inp["positions_angstrom"]),
                np.array(inp["cell_angstrom"]) if periodic else None)
        kw = dict(skdir=skdir, max_l=inp["max_l"], kpts=kpts, scc=case.scc, filling=case.filling,
                  shell_resolved=case.shell_resolved, repulsive=case.repulsive, extra_ham=extra,
                  scc_tol=case.scc_tol, threads=threads)
        if not (case.klines or case.regions or case.dp_dos):
            return dftbplus.run(*args, **kw)
        shells = [list(range(inp["max_l"][inp["labels"][r["atoms"][0]]] + 1))
                  for r in case.regions or []]
        return dftbplus.run_bands(*args, klines=case.klines, regions=case.regions,
                                  region_shells=shells, dos=case.dp_dos, **kw)



# ---- band structures, projections and DOS curves (milestone M2e) ---------------
def _band_reference(case: Case, ref: dict) -> dict:
    """What a band/PDOS case freezes from DFTB+ (see dftbplus.run_bands)."""
    out = {"eigenvalues": ref["eigenvalues"].tolist(), "filling": ref["filling"].tolist(),
           "fermi_levels": ref["fermi_levels"].tolist()}
    if case.regions:
        out["regions"] = {name: vals[..., 1].tolist() for name, (_, vals) in ref["regions"].items()}
    if case.klines:
        out["band"] = {k: np.asarray(ref["band"][k]).tolist()
                       for k in ("eigenvalues", "filling", "fermi_levels")}
    if case.dp_dos:
        files = {"band.out": ref["band_out"], **ref["regions"]}
        out["levels"] = {name: {"kweights": files[name][0].tolist(), "values": files[name][1].tolist()}
                         for name in dict.fromkeys(name for name, _ in case.dp_dos)}
        out["dp_dos"] = [{"file": name, "args": args, "x0": float(cols[0, 0]), "nx": len(cols),
                          "y": cols[:, 1:].T.tolist()} for name, args, cols in ref["dp_dos"]]
    return out


def dp_dos_options(args) -> tuple[bool, dict]:
    """dp_dos command-line arguments -> (weight by the second column?, pydftb.bands.broaden keywords)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", type=float, default=0.01)
    ap.add_argument("-w", action="store_true")
    ap.add_argument("-f", default="gauss")
    ap.add_argument("-o", type=int, default=1)
    ap.add_argument("-b", type=float, default=None)
    ap.add_argument("-s", type=float, default=None)
    a = ap.parse_args(args)
    return a.w, {"grid": a.g, "kind": a.f, "mp_order": a.o, "width": a.b, "cutoff": a.s}


def _cluster_dev(evals, w, w_ref, tol=1e-7) -> float:
    """Largest difference of the weights summed over degenerate groups (per state)."""
    worst = 0.0
    for e, a, b in zip(evals.reshape(-1, evals.shape[-1]), w.reshape(-1, evals.shape[-1]),
                       w_ref.reshape(-1, evals.shape[-1])):
        gid = np.cumsum(np.concatenate([[True], np.diff(e) > tol])) - 1
        size = np.bincount(gid)
        diff = np.abs(np.bincount(gid, weights=a) - np.bincount(gid, weights=b)) / size
        worst = max(worst, float(diff.max()))
    return worst


def _fermi_dev(band, ref_band) -> float:
    """Fermi level of the band run vs DFTB+'s.  For an insulator any level in the gap fills
    the same states: pydftb takes the middle of the gap (occupations.py) and is compared
    with the middle of DFTB+'s gap, because DFTB+'s own value is where its bisection ends
    after middleGap missed the gap (water_box_bands: just below the conduction band)."""
    occ = np.array(ref_band["filling"])
    full = 2.0 if band.nspin == 1 else 1.0
    if not np.all((occ < 1e-9) | (occ > full - 1e-9)):
        return float(np.abs(band.fermi - np.array(ref_band["fermi_levels"])).max())
    e = np.array(ref_band["eigenvalues"])
    filled = occ > 0.5 * full
    if np.ptp(band.fermi) == 0.0:                   # one Fermi level for all channels
        mids = [0.5 * (e[filled].max() + e[~filled].min())] * band.nspin
    else:
        mids = [0.5 * (e[s][filled[s]].max() + e[s][~filled[s]].min()) for s in range(band.nspin)]
    return float(np.abs(band.fermi - np.array(mids)).max())


def _compare_states(case: Case, inp: dict, ref: dict, res, engine) -> dict:
    """Deviations of eigenvalues, occupations, Fermi levels, region weights, DOS curves."""
    periodic = inp["cell_angstrom"] is not None
    geo = (inp["labels"], np.array(inp["positions_angstrom"]) / BOHR,
           np.array(inp["cell_angstrom"]) / BOHR if periodic else None,
           np.ones(3, bool) if periodic else None)
    fermi = res.fermi_spin if res.fermi_spin is not None else res.fermi
    st = eigenstates(engine, *geo, res.q_in, projections=bool(case.regions), scc_fermi=fermi)
    dev = {"eigenvalues": float(np.abs(st.eigenvalues - np.array(ref["eigenvalues"])).max()),
           "occupations": float(np.abs(st.occupations - np.array(ref["filling"])).max())}
    if case.regions:
        shells = [list(range(inp["max_l"][inp["labels"][r["atoms"][0]]] + 1)) for r in case.regions]
        dev["regions"] = max(
            _cluster_dev(st.eigenvalues, st.weights(case.regions[i]["atoms"], ish, m),
                         np.array(ref["regions"][name]))
            for name, i, ish, m in dftbplus.region_files(case.regions, shells))
    if case.klines:
        band = eigenstates(engine, *geo, res.q_in, kpts=np.array(inp["band_kpoints"]),
                           scc_fermi=fermi)
        b = ref["band"]
        dev["eigenvalues"] = max(dev["eigenvalues"],
                                 float(np.abs(band.eigenvalues - np.array(b["eigenvalues"])).max()))
        dev["occupations"] = max(dev["occupations"],
                                 float(np.abs(band.occupations - np.array(b["filling"])).max()))
        dev["fermi"] = _fermi_dev(band, b)
    if ref.get("dp_dos"):
        worst = 0.0
        for curve in ref["dp_dos"]:
            occ, kw = dp_dos_options(curve["args"])
            lev = ref["levels"][curve["file"]]
            vals = np.array(lev["values"])
            x, y = broaden(vals[..., 0], np.array(lev["kweights"]), vals[..., 1] if occ else None,
                           **kw)
            cols = y if len(y) == 1 else np.vstack([y.sum(0), y])    # dp_dos: total, up, down
            yref = np.array(curve["y"])
            if len(x) != curve["nx"] or abs(x[0] - curve["x0"]) > 1e-9:
                worst = np.inf
            else:
                worst = max(worst, float(np.abs(cols - yref).max() / np.abs(yref).max()))
        dev["dp_dos"] = worst
    return dev


# ----------------------------------------------------------------------------
def regenerate(names=None, threads: int = 1):
    os.makedirs(REF_DIR, exist_ok=True)
    for case in CASES:
        if names and case.name not in names:
            continue
        inp = _inputs(case)
        ref = run_dftbplus(case, inp, threads)
        doc = {
            "schema": SCHEMA,
            "case": case.name,
            "note": case.note,
            "dftbplus": {"version": ref["version"], "parser_version": dftbplus.PARSER_VERSION,
                         "differentiation": "FiniteDiff (default)"},
            "skf_set": case.skf_set,
            "skf_sha256": _skf_hashes(case),
            "settings": {"scc": case.scc, "filling": list(case.filling),
                         "shell_resolved": case.shell_resolved, "repulsive": case.repulsive,
                         "typed": case.typed, "dftb3": case.dftb3, "spin": case.spin,
                         "scc_tol": case.scc_tol, "ewald_tolerance": case.ewald_tolerance},
            "libraries": ({lib: library_versions()[lib]} if (lib := dispersion_library(inp))
                          else {}),
            "input": inp,
            "hsd": ref["hsd"],
            "reference": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ref.items()
                          if k in ("energy", "free_energy", "e_zero", "e_band", "e_scc", "e_rep",
                                   "ts", "fermi", "forces", "charges_net", "n_scc", "stress",
                                   "e_disp")},
            "tolerances": case.tol,
        }
        if case.klines or case.regions or case.dp_dos:
            doc["reference"].update(_band_reference(case, ref))
            doc["band_hsd"] = ref.get("band", {}).get("hsd")
        if case.dp_dos:
            doc["dp_dos"] = ("dp_dos of DFTB+ 25.1 dptools (LGPL), run only to freeze these "
                             "curves; inputs in reference.levels")
        path = os.path.join(REF_DIR, f"{case.name}.json")
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=1)
        print(f"wrote {path}  (E = {ref['energy']:.10f} Ha, nSCC = {ref['n_scc']})")


def load(name: str) -> dict:
    with open(os.path.join(REF_DIR, f"{name}.json")) as fh:
        return json.load(fh)


def compare(doc: dict, solver: str = "eigh", fd_delta: float | None = None,
            **backend) -> dict:
    """Run pydftb on a frozen case; return the deviations and pass/fail.

    ``fd_delta`` = DFTB+'s finite-difference step (1.22e-4 by default in the
    reference runs) makes pydftb reproduce DFTB+'s approximate forces.
    ``backend``: ``backend``/``device``/``dtype`` of :class:`pydftb.DFTB`.
    """
    case = CASES_BY_NAME[doc["case"]]
    inp = doc["input"]
    t0 = time.perf_counter()
    res, engine = run_pydftb(case, inp, solver=solver, fd_delta=fd_delta, engine=True,
                             **backend)
    wall = time.perf_counter() - t0
    ref = doc["reference"]
    sk_q0 = _neutral_populations(case, inp)
    dev = {
        "energy": abs(res.energy - ref["energy"]),
        "free_energy": abs(res.free_energy - ref["free_energy"]),
        "e_zero": abs(res.e_zero - ref["e_zero"]) if "e_zero" in ref else 0.0,
        "stress": (float(np.abs(res.stress - np.array(ref["stress"])).max())
                   if ref.get("stress") is not None else 0.0),
        "forces": float(np.abs(res.forces - np.array(ref["forces"])).max()),
        "charges": float(np.abs((sk_q0 - res.charges) - np.array(ref["charges_net"])).max()),
        "e_disp": abs(res.e_disp - ref["e_disp"]) if "e_disp" in ref else 0.0,
    }
    if "eigenvalues" in ref:
        dev.update(_compare_states(case, inp, ref, res, engine))
    tol = doc["tolerances"]
    ok = (all(dev[k] <= tol[k] for k in ("energy", "forces", "charges"))
          and max(dev["free_energy"], dev["e_zero"], dev["e_disp"]) <= tol["energy"]
          and dev["stress"] <= tol.get("stress", 1e-7)
          and all(dev[k] <= tol[k] for k in STATE_KEYS if k in dev))
    return {"case": doc["case"], "dev": dev, "ok": ok, "wall": wall, "n_scc": res.n_scc,
            "result": res,
            "n_scc_ref": ref.get("n_scc"), "natoms": len(inp["labels"])}


def _neutral_populations(case: Case, inp: dict) -> np.ndarray:
    params_dir = SKF_SETS[case.skf_set]
    base = case.typed or {}
    p = ParamSet(params_dir, max_l={base.get(k, k): v for k, v in inp["max_l"].items()})
    sk = p.skset([base.get(lab, lab) for lab in dict.fromkeys(inp["labels"])])
    return np.array([sk.occupation[base.get(lab, lab)].sum() for lab in inp["labels"]])


def check(names=None, solver: str = "eigh", **backend) -> bool:
    all_ok = True
    print(f"{'case':26s} {'N':>4s} {'|dE| Ha':>9s} {'|dF| Ha/Bohr':>12s} {'|dq| e':>9s} "
          f"{'|dsigma|':>9s} {'|dE_disp|':>9s} {'|deps| Ha':>9s} {'nSCC py/ref':>11s} {'t_py':>6s}  ok")
    for case in CASES:
        if names and case.name not in names:
            continue
        path = os.path.join(REF_DIR, f"{case.name}.json")
        if not os.path.exists(path):
            print(f"{case.name:26s} (no reference JSON)")
            all_ok = False
            continue
        doc = load(case.name)
        why = missing_requirement(doc)
        if why:
            print(f"{case.name:26s} skipped: {why}")
            continue
        r = compare(doc, solver, **backend)
        d = r["dev"]
        all_ok &= r["ok"]
        e_disp = f"{d['e_disp']:9.1e}" if "e_disp" in doc["reference"] else f"{'-':>9s}"
        eps = f"{d['eigenvalues']:9.1e}" if "eigenvalues" in d else f"{'-':>9s}"
        print(f"{case.name:26s} {r['natoms']:4d} {d['energy']:9.1e} {d['forces']:12.1e} "
              f"{d['charges']:9.1e} {d['stress']:9.1e} {e_disp} {eps} "
              f"{r['n_scc']:>5}/{str(r['n_scc_ref']):<5} {r['wall']:6.2f}  {'yes' if r['ok'] else 'NO'}")
    return all_ok


def live(names=None):
    print(f"{'case':26s} {'|dE| Ha':>9s} {'|dF|':>9s} {'t_py':>6s} {'t_dftb+':>7s}")
    for case in CASES:
        if names and case.name not in names:
            continue
        inp = _inputs(case)
        t0 = time.perf_counter()
        res = run_pydftb(case, inp)
        t_py = time.perf_counter() - t0
        ref = run_dftbplus(case, inp)
        print(f"{case.name:26s} {abs(res.energy - ref['energy']):9.1e} "
              f"{np.abs(res.forces - ref['forces']).max():9.1e} {t_py:6.2f} {ref['wall']:7.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--regenerate", action="store_true", help="run dftb+ and write reference JSON")
    g.add_argument("--check", action="store_true", help="compare pydftb with the frozen JSON")
    g.add_argument("--live", action="store_true", help="compare pydftb with a fresh dftb+ run")
    ap.add_argument("cases", nargs="*", help="case names (default: all)")
    ap.add_argument("--solver", default="eigh")
    ap.add_argument("--backend", default=None, help="numpy (default) or torch")
    ap.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    ap.add_argument("--dtype", default=None, help="float64 (default) or float32")
    a = ap.parse_args(argv)
    if a.regenerate:
        regenerate(a.cases)
    elif a.check:
        sys.exit(0 if check(a.cases, a.solver, backend=a.backend, device=a.device,
                            dtype=a.dtype) else 1)
    else:
        live(a.cases)


if __name__ == "__main__":
    main()
