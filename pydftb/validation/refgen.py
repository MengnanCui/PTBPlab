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

from ..calculator import DFTB
from ..constants import BOHR
from ..params import ParamSet
from ..skf import write_typed_copies
from . import dftbplus
from .cases import CASES, CASES_BY_NAME, MAXL, REF_DIR, SKF_SETS, Case

SCHEMA = 1


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
    return {
        "labels": list(labels),
        "positions_angstrom": atoms.positions.tolist(),
        "cell_angstrom": atoms.cell.array.tolist() if periodic else None,
        "kpoints": None if k is None else k[0].tolist(),
        "kweights": None if k is None else k[1].tolist(),
        "max_l": {lab: MAXL[base.get(lab, lab).rstrip("0123456789")] for lab in dict.fromkeys(labels)},
    }


def run_pydftb(case: Case, inp: dict, skdir: str | None = None, solver: str = "eigh",
               fd_delta: float | None = None):
    """Run pydftb for a case.  ``skdir`` defaults to the (compressed) test set."""
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
        calc = DFTB(sk, scc=case.scc, kpts=kpts, smearing=tuple(case.filling), scc_tol=1e-10,
                    max_scc=500, solver=solver, repulsive=case.repulsive, fd_delta=fd_delta)
        cell = np.array(inp["cell_angstrom"]) / BOHR if periodic else None
        return calc.calculate(labels, np.array(inp["positions_angstrom"]) / BOHR, cell,
                              np.ones(3, bool) if periodic else None)
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
        return dftbplus.run(inp["labels"], np.array(inp["positions_angstrom"]),
                            np.array(inp["cell_angstrom"]) if periodic else None, skdir=skdir,
                            max_l=inp["max_l"], kpts=kpts, scc=case.scc, filling=case.filling,
                            shell_resolved=case.shell_resolved, repulsive=case.repulsive,
                            threads=threads)


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
                         "typed": case.typed, "scc_tol": 1e-10},
            "input": inp,
            "hsd": ref["hsd"],
            "reference": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ref.items()
                          if k in ("energy", "free_energy", "e_zero", "e_band", "e_scc", "e_rep",
                                   "ts", "fermi", "forces", "charges_net", "n_scc", "stress")},
            "tolerances": case.tol,
        }
        path = os.path.join(REF_DIR, f"{case.name}.json")
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=1)
        print(f"wrote {path}  (E = {ref['energy']:.10f} Ha, nSCC = {ref['n_scc']})")


def load(name: str) -> dict:
    with open(os.path.join(REF_DIR, f"{name}.json")) as fh:
        return json.load(fh)


def compare(doc: dict, solver: str = "eigh", fd_delta: float | None = None) -> dict:
    """Run pydftb on a frozen case; return the deviations and pass/fail.

    ``fd_delta`` = DFTB+'s finite-difference step (1.22e-4 by default in the
    reference runs) makes pydftb reproduce DFTB+'s approximate forces.
    """
    case = CASES_BY_NAME[doc["case"]]
    inp = doc["input"]
    t0 = time.perf_counter()
    res = run_pydftb(case, inp, solver=solver, fd_delta=fd_delta)
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
    }
    tol = doc["tolerances"]
    ok = (all(dev[k] <= tol[k] for k in ("energy", "forces", "charges"))
          and max(dev["free_energy"], dev["e_zero"]) <= tol["energy"]
          and dev["stress"] <= tol.get("stress", 1e-7))
    return {"case": doc["case"], "dev": dev, "ok": ok, "wall": wall, "n_scc": res.n_scc,
            "result": res,
            "n_scc_ref": ref.get("n_scc"), "natoms": len(inp["labels"])}


def _neutral_populations(case: Case, inp: dict) -> np.ndarray:
    params_dir = SKF_SETS[case.skf_set]
    base = case.typed or {}
    p = ParamSet(params_dir, max_l={base.get(k, k): v for k, v in inp["max_l"].items()})
    sk = p.skset([base.get(lab, lab) for lab in dict.fromkeys(inp["labels"])])
    return np.array([sk.occupation[base.get(lab, lab)].sum() for lab in inp["labels"]])


def check(names=None, solver: str = "eigh") -> bool:
    all_ok = True
    print(f"{'case':26s} {'N':>4s} {'|dE| Ha':>9s} {'|dF| Ha/Bohr':>12s} {'|dq| e':>9s} "
          f"{'|dsigma|':>9s} {'nSCC py/ref':>11s} {'t_py':>6s}  ok")
    for case in CASES:
        if names and case.name not in names:
            continue
        path = os.path.join(REF_DIR, f"{case.name}.json")
        if not os.path.exists(path):
            print(f"{case.name:26s} (no reference JSON)")
            all_ok = False
            continue
        r = compare(load(case.name), solver)
        d = r["dev"]
        all_ok &= r["ok"]
        print(f"{case.name:26s} {r['natoms']:4d} {d['energy']:9.1e} {d['forces']:12.1e} "
              f"{d['charges']:9.1e} {d['stress']:9.1e} {r['n_scc']:>5}/{str(r['n_scc_ref']):<5} "
              f"{r['wall']:6.2f}  {'yes' if r['ok'] else 'NO'}")
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
    a = ap.parse_args(argv)
    if a.regenerate:
        regenerate(a.cases)
    elif a.check:
        sys.exit(0 if check(a.cases, a.solver) else 1)
    else:
        live(a.cases)


if __name__ == "__main__":
    main()
