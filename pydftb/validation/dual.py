"""Both engines against the frozen DFTB+ references (P2).

    pydftb dual --inputs     # the DFTB+ engine's input for every frozen case (no dftb+ needed)
    pydftb dual --pydftb     # the pydftb engine (make_calculator) vs refgen's direct DFTB path
    pydftb dual --live       # the DFTB+ engine run live vs the frozen references, and the two
                             # engines' results vs each other (needs dftb+)
    pydftb dual --drivers    # ASE's drivers on the DFTB+ engine vs DFTB+'s own (34 cases)

(``python -m pydftb.validation.dual ...`` is the same.)  Every frozen case is translated by
:func:`case_options` into the engine-neutral options of :func:`pydftb.engines.make_calculator`
(PyDFTB's names and units).  ``--inputs`` compares the inputs DFTB+ would read: the children of
a block in any order (same-named ones in order), numbers to 1e-12 relative, SK files by name;
the single points, the optimisations and MD runs (DFTB+'s drivers), the band runs and the
SecondDerivatives runs.  The development and ELSI cases need ``$DFTBPLUS_DEV`` and
``$DFTBPLUS_ELSI`` for ``--live`` (they are skipped otherwise).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import tempfile
import time
import traceback
import warnings

import numpy as np
from ase import Atoms, units

from ..advisor import hsd as _hsd
from ..constants import BOHR, HARTREE
from ..system import element_of_label
from . import dftbplus as _dp
from . import refgen
from .cases import CASES, CASES_BY_NAME, DATA, REF_DIR, SKF_SETS

FD_DELTA = float(np.finfo(float).eps) ** 0.25
# Casida's state of interest: DFTB+'s driver energy is E0 + w (dynamics), its MD the same
COMPAT_FLAGS = {"gamma_q_in_compat": "_dftbp_gamma_q_in",
                "stale_blocks_compat": "_dftbp_stale_blocks"}


# ---- the translation ---------------------------------------------------------------------------
def case_atoms(inp: dict, charge: float = 0.0) -> Atoms:
    """The ASE atoms of a frozen input (elements of the type labels; ``charge`` on atom 0)."""
    periodic = inp["cell_angstrom"] is not None
    atoms = Atoms([element_of_label(lab) for lab in inp["labels"]],
                  positions=np.array(inp["positions_angstrom"]),
                  cell=np.array(inp["cell_angstrom"]) if periodic else None, pbc=periodic)
    if charge:
        q = np.zeros(len(atoms))
        q[0] = charge
        atoms.set_initial_charges(q)
    return atoms


def sk_dir(case, tmp: str) -> str:
    """The SK directory of a case (its typed copies in ``tmp`` when it has type labels)."""
    if not case.typed:
        return SKF_SETS[case.skf_set]
    from ..dftbplus_calc import plain_sk_dir
    from ..skf import write_typed_copies
    typed = os.path.join(tmp, "typed")
    if not os.path.isdir(typed):
        plain = plain_sk_dir(SKF_SETS[case.skf_set], os.path.join(tmp, "plain"))
        write_typed_copies(plain, typed, case.typed)
    return typed


def case_options(case, inp: dict, tmp: str, engine: str = "dftbplus"):
    """(atoms, skdir, options, properties, notes): a frozen case as ``make_calculator``'s
    options (PyDFTB's names and units) for ``engine``; ``tmp`` holds files the options name
    (typed SK copies, the charge file of a restart)."""
    notes = []
    atoms = case_atoms(inp, case.charge)
    skdir = sk_dir(case, tmp)
    labels = inp["labels"]
    periodic = inp["cell_angstrom"] is not None
    sel = case.selected_shells or {}
    o = {"max_l": {k: v for k, v in inp["max_l"].items() if k not in sel},
         "labels": list(labels), "smearing": tuple(case.filling), "scc": case.scc,
         "scc_tol": case.scc_tol, "max_scc": case.fixed_scc or 500,
         "repulsive": case.repulsive, "shell_resolved": case.shell_resolved}
    if periodic:
        if case.folding is not None:
            o["kpts"] = {"size": tuple(case.folding), "gamma": True}
        elif inp["kpoints"] is not None:
            o["kpts"] = {"kpts": inp["kpoints"], "weights": inp["kweights"]}
            o["time_reversal"] = False
    if case.fd_delta is not None:
        o["fd_delta"] = case.fd_delta
    o.update(case.dftb3 or {})
    if case.spin:
        o.update(spin=True, **case.spin)
    if case.noncolinear:
        nc = case.noncolinear
        o.update(spin="noncolinear", initial_spins=nc["initial_spins"],
                 spin_constants={k: v for k, v in nc["spin_constants"].items() if k in labels})
    if case.spin_orbit:
        so = case.spin_orbit
        o.update(spin_orbit={k: v for k, v in so["xi"].items() if k in labels},
                 spin_orbit_dual=so.get("dual", True))
    if inp.get("dispersion") is not None:
        o["dispersion"] = inp["dispersion"]
    o.update(refgen.block_options(case, labels))
    if case.efield:
        o["electric_field"] = dict(case.efield)
    if case.point_charges:
        o["point_charges"] = dict(case.point_charges)
    if case.site_potentials:
        o["site_potentials"] = case.site_potentials
    if case.solvation:
        spec = dict(case.solvation)
        if "param_file" in spec:
            spec["param_file"] = os.path.join(DATA, spec["param_file"])
        o["solvation"] = spec
        o["rescale_solvated_fields"] = case.rescale_fields
    if case.casida:
        c = case.casida
        o["casida"] = {"n_excitations": c["n"], "symmetry": c.get("symmetry") or "singlet",
                       "state_of_interest": c.get("state", 0),
                       "energy_window": c.get("energy_window"),
                       "oscillator_window": c.get("oscillator_window"),
                       "solver": ("arpack" if str(c.get("solver", "")).lower() == "arpack"
                                  else "auto"),
                       "excited_forces": bool(c.get("excited_forces")),
                       "excited_charges": bool(c.get("excited_charges")),
                       "transition_charges": bool(c.get("transition_charges"))}
        if c.get("spin_constants") and not case.spin:
            o["spin_constants"] = c["spin_constants"]
    o.update(refgen.scc_start_options(case))
    if case.read_charges is not None and "charges_file" in inp:
        name = "charges.dat" if inp.get("charges_text") else "charges.bin"
        path = os.path.join(tmp, name)
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(inp["charges_file"]))
        o["read_charges"] = path
    fermi = case.fermi or {}
    if fermi.get("fixed") is not None:
        o["fixed_fermi_level"] = (np.asarray(fermi["fixed"], float) * HARTREE).tolist()
    if fermi.get("independent_k"):
        o["independent_k_filling"] = True
    if case.hubbard:
        o["hubbard"] = {k: v for k, v in case.hubbard.items() if k in labels}
    if case.occupations:
        o["reference_occupations"] = case.occupations
    if case.truncate_sk:
        t = dict(case.truncate_sk)
        o["truncate_sk_range"] = {**t, "max_distance": t["max_distance"] * BOHR}
    o.update(refgen.corrections_ase_options(case))
    if case.analysis:
        spec = {k: v for k, v in case.analysis.items() if k in refgen.ANALYSIS_KEYS}
        if engine == "dftbplus" and not case.analysis.get("dftbplus_coord", True):
            spec.pop("coord_derivatives", None)
            notes.append("coord_derivatives: refused or wrong in DFTB+ (dftbplus_coord False)")
        if spec:
            o["analysis"] = spec
    if case.selected_shells:
        o["selected_shells"] = case.selected_shells
    o.update(refgen.hybrid_ase_options(case))
    if case.solver is not None:
        o["solver"] = case.solver
    if engine == "dftbplus":
        o["ewald_tolerance"] = case.ewald_tolerance       # None: DFTB+'s, as the references
        if case.fixed_scc:
            o["convergent_scc_only"] = False
        if case.dftbplus in ("dev", "elsi"):
            try:
                o["binary"] = refgen.dftbplus_binary(case)
            except FileNotFoundError as err:       # the inputs need no binary (--inputs)
                notes.append(str(err))
        if case.regions:
            o["regions"] = case.regions
        forces = not (case.casida and periodic) and case.dftbp_forces
    else:
        if case.solver is None:
            o["solver"] = "eigh"                  # refgen's (PyDFTB's default: batched)
        forces = case.pydftb_forces
    properties = ("energy", "forces") if forces else ("energy",)
    if case.type_switching:
        notes.append("type switching: pydftb only (the fixed type labels here)")
    for flag in COMPAT_FLAGS:
        if getattr(case, flag):
            notes.append(flag)
    return atoms, skdir, o, properties, notes


def split_options(options: dict, engine: str):
    """``make_calculator``'s (common, own section) of translated options."""
    from ..engines import OPTIONS
    other = "pydftb" if engine == "dftbplus" else "dftbplus"
    own = {k: v for k, v in options.items() if not OPTIONS[k][other]}
    return {k: v for k, v in options.items() if k not in own}, own


def make(engine: str, skdir: str, options: dict):
    from ..engines import make_calculator
    common, own = split_options(options, engine)
    return make_calculator(skdir, engine, **{engine: own}, **common)


# ---- comparing inputs --------------------------------------------------------------------------
_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?$")


def _tok_equal(a: str, b: str, rtol: float) -> bool:
    if a == b:
        return True
    if _NUM.match(a) and _NUM.match(b):
        x = float(a.replace("d", "e").replace("D", "e"))
        y = float(b.replace("d", "e").replace("D", "e"))
        return abs(x - y) <= rtol * max(abs(x), abs(y))
    return a.lower() == b.lower() and a.lower() in ("yes", "no")


def _norm_value(name, value):
    if value is None:
        return None
    if name.lower() == "prefix":
        return "<SKDIR>/"
    if name.lower() == "paramfile":
        return os.path.basename(value)
    return value


def hsd_diff(a: str, b: str, rtol: float = 1e-12, allow=()) -> list:
    """What DFTB+ would read differently in two HSD texts (paths like ``/analysis/printforces``
    in ``allow`` are left out): the children of a block in any order, same-named ones in order;
    numbers to ``rtol``; SK and parameter files by name."""
    out = []

    def toks(text):
        return [t[1] for t in _hsd._tokens(text) if t[0] != "nl"] if text else []

    def walk(x, y, path):
        if path in allow:
            return
        if (x.modifier or "").lower() != (y.modifier or "").lower():
            out.append(f"{path}: modifier {x.modifier!r} != {y.modifier!r}")
        va, vb = _norm_value(x.name, x.value), _norm_value(y.name, y.value)
        if path.startswith("/hamiltonian/dftb/slaterkosterfiles/") and va and vb:
            va = " ".join(os.path.basename(t.strip('"')) for t in va.split())
            vb = " ".join(os.path.basename(t.strip('"')) for t in vb.split())
        ka, kb = toks(va), toks(vb)
        if len(ka) != len(kb) or not all(_tok_equal(p, q, rtol) for p, q in zip(ka, kb)):
            out.append(f"{path}: {va!r} != {vb!r}")
        da = [t for _, ts in x.data for t in ts]
        db = [t for _, ts in y.data for t in ts]
        if len(da) != len(db) or not all(_tok_equal(p, q, rtol) for p, q in zip(da, db)):
            out.append(f"{path}: data differ")
        ca, cb = {}, {}
        for node, d in ((x, ca), (y, cb)):
            seen = {}
            for c in node.children:
                key = c.name.lower()
                d[(key, seen.get(key, 0))] = c
                seen[key] = seen.get(key, 0) + 1
        for key in sorted(set(ca) | set(cb)):
            name = f"{path}/{key[0]}" + (f"[{key[1]}]" if key[1] else "")
            if name in allow:
                continue
            if key not in ca:
                out.append(f"{name}: only in the second")
            elif key not in cb:
                out.append(f"{name}: only in the first")
            else:
                walk(ca[key], cb[key], name)

    walk(_hsd.parse(a), _hsd.parse(b), "")
    return out


def _band_klines(case):
    return {"klines": [(int(n), tuple(float(x) for x in k)) for n, k in case.klines]}


def case_inputs(case, doc: dict, tmp: str) -> list:
    """[(run, Input, frozen HSD, allowed differences)] of a frozen refgen case: the single point
    and, when it has them, its band run and its SecondDerivatives run."""
    atoms, skdir, opts, props, _ = case_options(case, doc["input"], tmp)
    calc = make("dftbplus", skdir, opts)
    labels = calc._labels(atoms)
    main = calc._input(atoms, props, charges=calc._start_charges(labels))
    out = [("single point", main, doc["hsd"], ())]
    if case.klines:
        calc.input = main
        band = calc.band_input(atoms, _band_klines(case), None, ("charges.bin", b""))
        # the frozen band run prints forces it does not need; DFTB+ ignores the flag otherwise
        out.append(("band run", band, doc["band_hsd"], ("/analysis/printforces",)))
    if case.hessian is not None:
        spec = case.hessian
        h = calc.hessian_input(atoms, spec.get("delta", 1e-4), spec.get("atoms"),
                               spec.get("moved_atoms"),
                               polarisability=bool(spec.get("polarisability")))
        out.append(("SecondDerivatives", h, doc["reference"]["hessian"]["hsd"], ()))
    return out


def dynamics_input(dcase, doc: dict, tmp: str):
    """(Input, frozen HSD) of a frozen dynamics case: DFTB+'s own driver on the base case."""
    from . import dynamics
    base = CASES_BY_NAME[dcase.base]
    inp = doc["input"]
    atoms, skdir, opts, _, _ = case_options(base, inp, tmp)
    opts["ewald_tolerance"] = dcase.ewald_tolerance or base.ewald_tolerance
    opts.update(scc_tol=1e-10, max_scc=500)        # dynamics.run_dftbplus: make_hsd's defaults
    calc = make("dftbplus", skdir, opts)
    if dcase.kind == "opt":
        i = calc.relax_input(atoms, fmax=dcase.grad_elem * dynamics.HA_BOHR, steps=3000,
                             cell=dcase.lattice, isotropic=dcase.isotropic,
                             optimiser=dcase.optimiser)
    else:
        i = calc.md_input(atoms, dcase.steps, dcase.time_step,
                          velocities=np.array(inp["velocities_aaps"]), xlbomd=dcase.xlbomd,
                          force_evaluation=dcase.force_evaluation)
    return i, doc["hsd"]


def _gen(labels, pos, cell) -> str:
    with tempfile.TemporaryDirectory(prefix="pydftb_dual_") as tmp:
        path = os.path.join(tmp, "geo.gen")
        _dp.write_gen(path, labels, np.asarray(pos), None if cell is None else np.asarray(cell))
        with open(path) as fh:
            return fh.read()


def _byte_norm(text: str) -> str:
    text = re.sub(r'Prefix = "[^"]*"', 'Prefix = "<SK>/"', text)
    text = re.sub(r'"/[^"]*/([^/"]+\.skf)"', r'"<SK>/\1"', text)
    return re.sub(r'ParamFile = "[^"]*/([^/"]+)"', r'ParamFile = "\1"', text)


def check_inputs(names=None, verbose: bool = True) -> dict:
    """V1: every frozen input against the DFTB+ engine's; {run: differences}."""
    from . import dynamics
    report, identical = {}, 0
    for case in CASES:
        if names and case.name not in names:
            continue
        path = os.path.join(REF_DIR, f"{case.name}.json")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            doc = json.load(fh)
        with tempfile.TemporaryDirectory(prefix="pydftb_dual_") as tmp:
            try:
                runs = case_inputs(case, doc, tmp)
            except Exception as err:          # noqa: BLE001
                report[case.name] = [f"error: {type(err).__name__}: {err}"]
                continue
            for run, i, frozen, allow in runs:
                key = case.name if run == "single point" else f"{case.name} ({run})"
                diff = hsd_diff(i.hsd, frozen, allow=allow)
                if run == "single point":
                    g = _gen(doc["input"]["labels"], doc["input"]["positions_angstrom"],
                             doc["input"]["cell_angstrom"])
                    if i.gen() != g:
                        diff.append("geo.gen differs")
                    identical += _byte_norm(i.hsd) == _byte_norm(frozen)
                report[key] = diff
    for dcase in dynamics.DYN_CASES:
        if names and dcase.name not in names:
            continue
        path = os.path.join(dynamics.REF_DIR, f"{dcase.name}.json")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            doc = json.load(fh)
        with tempfile.TemporaryDirectory(prefix="pydftb_dual_") as tmp:
            try:
                i, frozen = dynamics_input(dcase, doc, tmp)
                report[f"{dcase.name} (driver)"] = hsd_diff(i.hsd, frozen)
            except Exception as err:          # noqa: BLE001
                report[f"{dcase.name} (driver)"] = [f"error: {type(err).__name__}: {err}"]
    bad = {k: v for k, v in report.items() if v}
    if verbose:
        print(f"{len(report)} DFTB+ inputs, {len(report) - len(bad)} as DFTB+ reads the frozen "
              f"ones; single points byte-identical (SK paths aside): {identical}")
        for k, v in bad.items():
            print(f"  {k}: " + "; ".join(v[:4]))
    return report


# ---- the pydftb engine ------------------------------------------------------------------------
_FIELDS = ("energy", "free_energy", "e_band", "e_scc", "e_rep", "ts", "fermi", "e_3rd", "e_spin",
           "e_disp", "e_ls", "e_plus_u", "e_onsite", "e_ext", "e_solv", "e_halogen", "e_fock",
           "excitation_energy", "excited_energy")
_ARRAYS = ("charges", "forces", "stress", "magmoms", "dipole", "point_charge_forces",
           "eigenvalues", "occupations", "scc_errors", "q_in")


def _diff(a, b) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.shape != y.shape:
        return float("inf")
    return float(np.abs(x - y).max()) if x.size else 0.0


class _compat:
    """DFTB+ 25.1 compatibility switches of a case (refgen sets them on its DFTB object; the
    engine builds its own, so they are set on the class for the run)."""

    def __init__(self, case):
        from ..calculator import DFTB
        self.cls = DFTB
        self.attrs = [a for f, a in COMPAT_FLAGS.items() if getattr(case, f)]
        self.saved = {}

    def __enter__(self):
        for a in self.attrs:
            self.saved[a] = self.cls.__dict__.get(a)
            setattr(self.cls, a, True)

    def __exit__(self, *exc):
        for a, v in self.saved.items():
            setattr(self.cls, a, v if v is not None else False)


def _pydftb_one(name):
    with open(os.path.join(REF_DIR, f"{name}.json")) as fh:
        doc = json.load(fh)
    case = CASES_BY_NAME[name]
    inp = doc["input"]
    rec = {"case": name}
    if refgen.missing_requirement(doc):
        rec["skipped"] = refgen.missing_requirement(doc)
        return rec
    with tempfile.TemporaryDirectory(prefix="pydftb_dual_") as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            t0 = time.perf_counter()
            ref = refgen.run_pydftb(case, inp, fd_delta=case.fd_delta)
            t1 = time.perf_counter()
            atoms, skdir, opts, props, notes = case_options(case, inp, tmp, engine="pydftb")
            props = list(props)
            if atoms.pbc.any() and "forces" in props and case.hybrid is None:
                props.append("stress")
            with _compat(case):
                calc = make("pydftb", skdir, opts)
                calc.calculate(atoms, props)
            r = calc.raw
            dev = {k: abs((getattr(r, k) or 0.0) - (getattr(ref, k) or 0.0)) for k in _FIELDS}
            dev.update({k: _diff(getattr(r, k), getattr(ref, k)) for k in _ARRAYS})
            dev["n_scc"] = abs(r.n_scc - ref.n_scc)
            rec.update(dev=dev, notes=notes, t_refgen=t1 - t0, t=time.perf_counter() - t1)
        except Exception as err:          # noqa: BLE001
            rec.update(error=f"{type(err).__name__}: {str(err)[:400]}",
                       trace=traceback.format_exc()[-2000:])
    return rec


def _pool(fn, names, workers):
    if workers <= 1:
        return [fn(n) for n in names]
    from multiprocessing import get_context
    with get_context("fork").Pool(workers) as pool:
        return pool.map(fn, names, chunksize=1)


def check_pydftb(names=None, workers: int = 1, tol: float = 1e-11, verbose: bool = True) -> list:
    """V2: the pydftb engine through ``make_calculator`` against refgen's direct path (the same
    DFTB settings): every field to ``tol`` (absolute; most agree bit for bit)."""
    names = names or [c.name for c in CASES if os.path.exists(os.path.join(REF_DIR,
                                                                           f"{c.name}.json"))]
    recs = _pool(_pydftb_one, names, workers)
    if verbose:
        same = [r for r in recs if "dev" in r and not any(r["dev"].values())]
        bad = [r for r in recs if "dev" in r and max(r["dev"].values()) > tol]
        err = [r for r in recs if "error" in r]
        worst = max(((max(r["dev"].values()), r["case"]) for r in recs if "dev" in r),
                    default=(0.0, "-"))
        print(f"{len(recs)} cases: {len(same)} bit for bit, largest difference {worst[0]:.1e} "
              f"({worst[1]}), {len(bad)} above {tol:g}, {len(err)} errors, "
              f"{sum('skipped' in r for r in recs)} skipped")
        for r in bad + err:
            print(f"  {r['case']}: " + (r.get("error") or
                                        ", ".join(f"{k} {v:.1e}" for k, v in r["dev"].items()
                                                  if v > tol)))
    return recs


# ---- the DFTB+ engine live -------------------------------------------------------------------
_SKIP = {"run1", "charges_file", "stress_fd", "forces_fd", "field_fd", "hessian", "regions",
         "band", "levels", "dp_dos", "exc_tags", "scc_eigenvalues", "atq_file", "transq_sqr",
         "excited_charges_net", "excited_dipole_debye"}


def _maxdiff(a, b) -> float:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return float("inf")
        return max([_maxdiff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, str) or isinstance(b, str):
        return 0.0 if a == b else float("inf")
    if a is None or b is None:
        return 0.0 if a is None and b is None else float("inf")
    try:
        x, y = np.asarray(a, float), np.asarray(b, float)
    except (TypeError, ValueError):
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
            return max([_maxdiff(p, q) for p, q in zip(a, b)] or [0.0])
        return 0.0 if a == b else float("inf")
    if x.shape != y.shape:
        return float("inf")
    return float(np.nanmax(np.abs(x - y))) if x.size else 0.0


def _live_one(name):
    with open(os.path.join(REF_DIR, f"{name}.json")) as fh:
        doc = json.load(fh)
    case = CASES_BY_NAME[name]
    inp, ref = doc["input"], doc["reference"]
    rec = {"case": name}
    why = refgen.missing_requirement(doc)
    if why is None and case.dftbplus in ("dev", "elsi"):
        try:
            refgen.dftbplus_binary(case)
        except FileNotFoundError as err:
            why = str(err)
    if why:
        rec["skipped"] = why
        return rec
    with tempfile.TemporaryDirectory(prefix="pydftb_dual_") as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            atoms, skdir, opts, props, notes = case_options(case, inp, tmp)
            props = list(props)
            if atoms.pbc.any() and "forces" in props and case.hybrid is None:
                props.append("stress")
            dp = make("dftbplus", skdir, opts)
            dp.calculate(atoms, props)
            out, raw = dp.output, dp.raw
            dev = {}
            for k, v in ref.items():
                if k in _SKIP:
                    continue
                if k not in out:
                    # the analysis left out for DFTB+ (its refused coordinate derivatives):
                    # the frozen run parsed nothing either
                    empty = k == "analysis" and not any(v.values())
                    dev[k] = 0.0 if empty else float("inf")
                    continue
                o = out[k].tolist() if isinstance(out[k], np.ndarray) else out[k]
                if k == "analysis":
                    common = set(o) & set(v)
                    o, v = {q: o[q] for q in common}, {q: v[q] for q in common}
                dev[k] = _maxdiff(o, v)
            # the pydftb engine on the same options, against the DFTB+ engine (V4)
            pa, pskdir, popts, pprops, _ = case_options(case, inp, tmp, engine="pydftb")
            pprops = list(pprops)
            if pa.pbc.any() and "forces" in pprops and case.hybrid is None:
                pprops.append("stress")
            with _compat(case):
                py = make("pydftb", pskdir, popts)
                py.calculate(pa, pprops)
            cross = _cross(case, ref, py.raw, raw)
            rec.update(dev=dev, cross=cross, notes=notes, wall=out["wall"],
                       version=out["version"], n_scc=out["n_scc"], n_scc_ref=ref.get("n_scc"))
        except Exception as err:          # noqa: BLE001
            rec.update(error=f"{type(err).__name__}: {str(err)[:400]}",
                       trace=traceback.format_exc()[-2000:])
    return rec


def _cross(case, ref, py, dp) -> dict:
    """The two engines' results against each other with the case's tolerances; the fields
    refgen takes from DFTB+'s differences instead of its analytic values are left out."""
    tol = dict(case.tol)
    out = {"energy": (abs(py.energy - dp.energy), tol["energy"]),
           "free_energy": (abs(py.free_energy - dp.free_energy), tol["energy"]),
           "charges": (_diff(py.charges, dp.charges), tol["charges"])}
    if py.forces is not None and dp.forces is not None and "forces_fd" not in ref:
        out["forces"] = (_diff(py.forces, dp.forces), tol["forces"])
    if py.stress is not None and dp.stress is not None and "stress_fd" not in ref:
        s_py = np.asarray(py.stress)
        s_dp = np.asarray(dp.stress)
        if case.efield:                       # DFTB+'s field stress is not symmetrised
            s_dp = 0.5 * (s_dp + s_dp.T)
        out["stress"] = (_diff(s_py, s_dp), tol.get("stress", 1e-7))
    if py.magmoms is not None and dp.magmoms is not None:
        out["magmoms"] = (_diff(py.magmoms, dp.magmoms), tol.get("spins", tol["charges"]))
    if py.dipole is not None and dp.dipole is not None:
        out["dipole"] = (_diff(py.dipole, dp.dipole), tol.get("dipole", 1e-6))
    return {k: [float(v), float(t)] for k, (v, t) in out.items()}


def check_live(names=None, workers: int = 1, verbose: bool = True) -> list:
    """V3 and V4: the DFTB+ engine run live (its outputs against the frozen reference, which the
    same DFTB+ made from the same input) and the two engines' results against each other with
    the case's tolerances."""
    names = names or [c.name for c in CASES if os.path.exists(os.path.join(REF_DIR,
                                                                           f"{c.name}.json"))]
    recs = _pool(_live_one, names, workers)
    if verbose:
        run = [r for r in recs if "dev" in r]
        same = [r for r in run if not any(v for v in r["dev"].values())]
        worst = max(((max(r["dev"].values() or [0.0]), r["case"]) for r in run),
                    default=(0.0, "-"))
        print(f"{len(recs)} cases, {len(run)} run: outputs bit for bit as frozen {len(same)}, "
              f"largest difference {worst[0]:.1e} ({worst[1]}); "
              f"{sum('skipped' in r for r in recs)} skipped, {sum('error' in r for r in recs)} "
              "errors")
        over = [(r["case"], k, v, t) for r in run for k, (v, t) in r["cross"].items() if v > t]
        print(f"  engines vs each other: {len(run) - len({c for c, *_ in over})} of {len(run)} "
              "within the case tolerances")
        for c, k, v, t in over:
            print(f"    {c}: {k} {v:.1e} (tolerance {t:.0e})")
        for r in recs:
            if "error" in r:
                print(f"  {r['case']}: {r['error']}")
    return recs


# ---- drivers -----------------------------------------------------------------------------------
def _driver_one(name):
    from ase.md.verlet import VelocityVerlet
    from ase.optimize import BFGS

    from . import dynamics
    doc = dynamics.load(name)
    dcase = dynamics.DYN_BY_NAME[name]
    base = CASES_BY_NAME[dcase.base]
    inp, ref = doc["input"], doc["reference"]
    rec = {"case": name, "kind": dcase.kind}
    with tempfile.TemporaryDirectory(prefix="pydftb_dual_") as tmp:
        try:
            atoms, skdir, opts, _, _ = case_options(base, inp, tmp)
            opts["ewald_tolerance"] = dcase.ewald_tolerance or base.ewald_tolerance
            opts.update(scc_tol=1e-10, max_scc=500)
            calc = make("dftbplus", skdir, opts)
            atoms.set_masses(calc.masses(atoms=atoms))
            native = dcase.xlbomd is not None or (base.noncolinear is not None
                                                  and base.hybrid is not None)
            n_scc = []
            t0 = time.perf_counter()
            if native and dcase.kind == "md":
                atoms.set_velocities(np.array(inp["velocities_aaps"]) / (1000.0 * units.fs))
                out = calc.md(atoms, dcase.steps, dcase.time_step, xlbomd=dcase.xlbomd,
                              force_evaluation=dcase.force_evaluation)
                pos = np.array([f["positions"] for f in out["trajectory"]])
                dev = {"positions": _diff(pos, ref["positions"]),
                       "energy": max(_diff(out["md"]["potential"], ref["potential"]),
                                     _diff(out["md"]["kinetic"], ref["kinetic"]))}
                n_scc = out["scc_iterations"]
                rec["driver"] = "DFTB+"
            else:
                calc.options["reuse_charges"] = True
                atoms.calc = calc
                if dcase.kind == "opt":
                    opt = BFGS(dynamics._target(atoms, dcase), logfile=None)
                    opt.attach(lambda: n_scc.append(calc.output["n_scc"]))
                    opt.run(fmax=dcase.grad_elem * dynamics.HA_BOHR, steps=5000)
                    e = atoms.get_potential_energy(force_consistent=True) / units.Hartree
                    dev = {"energy": abs(e - ref["free_energy"])}
                    if inp["cell_angstrom"] is None:
                        dev["positions"] = dynamics.kabsch_rmsd_max(np.array(ref["positions"]),
                                                                    atoms.positions)
                    else:
                        cref = np.array(ref["cell"])
                        frac = atoms.positions @ np.linalg.inv(atoms.cell.array)
                        d = frac - np.array(ref["positions"]) @ np.linalg.inv(cref)
                        d -= np.round(d - d[0]) + d[0]
                        dev["positions"] = float(np.abs(d @ atoms.cell.array).max())
                        dev["cell"] = float(np.abs(dynamics._cell_params(atoms.cell.array)
                                                   - dynamics._cell_params(cref))[:3].max())
                else:
                    atoms.set_velocities(np.array(inp["velocities_aaps"]) / (1000.0 * units.fs))
                    fr = {"p": [], "e": [], "k": []}

                    def record():
                        fr["p"].append(atoms.positions.copy())
                        fr["e"].append(atoms.get_potential_energy(force_consistent=True)
                                       / units.Hartree)
                        fr["k"].append(atoms.get_kinetic_energy() / units.Hartree)
                        n_scc.append(calc.output["n_scc"])

                    md = VelocityVerlet(atoms, timestep=dcase.time_step * units.fs,
                                        logfile=None)
                    md.attach(record, interval=1)
                    md.run(dcase.steps)
                    dev = {"positions": _diff(fr["p"], ref["positions"]),
                           "energy": max(_diff(fr["e"], ref["potential"]),
                                         _diff(fr["k"], ref["kinetic"]))}
                rec["driver"] = "ASE"
            k = min(len(n_scc), len(ref["n_scc"]))
            dev["n_scc"] = (int(np.abs(np.array(n_scc[:k]) - np.array(ref["n_scc"][:k])).max())
                            if k else 0)
            tol = doc["tolerances"]
            rec.update(dev=dev, tol=tol, calls=calc.calls, t=time.perf_counter() - t0,
                       t_ref=ref.get("wall"),
                       ok=all(dev[q] <= tol[q] for q in tol if q in dev and q != "n_scc"))
        except Exception as err:          # noqa: BLE001
            rec.update(error=f"{type(err).__name__}: {str(err)[:400]}",
                       trace=traceback.format_exc()[-2000:])
    return rec


def check_drivers(names=None, workers: int = 1, verbose: bool = True) -> list:
    """V5: the 34 dynamics cases on the DFTB+ engine: ASE's BFGS and velocity Verlet (charges
    reused from step to step), or DFTB+'s own driver for XL-BOMD and non-collinear hybrids
    (no charge file), against the frozen DFTB+-driver references and their tolerances."""
    from . import dynamics
    names = names or [c.name for c in dynamics.DYN_CASES]
    recs = _pool(_driver_one, names, workers)
    if verbose:
        for r in recs:
            if "error" in r:
                print(f"{r['case']:30s} ERROR {r['error'][:160]}")
                continue
            d = r["dev"]
            print(f"{r['case']:30s} {r['kind']:3s} {r['driver']:5s} dE {d['energy']:.1e} "
                  f"dr {d['positions']:.1e} SCC/step diff {d['n_scc']} {r['calls']} runs "
                  f"{r['t']:.1f} s  {'ok' if r['ok'] else 'NO'}")
        print(f"{sum(r.get('ok', False) for r in recs)} of {len(recs)} within the tolerances")
    return recs


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inputs", action="store_true", help="V1: every frozen input (no dftb+)")
    g.add_argument("--pydftb", action="store_true",
                   help="V2: the pydftb engine vs refgen's direct path")
    g.add_argument("--live", action="store_true",
                   help="V3/V4: the DFTB+ engine live vs the frozen references and vs pydftb")
    g.add_argument("--drivers", action="store_true", help="V5: optimisations and MD")
    ap.add_argument("cases", nargs="*", help="case names (default: all)")
    ap.add_argument("--workers", type=int, default=1, help="processes (default 1)")
    ap.add_argument("--json", help="write the records to this file")
    a = ap.parse_args(argv)
    if a.inputs:
        report = check_inputs(a.cases)
        ok = not any(report.values())
        recs = report
    elif a.pydftb:
        recs = check_pydftb(a.cases, a.workers)
        ok = all("error" not in r and max(r.get("dev", {"x": 0.0}).values()) <= 1e-11
                 for r in recs)
    elif a.live:
        recs = check_live(a.cases, a.workers)
        ok = all("error" not in r and all(v <= t for v, t in r.get("cross", {}).values())
                 for r in recs)
    else:
        recs = check_drivers(a.cases, a.workers)
        ok = all(r.get("ok", False) for r in recs)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(recs, fh, indent=1, default=float)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
