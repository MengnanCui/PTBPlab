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
import base64
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
from ..casida import AU_DEBYE, Casida
from ..constants import BOHR, HARTREE, VOLT_PER_ANGSTROM_AU
from ..dispersion import library_versions
from ..external import PointCharges
from ..params import ParamSet
from ..skf import write_typed_copies
from . import dftbplus
from .cases import CASES, CASES_BY_NAME, DATA, MAXL, REF_DIR, SKF_SETS, Case, solvation_model

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
    extra = {}
    if case.read_charges is not None:       # M6f-1: run 1's geometry (the file is added later)
        extra["run1_positions_angstrom"] = case.run1_positions(atoms).tolist()
    return extra | {
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
               backend=None, device=None, dtype=None, casida_solver: str = "auto", **options):
    """Run pydftb for a case.  ``skdir`` defaults to the (compressed) test set.
    ``engine``: also return the pydftb.DFTB object (for band structures).
    ``backend``/``device``/``dtype``: the array backend (milestone M4, default NumPy);
    ``casida_solver``: the eigensolver of a Casida case (M6d);
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
                    **(case.dftb3 or {}), **({"spin": True, **case.spin} if case.spin else {}),
                    **pauli_options(case, labels), **block_options(case, labels),
                    **environment_options(case), **casida_options(case, casida_solver),
                    **scc_start_options(case), **fill_options(case, labels),
                    **corrections_options(case))
        q_init = options.pop("q_init", None)
        calc = DFTB(sk, **{**opts, **options})
        if q_init is None and case.read_charges is not None and "charges_file" in inp:
            q_init = charge_file_data(inp, calc._charge_layout(labels)[0].norb_atom)
        cell = np.array(inp["cell_angstrom"]) / BOHR if periodic else None
        res = calc.calculate(labels, np.array(inp["positions_angstrom"]) / BOHR, cell,
                             np.ones(3, bool) if periodic else None, q_init=q_init)
        return (res, calc) if engine else res
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def scc_start_options(case: Case) -> dict:
    """:class:`pydftb.DFTB` options of an M6f-1 case: mixer and initial charges (initial
    spins are in the spin dict)."""
    out = {}
    if case.mixer is not None:
        out["mixer"] = case.mixer
    if case.initial_charges is not None:
        out["initial_charges"] = case.initial_charges
    return out


def fill_options(case: Case, labels) -> dict:
    """:class:`pydftb.DFTB` options of an M6f-2 case: fixed Fermi level, independent k-point
    filling, custom Hubbard U (for these labels), reference occupations, SK truncation."""
    out = {}
    fermi = case.fermi or {}
    if fermi.get("fixed") is not None:
        out["fixed_fermi_level"] = fermi["fixed"]
    if fermi.get("independent_k"):
        out["independent_k_filling"] = True
    if case.hubbard:
        out["hubbard"] = {k: v for k, v in case.hubbard.items() if k in labels}
    if case.occupations:
        out["reference_occupations"] = case.occupations
    if case.truncate_sk:
        out["truncate_sk_range"] = dict(case.truncate_sk)
    return out


def corrections_options(case: Case) -> dict:
    """:class:`pydftb.DFTB` options of an M6f-3 case: H5, the halogen correction, the dipole
    correction (its layer position in Bohr)."""
    out = {}
    if case.h5 is not None:
        out["h5"] = case.h5
    if case.halogen is not None:
        out["halogen_correction"] = True
    if case.dipole_correction is not None:
        dc = case.dipole_correction
        if isinstance(dc, dict) and dc.get("layer_position") is not None:
            dc = {**dc, "layer_position": float(dc["layer_position"]) / BOHR}
        out["dipole_correction"] = dc
    return out


def corrections_ase_options(case: Case) -> dict:
    """:class:`pydftb.ase_calc.PyDFTB` options of an M6f-3 case (Angstrom)."""
    out = {}
    if case.h5 is not None:
        out["h5"] = case.h5
    if case.halogen is not None:
        out["halogen_correction"] = True
    if case.dipole_correction is not None:
        out["dipole_correction"] = case.dipole_correction
    return out


def dftbplus_binary(case: Case) -> str | None:
    """The DFTB+ executable of a case: the development build ($DFTBPLUS_DEV) for what 25.1
    does not have (M6f-3), else the default."""
    if case.dftbplus == "dev":
        path = os.environ.get("DFTBPLUS_DEV")
        if not path:
            raise FileNotFoundError(f"{case.name} needs the development DFTB+: set $DFTBPLUS_DEV")
        return path
    return None


def fermi_hsd(case: Case) -> dict | None:
    """DFTB+'s Filling options of an M6f-2 case (two fixed levels with collinear spin)."""
    if not case.fermi:
        return None
    f = dict(case.fermi)
    if f.get("fixed") is not None and case.spin and np.size(f["fixed"]) == 1:
        f["fixed"] = [float(np.ravel(f["fixed"])[0])] * 2
    return f


def charge_file_data(inp: dict, norb_atom):
    """The DFTB+ charge file frozen in a ``read_charges`` case's input (M6f-1)."""
    from ..dftbplus_io import read_charges
    text = bool(inp.get("charges_text"))
    with tempfile.TemporaryDirectory(prefix="pydftb_charges_") as tmp:
        path = os.path.join(tmp, "charges.dat" if text else "charges.bin")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(inp["charges_file"]))
        return read_charges(path, norb_atom, text=text)


def pauli_options(case: Case, labels) -> dict:
    """:class:`pydftb.DFTB` options of a two-component case (M6a), for these type labels."""
    out = {}
    if case.noncolinear:
        nc = case.noncolinear
        out.update(spin="noncolinear", initial_spins=nc["initial_spins"],
                   spin_constants={k: v for k, v in nc["spin_constants"].items() if k in labels})
    if case.spin_orbit:
        so = case.spin_orbit
        out.update(spin_orbit={k: v for k, v in so["xi"].items() if k in labels},
                   spin_orbit_dual=so.get("dual", True))
    return out


def block_options(case: Case, labels) -> dict:
    """:class:`pydftb.DFTB` options of a DFTB+U / onsite-correction case (M6b)."""
    out = {}
    if case.plus_u:
        out.update(plus_u={k: v for k, v in case.plus_u["groups"].items() if k in labels},
                   plus_u_functional=case.plus_u.get("functional", "fll"))
    if case.onsite:
        out["onsite_corrections"] = {k: v for k, v in case.onsite.items() if k in labels}
    return out


def block_hsd(case: Case, inp: dict) -> str:
    """DFTB+ ``OrbitalPotential`` and ``OnSiteCorrection`` blocks of a case (M6b)."""
    shells = {lab: list(range(inp["max_l"][lab] + 1)) for lab in inp["max_l"]}
    return (dftbplus.orbital_potential_block(case.plus_u, shells)
            + dftbplus.onsite_block(case.onsite, list(dict.fromkeys(inp["labels"]))))


def environment_options(case: Case) -> dict:
    """:class:`pydftb.DFTB` options of a field / point-charge / site-potential / solvation
    case (M6c), converted with DFTB+'s constants; also the net charge."""
    out = {}
    if case.charge:
        out["charge"] = case.charge
    if case.efield:
        d = np.asarray(case.efield["direction"], float)
        out["electric_field"] = case.efield["strength"] * VOLT_PER_ANGSTROM_AU * d / np.linalg.norm(d)
    if case.point_charges:
        pc = case.point_charges
        out["point_charges"] = PointCharges(np.asarray(pc["positions"], float) / BOHR,
                                            np.asarray(pc["charges"], float),
                                            np.full(len(pc["charges"]), pc.get("blur", 0.0) / BOHR))
    if case.site_potentials:
        out["site_potentials"] = {kind: {int(a): v / HARTREE for a, v in sites.items()}
                                  for kind, sites in case.site_potentials.items()}
    if case.solvation:
        out["solvation"] = solvation_model(case.solvation)
        out["rescale_solvated_fields"] = case.rescale_fields
    return out


def environment_ase_options(case: Case) -> dict:
    """:class:`pydftb.ase_calc.PyDFTB` options of an M6c case, in ASE's units (V/A, A, eV)."""
    out = {}
    if case.efield:
        d = np.asarray(case.efield["direction"], float)
        out["electric_field"] = case.efield["strength"] * d / np.linalg.norm(d)
    if case.point_charges:
        pc = case.point_charges
        out["point_charges"] = (pc["positions"], pc["charges"], pc.get("blur", 0.0))
    if case.site_potentials:
        out["site_potentials"] = case.site_potentials
    if case.solvation:
        out["solvation"] = solvation_model(case.solvation)
        out["rescale_solvated_fields"] = case.rescale_fields
    return out


def casida_options(case: Case, solver: str = "auto") -> dict:
    """:class:`pydftb.DFTB` options of a Casida case (M6d): the forces of the state of interest
    when the spec asks for them ("excited_forces", M6e; else the ground state's), and the spin
    constants of spin-restricted triplets; ``solver`` is pydftb's (the case's "solver" is
    DFTB+'s)."""
    spec = case.casida
    if not spec:
        return {}
    out = {"casida": Casida(n_excitations=spec["n"], symmetry=spec.get("symmetry") or "singlet",
                            state_of_interest=spec.get("state", 0),
                            energy_window=spec.get("energy_window"),
                            oscillator_window=spec.get("oscillator_window"),
                            solver=solver, excited_forces=bool(spec.get("excited_forces")),
                            excited_charges=bool(spec.get("excited_charges")))}
    if spec.get("spin_constants") and not case.spin:
        out["spin_constants"] = spec["spin_constants"]
    return out


def environment_hsd(case: Case) -> str:
    """DFTB+ ``ElectricField``, ``AtomSitePotential`` and ``Solvation`` blocks (M6c)."""
    return (dftbplus.electric_field_block(case.efield, case.point_charges)
            + dftbplus.site_potential_block(case.site_potentials)
            + dftbplus.solvation_block(case.solvation, case.rescale_fields, DATA))


def run_dftbplus(case: Case, inp: dict, threads: int = 1) -> dict:
    with tempfile.TemporaryDirectory(prefix="pydftb_ref_") as tmp:
        skdir = _plain_skf_dir(case, tmp)
        periodic = inp["cell_angstrom"] is not None
        kpts = None
        if periodic and inp["kpoints"] is not None:
            kpts = (np.array(inp["kpoints"]), np.array(inp["kweights"]))
        extra = (dftbplus.dftb3_block(case.dftb3, set(inp["labels"]))
                 + dftbplus.spin_block(case.spin, set(inp["labels"]), case.shell_resolved)
                 + dftbplus.noncolinear_block(case.noncolinear, set(inp["labels"]),
                                              case.shell_resolved)
                 + dftbplus.spinorbit_block(case.spin_orbit, {lab: inp["max_l"][lab] + 1
                                                              for lab in inp["max_l"]})
                 + dftbplus.dispersion_block(inp.get("dispersion"), inp["labels"])
                 + block_hsd(case, inp) + environment_hsd(case)
                 + dftbplus.mixer_block(case.mixer)
                 + dftbplus.initial_charges_block(case.initial_charges)
                 + dftbplus.custom_hubbards_block(case.hubbard, set(inp["labels"]))
                 + dftbplus.custom_occupations_block(case.occupations)
                 + dftbplus.truncate_block(case.truncate_sk)
                 + dftbplus.corrections_block(case.h5, case.halogen, case.dipole_correction,
                                              inp["labels"]))
        if case.casida and case.casida.get("spin_constants") and not case.spin:
            extra += dftbplus.spin_constants_block(case.casida["spin_constants"],
                                                   set(inp["labels"]))
        if case.ewald_tolerance is not None:
            extra += f"  EwaldTolerance = {dftbplus._f(case.ewald_tolerance)}\n"
        args = (inp["labels"], np.array(inp["positions_angstrom"]),
                np.array(inp["cell_angstrom"]) if periodic else None)
        # DFTB+ has no excited-state stress: no forces for a periodic Casida case
        forces = not (case.casida and periodic)
        kw = dict(skdir=skdir, max_l=inp["max_l"], kpts=kpts, scc=case.scc, filling=case.filling,
                  shell_resolved=case.shell_resolved, repulsive=case.repulsive, extra_ham=extra,
                  scc_tol=case.scc_tol, threads=threads, charge=case.charge,
                  binary=dftbplus_binary(case),
                  **({"extra_top": dftbplus.casida_block(case.casida, forces), "forces": forces,
                      "autotest": True} if case.casida else {}),
                  **({"fermi": fermi_hsd(case)} if case.fermi else {}))
        if case.read_charges is not None:             # M6f-1: two runs
            return dftbplus.run_restart(args[0], np.array(inp["run1_positions_angstrom"]),
                                        args[1], args[2],
                                        text=bool(case.read_charges.get("text")), **kw)
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
    full = 1.0 if band.nspin == 2 or band.two_component else 2.0
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
        if case.read_charges is not None:   # the file run 2 started from is part of its input
            inp["charges_file"] = ref["charges_file"]
            inp["charges_text"] = bool(case.read_charges.get("text"))
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
                         "scc_tol": case.scc_tol, "ewald_tolerance": case.ewald_tolerance,
                         **({"noncolinear": case.noncolinear, "spin_orbit": case.spin_orbit,
                             "time_reversal": case.time_reversal} if case.pauli else {}),
                         **({"plus_u": case.plus_u, "onsite": case.onsite} if case.block else {}),
                         **({"efield": case.efield, "point_charges": case.point_charges,
                             "site_potentials": case.site_potentials,
                             "solvation": case.solvation, "rescale_fields": case.rescale_fields,
                             "charge": case.charge} if case.environment else {}),
                         **({"casida": case.casida, "charge": case.charge}
                            if case.casida else {}),
                         **({"mixer": case.mixer, "initial_charges": case.initial_charges,
                             "read_charges": case.read_charges, "charge": case.charge}
                            if case.scc_start else {}),
                         **({"fermi": case.fermi, "hubbard": case.hubbard,
                             "occupations": case.occupations, "truncate_sk": case.truncate_sk,
                             "stress_fd": case.stress_fd, "charge": case.charge}
                            if case.overrides else {}),
                         **({"h5": case.h5, "halogen": case.halogen,
                             "dipole_correction": case.dipole_correction,
                             "stress_fd": case.stress_fd, "dftbplus": case.dftbplus}
                            if case.corrections else {})},
            "libraries": ({lib: library_versions()[lib]} if (lib := dispersion_library(inp))
                          else {}),
            "input": inp,
            "hsd": ref["hsd"],
            "reference": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ref.items()
                          if k in ("energy", "free_energy", "e_zero", "e_band", "e_scc", "e_rep",
                                   "ts", "fermi", "forces", "charges_net", "n_scc", "stress",
                                   "e_disp")
                          or case.pauli and k in ("e_spin", "e_ls", "e_3rd", "spins",
                                                  "angular_spin", "angular_orbital")
                          or case.block and k in ("e_plus_u", "e_onsite", "e_spin", "e_ls",
                                                  "e_3rd", "spins", "angular_spin",
                                                  "angular_orbital", "orbital_charges")
                          or case.environment and k in ("e_ext", "e_solv", "dipole",
                                                        "point_charge_forces", "e_spin",
                                                        "e_ls", "e_3rd", "spins")
                          or case.casida and k in ("exc_energies", "exc_symmetries",
                                                   "exc_rows", "spx_rows", "excitation_energy",
                                                   "e_spin", "spins")
                          or case.scc_start and k in ("scc_errors", "run1", "e_spin", "spins",
                                                      "e_3rd", "e_plus_u", "e_onsite")
                          or case.overrides and k in ("scc_errors", "e_spin", "spins", "e_3rd",
                                                      "e_plus_u", "e_onsite", "e_ls",
                                                      "e_force_related",
                                                      "number_of_electrons", "fermi_levels")
                          or case.corrections and k in ("scc_errors", "e_3rd", "e_halogen")},
            "tolerances": case.tol,
        }
        if case.block and not (case.klines or case.regions or case.dp_dos):
            doc["reference"]["scc_eigenvalues"] = np.asarray(ref["eigenvalues"]).tolist()
        if case.casida:
            doc["reference"]["exc_tags"] = {k: np.asarray(v).tolist()
                                            for k, v in ref["exc_tags"].items()}
            for k in ("excited_charges_net", "excited_dipole_debye"):         # M6e
                if k in ref:
                    doc["reference"][k] = np.asarray(ref[k]).tolist()
            if case.casida.get("forces_fd"):
                doc["reference"]["forces_fd"] = excited_forces_fd(case, inp, threads).tolist()
                doc["forces_fd"] = ("-d(E_Mermin + Excitation Energy)/dR of DFTB+ (autotest.tag "
                                    f"mermin_energy), 4-point central differences, h = {FD_STEP} "
                                    "Bohr")
        if case.stress_fd:                                              # M6f-2
            doc["reference"]["stress_fd"] = stress_fd(case, inp, threads,
                                                      analytic=ref.get("stress")).tolist()
            doc["stress_fd"] = ("-(1/V) dE_FR/d(strain) of DFTB+ (results.tag "
                                "forcerelated_energy), 4-point central differences of a "
                                f"symmetric strain, h = {STRESS_FD_STEP}")
            if isinstance(case.stress_fd, tuple):                       # M6f-3
                doc["stress_fd"] += (f"; components {list(case.stress_fd)} only, the others "
                                     "DFTB+'s analytic stress")
        if case.halogen and case.halogen.get("forces_fd"):            # M6f-3
            doc["reference"]["forces_fd"] = forces_fd(case, inp, threads).tolist()
            doc["forces_fd"] = ("-dE_Mermin/dR of DFTB+ (results.tag mermin_energy), 4-point "
                                f"central differences, h = {FD_STEP} Bohr")
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


FD_STEP = 1e-4          # Bohr: differences of DFTB+'s excited-state energy (M6e)


def excited_forces_fd(case: Case, inp: dict, threads: int = 1, h: float = FD_STEP) -> np.ndarray:
    """-d(E_Mermin + Omega)/dR of DFTB+ (Ha/Bohr) by 4-point central differences: where its
    analytic excited-state forces are not this derivative (M6e)."""
    pos0 = np.array(inp["positions_angstrom"])
    out = np.zeros_like(pos0)
    for a in range(len(pos0)):
        for k in range(3):
            e = {}
            for s in (-2, -1, 1, 2):
                p = pos0.copy()
                p[a, k] += s * h * BOHR
                r = run_dftbplus(case, dict(inp, positions_angstrom=p.tolist()), threads)
                e[s] = r["excited_mermin_tag"]   # autotest.tag: all digits (detailed.out: 10)
            out[a, k] = -(8.0 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12.0 * h)
    return out


STRESS_FD_STEP = 1e-4   # strain: differences of DFTB+'s force-related energy (M6f-2)


def stress_fd(case: Case, inp: dict, threads: int = 1, h: float = STRESS_FD_STEP,
              analytic=None) -> np.ndarray:
    """-(1/V) dE_FR/d(strain) of DFTB+ (Hartree/Bohr^3, its sign convention) by 4-point
    central differences of a symmetric strain (24 runs; fractional coordinates and k-points
    kept; two points leave an h^2 error of 1e-9 in a metal): where its analytic stress of a
    charged cell is not this derivative (M6f-2).  A tuple ``case.stress_fd`` limits the
    components (M6f-3: DFTB+ refuses slabs sheared out of their plane); the others are then
    ``analytic`` (DFTB+'s stress)."""
    pos0, cell0 = np.array(inp["positions_angstrom"]), np.array(inp["cell_angstrom"])
    vol = abs(np.linalg.det(cell0 / BOHR))
    comps = (case.stress_fd if isinstance(case.stress_fd, tuple)
             else [(a, b) for a in range(3) for b in range(a, 3)])
    out = np.zeros((3, 3)) if analytic is None else np.array(analytic, float)
    for a, b in comps:
        e = {}
        for sgn in (-2, -1, 1, 2):
            eps = np.zeros((3, 3))
            eps[a, b] += 0.5 * sgn * h
            eps[b, a] += 0.5 * sgn * h
            m = np.eye(3) + eps
            r = run_dftbplus(case, dict(inp, positions_angstrom=(pos0 @ m.T).tolist(),
                                        cell_angstrom=(cell0 @ m.T).tolist()), threads)
            e[sgn] = r["e_force_related"]
        de = (8.0 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12.0 * h)
        out[a, b] = out[b, a] = -de / vol
    return out


def forces_fd(case: Case, inp: dict, threads: int = 1, h: float = FD_STEP) -> np.ndarray:
    """-dE_Mermin/dR of DFTB+ (Ha/Bohr) by 4-point central differences of results.tag's
    full-precision mermin_energy: where its analytic forces are not this derivative (M6f-3:
    the halogen correction's switching region)."""
    pos0 = np.array(inp["positions_angstrom"])
    out = np.zeros_like(pos0)
    for a in range(len(pos0)):
        for k in range(3):
            e = {}
            for s in (-2, -1, 1, 2):
                p = pos0.copy()
                p[a, k] += s * h * BOHR
                r = run_dftbplus(case, dict(inp, positions_angstrom=p.tolist()), threads)
                e[s] = float(np.atleast_1d(r["mermin_tag"])[0])
            out[a, k] = -(8.0 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12.0 * h)
    return out


def load(name: str) -> dict:
    with open(os.path.join(REF_DIR, f"{name}.json")) as fh:
        return json.load(fh)


def compare(doc: dict, solver: str = "eigh", fd_delta: float | None = None,
            casida_solver: str = "auto", **backend) -> dict:
    """Run pydftb on a frozen case; return the deviations and pass/fail.

    ``fd_delta`` = DFTB+'s finite-difference step (1.22e-4 by default in the
    reference runs) makes pydftb reproduce DFTB+'s approximate forces.
    ``casida_solver``: pydftb's Casida eigensolver (M6d).
    ``backend``: ``backend``/``device``/``dtype`` of :class:`pydftb.DFTB`.
    """
    case = CASES_BY_NAME[doc["case"]]
    inp = doc["input"]
    t0 = time.perf_counter()
    res, engine = run_pydftb(case, inp, solver=solver, fd_delta=fd_delta, engine=True,
                             casida_solver=casida_solver, **backend)
    wall = time.perf_counter() - t0
    ref = doc["reference"]
    sk_q0 = _neutral_populations(case, inp)
    ref_stress = None if ref.get("stress") is None else np.array(ref["stress"])
    if ref_stress is not None and case.efield:      # DFTB+'s field stress is not symmetrised
        ref_stress = 0.5 * (ref_stress + ref_stress.T)
    if "stress_fd" in ref:                          # M6f-2: a charged cell
        ref_stress = np.array(ref["stress_fd"])
    dev = {
        "energy": abs(res.energy - ref["energy"]),
        "free_energy": abs(res.free_energy - ref["free_energy"]),
        "e_zero": abs(res.e_zero - ref["e_zero"]) if "e_zero" in ref else 0.0,
        "stress": (float(np.abs(res.stress - ref_stress).max())
                   if ref_stress is not None else 0.0),
        "forces": (float(np.abs(res.forces - np.array(ref["forces"])).max())
                   if ref.get("forces") is not None else 0.0),
        "charges": float(np.abs((sk_q0 - res.charges) - np.array(ref["charges_net"])).max()),
        "e_disp": abs(res.e_disp - ref["e_disp"]) if "e_disp" in ref else 0.0,
    }
    if "eigenvalues" in ref:
        dev.update(_compare_states(case, inp, ref, res, engine))
    if case.pauli:
        dev.update(_compare_pauli(ref, res))
    if case.block:
        block = _compare_block(ref, res, inp)
        block["e_terms"] = max(block["e_terms"], dev.get("e_terms", 0.0))
        dev.update(block)
    if case.environment:
        env = _compare_environment(case, ref, res)
        env["e_terms"] = max(env["e_terms"], dev.get("e_terms", 0.0))
        dev.update(env)
    if case.casida:
        dev.update(_compare_casida(ref, res))
        dev.update(_compare_excited(case, inp, ref, res, sk_q0, solver=solver, fd_delta=fd_delta,
                                    casida_solver=casida_solver, **backend))
    if case.track_scc:
        dev.update(_compare_scc_start(case, inp, ref, res, solver=solver, fd_delta=fd_delta,
                                      **backend))
    if case.overrides:
        dev.update(_compare_fill(case, ref, res))
    if case.corrections:
        dev.update(_compare_corrections(case, inp, ref, res, solver=solver, fd_delta=fd_delta,
                                        **backend))
    tol = doc["tolerances"]
    ok = (all(dev[k] <= tol[k] for k in ("energy", "forces", "charges"))
          and max(dev["free_energy"], dev["e_zero"], dev["e_disp"],
                  dev.get("e_terms", 0.0)) <= tol["energy"]
          and dev["stress"] <= tol.get("stress", 1e-7)
          and all(dev[k] <= tol[k] for k in STATE_KEYS if k in dev)
          and all(dev[k] <= tol[k] for k in ("spins", "moments", "orbital", "pc_forces",
                                              "dipole", "exc", "exc_osc", "exc_tdip",
                                              "exc_table", "spx", "exc_state", "exc_charges",
                                              "exc_dipole", "forces_dftbp", "n_scc",
                                              "scc_errors", "charges_file", "e_fr",
                                              "n_electrons", "fermi_level", "e_halogen")
                  if k in dev))
    return {"case": doc["case"], "dev": dev, "ok": ok, "wall": wall, "n_scc": res.n_scc,
            "result": res,
            "n_scc_ref": ref.get("n_scc"), "natoms": len(inp["labels"])}


SCC_ERROR_FLOOR = 1e-7       # M6f-1: printed SCC errors compared above this (8 digits printed)


def _compare_scc_start(case: Case, inp: dict, ref: dict, res, **run) -> dict:
    """Deviations of an M6f-1 case: the SCC iterations (exactly), the SCC error of every
    iteration (relative, where DFTB+'s printed error exceeds ``SCC_ERROR_FLOOR``), collinear
    moments, and for a charge file that its class populations are pydftb's own input
    populations of run 1."""
    dev = {"n_scc": abs(int(res.n_scc) - int(ref["n_scc"]))}
    t, e = np.array(ref["scc_errors"]), np.asarray(res.scc_errors)
    n = min(len(t), len(e))
    keep = t[:n] > SCC_ERROR_FLOOR
    dev["scc_errors"] = (float((np.abs(e[:n] - t[:n]) / t[:n])[keep].max()) if keep.any()
                         else 0.0)
    if "spins" in ref and not (case.pauli or case.block):     # collinear moments
        dev["spins"] = float(np.abs(res.magmoms - np.ravel(ref["spins"])).max())
    if case.read_charges is not None:
        inp1 = {k: v for k, v in inp.items() if k != "charges_file"}
        inp1["positions_angstrom"] = inp["run1_positions_angstrom"]
        r1, calc = run_pydftb(case, inp1, engine=True, **run)
        data = charge_file_data(inp, calc._charge_layout(inp["labels"])[0].norb_atom)
        dev["charges_file"] = float(np.abs(calc.populations_from_file(inp["labels"], data)
                                           - r1.q_in).max())
    return dev


def _compare_fill(case: Case, ref: dict, res) -> dict:
    """Deviations of an M6f-2 case: the force-related energy and the electron count of
    results.tag (per spin channel with collinear spin); with stress_fd also DFTB+'s own
    stress (for the record, not a criterion)."""
    dev = {}
    if "e_force_related" in ref:
        dev["e_fr"] = abs(res.e_force_related - ref["e_force_related"])
    if "number_of_electrons" in ref:
        n_ref = np.ravel(ref["number_of_electrons"])
        n = np.array(res.n_electrons_spin if len(n_ref) == 2 else [res.n_electrons])
        dev["n_electrons"] = float(np.abs(n - n_ref).max())
    if case.fermi and "fermi_levels" in ref:   # the given levels, or the k-weighted mean
        mine = np.ravel(res.fermi_spin if res.fermi_spin is not None else [res.fermi])
        dev["fermi_level"] = float(np.abs(mine - np.ravel(ref["fermi_levels"])).max())
    if "stress_fd" in ref and ref.get("stress") is not None:
        dev["stress_dftbp"] = float(np.abs(res.stress - np.array(ref["stress"])).max())
    return dev


def _compare_corrections(case: Case, inp: dict, ref: dict, res, **run) -> dict:
    """Deviations of an M6f-3 case: the halogen energy (detailed.out, 10 decimals); with
    ``forces_fd`` the forces against differences of DFTB+'s energy, and pydftb's forces with
    DFTB+'s sign of the halogen switching terms (``compat``) against DFTB+'s own; with a
    ``stress_fd`` also DFTB+'s own stress (for the record, not a criterion)."""
    dev = {}
    if "e_halogen" in ref:
        dev["e_halogen"] = abs(res.e_halogen - ref["e_halogen"])
    if "forces_fd" in ref and not case.casida:
        dev["forces"] = float(np.abs(res.forces - np.array(ref["forces_fd"])).max())
        if (case.halogen or {}).get("compat"):
            DFTB._halogen_compat = True
            try:
                compat = run_pydftb(case, inp, **run)
            finally:
                DFTB._halogen_compat = False
            dev["forces_dftbp"] = float(np.abs(compat.forces - np.array(ref["forces"])).max())
    if "stress_fd" in ref and ref.get("stress") is not None:
        dev["stress_dftbp"] = float(np.abs(res.stress - np.array(ref["stress"])).max())
    return dev


G_FACTOR = 2.00231930436153  # DFTB+ gfac: detailed.out prints -g/4 m as the spin momentum


def _compare_pauli(ref: dict, res) -> dict:
    """Deviations of a two-component case (M6a): the energy terms E_spin, E_LS (and E_3rd),
    the magnetisation vectors (results.tag) and the spin and orbital angular momenta per
    shell (detailed.out: -g/4 m and -<L>, 8 decimals)."""
    dev = {"e_terms": max([abs(getattr(res, k) - ref[k]) for k in ("e_spin", "e_ls", "e_3rd")
                           if k in ref] or [0.0])}
    if "spins" in ref:
        dev["spins"] = float(np.abs(res.magmoms.T - np.array(ref["spins"])).max())
    moments = []
    if "angular_spin" in ref:
        moments.append(np.abs(-0.25 * G_FACTOR * res.q_spin[1:].T
                              - np.array(ref["angular_spin"])).max())
    if "angular_orbital" in ref:
        moments.append(np.abs(-res.orbital_moments - np.array(ref["angular_orbital"])).max())
    if moments:
        dev["moments"] = float(max(moments))
    return dev


def _compare_block(ref: dict, res, inp: dict) -> dict:
    """Deviations of a DFTB+U / onsite case (M6b): the energy terms (E_U, E_onsite and those
    of spin, spin-orbit and DFTB3), the spins (collinear or vectors), every orbital's electron
    population (results.tag; collinear spin: up + down) and the eigenvalues of the SCC."""
    dev = {"e_terms": max([abs(getattr(res, k) - ref[k])
                           for k in ("e_plus_u", "e_onsite", "e_spin", "e_ls", "e_3rd")
                           if k in ref] or [0.0])}
    if "spins" in ref:
        dev["spins"] = float(np.abs(res.magmoms.T - np.array(ref["spins"])).max())
    if "orbital_charges" in ref:
        oc = np.array(ref["orbital_charges"])
        oc = oc[0] + oc[1] if len(oc) == 2 else oc[0]
        norb = [(inp["max_l"][lab] + 1) ** 2 for lab in inp["labels"]]    # shells 0..l_max
        flat = np.concatenate([oc[a, :n] for a, n in enumerate(norb)])
        dev["orbital"] = float(np.abs(res.q_orbital - flat).max())
    if "scc_eigenvalues" in ref:
        ev = np.array(ref["scc_eigenvalues"])
        dev["eigenvalues"] = float(np.abs(np.array(res.eigenvalues).reshape(ev.shape) - ev).max())
    return dev


def _compare_environment(case: Case, ref: dict, res) -> dict:
    """Deviations of a field / point-charge / site-potential / solvation case (M6c): E_ext,
    E_solv and E_SCC (with the point charges), the dipole moment (molecules; DFTB+ prints it
    times 3 eps/(2 eps + 1) in a solvent with rescaled fields) and the forces on the point
    charges."""
    dev = {"e_terms": max([abs(getattr(res, k) - ref[k])
                           for k in ("e_ext", "e_solv", "e_scc", "e_spin", "e_ls", "e_3rd")
                           if k in ref] or [0.0])}
    if res.dipole is not None and "dipole" in ref:
        scale = 1.0
        if case.solvation and case.rescale_fields and case.solvation["model"] != "sasa":
            scale = solvation_model(case.solvation).field_scaling()
        dev["dipole"] = float(np.abs(res.dipole * scale - np.array(ref["dipole"])).max())
    if "point_charge_forces" in ref:
        dev["pc_forces"] = float(np.abs(res.point_charge_forces
                                        - np.array(ref["point_charge_forces"])).max())
    return dev


def _runs(values, tol: float) -> list:
    """[start, stop) of the runs of consecutive values that differ by at most ``tol``."""
    runs, start = [], 0
    for k in range(1, len(values) + 1):
        if k == len(values) or abs(values[k] - values[k - 1]) > tol:
            runs.append((start, k))
            start = k
    return runs


def _rotation_free(x, tol: float = 1e-8, share: float = 1e-4) -> np.ndarray:
    """Per state: True if its X + Y has (practically) nothing on degenerate transitions (an
    energy that another transition of the same spin shares).  Degenerate orbitals are fixed
    only up to a rotation, which DFTB+ and pydftb choose differently; it changes the weights
    (X+Y)^2 / ||(X+Y)^2||_2 unless (sum of (X+Y)^2 over them)^2 is small against ||(X+Y)^2||^2."""
    w, spin = x.single_particle.energies, x.pairs[:, 2]
    deg = np.zeros(len(w), bool)
    for s in np.unique(spin):
        idx = np.flatnonzero(spin == s)
        idx = idx[np.argsort(w[idx], kind="stable")]
        close = np.abs(np.diff(w[idx])) <= tol
        deg[idx[1:][close]] = deg[idx[:-1][close]] = True
    sq = x.xpy ** 2
    return sq[deg].sum(axis=0) ** 2 <= share * (sq ** 2).sum(axis=0)


def _compare_casida(ref: dict, res) -> dict:
    """Casida excitations against DFTB+ (M6d): the degenerate groups of autotest.tag (Omega,
    oscillator strengths, |mu|; full precision), every state's energy from TRA.DAT, the rows
    of EXC.DAT (oscillator strength, transition, weight, KS energy, symmetry or D<S^2>;
    printed with 8 and 3 decimals) and of SPX.DAT, and the state of interest's energy.
    ``exc_table`` and ``spx`` count the printed values that differ by more than their last
    digit.  Degenerate states and orbitals are fixed only up to a rotation that the two codes
    choose differently, so the tables are compared through what does not depend on it: per
    group of degenerate states the summed oscillator strength and D<S^2> and the sorted KS
    energies, the transition and weight only of a non-degenerate state that is free of
    degenerate transitions (:func:`_rotation_free`), and in SPX.DAT the summed oscillator
    strength of every group of degenerate transitions."""
    xs = res.excitations if isinstance(res.excitations, list) else [res.excitations]
    tags = ref["exc_tags"]
    g = xs[-1].degenerate_groups()             # the tags hold the last symmetry (DFTB+)
    dev = {"exc_groups": float(len(g["omega2"]) != len(tags["omega2"]))}
    n = min(len(g["omega2"]), len(tags["omega2"]))
    dev["exc"] = float(np.abs(np.sqrt(np.abs(g["omega2"][:n]))
                              - np.sqrt(np.abs(np.array(tags["omega2"][:n])))).max())
    dev["exc_osc"] = float(np.abs(g["oscillator"][:n] - np.array(tags["oscillator"][:n])).max())
    dev["exc_tdip"] = float(np.abs(g["transdip"][:n] - np.array(tags["transdip"][:n])).max())
    energies = np.concatenate([x.energies for x in xs])
    if "exc_energies" in ref:
        e_ref = np.array(ref["exc_energies"])
        dev["exc"] = max(dev["exc"], float(np.abs(energies[:len(e_ref)] - e_ref).max())
                         if len(e_ref) == len(energies) else np.inf)
    ref_rows = ref.get("exc_rows", [])
    bad = int(len(ref_rows) != sum(len(x.omega2) for x in xs))
    offset = 0
    for x in xs if not bad else []:
        rows = ref_rows[offset:offset + len(x.omega2)]
        offset += len(x.omega2)
        free = _rotation_free(x)
        for a, b in x.degenerate_groups()["groups"]:
            grp = rows[a:b]
            bad += abs(x.oscillator_strengths[a:b].sum() - sum(r[1] for r in grp)) > 6e-9 * (b - a)
            bad += np.abs(np.sort(x.ks_energies[a:b]) * HARTREE
                          - np.sort([r[5] for r in grp])).max() > 6e-4
            if x.symmetries is None:
                bad += abs(x.spin_contamination[a:b].sum()
                           - sum(float(r[6]) for r in grp)) > 6e-4 * (b - a)
            else:
                bad += sum(s != r[6] for s, r in zip(x.symmetries[a:b], grp))
            if b - a == 1 and free[a]:
                i, v, _ = x.transitions[a]
                bad += (i + 1, v + 1) != (grp[0][2], grp[0][3])
                bad += abs(x.weights[a] - grp[0][4]) > 6e-4
    dev["exc_table"] = float(bad)
    spx_ref = ref.get("spx_rows", [])
    sp = xs[-1].single_particle
    bad = int(len(spx_ref) != len(sp.energies))
    channel = {"": 0, "U": 0, "D": 1}
    for a, b in _runs(sp.energies, 1e-8) if not bad else []:
        grp = spx_ref[a:b]
        bad += np.abs(sp.energies[a:b] * HARTREE - np.array([r[0] for r in grp])).max() > 6e-4
        bad += abs(sp.oscillator_strengths[a:b].sum() - sum(r[1] for r in grp)) > 6e-8 * (b - a)
        bad += (sorted((int(i) + 1, int(v) + 1, int(s)) for i, v, s in sp.pairs[a:b])
                != sorted((r[2], r[3], channel[r[4]]) for r in grp))
    dev["spx"] = float(bad)
    if ref.get("excitation_energy") is not None:
        dev["exc_state"] = abs(res.excitation_energy - ref["excitation_energy"])
    dev["exc_table"] += dev.pop("exc_groups")
    return dev


def _compare_excited(case: Case, inp: dict, ref: dict, res, sk_q0, **run) -> dict:
    """The state of interest against DFTB+ (M6e): its Mulliken net charges (XCH.DAT) and dipole
    (XREST.DAT, Debye; compared in e Bohr).  Where DFTB+'s excited-state forces are not
    -d(E0 + Omega)/dR (``forces_fd`` in the reference) the forces are compared with that
    derivative of DFTB+'s energies instead, and pydftb leaving out what DFTB+ leaves out (the
    case's ``compat``) with DFTB+'s forces (``forces_dftbp``, and its point-charge forces)."""
    dev = {}
    exc = res
    if "forces_fd" in ref:
        dev["forces"] = float(np.abs(res.forces - np.array(ref["forces_fd"])).max())
        DFTB._excited_compat = frozenset(case.casida.get("compat", ()))
        try:
            exc = run_pydftb(case, inp, **run)
        finally:
            DFTB._excited_compat = frozenset()
        dev["forces_dftbp"] = float(np.abs(exc.forces - np.array(ref["forces"])).max())
        if "point_charge_forces" in ref:
            dev["pc_forces"] = float(np.abs(exc.point_charge_forces
                                            - np.array(ref["point_charge_forces"])).max())
    # charges and dipole as DFTB+ computes them (with an energy window from its Z vector)
    if "excited_charges_net" in ref:
        dev["exc_charges"] = float(np.abs((sk_q0 - exc.excited_charges)
                                          - np.array(ref["excited_charges_net"])).max())
    if "excited_dipole_debye" in ref:
        dev["exc_dipole"] = float(np.abs(exc.excited_dipole - np.array(ref["excited_dipole_debye"])
                                         / AU_DEBYE).max())
    return dev


def _neutral_populations(case: Case, inp: dict) -> np.ndarray:
    params_dir = SKF_SETS[case.skf_set]
    base = case.typed or {}
    p = ParamSet(params_dir, max_l={base.get(k, k): v for k, v in inp["max_l"].items()})
    sk = p.skset([base.get(lab, lab) for lab in dict.fromkeys(inp["labels"])])
    q0 = np.array([sk.occupation[base.get(lab, lab)].sum() for lab in inp["labels"]])
    for atom, shells in (case.occupations or {}).items():     # M6f-2: CustomisedOccupations
        lab = inp["labels"][int(atom)]
        occ = sk.occupation[base.get(lab, lab)].copy()
        for name, x in shells.items():
            occ["spdf".index(name)] = x
        q0[int(atom)] = occ.sum()
    return q0


def check(names=None, solver: str = "eigh", casida_solver: str = "auto", **backend) -> bool:
    all_ok = True
    print(f"{'case':26s} {'N':>4s} {'|dE| Ha':>9s} {'|dF| Ha/Bohr':>12s} {'|dq| e':>9s} "
          f"{'|dsigma|':>9s} {'|dE_disp|':>9s} {'|deps| Ha':>9s} {'|dm|':>9s} {'nSCC py/ref':>11s} "
          f"{'t_py':>6s}  ok")
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
        if backend.get("backend") == "torch" and not case.torch:
            print(f"{case.name:26s} skipped: NumPy-only physics (M6a two-component, M6b blocks, "
                  "M6c environment, M6d excitations, M6f SCC start, mixers, occupations, "
                  "overrides and corrections)")
            continue
        r = compare(doc, solver, casida_solver=casida_solver, **backend)
        d = r["dev"]
        all_ok &= r["ok"]
        e_disp = f"{d['e_disp']:9.1e}" if "e_disp" in doc["reference"] else f"{'-':>9s}"
        eps = f"{d['eigenvalues']:9.1e}" if "eigenvalues" in d else f"{'-':>9s}"
        dm = (f"{max(d.get('spins', 0.0), d.get('moments', 0.0)):9.1e}"
              if "spins" in d or "moments" in d else f"{'-':>9s}")       # M6a: magnetisation
        dq = max(d["charges"], d.get("orbital", 0.0))     # M6b: orbital populations too
        print(f"{case.name:26s} {r['natoms']:4d} {d['energy']:9.1e} {d['forces']:12.1e} "
              f"{dq:9.1e} {d['stress']:9.1e} {e_disp} {eps} {dm} "
              f"{r['n_scc']:>5}/{str(r['n_scc_ref']):<5} {r['wall']:6.2f}  {'yes' if r['ok'] else 'NO'}")
        if "exc" in d:                                    # M6d: the excitations
            state = f", |dE_exc| {d['exc_state']:.1e} Ha" if "exc_state" in d else ""
            print(f"{'':26s}   Casida: |dw| {d['exc']:.1e} Ha, |df| {d['exc_osc']:.1e}, "
                  f"|dmu| {d['exc_tdip']:.1e} e Bohr, table/SPX differences "
                  f"{int(d['exc_table'])}/{int(d['spx'])}{state}")
        if "exc_charges" in d:                            # M6e: the state of interest
            fd = ("forces vs DFTB+'s energy differences, as DFTB+ (compat) "
                  f"{d['forces_dftbp']:.1e}; " if "forces_dftbp" in d else "")
            print(f"{'':26s}   Excited: {fd}|dq_ex| {d['exc_charges']:.1e} e, "
                  f"|dmu_ex| {d['exc_dipole']:.1e} e Bohr")
        if "scc_errors" in d:                             # M6f-1: the SCC path
            cf = (f", charge file vs run 1 {d['charges_file']:.1e} e" if "charges_file" in d
                  else "")
            print(f"{'':26s}   SCC: errors per iteration (relative) {d['scc_errors']:.1e}{cf}")
        if "e_fr" in d or "n_electrons" in d:             # M6f-2
            sd = (f", DFTB+'s own stress {d['stress_dftbp']:.1e} (vs stress_fd)"
                  if "stress_dftbp" in d else "")
            ef = f", |dE_f| {d['fermi_level']:.1e} Ha" if "fermi_level" in d else ""
            print(f"{'':26s}   Filling: |dE_FR| {d.get('e_fr', 0.0):.1e} Ha, "
                  f"|dN| {d.get('n_electrons', 0.0):.1e}{ef}{sd}")
        if case.corrections and ("e_halogen" in d or "stress_dftbp" in d
                                 or ("forces_dftbp" in d and not case.casida)):   # M6f-3
            parts = []
            if "e_halogen" in d:
                parts.append(f"|dE_halogen| {d['e_halogen']:.1e} Ha")
            if "forces_dftbp" in d and not case.casida:
                parts.append(f"forces vs DFTB+'s energy differences, as DFTB+ (compat) "
                             f"{d['forces_dftbp']:.1e}")
            if "stress_dftbp" in d:
                parts.append(f"DFTB+'s own stress {d['stress_dftbp']:.1e} (vs stress_fd)")
            print(f"{'':26s}   Corrections: " + ", ".join(parts))
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
    ap.add_argument("--casida-solver", default="auto",
                    help="pydftb's Casida eigensolver: auto, exact, davidson or arpack (M6d)")
    ap.add_argument("--backend", default=None, help="numpy (default) or torch")
    ap.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    ap.add_argument("--dtype", default=None, help="float64 (default) or float32")
    a = ap.parse_args(argv)
    if a.regenerate:
        regenerate(a.cases)
    elif a.check:
        sys.exit(0 if check(a.cases, a.solver, casida_solver=a.casida_solver,
                            backend=a.backend, device=a.device, dtype=a.dtype) else 1)
    else:
        live(a.cases)


if __name__ == "__main__":
    main()
