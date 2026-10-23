"""Geometry optimisation and molecular dynamics against DFTB+'s own drivers (milestone M2f).

DFTB+ runs its ``GeometryOptimisation`` or ``VelocityVerlet`` driver; pydftb runs the same
task through ASE (BFGS, with ``FrechetCellFilter`` for the cell; ``VelocityVerlet``) with
``PyDFTB(fd_delta=2**-13, reuse_charges=True)``.  The finite-difference step gives DFTB+'s
forces (it differentiates H0/S numerically); charge reuse gives DFTB+'s SCC starting points,
so the MD steps take the same SCC paths.  XL-BOMD (M6e) runs the same way, DFTB+'s
``Xlbomd``/``XlbomdFast`` against ``PyDFTB(xlbomd=...)``, and compares the SCC iterations of
every step as well.

    python -m pydftb.validation.dynamics --regenerate   # needs dftb+; writes JSON
    python -m pydftb.validation.dynamics --check        # pydftb vs the frozen JSON

Optimisations are compared at the end point (energy, structure after alignment, cell).
They start from the base structure relaxed to 0.02 eV/A (frozen in the JSON), so both
optimisers polish the same minimum: from a far start different optimisers can end in
different local minima, which says nothing about the forces (and DFTB+'s LBFGS has no step
limit; on the soft water dimer it diverged).  MD trajectories are compared frame by frame
(positions, Mermin and kinetic energy).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms, units
from ase.filters import FrechetCellFilter
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

from ..ase_calc import PyDFTB
from ..constants import AMU_AU, AU_FS, BOHR
from ..dispersion import library_versions
from . import dftbplus, refgen
from .cases import CASES_BY_NAME, DATA, SKF_SETS

REF_DIR = os.path.join(DATA, "ref_dynamics")
# DFTB+'s default finite-difference step, epsilon**(1/4) = 2**-13 = 1.2207e-4 Bohr (the rounded
# 1.22e-4 of M2f-M6d gave up to 2.4e-8 Ha/Bohr where the two points straddle an SK grid node)
FD_DELTA = float(np.finfo(float).eps) ** 0.25
HA_BOHR = units.Hartree / units.Bohr  # eV/Angstrom per Ha/Bohr (ASE, as PyDFTB)


@dataclass
class DynCase:
    name: str
    base: str                       # validation.cases.Case: structure and Hamiltonian
    kind: str                       # "opt" | "md"
    lattice: bool = False           # optimise the cell as well
    isotropic: bool = False         # ... by a uniform scaling only
    ewald_tolerance: float | None = None   # DFTB+ EwaldTolerance (pydftb: 1e-12)
    grad_elem: float = 1e-7         # DFTB+ GradElem (Ha/Bohr); ASE fmax is the same in eV/A
    optimiser: str = "LBFGS {}"     # DFTB+ optimiser block
    prerelax: float | None = 0.02   # opt: start from pydftb + BFGS relaxed to this fmax (eV/A)
    temperature: float = 500.0      # K: initial velocities (MD)
    seed: int = 1
    steps: int = 40                 # MD steps
    time_step: float = 0.5          # fs
    xlbomd: dict | None = None      # M6e: XL-BOMD (pydftb.xlbomd.XLBOMD options)
    force_evaluation: str | None = None    # "dynamics_t0" | "dynamics" (DFTB+ ForceEvaluation)
    tol: dict = field(default_factory=dict)
    note: str = ""


# DFTB+ ForceEvaluation names of pydftb's force_evaluation
FORCE_EVALUATION_HSD = {"traditional": "Traditional", "dynamics_t0": "DynamicsT0",
                        "dynamics": "Dynamics"}
TOL_OPT = {"energy": 1e-9, "positions": 1e-4, "cell": 1e-5}
# MD through ASE: its unit constants (CODATA 2014) differ from DFTB+'s by ~1e-7, which over 40
# steps gives up to 4e-8 A and 1.1e-8 Ha.  *_exact: the same velocity Verlet in atomic units with
# DFTB+'s constants (verlet_dftbplus_units), limited by the 8 decimals of DFTB+'s xyz output
TOL_MD = {"positions": 1e-7, "energy": 2e-8, "positions_exact": 2e-8, "energy_exact": 2e-9}
# XL-BOMD (M6e): also the velocities (A/ps; 8 decimals in DFTB+'s xyz) and the SCC iterations of
# every step (largest difference; the closest convergence decision is 5e-4 from its tolerance).
# The exact Verlet agrees to DFTB+'s printed digits (5e-9 A, 5e-11 Ha, 7e-9 A/ps over 60
# steps); through ASE 60 steps give up to 1.1e-7 A, 2.6e-8 Ha and 7e-5 A/ps
TOL_XL = {"positions": 3e-7, "energy": 6e-8, "velocities": 2e-4, "positions_exact": 1e-8,
          "energy_exact": 2e-10, "velocities_exact": 2e-8, "n_scc": 0}

DYN_CASES = [
    DynCase("h2o_opt", "h2o", "opt", tol=TOL_OPT),
    DynCase("ch3oh_opt", "ch3oh", "opt", tol=TOL_OPT),
    DynCase("water_dimer_opt", "water_dimer_hdamp", "opt", tol=TOL_OPT,
            note="soft hydrogen bond (from the unrelaxed start DFTB+'s LBFGS steps grow until the "
                 "SCC diverges)"),
    DynCase("o2_triplet_opt", "o2_triplet_wmatrix", "opt", tol=TOL_OPT),
    DynCase("cuo_chain_opt", "cuo_chain_k422", "opt", ewald_tolerance=1e-12, tol=TOL_OPT,
            note="k-points, atoms only"),
    DynCase("li_bcc_cellopt", "li_bcc_cubic_k444_1000K", "opt", lattice=True, isotropic=True,
            grad_elem=1e-8, tol=TOL_OPT,
            note="lattice constant of the cubic cell (uniform scaling).  In the primitive cell "
                 "DFTB+'s LBFGS drifts along a very soft shear (6.6e-9 Ha above the symmetric "
                 "minimum), and its Isotropic mode shears the cell too: it converges only the "
                 "pressure and steps along L^-T, which is not parallel to L"),
    DynCase("cu_fcc_cellopt", "cu_fcc_k6", "opt", lattice=True, grad_elem=1e-8, tol=TOL_OPT,
            note="cell only (one atom), Fermi 1000 K"),
    DynCase("h2o_md", "h2o", "md", tol=TOL_MD),
    DynCase("water_dimer_md", "water_dimer_hdamp", "md", tol=TOL_MD),
    DynCase("o2_triplet_md", "o2_triplet_wmatrix", "md", temperature=1000.0, tol=TOL_MD,
            note="spin; two atoms"),
    DynCase("cuo_chain_md", "cuo_chain_k422", "md", ewald_tolerance=1e-12, tol=TOL_MD,
            note="k-points, Fermi 500 K"),
    # M6b: DFTB+ leaves the onsite corrections out of the first SCC iteration of every
    # geometry step, pydftb only when it starts without charges (reused charges count as
    # read in): the SCC paths differ, the trajectory does not
    DynCase("h2o_ons_md", "h2o_ons", "md", tol=TOL_MD,
            note="onsite corrections; the block vector carried between steps"),
    DynCase("h2o_gbsa_md", "h2o_gbsa", "md", tol=TOL_MD,
            note="GBSA solvation (Born radii, surface areas, hydrogen-bond correction) (M6c)"),
    # M6e: DFTB+'s drivers in an excited state follow E0 + w and its forces
    DynCase("h2co_s1_opt", "h2co_exc_forces_s1", "opt", tol=TOL_OPT,
            note="first excited singlet: the molecule pyramidalises"),
    DynCase("h2co_s2_md", "h2co_exc_forces", "md", temperature=300.0, steps=20, tol=TOL_MD,
            note="second excited singlet, 20 steps"),
    # M6e: XL-BOMD.  Xlbomd converges every step to its own tolerance from the integrated
    # populations, XlbomdFast takes one SCC iteration per step (its energy is the shadow
    # energy); before the integration runs (PreSteps, K + 1 filling steps) ordinary BO steps
    DynCase("water_dimer_xl5_md", "water_dimer_hdamp", "md", steps=60,
            xlbomd={"integration_steps": 5, "scc_tol": 1e-6}, force_evaluation="dynamics_t0",
            tol=TOL_XL, note="Xlbomd, K = 5, SCC tolerance 1e-6"),
    DynCase("water_dimer_xl6_presteps_md", "water_dimer_hdamp", "md",
            xlbomd={"integration_steps": 6, "pre_steps": 3, "scc_tol": 1e-6},
            force_evaluation="dynamics_t0", tol=TOL_XL, note="Xlbomd, K = 6 after 3 BO steps"),
    DynCase("water_dimer_xl_minscc_md", "water_dimer_hdamp", "md",
            xlbomd={"min_scc": 4, "scc_tol": 1e-3}, force_evaluation="dynamics_t0", tol=TOL_XL,
            note="Xlbomd to 1e-3: MinSccIterations 4 has no effect (DFTB+ applies it on the first "
                 "geometry step only, an ordinary BO step)"),
    DynCase("water_dimer_xlfast_md", "water_dimer_hdamp", "md", steps=60,
            xlbomd={"fast": True, "scale": 0.5}, force_evaluation="dynamics_t0", tol=TOL_XL,
            note="XlbomdFast, K = 5, Scale 0.5 (1.0 diverges here, in DFTB+ as in pydftb)"),
    DynCase("water_dimer_xlfast7_md", "water_dimer_hdamp", "md",
            xlbomd={"integration_steps": 7, "fast": True, "transient_steps": 5, "scale": 0.25},
            force_evaluation="dynamics_t0", tol=TOL_XL,
            note="XlbomdFast, K = 7 (DFTB+'s single-precision alpha), 5 transient steps"),
    DynCase("water_dimer_dftb3_xlfast_md", "water_dimer_dftb3_nodamp", "md",
            xlbomd={"fast": True, "scale": 0.5}, force_evaluation="dynamics_t0", tol=TOL_XL,
            note="DFTB3 (full third order): its shadow energy and forces"),
    DynCase("cuo_chain_xl_md", "cuo_chain_k422", "md", ewald_tolerance=1e-12,
            xlbomd={"scc_tol": 1e-6}, force_evaluation="dynamics", tol=TOL_XL,
            note="k-points, Fermi 500 K: the Dynamics force type"),
]
DYN_BY_NAME = {c.name: c for c in DYN_CASES}


# ----------------------------------------------------------------------------
def _inputs(case: DynCase) -> dict:
    base = CASES_BY_NAME[case.base]
    inp = refgen._inputs(base)
    if case.kind == "opt" and case.prerelax:
        atoms = _atoms(inp)
        atoms.calc = pydftb_calculator(case, inp)
        opt = BFGS(_target(atoms, case), logfile=None)
        opt.run(fmax=case.prerelax, steps=1000)
        inp["positions_angstrom"] = atoms.positions.tolist()
        if case.lattice:
            inp["cell_angstrom"] = atoms.cell.array.tolist()
        inp["prerelaxed"] = {"fmax_eV_A": case.prerelax, "bfgs_steps": opt.nsteps}
    if case.kind == "md":
        atoms = _atoms(inp)
        MaxwellBoltzmannDistribution(atoms, temperature_K=case.temperature,
                                     rng=np.random.default_rng(case.seed))
        Stationary(atoms)
        if not atoms.pbc.any():
            ZeroRotation(atoms)
        inp["velocities_aaps"] = (atoms.get_velocities() * 1000.0 * units.fs).tolist()
    return inp


def _target(atoms: Atoms, case: DynCase):
    """What the ASE optimiser moves: the atoms, or atoms and cell."""
    if not case.lattice:
        return atoms
    return FrechetCellFilter(atoms, hydrostatic_strain=case.isotropic)


def _atoms(inp: dict) -> Atoms:
    from ..system import element_of_label
    periodic = inp["cell_angstrom"] is not None
    return Atoms([element_of_label(lab) for lab in inp["labels"]],
                 positions=np.array(inp["positions_angstrom"]),
                 cell=np.array(inp["cell_angstrom"]) if periodic else None, pbc=periodic)


def pydftb_calculator(case: DynCase, inp: dict, **overrides) -> PyDFTB:
    """PyDFTB with the base case's settings, DFTB+'s finite differences and charge reuse."""
    base = CASES_BY_NAME[case.base]
    kpts = None
    if inp["kpoints"] is not None:
        kpts = {"kpts": inp["kpoints"], "weights": inp["kweights"]}
    opts = dict(max_l=inp["max_l"], kpts=kpts, time_reversal=False, smearing=tuple(base.filling),
                scc=base.scc, scc_tol=1e-10, max_scc=500, repulsive=base.repulsive,
                shell_resolved=base.shell_resolved, labels=inp["labels"], fd_delta=FD_DELTA,
                dispersion=inp.get("dispersion"), reuse_charges=True,
                **(base.dftb3 or {}), **({"spin": True, **base.spin} if base.spin else {}),
                **refgen.block_options(base, inp["labels"]),
                **refgen.environment_ase_options(base),
                **refgen.casida_options(base))              # M6e: an excited state's forces
    if case.xlbomd is not None:                                 # M6e
        opts.update(xlbomd=dict(case.xlbomd), force_evaluation=case.force_evaluation)
    opts.update(overrides)
    return PyDFTB(SKF_SETS[base.skf_set], **opts)


def _potential(calc: PyDFTB) -> float:
    """The potential energy the driver sees (Ha): the Mermin free energy, plus Omega in an
    excited state (as DFTB+'s md.out and optimiser, M6e)."""
    raw = calc.raw
    return raw.excited_free_energy if calc.excited else raw.free_energy


def _with_masses(atoms: Atoms, calc: PyDFTB, labels) -> Atoms:
    """Masses from the SKF files, as DFTB+ (equal to ASE's for the test sets)."""
    sk = calc.params.skset(labels)
    atoms.set_masses([sk.mass[lab] for lab in labels])
    return atoms


def run_pydftb(case: DynCase, inp: dict, steps: int | None = None, **overrides) -> dict:
    """Optimisation or MD through ASE; returns the end point or the trajectory (A, Ha)."""
    atoms = _atoms(inp)
    calc = pydftb_calculator(case, inp, **overrides)
    _with_masses(atoms, calc, inp["labels"])
    atoms.calc = calc
    t0 = time.perf_counter()
    n_scc = []
    if case.kind == "opt":
        opt = BFGS(_target(atoms, case), logfile=None)
        opt.attach(lambda: n_scc.append(calc.raw.n_scc))
        opt.run(fmax=case.grad_elem * HA_BOHR, steps=5000)
        return {"positions": atoms.positions.copy(), "cell": atoms.cell.array.copy(),
                "free_energy": _potential(calc), "steps": opt.nsteps,
                "n_scc": n_scc, "wall": time.perf_counter() - t0}
    atoms.set_velocities(np.array(inp["velocities_aaps"]) / (1000.0 * units.fs))
    frames = {"positions": [], "velocities": [], "potential": [], "kinetic": []}

    def record():
        frames["positions"].append(atoms.positions.copy())
        frames["velocities"].append(atoms.get_velocities() * 1000.0 * units.fs)
        frames["potential"].append(atoms.get_potential_energy(force_consistent=True) / units.Hartree)
        frames["kinetic"].append(atoms.get_kinetic_energy() / units.Hartree)
        n_scc.append(calc.raw.n_scc)

    md = VelocityVerlet(atoms, timestep=case.time_step * units.fs, logfile=None)
    md.attach(record, interval=1)
    md.run(case.steps if steps is None else steps)
    return {k: np.array(v) for k, v in frames.items()} | {"n_scc": n_scc,
                                                         "wall": time.perf_counter() - t0}


def verlet_dftbplus_units(case: DynCase, inp: dict, steps: int | None = None, **overrides) -> dict:
    """DFTB+'s velocity Verlet (md/velocityverlet.F90) in atomic units with DFTB+'s constants,
    pydftb forces: separates the integrator and the forces from ASE's unit constants."""
    atoms = _atoms(inp)
    calc = pydftb_calculator(case, inp, **overrides)
    atoms.calc = calc
    sk = calc.params.skset(inp["labels"])
    m = np.array([sk.mass[lab] for lab in inp["labels"]])[:, None] * AMU_AU
    dt = case.time_step / AU_FS
    x = np.array(inp["positions_angstrom"]) / BOHR
    v = np.array(inp["velocities_aaps"]) / BOHR * AU_FS / 1000.0     # Bohr per atomic time unit

    frames = {"positions": [], "velocities": [], "potential": [], "kinetic": [], "n_scc": []}

    def forces(x):
        atoms.positions = x * BOHR
        atoms.get_forces()
        frames["n_scc"].append(calc.raw.n_scc)
        return calc.raw.forces, _potential(calc)

    def record(x, v, e):
        frames["positions"].append(x * BOHR)
        frames["velocities"].append(v * BOHR / AU_FS * 1000.0)           # A/ps
        frames["potential"].append(e)
        frames["kinetic"].append(0.5 * float((m * v * v).sum()))

    f, e = forces(x)
    record(x, v, e)
    for _ in range(case.steps if steps is None else steps):
        v_half = v + 0.5 * f / m * dt
        x = x + v_half * dt
        f, e = forces(x)
        v = v_half + 0.5 * f / m * dt
        record(x, v, e)
    return {k: np.array(val) for k, val in frames.items()}


def run_dftbplus(case: DynCase, inp: dict, threads: int = 1) -> dict:
    base = CASES_BY_NAME[case.base]
    periodic = inp["cell_angstrom"] is not None
    kpts = None if inp["kpoints"] is None else (np.array(inp["kpoints"]), np.array(inp["kweights"]))
    if case.kind == "opt":
        driver = dftbplus.opt_driver(case.optimiser, case.lattice, case.grad_elem,
                                     isotropic=case.isotropic)
    else:
        driver = dftbplus.md_driver(inp["velocities_aaps"], case.time_step, case.steps,
                                    xlbomd=case.xlbomd)
    with tempfile.TemporaryDirectory(prefix="pydftb_dyn_") as tmp:
        skdir = refgen._plain_skf_dir(base, tmp)
        extra = (dftbplus.dftb3_block(base.dftb3, set(inp["labels"]))
                 + dftbplus.spin_block(base.spin, set(inp["labels"]), base.shell_resolved)
                 + dftbplus.dispersion_block(inp.get("dispersion"), inp["labels"])
                 + refgen.block_hsd(base, inp) + refgen.environment_hsd(base))
        ewald = case.ewald_tolerance or base.ewald_tolerance
        if ewald is not None:
            extra += f"  EwaldTolerance = {dftbplus._f(ewald)}\n"
        if case.force_evaluation is not None:
            extra += f"  ForceEvaluation = {FORCE_EVALUATION_HSD[case.force_evaluation]}\n"
        if base.casida and base.casida.get("spin_constants") and not base.spin:
            extra += dftbplus.spin_constants_block(base.casida["spin_constants"],
                                                   set(inp["labels"]))
        res = dftbplus.run(inp["labels"], np.array(inp["positions_angstrom"]),
                           np.array(inp["cell_angstrom"]) if periodic else None, skdir=skdir,
                           max_l=inp["max_l"], kpts=kpts, scc=base.scc, filling=base.filling,
                           shell_resolved=base.shell_resolved, repulsive=base.repulsive,
                           extra_ham=extra, threads=threads, driver=driver, charge=base.charge,
                           extra_top=dftbplus.casida_block(base.casida), autotest=bool(base.casida))
        if base.casida and "excited_mermin_tag" in res:
            res["free_energy"] = res["excited_mermin_tag"]     # an excited state: E0 + w (M6e)
        return res


# ----------------------------------------------------------------------------
def kabsch_rmsd_max(a: np.ndarray, b: np.ndarray) -> float:
    """Largest atomic deviation after optimally superimposing b onto a (Kabsch)."""
    pa, pb = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(pb.T @ pa)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.linalg.norm(pb @ rot - pa, axis=1).max())


def _cell_params(cell: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(cell, axis=1)
    cos = [cell[1] @ cell[2] / (lengths[1] * lengths[2]), cell[0] @ cell[2] / (lengths[0] * lengths[2]),
           cell[0] @ cell[1] / (lengths[0] * lengths[1])]
    return np.concatenate([lengths, np.degrees(np.arccos(np.clip(cos, -1, 1)))])


def compare(doc: dict, **overrides) -> dict:
    case = DYN_BY_NAME[doc["case"]]
    inp, ref = doc["input"], doc["reference"]
    res = run_pydftb(case, inp, **overrides)
    if case.kind == "opt":
        dev = {"energy": abs(res["free_energy"] - ref["free_energy"])}
        if inp["cell_angstrom"] is None:
            dev["positions"] = kabsch_rmsd_max(np.array(ref["positions"]), res["positions"])
        else:
            frac_ref = np.array(ref["positions"]) @ np.linalg.inv(np.array(ref["cell"]))
            frac = res["positions"] @ np.linalg.inv(res["cell"])
            diff = frac - frac_ref
            diff -= np.round(diff - diff[0]) + diff[0]          # same origin, wrap
            dev["positions"] = float(np.abs(diff @ res["cell"]).max())
            p_ref, p = _cell_params(np.array(ref["cell"])), _cell_params(res["cell"])
            dev["cell"] = float(np.abs(p - p_ref)[:3].max())
            dev["angles"] = float(np.abs(p - p_ref)[3:].max())
    else:
        exact = verlet_dftbplus_units(case, inp, **overrides)
        dev = {}
        for suffix, r in (("", res), ("_exact", exact)):
            dev["positions" + suffix] = float(np.abs(r["positions"] - np.array(ref["positions"])).max())
            dev["energy" + suffix] = float(max(np.abs(r["potential"] - np.array(ref["potential"])).max(),
                                               np.abs(r["kinetic"] - np.array(ref["kinetic"])).max()))
            if "velocities" in ref:                         # M6e references
                dev["velocities" + suffix] = float(np.abs(r["velocities"]
                                                          - np.array(ref["velocities"])).max())
        if case.xlbomd is not None:     # the SCC iterations of every step (both runs)
            dev["n_scc"] = max(int(np.abs(np.array(r["n_scc"]) - np.array(ref["n_scc"])).max())
                               for r in (res, exact))
    tol = doc["tolerances"]
    ok = all(dev[k] <= tol[k] for k in tol if k in dev)
    return {"case": case.name, "dev": dev, "ok": ok, "result": res}


# ----------------------------------------------------------------------------
def regenerate(names=None, threads: int = 1):
    os.makedirs(REF_DIR, exist_ok=True)
    for case in DYN_CASES:
        if names and case.name not in names:
            continue
        inp = _inputs(case)
        ref = run_dftbplus(case, inp, threads)
        reference = {"n_scc": ref["scc_iterations"], "geometry_steps": ref["geometry_steps"],
                     "wall": ref["wall"]}
        if case.kind == "opt":
            if not ref["converged_geometry"]:
                raise RuntimeError(f"{case.name}: DFTB+ did not converge the geometry")
            _, pos, cell = ref["final_geometry"]
            reference.update(free_energy=ref["free_energy"], positions=pos.tolist(),
                             cell=None if cell is None else cell.tolist())
        else:
            md = ref["md"]
            frames = ref["trajectory"]
            if len(ref["scc_iterations"]) != len(frames):
                raise RuntimeError(f"{case.name}: SCC iterations of {len(ref['scc_iterations'])} "
                                   f"steps for {len(frames)} frames")
            reference.update(positions=[f["positions"].tolist() for f in frames],
                             velocities=[f["velocities"].tolist() for f in frames],
                             potential=md["potential"].tolist(), kinetic=md["kinetic"].tolist(),
                             total=md["total"].tolist())
        lib = refgen.dispersion_library(inp)
        doc = {"schema": refgen.SCHEMA, "case": case.name, "base": case.base, "kind": case.kind,
               "note": case.note,
               "dftbplus": {"version": ref["version"], "parser_version": dftbplus.PARSER_VERSION},
               "settings": {"lattice": case.lattice, "isotropic": case.isotropic,
                            "ewald_tolerance": case.ewald_tolerance, "grad_elem": case.grad_elem,
                            "optimiser": case.optimiser, "temperature": case.temperature,
                            "seed": case.seed, "steps": case.steps, "time_step_fs": case.time_step,
                            "xlbomd": case.xlbomd, "force_evaluation": case.force_evaluation},
               "libraries": {lib: library_versions()[lib]} if lib else {},
               "input": inp, "hsd": ref["hsd"], "reference": reference,
               "tolerances": case.tol}
        path = os.path.join(REF_DIR, f"{case.name}.json")
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=1)
        print(f"wrote {path}  (geometry steps {ref['geometry_steps']}, {ref['wall']:.1f} s)")


def load(name: str) -> dict:
    with open(os.path.join(REF_DIR, f"{name}.json")) as fh:
        return json.load(fh)


def check(names=None, **overrides) -> bool:
    """``overrides``: PyDFTB options, e.g. ``backend="torch"`` (milestone M4)."""
    all_ok = True
    print(f"{'case':28s} {'kind':4s} {'|dE| Ha':>9s} {'|dr| A':>9s} {'|da| A':>9s} "
          f"{'steps py/ref':>12s} {'SCC py/ref':>11s} {'t_py':>7s} {'t_ref':>7s}  ok")
    for case in DYN_CASES:
        if names and case.name not in names:
            continue
        path = os.path.join(REF_DIR, f"{case.name}.json")
        if not os.path.exists(path):
            print(f"{case.name:28s} (no reference JSON)")
            all_ok = False
            continue
        doc = load(case.name)
        why = refgen.missing_requirement(doc)
        if why:
            print(f"{case.name:28s} skipped: {why}")
            continue
        r = compare(doc, **overrides)
        d, res, ref = r["dev"], r["result"], doc["reference"]
        all_ok &= r["ok"]
        steps = (f"{res['steps']}/{ref['geometry_steps']}" if case.kind == "opt"
                 else f"{len(res['potential']) - 1}/{ref['geometry_steps'] - 1}")
        scc = f"{sum(res['n_scc'])}/{sum(ref['n_scc'])}"
        cell = f"{d['cell']:9.1e}" if "cell" in d else f"{'-':>9s}"
        print(f"{case.name:28s} {case.kind:4s} {d['energy']:9.1e} {d['positions']:9.1e} {cell} "
              f"{steps:>12s} {scc:>11s} {res['wall']:7.2f} {ref['wall']:7.2f}  "
              f"{'yes' if r['ok'] else 'NO'}")
        if case.xlbomd is not None:
            print(f"{'':28s} XL: velocities {d['velocities']:.1e} A/ps (exact Verlet: "
                  f"{d['velocities_exact']:.1e}), SCC iterations per step differ by at most "
                  f"{d['n_scc']}")
    return all_ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--regenerate", action="store_true", help="run dftb+ and write reference JSON")
    g.add_argument("--check", action="store_true", help="compare pydftb with the frozen JSON")
    ap.add_argument("cases", nargs="*", help="case names (default: all)")
    ap.add_argument("--backend", default=None, help="numpy (default) or torch")
    ap.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    ap.add_argument("--dtype", default=None, help="float64 (default) or float32")
    a = ap.parse_args(argv)
    if a.regenerate:
        regenerate(a.cases)
    else:
        opts = {k: v for k, v in (("backend", a.backend), ("device", a.device),
                                  ("dtype", a.dtype)) if v is not None}
        sys.exit(0 if check(a.cases, **opts) else 1)


if __name__ == "__main__":
    main()
