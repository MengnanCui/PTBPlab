"""``pydftb run``: one command for a calculation on either engine (P2).

    pydftb run STRUCTURE --skdir DIR [--engine pydftb|dftbplus] [--task TASK] [options]

Tasks: ``energy`` (single point: energy, forces, stress, charges, ...), ``relax`` (ASE's BFGS,
``--cell`` with the cell), ``md`` (velocity Verlet), ``bands`` (band structure along a path),
``dos`` (density of states, PDOS regions), ``hessian`` (second derivatives and normal modes).
The structure is any file ASE reads.  Calculator options are PyDFTB's (eV, Angstrom, K) from
``--options FILE.json`` (both engines' options at the top level, one engine's in its
``"pydftb"`` or ``"dftbplus"`` section), ``--set KEY=VALUE`` (the value read as JSON, else a
string) and the flags below; only ``--engine`` changes between the two engines.  ``--native``
runs DFTB+'s own optimiser or MD driver (the DFTB+ engine; XL-BOMD always does).  Results go to
the screen and, with ``--json``, to a JSON file whose fields are the same for both engines.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from ase import units

from . import __version__
from .engines import EngineOptionError, common_results, engine_name, make_calculator

TASKS = ("energy", "relax", "md", "bands", "dos", "hessian")
# the hessian task's SCC tolerance unless one is given: its finite differences of the forces
# (step 1e-4 Bohr) turn an SCC error of 1e-6 into 1e-2 Ha/Bohr^2
HESSIAN_SCC_TOL = 1e-10


def _value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_options(args) -> tuple[dict, dict, dict]:
    """(both engines' options, pydftb's own, DFTB+'s own) from the options file, ``--set`` and
    the flags (later ones win)."""
    common, own = {}, {"pydftb": {}, "dftbplus": {}}
    if args.options:
        with open(args.options) as fh:
            data = json.load(fh)
        for key in ("pydftb", "dftbplus"):
            own[key].update(data.pop(key, None) or {})
        common.update(data)
    for item in args.set or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, val = item.split("=", 1)
        section, _, name = key.rpartition(".")
        if section in ("pydftb", "dftbplus"):
            own[section][name] = _value(val)
        elif section:
            raise SystemExit(f"--set {key}: the section must be pydftb or dftbplus")
        else:
            common[key] = _value(val)
    if args.kpts is not None:
        common["kpts"] = tuple(args.kpts)
    if args.temperature is not None:
        common["smearing"] = ("fermi", float(args.temperature))
    if args.scc_tol is not None:
        common["scc_tol"] = args.scc_tol
    if args.binary is not None:
        own["dftbplus"]["binary"] = args.binary
    if args.threads is not None:
        common["threads"] = args.threads
    if isinstance(common.get("smearing"), list):
        common["smearing"] = tuple(common["smearing"])
    return common, own["pydftb"], own["dftbplus"]


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def _needs_native(engine: str, options: dict) -> str | None:
    """Why the DFTB+ engine runs this MD with DFTB+'s own driver, or None."""
    if engine != "dftbplus":
        return None
    if options.get("xlbomd") is not None:
        return "XL-BOMD is an option of DFTB+'s own MD driver"
    spin = options.get("spin")
    if (isinstance(spin, str) and spin.lower() in ("noncolinear", "noncollinear")
            and options.get("hybrid") is not None):
        return ("DFTB+ writes no charge file for a non-collinear hybrid functional, so the "
                "steps of an ASE driver could not carry the density from one to the next")
    return None


# ---- tasks ---------------------------------------------------------------------------------------
def task_energy(atoms, calc, args, info):
    props = ("energy", "forces") + (("stress",) if atoms.pbc.any() and not args.no_stress else ())
    return {"results": common_results(calc, atoms, properties=props)}


def task_relax(atoms, calc, args, info):
    if args.native:
        new, out = calc.relax(atoms, fmax=args.fmax, steps=args.steps, cell=args.cell,
                              isotropic=args.isotropic)
        atoms.positions = new.positions
        if args.cell:
            atoms.set_cell(new.cell, scale_atoms=False)
        steps, converged = out["geometry_steps"], out["converged_geometry"]
        calc.reset()
    else:
        from ase.optimize import BFGS, FIRE, LBFGS
        target = atoms
        if args.cell:
            from ase.filters import FrechetCellFilter
            target = FrechetCellFilter(atoms, hydrostatic_strain=args.isotropic)
        opt = {"bfgs": BFGS, "fire": FIRE, "lbfgs": LBFGS}[args.optimizer](target, logfile=None)
        converged = bool(opt.run(fmax=args.fmax, steps=args.steps))
        steps = opt.nsteps
    out = {"steps": int(steps), "converged": bool(converged),
           "positions": atoms.positions.tolist(),
           "cell": atoms.cell.array.tolist() if atoms.pbc.any() else None}
    out["results"] = common_results(calc, atoms, properties=("energy", "forces"))
    if args.output:
        from ase.io import write
        write(args.output, atoms)
        out["output"] = args.output
    return out


def task_md(atoms, calc, args, info, options):
    import warnings

    from ase.md.velocitydistribution import (
        MaxwellBoltzmannDistribution,
        Stationary,
        ZeroRotation,
    )
    atoms.set_masses(calc.masses(atoms=atoms))
    with warnings.catch_warnings():      # newer ASE prefers thermalize_momenta (same result)
        warnings.simplefilter("ignore", DeprecationWarning)
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.md_temperature,
                                     rng=np.random.default_rng(args.seed))
    Stationary(atoms)
    if not atoms.pbc.any():
        ZeroRotation(atoms)
    native = args.native or bool(_needs_native(info["engine"], options))
    frames = {"potential": [], "kinetic": [], "positions": []}
    if native:
        out = calc.md(atoms, steps=args.steps, time_step=args.timestep,
                      xlbomd=options.get("xlbomd"),
                      force_evaluation=options.get("force_evaluation"))
        frames["potential"] = (np.asarray(out["md"]["potential"]) * units.Hartree).tolist()
        frames["kinetic"] = (np.asarray(out["md"]["kinetic"]) * units.Hartree).tolist()
        frames["positions"] = [f["positions"].tolist() for f in out["trajectory"]]
        if out["trajectory"]:
            atoms.positions = out["trajectory"][-1]["positions"]
        info["native"] = True
    else:
        from ase.md.verlet import VelocityVerlet

        def record():
            frames["potential"].append(float(atoms.get_potential_energy(force_consistent=True)))
            frames["kinetic"].append(float(atoms.get_kinetic_energy()))
            frames["positions"].append(atoms.positions.tolist())

        md = VelocityVerlet(atoms, timestep=args.timestep * units.fs, logfile=None)
        md.attach(record, interval=1)
        md.run(args.steps)
    total = (np.asarray(frames["potential"]) + np.asarray(frames["kinetic"])).tolist()
    out = {"steps": args.steps, "time_step_fs": args.timestep, "potential": frames["potential"],
           "kinetic": frames["kinetic"], "total": total,
           "drift": float(max(total) - min(total)) if total else 0.0}
    if args.output:
        from ase import Atoms
        from ase.io import write
        traj = [Atoms(atoms.get_chemical_symbols(), positions=p, cell=atoms.cell,
                      pbc=atoms.pbc) for p in frames["positions"]]
        write(args.output, traj)
        out["output"] = args.output
    return out


def task_bands(atoms, calc, args, info):
    if not atoms.pbc.any():
        raise SystemExit("bands: needs a periodic structure")
    calc.get_potential_energy(atoms)
    bs = calc.band_structure(path=args.path, npoints=args.npoints, atoms=atoms)
    path = bs.path
    x, xs, names = path.get_linear_kpoint_axis()
    return {"path": args.path, "kpoints": np.asarray(path.kpts).tolist(),
            "x": np.asarray(x).tolist(), "special_x": np.asarray(xs).tolist(),
            "special_labels": list(names), "reference": float(bs.reference),
            "energies": np.asarray(bs.energies).tolist(),
            "results": common_results(calc)}


def task_dos(atoms, calc, args, info):
    regions = None
    if args.regions:
        with open(args.regions) as fh:
            regions = json.load(fh)
    calc.get_potential_energy(atoms)
    st = calc.states(regions=regions, atoms=atoms)
    x, dos = st.dos(width=args.width)
    out = {"energies": np.asarray(x).tolist(), "dos": np.asarray(dos).tolist(),
           "fermi_levels": (np.atleast_1d(st.scc_fermi) * units.Hartree).tolist()}
    if regions:
        from .bands import broaden
        from .constants import HARTREE
        pdos = {}
        for name, w in st.region_weights.items():
            _, y = broaden(st.eigenvalues * HARTREE, st.kweights, w, args.width)
            pdos[name] = np.asarray(y).tolist()
        out["pdos"] = pdos
    out["results"] = common_results(calc)
    return out


def task_hessian(atoms, calc, args, info):
    from .constants import BOHR
    from .vibrations import normal_modes
    sd = calc.second_derivatives(atoms, delta=args.delta)
    out = {"hessian": np.asarray(sd.hessian).tolist(), "delta_bohr": args.delta,
           "units": "Hartree/Bohr^2"}
    if not sd.partial:
        m = normal_modes(sd.hessian, calc.masses(atoms=atoms), atoms.positions / BOHR,
                         born=sd.dipole_derivatives, remove_translation=args.project,
                         remove_rotation=args.project and not atoms.pbc.any())
        out["wavenumbers_cm"] = np.asarray(m.wavenumbers).tolist()
        if m.ir is not None:
            out["ir"] = np.asarray(m.ir).tolist()
    return out


# ---- command -------------------------------------------------------------------------------------
def build_parser(prog=None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", help="structure file (any format ASE reads)")
    ap.add_argument("--skdir", required=True, help="directory of the SK files (A-B.skf[.gz])")
    ap.add_argument("--engine", default="pydftb", help="pydftb (default) or dftbplus")
    ap.add_argument("--task", default="energy", choices=TASKS)
    ap.add_argument("--options", help="calculator options (JSON)")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="one option (the value as JSON, else a string); pydftb.KEY or "
                         "dftbplus.KEY for one engine; repeatable")
    ap.add_argument("--kpts", type=int, nargs=3, help="Monkhorst-Pack grid")
    ap.add_argument("--temperature", type=float, help="electronic temperature (K, Fermi)")
    ap.add_argument("--scc-tol", type=float, help="SCC tolerance")
    ap.add_argument("--charge", type=float, default=0.0, help="net charge (e)")
    ap.add_argument("--threads", type=int, help="threads of the engine")
    ap.add_argument("--binary", help="the DFTB+ executable (DFTB+ engine)")
    ap.add_argument("--json", help="write the results to this JSON file")
    ap.add_argument("--output", help="relax: the final structure; md: the trajectory")
    ap.add_argument("--no-stress", action="store_true", help="energy: no stress (periodic)")
    g = ap.add_argument_group("relax")
    g.add_argument("--fmax", type=float, default=0.01, help="force criterion (eV/A)")
    g.add_argument("--steps", type=int, default=None, help="relax: at most this many steps "
                   "(default 1000); md: the number of steps (default 100)")
    g.add_argument("--cell", action="store_true", help="relax the cell too")
    g.add_argument("--isotropic", action="store_true", help="with --cell: uniform scaling")
    g.add_argument("--optimizer", default="bfgs", choices=("bfgs", "fire", "lbfgs"))
    g.add_argument("--native", action="store_true",
                   help="DFTB+'s own optimiser or MD driver (DFTB+ engine)")
    g = ap.add_argument_group("md")
    g.add_argument("--timestep", type=float, default=0.5, help="time step (fs)")
    g.add_argument("--md-temperature", type=float, default=300.0,
                   help="Maxwell-Boltzmann initial velocities (K)")
    g.add_argument("--seed", type=int, default=1)
    g = ap.add_argument_group("bands, dos, hessian")
    g.add_argument("--path", default=None, help="bands: special points, e.g. GXWLGK "
                   "(default: ASE's path for the cell)")
    g.add_argument("--npoints", type=int, default=100)
    g.add_argument("--width", type=float, default=0.1, help="dos: Gaussian width (eV)")
    g.add_argument("--regions", help="dos: PDOS regions (JSON list of DFTB+ ProjectStates "
                   "regions: label, atoms (0-based), shell_resolved, orbital_resolved)")
    g.add_argument("--delta", type=float, default=1e-4,
                   help="hessian: step (Bohr); the SCC tolerance is 1e-10 unless given")
    g.add_argument("--project", action="store_true",
                   help="hessian: remove translations and rotations")
    return ap


def run(args) -> dict:
    from ase.io import read
    engine = engine_name(args.engine)
    atoms = read(args.structure)
    if args.charge:
        q = np.zeros(len(atoms))
        q[0] = args.charge
        atoms.set_initial_charges(q)
    common, own_py, own_dp = load_options(args)
    options = {**common, **(own_py if engine == "pydftb" else own_dp)}
    md_native = {}
    if engine == "dftbplus" and args.task == "md":
        for key in ("xlbomd", "force_evaluation"):        # options of DFTB+'s MD driver
            if key in common:
                md_native[key] = common.pop(key)
    if args.task in ("relax", "md") and args.native and engine != "dftbplus":
        raise SystemExit("--native: DFTB+'s own drivers need --engine dftbplus")
    if args.steps is None:
        args.steps = 1000 if args.task == "relax" else 100
    if args.task == "bands" and args.path is None:
        args.path = atoms.cell.bandpath().path
    if args.task == "hessian" and "scc_tol" not in common and "scc_tol" not in own_py \
            and "scc_tol" not in own_dp:
        common["scc_tol"] = HESSIAN_SCC_TOL          # differences of forces need tight charges
        options["scc_tol"] = HESSIAN_SCC_TOL
    calc = make_calculator(args.skdir, engine, pydftb=own_py, dftbplus=own_dp, **common)
    if args.task in ("relax", "md") and not args.native and engine == "dftbplus":
        calc.options.setdefault("reuse_charges", True)      # as DFTB+'s own drivers
    elif args.task in ("relax", "md") and engine == "pydftb":
        calc.reuse_charges = True
    atoms.calc = calc
    info = {"engine": engine, "task": args.task, "structure": os.path.abspath(args.structure),
            "pydftb": __version__}
    t0 = time.perf_counter()
    if args.task == "md":
        out = task_md(atoms, calc, args, info, {**options, **md_native})
    else:
        out = globals()[f"task_{args.task}"](atoms, calc, args, info)
    info["wall"] = time.perf_counter() - t0
    if engine == "dftbplus" and calc.output is not None:
        info["dftbplus"] = calc.output.get("version")
        info["dftbplus_runs"] = calc.calls
    info["options"] = _jsonable(options | md_native)
    return {**info, **out}


def summary(res: dict) -> str:
    lines = [f"pydftb run  engine {res['engine']}  task {res['task']}  {res['wall']:.2f} s"
             + (f"  ({res['dftbplus']}, {res['dftbplus_runs']} runs)" if "dftbplus" in res
                else "")]
    r = res.get("results") or {}
    if "energy" in r:
        lines.append(f"  energy {r['energy']:.10f} eV   free energy {r.get('free_energy', 0):.10f} eV")
    if "forces" in r:
        f = np.asarray(r["forces"])
        lines.append(f"  max |force| {np.abs(f).max():.3e} eV/A")
    if "stress" in r:
        lines.append("  stress (eV/A^3) " + " ".join(f"{x:.4e}" for x in r["stress"]))
    if "charges" in r:
        lines.append("  charges " + " ".join(f"{q:+.5f}" for q in r["charges"][:12])
                     + (" ..." if len(r["charges"]) > 12 else ""))
    if r.get("excitation_energies"):
        lines.append("  excitations (eV) " + " ".join(f"{e:.4f}" for e in
                                                     r["excitation_energies"][:8]))
    if res["task"] == "relax":
        lines.append(f"  {res['steps']} steps, converged {res['converged']}")
    if res["task"] == "md":
        lines.append(f"  {res['steps']} steps of {res['time_step_fs']} fs, total-energy drift "
                     f"{res['drift']:.3e} eV")
    if res["task"] == "hessian" and "wavenumbers_cm" in res:
        lines.append("  wavenumbers (cm^-1) " + " ".join(f"{w:.1f}" for w in
                                                        res["wavenumbers_cm"]))
    if res["task"] == "bands":
        lines.append(f"  {len(res['kpoints'])} k-points along {res['path']}")
    if res["task"] == "dos":
        lines.append(f"  DOS on {len(res['energies'])} points"
                     + (f", {len(res['pdos'])} PDOS regions" if "pdos" in res else ""))
    return "\n".join(lines)


def main(argv=None, prog=None) -> int:
    args = build_parser(prog).parse_args(argv)
    try:
        res = run(args)
    except EngineOptionError as err:
        print(f"{prog or 'pydftb run'}: {err}", file=sys.stderr)
        return 2
    print(summary(res))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(_jsonable(res), fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
