"""DFTB MD of the Li | Li2O interface with a fitted parameter set (M5c, design section 5).

    PYTHONPATH=<repo> python md_interface.py IFACE.vasp --skf DIR [--types JSON] --out DIR
        [--steps 2000] [--temperature 300] [--snapshots 5]

1. relax the positions at fixed cell (FIRE, fmax 0.05 eV/A, at most 500 steps), writing
   relax.traj and relax_weights.csv; if a calculation fails (SCC not converged) the last
   geometry goes to failed.vasp and the script exits with 2;
2. Langevin NVT at --temperature, 1 fs, --steps steps (friction 0.02 / fs);
3. 1 ps of NVE after it, to check the energy drift;
4. writes md.traj (every 10 steps), weights.csv (the Li_ox weight of every Li per frame, with
   switching), drift.txt and --snapshots frames evenly spaced over the NVT part as
   snap_XX.vasp for the DFT check.

Electronic temperature 300 K (Fermi) and exact SK derivatives (fd_delta None): the MD needs
forces consistent with the energy; PTBP's fit used T = 0 and DFTB+'s 0.01 Bohr differences.
"""
from __future__ import annotations

import argparse
import json
import os

import ase.io
import numpy as np
from ase import units
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.calculators.calculator import CalculationFailed
from ase.optimize import FIRE

from pydftb.ase_calc import PyDFTB
from utils.calculator import monkhorstpack2kptdensity
from utils.typemodel import switch_from_declaration


def make_calc(skf, switches, atoms):
    _, kpts = monkhorstpack2kptdensity(atoms, 5.0)          # PTBP's k-point density
    extra = {"type_switching": switches} if switches else {}
    return PyDFTB(skf if skf.endswith("/") else skf + "/", kpts=tuple(int(k) for k in kpts),
                  smearing=("fermi", 300.0), scc=True, scc_tol=1e-6, max_scc=500,
                  repulsive="skf", shell_resolved=True, fd_delta=None, **extra)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure")
    ap.add_argument("--skf", required=True)
    ap.add_argument("--types", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--nve", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--snapshots", type=int, default=5)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    switches = [switch_from_declaration(d) for d in json.loads(a.types)] if a.types else []

    atoms = ase.io.read(a.structure)
    atoms.calc = make_calc(os.path.abspath(a.skf), switches, atoms)
    print("k-points", atoms.calc.parameters.get("kpts") if hasattr(atoms.calc, "parameters") else "")
    li = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Li"]

    def weight_writer(name):
        wfile = open(os.path.join(a.out, name), "w")
        wfile.write("step," + ",".join(f"Li{i}" for i in li) + "\n")

        def write(step):
            raw = getattr(atoms.calc, "raw", None)
            tw = getattr(raw, "type_weights", None) if raw is not None else None
            if tw and "Li_ox" in tw:
                wfile.write(f"{step}," + ",".join(f"{tw['Li_ox'][i]:.4f}" for i in li) + "\n")
                wfile.flush()
        return write

    # relax: relax.traj every step and the Li_ox weights per step (relax_weights.csv); a
    # calculation that fails (e.g. SCC not converged) leaves its last geometry in failed.vasp
    opt = FIRE(atoms, logfile=os.path.join(a.out, "relax.log"),
               trajectory=os.path.join(a.out, "relax.traj"))
    rw = weight_writer("relax_weights.csv")
    opt.attach(lambda: rw(opt.nsteps))
    try:
        opt.run(fmax=0.05, steps=500)
    except CalculationFailed as err:
        ase.io.write(os.path.join(a.out, "failed.vasp"), atoms, format="vasp", direct=True,
                     sort=True)
        print(f"relax stopped at step {opt.nsteps}: {err}")
        return 2
    ase.io.write(os.path.join(a.out, "relaxed.vasp"), atoms, format="vasp", direct=True, sort=True)

    weights = weight_writer("weights.csv")

    MaxwellBoltzmannDistribution(atoms, temperature_K=a.temperature, rng=np.random.default_rng(7))
    traj = Trajectory(os.path.join(a.out, "md.traj"), "w", atoms)
    dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=a.temperature, friction=0.02 / units.fs,
                   rng=np.random.default_rng(11), logfile=os.path.join(a.out, "nvt.log"),
                   loginterval=10)
    snaps = set(np.linspace(a.steps // a.snapshots, a.steps, a.snapshots).astype(int))
    count = {"n": 0}

    def every():
        n = count["n"] = count["n"] + 10
        traj.write()
        weights(n)
        if n in snaps:
            k = sorted(snaps).index(n)
            ase.io.write(os.path.join(a.out, f"snap_{k:02d}.vasp"), atoms, format="vasp",
                         direct=True, sort=True)

    dyn.attach(every, interval=10)
    dyn.run(a.steps)

    etot = []
    nve = VelocityVerlet(atoms, 1.0 * units.fs, logfile=os.path.join(a.out, "nve.log"),
                         loginterval=10)
    nve.attach(lambda: etot.append(atoms.get_potential_energy() + atoms.get_kinetic_energy()),
               interval=10)
    nve.run(a.nve)
    etot = np.array(etot)
    t = np.arange(len(etot)) * 10e-3                          # ps
    slope = np.polyfit(t, etot, 1)[0] if len(etot) > 2 else float("nan")
    with open(os.path.join(a.out, "drift.txt"), "w") as fh:
        fh.write(f"NVE {a.nve} fs: E_tot drift {slope / len(atoms) * 1e3:.4f} meV/atom/ps, "
                 f"std {etot.std() / len(atoms) * 1e3:.4f} meV/atom\n")
    print(open(os.path.join(a.out, "drift.txt")).read())


if __name__ == "__main__":
    raise SystemExit(main())
