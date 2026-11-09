"""Milestone M8a: NVE molecular dynamics with fitted environment-adaptive Cu parameters.

Reads the fitted model of ``pydftb.bench.adaptive_cu`` (its JSON), checks the forces of a
distorted Cu13 icosahedron (surface CN about 6, centre 12) against central differences of the
free energy, then runs velocity Verlet at 600 K and reports the drift of the total energy.

    python -m pydftb.bench.adaptive_md adaptive_cu.json [--model adaptive] [--steps 2000]
        [--structure cu13|cu38|cu55|fcc32] [--kpts 3 3 3]

``--structure``: the Cu13 icosahedron (default), a Cu38 truncated octahedron, a Cu55
icosahedron or a periodic fcc 2x2x2 cubic supercell (32 atoms, ``--kpts``, default 3x3x3),
each rattled by 0.05 A.  The trace records the shortest bond and the largest nearest-neighbour
distance (an atom leaving the cluster), periodic distances by minimum image.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from ase import units
from ase.cluster import Icosahedron
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet

from pydftb.ase_calc import PyDFTB
from pydftb.bench import adaptive_cu as ac


def fitted_model(path: str, kind: str):
    with open(path) as fh:
        rep = json.load(fh)
    if "wall" in rep:                    # a cu_dft_fit model: its core wall (A), not M8a's
        from pydftb.bench.cu_dft_fit import models
        m = models(kind, rep["wall"], rep.get("cn_clamp"))[0]
    else:
        m = ac.models(kind)[0]
    return m.with_vector(np.array(rep["models"][kind]["x"]))


def structure(name: str):
    from ase.build import bulk
    from ase.cluster import Octahedron
    if name == "cu13":
        at = Icosahedron("Cu", 2, latticeconstant=3.61)
    elif name == "cu55":
        at = Icosahedron("Cu", 3, latticeconstant=3.61)
    elif name == "cu38":
        at = Octahedron("Cu", 4, cutoff=1, latticeconstant=3.61)
    else:
        at = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((2, 2, 2))
    assert len(at) == {"cu13": 13, "cu38": 38, "cu55": 55, "fcc32": 32}[name]
    if not at.pbc.any():
        at.center(vacuum=8.0)
    return at


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fit_json")
    ap.add_argument("--model", default="adaptive")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--timestep", type=float, default=1.0, help="fs")
    ap.add_argument("--temperature", type=float, default=600.0, help="K")
    ap.add_argument("--out", default="adaptive_md.json")
    ap.add_argument("--traj", default=None, help="write every 10th step (extxyz)")
    ap.add_argument("--structure", choices=("cu13", "cu38", "cu55", "fcc32"), default="cu13")
    ap.add_argument("--kpts", type=int, nargs=3, default=(3, 3, 3), help="fcc32 only")
    args = ap.parse_args(argv)
    model = fitted_model(args.fit_json, args.model)
    with open(args.fit_json) as fh:
        skdir = json.load(fh).get("skf") or f"{ac.REPO}/pydftb/tests/data/skf/ptbp_hcocu"

    at = structure(args.structure)
    periodic = bool(at.pbc.any())

    def calc(**kw):
        if periodic:
            kw["kpts"] = tuple(args.kpts)
        return PyDFTB(skdir, max_l={"Cu": 2}, shell_resolved=True, temperature=300.0,
                      adaptive=model, **kw)

    rng = np.random.default_rng(3)
    at.positions += rng.normal(scale=0.05, size=at.positions.shape)
    at.calc = calc(scc_tol=1e-10)
    f = at.get_forces()
    cn = at.calc.raw.adaptive.cn
    fd = np.zeros_like(f)
    d = 1e-4                                                     # Angstrom
    for i in (0, 1, 7):
        for k in range(3):
            p = at.copy()
            p.calc = calc(scc_tol=1e-10)
            p.positions[i, k] += d
            ep = p.get_potential_energy(force_consistent=True)
            p.positions[i, k] -= 2 * d
            em = p.get_potential_energy(force_consistent=True)
            fd[i, k] = -(ep - em) / (2 * d)
    ferr = float(np.abs(f[[0, 1, 7]] - fd[[0, 1, 7]]).max())
    print(f"CN {np.round(cn, 2)}; forces vs FD {ferr:.2e} eV/A (|F| {np.abs(f).max():.2f})",
          flush=True)
    at.calc = calc(scc_tol=1e-8, reuse_charges=True)
    MaxwellBoltzmannDistribution(at, temperature_K=args.temperature, rng=np.random.default_rng(1))
    Stationary(at)
    ZeroRotation(at)
    md = VelocityVerlet(at, args.timestep * units.fs)
    rows, t0 = [], time.perf_counter()

    def record():
        ep = at.get_potential_energy(force_consistent=True)
        ek = at.get_kinetic_energy()
        dd = at.get_all_distances(mic=periodic)
        np.fill_diagonal(dd, np.inf)
        rows.append((md.nsteps * args.timestep, ep, ek, ep + ek,
                     float(at.calc.raw.adaptive.cn.min()), float(at.calc.raw.adaptive.cn.max()),
                     float(dd.min()), float(dd.min(axis=1).max())))

    md.attach(record, interval=10)
    if args.traj:
        from ase.io import write

        def frame():
            write(args.traj, at, append=True)

        md.attach(frame, interval=10)
    md.run(args.steps)
    a = np.array(rows)
    etot = a[:, 3]
    drift = np.polyfit(a[:, 0], etot, 1)[0] * 1000.0             # eV per ps
    out = {"model": args.model, "forces_vs_fd": ferr, "cn": cn.tolist(), "steps": args.steps,
           "timestep_fs": args.timestep, "temperature_K": args.temperature,
           "etot_std_meV": float(etot.std() * 1000), "etot_range_meV": float(np.ptp(etot) * 1000),
           "drift_meV_per_ps": float(drift * 1000), "epot_std_meV": float(a[:, 1].std() * 1000),
           "cn_range": [float(a[:, 4].min()), float(a[:, 5].max())],
           "min_distance": float(a[:, 6].min()),
           "max_nearest_neighbour": float(a[:, 7].max()), "structure": args.structure,
           "atoms": len(at),
           "seconds": time.perf_counter() - t0, "trace": a.tolist()}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"NVE {args.steps} x {args.timestep} fs: E_tot std {out['etot_std_meV']:.2f} meV, "
          f"range {out['etot_range_meV']:.2f} meV, drift {out['drift_meV_per_ps']:.2f} meV/ps "
          f"(E_pot std {out['epot_std_meV']:.0f} meV), CN {out['cn_range']}, "
          f"bonds {out['min_distance']:.2f} A .. nearest neighbour <= "
          f"{out['max_nearest_neighbour']:.2f} A, "
          f"{out['seconds']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
