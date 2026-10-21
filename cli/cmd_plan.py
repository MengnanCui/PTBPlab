"""``ptbp plan DATASET``: the computation strategy of a run, without running it (M3c).

Prints what one BO step computes (structures, Gamma-only ones, ranges of atoms,
orbitals and k-points, hotcent tables), the plan ``ptbp optimize`` would use with the
same options (processes x threads per stage and why), the estimated time of a BO step
and of the whole run, the memory of a DFTB process, and suggestions.  ``--json``
prints the plan as ``plan.json`` holds it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def add_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        'plan',
        help="Show the parallel plan and estimated run time of an optimisation, without running it.",
        description="Estimate what `ptbp optimize DATASET` with the same options would do: "
                    "the processes and threads of each stage of a BO step, its estimated "
                    "time, the memory per DFTB process and suggestions.",
    )
    p.add_argument('dataset', metavar='DATASET', help="Path to the DFT reference (xyz/extxyz/traj).")
    p.add_argument('--mode', default='auto', choices=['auto', 'energygeometry', 'dataset', 'reaction'],
                   help="Loss of the optimisation (as `ptbp optimize --mode`). Default: auto.")
    p.add_argument('--engine', default='dftbplus', choices=['dftbplus', 'pydftb'],
                   help="DFTB engine. Default: %(default)s.")
    p.add_argument('--kpts', type=float, default=5.0, metavar='F',
                   help="k-point density. Default: %(default)s.")
    p.add_argument('--eos-points', type=int, default=11, metavar='N', dest='eos_points',
                   help="Structures per EOS group. Default: %(default)s.")
    p.add_argument('--optimizer', default='bo', choices=['bo', 'parallel_bo', 'pso'],
                   help="Outer-loop optimiser. Default: bo.")
    p.add_argument('--n-particles', type=int, default=4, metavar='N', dest='n_particles',
                   help="Batch width (parallel_bo) or swarm size (pso). Default: 4.")
    p.add_argument('--n-calls', type=int, default=None, metavar='N', dest='n_calls',
                   help="BO/PSO evaluations. Default: 11.")
    p.add_argument('--ncores', default='auto', metavar='N|auto',
                   help="Core budget (as `ptbp optimize --ncores`). Default: auto.")
    p.add_argument('--multi-element', nargs='+', default=None, metavar='E', dest='multi_element',
                   help="Elements to optimise in a multi-element dataset.")
    p.add_argument('--probe', action='store_true',
                   help="Measure this machine's speed first (about 1 s; cached per machine).")
    p.add_argument('--json', action='store_true', help="Print the plan as JSON.")
    p.set_defaults(func=run)
    return p


def run(args: argparse.Namespace) -> int:
    import ase.io

    from utils.parallel import detect_cpu
    from utils.recommender import bolt_for, make_plan, render_report, workload_from_dataset

    path = Path(args.dataset)
    if not path.exists():
        print(f"ptbp plan: {path} does not exist")
        return 2
    mode = args.mode
    if mode == 'auto':
        from cli import auto_mode
        mode, why = auto_mode.detect_mode(path)
        if not args.json:
            print(f"[auto] mode = {mode} ({why})")
    if mode == 'bandstructure':
        print("ptbp plan: the band-structure mode runs serially; there is nothing to plan")
        return 2
    try:
        budget = detect_cpu(args.ncores)
    except ValueError as err:
        print(f"ptbp plan: {err}")
        return 2
    structures = ase.io.read(path, ':')
    bolt = bolt_for(structures, mode, args.eos_points, args.multi_element)
    workload = workload_from_dataset(structures, bolt, kpts_density=args.kpts,
                                     eos_points=args.eos_points, engine=args.engine, mode=mode,
                                     opt_elem=args.multi_element or ())
    plan = make_plan(budget, workload, optimizer=args.optimizer, n_particles=args.n_particles,
                     n_calls=args.n_calls if args.n_calls is not None else 11, probe=args.probe)
    if args.json:
        print(json.dumps(plan.to_json(), indent=2))
    else:
        print(render_report(plan))
    return 0
