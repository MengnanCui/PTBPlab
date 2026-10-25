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
    p.add_argument('--optimizer', default='bo',
                   choices=['bo', 'parallel_bo', 'pso', 'gradient', 'bo_grad'],
                   help="Outer-loop optimiser. Default: bo (the step estimates are those of "
                        "a BO step; a gradient evaluation costs more, see the M4c report).")
    p.add_argument('--n-particles', type=int, default=4, metavar='N', dest='n_particles',
                   help="Batch width (parallel_bo) or swarm size (pso). Default: 4.")
    p.add_argument('--n-calls', type=int, default=None, metavar='N', dest='n_calls',
                   help="BO/PSO evaluations. Default: as `ptbp optimize`, 11 for up to 2 "
                        "optimised dimensions (one parameter set), else 10 + 5 x dims.")
    p.add_argument('--ncores', default='auto', metavar='N|auto',
                   help="Core budget (as `ptbp optimize --ncores`). Default: auto.")
    p.add_argument('--multi-element', nargs='+', default=None, metavar='E', dest='multi_element',
                   help="Elements to optimise in a multi-element dataset (the others keep "
                        "PTBP's published parameters).")
    p.add_argument('--shared-parameters', action='store_true', dest='shared_parameters',
                   help="As `ptbp optimize --shared-parameters` (one parameter set for all "
                        "elements, 2 dimensions): the default --n-calls is then 11.")
    p.add_argument('--switch', nargs=4, action='append', default=None,
                   metavar=('ELEMENT', 'LABELS', 'CENTRE', 'WIDTH'), dest='switch',
                   help="Switched types, as `ptbp optimize --switch` (M5b): hotcent computes "
                        "the tables of every pair of type labels.")
    p.add_argument('--probe', action='store_true',
                   help="Measure this machine's speed first (about 1 s; cached per machine) "
                        "and look for GPUs pydftb could use.")
    p.add_argument('--pydftb-backend', default='numpy', choices=['numpy', 'torch'],
                   dest='pydftb_backend', help="As `ptbp optimize`. Default: %(default)s.")
    p.add_argument('--device', default='cpu', metavar='DEV',
                   help="Torch device for --pydftb-backend torch. Default: %(default)s.")
    p.add_argument('--dtype', default='float64', choices=['float64', 'float32'],
                   help="Precision for --pydftb-backend torch. Default: %(default)s.")
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
    from utils.typemodel import TypeModel, default_n_calls, parse_switch_args
    labels, dims = None, 2
    try:
        model = TypeModel.from_dataset(
            structures, shared=bool(getattr(args, 'shared_parameters', False)),
            declarations=parse_switch_args(getattr(args, 'switch', None)),
            multi_element=args.multi_element)
    except ValueError as err:
        if getattr(args, 'switch', None) or getattr(args, 'shared_parameters', False):
            print(f"ptbp plan: {err}")
            return 2
        model = None                                      # the default --n-calls of 2 dimensions
    if model is not None:
        dims = model.n_params                             # M5b-2: the default --n-calls
        if model.switches:                                # M5b: the tables of the type labels
            labels = model.element_of
    n_calls = args.n_calls if args.n_calls is not None else default_n_calls(dims)
    # M5b: every pair of types gets its tables (--multi_element no longer restricts them)
    workload = workload_from_dataset(structures, bolt, kpts_density=args.kpts,
                                     eos_points=args.eos_points, engine=args.engine, mode=mode,
                                     **({} if labels is None else {'labels': labels}))
    backend = accelerators = None
    if args.engine == 'pydftb' and getattr(args, 'pydftb_backend', 'numpy') == 'torch':   # M4d
        import importlib.metadata
        try:
            version = importlib.metadata.version('torch')
        except importlib.metadata.PackageNotFoundError:
            version = None
        backend = {'pydftb_backend': 'torch', 'device': args.device, 'dtype': args.dtype,
                   'torch': version}
    elif args.engine == 'pydftb' and args.probe:
        from pydftb.backend.devices import probe_accelerators
        accelerators = probe_accelerators()
    plan = make_plan(budget, workload, optimizer=args.optimizer, n_particles=args.n_particles,
                     n_calls=n_calls, probe=args.probe,
                     backend=backend, accelerators=accelerators)
    if args.json:
        print(json.dumps(plan.to_json(), indent=2))
    else:
        print(render_report(plan))
    return 0
