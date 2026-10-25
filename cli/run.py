

import copy
import math
import os

import ase
import numpy as np
from ase import Atoms

np.random.seed(66)
import random
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from ase.data import atomic_numbers, chemical_symbols, covalent_radii
from ase.units import *
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from skopt import gp_minimize, load
from skopt.callbacks import CheckpointSaver
from skopt.plots import (plot_convergence, plot_evaluations,
                         plot_gaussian_process, plot_objective,
                         plot_objective_2D)
from skopt.space import Categorical, Integer, Real
from scipy.optimize import Bounds, basinhopping, brute, fmin


import multiprocessing as mp
from tools.physrep import physrep
# from hotpy import hotpy
from math import floor
import argparse
from argparse import RawTextHelpFormatter


from cli.arg_parser import build_default_arg_parser
import sys
from pathlib import Path
import ase.io
import utils.data as data 
from utils.hotpy import hotpy
from utils.loss import Loss, penalty_radius
from utils.parameters_set import get_PTBP, get_QNplusRep, get_PRIOR
hp = hotpy()

# os.system('export ASE_DFTB_COMMAND="/home/mengnancui/Software/dftbplus/dftbplus-23.1.x86_64-linux/bin/dftb+" > PREFIX.out')
# os.system('export PATH=$PATH:/home/mengnancui/Software/dftbplus/dftbplus-23.1.x86_64-linux/bin')

# Debug-only hardcoded paths — disabled so the real CLI args are honoured.
# Set ASE_DFTB_COMMAND in your shell if dftb+ is not on PATH.
# os.environ['ASE_DFTB_COMMAND'] = "/home/mengnancui/Software/dftbplus/dftbplus-23.1.x86_64-linux/bin/dftb+"

# sys.argv = ['run.py', '--skf_generator', 'full', '--symbols', 'H', 'O', '--known_parameters', 'ptbp']

# sys.argv = ['./cli/run.py', '--optimization_option', 'energygeometry', '--parameters', 'r0_w', 'r0_d', 'sigma_rep', '--superposition', 'density']

ptbp_args = build_default_arg_parser()
print("PTBP args: ", ptbp_args)

# DFTB engine + electronic smearing for every calculation of this run
# (module-level settings, inherited by the forked workers of parallel_bo).
from utils.engines import configure as _configure_engine
_device = getattr(ptbp_args, 'device', None)
if _device == 'auto' and getattr(ptbp_args, 'pydftb_backend', None) == 'torch':
    # M4d: look for a GPU in a subprocess, so that this process does not initialise CUDA
    # before it forks workers (with NumPy, configure refuses 'auto' on every machine)
    from pydftb.backend.devices import best_device as _best_device, probe_accelerators as _probe
    _found = _probe()
    _device = _best_device(_found)
    print(f"[engine] --device auto: {_device} (torch {_found.get('torch')}, GPUs "
          f"{_found.get('cuda') or _found.get('xpu') or ('mps' if _found.get('mps') else 'none')})")
print("[engine]", _configure_engine(
    engine=getattr(ptbp_args, 'engine', None),
    smearing=getattr(ptbp_args, 'smearing', None),
    smearing_temperature=getattr(ptbp_args, 'smearing_temperature', None),
    mp_order=getattr(ptbp_args, 'mp_order', None),
    pydftb_forces=getattr(ptbp_args, 'pydftb_forces', None),
    pydftb_backend=getattr(ptbp_args, 'pydftb_backend', None),
    pydftb_device=_device,
    pydftb_dtype=getattr(ptbp_args, 'dtype', None),
))
# hotcent's confinement scheme (M4e): self-consistent (hotcent 1.x, r0_d confines the density)
# unless --confinement perturbative; process-wide, so the forked hotcent workers inherit it
from utils.hotpy import set_confinement as _set_confinement
_set_confinement(getattr(ptbp_args, 'confinement', None) or 'selfconsistent')
print(f"[hotcent] confinement {getattr(ptbp_args, 'confinement', None) or 'selfconsistent'}")
from utils.engines import is_gpu as _is_gpu
if _is_gpu() and getattr(ptbp_args, 'optimizer', 'bo') in ('parallel_bo', 'pso'):
    raise SystemExit("--optimizer parallel_bo/pso evaluate BO points in parallel processes; "
                     "with a GPU (--device) use --optimizer bo (M4d)")
if getattr(ptbp_args, 'optimizer', 'bo') in ('gradient', 'bo_grad'):
    # the gradient optimisers (M4c) need pydftb.autodiff (torch) and an r0 loss
    from utils.gradient_opt import check_requirements as _check_gradient
    _check_gradient(getattr(ptbp_args, 'engine', None) or 'dftbplus',
                    ptbp_args.optimization_option.lower(),
                    getattr(ptbp_args, 'pydftb_backend', None) or 'numpy')

from utils.parallel import detect_cpu as _detect_cpu, set_plan as _set_plan, simple_plan as _simple_plan


def _type_model(dft_input):
    """The types of this run and their parameter sets (milestone M5b, utils.typemodel): one
    set per element by default, one for all with --shared_parameters, switched types from
    --types, --multi_element fits only those elements.  Switched types need pydftb (DFTB+
    through ASE writes element symbols only) and BO or PSO (pydftb.autodiff has no mixing)."""
    from utils.typemodel import TypeModel, load_declarations, parse_starts
    try:
        model = TypeModel.from_dataset(
            dft_input, shared=bool(getattr(ptbp_args, 'shared_parameters', None)),
            declarations=load_declarations(getattr(ptbp_args, 'types', None)),
            starts=parse_starts(getattr(ptbp_args, 'type_start', None)),
            multi_element=ptbp_args.multi_element)
    except ValueError as err:
        raise SystemExit(f"types: {err}")
    if model.switches:
        from utils.engines import DFTBPLUS_SWITCHING_ERROR, set_type_switching
        if (getattr(ptbp_args, 'engine', None) or 'dftbplus') != 'pydftb':
            raise SystemExit(DFTBPLUS_SWITCHING_ERROR)
        if getattr(ptbp_args, 'pydftb_backend', 'numpy') == 'torch':
            raise SystemExit("type switching (M5b) runs on pydftb's NumPy engine: drop "
                             "--pydftb_backend torch")
        if getattr(ptbp_args, 'optimizer', 'bo') in ('gradient', 'bo_grad'):
            raise SystemExit("--optimizer gradient/bo_grad: pydftb.autodiff has no type "
                             "switching (the mixed blocks and the weights' terms); use --optimizer "
                             "bo, parallel_bo or pso with switched types")
        set_type_switching(model.switches)
    print(model.describe())
    # M5b-2: switched types with the energygeometry loss may leave a type without data
    from utils.typemodel import energygeometry_warning
    warning = energygeometry_warning(model, ptbp_args.optimization_option)
    if warning:
        print(warning, flush=True)
    return model


def _resolve_n_calls(types):
    """--n-calls, or its default (M5b-2): 11 for up to 2 optimised dimensions, 10 + 5 x dims
    above (utils.typemodel.default_n_calls)."""
    from utils.typemodel import default_n_calls
    if ptbp_args.n_calls is not None:
        return ptbp_args.n_calls
    dims = types.n_params if types is not None else 2
    n = default_n_calls(dims)
    if dims > 2:
        print(f"[n_calls] {n} evaluations: the default for {dims} optimised dimensions "
              f"(10 + 5 x {dims}; 11 up to 2 dimensions); --n-calls sets another number")
    return n


def _parallel_plan(dft_input, bolt, opt_elem=(), labels=None, n_calls=None):
    """CPU budget and parallel plan of this run (milestone M3): chosen by the cost-model
    recommender (M3c), printed, saved as plan.json and set process-wide, so every BO
    step follows it."""
    budget = _detect_cpu(getattr(ptbp_args, 'ncores', None))
    engine = getattr(ptbp_args, 'engine', None) or 'dftbplus'
    try:
        from utils.recommender import make_plan, workload_from_dataset
        from utils.engines import settings as _engine_settings
        workload = workload_from_dataset(
            dft_input, bolt, kpts_density=ptbp_args.kpt_density, eos_points=ptbp_args.eos_points,
            engine=engine, mode=ptbp_args.optimization_option.lower(), opt_elem=opt_elem,
            **({} if labels is None else {'labels': labels}))
        es, backend, accelerators = _engine_settings(), None, None
        if engine == 'pydftb' and es['pydftb_backend'] == 'torch':      # M4d
            import torch
            backend = {'pydftb_backend': 'torch', 'device': es['pydftb_device'],
                       'dtype': es['pydftb_dtype'], 'torch': torch.__version__}
        elif engine == 'pydftb':
            import importlib.util
            if importlib.util.find_spec('torch') is not None:            # a GPU to suggest?
                from pydftb.backend.devices import probe_accelerators
                accelerators = probe_accelerators()
        if n_calls is None:
            n_calls = ptbp_args.n_calls if ptbp_args.n_calls is not None else 11
        plan = make_plan(budget, workload, optimizer=ptbp_args.optimizer,
                         n_particles=ptbp_args.n_particles, n_calls=n_calls,
                         backend=backend, accelerators=accelerators)
    except Exception as err:                    # planning must never stop a run
        print(f"[plan] cost model unavailable ({type(err).__name__}: {err}); simple rule")
        eos = max(1, int(ptbp_args.eos_points))
        plan = _simple_plan(budget, int(len(dft_input) / eos) * eos, engine=engine,
                            optimizer=ptbp_args.optimizer, particles=ptbp_args.n_particles)
    import hotcent as _hotcent
    from utils.hotpy import confinement as _confinement
    plan.hotcent = {'confinement': _confinement(), 'version': _hotcent.__version__}   # M4e
    _set_plan(plan)
    print(plan.render())
    if engine == 'dftbplus':
        # DFTB+ configuration advisor (M3d): checks the first DFTB+ run, reports only
        from utils import advisor_hook as _advisor_hook
        from utils.engines import settings as _engine_settings
        _advisor_hook.configure(path, legacy=_engine_settings()['smearing'] == 'none')
    try:
        plan.save(path / 'plan.json')
    except OSError as err:
        print(f"[plan] plan.json not written: {err}")
    return plan


# Define the OptimizationScheme
# (1) Elementary system 
# opt_elem = []

# (2) Binary system(eg. C H O Cu)
# - I want to optimize the parameters for Cu, then keep others the same as PTBP
# opt_elem = ['Cu']

path = Path.cwd()
if ptbp_args.multi_element is None:
    opt_elem = []
else:
    # atomic numbers: the helpers compare opt_elem with Atoms.numbers and physrep's keys (the
    # symbols of --multi_element used to be passed as they are); hotcent takes the symbols
    opt_elem = [atomic_numbers[e] if isinstance(e, str) else int(e) for e in ptbp_args.multi_element]
opt_symbols = [chemical_symbols[z] for z in opt_elem]

checkpoint_saver = CheckpointSaver(f"{path}/checkpoint.pkl", compress=9, store_objective=False)  # lifesaver


if ptbp_args.optimization_option is not None:
    _opt = ptbp_args.optimization_option.lower()
    # 'reaction' shares the same plumbing as 'dataset' (per-structure energy +
    # forces loss); it differs in expected dataset shape (multiple chemical
    # formulas) and gets per-formula reporting via the Loss class.
    if _opt in ('energygeometry', 'dataset', 'reaction'):

        # In dataset / reaction mode each Atoms object is its own data point —
        # there is no phase / volume scan, so eos_points defaults to 1 unless
        # the user is deliberately treating the input as a stack of EOS scans.
        if _opt in ('dataset', 'reaction') and ptbp_args.eos_points == 11:
            ptbp_args.eos_points = 1

        # --E0s arrives as a string. JSON dict ({"6": -37.8, ...}) gives the
        # atomic reference energies used in formation-energy loss. The legacy
        # 'ref.xyz' string remains supported.
        if isinstance(ptbp_args.E0s, str) and ptbp_args.E0s.lstrip().startswith('{'):
            import json as _json
            ptbp_args.E0s = {int(k): float(v) for k, v in _json.loads(ptbp_args.E0s).items()}

        # In dataset / reaction mode, when no --E0s was supplied, look for a
        # cached fit on disk first; if missing, regress E0s from the dataset
        # itself and persist the result. The cache lives next to the dataset
        # so a given DFT dataset reuses the same fit across runs.
        _e0s_cache_path = str(path / ptbp_args.ref_dir / 'E0s_dft.json')
        if _opt in ('dataset', 'reaction') and not isinstance(ptbp_args.E0s, dict):
            cached = data.load_E0s_cache(_e0s_cache_path)
            if cached is not None:
                print(f"[E0s] Loaded cached atomic reference energies from {_e0s_cache_path}")
                ptbp_args.E0s = cached
            else:
                print(f"[E0s] No --E0s and no cache; fitting from {ptbp_args.dft_file} ...")
                _atoms_for_fit = ase.io.read(path / ptbp_args.ref_dir / ptbp_args.dft_file, ':')
                ptbp_args.E0s = data.fit_atomic_E0s(_atoms_for_fit)
                data.save_E0s_cache(ptbp_args.E0s, _e0s_cache_path)
                print(f"[E0s] Fitted: {ptbp_args.E0s}")
                print(f"[E0s] Cached to {_e0s_cache_path}")

        # Optimize the Confinement potential and Repulsion part together!
        if ptbp_args.parameters == ["r0_w", "r0_d", "sigma_rep"]:
            print("Optimizing the Confinement potential and Repulsion part together! ")
            dft_infile = path / ptbp_args.ref_dir / ptbp_args.dft_file
            dft_input = ase.io.read(dft_infile, ':') if dft_infile.exists() else None
            types = _type_model(dft_input) if dft_input is not None else None
            typed = types is not None and not types.legacy        # M5b: parameters per type
            n_calls = _resolve_n_calls(types)                     # M5b-2: default by dimensions
            with open(ptbp_args.para_file, "w") as para:
                para.write(types.par_header() if typed else '#r0_w,r0_d,c_rep,kxc,cov_radius,mu\n')

            with open(ptbp_args.log_file, 'w') as output:
                    output.write(types.log_header(_opt) if typed else
                                 '#r0_w,r0_d,c_rep,kxc,cov_radius,mu_E,mu_V,mu_B,penalty,mu\n')

            assert dft_infile.exists(), f'Seems that the input path {dft_infile} does not exist!'
            dft_center = dft_input[floor(ptbp_args.eos_points/2)::ptbp_args.eos_points]
            num_center_atoms = len(dft_center)


            if _opt in ('dataset', 'reaction'):
                print(f"Set boltzmann factor for {_opt.title()} Mode (identical weights)!")
                bolt_f, bolt_f11 = data.get_bolt_factor(dft_center, 'identical', 3000)
                if _opt == 'reaction':
                    from collections import Counter as _Counter
                    _formulas = _Counter(a.get_chemical_formula() for a in dft_input)
                    print(f"Reaction mode: {len(_formulas)} unique formulas detected:")
                    for _f, _n in _formulas.most_common():
                        print(f"  {_f:<24s} {_n} structures")
            elif ptbp_args.multi_element is None:
                print("Set boltmann factor for Elementary System!")
                bolt_f, bolt_f11 = data.get_bolt_factor(dft_center, 'boltzmann', 3000)
            else:
                print("Set boltmann factor for Binary System!")
                bolt_f, bolt_f11 = data.get_bolt_factor(dft_center, 'identical', 3000)


            # Get the atomic number of the system
            atomic_numbers, atomic_numbers_exclusive , atomic_symbols, atomic_symbols_exclusive = data.get_elements_from_dataset(dft_input, opt_elem)
            cov_radius = penalty_radius(dft_input, opt_elem)

            print(f"""\n#=================Input=================#:
Optimization target: {ptbp_args.optimization_option}
Input file: {dft_infile}
Number of structure: {len(dft_input)}
Number of center atoms: {num_center_atoms}
Atomic number: {atomic_numbers}
Covalent radius: {cov_radius}
Boltzmann factor: {bolt_f.tolist()}
            """)
            if typed:
                # every pair of types gets its tables (fixed types included), so no opt_elem
                _parallel_plan(dft_input, bolt_f11, (),
                               labels=types.element_of if types.switches else None,
                               n_calls=n_calls)
            else:
                _parallel_plan(dft_input, bolt_f11, opt_elem, n_calls=n_calls)

            loss = Loss(args_input=ptbp_args,
                        dft_input=dft_input,
                        bolt_f11=bolt_f11,
                        opt_elem=[] if typed else opt_elem,
                        cov_radius=cov_radius,
                        path=path,
                        types=types)

            _result_pkl = path / 'result.pkl'
            if ptbp_args.postprocess_only:
                # Reuse a previous run's optimisation snapshot — skip the
                # expensive hotcent + DFTB+ + brute pipeline entirely.
                import pickle as _pickle
                if not _result_pkl.exists():
                    raise FileNotFoundError(
                        f"--postprocess_only requested but no {_result_pkl} "
                        "found in cwd. Run the optimisation once without "
                        "--postprocess_only first.")
                with open(_result_pkl, 'rb') as _fh:
                    res = _pickle.load(_fh)
                print(f"[postprocess] Loaded prior result from {_result_pkl}: "
                      f"best x={res.x}, fun={res.fun}, "
                      f"{len(getattr(res, 'x_iters', [])) or 0} evaluation(s).")
            else:
                res = loss.optimize(cov_radius=cov_radius,
                                    checkpoint_saver=checkpoint_saver,
                                    n_calls=n_calls)
                # Snapshot the result so post-processing can be redone later
                # via `--postprocess_only` without rerunning the optimiser.
                import pickle as _pickle
                _snapshot = {
                    'x':         list(map(float, res.x)),
                    'fun':       float(res.fun),
                    'x_iters':   [list(map(float, p)) for p in getattr(res, 'x_iters', [])],
                    'func_vals': list(map(float, getattr(res, 'func_vals', []))),
                    'optimizer': ptbp_args.optimizer,
                }
                if getattr(res, 'best_dir', None):       # gradient optimisers (M4c)
                    _snapshot['best_dir'] = res.best_dir
                from types import SimpleNamespace as _SN
                with open(_result_pkl, 'wb') as _fh:
                    _pickle.dump(_SN(**_snapshot), _fh)
                print(f"[result.pkl] Snapshot of optimisation written to {_result_pkl}")

            print('Best parameter:\n', res.x,'\n', res.fun)
            if loss.failed_points:                  # M5b-2: failed evaluations of this run
                print(f"[failures] {len(loss.failed_points)} failed evaluation(s): each counts "
                      f"2 x the worst successful loss before it (a fixed value before the first "
                      f"success; see the [failure] lines)")
                if loss.is_failure(res.x):
                    print("[failures] WARNING: the lowest loss belongs to a failed evaluation "
                          "(no evaluation succeeded with a lower loss): these parameters have "
                          "no loss of the whole dataset")
            if typed:
                for _lab, (_w, _d) in types.r0(res.x).items():
                    print(f"  {_lab}: r0_w = {_w:.6f}, r0_d = {_d:.6f}")
            else:
                r0_w_best, r0_d_best = res.x

            os.chdir(path)
            # skopt's diagnostic plots only work with skopt OptimizeResult
            # objects; PSO and parallel_bo paths return lightweight stand-ins,
            # so we draw a plain matplotlib convergence curve from
            # res.func_vals instead.
            if ptbp_args.optimizer == 'bo':
                plot_convergence(res)
                plt.savefig('convergence.png', dpi=300)
                plt.clf()

                eval = plot_evaluations(res, bins=20)
                eval[0][0].figure.savefig('eval.png', dpi=300, bbox_inches='tight')

                obje = plot_objective(res, sample_source='result')
                obje[0][0].figure.savefig('obj.png', dpi=300, bbox_inches='tight')
            else:
                # Generic best-so-far convergence curve usable for any optimiser.
                vals = list(res.func_vals)
                if vals:
                    best_so_far = []
                    cur = float('inf')
                    for v in vals:
                        cur = min(cur, float(v))
                        best_so_far.append(cur)
                    plt.figure()
                    plt.plot(range(1, len(vals) + 1), vals, 'o-', label='per-eval μ', alpha=0.5)
                    plt.plot(range(1, len(vals) + 1), best_so_far, 'r-', label='best so far')
                    plt.xlabel('evaluation')
                    plt.ylabel('μ')
                    plt.yscale('log')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig('convergence.png', dpi=300)
                    plt.clf()

            # gradient optimisers name their directories with more digits (best_dir)
            best_name = getattr(res, 'best_dir', None) or (
                types.directory_name(res.x) if typed else f"opt_{res.x[0]:.3f}_{res.x[1]:.3f}")
            best_folder = path / ptbp_args.results_dir / best_name
            backup_dir  = path / best_name

            os.system(f"cp -r {best_folder} {backup_dir}")
            os.system("tar -zcvf results.tgz results && rm -r results")


        # Optimize the Repulsion part!
        elif ptbp_args.parameters == ["sigma_rep"]:
            print("Optimizing the Repulsion part! ")
            dft_infile = path / ptbp_args.ref_dir / ptbp_args.dft_file
            assert dft_infile.exists(), f'Seems that the input path {dft_infile} does not exist!'
            dft_input = ase.io.read(dft_infile, ':')
            types = _type_model(dft_input)
            typed = not types.legacy                 # M5b: r0_w r0_d of every parameter set
            if typed:
                x_fixed = list(ptbp_args.confinement_parameters or [])
                if len(x_fixed) != types.n_params:
                    raise SystemExit(f"--confinement_parameters: {types.n_params} values "
                                     f"({types.describe_sets()}) for these types")
            else:
                r0_w_best, r0_d_best, p = ptbp_args.confinement_parameters

            with open(ptbp_args.para_file, "w") as para:
                para.write(types.par_header() if typed else '#r0_w,r0_d,c_rep,kxc,cov_radius,mu\n')

            with open(ptbp_args.log_file, 'w') as output:
                    output.write(types.log_header(_opt) if typed else
                                 '#r0_w,r0_d,c_rep,kxc,cov_radius,mu_E,mu_V,mu_B,penalty,mu\n')

            dft_center = dft_input[floor(ptbp_args.eos_points/2)::ptbp_args.eos_points]
            num_center_atoms = len(dft_center)

            
                    
            if ptbp_args.multi_element is None:
                print("Set boltmann factor for Elementary System!")
                bolt_f, bolt_f11 = data.get_bolt_factor(dft_center, 'boltzmann', 3000)
            else:
                print("Set boltmann factor for Binary System!")
                bolt_f, bolt_f11 = data.get_bolt_factor(dft_center, 'identical', 3000)


            # Get the atomic number of the system
            atomic_numbers, atomic_numbers_exclusive , atomic_symbols, atomic_symbols_exclusive = data.get_elements_from_dataset(dft_input, opt_elem)
            cov_radius = penalty_radius(dft_input, opt_elem)

            print(f"""\n#=================Input=================#:
Optimization target: {ptbp_args.optimization_option}
Input file: {dft_infile}
Number of structure: {len(dft_input)}
Number of center atoms: {num_center_atoms}
Atomic number: {atomic_numbers}
Covalent radius: {cov_radius}
Boltzmann factor: {bolt_f.tolist()}
            """)
            if typed:
                _parallel_plan(dft_input, bolt_f11, (),
                               labels=types.element_of if types.switches else None)
            else:
                _parallel_plan(dft_input, bolt_f11, opt_elem)

            loss = Loss(args_input=ptbp_args,
                        dft_input=dft_input,
                        bolt_f11=bolt_f11,
                        opt_elem=[] if typed else opt_elem,
                        cov_radius=cov_radius,
                        path=path,
                        types=types)

            # Loss_EnergyGeometry takes one list (it was called with two arguments)
            mu = loss.Loss_EnergyGeometry(x_fixed if typed else [r0_w_best, r0_d_best])
            print("mu: ", mu)
        

    if ptbp_args.optimization_option.lower() == "bandstructure":
            
            # Data preparation
            DEFAULT_DFT_BAND_FILE = ptbp_args.dft_file #= 'dft.json'
            with open(ptbp_args.parameter_file, 'w') as para:
                    para.write('#r0_w,r0_d,p,cov_radius,mu_homo,mu_lumo,mu\n')

            dft_infile = p / ptbp_args.ref_dir / ptbp_args.dft_file
            dft_bs = p / ptbp_args.ref_dir / ptbp_args.dft_band_file
            assert dft_infile.exists(), f'Seems that the input path {dft_infile} does not exist!'
            assert dft_bs.exists(), f'Seems that the input path {dft_bs} does not exist!'
            dft_input = ase.io.read(dft_infile)

            
            if ptbp_args.multi_element is not None:
                opt_elem = [x for x in ptbp_args.multi_element]

            atomic_numbers, atomic_numbers_exclusive , atomic_symbols, atomic_symbols_exclusive = loss.get_elements_from_dataset(dft_input, opt_elem)
            cov_radius = penalty_radius(dft_input, opt_elem)

            bolt_f, bolt_f11 = data.get_bolt_factor(dft_center, 'identical', 3000)
            

            print(f"""\n#===================================Input===================================#:
Optimization target: {ptbp_args.optimization_option}
Input file: {dft_infile}
Number of structure: {len(dft_input)}
Number of center atoms: {num_center_atoms}
Atomic number: {atomic_numbers}
Covalent radius: {cov_radius}
Boltzmann factor: {bolt_f.tolist()}
            """)

            loss = Loss(args_input=ptbp_args, 
                        dft_input=dft_input,
                        bolt_f11=bolt_f11,
                        opt_elem=opt_elem, 
                        cov_radius=cov_radius,
                        path=path)

            # Optimization
            _n_calls = ptbp_args.n_calls if ptbp_args.n_calls is not None else 100
            res = loss.optimize(cov_radius, checkpoint_saver, n_calls=_n_calls)

            print('Best parameter:\n', res.x,'\n', res.fun)
            r0_best, p_best = res.x

            os.chdir(p)
            plot_convergence(res)
            plt.savefig('convergence.png', dpi=300)
            plt.clf()

            eval = plot_evaluations(res, bins=20)
            eval[0][0].figure.savefig('eval.png', dpi=300, bbox_inches='tight')

            obje = plot_objective(res, sample_source='result')
            obje[0][0].figure.savefig('obj.png', dpi=300, bbox_inches='tight')


            best_folder = path / ptbp_args.results_dir / f"opt_{res.x[0]:.3f}_{res.x[1]:.3f}"
            backup_dir  = path / f"opt_{res.x[0]:.3f}_{res.x[1]:.3f}"

            os.system(f"cp -r {best_folder} {backup_dir}")
            os.system("tar -zcvf results.tgz results && rm -r results")


                
if ptbp_args.skf_generator is not None:
    print("Generate the skf file with Existed Parameter sets!")

    from ase.data import atomic_numbers as _Z     # `atomic_numbers` is reassigned above
    if ptbp_args.symbols is not None:
        if ptbp_args.multi_element is not None:
            symbol_list = [s for s in ptbp_args.symbols if s not in opt_symbols] + opt_symbols
        else:
            symbol_list = [s for s in ptbp_args.symbols]
        atomic_numbers_list = [_Z[s] for s in symbol_list]

    elif ptbp_args.dft_file is not None:
        dft_infile = path / ptbp_args.ref_dir / ptbp_args.dft_file
        assert dft_infile.exists(), f'Seems that the input path {dft_infile} does not exist!'
        dft_input = ase.io.read(dft_infile, ':')
        # (this used loss.get_elements_from_dataset, but there is no `loss` here)
        atomic_numbers_list, atomic_numbers_exclusive , atomic_symbols, atomic_symbols_exclusive = data.get_elements_from_dataset(dft_input, opt_elem)
        symbol_list = atomic_symbols


    if ptbp_args.skf_generator.lower() in ["full", "band"]:
        if ptbp_args.known_parameters.lower() == "ptbp":

            r0_dict = {symbol: [get_PTBP()[symbol]['r0_w'], get_PTBP()[symbol]['r0_d']] for symbol in symbol_list}

            # Get p
            p_dict = {symbol: 2.0 for symbol in symbol_list}

            # Get c_rep for repulsion if exist
            sigma_rep_list = [get_PTBP()[symbol]['sigma_rep'] for symbol in symbol_list]
            kxc_list = [0.0 for symbol in symbol_list]
            superposition = 'density'


            print('r0_dict', r0_dict)
            print('p_dict', p_dict)
            print('sigma_rep_list', sigma_rep_list)
        
        elif ptbp_args.known_parameters.lower() in ["qnplusrep", "quasinano2013"]:
            
            # Get r_wave and r_dens for wavefunction and density function, respectively
            r0_dict = {symbol: [get_QNplusRep()[symbol]['r0_quasinano'], get_QNplusRep()[symbol]['r0_quasinano']] for symbol in symbol_list}

            # Get p
            p_dict = {symbol: get_QNplusRep()[symbol]['p_quasinano'] for symbol in symbol_list}

            # Get c_rep for repulsion if exist
            sigma_rep_list = [get_QNplusRep()[symbol]['sigma_rep'] for symbol in symbol_list]
            kxc_list = [0.0 for symbol in symbol_list]

            superposition = 'potential'

            print('r0_dict', r0_dict)
            print('p_dict', p_dict)
            print('sigma_rep_list', sigma_rep_list)

        elif ptbp_args.known_parameters.lower() == "prior":

            r0_dict = {symbol: [get_PRIOR()[symbol]['r0_w'], get_PRIOR()[symbol]['r0_d']] for symbol in symbol_list}

            # Get p
            p_dict = {symbol: 2.0 for symbol in symbol_list}

            # Get c_rep for repulsion if exist
            sigma_rep_list = [get_PRIOR()[symbol]['sigma_rep'] for symbol in symbol_list]
            kxc_list = [0.0 for symbol in symbol_list]
            superposition = 'density'


            print('r0_dict', r0_dict)
            print('p_dict', p_dict)
            print('sigma_rep_list', sigma_rep_list)

        elif ptbp_args.known_parameters.lower() == "personal":

            # Get r_wave and r_dens for wavefunction and density function, respectively
            # r0_list = list(ptbp_args.confinement_r0)
            # p_list = list(args.confinement_p)
            confinement_parameters = ptbp_args.confinement_parameters
            r0_dict = {symbol: [confinement_parameters[3*i], confinement_parameters[3*i+1]] for i, symbol in enumerate(symbol_list)}
            p_dict = {symbol: confinement_parameters[3*i+2] for i, symbol in enumerate(symbol_list)}

            # Get c_rep for repulsion if exist (pairs sigma_rep, kxc per element; physrep
            # reads kxc_list, which this branch used to leave undefined)
            if ptbp_args.repulsive_parameters is not None:
                sigma_rep_list = ptbp_args.repulsive_parameters[0::2]
                kxc_list = ptbp_args.repulsive_parameters[1::2]
            elif ptbp_args.skf_generator.lower() == "full":
                raise SystemExit("--known_parameters personal with --skf_generator full needs "
                                 "--repulsive_parameters (sigma_rep kxc per element)")

            superposition = 'density'

            print('r0_dict', r0_dict)
            print('p_dict', p_dict)

        else:
            raise ValueError(f"The known parameters {ptbp_args.known_parameters} are not defined!")
    
    hp = hotpy(r0_dict, p_dict, workdir=path, slator_p=path)
    hp.full_hotcent(superposition,
                    opt_symbols,                          # hotcent takes symbols
                    ptbp_args.known_parameters.lower())   # run Hotcent
    

    if ptbp_args.skf_generator.lower() in ["full", "rep"]:
        pr = physrep(atomic_numbers_list, 
                 rmin=0.5,
                 rmax=20.0,
                 npoints=700,
                 scale_cov=sigma_rep_list,
                 k_xc=kxc_list)
        
        if len(opt_elem) == 0: 
            pr.write_skf(ptbp_args.known_parameters)
        else:
            pr.write_specific_skf(opt_elem, ptbp_args.known_parameters)

         

        
        
