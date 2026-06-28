

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
from utils.loss import Loss 
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
    opt_elem = ptbp_args.multi_element

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
            with open(ptbp_args.para_file, "w") as para:
                para.write('#r0_w,r0_d,c_rep,kxc,cov_radius,mu\n')

            with open(ptbp_args.log_file, 'w') as output:
                    output.write('#r0_w,r0_d,c_rep,kxc,cov_radius,mu_E,mu_V,mu_B,penalty,mu\n')

            dft_infile = path / ptbp_args.ref_dir / ptbp_args.dft_file
            assert dft_infile.exists(), f'Seems that the input path {dft_infile} does not exist!'
            dft_input = ase.io.read(dft_infile, ':')
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
            cov_radius = covalent_radii[atomic_numbers[-1]] / Bohr

            print(f"""\n#=================Input=================#:
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
                        path=path)

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
                _n_calls = ptbp_args.n_calls if ptbp_args.n_calls is not None else 11
                res = loss.optimize(cov_radius=cov_radius,
                                    checkpoint_saver=checkpoint_saver,
                                    n_calls=_n_calls)
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
                from types import SimpleNamespace as _SN
                with open(_result_pkl, 'wb') as _fh:
                    _pickle.dump(_SN(**_snapshot), _fh)
                print(f"[result.pkl] Snapshot of optimisation written to {_result_pkl}")

            print('Best parameter:\n', res.x,'\n', res.fun)
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

            best_folder = path / ptbp_args.results_dir / f"opt_{res.x[0]:.3f}_{res.x[1]:.3f}"
            backup_dir  = path / f"opt_{res.x[0]:.3f}_{res.x[1]:.3f}"

            os.system(f"cp -r {best_folder} {backup_dir}")
            os.system("tar -zcvf results.tgz results && rm -r results")


        # Optimize the Repulsion part!
        elif ptbp_args.parameters == ["sigma_rep"]:
            print("Optimizing the Repulsion part! ")
            r0_w_best, r0_d_best, p = ptbp_args.confinement_parameters

            with open(ptbp_args.para_file, "w") as para:
                para.write('#r0_w,r0_d,c_rep,kxc,cov_radius,mu\n')

            with open(ptbp_args.log_file, 'w') as output:
                    output.write('#r0_w,r0_d,c_rep,kxc,cov_radius,mu_E,mu_V,mu_B,penalty,mu\n')

            dft_infile = path / ptbp_args.ref_dir / ptbp_args.dft_file 
            assert dft_infile.exists(), f'Seems that the input path {dft_infile} does not exist!'
            dft_input = ase.io.read(dft_infile, ':')
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
            cov_radius = covalent_radii[atomic_numbers[-1]] / Bohr

            print(f"""\n#=================Input=================#:
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
                        path=path)

            mu = loss.Loss_EnergyGeometry(r0_w_best, r0_d_best)
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
            cov_radius = covalent_radii[atomic_numbers[-1]] / Bohr

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

    if ptbp_args.symbols is not None: 
        if ptbp_args.multi_element is not None:
            symbol_list = [s for s in ptbp_args.symbols if s not in opt_elem] + opt_elem
        else:
            symbol_list = [s for s in ptbp_args.symbols]
            atomic_numbers_list = [atomic_numbers[s] for s in symbol_list]

    elif ptbp_args.dft_file is not None:
        dft_infile = path / ptbp_args.ref_dir / ptbp_args.dft_file 
        assert dft_infile.exists(), f'Seems that the input path {dft_infile} does not exist!'
        dft_input = ase.io.read(dft_infile, ':')
        atomic_numbers_list, atomic_numbers_exclusive , atomic_symbols, atomic_symbols_exclusive = loss.get_elements_from_dataset(dft_input, opt_elem)
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

            # Get c_rep for repulsion if exist
            if ptbp_args.repulsive_parameters is not None:
                sigma_rep_list = ptbp_args.repulsive_parameters[0::2]
                sigma_kxc_list = ptbp_args.repulsive_parameters[1::2]

            superposition = 'density'

            print('r0_dict', r0_dict)
            print('p_dict', p_dict)

        else:
            raise ValueError(f"The known parameters {ptbp_args.known_parameters} are not defined!")
    
    hp = hotpy(r0_dict, p_dict, workdir=path, slator_p=path)
    hp.full_hotcent(superposition, 
                    opt_elem, 
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

         

        
        
