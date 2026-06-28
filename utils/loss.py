
import copy
import json
import math
import os
import re

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
from scipy.optimize import minimize
from skopt import gp_minimize, load
from skopt.space import Categorical, Integer, Real
from scipy.optimize import Bounds, basinhopping, brute, fmin


import multiprocessing as mp
import argparse
from argparse import RawTextHelpFormatter

import copy
import json
import math
import os
import pickle
import pprint as pp
import re
import time
from math import ceil
from pathlib import Path
from typing import Tuple

import ase.io
from ase.data import atomic_numbers, chemical_symbols, covalent_radii
from ase.eos import EquationOfState as EOS
from ase.io import Trajectory
from ase.io.jsonio import read_json
from ase.units import Bohr, kB, kJ
from ase.utils.deltacodesdft import delta
from typing import List, Dict, Tuple
import multiprocessing as mp
from itertools import combinations



from math import pi, sqrt
from pathlib import Path
from utils.hotpy import hotpy  
from utils.calculator import run_calculator, dftbplus_calculator, calc_band_binary, delta_band_binary
from math import pi, sqrt, floor
from tools.physrep import physrep
from utils.parameters_set import get_PTBP, get_QNplusRep, get_PRIOR
import utils.data as data
import utils.plot as plot



class Loss():
    """
    For storing multiple loss functions and related functions
    """

    def __init__(self,
                 args_input: argparse.Namespace=None,
                 dft_input: list=[],
                 bolt_f11:list=[],
                 opt_elem: list=[],
                 superposition: str='density',
                 cov_radius: float=0.5,
                 path:Path='./',
                 ):
        self.args_input = args_input
        self.dft_input = dft_input
        self.bolt_f11 = bolt_f11
        self.superposition = superposition
        self.cov_radius = cov_radius
        self.path = path
        self.hp = hotpy()
        self.opt_elem = opt_elem
    

    def get_forces(self, atoms:list, prefix:str):

            if prefix == '_':
                prefix = ''

            forces_label = f"{prefix}forces"
            forces = []
            for a in atoms:
                forces.extend(a.get_array(forces_label).flatten())

            return forces
    def get_elements_from_dataset(self, dataset, opt_elem):
    
        tot_atomic_numbers = np.unique(np.array([np.unique(mole.numbers).tolist() for mole in dataset]).flatten()) 
        atomic_numbers_exclusive = [i for i in tot_atomic_numbers if i not in opt_elem]
        atomic_numbers = atomic_numbers_exclusive + opt_elem

        atomic_symbols_exclusive = [chemical_symbols[i] for i in atomic_numbers_exclusive]
        atomic_symbols = atomic_symbols_exclusive + [chemical_symbols[num] for num in opt_elem]

        return atomic_numbers, atomic_numbers_exclusive, atomic_symbols, atomic_symbols_exclusive


    def get_energy_per_atom(self, dft:list,dftb:list, bolt_fxx:list=None):
        """
        Calculate the energy per atom

        Note: bolt_fxx is the Boltzmann factor for each structure, it's length is the same as dftb
        provide it while you want to calculate specific structures
        None while you want to calculate all structures, then it will automatically find from files "search_index.txt"

        """
        dftb_energy_per_atom = [dftb_mole.info['energy']/ dftb_mole.get_global_number_of_atoms() for dftb_mole in dftb]
        dft_energy_per_atom  = [dft_mole.get_total_energy()/ dft_mole.get_global_number_of_atoms() for dft_mole in dft]
        min_num = np.argmin(dft_energy_per_atom); print('min_num:', min_num)

        dftb_energy_per_atom_relative = [dftb_energy_per_atom[i] - dftb_energy_per_atom[min_num] for i in range(len(dftb_energy_per_atom))]
        dft_energy_per_atom_relative  = [dft_energy_per_atom[i]  - dft_energy_per_atom[min_num] for i in range(len(dft_energy_per_atom))]

        # delta_energy_per_atom = np.mean(np.abs(np.array(dftb_energy_per_atom_relative) - np.array(dft_energy_per_atom_relative)) )
        if bolt_fxx == None: # which means we want to calculate all structures all at once
            with open ("search_index.txt", 'r') as search_index_file:
                search_index = search_index_file.read().strip().split(',')
            bolt_fxx = [self.bolt_f11[int(i)] for i in search_index]

        delta_energy_per_atom = np.sum(np.abs([dftb_energy_per_atom_relative[m] - dft_energy_per_atom_relative[m] for m in range(len(bolt_fxx)) if bolt_fxx[m] >= 0.1]) / np.sum([bolt_fxx[m] for m in range(len(bolt_fxx)) if bolt_fxx[m] >= 0.1]))
        # This is designed for normalizing structures in this subgroup

        # return mu_E
        return dftb_energy_per_atom_relative, dft_energy_per_atom_relative, delta_energy_per_atom
    

    def get_formation_energy(self, atoms:list, E0:dict, prefix:str):

            if prefix == '_':
                prefix = ''

            energy_label = f"{prefix}energy"

            def _energy_of(mole):
                # Robust to whether energy lives in atoms.info (legacy ASE
                # convention, mirrored back by utils._compat) or only in
                # atoms.calc.results (modern ASE singlepoint). Either ase.io
                # write+read or the inner DFTB+ call may strip one side.
                if energy_label in mole.info:
                    return mole.info[energy_label]
                if mole.calc is not None and energy_label in getattr(mole.calc, 'results', {}):
                    return mole.calc.results[energy_label]
                if not prefix:
                    try:
                        return mole.get_potential_energy()
                    except Exception:
                        pass
                raise KeyError(f"{energy_label!r} not found on Atoms.info, "
                               f".calc.results, nor as get_potential_energy()")

            if isinstance(E0, dict):
                print("Calculating formation energy by E0s...")
                elemental_numbers = [int(i) for i in E0.keys()]
                formation_energy = [
                    _energy_of(mole) - sum(E0[elem] * mole.get_atomic_numbers().tolist().count(elem)
                                           for elem in elemental_numbers)
                    for mole in atoms
                ]
                formation_energy_per_atom = [energy / mole.get_global_number_of_atoms()
                                             for energy, mole in zip(formation_energy, atoms)]

            elif E0 == 'ref.xyz':
                print("Calculating formation energy by reference structure...")
                ref = ase.io.read('ref.xyz')
                E0_energy = _energy_of(ref)
                formation_energy = [_energy_of(mole) - E0_energy for mole in atoms]
                formation_energy_per_atom = [energy / mole.get_global_number_of_atoms()
                                             for energy, mole in zip(formation_energy, atoms)]

            return formation_energy, formation_energy_per_atom
    
    

    def get_atomic_energy(self, numbers, ref_dataset=None):
        """DFTB+ atomic reference energies for each unique element.

        For an isolated atom in a 20 Å cubic box every neighbour is past
        physrep's cutoff, so the result depends on the band-structure SKFs
        (r0_w / r0_d) but not on sigma_rep — meaning we can cache it across
        the inner brute / L-BFGS-B sigma_rep scan within a single BO outer
        iteration and avoid 11+ redundant DFTB+ runs per element.

        Parameters
        ----------
        numbers : iterable[int] | iterable[ase.Atoms]
            Either a list of atomic numbers, or the dataset itself (Atoms
            list) — in the latter case the unique elements are extracted.
        ref_dataset : optional
            Kept for backwards compatibility; unused.
        """
        # Normalise `numbers` to a sorted list of unique atomic numbers.
        if numbers is None:
            unique_Z = []
        elif hasattr(numbers, '__iter__'):
            try:
                unique_Z = sorted({int(z) for z in numbers})
            except (TypeError, ValueError):
                # `numbers` was a dataset (Atoms list) — extract elements
                unique_Z = sorted({int(z) for atoms in numbers
                                   for z in atoms.numbers})
        else:
            raise TypeError(f"Cannot derive elements from {type(numbers)}")

        cache = getattr(self, '_E0_dftb_cache', None)
        if cache is not None and frozenset(unique_Z) in cache:
            return cache[frozenset(unique_Z)]

        dftb_args = {'kpts': [1, 1, 1],
                     'poly_rep': 'No',
                     'SCC': 'Yes',
                     'slater_p': './',
                     'calc_type': 'static'}

        E0s = {}
        for Z in unique_Z:
            atoms = ase.Atoms(chemical_symbols[int(Z)], positions=[[0, 0, 0]])
            atoms.set_cell(np.eye(3) * 20.0)  # 20 Å cubic vacuum
            atoms.set_pbc([True, True, True])
            calc = dftbplus_calculator(atoms, dftb_args)
            atoms.calc = calc
            E0s[int(Z)] = atoms.get_potential_energy()

        if not hasattr(self, '_E0_dftb_cache'):
            self._E0_dftb_cache = {}
        self._E0_dftb_cache[frozenset(unique_Z)] = E0s

        print(f"[E0_dftb] Computed isolated-atom energies: {E0s}")
        return E0s


    def run_calculator_seperately_parallel(self, dft_incopy:Atoms, calculator_code:str, tag:str):
        # def prep_gnuparallel(dataset_infile:str, num_parall, calculator_code:str, tag:str):
        """

        Seperate the dataset and run DFTB calculation parallelly!

        The reason is due to the trivial parallelization setting of DFTB+, basically we simply run each DFTB+ calculation with one CUP core.

        The parallelization is based on the number of eos_points, defaultly, self.args_input.eos_points = 11, therefore, if we have 66 structures, the number of parallelized calculators will be 66/11 = 6.
        Be parallelized respect to args_input.eos_points

        """

        # 1. Divide input file into num_parall folders
        # initialize information
        dataset = dft_incopy#ase.io.read(dataset_infile, ':')
        dataset_file = "ref.xyz"
        # dataset_file = dataset_infile.strip().split('/')[-1]
        # dataset_file_output = dataset_file.replace('.xyz', f'_{tag}.xyz')
        dataset_file_output = "dftb_band.xyz"
        num_atoms = len(dft_incopy)
        num_parall = int(num_atoms / self.args_input.eos_points)
        if num_parall == 0:
            num_parall = 1
        

        # seperate val_dataset into num_parall parts
        size_subs = math.ceil(num_atoms / num_parall)

        subdataset_names = [f"{tag}_{i}/tmp.{dataset_file.split('.')[0]}.xyz" for i in range(num_parall)]
        subdataset_gap_names = [f"{tag}_{i}/tmp.{dataset_file.split('.')[0]}_dftb.xyz" for i in range(num_parall)]
        print(f"subdataset_names: {'  '.join(subdataset_names)}")
        print(f"subdataset_gap_names: {'  '.join(subdataset_gap_names)}")


        # write subdataset
        cmd_file = open("cmd.lst", 'w')
        for n in range(num_parall):
            if self.bolt_f11[n*self.args_input_eos_points + floor(self.args_input_eos_points/2)] >= 0.1:
                k_dens = self.args_input.kpt_density
            else:
                k_dens = 1
            start_index = n * size_subs
            end_index = (n + 1) * size_subs
            subdataset = dataset[start_index: end_index]

            subfolder_name = f"{tag}_{n}"
            if not os.path.exists(subfolder_name):
                os.system(f"mkdir {subfolder_name}")
                os.system(f"cp {self.path}/{calculator_code} {subfolder_name}")
                if tag.lower() != "xtb":
                    os.system(f"cp *.skf {subfolder_name}")
            else:
                os.system(f"cp {self.path}/{calculator_code} {subfolder_name}")
                if tag.lower() != "xtb":
                    os.system(f"cp *.skf {subfolder_name}")
            subdataset_name = subdataset_names[n]  
            ase.io.write(subdataset_name, subdataset)


            # write cmd file to cmd.lst, here i run calculator with python
            cmd_run = f"python  {calculator_code} -sp tmp.{dataset_file.split('.')[0]}.xyz {tag} -d {k_dens}"

            cmd_file.write(f"cd {subfolder_name} ; {cmd_run} >> output\n")
        cmd_file.close()

        print("Seperate validation dataset into " + str(num_parall) + " parts for parallel calculation.")


        # 2. Run GAP on each part
        os.system(f"parallel --delay 0.2 --joblog task.log --progress --resume -j {num_parall} < cmd.lst")


        # 3. Collect data after finished
        os.system(f"cat {' '.join(subdataset_gap_names)} > {dataset_file_output}")
        os.system("mv task.log task_backup.log")

        output_dataset = ase.io.read(dataset_file_output, ':')
        dft_final, search_index = data.find_ref_by_structure(output_dataset, dft_incopy, "_", "_")
        
        # bolt_f11_final = [bolt_f11[i] for i in search_index]
        # write the search index into a file
        with open("search_index.txt", 'w') as search_index_file:
            search_index_file.write(','.join([str(i) for i in search_index]))

        ase.io.write("dft_final.xyz", dft_final)


        if len(output_dataset) == len(dataset):
            os.system(f"rm -r {tag}_*")
        else:
            print(f"WARNING: len(ref_dataset):{len(dataset)} != len(gap_dataset):{len(output_dataset)}")
        
        return output_dataset, dft_final

    def run_calculator_seperately(self, dft_incopy:Atoms):
        """
        Seperate the dataset is aims to save most of caculation rather than ruin all by a few failed DFTB+ calculation.
        
        """

        dft_subincopy_final = []
        dftb_bandstructure_final = []
        kept_indices = []
        eos = self.args_input.eos_points
        for n in np.arange(int(len(dft_incopy) / eos)):
            dft_sub = dft_incopy[n*eos:(n+1)*eos]
            bolt_subf = self.bolt_f11[n*eos:(n+1)*eos]
            try:
                dftb_bandstructure = run_calculator(dft_sub,
                                                    poly_rep='Yes',
                                                    SCC='Yes',
                                                    kpts_dens=self.args_input.kpt_density,
                                                    bolt_f=bolt_subf)
                # poly_rep = 'Yes': switch polynomial repulsion (default 0.0 for hotcent skf generator)
                # SCC = 'Yes': switch the self charge consistent calculation(DFTB2)

                dft_subincopy_final.extend(dft_sub)
                dftb_bandstructure_final.extend(dftb_bandstructure)
                kept_indices.extend(range(n*eos, (n+1)*eos))
            except FileNotFoundError as err:
                print("DFTB+ Warning:", err)
                print("The problem caused by failed generation of SKF with this parameter. We will overlook these failed structures in this round of parameterization.")

        ase.io.write('dft_final.xyz', dft_subincopy_final)
        ase.io.write('dftb_band.xyz', dftb_bandstructure_final)
        with open("search_index.txt", 'w') as search_index_file:
            search_index_file.write(','.join(str(i) for i in kept_indices))

        return dftb_bandstructure_final, dft_subincopy_final
    
    def set_parameters(self, conf_parameters:list):
        """
        Set parameters as the input for Hotcent and Physrep
        """

        kxc = 0.0
        r0, r0_dens, p = conf_parameters

        atomic_numbers, atomic_numbers_exclusive, atomic_symbols, atomic_symbols_exclusive = self.get_elements_from_dataset(self.dft_input, self.opt_elem)
        
        if self.args_input.multi_element is None:
            #! For Single element system
            confinement_r0 = {s: [r0, r0_dens] for s in atomic_symbols}
            confinement_sigma = {s: p for s in atomic_symbols}

        else:
            #! For Binary system
            # For multi-element system, we need to define the cons and sigs and c_rep for each element
            confinement_r0 = {s: [get_PTBP()[s]['r_wave'], get_PTBP()[s]['r_dens']] for s in atomic_symbols_exclusive}
            confinement_r0[chemical_symbols[self.opt_elem[0]]] = [r0, r0_dens]
            confinement_sigma = {s: p for s in atomic_symbols}
        
        return confinement_r0, confinement_sigma


    def collectAndcalculate_muEF(self, dft_incopy:list, dftb_full:list, sigma_rep:float, r0_w:float, r0_d:float):

        # Compute DFTB+ atomic reference energies for every unique element in
        # the dataset (cached across the inner brute scan within this BO iter).
        E0_dftb = self.get_atomic_energy(dft_incopy)

        _, dftb_energy_per_atom_relative = self.get_formation_energy(dftb_full, E0_dftb, '_')
        _, dft_energy_per_atom_relative = self.get_formation_energy(dft_incopy, self.args_input.E0s, '_')

        diffs = np.abs(np.array(dftb_energy_per_atom_relative) - np.array(dft_energy_per_atom_relative))
        delta_energy_per_atom = np.sum(diffs)

        mu_E = delta_energy_per_atom

        dft_forces = self.get_forces(dft_incopy, '_')
        dftb_forces = self.get_forces(dftb_full, '_')

        # RMSE & MAE for forces
        forces_rmse = np.sqrt(np.mean(np.square(np.array(dft_forces) - np.array(dftb_forces))))
        forces_mae = np.mean(np.abs(np.array(dft_forces) - np.array(dftb_forces)))

        mu_F = forces_mae
        # Penalty for overlarge parameters (regularization)
        penalty = ((r0_w / (2*self.cov_radius) - 1)**2 + 1)**2 * ((r0_d / (4*self.cov_radius) - 1)**2 + 2)**2

        c = 0.1

        mu = (mu_E * (1-c) + mu_F * c) * penalty

        with  open(self.args_input.log_file, 'a') as output:
            output.write(f"{r0_w:15.8f}, {r0_d:15.8f}, {sigma_rep[0]:15.8f}, {0.0:20.8e}, {self.cov_radius:15.8f}, {mu_E:15.8f}, {mu_F:20.8f}, {0.0: 20.8f}, {0.0:15.5f}, {mu:25.11f}\n")

        # Reaction mode: report per-formula |E_dft - E_dftb| breakdown alongside
        # the aggregate loss so the user can spot which compositions are well-fit
        # vs which are dragging the score.
        if self.args_input.optimization_option.lower() == 'reaction':
            from collections import defaultdict as _dd
            buckets = _dd(list)
            for atoms, d in zip(dft_incopy, diffs):
                buckets[atoms.get_chemical_formula()].append(float(d))
            with open('reaction_per_formula.csv', 'a') as fh:
                fh.write(f"# r0_w={r0_w:.6f} r0_d={r0_d:.6f} sigma_rep={sigma_rep[0]:.6f} mu={mu:.6e}\n")
                fh.write("formula,n_structures,mean_abs_dE_per_atom,max_abs_dE_per_atom\n")
                for formula in sorted(buckets):
                    vals = np.array(buckets[formula])
                    fh.write(f"{formula},{len(vals)},{vals.mean():.6e},{vals.max():.6e}\n")

        return mu



    # ---------------------------------------------------------------
    # EOS helpers (ported from tools/hotpy.py + utils/calculator.py)
    # ---------------------------------------------------------------
    def mk_traj(self, atoms: Atoms, traj_label: str, default_eos_points: int) -> None:
        """Write one .traj per phase (atoms are flat-concatenated, eos_points each)."""
        num_phases = int(len(atoms) / default_eos_points)
        for n in range(num_phases):
            traj = Trajectory('{}_{}.traj'.format(traj_label, n), 'w')
            sub_atoms = atoms[n * default_eos_points: (n + 1) * default_eos_points]
            for a in sub_atoms:
                traj.write(a)

    def fit(self, na: str):
        """Fit a Birch-Murnaghan EOS to the configurations in `<na>.traj`."""
        V, E = [], []
        for atoms in ase.io.read('{}.traj@:'.format(na)):
            V.append(atoms.get_volume() / len(atoms))
            E.append(atoms.get_potential_energy() / len(atoms))
        eos = EOS(V, E, 'birchmurnaghan')
        eos.fit(warn=False)
        e0, B, Bp, v0 = eos.eos_parameters
        return e0, v0, B, Bp

    def eos_atoms(self, atoms_input, default_eos_points: int, calc_label: str):
        """Run EOS fit on each phase, return dict of {label_n: {energy,volume,B,Bp}}."""
        if isinstance(atoms_input, str):
            atoms_list = ase.io.read(atoms_input, ':')
        else:
            atoms_list = atoms_input

        num_phases = int(len(atoms_list) / default_eos_points)
        atomic_nums = []
        for a in atoms_list:
            atomic_nums.extend(a.numbers)
        atomic_nums = np.unique(atomic_nums)
        atomic_syms = [chemical_symbols[i] for i in atomic_nums]
        traj_label = f"eos_{'_'.join(atomic_syms)}"

        self.mk_traj(atoms_list, traj_label, default_eos_points)

        out = {}
        for n in range(num_phases):
            na = traj_label + '_' + str(n)
            try:
                e0, v0, B, Bp = self.fit(na)
            except RuntimeError:
                e0, v0, B, Bp = 1000, 1000, 1000, 1000
            out[na] = {f'{calc_label}_energy': e0,
                       f'{calc_label}_volume': v0,
                       f'{calc_label}_B': B,
                       f'{calc_label}_Bp': Bp}
            print(f"\n{na}\n{out[na]}")
        Path(f'{calc_label}.json').write_text(json.dumps(out))
        return out

    def eos_cal(self, dataset, bolt_factor: list):
        """Compute EOS-based loss components: (delta_tot, mu_v_tot, mu_logb_tot).

        delta_tot = Σ_phase  Boltzmann(phase) * (Δv * Δlog10B),
        with Δv  = |v0_dftb / v0_dft - 1|,
             ΔlogB = |log10(b_dftb) / log10(b_dft) - 1| + 1.
        Phases with bolt_factor < 0.1 are skipped.
        """
        for name in ['volume', 'B', 'Bp']:
            with open(name + '.csv', 'w') as f:
                print('#symbol, dft, dftb', file=f)
                for symbol, dct in dataset.items():
                    values = [dct[code + '_' + name] for code in ['dft', 'dftb']]
                    if name == 'B':
                        values = [val * 1e24 / kJ for val in values]
                    print('{},'.format(symbol),
                          ', '.join('{:.2f}'.format(v) for v in values), file=f)

        delta_tot, mu_v_tot, b_tot, n_i = 0.0, 0.0, 0.0, 0
        bolt_left = []
        with open('delta.csv', 'w') as f:
            print('#symbol,delta(dft-dftb),bolt_factor,delta*bolt', file=f)
            for symbol, dct in dataset.items():
                v0_dftb = dct['dftb_volume']
                b_dftb = dct['dftb_B'] * 1e24 / kJ
                v0_dft = dct['dft_volume']
                b_dft = dct['dft_B'] * 1e24 / kJ

                mu_v = abs(v0_dftb / v0_dft - 1)
                if math.isnan(mu_v):
                    mu_v = 100

                try:
                    mu_logb = abs(np.log10(b_dftb) / np.log10(b_dft) - 1) + 1
                except ZeroDivisionError:
                    mu_logb = 100
                if math.isnan(mu_logb):
                    mu_logb = 100

                delta_eos = mu_v * mu_logb
                delta_eos_bolt = delta_eos * bolt_factor[n_i]

                if bolt_factor[n_i] < 0.1:
                    n_i += 1
                    continue
                mu_v_tot += mu_v * bolt_factor[n_i]
                b_tot += mu_logb * bolt_factor[n_i]
                delta_tot += delta_eos_bolt
                bolt_left.append(bolt_factor[n_i])
                print('{}, {:5.2f}, {:5.2f}, {:5.2f}'.format(
                    symbol, delta_eos, bolt_factor[n_i], delta_eos_bolt), file=f)
                n_i += 1

        return delta_tot, mu_v_tot, b_tot

    # ---------------------------------------------------------------

    def collectAndcalculate_muEV(self, dft_incopy:Atoms, sigma_rep, r0_w, r0_d, bolt_f) -> float:

        """
        Collected energy forces and calculate mu_E and mu_V

        Parameters
        ------
        dft_incopy: Atoms
            Atoms object of the whole dataset
        
        sigma_rep: list
            Parameter for repulsive potential
            eg: sigma_rep = array([0.4321])
        
        r0_w: float
            Parameter for wavefunction cutoff

        r0_d: float
            Parameter for density function cutoff
        
        """

        # Read dataset
        num_parall = int(len(dft_incopy) / self.args_input.eos_points)

        dftb_full, mu_Es, mu_Vs, mu_logbs = [], [], [], []
        for num in range(num_parall):
            dft_sub = dft_incopy[num*self.args_input.eos_points:(num+1)*self.args_input.eos_points]
            dftb_sub = ase.io.read(f'dftb_sub_{num}.xyz', ':')
            dftb_full.extend(dftb_sub)

            eos_dftb = self.eos_atoms(dftb_sub, self.args_input.eos_points, 'dftb')
            eos_dft  = self.eos_atoms(dft_sub, self.args_input.eos_points, 'dft')
            print(f"\neos_dftb: {eos_dftb}\neos_dft: {eos_dft}")


            # Calculate the mu_V
            [eos_dft[conf].update(eos_dftb[conf]) for conf in eos_dftb] # input eos of dft
            mu_cal, mu_V, mu_logb = self.eos_cal(eos_dft, [bolt_f[num]]) 
            mu_Vs.append(mu_V); mu_logbs.append(mu_logb)
            
            mu_Es.append(dftb_sub[0].info['mu_E'] * bolt_f[num]) 

        bolt_left = np.sum([bolt_f[i] for i in range(len(bolt_f)) if bolt_f[i] >= 0.1])
        mu_E, mu_V, mu_logb = np.sum(mu_Es)/bolt_left, np.sum(mu_Vs)/bolt_left, np.sum(mu_logb)/bolt_left

        ase.io.write('dftb_full.xyz', dftb_full)
        os.system("rm dftb_sub_*.xyz")

        # Penalty term(regularization)
        penalty = ((r0_w / (2*self.cov_radius) - 1)**2 + 1)**2 * ((r0_d / (4*self.cov_radius) - 1)**2 + 2)**2


        # Combine mu_E and mu_V
        c = 0.5
        mu = (c * mu_V + (1 - c) * mu_E) * penalty 


        # Log the results
        sigma_rep, kxc = sigma_rep[0], 0.0
        with open(self.args_input.log_file, 'a') as output:
            output.write(f'{r0_w:15.8f}, {r0_d:15.8f}, {sigma_rep:15.8f}, {kxc:20.8e}, {self.cov_radius:15.8f}, {mu_E:15.8f}, {mu_V:20.8f}, {mu_logb: 20.8f}, {penalty:15.5f}, {mu:25.11f}\n')


        return mu

    def generate_repulsion(self, input_parameters) -> list:

        """
        Generate repulsive potential for each structure, for parallelization

        Parameters
        ------
        input_parameters: list
            List of parameters for repulsive potential
            eg: input_parameters = [n, dftb_bandstructure, dft_sub, atomic_subnumbers, para_rep_tot, kxc_tot, write_rep]
        
        """

        # Log
        n, dftb_bandstructure, dft_sub, atomic_subnumbers_tot, para_rep_tot, kxc_tot, write_rep = input_parameters
        print(f"\n> Subparallel Physrep Info: {n}\n  atomic_subnumbers_tot:{atomic_subnumbers_tot}\n  para_rep_tot:{para_rep_tot}\n  kxc_tot:{kxc_tot}\n len(dftb_band):{len(dftb_bandstructure)}\n len(dft_sub):{len(dft_sub)}\n write_rep:{write_rep}")


        pr = physrep(elements=atomic_subnumbers_tot,
                    rmin=0.5,
                    rmax=20.0,
                    npoints=700,
                    scale_cov=para_rep_tot,
                    k_xc=kxc_tot,
                    )
        
        if write_rep:
            if self.args_input.multi_element is not None:
                pr.write_specific_skf(self.opt_elem, 'ptbp')
                print("Write specific SKF for Binarys")
            else:
                pr.write_skf('ptbp')
                print("Write SKF for Elementary")

        
        atoms_total = data.get_full_dftb(dftb_bandstructure, dft_sub, pr, True)


        # Incase some structures faild in band energy calculation part
        with open ("search_index.txt", 'r') as search_index_file:
            search_index = search_index_file.read().strip().split(',')
            bolt_f11_final = [self.bolt_f11[int(i)] for i in search_index]


        dftb_energy_per_atom_relative, dft_energy_per_atom_relative, delta_energy_per_atom = self.get_energy_per_atom(dft_sub, atoms_total, bolt_f11_final[n*self.args_input.eos_points:(n+1)*self.args_input.eos_points])
        for atoms in atoms_total:
            atoms.info['mu_E'] = delta_energy_per_atom
    

        ase.io.write(f'dftb_sub_{n}.xyz', atoms_total)
        print(f"Write dftb_sub_{n}.xyz")


        return atoms_total, delta_energy_per_atom



    def parallel_repulsion(self, sigma_rep:list, r0_w:float, r0_d:float, write_rep:bool=False):
        """
        Calculator of repulsion potential, which is parallelized respect to number of structures

        Parameters
        ------
        sigma_rep: list
            Parameter for repulsive potential
            eg: sigma_rep = array([0.4321])
        
        r0_w: float
            Parameter for wavefunction cutoff
        
        r0_d: float
            Parameter for density function cutoff
        
        write_rep: bool
            Whether write the repulsive potential to SKF file

        """

        sigma_rep = sigma_rep.tolist()
        pool = mp.Pool()

        # Prepare data
        dftb_bandstructure_final = ase.io.read('dftb_band.xyz', ':')
        dft_final = ase.io.read('dft_final.xyz', ':')
        dft_incopy = copy.deepcopy(dft_final)


        num_parall = int(len(dft_final) / self.args_input.eos_points)
        for num in range(num_parall):
            dft_sub = copy.deepcopy(dft_incopy[num*self.args_input.eos_points:(num+1)*self.args_input.eos_points])
            dftb_bandstructure = dftb_bandstructure_final[num*self.args_input.eos_points:(num+1)*self.args_input.eos_points]
            atomic_subnumbers, atomic_subnumbers_exclusive , _, _ = self.get_elements_from_dataset(dft_sub, self.opt_elem)


            if self.args_input.multi_element is None:
                para_rep_tot = sigma_rep
            else:
                para_rep_tot = [get_PTBP()[chemical_symbols[n]]['sigma_rep'] for n in atomic_subnumbers_exclusive] + sigma_rep

            kxc_tot = [0.0] * len(atomic_subnumbers)

            
            # Prepare input for parallel
            input_parameters = [(num, dftb_bandstructure, dft_sub, atomic_subnumbers, para_rep_tot, kxc_tot, write_rep)]
            outputs = list(pool.imap(self.generate_repulsion, input_parameters))

        pool.close()
        pool.join()

        # Judge which loss function to use

        bolt_f = self.bolt_f11[floor(self.args_input.eos_points/2)::self.args_input.eos_points]
        mu = self.collectAndcalculate_muEV(dft_incopy, sigma_rep, r0_w, r0_d, bolt_f)

        return mu



    def unparallel_repulsion(self, sigma_rep:list, r0_w:float, r0_d:float, write_rep:bool=True):

        """
        Calculator of repulsion potential

        Parameters
        ------
        sigma_rep: list
            Parameter for repulsive potential
            eg: sigma_rep = array([0.4321])
        
        r0_w: float
            Parameter for wavefunction cutoff
        
        r0_d: float
            Parameter for density function cutoff
        
        write_rep: bool
            Whether write the repulsive potential to SKF file

        """
        
        # Read dataset from band energy calculation
        dftb_bandstructure_final = ase.io.read('dftb_band.xyz', ':')
        dft_final = ase.io.read('dft_final.xyz', ':')
        dft_incopy = copy.deepcopy(dft_final)


        atomic_numbers_tot, atomic_subnumbers_exclusive , _, _ = self.get_elements_from_dataset(dft_incopy, self.opt_elem)
        # The outer brute / L-BFGS-B in Repulsion_optimization gives a single
        # scalar sigma_rep. For datasets that span multiple elements (i.e.
        # dataset / reaction mode), broadcast that scalar to all elements so
        # physrep can build one repulsive spline per element.
        scale_cov_per_elem = [float(np.atleast_1d(sigma_rep)[0])] * len(atomic_numbers_tot)
        kxc_tot = [0.0] * len(atomic_numbers_tot)


        # Repuslive potential Calculator
        pr = physrep(elements=atomic_numbers_tot,
                    rmin=0.5,
                    rmax=20.0,
                    npoints=700,
                    scale_cov=scale_cov_per_elem,
                    k_xc=kxc_tot,
                    )
        

        # implement repulsive potential to dftb_bandstructure part 
        dftb_full = data.get_full_dftb(dftb_bandstructure_final, dft_incopy, pr, True)

        ase.io.write('dftb_full.xyz', dftb_full)

        if write_rep:
            if self.args_input.multi_element is not None:
                pr.write_specific_skf(self.opt_elem, 'ptbp')
                print("Write specific SKF for Binarys")
            else:
                pr.write_skf('ptbp')
                print("Write SKF for Elementary")

        mu = self.collectAndcalculate_muEF(dft_incopy, dftb_full, sigma_rep, r0_w, r0_d)

        return mu  
    



    def Repulsion_optimization(self, conf_parameters:list, c0:float, c1:float) -> list:
        """
        Ready for the optimization of repulsion parameters sigma_rep
        
        Parameters
        -------
        conf_parameters: list
            List of parameters to be optimized
            eg: conf_parameters = [r0_w, r0_d, p]
            V_conf = (r_cov/r0)^p

        c0: float
            Lower boundary for sigma
        
        c1: float
            Upper boundary for sigma

        Note that please set a bigger upper boundary for Hydrogen eg. c1 = 1.5

        """
        # Set search boundary
        r0_w, r0_d, p = conf_parameters
        srange = (c0, c1)
        bounds = (Bounds(srange[0], srange[1], keep_feasible=True))

        
        # # Search time 1: Gride search
        conf_args = (r0_w, r0_d, True)
        if self.args_input.optimization_option.lower() in ('dataset', 'reaction'):
            res_brute = brute(self.unparallel_repulsion, (srange,), Ns=11, full_output=True,  disp=True, workers=1, args=conf_args)

        elif self.args_input.optimization_option.lower() == 'energygeometry':
            res_brute = brute(self.parallel_repulsion, (srange,), Ns=11, full_output=True,  disp=True, workers=1, args=conf_args)

        print(f"Search time 1: sigma_rep = {res_brute[0]}, mu = {res_brute[1]}")


        # Search time 2: L-BFGS-B
        # scipy >=1.11 makes brute return res_brute[0] as a 0-d ndarray; wrap to a
        # plain 1-d list so minimize's x0-shape check accepts it.
        x0_bfgs = [float(np.atleast_1d(res_brute[0])[0])]
        if self.args_input.optimization_option.lower() in ('dataset', 'reaction'):
            res_bfgs = minimize(self.unparallel_repulsion, x0_bfgs, method='L-BFGS-B',bounds=bounds, options={'disp':True,'maxfun':20,'ftol':1e-3}, args=conf_args)

        elif self.args_input.optimization_option.lower() == 'energygeometry':
            res_bfgs = minimize(self.parallel_repulsion, x0_bfgs, method='L-BFGS-B',bounds=bounds, options={'disp':True,'maxfun':20,'ftol':1e-3}, args=conf_args)


        sigma_rep = res_bfgs.x[0]
        mu = res_bfgs.fun
        print(f"Search time 2: sigma_rep = {sigma_rep}, mu = {mu}")


        return sigma_rep, mu






    def EnergyGeometry(self, conf_parameters:list,
                       xc:str,
                       superposition:str,
                    ) -> float:
        """
        Calculate the loss function for Energy and Geometry

        Parameters:
        -------
        conf_parameters: list
            List of parameters to be optimized
            eg: conf_parameters = [r0_w, r0_d, p]
            V_rep = (r_cov/r0)^p
        
        xc: str
            Exchange and Correlation functional
            eg: xc = 'GGA_X_PBE+GGA_C_PBE'

        
        superposition: str
            Density or Potential superposition
            default: 'Density' for PTBP, 'Potential' for QN(QUASINANO2013)+rep
        """

        kxc = 0.0
        r0_w, r0_d, p = conf_parameters

        # SKF parameters changed → invalidate the per-iter E0_dftb cache so
        # the inner brute scan recomputes against the new band structure.
        self._E0_dftb_cache = {}

        opt_directory = self.path / self.args_input.results_dir / f"opt_{r0_w:.3f}_{r0_d:.3f}"
        opt_directory.mkdir(exist_ok=True, parents = True)

        os.chdir(opt_directory)


        #1. Generate SKF 
        t_ini = perf_counter()
        confinement_r0, confinement_p = self.set_parameters(conf_parameters)

        print(f"Confinement_r0:{confinement_r0}, Confinement_p:{confinement_p}")

        self.hp.set_variables(confinement_r0,
                        confinement_p, 
                        xc=xc,
                        superposition=self.args_input.superposition,
                        )

        try:
            # Write  SKF file
            self.hp.full_hotcent(superposition=superposition,
                            opt_elem=self.opt_elem,
                            )        

            t_hotcent = perf_counter()

        
        #2. DFTB Calculation for Band Structure Energy term
            dft_incopy = copy.deepcopy(self.dft_input)

            # Calculate energy/forces for the band energy part  
            # dftb_bandstructure_final, dft_subincopy_final = self.run_calculator_seperately_parallel(dft_incopy, self.args_input.kpt_density, 'calc_dftb.py', 'static')
            self.run_calculator_seperately(dft_incopy)

            # dft_subincopy_final = ase.io.read('dft_final.xyz', ':'); dftb_bandstructure_final = ase.io.read('dftb_band.xyz', ':')
            t_dftb = perf_counter()


        #3. Parameterization for the Repulsion term, which is parallelized respect to
            sigma_rep_best, mu = self.Repulsion_optimization(conf_parameters, 0.1, 0.8)
            # Downstream print/plot/par.out blocks read the bare `sigma_rep`
            # name; alias it to the optimised scalar so they don't NameError.
            sigma_rep = float(np.atleast_1d(sigma_rep_best)[0])


            # Write REP into SKF file
            # Calculate repulsive potential based on the optimized parameter
            # tot_para_rep = para_rep + [sigma_rep]
            atomic_numbers, atomic_numbers_exclusive, atomic_symbols, atomic_symbols_exclusive = \
                data.get_elements_from_dataset(self.dft_input, self.opt_elem)
            if self.args_input.multi_element is None:
                # Elementary / dataset / reaction modes: broadcast the brute-
                # optimised scalar sigma_rep to every element in the dataset
                # so physrep can build one repulsive spline per element.
                tot_para_rep = [float(np.atleast_1d(sigma_rep_best)[0])] * len(atomic_numbers)
            else:
                tot_para_rep = [get_PTBP()[chemical_symbols[n]]['sigma_rep']
                                for n in atomic_numbers_exclusive] + [float(np.atleast_1d(sigma_rep_best)[0])]

            pr = physrep(atomic_numbers,
                        rmin=0.5,
                        rmax=20.0,
                        npoints=700,
                        scale_cov=tot_para_rep,
                        k_xc=[0.0] * len(atomic_numbers),
                        )

            if self.args_input.multi_element is not None:
                pr.write_specific_skf(self.opt_elem, 'ptbp')
            else:
                pr.write_skf('ptbp')


            dft_subincopy_final = ase.io.read('dft_final.xyz', ':'); dftb_bandstructure_final = ase.io.read('dftb_band.xyz', ':')
            dftb_full = data.get_full_dftb(dftb_bandstructure_final, dft_subincopy_final, pr, True)
            ase.io.write('dftb_full.xyz', dftb_full)
            
            
            t_physrep = perf_counter()

    # ============================================================================ #
    #  4. Data processing and output

            # Calculate the energy per atom and plot
            with open ("search_index.txt", 'r') as search_index_file:
                search_index = search_index_file.read().strip().split(',')
            bolt_f11_final = [self.bolt_f11[int(i)] for i in search_index]

            if self.args_input.optimization_option.lower() == 'energygeometry':
                dftb_energy_per_atom_relative, dft_energy_per_atom_relative, delta_energy_per_atom = self.get_energy_per_atom(dft_subincopy_final, dftb_full, bolt_f11_final)

            elif self.args_input.optimization_option.lower() in ('dataset', 'reaction'):
                E0_dftb = self.get_atomic_energy(dft_incopy)
                _, dftb_energy_per_atom_relative = self.get_formation_energy(dftb_full, E0_dftb, '_')
                _, dft_energy_per_atom_relative = self.get_formation_energy(dft_incopy, self.args_input.E0s, '_')



            # Plot
            if self.args_input.optimization_option.lower() == 'energygeometry' and self.args_input.multi_element is None:
                
                bolt_f = self.bolt_f11[floor(self.args_input.eos_points/2)::self.args_input.eos_points]

                plot.plot_vs(atomic_symbols[0], [r0_w, r0_d, sigma_rep, mu], dft_incopy, dft_energy_per_atom_relative, dftb_energy_per_atom_relative, bolt_f)

                plot.plot_tradition(atomic_symbols[0], [r0_w, r0_d, sigma_rep, 2.0],  mu, dft_energy_per_atom_relative, dftb_energy_per_atom_relative, bolt_f)
            

            penalty = ((r0_w / (2*self.cov_radius) - 1)**2 + 1)**2 * ((r0_d / (4*self.cov_radius) - 1)**2 + 2)**2

            with open(self.args_input.para_file, 'a') as para:
                para.write(f'{r0_w:15.8f}, {r0_d:15.8f}, {sigma_rep:15.8f}, {kxc:20.8e}, {self.cov_radius:15.7f}, {mu:25.11f}\n')


            t_end = perf_counter()

            # os.chdir(p / DEFAULT_OUT_DIR)        
            print(f"""\nTime:
                \nHotcent: {t_hotcent - t_ini} 
                \nDFTB: {t_dftb - t_hotcent}
                \nPhysrep: {t_physrep - t_dftb}
                \nDataProcessing: {t_end - t_physrep}
                \nTotal: {t_end - t_ini}""")              
            return mu
            

        # Except Errors caused by unphysical parameters
        except(RuntimeError, AssertionError, UnboundLocalError, ase.io.formats.UnknownFileTypeError) as error:
            print("Error:", error)
            sigma_rep, kxc = [0.4321, 0.0]
            mu_E, mu_V, mu_logb, penalty = [10, 10, 10, 10]
            mu = random.randint(97,99)

            with open(self.args_input.para_file, 'a') as para:
                para.write(f'{r0_w:15.8f}, {r0_d:15.8f}, {sigma_rep:15.8f}, {kxc:20.8e}, {self.cov_radius:15.7f}, {mu:25.11f}\n')
            
            with open(self.args_input.log_file, 'a') as output:
                output.write(f'{r0_w:15.8f}, {r0_d:15.8f}, {sigma_rep:15.8f}, {kxc:20.8e}, {self.cov_radius:15.8f}, {mu_E:15.8f}, {mu_V:20.8f}, {mu_logb: 20.8f}, {penalty:15.5f}, {mu:25.11f}\n')
            return mu
        

    def BandStructure(self, conf_parameters:list,
                        xc:str,
                        superposition:str,
                        ) -> float:
        
        """
        Calculate the loss function for Band Structure

        Parameters:
        -------
        conf_parameters: list
            List of parameters to be optimized
            eg: conf_parameters = [r0_w, r0_d, p]
        In band structure optimization, r0_d == r0_w

        xc: str
            Exchange and Correlation functional
            eg: xc = 'GGA_X_PBE+GGA_C_PBE'

        
        superposition: str
            density or potential

        """

        # Data preparation
        r0_w, r0_d, p = conf_parameters
        opt_directory = self.path / self.args_input.results_dir / f"opt_{r0_w:.3f}_{p:.3f}"
        opt_directory.mkdir(exist_ok=True, parents = True)
        os.chdir(opt_directory)


        # Generate SKF
        t_ini = perf_counter()
        confinement_r0, confinement_p = self.set_parameters(conf_parameters)
        print(f"Confinement_r0:{confinement_r0}, Confinement_p:{confinement_p}")

        self.hp.set_variables(confinement_r0,
                            confinement_p, 
                            xc=xc,
                            superposition=superposition,
                            )
        try:
            # Write  SKF file
            self.hp.full_hotcent(superposition=superposition,
                            opt_elem=self.opt_elem,
                            )        

            t_hotcent = perf_counter()

    #       DFTB BAND STRUCTURE CALCULATION
            dft_incopy = copy.deepcopy(self.dft_input)

            # Weight of each structure
            bolt_f = [1.0]

            # The dftb_bandstructure is refer to DFTB1 not real band structure
            calc_band_binary(conf_parameters, dft_incopy, self.args_input.kpt_density, bolt_f)

    #       DELAT BAND STRUCTURE CALCULATION
            dft_incopy = copy.deepcopy(self.dft_input)
            mu_homo, mu_lumo = delta_band_binary(conf_parameters, self.args_input.dft_band_file, dft_incopy, bolt_f)

            mu = mu_homo + mu_lumo

            with open(self.args_input.para_file, 'a') as para:
                para.write(f"{r0_w:15.8f}, {r0_d:15.8f}, {p:15.8f}, {self.cov_radius:15.7}, {mu_homo:20.11f}, {mu_lumo:20.11f}, {mu:25.11f}\n")

            return mu
        

        except(RuntimeError, AssertionError, UnboundLocalError, ase.io.formats.UnknownFileTypeError) as error:
            print(f"Error in the bandstruture calculation:{error}")
            mu_homo, mu_lumo, mu = 10, 10, 20
            with open(self.args_input.para_file, 'a') as para:
                para.write(f"{r0_w:15.8f}, {r0_d:15.8f}, {p:15.8f}, {self.cov_radius:15.7}, {mu_homo:20.11f}, {mu_lumo:20.11f}, {mu:25.11f}\n")
            return mu
        

    def Loss_EnergyGeometry(self, parameters:list) -> float:
        """
        Calculate Loss function based on Energy and Geometry

        Parameters:
        -------
        parameters : list
            List of parameters to be optimized
            eg: parameters = [r0_w, r0_d]
        """

        superposition = self.args_input.superposition
        xc = self.args_input.xc


        default_p = 2.0
        r0_w, r0_d = parameters
        conf_parameters = [r0_w, r0_d, default_p]

        mu = self.EnergyGeometry(conf_parameters,
                            xc=xc,
                            superposition=superposition,
                            )

        return mu


    def Loss_BandStructure(self, parameters:list) -> float:
        """
        Calculate Loss function based on Band Structure

        Parameters:
        -------
        parameters : list
            List of parameters to be optimized
            eg: conf_parameters = [r0, p]
        """
        
        superposition = self.args_input.superposition
        xc = self.args_input.xc

        r0, p = parameters

        # In band structure optimization, r0_d == r0_w
        conf_parameters = [r0, r0, p]

        mu = self.BandStructure(conf_parameters,
                            xc=xc,
                            superposition=superposition,
                            )

        return mu



    # =========================================================
    # Outer-loop optimisers
    # =========================================================
    def optimize(self, cov_radius, checkpoint_saver, n_calls):
        """Dispatch to the optimiser selected by --optimizer.

        Returns an object with at minimum `.x` (best point as list) and
        `.fun` (best loss). The skopt path also exposes `.x_iters` /
        `.func_vals` for plotting; the PSO path provides equivalents.
        """
        opt = getattr(self.args_input, 'optimizer', 'bo')
        if opt == 'bo':
            return self.Bayesian_optimization(cov_radius, checkpoint_saver, n_calls)
        if opt == 'parallel_bo':
            return self._optimize_parallel_bo(cov_radius, checkpoint_saver, n_calls)
        if opt == 'pso':
            return self._optimize_pso(cov_radius, n_calls)
        raise ValueError(f"Unknown optimizer: {opt!r}")

    def _bo_dimensions_and_cost(self, cov_radius):
        """Shared helper: returns (dimensions, bounds_lo, bounds_hi, cost_fn).

        Both BO and PSO use the same parameter ranges; the data shape is the
        only thing that differs (skopt wants Real() dimensions, pyswarms
        wants raw lower/upper arrays).
        """
        opt_opt = self.args_input.optimization_option.lower()
        if opt_opt in ('energygeometry', 'dataset', 'reaction'):
            r0_w_lower, r0_w_upper = 2.1, 7.0 * cov_radius
            r0_d_lower, r0_d_upper = 3.4, 9.0 * cov_radius
            dims = [Real(name='r0_w', low=r0_w_lower, high=r0_w_upper, prior='uniform'),
                    Real(name='r0_d', low=r0_d_lower, high=r0_d_upper, prior='uniform')]
            lo = [r0_w_lower, r0_d_lower]
            hi = [r0_w_upper, r0_d_upper]
            cost_fn = self.Loss_EnergyGeometry
        else:  # bandstructure
            r0_lower, r0_upper = 2.1, 7.0 * cov_radius
            p_lower, p_upper = 1.01, 11.0
            dims = [Real(name='r0', low=r0_lower, high=r0_upper, prior='uniform'),
                    Real(name='p', low=p_lower, high=p_upper, prior='uniform')]
            lo = [r0_lower, p_lower]
            hi = [r0_upper, p_upper]
            cost_fn = self.Loss_BandStructure
        return dims, lo, hi, cost_fn

    def _optimize_pso(self, cov_radius, n_calls):
        """Particle Swarm Optimization via pyswarms.

        Population × iterations gives the total evaluation budget. We aim
        for `n_particles * iters ≈ n_calls`, with `n_particles` from
        --n_particles (default 4) and `iters` derived (≥2).
        """
        import pyswarms as ps

        n_particles = max(2, int(getattr(self.args_input, 'n_particles', 4)))
        iters = max(2, n_calls // n_particles)
        total = n_particles * iters

        _dims, lo, hi, cost_fn = self._bo_dimensions_and_cost(cov_radius)
        bounds = (np.array(lo, dtype=float), np.array(hi, dtype=float))

        # Standard PSO hyperparameters; tunable later via CLI if needed.
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        def batch_cost(swarm):
            return np.array([float(cost_fn(p.tolist())) for p in swarm])

        print(f"[PSO] particles={n_particles}, iters={iters}, "
              f"total evaluations={total} (bounds lo={lo}, hi={hi})")
        opt = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=len(lo),
            options=options,
            bounds=bounds,
        )
        cost, pos = opt.optimize(batch_cost, iters=iters)

        # Wrap into a skopt-result-like object so cli/run.py downstream stays
        # the same. Flatten pos_history (per-iter swarm arrays) into a single
        # x_iters list.
        x_iters_flat = [list(map(float, p)) for swarm in opt.pos_history for p in swarm]
        # cost_history is best-so-far per iter; expand to per-evaluation by
        # repeating each iteration's snapshot n_particles times so plotting
        # over `func_vals` still makes sense.
        func_vals_flat = []
        for c in opt.cost_history:
            func_vals_flat.extend([float(c)] * n_particles)

        # SimpleNamespace is picklable (unlike a class defined inside a
        # method), so cli/run.py can dump the result to result.pkl for
        # `--postprocess_only` reuse.
        from types import SimpleNamespace
        return SimpleNamespace(
            x=list(map(float, pos)),
            fun=float(cost),
            x_iters=x_iters_flat,
            func_vals=func_vals_flat,
            optimizer='pso',
        )

    def _optimize_parallel_bo(self, cov_radius, checkpoint_saver, n_calls):
        """Batched BO via skopt.Optimizer.ask(n_points, strategy='cl_min')
        with TRUE parallel evaluation via a forked Process per particle.

        Each round:
          1. opt.ask(n_points=n_particles) proposes a batch using the
             constant-liar strategy.
          2. n_particles forked subprocesses each evaluate Loss_EnergyGeometry
             on one batch member concurrently. Forked workers inherit the
             Loss instance state (dft_input, hp, E0 cache, etc.) without
             needing to pickle bound methods.
          3. Results are collected via Queue and fed back via opt.tell.

        Notes:
        - We use `multiprocessing.Process` (not `Pool`) because Pool workers
          are daemonic and would forbid the nested mp.Pool that
          `hotpy.full_hotcent` and `parallel_repulsion` create internally.
        - par.out and log.out at the parent cwd are appended-to by every
          worker concurrently — line interleaving is possible but writes
          are atomic per-line on POSIX. The per-evaluation
          `results/opt_<r0_w>_<r0_d>/` directories are unique so no
          collision there.
        """
        from skopt import Optimizer
        import multiprocessing as mp

        n_particles = max(2, int(getattr(self.args_input, 'n_particles', 4)))
        rounds = max(1, n_calls // n_particles)

        dims, _lo, _hi, cost_fn = self._bo_dimensions_and_cost(cov_radius)
        opt = Optimizer(dimensions=dims, base_estimator='GP', acq_func='EI',
                        random_state=getattr(self.args_input, 'seed', 123))

        ctx = mp.get_context('fork')

        def _worker(x_arg, q, idx):
            # Closure over `cost_fn` works under fork because the child
            # inherits the parent's full Python state; no pickling involved.
            try:
                val = float(cost_fn(x_arg))
                q.put((idx, val, None))
            except Exception:
                import traceback as _tb
                q.put((idx, None, _tb.format_exc()))

        print(f"[parallel_BO] batch_width={n_particles} (forked workers), "
              f"rounds={rounds}, total evaluations={n_particles*rounds}")
        x_iters_flat, func_vals_flat = [], []
        best_x, best_fun = None, float('inf')
        for r in range(rounds):
            xs = opt.ask(n_points=n_particles, strategy='cl_min')
            queue = ctx.Queue()
            procs = [ctx.Process(target=_worker, args=(xs[i], queue, i))
                     for i in range(len(xs))]
            for p in procs:
                p.start()
            results = [None] * len(xs)
            for _ in xs:
                idx, val, err = queue.get()
                if err is not None:
                    for p in procs:
                        if p.is_alive():
                            p.terminate()
                    raise RuntimeError(
                        f"parallel_bo worker {idx} crashed:\n{err}")
                results[idx] = val
            for p in procs:
                p.join()
            opt.tell(xs, results)
            for x, y in zip(xs, results):
                x_iters_flat.append(list(map(float, x)))
                func_vals_flat.append(y)
                if y < best_fun:
                    best_fun, best_x = y, list(map(float, x))
            print(f"[parallel_BO] round {r+1}/{rounds}: best so far "
                  f"x={best_x}, fun={best_fun:.6e}")

        from types import SimpleNamespace
        return SimpleNamespace(
            x=best_x,
            fun=best_fun,
            x_iters=x_iters_flat,
            func_vals=func_vals_flat,
            optimizer='parallel_bo',
        )

    def Bayesian_optimization(self,
                              cov_radius,
                              checkpoint_saver,
                              n_calls):
        """
        Initialize the Bayesian optimization
        """

        # if self.args_input.checkpoint.exists():
        if os.path.exists(self.args_input.checkpoint):
            res = load(self.args_input.checkpoint)
            x0 = res.x_iters
            y0 = res.func_vals
            done = len(x0)
            # Note: when x0/y0 are passed, gp_minimize treats `n_calls` as the
            # number of *additional* fresh evaluations on top of the prior
            # ones. So `--n_calls 11` means "do 11 more BO steps", giving a
            # total stored history of `done + n_calls`.
            print(f"[checkpoint] Resuming from {self.args_input.checkpoint}: "
                  f"{done} prior iteration(s) loaded; will run {n_calls} more "
                  f"(total stored: {done + n_calls}).")
        else:
            # Smart seed: for an elementary system whose element already has a
            # PTBP entry, start the BO at those published params instead of a
            # generic cov_radius * (3, 5) heuristic. Saves several BO iterations
            # of cold-start exploration.
            x0 = None
            try:
                symbols = sorted({chemical_symbols[int(z)]
                                  for atoms in self.dft_input
                                  for z in atoms.numbers})
                if (len(symbols) == 1
                        and symbols[0] in get_PTBP()
                        and self.args_input.optimization_option.lower() in
                            ('energygeometry', 'dataset', 'reaction')):
                    p = get_PTBP()[symbols[0]]
                    x0 = [p['r0_w'], p['r0_d']]
                    print(f"BO seed from PTBP[{symbols[0]}]: "
                          f"r0_w={x0[0]}, r0_d={x0[1]}")
            except Exception as _e:
                print(f"PTBP-seed lookup failed ({_e}); falling back to heuristic")
            if x0 is None:
                x0 = [cov_radius * 3, cov_radius * 5]  # DEFAULT_START_PARA
                print(f"BO seed from heuristic: r0_w={x0[0]}, r0_d={x0[1]}")
            y0 = None
            n_calls = n_calls

        if self.args_input.optimization_option.lower() in ('energygeometry', 'dataset', 'reaction'):
            r0_w_lower, r0_w_upper = 2.1, 7.0 * cov_radius            # Wavefunction cutoff
            r0_d_lower, r0_d_upper = 3.4, 9.0 * cov_radius            # Density function cutoff
            print(f"Bounds: r0_w: {r0_w_lower} ~ {r0_w_upper}, r0_d: {r0_d_lower} ~ {r0_d_upper}")

            dr0_w   = Real(name='r0_w',  low=r0_w_lower,  high=r0_w_upper,   prior='uniform')
            dr0_d   = Real(name='r0_d',  low=r0_d_lower,  high=r0_d_upper,   prior='uniform')
            dimensions = [dr0_w, dr0_d]

            res = gp_minimize(func=self.Loss_EnergyGeometry,
                            dimensions=dimensions,
                            n_calls=n_calls,
                            acq_func="EI",
                            x0=x0,
                            y0=y0,
                            callback=[checkpoint_saver],
                            random_state=getattr(self.args_input, 'seed', 123))

        elif self.args_input.optimization_option.lower() == 'bandstructure':
            r0_lower, r0_upper = 2.1, 7.0 * cov_radius
            p_lower, p_upper = 1.01, 11.0

            # V_conf = (r0/r)^p

            print(f"Bounds: r0: {r0_lower} ~ {r0_upper}, p: {p_lower} ~ {p_upper}")
            print(f"Default r0: {x0[0]}, p: {x0[1]}")

            dr0   = Real(name='r0',  low=r0_lower,  high=r0_upper,   prior='uniform')
            dp   = Real(name='p',  low=p_lower,  high=p_upper,   prior='uniform')
            dimensions = [dr0, dp]

            res = gp_minimize(func=self.Loss_BandStructure,
                            dimensions=dimensions,
                            n_calls=n_calls,
                            acq_func="EI",
                            x0=x0,
                            y0=y0,
                            callback=[checkpoint_saver],
                            random_state=getattr(self.args_input, 'seed', 123))

        
        return res
