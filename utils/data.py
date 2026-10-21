# For dealing with data

from ase import Atoms
import numpy as np
import math
from tools import physrep
import ase
import ase.io
from ase.data import covalent_radii, chemical_symbols
from ase.units import Bohr, kB
import os
import json
from typing import Optional, Dict


def fit_atomic_E0s(atoms_list: list,
                   energy_label: str = 'energy') -> Dict[int, float]:
    """Least-squares fit of per-element atomic reference energies from a dataset.

    For each structure  E_total ≈ Σ_Z N_Z · E0[Z]  is solved jointly via
    np.linalg.lstsq. Useful when the user has not supplied `--E0s` for
    `dataset` / `reaction` mode — the resulting E0[Z] are *self-consistent
    with the dataset* (the constant offset cancels out of formation energies),
    even though they are not "true" isolated-atom energies.

    Parameters
    ----------
    atoms_list : list[ase.Atoms]
        Structures with energies stored in `atoms.info[energy_label]` or on
        `atoms.calc.results['energy']`.
    energy_label : str
        Where to read the total energy. Defaults to `'energy'`.

    Returns
    -------
    dict[int, float]
        Atomic-number → fitted E0 (eV).
    """
    elements = sorted({int(z) for atoms in atoms_list for z in atoms.numbers})
    z_to_col = {z: i for i, z in enumerate(elements)}

    A = np.zeros((len(atoms_list), len(elements)))
    b = np.zeros(len(atoms_list))
    for i, atoms in enumerate(atoms_list):
        for z in atoms.numbers:
            A[i, z_to_col[int(z)]] += 1
        if energy_label in atoms.info:
            b[i] = atoms.info[energy_label]
        elif atoms.calc is not None and energy_label in atoms.calc.results:
            b[i] = atoms.calc.results[energy_label]
        else:
            raise KeyError(
                f"Structure {i} has no '{energy_label}' on info or calc.results"
            )

    coeffs, _residuals, rank, _sv = np.linalg.lstsq(A, b, rcond=None)
    if rank < len(elements):
        print(f"[fit_atomic_E0s] WARNING: design matrix rank-deficient "
              f"({rank} < {len(elements)} elements). The fit is "
              f"underdetermined — datasets dominated by a single composition "
              f"cannot uniquely separate per-element E0s. Add more variety "
              f"or pass --E0s explicitly.")

    return {int(z): float(coeffs[z_to_col[z]]) for z in elements}


def load_E0s_cache(path: str) -> Optional[Dict[int, float]]:
    """Read a previously-saved E0s JSON cache. Returns None if file missing
    or unreadable. Keys are coerced back to int."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return None
    return {int(k): float(v) for k, v in raw.items()}


def save_E0s_cache(E0s: Dict[int, float], path: str) -> None:
    """Write E0s to disk as a JSON dict keyed by atomic number (string)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        json.dump({str(int(k)): float(v) for k, v in E0s.items()},
                  fh, indent=2, sort_keys=True)



def dist_min(atoms:Atoms) -> list:
    """
    Get the minimum atomic distance in a bulk structure

    Parameters:
    -----------
    atoms(Atoms): list of Atoms object
    
    Return:
        d_min(list): the minimum atomic distance in a bulk
    """
    
    d_min = []
    # file = open('dist_min.txt', 'w')

    for num, mole in enumerate(atoms):

        if mole.get_global_number_of_atoms() > 1: 
            dist = mole.get_all_distances(mic=True).flatten()
            dist = dist[dist!=0.]
            min_dist = np.min(dist)
            d_min.append(min_dist)
            # file.write(str(num) + ' ' + str(min_dist) + '\n')

        #For one atom bulk, the minimum distance is the minimum value of abc
        else:
            abc = mole.cell.cellpar()[:3]
            min_dist = np.min(abc)
            d_min.append(min_dist)
            # file.write(str(num) + ' ' + str(min_dist) + '\n')
    # file.close()
    return(d_min)

def dist_coor(coor):

    dist = 0
    for xyz in coor[:]:
        x, y, z = xyz
        dist += math.sqrt(x**2 + y**2 + z**2)
    return dist


def merge_atoms(atoms_a:Atoms, atoms_b:Atoms, prefixs=['band_', 'rep_']):
    """
    Merge two atoms object
    """
    prefix_a, prefix_b = prefixs
    atoms_total = []
    for num, atoms in enumerate(atoms_a):
        atoms_new = ase.Atoms(atoms.get_chemical_formula(), atoms.get_positions())
        atoms_new.set_cell(atoms.get_cell())
        atoms_new.set_pbc(atoms.get_pbc())
        
        atoms_new.info['energy'] = atoms.info[prefix_a + 'energy'] + atoms_b[num].info[prefix_b + 'energy']
        atoms_new.info[prefix_a + 'energy'] = atoms.info[prefix_a + 'energy']
        atoms_new.info[prefix_b + 'energy'] = atoms_b[num].info[prefix_b + 'energy']

        atoms_new.set_array('forces', atoms.get_array(prefix_a + 'forces') + atoms_b[num].get_array(prefix_b + 'forces'))
        atoms_new.set_array(prefix_a + 'forces', atoms.get_array(prefix_a + 'forces'))
        atoms_new.set_array(prefix_b + 'forces', atoms_b[num].get_array(prefix_b + 'forces'))

        atoms_new.info['structure_name'] = atoms.info['structure_name']

        atoms_total.append(atoms_new)
    return atoms_total


def get_full_dftb(dftb_bandstructure:list, dft:Atoms, pr:physrep, calc_forces:bool=False) -> list:
    """
    Implement repulsive potential to dftb_bandstructure part

    Parameters
    ------
    dftb_bandstructure: list[Atoms]
        the Atoms list with results only band energy part

    pr: physrep
        physrep object
    
    calc_forces: bool
        whether calculate forces or not  
        - False: when the optimization_option == EnergyGeomerty
        - True: when the optimization_option == Dataset

    """


    atoms_total = []
    print("Calculate the repulsive energy!")
    for num, atoms in enumerate(dft):
        ref_e, ref_f = pr.evaluate_pbc(atoms)
        atoms_new = ase.Atoms(atoms.get_chemical_formula(), 
                              atoms.get_positions())
        atoms_new.set_cell(atoms.get_cell())
        atoms_new.set_pbc(atoms.get_pbc())
        
        atoms_new.info['energy'] = ref_e + dftb_bandstructure[num].info['energy']
        atoms_new.info['band_energy'] = dftb_bandstructure[num].info['energy']
        atoms_new.info['rep_energy'] = ref_e

        if calc_forces:
            atoms_new.set_array('forces', ref_f + dftb_bandstructure[num].get_array('forces'))
            atoms_new.set_array('band_forces', dftb_bandstructure[num].get_array('forces'))
            atoms_new.set_array('rep_forces', ref_f)

        try:
            atoms_new.info['structure_name'] = atoms.info['structure_name']
        except:
            pass
        atoms_total.append(atoms_new)

    return atoms_total


def get_energy_per_atom(dft:Atoms,dftb:Atoms, bolt_all, bolt_fxx:list=None):
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
        bolt_fxx = [bolt_all[int(i)] for i in search_index]

    delta_energy_per_atom = np.sum(np.abs([dftb_energy_per_atom_relative[m] - dft_energy_per_atom_relative[m] for m in range(len(bolt_fxx)) if bolt_fxx[m] >= 0.1]) / np.sum([bolt_fxx[m] for m in range(len(bolt_fxx)) if bolt_fxx[m] >= 0.1]))
    # This is designed for normalizing structures in this subgroup

    # return mu_E
    return dftb_energy_per_atom_relative, dft_energy_per_atom_relative, delta_energy_per_atom


def get_elements_from_dataset(dataset, opt_elem):
    
    tot_atomic_numbers = np.unique(np.array([np.unique(mole.numbers).tolist() for mole in dataset]).flatten()) 
    atomic_numbers_exclusive = [i for i in tot_atomic_numbers if i not in opt_elem]
    atomic_numbers = atomic_numbers_exclusive + opt_elem

    atomic_symbols_exclusive = [chemical_symbols[i] for i in atomic_numbers_exclusive]
    atomic_symbols = atomic_symbols_exclusive + [chemical_symbols[num] for num in opt_elem]

    return atomic_numbers, atomic_numbers_exclusive, atomic_symbols, atomic_symbols_exclusive



def find_ref_by_structure(atoms_input1, atoms_input2, in_prefix, out_prefix):

    if in_prefix == '_':
        in_prefix = ''
    if out_prefix == '_':
        out_prefix = ''

    atoms_output = []
    search_index = []
    for i, atoms1 in enumerate(atoms_input1):  # this is the reference dataset

        read_energy = f"{in_prefix}energy"
        read_forces = f"{in_prefix}forces"
        write_energy = f"{out_prefix}energy"
        write_forces = f"{out_prefix}forces"

        for j, atoms2 in enumerate(atoms_input2):

            if np.array_equal(atoms2.get_positions(),atoms1.get_positions()) and np.array_equal(atoms2.get_cell(), atoms1.get_cell()) and np.array_equal(atoms2.get_atomic_numbers(), atoms1.get_atomic_numbers()):
                print(f"Found a match from no.{j} of Dataset2 to no.{i} of Dataset1")
                atoms_output.append(atoms2)
                search_index.append(j)
                break
    return atoms_output, search_index


def get_bolt_factor(atoms:Atoms, option:str, temperature:float) -> tuple:
    '''
    Calculate the Boltzmann factor as the weight of each molecule
    Parameters:
    ----------
        atoms(Atoms): a list of Atoms object

        option(str): whether ouput boltzmann factor weight or uniformly equal to 1/num_atoms or 1 for all
                        eg. "boltzmann" or "uniform" and "identical"

        temperature(float): temperature in K
    
    Returns:
        bf_weight_norm(List): list of boltzmann factor weight for each structure
        bf_weight_norm11(List): list of Boltzmann factors weight for each structure's EOS expansion

    '''
    T = temperature
    atoms_number = len(atoms)
    if option == "boltzmann":
        energy_per_atom = [dft_mole.get_total_energy() / dft_mole.get_global_number_of_atoms() for dft_mole in atoms]
        energy_per_atom_min = np.array(energy_per_atom).min()

        """Bf = exp(-(E-Emin)/(kB*T)) """
        bf_weights = [np.exp(-(energy_per_atom[i] - energy_per_atom_min)/(kB * T)) for i in range(atoms_number)]

        """Bf_norm = Bf/sum(Bf)"""
        bf_weight_norm = bf_weights/sum(bf_weights)

        # When fitting the EOS, the number of data points is 11 times the number of atoms
        bf_weight_norm11 = np.array([[bf_weight_norm[n]] * 11 for n in range(atoms_number)]).flatten()

        """Bf = 1/num_of_structures"""
    elif option == "uniform":
        bf_weight_norm = np.array([1/atoms_number] * atoms_number)
        bf_weight_norm11 = np.array([1/atoms_number] * atoms_number * 11)
    elif option == "identical":
        bf_weight_norm = np.array([1] * atoms_number)
        bf_weight_norm11 = np.array([1] * atoms_number * 11)
    else:
        raise ValueError('option should be "boltzmann", "uniform" or "identical"')
    return(bf_weight_norm, bf_weight_norm11)