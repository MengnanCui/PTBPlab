# import json
# import math
# import os
# import re
import argparse
import ase
import numpy as np

np.random.seed(237)
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from ase.data import atomic_numbers, chemical_symbols, covalent_radii
from ase.units import kB, Hartree, Bohr
from scipy.optimize import Bounds, minimize

from hotpy import hotpy
from physrep import physrep
from argparse import RawTextHelpFormatter

 
import sys

#sys.argv = ['skf_generator.py', '-s', 'C', 'O', '-p', 'quasinano', '-skf', '-rep']
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('-s', '--symbol', 
                        default=None,
                        nargs='+', 
                        help='The chemical symbol of the atom. \
                        \ne.g. C H O')
    
    parser.add_argument('-d', '--data', type=str, 
                        help='Give dataset file while you are lazy to input symbols \
                            \n -d dft.xyz')
    
    parser.add_argument('-p', '--parameter_set', type=str, 
                        help='Apply confinement and repulsive parameters from either Periodic table baseline parameters set or QUASINANO. \
                            \ne.g. PTBP or QUASINANO')
    
    parser.add_argument('-skf', '--skf_genetator', action='store_true', 
                        help='Activate the skf generator.')
    
    parser.add_argument('-rep', '--repulsion_generator', 
                        default=None,
                        nargs='+', 
                        type=float, 
                        help='Activate the repulsive fitting. \
                            \n e.g. -rep 0.5 0.6 for C and H')
    
    parser.add_argument('-r0', '--confinement_r0', 
                        default=None,
                        nargs='+', 
                        type=float,
                        help='The confinement parameter for wavefunction and density function for each atom. \n e.g. -r0 2.0 4.0 3.0 6.0 for -s H C') 
    
    parser.add_argument('-sigma', '--confinement_sigma', 
                        default=None,
                        nargs='+', 
                        type=float,
                        help='The confinement parameter sigma for each atom. \n e.g. -sigma 2.0 2.0 for -s H C')

args = parser.parse_args()



def set_parameters(parameters:list,
                   sigma:float):
    """
    Set parameters as the input for Hotcent and Physrep
    """

    kxc = 0.0
    r0, r0_dens = parameters


    if opt_elem == None or len(opt_elem) == 0:

        confinement_r0 = {s: [r0, r0_dens] for s in atomic_symbols}
        confinement_sigma = {s: sigma for s in atomic_symbols}


    elif len(opt_elem) > 0:

        confinement_r0 = {s: [hp.atoms_info[s]['r_wave'], hp.atoms_info[s]['r_dens']] for s in atomic_symbols_exclusive}
        confinement_r0[chemical_symbols[opt_elem[0]]] = [r0, r0_dens]
        # confinement_sigma = {s: sigma for s in atomic_symbols_exclusive}
        confinement_sigma = {s: sigma for s in atomic_symbols}

    return confinement_r0, confinement_sigma

def get_elements_from_dataset(dataset, opt_elem):
    
    tot_atomic_numbers = np.unique(np.array([np.unique(mole.numbers).tolist() for mole in dataset]).flatten()) 
    atomic_numbers_exclusive = [i for i in tot_atomic_numbers if i not in opt_elem]
    atomic_numbers = atomic_numbers_exclusive + opt_elem

    atomic_symbols_exclusive = [chemical_symbols[i] for i in atomic_numbers_exclusive]
    atomic_symbols = atomic_symbols_exclusive + [chemical_symbols[num] for num in opt_elem]

    return atomic_numbers, atomic_numbers_exclusive, atomic_symbols, atomic_symbols_exclusive



#==============================================================#
hp = hotpy()
#symbol_list = [s for s in sys.argv[1:]]
workdir, slator_path = './', './'
xc = 'GGA_X_PBE+GGA_C_PBE'
opt_elem=[]

if args.symbol is not None: 
    if len(opt_elem) >= 1:
        symbol_list = [s for s in args.symbol if s not in chemical_symbols[opt_elem[0]]] + [chemical_symbols[num] for num in opt_elem]
    else:
        symbol_list = [s for s in args.symbol]
        atomic_numbers_list = [atomic_numbers[s] for s in symbol_list]

elif args.data is not None:
    dft_input = ase.io.read(args.data, ':')
    atomic_numbers_list, atomic_numbers_exclusive , atomic_symbols, atomic_symbols_exclusive = get_elements_from_dataset(dft_input, opt_elem)
    symbol_list = atomic_symbols



#================================================================#
if args.parameter_set.lower() == 'ptbp':
    # Get r_wave and r_dens for wavefunction and density function, respectively
    r0_dict = {symbol: [hp.atoms_info[symbol]['r_wave'], hp.atoms_info[symbol]['r_dens']] for symbol in symbol_list}

    # Get sigma
    sigma_dict = {symbol: 2.0 for symbol in symbol_list}

    # Get c_rep for repulsion if exist
    crep_list = [hp.atoms_info[symbol]['c_rep'] for symbol in symbol_list]
    kxc_list = [0.0 for symbol in symbol_list]
    superposition = 'density'


    print('r0_dict', r0_dict)
    print('sigma_dict', sigma_dict)
    print('crep_list', crep_list)

elif args.parameter_set.lower() == 'quasinano':
    # Get r_wave and r_dens for wavefunction and density function, respectively
    r0_dict = {symbol: [hp.atoms_info_quasinano[symbol]['r_quasinano'], hp.atoms_info[symbol]['r_quasinano']] for symbol in symbol_list}

    # Get sigma
    sigma_dict = {symbol: hp.atoms_info_quasinano[symbol]['sigma_quasinano'] for symbol in symbol_list}

    # Get c_rep for repulsion if exist
    crep_list = [hp.atoms_info_quasinano[symbol]['c_rep'] for symbol in symbol_list]
    kxc_list = [0.0 for symbol in symbol_list]

    superposition = 'potential'


    print('r0_dict', r0_dict)
    print('sigma_dict', sigma_dict)
    print('crep_list', crep_list)

elif args.parameter_set.lower() == 'personal':
    # Get r_wave and r_dens for wavefunction and density function, respectively
    r0_list = list(args.confinement_r0)
    sigma_list = list(args.confinement_sigma)
    r0_dict = {symbol: [r0_list[2*i], r0_list[2*i+1]] for i, symbol in enumerate(symbol_list)}
    sigma_dict = {symbol: sigma_list[i] for i, symbol in enumerate(symbol_list)}

    # Get c_rep for repulsion if exist
    if args.repulsion_generator is not None:
        crep_list = [float(c) for c in args.repulsion_generator]


    superposition = 'density'


    print('r0_dict', r0_dict)
    print('sigma_dict', sigma_dict)

else:
    print('Please choose the parameter set from either PTBP or QUASINANO.')


#================================================================#
# Generate skf
if args.skf_genetator:
    hp = hotpy(r0_dict, sigma_dict, workdir=workdir, slator_p=slator_path)
    # except_symbol = ''
    hp.full_hotcent(superposition, opt_elem, args.parameter_set.lower())   # run Hotcent

#================================================================#
# Fit repulsion for elementary atoms
if args.repulsion_generator is not None:
    # convert symbol_list to atomic_num
    # atomic_number_list = [hp.atoms_info[symbol]['atomic_number'] for symbol in symbol_list]
    kxc_list = [0.0 for symbol in symbol_list]
    pr = physrep(atomic_numbers_list, 
                 rmin=0.5,
                 rmax=20.0,
                 npoints=666,
                 scale_cov=crep_list,
                 k_xc=kxc_list)
    
    if len(opt_elem) == 0: 
        pr.write_skf()
    else:
        pr.write_specific_skf(opt_elem)
