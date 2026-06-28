
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
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pylibxc
from ase import Atoms
from ase.build import molecule
from ase.calculators.aims import Aims
from ase.calculators.dftb import Dftb
from ase.data import atomic_numbers, chemical_symbols, covalent_radii
from ase.eos import EquationOfState as EOS
from ase.io import Trajectory
from ase.io.jsonio import read_json
from ase.units import Bohr, kB, kJ
from ase.utils.deltacodesdft import delta
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import PowerConfinement
# from hotcent.slako import SlaterKosterTable 
from hotcent.offsite_twocenter import Offsite2cTable as SlaterKosterTable
from typing import List, Dict, Tuple
import multiprocessing as mp
from itertools import combinations
from math import pi, sqrt

from utils.parameters_set import get_PTBP, get_QNplusRep


class hotpy():
    def __init__(self,
                 confinement_r0: Dict[str, List[float]] = {'H': [1.0, 2.0], 'C': [3.0, 4.0], 'O': [5.0, 6.0]},
                 confinement_p: Dict[str, float] = {'H': 2.0, 'C': 2.0, 'O': 2.0},
                 workdir: str = './',
                 slator_p: str = './',
                 xc: str = 'GGA_X_PBE+GGA_C_PBE',
                 superposition: str = 'density',) -> None:
        """Initializes a hotpy object."""
        self.confinement_r0 = confinement_r0
        self.confinement_p = confinement_p
        self.symbols = list(confinement_r0.keys())
        self.workdir = workdir
        self.slator_p = slator_p
        self.xc = xc
        self.superposition = superposition
    

    def set_variables(self, 
                      confinement_r0: dict,
                      confinement_p: dict,
                      xc: str,
                      superposition: str,
                      workdir: str='./',
                      slator_p: float='./',
    ):
        self.confinement_r0 = confinement_r0
        self.confinement_p = confinement_p
        self.symbols = list(confinement_r0.keys())
        self.workdir = workdir
        self.slator_p = slator_p
        self.xc = xc
        self.superposition = superposition
     
    def get_orbit(self, valence_shell:str) -> tuple:
                """
                Parser valence_shell string occupation, valances, orbits

                Parameters:
                -----------
                    valence_shell(str): valence shell of an element
                    eg. '4d8 5s1 5p0' for 'Rh' 

                Return:
                    eg. (occupations,valences,orbits,valence_electron_numbers) = ({'4d': 8, '5s': 1, '5p': 0}, ['4d', '5s', '5p'],['d', 's', 'p'], 9)
                """

                info, occupations, valences, orbits = [], {}, [], []


                [info.append(i) for i in re.split(r'\s+', valence_shell.strip())]

                valence_electron_numbers = 0  # number of valence electrons

                for orb in info:
                    # maches numbers from the end
                    num_elec = re.findall(r'\d+$', orb)
                    occupations[orb[:2]] = int(num_elec[0])
                    valence_electron_numbers += int(num_elec[0])
                    # append valences to [valences] and
                    orbits.append(orb[1])
                    valences.append(orb[0:2])      # [orbits]
                return(occupations, valences, orbits, valence_electron_numbers)  

    # def hot_hamiltonian(self, const, sig, xc, element) -> dict:
    def hot_hamiltonian(self, r0_x:list, p_x:list, xc:str, symbol_x:str, label:str='ptbp'): 
            """
            Atomic DFT calculation for isolated atom

            Parameter:
            ----------
                r0_x(list): r0 parameters for confinement potential, [wavefunction r0, density function r0]
                    e.g. [1,2]

                p_x(list): p value for confinement potential
                    e.g. 2.0

                V_conf = (r_cov/r0)^p

                xc(str): exchange correlation method
                    e.g. 'GGA_X_PBE+GGA_C_PBE'

                symbol_x(str): element symbol
                    e.g. 'H'

            Returns:
            --------
                Return AtomicDFT, 
            """

            # Get the valence shell information store in get_PTBP()
            if label.lower() in ['ptbp', 'prior', 'personal']:
                occupations, valences, orbits, outer_ele = self.get_orbit(
                    get_PTBP()[symbol_x]['valence_shell'])
                
                config = get_PTBP()[symbol_x]['configuration']
                print("Occupations:", occupations, "\nValences:",
                    valences, "\nOrbits:", orbits)


                # Get previous calculated eigenvalues and hubbard values store in get_PTBP()
                atom_info = {'hubbardvalues':{}, 'eigenvalues':{}}
                Ed, Ep, Es = get_PTBP()[symbol_x]['Ed'], get_PTBP()[symbol_x]['Ep'], get_PTBP()[symbol_x]['Es']
                Ud, Up, Us = get_PTBP()[symbol_x]['Ud'], get_PTBP()[symbol_x]['Up'], get_PTBP()[symbol_x]['Us']
            elif label.lower() in ['qnplusrep', 'quasinano', 'quasinano2013']:
                occupations, valences, orbits, outer_ele = self.get_orbit(
                    get_QNplusRep()[symbol_x]['valence_shell'])
                
                config = get_QNplusRep()[symbol_x]['configuration']
                print("Occupations:", occupations, "\nValences:",
                    valences, "\nOrbits:", orbits)
                

                # Get previous calculated eigenvalues and hubbard values store in self.atoms_info
                atom_info = {'hubbardvalues':{}, 'eigenvalues':{}}
                Ed, Ep, Es = get_QNplusRep()[symbol_x]['Ed'], get_QNplusRep()[symbol_x]['Ep'], get_QNplusRep()[symbol_x]['Es']
                Ud, Up, Us = get_QNplusRep()[symbol_x]['Ud'], get_QNplusRep()[symbol_x]['Up'], get_QNplusRep()[symbol_x]['Us']
            else:
                raise ValueError('label should be "ptbp" or "quasinano"')
            

            for i in range(len(orbits)):
                if orbits[i] == 'd':
                    atom_info['hubbardvalues'][valences[i]] = Ud
                    atom_info['eigenvalues'][valences[i]] = Ed
                elif orbits[i] == 'p':
                    atom_info['hubbardvalues'][valences[i]] = Up
                    atom_info['eigenvalues'][valences[i]] = Ep
                elif orbits[i] == 's':
                    atom_info['hubbardvalues'][valences[i]] = Us
                    atom_info['eigenvalues'][valences[i]] = Es
            print('info:', atom_info)

            atom_info['occupations'] = occupations



            # Set wavefunction confinement parameter for each valence        
            conf_wf = {}
            for val in valences:  # set confinement for each valence
                conf_wf[val] = PowerConfinement(r0=(r0_x[0]), s=p_x)

            # Set density confinement parameter
            conf_dens = PowerConfinement(r0=r0_x[1], s=p_x)


            # Apply confinement potential and solving the pseudo-atom orbitals
            atom = AtomicDFT(symbol_x,
                            xc=xc,
                            configuration=config,
                            valence=valences,
                            scalarrel=True,
                            confinement=conf_dens,
                            wf_confinement=conf_wf,
                            txt='-',
                            )
            atom.run()

            atom.info = {'hubbardvalues': {}}
            atom.info['hubbardvalues'] = atom_info['hubbardvalues']
            atom.info['occupations'] = atom_info['occupations']
            atom.info['eigenvalues'] = atom_info['eigenvalues']

            return atom, config, r0_x
        

    def generate_skf(self, symbol_pair:list, r0_pair:dict, p_pair:dict, xc:str, superposition:str, label:str='ptbp'):
        

            """
            To generate skf file for one symbol pair [symbol_a, symbol_b]
            
            Parameters:
            ----------
                Theory:
                    V_conf = (r_cov/r0)^p
                    where, r_cov = covalent_radii[atomic_numbers[element]] / Bohr


                symbol_pair(list): element symbol list 
                    eg. ['H', 'C']

                r0_pair(dict): r0 parameters for confinement potential, eg. {'H': [wavefunction r0, density function r0], 'C': [wavefunction r0, density function r0]}
                    eg. {'H': [1,2], 'C': [2,3]}

                p_pair(dict): p value for confinement potential
                    eg. {'H': 2.0, 'C': 2.0}

                xc: exchange correlation method 
                    eg. 'GGA_X_PBE+GGA_C_PBE' 

                superposition: superposition method
                    eg. 'denisty' or 'potential'
                label: str,
                    eg. 'ptbp', 'prior', or 'quasinano'

            Return:
            -------
                skf file
            """


            # Build subfolder to avoid confilication of multiprocessing 
            name_subfolder = 'SKF_' + '_'.join(symbol_pair)
            if not os.path.exists(name_subfolder):
                os.mkdir(name_subfolder)
                os.chdir(name_subfolder)
            else:
                os.chdir(name_subfolder)


            # Calculate hamiltonian for each symbol(element)
            # To get pseudo-atom orbitals
            hamitonian_dict = {}  # store hamiltonian for each symbol
            hot_log = open('../pot.out', 'a') 

            if len(symbol_pair) == 2 and symbol_pair[0] == symbol_pair[1]:
                hot_log.write("calculate {}'s hamitonian for elementary\n".format(symbol_pair[0]))
                hot_log.flush()
                r0_x = r0_pair[symbol_pair[0]]
                p_x = p_pair[symbol_pair[0]]

                # Call hot_hamiltonian function to calculate hamiltonian for each symbol(element)
                atom_x, config_x, r0_x = self.hot_hamiltonian(r0_x, p_x, xc, symbol_pair[0], label)

                hamitonian_dict[symbol_pair[0]] = [atom_x, config_x, r0_x]

            else:
                for i, symbol_x in enumerate(symbol_pair):
                    hot_log.write("calculate {}'s hamitonian\n".format(symbol_x))
                    hot_log.flush()
                    r0_x = r0_pair[symbol_x]
                    p_x = p_pair[symbol_x]

                    # Call hot_hamiltonian function to calculate hamiltonian for each symbol(element)
                    atom_x, config_x, r0_x = self.hot_hamiltonian(r0_x, p_x, xc, symbol_x, label)
                    hamitonian_dict[symbol_x] = [atom_x, config_x, r0_x]



            # Write skf file
            hot_log.write("###Write skf file###\n")
            for i, symbol_a in enumerate(symbol_pair):
                for k in range(i+1):
                    symbol_b = symbol_pair[k]

                    
                    hot_na = symbol_a + '-' + symbol_b + '.skf'
                    print("\nWorking on {}\n".format(hot_na))
                    hot_log.flush()

                    atom_a, config_a, r0_a = hamitonian_dict[symbol_a][0], hamitonian_dict[symbol_a][1], hamitonian_dict[symbol_a][2]
                    atom_b, config_b, r0_b = hamitonian_dict[symbol_b][0], hamitonian_dict[symbol_b][1], hamitonian_dict[symbol_b][2]


                    # #==============================================================#
                    # # The code will not overwrite old skf file                     #
                    # #==============================================================#
                    if os.path.exists(hot_na): #and os.path.getsize(hot_na):
                        # print("{} existed! I am gonna backup it to {}".format(hot_na, hot_na.lower()))
                        # os.rename(hot_na, hot_na.lower())
                        print(f"There are old skf files:{hot_na}, I am gonna remove them")
                        os.system(f"rm {hot_na}")

                    rmin, dr, N = 0.4, 0.02, 700

                    sk = SlaterKosterTable(atom_a, atom_b)
                    sk.run(rmin, dr, N, superposition=superposition, xc=xc)

                    if atom_a == atom_b:
                        print(f'atom_a.info: {atom_a.info}')
                        sk.write(eigenvalues=atom_a.info['eigenvalues'],
                                 hubbardvalues=atom_a.info['hubbardvalues'],
                                 occupations=atom_a.info['occupations'], 
                                 spe=0.
                                 )
                        
                        # For Hotcent 2.0, the skf file needs to be renamed as "symbol-symbol_offsite2c.skf" to "symbol-sybmol.skf"
                        hotcent_name = f"{symbol_a}-{symbol_b}_offsite2c.skf"
                        os.system(f"mv {hotcent_name}  {hotcent_name.split('_')[0]}.skf")
                        print(f"Rename {hotcent_name} to {hotcent_name.split('_')[0]}.skf")
                    else:
                        sk.write()
                        # For Hotcent 2.0, the skf file needs to be renamed as "symbol-symbol_offsite2c.skf" to "symbol-sybmol.skf"
                        hotcent_name0 = f"{symbol_a}-{symbol_b}_offsite2c.skf"
                        hotcent_name1 = f"{symbol_b}-{symbol_a}_offsite2c.skf"
                        os.system(f"mv {hotcent_name0}  {hotcent_name0.split('_')[0]}.skf")
                        os.system(f"mv {hotcent_name1}  {hotcent_name1.split('_')[0]}.skf")

                        print(f"Rename {hotcent_name0} to {hotcent_name0.split('_')[0]}.skf")
                        print(f"Rename {hotcent_name1} to {hotcent_name1.split('_')[0]}.skf")
                    
                    hot_log.write('config_a=' + str(config_a) + '\n')
                    hot_log.write('config_b=' + str(config_b) + '\n')
                    hot_log.write('#r0 for eigenvalue calculation:a,b=' + str(r0_a) + ',' + str(r0_b) + '\n')
                    hot_log.flush()


            os.system("mv *.skf ../")
            os.chdir('..')
            os.rmdir(name_subfolder)

            hot_log.close()


    def error_callback(self, exception):
        print('Exception:', exception)

    def full_hotcent(self, superposition:str, opt_elem:list=[], label:str='ptbp'):
            """
            Generate full SKF files (band+repulsion) based on existed parameters

            Parameters:
                xc(str): exchange correlation method 
                    eg. 'GGA_X_PBE+GGA_C_PBE'

                superposition(str): superposition method 
                    eg. 'denisty' or 'potential'

                opt_symbol: list of symbol that need to optimize confinement parameters
                    eg. ['O']

            """
            xc = self.xc
            opt_symbol = [i for i in opt_elem]
            r0 = self.confinement_r0
            p = self.confinement_p
            symbols = self.symbols

            hot_log = open('pot.out', 'w') 

            # if opt_symbol != None:
            if len(opt_symbol) == 0:
                if len(symbols) == 1:
                    symbol_pairs = [(symbols[0], symbols[0])]
                else:
                    symbol_skf = symbols
                    symbol_pairs = [comb for comb in combinations(symbol_skf, 2)]
            elif len(opt_symbol) >= 1:
                symbol_skf = [i for i in symbols if i not in opt_symbol] 

                symbol_pairs = []
                [symbol_pairs.append([elx] + opt_symbol) for elx in symbol_skf]

            hot_log.write('#---------------rconf=' + str(r0) + '---------------\n')
            hot_log.close()

            num_cores = int(mp.cpu_count())
            print("CPU cores: " + str(num_cores))
            print("Number of jobs: " + str(len(symbol_pairs)))
            
            
            if len(symbol_pairs) < num_cores:
                pool = mp.Pool(processes=len(symbol_pairs)) if len(symbol_pairs) != 0 else mp.Pool(processes=1)
            else:
                pool = mp.Pool(processes=num_cores)


            for i, pair in enumerate(symbol_pairs):
                symbol_pair = list(pair)

                r0_pair = {sym: r0[sym] for sym in symbol_pair}
                p_pair = {sym: p[sym] for sym in symbol_pair}

                pool.apply_async(self.generate_skf, args=(symbol_pair, r0_pair, p_pair, xc, superposition, label), error_callback=self.error_callback)
            
            pool.close()
            pool.join()



    
    
