import os
import json
import math
import copy
import numpy as np
from ase import Atoms
from ase.calculators.dftb import Dftb
from ase.units import kJ
from math import pi, sqrt, ceil
import ase.io
from utils.parameters_set import get_PTBP, get_QNplusRep
from utils.hotpy import hotpy
from ase.utils.deltacodesdft import delta
from ase.data import chemical_symbols, atomic_numbers
from ase.io.jsonio import read_json
import pprint as pp
import matplotlib.pyplot as plt
import pickle
hp = hotpy()



path_dict = {'dimer':  {'path':'MRGXMGR','npoints':101},
                'fcc':    {'path':'WXGLKGL','npoints':101},
                'bcc':    {'path':'PGHPGN','npoints':101}, #PGHPGN
                'beta-Sn':{'path':'GXMGZPNZ1XP','npoints':101},
                'diamond':{'path':'WXGLKGL','npoints':101},
                'AA-graphite':{'path':'GKHAGMLA','npoints':101},
                'wz_ZnO':{'path':'GMKGALHA,LM,KH','npoints':101},
                'CuZn': {'path':'GXMGRX,MR', 'npoints': 101},
                'CuZnO3': {'path': 'GXMGRX,MR', 'npoints': 101},
                'ZnAg2O4':{'path': 'GXMGZRAZ,XR,MA', 'npoints': 101},
                'ZnCu2O4':{'path': 'GXMGZRAZ,XR,MA', 'npoints': 101},
}

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


def get_bandpath_for_dftb(atoms:Atoms, kpts:int):
    """This function sets up the band path according to Setyawan-Curtarolo conventions.

    Parameters:
    -----------
    atoms(Atoms): ase.Atoms object
        The molecule or crystal structure.
    kpts(int): The number of k-points among two special kpoint positions.

    Returns(Dict):
    -------------

    """
    from ase.dft.kpoints import kpoint_convert, parse_path_string

    # atoms.pbc = pbc
    # path = parse_path_string(
    #     atoms.cell.get_bravais_lattice(pbc=atoms.pbc).bandpath().path
    # )

    path = parse_path_string(kpts['path'])
    # lists of path segments
    points = atoms.cell.get_bravais_lattice(pbc=atoms.pbc).bandpath().special_points
    segments = []
    for seg in path:
        section = [(i, j) for i, j in zip(seg[:-1], seg[1:])]
        segments.append(section)


    output_bands = []
    output_bands = np.empty(shape=(0, 3))
    index = kpts['npoints']
    for seg in segments:
        # output_bands.append("## Brillouin Zone section Nr. {:d}\n".format(index))
        for num, sec in enumerate(seg):
            dist = np.array(points[sec[1]]) - np.array(points[sec[0]])
            npoints = index
            if num == 0:
                dist_matrix = np.linspace(points[sec[0]], points[sec[1]], npoints)
            else:
                dist_matrix = np.linspace(points[sec[0]], points[sec[1]], npoints)[1:,:]
            output_bands = np.vstack((output_bands, dist_matrix))

    return {'path':path, 'kpts':output_bands}




def monkhorstpack2kptdensity(atoms, k_density):

    """
    Convert Monkhorst-Pack grid to k-point density.
    atoms (ase.atoms.Atoms): Atoms object.
    k_grid (list): [nx, ny, nz].
    Returns:
        float: Smallest line-density.
    """

    # assert len(k_grid) == 3, "Size of k_grid is not 3."

    recipcell = atoms.cell.reciprocal()
    kd = [] # k-point density
    ks = [] # k-point space
    # initial a array for k_grids, with 3 columns
    # k_grids = np.empty((1,3))
    k_grid_abc = []
    for i in range(3):
        if atoms.pbc[i]:
            k_grida = int(k_density * (2 * pi * sqrt((recipcell[i] ** 2).sum())))
            if k_grida == 0:
                k_grida = 1
            k_grid_abc.append(k_grida) # current k_grid
            # k_grids = np.vstack((k_grids, k_grid_abc))
            # kptdensity = k_grid[i] / (2 * pi * sqrt((recipcell[i] ** 2).sum()))
            kptspace =  sqrt((recipcell[i] ** 2).sum()) / k_grida
            ks.append(kptspace)
    return ks, k_grid_abc


def run_calculator(dft_data:Atoms, poly_rep:str, SCC:str, kpts_dens, bolt_f):
        """
        Calculate DFTB+ and delta forces between DFT and DFTB

        Parameters:
            dft_data: dft Atoms object
            opt: 'Yes' or 'No'. 'yes', when skf without repulsive pot.  
            kpts: list eg. kpts = [11,11,11]
            bolt_f: Boltzmann factor 1x6 array

        Return:
            Atoms object
        """
        dftb_args = {"calc_type": "static",
                     "poly_rep" : poly_rep,
                     "SCC"      : SCC,
                     "slater_p" : './',
        }

        atoms = []
        for num, mole_dft in enumerate(dft_data):
            atoms_new = Atoms(mole_dft.get_chemical_symbols(), positions=mole_dft.get_positions())
            atoms_new.set_cell(mole_dft.get_cell())
            atoms_new.set_pbc(mole_dft.get_pbc())


            # formula = mole_dft.get_chemical_formula()
            # force_dft = mole_dft.get_forces()

            # calculate dftb forces
            if bolt_f[num] < 0.1:
                kpts_dftb = [1,1,1]
            else:
                ks, kpts_dftb = monkhorstpack2kptdensity(mole_dft, kpts_dens)

            dftb_args['kpts'] = kpts_dftb
            calc = dftbplus_calculator(atoms_new, dftb_args)
            atoms_new.calc = calc
            # energy     = atoms_new.get_potential_energy()
            force_dftb = atoms_new.get_forces()
            atoms.append(atoms_new)
        return atoms



def dftbplus_calculator(mole:Atoms, dftb_args:dict):
    '''
    mole is atoms object
    formula is the str name of molecules
    Parameters:
        type: 'band' or 'static' or 'opt_geometry'
        opt: Yes when slater file without repulsive pot
        formula: whatever string lable
        h_scc = No when don't using SCC calculation
        slater_p is the directory of slator-koster file
        kpts: (20, 20, 20)
    '''
    # poly_rep = 'SetForAll {' + str(opt) + '}'
    calc_type, poly_rep, SCC, kpts, slater_p = dftb_args['calc_type'], dftb_args['poly_rep'], dftb_args['SCC'], dftb_args['kpts'], dftb_args['slater_p']
    formula = mole.get_chemical_formula()
    
    if calc_type == 'static':
        calc = Dftb(atoms=mole, label=str(formula), kpts=kpts,

                    Hamiltonian_='DFTB',
                    Hamiltonian_SCC=SCC,
                    Hamiltonian_SCCTolerance=1e-5,
                    Hamiltonian_MaxSCCIterations=1000,
                    Hamiltonian_ConvergentSCCOnly = 'No',
                    Hamiltonian_ReadInitialCharges='No', # Static calculation
                    Hamiltonian_ForceEvaluation='dynamics',


                    # ONLY FOR SCC CONVERGENCE
                    Hamiltonian_Fermi_='', Hamiltonian_Fermi_Temperature=2000,
                    Hamiltonian_MethfesselPaxton_='', Hamiltonian_MethfesselPaxton_Order=2,
                    Parallel_='', Parallel_Groups=1,


                    # SlaterKosterFiles
                    Hamiltonian_SlaterKosterFiles_='Type2FileNames',
                    Hamiltonian_SlaterKosterFiles_Prefix=slater_p,
                    Hamiltonian_SlaterKosterFiles_Separator='"-"',
                    Hamiltonian_SlaterKosterFiles_Suffix='".skf"',
                    Hamiltonian_MaxAngularMomentum_='',


                    # Repulsive Potential
                    Hamiltonian_PolynomialRepulsive=f'SetForAll {{{poly_rep}}}',


                    # Options
                    Options_='', Options_WriteResultsTag='Yes',        # Default:No
                    Options_WriteDetailedOut='Yes', # Default:
                    )


    elif calc_type == 'band':
                calc = Dftb(atoms=mole, label=str(formula), kpts=kpts,

                Hamiltonian_='DFTB',
                Hamiltonian_SCC=SCC, Hamiltonian_SCCTolerance=1e-5,
                Hamiltonian_MaxSCCIterations=1,
                Hamiltonian_ReadInitialCharges='Yes',
                Hamiltonian_SlaterKosterFiles_Prefix=slater_p,
                Hamiltonian_SlaterKosterFiles_Separator='"-"',
                Hamiltonian_SlaterKosterFiles_Suffix='".skf"',
                Hamiltonian_PolynomialRepulsive=f'SetForAll {{{poly_rep}}}',
                Hamiltonian_ForceEvaluation='dynamics',

                Options_='', Options_WriteResultsTag='Yes',        # Default:No
                Options_WriteDetailedOut='Yes',
                )
    elif calc_type == 'xtb':
        calc = Dftb(atoms=mole, label=str(formula),
                kpts=kpts,
                Hamiltonian_='xTB',
                Hamiltonian_Method="GFN1-xTB",
                Hamiltonian_SCC=SCC,
                Hamiltonian_SCCTolerance=1e-5,
                Hamiltonian_MaxSCCIterations=1000,
                Hamiltonian_ConvergentSCCOnly = 'No',
                Hamiltonian_ReadInitialCharges='No', # Static calculation
            
                
                #Hamiltonian_Filling_='',
                #Hamiltonian_Filling_MethfesselPaxton_='', 
                #Hamiltonian_Filling_MethfesselPaxton_Order=2,
                #Hamiltonian_Filling_MethfesselPaxton_Temperature=2000,
                Hamiltonian_SlaterKosterFiles_='',
                Hamiltonian_MaxAngularMomentum_='',


                Options_='', Options_WriteResultsTag='Yes',        # Default:No
                Options_WriteDetailedOut='Yes', # Default:
                Parallel_='', Parallel_Groups=64,
    )
    else:
        raise ValueError('type must be band or static')
    return calc


def calc_band_binary(conf_parameters, dft_atoms, kpt_dens, bolt_f):
    """
    Band calculation for DFTB+

    Parameters
    ----------
    para_list : list
        List of parameters.

    """
    
    # Label
    para_name = '_'.join([str(round(i,3)) for i in conf_parameters])


    # Calculate the band structure for each structure
    for num, mole in enumerate(dft_atoms):

        # # config = mole.info['structure_name']
        # print("\n#===========Electronic calculation for {}============#".format(config))
        print(f"\n#===========Electronic calculation for {num}/{len(dft_atoms)}============#")
        print("bolt_f:", bolt_f[num])
        try:
            configuration = mole.info['structure_name']
        except:
            configuration = 'band_' + str(num)
        print('Band structure:', configuration)

        os.system('rm charges.bin')

        if bolt_f[num] < 0.1:
            continue
        else:
    # ============================================================================ #
    #       # Static calculation
            # Name the output file also for getting kpts path
            # The input file need to have a info named "structure_name"
            # To (1) label output file (2) get the kpts path for path_dict


            ks, kpts = monkhorstpack2kptdensity(mole, kpt_dens)

            dftb_args = {"calc_type": "static",
                            "poly_rep" : 'Yes',
                            "SCC"      : 'Yes',
                            "slater_p" : './',
                            "kpts"     : kpts,
            }

            mole_static = copy.deepcopy(mole)

            mole_static.calc = dftbplus_calculator(mole_static, dftb_args)

            # Attention: 
            # (1) First Yes means read repulsion from polynomial, however, it is 0 in skfiles, shortly, means no repulsion
            # (2) Second Yes means open "SCC" calculation
            # mole.calc = self.dftbplus_calculator('static', mole, 'res', 'Yes', 'Yes', './', kpts)

            print('------------\nStep1: Static calculation\nEnergy:', mole_static.get_potential_energy())
            # mole.get_forces()
            
            fermi_static = mole_static.calc.get_fermi_level()
            print('Static Fermi_level:', fermi_static)

        # ============================================================================ #
        #   # Band structure calculation

            # Two ways to get kpts path
            # (1) Use the known path_dict
            # (2) Use the default path from ase.dft.kpoints.bandpath
            try:
                kpts_spe = path_dict[configuration]
                kpts_path= get_bandpath_for_dftb(mole, kpts=kpts_spe)
            except Exception as e:
                print('Find kpts path Error:')
                path_dict = mole.cell.get_bravais_lattice().get_special_points()
                kpts_path = ase.dft.kpoints.bandpath(path_dict, mole.cell)

            mole_band = copy.deepcopy(mole_static)
            # mole_band.calc = self.dftbplus_calculator('band', mole_band, 'res', 'Yes', 'No', './',kpts_path)
            dftb_args['kpts'] = kpts_path
            mole_band.calc = dftbplus_calculator(mole_band, dftb_args)
            print('------------\nStep2: Band structure calculation, \nEnergy:', mole_band.get_potential_energy())

            bs = mole_band.calc.band_structure()
            fermi_band = mole_band.calc.get_fermi_level()
            print("Band Fermi_level:", fermi_band, '\n')

        # ============================================================================ #
        #   # Attention to compare the fermi level between static and band
            # The density of kpoints could casually cause the difference of fermi level
            # Here, we use fermi level from static calculation!
            if abs(fermi_band - fermi_static)/abs(fermi_static) > 0.1:
                print('Warning!!! The Fermi_level between static and band is not consistent more than 10%!!!')

        # ========================================================================== #
            # Correct the Fermi level to HVB(highest valence band) = 0
            # PS: for semiconductor or insulator, fermi level from DFTB+ tends to located in the center of band gap
            # So, It is necessary to shift.

            # Get band gap from ASE bandgap as reference
            from ase.dft import bandgap
            band_gap = round(bandgap.bandgap(mole_band.calc)[0],5)  # Band gap from DFTB+
            print('Band gap from bandgap.bandgap(mole.calc):', band_gap, '\n')

            # When semiconductor or insulator, one need to find the HVB and LCB (lowest conduction band)
            if band_gap != 0.0: 

                # Collect the valence shell information 
                dftb_tot_chemical_symbols = mole.get_chemical_symbols() #eg:['Zn', 'Zn', 'O', 'O']
                dftb_num_valence_electron = np.sum(hp.get_orbit(get_PTBP()[syms]['valence_shell'])[3] for syms in dftb_tot_chemical_symbols)

                # Get the band for HVB and LCB
                dftb_json_energies = bs.energies
                dftb_LCB_e = dftb_json_energies[0, :, ceil(dftb_num_valence_electron/2)].min()
                dftb_HVB_e = dftb_json_energies[0, :, ceil(dftb_num_valence_electron/2) - 1].max()
            # Assure the band gap is consistent with ASE bandgap
                band_from_json = round(dftb_LCB_e - dftb_HVB_e,3) if dftb_LCB_e - dftb_HVB_e > 0 else 0.0
                print("LCB_e={:.4f}\nHVB_e={:.4f}\nDFTB_gap:{:.5f}".format(dftb_LCB_e, dftb_HVB_e, band_from_json))
                assert round(band_gap, 3) == round(band_from_json, 3), "The band gap of bandgap.bandgap() is not consistent with json file!"

            # Correction information
                print("Here I correct the fermi level from {} to HVB({})".format(fermi_band, dftb_HVB_e))


                # Save both orign and corrected one
                origin_band = copy.deepcopy(bs)
                origin_band._reference = fermi_static
                origin_band.write(configuration + '_' + para_name + '_o.json') # save the original band structure just for reference

                bs._reference = dftb_HVB_e
                bs.write(configuration + '_' + para_name + '.json')

            else:
                bs.write(configuration + '_' + para_name + '.json')

    print('Output band.xyz and json files!')
    return(bs)



def eos_cal(self, dataset: json, bolt_factor: list):
    """
    Calculate eos value

    Parameters:
        dataset: include dftb and dft data
        bolt_factor: bolt_factor of different configuration

    Return:
        delta_eos = (V_{0, DFTB} / V_{0, DFT} - 1)**2 + (B_{DFTB} / B_{DFT} - 1)**2
        mu = delta_eos * ((r_{cut} / r_{ref} - 1)**2 + 1)
    """

    # Write LogFile
    for name in ['volume', 'B', 'Bp']:
        with open(name + '.csv', 'w') as f:
            print('#symbol, dft, dftb', file=f)
            for symbol, dct in dataset.items():
                values = [dct[code + '_' + name]
                            for code in ['dft', 'dftb']]
                if name == 'B':
                    values = [val * 1e24 / kJ for val in values]
                print('{},'.format(symbol), ', '.join(
                    '{:.2f}'.format(value) for value in values), file=f)

    # calculate mu_eos
    delta_tot, mu_v_tot, b_tot, n_i = 0, 0, 0, 0
    # n_i = 0  # for multiply boltzmann factor

    bolt_left = []
    with open('delta.csv', 'w') as f:
        print('#symbol,delta(dft-dftb),bolt_factor,delta*bolt', file=f)
        for symbol, dct in dataset.items():
            # Get v0, B, Bp:

            # Read data from json file
            dft, dftb = [(dct[code + '_volume'],
                            dct[code + '_B'],
                            dct[code + '_Bp'])
                            for code in ['dft', 'dftb']]
            
            # Convert units
            v0_dftb, b_dftb, v0_dft, b_dft = dftb[0], dftb[1] * \
                1e24 / kJ, dft[0], dft[1] * 1e24 / kJ


            # Calculate LOSS function
            # If v0_dftb in 80% ~ 120% of v0_dft, we use it.

            # 1. Volume difference
            mu_v = abs(v0_dftb / v0_dft - 1)
            print(f'mu_v   =|{v0_dftb:16.3f} / {v0_dft:17.3f} -1| = {mu_v:10.5f}')
            if math.isnan(mu_v):  # To avoid unexpected parameters/configurations
                mu_v = 100

            # 2. Bulk modulus difference
            try:
                mu_logb = abs(np.log10(b_dftb) / np.log10(b_dft) - 1) + 1
                print(f'mu_dftb=|log10({b_dftb:10.3f})/log10({b_dft:10.3f}) - 1| + 1 = {mu_logb:10.5f}\n')
            # Except unphysical log(b) value
            # B: bulk module will be deprecated in the future
            except ZeroDivisionError:
                print('The birchmurnaghan EOS fitting for DFT failed')
                mu_logb = 100
            if math.isnan(mu_logb):  # To avoid unexpected parameters/configurations
                mu_logb = 100

            # 3. Multiplication between them (will be deprecated in the future)
            delta_eos = mu_v * mu_logb
            delta_eos_bolt_factor = delta_eos * bolt_factor[n_i]

            # Summary and Boltzmann factor additionif boltzmann factor larger >0.1, we consider its effects
            # otherwise, we neglect its influences
            if bolt_factor[n_i] < 0.1:
                n_i += 1
                continue
            else:
                mu_v_tot += mu_v * bolt_factor[n_i]
                b_tot += mu_logb * bolt_factor[n_i]
                delta_tot += delta_eos_bolt_factor
                bolt_left.append(bolt_factor[n_i])

                print('{}, {:5.2f}, {:5.2f}, {:5.2f}'.format(
                    symbol, delta_eos, bolt_factor[n_i], delta_eos_bolt_factor), file=f)
                n_i += 1
                

                # mu_v_tot  = mu_v_tot  / np.sum(bolt_left)
                # delta_tot = delta_tot / np.sum(bolt_left)
                # b_tot = b_tot / np.sum(bolt_left)
    return(delta_tot, mu_v_tot, b_tot)

def delta_eos(self, dataset: json, bolt_factor: list):
    '''
    Calculate delta between AIMS and DFTB+
    '''

    for name in ['volume', 'B', 'Bp']:
        with open(name + '.csv', 'w') as f:
            print('# symbol, aims, dftb', file=f)
            for symbol, dct in dataset.items():
                values = [dct[code + '_' + name]
                            for code in ['aims', 'dftb']]
                if name == 'B':
                    values = [val * 1e24 / kJ for val in values]
                print('{},'.format(symbol), ', '.join(
                    '{:.2f}'.format(value) for value in values), file=f)

    # calculate delta
    delta_tot = 0
    n_i = 0  # for multiply boltzmann factor
    with open('delta.csv', 'w') as f:
        print('# symbol, aims-dftb', 'bolt_factor', 'delta * bolt', file=f)
        for symbol, dct in dataset.items():
            # Get v0, B, Bp:
            aims, dftb = [(dct[code + '_volume'],
                            dct[code + '_B'],
                            dct[code + '_Bp'])
                            for code in ['aims', 'dftb']]

            delta_val = delta(*aims, *dftb) * 1000
            if math.isnan(delta_val):  # To avoid unexpected parameters/configurations
                delta_val = 500
            delta_val_bolt_factor = delta_val * bolt_factor[n_i]
            n_i += 1
            delta_tot += delta_val_bolt_factor
            print('{}, {:5.2f}, {:5.2f}, {:5.2f}'.format(
                symbol, delta_val, bolt_factor[n_i-1], delta_val_bolt_factor), file=f)
    return(delta_tot)



def get_kpt_position(self, spe_kpts, kpts):
    """
    Get the position of the special points in the path.

    Parameters:
    spe_kpts: dict of special points
        eg. {'G': [0.0, 0.0, 0.0], 'K': [0.5, 0.5, 0.0], 'L': [0.5, 0.0, 0.5], 'U': [0.0, 0.5, 0.5], 'W': [0.5, 0.5, 0.5], 'X': [0.5, 0.5, 0.5]}
    kpts: ndarray of kpoints, shape=(nkpt, 3)

    Returns:
    kpt_pos: dict of kpoint indexes by special points
        eg. {'G': 40, 'K': 80, 'L': 60, 'W': 0, 'X': 20}
    """
    kpt_pos = {}
    for special_point in spe_kpts.keys():
        # print(special_point)
        for num, kpt in enumerate(kpts):
            if list(np.around(spe_kpts[special_point],6)) == list(np.around(kpt,6)):
                # print(spe_kpts[special_point], kpt)
                kpt_pos[special_point] = num
                break
    return kpt_pos



def delta_band(self, r_wave, r_dens, dft_folder, dft101, bolt_factor):
    '''
    Calculate delta between AIMS and DFTB+

    Parameters:
    r_wave, r_dens: the parameters of wavefunction and density
        Only used for label savefile here

    dft_directory: directory of DFT
        eg. '../../'

    dft101: dft six configurations, equivalent states, sorted by coordination.

    bolt_factor: list of boltzmann factor
        eg. [0.0, 0.0, 0.0, 0.0,0.86,  0.13]
    
    '''
    # odft101 = ase.io.read(dft_directory + '101.xyz',':')
    # dft101 = [odft101[0], odft101[5], odft101[4], odft101[3], odft101[2], odft101[1]]

    elements = np.unique(dft101[0].numbers)
    elem_syms = [chemical_symbols[i] for i in elements] # get the element symbols ['Si']

    # bolt_factor = {'dimer':0.0, 'AA-graphite':0.0, 'diamond': 0.0, 'beta-Sn': 0.0, 'bcc':0.86, 'fcc': 0.13}

    tot_ele = self.atoms_info[elem_syms[0]]['atomic_number']
    _, _, _, outer_ele = self.get_orbit(self.atoms_info[elem_syms[0]]['valence_shell']) # outer_ele is the num of valence electron
    inner_ele = int(tot_ele - outer_ele)
    print("inner_ele:", inner_ele, "outer_ele:", outer_ele)


    dft_info = {'dimer':{'special_points':'', 'energies':'', 'bolt_factor':''},   'AA-graphite':{'special_points':'', 'energies':'', 'bolt_factor':''},
                'diamond':{'special_points':'', 'energies':'', 'bolt_factor':''}, 'beta-Sn':{'special_points':'', 'energies':'', 'bolt_factor':''},
                'bcc':{'special_points':'', 'energies':'', 'bolt_factor':''}, 'fcc':{'special_points':'', 'energies':'', 'bolt_factor':''}}
    dftb_info = {'dimer':{'special_points':'', 'energies':'', 'bolt_factor':''},   'AA-graphite':{'special_points':'', 'energies':'', 'bolt_factor':''},
                'diamond':{'special_points':'', 'energies':'', 'bolt_factor':''}, 'beta-Sn':{'special_points':'', 'energies':'', 'bolt_factor':''},
                'bcc':{'special_points':'', 'energies':'', 'bolt_factor':''}, 'fcc':{'special_points':'', 'energies':'', 'bolt_factor':''}}

    kband_info={}
    delta_band = []
    delta_homos, delta_lumos = [], []
    delta_fermi_homos, delta_fermi_lumos = [], []
    bolt_left = []


    for n, mole in enumerate(dft101):
        config = mole.info['structure_name']
        print("\n#===========Delta band structure for {}===========#".format(config))
        print(config, "\nbolt_factor:", bolt_factor[n])
        # if bolt_factor > 0.1
        if bolt_factor[n] < 0.1:
            delta_homos.append(0.0)
            delta_lumos.append(0.0)
            delta_fermi_homos.append(0.0)
            delta_fermi_lumos.append(0.0)
            continue
        else:
            num_atom = mole.get_global_number_of_atoms()
            dft_band_energies = np.empty(shape=(0, ceil(outer_ele/2) * num_atom))
            dftb_band_energies = np.empty(shape=(0, ceil(outer_ele/2) * num_atom))

        # ============================================================================ #
        # # 1. read information

            # do a check for dft_json, in order to keep fermi level always loacated max(homo) for insulator and semi-conductor
            dft_json = read_json(dft_folder / str(config+'.json'))
            dft_json_check = dft_json.energies
            check_homo_e = dft_json_check[0, :, ceil(inner_ele/2)*num_atom + ceil(outer_ele/2)*num_atom - 1]
            check_lumo_e = dft_json_check[0, :, ceil(inner_ele/2)*num_atom + ceil(outer_ele/2)*num_atom]
            dft_bandgap = min(check_lumo_e) - max(check_homo_e) if min(check_lumo_e) - max(check_homo_e) > 0.0 else 0.0
            if dft_bandgap > 0.0: # insulator or semi-conductor
                dft_json._reference = max(check_homo_e)
            

            dft_json_energies = dft_json.subtract_reference().energies
            dftb_json = read_json(config  +'_' + str(r_wave) +'_' +  str(r_dens) + '.json')
            dftb_json_energies = dftb_json.subtract_reference().energies

            
            # print("Dftb_fermi(after correction):", 0.0)
            # lumo_e = dftb_json_energies[0, :, ceil((outer_ele*num_atom)/2)].min()
            # homo_e = dftb_json_energies[0, :, ceil((outer_ele*num_atom)/2) - 1].max()
            # print("lumo_e={:.4f}\nhomo_e={:.4f}\nDFTB_gap:{}".format(lumo_e, homo_e, lumo_e - homo_e))
            # if round(lumo_e + homo_e,4) == 0.0:
            #     print("WARNING: The Fermi level of DFTB is not corrected, but I corrected it!")
            #     fermi_update = dftb_json.reference + homo_e
            #     dftb_json._reference = fermi_update
            #     dftb_json_energies = dftb_json.subtract_reference().energies

        # ============================================================================ #
        # # 2. get special points and index

            dft_spe_kpts, dft_kpts = dft_json.path.special_points, dft_json.path.kpts
            dftb_spe_kpts, dftb_kpts = dftb_json.path.special_points, dftb_json.path.kpts
            print("\nDFT_SPECIAL_KPTS:")
            pp.pprint(dft_spe_kpts)
            print("\nDFTB_SPECIAL_KPTS:")
            pp.pprint(dftb_spe_kpts)


            dft_kpt_idx = self.get_kpt_position(dft_spe_kpts, dft_kpts)
            dftb_kpt_idx = self.get_kpt_position(dftb_spe_kpts, dftb_kpts)
            # print("\nThe Index:", dft_kpt_idx, dftb_kpt_idx)
            

        # ============================================================================ #
        # # 3. get energies
            
            dft_homo_e = dft_json_energies[0, :, ceil(inner_ele/2)*num_atom + ceil(outer_ele/2)*num_atom - 1]
            dftb_homo_e = dftb_json_energies[0, :, ceil(outer_ele/2)*num_atom - 1]
            dft_homo_rel_e = dft_homo_e - max(dft_homo_e)
            dftb_homo_rel_e = dftb_homo_e - max(dftb_homo_e)


            dft_lumo_e = dft_json_energies[0, :, ceil(inner_ele/2)*num_atom + ceil(outer_ele/2)*num_atom]
            dftb_lumo_e = dftb_json_energies[0, :, ceil(outer_ele/2)*num_atom]
            dft_lumo_rel_e = dft_lumo_e - min(dft_lumo_e)
            dftb_lumo_rel_e = dftb_lumo_e - min(dftb_lumo_e)


            # save relative energies < +-1 eV to a new list
            dft_homo_ee, dftb_homo_ee = [], []
            dft_lumo_ee, dftb_lumo_ee = [], []
            dft_fermi_homo_ee, dftb_fermi_homo_ee = [], []
            dft_fermi_lumo_ee, dftb_fermi_lumo_ee = [], []


            for nn, ee in enumerate(dft_homo_e):
                if abs(ee) <= 1.0:
                    dft_fermi_homo_ee.append(ee)
                    dftb_fermi_homo_ee.append(dftb_homo_e[nn])
            
            for nn, ee in enumerate(dft_lumo_e):
                if abs(ee) <= 1.0:
                    dft_fermi_lumo_ee.append(ee)
                    dftb_fermi_lumo_ee.append(dftb_homo_e[nn])
                else:
                    dft_fermi_lumo_ee.append(0.0)
                    dftb_fermi_lumo_ee.append(0.0)


            for nn, ee in enumerate(dft_homo_rel_e):
                if abs(ee) <= 1.0:
                    dft_homo_ee.append(ee)
                    dftb_homo_ee.append(dftb_homo_rel_e[nn])
            
            for nn, ee in enumerate(dft_lumo_rel_e):
                if abs(ee) <= 1.0:
                    dft_lumo_ee.append(ee)
                    dftb_lumo_ee.append(dftb_lumo_rel_e[nn])


    # ============================================================================ #
    # #     # Plot band structure
            emin, emax = -20, 20
            fig, ax = plt.subplots(1,1, dpi=150)
            fig.suptitle("Parameters: r_wave={}, r_dens={}".format(r_wave, r_dens))
            ax.set_ylabel('Energies (eV)', fontsize=13)
            dftb_json.subtract_reference().plot(ax=ax, emin=emin, emax=emax, filename=config+'_' + str(r_wave) +'_' +  str(r_dens) + '.png', colors=['g'])
            dft_json.subtract_reference().plot(ax=ax, emin=emin, emax=emax, filename=config+'_' + str(r_wave) +'_' +  str(r_dens) + '.png', colors=['r'])
            fig.clf()                

            # for Gamma in dft_kpt_idx.keys():
            #     dft_e = dft_json_energies[0, dft_kpt_idx[Gamma], ceil(inner_ele/2)*num_atom: ceil(inner_ele/2)*num_atom + ceil(outer_ele/2)*num_atom]
            #     dftb_e = dftb_json_energies[0, dftb_kpt_idx[Gamma], 0: ceil(outer_ele/2)*num_atom]
            
            #     dft_band_energies = np.vstack((dft_band_energies, dft_e))
            #     dftb_band_energies = np.vstack((dftb_band_energies, dftb_e))
            # # dft_homo_e = dft_e[ceil(outer_ele * num_atom / 2) - 1 ,:]
            # # dftb_homo_e = dftb_e[ceil(outer_ele * num_atom / 2) - 1 ,:]
            # dft_homo_e = dft_band_energies[:,-1] 
            # dftb_homo_e = dftb_band_energies[:,-1]
            # dft_homo_e_rel = dft_homo_e - max(dft_homo_e)  # relative to the highest of homo
            # dftb_homo_e_rel = dftb_homo_e - max(dftb_homo_e)

            # # constrain the loss within 1 eV
            # dft_ee, dftb_ee = [], []
            # for nn, ee in enumerate(dft_homo_e_rel):
            #     if abs(ee) <=1.0:
            #         dft_ee.append(ee)
            #         dftb_ee.append(dftb_homo_e_rel[nn])

            # dft_homo_e_rel = np.array(dft_ee)
            # dftb_homo_e_rel = np.array(dftb_ee)
        # ============================================================================ #
        # # 4. get band gap
            # bg_dft = dft_json_energies[0, :, ceil(inner_ele/2)*num_atom + ceil(outer_ele/2)*num_atom + 1].min() - dft_json_energies[0, :, ceil(inner_ele/2)*num_atom + ceil(outer_ele/2)*num_atom].max()
            # bg_dftb = dftb_json_energies[0, :, ceil(outer_ele/2 + 1)*num_atom].min() - dftb_json_energies[0, :, ceil(outer_ele/2)*num_atom].max()
            # print("DFT band gap:", bg_dft, "\nDFTB band gap:", bg_dftb)


        # ============================================================================ #
        # # 4. save information

            # assert dft_kpt_idx.keys() == dftb_kpt_idx.keys()
            dft_info[config]['special_points'] = ' '.join(dft_kpt_idx.keys())
            dft_info[config]['energies'] = dft_band_energies
            dft_info[config]['bolt_factor'] = bolt_factor[n]

            dftb_info[config]['special_points'] = ' '.join(dftb_kpt_idx.keys())
            dftb_info[config]['energies'] = dftb_band_energies
            dftb_info[config]['bolt_factor'] = bolt_factor[n]

            delta_energies = np.mean(np.abs(np.array(dft_band_energies) - np.array(dftb_band_energies))) * bolt_factor[n]

            delta_homo = np.mean(np.abs(np.array(dft_homo_ee).flatten() - np.array(dftb_homo_ee).flatten())) * bolt_factor[n]
            delta_lumo = np.mean(np.abs(np.array(dft_lumo_ee).flatten() - np.array(dftb_lumo_ee).flatten())) * bolt_factor[n]

            delta_fermi_homo = np.mean(np.abs(np.array(dft_fermi_homo_ee).flatten() - np.array(dftb_fermi_homo_ee).flatten())) * bolt_factor[n]
            delta_fermi_lumo = np.mean(np.abs(np.array(dft_fermi_lumo_ee).flatten() - np.array(dftb_fermi_lumo_ee).flatten())) * bolt_factor[n]
            
            # delta_band.append(delta_energies)
            delta_homos.append(delta_homo)
            delta_lumos.append(delta_lumo)

            delta_fermi_homos.append(delta_fermi_homo)
            delta_fermi_lumos.append(delta_fermi_lumo)

            bolt_left.append(bolt_factor[n])

        # bs_dft = read_json(dft_folder / str(configuration + '.json'))
        # ============================================================================ #
        # # 5. plot band structure
            # emin, emax = -20, 20
            # fig, ax = plt.subplots(1,1, dpi=150)
            # fig.suptitle("Parameters: r_wave={}, r_dens={}".format(r_wave, r_dens))
            # ax.set_ylabel('Energies (eV)', fontsize=13)
            # bs.subtract_reference().plot(ax=ax, emin=emin, emax=emax, filename=configuration+'_' + str(r_wave) +'_' +  str(r_dens) + '.png', colors=['g'])
            # bs_dft.subtract_reference().plot(ax=ax, emin=emin, emax=emax, filename=configuration+'_' + str(r_wave) +'_' +  str(r_dens) + '.png', colors=['r'])
            # fig.clf()

    kband_info = {'dft':dft_info, 'dftb':dftb_info, 'delta':np.mean(delta_homos)}
    # mu_band = np.sum(delta_band) / np.sum(bolt_left)
    # delta_homos = np.sum(delta_homo) / np.sum(bolt_left)
    # mu_band_homo = np.sum(delta_homos) / np.sum(bolt_left)  # homo_rel
    # mu_band_lumo = np.sum(delta_lumos) / np.sum(bolt_left)    # lumo_rel

    mu_band_homo = np.sum(delta_fermi_homos) / np.sum(bolt_left)
    mu_band_lumo = np.sum(delta_fermi_lumos) / np.sum(bolt_left)

    with open('kband_info' + '_' + str(r_wave) +'_' +  str(r_dens) + '.pkl', 'wb') as f:
        pickle.dump(kband_info, f)
    return kband_info, delta_fermi_homos, delta_fermi_lumos, mu_band_homo, mu_band_lumo


# def delta_band_binary(self, para_list, json_file, atoms_list, bolt_factor):

def delta_band_binary(self, conf_parameters, json_file, dft_atoms, bolt_f):
    '''
    Calculate delta between AIMS and DFTB+

    Parameters:
    r_wave, r_dens: the parameters of wavefunction and density
        Only used for label savefile here

    dft_directory: directory of DFT
        eg. '../../'

    dft101: dft six configurations, equivalent states, sorted by coordination.

    bolt_factor: list of boltzmann factor weight
        eg. [0.0, 0.0, 0.0, 0.0,0.86,  0.13]
    
    '''

    # dft_info = {'DFT':{'special_points':'', 'energies':'', 'bolt_factor':''},   'AA-graphite':{'special_points':'', 'energies':'', 'bolt_factor':''}}
    # dftb_info = {'DFTB':{'special_points':'', 'energies':'', 'bolt_factor':''},   'AA-graphite':{'special_points':'', 'energies':'', 'bolt_factor':''}}

    # kband_info={}
    # delta_band = []
    delta_homos, delta_lumos = [], []
    delta_fermi_homos, delta_fermi_lumos = [], []
    bolt_left = []


    for n, mole in enumerate(dft_atoms):
        config = mole.info['structure_name']
        print("\n#===========Delta band structure for {}===========#".format(config))
        print(config, "\nbolt_factor:", bolt_f[n])

        if bolt_f[n] < 0.1:
            delta_homos.append(0.0)
            delta_lumos.append(0.0)
            delta_fermi_homos.append(0.0)
            delta_fermi_lumos.append(0.0)
            continue


        else:
            # Get the valence electron info for DFTB and DFT
            dftb_tot_chemical_symbols = mole.get_chemical_symbols()
            dftb_num_valence_electron = np.sum(self.get_orbit(self.atoms_info[syms]['valence_shell'])[3] for syms in dftb_tot_chemical_symbols)


            dft_tot_chemical_symbols = mole.get_chemical_symbols()
            dft_num_valence_electron = np.sum(self.get_orbit(self.atoms_info_qe[syms]['valence_shell'])[3] for syms in dft_tot_chemical_symbols)

            # dft_band_energies = np.empty(shape=(0, ceil(outer_ele/2) * num_binary))
            # dft_band_energies = np.empty(shape=(0, ceil(dft_num_valence_electron/2)))


            # dftb_band_energies = np.empty(shape=(0, ceil(dftb_num_valence_electron/2)))

        # ============================================================================ #
        # # 1. read information

            # do a check for dft_json, in order to keep fermi level always loacated max(homo) for insulator and semi-conductor
            dft_json = read_json(json_file)
            dft_json_check = dft_json.energies

            # outer_ele_dft, virtual_ele_dft = ceil(outer_ele/2) * num_binary , 4
            # outer_ele_dft, virtual_ele_dft = ceil(outer_ele_qe/2) * num_binary , 4

            # check_homo_e = dft_json_check[0, :, ceil(inner_ele_dft/2)*num_binary + ceil(outer_ele_dft/2)*num_binary - 1]
            # check_lumo_e = dft_json_check[0, :, ceil(inner_ele_dft/2)*num_binary + ceil(outer_ele_dft/2)*num_binary]

            # Make sure find the right band
            dft_lumo_e_check = dft_json_check[0, :, ceil(dft_num_valence_electron/2)]
            dft_homo_e_check = dft_json_check[0, :, ceil(dft_num_valence_electron/2) - 1]


            dft_bandgap = min(dft_lumo_e_check) - max(dft_homo_e_check) if min(dft_lumo_e_check) - max(dft_homo_e_check) > 0.0 else 0.0
            if dft_bandgap > 0.0: # insulator or semi-conductor
                dft_json._reference = max(dft_homo_e_check)

            # After correction
            dft_json_energies = dft_json.subtract_reference().energies
            dft_lumo_e = dft_json_energies[0, :, ceil(dft_num_valence_electron/2)]
            dft_homo_e = dft_json_energies[0, :, ceil(dft_num_valence_electron/2) - 1]


            # Read DFTB JSON
            para_name = '_'.join([str(round(i,3)) for i in conf_parameters])
            dftb_json = read_json(config  + '_' + para_name  + '.json')
            dftb_json_energies = dftb_json.subtract_reference().energies

        # ============================================================================ #
        # # 2. get special points and index

            dft_spe_kpts, dft_kpts = dft_json.path.special_points, dft_json.path.kpts
            dftb_spe_kpts, dftb_kpts = dftb_json.path.special_points, dftb_json.path.kpts
            print("\nDFT_SPECIAL_KPTS:")
            pp.pprint(dft_spe_kpts)
            print("\nDFTB_SPECIAL_KPTS:")
            pp.pprint(dftb_spe_kpts)
            assert dft_spe_kpts.keys() == dftb_spe_kpts.keys(), "The special points are not the same for DFT and DFTB"

            # dft_kpt_idx = self.get_kpt_position(dft_spe_kpts, dft_kpts)
            # dftb_kpt_idx = self.get_kpt_position(dftb_spe_kpts, dftb_kpts)
            # assert dft_kpt_idx == dftb_kpt_idx, "The special points are not the same for DFT and DFTB"
            # print("\nThe Index:", dft_kpt_idx, dftb_kpt_idx)
            

        # ============================================================================ #
        # # 3. get energies
            
            # dft_homo_e = dft_json_energies[0, :, ceil(inner_ele/2)*num_atom + ceil(outer_ele/2)*num_atom - 1]
            dftb_lumo_e = dftb_json_energies[0, :, ceil(dftb_num_valence_electron/2)]
            dftb_homo_e = dftb_json_energies[0, :, ceil(dftb_num_valence_electron/2) - 1]

            # Plot only homo and lumo
            emin, emax = -10, 10
            fig, ax = plt.subplots(1,1, dpi=80)
            fig.suptitle("Parameters:" + str(para_name))
            xx = np.linspace(0,1,len(dft_lumo_e))
            ax.plot(xx, dft_lumo_e, color='r', linestyle='-', label='DFT')
            ax.plot(xx, dft_homo_e, color='r', linestyle='--')
            ax.plot(xx, dftb_lumo_e, color='g', linestyle='-', label='DFTB')
            ax.plot(xx, dftb_homo_e, color='g', linestyle='--')

            ax.set_ylim(emin, emax)
            ax.legend()
            fig.tight_layout()
            fig.savefig('pic_' + para_name + '.png')
            fig.clf()


            # save relative energies < +-1 eV to a new list
            dft_fermi_homo_ee, dftb_fermi_homo_ee = [], []
            dft_fermi_lumo_ee, dftb_fermi_lumo_ee = [], []

            for nn, ee in enumerate(dft_homo_e):
                if abs(ee) <= 1.0:
                    dft_fermi_homo_ee.append(ee)
                    dftb_fermi_homo_ee.append(dftb_homo_e[nn])

            
            for nn, ee in enumerate(dft_lumo_e):
                if abs(ee) <= 1.0:
                    dft_fermi_lumo_ee.append(ee)
                    dftb_fermi_lumo_ee.append(dftb_lumo_e[nn])
                # else:
                #     dft_fermi_lumo_ee.append(0.0)
                #     dftb_fermi_lumo_ee.append(0.0)

            delta_fermi_homo = np.mean(np.abs(np.array(dft_fermi_homo_ee).flatten() - np.array(dftb_fermi_homo_ee).flatten())) * bolt_f[n]
            delta_fermi_lumo = np.mean(np.abs(np.array(dft_fermi_lumo_ee).flatten() - np.array(dftb_fermi_lumo_ee).flatten())) * bolt_f[n]

            delta_fermi_homos.append(delta_fermi_homo)
            delta_fermi_lumos.append(delta_fermi_lumo)

            bolt_left.append(bolt_f[n])


            # Not really use here
            # dft_homo_ee, dftb_homo_ee = [], []
            # dft_lumo_ee, dftb_lumo_ee = [], []
            # for nn, ee in enumerate(dft_homo_rel_e):
            #     if abs(ee) <= 1.0:
            #         dft_homo_ee.append(ee)
            #         dftb_homo_ee.append(dftb_homo_rel_e[nn])
            
            # for nn, ee in enumerate(dft_lumo_rel_e):
            #     if abs(ee) <= 1.0:
            #         dft_lumo_ee.append(ee)
            #         dftb_lumo_ee.append(dftb_lumo_rel_e[nn])

            # delta_homo = np.mean(np.abs(np.array(dft_homo_ee).flatten() - np.array(dftb_homo_ee).flatten())) * bolt_factor[n]
            # delta_lumo = np.mean(np.abs(np.array(dft_lumo_ee).flatten() - np.array(dftb_lumo_ee).flatten())) * bolt_factor[n]



        # ============================================================================ #
        # # 4. save information

            # assert dft_kpt_idx.keys() == dftb_kpt_idx.keys()
            # dft_info['DFT']['special_points'] = ' '.join(dft_kpt_idx.keys())
            # dft_info['DFT']['energies'] = dft_band_energies
            # dft_info['DFT']['bolt_factor'] = bolt_factor[n]

            # dftb_info['DFTB']['special_points'] = ' '.join(dftb_kpt_idx.keys())
            # dftb_info['DFTB']['energies'] = dftb_band_energies
            # dftb_info['DFTB']['bolt_factor'] = bolt_factor[n]

            # delta_energies = np.mean(np.abs(np.array(dft_band_energies) - np.array(dftb_band_energies))) * bolt_factor[n]

            
            # delta_band.append(delta_energies)
            # delta_homos.append(delta_homo)
            # delta_lumos.append(delta_lumo)



    # kband_info = {'dft':dft_info, 'dftb':dftb_info, 'delta':np.mean(delta_homos)}
    # mu_band = np.sum(delta_band) / np.sum(bolt_left)
    # delta_homos = np.sum(delta_homo) / np.sum(bolt_left)
    # mu_band_homo = np.sum(delta_homos) / np.sum(bolt_left)  # homo_rel
    # mu_band_lumo = np.sum(delta_lumos) / np.sum(bolt_left)    # lumo_rel

    mu_band_homo = np.sum(delta_fermi_homos) / np.sum(bolt_left)
    mu_band_lumo = np.sum(delta_fermi_lumos) / np.sum(bolt_left)

    # with open('kband_info' + '_' + str(round(r_wave_zn,3)) +'_' +  str(round(r_dens_zn,3)) +'_'+ str(round(r_wave_o,3)) +'_' +  str(round(r_dens_o,3))  + '.pkl', 'wb') as f:
    #     pickle.dump(kband_info, f)
    return mu_band_homo, mu_band_lumo


def get_curvature(self, r_wave, r_dens, dft_folder, dft101, bolt_factor):
    '''
    Calculate delta between AIMS and DFTB+

    Parameters:
    r_wave, r_dens: the parameters of wavefunction and density
        Only used for label savefile here

    dft_directory: directory of DFT
        eg. '../../'

    dft101: dft six configurations, equivalent states, sorted by coordination.

    bolt_factor: list of boltzmann factor
        eg. [0.0, 0.0, 0.0, 0.0,0.86,  0.13]
    
    '''
    # odft101 = ase.io.read(dft_directory + '101.xyz',':')
    # dft101 = [odft101[0], odft101[5], odft101[4], odft101[3], odft101[2], odft101[1]]

    elements = np.unique(dft101[0].numbers)
    elem_syms = [chemical_symbols[i] for i in elements] # get the element symbols ['Si']

    # bolt_factor = {'dimer':0.0, 'AA-graphite':0.0, 'diamond': 0.0, 'beta-Sn': 0.0, 'bcc':0.86, 'fcc': 0.13}

    tot_ele = self.atoms_info[elem_syms[0]]['atomic_number']
    _, _, _, outer_ele = self.get_orbit(self.atoms_info[elem_syms[0]]['valence_shell']) # outer_ele is the num of valence electron
    inner_ele = int(tot_ele - outer_ele)
    print("inner_ele:", inner_ele, "outer_ele:", outer_ele)


    dft_info = {'dimer':{'special_points':'', 'energies':'', 'bolt_factor':''},   'AA-graphite':{'special_points':'', 'energies':'', 'bolt_factor':''},
                'diamond':{'special_points':'', 'energies':'', 'bolt_factor':''}, 'beta-Sn':{'special_points':'', 'energies':'', 'bolt_factor':''},
                'bcc':{'special_points':'', 'energies':'', 'bolt_factor':''}, 'fcc':{'special_points':'', 'energies':'', 'bolt_factor':''}}
    dftb_info = {'dimer':{'special_points':'', 'energies':'', 'bolt_factor':''},   'AA-graphite':{'special_points':'', 'energies':'', 'bolt_factor':''},
                'diamond':{'special_points':'', 'energies':'', 'bolt_factor':''}, 'beta-Sn':{'special_points':'', 'energies':'', 'bolt_factor':''},
                'bcc':{'special_points':'', 'energies':'', 'bolt_factor':''}, 'fcc':{'special_points':'', 'energies':'', 'bolt_factor':''}}

    kband_info={}
    delta_band = []
    delta_homos = []
    bolt_left = []
    delta_curvatures = []


    for n, mole in enumerate(dft101):
        config = mole.info['structure_name']
        print("\n#===========Delta band structure for {}===========#".format(config))
        print(config, "\nbolt_factor:", bolt_factor[n])
        # if bolt_factor > 0.1
        if bolt_factor[n] < 0.1:
            delta_homos.append(0.0)
            continue
        else:
            num_atom = mole.get_global_number_of_atoms()
            dft_band_energies = np.empty(shape=(0, ceil(outer_ele/2) * num_atom))
            dftb_band_energies = np.empty(shape=(0, ceil(outer_ele/2) * num_atom))

        # ============================================================================ #
        # # 1. read information

            dft_json = read_json(dft_folder / str(config+'.json'))
            dftb_json = read_json('dat/' + config  +'_' + str(r_wave) +'_' +  str(r_dens) + '.json')
            dft_json_energies = dft_json.subtract_reference().energies
            dftb_json_energies = dftb_json.subtract_reference().energies
            
            print("Dftb_fermi(after correction):", 0.0)
            lumo_e = dftb_json_energies[0, :, ceil((outer_ele*num_atom)/2)].min()
            homo_e = dftb_json_energies[0, :, ceil((outer_ele*num_atom)/2) - 1].max()
            print("lumo_e={:.4f}\nhomo_e={:.4f}\nDFTB_gap:{}".format(lumo_e, homo_e, lumo_e - homo_e))
            if round(lumo_e + homo_e,4) == 0.0:
                print("WARNING: The Fermi level of DFTB is not corrected, but I corrected it!")
                fermi_update = dftb_json.reference + homo_e
                dftb_json._reference = fermi_update
                dftb_json_energies = dftb_json.subtract_reference().energies

        # ============================================================================ #
        # # 2. get special points and index

            dft_spe_kpts, dft_kpts = dft_json.path.special_points, dft_json.path.kpts
            dftb_spe_kpts, dftb_kpts = dftb_json.path.special_points, dftb_json.path.kpts
            print("\nDFT_SPECIAL_KPTS:")
            pp.pprint(dft_spe_kpts)
            print("\nDFTB_SPECIAL_KPTS:")
            pp.pprint(dftb_spe_kpts)


            dft_kpt_idx = self.get_kpt_position(dft_spe_kpts, dft_kpts)
            dftb_kpt_idx = self.get_kpt_position(dftb_spe_kpts, dftb_kpts)
            # print("\nThe Index:", dft_kpt_idx, dftb_kpt_idx)
            

        # ============================================================================ #
        # # 3. get energies
            delta_curva_kpts = []
            for Gamma in dft_kpt_idx.keys():
                dft_e = dft_json_energies[0, dft_kpt_idx[Gamma], ceil(inner_ele/2)*num_atom: ceil(inner_ele/2)*num_atom+ceil(outer_ele/2)*num_atom]
                dftb_e = dftb_json_energies[0, dftb_kpt_idx[Gamma], 0: ceil(outer_ele/2)*num_atom]
            
                dft_band_energies = np.vstack((dft_band_energies, dft_e))
                dftb_band_energies = np.vstack((dftb_band_energies, dftb_e))



                # get curvature
                dftb_homo_e = dftb_json_energies[0, :, ceil((outer_ele*num_atom)/2) - 1]
                dft_homo_e = dft_json_energies[0, :, ceil(inner_ele/2)*num_atom + ceil((outer_ele*num_atom)/2) - 1]
                # collect two points around kpt_idx to calculate the curvature
                if dftb_kpt_idx[Gamma] == 0:
                    assert dft_kpt_idx[Gamma] == 0, "The end kpt_idx of DFT and DFTB are not the same!"
                    dftb_cpoints = [dftb_homo_e[dftb_kpt_idx[Gamma]], dftb_homo_e[dftb_kpt_idx[Gamma] + 3], dftb_homo_e[dftb_kpt_idx[Gamma]+6]]
                    dftb_curva = ((dftb_cpoints[2] - dftb_cpoints[1]) - (dftb_cpoints[1] - dftb_cpoints[0]))/2

                    dft_cpoints = [dft_homo_e[dft_kpt_idx[Gamma]], dft_homo_e[dft_kpt_idx[Gamma]+2], dft_homo_e[dft_kpt_idx[Gamma]+4]]
                    dft_curva = ((dft_cpoints[2] - dft_cpoints[1]) - (dft_cpoints[1] - dft_cpoints[0]))/2

                    if np.sign(dftb_curva) == np.sign(dft_curva):
                        delta_curva = abs(dft_curva - dftb_curva)
                    else:
                        delta_curva = abs(dft_curva - dftb_curva) * 10
                    delta_curva_kpts.append(delta_curva)
                    print("Gamma:{:10}dft_curva:{:12.5f}dftb_curva:{:12.5f}dftb_curva:{:12.5f}".format(Gamma, dft_curva, dftb_curva, delta_curva))

                elif dftb_kpt_idx[Gamma] == len(dftb_kpts) - 1:
                    assert dft_kpt_idx[Gamma] == len(dft_kpts) - 1, "The end kpt_idx of DFT and DFTB are not the same!"
                    dftb_cpoints = [dftb_homo_e[dftb_kpt_idx[Gamma]-6], dftb_homo_e[dftb_kpt_idx[Gamma]-3], dftb_homo_e[dftb_kpt_idx[Gamma]]]
                    dftb_curva = ((dftb_cpoints[2] - dftb_cpoints[1]) - (dftb_cpoints[1] - dftb_cpoints[0]))/2

                    dft_cpoints = [dft_homo_e[dft_kpt_idx[Gamma]-4], dft_homo_e[dft_kpt_idx[Gamma]-2], dft_homo_e[dft_kpt_idx[Gamma]]]
                    dft_curva = ((dft_cpoints[2] - dft_cpoints[1]) - (dft_cpoints[1] - dft_cpoints[0]))/2

                    if np.sign(dftb_curva) == np.sign(dft_curva):
                        delta_curva = abs(dft_curva - dftb_curva)
                    else:
                        delta_curva = abs(dft_curva - dftb_curva) * 10
                    delta_curva_kpts.append(delta_curva)
                    print("Gamma:{:10}dft_curva:{:12.5f}dftb_curva:{:12.5f}dftb_curva:{:12.5f}".format(Gamma, dft_curva, dftb_curva, delta_curva))


                else:                
                    dftb_cpoints = [dftb_homo_e[dftb_kpt_idx[Gamma]-10], dftb_homo_e[dftb_kpt_idx[Gamma]], dftb_homo_e[dftb_kpt_idx[Gamma]+10]]
                    dftb_curva = ((dftb_cpoints[2] - dftb_cpoints[1]) - (dftb_cpoints[1] - dftb_cpoints[0]))/2

                    dft_cpoints = [dft_homo_e[dft_kpt_idx[Gamma]-3], dft_homo_e[dft_kpt_idx[Gamma]], dft_homo_e[dft_kpt_idx[Gamma]+3]]
                    dft_curva = ((dft_cpoints[2] - dft_cpoints[1]) - (dft_cpoints[1] - dft_cpoints[0]))/2

                    if np.sign(dftb_curva) == np.sign(dft_curva):
                        delta_curva = abs(dft_curva - dftb_curva)
                    else:
                        delta_curva = abs(dft_curva - dftb_curva) * 10
                    delta_curva_kpts.append(delta_curva)
                    print("Gamma:{:10}dft_curva:{:12.5f}dftb_curva:{:12.5f}dftb_curva:{:12.5f}".format(Gamma, dft_curva, dftb_curva, delta_curva))


            # dft_homo_e = dft_e[ceil(outer_ele * num_atom / 2) - 1 ,:]
            # dftb_homo_e = dftb_e[ceil(outer_ele * num_atom / 2) - 1 ,:]
            dft_homo_e = dft_band_energies[:,-1] 
            dftb_homo_e = dftb_band_energies[:,-1]
            dft_homo_e_rel = dft_homo_e - max(dft_homo_e)
            dftb_homo_e_rel = dftb_homo_e - max(dftb_homo_e)
        # ============================================================================ #
        # # 4. get band gap
            # bg_dft = dft_json_energies[0, :, ceil(inner_ele/2)*num_atom + ceil(outer_ele/2)*num_atom + 1].min() - dft_json_energies[0, :, ceil(inner_ele/2)*num_atom + ceil(outer_ele/2)*num_atom].max()
            # bg_dftb = dftb_json_energies[0, :, ceil(outer_ele/2 + 1)*num_atom].min() - dftb_json_energies[0, :, ceil(outer_ele/2)*num_atom].max()
            # print("DFT band gap:", bg_dft, "\nDFTB band gap:", bg_dftb)


        # ============================================================================ #
        # # 4. save information

            assert dft_kpt_idx.keys() == dftb_kpt_idx.keys()
            dft_info[config]['special_points'] = ' '.join(dft_kpt_idx.keys())
            dft_info[config]['energies'] = dft_band_energies
            dft_info[config]['bolt_factor'] = bolt_factor[n]

            dftb_info[config]['special_points'] = ' '.join(dftb_kpt_idx.keys())
            dftb_info[config]['energies'] = dftb_band_energies
            dftb_info[config]['bolt_factor'] = bolt_factor[n]

            delta_energies = np.mean(np.abs(np.array(dft_band_energies) - np.array(dftb_band_energies))) * bolt_factor[n]

            delta_homo = np.mean(np.abs(dft_homo_e_rel - dftb_homo_e_rel)) * bolt_factor[n]
            delta_curva_config = np.mean(delta_curva_kpts) * bolt_factor[n] # this config
            
            delta_band.append(delta_energies)
            delta_homos.append(delta_homo)
            bolt_left.append(bolt_factor[n])
            delta_curvatures.append(delta_curva_config)

        # bs_dft = read_json(dft_folder / str(configuration + '.json'))
        # ============================================================================ #
        # # 5. plot band structure
            # emin, emax = -20, 20
            # fig, ax = plt.subplots(1,1, dpi=150)
            # fig.suptitle("Parameters: r_wave={}, r_dens={}".format(r_wave, r_dens))
            # ax.set_ylabel('Energies (eV)', fontsize=13)
            # bs.subtract_reference().plot(ax=ax, emin=emin, emax=emax, filename=configuration+'_' + str(r_wave) +'_' +  str(r_dens) + '.png', colors=['g'])
            # bs_dft.subtract_reference().plot(ax=ax, emin=emin, emax=emax, filename=configuration+'_' + str(r_wave) +'_' +  str(r_dens) + '.png', colors=['r'])
            # fig.clf()

    kband_info = {'dft':dft_info, 'dftb':dftb_info, 'delta':np.mean(delta_band)}
    # mu_band = np.sum(delta_band) / np.sum(bolt_left)
    # delta_homos = np.sum(delta_homo) / np.sum(bolt_left)
    mu_band = np.mean(delta_homos) / np.sum(bolt_left)
    mu_curvatures = np.sum(delta_curvatures) / np.sum(bolt_left)

    with open('kband_info' + '_' + str(r_wave) +'_' +  str(r_dens) + '.pkl', 'wb') as f:
        pickle.dump(kband_info, f)
    return kband_info, delta_band, delta_homos, mu_band, delta_curvatures, mu_curvatures
