# %%

import copy
import math
import os
import sys
from pathlib import Path

import ase
import ase.cluster as cluster
import numpy as np
from ase.atoms import Atom, Atoms
from ase.calculators.aims import Aims
from ase.calculators.dftb import Dftb
from ase.constraints import FixAtoms
from ase.optimize import BFGS, LBFGS
from numpy import pi, sqrt
from ase.calculators.dftb import Dftb
from ase.calculators.mixing import SumCalculator
from ase.calculators.mixing import MixedCalculator
from ase.md.langevin import Langevin
from ase import units
import argparse
import time

try:
    from quippy.potential import Potential
    from ase.calculators.mixing import SumCalculator
except:
    print("Import quippy unsuccessfully, proceeding running!")


from ase.calculators.calculator import (Calculator,
                                        PropertyNotImplementedError,
                                        all_changes)

from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                        Stationary, ZeroRotation)
from ase import units
from ase.io.trajectory import Trajectory

def monkhorstpack2kptdensity(atoms:Atoms, k_density:float):

    """Convert Monkhorst-Pack grid to k-point density.
    atoms (ase.atoms.Atoms): Atoms object.
    k_grid (list): [nx, ny, nz].
    Returns:
        float: Smallest line-density.

    """

    recipcell = atoms.cell.reciprocal()
    kd = [] # k-point density
    ks = [] # k-point space
    # initial a array for k_grids, with 3 columns
    # k_grids = np.empty((1,3))


    k_grid_abc = []
    for i in range(3):
        if atoms.pbc[i]:
            k_grida = int(int(k_density) * (2 * pi * sqrt((recipcell[i] ** 2).sum())))
            if k_grida == 0:
                k_grida = 1
            k_grid_abc.append(k_grida) # current k_grid

            # k_grids = np.vstack((k_grids, k_grid_abc))
            # kptdensity = k_grid[i] / (2 * pi * sqrt((recipcell[i] ** 2).sum()))

            kptspace =  sqrt((recipcell[i] ** 2).sum()) / k_grida
            ks.append(kptspace)
    return ks, k_grid_abc



# def calc_dftbplus(calc_type:str, mole:Atoms, formula:str, poly_rep:bool, SCC:bool, kpts:list, slater_p:str='./'):
def calc_dftbplus(mole:Atoms, args:dict):
    '''
    Calculation with DFTB+ using ase.calculator.dftb

    Parameters:
    ----------
    Parameters:
        calc_type: define the type of calculation
            "static", "band", "opt_geometry", "molecule"
            Attention: "opt_geometry" actually not working here
        
        mole: ase.Atoms object

        formula: str, whatever string lable

        rep_option: bool 
            Yes: Use repulsion potetnial form polynomial fit, third line in skf file, becarefull if you really have repulsion potential there
            No: Use spline interpolation to get the repulsive potential, after `Splines` in skf file

        SCC: bool
            Yes: when using SCC calculation
            No: when don't using SCC calculation

        kpts: list
            kpts = [kx, ky, kz]
        
        slater_p: str
            The directory of slator-koster file
    '''
    calc_type, poly_rep, SCC, kpts_dens, slater_p = args['calc_type'], args['poly_rep'], args['SCC'], args['kpts_dens'], args['slater_p']
    formula = mole.get_chemical_formula()

    if np.array_equal(mole.get_pbc(), [True, True, True]):
        print(f"Calculate Bulk with {calc_type} method!")
        ks, kpts = monkhorstpack2kptdensity(mole, kpts_dens)

        if calc_type.lower() == 'static':
                calc = Dftb(atoms=mole, label=str(formula), kpts=kpts,
                        Hamiltonian_='DFTB',
                        Hamiltonian_SCC=SCC,  
                        Hamiltonian_SCCTolerance=1e-5,
                        Hamiltonian_MaxSCCIterations=1000,
                        Hamiltonian_ConvergentSCCOnly = 'No',
                        Hamiltonian_ReadInitialCharges='No', # Static calculation
                        Hamiltonian_ForceEvaluation='dynamics',
                        

                        # ONLY FOR SCC CONVERGENCE
                        #Hamiltonian_Fermi_='', Hamiltonian_Fermi_Temperature=2000,
                        #Hamiltonian_MethfesselPaxton_='', Hamiltonian_MethfesselPaxton_Order=2,
                        #Parallel_='', Parallel_Groups=64,


                        # SlaterKosterFiles
                        Hamiltonian_SlaterKosterFiles_='Type2FileNames',
                        Hamiltonian_SlaterKosterFiles_Prefix=slater_p,
                        Hamiltonian_SlaterKosterFiles_Separator='"-"',
                        Hamiltonian_SlaterKosterFiles_Suffix='".skf"',
                        Hamiltonian_MaxAngularMomentum_='',
                       
                        # # For D3-dispersion
                        # Hamiltonian_Dispersion_='SimpleDftD3',
                        # # # Hamiltonian_Dispersion_SimpleDftD3_='',
                        # Hamiltonian_Dispersion_a1=0.413,
                        # Hamiltonian_Dispersion_a2=4.429,
                        # Hamiltonian_Dispersion_s6=1.0,
                        # Hamiltonian_Dispersion_s8=0.700,


                        # Repulsive Potential
                        Hamiltonian_PolynomialRepulsive=f'SetForAll {{{poly_rep}}}',


                        # Options
                        Options_='', Options_WriteResultsTag='Yes',        # Default:No
                        Options_WriteDetailedOut='Yes', # Default:


                        # For 3OB
                        #Hamiltonian_ThirdOrderFull='Yes',
                        #Hamiltonian_HubbardDerivs_='',
                        #Hamiltonian_HCorrection_='', Hamiltonian_HCorrection_Damping_='', Hamiltonian_HCorrection_Damping_Exponent=4.05,
                        ## set hubbardderivs for hubbardderivs for H, C, O, N, F,  Na, Mg, S, P, Cl, K, Ca, Zn, Br, I
                        #Hamiltonian_HubbardDerivs_H=-0.1857, Hamiltonian_HubbardDerivs_C=-0.1492,
                        #amiltonian_HubbardDerivs_O=-0.1575, Hamiltonian_HubbardDerivs_N=-0.1535,
                        #Hamiltonian_HubbardDerivs_F=-0.1623, Hamiltonian_HubbardDerivs_Na=-0.0454,
                        #Hamiltonian_HubbardDerivs_Mg=-0.02 , Hamiltonian_HubbardDerivs_S=-0.11,
                        #Hamiltonian_HubbardDerivs_P=-0.14  , Hamiltonian_HubbardDerivs_Cl=-0.0697,
                        #Hamiltonian_HubbardDerivs_K=-0.0339, Hamiltonian_HubbardDerivs_Ca=-0.0340,
                        #Hamiltonian_HubbardDerivs_Zn=-0.03, Hamiltonian_HubbardDerivs_Br=-0.0573,
                        #Hamiltonian_HubbardDerivs_I=-0.0433,
                        # set MaxAngularMomentum for H, C, O, N, F,  Na, Mg, S, P, Cl, K, Ca, Zn, Br, I
                        # Hamiltonian_MaxAngularMomentum_='', 
                        #Hamiltonian_MaxAngularMomentum_H='s',
                        #Hamiltonian_MaxAngularMomentum_C='p', Hamiltonian_MaxAngularMomentum_O='p',
                        #Hamiltonian_MaxAngularMomentum_N='p', Hamiltonian_MaxAngularMomentum_F='p',
                        #Hamiltonian_MaxAngularMomentum_Na='p', Hamiltonian_MaxAngularMomentum_Mg='p',
                        #Hamiltonian_MaxAngularMomentum_S='d', Hamiltonian_MaxAngularMomentum_P='d',
                        #Hamiltonian_MaxAngularMomentum_Cl='d', Hamiltonian_MaxAngularMomentum_K='p',
                        #Hamiltonian_MaxAngularMomentum_Ca='p', Hamiltonian_MaxAngularMomentum_Zn='d',
                        #Hamiltonian_MaxAngularMomentum_Br='d', Hamiltonian_MaxAngularMomentum_I='d'
                        )


        elif calc_type.lower() == 'band':
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
                    Options_WriteDetailedOut='Yes'
                    )
        elif calc_type.lower()[:3] == 'xtb':
            if calc_type.lower() == 'xtb1':
                Method = 'GFN1-xTB'
            elif calc_type.lower() == 'xtb2':
                Method = 'GFN2-xTB'
            else:
                print("Please set calc_type is either xtb1 or xtb2!")

            calc = Dftb(atoms=mole, label=str(formula),
                kpts=kpts,
                Hamiltonian_='xTB',
                # Hamiltonian_xTB='',
                Hamiltonian_Method=Method,
                Hamiltonian_SCC=SCC,
                Hamiltonian_SCCTolerance=1e-5,
                Hamiltonian_MaxSCCIterations=1500,
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
            raise ValueError('Calc_type for Bulk must be either STATIC, BAND or XTB!')
            

    # Molcule
    else:
        print(f"Calculate Molecule with {calc_type} method!")
        if calc_type.lower() == 'static':
            calc = Dftb(atoms=mole, label=str(formula),

                        Hamiltonian_='DFTB',
                        Hamiltonian_SCC=SCC,  
                        Hamiltonian_SCCTolerance=1e-5,
                        Hamiltonian_MaxSCCIterations=1000,
                        Hamiltonian_ConvergentSCCOnly = 'No',
                        Hamiltonian_ReadInitialCharges='No', # Static calculation
                        Hamiltonian_ForceEvaluation='dynamics',
                        

                        # ONLY FOR SCC CONVERGENCE
                        # Hamiltonian_Fermi_='', Hamiltonian_Fermi_Temperature=2000,
                        # Hamiltonian_MethfesselPaxton_='', Hamiltonian_MethfesselPaxton_Order=2,
                        # Parallel_='', Parallel_Groups=64,


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


                        # For 3OB
                        #Hamiltonian_ThirdOrderFull='Yes',
                        #Hamiltonian_HubbardDerivs_='',
                        #Hamiltonian_HCorrection_='', Hamiltonian_HCorrection_Damping_='', Hamiltonian_HCorrection_Damping_Exponent=4.05,
                        # set hubbardderivs for hubbardderivs for H, C, O, N, F,  Na, Mg, S, P, Cl, K, Ca, Zn, Br, I
                        #Hamiltonian_HubbardDerivs_H=-0.1857, Hamiltonian_HubbardDerivs_C=-0.1492,
                        #Hamiltonian_HubbardDerivs_O=-0.1575, Hamiltonian_HubbardDerivs_N=-0.1535,
                        #Hamiltonian_HubbardDerivs_F=-0.1623, Hamiltonian_HubbardDerivs_Na=-0.0454,
                        #Hamiltonian_HubbardDerivs_Mg=-0.02 , Hamiltonian_HubbardDerivs_S=-0.11,
                        #Hamiltonian_HubbardDerivs_P=-0.14  , Hamiltonian_HubbardDerivs_Cl=-0.0697,
                        #Hamiltonian_HubbardDerivs_K=-0.0339, Hamiltonian_HubbardDerivs_Ca=-0.0340,
                        #Hamiltonian_HubbardDerivs_Zn=-0.03, Hamiltonian_HubbardDerivs_Br=-0.0573,
                        #Hamiltonian_HubbardDerivs_I=-0.0433,
                        # set MaxAngularMomentum for H, C, O, N, F,  Na, Mg, S, P, Cl, K, Ca, Zn, Br, I
                        # Hamiltonian_MaxAngularMomentum_='', 
                        #Hamiltonian_MaxAngularMomentum_H='s',
                        #Hamiltonian_MaxAngularMomentum_C='p', Hamiltonian_MaxAngularMomentum_O='p',
                        #Hamiltonian_MaxAngularMomentum_N='p', Hamiltonian_MaxAngularMomentum_F='p',
                        #Hamiltonian_MaxAngularMomentum_Na='p', Hamiltonian_MaxAngularMomentum_Mg='p',
                        #Hamiltonian_MaxAngularMomentum_S='d', Hamiltonian_MaxAngularMomentum_P='d',
                        #Hamiltonian_MaxAngularMomentum_Cl='d', Hamiltonian_MaxAngularMomentum_K='p',
                        #Hamiltonian_MaxAngularMomentum_Ca='p', Hamiltonian_MaxAngularMomentum_Zn='d',
                        #Hamiltonian_MaxAngularMomentum_Br='d', Hamiltonian_MaxAngularMomentum_I='d'
                        )
        elif calc_type.lower() == 'xtb':
            calc = Dftb(atoms=mole, label=str(formula),
                Hamiltonian_='xTB',
                Hamiltonian_Method="GFN2-xTB",
                Hamiltonian_SCC=SCC,
                Hamiltonian_SCCTolerance=1e-5,
                Hamiltonian_MaxSCCIterations=1000,
                Hamiltonian_ConvergentSCCOnly = 'No',
                Hamiltonian_ReadInitialCharges='No', # Static calculation
                
                # Hamiltonian_='xTB',
                # Hamiltonian_Method="GFN1-xTB",

                # Hamiltonian_SCCTolerance=1e-5,
                # Hamiltonian_MaxSCCIterations=500,
                # Hamiltonian_ConvergentSCCOnly = 'Yes',
                # Hamiltonian_ReadInitialCharges='No', # Static calculation
                
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
            raise ValueError('Calc_type for Molecules must be eigher STATIC or XTB!')
    return calc


def run_calculator(atoms:Atoms, args:dict):
    """
    args: dict
        eg. {calc_type: 'static',
             opt: 'No',
             kpts: [1, 1, 1],
             h_scc: 'Yes',
             slater_p: './',}
    
    """

    # If one want to do geometry optimization based on ASE
    try: 
        opt_type = args['opt_type']
        print(f"Do structual optimization with opt_type={args['opt_type']}, steps={args['steps']}!")

        atoms.calc = calc_dftbplus(atoms, args)

        dyn = LBFGS(atoms, trajectory='steps.traj')
        dyn.run(fmax=0.01, steps=args['steps'])
        
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

    # print all exception 
    except Exception as err:
    
        print(f"Error: {err}, No optimization is needed!")
        atoms.calc = calc_dftbplus(atoms, args)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        # stress = atoms.get_stress()
    return atoms, energy, forces


def run_one_by_one(atoms_file:str, num_list:list, args:dict):
    #     """
    #     Run structure one by one for easily tracking and continue running

    #     Parameters:
    #     ----------
    #     atoms: ase.Atoms object
    #         atoms for continue the calculation from finished structures, empty if start from the beginning
    #     args: dict
    #         eg. {calc_type: 'static',
    #              opt: 'No',
    #              kpts_dens: 5.0,
    #              h_scc: 'Yes',
    #              slater_p: './',}
    #     """
    atoms_output = []
    atoms_error = []
    atoms_input = ase.io.read(atoms_infile, ':')
    atoms_outfile = atoms_infile.split('/')[-1].replace('.xyz', '_dftb.xyz')


    if len(num_list) > 0:
        for n, mole in enumerate(atoms_input):


            # Collect old atoms
            if n <= np.max(num_list):
                old_mole = ase.io.read('g_' + str(n) + '.xyz')
                atoms_output.append(old_mole)
                continue

            else:
                new_mole = ase.Atoms(mole.get_chemical_symbols(), mole.get_positions())
                new_mole.set_pbc(mole.get_pbc())
                if np.array_equal(mole.get_pbc(), [True, True, True]):
                    new_mole.set_cell(mole.get_cell())

                try:
                    calc_mole, energy, _ = run_calculator(new_mole, args)
                    ase.io.write('g_' + str(n) + '.xyz', calc_mole)
                    atoms_output.append(calc_mole)
                except (RuntimeError, FileNotFoundError) as err:
                    print('Error:', err)
                    atoms_error.append(new_mole)
                    ase.io.write('e_' + str(n) + '.xyz', new_mole)
                    continue

    else:
        for n, mole in enumerate(atoms_input):
            new_mole = ase.Atoms(mole.get_chemical_symbols(), mole.get_positions())
            new_mole.set_pbc(mole.get_pbc())
            if np.array_equal(mole.get_pbc(), [True, True, True]):
                new_mole.set_cell(mole.get_cell())
            try:
                # calc_mole, forces = run_calculator(calc_type, new_mole, kpts)
                calc_mole, energy, _ = run_calculator(new_mole, args)
                ase.io.write('g_' + str(n) + '.xyz', calc_mole)
                atoms_output.append(calc_mole)
            except (RuntimeError, FileNotFoundError) as err:
                print('Error:', err)
                atoms_error.append(new_mole)
                ase.io.write('e_' + str(n) + '.xyz', new_mole)
                continue


        ase.io.write(atoms_outfile, atoms_output)
        ase.io.write('error_moles.xyz', atoms_error)


        # Delte log file
        if len(atoms_output) == len(atoms_input):
            os.system('rm g_*.xyz')


    return atoms_output, atoms_error


# def md_dftb_calculator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--single_point', nargs='*',
                      default=False,
                      help='Single point calculation, input: atoms_infile, calc_type')
    
    parser.add_argument('-opt', '--optimization', 
                        nargs='*',
                        default=None,
                        help="Optimization geometry: -opt atoms 1000")

    parser.add_argument('-d', '--density', 
                        default=None,
                        type=int,
                        help='kpts density for bulk calculation')
    
    parser.add_argument('-rep', '--repulsion',
                        default=None,
                        type=str,
                        help='Repulsion potential from Spline part, input: Yes or No')

    parser.add_argument('-md', '--molecular_dynamics', 
                        nargs='*',
                      default=False,
                      help='Molecular dynamics calculation, input: atoms_infile, calc_type, gap.xml')
    
    parsergs = parser.parse_args()



    if parsergs.density is not None:
        kpts_dens = parsergs.density
    else:
        kpts_dens  = 5   # kpts per inverse Angstrom
        print(f"Attention: without kpoints density input, if not molecules, set defaultly: \nkpts_dens={kpts_dens} per inverse Angstrom!")    
    
    if parsergs.repulsion is not None:
        rep_option = parsergs.repulsion    
    else:
        rep_option = "Yes" # calculate repulsion potential from repulsive fitting, after "Spline"
        print(f"Implement spline repulsive under skf files!")
        

    if rep_option == 'Yes':
        poly_rep = 'No'
    else:
        poly_rep = 'Yes'  # calculate repulsion potential from polynomial fitting, equal to zero in my case

    args = {
        "poly_rep" : poly_rep,
        "SCC"      : 'Yes',
        "slater_p" : './',
        "kpts_dens": kpts_dens,
    }


    if parsergs.single_point:
        atoms_infile, calc_type = parsergs.single_point
        args['calc_type'] = calc_type

        if parsergs.optimization:
            opt_type, opt_steps = parsergs.optimization
            # atoms means only optimize the atomic position
            # stress means optimize the bulk size as well, not yet
            args['opt_type'] = opt_type
            args['steps'] = int(opt_steps)

        p = Path.cwd()
        num_list = []
        num_list = [int(str(path).split('_')[-1].split('.')[0]) for path in p.glob('g_*.xyz')]

        atoms_output, atoms_error = run_one_by_one(atoms_infile, num_list, args)
        # ase.io.write(atoms_outfile, atoms_output)
        # ase.io.write('error_moles.xyz', atoms_error)


    if parsergs.molecular_dynamics:
        t1 = time.perf_counter()
        print(f"t1:{t1}")

        T = 300
        steps = 20000
        step_time = 0.5  # units.fs
        friction  = 0.001
        log_step, save_step = 10, 50
        print(f"\nMD Attention: T={T}, steps={steps}, t={step_time} fs, friction={friction}\nlog: per {step_time * log_step} fs\n\save: per {step_time * save_step} fs!")

        if parsergs.molecular_dynamics[-1].split('.')[-1] == 'xml':
            if len(parsergs.molecular_dynamics) == 3:

                atoms_infile, calc_type, param_filename = parsergs.molecular_dynamics
                atoms_outfile = atoms_infile.replace('.xyz', '_dftbgap_md.traj')
                atoms_input = ase.io.read(atoms_infile)

                def printenergy(a=atoms_input):  # store a reference to atoms in the definition.
                    """Function to print the potential, kinetic and total energy."""
                    epot = a.get_potential_energy() / len(a)
                    ekin = a.get_kinetic_energy() / len(a)
                    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
                        'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

        #        ks, kpts = monkhorstpack2kptdensity(atoms_input, kpts_dens)
                args['calc_type'] = calc_type
                args['kpts_dens'] = kpts_dens


                dftb_calc = calc_dftbplus(atoms_input, args)

                # gap calculator
                gap_soap_calc = Potential(param_filename=param_filename)

                sum_calc = MixedCalculator(dftb_calc, gap_soap_calc, 1, 0.04336)
                atoms_input.calc = sum_calc

                dyn = Langevin(atoms_input, step_time * units.fs, temperature_K=T * units.kB, friction=friction)

                dyn.attach(printenergy, interval=log_step)

                MaxwellBoltzmannDistribution(atoms_input, temperature_K=2*T)
                Stationary(atoms_input)  # zero linear momentum
                ZeroRotation(atoms_input)  # zero angular momentum


                traj = Trajectory(atoms_outfile, 'w', atoms_input)
                dyn.attach(traj.write, interval=save_step)

                # Now run the dynamics
                printenergy()
                dyn.run(steps)
            
            else:

                atoms_infile, param_filename = parsergs.molecular_dynamics
                atoms_outfile = atoms_infile.replace('.xyz', '_gap_md.traj')
                atoms_input = ase.io.read(atoms_infile)

                def printenergy(a=atoms_input):  # store a reference to atoms in the definition.
                    """Function to print the potential, kinetic and total energy."""
                    epot = a.get_potential_energy() / len(a)
                    ekin = a.get_kinetic_energy() / len(a)
                    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
                        'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

                # gap calculator
                gap_soap_calc = Potential(param_filename=param_filename)

                atoms_input.calc = gap_soap_calc

                dyn = Langevin(atoms_input, step_time * units.fs, temperature_K=T * units.kB, friction=friction)

                dyn.attach(printenergy, interval=log_step)

                MaxwellBoltzmannDistribution(atoms_input, temperature_K=2*T)
                Stationary(atoms_input)  # zero linear momentum
                ZeroRotation(atoms_input)  # zero angular momentum


                traj = Trajectory(atoms_outfile, 'w', atoms_input)
                dyn.attach(traj.write, interval=save_step)

                # Now run the dynamics
                printenergy()
                dyn.run(steps)


        else:
            assert len(parsergs.molecular_dynamics) == 2, "Hi, you dont want to do run DFTB with GAP.xml right?"
            atoms_infile, calc_type = parsergs.molecular_dynamics
            atoms_outfile = atoms_infile.replace('.xyz', '_dftb_md.traj')
            atoms_input = ase.io.read(atoms_infile)


            def printenergy(a=atoms_input):  # store a reference to atoms in the definition.
                """Function to print the potential, kinetic and total energy."""
                epot = a.get_potential_energy() / len(a)
                ekin = a.get_kinetic_energy() / len(a)
                print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
                    'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

            # ks, kpts = monkhorstpack2kptdensity(atoms_input, kpts_dens)
            args['calc_type'] = calc_type
            args['kpts_dens'] = kpts_dens

            atoms_input.set_calculator(calc_dftbplus(atoms_input, args))
            dyn = Langevin(atoms_input, step_time * units.fs, T * units.kB, friction)
            dyn.attach(printenergy, interval=log_step)

            MaxwellBoltzmannDistribution(atoms_input, temperature_K=2*T)
            Stationary(atoms_input)  # zero linear momentum
            ZeroRotation(atoms_input)  # zero angular momentum

            traj = Trajectory(atoms_outfile, 'w', atoms_input)
            dyn.attach(traj.write, interval=save_step)

            # Now run the dynamics
            printenergy()
            dyn.run(steps)
        
        t2 = time.perf_counter()
        print(f"t2-t1:{t2-t1}")


    

