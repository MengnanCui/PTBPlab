import math
# from cv2 import dft
from pathlib import Path

import ase.io
import numpy as np
from ase import Atoms
from ase.calculators.aims import Aims
from ase.optimize import BFGS
from numpy import pi, sqrt


def monkhorstpack2kptdensity(atoms, k_density):

    """Convert Monkhorst-Pack grid to k-point density.
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

# %%
def calc_fhiaims(kpts):
    calc = Aims(aims_command = "mpirun -n 5 /u/mncui/software/fhiaims_code/bin/aims.201231.scalapack.mpi.x > aims.out",
                outfilename = "aims.out",
                species_dir = "/u/mncui/software/FHIAIMS/fhiaims_code/species_defaults/really_tight",
                xc='PBE',
                spin='none',
                relativistic='atomic_zora scalar', # default
                k_grid=kpts,
                basis_threshold=1e-5,            #default
                override_illconditioning='true', #default
                relax_unit_cell='none',
                sc_accuracy_etot=1e-5,
                sc_accuracy_eev=1e-3,     # Usually, value=10−3 eV is enough to indicate a reliable total-energy and force convergence
                sc_accuracy_rho=1e-4,     # value=10−6·n_atoms/6
                sc_accuracy_forces=1e-4,  # criterion for scf / relaxation
                n_max_pulay=8,
                charge_mix_param=0.02,
                occupation_type='gaussian 0.1')
    return calc

def run_calculator(atoms, k_grid):
    # atoms.calc = calc_dftbplus('static', atoms, 'res', 'No', 'Yes', './', k_grid)
    # first No: we use spline interpolation to get the repulsive potential
    # second No: we drop SCC calculation

    atoms.calc = calc_fhiaims(k_grid)
    #dyn    = BFGS(atoms)
    #dyn.run(fmax=0.01, steps=300)
    energy = atoms.get_total_energy()
    # forces = atoms.get_forces()
    return atoms, energy
import sys
filename = sys.argv[1]
moles   = ase.io.read(filename, ':')
length  = len(moles)
k_density = 1.0 # kpts per inverse Angstrom
dft_moles = []
err_moles = []

# continue from the last calculation g_*.xyz
p = Path.cwd()
num_list = []
num_list = [int(str(path).split('_')[-1].split('.')[0]) for path in p.glob('g_*.xyz')]

if len(num_list) > 0:
    for n, mole in enumerate(moles):
        if n <= np.max(num_list):
            old_mole = ase.io.read('g_' + str(n) + '.xyz')
            dft_moles.append(old_mole)
            continue
        else:
            new_mole = ase.Atoms(mole.get_chemical_symbols(), mole.get_positions())
            new_mole.set_pbc(True)
            new_mole.set_cell(mole.get_cell())
            ks, kpts = monkhorstpack2kptdensity(new_mole, k_density)
            try:
                calc_mole, kpts = run_calculator(new_mole, kpts)
                ase.io.write('g_' + str(n) + '.xyz', calc_mole)
                dft_moles.append(calc_mole)
            except RuntimeError as err:
                print('Error:', err)
                err_moles.append(new_mole)
                ase.io.write('e_' + str(n) + '.xyz', new_mole)
                continue
else:
    for n, mole in enumerate(moles):
        new_mole = ase.Atoms(mole.get_chemical_symbols(), mole.get_positions())
        new_mole.set_pbc(True)
        new_mole.set_cell(mole.get_cell())
        ks, kpts = monkhorstpack2kptdensity(new_mole, k_density)
        try:
            calc_mole, kpts = run_calculator(new_mole, kpts)
            ase.io.write('g_' + str(n) + '.xyz', calc_mole)
            dft_moles.append(calc_mole)
        except RuntimeError as err:
            print('Error:', err)
            err_moles.append(new_mole)
            ase.io.write('e_' + str(n) + '.xyz', new_mole)
            continue
ase.io.write('fhiaims.xyz', dft_moles)
ase.io.write('error_moles.xyz', err_moles)
