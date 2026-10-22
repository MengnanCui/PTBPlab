"""Generate the DFT reference of example 07 with GPAW.

Writes, next to this script:

- ref/cu_fcc.xyz: the structure PTBP fits, the fcc frame at the centre of the
  bundled EOS scan (dft/dft.xyz, a = 3.637 Å), one Cu atom, no magnetic moment;
- ref/cu_fcc_band.json: its band structure as an ase BandStructure.

DFT settings: PBE, plane waves (500 eV), 12x12x12 Gamma-centred k-points,
Fermi-Dirac smearing of 0.1 eV. The bands are computed at the self-consistent
density along W-X-G-L-K-G-L (the path of `path_dict['fcc']` in
utils/calculator.py) with 101 k-points and 16 bands, and are referenced to the
Fermi level of the self-consistent run.

GPAW's default Cu setup has 11 valence electrons (3d10 4s1), the valence shell
PTBP uses for Cu, so band n of the reference is band n of DFTB.

Usage (needs ase and gpaw, e.g. `conda install -c conda-forge gpaw gpaw-data`):

    python make_reference.py
"""
from pathlib import Path

import ase.io
import numpy as np
from ase import Atoms
from ase.spectrum.band_structure import BandStructure
from gpaw import GPAW, PW, FermiDirac

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / 'ref'

frames = ase.io.read(REPO / 'dft' / 'dft.xyz', ':')
fcc = [atoms for atoms in frames if atoms.info.get('structure_name') == 'fcc']
centre = fcc[len(fcc) // 2]

# A fresh Atoms object: the frame in dft.xyz carries initial magnetic moments,
# which would make GPAW run spin-polarised.
atoms = Atoms('Cu', cell=centre.get_cell(), pbc=True)
atoms.info['structure_name'] = 'fcc'
OUT.mkdir(exist_ok=True)
ase.io.write(OUT / 'cu_fcc.xyz', atoms)

atoms.calc = GPAW(mode=PW(500),
                  xc='PBE',
                  kpts={'size': (12, 12, 12), 'gamma': True},
                  occupations=FermiDirac(0.1),
                  txt=None)
energy = atoms.get_potential_energy()
fermi_level = atoms.calc.get_fermi_level()
print(f"SCF: E = {energy:.6f} eV, Fermi level = {fermi_level:.6f} eV")

bandpath = atoms.cell.bandpath('WXGLKGL', npoints=101)
band_calc = atoms.calc.fixed_density(kpts=bandpath,
                                     symmetry='off',
                                     nbands=16,
                                     convergence={'bands': 10},
                                     txt=None)
kpts = band_calc.get_ibz_k_points()
assert np.allclose(kpts, bandpath.kpts), "GPAW changed the k-points of the band path"

energies = np.array([[band_calc.get_eigenvalues(kpt=k, spin=s)
                      for k in range(len(kpts))]
                     for s in range(band_calc.get_number_of_spins())])
bs = BandStructure(path=bandpath, energies=energies, reference=fermi_level)
bs.write(OUT / 'cu_fcc_band.json')
print(f"Wrote {OUT / 'cu_fcc_band.json'}: {energies.shape[1]} k-points x "
      f"{energies.shape[2]} bands, path {bandpath.path}")
