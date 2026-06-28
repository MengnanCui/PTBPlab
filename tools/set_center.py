import ase.io
from ase import Atoms
import sys

atoms_infile = sys.argv[1]
atoms_input = ase.io.read(atoms_infile, ':')


for atoms in atoms_input:
    atoms.center()

ase.io.write('centered.xyz', atoms_input, format='xyz')