import ase.io
from ase import Atoms
import sys

print("python get_slice_atom.py f.xyz 0:1000:2\n Means slice f.xyz from 0 to 1000, 2 interval!")

inputs = sys.argv[1:]

atoms_infile, slice = inputs
atoms_outfile = atoms_infile.replace('.xyz', f'_s{slice}.xyz')
print("Input file: ", atoms_infile)
print("Slice: ", slice)


atoms_input = ase.io.read(atoms_infile, index=slice)

ase.io.write(atoms_outfile, atoms_input)
print("Output file: ", atoms_outfile)
