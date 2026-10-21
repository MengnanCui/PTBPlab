# pydftb

A from-scratch, readable SCC-DFTB engine in Python, validated against DFTB+.
It lives inside PTBPlab as an independent sub-package.

- **Why it exists:** [`docs/WHY.md`](docs/WHY.md)
- **Roadmap and DFTB+ feature checklist:** [`docs/ROADMAP.md`](docs/ROADMAP.md)
- **Learning map** (each module next to the DFTB+ source file that implements the same physics): [`docs/dftbplus_map.md`](docs/dftbplus_map.md)
- **Development docs (HTML, open in a browser):** [`docs/index.html`](docs/index.html): every stage's design notes and report, key results first, development details folded
- **Glossary** (Chinese, plain language): [`docs/glossary.md`](docs/glossary.md)
- **Prototype research report** (performance experiments): [`RESEARCH.md`](RESEARCH.md)
- **License:** MIT, like the rest of PTBPlab (see [`LICENSE`](../LICENSE)). The optional D3/D4 dispersion packages (`dftd3`, `dftd4`) are LGPL-3.0 and are only imported at run time.

## Quick start

```python
import pydftb
from ase.build import molecule

params = pydftb.ParamSet("path/to/skf")                  # A-B.skf or A-B.skf.gz files
mf = pydftb.SCC(pydftb.System.from_ase(molecule("H2O")), params)
e = mf.kernel()      # total energy (Hartree)
f = mf.forces()      # forces (Hartree/Bohr)
print(mf.mulliken)   # net Mulliken charges
```

Periodic systems take a Monkhorst-Pack grid, e.g. `pydftb.SCC(system, params, kpts=(4, 4, 4), temperature=1000)`.

ASE users can call `pydftb.ase_calc.PyDFTB` instead; it works in eV and Angstrom.

### Several parameter sets for one element

Give atoms type labels such as `"O2"` or `"Li1"`. The engine then reads `O2-H.skf` and so on, following DFTB+'s `Type2FileNames` convention:

```python
labels = atoms.get_chemical_symbols(); labels[3] = "O2"
mf = pydftb.SCC(pydftb.System.from_ase(atoms, labels=labels), params)
```

`pydftb.skf.write_typed_copies` creates the files for a new type from existing ones.

### f orbitals

Parameter sets with f electrons (lanthanides, actinides) use DFTB+'s extended SKF format
(first line `@`); simple and extended files can be mixed.  `max_l` is read from the files
as before (an f shell counts when its occupation is non-zero), or given explicitly, e.g.
`max_l={"Gd": 3, "O": 1}`.  The test set `tests/data/skf/hotcent_gdo` (Gd, O; no repulsive)
is made by `python -m pydftb.validation.make_f_set`; details in
[`docs/reports/M2g-f-orbitals.html`](docs/reports/M2g-f-orbitals.html).

### Dispersion

`dispersion=` takes a dict that mirrors DFTB+'s `Dispersion` block (Bohr, Hartree):

```python
d3 = {"model": "dftd3", "damping": "beckejohnson", "a1": 0.5719, "a2": 3.6017, "s6": 1.0, "s8": 0.5883}
mf = pydftb.SCC(system, params, dispersion=d3)
```

Lennard-Jones/UFF and Slater-Kirkwood are built in.  D3 and D4 call the reference libraries, which are
optional: `conda install -c conda-forge dftd3-python dftd4-python` (or `pip install dftd3 dftd4`).
Models, defaults and the differences found in DFTB+'s own D4: [`docs/design/M2d-dispersion.html`](docs/design/M2d-dispersion.html).

### Band structures, DOS and PDOS

After an SCC run, the states at other k-points come from the converged charges, as in DFTB+'s
band-structure run (`ReadInitialCharges` + `Klines`):

```python
cu.calc = PyDFTB("path/to/skf", kpts=(6, 6, 6), temperature=1000.0)
cu.get_potential_energy()
bs = cu.calc.band_structure(path="GXWLGK", npoints=100)     # ASE BandStructure (eV)
st = cu.calc.states(kpts=(24, 24, 24), projections=True)    # pydftb.bands.States
e, dos = st.dos()                                           # dp_dos rules: Gaussian 0.1 eV
e, d = st.pdos(atoms=[0], shell=2)                          # d shell of atom 0
```

`pydftb.SCC.states()` does the same in atomic units; `pydftb.bands.klines` turns a DFTB+ `Klines`
block into k-points.  Details: [`docs/reports/M2e-bands.html`](docs/reports/M2e-bands.html).

### Geometry optimisation and MD

`PyDFTB` works with ASE's optimisers, cell filters and integrators.  For optimisations and MD,
`reuse_charges=True` starts each SCC from the previous step's charges, as DFTB+'s own drivers do:

```python
from ase.optimize import BFGS
atoms.calc = PyDFTB("path/to/skf", reuse_charges=True)
BFGS(atoms).run(fmax=0.01)
```

Comparisons with DFTB+'s `GeometryOptimisation` and `VelocityVerlet`:
`python -m pydftb.validation.dynamics --check`; details in
[`docs/reports/M2f-opt-md.html`](docs/reports/M2f-opt-md.html).

### Threads over k-points

`k_threads` (for `pydftb.DFTB`, `PyDFTB` and `pydftb.SCC`) splits the k-points of a periodic
calculation into chunks that run in threads: the reduction to standard form, the eigensolver
and the density matrices (NumPy releases the GIL inside LAPACK/BLAS).  The results are bitwise
identical for any number of threads; sums over k are never split, and BLAS runs on one thread
while the k threads work.

```python
atoms.calc = PyDFTB("path/to/skf", kpts=(14, 14, 14), k_threads=4)   # or "auto"
```

On 4 threads the eigensolver part runs 1.9–2.9× faster and a whole calculation 1.3–1.5×
(the Ewald setup, the forces and the Fermi level are not threaded; see
[`docs/reports/M3a-dispatch.html`](docs/reports/M3a-dispatch.html)).  Use the default
`solver="batched"`: the `"eigh"` solver calls SciPy, which keeps the GIL.

### Cost model

`pydftb.perfmodel` estimates the time and peak memory of an energy-and-forces
calculation before it runs, from the orbitals, the irreducible k-points and the pairs within
the Slater-Koster and Coulomb cut-offs (PTBP plans its parallel runs with it):

```python
from pydftb.perfmodel import CostModel, Species, shape

cu = Species(((2, 0.5095), (1, 0.0800), (0, 0.2403)))      # (l, Hubbard U) of each shell
s = shape(atoms.get_chemical_symbols(), {"Cu": cu}, cell=atoms.cell, pbc=atoms.pbc, grid=(8, 8, 8))
model = CostModel()                                         # CostModel(speed=...) from probe()
model.pydftb_time(s, k_threads=4), model.memory("pydftb", s), model.dftbplus_time(s)
```

The constants were fitted to calibration runs (Cu, C, CuO, Li, GdO cells of 2–216 atoms
and PTBP's Cu dataset): median error 12 % for the time (DFTB+ 7 %), 2 % for pydftb's
memory.  With shell-resolved charges the Coulomb sum reaches as far as the smallest Hubbard
U requires (96 Bohr for U = 0.08 Ha, the 4p shell of Cu): a periodic cell of 64 Cu atoms
has 3 million atom pairs within it and pydftb needs about 2.2 GB (DFTB+ 0.3 GB); see
[`docs/reports/M3c-recommender.html`](docs/reports/M3c-recommender.html).

## Tests and validation

```bash
PYTHONPATH=. pytest pydftb/tests                    # offline, uses frozen DFTB+ references
python -m pydftb.validation.refgen --check          # table of deviations from DFTB+
python -m pydftb.validation.refgen --regenerate     # needs dftb+; rewrites the references
python -m pydftb.validation.refgen --live           # fresh comparison with dftb+
python -m pydftb.validation.dynamics --check        # optimisation and MD vs DFTB+'s drivers
```
