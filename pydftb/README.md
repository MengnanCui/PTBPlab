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

## Tests and validation

```bash
PYTHONPATH=. pytest pydftb/tests                    # offline, uses frozen DFTB+ references
python -m pydftb.validation.refgen --check          # table of deviations from DFTB+
python -m pydftb.validation.refgen --regenerate     # needs dftb+; rewrites the references
python -m pydftb.validation.refgen --live           # fresh comparison with dftb+
```
