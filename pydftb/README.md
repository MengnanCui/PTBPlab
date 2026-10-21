# pydftb

A from-scratch, readable SCC-DFTB engine in Python, validated against DFTB+.
It lives inside PTBPlab as an independent sub-package.

- **Why it exists:** [`docs/WHY.md`](docs/WHY.md)
- **Roadmap and DFTB+ feature checklist:** [`docs/ROADMAP.md`](docs/ROADMAP.md)
- **Learning map** (each module next to the DFTB+ source file that implements the same physics): [`docs/dftbplus_map.md`](docs/dftbplus_map.md)
- **Design notes:** [`docs/design/`](docs/design)
- **Milestone reports:** [`docs/reports/`](docs/reports)
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

## Tests and validation

```bash
PYTHONPATH=. pytest pydftb/tests                    # offline, uses frozen DFTB+ references
python -m pydftb.validation.refgen --check          # table of deviations from DFTB+
python -m pydftb.validation.refgen --regenerate     # needs dftb+; rewrites the references
python -m pydftb.validation.refgen --live           # fresh comparison with dftb+
```
