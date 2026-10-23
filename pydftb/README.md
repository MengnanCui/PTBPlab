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

ASE users can call `pydftb.ase_calc.PyDFTB` instead; it works in eV and Angstrom.  When the
engine fails where DFTB+ would stop (a non-positive-definite overlap, e.g. very diffuse
basis functions), it raises ASE's `CalculationFailed`, as `ase.calculators.dftb.Dftb` does.

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

### Non-collinear spin and spin-orbit coupling

Two-component (spinor) calculations follow DFTB+'s `SpinPolarisation = NonColinear` and
`SpinOrbit` blocks (NumPy engine only):

```python
mf = pydftb.SCC(system, params, temperature=300, spin="noncolinear",
                initial_spins=[[0, 0, 1], [0, 0, 0]],          # a vector per atom
                spin_constants={"O": -0.028, "H": -0.072},
                spin_orbit={"O": [0.0, 0.0004]},              # xi per shell (Hartree), s = 0
                mixer="pulay")                                 # see below
mf.kernel()
mf.result.magmoms, mf.result.e_ls, mf.result.orbital_moments  # (N, 3), E_LS, <L> per shell
```

`spin_orbit` also works without spin polarisation; `spin_orbit_dual=False` selects DFTB+'s
onsite model instead of the dual one (its default).  Spin-orbit coupling needs the full
k-point grid (`api.SCC` and `PyDFTB` use it automatically for two-component runs).  With
`PyDFTB(spin="noncolinear")` the atoms' (N, 3) initial magnetic moments are the initial
spins.

The direction of free magnetic moments is a soft mode of the SCC: without spin-orbit
coupling a global rotation costs nothing, and weak spin-orbit coupling hardly pins it.
DFTB+'s (and pydftb's default) Broyden mixer can then diverge, e.g. above with the default
mixer, or for periodic cells whose moments are not along a coordinate axis; the Pulay mixer
(`mixer="pulay"`) converged in these cases.  Details:
[`docs/reports/M6a-spin-orbit.html`](docs/reports/M6a-spin-orbit.html).

### DFTB+U and onsite corrections

Block potentials, DFTB+'s `OrbitalPotential` and `OnSiteCorrection` (NumPy engine only):

```python
from pydftb.blockpot import ONSITE_PBE
mf = pydftb.SCC(system, params, temperature=300,
                plus_u={"Cu": [("d", 0.15)]},       # groups of shells with U - J (Hartree)
                plus_u_functional="fll",            # or "psic"
                onsite_corrections=ONSITE_PBE)      # f_uu, f_ud per type (DFTB+ manual, PBE)
mf.kernel()
mf.result.e_plus_u, mf.result.e_onsite             # "Energy DFTB+U", "Energy onsite"
```

Shells are angular momentum letters (`"pd"` couples the p and d shells in one group) or
0-based shell indices (DFTB+'s `Shells` count from 1).  Onsite corrections need an entry for
every type, zeros for H; `ONSITE_PBE` has H, C, N, O, S, Ti and Au for s / sp / spd bases.
Both work with collinear and non-collinear spin, spin-orbit coupling (+U only with the dual
model, as in DFTB+), DFTB3, k-points, forces, stress and band structures.  The SCC mixes
what DFTB+ mixes then (the corrected orbitals one by one, the blocks' off-diagonal elements),
so it takes the same iterations; given charges (`q_init`, `reuse_charges`) count as read in,
so the onsite potential is in the first iteration, while DFTB+'s own drivers leave it out of
the first iteration of every geometry step.  Details:
[`docs/reports/M6b-dftbu-onsite.html`](docs/reports/M6b-dftbu-onsite.html).

### External fields, point charges and implicit solvation

DFTB+'s `ElectricField` (`External`, `PointCharges`), `AtomSitePotential` and `Solvation`
(`GeneralisedBorn`, `SASA`, `CM5`) blocks (NumPy engine only):

```python
from pydftb.solvation import GeneralizedBorn, SASA, CM5
from pydftb.constants import BOHR, VOLT_PER_ANGSTROM_AU

mf = pydftb.SCC(system, params,
                electric_field=(0.5 * VOLT_PER_ANGSTROM_AU, 0, 0),         # Ha/(e Bohr)
                point_charges=(r_bohr, q, 0.5),                            # positions, charges, blur
                site_potentials={"gross": {0: 0.01}, "net": {3: -0.02}})   # Hartree
mf.forces()
mf.result.e_ext, mf.result.dipole, mf.result.point_charge_forces

gbsa = GeneralizedBorn.from_param_file("param_gbsa_water.txt")   # xtb-format file (DFTB+ ParamFile)
alpb = GeneralizedBorn("water", free_energy_shift=0.0, born_scale=1.5, born_offset=0.0,
                       descreening="unity", kernel="p16", alpb=True, cm5=CM5(),
                       sasa=SASA(probe_radius=1.4 / BOHR,
                                 surface_tension={"H": -0.33, "C": -0.75, "O": 0.92}))
pydftb.SCC(molecule, params, solvation=gbsa).kernel()                # result.e_solv
```

Energies go where DFTB+ puts them: the field and the site potentials in `e_ext`, the
solvent in `e_solv`, the point charges' energy in `e_scc`.  Periodic cells get DFTB+'s
saw-tooth field (with its warnings) and Ewald sums over the point charges' images; the
forces on the point charges are `result.point_charge_forces`.  `PyDFTB` takes the field in
V/Å, the charges in Å and e, the site potentials in eV, and `embed()` makes it a QM
calculator for ASE's `EIQMMM`.  Point charges, site potentials and solvents need SCC;
solvents are for molecules only; periodic cells with point charges have no stress, as in
DFTB+.  With a solvent, an external field is scaled by 3ε/(2ε+1) unless
`rescale_solvated_fields=False` (DFTB+'s `RescaleSolvatedFields`).  `CM5()` follows the CM5
paper; DFTB+ 25.1 adds the CM5 correction twice in its GB model, which
`CM5(double_count=True)` reproduces.  Details:
[`docs/reports/M6c-fields-solvation.html`](docs/reports/M6c-fields-solvation.html).

### Excited states (Casida linear response)

DFTB+'s `ExcitedState { Casida }` (NumPy engine only): vertical excitation energies,
oscillator strengths, transition dipoles and the dominant single-particle transitions of
molecules (or the Gamma point of a cell), after the SCC:

```python
from pydftb.casida import Casida

mf = pydftb.SCC(system, params,
                spin_constants={"H": -0.072, "C": -0.023, "O": -0.028},   # triplets, spin polarisation
                casida=Casida(n_excitations=8, symmetry="singlet"))        # "triplet", "both"
mf.kernel()
x = mf.result.excitations
x.energies, x.oscillator_strengths, x.transition_dipoles     # Hartree, -, e Bohr
x.transitions, x.weights, x.spin_contamination               # dominant (i, a, spin), 0-based; D<S^2>
print(x.table())                                             # EXC.DAT; x.single_particle.table(): SPX.DAT
```

Singlets, triplets (`symmetry="both"` gives `[triplets, singlets]`) and collinear spin
polarisation (both channels coupled, with DFTB+'s D<S^2>); fractional occupations, onsite
corrections, energy and oscillator windows, a state of interest (`state_of_interest=k`,
or `-1` for the brightest singlet: `result.excitation_energy`, while `energy` stays the
ground state's, as in DFTB+'s output).  The restrictions are DFTB+'s: SCC, atom-resolved,
no DFTB3, two-component spin, DFTB+U or solvent, Gamma point only; fields, point charges
and dispersion are fine.  `solver="auto"` diagonalises Omega densely up to 500
transitions and uses a block Davidson solver beyond (C60's 14,400 transitions in 0.46 s,
DFTB+'s fastest solver 1.46 s); `"arpack"` is an independent check.  In degenerate
manifolds only group sums are defined (`x.degenerate_groups()`).  `PyDFTB(...,
casida=...)` gives `calc.excitations`.  Details:
[`docs/reports/M6d-casida.html`](docs/reports/M6d-casida.html).

### Excited-state forces, charges and XL-BOMD

With a state of interest the forces are those of the excited state, -d(E0 + Omega)/dR, by
the Z-vector method (DFTB+'s `ExcitedStateForces`; NumPy engine only), and the relaxed
density gives the excited state's Mulliken charges and dipole (DFTB+'s XCH.DAT, XREST.DAT):

```python
calc = DFTB(sk, casida=Casida(6, state_of_interest=2))   # excited_forces=True by default
r = calc.calculate(labels, positions)
r.forces                                     # -grad(E0 + Omega_2)
r.excited_energy, r.excited_free_energy      # E0 + Omega (r.energy stays the ground state's)
r.excited_charges, r.excited_dipole          # Casida(..., excited_charges=True) without forces
atoms.calc = PyDFTB(skdir, casida={"n_excitations": 6, "state_of_interest": 1})
BFGS(atoms).run(fmax=0.01)                   # ASE sees E0 + Omega: optimisation and MD in S1
```

Singlets, triplets and collinear spin polarisation; fields, point charges, site potentials,
dispersion, H-X damping and energy windows are included exactly (pydftb's forces agree with
differences of its own E0 + Omega to 1e-10 Ha/Bohr; DFTB+'s miss terms in five of these
combinations, see the report).  Restrictions as in DFTB+: molecules, integer occupations,
no oscillator window or onsite corrections; spin constants that differ between shells are
refused.

XL-BOMD (DFTB+'s `Xlbomd` / `XlbomdFast`): the SCC populations are integrated with the
atoms and start every step's SCC, so a step needs a few SCC iterations (or one):

```python
from pydftb.xlbomd import XLBOMD
atoms.calc = PyDFTB(skdir, xlbomd=XLBOMD(integration_steps=5, scc_tol=1e-6))
VelocityVerlet(atoms, 0.5 * units.fs).run(1000)            # one calculation per MD step
PyDFTB(skdir, xlbomd=XLBOMD(fast=True, scale=0.5))         # one SCC iteration per step
```

The energy is the shadow energy E_band[D(n)] + 1/2 (2 q_out - n).gamma.n and the forces its
derivatives, as in DFTB+; `force_evaluation` is DFTB+'s `ForceEvaluation` ("dynamics_t0" at
T = 0, "dynamics" otherwise, chosen automatically).  Against DFTB+'s own XL-BOMD (K = 5, 6,
7, PreSteps, XlbomdFast, DFTB3, k-points at 500 K) the trajectories agree to the digits
DFTB+ prints and every step takes the same number of SCC iterations.  The restrictions are
DFTB+'s (SCC, Fermi filling, no spin, +U, onsite, fields, charges, solvent or excited-state
forces).  Details:
[`docs/reports/M6e-excited-forces-xlbomd.html`](docs/reports/M6e-excited-forces-xlbomd.html).

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

### DFTB+ configuration advisor

`python -m pydftb.advisor RUN_DIR` reads a DFTB+ run -- `dftb_in.hsd`, and if present
`dftb_pin.hsd` (what DFTB+ really used), the screen output and the SKF files -- and reports
the pitfalls found while validating pydftb against DFTB+, with the file and line of the
evidence and advice.  It only reads; DFTB+ is not needed.

```text
$ python -m pydftb.advisor run/
[P01] warning old ParserVersion: DFTB+ rewrites the input  (dftb_in.hsd:46)
    ParserVersion = 1: on the way to version 14, ShellResolvedSCC = Yes is added ...;
    Differentiation = FiniteDiff { Delta = 0.01 } is added ...
[P03] warning input ignored by the parser  (dftb_in.hsd:10-12)
    Hamiltonian/DFTB/Fermi is not inside Filling and is ignored ...; DFTB+ used Filling =
    Fermi with T = 0 Ha (zero temperature)
```

18 rules (`--list-rules`, `--explain P03`): old parser versions (P01, P02), ignored input
(P03), temperatures without a unit (F01), nearly equal Hubbard U values that break DFTB+'s
gamma formula (H01), SKF files without a Spline repulsive (K01), band runs that stop (B01),
threads and MPI groups (T01-T04), unconverged SCC results (S01), DFTB+ errors (X01), the
Ewald tolerance (E01), lattice optimisation and LBFGS (G01, G02), periodic D4 (D02, D03).
`--json` for machines; the exit status is 1 when there is a finding of `--fail-on`
severity (default `error`).  Frozen DFTB+ 25.1 runs for every rule are in
`tests/data/advisor/` (`python -m pydftb.validation.advisor_cases --regenerate`); see
[`docs/reports/M3d-advisor.html`](docs/reports/M3d-advisor.html).

### PyTorch backend

`backend="torch"` (for `pydftb.DFTB`, `PyDFTB` and `pydftb.SCC`) runs the heavy linear
algebra of every SCC iteration on a torch device: the folding to k-points, the Cholesky
factor of S, the eigensolver, the density matrices, the Mulliken populations, the traces of
the energy and the real-space densities of the forces.  Geometry, parameters, gamma, the
mixer and the Fermi level stay NumPy; results come back as NumPy arrays and floats.  torch is
optional (`pip install "torch>=2.14" --index-url https://download.pytorch.org/whl/cpu`);
`import pydftb` never imports it.

```python
atoms.calc = PyDFTB("path/to/skf", kpts=(8, 8, 8), backend="torch")                 # CPU
atoms.calc = PyDFTB("path/to/skf", kpts=(8, 8, 8), backend="torch", device="cuda")  # GPU
atoms.calc = PyDFTB("path/to/skf", kpts=(8, 8, 8), backend="torch", dtype="float32")
```

On the 74 frozen DFTB+ references it passes with the same tolerances and the same SCC
iteration counts as DFTB+; against the NumPy backend the largest differences are
4e-14 Ha (energy), 4e-14 Ha/Bohr (forces) and 1.4e-13 e (charges).  `dtype="float32"` runs
H, the eigensolver and the densities in single precision (the Cholesky factor of S stays
double): energies within about 4e-6 Ha.  `threads` sets torch's thread count too; set it in
processes forked from one that ran torch on several threads.  GPU speed-ups are not
measured yet (see *Devices and the GPU self-check* below); see
[`docs/reports/M4a-torch-backend.html`](docs/reports/M4a-torch-backend.html).

### Derivatives (`pydftb.autodiff`)

`pydftb.autodiff` (needs torch) differentiates a converged calculation with respect to the
parameters -- every value of the SK tables, on-site energies, Hubbard U and derivatives,
spin constants, repulsive coefficients -- and to the positions and the cell.  The engine
(NumPy or torch) converges the SCC with DFTB+'s rules; the returned values are its own
numbers, carrying derivatives:

```python
from pydftb.autodiff import Params, evaluate

params = Params.from_skset(calc.sk, calc.repulsive, requires_grad=("tables", "hubbard"))
res = evaluate(calc, species, positions, cell, pbc, params)              # mode="hf"
res.free_energy.backward()             # params.tables[("Cu", "O")].grad, params.hubbard["Cu"].grad

res = evaluate(calc, species, positions, cell, pbc, params, mode="implicit")
loss = ((res.forces - f_ref) ** 2).sum() + (res.energy - e_ref) ** 2
loss.backward()                        # derivatives of energies and forces
```

`mode="hf"` uses the variational property of the Mermin free energy F: dF/dtheta, forces
and stress without differentiating the eigensolver.  `mode="implicit"` attaches the SCC
fixed point by the implicit function theorem, so the charges, eigenvalues, Fermi levels,
forces and stress have full derivatives too (forces use DFTB+'s finite-difference kernel
when `fd_delta` is set, as PTBP runs).  Against central differences of the engine
(`python -m pydftb.validation.gradcheck --check`, 11 cases, 346 comparisons) the median
relative deviation is 2e-9.  The l1 = l2 integrals that both files A-B and B-A store, and
equal Hubbard U values, only have derivatives along directions that move them together.
With Methfessel-Paxton smearing and DFTB+'s TS, F is not variational: mode "hf" refuses dF
(use `Smearing(..., ts_convention="variational")` or mode "implicit").
`Smearing(fermi_level="exact")` replaces DFTB+'s mid-gap Fermi level (which lets the
electron count drift by up to 1e-7) by the exact count.  See
[`docs/reports/M4b-autodiff.html`](docs/reports/M4b-autodiff.html).

The ASE calculator has the same entry point: `PyDFTB.evaluate(atoms, requires_grad=("tables",),
mode="hf")` runs the calculation of `calculate` -- its `results` are bit for bit the same --
and returns `(Result, Params)` with the derivatives.  PTBP builds the gradient of its loss on
it (M4c, [`docs/reports/M4c-ptbp-gradient.html`](docs/reports/M4c-ptbp-gradient.html)).

### Devices and the GPU self-check

`device` is `"cpu"`, `"cuda"`/`"cuda:i"` (NVIDIA, or AMD with a ROCm build of torch), `"mps"`
(Apple), `"xpu"` (Intel) or `"auto"` (the first of these that is available, else the CPU).
What the device can do decides where each step runs (`pydftb.backend.devices`):

| Device | Precision | Cholesky, eigh | float64 arrays and sums |
|---|---|---|---|
| CPU, CUDA (NVIDIA, ROCm) | float64, float32 | device | device |
| MPS | float32 only | device if a probe works, else CPU | CPU |
| XPU | float64 if a probe works | device if a probe works, else CPU | CPU without float64 |

MPS and XPU are probed once with 2 x 2 matrices instead of a list of supported operations.
On CUDA a calculation runs with torch's deterministic algorithms (switched on for the
calculation, the previous setting restored afterwards; `CUBLAS_WORKSPACE_CONFIG=:4096:8` is
set before the first CUDA call unless already set), so a repeated run gives the same bits;
GPU and CPU differ in the last digits.  `pydftb.autodiff` computes on the device of its
`Params` (`Params.from_skset(..., device="cuda")`).

No GPU was available while this was written.  The device logic was checked on the CPU: the
branches that move work to the CPU give the same bits as the plain path, and under
`torch.device("meta")` no tensor is created off its device.  GPU correctness and speed are
**not measured**.  To check a GPU:

```bash
python -m pydftb.validation.gpu_check --json gpu_check.json      # first GPU, everything
python -m pydftb.validation.gpu_check --device cuda:1 --dtype both --json gpu_check.json
python -m pydftb.validation.gpu_check --quick                    # a subset, a few minutes
```

It recomputes the frozen DFTB+ references on the device and compares them with torch on the
CPU (float64: within 1e-10), runs calculations twice (bit for bit), compares autodiff
gradients of both modes with the CPU (1e-9), checks that a process that used CUDA can fork a
CPU-only worker, and times `eigh` (N = 9-288, batches of 1-2048), Cu cells as PTBP computes
them, and stacks of several structures.  It exits with status 1 if a check fails; please
send the JSON back.  See [`docs/reports/M4d-devices.html`](docs/reports/M4d-devices.html).

## Tests and validation

```bash
PYTHONPATH=. pytest pydftb/tests                    # offline, uses frozen DFTB+ references
python -m pydftb.validation.refgen --check          # table of deviations from DFTB+
python -m pydftb.validation.refgen --regenerate     # needs dftb+; rewrites the references
python -m pydftb.validation.refgen --live           # fresh comparison with dftb+
python -m pydftb.validation.dynamics --check        # optimisation and MD vs DFTB+'s drivers
python -m pydftb.validation.refgen --check --backend torch    # the same on the torch backend
python -m pydftb.validation.golden --base COMMIT    # NumPy path bitwise unchanged since COMMIT
python -m pydftb.validation.gradcheck --check       # pydftb.autodiff vs finite differences
python -m pydftb.validation.gpu_check --device cpu --quick    # the GPU self-check, on the CPU
```
