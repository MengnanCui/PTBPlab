# pydftb

A from-scratch, readable SCC-DFTB engine in Python, validated against DFTB+.
It lives inside PTBPlab and installs as a package of its own, with the `pydftb` command.

- **Why it exists:** [`docs/WHY.md`](docs/WHY.md)
- **Roadmap and DFTB+ feature checklist:** [`docs/ROADMAP.md`](docs/ROADMAP.md)
- **Learning map** (each module next to the DFTB+ source file that implements the same physics): [`docs/dftbplus_map.md`](docs/dftbplus_map.md)
- **Development docs (HTML, open in a browser):** [`docs/index.html`](docs/index.html): every stage's design notes and report, key results first, development details folded
- **Glossary** (Chinese, plain language): [`docs/glossary.md`](docs/glossary.md)
- **Prototype research report** (performance experiments): [`RESEARCH.md`](RESEARCH.md)
- **License:** MIT, like the rest of PTBPlab (see [`LICENSE`](LICENSE)). The optional D3/D4 dispersion packages (`dftd3`, `dftd4`) are LGPL-3.0 and are only imported at run time.

## Installation

pydftb needs Python 3.10 or later, NumPy, SciPy, ASE and threadpoolctl.  From a checkout of
PTBPlab:

```bash
pip install ./pydftb                     # or "./pydftb[torch,dispersion]"; -e ./pydftb to edit it
pydftb --help                            # the command-line tools (table below)
python -m pytest --pyargs pydftb.tests   # the offline tests, from any folder (needs pytest)
```

Extras: `torch` (the PyTorch backend, GPUs and `pydftb.autodiff`; GPU builds from
pytorch.org) and `dispersion` (the D3/D4 libraries; on conda-forge `dftd3-python` and
`dftd4-python`).  DFTB+ is only needed to regenerate the references or to compare live.
PTBP depends on pydftb: `pip install ./pydftb .` installs both.  A checkout also works
without installing, as before (`PYTHONPATH=.` at the repository root).

| Command | What it does |
|---|---|
| `pydftb advisor RUN_DIR` | checks a DFTB+ input and run for known pitfalls |
| `pydftb refgen --check` | pydftb against the frozen DFTB+ references (`--live`, `--regenerate` need dftb+) |
| `pydftb dynamics --check` | optimisations and MD against DFTB+'s drivers |
| `pydftb gpu-check` | the GPU self-check (needs torch) |
| `pydftb gradcheck --check` | `pydftb.autodiff` against finite differences (needs torch) |
| `pydftb golden --base COMMIT` | the NumPy path bitwise unchanged since COMMIT (in a git checkout) |

`pydftb COMMAND --help` lists a command's options, and `python -m pydftb COMMAND` is the same
command.  The module forms keep working (`python -m pydftb.validation.refgen --check`,
`python -m pydftb.advisor RUN_DIR`, ...).  The `--regenerate` modes, `make_f_set`,
`make_poly_set` and `advisor_cases` write into the package's `tests/data`: run them in a
checkout, not in an installed copy.

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

The types can also switch smoothly with the environment (milestone M5a; DFTB+ fixes each
atom's type): every atom of the element mixes its types with weights from its coordination
number, H0, S and the repulsive are mixed, and forces and stress include the weights'
derivatives, so geometry optimisation and MD work:

```python
from pydftb import DFTB, ParamSet, TypeSwitch
sw = TypeSwitch("Cu", types=("Cu2", "Cu"),   # by rising CN: low-coordinated Cu2, bulk Cu
                centre=10.5, width=1.0)      # switching window CN 10-11
sk = ParamSet(skdir).skset(["Cu", "Cu2"])    # every type, with Cu-Cu2.skf and Cu2-Cu.skf
r = DFTB(sk, kpts=k, type_switching=[sw]).calculate(["Cu"] * n, pos, cell, pbc)
r.type_weights["Cu2"]                        # each atom's weight of Cu2
atoms.calc = PyDFTB(skdir, kpts=(4, 4, 1), type_switching=[sw])
```

The types of an element need equal on-site energies, Hubbard values, occupations and basis
(PTBP's types have them); the files between two types must not start with an on-site line
(`pydftb/validation/make_cu_types_set.py` makes such a set with hotcent).  NumPy engine only;
not with Casida, XL-BOMD or the analysis outputs.  Details:
[`docs/reports/M5a-type-switching.html`](docs/reports/M5a-type-switching.html).

### Shells from several SK files (SelectedShells)

DFTB+'s `MaxAngularMomentum { O = SelectedShells { "sp" "s" } }` builds an atom's basis from
several SK files, so one angular momentum can appear twice (an sps* basis); milestone M6f-5:

```python
p = ParamSet(skdir, selected_shells={"O": ("sp", "s")})   # the s group from O2-O2.skf, O-O2.skf, O2-H.skf ...
p = ParamSet(skdir, selected_shells={"O": (("O", "sp"), ("Ox", "s"))})   # name each group's files
r = DFTB(p.skset(["O", "H"]), reference_occupations={0: {"s2": 0.0}}).calculate(labels, pos)
atoms.calc = PyDFTB(skdir, selected_shells={"Cu": ("sd", "p")})
```

The k-th group's files are named after the label followed by k (DFTB+'s manual: Si, Si2), or
after the file label given.  As in DFTB+: the orbitals follow the groups in order ("sd" "p" is
s, d, p); each shell's on-site energy, occupation and U come from its group's homonuclear file;
atom-resolved SCC uses the first shell's U; the repulsive and the mass come from the first file;
the files between two groups of one label are heteronuclear (no on-site line;
`pydftb.skf.write_typed_copies` writes them so); all files of a label pair share one grid.  A
repeated angular momentum is named "s2" (`reference_occupations`, eigenvec.out).  Spin constants
and shell-resolved Hubbard derivatives follow the shells' order.  NumPy engine only; not with
spin-orbit coupling, DFTB+U, onsite corrections, Casida or type switching.  Details:
[`docs/reports/M6f5-selected-shells.html`](docs/reports/M6f5-selected-shells.html).

### Polarisability and response kernel

DFTB+'s coupled-perturbed response (`Analysis { Polarisability; ResponseKernel }`), static and at
finite frequencies; milestone M6g-1:

```python
r = DFTB(sk, analysis={"polarisability": {"frequencies": [0.05]},   # Hartree; static too
                       "response_kernel": True}).calculate(labels, pos)
p = r.analysis.polarisability
p.tensor[0]                 # static polarisability d mu_j / d E_i (a.u.); None if periodic
p.charge_derivatives[-1]    # (3, n_qm, n_atom) dN_A / dE_i at the last frequency
k = r.analysis.response_kernel
k.atom[0, :, 0, :]          # [A, B] = dN_B / dV_A (static, SCC screened)
print(pydftb.dftbplus_io.response_tables(r))   # DFTB+'s detailed.out blocks
```

The same spectral sum as DFTB+ (pairs of filled and empty levels weighted by
(f_i - f_a)/(e_i - e_a [+- omega + i eta]), DFTB+'s window and degenerate runs, the move of the
Fermi level of metals), solved as a linear system instead of DFTB+'s iterations (DFTB+'s default
`PerturbSccTol` of 1e-5 leaves 2.5e-7 in water's polarisability).  Options: "static",
"frequencies", "frequency_range" (start, stop, step), the kernel's "rpa" (the bare response), and
`analysis["perturbation"] = {"degeneracy_tol": 1e-9, "eta": 1e-8, "filling_tol": None}` (None:
DFTB+'s window, which leaves out levels within 1e-7 of empty or full, about 1e-7 of a metal's
response at 3000 K; 1e-15 keeps every partly filled level).  Also the eigenvalue and
Fermi-level derivatives (dE_F/dE the physical value, DFTB+ prints it divided by the full
filling) and `dftbplus_io.write_band_derivatives` (dE_band.out, dV_band.out).  With collinear
spin, metals, degenerate levels, the full third order, k-points (charge response only; the
finite-frequency eta term that DFTB+ keeps for time-reversal reduced k-point sets is dropped, so
reduced and full sets agree), dispersion, DFTB+U and onsite corrections (also both together,
which DFTB+ refuses; milestone M6g-3a: the atom blocks' elements become response channels).
NumPy engine only; not with non-Fermi smearing, solvation, two-component spin (milestone
M6g-3b), the dipole correction or selected shells.  Details:
[`docs/reports/M6g1-response.html`](docs/reports/M6g1-response.html).

### Coordinate derivatives, Hessian and normal modes

DFTB+'s `Analysis { CoordDerivatives }`, `Driver = SecondDerivatives` and its `modes` program;
milestone M6g-2:

```python
from pydftb.vibrations import second_derivatives, normal_modes, masses_of

r = DFTB(sk, analysis={"coord_derivatives": True}).calculate(labels, pos)
c = r.analysis.coord_derivatives
c.born                      # (n_K, 3, 3) [K, i, j] = d mu_j / d x_Ki (physical sign, e)
c.atom                      # (n_K, 3, n_qm, n_atom) dN_B / dx_Ki (e/Bohr)
c.eigenvalue_derivatives    # (n_K, 3, n_block, n_level) d eps / dx (Ha/Bohr)

calc = DFTB(sk, analysis={"polarisability": True})        # also d alpha / dx
sd = second_derivatives(calc, labels, pos)                # 6N + 1 calculations
nm = normal_modes(sd.hessian, masses_of(calc, labels), pos, born=sd.dipole_derivatives,
                  born_derivatives=sd.polarisability_derivatives,
                  remove_translation=True, remove_rotation=True)
nm.wavenumbers, nm.ir, nm.raman                            # cm^-1 (imaginary: negative)
```

The coordinate response builds on the polarisability's: the derivatives of H0 and S, of the
ground state's potentials and of gamma, the third order, the H5 correction, external fields
and point charges, with the full finite-temperature first-order density (Fermi-surface terms,
the move of a metal's Fermi level, degenerate runs).  So it also covers what DFTB+ refuses
(spin, metals, degenerate levels) or gets wrong (fields and point charges, whose position
derivatives DFTB+ leaves out, and the third order in 25.1); DFTB+'s Born charges are
-d mu / dx (`dftbplus_io.coord_derivative_tables` writes them that way).  `{"atoms": [...]}`
displaces only some atoms.  Also with DFTB+U and onsite corrections, which DFTB+ refuses
(milestone M6g-3a).  Molecules only (as DFTB+); NumPy engine only; not with two-component spin
(milestone M6g-3b).  `second_derivatives` follows
DFTB+'s order of displacements (`atoms`, `moved_atoms` for partial Hessians, `delta` 1e-4 Bohr)
and only calls `calculate`, so it works for anything that gives forces; `normal_modes`
follows `modes` (projection, intensities; it symmetrises the Hessian unless
`triangle="modes"`).  DFTB+'s formats: `write_band_derivatives(path, r, "coordinate")`,
`write_hessian`, `write_born`, `write_born_derivatives`, `write_vibrations_tag`.  Finite
differences see the jump of the SK interpolation's derivative at grid points (water's H-H
distance lies 4.8e-6 Bohr from one).  Details:
[`docs/reports/M6g2-coord-hessian.html`](docs/reports/M6g2-coord-hessian.html).

### Derivatives with respect to the point charges' positions

The development DFTB+'s `Analysis { ExternalChargeDerivatives }`; milestone M6g-3a:

```python
calc = DFTB(sk, point_charges={"positions": R, "charges": Q},       # Bohr
            analysis={"external_charge_derivatives": True})        # or {"charges": [0, 2]}
r = calc.calculate(labels, pos)
e = r.analysis.external_charge_derivatives
e.atom                      # (n_j, 3, n_qm, n_atom) dN_B / dR_j,c (e/Bohr)
print(pydftb.dftbplus_io.external_charge_tables(r))   # the development DFTB+'s block
```

Moving a point charge only changes the external potential on the atoms (H0 and S stay), so it
is solved like the field; molecules and periodic systems (Ewald), with collinear spin, metals,
DFTB3, DFTB+U and onsite corrections.  DFTB+'s tag and detailed.out block are dN for molecules
and -dN for periodic systems (its Ewald derivative is taken with respect to the atom); pydftb
gives dN for both.  NumPy engine only.  Details:
[`docs/reports/M6g3a-blocks-external.html`](docs/reports/M6g3a-blocks-external.html).

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

### SCC start, mixers and DFTB+ charge files

DFTB+'s four mixers are exact ports, so the SCC takes the same iterations as in DFTB+; the
error of every iteration (DFTB+'s "SCC error" column) is in `r.scc_errors`:

```python
DFTB(sk, mixer="anderson")                  # "broyden" (default), "simple", "anderson", "diis"
DFTB(sk, mixer={"kind": "diis", "generations": 4, "use_from_start": False})
DFTB(sk, mixer={"kind": "anderson", "dynamic_mixing": [(1e-2, 0.3)]}, mixing=0.1)
```

The options are DFTB+'s keywords in snake case (`inverse_jacobi_weight`, `init_mixing`,
`diagonal_rescaling`, ...); `mixing` (DFTB+ `MixingParameter`) defaults to the mixer's DFTB+
default (Broyden 0.2, as before).  **Note:** `mixer="anderson"` now means DFTB+'s Anderson
mixer; before M6f it was an alias of pydftb's Pulay-type mixer, which stays `mixer="pulay"`.

The start of the SCC as DFTB+'s `InitialCharges` and `InitialSpins`, and DFTB+'s charge
restart files (`charges.dat` byte for byte as DFTB+ writes it):

```python
DFTB(sk, initial_charges={0: 0.3, 1: -0.3})          # electrons fill each atom's shells in order
DFTB(sk, spin=True, unpaired_electrons=0, initial_spins=[1, 0, -1, 0])   # an antiferromagnetic start
from pydftb import dftbplus_io
r = calc.calculate(labels, pos, q_init=dftbplus_io.read_charges("charges.bin", norb_atom))
dftbplus_io.write_charges("charges.dat", calc.charge_data(labels, r))
atoms.calc = PyDFTB(skdir, read_charges="charges.bin")      # and atoms.calc.write_charges(path)
```

With collinear spin, `PyDFTB` starts from the atoms' initial magnetic moments when they are
not all zero.  NumPy engine only.  Details:
[`docs/reports/M6f1-scc-start-mixers.html`](docs/reports/M6f1-scc-start-mixers.html).

### Fixed Fermi levels and overrides of the Hamiltonian block

DFTB+'s `FixedFermiLevel`, `IndependentKFilling`, `CustomisedHubbards`,
`CustomisedOccupations` and `TruncateSKRange`:

```python
r = DFTB(sk, smearing=("fermi", 300), fixed_fermi_level=-0.14).calculate(labels, pos)
r.e_force_related, r.n_electrons     # F - N E_f (what the forces are the gradient of), N
DFTB(sk, spin=True, ..., fixed_fermi_level=(-0.13, -0.24))     # up, down
DFTB(sk, kpts=..., independent_k_filling=True)                 # every k-point on its own
DFTB(sk, hubbard={"O": 0.5})                    # one U per type, or per shell; <= 0 keeps the SKF's
DFTB(sk, reference_occupations={1: {"p": 4.2}})  # atom -> shell electrons: q0 and N change
DFTB(sk, truncate_sk_range=6.0)                  # Bohr; {"max_distance": 6.0, "hard_cutoff": False}
```

With a fixed Fermi level the electron count follows the SCC (a reservoir); `energy` and
`free_energy` keep DFTB+'s meaning and the forces and stress are those of `e_force_related`
(also for the charged periodic cells, where DFTB+'s stress is 1e-8 to 4e-7 off).  `PyDFTB`
takes the Fermi level in eV and the truncation in Angstrom, and reports `e_force_related` as
the force-consistent `free_energy`.  NumPy engine only.  Details:
[`docs/reports/M6f2-occupations-overrides.html`](docs/reports/M6f2-occupations-overrides.html).

### H5, halogen and dipole corrections

DFTB+'s `HCorrection = H5`, `HalogenXCorr` (DFTB3) and `DipoleCorrection` (slabs; DFTB+ after
25.1):

```python
DFTB(sk, h5=True)                                # the paper's values; or {"r_scaling", "w_scaling",
                                                 #   "scaling": {"O": 0.06, "N": 0.18, "S": 0.21}}
r = DFTB(sk, third_order="full", hubbard_derivs=ud, halogen_correction=True).calculate(labels, pos)
r.e_halogen                                      # O/N - Cl/Br/I pairs, in energy and free_energy
r = DFTB(sk, kpts=(k, w), dipole_correction=True).calculate(labels, pos, cell, pbc)
r.e_dipole_corr                                  # 2 pi D^2 / V, a part of e_scc as in DFTB+
DFTB(sk, kpts=(k, w), dipole_correction={"layer_position": 9.45})      # Bohr; PyDFTB: Angstrom
```

Forces and stress are the derivatives of the energy, also where DFTB+'s are not: the halogen
correction's switching region (DFTB+ has the wrong sign there), excited-state forces with H5
(DFTB+ leaves H5 out of gamma') and the dipole correction's stress (DFTB+ has no stress term).
H5 and the halogen correction recognise elements, DFTB+ species names (type labels such as
`O2` differ).  NumPy engine only.  Details:
[`docs/reports/M6f3-corrections.html`](docs/reports/M6f3-corrections.html).

### Analysis outputs

DFTB+'s `Analysis` outputs (atom-resolved energies, net populations, CM5 charges, the
electrostatic potential, eigenvectors) and its files:

```python
from pydftb import DFTB, Analysis, dftbplus_io as io

r = DFTB(sk, analysis=True).calculate(labels, pos)   # atom energies, net populations, CM5
a = r.analysis
a.atom_energies["total"]                 # (n_atom,), sums to r.energy; also "electronic",
                                         # "repulsive", "dispersion" and every term ("h0", ...)
a.net_populations, a.hybridisation, a.cm5_charges
r = DFTB(sk, analysis={"esp": {"points": pts}, "eigenvectors": True}).calculate(labels, pos)
r.analysis.esp.internal                  # Hartree/e (physical potential), points in Bohr
io.write_esp("ESP.dat", r.analysis.esp)  # DFTB+'s formats: ESP.dat (V), eigenvec.out/.bin,
io.write_eigenvectors("eigenvec.out", r) # band.out and the analysis blocks of detailed.out
io.write_band_out("band.out", r)
print(io.detailed_tables(r))
atoms.calc = PyDFTB(skdir, analysis={"atom_energies": True})
atoms.get_potential_energies()           # eV, sums to the potential energy
```

Every term is split over the atoms as DFTB+ does, except the D3 library's dispersion energy,
which DFTB+ spreads evenly (E/N) and pydftb resolves by atom pairs.  Grids:
`{"grid": {"spacing", "origin", "points", "directions"}}`, the last index running fastest,
or `"fractional": True` in a periodic cell.  NumPy engine only.  Details:
[`docs/reports/M6f4a-analysis.html`](docs/reports/M6f4a-analysis.html).

Pipek-Mezey localised orbitals, bond populations, bond energies and Mayer bond orders, and
Casida transition charges (milestone M6f-4b):

```python
r = DFTB(sk, analysis={"localise": True,  # or {"max_iterations": 1000, "tolerance": 1e-8}
                       "bond_populations": True, "bond_energies": True,
                       "bond_orders": True}).calculate(labels, pos)
loc = r.analysis.localisation
loc.orbitals[0][:, :loc.n_localised[0]]  # localised orbitals (columns) of the first spin channel
loc.populations[0], loc.value            # their Mulliken populations; DFTB+'s pm_localisation
r.analysis.bond_populations              # (n_atom, n_atom): rows sum to the Mulliken populations
r.analysis.bond_energies, r.analysis.bond_orders   # sum to E_H0; Mayer orders (molecules)
io.write_eigenvectors("localOrbs.out", r, localised=True)
io.write_bond_files(r, ".")              # bond_pop.dat, bond_energy.dat, bond_order.dat

from pydftb.casida import Casida
r = DFTB(sk, casida=Casida(6, state_of_interest=2, transition_charges=True)).calculate(labels, pos)
r.excitations.transition_charges         # (n_atom,), overall sign arbitrary (as in DFTB+)
io.write_transition_charges("ATQ.DAT", r)
```

The localisation is DFTB+'s dense algorithm, step by step; it needs real orbitals (molecules,
the Gamma point): DFTB+'s k-point version multiplies unconjugated coefficients, so its result
depends on the eigensolver's arbitrary phases.  Details:
[`docs/reports/M6f4b-localisation.html`](docs/reports/M6f4b-localisation.html).

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
`pydftb dynamics --check`; details in
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

`pydftb advisor RUN_DIR` reads a DFTB+ run -- `dftb_in.hsd`, and if present
`dftb_pin.hsd` (what DFTB+ really used), the screen output and the SKF files -- and reports
the pitfalls found while validating pydftb against DFTB+, with the file and line of the
evidence and advice.  It only reads; DFTB+ is not needed.

```text
$ pydftb advisor run/
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
optional (the `torch` extra, or the CPU build:
`pip install "torch>=2.14" --index-url https://download.pytorch.org/whl/cpu`);
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
(`pydftb gradcheck --check`, 11 cases, 346 comparisons) the median
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
pydftb gpu-check --json gpu_check.json      # first GPU, everything
pydftb gpu-check --device cuda:1 --dtype both --json gpu_check.json
pydftb gpu-check --quick                    # a subset, a few minutes
```

It recomputes the frozen DFTB+ references on the device and compares them with torch on the
CPU (float64: within 1e-10), runs calculations twice (bit for bit), compares autodiff
gradients of both modes with the CPU (1e-9), checks that a process that used CUDA can fork a
CPU-only worker, and times `eigh` (N = 9-288, batches of 1-2048), Cu cells as PTBP computes
them, and stacks of several structures.  It exits with status 1 if a check fails; please
send the JSON back.  See [`docs/reports/M4d-devices.html`](docs/reports/M4d-devices.html).

## Tests and validation

```bash
python -m pytest --pyargs pydftb.tests   # offline, uses frozen DFTB+ references (any folder)
PYTHONPATH=. pytest pydftb/tests          # the same at the repository root, without installing
pydftb refgen --check                     # table of deviations from DFTB+
pydftb refgen --regenerate                # needs dftb+; rewrites the references (in a checkout)
pydftb refgen --live                      # fresh comparison with dftb+
pydftb dynamics --check                   # optimisation and MD vs DFTB+'s drivers
pydftb refgen --check --backend torch     # the same on the torch backend
pydftb golden --base COMMIT               # NumPy path bitwise unchanged since COMMIT (checkout)
pydftb gradcheck --check                  # pydftb.autodiff vs finite differences
pydftb gpu-check --device cpu --quick     # the GPU self-check, on the CPU
```
