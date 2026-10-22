# Example 7 — band-structure optimisation on Cu

Fits the confinement of the Cu pseudo-atomic orbitals, V_conf = (r/r0)^p
(the same r0 for the wavefunctions and the density), to the DFT band
structure of fcc Cu.

```
./run.sh
```

Inputs (in `ref/`):
- `cu_fcc.xyz` — the structure: fcc Cu at a = 3.637 Å, the centre of the
  bundled EOS scan (`dft/dft.xyz`), with `structure_name=fcc`.
- `cu_fcc_band.json` — its band structure from GPAW (PBE, 500 eV plane waves,
  12×12×12 k-points), stored as an ase `BandStructure`: 101 k-points along
  W-X-Γ-L-K-Γ-L, 16 bands, referenced to the Fermi level.
  `make_reference.py` regenerates it (needs `gpaw`).

Each Bayesian-optimisation step (r0, p):
1. hotcent writes `Cu-Cu.skf`;
2. DFTB+ runs self-consistently on a Monkhorst-Pack grid (`--kpt_density`);
3. DFTB+ computes the bands at the k-points of the reference, in that
   self-consistent potential;
4. the loss is μ = μ_homo + μ_lumo. Each term is the mean |E_DFT − E_DFTB| of
   one band over the k-points where the DFT band lies within ±1 eV of the Fermi
   level (of the valence-band maximum, for a reference with a gap). The bands
   are the highest occupied one (band 5: Cu has 11 valence electrons, 3d10 4s1)
   and the next one. For Cu only band 5 comes within 1 eV of the Fermi level,
   so μ_lumo is 0.

The search starts at the QUASINANO2013 parameters of Cu, (r0, p) = (5.2, 2.2),
within r0 ∈ [2.1, 7·r_cov] Bohr and p ∈ [1.01, 11]. Parameters at which hotcent
or DFTB+ fail score μ = 20. At large r0, for instance, the orbitals are so
diffuse that the overlap matrix is not positive definite.

Expected runtime: about 1 min per step, so ~13 min for the 11 steps
(4 cores; hotcent dominates). In our run μ went from 1.49 eV at the seed to
0.55 eV at (r0, p) = (8.41, 6.80). The six steps with r0 ≥ 9.6 Bohr failed
(non-positive-definite overlap) and scored 20; pass `--n_calls` for a longer
search.

Output (in `./run/`):
- the console log: the DFTB+ Fermi levels and μ of every step, then
  `Best parameter: [r0, p] μ`;
- `convergence.png`, `eval.png`, `obj.png` — skopt plots;
- `opt_<r0>_<p>/` — the best step:
  - `Cu-Cu.skf` and the DFTB+ files;
  - `fcc_<r0>_<r0>_<p>.json` — the DFTB band structure;
  - `pic_*.png` — DFT vs DFTB bands 5 and 6;
  - `par.out` — r0_w, r0_d, p, r_cov, μ_homo, μ_lumo, μ;
- `results.tgz` — every step.

Requirements:
- `dftb+` on `PATH`. With ase ≥ 3.23, `DFTB_COMMAND` sets another command;
  `ASE_DFTB_COMMAND` is no longer read.
- `DFTB_PREFIX` unset (`run.sh` unsets it).

## Your own reference

Any ase `BandStructure` json of one structure works, e.g.
`calc.band_structure().write('bands.json')`. DFTB+ is run at its k-points,
so the two band structures are compared point by point. At the start of
every step PTBP checks that the reference:
- has the lattice parameters of the structure in `--dft_file`;
- has at least N/2 + 1 bands, for N valence electrons;
- counts the same electrons as valence as PTBP's valence shells
  (`utils/parameters_set.py`). A reference with semicore states fails this
  check.
