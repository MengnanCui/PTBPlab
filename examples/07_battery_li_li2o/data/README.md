# Li | Li₂O DFT data (M5c, M5d)

The DFT settings are the same for every file: VASP 6.6.0, PBE (`potpaw_PBE.54`, Li_sv and O), PREC Accurate, ENCUT 800 eV, EDIFF 1e-7, ISMEAR 0 with SIGMA 0.05, ISPIN 1. Settings that change between files are given in the table.

- `energy` is VASP's `energy(sigma->0)` in eV, and `forces` are in eV/Å; ASE reads both into `atoms.calc`.
- Each structure has `structure_name` and `group` in `atoms.info`.
- `collect.py` builds these files from the cluster runs.

| File | Structures | Contents | k-points | Origin |
|---|---|---|---|---|
| `train.xyz` | 68 | 11-point EOS (0.88–1.12 V₀) for bcc Li, fcc Li, Li₂O, Li₂O₂ (44), plus rattled cells at σ 0.05/0.10 Å with seeds 1–3: bcc Li16, Li₂O 12-atom, Li16O, Li8O3 (24) | KSPACING 0.15, Γ-centred | M5c stages 1–3 |
| `test.xyz` | 8 | the same four rattled groups with seed 101, 2 each | KSPACING 0.15 | M5c |
| `short_range.xyz` | 43 | Li₂, LiO and O₂ dimers from 1.0 to 3.5 Å (Γ point, ISYM 0, in the 15 × 15.5 × 16 Å box of the isolated atoms), and bcc Li, Li₂O, Li₂O₂ compressed to 0.70/0.75/0.80/0.85 V₀ | Γ for the dimers, KSPACING 0.15 for the compressed cells | M5d, jobs 981219 and 981265 (about 0.60 yuan billed) |
| `interface.xyz` | 3 | the 80-atom Li64O16 interface: the initial structure, plus the collapsed snapshots md02 and md04 from a DFTB MD | KSPACING 0.25, ISYM 0 | M5c interface check (4.49 yuan billed) |
| `E0s_dft.json` | — | isolated atoms without spin polarization (Li −0.01707, O −0.05047 eV), keyed by atomic number. These are the reference for the atomization energies, the same convention as DFTB's neutral, unpolarized atoms | Γ | M5c |
| `E0s_spin_polarized.json` | — | spin-polarized isolated atoms (Li −0.2974, O −1.5468 eV), kept for the record | Γ | M5c |

The dimers and the compressed cells in `short_range.xyz` are also unpolarized (ISPIN 1). The O₂ curve is therefore the closed-shell singlet, which is consistent with spin-free DFTB.

`short_range.xyz` and `interface.xyz` were copied on 2026-11-09 from the session scratch directory, where `collect.py` had written them. Before that they were not in the repository.
