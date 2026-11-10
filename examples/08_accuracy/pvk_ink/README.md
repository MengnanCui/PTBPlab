# A1.3 — Pb–Sn perovskite precursor ink: data, references and the single-set scan

The design is `pydftb/docs/design/A0-accuracy-programme.html`, sections 2–3 and 7.

The system is the M409 ink of the perovskite ink project (`PTBP_PEROVSKITE`): Cs₀.₂₅FA₀.₇₅Pb₀.₅Sn₀.₅I₃ in DMSO/DMF, with the elements H C N O S I Cs Sn Pb.  By the user's decision (2026-11-09) the precursor solution comes first.  Its "bands" are the orbital energies of molecules and clusters from HOMO − 3 eV to LUMO + 1 eV.

Nothing here submits a job.  Every script logs to `<out>/logs/<script>/<time>-<host>-<pid>/` (run.log, meta.json with the git commit and package versions, events.jsonl, status.json).

## Order of the steps

| step | script | where | output | cost |
|---|---|---|---|---|
| 1 | `prepare_structures.py` | anywhere (seconds) | `data/structures.xyz` (148 structures), `manifest.csv`, `exclusions.csv` | 0 |
| 2 | `orbitals_gpaw.py` | locally or on the server, GPAW 26 (`conda activate gpaw`), `mpirun -np 8 gpaw python orbitals_gpaw.py` | `bands/<name>.json` (BandRef), `bands/energies.json` | about 0.1–2 h per complex on 8 cores; `--dry-run` lists them |
| 3 | `make_d4.py` | anywhere | `d4/{sp,relax,atoms}/…`, `job_NN.sh`, `run_cases.sh`, `submit_all.sh`, `cost_estimate.json` | about 510 core-h, about 51 yuan list price (rough: calibrate first) |
| 4 | (user) `bash d4/submit_all.sh` on dghpc | dg83 | OUTCARs | — |
| 5 | `collect_d4.py d4 [--bands] [--d3]` | anywhere | `data/structures_dft_{train,test,ood}.xyz`, `data/E0s_dft.json`, `bands_vasp/` | 0 |
| 6 | `k5_fa_iodide_gap.py --jobs 8` | server (hotcent + pydftb) | `k5/k5_summary.md`, `k5_results.json` | 25 hotcent runs |
| 7 | `ptbp scan points/run/losses/report scan_ink.yaml` | server | `runs/scan_ink/report.md`, `losses.csv` | 129 points × (45 SK tables + about 300 DFTB) |
| 8 | `make_d5.py` (needs the M409 trajectories) | anywhere | `d5/clusters.xyz`, VASP inputs | about 3000 core-h, about 300 yuan (rough) |
| 9 | (user) `bash d5/submit_all.sh`, then `collect_d4.py d5 --subdirs vasp --name clusters_dft --out data_d5` | dg83 | `data_d5/clusters_dft_*.xyz` | — |

Step 2 (local GPAW) and step 3 (VASP inputs) are independent of each other.  K5 (step 6) needs only the structure `ion_fa_i` and, for the DFT gap, `bands/ion_fa_i.json`.

## The data set (`prepare_structures.py`)

| family | what | role |
|---|---|---|
| `mol_<id>_<name>` | 14 neutral molecules of `data/stage1/molecule_list.csv` (relaxed with MACE-OMOL), each with 2 rattled copies (σ 0.05 Å) | train; the optional solvents R15–R20 are test |
| `cplx_<M>I2_<L><n>` | 36 MI₂(L)ₙ complexes, M = Pb/Sn, L = DMSO/DMF, n = 1–3, 3 orientations (MACE-OMOL, else MACE-MPA) | n = 1, 2 train; n = 3 test |
| `ion_*` | FA⁺·I⁻, CsI, FA⁺·I⁻·DMSO, FA⁺·I⁻·DMF, PbI₃⁻·FA⁺, SnI₃⁻·FA⁺, PbI₄²⁻·2FA⁺, SnI₄²⁻·2FA⁺ (+1 rattled copy each; built with N–H···I 2.6 Å) | train; the last two are test |
| `scan_pbo`, `scan_sno` | M–O of MI₂(DMSO), 9 points, ligand moved rigidly (d₀ − 0.35 to d₀ + 2.0 Å) | train |
| `scan_pbi`, `scan_sni` | M–I of MI₂, 9 points (2.5–4.6 Å) | train |
| `scan_pt_fa_i` | FA⁺→I⁻ proton transfer: H on the N···I line, d(N–H) from 1.02 Å to d(N···I) − 1.62 Å | train |
| `scan_pt_fa_o` | FA⁺→O(DMSO) proton transfer in FA⁺·I⁻·DMSO | test |

The charged species (FA⁺, Cs⁺, I⁻, PbI₃⁻, SnI₃⁻) and SnF₂ (F is not in the model) are left out; `exclusions.csv` lists them.  Every structure sits in a cubic box with 8 Å of vacuum and pbc False: VASP adds the dipole correction, and DFTB treats it as a molecule.

## DFT conventions

VASP PBE with Pb_d, Sn_d, Cs_sv, I, S, O, N, C, H; ENCUT 520; the Γ point (vasp_gam); ISPIN 1; ISMEAR 0 with SIGMA 0.01; EDIFF 1e-6.

There is no D3 in the references.  The fits compare plain PBE with DFTB, and the same D3(BJ) is added to both sides when the models are used; `collect_d4.py --d3` stores the D3 terms for checks.  Isolated atoms are spin-unpolarized, in a 20 × 20.5 × 21 Å box.

GPAW (D1) uses PBE in PW mode at 600 eV, Γ, Gaussian 0.01 eV and an orthorhombic box with 6 Å of vacuum; its energies are never mixed with the VASP energies.  `run_cases.sh` reads `VASP_BIN_DIR`, `VASP_PP_PATH` and `VASP_MODULES` (the defaults are the dghpc paths).

## The scan (`scan_ink.yaml`)

The scan has these settings:

- **Space:** fit (r0_w, r0_d) of I, Pb, Sn, Cs in a box of 0.7–1.4 × the published values, with 128 Sobol points plus the published set; H C N O S stay as published.
- **max_l:** S, I, Sn, Cs = d; Pb = p (the shells PTBP's files have).
- **Strategies:** S1, S1o, S3, S4.  S2 needs EOS families, which this data has none of.
- **Reactions:** `reactions_ink.yaml`, 28 reactions:
  - binding of MI₂(L)ₙ;
  - the Pb/Sn selectivity for DMSO (decisive);
  - solvent exchange;
  - the two proton transfers (decisive);
  - iodide affinities of Pb and Sn.
- **R₂b control:** in utils/accuracy.py its splines are the whole repulsive, so `repfit.pairs` lists all 40 pairs that occur.

## Open points

- The ion clusters and the proton-transfer scans are hand-built (N–H···I 2.6 Å, linear proton paths with N and X fixed).  D4's relaxations give their relaxed versions (`*_relaxed`), and the reactions take the lowest structure of each family.
- The cost model (3 core-h per 40 atoms, ∝ N²) is a guess: run one complex and one cluster first.
- D5 needs `runs/M409/T02/{mace_mpa_B,mace_mpa_A2}/traj.extxyz`.  They are 218 MB each and are not in the repository, so they must be copied to the server.
