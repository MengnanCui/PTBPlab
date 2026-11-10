# Li / Li–O data generators (A series, A0 section 2)

These scripts write DFT inputs and band references for A1 (Li and Li–O). None of them submits a job. Each one writes `job_XX.sh`, `submit_all.sh` and `cost_estimate.json`, and you submit by hand after checking the cost.

Every run logs to `<out>/logs/<name>/<time>-<host>-<pid>/` (`utils/runlog.py`):
- `run.log` (stdout and stderr);
- `meta.json` (git commit, package versions, host, arguments);
- `events.jsonl`;
- `status.json`.

DFT settings are those of the M5c/M5d training data (`examples/07_battery_li_li2o/vasp_common.py`):
- VASP 6.6, PBE, Li_sv/O;
- ENCUT 800 eV, PREC Accurate, EDIFF 1e-7;
- ISMEAR 0 with SIGMA 0.05, ISPIN 1;
- KSPACING 0.15 Å⁻¹, Γ-centred. Interface frames use 0.25 with ISYM 0.

| Script | Batch | Runs where | What it writes | Cost |
|---|---|---|---|---|
| `bands_gpaw.py` | D1 (GPAW) | local GPAW 26.7 (conda env `gpaw`) or the server | `bands/<name>.json` (pydftb `BandRef`) for 15 crystals (bcc/fcc/hcp Li, Li₂O, Li₂O₂ at 0.952/1.000/1.048 V₀) and 15 dimers (Li₂, LiO, O₂ at 5 distances); `bands/energies.json` with the SCF energies (cross-code check only) | 0 yuan; about 1–2 days on 10 cores (PW 800 eV) |
| `vasp_bands.py` | D1 (VASP) | cluster | per crystal `scf/` (LCHARG) + `band/` (ICHARG 11, explicit k-points, `kpath.json`); per dimer `scf/` (Γ) | rough estimate in `cost_estimate.json` (a few yuan) |
| `collect_bands.py` | D1 | anywhere | `BandRef` JSON from finished VASP band runs (k-point order checked, Fermi level from the SCF, `n_semicore` from ZVAL); checks and copies GPAW JSON | — |
| `make_d2.py write` | D2 | cluster | about 80 single points, unrelaxed or rattled, plus `manifest.csv` (name, family, role, k-grid, estimated core-h) | about 30–40 yuan (`--dry-run` prints the table) |
| `make_d3.py sample` | D3 | server (mace-torch; GPU optional) | MACE-MP-0 Langevin trajectories of the 80-atom interface and two ±2 % strained variants at 300/600/900 K, in `--traj-dir` (default `$PTBP_OFFLOAD/pydftb/a1_interface`, a directory git ignores) | GPU or CPU time only |
| `make_d3.py select` | D3 | anywhere | `selected.extxyz`: farthest-point sampling of 80 frames on RDF + Li–O coordination descriptors; `--extra-traj` adds DFTB collapse frames | — |
| `make_d3.py write` | D3 | cluster | VASP single points of the selected frames, `manifest.csv`, job scripts | about 1.5 yuan per frame (M5c price), 80 frames about 120 yuan |

## Order

1. `python bands_gpaw.py --out ../runs/d1_li_o --dry-run`, then on the GPAW machine run `mpirun -np 8 gpaw python bands_gpaw.py --out ../runs/d1_li_o`. Point `ptbp scan` at the result with `bands.refs: [../runs/d1_li_o/bands/*.json]`.
   - The GPAW and VASP energies are never mixed into one fit.
   - `energies.json` only checks the GPAW relative energies along each EOS against `train.xyz`.
2. Optionally `python vasp_bands.py --out ../runs/d1_vasp`. On the cluster, check the cost and run `bash submit_all.sh`. Then `python collect_bands.py --vasp ../runs/d1_vasp --out ../data/bands_vasp`. This gives band references from the same code as the training energies.
3. `python make_d2.py write --out ../runs/d2_vasp --dry-run`, then without `--dry-run`. Submit by hand.
4. On the server, `python make_d3.py sample --out ../runs/d3 --device cuda`, then `select`, then `write`. Submit by hand.

## Collecting D2 and D3 energies and forces

`examples/07_battery_li_li2o/collect.py RUNS --out DIR` reads any single-point directory: `energy(sigma->0)` and forces from OUTCAR. Two things in it do not fit the new data:
- it names groups by fixed prefixes, so every new family comes out as `other`;
- it splits train and test by whether the name contains `test`.

Give it its own `--out`, then set the tags from the manifest:

```bash
python ../../07_battery_li_li2o/collect.py ../runs/d2_vasp --out ../runs/d2_collected
python make_d2.py retag --manifest ../runs/d2_vasp/manifest.csv \
    --frames ../runs/d2_collected/train.xyz ../runs/d2_collected/test.xyz --out ../data/d2.xyz
```

`retag` sets `name`, `family`, `role` (train/test/ood, from `manifest.csv`), `kspacing` and `source`. `ptbp scan` reads `kspacing` to build the same Γ-centred k-grid for DFTB. D3 works the same way with `../runs/d3/vasp/manifest.csv`.

## Roles

**D2:**
- `ood` (transfer tests): hcp EOS and rattled hcp, Li and Li₂O surfaces, the vacancy and the 54-atom cell, and the Li₂O point defects.
- `train`: Li₂O₂ rattles and the hot and compressed rattles. Their seed-101 copies are `test`.

**D3:**
- `ood`: frames of the unstrained interface and of `--extra-traj`.
- `train`: frames of the strained variants. `--ood-variants` changes the split.
