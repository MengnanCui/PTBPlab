# S5: refitting the SK tables themselves (optional A1 diagnostic, A0 section 3)

**Question.** A1 changes the SK tables only through the confinement radii. Suppose the best A1 point misses the acceptance line even with a flexible two-body repulsive (the scan's R₂b and oracle rows). Is the limit the confinement-radius parameterisation, or the two-centre form itself?

S5 keeps the two-centre form, lets the H and S tables of that point change freely but smoothly, and refits the repulsive at every step. Nothing is submitted; the Slurm template is `slurm/s5.sh`.

## What is fitted

- **Corrections.** Every SK integral column the basis uses gets a correction c(r) = Σₘ βₘ Bₘ(r), H and S separately, per element pair and channel (ss0, sp0, pp0, pp1, …).
  - The Bₘ are `n_coef` uniform cubic B-splines whose supports lie inside the pair's range `[r_lo, r_hi]`, so c is C² and exactly zero outside.
  - The form is T(r)(1 + c(r)) (`multiplicative`, the default) or T(r) + c(r)·max|T| (`additive`).
- **Shared coefficients.** For A ≠ B, the l₁ = l₂ columns of `A-B.skf` and `B-A.skf` are the same integral and share their coefficients.
- **What never changes.** On-site lines, grids, the Spline/polynomial lines and every other file stay as they are. The start point's files are never touched: each accepted step writes a full new SK set to `steps/kNNN/skf/`.
- **Size.**
  - Li s / O p (`scan_li_o.yaml`): 14 channels, 56 coefficients with `n_coef: 4`.
  - Li sp: 26 channels, 104 coefficients.
- **Penalties.** Ridge on β, and second differences of β within a channel. `beta_max` bounds |β|.

## Repulsive and loss

- **`r2b`** (default): the scan's `repfit` pair splines, refitted on train at every outer step, exactly as the scan's R₂b row.
- **`physrep`**: one σ per fitted element by L_EF, as S1. At step 0 the run prints the scan's own S1 loss for comparison; the two are equal unless `--max-structures` is used.
- **Loss.** Accept/reject decisions and reports use the scan's L_EF on train exactly (`utils.accuracy.loss_ef`). The optimiser sees the same per-structure terms with |x| smoothed by `abs_smoothing`.
- **Band term.** `bands.weight` adds the scan's band loss of the corrected tables to the accept/reject objective. It is value only: it gets no gradient.

## Optimisation and gradients

**Block coordinates with a trust region.** Each outer step:
1. refits the repulsive;
2. runs `inner` L-BFGS-B iterations on β with that repulsive frozen, inside |β − β_k| ≤ trust;
3. accepts the step only if the full objective decreased (repulsive refitted, exact L_EF, penalties, band term). The trust radius grows ×1.5 on accept and halves on reject.

With physrep, the frozen-σ gradient is the gradient of the full objective. With R₂b it is not, which is why each step is checked after the refit.

**Gradients:**

| `optimizer.gradient` | how | needs |
|---|---|---|
| `autodiff` (default) | `pydftb.autodiff`, mode `"implicit"`. The SK tables are leaves of `Params`, and the implicit SCC derivative gives d(E − TS/2)/dβ and d(forces)/dβ exactly, one backward pass per structure | torch (CPU wheel is enough) |
| `fd` | central differences of the full SCC, only in the coefficients whose support holds a pair distance of the structure | nothing extra; 2 SCCs per such coefficient and structure |
| `none` | derivative-free Powell on the inner problem | — |

## Run

From `examples/08_accuracy/li_o`, after the Li–O scan is scored (`ptbp scan run` + `losses`):

```bash
python s5_sk_tables.py dry-run   s5_li_o.yaml --jobs 64   # channels, ranges, parameter count, cost per evaluation
python s5_sk_tables.py gradcheck s5_li_o.yaml             # gradient vs central differences, and its cost
python s5_sk_tables.py fit       s5_li_o.yaml --jobs 64   # rerun the same command to resume (state.json)
python s5_sk_tables.py report    s5_li_o.yaml --jobs 64   # s5_report.md / .json
python s5_sk_tables.py export    s5_li_o.yaml --dest models/S5   # tables + repulsive Spline blocks, model.json
python md_check.py --model models/S5 --out runs/md_S5            # as for the A1 models
```

Or run `sbatch slurm/s5.sh` from `examples/08_accuracy` (one dg83 node).

**Cost.** The dry run takes the time per structure from the scan's own timing of this point. Expect roughly 1–3 node-h for 20 outer steps, i.e. 6–20 yuan at most.

**Logs.** Every stage writes `runs/s5_li_o/logs/s5-<stage>/<time>-<host>-<pid>/`, with `run.log`, `meta.json`, `events.jsonl` (one line per inner evaluation and outer step) and `status.json`.

**Other options.**
- `--gradient` and `--outer` override the config.
- `--restart` starts over and keeps a `.bak` of the old state.
- `--max-structures N` takes only the first N structures of every role (smoke tests only).

## Reading the report

`s5_report.md` puts S5 next to:
- the scan's S1, R₂b and oracle rows at the start point;
- the scan's best S1 point;
- the start tables with the same refit, a consistency check that should reproduce the scan's R₂b row (r2b) or S1 row (physrep) on full data.

Each row gives train/test/ood within-family MAE (meV/atom), force MAE (eV/Å), tier and the decisive signs. It also reports:
- the band loss before and after;
- a file check that the written tables reproduce the in-memory corrections (to about 1e-15 eV).

**Decision (fixed in advance):**

| result | meaning |
|---|---|
| S5 reaches tier A/B on test/ood and the scan's R₂b does not | the confinement-radius parameterisation limits A1 |
| S5 no better than R₂b | the two-centre form plus a two-body repulsive is the limit; go to A2/A3 |
| S5 better on train only | overfitting: raise `penalties`, narrow `ranges` or use `kinds: [H]` |

## Limits

- **Many parameters.** 56–104 against A1's four. Judge only on test/ood.
- **Physical meaning.** The corrected tables are no longer the integrals of any confined basis, so S5 is a diagnostic, not a parameter set to publish.
- **Fixed tables.** S5 changes neither the isolated atoms (no pairs) nor the on-site energies and U. The atomization energies therefore use the start point's DFTB atoms.
- **SCC failures.** The scan's slower mixings are tried; a structure that still fails makes the trial step fail, and it is rejected.

## Local smoke test (2026-11-10)

Scratchpad smoke scan: Li only, 2 coefficients (Li-Li H/S ss0), 2 train structures.

- **Gradcheck.**
  - Autodiff vs central differences: 4.7e-9 (EOS cells and Li₂), and 4e-11 on a rattled periodic Li₁₆ cell with forces. This ran in the conda env `perov`, which has torch.
  - fd: 4.9e-7.
- **Fit.**
  - fd: 4 outer steps, L_EF 84 → 56 (2 structures only).
  - Resume works with `--jobs 2` (spawn) and when switching to autodiff.
  - physrep + Powell + band term + auto ranges ran.
- **Report and export.** Both ran. On-site lines and the rest of the files are unchanged; the table check passed at ≤ 1.8e-15 eV.
