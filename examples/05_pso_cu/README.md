# Example 5 — PSO optimiser on Cu

Same Cu dataset as example 3, but uses Particle Swarm Optimisation
(via `pyswarms.single.GlobalBestPSO`) instead of sequential
Gaussian-process Bayesian optimisation. Useful when the loss surface
has multiple basins / non-smooth regions where the GP smoothness
assumption hurts.

```
./run.sh
```

Settings:
- `--n_particles 4` — swarm size
- `--n_calls 12` — total evaluation budget (3 PSO iterations of 4 each)
- pyswarms hyperparameters: `c1=0.5, c2=0.3, w=0.9` (standard defaults)

Output (same shape as the BO examples):
- `convergence.png` — best-so-far μ vs. evaluation index
- `result.pkl` — the optimisation snapshot, reusable via
  `--postprocess_only` to redo plots/repackaging without rerunning PSO
- `results/opt_<r0_w>_<r0_d>/` — one folder per evaluated particle
  position, with the SKF and DFTB+ artefacts
- `opt_<best_r0_w>_<best_r0_d>/` — copy of the best-particle folder
- `results.tgz` — packed archive

### Re-running just post-processing

If only the plotting / packaging step failed, or you want to redo the
output with no optimiser call:

```
./run.sh --postprocess_only           # ← any extra args go to ptbp
```
