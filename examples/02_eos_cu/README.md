# Example 2 — EOS workflow on Cu (the original PTBP recipe)

Reproduces the original PTBP paper workflow on the bundled Cu dataset:
6 phases × 11 EOS volumes = 66 reference structures. Loss is
μ_E + μ_V + μ_B (energy / volume / bulk-modulus mismatch from the
Birch-Murnaghan fit), Boltzmann-weighted across phases.

```
./run.sh
```

Expected runtime: ~5-15 min for the default 11 BO iterations
(cpu-bound on hotcent + DFTB+).

Output:
- `par.out` — best `(r0_w, r0_d, sigma_rep, μ)` per BO iteration.
- `log.out` — full inner-loop trajectory of (μ_E, μ_V, μ_B, penalty, μ).
- `results/opt_<r0_w>_<r0_d>/` — one subfolder per evaluated point with
  the SKF, DFTB+ inputs, and EOS json files.
