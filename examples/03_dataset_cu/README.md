# Example 3 — dataset workflow on Cu with auto-fit E0s

Same Cu dataset as example 2, but used as a flat dataset (no EOS
slicing). Loss is `(1−c)·μ_E + c·μ_F` over per-structure formation
energies. `--E0s` is intentionally omitted to demonstrate the auto-fit +
on-disk cache flow.

```
./run.sh        # first time: fits E0s, writes dft/E0s_dft.json
./run.sh        # second time: cache hit, no refit
```

The Cu dataset auto-fit yields E0[Cu] ≈ -45 245 eV. Compared with a
hand-fed wrong value (e.g. -22 622), the fitted E0 brings the inner-loop
μ down by ~3 orders of magnitude, because the constant offset between
DFT-side and DFTB-side formation energies cancels.

Output:
- `dft/E0s_dft.json` — cached fit `{29: -45244.94}`.
- `par.out` / `log.out` / `results/opt_*/` as in example 2.
