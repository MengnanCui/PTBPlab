# Example 4 — reaction mode on a multi-formula dataset

Demonstrates `--optimization_option reaction` on a real multi-element,
multi-formula dataset: 22 structures of B2N2 (11) + N3Na (11) drawn from
the bundled `nitrides.xyz`.

```
./run.sh
```

The first invocation runs `prepare.py` to extract the subset into
`./dft/dft.xyz`; subsequent invocations skip that step.

Compared to dataset mode, reaction mode adds:

1. **Auto-detection of formulas**, logged at startup:
   ```
   Reaction mode: 2 unique formulas detected:
     B2N2                     11 structures
     N3Na                     11 structures
   ```

2. **Per-formula loss breakdown** written under
   `results/opt_<r0_w>_<r0_d>/reaction_per_formula.csv`, with one block
   per inner-loop sigma_rep evaluation:

   ```
   formula,n_structures,mean_abs_dE_per_atom,max_abs_dE_per_atom
   B2N2,11,8.60e+00,1.00e+01
   N3Na,11,7.61e+00,8.99e+00
   ```

   Useful for spotting compositions that are dragging the global μ.
