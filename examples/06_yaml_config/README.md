# Example 6 — YAML config for a reproducible run

Demonstrates the `--config FILE.yaml` flow. All settings live in
`config.yaml`; the CLI invocation is one line. CLI flags override
individual YAML fields, so the same config can be reused for sweeps:

```bash
# Default settings (12 evals)
./run.sh

# Smoke check, override n_calls to 4 without editing the YAML
ptbp optimize ../../dft/dft.xyz --config config.yaml --n-calls 4

# Try parallel BO instead of PSO (overriding optimizer)
ptbp optimize ../../dft/dft.xyz --config config.yaml --optimizer parallel_bo
```

After every successful run, the **effective** settings (YAML + CLI
overrides + auto-detection) are written to `<output>/ptbp_run.yaml` —
making it easy to re-run with `--config <output>/ptbp_run.yaml`, or to
hand it to `ptbp postprocess <output>` (which reads `mode` and
`dataset` from the snapshot, no extra flags needed).
