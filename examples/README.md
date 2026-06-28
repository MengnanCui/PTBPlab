# PTBP examples

Four end-to-end recipes covering each mode of the toolkit. Each folder
has a self-contained `run.sh` and a README explaining what the example
does, what it produces, and how long it takes.

| folder | mode | input | DFTB+ needed |
|---|---|---|---|
| `01_skf_generator/` | `--skf_generator` | published H/O parameters | ❌ |
| `02_eos_cu/` | `--optimization_option energygeometry` | EOS scan: 6 phases × 11 vols | ✅ |
| `03_dataset_cu/` | `--optimization_option dataset` | Cu trajectory + auto-fit E0s | ✅ |
| `04_reaction_b2n2_n3na/` | `--optimization_option reaction` | 2 formulas, 3 elements | ✅ |
| `05_pso_cu/` | `--optimizer pso` (dataset mode) | Cu trajectory, particle swarm | ✅ |
| `06_yaml_config/` | `--config FILE.yaml` (dataset+pso) | every setting in YAML, reproducible | ✅ |

Run any example:

```
cd 04_reaction_b2n2_n3na
./run.sh
```

`run.sh` scripts each `chmod +x`-able; if you cloned the repo without
preserving exec bits use `bash run.sh` instead.
