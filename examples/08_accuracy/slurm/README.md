# Slurm templates (A series)

Nothing in examples/08_accuracy submits a job.  These templates are for you to `sbatch` after
checking them.  `env.sh` sets the environment: repo, conda env, SK cache, one BLAS thread per
process.  Edit it once per server.

| template | what | queue | rough cost |
|---|---|---|---|
| `scan_node.sh CONFIG` | `ptbp scan` points → run → losses → report on one node | dg83, 76 cores | Li grid ≈ 1 node-h ≈ 7.6 yuan at most |
| `scan_array.sh CONFIG N` (`--array=0-(N-1)`) | the run stage spread over N tasks | dg83 | Li–O Sobol 257 points ≈ 10–17 core-h ≈ 1–2 yuan |
| `scan_losses.sh CONFIG` | losses + report after an array (`--dependency=afterok:ID`) | dg83 | < 1 node-h |
| `kchecks.sh` | K1, K4, K6 | dg83, 16 cores | < 1 yuan |
| `gpaw_bands.sh [ink]` | D1 band and orbital references with GPAW | dg83 | a few core-h |
| `mace_gpu.sh d3\|k7` | D3 interface sampling with MACE-MP-0; K7 training | dg50, 1 GPU | 1–3 GPU-h = 6–18 yuan |

The VASP batches D2–D5 bring their own `job.sh` and `submit_all.sh` (written by `make_d*.py`)
and need the user's approval of their cost first (A0 section 2).

Prices: dg83 0.1 yuan/core-h; dg50 6 yuan/GPU-h.  The scan estimates assume the dense hotcent
grid (nr 100, ntheta 300); K1 (b) tells whether it is needed.
