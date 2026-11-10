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
| `baselines.sh CONFIG OUT` (`METHODS=...`) | common/baselines.py: PTBP published, GFN1-xTB, MACE-MP-0 (+ MACE-OMOL for the ink) | dg83, 16 cores | < 1 node-h, < 8 yuan |
| `a3_node.sh r3\|r4 CONFIG PARENT OUT [...]` (`A3_MD=1`) | a3/a3_repulsive.py: R3/R4 fit on an A1 export, score, optional MD | dg83, 76 cores | Li–O R3 ≈ 2–3 node-h ≈ 15–23 yuan (rough) |
| `a4_delta_gpu.sh CONFIG MODEL OUT` | a4/delta_ml.py: prepare (DFTB on 20 cores), 8 MACE trainings, evaluate | dg50, 1 GPU | 2–4 GPU-h ≈ 12–24 yuan (rough) |

The VASP batches D2–D5 bring their own `job.sh` and `submit_all.sh` (written by `make_d*.py`)
and need the user's approval of their cost first (A0 section 2).

Prices: dg83 0.1 yuan/core-h; dg50 6 yuan/GPU-h.  The scan estimates assume the dense hotcent
grid (nr 100, ntheta 300); K1 (b) tells whether it is needed.

The A3/A4/baselines templates are submitted from `examples/08_accuracy` (paths relative to it) and look for `env.sh` in `$SLURM_SUBMIT_DIR/slurm`, `$SLURM_SUBMIT_DIR` and `$PTBP_REPO/examples/08_accuracy/slurm` (or take `$PTBP_ENV_SH`): Slurm runs a spooled copy of the script, so `$(dirname "$0")` does not point to this directory inside a job.
