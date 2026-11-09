#!/bin/bash
# D1 band references with GPAW on dg83 (if not done on the workstation).
#   sbatch slurm/gpaw_bands.sh            # Li / Li-O  (li_o/bands_gpaw.py)
#   sbatch slurm/gpaw_bands.sh ink        # precursor ink orbitals (pvk_ink/orbitals_gpaw.py)
# Needs a GPAW environment ($GPAW_CONDA_ENV, default gpaw).  Estimate: Li/Li-O 30 structures,
# a few core-h in total.
#SBATCH -J a1_gpaw
#SBATCH -p dg83
#SBATCH -N 1 -n 32
#SBATCH -t 12:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
set -euo pipefail
source "$(dirname "$0")/env.sh"
conda activate "${GPAW_CONDA_ENV:-gpaw}" 2>/dev/null || true
if [ "${1:-lio}" = "ink" ]; then
  cd "$PTBP_REPO/examples/08_accuracy/pvk_ink" && mpirun -np "${SLURM_NTASKS:-32}" gpaw python orbitals_gpaw.py --out .
else
  cd "$PTBP_REPO/examples/08_accuracy/li_o" && mpirun -np "${SLURM_NTASKS:-32}" gpaw python bands_gpaw.py --out .
fi
