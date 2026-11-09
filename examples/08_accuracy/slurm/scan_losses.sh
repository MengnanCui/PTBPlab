#!/bin/bash
# The `losses` and `report` stages after an array run (dependency: --dependency=afterok:<array id>).
#   sbatch --dependency=afterok:123456 slurm/scan_losses.sh li_o/scan_li_o.yaml
#SBATCH -J a1_losses
#SBATCH -p dg83
#SBATCH -N 1 -n 76
#SBATCH -t 04:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
set -euo pipefail
CONFIG=${1:?scan config}
source "$(dirname "$0")/env.sh"
cd "$(dirname "$CONFIG")"
C="$(basename "$CONFIG")"
$PTBP scan losses "$C" --jobs "${SLURM_NTASKS:-76}"
$PTBP scan report "$C"
