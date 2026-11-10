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
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ENV_SH="${PTBP_ENV_SH:-}"
if [ -z "$ENV_SH" ]; then          # Slurm runs a spooled copy: $0 is not this file inside a job
  for d in "$HERE/slurm" "$HERE" "${PTBP_REPO:-}/examples/08_accuracy/slurm"; do
    if [ -f "$d/env.sh" ]; then ENV_SH="$d/env.sh"; break; fi
  done
fi
[ -n "$ENV_SH" ] || { echo "env.sh not found: submit from examples/08_accuracy or set PTBP_ENV_SH" >&2; exit 1; }
source "$ENV_SH"
cd "$(dirname "$CONFIG")"
C="$(basename "$CONFIG")"
$PTBP scan losses "$C" --jobs "${SLURM_NTASKS:-76}"
$PTBP scan report "$C"
