#!/bin/bash
# The `run` stage of a scan as a Slurm array on dg83: task i does points i, i+N, i+2N, ...
#   sbatch --array=0-31 slurm/scan_array.sh li_o/scan_li_o.yaml 32
# Cost (dg83, 0.1 yuan/core-h): about 2-4 core-min per Li-O point with the dense hotcent grid,
# so 257 points ~ 10-17 core-h ~ 1-2 yuan.  Finished points are skipped on a resubmission.
#SBATCH -J a1_scan
#SBATCH -p dg83
#SBATCH -N 1 -n 1 -c 2
#SBATCH -t 12:00:00
#SBATCH -o logs/%x.%A_%a.out
#SBATCH -e logs/%x.%A_%a.err
set -euo pipefail
CONFIG=${1:?scan config}; STRIDE=${2:-${SLURM_ARRAY_TASK_COUNT:-1}}
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
$PTBP scan run "$(basename "$CONFIG")" --index "${SLURM_ARRAY_TASK_ID:-0}" --stride "$STRIDE"
