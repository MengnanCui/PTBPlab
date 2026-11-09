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
source "$(dirname "$0")/env.sh"
cd "$(dirname "$CONFIG")"
$PTBP scan run "$(basename "$CONFIG")" --index "${SLURM_ARRAY_TASK_ID:-0}" --stride "$STRIDE"
