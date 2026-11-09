#!/bin/bash
# A whole scan on one dg83 node (76 cores): run, losses, report.
#   sbatch slurm/scan_node.sh li_o/scan_li.yaml
# Li grid (145 points): about 1 h on a node = 76 core-h = 7.6 yuan at most (it stops when done).
#SBATCH -J a1_node
#SBATCH -p dg83
#SBATCH -N 1 -n 76
#SBATCH -t 08:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
set -euo pipefail
CONFIG=${1:?scan config}; JOBS=${2:-${SLURM_NTASKS:-76}}
source "$(dirname "$0")/env.sh"
cd "$(dirname "$CONFIG")"
C="$(basename "$CONFIG")"
$PTBP scan points "$C"
$PTBP scan run "$C" --jobs "$JOBS"
$PTBP scan losses "$C" --jobs "$JOBS"
$PTBP scan report "$C"
$PTBP scan status "$C"
