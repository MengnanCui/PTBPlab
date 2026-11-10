#!/bin/bash
# A3 on one dg83 node: R3 or R4 on an A1 export (a3/a3_repulsive.py), optionally its MD check.
# Submit from examples/08_accuracy (the script finds slurm/env.sh through $SLURM_SUBMIT_DIR:
# Slurm runs a spooled copy, so $0 is not this file):
#   sbatch slurm/a3_node.sh r3 li_o/scan_li_o.yaml li_o/models/S1 a3/runs/r3_S1 [a3_repulsive.py options]
#   A3_MD=1 sbatch slurm/a3_node.sh r4 li_o/scan_li_o.yaml li_o/models/S1 a3/runs/r4_S1 --band-weight 1.0
# Cost (rough, calibrate with --max-evaluations 5 first): one evaluation with forces is
# (train structures) x (1 + 2 x coefficients) SCC runs; Li-O R3 (8 coefficients, 111 structures)
# ~ 1900 SCC runs ~ 2-3 core-h, ~ 2-3 min on 76 cores; 60 evaluations ~ 2-3 node-h ~ 15-23 yuan.
# The MD check (A3_MD=1, 3 systems x 2 temperatures, 2300 steps, the 80-atom interface at
# its DFT k-grid) runs one process per run and can take many hours: give it its own job.
#SBATCH -J a3_node
#SBATCH -p dg83
#SBATCH -N 1 -n 76
#SBATCH -t 12:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
set -euo pipefail
VARIANT=${1:?r3 or r4}; CONFIG=${2:?scan config}; PARENT=${3:?A1 export dir}; OUT=${4:?output dir}
shift 4
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ENV_SH="${PTBP_ENV_SH:-}"
if [ -z "$ENV_SH" ]; then          # Slurm runs a spooled copy: $0 is not this file inside a job
  for d in "$HERE/slurm" "$HERE" "${PTBP_REPO:-}/examples/08_accuracy/slurm"; do
    if [ -f "$d/env.sh" ]; then ENV_SH="$d/env.sh"; break; fi
  done
fi
[ -n "$ENV_SH" ] || { echo "env.sh not found: submit from examples/08_accuracy or set PTBP_ENV_SH" >&2; exit 1; }
source "$ENV_SH"
cd "$PTBP_REPO/examples/08_accuracy"      # the paths below are relative to it
python a3/a3_repulsive.py "$VARIANT" --config "$CONFIG" --parent "$PARENT" --out "$OUT" \
    --jobs "${SLURM_NTASKS:-76}" "$@"
if [ "${A3_MD:-0}" = "1" ]; then
  python a3/a3_repulsive.py md --config "$CONFIG" --model "$OUT/model" --out "$OUT/md"
fi
