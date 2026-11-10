#!/bin/bash
# Baselines (common/baselines.py) on dg83: PTBP published, GFN1-xTB (tblite), MACE-MP-0 (CPU).
# Submit from examples/08_accuracy:
#   sbatch slurm/baselines.sh li_o/scan_li_o.yaml li_o/runs/baselines
#   METHODS="ptbp_published gfn1_xtb mace_mp mace_omol" sbatch slurm/baselines.sh pvk_ink/scan_ink.yaml pvk_ink/runs/baselines
# ptbp_published needs the scan's published point (ptbp scan run CONFIG --points 0).  GFN1-xTB runs
# periodic cells as Gamma-point supercells of the k-grid (--xtb-max-atoms caps them).
# Cost: < 1 node-h, < 8 yuan.
#SBATCH -J a1_baselines
#SBATCH -p dg83
#SBATCH -N 1 -n 16
#SBATCH -t 06:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
set -euo pipefail
CONFIG=${1:?scan config}; OUT=${2:?output dir}
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ENV_SH="${PTBP_ENV_SH:-}"
if [ -z "$ENV_SH" ]; then          # Slurm runs a spooled copy: $0 is not this file inside a job
  for d in "$HERE/slurm" "$HERE" "${PTBP_REPO:-}/examples/08_accuracy/slurm"; do
    if [ -f "$d/env.sh" ]; then ENV_SH="$d/env.sh"; break; fi
  done
fi
[ -n "$ENV_SH" ] || { echo "env.sh not found: submit from examples/08_accuracy or set PTBP_ENV_SH" >&2; exit 1; }
source "$ENV_SH"
export OMP_NUM_THREADS=16
cd "$PTBP_REPO/examples/08_accuracy"      # the paths below are relative to it
python common/baselines.py --config "$CONFIG" --out "$OUT" \
    --methods ${METHODS:-ptbp_published gfn1_xtb mace_mp} --jobs "${SLURM_NTASKS:-16}"
