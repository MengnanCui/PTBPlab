#!/bin/bash
# A4 Delta-ML on dg50 (RTX 5090, 6 yuan/GPU-h, CPU and memory free): a4/delta_ml.py prepare
# (the base DFTB model on every structure, 20 CPU processes), train (8 MACE trainings: Delta and
# pure at N = 25, 50, 100, all), evaluate.  Submit from examples/08_accuracy:
#   sbatch slurm/a4_delta_gpu.sh li_o/scan_li_o.yaml li_o/models/S1 a4/runs/d1_S1
#   sbatch slurm/a4_delta_gpu.sh li_o/scan_li_o.yaml a2/runs/<A2 model dir> a4/runs/d3_E2   # Delta3
# Needs mace-torch + torch with CUDA and pydftb's dependencies in $MACE_CONDA_ENV (default perov).
# Cost (rough, calibrate on the first training): ~15-30 min per training at 400 epochs ->
# 2-4 GPU-h ~ 12-24 yuan.  Finished trainings are skipped on a resubmission.
#SBATCH -J a4_delta
#SBATCH -p dg50
#SBATCH --gres=gpu:1
#SBATCH -n 20
#SBATCH -t 12:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
set -euo pipefail
CONFIG=${1:?scan config}; MODEL=${2:?base model dir}; OUT=${3:?output dir}
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ENV_SH="${PTBP_ENV_SH:-}"
if [ -z "$ENV_SH" ]; then          # Slurm runs a spooled copy: $0 is not this file inside a job
  for d in "$HERE/slurm" "$HERE" "${PTBP_REPO:-}/examples/08_accuracy/slurm"; do
    if [ -f "$d/env.sh" ]; then ENV_SH="$d/env.sh"; break; fi
  done
fi
[ -n "$ENV_SH" ] || { echo "env.sh not found: submit from examples/08_accuracy or set PTBP_ENV_SH" >&2; exit 1; }
source "$ENV_SH"
module load cuda/12.9 2>/dev/null || true
conda activate "${MACE_CONDA_ENV:-perov}" 2>/dev/null || true
cd "$PTBP_REPO/examples/08_accuracy"      # the paths below are relative to it
python a4/delta_ml.py prepare --config "$CONFIG" --model "$MODEL" --out "$OUT" --jobs "${SLURM_NTASKS:-20}"
python a4/delta_ml.py train --out "$OUT" --device cuda
python a4/delta_ml.py evaluate --config "$CONFIG" --out "$OUT" --device cuda
