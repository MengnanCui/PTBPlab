#!/bin/bash
# MACE on dg50 (RTX 5090, 6 yuan/GPU-h, CPU free): D3 sampling and the K7 training.
#   sbatch slurm/mace_gpu.sh d3       # li_o/make_d3.py sample (MACE-MP-0 MD of the interface)
#   sbatch slurm/mace_gpu.sh k7       # li_o/k7_mace.py prepare + train + evaluate
# Needs mace-torch + torch with CUDA ($MACE_CONDA_ENV, default perov).  Estimate: 1-3 GPU-h.
#SBATCH -J a1_mace
#SBATCH -p dg50
#SBATCH --gres=gpu:1
#SBATCH -n 20
#SBATCH -t 06:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
set -euo pipefail
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
cd "$PTBP_REPO/examples/08_accuracy/li_o"
case "${1:-k7}" in
  d3) python make_d3.py sample --out runs/d3 --device cuda ;;
  k7) python k7_mace.py prepare --out runs/k7 && python k7_mace.py train --out runs/k7 --device cuda \
        && python k7_mace.py evaluate --out runs/k7 --device cuda ;;
esac
