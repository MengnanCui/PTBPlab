#!/bin/bash
# The cheap checks K1, K4 (and K6 once a scan is scored) on one node.
#   sbatch slurm/kchecks.sh
#SBATCH -J a1_kchecks
#SBATCH -p dg83
#SBATCH -N 1 -n 16
#SBATCH -t 04:00:00
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
export OMP_NUM_THREADS=8
cd "$PTBP_REPO/examples/08_accuracy/li_o"
python k1_numerics.py --config scan_li_o.yaml --out runs/k1
python k4_dftb3_u.py --config scan_li_o.yaml --out runs/k4
[ -f runs/li_o_s_sobol/losses.csv ] && python k6_per_structure_r0.py --config scan_li_o.yaml --out runs/k6 || true
