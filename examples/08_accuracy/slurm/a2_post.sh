#!/bin/bash
# A2 after the fits of a run: the transfer check (electronic energies and forces of the fitted
# model on every structure, then physrep sigma and R2b refitted: the scan's own scoring) and the
# summary (a2-result/1 files for common/gates.py G2).
#   mkdir -p logs && sbatch slurm/a2_post.sh a2/a2_li_o.yaml E3-L1-linear
# Cost: the A1 data once (about 120 structures) plus the scoring: < 0.5 h on 32 cores, < 1.6 yuan.
#SBATCH -J a2_post
#SBATCH -p dg83
#SBATCH -N 1 -n 32
#SBATCH -t 02:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
set -euo pipefail
CONFIG=${1:?A2 config}; RUN=${2:?run name}
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ENV_SH="${PTBP_ENV_SH:-}"
if [ -z "$ENV_SH" ]; then          # Slurm runs a spooled copy: $0 is not this file inside a job
  for d in "$HERE/slurm" "$HERE" "${PTBP_REPO:-}/examples/08_accuracy/slurm"; do
    if [ -f "$d/env.sh" ]; then ENV_SH="$d/env.sh"; break; fi
  done
fi
[ -n "$ENV_SH" ] || { echo "env.sh not found: submit from examples/08_accuracy or set PTBP_ENV_SH" >&2; exit 1; }
source "$ENV_SH"
A2="$PTBP_REPO/examples/08_accuracy/a2/a2.py"
python "$A2" transfer "$CONFIG" --run "$RUN" --jobs "${SLURM_NTASKS:-32}"
python "$A2" summary "$CONFIG"
python "$A2" status "$CONFIG"
