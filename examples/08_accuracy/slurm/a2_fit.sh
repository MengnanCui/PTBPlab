#!/bin/bash
# A2: one fit per array task (task 0 = the full fit, task k = the leave-one-environment-out fit
# of band group k; `a2.py plan` lists the groups), then nothing else.
#   mkdir -p logs && sbatch --array=0-7 slurm/a2_fit.sh a2/a2_li_o.yaml E3-L1-linear
# then, after the array: sbatch --dependency=afterok:<id> slurm/a2_post.sh a2/a2_li_o.yaml E3-L1-linear
# Cost: a band fit is 40 evaluations x (band SCCs + frozen-charge derivatives) on 8 cores,
# about 0.2-1 h, i.e. 0.2-0.8 yuan per task (dg83, 0.1 yuan per core-hour).
#SBATCH -J a2_fit
#SBATCH -p dg83
#SBATCH -N 1 -n 8
#SBATCH -t 04:00:00
#SBATCH -o logs/%x.%A_%a.out
#SBATCH -e logs/%x.%A_%a.err
set -euo pipefail
CONFIG=${1:?A2 config}; RUN=${2:?run name (a2.py plan)}
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ENV_SH="${PTBP_ENV_SH:-}"
if [ -z "$ENV_SH" ]; then          # Slurm runs a spooled copy: $0 is not this file inside a job
  for d in "$HERE/slurm" "$HERE" "${PTBP_REPO:-}/examples/08_accuracy/slurm"; do
    if [ -f "$d/env.sh" ]; then ENV_SH="$d/env.sh"; break; fi
  done
fi
[ -n "$ENV_SH" ] || { echo "env.sh not found: submit from examples/08_accuracy or set PTBP_ENV_SH" >&2; exit 1; }
source "$ENV_SH"
python "$PTBP_REPO/examples/08_accuracy/a2/a2.py" fit "$CONFIG" --run "$RUN" \
  --holdout "${SLURM_ARRAY_TASK_ID:-0}" --processes "${SLURM_NTASKS:-8}"
