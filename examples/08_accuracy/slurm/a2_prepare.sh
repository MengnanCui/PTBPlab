#!/bin/bash
# A2 SK sets: E1 type copies (seconds) and the E2 typed sets of r0 nodes (hotcent; `a2.py plan`
# prints the tables and core-hours, e.g. Li 7 nodes + O: 36 tables, about 1 core-hour).
#   mkdir -p logs && sbatch slurm/a2_prepare.sh a2/a2_li_o.yaml [RUN]
#SBATCH -J a2_prep
#SBATCH -p dg83
#SBATCH -N 1 -n 16
#SBATCH -t 04:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
set -euo pipefail
CONFIG=${1:?A2 config}; RUN=${2:-}
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ENV_SH="${PTBP_ENV_SH:-}"
if [ -z "$ENV_SH" ]; then          # Slurm runs a spooled copy: $0 is not this file inside a job
  for d in "$HERE/slurm" "$HERE" "${PTBP_REPO:-}/examples/08_accuracy/slurm"; do
    if [ -f "$d/env.sh" ]; then ENV_SH="$d/env.sh"; break; fi
  done
fi
[ -n "$ENV_SH" ] || { echo "env.sh not found: submit from examples/08_accuracy or set PTBP_ENV_SH" >&2; exit 1; }
source "$ENV_SH"
python "$PTBP_REPO/examples/08_accuracy/a2/a2.py" prepare "$CONFIG" ${RUN:+--run "$RUN"}
