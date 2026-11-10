#!/bin/bash
# S5 (optional A1 diagnostic): refit the SK tables of the best S1 point on one dg83 node.
#   cd examples/08_accuracy && mkdir -p logs && sbatch slurm/s5.sh [li_o/s5_li_o.yaml] [JOBS]
# Needs a scored Li-O scan (ptbp scan losses li_o/scan_li_o.yaml) and, for gradient autodiff,
# torch in the conda env (pip install torch --index-url https://download.pytorch.org/whl/cpu);
# without torch set optimizer.gradient: fd (much slower) in the config.
# Cost: the dry run prints the time per evaluation from the scan's own timing; 20 outer steps are
# roughly 1-3 node-h = 64-200 core-h = 6-20 yuan at most (it stops early when the trust radius
# collapses).  Rerunning the same command resumes from state.json.
#SBATCH -J a1_s5
#SBATCH -p dg83
#SBATCH -N 1 -n 64
#SBATCH -t 12:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
set -euo pipefail
CONFIG=${1:-li_o/s5_li_o.yaml}; JOBS=${2:-${SLURM_NTASKS:-64}}
# sbatch runs a spooled copy of this file, so find env.sh from the submit directory first
HERE="${SLURM_SUBMIT_DIR:-}/slurm"
[ -f "$HERE/env.sh" ] || HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$PTBP_REPO/examples/08_accuracy"
python li_o/s5_sk_tables.py dry-run "$CONFIG" --jobs "$JOBS"
python li_o/s5_sk_tables.py gradcheck "$CONFIG" --structures 2 --coefs 4
python li_o/s5_sk_tables.py fit "$CONFIG" --jobs "$JOBS"
python li_o/s5_sk_tables.py report "$CONFIG" --jobs "$JOBS"
python li_o/s5_sk_tables.py export "$CONFIG" --dest li_o/models/S5
