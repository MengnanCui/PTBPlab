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
source "$(dirname "$0")/env.sh"
export OMP_NUM_THREADS=8
cd "$PTBP_REPO/examples/08_accuracy/li_o"
python k1_numerics.py --config scan_li_o.yaml --out runs/k1
python k4_dftb3_u.py --config scan_li_o.yaml --out runs/k4
[ -f runs/li_o_s_sobol/losses.csv ] && python k6_per_structure_r0.py --config scan_li_o.yaml --out runs/k6 || true
