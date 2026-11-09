#!/usr/bin/env bash
# The fits of the M5c battery demo (design M5c, section 4), on the data collected by collect.py:
#   b   one Li parameter set                       (2 dims, --n-calls 30)
#   c   two Li types, Li -> Li_ox by CN_O          (4 dims, --n-calls 30 and 60)
# O keeps PTBP's published parameters in every fit (--multi-element Li).
#
#   ./run_fits.sh DATA_DIR OUT_DIR [NCORES_PER_FIT]
#
# DATA_DIR holds train.xyz and E0s_dft.json; PTBP runs from the repository (PYTHONPATH).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
data=$(cd "$1" && pwd)
out=$2
nc=${3:-5}
mkdir -p "$out"
out=$(cd "$out" && pwd)
PY=${PYTHON:-python}
E0=$(tr -d '\n ' < "$data/E0s_dft.json")
TYPES='[{"element":"Li","labels":["Li","Li_ox"],"centre":1.5,"width":2.0,"cutoff":{"O":[4.03,4.77]}}]'

fit() {   # name n_calls [extra args]
  local name=$1 n=$2; shift 2
  mkdir -p "$out/$name/dft"
  cp "$data/train.xyz" "$out/$name/dft/dft.xyz"
  # PTBP's evaluation plot (cli/run.py:401) crashes after the results are written: record the
  # exit code instead of letting set -e end the subshell; the SKFs are in run/results/opt_*.
  ( cd "$out/$name" && rc=0 && PYTHONPATH="$repo" "$PY" -m cli.main optimize dft/dft.xyz \
      --mode reaction --engine pydftb --E0s "$E0" --multi-element Li \
      --n-calls "$n" --ncores "$nc" --output run "$@" > fit.log 2>&1 || rc=$?
    echo "exit $rc" >> fit.log ) || true
}

fit b30 30 &
fit c30 30 --types "$TYPES" &
wait
fit c60 60 --types "$TYPES"
echo "fits done: $out"
