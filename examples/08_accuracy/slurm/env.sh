#!/bin/bash
# Sourced by every template: the remote environment.  Edit once for your server.
#   PTBP_REPO        the PTBPlab checkout (default: two levels above this file)
#   PTBP_CONDA_ENV   conda environment with numpy/scipy/ase/hotcent/pyyaml + `pip install -e pydftb`
#   PTBP_SKCACHE     the shared SK cache (large: keep it off Nextcloud-synced disks)
#   PTBP_LOG_ROOT    optional: one place for all run logs
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PTBP_REPO="${PTBP_REPO:-$(cd "$HERE/../../.." && pwd)}"
export PTBP_CONDA_ENV="${PTBP_CONDA_ENV:-ptbp}"
export PTBP_SKCACHE="${PTBP_SKCACHE:-$HOME/ptbp_skcache}"
# one BLAS thread per process: the scan runs one point per process
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" && conda activate "$PTBP_CONDA_ENV"
fi
export PYTHONPATH="$PTBP_REPO:${PYTHONPATH:-}"
PTBP="python $PTBP_REPO/parameterize.py"
echo "[env] repo $PTBP_REPO ($(git -C "$PTBP_REPO" rev-parse --short HEAD 2>/dev/null)), env $PTBP_CONDA_ENV, cache $PTBP_SKCACHE, host $(hostname)"
