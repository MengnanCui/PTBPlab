#!/usr/bin/env bash
# Example 7 — band-structure optimisation of the Cu confinement (r0, p)
# against a GPAW (PBE) band structure of fcc Cu. Outputs go into ./run/.
set -euo pipefail
cd "$(dirname "$0")"

# ASE reads the SKF headers from $DFTB_PREFIX when it is set, while PTBP
# writes the SKFs of every step into that step's own directory.
unset DFTB_PREFIX

mkdir -p run
cd run
ptbp --optimization_option bandstructure \
     --ref_dir ../ref \
     --dft_file cu_fcc.xyz \
     --dft_band_file cu_fcc_band.json \
     --superposition density \
     --n_calls 11

echo
echo "Best (r0, p) is printed above; every step is archived in ./run/results.tgz"
