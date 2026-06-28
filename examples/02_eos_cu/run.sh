#!/usr/bin/env bash
# Example 2 — original PTBP EOS workflow on Cu
# Auto-detected as `energygeometry` because dft/fit.json sits alongside
# dft.xyz. Outputs go into ./run/.
set -euo pipefail
cd "$(dirname "$0")"

ptbp optimize ../../dft/dft.xyz \
     --mode energygeometry \
     --eos-points 11 \
     --output ./run

echo
echo "Best parameters logged to ./run/par.out and ./run/results/opt_*/"
