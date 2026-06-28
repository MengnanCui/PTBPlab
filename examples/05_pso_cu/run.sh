#!/usr/bin/env bash
# Example 5 — PSO optimiser on Cu, with auto-fit E0s
# Same Cu dataset as example 3, but optimised by particle swarm via
# pyswarms.single.GlobalBestPSO instead of skopt's GP-BO.
set -euo pipefail
cd "$(dirname "$0")"

ptbp optimize ../../dft/dft.xyz \
     --mode dataset \
     --optimizer pso \
     --n-particles 4 \
     --n-calls 12 \
     --output ./run

echo
echo "PSO snapshot written to ./run/result.pkl."
echo "Re-run post-processing without recomputing:"
echo "  ptbp postprocess ./run"
