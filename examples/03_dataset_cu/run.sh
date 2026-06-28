#!/usr/bin/env bash
# Example 3 — dataset workflow with auto-fit E0s
# Forces --mode dataset (auto-detection would pick `energygeometry` because
# fit.json exists alongside dft.xyz; this example demonstrates the
# alternative dataset-mode loss). E0s are auto-fitted from the dataset's
# DFT energies on the first run and cached.
set -euo pipefail
cd "$(dirname "$0")"

ptbp optimize ../../dft/dft.xyz \
     --mode dataset \
     --output ./run

echo
echo "Auto-fitted E0s cached at: ../../dft/E0s_dft.json"
echo "Best parameters under: ./run/results/opt_*/"
