#!/usr/bin/env bash
# Example 4 — reaction mode on a multi-formula multi-element dataset
# 22 structures spanning 2 formulas (B2N2 + N3Na) and 3 elements (B, N, Na),
# extracted from the bundled nitrides.xyz. Auto-detected as `reaction`
# because the subset spans more than one formula.
set -euo pipefail
cd "$(dirname "$0")"

# Build dft/dft.xyz on first run; idempotent.
[ -f dft/dft.xyz ] || python prepare.py

ptbp optimize ./dft/dft.xyz --output ./run

echo
echo "Per-formula breakdown across the BO trajectory:"
echo "  ./run/results/opt_*/reaction_per_formula.csv"
