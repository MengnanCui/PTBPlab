#!/usr/bin/env bash
# Example 1 — generate SKF files from PTBP published parameters
# Output: H-H.skf, H-O.skf, O-H.skf, O-O.skf in this directory.
# DFTB+ NOT required.
set -euo pipefail
cd "$(dirname "$0")"

ptbp gen H O --params PTBP

echo
echo "Generated: $(ls -1 *.skf | tr '\n' ' ')"
