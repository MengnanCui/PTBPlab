#!/usr/bin/env bash
# Example 6 — drive a run from a YAML config file
# Same effect as example 5 but every setting is in config.yaml. CLI
# arguments still override individual fields, e.g. to do a 4-evaluation
# smoke check:
#
#   ptbp optimize $(grep '^dataset' config.yaml | awk '{print $2}') \
#        --config config.yaml --n-calls 4
#
# After a successful run, the resolved settings are written back to
# ./run/ptbp_run.yaml (handy as input to `--config` for a follow-up run).
set -euo pipefail
cd "$(dirname "$0")"

# Read dataset from config (POSITIONAL is still required by argparse so we
# pass it explicitly; the dataset key in YAML is informational + used by
# `ptbp postprocess`).
DATASET=$(awk '/^dataset:/ {print $2}' config.yaml)

ptbp optimize "$DATASET" --config config.yaml

echo
echo "Effective settings snapshot: ./run/ptbp_run.yaml"
echo "Re-run post-processing: ptbp postprocess ./run"
