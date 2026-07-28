#!/usr/bin/env bash
set -euo pipefail

# Run the Python pipeline for multiple examples (build hmMDP, solve with SARSOP, copy artifacts).
#
# Usage:
#   bash scripts/run_compare_examples.sh
#   bash scripts/run_compare_examples.sh potoroo examples2states2actions
#
# Defaults (if no args provided):
#   potoroo examples2states2actions examples2states4actions
#   examples2states6actions examples2states10actions gouldian
#
# Optional env vars:
#   UAMS_SEED=123
#   UAMS_PRECISION=0.1
#   UAMS_TIMEOUT=3600

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$#" -gt 0 ]]; then
  EXAMPLES=("$@")
else
  EXAMPLES=(
    potoroo
    examples2states2actions
    examples2states4actions
    examples2states6actions
    examples2states10actions
    gouldian
  )
fi

echo "Running Python pipeline for: ${EXAMPLES[*]}"
echo "UAMS_SEED=${UAMS_SEED:-123} UAMS_PRECISION=${UAMS_PRECISION:-0.1} UAMS_TIMEOUT=${UAMS_TIMEOUT:-3600}"
echo

for ex in "${EXAMPLES[@]}"; do
  echo "==================== ${ex} ===================="
  bash scripts/run_python_version.sh "$ex"
  echo
done

echo "Done. Artifacts are under:"
echo "  comparison/python/<example>/"
