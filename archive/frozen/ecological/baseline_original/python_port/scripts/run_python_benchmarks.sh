#!/usr/bin/env bash
set -euo pipefail

# Python-only benchmark runner:
# - builds MC-UAMS
# - solves with SARSOP
# - simulates MC-UAMS vs baselines (PUBD + optimal)
# - saves everything under python_port/results/<example>/
#
# Usage:
#   bash python_port/scripts/run_python_benchmarks.sh [example]
#   bash python_port/scripts/run_python_benchmarks.sh potoroo
#
# Optional env:
#   UAMS_SEED=123
#   UAMS_TRIALS=10000
#   UAMS_N_MDP=100
#   UAMS_N_SIM_IT=100
#   UAMS_PRECISION=0.1
#   UAMS_TIMEOUT=3600

EXAMPLE="${1:-gouldian}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python "python_port/scripts/run_python_benchmarks.py" \
  --example "$EXAMPLE" \
  --seed "${UAMS_SEED:-123}" \
  --n-trials "${UAMS_TRIALS:-10000}" \
  ${UAMS_N_MDP:+--n-mdp "$UAMS_N_MDP"} \
  --n-sim-it "${UAMS_N_SIM_IT:-100}" \
  --precision "${UAMS_PRECISION:-0.1}" \
  --timeout "${UAMS_TIMEOUT:-3}"
