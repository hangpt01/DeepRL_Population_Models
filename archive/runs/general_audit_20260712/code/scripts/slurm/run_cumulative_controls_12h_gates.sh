#!/usr/bin/env bash
#SBATCH --job-name=cumctrl-gates
#SBATCH --partition=comp
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/cumctrl_gates_%j.out
#SBATCH --error=logs/cumctrl_gates_%j.err

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
CONFIG="${CONFIG:-$ROOT/configs/cumulative_controls_12h.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/cumulative_controls_12h}"

mkdir -p "$ROOT/logs" "$OUTPUT_ROOT/gates"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

for ENV in allee theta regime; do
  for NA in 5 10; do
    for SIGMA in 0.0 0.1 0.2 0.4; do
      python -m real_ecology_benchmark.cli gate \
        --config "$CONFIG" \
        --environment "$ENV" \
        --actions "$NA" \
        --sigma "$SIGMA" \
        --episodes 20 \
        --output "$OUTPUT_ROOT/gates/${ENV}_${NA}a_sigma${SIGMA}.json"
    done
  done
done
