#!/usr/bin/env bash
#SBATCH --job-name=cumctrl-gate
#SBATCH --partition=comp
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/cumctrl_gate_%A_%a.out
#SBATCH --error=logs/cumctrl_gate_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
CONFIG="${CONFIG:-$ROOT/configs/cumulative_controls_12h.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/cumulative_controls_12h}"
INDEX="${SLURM_ARRAY_TASK_ID:?submit this script as an array job}"

KINDS=(allee theta regime)
ACTIONS=(5 10)
SIGMAS=(0.0 0.1 0.2 0.4)

KIND_INDEX=$((INDEX / 8))
REM=$((INDEX % 8))
ACTION_INDEX=$((REM / 4))
SIGMA_INDEX=$((REM % 4))

ENV="${KINDS[$KIND_INDEX]}"
NA="${ACTIONS[$ACTION_INDEX]}"
SIGMA="${SIGMAS[$SIGMA_INDEX]}"

mkdir -p "$ROOT/logs" "$OUTPUT_ROOT/gates"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m real_ecology_benchmark.cli gate \
  --config "$CONFIG" \
  --environment "$ENV" \
  --actions "$NA" \
  --sigma "$SIGMA" \
  --episodes 20 \
  --output "$OUTPUT_ROOT/gates/${ENV}_${NA}a_sigma${SIGMA}.json"
