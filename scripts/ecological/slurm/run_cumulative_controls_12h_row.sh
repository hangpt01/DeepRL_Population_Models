#!/usr/bin/env bash
#SBATCH --job-name=cumctrl-row
#SBATCH --partition=comp
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/cumctrl_%A_%a.out
#SBATCH --error=logs/cumctrl_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
CONFIG="${CONFIG:-$ROOT/configs/cumulative_controls_12h.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/cumulative_controls_12h}"
MANIFEST="${1:-$OUTPUT_ROOT/manifest_probe.csv}"
INDEX="${SLURM_ARRAY_TASK_ID:?submit this script as an array job}"

mkdir -p "$ROOT/logs" "$OUTPUT_ROOT"
cd "$ROOT"
export REQUIRE_GATE=false
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python "$ROOT/scripts/run_manifest_row.py" \
  "$MANIFEST" \
  "$INDEX" \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --allow-uncalibrated
