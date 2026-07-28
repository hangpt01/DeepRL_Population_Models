#!/usr/bin/env bash
#SBATCH --job-name=adapted-fit
#SBATCH --partition=comp
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/adapted_fit_%A_%a.out
#SBATCH --error=logs/adapted_fit_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MANIFEST="${1:?usage: run_adapted_fit_row.sh FIT_MANIFEST.csv}"
INDEX="${SLURM_ARRAY_TASK_ID:?submit this script as an array job}"
CONFIG="${CONFIG:-$ROOT/configs/paper_faithful_hidden.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(cd "$(dirname "$MANIFEST")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-paper-faithful/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "adapted-fit Python is unavailable: $PYTHON_BIN" >&2
  exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

"$PYTHON_BIN" "$ROOT/scripts/run_adapted_fit_row.py" \
  "$MANIFEST" "$INDEX" \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT"
