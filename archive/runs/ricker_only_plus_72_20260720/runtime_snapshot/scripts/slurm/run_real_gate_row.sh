#!/usr/bin/env bash
#SBATCH --job-name=real-gate
#SBATCH --partition=comp
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=real_gate_%A_%a.out
#SBATCH --error=real_gate_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/src/real_ecology_benchmark" ]]; then
    ROOT="$SLURM_SUBMIT_DIR"
  else
    ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  fi
fi
MANIFEST="${1:?usage: run_real_gate_row.sh MANIFEST.csv}"
INDEX="${SLURM_ARRAY_TASK_ID:?submit this script as an array job}"
CONFIG="${CONFIG:-$ROOT/configs/real_experiment.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(cd "$(dirname "$MANIFEST")" && pwd)}"
EPISODES="${GATE_EPISODES:-20}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

ARGS=(
  "$ROOT/scripts/run_real_gate_row.py"
  "$MANIFEST" \
  "$INDEX"
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --episodes "$EPISODES"
)
if [[ -n "${BACKEND:-}" ]]; then
  ARGS+=(--backend "$BACKEND")
fi
if [[ -n "${DEVICE:-}" ]]; then
  ARGS+=(--device "$DEVICE")
fi
if [[ -n "${BACKEND_STRICT:-}" ]]; then
  ARGS+=(--backend-strict "$BACKEND_STRICT")
fi

python "${ARGS[@]}"
