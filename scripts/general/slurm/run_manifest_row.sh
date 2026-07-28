#!/usr/bin/env bash
#SBATCH --job-name=synthetic-cell
#SBATCH --partition=comp
#SBATCH --qos=rtq
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
MANIFEST="${1:-$ROOT/outputs/manifest.csv}"
INDEX="${SLURM_ARRAY_TASK_ID:?submit this script as an array job}"
mkdir -p "$ROOT/logs"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
GATE_ARGS=()
if [[ "${REQUIRE_GATE:-true}" == "true" ]]; then
  GATE_ARGS+=(--require-gate)
fi
python "$ROOT/scripts/run_manifest_row.py" "$MANIFEST" "$INDEX" --config "$ROOT/configs/synthetic_full.yaml" "${GATE_ARGS[@]}"
