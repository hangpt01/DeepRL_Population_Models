#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:?CODE_ROOT is required}"
MANIFEST="${MANIFEST:?MANIFEST is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
STAGE="${STAGE:?STAGE is required}"
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN is required}"
INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
CONFIG="${CONFIG:-$CODE_ROOT/configs/paper_faithful_hidden_ricker_only_plus_v1.yaml}"

case "$STAGE" in
  fit) RUNNER="$CODE_ROOT/scripts/run_ricker_only_fit_locked.py" ;;
  plan) RUNNER="$CODE_ROOT/scripts/run_real_manifest_row.py" ;;
  *) echo "unsupported stage: $STAGE" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" "$CODE_ROOT/scripts/ricker_only_thread_entry.py" \
  "$RUNNER" "$MANIFEST" "$INDEX" \
  --config "$CONFIG" --output-root "$OUTPUT_ROOT"
