#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:?CODE_ROOT is required}"
FIT_MANIFEST="${FIT_MANIFEST:?FIT_MANIFEST is required}"
PLAN_MANIFEST="${PLAN_MANIFEST:?PLAN_MANIFEST is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN is required}"
INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
CONFIG="${CONFIG:-$CODE_ROOT/configs/paper_faithful_hidden_ricker_only_plus_v1.yaml}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# The shell's errexit is the cell-local success dependency: plan row i is never
# invoked unless fit row i exits zero. Other array elements remain independent.
"$PYTHON_BIN" "$CODE_ROOT/scripts/ricker_only_thread_entry.py" \
  "$CODE_ROOT/scripts/run_ricker_only_fit_locked.py" "$FIT_MANIFEST" "$INDEX" \
  --config "$CONFIG" --output-root "$OUTPUT_ROOT"

exec "$PYTHON_BIN" "$CODE_ROOT/scripts/ricker_only_thread_entry.py" \
  "$CODE_ROOT/scripts/run_ricker_only_plan_with_receipt.py" "$PLAN_MANIFEST" "$INDEX" \
  --config "$CONFIG" --output-root "$OUTPUT_ROOT"
