#!/usr/bin/env bash
set -euo pipefail

OVERLAY_ROOT="/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/ricker_only_plus_72_20260720/trajectory_overlay"
PYTHON_BIN="/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python"
INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "$PYTHON_BIN" "$OVERLAY_ROOT/capture_ricker_only_plus_trajectories.py" "$INDEX"
