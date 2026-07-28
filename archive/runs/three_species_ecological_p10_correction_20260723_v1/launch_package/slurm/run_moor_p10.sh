#!/usr/bin/env bash
set -euo pipefail

readonly WORKSPACE=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models
readonly RUN_ROOT="${WORKSPACE}/real_ecology_runs/three_species_ecological_p10_correction_20260723_v1"
readonly PYTHON="${WORKSPACE}/.venv-paper-faithful/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "${PYTHON}" \
  "${RUN_ROOT}/launch_package/scripts/dispatch_p10_plan.py" \
  --method moor \
  --position "${SLURM_ARRAY_TASK_ID}"
