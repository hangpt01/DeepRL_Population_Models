#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
exec /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python \
  /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/option_a_three_species_safe_completion_20260723_v1/launch_package/scripts/option_a_dispatch.py \
  --method plus --position "${SLURM_ARRAY_TASK_ID:?}"
