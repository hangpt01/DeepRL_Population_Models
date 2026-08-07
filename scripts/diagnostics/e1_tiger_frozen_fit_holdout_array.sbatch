#!/bin/bash
#SBATCH --job-name=e1_tiger_holdout
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --constraint=EPYC9534
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/fs04/scratch2/ce25/DeepRL_Population_Models/outputs/e1_tiger_frozen_fit_holdout_predictive_20260805_v1/logs/array_%A_%a.out
#SBATCH --error=/fs04/scratch2/ce25/DeepRL_Population_Models/outputs/e1_tiger_frozen_fit_holdout_predictive_20260805_v1/logs/array_%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd /fs04/scratch2/ce25/DeepRL_Population_Models
exec /fs04/scratch2/ce25/hphung/conda/envs/poprl/bin/python3.10 \
  scripts/diagnostics/e1_tiger_frozen_fit_holdout_run.py \
  --task-id "${SLURM_ARRAY_TASK_ID}"
