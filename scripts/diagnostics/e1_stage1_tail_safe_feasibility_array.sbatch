#!/bin/bash
#SBATCH --job-name=e1_tail_safe_feas
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --constraint=EPYC9534
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --array=0-15%16
#SBATCH --output=/fs04/scratch2/ce25/DeepRL_Population_Models/.verification/e1_stage1_tail_safe_feasibility_20260806_v1/logs/array_%A_%a.out
#SBATCH --error=/fs04/scratch2/ce25/DeepRL_Population_Models/.verification/e1_stage1_tail_safe_feasibility_20260806_v1/logs/array_%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec /fs04/scratch2/ce25/hphung/conda/envs/poprl/bin/python3.10 \
  /fs04/scratch2/ce25/DeepRL_Population_Models/scripts/diagnostics/e1_stage1_tail_safe_feasibility.py \
  --task-id "${SLURM_ARRAY_TASK_ID}"
