#!/bin/bash
#SBATCH --job-name=i2b_arm_t_sigma02_r1
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --account=ce25
#SBATCH --constraint=xenon-8452Y
#SBATCH --cpus-per-task=1
#SBATCH --array=0-11
#SBATCH --time=12:00:00
#SBATCH --output=outputs/i2b_stageb_arm_t_sigma02_20260809/slurm/arm_t_%A_%a.out
#SBATCH --error=outputs/i2b_stageb_arm_t_sigma02_20260809/slurm/arm_t_%A_%a.err
set -euo pipefail
export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
cd /home/hphung/ce25_scratch2/DeepRL_Population_Models
runner=docs/true_noisy_state_real_methods/i2b_fasttrack_integration_canary_20260808/run_arm_t_task_recovery1.py
case "${SLURM_ARRAY_TASK_ID}" in
  0|1|6|7) interpreter=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python ;;
  *) interpreter=/usr/bin/python ;;
esac
"${interpreter}" -B "${runner}" "${SLURM_ARRAY_TASK_ID}"
