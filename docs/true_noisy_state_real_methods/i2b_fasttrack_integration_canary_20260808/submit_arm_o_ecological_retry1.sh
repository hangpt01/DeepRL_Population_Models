#!/bin/bash
#SBATCH --job-name=i2b_arm_o_eco_r1
#SBATCH --partition=m3h
#SBATCH --account=ce25
#SBATCH --qos=m3h
#SBATCH --constraint=xenon-8452Y
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0,1,6,7
#SBATCH --output=/home/hphung/ce25_scratch2/DeepRL_Population_Models/outputs/i2b_fasttrack_integration_canary_20260808/slurm/i2b_arm_o_eco_retry1_%A_%a.out
#SBATCH --error=/home/hphung/ce25_scratch2/DeepRL_Population_Models/outputs/i2b_fasttrack_integration_canary_20260808/slurm/i2b_arm_o_eco_retry1_%A_%a.err

set -eu
export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cd /home/hphung/ce25_scratch2/DeepRL_Population_Models
/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python -B docs/true_noisy_state_real_methods/i2b_fasttrack_integration_canary_20260808/run_arm_o_ecological_retry1.py "${SLURM_ARRAY_TASK_ID}"
