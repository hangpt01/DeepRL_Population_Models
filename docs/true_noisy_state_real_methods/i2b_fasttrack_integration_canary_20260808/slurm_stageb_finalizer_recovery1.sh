#!/bin/bash
#SBATCH --job-name=i2b_arm_t_finalize_r1
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --account=ce25
#SBATCH --constraint=xenon-8452Y
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=outputs/i2b_stageb_arm_t_sigma02_20260809/slurm/finalizer_%j.out
#SBATCH --error=outputs/i2b_stageb_arm_t_sigma02_20260809/slurm/finalizer_%j.err
set -euo pipefail
export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 ARM_T_ARRAY_JOB_ID ARM_O_GATE_JOB_ID" >&2
  exit 64
fi
cd /home/hphung/ce25_scratch2/DeepRL_Population_Models
/usr/bin/python -B docs/true_noisy_state_real_methods/i2b_fasttrack_integration_canary_20260808/finalize_stageb_recovery1.py "$1" "$2"
