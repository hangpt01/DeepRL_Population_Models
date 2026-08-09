#!/bin/bash
#SBATCH --job-name=i2b_arm_o_gate_r1
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --account=ce25
#SBATCH --constraint=xenon-8452Y
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=outputs/i2b_stageb_arm_t_sigma02_20260809/slurm/gate_%j.out
#SBATCH --error=outputs/i2b_stageb_arm_t_sigma02_20260809/slurm/gate_%j.err
set -euo pipefail
export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
cd /home/hphung/ce25_scratch2/DeepRL_Population_Models
/usr/bin/python -B docs/true_noisy_state_real_methods/i2b_fasttrack_integration_canary_20260808/arm_o_gate_recovery1.py
