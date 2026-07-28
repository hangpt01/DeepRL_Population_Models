#!/bin/bash
#SBATCH --job-name=h12_plus_repair
#SBATCH --partition=m3h
#SBATCH --account=ce25
#SBATCH --qos=m3h
#SBATCH --constraint=xenon-8452Y
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=06:00:00
#SBATCH --array=0-47
#SBATCH --output=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_followups_20260727_v1/H12_out/slurm-%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

CELLS=(A1 A2 A3 A4 B2 B3)
ARMS=(0.5 0.7 0.85 1.2 1.5 2.0 3.0 kref)
task=$SLURM_ARRAY_TASK_ID
arm_index=$((task % 8))
cell_index=$((task / 8))
ROOT=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_followups_20260727_v1
cd "$ROOT/scripts"
srun /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python \
  run_h12_arm.py "${CELLS[$cell_index]}" plus "${ARMS[$arm_index]}"
