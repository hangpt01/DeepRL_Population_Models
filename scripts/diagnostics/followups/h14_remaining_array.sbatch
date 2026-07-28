#!/bin/bash
#SBATCH --job-name=h14_remaining
#SBATCH --partition=m3h
#SBATCH --account=ce25
#SBATCH --qos=m3h
#SBATCH --constraint=xenon-8452Y
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:15:00
#SBATCH --array=0-7
#SBATCH --output=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_followups_20260727_v1/H14_out/slurm-%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

CELLS=(B1 B1 B2 B2 B2 B3 B3 B3)
METHODS=(ogsrl bamcts refplan ogsrl bamcts refplan ogsrl bamcts)
ROOT=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_followups_20260727_v1
mkdir -p "$ROOT/H14_out"
cd "$ROOT/scripts"
srun python3 run_h14.py "${CELLS[$SLURM_ARRAY_TASK_ID]}" "${METHODS[$SLURM_ARRAY_TASK_ID]}"
