#!/bin/bash
#SBATCH --job-name=h12_m1_gate
#SBATCH --partition=m3h
#SBATCH --account=ce25
#SBATCH --qos=m3h
#SBATCH --constraint=xenon-8452Y
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_followups_20260727_v1/H12_out/m1_anchor/slurm-%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
CELLS=(A1 A1 A2 A2 A3 A3 A4 A4 B2 B2 B3 B3)
SIDES=(plus moor plus moor plus moor plus moor plus moor plus moor)
ROOT=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_followups_20260727_v1
mkdir -p "$ROOT/H12_out/m1_anchor"
cd "$ROOT/scripts"
srun /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python run_h12_m1_anchor.py "${CELLS[$SLURM_ARRAY_TASK_ID]}" "${SIDES[$SLURM_ARRAY_TASK_ID]}"
