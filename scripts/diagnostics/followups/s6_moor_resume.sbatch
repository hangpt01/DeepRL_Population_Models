#!/bin/bash
#SBATCH --job-name=s6_moor_resume
#SBATCH --partition=m3h
#SBATCH --account=ce25
#SBATCH --qos=m3h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_followups_20260727_v1/S6_out/slurm-%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_followups_20260727_v1
cd "$ROOT/scripts"
srun /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python \
  resume_s6_moor.py
