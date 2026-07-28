#!/bin/bash
#SBATCH --job-name=s6_fit_probe
#SBATCH --partition=m3h
#SBATCH --account=ce25
#SBATCH --qos=m3h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_followups_20260727_v1/S6_out/slurm-%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_followups_20260727_v1
mkdir -p "$ROOT/S6_out"
cd "$ROOT/scripts"
srun /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python run_s6_fit_probe.py
