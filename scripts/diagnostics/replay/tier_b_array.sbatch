#!/bin/bash
# Tier-B diagnostic replay: B1-B3 x four frozen general methods. The accepted
# general-run code/config/data path is imported by run_tier_b_replay.py.
#SBATCH --job-name=diag_replay_B
#SBATCH --partition=m3h
#SBATCH --account=ce25
#SBATCH --qos=m3h
#SBATCH --constraint=xenon-8452Y
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_replay_20260726_v1/parity/slurm-%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
METHODS=(refplan ogsrl bamcts ensemble_value_disagreement_pessimism)
CELL_INDEX=$((SLURM_ARRAY_TASK_ID / 4 + 1))
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
CELL="B${CELL_INDEX}"
METHOD="${METHODS[$METHOD_INDEX]}"
ROOT=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_replay_20260726_v1
cd "$ROOT/scripts"
echo "task=$SLURM_ARRAY_TASK_ID cell=$CELL method=$METHOD host=$(hostname)"
lscpu | grep -E "Model name" || true
srun python3 run_tier_b_replay.py "$CELL" "$METHOD"
