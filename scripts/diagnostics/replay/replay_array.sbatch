#!/bin/bash
# Tier-A fleet: A1-A5 x {PLUS, MOOR} = 10 tasks. One CPU/task, threads=1, pinned to the
# accepted run's CPU (xenon-8452Y). Each task is an independent 1-threaded process, so parity
# is preserved even if several pack onto one node.
#SBATCH --job-name=diag_replay_A
#SBATCH --partition=m3h
#SBATCH --account=ce25
#SBATCH --qos=m3h
#SBATCH --constraint=xenon-8452Y
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_replay_20260726_v1/parity/slurm-%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PAIRS=(A1:plus A2:plus A3:plus A4:plus A5:plus A1:moor A2:moor A3:moor A4:moor A5:moor)
IFS=: read -r CELL SIDE <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"
ROOT=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_replay_20260726_v1
cd "$ROOT/scripts"
echo "task=$SLURM_ARRAY_TASK_ID cell=$CELL side=$SIDE host=$(hostname)"
lscpu | grep -E "Model name" || true
srun python3 run_diagnostic_replay.py "$CELL" "$SIDE"
