#!/usr/bin/env bash
#SBATCH --job-name=hidden-rk-traces
#SBATCH --partition=comp
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -euo pipefail

RUN="/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/hidden_rk_comparison_20260716"
export PYTHONPATH="$RUN/code/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$RUN/analysis/.mplconfig"
mkdir -p "$MPLCONFIGDIR"
python "$RUN/analysis/capture_report_trajectories.py" "$SLURM_ARRAY_TASK_ID"
