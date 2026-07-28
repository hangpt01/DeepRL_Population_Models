#!/usr/bin/env bash
#SBATCH --job-name=synthetic-traj
#SBATCH --partition=comp
#SBATCH --qos=rtq
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/traj_%A_%a.out
#SBATCH --error=logs/traj_%A_%a.err

set -euo pipefail
ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
INDEX="${SLURM_ARRAY_TASK_ID:?submit as an array job}"
mkdir -p "$ROOT/logs"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$ROOT/scripts/capture_trajectories.py" "$INDEX" --episodes 5
