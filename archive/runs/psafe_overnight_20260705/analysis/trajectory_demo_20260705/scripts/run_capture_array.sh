#!/usr/bin/env bash
#SBATCH --job-name=real-traj
#SBATCH --partition=comp
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/capture_%A_%a.out
#SBATCH --error=logs/capture_%A_%a.err

set -euo pipefail

if [[ -n "${TRAJ_DEMO_ROOT:-}" ]]; then
  DEMO_ROOT="$TRAJ_DEMO_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/manifest.csv" ]]; then
  DEMO_ROOT="$SLURM_SUBMIT_DIR"
else
  DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
MANIFEST="${MANIFEST:-$DEMO_ROOT/manifest.csv}"
INDEX="${SLURM_ARRAY_TASK_ID:?submit this script as an array job}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "$DEMO_ROOT"
python "$DEMO_ROOT/scripts/capture_real_trajectory_row.py" "$MANIFEST" "$INDEX"
