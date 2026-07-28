#!/usr/bin/env bash
#SBATCH --job-name=real-traj-plot
#SBATCH --partition=comp
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/plot_%j.out
#SBATCH --error=logs/plot_%j.err

set -euo pipefail

if [[ -n "${TRAJ_DEMO_ROOT:-}" ]]; then
  DEMO_ROOT="$TRAJ_DEMO_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/manifest.csv" ]]; then
  DEMO_ROOT="$SLURM_SUBMIT_DIR"
else
  DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$DEMO_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$DEMO_ROOT/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"
python "$DEMO_ROOT/scripts/plot_real_trajectory_demo.py"
