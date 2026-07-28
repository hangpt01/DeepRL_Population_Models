#!/usr/bin/env bash
#SBATCH --job-name=real-agg
#SBATCH --partition=comp
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=real_aggregate_%j.out
#SBATCH --error=real_aggregate_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/src/real_ecology_benchmark" ]]; then
    ROOT="$SLURM_SUBMIT_DIR"
  else
    ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  fi
fi
OUTPUT_ROOT="${1:-$ROOT/outputs/real_reward_modes_20260703}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python -m real_ecology_benchmark.cli aggregate \
  --root "$OUTPUT_ROOT/evaluation" \
  --output "$OUTPUT_ROOT/aggregate.json"
python "$ROOT/scripts/summarize_real_outputs.py" \
  "$OUTPUT_ROOT" \
  --aggregate "$OUTPUT_ROOT/aggregate.json" \
  --output "$OUTPUT_ROOT/summary.txt"
