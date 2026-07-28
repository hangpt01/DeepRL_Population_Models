#!/usr/bin/env bash
#SBATCH --job-name=synthetic-gates
#SBATCH --partition=comp
#SBATCH --qos=rtq
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/%j_gate.out
#SBATCH --error=logs/%j_gate.err

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
mkdir -p "$ROOT/logs" "$ROOT/outputs/gates"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python "$ROOT/scripts/run_gate_matrix.py" \
  --config "$ROOT/configs/synthetic_full.yaml" \
  --output-root "$ROOT/outputs/gates" \
  --episodes 20
