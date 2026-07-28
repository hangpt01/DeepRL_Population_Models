#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
mkdir -p "$ROOT/logs" "$ROOT/outputs/gates"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m real_ecology_benchmark.cli manifest --output "$ROOT/outputs/manifest.csv"
ROWS=$(($(wc -l < "$ROOT/outputs/manifest.csv") - 1))
GATE_JOB=$(sbatch --parsable "$ROOT/scripts/slurm/run_gate_matrix.sh")
MATRIX_JOB=$(sbatch \
  --parsable \
  --dependency="afterok:$GATE_JOB" \
  --array="0-$((ROWS - 1))%72" \
  "$ROOT/scripts/slurm/run_manifest_row.sh" \
  "$ROOT/outputs/manifest.csv")
echo "gate_job=$GATE_JOB"
echo "matrix_job=$MATRIX_JOB dependency=afterok:$GATE_JOB rows=$ROWS concurrency=72"
