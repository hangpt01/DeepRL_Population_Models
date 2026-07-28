#!/usr/bin/env bash
#SBATCH --job-name=gaudit-rows
#SBATCH --partition=comp
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Run an explicit LIST of manifest rows in one job.  Used for the canary: the canary's
# rows are the 24 configs of a single cell, whose manifest indices run past Slurm's
# MaxArraySize (1001), so they cannot be addressed as array task ids.  (The main sweep
# packs contiguous blocks and its array indices max out at 527, well inside the limit.)
set -uo pipefail

MANIFEST="${1:?usage: run_rows.sh MANIFEST ROWS(comma-separated)}"
ROWS="${2:?usage: run_rows.sh MANIFEST ROWS}"
ROOT="${ROOT:?set ROOT}"
CONFIG="${CONFIG:?set CONFIG}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

rc=0
IFS=',' read -ra IDX <<< "$ROWS"
for i in "${IDX[@]}"; do
  echo "--- row $i"
  if ! python "$ROOT/scripts/run_real_manifest_row.py" "$MANIFEST" "$i" \
        --config "$CONFIG" --output-root "$OUTPUT_ROOT" \
        --dataset-root "$DATASET_ROOT" --backend numpy; then
    echo "ROW $i FAILED" >&2
    rc=1
  fi
done
exit "$rc"
