#!/usr/bin/env bash
# One array task runs one complete five-method ecological cell.
#SBATCH --partition=comp
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -uo pipefail

MANIFEST="${1:?usage: run_block.sh MANIFEST.csv}"
ROOT="${ROOT:?set ROOT to frozen snapshot}"
CONFIG="${CONFIG:?set CONFIG}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
BS="${BS:-5}"
B="${SLURM_ARRAY_TASK_ID:?submit as an array job}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

NROWS=$(($(wc -l < "$MANIFEST") - 1))
START=$((B * BS))
END=$((START + BS))
(( END > NROWS )) && END=$NROWS

echo "block $B: rows [$START,$END) of $NROWS config=$(basename "$CONFIG") out=$OUTPUT_ROOT"
rc=0
for ((i = START; i < END; i++)); do
  echo "--- row $i"
  if ! python "$ROOT/scripts/run_real_manifest_row.py" "$MANIFEST" "$i" \
      --config "$CONFIG" --output-root "$OUTPUT_ROOT" --backend numpy; then
    echo "ROW $i FAILED" >&2
    rc=1
  fi
done
exit "$rc"

