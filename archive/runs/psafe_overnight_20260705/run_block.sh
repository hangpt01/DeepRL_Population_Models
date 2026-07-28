#!/usr/bin/env bash
#SBATCH --job-name=psafe-block
#SBATCH --partition=comp
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# Process a contiguous BLOCK of manifest rows in one array task, to stay under
# the account's MaxSubmitJobs=1000 task cap. Env: ROOT, CONFIG, OUTPUT_ROOT, BS.
set -uo pipefail

MANIFEST="${1:?usage: run_block.sh MANIFEST}"
BS="${BS:?set BS (block size)}"
ROOT="${ROOT:?set ROOT}"
CONFIG="${CONFIG:?set CONFIG}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
B="${SLURM_ARRAY_TASK_ID:?submit as an array job}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

NROWS=$(($(wc -l < "$MANIFEST") - 1))
START=$((B * BS))
END=$((START + BS))
(( END > NROWS )) && END=$NROWS

echo "block $B: rows [$START,$END) of $NROWS  config=$(basename "$CONFIG")  out=$OUTPUT_ROOT"
rc=0
for ((idx = START; idx < END; idx++)); do
  if ! python "$ROOT/scripts/run_real_manifest_row.py" "$MANIFEST" "$idx" \
        --config "$CONFIG" --output-root "$OUTPUT_ROOT" --allow-uncalibrated \
        > /dev/null; then
    echo "ROW FAIL idx=$idx" >&2
    rc=1
  fi
done
echo "block $B done (rc=$rc)"
exit 0   # never fail the task on a single bad row; failures are logged above
