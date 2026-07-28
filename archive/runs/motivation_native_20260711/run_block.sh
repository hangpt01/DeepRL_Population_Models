#!/usr/bin/env bash
#SBATCH --job-name=motiv-block
#SBATCH --partition=comp
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#
# One array task = BS contiguous manifest rows.  Packing is mandatory: the account's
# MaxSubmitJobs is 1000 and the main manifest has 1440 rows.
#
# NO GPU IS REQUESTED, deliberately.  backend.py exposes a CuPy kernel only for
# mechanistic_transition / method:plus / oracle_ablation:plus; every method in this
# run (refplan, bamcts, ogsrl, plus_native, moor_native) is NumPy-only, so a GPU
# allocation would buy nothing and would queue behind GPU fair-share.
#
# Env: ROOT (frozen snapshot), CONFIG, OUTPUT_ROOT, BS.
set -uo pipefail

MANIFEST="${1:?usage: run_block.sh MANIFEST}"
BS="${BS:?set BS (block size)}"
ROOT="${ROOT:?set ROOT (frozen snapshot)}"
CONFIG="${CONFIG:?set CONFIG}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
B="${SLURM_ARRAY_TASK_ID:?submit as an array job}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# One row = one core is the accounting the 250-CPU cap assumes; do not let BLAS
# oversubscribe and silently consume the quota.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

NROWS=$(($(wc -l < "$MANIFEST") - 1))
START=$((B * BS))
END=$((START + BS))
(( END > NROWS )) && END=$NROWS

echo "block $B: rows [$START,$END) of $NROWS  config=$(basename "$CONFIG")  out=$OUTPUT_ROOT"
rc=0
for (( i = START; i < END; i++ )); do
  echo "--- row $i"
  if ! python "$ROOT/scripts/run_real_manifest_row.py" "$MANIFEST" "$i" \
        --config "$CONFIG" --output-root "$OUTPUT_ROOT" --backend numpy; then
    echo "ROW $i FAILED" >&2
    rc=1
  fi
done
exit "$rc"
