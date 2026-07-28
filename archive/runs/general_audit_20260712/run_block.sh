#!/usr/bin/env bash
#SBATCH --job-name=gaudit-block
#SBATCH --partition=comp
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#
# One array task = BS contiguous manifest rows.  --time is set PER ARRAY at submit
# (refplan 04:00, bamcts 12:00, ogsrl 04:00) because a BA-MCTS task at the expensive
# corner can run ~5.4 h on the slowest cell; a mean-sized walltime would silently kill
# exactly the tail we most need.
#
# NO GPU, deliberately: refplan/bamcts/ogsrl are all NumPy-only (backend.py exposes a
# CuPy kernel only for mechanistic_transition / method:plus / oracle_ablation:plus).
#
# Datasets are READ from the motivation run and never regenerated -- see --dataset-root.
set -uo pipefail

MANIFEST="${1:?usage: run_block.sh MANIFEST}"
BS="${BS:?set BS}"
ROOT="${ROOT:?set ROOT (frozen snapshot)}"
CONFIG="${CONFIG:?set CONFIG}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT}"
B="${SLURM_ARRAY_TASK_ID:?submit as an array job}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

NROWS=$(($(wc -l < "$MANIFEST") - 1))
START=$((B * BS))
END=$((START + BS))
(( END > NROWS )) && END=$NROWS

echo "block $B: rows [$START,$END) of $NROWS  manifest=$(basename "$MANIFEST")"
rc=0
for (( i = START; i < END; i++ )); do
  echo "--- row $i"
  if ! python "$ROOT/scripts/run_real_manifest_row.py" "$MANIFEST" "$i" \
        --config "$CONFIG" \
        --output-root "$OUTPUT_ROOT" \
        --dataset-root "$DATASET_ROOT" \
        --backend numpy; then
    echo "ROW $i FAILED" >&2
    rc=1
  fi
done
exit "$rc"
