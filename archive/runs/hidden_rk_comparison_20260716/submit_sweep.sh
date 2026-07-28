#!/usr/bin/env bash
# Submit matched full and hidden arrays plus an automatic completeness/analysis job.
set -euo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAP="$RUN/code"
CONFIG="$SNAP/configs/hidden_rk.yaml"
HIDDEN_MANIFEST="$RUN/manifests/manifest_hidden.csv"
FULL_MANIFEST="$RUN/manifests/manifest_full.csv"

[[ -d "$SNAP" ]] || { echo "run prepare_run.sh first" >&2; exit 1; }
[[ -f "$HIDDEN_MANIFEST" && -f "$FULL_MANIFEST" ]] || {
  echo "matched manifests are missing" >&2
  exit 1
}

cd "$RUN/logs"
HIDDEN_JOB=$(sbatch --parsable --job-name=hidden-rk-hidden \
  --array=0-287%128 \
  --export=ALL,ROOT="$SNAP",CONFIG="$CONFIG",OUTPUT_ROOT="$RUN/outputs/hidden",BS=5 \
  "$RUN/run_block.sh" "$HIDDEN_MANIFEST")
FULL_JOB=$(sbatch --parsable --job-name=hidden-rk-full \
  --array=0-287%112 \
  --export=ALL,ROOT="$SNAP",CONFIG="$CONFIG",OUTPUT_ROOT="$RUN/outputs/full",BS=5 \
  "$RUN/run_block.sh" "$FULL_MANIFEST")
ANALYSIS_JOB=$(sbatch --parsable --dependency="afterany:$HIDDEN_JOB:$FULL_JOB" \
  --export=ALL,ROOT="$SNAP",RUN="$RUN" "$RUN/run_analysis.sh")

{
  echo "hidden=$HIDDEN_JOB"
  echo "full=$FULL_JOB"
  echo "analysis=$ANALYSIS_JOB"
} > "$RUN/job_ids.txt"

echo "hidden -> $HIDDEN_JOB"
echo "full -> $FULL_JOB"
echo "analysis -> $ANALYSIS_JOB"

