#!/usr/bin/env bash
#SBATCH --job-name=cumctrl-launch
#SBATCH --partition=comp
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/cumctrl_launch_%j.out
#SBATCH --error=logs/cumctrl_launch_%j.err

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/cumulative_controls_12h}"
PROBE_JOB_ID="${PROBE_JOB_ID:?set PROBE_JOB_ID to the probe array task, e.g. 57913787_0}"
GATE_JOB_ID="${GATE_JOB_ID:-}"

mkdir -p "$ROOT/logs" "$OUTPUT_ROOT"
cd "$ROOT"

read -r STATE EXIT_CODE ELAPSED_RAW < <(
  sacct -n -P -X -j "$PROBE_JOB_ID" -o State,ExitCode,ElapsedRaw |
    awk -F'|' 'NF>=3 {print $1, $2, $3; exit}'
)

if [[ -z "${STATE:-}" ]]; then
  echo "Could not read probe accounting for $PROBE_JOB_ID" >&2
  exit 2
fi

cat > "$OUTPUT_ROOT/probe_decision.txt" <<EOF
probe_job=$PROBE_JOB_ID
state=$STATE
exit_code=$EXIT_CODE
elapsed_raw=$ELAPSED_RAW
EOF

if [[ "$STATE" != COMPLETED || "$EXIT_CODE" != 0:0 ]]; then
  echo "Probe did not complete cleanly; not launching arrays." | tee -a "$OUTPUT_ROOT/probe_decision.txt"
  exit 3
fi

if (( ELAPSED_RAW <= 10800 )); then
  MAIN_CONCURRENCY=48
elif (( ELAPSED_RAW <= 14400 )); then
  MAIN_CONCURRENCY=32
else
  echo "Probe exceeded 4h; not launching arrays." | tee -a "$OUTPUT_ROOT/probe_decision.txt"
  exit 4
fi

echo "main_concurrency=$MAIN_CONCURRENCY" >> "$OUTPUT_ROOT/probe_decision.txt"

PREWARM_JOB=$(
  timeout 60 sbatch --parsable \
    --array=0-47%24 \
    "$ROOT/scripts/slurm/run_cumulative_controls_12h_row.sh" \
    "$OUTPUT_ROOT/manifest_prewarm.csv"
)
echo "prewarm_job=$PREWARM_JOB" >> "$OUTPUT_ROOT/probe_decision.txt"

MAIN_JOB=$(
  timeout 60 sbatch --parsable \
    --dependency="afterok:$PREWARM_JOB" \
    --array="0-167%$MAIN_CONCURRENCY" \
    "$ROOT/scripts/slurm/run_cumulative_controls_12h_row.sh" \
    "$OUTPUT_ROOT/manifest_main.csv"
)
echo "main_job=$MAIN_JOB" >> "$OUTPUT_ROOT/probe_decision.txt"

AGG_DEP="afterany:$MAIN_JOB"
if [[ -n "$GATE_JOB_ID" ]]; then
  AGG_DEP="$AGG_DEP:$GATE_JOB_ID"
fi
AGG_JOB=$(
  timeout 60 sbatch --parsable \
    --dependency="$AGG_DEP" \
    "$ROOT/scripts/slurm/run_cumulative_controls_12h_aggregate.sh"
)
echo "aggregate_job=$AGG_JOB" >> "$OUTPUT_ROOT/probe_decision.txt"
echo "aggregate_dependency=$AGG_DEP" >> "$OUTPUT_ROOT/probe_decision.txt"

cat "$OUTPUT_ROOT/probe_decision.txt"
