#!/usr/bin/env bash
set -euo pipefail
GROUP="${1:?method group required}"
WORKER_JOB_ID="${2:?worker job id required}"
ROOT="/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/option_a_three_species_safe_completion_20260723_v1"
if [[ "$GROUP" == "plus" ]]; then
  RUN_ROOT="$ROOT/plus_alignment"
  FIT="$ROOT/launch_package/manifests/plus_alignment_fit_2.csv"
  PLAN="$ROOT/launch_package/manifests/plus_alignment_plan_2.csv"
  OUTPUT="$ROOT/acceptance_plus_alignment.json"
elif [[ "$GROUP" == "moor" ]]; then
  RUN_ROOT="$ROOT/moor_crab"
  FIT="$ROOT/launch_package/manifests/moor_crab_fit_8.csv"
  PLAN="$ROOT/launch_package/manifests/moor_crab_plan_8.csv"
  OUTPUT="$ROOT/acceptance_moor_crab.json"
else
  echo "unsupported method group: $GROUP" >&2
  exit 2
fi
exec /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python \
  "$ROOT/launch_package/scripts/option_a_acceptance.py" \
  --method "$GROUP" --run-root "$RUN_ROOT" \
  --fit-manifest "$FIT" --plan-manifest "$PLAN" \
  --worker-job-id "$WORKER_JOB_ID" --output "$OUTPUT"
