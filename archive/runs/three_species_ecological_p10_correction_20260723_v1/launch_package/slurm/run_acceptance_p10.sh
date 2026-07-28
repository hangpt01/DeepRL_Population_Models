#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 {plus|moor} WORKER_JOB_ID" >&2
  exit 2
fi

readonly METHOD="$1"
readonly WORKER_JOB_ID="$2"
readonly WORKSPACE=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models
readonly ROOT="${WORKSPACE}/real_ecology_runs/three_species_ecological_p10_correction_20260723_v1"
readonly PACKAGE="${ROOT}/launch_package"
readonly PYTHON="${WORKSPACE}/.venv-paper-faithful/bin/python"

case "${METHOD}" in
  plus)
    readonly METHOD_ROOT="${ROOT}/plus"
    readonly MANIFEST="${PACKAGE}/manifests/plus_p10_plan_24.csv"
    readonly CONFIG="${PACKAGE}/configs/plus_ricker_only_p10.yaml"
    readonly OUTPUT="${ROOT}/acceptance_plus_p10.json"
    ;;
  moor)
    readonly METHOD_ROOT="${ROOT}/moor"
    readonly MANIFEST="${PACKAGE}/manifests/moor_p10_plan_24.csv"
    readonly CONFIG="${PACKAGE}/configs/moor_ricker_p10.yaml"
    readonly OUTPUT="${ROOT}/acceptance_moor_p10.json"
    ;;
  *)
    echo "unsupported method: ${METHOD}" >&2
    exit 2
    ;;
esac

exec "${PYTHON}" "${PACKAGE}/scripts/accept_p10_method.py" \
  --method "${METHOD}" \
  --run-root "${METHOD_ROOT}" \
  --manifest "${MANIFEST}" \
  --config "${CONFIG}" \
  --ledger "${PACKAGE}/provenance/fit_reuse_ledger_216.csv" \
  --worker-job-id "${WORKER_JOB_ID}" \
  --output "${OUTPUT}"
