#!/usr/bin/env bash
# TEMPLATE ONLY — DO NOT RUN FROM THE MAC.
set -euo pipefail
: "${STAGEB_SLURM_TEMPLATE_ROOT:?configure template directory on M3}"
: "${STAGEB_REPO_ROOT:?configure the clean registered M3 clone}"
: "${STAGEB_REGISTRATION_BUNDLE:?configure the frozen registration bundle}"

mapfile -t stageb_log_paths < <(/usr/bin/python - \
  "${STAGEB_REPO_ROOT}" "${STAGEB_REGISTRATION_BUNDLE}" <<'PY'
import sys
from pathlib import Path

repository, registration_path = sys.argv[1:]
sys.path.insert(0, repository)
from docs.true_noisy_state_real_methods.stageb_sigma02_corrected_local_20260809.common import strict_json_loads
from docs.true_noisy_state_real_methods.stageb_sigma02_corrected_local_20260809.registration import freeze_registration_bundle
from docs.true_noisy_state_real_methods.stageb_sigma02_corrected_local_20260809.submission import validate_durable_log_plan

bundle = strict_json_loads(Path(registration_path).read_bytes())
frozen = freeze_registration_bundle(bundle, repository_root=Path(repository))
plan = validate_durable_log_plan(
    frozen.bundle()["scientific_log_plan"],
    require_existing=True,
    require_writable=True,
)
for role in ("arm_o", "arm_o_gate", "arm_t", "finalizer"):
    print(plan["templates"][role]["stdout"])
    print(plan["templates"][role]["stderr"])
PY
)
if (( ${#stageb_log_paths[@]} != 8 )); then
  printf 'registered durable log-plan resolution failed\n' >&2
  exit 2
fi

arm_o_job="$(sbatch --parsable --chdir="${STAGEB_REPO_ROOT}" \
  --output="${stageb_log_paths[0]}" --error="${stageb_log_paths[1]}" \
  "${STAGEB_SLURM_TEMPLATE_ROOT}/arm_o_array.sbatch")"
gate_job="$(sbatch --parsable --chdir="${STAGEB_REPO_ROOT}" \
  --output="${stageb_log_paths[2]}" --error="${stageb_log_paths[3]}" \
  --dependency="afterok:${arm_o_job}" \
  "${STAGEB_SLURM_TEMPLATE_ROOT}/arm_o_gate.sbatch")"
arm_t_job="$(sbatch --parsable --chdir="${STAGEB_REPO_ROOT}" \
  --output="${stageb_log_paths[4]}" --error="${stageb_log_paths[5]}" \
  --dependency="afterok:${gate_job}" \
  "${STAGEB_SLURM_TEMPLATE_ROOT}/arm_t_array.sbatch")"
finalizer_job="$(sbatch --parsable --chdir="${STAGEB_REPO_ROOT}" \
  --output="${stageb_log_paths[6]}" --error="${stageb_log_paths[7]}" \
  --dependency="afterany:${arm_t_job}" \
  "${STAGEB_SLURM_TEMPLATE_ROOT}/finalizer.sbatch")"

printf 'arm_o=%s\ngate=%s\narm_t=%s\nfinalizer=%s\n' \
  "${arm_o_job}" "${gate_job}" "${arm_t_job}" "${finalizer_job}"
