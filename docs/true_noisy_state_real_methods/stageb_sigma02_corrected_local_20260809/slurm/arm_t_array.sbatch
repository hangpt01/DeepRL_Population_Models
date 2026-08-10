#!/usr/bin/env bash
#SBATCH --job-name=stageb_corr_t
#SBATCH --array=0-11
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --account=ce25
#SBATCH --constraint=xenon-8452Y
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

set -euo pipefail
: "${STAGEB_REPO_ROOT:?configure the clean M3 clone path}"
: "${STAGEB_DRIVER:?configure the corrected M3 driver path}"
: "${STAGEB_REGISTRATION_BUNDLE:?configure the frozen registration bundle}"
: "${STAGEB_ARM_O_GATE_RECEIPT:?configure the successful Arm O gate receipt}"
: "${STAGEB_GATE_PUBLIC_KEY_FILE:?configure the 32-byte Ed25519 public verification key}"
: "${STAGEB_OUTPUT_ROOT:?configure the output namespace}"

cd "${STAGEB_REPO_ROOT}"
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
interpreter="$(/usr/bin/python - "${STAGEB_REPO_ROOT}" "${STAGEB_REGISTRATION_BUNDLE}" \
  "${SLURM_ARRAY_TASK_ID}" <<'PY'
import sys
from pathlib import Path

root, registration_path, task_index = sys.argv[1:]
sys.path.insert(0, root)
from docs.true_noisy_state_real_methods.stageb_sigma02_corrected_local_20260809.common import strict_json_loads
from docs.true_noisy_state_real_methods.stageb_sigma02_corrected_local_20260809.registration import freeze_registration_bundle, interpreter_binding_for_task

payload = Path(registration_path).read_bytes()
frozen = freeze_registration_bundle(strict_json_loads(payload), repository_root=Path(root))
_task, binding = interpreter_binding_for_task(frozen, arm="T", task_index=int(task_index))
print(binding["absolute_interpreter_path"])
PY
)"
exec "${interpreter}" "${STAGEB_DRIVER}" arm-t \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --registration "${STAGEB_REGISTRATION_BUNDLE}" \
  --arm-o-gate "${STAGEB_ARM_O_GATE_RECEIPT}" \
  --gate-public-key-file "${STAGEB_GATE_PUBLIC_KEY_FILE}" \
  --output-root "${STAGEB_OUTPUT_ROOT}"
