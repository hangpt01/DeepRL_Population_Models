#!/usr/bin/env bash
#SBATCH --job-name=stageb_corr_gate
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --account=ce25
#SBATCH --constraint=xenon-8452Y
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

set -euo pipefail
: "${STAGEB_REPO_ROOT:?configure the clean M3 clone path}"
: "${STAGEB_DRIVER:?configure the corrected M3 driver path}"
: "${STAGEB_REGISTRATION_BUNDLE:?configure the frozen registration bundle}"
: "${STAGEB_GATE_SIGNING_SEED_FILE:?configure the owner-only 32-byte Ed25519 signing seed}"
: "${STAGEB_OUTPUT_ROOT:?configure the output namespace}"

cd "${STAGEB_REPO_ROOT}"
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1
interpreter="$(/usr/bin/python - "${STAGEB_REPO_ROOT}" "${STAGEB_REGISTRATION_BUNDLE}" <<'PY'
import sys
from pathlib import Path

root, registration_path = sys.argv[1:]
sys.path.insert(0, root)
from docs.true_noisy_state_real_methods.stageb_sigma02_corrected_local_20260809.common import strict_json_loads
from docs.true_noisy_state_real_methods.stageb_sigma02_corrected_local_20260809.registration import freeze_registration_bundle, interpreter_binding_for_command

payload = Path(registration_path).read_bytes()
frozen = freeze_registration_bundle(strict_json_loads(payload), repository_root=Path(root))
binding = interpreter_binding_for_command(frozen, "arm-o-gate")
print(binding["absolute_interpreter_path"])
PY
)"
exec "${interpreter}" "${STAGEB_DRIVER}" arm-o-gate \
  --registration "${STAGEB_REGISTRATION_BUNDLE}" \
  --gate-signing-seed-file "${STAGEB_GATE_SIGNING_SEED_FILE}" \
  --output-root "${STAGEB_OUTPUT_ROOT}"
