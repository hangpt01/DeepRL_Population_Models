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
: "${STAGEB_GATE_PYTHON:?configure the gate interpreter}"
: "${STAGEB_REGISTRATION_BUNDLE:?configure the frozen registration bundle}"
: "${STAGEB_GATE_SIGNING_SEED_FILE:?configure the owner-only 32-byte Ed25519 signing seed}"
: "${STAGEB_OUTPUT_ROOT:?configure the output namespace}"

cd "${STAGEB_REPO_ROOT}"
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1
exec "${STAGEB_GATE_PYTHON}" "${STAGEB_DRIVER}" arm-o-gate \
  --registration "${STAGEB_REGISTRATION_BUNDLE}" \
  --gate-signing-seed-file "${STAGEB_GATE_SIGNING_SEED_FILE}" \
  --output-root "${STAGEB_OUTPUT_ROOT}"
