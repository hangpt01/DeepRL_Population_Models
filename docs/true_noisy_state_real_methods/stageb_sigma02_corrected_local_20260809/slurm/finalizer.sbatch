#!/usr/bin/env bash
#SBATCH --job-name=stageb_corr_final
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --account=ce25
#SBATCH --constraint=xenon-8452Y
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

set -euo pipefail
: "${STAGEB_REPO_ROOT:?configure the clean M3 clone path}"
: "${STAGEB_DRIVER:?configure the corrected M3 driver path}"
: "${STAGEB_FINALIZER_PYTHON:?configure the finalizer interpreter}"
: "${STAGEB_REGISTRATION_BUNDLE:?configure the frozen registration bundle}"
: "${STAGEB_OUTPUT_ROOT:?configure the output namespace}"

cd "${STAGEB_REPO_ROOT}"
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1
exec "${STAGEB_FINALIZER_PYTHON}" "${STAGEB_DRIVER}" finalize-inspection-only \
  --registration "${STAGEB_REGISTRATION_BUNDLE}" \
  --output-root "${STAGEB_OUTPUT_ROOT}"
