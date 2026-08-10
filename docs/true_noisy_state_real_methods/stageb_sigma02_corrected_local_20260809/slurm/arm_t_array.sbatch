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
: "${STAGEB_PYTHON:?configure the registered per-task interpreter wrapper}"
: "${STAGEB_REGISTRATION_BUNDLE:?configure the frozen registration bundle}"
: "${STAGEB_ARM_O_GATE_RECEIPT:?configure the successful Arm O gate receipt}"
: "${STAGEB_GATE_PUBLIC_KEY_FILE:?configure the 32-byte Ed25519 public verification key}"
: "${STAGEB_OUTPUT_ROOT:?configure the output namespace}"

cd "${STAGEB_REPO_ROOT}"
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec "${STAGEB_PYTHON}" "${STAGEB_DRIVER}" arm-t \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --registration "${STAGEB_REGISTRATION_BUNDLE}" \
  --arm-o-gate "${STAGEB_ARM_O_GATE_RECEIPT}" \
  --gate-public-key-file "${STAGEB_GATE_PUBLIC_KEY_FILE}" \
  --output-root "${STAGEB_OUTPUT_ROOT}"
