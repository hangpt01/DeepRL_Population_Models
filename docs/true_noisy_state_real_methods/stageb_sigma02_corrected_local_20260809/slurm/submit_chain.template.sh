#!/usr/bin/env bash
# TEMPLATE ONLY — DO NOT RUN FROM THE MAC.
set -euo pipefail
: "${STAGEB_SLURM_TEMPLATE_ROOT:?configure template directory on M3}"

arm_o_job="$(sbatch --parsable "${STAGEB_SLURM_TEMPLATE_ROOT}/arm_o_array.sbatch")"
gate_job="$(sbatch --parsable --dependency="afterok:${arm_o_job}" \
  "${STAGEB_SLURM_TEMPLATE_ROOT}/arm_o_gate.sbatch")"
arm_t_job="$(sbatch --parsable --dependency="afterok:${gate_job}" \
  "${STAGEB_SLURM_TEMPLATE_ROOT}/arm_t_array.sbatch")"
finalizer_job="$(sbatch --parsable --dependency="afterany:${arm_t_job}" \
  "${STAGEB_SLURM_TEMPLATE_ROOT}/finalizer.sbatch")"

printf 'arm_o=%s\ngate=%s\narm_t=%s\nfinalizer=%s\n' \
  "${arm_o_job}" "${gate_job}" "${arm_t_job}" "${finalizer_job}"
