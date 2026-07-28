#!/usr/bin/env bash
# GPU variant of run_real_row.sh: requests one GPU and asks the row runner for
# the CuPy backend.  Use this for PLUS-only manifests; other full method rows
# still fail loudly in strict mode rather than being mislabeled as GPU runs.
# The GPU partition must be supplied to sbatch; Slurm does not expand an
# environment variable inside #SBATCH lines after submission.
#
#   sbatch --parsable --partition=<gpu-partition> --array=0-N%K \
#       --time=... scripts/slurm/run_real_row_gpu.sh MANIFEST.csv
#
# BACKEND_STRICT defaults to true.  PLUS real-setpoint rows are allowed because
# their candidate-bank mechanistic-transition hot path is CuPy-backed.  Other
# methods fail loudly instead of being mislabeled.  Set BACKEND_STRICT=false only
# for an explicitly recorded CuPy->NumPy fallback run.
#
#SBATCH --job-name=real-row-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=real_row_gpu_%A_%a.out
#SBATCH --error=real_row_gpu_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/src/real_ecology_benchmark" ]]; then
    ROOT="$SLURM_SUBMIT_DIR"
  else
    ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  fi
fi
MANIFEST="${1:?usage: run_real_row_gpu.sh MANIFEST.csv}"
INDEX="${SLURM_ARRAY_TASK_ID:?submit this script as an array job}"

if [[ -z "${CONFIG:-}" ]]; then
  case "$(basename "$MANIFEST")" in
    *probe*) CONFIG="$ROOT/configs/real_probe.yaml" ;;
    *) CONFIG="$ROOT/configs/real_experiment.yaml" ;;
  esac
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$(cd "$(dirname "$MANIFEST")" && pwd)}"

cd "$ROOT"
module load "${MINIFORGE_MODULE:-miniforge3}"
mamba activate "${CONDA_ENV:-pytorchrl}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# One BLAS thread; GPU does the heavy lifting.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export BACKEND="cupy"

ARGS=(
  "$ROOT/scripts/run_real_manifest_row.py"
  "$MANIFEST"
  "$INDEX"
  --config "$CONFIG"
  --output-root "$OUTPUT_ROOT"
  --backend cupy
  --backend-strict "${BACKEND_STRICT:-true}"
  --allow-uncalibrated
)
if [[ -n "${DEVICE:-}" ]]; then
  ARGS+=(--device "$DEVICE")
fi
if [[ "${REGENERATE:-false}" == "true" ]]; then
  ARGS+=(--regenerate)
fi
if [[ "${REQUIRE_GATE:-false}" == "true" ]]; then
  ARGS+=(--require-gate)
fi
if [[ "${REQUIRE_BAND:-false}" == "true" ]]; then
  ARGS+=(--require-band)
fi

python "${ARGS[@]}"
