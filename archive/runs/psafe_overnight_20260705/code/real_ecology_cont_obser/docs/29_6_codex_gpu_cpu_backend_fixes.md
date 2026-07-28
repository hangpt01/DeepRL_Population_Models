# Codex Fix Note: GPU/CPU Backend Audit Items

Date: 2026-07-04

Scope: only `discrete_action_cont_obser/real_ecology_cont_obser/`.

This implements the fixes requested after `29_6_codex_gpu_cpu_backend_audit.md`.

## Fixes Applied

### 1. Backend labels are now workload-honest

`src/real_ecology_benchmark/backend.py` now has a workload support gate:

- `resolve_backend_for_workload(...)`
- `ensure_backend_supports_workload(...)`

CuPy is currently supported for the vectorized mechanistic transition kernel
(`mechanistic_transition`) and for PLUS real-setpoint method rows, whose hot path
is the candidate-bank mechanistic transition. Other full method/gate rows still
contain NumPy-only planners and learned dynamics, so:

- strict `backend=cupy` for non-PLUS method/gate rows raises `BackendUnavailable`;
- non-strict `backend=cupy` for unsupported rows downgrades to `effective=numpy`
  and records the fallback reason.
- strict `backend=cupy` for `method:plus` on `real_setpoint` cells records
  `effective=cupy` and `compute_backend_acceleration_scope=plus_mechanistic_transition`.

`pipeline.run_method`, `pipeline.run_oracle_state_ablation`, and `gate.run_decision_gate` all use this workload-aware resolver.

### 2. Backend metadata is carried into episode rows and aggregates

`ContinuousEvaluator` now writes backend metadata into `episodes.csv`, not only `summary.json`.

`manifest.aggregate_summaries` now groups by `compute_backend_effective` for:

- `model_return_mean`
- `efficiency_mean`
- paired seed comparisons
- `beats_both_cells`
- `beats_both_rate`

`scripts/summarize_real_outputs.py` also reports backend-separated efficiency and safe/yield deltas.

### 3. Gate artifacts are backend-namespaced

Gate row outputs now write to:

```text
gates/backend_<effective>/reward_<mode>/<population>/<family>/sigma_<x>.json
```

`run_real_manifest_row.py --require-gate` looks in the same backend namespace, preventing CPU/GPU gate collisions.

### 4. Gate runner accepts backend overrides

`scripts/run_real_gate_row.py` now supports:

- `--backend`
- `--device`
- `--backend-strict`

It shares the same override precedence as method rows: CLI flag, then `$BACKEND`, then config.

`scripts/slurm/run_real_gate_row.sh` forwards `$BACKEND`, `$DEVICE`, and `$BACKEND_STRICT`.

### 5. Slurm wrappers were repaired

`scripts/slurm/run_real_row_gpu.sh` now uses the same `SLURM_SUBMIT_DIR` root detection as the CPU wrappers, avoiding the spool-copy root bug.

The GPU wrapper documentation no longer implies that `GPU_PARTITION` changes the Slurm partition after submission. It documents the real launch shape:

```bash
sbatch --partition=<gpu-partition> --array=0-N%K \
  --time=... scripts/slurm/run_real_row_gpu.sh MANIFEST.csv
```

The CPU row/gate wrappers now also forward backend env overrides when set.

## Verification

Unit tests:

```text
PYTHONPATH=src python -m unittest discover -s tests -v
Ran 25 tests in 3.287s
OK
```

Smoke, NumPy:

```text
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml --backend numpy
status: ok
compute_backend_requested: numpy
compute_backend_effective: numpy
output_dir includes backend_numpy
```

Smoke, requested CuPy with non-strict fallback on this CPU-only node:

```text
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml --backend cupy --backend-strict false
status: ok
compute_backend_requested: cupy
compute_backend_effective: numpy
compute_backend_fallback_reason: No module named 'cupy'
output_dir includes backend_numpy
```

Gate row smoke:

```text
PYTHONPATH=src python scripts/run_real_gate_row.py outputs/real_reward_modes_20260703/manifest_gates.csv 0 \
  --config configs/real_probe.yaml \
  --output-root /tmp/real_gate_backend_smoke \
  --episodes 1 \
  --backend numpy
```

Wrote:

```text
/tmp/real_gate_backend_smoke/gates/backend_numpy/reward_safe/egyptian_vulture/ricker/sigma_0.json
```

Shell/script syntax:

```text
bash -n scripts/slurm/run_real_row.sh
bash -n scripts/slurm/run_real_gate_row.sh
bash -n scripts/slurm/run_real_row_gpu.sh
python -m py_compile scripts/run_real_manifest_row.py scripts/run_real_gate_row.py scripts/summarize_real_outputs.py
```

All passed.

## Remaining Intentional Limitation

This does not make the full methods GPU-native. It prevents misleading GPU labels until the full planner/model paths are ported. The current fast path remains the NumPy vectorized implementation, which is the honest backend for benchmark rows today.
