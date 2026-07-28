# Codex Audit: GPU/CPU Backend Implementation

Date: 2026-07-04

Scope: `discrete_action_cont_obser/real_ecology_cont_obser/` only. I audited Claude's report in `docs/29_6_claude_gpu_cpu_backend_implementation.md` against the live code and ran the local tests/smoke checks.

## Verdict

The NumPy backend selector and the vectorized mechanistic transition are real and useful. The test suite passes, NumPy smoke passes, and the non-strict CuPy fallback path records the fallback correctly on this node.

However, I would not launch a fair CPU-vs-GPU benchmark yet. The implementation currently has four launch-blocking issues for GPU/fair-backend runs:

1. `compute_backend_effective=cupy` can be recorded even for methods whose policy/planner/dynamics still run almost entirely on NumPy/CPU.
2. Aggregation and summary code pool CPU/GPU outputs together instead of grouping by backend.
3. The GPU Slurm wrapper likely fails under `sbatch` because it lacks the CPU wrapper's `SLURM_SUBMIT_DIR` root handling.
4. Gate runs cannot be selected/namespaced by backend from the row wrapper, so CPU/GPU gate outputs can collide.

So: approve the CPU vectorization/foundation, but fix the above before using GPU results or comparing backend fairness.

## Commands Run

From `discrete_action_cont_obser/real_ecology_cont_obser/`:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Result:

```text
Ran 23 tests in 3.326s
OK
```

Smoke, NumPy backend:

```bash
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml --backend numpy
```

Result: `status: ok`, `compute_backend_requested: numpy`, `compute_backend_effective: numpy`, output path under `.../backend_numpy/...`.

Smoke, requested CuPy with non-strict fallback:

```bash
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml --backend cupy --backend-strict false
```

Result: `status: ok`, `compute_backend_requested: cupy`, `compute_backend_effective: numpy`, fallback reason records `No module named 'cupy'`, output path under `.../backend_numpy/...`.

Wrapper syntax:

```bash
bash -n scripts/slurm/run_real_row_gpu.sh
```

Result: syntax OK.

## Verified Correct

### Backend resolver exists and records fallback

`src/real_ecology_benchmark/backend.py:33-45` imports CuPy with a real-device check and returns a fallback reason. `resolve_backend` handles strict and non-strict behavior in `backend.py:97-120`. Metadata fields are exposed by `Backend.to_dict` in `backend.py:78-85`.

This supports Claude's claim that strict missing-CuPy raises and non-strict fallback is recorded.

### Config has a compute block

`src/real_ecology_benchmark/config.py:221-236` defines `ComputeConfig(backend, device, strict)` and validates the backend/device. `BenchmarkConfig` includes it in `config.py:239-257`, and `load_config` parses the YAML `compute:` block in `config.py:268-291`.

### Pipeline resolves backend and namespaces method outputs

`src/real_ecology_benchmark/pipeline.py:207-210` resolves and activates the backend at the start of `run_method`. `pipeline.py:249-254` records `compute_backend_requested`, `compute_backend_effective`, `compute_backend_device`, `compute_backend_strict`, and fallback reason in `summary.json`.

Output path namespacing by effective backend is implemented in `_reward_mode_output_root` at `pipeline.py:39-44`, producing paths of the form:

```text
evaluation/<population>/<family>/sigma_<x>/backend_<effective>/reward_<mode>/<method>/<filter>/
```

### Vectorized mechanistic transition is implemented

`src/real_ecology_benchmark/beliefs.py:27-81` defines `real_next_states`, using the active backend's `xp` namespace for the batch computation and returning through `b.to_numpy(...)`.

`MechanisticProposal` captures the active backend at construction in `beliefs.py:353-357` and uses the vectorized helper in `beliefs.py:380-386`.

The gate's exact proposal uses the same helper in `src/real_ecology_benchmark/gate.py:30-52`, so the exact/clairvoyant proposal and the mechanistic proposal now share the transition implementation.

### Tests cover the main NumPy/fallback surface

The 23-test suite includes backend config/metadata/fallback tests and vectorized-vs-scalar parity. I did not independently reproduce the reported `212s -> 4.7s` PLUS timing, but the vectorization is real and the local smoke path supports the direction of the claim.

## Blocking Issues

### B1. Strict/fair GPU backend is not method-supported

The backend flag is global, but only the mechanistic proposal/exact gate transition is backend-aware. Several methods can record `compute_backend_effective=cupy` while still running their planner/model hot paths on NumPy/CPU.

Evidence:

- `pipeline.py:207-210` resolves a single active backend and `pipeline.py:249-254` records it in every method summary.
- `src/real_ecology_benchmark/planning.py:8` imports NumPy only, and `planning.py:58-151` uses NumPy arrays/operators throughout `ParticleMPC`.
- `src/real_ecology_benchmark/dynamics.py:1` imports NumPy only. `_design` and prediction paths use NumPy in `dynamics.py:17-58` and `dynamics.py:135-151`.
- `src/real_ecology_benchmark/methods/mopo.py:21-48` uses `ContinuousDynamicsEnsemble` plus `ParticleMPC`, so its learned-dynamics/planning path is still NumPy/CPU.
- `src/real_ecology_benchmark/methods/bamcts.py:61-118` is recursive Python/NumPy-style tree search, not backend-aware.

Why this matters: with CuPy installed, a run can be labeled `backend_cupy` even though MOPO/BAMCTS/learned-dynamics planning are not actually using GPU kernels. That violates the intended fair-backend comparison and makes backend-tagged results misleading.

Recommended fix:

- Add an explicit per-method backend support matrix.
- In `strict` CuPy mode, fail methods that do not have a GPU-supported hot path.
- In non-strict mode, either downgrade unsupported methods to `effective=numpy` or record both a global backend and a per-method/per-component effective backend.
- Do not treat the current implementation as "all seven methods GPU-ready."

### B2. Aggregation pools CPU/GPU outputs

Output directories include `backend_<effective>`, but aggregation and summary metrics do not group by backend. If both CPU and GPU runs exist under one output root, some metrics will be pooled or overwritten conceptually.

Evidence:

- `src/real_ecology_benchmark/manifest.py:96-108` builds `model_return_mean` keyed by `(reward_mode, model)`, with no backend key.
- `manifest.py:124-148` builds efficiency metrics keyed by `(reward_mode, model, filter)`, with no backend key.
- `manifest.py:149-158` constructs paired episode keys from `population`, `family`, `sigma_obs`, `reward_mode`, `method`, `filter`, `seed`, and `episode`, but not backend.
- `manifest.py:190-226` computes `beats_both` without backend in the grouping.
- `src/real_ecology_benchmark/evaluator.py:160-198` writes episode rows without backend metadata, so `episodes.csv` cannot be cleanly grouped by backend except by parsing the path.
- `scripts/summarize_real_outputs.py:34-44` groups safe/yield deltas without backend, and `summarize_real_outputs.py:60-84` prints efficiency without backend.

Recommended fix:

- Add backend metadata to episode rows, or carry it explicitly when loading summaries/episode CSVs.
- Include backend in aggregate keys for `model_return_mean`, efficiency, paired comparisons, and any safe-vs-yield deltas.
- Make the summarizer print backend-separated tables.

### B3. GPU Slurm wrapper likely has the old spool-root bug

The CPU wrapper handles Slurm's copied/spooled script behavior; the GPU wrapper does not.

Evidence:

- `scripts/slurm/run_real_row.sh:12-18` uses `SLURM_SUBMIT_DIR` when present, otherwise derives the root from `BASH_SOURCE`.
- `scripts/slurm/run_real_row_gpu.sh:25-26` always derives `ROOT` from the script path:

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
```

Under `sbatch`, `BASH_SOURCE[0]` may point to Slurm's spool copy rather than the repository script. Then `ROOT/scripts/run_real_manifest_row.py` in `run_real_row_gpu.sh:46-53` can resolve to the wrong location.

Recommended fix: copy the CPU wrapper's `SLURM_SUBMIT_DIR` fallback logic into `run_real_row_gpu.sh`.

### B4. `GPU_PARTITION` is documented but not used

The GPU wrapper comments say to set `GPU_PARTITION`, but the script never passes it to Slurm.

Evidence:

- `scripts/slurm/run_real_row_gpu.sh:3-7` documents `GPU_PARTITION`.
- `run_real_row_gpu.sh:19-23` checks that the variable is non-empty.
- There is no `#SBATCH --partition=...` directive and no way for the script itself to set a partition at runtime.

This means `GPU_PARTITION=a100 sbatch scripts/slurm/run_real_row_gpu.sh ...` will not request the `a100` partition unless the `sbatch` command itself also includes `--partition=a100`.

Recommended fix:

- Either remove the misleading env-var behavior and document `sbatch --partition=<gpu_partition> ...`, or provide a small submission wrapper that passes `--partition="$GPU_PARTITION"` to `sbatch`.

### B5. Gate backend control and path isolation are incomplete

Method rows can select backend via CLI/env, but gate rows cannot.

Evidence:

- `scripts/run_real_gate_row.py:20-26` accepts only manifest/index/config/output-root/episodes. It has no `--backend`, `--device`, or `--backend-strict` override.
- `scripts/run_real_gate_row.py:31-34` writes `gates/<cell>.json`; there is no `backend_<effective>` namespace in the gate output path.
- `scripts/slurm/run_real_gate_row.sh` is CPU-only and does not expose backend env overrides.
- `src/real_ecology_benchmark/gate.py:176-208` does resolve `cfg.compute` and records backend in the result, but the row runner cannot align gate backend with method backend unless a separate config file is created per backend.

Recommended fix:

- Add the same backend/device/strict overrides to the gate row runner.
- Namespace gate outputs by effective backend, for example `gates/backend_<effective>/<cell>.json`.
- Include backend in any gate aggregation/reporting.

## Medium / Low Issues

### M1. Report overstates "shared particle-MPC scoring" backend support

`backend.py:1-5` says the backend abstraction is for "shared particle-MPC scoring." In the live code, `ParticleMPC` itself is still pure NumPy (`planning.py:8`, `planning.py:58-151`). The backend-aware piece is the mechanistic transition used inside proposal sampling, not the whole MPC scorer.

This is a documentation/reporting mismatch unless `ParticleMPC` is actually ported to the backend namespace.

### M2. CuPy path is not numerically validated on a GPU node

Claude's caveat is fair: this node lacks CuPy/GPU, so strict CuPy execution has not been run. The code has import/fallback tests, but no actual GPU numeric test yet.

Before using GPU outputs, run at least:

- one strict CuPy smoke row,
- vectorized transition parity on GPU against the scalar reference,
- one PLUS row and one learned-dynamics row, with backend metadata checked.

### M3. PLUS candidate batching is still deferred

`src/real_ecology_benchmark/methods/plus.py:87-93` still loops over all 21 candidates in Python. This is scientifically fine because the paper's 21 candidates are preserved, and the vectorized scorer already removes the biggest CPU bottleneck. But it means the plan item "batch all 21 PLUS candidates into one kernel" was not implemented. Claude's report acknowledges this as unnecessary for the speedup, which is a reasonable engineering choice, but it should be recorded as deferred rather than complete.

## Report Accuracy

Accurate claims:

- Backend resolver, config block, CLI flags, and summary metadata exist.
- NumPy/fallback behavior is tested and passes locally.
- Mechanistic proposal and exact gate proposal use a shared vectorized transition helper.
- Output paths for method rows include `backend_<effective>` and `reward_<mode>`.
- Local test count is correct: 23 tests pass.
- CuPy/GPU execution has not been validated on this node.

Overstated or incomplete claims:

- "Compute backend selector you asked for, fair and tested" is too strong. The selector exists, but backend fairness is not enforced per method and aggregation pools backends.
- "GPU wrapper with `--gres=gpu:1`; set `GPU_PARTITION` on submit" is incomplete. `GPU_PARTITION` is checked but not used by the script; partition must be passed to `sbatch` or implemented by a submission helper.
- "Shared particle-MPC scoring" is not accurate for the current code. `ParticleMPC` remains NumPy-only.
- The PLUS timing improvement claim was not independently reproduced in this audit. I verified the vectorization/tests/smoke, not the full timing table.

## Required Fixes Before Launching GPU/Fair-Backend Jobs

1. Add per-method backend support enforcement so strict CuPy cannot silently label CPU-only methods as GPU runs.
2. Add backend to aggregation keys and episode/summary records.
3. Fix `run_real_row_gpu.sh` root handling using the CPU wrapper's `SLURM_SUBMIT_DIR` logic.
4. Make GPU partition selection real in the launch path.
5. Add backend CLI/env overrides and backend-namespaced outputs for gate rows.
6. Run a strict CuPy smoke/probe on a real GPU node before trusting GPU results.

After those fixes, the CPU vectorized implementation looks like a strong baseline and may already be fast enough to rerun the experiment without requiring GPU acceleration.
