# 29/6 Real-Ecology GPU/CPU Backend Implementation Plan

## Purpose

Add an explicit CPU/GPU backend selection layer for the real-ecology benchmark so
experiments can be run fairly under a single declared compute backend. The goal
is not to reduce PLUS's paper setting of 21 candidate models. The goal is to
preserve that setting while making the expensive PLUS planning path batchable on
GPU.

This is a plan only. No implementation should happen until the plan is audited
and approved.

## Current Code Facts

- The package currently uses NumPy throughout and has no live `torch`, `cupy`,
  `jax`, `numba`, or CUDA backend imports.
- `PLUSPolicy` is expensive because it loops over 21 candidate Ricker/K models
  and scores the same MPC sequence set for each candidate.
- `ParticleMPC.score_sequences()` is the shared hot path for PLUS, MOPO, MOOR,
  RefPlan, and the decision gate.
- The current PLUS cost is approximately:

```text
21 candidates * 96 sequences * 32 particles * 5 horizon steps * evaluation decisions
```

- With the current pilot settings, PLUS rows have averaged about 4.2 hours on
  CPU, while fast non-PLUS rows averaged about 6 minutes.
- Several methods are not actually neural-network deep RL in this implementation:
  OGSRL is a NumPy linear softmax policy with rollouts/KNN guardian, Delphic is
  mostly linear/ridge heads, and BAMCTS is recursive Python tree search.

## Fairness Rule

Final ranking tables must not silently mix CPU and GPU implementations.

For a ranking run:

- Use one declared backend for all included methods in that table.
- Record both requested and effective backend in every row summary.
- If a method cannot honor the requested backend under strict mode, the row
  should fail rather than silently falling back to CPU.
- If a method is deliberately CPU-only, either exclude it from the GPU table or
  report it in a separate CPU-only/unsupported section.

## Proposed Config Surface

Add a new compute config:

```yaml
compute:
  backend: numpy   # numpy | cupy
  device: 0
  strict: true
```

Semantics:

- `backend: numpy`: current CPU behavior and default.
- `backend: cupy`: request CuPy/CUDA GPU execution for supported hot paths.
- `device`: GPU device id used by the backend.
- `strict: true`: fail if the requested backend is unavailable or unsupported by
  the requested method.
- `strict: false`: allow explicit CPU fallback, but record fallback in
  `summary.json` and logs.

Recommended final setting for fair comparisons:

```yaml
compute:
  backend: cupy
  device: 0
  strict: true
```

## Files To Modify

Expected new file:

- `src/real_ecology_benchmark/backend.py`

Expected modified files:

- `src/real_ecology_benchmark/config.py`
- `src/real_ecology_benchmark/planning.py`
- `src/real_ecology_benchmark/methods/plus.py`
- `src/real_ecology_benchmark/pipeline.py`
- `src/real_ecology_benchmark/cli.py`
- `src/real_ecology_benchmark/manifest.py`
- `scripts/run_real_manifest_row.py`
- `scripts/run_real_gate_row.py`
- `scripts/slurm/run_real_row.sh`
- potentially add `scripts/slurm/run_real_row_gpu.sh`
- `scripts/summarize_real_outputs.py`
- tests under `tests/`

No files outside `discrete_action_cont_obser/real_ecology_cont_obser/` should be
modified unless explicitly approved.

## Backend Layer Design

Create `backend.py` with small helpers:

- `resolve_backend(compute_cfg)`
- `xp()` or returned module object (`numpy` or `cupy`)
- `to_device(array)`
- `to_numpy(array)`
- `using_gpu()`
- `backend_name_requested`
- `backend_name_effective`

The backend layer should keep the CPU path as the reference implementation.

If `backend=cupy` and CuPy is unavailable:

- fail immediately in strict mode;
- optionally fall back to NumPy only when `strict=false`, while recording the
  fallback.

## Output Isolation

Backend must be included in output paths to avoid CPU/GPU clobbering.

Current style:

```text
evaluation/<population>/<family>/<sigma>/reward_safe/<method>/<filter>/
```

Proposed:

```text
evaluation/backend_numpy/<population>/<family>/<sigma>/reward_safe/<method>/<filter>/
evaluation/backend_cupy/<population>/<family>/<sigma>/reward_safe/<method>/<filter>/
```

Every `summary.json` should include:

```json
{
  "compute_backend_requested": "cupy",
  "compute_backend_effective": "cupy",
  "compute_device": 0,
  "compute_backend_strict": true
}
```

Aggregation should preserve backend as a grouping key.

## Phase 1: CPU Reference Lock

Before adding GPU behavior, lock current CPU behavior:

1. Add backend config with default `numpy`.
2. Ensure all existing tests pass unchanged under `backend=numpy`.
3. Add tests that `backend=numpy` summaries record backend metadata.
4. Confirm output path namespacing does not break aggregation.

Acceptance:

- CPU run results are unchanged except for new backend metadata/path prefix.
- Existing real-ecology tests pass.

## Phase 2: Shared ParticleMPC Backend

Port the shared MPC hot path in `planning.py` to backend-aware array operations.

Target function:

- `ParticleMPC.score_sequences()`

Methods benefiting from this shared path:

- MOPO
- MOOR
- RefPlan
- PLUS, partially before PLUS-specific batching
- decision gate

Implementation rule:

- Keep the NumPy branch as the reference.
- The CuPy branch should convert only the scoring tensors to device arrays and
  return NumPy outputs at the API boundary.
- Avoid leaking device arrays into public dataset/evaluator objects.

Acceptance:

- Tiny deterministic test compares NumPy vs CuPy scores/actions within tolerance.
- Reward, cost, collapse, and public control updates match CPU within tolerance.

## Phase 3: Batched PLUS GPU Path

PLUS currently loops over the 21 candidate models:

```text
for candidate in candidates:
    score MPC sequences
```

The GPU path should batch candidates:

```text
[candidate=21, sequence=96, particle=32, horizon=5]
```

This preserves the paper's 21 candidate setting.

Expected implementation:

- Add a PLUS-specific batched scorer for `control_mode="real_setpoint"`.
- Batch candidate `K` values and candidate belief banks together.
- Reuse one generated sequence matrix per decision, as the current code does.
- Combine candidate scores by posterior weights exactly as CPU does.
- Return the selected action and diagnostics.

Important parity details:

- Candidate posterior updates in `observe()` must remain equivalent.
- The public control state `rho`, `kappa`, `K_eff` must remain public only.
- No true latent state/private noise should be introduced into policy inputs.

Acceptance:

- On small deterministic cells, CPU PLUS and GPU PLUS choose the same action or
  have score differences within tolerance.
- `candidate_count` remains 21 by default.
- No PLUS-lite or reduced-candidate behavior is used unless separately flagged
  and reported.

## Phase 4: Method Support Matrix

Initial support expectation:

| Method | GPU Backend Plan | Notes |
| --- | --- | --- |
| PLUS | Full batched GPU path | Highest priority and biggest speedup. |
| MOPO | Shared GPU `ParticleMPC` | Medium benefit. |
| MOOR | Shared GPU `ParticleMPC` | Medium benefit. |
| RefPlan | Shared GPU `ParticleMPC` | Medium benefit, still loops over selected members unless further batched. |
| Gate | Shared GPU `ParticleMPC` | Useful for GPU-mode gate parity. |
| OGSRL | Optional later | Mostly NumPy linear policy/rollouts/KNN, not neural deep RL here. |
| Delphic | Optional later | Mostly small linear/ridge heads, low expected speedup. |
| BAMCTS | CPU-only initially | Recursive tree search with Python dicts; poor GPU fit. |

Strict GPU behavior:

- If `compute.backend=cupy` and `strict=true`, rows for unsupported methods
  should fail cleanly with an explicit unsupported-backend message.
- The manifest should optionally be able to exclude unsupported methods from GPU
  ranking runs.

## Phase 5: CLI And Slurm

Add CLI flags:

```bash
--backend numpy|cupy
--device 0
--backend-strict true|false
```

Update manifest runners so backend can come from:

1. config file;
2. environment variables such as `BACKEND=cupy`;
3. explicit CLI override.

Slurm:

- Keep CPU wrapper with `backend=numpy`.
- Add a GPU wrapper or GPU mode with:

```bash
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
BACKEND=cupy
```

The GPU wrapper should also set one CPU thread for BLAS unless profiling shows
otherwise:

```bash
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
```

## Phase 6: Tests

Add or update tests for:

1. Backend config parses and validates.
2. `backend=numpy` preserves existing behavior.
3. `backend=cupy` missing dependency fails in strict mode with a clear message.
4. `backend=cupy, strict=false` records CPU fallback if fallback is allowed.
5. Output paths include backend namespace.
6. `summary.json` records requested/effective backend.
7. CPU/GPU parity for `ParticleMPC.score_sequences()` on a tiny deterministic
   fixture.
8. CPU/GPU parity for PLUS action selection on a tiny deterministic fixture.
9. Leakage guard still passes: no true latent state/private noise is exposed to
   methods.
10. Aggregation groups or preserves backend.

GPU tests should skip if CuPy/CUDA is not available, except strict-missing tests
which should run everywhere.

## Phase 7: Validation Run

After implementation and tests:

1. Run CPU smoke:

```bash
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke \
  --config configs/real_smoke.yaml
```

2. Run GPU smoke on one supported method:

```bash
BACKEND=cupy PYTHONPATH=src python -m real_ecology_benchmark.cli run \
  --config configs/real_probe.yaml \
  --method plus \
  --filter learned
```

3. Run one CPU/GPU paired row for PLUS on the same cell and compare:

- chosen actions distribution;
- reward metrics;
- collapse metrics;
- runtime;
- backend metadata.

4. Only after the paired probe passes, submit a GPU pilot.

## Experiment Reporting Rules

For any result table:

- Include backend in the table caption or manifest metadata.
- Do not compare CPU and GPU rows as if they are one backend unless parity has
  been explicitly validated and the table says so.
- If unsupported methods are excluded from the GPU table, state that directly.
- Report runtime/peak memory separately, since backend choice affects efficiency
  and resource usage.

## Open Questions For Claude Audit

1. Should strict GPU mode exclude BAMCTS by manifest construction, or should it
   submit BAMCTS and fail loudly?
2. Should OGSRL be ported in phase 1, or is it acceptable to mark it CPU-only
   for the first GPU pilot?
3. Is CuPy the preferred backend on this cluster, or should the implementation
   target PyTorch/JAX instead if the module environment supports those better?
4. Should backend be part of dataset/cache paths? The transitions are backend
   independent, but learned filter caches and belief caches may differ if a GPU
   path changes numerical details.
5. What tolerance should define CPU/GPU parity for stochastic planning scores?

## Recommended MVP

Implement:

1. `compute.backend` config with output namespacing and metadata.
2. NumPy reference behavior unchanged.
3. CuPy strict/fallback availability checks.
4. GPU-aware shared `ParticleMPC.score_sequences()`.
5. Batched PLUS GPU scorer preserving `candidate_count=21`.
6. Tests for backend metadata, CPU reference, missing-GPU behavior, and tiny
   CPU/GPU parity when CuPy is available.
7. One GPU timing probe before any full rerun.

Do not reduce PLUS candidates. Do not silently mix CPU and GPU rows in one
ranking table.
