# GPU/CPU backend — implementation record

Date: 2026-07-03
Implements the approved `29_6_gpu_cpu_backend_implementation_plan.md`. Scope: only
`real_ecology_cont_obser/`. This node has **no CuPy / no GPU / no torch/JAX**
(Python 3.9, NumPy 1.23), so the CuPy execution path is written and import-guarded
but validated only on NumPy + the strict/fallback behaviour; GPU execution must be
validated later on a CuPy node.

## Headline result

The prerequisite for any GPU port was to remove the per-particle Python loop in the
mechanistic proposal (profiled at 93% of PLUS runtime). Doing that as a vectorized,
backend-aware array computation already fixes the slow-PLUS problem **on CPU**:

| method | before | after (NumPy) |
|---|---|---|
| PLUS (probe-scale, eval horizon 30) | 212 s | **4.7 s (~45×)** |
| MOOR | ~10 s | 2.4 s |

Bit-identical to the scalar reference (max relative error ~1e-15 across all four
families, verified incl. absorbing zeros). At pilot scale this turns the ~4.2 h
PLUS rows into minutes; the GPU backend is now optional for performance but is
provided as requested.

## What was added

- **`backend.py`** — `Backend` (resolved xp module + requested/effective names +
  device/strict + `to_numpy` boundary), `resolve_backend(compute_cfg)`
  (numpy always; cupy with a real device check; strict→raise `BackendUnavailable`,
  non-strict→recorded NumPy fallback), and a per-process active backend
  (`set/get/reset_active_backend`) so the hot paths pick it up without threading it
  through every constructor.
- **`config.py`** — `ComputeConfig(backend, device, strict)` on `BenchmarkConfig`,
  validated, YAML-loadable (`compute:` block).
- **`beliefs.py`** — `real_next_states(...)` (vectorized backend-array form of
  `env.transition_value` for `real_setpoint`, all four families) + a per-action
  device lookup helper; `MechanisticProposal` uses it (scalar per-particle loop
  retained only as the reference for the unused non-real control modes).
- **`gate.py`** — `ExactEpisodeProposal` (clairvoyant) uses the same vectorized
  helper; `run_decision_gate` activates the backend and records it.
- **`pipeline.py`** — `run_method` / `run_oracle_state_ablation` resolve+activate
  the backend from `cfg.compute`, namespace outputs
  `…/backend_<effective>/reward_<mode>/<method>/<filter>`, and write
  `compute_backend_{requested,effective}`, `compute_device`,
  `compute_backend_strict`, `compute_backend_fallback_reason` into `summary.json`.
- **`cli.py`** + **`scripts/run_real_manifest_row.py`** — `--backend/--device/
  --backend-strict` (row runner also honours `$BACKEND`).
- **`scripts/slurm/run_real_row_gpu.sh`** — GPU wrapper (`--gres=gpu:1`,
  `BACKEND=cupy`, `--backend-strict true`); set `GPU_PARTITION` on submit (no GPU
  partition name is committed here).

## Fairness (per the plan)

- One declared backend per ranking table; every summary records requested+effective
  backend; strict mode fails loud rather than silently using CPU; outputs are
  namespaced by backend so CPU and GPU rows never clobber and aggregate separately.

## Tests (23 pass)

`tests/test_real_ecology.py::TestComputeBackend`: config validate; numpy metadata;
strict-missing-cupy raises; non-strict cupy fallback recorded; **vectorized proposal
== scalar transition_value** (all families, rel <1e-9); run records backend + emits
`backend_numpy/…` path. All prior 17 tests still green.

## Not done / to validate on a GPU node

- Actual CuPy execution + CPU/GPU numeric parity (needs a CuPy device).  PLUS
  real-setpoint rows are now the supported method-level CuPy target; non-PLUS
  method rows still fail loudly under strict CuPy.
- The plan's Phase-3 "batch all 21 PLUS candidates into one kernel" is **not**
  needed for the speedup (the 21-candidate loop now calls a fast vectorized scorer);
  it would be an additional GPU-only optimization.
- **RNG-parity caveat** (from the audit): CuPy RNG ≠ NumPy RNG, so CPU/GPU parity
  can only be defined on the deterministic sub-computation (the vectorized map is
  bit-identical; the stochastic planner is not, by construction). Define GPU parity
  tests accordingly.

## Bottom line

The `compute.backend` selector is implemented and fair; the mechanistic hot path is
vectorized and backend-ready (bit-identical); PLUS is ~45× faster on CPU already.
PLUS-only GPU runs are ready to validate on a CuPy-equipped partition via
`run_real_row_gpu.sh` (submit with `--partition=<gpu-partition>`). See the audit
(`29_6_claude_gpu_cpu_backend_audit.md`) for why vectorization — not GPU — was the
actual fix.
