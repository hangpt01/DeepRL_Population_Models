# Plan: Single Real-Data Repo With Optional Dummy Set-Point Mode

Superseded after Claude audit. Use:

`discrete_action_cont_obser/docs/PLAN_single_real_data_repo_with_dummy_setpoint_REVISED_2026-07-09.md`

Audit file:

`discrete_action_cont_obser/docs/AUDIT_PLAN_single_real_data_repo_2026-07-09.md`

Date: 2026-07-09

Scope: `discrete_action_cont_obser/`

Status: planning document only. Do not implement, move, delete, or archive files
until this plan has been audited and explicitly approved.

## Goal

Keep one standalone folder/repo for the continuous-observation ecology benchmark.
The default and paper-facing setting should use real ecology data. A secondary
dummy/synthetic setting should remain available by argument/config for quick
experimentation, but it should use the same action semantics as the real setting:

- `r` is a per-action set-point, refreshed each step.
- `K` is cumulative through a public capacity accumulator.

This preserves the current real-ecology semantics while giving a small dummy
mode to play with.

## Current Code Facts To Preserve

The existing real package already implements the desired semantics:

- `real_ecology_cont_obser/src/real_ecology_benchmark/config.py`
  - `control_mode="real_setpoint"` is the default real mode.
  - `accumulator_decay_r=1.0`, `r_base_low=0.0`, and `r_base_high=0.0` make `rho`
    carry the selected action's set-point instead of accumulating growth.
  - `accumulator_decay_K=0.0` makes `kappa` cumulative.

- `real_ecology_cont_obser/src/real_ecology_benchmark/controls.py`
  - `advance_public_controls()` computes:
    - `next_rho = (1 - accumulator_decay_r) * rho + delta_r`
    - `next_kappa = (1 - accumulator_decay_K) * kappa + delta_K`
  - Under the real defaults this means `rho = delta_r(action)` each step and
    `kappa = cumulative sum(delta_K)`.

- `real_ecology_cont_obser/src/real_ecology_benchmark/actions.py`
  - Real actions are read from `real_ecology_data/action_effects_long.csv`.
  - `delta_r` is the set-point rate from the data table.
  - `delta_K` is the per-step cumulative capacity increment.
  - `stocking_delta` is direct translocation, nonzero only for the relevant real
    action.

- `real_ecology_cont_obser/src/real_ecology_benchmark/envs.py`
  - The real/control-enabled branch advances controls before growth.
  - It uses `private_r_eff(...)` for the set-point `r`.
  - It uses public `K_eff = clip(K_base + kappa, K_min, K_max)`.

The old top-level synthetic package is not the target behavior by default:

- `src/tier2_benchmark/` has `tier2_one_step` and `cumulative_capped`.
- `tier2_one_step` recomputes `r_base + delta_r` and `K_base + delta_K` per
  transition.
- `cumulative_capped` can accumulate both `r` and `K`.
- Neither is exactly the desired dummy mode unless we generalize the real
  set-point/cumulative-capacity branch.

## Recommended Architecture

Use the real package as the canonical codebase and add dummy data as a mode, not
as a separate copied package.

Target shape:

```text
discrete_action_cont_obser/
  pyproject.toml
  Makefile
  README.md
  configs/
    real_default.yaml
    real_smoke.yaml
    dummy_setpoint_default.yaml
    dummy_setpoint_smoke.yaml
  src/
    real_ecology_benchmark/
      ...
      realdata.py
      dummydata.py
  tests/
    test_real_ecology.py
    test_dummy_setpoint.py
  real_ecology_data/
    actions.csv
    species.csv
    species_lambda.csv
    action_effects_long.csv
    cost_sources.csv
    cost_anchors_portal.csv
    README.md
  docs/
    ...
```

The package name can stay `real_ecology_benchmark` for minimal disruption, or be
renamed later. The lower-risk first step is to keep the existing package name and
only change packaging/paths.

## Configuration Design

Add explicit setting fields:

```yaml
environment:
  data_mode: real        # real | dummy
  control_mode: setpoint_cumulative
```

Compatibility rule:

- Treat current `control_mode: real_setpoint` as an alias of
  `setpoint_cumulative` when `data_mode: real`.
- Keep existing real configs working while new configs use clearer names.

Behavior rule:

- `data_mode: real`
  - Reads `real_ecology_data/`.
  - Requires a real `population`.
  - Uses 11 real actions.
  - Default for all real experiment configs.

- `data_mode: dummy`
  - Uses synthetic/dummy profiles from code or small CSVs.
  - Should not read `real_ecology_data/`.
  - Uses a small action table, likely 5 or 10 actions.
  - Uses the same set-point `r` and cumulative `K` semantics.

## Dummy Data Design

Prefer a small `dummydata.py` first. It is easier to audit than creating another
CSV/table family before the semantics are stable.

Minimum dummy profile:

- `population = "Dummy baseline"`
- `N0 = 200.0`
- `K_base = 500.0`
- `K_max = 1000.0`
- `K_min = K_base`
- `r_min`, `r_max`, and dummy set-points chosen so all actions are valid.
- `safety_threshold` and `K_ref` set consistently with current reward code.

Dummy action contract:

- Action 0: no-op/baseline set-point.
- Growth actions set `r` to a new value for the current and following transition
  state, but repeated growth actions do not add on top of each other.
- Capacity actions add `delta_K` to cumulative `kappa`.
- Optional direct-state action can be omitted at first unless needed.

Example semantics:

```text
rho_t+1 = r_setpoint(action_t)
kappa_t+1 = clip_or_accumulate(kappa_t + delta_K(action_t))
r_eff_t+1 = clip(rho_t+1, r_min, r_max)
K_eff_t+1 = clip(K_base + kappa_t+1, K_min, K_max)
```

## Implementation Phases

### Phase 0: Audit Before Editing

Ask Claude to audit this plan against the live code before implementation.

Claude should specifically check:

- Whether `setpoint_cumulative` should be a new `control_mode` or a clearer alias
  for the existing `real_setpoint` branch.
- Whether `dummydata.py` is better than dummy CSV files for the first pass.
- Whether any method assumes `control_mode == "real_setpoint"` specifically and
  would need a compatibility helper.
- Whether package movement should happen before or after dummy mode is added.

### Phase 1: Package And Path Cleanup

Goal: make the real package runnable from the top-level `discrete_action_cont_obser/`.

Tasks:

- Move or copy `real_ecology_cont_obser/src/real_ecology_benchmark/` to
  `src/real_ecology_benchmark/`.
- Move `real_ecology_cont_obser/tests/test_real_ecology.py` to top-level
  `tests/`.
- Move real configs from `real_ecology_cont_obser/configs/` to top-level
  `configs/`.
- Update `pyproject.toml`:
  - project name can become `ecorl-real-ecology` or similar;
  - console script can become `ecology-benchmark`;
  - optionally keep `real-ecology` as a script alias.
- Update `Makefile` with real test/smoke targets.
- Update `realdata.DATA_DIR` so it reliably points to top-level
  `real_ecology_data/`.

Acceptance checks:

```bash
cd discrete_action_cont_obser
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
```

### Phase 2: Add Explicit Data Mode And Control Alias

Goal: make the intended semantics visible in config instead of hidden behind
`real_setpoint`.

Tasks:

- Add `data_mode` to `EnvironmentConfig`.
- Add `setpoint_cumulative` to accepted control modes.
- Keep `real_setpoint` accepted as a backward-compatible alias.
- Add helper functions such as:
  - `is_setpoint_cumulative(cfg)`
  - `uses_real_data(cfg)`
  - `uses_dummy_data(cfg)`
- Replace direct string checks where needed, especially in controls, reward,
  config loading, CLI overrides, pipeline cache validation, methods, and tests.

Acceptance checks:

- Existing real configs still pass unchanged.
- New real configs using `control_mode: setpoint_cumulative` pass.
- Dataset cache validation includes `data_mode` and effective control mode.

### Phase 3: Add Dummy Set-Point Data Mode

Goal: support `--data-mode dummy` or a dummy YAML config while preserving the real
default.

Tasks:

- Add `dummydata.py` or a small `dummy_ecology_data/` table.
- Add `dummy_environment(...)` builder analogous to `real_environment(...)`.
- Update `resolve_actions(cfg)` to dispatch:
  - real data -> `real_action_table(...)`
  - dummy data -> `dummy_action_table(...)`
- Ensure dummy `r` uses set-point semantics:
  - use `accumulator_decay_r=1.0`;
  - use `r_base_low=r_base_high=0.0`;
  - `private_r_eff()` returns/clips `rho`, not `r_base + rho`, under the generalized
    set-point mode.
- Ensure dummy `K` accumulates:
  - use `accumulator_decay_K=0.0`;
  - `K_eff = clip(K_base + kappa, K_min, K_max)`.

Acceptance checks:

```bash
cd discrete_action_cont_obser
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/dummy_setpoint_smoke.yaml
```

### Phase 4: Tests For Semantics

Add focused tests:

- Real data still reproduces `action_effects_long.csv`.
- Real `r` is set-point and does not accumulate across repeated rate actions.
- Real `K` accumulates and clips at `K_max`.
- Dummy mode does not read `real_ecology_data/`.
- Dummy `r` is set-point and does not accumulate across repeated rate actions.
- Dummy `K` accumulates and clips at `K_max`.
- Public/private leakage boundary remains unchanged:
  - methods can see public controls such as `rho`, `kappa`, `K_eff`;
  - evaluator-only truth remains private.

### Phase 5: Remove Or Archive Old Synthetic Package

Only after real and dummy set-point smokes pass:

- Archive or remove old `src/tier2_benchmark/`.
- Archive or remove old top-level synthetic configs/scripts that only target
  `tier2_benchmark`.
- Keep useful docs only if they explain historical design decisions.

Important: do not delete generated artifacts with a broad command until git
tracked status has been checked. Existing handoffs warn that some generated
outputs/logs may be tracked despite `.gitignore`.

### Phase 6: Generated Artifact Cleanup

Likely archive/remove candidates:

```text
outputs/
logs/
real_ecology_cont_obser/outputs/
real_ecology_cont_obser/log/
real_ecology_runs/
**/__pycache__/
**/.pytest_cache/
```

Keep or archive separately if provenance is still needed:

```text
real_ecology_runs/psafe_overnight_20260705/
```

Before cleanup, run:

```bash
cd /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models
git ls-files discrete_action_cont_obser/outputs discrete_action_cont_obser/logs \
  discrete_action_cont_obser/real_ecology_cont_obser/outputs \
  discrete_action_cont_obser/real_ecology_cont_obser/log \
  discrete_action_cont_obser/real_ecology_runs
```

If tracked files appear, decide whether to `git rm --cached` them, archive them,
or keep selected paper-facing artifacts.

## Final Validation Commands

From the final top-level folder:

```bash
cd discrete_action_cont_obser
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/dummy_setpoint_smoke.yaml
```

Optional static dependency checks:

```bash
rg -n "claude_build|from claude|import claude" discrete_action_cont_obser
rg -n "tier2_benchmark" discrete_action_cont_obser/src discrete_action_cont_obser/tests discrete_action_cont_obser/configs
```

Expected result:

- No runtime dependency on `claude_build`.
- No remaining runtime dependency on `tier2_benchmark` after old synthetic code is
  archived/removed.

## Suggested Audit Prompt For Claude

```text
Please audit this plan before any implementation:

discrete_action_cont_obser/docs/PLAN_single_real_data_repo_with_dummy_setpoint_2026-07-09.md

Check it against the live code under discrete_action_cont_obser/, especially:

1. Does the existing real_ecology_benchmark implementation already encode
   set-point r and cumulative K as the plan says?
2. Should the new shared mode be named control_mode="setpoint_cumulative", or
   should we keep/alias "real_setpoint" to reduce churn?
3. Is dummydata.py the safest first implementation for dummy mode, or should the
   dummy setting use CSVs matching real_ecology_data?
4. Which files/functions would need edits if real_ecology_cont_obser is promoted
   to the top-level src package?
5. Are there hidden method assumptions tied to control_mode == "real_setpoint",
   real population names, 11 actions, or realdata.NUM_REAL_ACTIONS?
6. Are the proposed tests sufficient to prove that r is set each step and K
   remains cumulative for both real and dummy modes?

Do not implement. Return findings, risks, and a corrected implementation plan.
```
