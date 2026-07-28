# Revised Plan: Single Real-Data Repo With Optional Dummy Set-Point Mode

Date: 2026-07-09

Scope: `discrete_action_cont_obser/`

Supersedes:
`discrete_action_cont_obser/docs/PLAN_single_real_data_repo_with_dummy_setpoint_2026-07-09.md`

Incorporates audit:
`discrete_action_cont_obser/docs/AUDIT_PLAN_single_real_data_repo_2026-07-09.md`

Status: planning document only. Do not implement, move, delete, or archive files
until this revised plan is approved.

## Goal

Keep one standalone folder/repo for the continuous-observation ecology benchmark.
The default and paper-facing setting should use real ecology data. A secondary
dummy/synthetic setting should remain available by argument/config for quick
experimentation.

Both real and dummy modes must share the same action semantics:

- `r` is a per-action set-point, refreshed each step.
- `K` is cumulative through a public capacity accumulator.

## Audit-Driven Design Decision

Do not introduce `control_mode: setpoint_cumulative` in the first implementation.

Keep:

```yaml
environment:
  control_mode: real_setpoint
```

as the shared semantic mode for both real and dummy data. In the current code,
many literal branches on `control_mode == "real_setpoint"` encode the desired
set-point `r` and cumulative `K` behavior. If dummy used a new control-mode
string too early, it could silently fall through to additive Tier-2/Tier-3
semantics and lose the main property we want.

Add only one new axis:

```yaml
environment:
  data_mode: real   # real | dummy
```

Meaning:

- `control_mode` answers: what are the transition/action semantics?
- `data_mode` answers: where do populations/actions/caps come from?

This is the core correction from the audit.

## Current Code Facts To Preserve

The existing real package already implements the desired semantics.

`real_ecology_cont_obser/src/real_ecology_benchmark/config.py`:

- `control_mode="real_setpoint"` is the default.
- `accumulator_decay_r=1.0`, `r_base_low=0.0`, and `r_base_high=0.0` make `rho`
  carry the selected action's set-point instead of accumulating growth.
- `accumulator_decay_K=0.0` makes `kappa` cumulative.

`real_ecology_cont_obser/src/real_ecology_benchmark/controls.py`:

- `next_rho = (1 - accumulator_decay_r) * rho + delta_r`
- `next_kappa = (1 - accumulator_decay_K) * kappa + delta_K`
- Under the real defaults, `rho = delta_r(action)` each step and `kappa` is the
  cumulative sum of `delta_K`.
- `private_r_eff(...)` returns/clips `rho` under `control_mode="real_setpoint"`.

`real_ecology_cont_obser/src/real_ecology_benchmark/actions.py`:

- Real actions are read from `real_ecology_data/action_effects_long.csv`.
- `delta_r` is the set-point rate from the data table.
- `delta_K` is the per-step cumulative capacity increment.
- `stocking_delta` is direct translocation, nonzero only for the real
  translocation action.

`real_ecology_cont_obser/src/real_ecology_benchmark/envs.py`:

- The control-enabled branch advances controls before growth.
- It uses `private_r_eff(...)` for set-point `r`.
- It uses `K_eff = clip(K_base + kappa, K_min, K_max)`.

The old top-level synthetic package is not the target dummy behavior:

- `src/tier2_benchmark/` has `tier2_one_step` and `cumulative_capped`.
- `tier2_one_step` applies one-step increments.
- `cumulative_capped` can accumulate both `r` and `K`.
- The existing Tier-2/Tier-3 action tables should not be reused as dummy
  set-point tables because their `delta_r` values are increments, not absolute
  set-points.

## Target Folder Shape

Use the real package as the canonical package and add dummy data as a mode.

Target:

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
    real/
      test_real_ecology.py
    dummy/
      test_dummy_setpoint.py
    legacy_tier2/
      ...
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

Keep the package name `real_ecology_benchmark` for the first implementation to
reduce churn. A later pure rename can happen after the real/dummy semantics are
stable.

## Configuration Design

Add `data_mode` to `EnvironmentConfig`:

```python
data_mode: str = "real"
```

Allowed values:

```text
real
dummy
```

Keep:

```python
control_mode: str = "real_setpoint"
```

for both real and dummy set-point/cumulative-capacity mode.

Suggested helpers:

```python
def uses_real_data(cfg) -> bool:
    return cfg.data_mode == "real"

def uses_dummy_data(cfg) -> bool:
    return cfg.data_mode == "dummy"
```

Do not add `is_setpoint_cumulative(...)` unless doing a full atomic rename of all
literal `real_setpoint` branches. That rename is cosmetic and should not be
bundled with dummy-mode work.

Behavior:

- `data_mode="real"`
  - Reads `real_ecology_data/`.
  - Requires a real `population`.
  - Uses `realdata.NUM_REAL_ACTIONS == 11`.
  - Default for existing configs.

- `data_mode="dummy"`
  - Uses `dummydata.py`.
  - Must not read `real_ecology_data/`.
  - May use 5 or 10 actions, or another explicit dummy action count.
  - Still uses `control_mode="real_setpoint"`.

## Must-Fix Audit Findings

### F1: Keep `real_setpoint` As The Shared Semantic Mode

Do not rename `control_mode` in this work. Add `data_mode` and branch only
data-source seams on it.

### F2: Fix `realdata.DATA_DIR` After Package Promotion

Current `realdata.DATA_DIR` depends on the current nested path. After moving the
package from:

```text
real_ecology_cont_obser/src/real_ecology_benchmark/
```

to:

```text
src/real_ecology_benchmark/
```

the current parent-count logic resolves one level too high.

Post-move fix:

```python
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PACKAGE_ROOT / "real_ecology_data"
```

Add a test that `DATA_DIR / "species.csv"` exists.

### F3: Gate The 11-Action Requirement On Real Data

Current validation rejects any `control_mode="real_setpoint"` config unless it
has 11 actions. That is correct for real data but wrong for dummy.

Required validation logic:

- if `data_mode == "real"` and `control_mode == "real_setpoint"`:
  - require `num_actions == realdata.NUM_REAL_ACTIONS`;
- if `data_mode == "dummy"` and `control_mode == "real_setpoint"`:
  - allow the chosen dummy action counts, likely `{5, 10}` first.

### F4: Route Real-Data Seams On `data_mode`

Dummy must avoid real-data reads at these seams:

- `actions.resolve_actions(...)`
  - real -> `real_action_table(...)`
  - dummy -> `dummy_action_table(...)`
- `config.real_environment(...)`
  - keep real-only.
- new `config.dummy_environment(...)`
  - build dummy configs without `realdata`.
- `config.load_config(...)`
  - route compact config expansion by `data_mode`.
- `config.environment_with_kind_defaults(...)`
  - refresh real caps from `realdata`, dummy caps from `dummydata`.
- `beliefs.MechanisticProposal`
  - do not call `realdata.pops_for(...)[population]` for dummy.
- `cli.build_parser(...)`
  - the current `--population choices=realdata.population_names()` pattern reads
    `species.csv` at parser-construction time;
  - make the population choices lazy, omit real-only choices under
    `data_mode="dummy"`, or validate population after config loading;
  - otherwise a dummy CLI run is not fully isolated from `real_ecology_data/`,
    and loader-monkeypatch isolation tests will fail before dispatch.

### F5: Include `data_mode` In Dataset Cache Validation

`pipeline._validate_dataset_cell(...)` must include `data_mode` in its metadata
comparison. Output namespacing should also include `data_mode` so real and dummy
runs do not collide.

### F6: Build Fresh Dummy Set-Point Tables

Do not reuse Tier-2/Tier-3 action tables as dummy set-point tables.

Dummy action table requirements:

- `delta_r` values are absolute set-points, not increments.
- every `delta_r` lies within dummy `[r_min, r_max]`;
- action 0 is the baseline/no-op set-point;
- capacity actions use `delta_K` as cumulative capacity increments;
- direct `stocking_delta` can be omitted initially unless needed.

### F7: Do Not Remove `tier2_benchmark` Until Tests/Configs Are Detached

Top-level tests and configs currently target `tier2_benchmark`. Either:

- keep legacy synthetic tests/configs in a separate discovery root; or
- migrate/retire them in the same phase that removes `tier2_benchmark`.

Do not run a mixed half-moved top-level `unittest discover -s tests` where old
tests still import removed packages.

## Dummy Data Design

Use `dummydata.py` first. It is easier to audit than new CSV files and avoids a
second data-loader schema before the semantics are stable.

Suggested dummy profile:

```text
population = "Dummy baseline"
N0 = 200.0
K_base = 500.0
K_min = 500.0
K_max = 1000.0
r_min = -0.15
r_max = 0.40
K_ref = 500.0
safety_threshold = 50.0
```

Suggested dummy actions:

```text
a0: baseline/no-op, r_setpoint = baseline_r, delta_K = 0
a1: lower growth/harvest pressure, r_setpoint < baseline_r, delta_K = 0
a2: recovery support, r_setpoint > baseline_r, delta_K = 0
a3: moderate restoration, r_setpoint = baseline_r, delta_K > 0
a4: integrated conservation, r_setpoint > baseline_r, delta_K > 0
```

Exact values should be chosen so smoke tests are stable and all set-points are
inside `[r_min, r_max]`.

Dummy control invariants:

```text
rho_next = r_setpoint(action)
kappa_next = kappa + delta_K(action)
r_eff = clip(rho_next, r_min, r_max)
K_eff = clip(K_base + kappa_next, K_min, K_max)
```

Repeated growth actions must not add `r`.

## Implementation Phases

### Phase 0: Approval Gate

This file is still a plan. Stop here until the user approves implementation.

### Phase 1: Promote Real Package To Top-Level Source

Goal: make the current real package runnable from top-level
`discrete_action_cont_obser/`.

Tasks:

- Move `real_ecology_cont_obser/src/real_ecology_benchmark/` to
  `src/real_ecology_benchmark/`.
- Move real configs from `real_ecology_cont_obser/configs/` to top-level
  `configs/`, using distinct names such as `real_smoke.yaml`.
- Move real tests into a distinct top-level root, such as `tests/real/`.
- Update `pyproject.toml`:
  - update project name if desired;
  - add console script for `real_ecology_benchmark.cli:main`;
  - keep old package coexistence during transition if needed.
- Update `Makefile` with real-only and all-test targets.
- Fix `realdata.DATA_DIR` per F2.

Acceptance:

```bash
cd discrete_action_cont_obser
PYTHONPATH=src python -m unittest discover -s tests/real -v
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
```

### Phase 2: Add `data_mode` Plumbing

Goal: make real-vs-dummy a data-source choice without changing the shared
`control_mode="real_setpoint"` semantics.

Tasks:

- Add `data_mode` to `EnvironmentConfig`, defaulting to `"real"`.
- Validate `data_mode in {"real", "dummy"}`.
- Gate the real 11-action requirement on `data_mode == "real"`.
- Include `data_mode` in environment metadata, dataset cache validation, and
  output namespacing.
- Add `uses_real_data(...)` and `uses_dummy_data(...)` helpers if they make the
  six data-source seams clearer.

Acceptance:

- Existing real configs that omit `data_mode` still pass.
- Real smoke still passes.
- Cache validation records and checks `data_mode`.

### Phase 3: Add Dummy Set-Point Data

Goal: support dummy configs without reading `real_ecology_data/`.

Tasks:

- Add `dummydata.py`.
- Add `dummy_environment(...)`.
- Add `dummy_action_table(...)`.
- Route `resolve_actions(...)` on `data_mode`.
- Route `load_config(...)` compact expansion on `data_mode`.
- Route `environment_with_kind_defaults(...)` on `data_mode`.
- Fix `beliefs.MechanisticProposal` so dummy caps come from dummy config/data, not
  from `realdata.pops_for(...)`.

Dummy config must pin:

```yaml
environment:
  data_mode: dummy
  control_mode: real_setpoint
  accumulator_decay_r: 1.0
  accumulator_decay_K: 0.0
  r_base_low: 0.0
  r_base_high: 0.0
```

Acceptance:

```bash
cd discrete_action_cont_obser
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/dummy_setpoint_smoke.yaml
```

### Phase 4: Tests

Add tests for:

- real data still reproduces `action_effects_long.csv`;
- real set-point non-accumulation:
  - apply the same growth action twice;
  - assert `rho == delta_r`, not `2 * delta_r`;
  - assert `r_eff` does not accumulate;
- real cumulative `K`:
  - repeated capacity actions increase `kappa`;
  - `K_eff` clips at `K_max`;
- dummy mode does not read `real_ecology_data/`:
  - monkeypatch `realdata.actions_for`, `realdata.effects_for`, and
    `realdata.pops_for` to raise, then construct/run a dummy env/config path;
  - if testing through the CLI, first fix the eager parser
    `realdata.population_names()` seam listed in F4;
- dummy set-point non-accumulation, same as real;
- dummy cumulative `K`, same as real;
- dummy out-of-cap `delta_r` raises through the `private_r_eff` guard;
- real/dummy dataset cache cross-use is rejected;
- `realdata.DATA_DIR / "species.csv"` exists after the package move;
- public/private leakage boundary remains unchanged:
  - public info may expose `rho`, `kappa`, and `K_eff`;
  - `state`, `r_eff_true`, hidden parameters, and process noise stay
    evaluator-only.

### Phase 5: Retire Or Archive Old Synthetic Code

Only after real and dummy set-point smokes pass:

- detach top-level tests/configs from `tier2_benchmark`;
- archive or remove old `src/tier2_benchmark/`;
- archive or remove synthetic-only scripts/configs that target `tier2_benchmark`;
- keep docs only if they explain historical design choices needed for the paper
  or future work.

Do not delete `tier2_benchmark` while tests or configs still import it.

### Phase 6: Generated Artifact Cleanup

Likely archive/remove candidates:

```text
outputs/
logs/
real_ecology_cont_obser/outputs/
real_ecology_cont_obser/log/
**/__pycache__/
**/.pytest_cache/
```

Paper/provenance bundle:

```text
real_ecology_runs/psafe_overnight_20260705/
```

Archive this bundle before removing it. Preserve at least paper-facing analysis,
figures, metrics, rollups, decision notes, manifests, and frozen code/data
snapshots if provenance still matters.

Before any cleanup, check tracked files:

```bash
cd /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models
git ls-files discrete_action_cont_obser/outputs discrete_action_cont_obser/logs \
  discrete_action_cont_obser/real_ecology_cont_obser/outputs \
  discrete_action_cont_obser/real_ecology_cont_obser/log \
  discrete_action_cont_obser/real_ecology_runs
```

If tracked files appear, decide explicitly whether to keep them, archive them, or
remove them from git. Do not rely on `.gitignore` alone.

Keep `real_ecology_data/` tracked. It is the authoritative real data.

## Final Validation Commands

From the final top-level folder:

```bash
cd discrete_action_cont_obser
PYTHONPATH=src python -m unittest discover -s tests/real -v
PYTHONPATH=src python -m unittest discover -s tests/dummy -v
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/dummy_setpoint_smoke.yaml
```

After `tier2_benchmark` is retired:

```bash
rg -n "tier2_benchmark" discrete_action_cont_obser/src discrete_action_cont_obser/tests discrete_action_cont_obser/configs
```

Expected result:

- no runtime dependency on `claude_build`;
- no runtime dependency on `tier2_benchmark` after retirement;
- real mode defaults to `data_mode="real"`;
- dummy mode runs with `data_mode="dummy"` and `control_mode="real_setpoint"`;
- both modes have set-point `r` and cumulative `K`.

## Paste-Ready Implementation Prompt

```text
Implement the revised plan only after reading these files:

1. discrete_action_cont_obser/docs/PLAN_single_real_data_repo_with_dummy_setpoint_REVISED_2026-07-09.md
2. discrete_action_cont_obser/docs/AUDIT_PLAN_single_real_data_repo_2026-07-09.md
3. discrete_action_cont_obser/docs/HANDOFF_repo_cleanup_discrete_action_cont_obser_2026-07-09.md
4. discrete_action_cont_obser/docs/HANDOFF_real_ecology_data_setting.md

Important constraints:

- Stay inside discrete_action_cont_obser/.
- Do not edit or delete claude_build/.
- Do not delete generated outputs/logs or real_ecology_runs until git tracked
  files have been checked and I approve cleanup.
- Preserve real set-point r + cumulative K behavior.
- Add data_mode: real|dummy as the data-source axis.
- Keep control_mode="real_setpoint" as the shared semantic mode for both real
  and dummy. Do not introduce control_mode="setpoint_cumulative" in this pass.
- Fix the audit findings F2-F7, including the CLI parser seam where eager
  population choices read realdata during dummy runs.
- Add tests proving r does not accumulate and K does accumulate for both real and
  dummy modes.

First give me a short implementation checklist and wait for approval before
editing code.
```
