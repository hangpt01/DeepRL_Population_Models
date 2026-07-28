# Server Handoff — Add a Hidden-r/K Experiment Setting

**Date:** 2026-07-15  
**Scope:** implementation handoff for a new experiment setting, not a rewrite of
the completed motivation-native run.  
**Status:** plan/handoff only. No code in this file has been implemented by this
handoff.  
**Primary repo:** `/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models`

## 0. Short Instruction for the Next Agent

You are continuing from the completed motivation-native experiment and the
hidden-parameter exposure audit. Implement a **new additive real-ecology setting**
that hides r/K/action-table dynamics structure from methods, while preserving the
current public-table setting and all completed results exactly.

Do **not** mutate the existing `motivation_native` setting into something else.
The current setting is now a documented negative result. The new setting should
live beside it, with a separate config, manifest, output root, and docs/results.

Recommended working name:

```text
information_mode = hidden_rk
```

Keep the existing behavior as the default:

```text
information_mode = public_tables
```

If the code already has a better naming convention, use it, but keep the
semantic distinction crisp: current public-table setting remains intact; new
hidden-r/K setting is separate.

## 1. Read These Files First

Read these in order before editing code:

1. `docs/benchmark/SERVER_AUDIT_hidden_parameter_usage_by_method.md`
2. `docs/benchmark/SERVER_AUDIT_hidden_parameter_usage_by_method.tex`
3. `docs/benchmark/09_motivation_native_results.tex`
4. `docs/benchmark/SERVER_METHOD_ADAPTATIONS_motivation_native.md`
5. `docs/benchmark/SERVER_RESULTS_motivation_native_run.md`
6. `docs/benchmark/SERVER_RESULTS_general_method_audit.md`
7. `docs/benchmark/01_problem_setting_and_design.tex`
8. `docs/benchmark/02_real_ecology_data_actions_and_costs.tex`
9. `docs/benchmark/03_state_action_reward_reference.tex`
10. `docs/benchmark/04_algorithm_adaptations_and_claims.tex`
11. `docs/benchmark/05_implementation_and_code_map.md`
12. `docs/benchmark/README.md`

The key point from the audit:

> The current benchmark does not appear to leak private sidecar arrays directly.
> The larger issue is that the public spec exposes action-specific growth
> set-points, capacity-control state, species-scale K quantities, and K-derived
> safety thresholds. General methods use these mostly as features; mechanistic
> and native methods use them as exact model inputs.

## 2. Why This New Setting Exists

The completed motivation-native experiment was negative for the original paper
motivation:

- best native ecological solver wins or ties in 860/864 paired comparisons;
- `refplan` and `bamcts` never beat both native solvers;
- `ogsrl` beats both native solvers in only 4/288 paired cells;
- even the single-Ricker `moor_native` baseline beats the best general method on
  recoverable safe cells.

The current diagnosis is that the native baselines are strong because the
benchmark gives them public ecological structure:

- action-specific growth set-points;
- public capacity accumulator and effective carrying capacity;
- species carrying-capacity scale;
- action-specific `dK_step` and `dN`;
- K-derived reward/safety scale.

The new setting should test whether the ordering changes when methods must infer
the transition-relevant ecological quantities from public data instead of
reading them from species/action tables.

This is not tuning the old result until it looks better. It is a new ablation /
new setting answering a different question:

> In the same real-ecology environment, what happens when r/K/action effects are
> simulator-private rather than method-facing public structure?

## 3. Settled Design Decisions

### 3.1 Preserve the current setting

The current setting remains:

```text
data_mode = real
control_mode = setpoint_cumulative
information_mode = public_tables  # new name, if implemented
```

Do not break existing configs, manifests, tests, docs, or result paths.

Required no-op proof:

- With `information_mode=public_tables` or when the field is absent, the current
  motivation-native code path must remain bit-identical or behaviorally
  unchanged for the smoke rows.
- Existing tests must pass before and after adding hidden mode.

### 3.2 Add an additive hidden-r/K setting

The new setting should use the same true simulator:

- same 9 populations;
- same 4 dynamics families;
- same 11 real management actions;
- same true action effects internally;
- same observation noise grid;
- same safe/yield reward modes;
- same offline/evaluation protocol when the full run is launched.

But methods should not receive exact transition table information:

- no public `rho`, `next_rho`;
- no public `kappa`, `next_kappa`;
- no public `K_eff`, `next_K_eff`;
- no method-facing `r_setpoint_ricker` / `r_setpoint_lgm`;
- no method-facing lambda profile;
- no method-facing exact `dK_step`;
- no method-facing exact `dN` / `stocking_delta`;
- no method-facing exact `K_base`, `K_max`, `r_min`, `r_max` for transition
  modelling;
- no method-facing exact population-to-species table lookup for transition
  modelling;
- no native solver construction from exact `resolve_actions(cfg)` in hidden
  mode.

### 3.3 Keep the objective public in the first version

For the first hidden-r/K implementation, keep the **objective** public:

- logged rewards remain public training targets;
- action costs remain public;
- reward mode remains known;
- `safe` vs `yield` remains known;
- safety/collapse metrics remain evaluator outputs.

However, be explicit in the report that a public safety threshold or
`K_ref=K_base` can still indirectly leak scale. If the next agent wants a
stricter variant that hides objective scale too, require a separate design doc
first. Do not silently mix that with this first implementation.

Practical reason: hiding the transition model is the direct test of the current
diagnosis. Hiding the reward/objective scale would become a different POMDP and
would require reward-model learning or a public surrogate objective.

## 4. Required Public/Private Contract for `hidden_rk`

### 4.1 Public dataset in hidden mode

Allowed public fields:

```text
observations
actions
rewards
next_observations
dones
episode_id
timestep
```

Not allowed in public dataset:

```text
rho
kappa
K_eff
next_rho
next_kappa
next_K_eff
r_eff_true
states
next_states
r_base
C
theta
regime
next_regime
entry
reward_true
initially_unsafe
```

Private sidecar may still contain truth and simulator diagnostics for evaluation
and audit.

### 4.2 Method-facing config in hidden mode

The hard part is not only removing dataset columns. Several methods receive
`env_cfg`, and today `env_cfg` contains exact species/action-derived scale.

Implement one of these patterns:

1. **Preferred:** construct a sanitized method-view config for policies,
   filters, learned dynamics, and method-side reward models. Keep the true env
   config for simulator/evaluator/dataset generation.
2. **Acceptable if simpler:** keep one config object but add explicit helper
   functions so method paths consult a public method view and simulator paths
   consult private truth. Add tests that fail if method code calls private action
   tables in hidden mode.

The method view should not allow transition modelling code to infer exact table
effects from:

- `cfg.population`;
- `cfg.kind`;
- `cfg.K_base`;
- `cfg.K_max`;
- `cfg.r_min`;
- `cfg.r_max`;
- `resolve_actions(cfg)`;
- `realdata.pops_for(...)`;
- `realdata.effects_for(...)`;
- `action_effects_long.csv`;
- `species.csv`;
- `species_lambda.csv`.

Keep enough public fields for generic algorithm operation:

- number of actions;
- horizon;
- observation noise;
- reward mode;
- action costs;
- optional public reward/safety objective parameters for version 1;
- generic normalization scale, if needed, derived from public observations or a
  non-table constant rather than `K_base`.

## 5. Implementation Surfaces to Inspect/Edit

Do not guess from this list; inspect the live code before editing. These are the
known exposure surfaces from the audit.

### 5.1 Config and mode semantics

Likely files:

- `src/real_ecology_benchmark/config.py`
- `configs/motivation_native.yaml`
- new config, probably `configs/motivation_hidden_rk.yaml`

Tasks:

- Add an information/publicness field, e.g. `information_mode`.
- Default to current behavior.
- Add validation for allowed values.
- Make it impossible to accidentally run hidden-r/K rows under a public-table
  output root.

### 5.2 Dataset schema and collection

Likely files:

- `src/real_ecology_benchmark/dataset.py`
- `src/real_ecology_benchmark/collector.py`
- `src/real_ecology_benchmark/pipeline.py`

Tasks:

- Split "simulator controls enabled" from "public controls exposed".
- In current public-table mode, preserve existing `rho/kappa/K_eff` public
  fields.
- In hidden-r/K mode, generate/load/save a public dataset without control fields.
- Add a schema assertion/gate that hidden-r/K public datasets do not contain
  public controls.
- Keep private truth sidecars available for evaluator metrics only.

### 5.3 Beliefs and feature builders

Likely files:

- `src/real_ecology_benchmark/types.py`
- `src/real_ecology_benchmark/beliefs.py`
- `src/real_ecology_benchmark/dynamics.py`
- `src/real_ecology_benchmark/methods/ogsrl.py`
- `src/real_ecology_benchmark/methods/delphic.py`

Tasks:

- In hidden-r/K mode, `BeliefState.features(...)` must not append `rho`,
  `kappa/K_ref`, or `K_eff/K_ref`.
- Learned filter proposal must not require or use `rho/kappa/K_eff/next_*`.
- `BeliefCache` for hidden-r/K mode must not store public controls.
- `ContinuousDynamicsEnsemble._design(...)` must not add exact control columns
  in hidden mode.
- OGSRL guardian features and policy features must not include exact controls.
- Delphic should only see the sanitized belief feature vector.

### 5.4 General planning path

Likely files:

- `src/real_ecology_benchmark/planning.py`
- `src/real_ecology_benchmark/rollout.py`
- `src/real_ecology_benchmark/methods/refplan.py`
- `src/real_ecology_benchmark/methods/bamcts.py`
- `src/real_ecology_benchmark/methods/ogsrl.py`
- `src/real_ecology_benchmark/methods/mopo.py`

Tasks:

- In hidden-r/K mode, learned model rollouts should use action identity and
  learned dynamics only, not exact public-control advance.
- `ParticleMPC` must not call `advance_public_controls` for hidden-r/K method
  planning.
- `bamcts` tree keys must not bucket `rho/kappa` in hidden-r/K mode.
- RefPlan member posterior update must not pass exact `belief.rho/kappa`.
- Keep reward/cost use consistent with the public-objective decision.

### 5.5 Mechanistic and native ecological baselines

Likely files:

- `src/real_ecology_benchmark/beliefs.py`
- `src/real_ecology_benchmark/native_solver.py`
- `src/real_ecology_benchmark/discretize.py`
- `src/real_ecology_benchmark/methods/plus.py`
- `src/real_ecology_benchmark/methods/moor.py`
- `src/real_ecology_benchmark/methods/plus_native.py`
- `src/real_ecology_benchmark/methods/moor_native.py`

Tasks:

- In hidden-r/K mode, do **not** let ecological baselines construct transition
  models from exact `resolve_actions(cfg)` or species/action tables.
- `moor_native` should fit its transition model from public `(o,a,o')` data. If
  it remains Ricker-form, it must estimate the action effects/scale it uses
  rather than reading them.
- `plus_native` should not build a form bank with exact table dynamics. It can
  maintain a candidate mechanistic form bank, but candidate parameters/action
  effects must be fit or inferred from public data.
- If a faithful hidden native PLUS is too large for the first pass, implement a
  conservative explicit placeholder only if it fails loudly and is excluded from
  the manifest. Do not silently run the old exact-table native solver and call it
  hidden-r/K.

### 5.6 Action and reward public views

Likely files:

- `src/real_ecology_benchmark/actions.py`
- `src/real_ecology_benchmark/reward.py`
- `src/real_ecology_benchmark/controls.py`

Tasks:

- Separate simulator action specs from method-visible action specs.
- Method-visible action specs in hidden-r/K mode may expose cost and action ID,
  but must not expose exact `delta_r`, `delta_K`, or `stocking_delta`.
- Any reward model handed to a method should not carry full private action specs
  unless the method only receives cost-only public action views.
- Add tests so `build_reward(method_view_cfg)` cannot be used to recover hidden
  transition effects through `reward.actions`.

### 5.7 Manifest, runner, analysis

Likely files:

- `src/real_ecology_benchmark/manifest.py`
- `scripts/run_real_manifest_row.py`
- `scripts/make_motivation_native_manifest.py`
- new manifest helper, probably `scripts/make_hidden_rk_manifest.py`
- analysis scripts under `scripts/` and `real_ecology_runs/.../analysis`

Tasks:

- Add a manifest column for `information_mode` or make the output root/config
  unambiguous.
- Output paths must include `information_hidden_rk` or equivalent.
- Aggregation must never pool public-table rows with hidden-r/K rows.
- Dataset cache keys/hashes must include information mode.
- Analysis should compare:
  - current public-table setting vs hidden-r/K setting;
  - general vs native within hidden-r/K;
  - per-family and per-noise differences.

## 6. Test Plan Before Running Any Big Experiment

### 6.1 Unit/schema tests

Add tests for:

- hidden-r/K public dataset has no `rho/kappa/K_eff/next_*`;
- hidden-r/K private sidecar still has evaluator truth;
- hidden-r/K belief cache has no control arrays;
- hidden-r/K belief feature dimension is lower than public-table feature
  dimension by the expected control columns;
- hidden-r/K learned dynamics design has no control columns;
- hidden-r/K OGSRL guardian feature vector has no exact control columns;
- hidden-r/K method-visible action specs expose costs but not exact
  `delta_r/delta_K/stocking_delta`;
- public-table mode remains unchanged.

### 6.2 No-op proof for old setting

Before trusting the new setting, prove the old setting was not accidentally
changed.

Suggested smoke rows:

- Amur tiger / ricker / sigma 0 / safe / `refplan`;
- Amur tiger / ricker / sigma 0 / safe / `moor_native`;
- Iberian lynx / allee / sigma 0.2 / safe / `plus_native`;
- one yield-mode row.

For existing public-table mode, compare key outputs against prior frozen
artifacts where available:

- method/filter route;
- dataset hash;
- fallback count;
- action histogram non-degenerate where expected;
- summary metric tolerances.

### 6.3 Hidden-r/K smoke tests

First hidden-r/K smoke cell:

```text
population = Amur tiger
family = ricker
sigma = 0.0
reward_mode = safe
methods = refplan, bamcts, ogsrl, moor_native, plus_native
```

Required checks:

- dataset public file has no public controls;
- methods do not crash with missing controls;
- `fallback_count == 0` for methods where fallback means policy exception;
- native methods do not call exact `NativeSolver.build` with exact table
  `resolve_actions`;
- summaries include `information_mode=hidden_rk`;
- output path contains hidden-r/K namespace;
- evaluator metrics still compute from private truth.

Then run a non-Ricker smoke:

```text
population = Iberian lynx
family = allee
sigma = 0.2
reward_mode = safe
```

This catches structural-family leakage and native form-bank shortcuts.

## 7. Experiment Shape After Gates Pass

Do not launch the full experiment until schema/no-op/smoke gates pass.

Recommended first substantive run:

```text
setting = hidden_rk
methods = refplan, bamcts, ogsrl, moor_native, plus_native
populations = 7 recoverable populations
families = ricker, allee, theta, regime
sigma = 0.0, 0.4
reward_mode = safe
seeds = same protocol as motivation-native
```

Reason: this 7 x 4 x 2 x 1 grid is small enough to debug and still tests the
two endpoints of observation noise on recoverable cells, where the motivation
claim is most interpretable.

If the pilot is clean and informative, run the full mirror of the motivation
experiment:

```text
populations = all 9
families = ricker, allee, theta, regime
sigma = 0.0, 0.1, 0.2, 0.4
reward_mode = safe, yield
methods = refplan, bamcts, ogsrl, moor_native, plus_native
eval = 5 seeds x 4 episodes x horizon 50
```

Use a separate root, e.g.

```text
real_ecology_runs/motivation_hidden_rk_<YYYYMMDD>/
```

## 8. Expected Interpretations

Pre-register the interpretation before seeing results.

### If general methods close most of the gap

This supports the informational-asymmetry diagnosis:

> current natives win because public tables give them a near-exact mechanistic
> planning model.

Then the paper can honestly say the original benchmark setting was too generous
to mechanistic ecological baselines, and the hidden-r/K setting better tests the
intended structural-uncertainty motivation.

### If natives still dominate

Then the diagnosis shifts:

> the advantage is not only public r/K tables; solver structure, discretized
> belief planning, objective geometry, or general-method model class may be the
> bottleneck.

In that case, simply hiding public ecological tables will not rescue the
original motivation.

### If everything degrades or becomes unstable

Then the hidden setting may be too hard or underidentified from 4000
transitions. Report this directly; do not tune the setting until it gives a
desired ordering.

## 9. Acceptance Criteria

Before launching any nontrivial hidden-r/K sweep, the next agent must produce a
short status doc proving:

- [ ] public-table mode is unchanged by default;
- [ ] hidden-r/K public datasets contain no control fields;
- [ ] hidden-r/K belief features/cache contain no exact r/K control fields;
- [ ] general methods no longer receive exact `rho/kappa/K_eff` in fitting or
      planning;
- [ ] `reward.actions` or any public method action view cannot reveal exact
      `delta_r/delta_K/stocking_delta`;
- [ ] native methods in hidden-r/K mode do not use exact-table
      `NativeSolver.build` / `resolve_actions` for transition construction;
- [ ] output paths and manifest rows include information mode;
- [ ] aggregation cannot pool public-table and hidden-r/K rows;
- [ ] `make test` and `make docs-check` pass;
- [ ] at least one Ricker and one non-Ricker hidden-r/K smoke cell complete;
- [ ] no full experiment is submitted until all gates pass.

## 10. Things Not to Do

- Do not overwrite the completed `motivation_native_20260711` outputs.
- Do not relabel the completed negative result as hidden-r/K.
- Do not remove `rho/kappa/K_eff` from the current public-table setting.
- Do not run old exact-table `plus_native`/`moor_native` under a hidden-r/K label.
- Do not change reward/objective publicness in the first implementation without a
  separate design note.
- Do not compare raw safe-mode and yield-mode returns as if they were the same
  objective.
- Do not pool recoverable populations with demographic sinks without a split.

## 11. First Implementation Order

1. Add the information-mode config/default and no-op tests for public-table mode.
2. Split simulator controls from public/method-exposed controls.
3. Make hidden-r/K datasets and belief caches drop public controls.
4. Sanitize method-visible action specs/config so transition effects cannot be
   read through reward/action helpers.
5. Update learned filters, learned dynamics, planning, OGSRL, and Delphic feature
   paths to respect hidden-r/K.
6. Rework native ecological baselines for hidden-r/K so they fit transition
   structure from public data instead of exact tables.
7. Add manifest/output namespacing and aggregation guards.
8. Run unit tests and public-table no-op proof.
9. Run the two smoke cells.
10. Write a launch-readiness doc before any pilot/full sweep.

## 12. The First Smoke Command Should Prove

The first successful hidden-r/K row should demonstrate all of the following in
its logs/summary:

```text
information_mode = hidden_rk
dataset_public_controls_present = false
belief_cache_controls_present = false
method_route = expected method/filter
native_exact_table_solver_used = false
dataset_hash emitted
fallback_count = 0  # except for explicitly designed OGSRL guardian fallback
summary metrics emitted
```

If any of those are missing, stop and patch the implementation before expanding
the run.

