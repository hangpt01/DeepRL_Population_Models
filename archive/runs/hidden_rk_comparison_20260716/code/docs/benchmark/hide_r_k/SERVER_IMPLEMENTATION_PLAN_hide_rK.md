# Server Implementation Plan - Hide r/K From Method-Facing Inputs

**Date:** 2026-07-16  
**Status:** Plan only; no implementation has been performed.  
**Primary contract:** `SERVER_FIX_BRIEF_hide_rK.md`  
**Audit resolution:** Incorporates the converged Round 2/3 decisions in
`SERVER_PLAN_REVIEW_hide_rK.md` and the PI choices recorded after that audit.  
**Purpose:** Provide a code-grounded plan that can be audited before implementation.

If this plan conflicts with `SERVER_FIX_BRIEF_hide_rK.md`, the fix brief wins.
`HANDOFF_rK_leak_error_and_discussion.md` is background only.

## 1. Root-Cause Confirmation

### 1.1 Verdict

The r/K exposure is primarily a **design/specification error faithfully routed
into code**, not an accidental read of evaluator-only arrays. The original
design conflated known population identity with known demographics. Later native
solver work amplified the consequence by turning the public tables into exact
mechanistic transition/reward tables.

The current private sidecar boundary is not the main failure. The public schema,
full environment config, shared features, action objects, reward objects, native
solver construction, and metadata all intentionally expose or derive quantities
that the corrected experiment needs to keep private.

### 1.2 Originating specification

`docs/history/29_6_Real_Ecology_Setting_Implementation_Plan.tex` states:

- E1, lines 231-238: expose population identity and `(r_eff, K_eff)` as known
  observed inputs.
- E8, lines 395-400: treat `r_base` and the lambda profile as known context and
  have methods read the already-known `(r_eff, K_eff)`.

`docs/real_ecology_history/29_6_algorithm_method_notes.tex`, lines 262-278,
then specifies `K_ref`-normalized state targets and public-control features
containing `rho`, `kappa`, and `K_eff`.

### 1.3 Git provenance

These commits establish the first tracked appearance or subsequent routing of
the relevant design/code in the current history. In particular,
`f62054f0b87ce456cfb4c2770e32ee26d4ef26fd` added the historical plan during a
duplicate-tree removal/refactor; it does not establish where the scientific
decision originated.

| Surface | Commit | Author | Date | Meaning |
|---|---|---|---|---|
| E1/E8 design text and original public schema | `f62054f0b87ce456cfb4c2770e32ee26d4ef26fd` | Hang Phung | 2026-07-09 | First tracked appearance of this design text in the current history; not proof of decision origin. |
| Real ecology data-mode/config plumbing | `320fa633a1bf377a70c8c85f5f48aabf4fe01e6b` | Hang Phung | 2026-07-09 | Population tables became environment config and method-visible controls. |
| Public-control readability/routing | `b873314e87c15ed42b30639c720487d7afac1d68` | Hang Phung | 2026-07-09 | The code explicitly describes rho/r_eff as non-leaky under known identity. |
| Native solvers | `5f9cf32a69d47e2d31b2ed6ee30ebbcfe84b8536` | Hang Phung | 2026-07-15 | Exact public demographic/action structure was converted into native solver tables. |

### 1.4 Confirmed exposure map

| Quantity | Current method-facing route | Source classification |
|---|---|---|
| `K_base`, `K_min`, `K_max`, `K_ref`, `r_min`, `r_max` | `config.real_environment()` reads `species.csv`; `BasePolicy` receives the complete `EnvironmentConfig`. | Spec design plus config routing |
| `safety_threshold` / `s_safe`, `C_low`, `C_high`, `regime_threshold_low`, `regime_threshold_high` | `real_environment()` computes/scales them from `K_base`; beliefs, planners, OGSRL, native grids, regime dynamics, reward and training metrics use them. | K-derived shared feature/objective aliases |
| `rho`, `kappa`, `K_eff`, and next aliases | `envs._public_info()` -> `collector.collect_dataset()` -> `TrajectoryDataset` -> filters, caches, dynamics and planners. | Public schema and shared feature builders |
| `r_setpoint`, lambda-derived rates, `dK_step`, `dN` | `realdata` -> `actions.real_action_table()` -> `resolve_actions()` -> controls, proposals, rewards and native tables. | Exact action-table routing |
| Reward decomposition | `build_reward()` gives methods exact action objects, `K_ref`, benefit formula and collapse/safety penalty. | Shared reward-model routing |
| `true_family` / `cfg.kind` | Full environment config reaches policies and filters; learned proposals branch on regime; `true_family` constructs the exact proposal. | Config and filter routing |
| Population identity | `cfg.population` selects species/action tables. Current runs are one population per cell, so identity is also implicit in the run. | Config/manifest metadata |
| Metadata aliases | Public dataset metadata contains `env.config_dict()` and `action_table_hash`; learned models serialize full environment JSON. | Loader/cache metadata |
| Manifest and filenames | Manifest `environment`, YAML `kind`, and `<population>/<family>/sigma_*` paths encode private cell labels. No current policy parses the filename, but caches and artifacts inherit it. | Manifest/filename side channel |
| Native solver tables | `NativeSolver.build()`, `assumption_config()`, `build_native_grid()` and `_build_tables()` use table-derived actions, K grids, reward and safety. | Exact native solver construction |

### 1.5 Why existing checks did not catch it

Existing tests check that true state and structural truth arrays do not enter the
public `.npz`, and that online policy feedback has no realized reward field.
They do not treat public controls, K-derived normalization, full config metadata,
action effects, family identity, or exact native tables as forbidden. Some tests
explicitly require the public controls to be present.

**Prevention rule:** Identity is not demographics. Every method-visible value
must be tested against the uncertainty the experiment claims to preserve.

## 2. Regime Split Design

### 2.1 Canonical flag

Add one canonical setting:

```text
expose_rk: hidden | full
```

- `hidden`: corrected hidden-demographics / structure-unknown regime and the
  default for real ecology.
- `full`: current public-demographics / parameter-known behavior, preserved as
  a reproducibility arm.
- `noisy`: not implemented in this repair.

Existing motivation/public-table configs will explicitly set `expose_rk: full`
so their behavior is not silently changed. Synthetic and dummy regression paths
retain their current semantics.

### 2.2 Simulator config versus method view

Keep `EnvironmentConfig` as the private simulator/evaluator configuration. Add a
separate sanitized `MethodContext` constructed by the pipeline.

In hidden mode, `MethodContext` may contain:

- number of actions and public action IDs;
- the mandatory public action-cost vector read from the sanitized public dataset;
- observation-noise model and horizon/planning budgets;
- an observation-derived numerical scale;
- an opaque population category token with no method-visible lookup to species,
  family, demographic, safety or action-effect tables;
- fitted public dynamics, reward, risk and ecological-model objects.

It must not contain population table keys, table paths, family truth, r/K values
or aliases, exact action effects, `K_ref`, K-derived thresholds (including
`C_low`, `C_high` and `regime_threshold_*`), reward decomposition, or private
sidecar data.

### 2.3 Settled scientific decisions for the first repair

These are binding choices, not remaining implementation discretion:

1. **Known population / unknown demographics.** Hidden datasets include a stable,
   opaque categorical `pop_id`. The private mapping from that token to the
   population name and its table rows remains simulator/evaluator-only. No loader,
   context, cache or model may use the token to call `species.csv`, action-effect
   tables, lambda profiles, family labels or any derived lookup. The schema and
   shared fit API support multiple tokens in one pooled training dataset. The
   controlled first repair retains the existing per-cell run shape and a target
   of 4000 transitions per cell; actual cross-cell pooled headline training is
   allowed by the interface but is not silently substituted for that registered
   run shape.
2. **Private safety objective.** The existing `c_safe * K_base` threshold and all
   quantities or labels derived from it stay private/evaluator-only. Hidden methods
   do not receive `s_safe`, `c_safe`, normalized safety distance, unsafe/safe
   occupancy or crossing labels, `initially_unsafe`, `entry`, or an independently
   reconstructed threshold. A new absolute public threshold is outside this first
   repair.
3. **One shared public reward/risk surrogate.** The pipeline fits one immutable
   surrogate artifact per sanitized public training unit (one cell today, or one
   explicitly pooled unit later), caches it by public-data hash/model version/seed,
   and supplies that exact artifact to every method. Methods may not refit it with
   method-specific labels, private data or evaluator feedback. Section 5 fixes its
   feature map, targets, split, loss, censoring and fallback behavior.
4. **Episode-preserving transition budget.** Treat 4000 rows as the collection
   target, not a hard truncation count. Complete the final episode, require
   `4000 <= actual_rows <= 4024` for the registered `episode_length=25`, and never
   increase or tune the target from Phase 0 outcomes. The canonical reporting
   language is **"4000 target transitions per cell, preserving complete
   episodes."** Record `target_rows`, `actual_rows`, `overshoot_rows`, and
   `episode_count` in manifests and summaries; require
   `overshoot_rows == actual_rows - target_rows` and, generally,
   `0 <= overshoot_rows < episode_length`.
5. **Safe-mode interpretation.** If the permitted public safety channels are weak,
   interpret safe-mode results as information-limited under the private safety
   objective, not as evidence that OGSRL intrinsically fails at safety. Keep all
   preregistered safe-mode rows. Do not add a post-hoc public threshold, tune any
   method/surrogate after seeing outcomes, or remove/demote safe-mode results.

### 2.4 Regime identity and isolation

Record `expose_rk` and a human-readable regime label in:

- YAML configs and resolved config records;
- every manifest row;
- public/private dataset schema versions and hashes;
- belief/model/native-fit cache keys;
- output directory roots;
- episode rows and summaries;
- aggregate grouping keys.

Every manifest row and summary must also record `target_rows`, `actual_rows`,
`overshoot_rows`, and `episode_count`. Aggregation must retain these fields and
must not silently normalize away or discard episode-preserving overshoot.

Aggregation must group by regime and fail on unlabeled or ambiguously mixed
results. Hidden and full caches must never share a cache key.

## 3. File and Function Change Map

Every package path in this map is anchored to the live package root
`src/real_ecology_benchmark/`; no `ecobench` or removed duplicate-tree path is an
implementation target.

| File(s) / functions | Planned change | Surface |
|---|---|---|
| `src/real_ecology_benchmark/config.py`: `EnvironmentConfig`, `validate`, environment factories, `load_config`, carryover helpers | Add `expose_rk`; default real ecology to hidden; add sanitized `MethodContext` and constructor. | Config / method inputs |
| `src/real_ecology_benchmark/cli.py`, `src/real_ecology_benchmark/__init__.py` | Add CLI/config support and export the new regime/context APIs. | CLI / public API |
| `src/real_ecology_benchmark/controls.py`: `control_fields_enabled`, initial/public/advance helpers | Split simulator-control availability from method/public-control exposure. Preserve old full behavior. | Control boundary |
| `src/real_ecology_benchmark/envs.py`: `_truth`, `_public_info`, `reset`, `step` | Keep exact controls and tables in simulator truth; omit them from hidden public feedback; expose genuine termination separately from horizon truncation. | Simulator / online interface |
| `src/real_ecology_benchmark/actions.py`: `ActionSpec`, `resolve_actions` | Keep rich action specs simulator-only; add a cost-only public action representation for methods. | Action boundary |
| `src/real_ecology_benchmark/reward.py`: `ContinuousReward`, `build_reward`; new shared surrogate API | Keep exact reward evaluator-only in hidden mode; fit the Section 5 surrogate once from public reward/event labels. | Reward boundary |
| `src/real_ecology_benchmark/dataset.py`: schemas, dataclasses, save/load, `assert_public_schema`, hash | Add regime-aware schemas; require per-transition costs and a validated public action-cost vector; add opaque `pop_id`, `terminated`, and `truncated`; remove hidden controls and sanitize metadata; store private controls plus evaluator-only `safety_penalty_applied` in the sidecar; recursively audit metadata. | Loader / schema |
| `src/real_ecology_benchmark/collector.py`: `collect_dataset`, metadata construction | Route control/safety truth to the private sidecar in hidden mode; emit only allowed public arrays and metadata; never collapse termination and truncation into the risk target; preserve complete episodes after the 4000-row target and record target/actual/overshoot/episode counts. | Collection |
| `src/real_ecology_benchmark/types.py`: `BeliefState`, `features`, `PublicTransition` | Use a public feature specification and data-derived scale; controls remain available only in full mode. | Shared method inputs |
| `src/real_ecology_benchmark/beliefs.py`: proposals, filters, `BeliefCache`, `cache_dataset_beliefs` | Remove hidden controls/K/safety/family from features and caches; make learned filtering family-independent; prohibit `true_family` in hidden; support data-fitted mechanistic/native filters. | Beliefs / shared features |
| `src/real_ecology_benchmark/dynamics.py`: `_design`, ensemble fit/save/load | Hidden design uses observations/beliefs and action IDs only; do not serialize full environment config. | Learned dynamics |
| `src/real_ecology_benchmark/planning.py`: `ParticleMPC` | Hidden planning uses fitted dynamics/reward/risk and no exact control advancement, K normalization or safety threshold. | Planning |
| `src/real_ecology_benchmark/rollout.py`: `AugmentedRolloutState` | Carry rho/kappa only in full mode; hidden rollout state contains public/history-derived state only. | Rollouts |
| New `src/real_ecology_benchmark/native_fit.py`: `fit_from_public_data`, fitted-model dataclasses | Fit candidate dynamics/action effects and nuisance parameters using sanitized data only. | Native estimation |
| `src/real_ecology_benchmark/native_solver.py`: `NativeSolver.build`, new `build_from_fitted` | Preserve exact-table `build` for full; hidden solvers accept fitted model/reward/grid objects and never call table loaders. | Native solver |
| `src/real_ecology_benchmark/discretize.py`: grid builders and kappa transitions | Preserve exact full grid; build hidden grids/control states from public observations and fitted model effects. | Native discretization |
| `src/real_ecology_benchmark/methods/base.py`: `BasePolicy.__init__` | Give policies `MethodContext`, not the private simulator config, in hidden mode; raise on hidden construction of an unconverted method. | All methods |
| `src/real_ecology_benchmark/methods/refplan.py` | Remove hidden controls from dynamics, planning and posterior updates; use fitted reward. | RefPlan |
| `src/real_ecology_benchmark/methods/bamcts.py` | Remove rho/kappa and private safety from keys/rollouts; use sanitized state/history and learned reward/risk. | BA-MCTS |
| `src/real_ecology_benchmark/methods/ogsrl.py` | Remove K/control/unsafe features; retain a public-support OOD guardian; consume the shared public termination-hazard estimate instead of private safety labels. | OGSRL |
| `src/real_ecology_benchmark/methods/mopo.py` | Use sanitized dynamics and reward interfaces while retaining ensemble pessimism. | MOPO |
| `src/real_ecology_benchmark/methods/delphic.py` | Use sanitized shared beliefs and reward labels only. | Delphic |
| `src/real_ecology_benchmark/methods/moor.py`, `src/real_ecology_benchmark/methods/plus.py`, `src/real_ecology_benchmark/methods/value.py` | In hidden mode construct mechanistic proposals/value models from data-fitted objects, not exact actions/K/reward/safety. | Adapted ecology methods |
| `src/real_ecology_benchmark/methods/moor_native.py` | Add single-Ricker public-data fit and build its filter/solver from the fitted model. | MOOR native |
| `src/real_ecology_benchmark/methods/plus_native.py` | Fit one parameterization per existing candidate form from public data; keep posterior only over those forms. | PLUS native |
| `src/real_ecology_benchmark/pipeline.py`: dataset validation, filter factory, method construction/run, cache/output paths | Construct private simulator and sanitized method views; fit the shared surrogate and native models before filters/policies; neutralize exact native filters in hidden; namespace all artifacts by regime. | Routing |
| `src/real_ecology_benchmark/evaluator.py`, `src/real_ecology_benchmark/gate.py` | Keep true simulator metrics private/evaluator-side; sanitize policy feedback; label evaluator results by regime. | Evaluator / gates |
| `src/real_ecology_benchmark/training_monitor.py` | Remove K/family/population-table metadata from method training artifacts; use public scaling in diagnostics. | Training artifacts |
| `src/real_ecology_benchmark/manifest.py`: manifest builders and aggregation | Add regime columns/builders, non-pooling aggregate keys, and target/actual/overshoot/episode-count fields. | Manifests / analysis |
| `scripts/run_real_manifest_row.py` | Validate manifest/config regime agreement, construct regime-separated paths, and persist target/actual/overshoot/episode counts. | Runner |
| `scripts/make_motivation_native_manifest.py`, `scripts/run_motivation_acceptance.py` | Explicitly label and verify the existing run as `full`. | Full-regime regression |
| New `scripts/make_hidden_rk_manifest.py`, `scripts/run_hidden_rk_acceptance.py` | Generate the corrected manifest and execute schema/table/family/smoke gates. | Hidden-regime runner |
| Real-row Slurm wrappers | Select the hidden config only for explicitly hidden manifests; retain explicit full config support. | Cluster launch |
| Existing `motivation_native*.yaml`; new hidden full/smoke YAML | Mark existing configs `full`; provide hidden default and smoke settings. | Configs |
| Existing real/native/schema/manifest tests; new `tests/real/test_hidden_rk.py` | Make old expectations regime-explicit and add the acceptance suite below. | Tests |

## 4. Per-Method Plan

| Method | Current / `full` behavior | Required `hidden` behavior |
|---|---|---|
| `refplan` | Control-aware ensemble and MPC use rho/kappa/K/K_ref and exact reward/safety. | Fit action-conditional dynamics from sanitized beliefs/actions; use a learned reward; remove controls from planning and posterior update. |
| `bamcts` | Dynamics and tree keys condition on rho/kappa; state buckets and rewards use K/safety. | Keys and rollouts use sanitized belief/history only. Private r/K changes must not alter a key or model input. |
| `ogsrl` | Dynamics, actor, guardian, rollouts and feasibility use K-normalized controls and unsafe thresholds. | OOD guardian uses public belief/action support. Actor uses sanitized features. Safety risk is the shared one-step genuine-termination hazard; combined `done`, truncation and K-derived labels are never risk targets. |
| `mopo` | Control-aware ensemble and exact-objective MPC. | Sanitized action-conditioned ensemble and learned reward, retaining MOPO uncertainty pessimism. |
| `delphic` | Inherits rho/kappa/K/safety through shared belief features. | Compatible worlds and Q fitting consume sanitized cache features and reward labels only. |
| Adapted `moor` | Fits mostly K while exact action growth/effects remain known. | Fit one Ricker model, including required action effects, from public data; use fitted reward/risk. |
| Adapted `plus` | Builds an exact-table mechanistic K bank. | Build a data-derived bootstrap/confidence bank without table initialization; report separately from the naive headline. |
| `moor_native` | Fits `K_hat` over exact action effects, exact grid and exact reward/safety. | Headline naive baseline: fit a single Ricker model and effects from public data, then solve it. |
| `plus_native` | Builds exact-table solvers for four forms and updates mainly the form posterior. | Separate closed-world baseline: fit parameters independently for the same four forms, then retain the form posterior. |

The `full` branch must continue through the current implementations and retain
current values, features, native tables, and outputs except for explicit regime
labels/namespacing.

## 5. Shared Public Reward/Risk Surrogate

### 5.1 Schema contract and permitted information

For hidden mode, the public transition schema is:

```text
(pop_id, o_t, a_t, cost_t, R_t, o_next, terminated, truncated,
 done, episode_id, timestep)
```

- `pop_id` is the opaque token from Section 2.3. It is mandatory for this first
  setting, even when constant in a per-cell dataset; the loader accepts multiple
  tokens for an explicitly pooled unit.
- `cost_t` is mandatory and equals a public `action_costs[a_t]` vector stored with
  the dataset. The loader verifies row/vector agreement and complete coverage of
  all public action IDs. No action-effect field may share that object.
- `R_t` is the simulator-emitted public reward label. It is a fitting target only,
  never a policy/belief/dynamics feature and never decomposed or inverted.
- `terminated` means genuine environment termination. For the current simulator
  that is the public episode event corresponding to extinction (`state_next == 0`),
  which this settled design permits as an event label without exposing state.
- `truncated` means the environment horizon or collector episode limit ended the
  trajectory without termination. Require `done == terminated | truncated` and
  reject rows where both event flags are true.
- Public history derived from these fields may be used. Private sidecars, table
  metadata, true state, `entry`, `reward_true`, `initially_unsafe`, safety
  occupancy/crossing labels and every r/K/family alias are forbidden.

### 5.2 Frozen feature map

Fit one feature specification from the training fold only. Let `q` be the median
strictly positive training observation, falling back to `1.0` if none exists, and
define `z(o) = log1p(max(o, 0) / q)`. For each transition use:

```text
z_prev       = z(previous observation in the same episode), or z(o_t) at t=0
z_t          = z(o_t)
delta_prev   = z_t - z_prev
z_next       = z(o_next)
delta_next   = z_next - z_t
tau          = timestep / max(public_horizon - 1, 1)
action_1hot  = one-hot(a_t)
cost_t       = public action cost
pop_1hot     = one-hot(opaque pop_id)
```

The common design row is
`[1, z_prev, z_t, delta_prev, z_next, delta_next, tau, action_1hot,
cost_t, pop_1hot]`. Continuous columns are standardized using training-fold
statistics only; the intercept and one-hot columns are not standardized. The
public vocabulary contains only action IDs and opaque tokens observed in the
sanitized training unit. Unknown tokens fail loudly rather than triggering a
private lookup.

During planning, `z_next` is formed from the public dynamics model's predicted or
sampled next observation. The caller carries `(z_prev, z_t, timestep, pop_id)` in
the sanitized rollout state. No method may replace these with true-state or
K-normalized features.

### 5.3 Shared fitting split and reward estimator

- Split by whole episode, deterministically from the dataset seed: 80% fit and
  20% holdout, stratified by opaque `pop_id` where a pooled unit has enough
  episodes. Never split transitions from one episode across folds.
- Fit the direct public reward label with L2 ridge regression on the Section 5.2
  design, squared-error loss, and fixed `ridge = 1e-3`; do not penalize the
  intercept. Do not tune against
  evaluator returns, private safety metrics or method performance.
- Report fit/holdout RMSE and MAE by opaque token and reward mode. The reward mode
  may label reports/caches but is not a feature exposing its private decomposition.
- As non-tuning diagnostics, also report holdout RMSE/MAE separately for the
  evaluator's private `safety_penalty_applied` strata and for the public low-reward
  tail. Define the tail before fitting as `R_t` at or below the fit-fold 10th
  percentile, then apply that fixed cutoff to holdout rows. Private strata are
  evaluator/report-only: they may not alter features, coefficients,
  hyperparameters, early stopping, model selection or method configuration.
- At planning time, the immutable model predicts the mean public reward for a
  candidate public transition. Methods must not add the exact benefit, private
  collapse penalty or a second table-derived cost term. Risk remains a separate
  constraint/diagnostic rather than being silently subtracted twice.

If there are fewer than two episodes, fit on all episodes, mark holdout metrics
unavailable and fail the benchmark acceptance configuration; this fallback is
only for unit/smoke fixtures.

### 5.4 Shared risk estimator and censoring

The risk target is the one-step genuine-termination indicator, not combined
`done` and not any K-derived unsafe label:

```text
y_term[t] = 1 if terminated[t] else 0
```

Fit L2-regularized logistic regression on the same training rows and frozen
feature map, with `l2 = 1e-3`, an unpenalized intercept, zero initialization,
maximum 200 Newton/IRLS iterations and convergence tolerance `1e-8`. Do not
class-reweight or tune a decision threshold against private evaluator outcomes.
Clip returned numerical probabilities to `[1e-6, 1 - 1e-6]` only when computing
log loss.

A truncated final transition is a valid observed non-termination for that one
completed step (`y_term = 0`), but contributes no inferred labels after the
truncation boundary. No future-window failure label is synthesized. If the fit
fold has only one target class, use the pre-registered constant Beta(1,1)
posterior mean `(n_terminated + 1) / (n_rows + 2)` and record
`risk_fallback=constant_single_class`; do not consult private events to repair the
class balance. Report holdout Brier score, log loss and termination prevalence;
there is no post-hoc calibration in this repair.

For a multi-step rollout, combine conditional one-step hazards as
`1 - product(1 - risk_hat_t)`. This estimates public extinction risk, not the
private `c_safe * K_base` safety objective. That distinction must be explicit in
method diagnostics and reporting.

**Pre-registered expectation:** genuine termination is known to be nearly absent
under the frozen collector. In the audited complete 288-cell sweep it occurred in
24 of 1,152,040 transitions (0.00208%); 280/288 cells had no event, and all events
were in `crab_eating_fox`/theta cells. Therefore
`risk_fallback=constant_single_class` is an expected experimental property in
most per-cell fits, not an implementation failure. The separate 721-file
prevalence calculation supports the same arithmetic but is **not an independent
replication**: it combines overlapping penalty sweeps whose matching ecological
trajectories are repeated. Report the complete-sweep result without claiming
independent replication.

### 5.5 OGSRL and shared-use contract

- Retain an OOD/support guardian fitted only from the sanitized Section 5.2
  feature space; remove its `K_ref`, safety-threshold, unsafe-bit and normalized
  safety-distance inputs.
- Replace OGSRL's private unsafe probability with the shared
  `risk_hat(public_history, action, predicted_next_observation, pop_id, timestep)`.
  Rollout and deployment constraints consume this probability and never
  `entry`, `reward_true`, `initially_unsafe`, `K_ref`, exact safety labels or exact
  reward decomposition.
- Other methods receive the identical reward/risk artifact. They may use the
  reward prediction according to their existing objective and may use the risk
  prediction only where their documented algorithm already has a risk/safety
  interface. No method-specific refit or private calibration is allowed.
- Keep OGSRL in the five-method headline comparison. For every hidden OGSRL row,
  disclose termination count/prevalence, `risk_fallback`, fitted risk prevalence,
  overall and low-reward-tail reward holdout error, and evaluator-only
  safe/unsafe-stratified reward error. If public signals are weak, use the
  information-limited interpretation in Section 2.3; do not characterize the
  result as intrinsic inability of OGSRL to handle safety.
- Fit once in `pipeline.py` before policy fitting. Serialize only the public
  feature vocabulary/scales, coefficients, fixed hyperparameters, public-data
  hash, split seed and public diagnostics. The artifact must contain no
  `EnvironmentConfig`, population-table mapping, family label, path or sidecar
  reference.

## 6. Native-Solver Plan

### 6.1 Public-data fitting entry point

Add `native_fit.fit_from_public_data()` (or an equivalently named API in that
module). It may accept only:

- sanitized `TrajectoryDataset` arrays;
- sanitized belief/observation features;
- public action IDs and costs;
- observation noise and numerical fitting budgets;
- an internally selected candidate-form label.

It may not accept `EnvironmentConfig`, population names, `data_dir`, private
sidecars, table-derived action specs, safety thresholds, `K_ref`, true family,
or any object produced by `realdata`/`resolve_actions`.

### 6.2 Fitted model

The fitted result should contain only estimates derived from public data:

- per-action growth effects;
- baseline/effective capacity parameters and fitted capacity-action effects;
- fitted direct-state action effects where identifiable;
- observation-derived state/grid scale;
- candidate-specific nuisance parameters needed by Allee, theta or regime forms;
- a fitted reward model or tabulated public-data reward estimate.

The hidden native grid and solver are built from this fitted result. The full
native grid and exact-table solver remain available only under `expose_rk=full`.

### 6.3 MOOR native

`moor_native` always requests the Ricker form, regardless of simulator truth.
Its public-data fit produces one fitted solver/grid. That exact object must be
used both for native likelihood/filter updates and policy action values; no
second fit or table-built filter may survive. The solver and discrete filter must
therefore be created after the public-data fit.

### 6.4 PLUS native scope

Keep the existing closed-world form set exactly:

```text
Ricker, Allee, theta-logistic, regime-switching
```

Fit one continuous parameterization for each form from the same sanitized data.
The posterior axis remains model form only. Do not add explicit r/K candidate
axes, distractor forms, truth-conditioned candidates, or open-world PLUS in this
repair.

Each candidate owns one fitted solver/grid and one belief bank. Its likelihood
updates and action values must use that same candidate-specific fitted object.
Do not route PLUS through one shared MOOR-style solver. In hidden mode the
pipeline's current exact-table `native_discrete` filter is removed, neutralized
to routing-only state, or replaced by these fitted candidate banks before policy
construction.

## 7. Test and Acceptance Plan

### 7.1 No-r/K-access test

- Recursively inspect hidden public arrays, metadata, method contexts, beliefs,
  caches, learned models and serialized artifacts for every forbidden name/alias.
- Assert that policies do not retain `EnvironmentConfig`, table paths, rich
  `ActionSpec` objects or K-derived reward/safety objects.
- Include `C_low`, `C_high`, `regime_threshold_low`,
  `regime_threshold_high`, and wildcard `regime_threshold_*` in the recursive
  forbidden-name/alias set.
- Construct two private simulator configs with different r/K/action effects but
  identical public histories; policy inputs, keys and chosen actions must match.
- Assert that constructing any not-yet-converted method under `hidden` raises a
  clear error before policy fit; partial rollout must never fall back to full
  `EnvironmentConfig` access.

### 7.2 Table-independence test

- Generate and freeze a hidden public dataset first.
- During method fit/planning, monkeypatch `species.csv`,
  `action_effects_long.csv`, `species_lambda.csv`, `realdata.pops_for`,
  `realdata.effects_for`, and `resolve_actions` to raise or return shuffled data.
- With fixed seeds, every hidden method's beliefs, fitted parameters, learned
  models and planning artifacts must be byte-identical.
- A paired `full` test confirms that the exact-table branch remains live and
  table-dependent.

### 7.3 Family-relabel test

- Hold the public dataset fixed.
- Relabel only private `true_family` metadata and all aliases, including manifest
  and evaluator grouping labels.
- Hidden training, fit, caches, solver tables and planning actions must remain
  byte-identical. Only evaluator/report grouping may change.
- Confirm `moor_native` remains Ricker and `plus_native` keeps the same four
  candidates independent of the relabel.

### 7.4 Regime-split-intact and full-equivalence test

- `full` retains public controls, current features and exact native table routing.
- `hidden` omits all forbidden fields and uses fitted models.
- Both regimes complete at least one Ricker and one non-Ricker smoke row.
- Gate the regime split before hidden logic lands: under fixed seeds, `full`
  preserves pre-change method-facing numerical inputs, RNG streams/states, solver
  parameters, feature values/shapes, chosen actions and metrics where
  reproducible.
- Byte identity applies to canonical payload arrays and explicitly unchanged
  artifacts only. Regime labels, output paths, cache namespaces and summaries are
  excluded because the repair requires them to change. Compare the legacy public
  array projection when the new schema adds `cost`, `pop_id`, `terminated` and
  `truncated`.

### 7.5 Reward/risk and schema tests

- Require `cost_t`, complete `action_costs`, opaque `pop_id`, `terminated` and
  `truncated`; verify cost row/vector consistency and `done == terminated |
  truncated` with mutually exclusive event flags.
- Prove population tokens have no reachable method-side mapping to species,
  family, demographics, safety or action-effect tables; exercise both one-token
  and explicitly pooled multi-token fixtures.
- Change only private safety thresholds/labels and confirm the frozen surrogate,
  method inputs and actions are unchanged for a fixed public dataset.
- Unit-test horizon truncation, genuine termination, one-class risk fallback,
  episode-level splitting, training-only scaling, deterministic coefficients,
  multi-step hazard composition and OGSRL's absence of private guardian fields.
- Verify overall reward holdout metrics, fit-fold-fixed public bottom-decile
  metrics, and evaluator-only `safety_penalty_applied` stratification. Assert that
  changing or withholding private diagnostic labels cannot change a fitted model,
  selected hyperparameter, method input or action.
- Fit the surrogate once and assert object/content-hash identity across every
  method in the same training unit; reject method-specific refits.

### 7.6 Manifest/output test

- Every row, dataset/cache key, output path, episode row and summary records
  `expose_rk`.
- Every manifest row and summary records `target_rows=4000`, `actual_rows`,
  `overshoot_rows=actual_rows-target_rows`, and `episode_count`; enforce
  `4000 <= actual_rows <= 4024` for the registered 25-step collector episodes.
- Hidden and full outputs are disjoint.
- Aggregation groups by regime and rejects missing/mixed labels.
- Method-side cache paths use public content hashes rather than family-derived
  method identities.

### 7.7 Existing tests to update

- `tests/real/test_real_ecology.py`: make public-control and guardian assertions
  explicitly `full`; add hidden schema assertions.
- `tests/real/test_native_baselines.py`: preserve full native tests and add
  data-fitted hidden-native tests.
- `tests/synthetic/test_cumulative_controls.py`: keep synthetic control semantics
  unchanged and explicit.
- Manifest/aggregation, telemetry and reward-leakage tests: add regime labels and
  hidden method-artifact checks; add direct public-reward-label and private
  reward-decomposition separation tests.
- `scripts/run_motivation_acceptance.py`: prove the historical arm is `full` and
  unchanged.

### 7.8 Phase 0 and runtime gates

Before the full refactor, run a cheap fixed-seed identifiability probe using only
the future hidden schema and candidate fitter. Its purposes are to falsify an
obviously broken estimator and characterize recovery of r/K/action effects. It
is **not** a success gate: weak/non-identifiability is reported as the intended
hidden-information uncertainty. Use the registered 4000-row target, preserve the
final complete episode, and do not increase or tune the target in response.

For every Phase 0 cell, also record:

- `target_rows`, `actual_rows`, `overshoot_rows`, and `episode_count`;
- genuine-termination count/prevalence by fit and holdout fold, target class
  counts, and whether `risk_fallback=constant_single_class` activates;
- reward-surrogate fit/holdout RMSE and MAE overall and by opaque token;
- holdout RMSE/MAE on the public low-reward tail defined by the fit-fold 10th
  percentile;
- evaluator-only reward RMSE/MAE stratified by private
  `safety_penalty_applied` status.

These are measure-and-document diagnostics, not gates for increasing data,
retuning the surrogate/methods, introducing a public threshold, or dropping a
reward mode. If both public extinction risk and reward-penalty association are
weak, retain the run and apply the pre-registered information-limited safe-mode
interpretation from Section 2.3.

After unit tests:

1. Run the full-mode no-op proof.
2. Run hidden Amur tiger / Ricker / sigma 0.0 / safe for `refplan`, `bamcts`,
   `ogsrl`, `moor_native`, and `plus_native`.
3. Run hidden Iberian lynx / Allee / sigma 0.2 / safe for the same methods.
4. Check finite fits/actions, intended fallback behavior, evaluator-only truth,
   regime labels and table/family invariance.
5. Run `make test` and `make docs-check` before any substantive experiment.

## 8. Risks and Open Confirmations

### 8.1 Conflicts between the brief and current code

1. The source design describes one pooled agent over nine populations, but the
   current runner trains one method per population/family cell.
2. The settled corrected schema requires opaque `pop_id` and action cost, but the
   current public dataset contains neither. Population is supplied through
   private config; cost comes from rich table-derived action specs. The new public
   cost channel is mandatory; only future cross-cell pooling is optional.
3. The current native filter is built from exact tables before the native policy
   is fitted. Hidden native fitting requires reversing that lifecycle.
4. The current public schema guard permits rho/kappa/K_eff and does not inspect
   metadata, serialized model config or rich action/reward objects.
5. General planners currently use exact `K_ref`, reward benefit, collapse
   penalty and safety threshold. The brief's reward-label-only contract therefore
   requires a learned method-side reward/risk model, not only feature deletion.
6. `true_family` is not merely evaluator metadata today: full config reaches
   methods and learned filters branch on regime behavior.
7. The collector already treats 4000 as a target and preserves complete episodes,
   so early-terminating cells can contain 4005 rows. This is not a hidden-r/K
   difference and does not inherently break paired full-mode equivalence, but the
   old "4000 transitions/cell" shorthand hid the distinction between target and
   actual rows. This plan resolves it by preserving existing episode-complete
   behavior and recording the bounded overshoot identically in both regimes.

### 8.2 PI decisions resolved for this repair

1. **Population identity:** expose an opaque categorical token with no demographic
   lookup; allow pooled-token training in the interface while retaining the first
   repair's controlled per-cell run shape.
2. **Safety information:** keep the current K-derived threshold and every derived
   safety label private/evaluator-only; do not introduce an absolute public
   threshold in this repair.
3. **Public-data reward/risk model:** use the single shared, frozen Section 5
   procedure. OGSRL uses its public termination hazard plus public-support OOD
   guardian, never private unsafe truth.
4. **Transition budget:** use 4000 target transitions per cell while preserving
   complete episodes; allow and record up to 24 overshoot rows for the registered
   25-step episode length. Do not tune the target from Phase 0.
5. **Safe-mode framing:** weak public safety signals mean information limitation
   under a private objective, not intrinsic OGSRL failure. Keep all safe-mode
   results and prohibit post-hoc thresholds, retuning or removal.

These decisions were supplied after audit and close the scientific-design gate.
Any departure requires a new documented setting or later ablation, not an
implementation-time substitution.

The `full`/`hidden` split, hidden default, no open-world PLUS, and no `noisy`
variant in this repair are already settled by the fix brief.

### 8.3 Items that require execution to confirm

- whether r/K/action effects are identifiable from the 4000-target,
  episode-preserving noisy dataset;
- per-cell termination prevalence, risk-fallback activation and whether the
  public reward surrogate recovers a useful penalty association;
- numerical stability and runtime of per-form native fitting;
- byte-identical artifacts under table corruption and family relabeling;
- whether hidden native models degrade under scarce/noisy data;
- smoke completion, fallback behavior and full-mode no-op equivalence;
- final performance changes, which are outcomes rather than acceptance gates.

## 9. Implementation Order After Approval

1. Run the non-gating Phase 0 identifiability probe and record its fixed-seed
   termination/reward diagnostics without changing the 4000-row target.
2. Capture the pre-change fixed-seed `full` canonical payload, method inputs, RNG
   states, solver parameters, representative actions and metrics.
3. Add `expose_rk`, regime labels and full-mode equivalence tests.
4. Split private simulator config from hidden-only sanitized `MethodContext`; add
   the fail-loud guard for unconverted hidden methods, then pass the `full`
   equivalence gate before adding hidden scientific logic.
5. Implement regime-aware dataset/public metadata/private sidecar schemas,
   including mandatory costs, opaque `pop_id`, separate termination/truncation,
   private diagnostic strata, and target/actual/overshoot/episode-count records.
6. Remove hidden controls, K scales, safety and family from shared beliefs,
   dynamics, caches and planning.
7. Add cost-only actions and the frozen Section 5 reward/risk fitting artifact.
8. Implement native public-data fitting, hidden grids and solver construction.
9. Update the five headline methods first. During this stage every other hidden
   method must fail loudly; then convert all remaining adapted methods before the
   schema repair is declared complete.
10. Rework pipeline native lifecycle and cache/output namespacing.
11. Add manifest/runner/aggregation guards and acceptance scripts.
12. Run unit/schema tests, full equivalence proof, two hidden smoke cells,
    `make test`,
    and `make docs-check`.

This document now records the resolved PI choices. Stop before source-code
implementation until the revised plan itself receives explicit approval.
