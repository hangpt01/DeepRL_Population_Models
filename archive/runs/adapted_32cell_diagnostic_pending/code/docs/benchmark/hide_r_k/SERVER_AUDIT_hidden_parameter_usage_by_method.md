# Hidden-Parameter Usage By Method Audit

**Date:** 2026-07-14  
**Scope:** read-only audit of the current real-ecology benchmark.  
**Requested output:** `docs/benchmark/SERVER_AUDIT_hidden_parameter_usage_by_method.md`  
**Code/experiment logic changes:** none.

## Executive Verdict

The current benchmark is internally consistent with its recent docs, but the docs
make several ecological quantities public that could be considered hidden under a
stricter POMDP design.

The important distinction is:

- **Private/evaluator-only:** true abundance, hidden structural parameters
  (`r_base`, `C`, `theta`, latent regime), true effective growth diagnostics,
  private reward diagnostics, and true safety-entry diagnostics.
- **Method-facing by current spec:** `rho`, `kappa`, `K_eff`, `next_rho`,
  `next_kappa`, `next_K_eff`, `K_ref`, `K_base`, `K_max`, `safety_threshold`,
  population identity, family/config identity, and action-table-derived
  `delta_r`, `delta_K`, and `stocking_delta`.

Consequently:

- `refplan`, `bamcts`, `ogsrl`, and `mopo` do **not** read private truth
  sidecars, but they do consume public r/K-derived controls in model fitting,
  belief features, rollout features, planning, or safety terms.
- `delphic` consumes public belief features and rewards; those belief features
  include `rho`, `kappa/K_ref`, and `K_eff/K_ref` when controls are enabled.
- Adapted `plus` and `moor` are mechanistic particle-MPC methods. They use exact
  action/species table structure through `MechanisticProposal` and
  `resolve_actions`.
- `plus_native` and `moor_native` are native discrete ecological solvers.
  `plus_native` reads exact public action/species table structure and estimates
  only a model-form posterior. `moor_native` reads exact public action/species
  structure but fits a scalar `K_hat`; it does not learn action growth set-points
  from `(o,a,o')`.

If the intended experiment requires r/K/action-table structure to be hidden from
methods, this is a **spec-level exposure faithfully implemented**, not a simple
accidental private-sidecar leak.

## Docs Read First

| Document | Relevant lines | Audit implication |
|---|---:|---|
| `docs/benchmark/01_problem_setting_and_design.tex` | 30-34 | Public history includes observations/actions plus public controls; true state, hidden structural parameters, regime, and true effective growth are evaluator-only. |
| `docs/benchmark/01_problem_setting_and_design.tex` | 46-61 | Set-point cells define public `rho`, `kappa`, `K_eff`, and `r_eff = clip(rho, r_min, r_max)`. |
| `docs/benchmark/01_problem_setting_and_design.tex` | 91-103 | Public dataset has rewards and public controls; private sidecar has truth and structural parameters. |
| `docs/benchmark/02_real_ecology_data_actions_and_costs.tex` | 26-29 | `species.csv`, `species_lambda.csv`, `actions.csv`, and `action_effects_long.csv` define population scale, lambda profiles, action menu, and precomputed effects. |
| `docs/benchmark/02_real_ecology_data_actions_and_costs.tex` | 81-85 | `a10` uses `0.10 * N0`; capacity actions use `(K_multiplier - 1) * K_base` and accumulate through `kappa`. |
| `docs/benchmark/02_real_ecology_data_actions_and_costs.tex` | 89-106 | Lambda is joined by population/action source; converted to Ricker/LGM set-points; real action table sets `rho` from that value and clips to species caps. |
| `docs/benchmark/03_state_action_reward_reference.tex` | 24-32 | Visibility table: `s`, `r_base`, `C`, `theta`, `z` private; `rho`, `kappa`, `K_eff` public. |
| `docs/benchmark/03_state_action_reward_reference.tex` | 37-51 | In set-point ecology, `rho` carries growth and `K` is public because action table/population identity are public. |
| `docs/benchmark/03_state_action_reward_reference.tex` | 59-79 | Reward uses true next state, `K_ref = K_base`, and safety penalty. |
| `docs/benchmark/03_state_action_reward_reference.tex` | 93-99 | Public reward is a value-fitting target; evaluator truth remains in private sidecar. |
| `docs/benchmark/04_algorithm_adaptations_and_claims.tex` | 43-48 | Shared infrastructure uses public controls and public belief cache. |
| `docs/benchmark/04_algorithm_adaptations_and_claims.tex` | 70-73, 102-104, 133-134, 254-256 | General methods are documented as using public data/beliefs/controls, not private truth. |
| `docs/benchmark/04_algorithm_adaptations_and_claims.tex` | 143-203 | Adapted `plus`/`moor` are mechanistic; `plus` spans candidate `K`, `moor` fits Ricker `K` in set-point mode. |
| `docs/benchmark/05_implementation_and_code_map.md` | 15-24 | Public/private dataset boundary and cache keys are intentional. |
| `docs/benchmark/05_implementation_and_code_map.md` | 33-51 | Code map for config, realdata, actions, controls, env, reward, dataset, filters, dynamics, planning, and methods. |
| `docs/benchmark/06_experiment_protocol_and_reproducibility.md` | 34-39 | Run flow: resolve env/action table, load/generate public dataset/private sidecar, fit public filter cache, fit policy, evaluate. |
| `docs/benchmark/06_experiment_protocol_and_reproducibility.md` | 62-72 | Public datasets and private truth are separate; manifests record population/family/noise/reward/filter/method. |
| `docs/benchmark/09_motivation_native_results.tex` | 54-81 | Motivation setting explicitly says action/species tables publicly determine growth set-points and capacity effects. |
| `docs/benchmark/09_motivation_native_results.tex` | 142-150 | Native methods build mechanistic transition tables from public ecological structure while general methods learn dynamics. |
| `docs/benchmark/09_motivation_native_results.tex` | 380-407 | Proposed next ablation must change planning-model source; `filter=ricker` or `filter=true_family` only changes state estimation. |
| `docs/benchmark/SERVER_METHOD_ADAPTATIONS_motivation_native.md` | 33-40, 77-85 | Native baselines receive public action-specific growth and species capacity; only abundance and regime remain hidden. |
| `docs/benchmark/SERVER_METHOD_ADAPTATIONS_motivation_native.md` | 174-231 | `moor_native` is single-Ricker but gets public structure; `plus_native` is model-identifying, not naive. |
| `docs/benchmark/SERVER_METHOD_ADAPTATIONS_motivation_native.md` | 279-288 | General methods learn approximate dynamics; native solvers read public ecological structure directly. |
| `docs/benchmark/README.md` | 53-57 | Method names are benchmark-native adaptations. |
| `docs/benchmark/README.md` | 65-85, 99-104 | Glossary marks `rho`, `kappa`, `K_eff`, rewards, and public controls as public; truth and structural parameters as evaluator-only. |

## Shared Code Paths

| Component | File/function/lines | What is exposed or consumed | Classification |
|---|---:|---|---|
| Public dataset schema | `src/real_ecology_benchmark/dataset.py:14-31` | Public fields include observations, actions, rewards, next observations, and public controls `rho`, `kappa`, `K_eff`, `next_rho`, `next_kappa`, `next_K_eff`. | Method-facing public data. |
| Private dataset schema | `src/real_ecology_benchmark/dataset.py:33-48` | Private fields include true states, `r_base`, `C`, `theta`, regime, reward truth, and optional `r_eff_true`. | Evaluator/audit-only. |
| Collection split | `src/real_ecology_benchmark/collector.py:295-322` | Collector writes public controls to public dataset and hidden truth to private sidecar. | Boundary implementation. |
| Environment public info | `src/real_ecology_benchmark/envs.py:148-152` | Runtime `public_info` includes controls. | Method-facing. |
| Environment private truth | `src/real_ecology_benchmark/envs.py:124-146`, `350-375` | Truth includes state, `r_base`, `C`, `theta`, regime, `r_eff_true`, safety diagnostics. | Evaluator-only. |
| Action table loading | `src/real_ecology_benchmark/realdata.py:195-222` | Loads `lambda`, `r_setpoint_ricker`, `r_setpoint_lgm`, `dK_step`, `dN`, cost. | Table source; method-facing if `resolve_actions` is called by method/proposal. |
| Action resolution | `src/real_ecology_benchmark/actions.py:140-168`, `204-216` | `real_action_table` maps effects to `ActionSpec(delta_r, delta_K, stocking_delta, cost)`. | Exact table lookup. |
| Public controls | `src/real_ecology_benchmark/controls.py:1-10`, `40-70` | `rho`, `kappa`, `K_eff`, and `next_*` are computed from `resolve_actions` and `K_base`. | Method-facing by current spec. |
| Effective r | `src/real_ecology_benchmark/controls.py:73-99` | `private_r_eff` clips `rho` in set-point mode; `r_base` ignored there. | Simulator/proposal construction; `r_eff_true` logged privately. |
| Belief state features | `src/real_ecology_benchmark/types.py:44-50`, `84-118` | `BeliefState.features` appends `rho`, `kappa/K_ref`, `K_eff/K_ref`; unsafe probability uses `safety_threshold`. | Method-facing feature channel. |
| Learned filter proposal | `src/real_ecology_benchmark/beliefs.py:168-215`, `248-291` | Requires dataset controls; feature matrix uses `next_rho`, `next_kappa/K_ref`, `next_K_eff/K_ref`. | Method-facing belief/filter construction. |
| Mechanistic proposal | `src/real_ecology_benchmark/beliefs.py:324-390` | Reads species caps for assumed family, calls `resolve_actions`, builds exact `delta_r`, `delta_K`, `stocking_delta` lookups, then simulates. | Mechanistic table lookup. |
| Belief cache | `src/real_ecology_benchmark/beliefs.py:676-775` | Stores belief features plus `rho`, `kappa`, `K_eff`, and next controls. | Method-facing cache. |
| Discrete native filter | `src/real_ecology_benchmark/beliefs.py:580-648` | Native belief stores `rho`, `kappa`, `K_eff`; update uses `NativeSolver.update_log_weights` and public control advance. | Native method-facing filter. |
| Learned dynamics | `src/real_ecology_benchmark/dynamics.py:40-58`, `92-120` | `_design` adds `rho_next`, `kappa_next/K_ref`, `K_eff_next/K_ref`; `fit` feeds cache controls. | Method-facing model-learning feature. |
| Particle MPC | `src/real_ecology_benchmark/planning.py:69-146` | Copies `belief.rho/kappa` into rollouts; advances public controls; reward/safety use `K_ref` and `safety_threshold`. | Planning input. |
| Reward model | `src/real_ecology_benchmark/reward.py:1-13`, `29-63`, `75-117` | Public logged reward is true-next-state based; `build_reward` uses action costs, `K_ref`, collapse penalty, safety threshold. | Public target and planning reward. |
| Filter routing | `src/real_ecology_benchmark/pipeline.py:146-198` | `learned`, `ricker`, `true_family`, and `native_discrete` construct different proposals/filters; `true_family` changes filter proposal only. | Method/filter configuration. |
| Native routing | `src/real_ecology_benchmark/pipeline.py:251-297` | Native methods require `filter='native_discrete'`; cache key includes native grid size. | Native method-facing. |

## Variable-Channel Classification

| Quantity / alias | Current source | Current visibility | Method-facing paths | Estimated from public data or exact value? | Consequence |
|---|---|---|---|---|---|
| `rho`, `next_rho` | `advance_public_controls`; action `delta_r` | Public by docs/code | Dataset, belief state, belief cache, learned dynamics, planners, native filters | Exact table-derived public control | Public r feature leak if r should be hidden. |
| `r_eff`, `r_eff_true` | `clip(rho, r_min, r_max)` in set-point mode | `r_eff_true` private; effective value implied by public `rho` and caps | Mechanistic proposals/native solvers compute it internally | Exact from public `rho` and species caps | Hidden-growth ambiguity mostly removed. |
| `r_setpoint_ricker`, `r_setpoint_lgm` | `action_effects_long.csv` | Table-derived public action effect by current docs | `real_action_table`, `resolve_actions`, proposals/native solvers | Exact table values | Mechanistic-planner advantage. |
| Lambda profile | `species_lambda.csv`, precomputed into effects | Not directly passed as lambda arrays to policies | Indirect through `r_setpoint_*` in action effects | Exact converted table value | Indirect r leak. |
| `r_base` | Env private draw/shared synthetic interface | Private | Sidecar/evaluator; not public dataset | Private simulator value | Harmless evaluator-only in current run. |
| `r_min`, `r_max` | `species.csv` caps via config | Method-facing config | Controls/proposals/native solver configs | Exact table/config values | Enables exact clipping of public `rho`. |
| `kappa`, `next_kappa` | Public accumulator | Public | Dataset, beliefs, dynamics, planning, native filter | Exact action-history-derived control | Public K feature leak if capacity should be hidden. |
| `K_base` | `species.csv` via `EnvironmentConfig` | Method-facing config | Config, controls, reward, grid, proposals, native solvers | Exact species table value, except `moor`/`moor_native` may fit a candidate `K` for assumed dynamics | Direct species-scale exposure. |
| `K_eff`, `next_K_eff` | `clip(K_base + kappa, K_min, K_max)` | Public | Dataset, beliefs, dynamics features, OGSRL guardian, native filter | Exact public derived value | Direct effective-capacity feature. |
| `K_ref` | Config, real default `K_ref=K_base` | Method-facing config | Belief features, learned dynamics normalization, reward utility, value functions | Exact config/species value | Indirect K leak via normalization and reward. |
| `K_max` | `species.csv` | Method-facing config | PLUS candidate ranges, native grids, config validation | Exact species table value | Species capacity-range exposure. |
| `dK_step` / `delta_K` | `action_effects_long.csv` | Public action effect by current docs | `resolve_actions`, public control advance, mechanistic proposals, native solvers | Exact table value | Mechanistic-planner advantage. |
| `dN` / `stocking_delta` | `action_effects_long.csv` | Public action effect by current docs | Env transition, mechanistic proposals, native solvers | Exact table value | Translocation magnitude exposed. |
| `s_safe` / `safety_threshold` | Derived from `K_base` and depletion rule | Method-facing config | Belief features, filters, reward/safety, planners, OGSRL guardian | Exact config-derived value | Indirect K leak and safety-feature channel. |
| `pop_id` / population | Manifest/config | Method-facing config | `real_environment`, `resolve_actions`, species/action lookup | Exact row identity | Enables species table lookup. |
| `true_family` / `cfg.kind` | Manifest/config | Method-facing config; true-family filter optional | Environment config, family-specific caps/action conversion; `filter=true_family` only if requested | Exact cell family in config | Structural-family leak risk; motivation generals used learned filter, but methods still receive `cfg.kind`. |
| `C`, `theta`, regime `z` | Env hidden parameters/state | Private/evaluator; regime hidden in native grid for regime candidate | Belief particles carry sampled contexts/regimes; native regime candidate includes hidden regime axis | Not direct truth except via candidate model assumptions | Mostly evaluator/private; candidate form may model it. |
| True state, reward decomposition, unsafe/collapse diagnostics | Env/private sidecar/evaluator | Private diagnostics; reward scalar public | Evaluation metrics; reward scalar used for value fitting | True state not method-facing; reward scalar public | Reward is supervised target, not filter observation. |

## Per-Method Audit Table

| Method | Consumes r/K or derived info? | Exact variables used | Where used | Private vs method-facing | Estimated vs exact | Short consequence |
|---|---|---|---|---|---|---|
| `refplan` | **Yes** | `rho`, `kappa`, derived `next_rho`, `next_kappa`, `next_K_eff`, `K_ref`, `safety_threshold`, reward/action costs | `ContinuousDynamicsEnsemble.fit` and `_design`; `ParticleMPC.score_sequences`; posterior update | Method-facing public features/config | Learns dynamics from public data; reads exact config for normalizers/reward/safety | Public feature leak; learned-model limitation remains. |
| `bamcts` | **Yes** | `rho`, `kappa`, `K_ref`, `safety_threshold`, reward/action costs; learned dynamics also gets derived next controls | Dynamics fit; tree key buckets `rho/kappa`; simulated transitions and disagreement | Method-facing public features/config | Learns dynamics from public data; reads exact config for reward/safety | Public feature leak; tree explicitly conditions on public controls. |
| `ogsrl` | **Yes** | `rho`, `kappa`, `K_eff`, `K_ref`, `safety_threshold`, reward/action costs, unsafe features | Dynamics fit; KNN guardian features; actor state features; rollouts; feasibility mask | Method-facing public features/config | Learns dynamics/guardian from public data; reads exact config for reward/safety | Strong public K/r feature channel plus safety feature channel. |
| `moor_native` | **Partial/Yes** | Exact `r_setpoint_*` through action specs, exact `dK_step`, `dN`, public `kappa`, exact `K_ref`, `K_max`, `safety_threshold`; fitted `K_hat` used as assumed `K_base` | `MOORNativePolicy.fit`; `predict_ricker_next`; `NativeSolver.build`; native discrete filter | Method-facing mechanistic table lookup plus one fitted scalar | Fits `K_hat`; does not estimate action r/dK/dN/safety/K_ref from scratch | Mechanistic-planner advantage; naive in form, not blind to tables. |
| `plus_native` | **Yes** | Exact `r_setpoint_*`, `dK_step`, `dN`, `K_base`, `K_max`, `K_ref`, `safety_threshold`, family candidate set, public `kappa` | `PLUSNativePolicy.fit`; `NativeSolver.build` per family; candidate belief bank and evidence update | Method-facing mechanistic table lookup/native solver | Does not estimate r/K; estimates only family posterior/evidence | Strong mechanistic-planner and form-identification advantage. |
| `mopo` | **Yes** | `rho`, `kappa`, derived next controls, `K_ref`, `safety_threshold`, reward/action costs | Dynamics fit; prediction RMSE; ParticleMPC planning/disagreement | Method-facing public features/config | Learns dynamics from public data; exact config for reward/safety | Same public-control feature leak as other learned planners. |
| `delphic` | **Yes, indirect** | Belief features include `rho`, `kappa/K_ref`, `K_eff/K_ref`, unsafe probability; rewards are public targets | Compatible-world features, CQL/Q fitting, greedy action features | Method-facing public belief features/reward | Does not fit mechanistic r/K; consumes cache features exactly as built | Indirect K/r leak through shared feature vector. |
| adapted `plus` | **Yes** | Exact action set-points through `resolve_actions`, `K_base`, `K_max`, `K_ref`, public `rho/kappa/K_eff`, `dK_step`, `dN`, reward/safety | Candidate `K` bank; `MechanisticProposal`; candidate belief bank; ParticleMPC | Method-facing mechanistic table lookup | Candidate bank spans `K`; action r is treated as known in set-point mode | Mechanistic-planner advantage. |
| adapted `moor` | **Yes** | Exact action set-points via `rho_next`, `dK_step`, `dN`, `K_ref`, `K_base/K_max` grid, public `kappa`, reward/safety | `_predict`; grid fit; `MechanisticProposal`; ParticleMPC | Method-facing mechanistic table lookup plus fitted `K_hat` | In set-point mode r is known from `rho`; fit mainly identifies `K` | Mechanistic-planner advantage; single-Ricker misspecification remains. |

## Method Details And References

### `refplan`

- `fit()` calls `ContinuousDynamicsEnsemble.fit(..., K_ref, env_cfg)`:
  `src/real_ecology_benchmark/methods/refplan.py:36-43`.
- The learned ensemble design includes `rho_next`, `kappa_next/K_ref`, and
  `K_eff_next/K_ref`: `src/real_ecology_benchmark/dynamics.py:40-58`.
- The ensemble fit pulls `cache.rho/cache.kappa`:
  `src/real_ecology_benchmark/dynamics.py:110-118`.
- Planning calls `ParticleMPC.score_sequences`, whose rollouts copy
  `belief.rho/belief.kappa` and advance controls:
  `src/real_ecology_benchmark/methods/refplan.py:69-73`,
  `src/real_ecology_benchmark/planning.py:69-146`.
- Posterior member update uses `member.mean_next(..., belief.rho, belief.kappa)`:
  `src/real_ecology_benchmark/methods/refplan.py:86-100`.

**Consequence:** public r/K-derived controls enter both model fitting and
planning. This is not private-sidecar leakage, but it is not hidden-r/K either.

### `bamcts`

- `fit()` calls the same `ContinuousDynamicsEnsemble.fit`:
  `src/real_ecology_benchmark/methods/bamcts.py:45-52`.
- Tree keys bucket `rho` and `kappa`:
  `src/real_ecology_benchmark/methods/bamcts.py:62-70`.
- `_simulate()` passes `rho/kappa` into learned dynamics, advances public
  controls, computes reward and safety from config, and recurses on next controls:
  `src/real_ecology_benchmark/methods/bamcts.py:72-117`.
- `act()` seeds simulations with `belief.rho/belief.kappa`:
  `src/real_ecology_benchmark/methods/bamcts.py:124-141`.

**Consequence:** BA-MCTS is explicitly conditioned on public r/K controls during
search. One implementation wrinkle: `observe()` does not pass `rho/kappa` into
`mean_next` at `src/real_ecology_benchmark/methods/bamcts.py:157`; that is an
update inconsistency, not an extra hidden-parameter exposure.

### `ogsrl`

- `KNNGuardian.features` includes `rho`, `kappa/K_ref`, and `K_eff/K_ref`:
  `src/real_ecology_benchmark/methods/ogsrl.py:35-77`.
- `_state_features` includes unsafe indicator plus `rho`, `kappa/K_ref`, and
  `K_eff/K_ref`: `src/real_ecology_benchmark/methods/ogsrl.py:164-175`.
- `_rollouts()` uses `rho/kappa`, learned dynamics, public-control advance,
  reward, safety, and guardian OOD:
  `src/real_ecology_benchmark/methods/ogsrl.py:191-245`.
- `fit()` trains dynamics and guardian with belief-cache controls:
  `src/real_ecology_benchmark/methods/ogsrl.py:297-314`.
- `act()` computes policy and action risks from these control-aware features:
  `src/real_ecology_benchmark/methods/ogsrl.py:393-435`.

**Consequence:** OGSRL has the broadest general-method public-feature exposure:
learned dynamics, guardian support, actor features, reward, and safety all see
K/r-derived public context.

### `moor_native`

- `fit()` reads public `dataset.kappa`, searches candidate `K`, and targets
  `dataset.next_observations` normalized by `env_cfg.K_ref`:
  `src/real_ecology_benchmark/methods/moor_native.py:24-47`.
- `predict_ricker_next` builds a Ricker assumption config, calls
  `resolve_actions`, uses initial public `rho`, public `kappa`, and
  `env.transition_value`: `src/real_ecology_benchmark/native_solver.py:73-105`.
- `NativeSolver.build(... assumed_K_base=self.K_hat)` constructs tabular
  transition/reward values with the fitted K:
  `src/real_ecology_benchmark/methods/moor_native.py:48-55`.
- Native table construction calls `resolve_actions`, `build_reward`, and
  `transition_value`: `src/real_ecology_benchmark/native_solver.py:120-136`,
  `src/real_ecology_benchmark/native_solver.py:189-255`.

**Consequence:** `moor_native` is naive in structural form (single Ricker), but
not uninformed. It estimates `K_hat`; it reads exact public action growth,
capacity increments, translocation magnitudes, reward costs, and safety scale.

### `plus_native`

- The module doc states the important assumption: growth set-point is public and
  `K_base` comes from the species table:
  `src/real_ecology_benchmark/methods/plus_native.py:1-16`.
- `fit()` ignores the dataset and builds one `NativeSolver` per candidate family:
  `src/real_ecology_benchmark/methods/plus_native.py:37-55`.
- `act()` combines per-candidate native solver action values:
  `src/real_ecology_benchmark/methods/plus_native.py:82-92`.
- `observe()` updates only the candidate model posterior/belief using
  action/observation evidence:
  `src/real_ecology_benchmark/methods/plus_native.py:94-126`.
- Each `NativeSolver` reads exact action/species/reward structure through
  `assumption_config`, `resolve_actions`, `build_reward`, and
  `transition_value`: `src/real_ecology_benchmark/native_solver.py:35-61`,
  `src/real_ecology_benchmark/native_solver.py:189-255`.

**Consequence:** `plus_native` does not estimate r/K from public transition data.
It is best described as a native model-identifying solver over four mechanistic
forms, with exact public table structure.

### `mopo`

- `fit()` calls `ContinuousDynamicsEnsemble.fit(..., K_ref, env_cfg)`:
  `src/real_ecology_benchmark/methods/mopo.py:18-30`.
- Fit diagnostics call `dynamics.predict(..., beliefs.rho, beliefs.kappa)`:
  `src/real_ecology_benchmark/methods/mopo.py:39-43`.
- `act()` calls `ParticleMPC.plan` and computes disagreement using
  `belief.rho/belief.kappa`: `src/real_ecology_benchmark/methods/mopo.py:45-60`.

**Consequence:** same public-control feature channel as the motivation general
planners, though `mopo` was not one of the three headline general methods in the
native-baseline result.

### `delphic`

- Compatible-world features are built from `cache.features`:
  `src/real_ecology_benchmark/methods/delphic.py:52-57`, `102-116`.
- `fit()` uses `beliefs.features`, `beliefs.next_features`, and
  `dataset.rewards`: `src/real_ecology_benchmark/methods/delphic.py:182-242`.
- `act()` calls `belief.features(K_ref, safety_threshold)`:
  `src/real_ecology_benchmark/methods/delphic.py:244-256`.
- The shared belief feature builder appends `rho`, `kappa/K_ref`, and
  `K_eff/K_ref`: `src/real_ecology_benchmark/types.py:84-118`.

**Consequence:** Delphic does not build a mechanistic planner, but it receives
r/K-derived controls indirectly through public belief features.

### Adapted `plus`

- In set-point mode, `fit()` states that per-action growth is known and candidate
  Ricker models span `K_base` to `K_max`:
  `src/real_ecology_benchmark/methods/plus.py:26-38`.
- Candidate proposals are `MechanisticProposal` objects, which call
  `resolve_actions` and precompute exact action lookups:
  `src/real_ecology_benchmark/beliefs.py:324-360`.
- Candidate belief banks preserve `rho`, `kappa`, and `K_eff`:
  `src/real_ecology_benchmark/methods/plus.py:71-80`.
- Planning scores sequences under those mechanistic candidates:
  `src/real_ecology_benchmark/methods/plus.py:82-101`.
- `observe()` advances public controls and updates candidate weights from
  observation evidence: `src/real_ecology_benchmark/methods/plus.py:115-163`.

**Consequence:** adapted `plus` is mechanistic and gets exact public action/species
structure, but it is still a particle-MPC adaptation rather than a native
discrete solver.

### Adapted `moor`

- `_predict()` calls `resolve_actions`, uses public-control advance, applies
  exact `stocking_delta`, clips `rho_next` to r caps, and computes `K_eff` from
  fitted `K` plus `kappa_next`:
  `src/real_ecology_benchmark/methods/moor.py:23-60`.
- In set-point mode, the code comment states that r is known from `rho`, so the
  fit identifies K only: `src/real_ecology_benchmark/methods/moor.py:38-43`.
- `fit()` searches K/r grids, but the r value is effectively unused in set-point
  prediction; it then constructs a `MechanisticProposal` and `ParticleMPC`:
  `src/real_ecology_benchmark/methods/moor.py:62-97`.
- `act()` plans with that proposal:
  `src/real_ecology_benchmark/methods/moor.py:99-115`.

**Consequence:** adapted `moor` is a single-Ricker mechanistic baseline with
exact public action/set-point structure and fitted K.

## Final Summary Matrix

Cells list the r/K or hidden-adjacent quantities consumed in each role.

| Method | Model learning | Belief/filter | Planning | Reward/safety | Native solver construction |
|---|---|---|---|---|---|
| `refplan` | `rho`, `kappa`, `next_rho`, `next_kappa`, `next_K_eff`, `K_ref` via learned dynamics | Shared cache features include `rho`, `kappa/K_ref`, `K_eff/K_ref` | `belief.rho`, `belief.kappa`, public control advance | `K_ref`, `safety_threshold`, action costs | none |
| `bamcts` | Same learned dynamics controls as `refplan` | Shared cache features; posterior particles | `rho/kappa` in tree keys and simulated rollouts | `K_ref`, `safety_threshold`, action costs | none |
| `ogsrl` | Same learned dynamics controls | Shared cache plus guardian anchors with `rho/kappa/K_eff` | Actor/rollouts use `rho/kappa/K_eff/K_ref` | `safety_threshold`, unsafe features, guardian OOD, reward costs | none |
| `moor_native` | Fits `K_hat` from public observations/actions/kappa; no learned general dynamics | `native_discrete` grid stores `rho/kappa/K_eff` | Tabular action values indexed by public kappa | Exact `K_ref`, `safety_threshold`, action costs | Exact `r_setpoint`, `dK_step`, `dN`; fitted assumed `K_base=K_hat`; exact config caps/ranges otherwise |
| `plus_native` | No learned dynamics; estimates family posterior only | Candidate native beliefs store public kappa/control state | Posterior mixture of native tabular action values | Exact `K_ref`, `safety_threshold`, action costs | Exact `r_setpoint`, `dK_step`, `dN`, `K_base`, `K_max`; candidate family bank |
| `mopo` | `rho/kappa/next_*`, `K_ref` via learned dynamics | Shared cache features | ParticleMPC with `belief.rho/kappa` | `K_ref`, `safety_threshold`, action costs, pessimism | none |
| `delphic` | No dynamics model; Q fitting uses public reward and cache features | Belief features include `rho`, `kappa/K_ref`, `K_eff/K_ref` | Greedy Q over belief features | Public rewards; `safety_threshold` appears in features | none |
| adapted `plus` | No learned dynamics; candidate mechanistic bank | Candidate belief bank keeps `rho/kappa/K_eff` | ParticleMPC under exact-table mechanistic proposals | `K_ref`, `safety_threshold`, action costs | none; mechanistic proposal uses exact `r_setpoint`, `dK_step`, `dN`, candidate `K` |
| adapted `moor` | Fits K-like Ricker proposal from public belief means/actions/kappa | Shared learned/raw filter cache | ParticleMPC under fitted Ricker proposal | `K_ref`, `safety_threshold`, action costs | none; mechanistic proposal uses exact action table and fitted K |

## Audit Conclusion

The current benchmark does not appear to leak private sidecar arrays directly into
training or deployment for the audited methods. The more important issue is that
the **public spec itself exposes** action-specific growth set-points,
capacity-control state, species-scale `K` quantities, and K-derived safety
thresholds. General methods consume these mostly as learned-model features and
planning context. Mechanistic and native ecological methods consume the same
public structure as exact transition/reward model inputs, which creates the
mechanistic-planner advantage observed in the motivation experiment.

No hidden-rK fix is implemented here. Any fix should first decide which of the
currently public quantities should remain legitimate manager-visible controls and
which should be withheld, randomized, or replaced by learned public surrogates.
