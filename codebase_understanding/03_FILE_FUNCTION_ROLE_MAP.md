# File/function role map

Primary paths below mean `src/tracks/general/real_ecology_benchmark/` unless
stated. “Risk” is impact if semantics change.

| File | Role; main symbols | Inputs → outputs; call graph | Affects / risk / why |
|---|---|---|---|
| `cli.py` | CLI, `_config`, `cmd_run`, `build_parser` | YAML+overrides → `run_method`; called by `__main__` | configs/all; high: precedence and entrypoint |
| `__main__.py` | module launcher | `python -m real_ecology_benchmark` → `cli.main()` | entrypoint only; low |
| `pipeline.py` | orchestration, `ensure_dataset`, `make_filter_factory`, `build_method`, `run_method` | config/cache → fitted policy+summary; calls collector/beliefs/evaluator | all; critical |
| `config.py` | all dataclasses, `real_environment(_like)`, `MethodContext` | YAML/CSV rows → validated `BenchmarkConfig` | environment/method visibility; critical |
| `envs.py` | `ContinuousEcologyEnv`, `transition_value`, `step` | config+action+RNG → observation/reward/private truth | scientific object; critical |
| `actions.py` | `ActionSpec`, `real_action_table`, `resolve_actions` | effect CSV → executable actions | environment/reward; critical |
| `controls.py` | `PublicControls`, `advance_public_controls`, `private_r_eff` | action history → rho/kappa/K/r | dynamics; critical |
| `realdata.py` | `RealAction/Population/Effect`, CSV loaders | `configs/ecology/*.csv` → typed lookup | inputs; critical |
| `dummydata.py` | in-code dummy tables | fixtures → typed lookup | tests only; low for accepted results |
| `reward.py` | `ContinuousReward`, indicator and penalty helpers | state/observation/action → three reward channels | evaluation/dataset; critical |
| `observation.py` | `LogNormalObservationModel` | state+sigma+RNG → survey/log likelihood | beliefs/environment; high |
| `collector.py` | `CollectorProfile`, `MixedDangerZonePolicy`, `collect_dataset` | simulator+seed → public/private trajectories | dataset; critical |
| `dataset.py` | `TrajectoryDataset`, `PrivateTrajectoryData`, serialization/hash | arrays ↔ `.npz` | privacy/cache identity; critical |
| `beliefs.py` | proposal/filter ladder and caches | public history/model → `BeliefState`/features | beliefs; critical |
| `public_surrogate.py` | `PublicRewardRiskSurrogate`, `fit_public_surrogate` | public train episodes → reward/risk predictor | hidden planning; critical |
| `public_models.py` | `PublicDynamicsMember/Ensemble`, `PublicParticlePlanner` | public beliefs/data → sampled plans | general methods; high |
| `behavior_model.py` | `CalibratedBehaviorModel`, `fit_reference_behavior` | belief features/actions → behavior prior | RefPlan; medium |
| `dynamics.py` | `LinearDynamicsMember`, `ContinuousDynamicsEnsemble` | full beliefs/data → bootstrap dynamics | full-mode methods; high |
| `planning.py` | `ParticleMPC`, `PlanDiagnostics` | belief+proposal+reward → action | full-mode plans; high |
| `planners/pbvi.py` | `PointBasedPlanner` | candidate POMDP+belief → finite-horizon Q | adapted methods; critical |
| `native_fit.py` | public four-form fits, `FittedNativeSolver` | hidden public data → tabular transition/reward/Q | native/legacy PLUS; high |
| `native_solver.py` | known-config `NativeSolver` | full config → grid/VI/filter | full native baselines; high |
| `discretize.py` | `NativeGrid`, `build_native_grid` | config → hidden state/control grid | tabular solvers; high |
| `faithful_fit.py` | `FitResult`, `CandidateBank`, LBFGS fitting/cache | ordered public episodes → mechanistic candidates | adapted methods; critical |
| `faithful_ecology.py` | `MechanisticModel`, regime law | parameters/action → candidate transition | adapted dynamics; critical |
| `faithful_pomdp.py` | `CandidateBelief/POMDP` | model+surrogate+history → belief/reward | adapted planning; critical |
| `faithful_artifacts.py` | model/planner serialization and validation | fitted candidates → hashes/artifacts | provenance; high |
| `evaluator.py` | `ContinuousEvaluator.run/summarize/save` | fitted policy+fresh seeds → rows/metrics | headline scores; critical |
| `gate.py` | `ExactEpisodeProposal`, `run_decision_gate` | env+clairvoyant proposal → ceiling/gate | verification; high |
| `manifest.py` | grid creation and `aggregate_summaries` | configs/summaries → CSV/JSON comparisons | experiment selection; high |
| `training_monitor.py` | episode split, `TrainingHistory`, artifact logging | dataset/cache/fit metrics → diagnostics | fit reporting; medium |
| `rollout.py` | `AugmentedRolloutState` helper contract | state/control context → rollout container | not on accepted `run_method()` path; low |
| `general_canary_acceptance.py` | general-only validity receipt construction/loading | row+summary+hashes → canary receipt | acceptance/provenance; high for general artifact admission |
| `telemetry.py` | elapsed time/RSS | process → timing fields | compute audit; low |
| `backend.py` | NumPy/CuPy resolution and strict fallback | compute config → backend | numerics/provenance; high |
| `privacy.py` | forbidden-name/path traversal | hidden artifacts → validation/error | leakage protection; high |
| `types.py` | reset/step/transition/belief protocols | shared data contracts | all; high |

## Methods

Every file is called through `methods/__init__.py:METHODS` unless noted.

| File / class | Fit → deployed selector | Risk / distinction |
|---|---|---|
| `mopo.py:MOPOPolicy` | bootstrap dynamics → pessimistic particle MPC | high; generic model uncertainty |
| `refplan.py:RefPlanPolicy` | ensemble+behavior prior → posterior-marginalized public planner | high; accepted general |
| `bamcts.py:BAMCTSPolicy` | public ensemble → online MCTS/model-belief update | high; accepted general |
| `ensemble_value_disagreement.py:EnsembleValueDisagreementPolicy` | bootstrap fitted-Q iterations → mean Q minus disagreement | high; accepted general |
| `delphic.py` | alias shim in general | low; unreachable as distinct implementation |
| `ogsrl.py:OGSRLPolicy` | public dynamics, OOD guardian, constrained actor → guarded/risk-ranked action | high; accepted general |
| `moor.py:MOORPolicy` | one Ricker proposal/Q → MPC/Q | high; legacy adapted |
| `plus.py:PLUSPolicy` | four public forms → candidate posterior/plans | high; legacy adapted |
| `moor_native.py:MOORNativePolicy` | fitted Ricker tabular solver → belief-weighted Q | high |
| `plus_native.py:PLUSNativePolicy` | four fitted solvers → posterior aggregate Q | high |
| `moor_faithful.py:MOORFaithfulRickerPBVIPolicy` | one mechanistic Ricker fit → PBVI | critical; accepted ecological |
| `plus_faithful.py:PLUSFaithfulPBVIPolicy` | mechanistic candidate bank → posterior aggregate PBVI | critical; ecological subclass supplies accepted Ricker-only key |
| `value.py` | `fit_mechanistic_q`, `evaluate_q` | helper used by legacy MOOR/PLUS | medium |
| `base.py:BasePolicy` | lifecycle/privacy guard | high |

The triplicated names are not equivalent wrappers: `plus.py` uses lightweight
public form fits and MPC, `plus_native.py` uses tabular VI, and
`plus_faithful.py` uses LBFGS mechanistic fits plus PBVI. MOOR has the same
three-level distinction.

## Scripts, diagnostics, tests, and reachability

`scripts/general/run_real_manifest_row.py` and its ecological copy own manifest
overlay/output paths; `pipeline.py` owns scientific execution; `manifest.py`
owns generic grids/aggregation. This boundary is unclear because runners also
validate caches/gates and rewrite configs (`apply_row_config()`), so changing a
pipeline default may be shadowed by a row.

`src/diagnostics/replay_analysis/{loader,metrics,run_analysis,figures}.py` loads
instrumented logs, computes M1–M15, runs self-tests, and plots. Routine
`scripts/general/make_*manifest.py` files can be grouped as grid builders;
load-bearing runners are `run_real_manifest_row.py`, `run_manifest_row.py`,
their Slurm wrappers, and accepted preparation scripts. Diagnostic replay
scripts reproduce policies; follow-ups S2/H12/H14/S6/reward-screen answer
post-hoc questions.

Tests are separated by track and by `real`, `synthetic`, `dummy`. Scientific
tests include `test_faithful_equations.py`, `test_real_ecology.py`, privacy
tests, and evaluator/gate tests; many other assertions are shape/schema/cache
regressions.

Files that look important but are not distinct live behavior:
general `methods/delphic.py` is only an alias; archive code and flat historical
docs are unreachable from `cli.main()`. `rollout.py` supplies helper state but
is not on the primary accepted `run_method()` path. `backend.py`, `telemetry.py`,
and `types.py` add indirection/contracts rather than new policy behavior.
