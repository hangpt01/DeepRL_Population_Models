# Artifact bundle evidence v3

This contract is based on source inspection and successful fit-only probes for both the
Amur-tiger and crab-eating-fox cells. The two cells have the same component applicability
and native widths; their fitted bytes remain cell-specific. A probe action was used only to
verify fresh-object serialization/reload replay. No evaluator, truth archive, runtime
`next_states`, scientific evaluation action, or return path was opened.

## Source-backed component applicability

`preprocessing` and `feature_transformations` are not separate fitted objects in any general
policy. Public feature construction is executable source, while fitted centering/scaling
values are fields of the actual fitted component that uses them. They are therefore absent,
not placeholder artifacts. The ecological planner has no fitted alpha-vector bank, and MOOR
has no candidate prior because it has one fitted model. These absent entries are excluded
from `REQUIRED_COMPONENTS`; no digest is fabricated for them.

The entries below apply independently to both registered cells.

| Method | Component | Real fitted class and object-graph path | Native fixture / intercept | Fit source; reload and replay | O/T status | Source evidence |
|---|---|---|---|---|---|---|
| RefPlan | `reward_surrogate` | `PublicRewardRiskSurrogate`; `root.method_context.surrogate` | 19; coefficient 0 is bias | Arm O public transitions/rewards; full root + surrogate save/load; `predict` | byte-identical O/T | `public_surrogate.py:51-280` |
| RefPlan | `dynamics_ensemble` | `PublicDynamicsEnsemble`; `root.dynamics` | 35; member coefficient 0 is bias | public observations/actions and belief cache; fresh root reload; `predict` | end-to-end refit by arm | `public_models.py:14-142` |
| RefPlan | `residual_process_scales` | `float` fields; `root.dynamics.members[*].residual_sigma` | non-feature | bootstrap dynamics residuals; exact scalar projection; dynamics replay | end-to-end refit by arm | `public_models.py:27-120` |
| RefPlan | `refplan_behavior_prior` | `CalibratedBehaviorModel`; `root.policy_prior` | 10; augmented column 0 is bias | public-belief features/actions; fresh root reload; `probabilities`/`act` | end-to-end refit by arm | `methods/refplan.py:39-63`; `behavior_model.py:10-56` |
| RefPlan | `planner_configuration` | `PublicParticlePlanner`; `root.planner`/`root.planner_cfg` | non-feature | registered configuration; fresh root reload; `plan_marginalized`/`act` | identical settings | `public_models.py:145-320` |
| OGSRL | `reward_surrogate` | `PublicRewardRiskSurrogate`; `root.surrogate`/`root.method_context.surrogate` | 19; coefficient 0 is bias | Arm O public transitions/rewards; full root + surrogate save/load; `predict` | byte-identical O/T | `public_surrogate.py:51-280` |
| OGSRL | `dynamics_ensemble` | `PublicDynamicsEnsemble`; `root.dynamics` | 35; member coefficient 0 is bias | public observations/actions and belief cache; fresh root reload; `predict` | end-to-end refit by arm | `public_models.py:14-142` |
| OGSRL | `residual_process_scales` | `float` fields; `root.dynamics.members[*].residual_sigma` | non-feature | bootstrap dynamics residuals; exact scalar projection; dynamics replay | end-to-end refit by arm | `public_models.py:27-120` |
| OGSRL | `ogsrl_actor` | `numpy.ndarray`; `root.actor_weights` | 2; actor column 0 is bias | model rollouts from public-belief cache; exact ndarray reconstruction; `_public_policy`/`act` | end-to-end refit by arm | `methods/ogsrl.py:407-420,616-682` |
| OGSRL | `ogsrl_guardian` | `PublicKNNGuardian`; `root.guardian` | 13; no intercept | public observations/actions; full guardian reload; `score`/`ood_probability`/`act` | end-to-end refit by arm | `methods/ogsrl.py:179-239,758-794` |
| OGSRL | `ogsrl_safety_calibration` | scalar policy fields; `root.s_low`/`root.safety_budget`/`root.guardian.threshold` | non-feature | public data and fitted guardian/dynamics; scalar projection; risk/`act` replay | rebuilt by arm | `methods/ogsrl.py:551-590,758-794` |
| BA-MCTS | `reward_surrogate` | `PublicRewardRiskSurrogate`; `root.method_context.surrogate` | 19; coefficient 0 is bias | Arm O public transitions/rewards; full root + surrogate save/load; `predict` | byte-identical O/T | `public_surrogate.py:51-280` |
| BA-MCTS | `dynamics_ensemble` | `PublicDynamicsEnsemble`; `root.dynamics` | 35; member coefficient 0 is bias | public observations/actions and belief cache; fresh root reload; `predict` | end-to-end refit by arm | `public_models.py:14-142` |
| BA-MCTS | `residual_process_scales` | `float` fields; `root.dynamics.members[*].residual_sigma` | non-feature | bootstrap dynamics residuals; exact scalar projection; dynamics replay | end-to-end refit by arm | `public_models.py:27-120` |
| BA-MCTS | `bamcts_model_bank` | `PublicDynamicsEnsemble`; `root.dynamics` logical alias | 35; member coefficient 0 is bias | same real dynamics object; canonical second logical view; simulation/`act` | end-to-end refit by arm | `methods/bamcts.py:46-67,134-183` |
| BA-MCTS | `bamcts_search_configuration` | scalar policy fields; `root.simulations`/`root.depth`/`root.exploration` | non-feature | registered configuration; scalar projection; `act` | identical settings | `methods/bamcts.py:26-67,237-296` |
| EVD | `evd_behavior_reference` | `CalibratedBehaviorModel`; `root.behavior_model` | 10; augmented column 0 is bias | public-belief features/actions; fresh root reload; `probabilities` | end-to-end refit by arm | `methods/ensemble_value_disagreement.py:132-177` |
| EVD | `evd_q_members` | `BootstrapQMember`; `root.q_members[*]` | 10; augmented column 0 is bias | episode bootstraps of public features/actions/raw rewards; fresh root reload; `values`/`act` | end-to-end refit by arm | `methods/ensemble_value_disagreement.py:26-177` |
| EVD | `evd_policy_configuration` | scalar policy fields; `root.disagreement_penalty`/`root.cql_alpha`/`root.fit_iterations` | non-feature | registered configuration; scalar projection; `act` | settings identical; Q fits differ | `methods/ensemble_value_disagreement.py:39-63,179-194` |
| PLUS | `ricker_fit_cache` | `MechanisticModel`; `root.candidate_bank.fits[*].model` | non-feature | ordered public survey/action histories, Ricker form; cache + root reload; transition/likelihood | byte-identical O/T | `faithful_fit.py:890-961` |
| PLUS | `residual_process_scales` | `float`; `root.candidate_bank.fits[*].model.process_scale` | non-feature | ordered public histories; exact scalar projection; POMDP transition | byte-identical O/T | `faithful_artifacts.py:37-119` |
| PLUS | `reward_surrogate` | `PublicRewardRiskSurrogate`; `root.method_context.surrogate` | 19; coefficient 0 is bias | Arm O public transitions/rewards; root + surrogate reload; expected public reward/`predict` | byte-identical O/T | `faithful_pomdp.py:209-229` |
| PLUS | `pbvi_policy` | `PointBasedPlanner`; `root.planners[*]` | non-feature | registered configuration and fitted Ricker POMDP; fresh root reload; action values/`act` | byte-identical O/T | `planners/pbvi.py:17-120` |
| PLUS | `pbvi_grids` | `CandidatePOMDP`; `root.pomdps[*]` grids | non-feature | fitted model and registered discretization; fit-artifact arrays; transition/observation | byte-identical O/T | `faithful_pomdp.py:41-66`; `faithful_artifacts.py:181-229` |
| PLUS | `pbvi_candidates` | `CandidateBank`; `root.candidate_bank.fits[*]` | non-feature | ordered public histories; complete root + fit artifacts; candidate/planner iteration | byte-identical O/T | `methods/plus_faithful.py:54-109` |
| PLUS | `pbvi_prior` | `numpy.ndarray`; `root.candidate_bank.initial_weights` | non-feature | registered uniform candidate prior; candidate-bank + root reload; posterior initialization | byte-identical O/T | `methods/plus_faithful.py:88-115` |
| MOOR | `ricker_fit_cache` | `MechanisticModel`; `root.fit_result.model` | non-feature | ordered public survey/action histories, Ricker form; cache + root reload; transition/likelihood | byte-identical O/T | `faithful_fit.py:890-961` |
| MOOR | `residual_process_scales` | `float`; `root.fit_result.model.process_scale` | non-feature | ordered public histories; exact scalar projection; POMDP transition | byte-identical O/T | `faithful_artifacts.py:37-119` |
| MOOR | `reward_surrogate` | `PublicRewardRiskSurrogate`; `root.method_context.surrogate` | 19; coefficient 0 is bias | Arm O public transitions/rewards; root + surrogate reload; expected public reward/`predict` | byte-identical O/T | `faithful_pomdp.py:209-229` |
| MOOR | `pbvi_policy` | `PointBasedPlanner`; `root.planner` | non-feature | registered configuration and fitted Ricker POMDP; fresh root reload; action values/`act` | byte-identical O/T | `planners/pbvi.py:17-120` |
| MOOR | `pbvi_grids` | `CandidatePOMDP`; `root.pomdp` grids | non-feature | fitted model and registered discretization; fit-artifact arrays; transition/observation | byte-identical O/T | `faithful_pomdp.py:41-66`; `faithful_artifacts.py:181-229` |
| MOOR | `pbvi_candidates` | `FitResult`; `root.fit_result` | non-feature | ordered public histories; complete singleton fit/root; planner use | byte-identical O/T | `methods/moor_faithful.py:36-79` |

Source paths in the table are relative to `src/tracks/general/real_ecology_benchmark/` or
`src/tracks/ecological/real_ecology_benchmark/`, as appropriate. The cell-specific probe
receipts preserve full feature names, exact classes, paths, fit sources, replay operations,
source lines, input hashes, serialized-root hashes, and parity hashes.

## V3 fixture contract

The authoritative domain is `REQUIRED_COMPONENTS[method] & FEATURE_COMPONENTS`. Each domain
member has exactly one three-row, two-dimensional, finite `float64` fixture. Keys must match
the domain exactly; non-feature components cannot accept fixtures.

For each `(component, row, column, feature_name)`, the generator hashes the UTF-8 bytes
`corrected-stageb-component-fixture-v1`, component name, decimal row, decimal column, and
exact serialized feature name separated by NUL bytes. It interprets the first eight SHA-256
bytes as unsigned big-endian, reduces modulo 2,000,001, subtracts 1,000,000, and divides by
65,536. The validator recomputes the matrix and compares C-order little-endian float64 bytes
exactly. A separately hashed canonical fixture manifest binds the generator recipe, row count,
feature order, shape, dtype, and each fixture-byte SHA-256.

The receipt schema is `corrected_stageb_artifact_bundle_evidence_v3`. V2 receipts do not
enter the V3 parser, and the V2 parser rejects V3. Component hashes still determine bundle
identity; fixture-manifest identity is separate. Real fitted-object graph serialization,
fresh reload/replay, cross-references, matched Arm-O surrogate rules, ecological shared-policy
rules, and information boundaries are unchanged.
