# Methods and planning objectives

There are fitted models and planners, not policy/critic neural networks.
`methods/__init__.py:METHODS` is the runtime registry.

## Registered methods

| key / class | Offline fit and deployed selector | objective / reward / uncertainty | accepted |
|---|---|---|---|
| `mopo` / `MOPOPolicy` | bootstrap public/full dynamics; particle MPC | sampled discounted surrogate/mechanistic reward minus configured pessimism×model disagreement | no |
| `refplan` / `RefPlanPolicy` | public dynamics ensemble + behavior prior; posterior-marginalized particle planner | `E_b[G]-pessimism*sqrt(Var_b[G])`; surrogate; posterior updated from transitions | yes, general |
| `bamcts` / `BAMCTSPolicy` | public ensemble; Bayes-adaptive MCTS at every act | UCT estimates discounted surrogate reward minus model disagreement; transition-likelihood model belief | yes, general |
| `ensemble_value_disagreement_pessimism` / `EnsembleValueDisagreementPolicy` | bootstrap ridge fitted-Q ensemble, 35 Bellman iterations/member | choose `argmax_a mean Q - penalty*std(Q)`; logged rewards | yes, general |
| `delphic` | alias to EVD in general | same as above; no latent-confounder model | no accepted alias |
| `ogsrl` / `OGSRLPolicy` | public ensemble, kNN OOD guardian, constrained linear softmax actor | actor objective `G_R-lambda_safety*G_cost-lambda_ood*G_ood`; surrogate reward; 25-step normalized low-abundance cost | yes, general |
| `moor` / `MOORPolicy` | one Ricker fit/proposal | MPC or fitted mechanistic Q; mechanistic/full or surrogate/hidden; no family posterior | no |
| `plus` / `PLUSPolicy` | four public form fits (`PUBLIC_FORM_CANDIDATES`) | candidate posterior and aggregated plan scores; surrogate in hidden | no |
| `moor_native` / `MOORNativePolicy` | one fitted public Ricker grid/VI | belief-weighted tabular Q, surrogate reward table | no |
| `plus_native` / `PLUSNativePolicy` | four fitted form grids/VI | posterior combination of candidate Q | no |
| `moor_adapted_ricker_misspec_pbvi` / `MOORFaithfulRickerPBVIPolicy` | one LBFGS mechanistic Ricker fit | finite-horizon PBVI expected public surrogate reward; one misspecified family | yes, ecological |
| `plus_adapted_mechanistic_pbvi` / `PLUSFaithfulPBVIPolicy` | mechanistic bootstrap candidate bank | posterior-weighted PBVI candidate scores | no under this key |
| ecological-only `plus_adapted_ricker_only_pbvi` | 8 Ricker bootstrap MAP candidates | posterior-weighted PBVI; within-Ricker parameter uncertainty, not four-family uncertainty | yes, ecological |

Relevant hyperparameters are defined in `config.ModelConfig`,
`PlannerConfig`, and `FaithfulConfig`, and accepted values are in the manifests:
dynamics-ensemble size 5 and ridge 1e-3 apply to RefPlan/MOPO/BA-MCTS through
`ModelConfig`. EVD's Q ensemble is not governed by `ModelConfig`:
`EnsembleValueDisagreementPolicy.__init__()` fixes constructor defaults
`ensemble_size=20`, `cql_alpha=0.5`, `disagreement_penalty=0.1`, and
`fit_iterations=35`; `pipeline.build_method()` passes only `seed`, so accepted
EVD used 20 bootstrap members. Other accepted values are MPC horizon 5, 96
sequences, 32 particles, discount .95, pessimism .5; BA-MCTS depth
8/simulations 256; OGSRL horizon 25 and 256 deployment rollouts; faithful 8
starts×100 LBFGS iterations, 16 MC paths, and PBVI horizon 5 with 41
state/observation bins, 9 capacity bins, 32 belief points and 7 observation
branches.

## What survives fit

EVD deploys its Q-member weights; RefPlan/MOPO/BA-MCTS deploy public dynamics
ensembles; OGSRL deploys actor, ensemble, guardian and safety scale; native
methods deploy transition/reward/Q tables; faithful methods deploy candidate
models, internal beliefs and PBVI planners. Intermediate regression design
matrices, LBFGS raw parameters/traces, bootstrap row selections, and diagnostic
holdout predictions are discarded or serialized only as fit diagnostics.

## PyTorch fitting

Only `faithful_fit.py:fit_mechanistic_model()` uses PyTorch gradients.
`_trajectory_objective()` simulates ordered public episodes using a fixed random
bank and minimizes mean squared error between normalized survey targets and
the model’s conditional survey mean; optional hierarchical penalties shrink
free action rates/capacities toward their group means. Parameters are
reparameterized by sigmoid/tanh into positive scales, bounded signed rates,
capacity increments, stocking and family thresholds. `_symmetric_positive()`
defines a custom autograd clamp with derivative 1 for positive, 0 for negative,
and 0.5 at zero, allowing the signed-rate split into growth/mortality.

`objective.backward()` is called inside the LBFGS closure, which is executed by
`optimizer.step(closure)` in
`src/tracks/general/real_ecology_benchmark/faithful_fit.py:fit_mechanistic_model()`.
`detach()` occurs
only when recording traces/gradients, converting the selected fitted model, and
computing curvature diagnostics after optimization; no detach is on the
objective path (`faithful_fit.py` lines 630–656). Gradient flow is therefore
present as intended. This statement does not prove numerical identifiability;
run `tests/*/real/test_faithful_equations.py` and inspect gradient norms.

## Family uncertainty versus accepted PLUS

`native_fit.py:PUBLIC_FORM_CANDIDATES` is exactly
`("ricker","allee","theta","regime")` and legacy/native PLUS fits one public
form per family. General faithful PLUS is also configured for four forms.
However, the accepted table uses ecological
`plus_adapted_ricker_only_pbvi`, whose manifest states
`candidate_family=ricker`, `candidate_allocation=ricker:8`,
`candidate_count=8`, `candidate_construction=ricker_only_episode_bootstrap_map_1full_7bootstrap_v1`,
and `prior=uniform` in all 24 rows. It maintains a posterior over eight Ricker
bootstrap-MAP parameter candidates (one full fit plus seven bootstrap fits),
**not** uncertainty over the four true families. This distinction is visible in
`experiments/accepted_p10/manifests/plus_p10_plan_24.csv`.

Methods registered but absent from accepted rows include MOPO, legacy
PLUS/MOOR, both native methods, four-form faithful PLUS, and the `delphic`
alias. Timing is cell-specific in external `summary.json` (`fit_seconds`,
`row_seconds`); the accepted CSV omits these fields, so a complete compute-cost
table cannot be derived from the copied headline artifact alone.
