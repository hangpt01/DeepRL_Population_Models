# Phase 2D authorization: final corrections before the general-RL canary

## Controlling documents

Read and apply, in this order:

1. `REVIEW_PHASE2B_GENERAL_RL_REMAINING_ISSUES_PLAN.md`
2. `AUDIT_PHASE2C_AGAINST_LATEST_REVIEW.md`
3. `PHASE2C_GENERAL_RL_FINAL_IMPLEMENTATION_REPORT.md`

The audit correctly concludes that the four-method canary remains NO-GO. This file resolves the
remaining scientific choices and authorizes implementation in the isolated worktree only.

## Settled decision: choose Delphic Path B

Do **not** attempt a rushed latent-generative Delphic reproduction in this phase. The estimated
4–8 engineering days plus numerical validation are outside the present experiment scope.

Replace the fourth method with the exact reader-facing scientific contract:

**Ensemble value-disagreement pessimism (Delphic-motivated)**

This is an idea-level proposal ingredient, not a Delphic baseline, not a compatible-world method,
and not an estimator of validated Delphic uncertainty.

### Required fourth-method implementation

Remove the direct random Q perturbation

`q_scale * unsupported * ambiguity * perturbation_w`

and remove the construction whose purpose is to engineer a large counterfactual/observed variance
ratio.

Instead:

1. Construct an ensemble of independently fitted conservative Q estimators using episode-level
   bootstrap resamples of the same 4,000-transition training split.
2. Use the same public features, public reward, action set, discount, and registered fitted-Q/CQL
   objective for every member.
3. Give each member only its bootstrap sample and deterministic registered seed; do not inject
   random values directly into its output Q.
4. Define disagreement as empirical variance across the independently fitted Q estimates.
5. Define the pessimistic action score as the ensemble mean Q minus the registered disagreement
   penalty. Keep the existing coefficient only if its units remain coherent after removing the
   direct perturbation; otherwise return for a scale decision rather than tuning from returns.
6. Any OOD/support diagnostic may be reported separately, but it must not multiply random Q
   perturbations or serve as a gate designed to force a variance ratio.
7. The calibrated public behavior model may remain as a descriptive behavior diagnostic. It must
   not be called a compatible-world gate and must not determine whether Q members are "worlds."
8. Remove G-D3/G-D4 as Delphic validity gates. Observed/counterfactual variance may be reported as
   an exploratory heuristic diagnostic only, with no target ratio and no pass/fail claim.

### Naming and compatibility

Prefer a new canonical internal ID such as:

`ensemble_value_disagreement_pessimism`

The old internal `delphic` ID may remain only as an explicitly deprecated alias if required to read
historical artifacts. The new unsubmitted manifest should use the new canonical ID. Do not mutate
old frozen manifests or result snapshots.

Rename operative code objects and fields so current artifacts do not claim Delphic fidelity:

- `SupportGatedWorld` -> a neutral fitted-Q ensemble-member name;
- `DelphicCQLPolicy` -> a neutral ensemble-disagreement policy name;
- `delphic_lambda` -> `disagreement_penalty` or another unit-clear name;
- `delphic_uncertainty*` -> `q_ensemble_disagreement*`;
- compatible-world/survivor/gate fields -> behavior-calibration or ensemble diagnostics as
  appropriate.

Update current README, benchmark documentation, probes, tests, registration, and prepared snapshot.
Do not rewrite immutable historical snapshots. Mark discoverable historical positive claims as
superseded where appropriate.

The report must state that Delphic motivated the use of cross-model value disagreement and
pessimism, while the implemented method does not reproduce Delphic's latent compatible-world
construction.

## OGSRL: unit-consistent public safety cost

Retain:

`s_low = Q_0.20(positive training observations)`

`c(o') = clip((s_low-o')/s_low, 0, 1)`.

This remains a public low-abundance proxy with no guarantee for the private safety objective.

### Registered cost functional

Use one normalized discounted occupancy helper:

`C_H(c_1:H) = ((1-gamma)/(1-gamma^H)) * sum_{t=0}^{H-1} gamma^t c_{t+1}`.

Properties that must be tested:

- all-zero costs produce 0;
- all-one costs produce 1 for every positive H;
- output remains in `[0,1]`;
- exact hand-calculated examples match;
- no division instability occurs when gamma is numerically near 1.

### Common constraint horizon

Register `H_cost = 25`, matching the complete offline training episodes. Use the same risk horizon
for:

- estimating the behavior-policy budget;
- hidden OGSRL actor/dual training;
- action-risk comparison at deployment through receding-horizon public-model rollouts.

At a deployment state with fewer than 25 episode steps remaining, use the actual remaining positive
horizon and the corresponding normalization. State clearly that this is a registered 25-step
receding public-risk constraint inside the benchmark's 50-step evaluation episode.

Do not compare one-step risk, six-step unnormalized return, and 25-step behavior cost against the
same threshold.

### Budget

For each cell, compute `C_25` for every complete training behavior episode and set:

`safety_budget = arithmetic mean_i C_25_i`.

Remove the scientific `[0.02,0.10]` floor/cap. A redundant final numerical clamp to `[0,1]` is
allowed only if tests prove it does not alter the computed value. Record episode count, standard
deviation, standard error, and a descriptive confidence interval; do not alter the budget using the
interval.

The previously measured reference values are approximately 0.0982 for Amur and 0.0546 for Egyptian
vulture. Treat them as verification targets for the same frozen public data, not performance-tuned
thresholds.

Recheck safety-dual movement and policy response using only constructed mechanism cases and
return-blind canaries. Failure to bind must be reported, not repaired by threshold tuning.

## RefPlan: history-conditioned trajectory prior

Retain the distinct calibrated public policy prior, epsilon `0.10`, separate model posterior,
posterior-marginalized scoring, common random numbers, and proposal-only use of the prior.

Change candidate generation so every continuation action uses

`pi_prior(a_t | simulated public history/state_t)`

rather than reusing the root-state action distribution across all depths.

Use one posterior-predictive public trajectory per candidate for proposal generation:

1. start from the current public belief representation;
2. evaluate the epsilon-mixed policy prior and sample the action at the current depth;
3. propagate the proposal state using the posterior-weighted predictive mean of the common public
   dynamics ensemble with registered common random numbers;
4. update the public proposal feature/history;
5. re-evaluate the prior and sample the next action;
6. repeat for the full planning horizon;
7. score the resulting fixed candidate sequence under every ensemble member using the existing
   posterior-marginalized reflected score.

Do not add an extra log-prior score. Preserve explicit first-action coverage rows.

Add tests with a state-dependent prior proving that later action distributions and candidate
sequences change when simulated histories differ while the root prior is identical.

## Privacy: complete lifecycle intervention test

Extend private-field intervention invariance across the complete hidden-method lifecycle:

1. `fit`;
2. initial `act`;
3. `observe(action, public_observation)`;
4. subsequent `act`;
5. comparison of complete fitted public artifacts, posterior state, diagnostics, and actions.

Apply different sentinels one private field at a time at:

- the high-level environment object;
- lower-level private table providers;
- global/config objects reachable during `fit`, `observe`, or `act`;
- data-directory and action-channel provider paths.

Do not merely compare a bounded scalar digest. Use deterministic complete serialization/hashing for
fitted numeric artifacts and lifecycle state, excluding only explicitly enumerated public runtime
metadata. Document every exclusion. Retain the existing blocked-private-API and relabel tests.

No current leak was found; this is verification hardening.

## Work and launch constraints

- Work only in `/fs04/scratch2/ce25/general_rl_phase2_iso`.
- Do not merge into the dirty main worktree.
- Do not modify or interfere with PLUS/MOOR jobs or artifacts.
- Do not inspect performance returns.
- Do not launch the four-method canary or full experiment.
- Do not tune thresholds or coefficients from policy performance.
- Keep matched 4,000-transition data, shared ensemble size 5, BA-MCTS 256/depth 8, one thread, and
  one allocated CPU.
- Preserve old frozen artifacts as immutable history.

## Verification gates

Before returning:

1. no direct support/ambiguity/random perturbation is added to Q;
2. the fourth method's ensemble members differ only through registered bootstrap data/seeds and
   fitted optimization;
3. no current operative file calls the fourth method Delphic-CQL, compatible worlds, or Delphic
   uncertainty;
4. OGSRL uses normalized `C_H` with `H_cost=25` consistently in budget fitting, actor constraints,
   and receding-horizon action-risk evaluation;
5. registered budgets equal train-only episode means and are not floor/cap selected;
6. OGSRL safety dual and action choice respond in constructed low-abundance tests;
7. RefPlan re-evaluates its prior at simulated state/history at every planning depth;
8. full lifecycle private intervention invariance passes all four methods;
9. thread parity remains deterministic;
10. full test suite passes;
11. refreshed snapshot, registration, manifest, and hashes use the corrected fourth-method ID and
    scientific label;
12. no jobs or returns were touched.

## Required return

Write `PHASE2D_FINAL_GENERAL_RL_CORRECTION_REPORT.md` containing:

1. exact files/functions changed;
2. fourth-method mathematical definition and naming cleanup;
3. OGSRL cost/budget equations and measured return-blind budgets;
4. RefPlan sequential prior algorithm;
5. privacy lifecycle test design and evidence;
6. all tests and results;
7. runtime changes versus Phase 2C;
8. refreshed snapshot/registration/manifest hashes;
9. remaining paper deviations and claims allowed;
10. GO/NO-GO recommendation for a limited corrected canary.

Stop after the report. A GO recommendation does not authorize submission.
