# Audit of Phase 2C Against the Latest Phase 2B Review

## Scope and verdict

This is a read-only reconciliation of the implemented Phase 2C state in the
isolated worktree against the later
`REVIEW_PHASE2B_GENERAL_RL_REMAINING_ISSUES_PLAN.md`. It does not treat the
Phase 2C implementation report as an approval. No code, configuration,
manifest, frozen snapshot, or run output was changed; this audit is the only
new file. No job was launched, no scheduler state or performance return was
inspected, and PLUS/MOOR were not touched.

**Revised verdict: NO-GO for a four-method canary.** BA-MCTS and the thread and
resource-accounting changes may remain. The current Delphic-motivated method
implements the direct support-gated Q perturbation explicitly rejected by F1;
OGSRL compares costs with three different horizon/unit conventions; RefPlan
uses one root-state prior distribution for all sequence positions rather than
re-evaluating the prior at simulated histories; and the privacy intervention
test does not exercise `observe` or lower-level global/config interventions.

## 1. Delphic construction

### 1.1 What the code constructs

World-specific disagreement is injected directly into Q. It is not induced by
complete fitted generative worlds.

The common base is fitted once by `_fit_shared_q` in
`src/real_ecology_benchmark/methods/delphic.py:211-224`. It is a public-feature
linear fitted-Q model shared byte-for-byte by every candidate. `_fit_world`
fits a world-specific behavior logistic head and creates a random action
perturbation, but fits no world-specific transition kernel or reward model
(`delphic.py:226-272`). The actual value function is

```text
Q_w(x,a) = Q_shared(x,a)
           + q_scale * unsupported(x,a)
                     * ambiguity(x) / ambiguity_scale
                     * perturbation_w(a),
```

implemented at `delphic.py:150-159`. `counterfactual_q` merely returns that
quantity (`delphic.py:161-168`). Thus behavior likelihood does not derive Q,
and no fitted world dynamics/reward can be rolled forward to derive Q. The
construction is the same mechanism criticized in latest-review F1: support is
small on logged state-action support and larger off support, so the design
mechanically suppresses observed-action variance and increases unlogged-action
variance. G-D3/G-D4 measure a property deliberately put into Q; they do not
establish observationally compatible causal worlds.

No further threshold or perturbation tuning should be used to make the 3:1
ratio pass. The healthy-cell miss is not a tuning target.

### 1.2 Remaining incompatible terminology and claims

The exact reader label at `delphic.py:173` is the approved downgrade text apart
from capitalization: `ensemble value-disagreement pessimism
(Delphic-motivated)`. Several operative claims and interfaces contradict it.

| Location | Remaining claim or misleading name |
|---|---|
| `README.md:33,84` | Calls the implementation Delphic-CQL and says Delphic uncertainty is variation over compatible latent worlds. |
| `src/real_ecology_benchmark/methods/delphic.py:115-138,171-186` | Names the objects `SupportGatedWorld`, the policy `DelphicCQLPolicy`, and its coefficient `delphic_lambda`. |
| `delphic.py:124-126,188-194,351-354` | Says candidates reproduce the observed action distribution for a Delphic observational-compatibility gate and calls survivors the compatible subset. |
| `delphic.py:294-316,361-386,401-416,419-435` | Names the manufactured variance `mean_delphic_uncertainty`, `delphic_uncertainty`, and `delphic_uncertainty_all`; the fit error also says `Delphic-CQL`. |
| `src/real_ecology_benchmark/delphic_compat.py:1-12` | Describes a Delphic compatibility gate over worlds and frames G-D3/G-D4 as an approximation to the paper mechanism. |
| `delphic_compat.py:149-220` | Uses `CompatibilityReport`, `compatible_mask`, `surviving_indices`, `gate_worlds`, and “candidate worlds”; diagnostic keys include `delphic_compatible_worlds`, `delphic_world_diversity`, and `delphic_compat_passes`. |
| `scripts/general_adequacy_probe.py:1-7,195-215` | Describes observational compatibility/world diversity and emits `compatible_worlds` and `compat_passes`. |
| `tests/real/test_general_paper_mechanisms.py:11-13,431-512` | Tests and comments explicitly call candidates observationally compatible worlds and the variance Delphic uncertainty. |
| `docs/benchmark/04_algorithm_adaptations_and_claims.tex:209-242` | Correctly disclaims validated Delphic uncertainty at the end, but still calls candidates random-feature latent worlds and the term in Q a “Delphic penalty.” |
| `docs/benchmark/SERVER_AUDIT_hidden_parameter_usage_by_method.md:131,243-250` and `.tex:276-279,354-359` | Call the inputs “compatible-world features.” These documents also describe an older feature path and are stale for hidden mode. |
| `docs/fix_implement_general_RL/PHASE2C_GENERAL_RL_FINAL_IMPLEMENTATION_REPORT.md:18,27-73` | Uses support-gated/world terminology and incorrectly states that no compatible-worlds claim remains. Its explicit disclaimer at lines 67-70 is correct, but does not cure the other claims. |

The prepared, unsubmitted snapshot repeats the same operative terms in
`real_ecology_runs/general_corrected_prepared/code/`: its `methods/delphic.py`,
`delphic_compat.py`, `scripts/general_adequacy_probe.py`, and
`tests/test_general_paper_mechanisms.py` contain the corresponding class,
comment, test, and diagnostic names. The snapshot must not be refreshed during
this audit, but it would have to be refreshed after an approved correction and
before any submission.

Historical material also contains positive compatible-world/Delphic-uncertainty
claims. These are archival rather than the Phase 2C reader contract, but must be
clearly marked superseded if they remain discoverable:

- `docs/history/methods.md`;
- `docs/history/22_6_Continuous_Observation_New_Baselines.tex`;
- `docs/history/22_6_Continuous_Observation_Experiment_Results.tex`;
- `docs/history/29_6_Improve_Continuous_Observations.tex`;
- `docs/history/29_6_tier2_continuous_state_implementation_handoff.md`;
- `docs/history/SPEC_documentation_bundle_2026-07-10.md` and
  `AUDIT_documentation_bundle_plan_2026-07-10.md`;
- `docs/history/codex_claude_conversation_md/claude_audit_22_6_tier2_continuous_implementation.md`,
  `claude_plots_and_results_handoff_for_codex.md`, and
  `claude_review_codex_tier2_continuous_plan.md`;
- `docs/history/planning/codex_stress_pomdp_22_6_tier2_continuous_implementation_plan.md`;
- `docs/real_ecology_history/29_6_algorithm_method_notes.tex`.

`docs/benchmark/04_algorithm_adaptations_and_claims.tex:241-242`,
`docs/real_ecology_history/AUDIT_algorithm_paper_implementation.md`,
`docs/real_ecology_history/29_6_algorithm_audit_reasonable_todo.md`, and
`docs/history/HANDOFF_real_ecology_data_setting.md` mention these phrases to
reject or criticize the claims; those are not residual positive claims. Older
frozen run snapshots at
`real_ecology_runs/general_audit_20260712/code/src/real_ecology_benchmark/methods/delphic.py`,
`real_ecology_runs/motivation_native_20260711/code/src/real_ecology_benchmark/methods/delphic.py`,
and the Delphic source/method-note/audit copies under
`real_ecology_runs/psafe_overnight_20260705/code/real_ecology_cont_obser/` also
preserve older terminology. They should remain immutable historical artifacts
rather than be mistaken for current implementation documentation.

Finally, the existing diagnostic output files preserve the misleading field
name `compatible_worlds`: both Phase 2C adequacy JSON files under
`real_ecology_runs/general_adequacy_phase2c/`, and both older files under
`real_ecology_runs/general_adequacy_canary/` (the older files additionally say
`compat_reason: "compatible and diverse"`). These are return-blind artifacts,
not source documentation, but their schema is another reader-visible residual
claim and must not be used as compatibility evidence.

### 1.3 Honesty under the narrower name

The numerical object is accurately describable as **ensemble
value-disagreement pessimism**: it is an ensemble of deliberately perturbed Q
values whose variance is subtracted from a linear Q estimate. “Delphic-motivated”
is defensible only as an idea-level provenance qualifier, with an explicit
statement that observed-support disagreement is misspecification contamination
and that the candidates are not fitted causal/generative worlds.

The current package is therefore **not yet scientifically honest as a whole,
even under the narrower label**. The label itself is honest, but the code,
diagnostics, tests, README, snapshot, and parts of the documentation still make
world/compatibility/Delphic-uncertainty claims. More importantly, Path B in the
latest review expressly says not to engineer the ratio through support-gated
random Q perturbations, and that exact construction remains active. G-D1/G-D2
can be retained as behavior-head calibration diagnostics, but they cannot
validate the Q candidates; G-D3/G-D4 cannot be presented as evidence of
Delphic validity or as a target to tune toward.

Before a canary, choose one scientific contract:

1. implement coherent latent generative worlds (Path A); or
2. keep a plainly disclosed heuristic disagreement penalty under the exact
   downgraded name, remove the direct ratio-engineering construction and all
   compatible-world/Delphic-uncertainty semantics, and treat any residual
   support/on-support variance measurements only as heuristic diagnostics.

This audit does not recommend trying to make the existing variance-ratio gate
pass.

Phase 2C did fill the formerly undefined numerical constants: G-D3 uses
`0.05 * scale^2` and G-D4 uses a positive floor of `1e-8 * scale^2` plus the
3:1 ratio (`delphic_compat.py:22-27,241-245`). It nevertheless fails latest
review F2's conceptual requirement because `passes` conjoins G-D1 through G-D4.
An observably valid zero-ambiguity candidate set therefore fails the overall
“compatibility” result when `V_cf` is zero, including the intended
`sigma_obs=0` negative control. World/generative validity and detected
ambiguity were not separated. Also, the ratio denominator in
`_value_variances` uses the module-level `DIVERSITY_FLOOR` rather than the
`diversity_floor` argument accepted by `gate_worlds`, leaving two nominally
related configuration paths. Path A must separate fit validity from uncertainty
presence; Path B should not retain these as a Delphic validity gate.

### 1.4 Scope of coherent latent generative worlds

A minimal coherent Path-A implementation is substantial new infrastructure,
not a local change to the taper. Each candidate would need a small latent state
`z` (for example 2--4 discrete states), a public-history behavior model
`pi_w(a_t | h_t,z_t)`, a next-observation model
`p_w(o_{t+1} | h_t,a_t,z_t)`, and either a fitted public reward model or the
shared public reward mapping with that sharing explicitly disclosed. The
latent coupling must affect behavior and outcomes. Candidates must be fitted
by episode-level observed-data marginal likelihood (EM or variational fitting
is sufficient), with held-out behavior, next-observation, and reward/generative
checks. Q must then be obtained only by policy evaluation or rollouts through
each retained candidate's transition/reward model; no direct Q perturbation is
permitted.

Expected code scope is replacement of most of `methods/delphic.py` and
`delphic_compat.py`, likely a new latent-world fitting module and serialized fit
artifacts, updates to the probe/registration interface, and new synthetic
recovery, likelihood, zero-ambiguity, and “no direct Q injection” tests. A
reasonable engineering estimate is roughly 500--1,000 new or replaced lines,
4--8 focused engineering days for a lightweight discrete-latent version, and
about 1--2 weeks including numerical validation and review. Compute would grow
from linear fits to multiple EM/variational starts plus per-world policy
evaluation; depending on retained worlds and rollout budget, a several-fold to
order-of-magnitude per-cell increase is plausible. These are design estimates,
not measurements from a job.

## 2. OGSRL cost units

### 2.1 Exact current functional

The approved per-transition public proxy is implemented exactly as

```text
s_low = empirical Q_0.20 of positive training observations
c(o') = clip((s_low - o') / s_low, 0, 1).
```

It remains a public low-abundance proxy with no guarantee for the private
safety objective. The rest of the current implementation uses three different
functionals:

1. **Behavior budget fit (H=25):** with `gamma=0.95`, for each complete
   25-transition training episode `i`,

   ```text
   B_i_raw = sum_{t=0}^{24} 0.95^t c(o_{i,t+1}),
   B_raw   = mean_i B_i_raw,
   B       = clip(B_raw, 0.02, 0.10).
   ```

   There is no division by 25 and no discounted-sum normalization
   (`methods/ogsrl.py:389-407`; configuration at `config.py:526,536-540`).

2. **Actor training (H=6):** `_public_rollouts` creates six costs by default.
   `_discounted_returns` computes, at rollout position `t`,

   ```text
   S_t = sum_{k=t}^{5} 0.95^(k-t) c(o_{k+1}).
   ```

   The actor objective uses these unnormalized return-to-go values, and the
   safety dual update compares `mean(S_0)` directly with `B` (`ogsrl.py:314-339,
   418-486`).

3. **Deployment (effective H=1):** for each action, the code obtains the public
   ensemble's mean next observation for every belief particle, applies `c` to
   that mean, belief-averages the resulting one-step costs, and compares that
   number in `[0,1]` with the same `B` (`ogsrl.py:533-550,716-752`). It does not
   compute a six-, 25-, or 50-step discounted occupancy. Gamma is irrelevant at
   this step. In particular, the registered environment horizon of 50 is not
   represented in the deployment safety constraint.

Thus Phase 2C's statement that training and deployment use “the same cost” is
true only at the pointwise `c(o')` level, not at the constrained-functional
level.

### 2.2 Unclipped behavior budgets

Recomputing only the deterministic public canary datasets and registered train
split (seed 116; split seed 117; 128 complete training episodes; no policy
returns) gives:

| Cell | `s_low` | Current `B_raw` | Episode-mean SE | Normalized `E[C_25]` | Normalized SE |
|---|---:|---:|---:|---:|---:|
| Amur tiger, ricker, sigma 0.2, n=4000 | 36.1525244506 | 1.4186959443 | 0.2135951454 | 0.0981646466 | 0.0147794121 |
| Egyptian vulture, ricker, sigma 0.2, n=4000 | 19.3053700356 | 0.7887461108 | 0.1172365614 | 0.0545761645 | 0.0081120170 |

Both registered budgets became `0.10` solely because both unnormalized raw
means exceed the configured cap `0.10`. The `0.02` floor did nothing. This also
explains why “the dual moved” does not establish a coherent constraint: the
dual was trained against a cap expressed on neither the H=25 raw-sum scale nor
an explicitly defined normalized H=6 scale.

### 2.3 Comparison with the latest review and exact correction

The latest review defines

```text
C_H = ((1-gamma)/(1-gamma^H))
      * sum_{t=0}^{H-1} gamma^t c(o_{t+1}).
```

For `c in [0,1]`, this is in `[0,1]` for every H. The current code omits the
factor `(1-gamma)/(1-gamma^H)` for budget fitting and actor training, and uses
only a one-step cost at deployment. It therefore does not implement the
reviewed functional.

The unit ambiguity can be removed without performance-based tuning by making
these exact changes:

1. Add one tested helper for normalized discounted occupancy, parameterized by
   the actual `gamma` and number of available costs `H`; define behavior-budget,
   actor-training, validation, diagnostic, and deployment costs through it.
2. In `_fit_public_safety_scale`, compute each complete train episode's `C_25`
   and set `safety_budget` to the arithmetic mean. Remove
   `ogsrl_budget_floor`/`ogsrl_budget_cap` and the `np.clip(..., 0.02, 0.10)`
   policy rule. Only a mathematically redundant `[0,1]` numerical guard may
   remain, if documented and tested not to alter measured budgets.
3. Record the episode count, sample standard deviation, standard error, and a
   preregistered interval for the behavior estimate. The two values above are
   the return-blind reference calculation, not thresholds.
4. Normalize the safety return-to-go in `_rollout_training_metrics`. With the
   existing actor rollout H=6, use `C_6` for the initial dual statistic and the
   corresponding normalized remaining-horizon quantity at later policy-gradient
   positions. Keep rewards in their existing units. State explicitly that H=6
   is a model-training approximation being compared with a behavior budget
   fitted at H=25; normalization makes the numerical range common but does not
   assert identical horizon distributions.
5. Replace `_public_belief_action_risks`' one-step calculation with a public-model
   estimate of `C_H` for each forced first action, followed by the learned actor,
   over the registered remaining deployment horizon (50 at reset, decreasing
   with `belief.timestep`). Average the normalized occupancy over belief/model
   draws and compare it with the same train-behavior budget. If a one-step guard
   is retained for operational reasons, it must have a separately named and
   separately justified immediate-cost threshold; it cannot reuse the OGSRL
   occupancy budget.
6. Update diagnostics, registration text, the report, and tests to state all
   three horizons and normalization. Add all-zero/all-one invariants, exact
   hand-calculated H tests, equality of the registered budget to the train-only
   `mean(C_25)`, and a constructed dual-binding test. Re-establish dual movement
   with return-blind mechanism canaries only after the unit correction; do not
   tune the budget by policy return.

## 3. RefPlan

The public policy prior and model posterior are separate fitted objects.
`RefPlanPolicy.fit` fits `policy_prior` with its own weights/scaler and stores
the dynamics-member posterior independently (`methods/refplan.py:43-63`).
`observe` updates only the dynamics posterior (`refplan.py:140-161`). The prior
enters proposal sampling, with no explicit log-prior score, while posterior
weights enter member-value marginalization. This separation satisfies that
part of F4.

The prior is not limited to the first array column: `_sequences` samples the
entire `[sequence_count, horizon]` array using `prior_probabilities`
(`public_models.py:139-157`). However, `plan_marginalized` evaluates
`policy_prior` exactly once on the root belief and passes that one fixed action
distribution to every column (`public_models.py:243-251`). Simulated states are
generated only later during scoring. Consequently:

- every continuation position is sampled from a prior-controlled distribution,
  not from uniform;
- it is the **same root-state distribution at every position**, not
  `pi_prior(a_t | simulated public history/state)`;
- the first action in each of the explicit action-coverage rows is overwritten
  by enumeration, as intended for coverage.

Therefore the Phase 2C report's broad statement that the prior controls
continuations is literally true but does not satisfy the latest review's
history-conditioned full-trajectory requirement. Candidate generation must be
made sequential: roll each candidate/particle forward with common random
numbers, evaluate the epsilon-mixed public prior at the simulated state at each
depth, and sample that depth's action before advancing. The prior should remain
proposal-only. Tests must vary a state-dependent prior and verify effects on
later actions, not merely swap a root prior and observe a changed final plan.

## 4. Privacy

`test_private_value_intervention_invariance` changes the listed private
environment fields, rebuilds `MethodContext`, holds public dataset/cache/
surrogate fixed, refits each method, resets it, and compares one `act` result.
It therefore covers the context boundary, current fit outputs as represented by
its scalar digest, and a single root action.

It does **not** establish intervention invariance for every private value
accessible indirectly during `fit`, `observe`, and `act`:

- it never calls `observe`, so RefPlan and BA-MCTS posterior updates and a
  subsequent action are outside the intervention test;
- it changes the high-level environment object but does not place sentinel
  private values in lower-level globals/configuration during the lifecycle;
- `_hidden_method_context` reads `cfg.environment.data_dir`,
  `realdata.DATA_DIR`, and `realdata.public_action_channels(...)`
  (`pipeline.py:212-234`). The private-field intervention does not vary or
  block that path;
- `MODEL_CFG` and `PLAN_CFG` are shared unchanged, and `model_cfg`,
  `planner_cfg`, and `env_cfg` are expressly excluded from the recursive scalar
  scan (`test_general_privacy.py:36-45`). They are public today, but the test
  would not catch future private data smuggled through one of those objects;
- the fitted digest is a bounded scalar traversal (depth 6, first 5,000 array
  entries), not byte serialization of the complete fitted object graph, so the
  report's “fitted numeric artifacts are byte-identical” claim is broader than
  the test.

The complementary `test_all_hidden_methods_fit_without_table_or_exact_native_access`
does patch `realdata.pops_for`, `effects_for`, `actions_for`,
`actions.resolve_actions`, the MOOR alias, and `NativeSolver.build` while it
fits and acts (`tests/real/test_hidden_rk.py:114-173`). That is valuable, and
code inspection finds no current hidden branch in the four methods that reads
private `env_cfg` fields. It still does not call `observe`, patch
`realdata.public_action_channels`/`DATA_DIR`, inject a lower-level config
sentinel, or automatically fail on a future, differently named global lookup.

The required correction is a lifecycle intervention test that constructs two
identical public inputs while placing different sentinels at both the
high-level environment boundary and all reachable lower-level config/table
providers, then compares complete fitted public artifacts, an `act -> observe
-> act` trace, posterior/diagnostic state, and actions. The lower-level access
blocklist should include the public-action-channel/data-directory path as well
as the currently patched private table and solver paths. This is a test change;
no current private leak was found by this audit.

## 5. Final classification and required disposition

| Method | Current scientific classification | May remain unchanged | Required before a canary |
|---|---|---|---|
| Delphic internal ID | **Ensemble value-disagreement pessimism (Delphic-motivated), not a Delphic/compatible-world baseline.** It uses a shared fitted Q plus direct randomized support/ambiguity-tapered disagreement. | Calibrated public behavior reference may remain as a behavior diagnostic; exact downgraded reader label may remain; linear conservative-Q machinery may remain if presented as a heuristic. | Choose coherent Path A or complete Path B: remove ratio-engineered direct Q construction, compatible-world and Delphic-uncertainty semantics/diagnostics/tests/docs, and any claim that G-D3/G-D4 validate Delphic uncertainty. Do not tune the 3:1 gate. |
| OGSRL internal ID | **OGSRL-inspired guarded constrained actor with a public low-abundance proxy, currently unit-inconsistent.** | `s_low=Q_0.20` on positive train observations, bounded shortfall `c`, public-only inputs, guardian, and no-private-safety-guarantee label may remain. | Implement normalized `C_H` everywhere, use unclipped train-behavior `mean(C_25)` and error bars, make deployment evaluate normalized remaining-horizon occupancy, update tests/registration, and recheck dual binding without returns. |
| RefPlan | **RefPlan-inspired posterior-marginalized public-model planner with a root-frozen policy-prior proposal; not yet a history-conditioned trajectory-prior implementation.** | Separate calibrated policy prior, separate model posterior and updates, epsilon 0.10 mixture, common random numbers, posterior marginalization, action coverage, and no log-prior term may remain. | Re-evaluate the prior at every simulated public history/state and test later-action effects. |
| BA-MCTS | **BA-MCTS-inspired public dynamics-ensemble tree search with in-tree Bayesian member-belief updates.** | Current Phase 2 mechanism, ensemble size 5, primary 256 simulations/depth 8, and its tests may remain. | No correction identified by the latest review beyond the strengthened shared privacy lifecycle test. |

Across methods, one-thread pinning, one allocated CPU per task, deterministic
thread parity, separate task-hours/actual CPU-hours/allocated core-hours/
elapsed-wall/queue-delay reporting, matched 4,000-transition data, the shared
ensemble size of five, MOPO exclusion, isolated-worktree policy, and the ban on
return-based threshold tuning can remain unchanged.

The four-method canary is **NO-GO** until the Delphic Path-A/Path-B correction,
OGSRL unit correction, RefPlan sequential history-conditioned prior, and
privacy lifecycle/global-access test are reviewed and implemented. After those
changes, only return-blind mechanism checks should be used for a new readiness
decision. This audit grants no authority to regenerate or submit the manifest,
launch jobs, inspect returns, merge the worktree, or alter running PLUS/MOOR
work.
