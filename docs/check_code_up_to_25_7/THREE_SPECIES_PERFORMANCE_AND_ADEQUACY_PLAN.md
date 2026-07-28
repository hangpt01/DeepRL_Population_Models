# Three-Species Performance and Adequacy Plan

## Purpose

Use a fixed three-species panel to establish that the current ecological and general offline-RL
implementations produce stable, interpretable performance before expanding to all nine species.

This is a staged plan. It prevents three common mistakes:

1. modifying algorithms before obtaining a clean frozen baseline;
2. interpreting weak 4,000-transition results without checking data adequacy; and
3. tuning many hyperparameters against the final evaluation returns.

No result from this panel should be described as representative of all nine populations. It is a
diagnostic and method-development panel.

## Selected species

| Role | Species | Evidence from the registered demographic table | Why selected |
|---|---|---|---|
| Recoverable but declining | **Amur tiger** | `N0=200`, `K=250`, baseline-action `lambda=0.9552`, `r_base=-0.0458`, strongest-recovery action set-point `r_max=+0.0707` | A relatively large population that declines under the baseline action but remains recoverable. It tests whether methods identify useful intervention rather than relying on passive growth. |
| Strong-growth recoverable | **Crab-eating fox** | `N0=41`, `K=41`, baseline-action `lambda=1.3813`, `r_base=+0.3230`, strongest-recovery action set-point `r_max=+0.4418` | A contrasting high-growth, near-capacity case. It tests whether methods avoid unnecessary or costly intervention and exposes ceiling/saturation behavior. |
| Severe demographic sink | **Egyptian vulture** | `N0=41`, `K=325`, baseline-action `lambda=0.9128`, `r_base=-0.0912`, strongest-recovery action set-point `r_max=-0.0098` | Even the strongest registered recovery-action set-point remains negative. It is the clearest sink and the known action-coverage bottleneck. |

The three cases intentionally span baseline decline with recovery potential, strong passive growth,
and structural decline. Do not pool the Egyptian-vulture results with the two recoverable species
without reporting both strata separately.

### Required Crab-eating-fox data check

Four earlier Crab-eating-fox/theta files contained five extra partial-episode rows. The current
general-RL matched package corrected this by retaining exactly 160 complete 25-transition episodes.
Before generating the three-species manifests, verify that every fox cell uses the corrected
complete-episode dataset and matches the ecological registry hash. No arbitrary row truncation is
allowed.

## Methods in scope

Use the current honestly named implementations:

1. `plus_adapted_ricker_only_pbvi` — eight fixed Ricker candidate POMDPs;
2. `moor_adapted_ricker_misspec_pbvi` — one fitted Ricker POMDP;
3. RefPlan-inspired posterior-adaptive planning;
4. OGSRL-inspired guarded constrained actor;
5. BA-MCTS-inspired Bayes-adaptive tree search; and
6. episode-bootstrap value-disagreement pessimism, Delphic-motivated only.

Add cheap evaluator-only reference policies where supported:

- behavior/logging policy;
- no-intervention action `a0`;
- a clearly labelled known-model or known-demographics oracle, if already implemented and kept
  outside deployable-method comparisons.

Do not use the cellwise `Best general` envelope as if it were one deployable method.

## Stage 0 — Freeze current status before new work

Before modifying code or submitting jobs:

1. verify current Slurm and artifact status for PLUS, MOOR, and the 576-row general-RL package;
2. record which rows are completed, accepted, failed, or still quarantined;
3. verify that the Ricker-only PLUS parser fix and structural smoke test have passed;
4. preserve all existing snapshots, manifests, receipts, and return quarantines;
5. record current runtime/configuration/dataset hashes; and
6. do not inspect final-test returns while designing sensitivities.

If valid matching artifacts already exist for a selected row, reuse them by exact hash and provenance
rather than refitting solely to create a new directory identity.

## Stage 1 — Current-configuration performance baseline

### Balanced data matrix

Use:

- 3 selected species;
- 4 hidden environment families: Ricker, Allee, theta-logistic, regime-switching;
- observation noise `sigma_obs in {0.1, 0.2}`;
- 4,000 transitions in 160 complete episodes;
- episode-preserving 80/20 split;
- safe and yield reward modes; and
- the existing five evaluation seeds, four episodes per seed, 50 steps per episode.

This gives:

- `3 × 4 × 2 = 24` unique public dataset cells;
- `24 × 2 rewards = 48` cell-reward conditions; and
- `48 × 6 methods = 288` primary method rows.

Safe mode is the conservation-primary result. Yield is necessary as a paired diagnostic because safe
mode is information-limited by the hidden private safety objective. A method that is weak only in
safe mode should not automatically be called an intrinsically poor controller.

### Reuse rules

- Dataset collection, train/holdout splits, dynamics fits, and other reward-independent objects
  should be reused across safe and yield when their cache identities are exact.
- Reward-dependent actors, fitted Q-functions, planners, and evaluations must be regenerated.
- Existing MOOR or general-RL rows may be reused only when method version, data, split, reward,
  hyperparameters, evaluation seeds, and artifact hashes match exactly.

### Performance outputs

For every method-cell, retain:

- paired operational and evaluator true return;
- survival/recovery or collapse outcome as appropriate;
- unsafe occupancy/frequency, clearly marked evaluator-only where private;
- final and minimum abundance;
- action cost and action composition;
- intervention switching/frequency;
- runtime and peak memory; and
- completion/acceptance status.

Primary comparisons must be paired by dataset and evaluation seed. Report:

- each species separately;
- recoverable species versus sink separately;
- Ricker versus each non-Ricker family;
- `sigma_obs=0.1` versus `0.2`;
- safe versus yield; and
- method-specific intervals, not only ranks or a best-method envelope.

With only three fixed species, uncertainty intervals describe episode/evaluation variability, not
population-level generalization to other species.

## Stage 2 — Instrumentation improvements before expensive reruns

These changes improve diagnosis without changing the scientific algorithms. They require new runtime
provenance and tests but should precede algorithmic upgrades.

### Common requirements

- Atomic fit, plan, and evaluation completion receipts with exact artifact allowlists and hashes.
- Per-stage deterministic seed ledger and cache identity.
- Train and untouched-holdout one-step and multi-step prediction errors.
- Calibration by horizon, observation noise, action, family, and species.
- Exact action/episode coverage for every fitted action coordinate.
- Repeated-run deterministic parity for fixed seeds.
- Strict proof that hidden family and private demographic/safety information never enter fitting or
  control.

### PLUS and MOOR

- Per-start optimizer objective, termination state, iterations, and finite/non-finite status.
- Projected gradient separated into free and bound-active coordinates.
- Parameter-boundary occupancy under a registered tolerance.
- Common holdout trajectory SSE and holdout/train ratio.
- Parameter and discretized-kernel stability across starts and data budgets.
- PLUS candidate diversity, effective rank, duplicate/near-duplicate detection, posterior entropy,
  effective candidate count, and posterior history on diagnostic rollouts.
- PBVI argmax margin, action agreement on a frozen public belief set, value stability, and
  deterministic-repeat diagnostics.

### General RL methods

- Shared dynamics-ensemble negative log likelihood, interval coverage, residual calibration, and
  member diversity over multi-step rollout horizons.
- RefPlan: posterior entropy/ESS, likelihood calibration, proposal diversity, effective planning
  horizon, and frequency with which the uncertainty penalty changes the selected action.
- OGSRL: guardian support coverage, actor/critic losses, both dual trajectories, constraint
  feasibility, predicted public-risk calibration, and frequency of infeasible-action fallback.
- BA-MCTS: simulations completed, depth reached, node count, bucket occupancy, posterior entropy by
  depth, root action visit distribution, and repeated-search action agreement.
- Value-disagreement method: ensemble variance scale, action-switch fraction relative to
  ensemble-mean Q, member diversity, and explicit `lambda_V=0` ablation support.

Do not reconstruct missing diagnostics by reading final return summaries during a return-blind gate.

## Stage 3 — Data-budget adequacy

The inherited 4,000-transition budget is not yet established as sufficient. Finite fitting objectives
are not evidence of adequacy.

### Nested-data construction

Collect once at the largest budget, preserve complete episode order, and use strict nested episode
prefixes. Never truncate arbitrary rows.

Required comparison:

- 4,000 versus 8,000 transitions.

Optional learning-curve context, if collection is cheap:

- 1,000 and 2,000 transitions as smaller nested prefixes.

Use one untouched common audit holdout that is not consumed by any nested training subset. If the
existing 80/20 holdout must change to achieve this, create a new preregistered dataset version rather
than silently changing the current split.

### Sentinel adequacy subset

Use the three selected species under:

- Ricker, as the in-family condition;
- regime-switching, as the structurally hardest misspecification condition; and
- `sigma_obs=0.2`, the harder selected survey level.

This gives `3 species × 2 families × 1 noise = 6` sentinel cells per budget.

First run PLUS/MOOR fitting and planning diagnostics return-blind. Also compute shared general-model
holdout calibration. Then evaluate all six methods at 4,000 and 8,000 on the same cells as a labelled
data-sensitivity analysis.

### Preregistered 4k-versus-8k adequacy gates

Retain the already proposed thresholds:

- every fitted action coordinate has at least 40 transitions and 20 distinct episodes for a clean
  pass; below 20 transitions or 10 episodes is inadequate;
- holdout-SSE improvement from 4k to 8k is less than 10%;
- mean fitted-kernel total-variation change is below 0.05 on a frozen public grid;
- PBVI action agreement is at least 95% on a frozen public belief/context set;
- at least two finite optimizer starts and no all-start failure; and
- no PLUS candidate-form collapse.

If any difficult sentinel cell is inadequate, either make 8,000 the new primary budget and rerun all
headline methods at 8,000, or retain 4,000 only with an explicit data-limited label. Never compare an
8,000-transition ecological method against a 4,000-transition general learner as a matched headline
result.

## Stage 4 — Focused hyperparameter adequacy

Do not run a full factorial sweep. Use one-factor-at-a-time or preregistered low/default/high settings
on the six sentinel cells, primarily in safe mode. Preserve an untouched final evaluation set.

### Ecological methods

| Method/component | Current | Focused sensitivity | Refit? |
|---|---:|---:|---|
| PLUS Ricker candidates | 8 | 4 / **8** / 16 | Yes for added/changed candidates |
| PLUS/MOOR process paths | 16 | **16** / 32 | Yes |
| Optimizer starts | 8 | **8** / 16 | Yes |
| PBVI belief points | 32 | **32** / 64 | No; rebuild/replan |
| PBVI horizon | 5 | **5** / 8 | No refit; replan |
| Abundance grid | 41 | **41** / 61 | No refit; rediscretize/replan |
| Observation branches | 7 | **7** / 11 | No refit; replan |

Use fit stability and frozen-belief action/value agreement to judge adequacy. Do not select the
headline setting by looking for the best final-test return.

### General methods

| Method | Minimum focused sensitivity |
|---|---|
| RefPlan-inspired | Ensemble 5/10; current horizon-5/96 proposals versus horizon-10/192 proposals; posterior particles 32/64; `lambda_ref=0` ablation versus 0.5. |
| OGSRL-inspired | Actor iterations 30/60; deployment paths 256/512; rollout starts 128/256; guardian-threshold and public-risk-proxy calibration. |
| BA-MCTS-inspired | Registered compute ladder: 128 simulations/depth 5, **256/depth 8**, 512/depth 12; optionally ensemble 5/10 after the search ladder. |
| Value disagreement | `lambda_V=0` versus **0.1**; members 10/**20**/40; FQI convergence 35/70 iterations. Report action-switch fraction. |

The EVD penalty currently changes very few actions. The zero-penalty ablation is necessary to show
whether its disagreement mechanism contributes anything. If a new normalized or calibrated penalty
is implemented, treat it as a separately named method version and evaluate it on a new frozen test;
do not silently replace the registered 0.1 score after viewing results.

## Stage 5 — Training-data and evaluation replication

Five evaluation seeds on one offline dataset do not measure sensitivity to the collected training
log. Offline-data variability is likely more important than adding many rollouts from one fit.

Before expansion to nine species:

1. generate at least three independent offline collection seeds for the six sentinel cells;
2. use identical collection seeds and nested budgets across all methods;
3. retain the existing five evaluation seeds as the primary matched set;
4. preregister up to five additional evaluation seeds if paired confidence intervals remain too
   wide, using a direction-free precision rule; and
5. report variance decomposition across data seed, evaluation seed, family, species, and method.

Do not select the best training seed or report only the most favorable replicate.

## Algorithmic improvements: after the frozen baseline

Do not make all of these changes before Stage 1; doing so would remove the current reference point.

### Highest-value near-term changes

1. **General latent dynamics:** separate latent process evolution from survey observation noise;
   current ridge-linear observation-space ensembles conflate the two.
2. **RefPlan:** recurrent history encoder, calibrated deployment likelihood, a genuinely conservative
   offline policy prior, and MPPI-style soft weighting.
3. **OGSRL:** trust-region/CPO-style constrained optimization, better calibrated guardian, and
   preregistered public-risk alternatives. Continue to state that no private-safety guarantee exists.
4. **BA-MCTS:** larger calibrated ensemble, better continuous-observation tree handling, learned
   policy/value priors, and optional outer distillation after the direct-search baseline.
5. **Value disagreement:** either keep the honest bootstrap-EVD scope and calibrate its scale, or
   implement full latent observationally compatible worlds before using the Delphic name.
6. **PLUS:** improve within-Ricker candidate coverage using registered grids or hierarchical/robust
   candidates; compare against the separate cross-family PLUS only as a named model-class sensitivity.
7. **MOOR:** parameter uncertainty or bootstrap-fit sensitivity may be added as diagnostics, but an
   online family posterior would change MOOR's one-model identity.

Every algorithmic modification requires a new method version, runtime digest, tests, manifests, and
return-blind acceptance. Preserve the current frozen methods as comparators.

## Code and configuration worklist

### Dataset and registry

- Add a three-species allowlist containing exactly Amur tiger, Crab-eating fox, and Egyptian vulture.
- Validate 24 unique base dataset keys for `3 × 4 × 2`.
- Enforce 160 complete 25-step episodes at 4k and 320 at 8k where applicable.
- Add the Crab-eating-fox/theta complete-episode regression test.
- Implement deterministic nested-prefix dataset identities and a common audit-holdout identity.
- Store population role (`recoverable` or `sink`) only as evaluator/report metadata, never as a
  policy input.

### Method configuration

- Preserve current primary method identifiers and hyperparameters for Stage 1.
- Add explicit sensitivity configuration IDs rather than overwriting defaults.
- Separate reward-independent fit caches from reward-dependent policy/plan artifacts.
- Ensure Ricker-only PLUS rejects non-Ricker candidates and treats regime fields as inapplicable.
- Ensure every general method receives only public history/features.

### Manifests

Create separate immutable manifests for:

1. Stage-1 4k fits/training;
2. Stage-1 safe plans/evaluations;
3. Stage-1 yield plans/evaluations;
4. 4k/8k sentinel adequacy diagnostics;
5. focused hyperparameter sensitivities;
6. independent training-data replicates; and
7. final test evaluation.

Every row must include dataset hash, population, family, noise, reward, budget, data seed, split,
method version, hyperparameters, fit seed, plan seed, evaluation seeds, output root, and expected
parent-artifact hashes.

### Tests

- Exact manifest cardinality and positional fit/plan pairing.
- No output collisions across reward, budget, data seed, or sensitivity version.
- Private-information intervention invariance through fit and sequential action selection.
- Dataset-prefix nesting and no holdout leakage.
- Candidate/model count and cache-identity validation.
- Deterministic one-thread repeatability.
- Method-specific mechanism tests already registered in the two explanation documents.
- Acceptance parser rejection of return/ranking fields during structural gating.
- Atomic receipt and cancellation-integrity tests.

### Scheduler and resources

- Measure task-level runtime and memory from accepted existing jobs before estimating the new run.
- Use method-specific arrays and concurrency caps so PLUS/BA-MCTS do not starve other methods.
- Pipeline only where dependencies and global CPU caps remain explicit.
- Use no automatic retries unless the retry policy and seed/artifact identity are preregistered.
- Monitor completion, failure, timeout, memory, and core-hours without opening returns.

### Acceptance

Run return-blind acceptance before performance analysis. It must verify:

- exact code/config/manifest/dataset identity;
- privacy and family-label exclusion;
- finite fits and method-specific structural diagnostics;
- complete artifact allowlists and hashes;
- legal actions and deterministic seed behavior;
- exact cell/method/evaluation completion; and
- declared resource ceilings.

Unavailable diagnostics must be marked `NOT EVALUABLE`, not passed.

## Expansion gate to all nine species

Expand only after the three-species panel establishes:

1. all registered rows complete or failures are scientifically localized;
2. no privacy, provenance, cache, or routing failure;
3. the primary data budget is selected from the preregistered adequacy criteria;
4. fit/kernel/planner stability is acceptable on the sentinel cells;
5. performance conclusions are not driven solely by one offline dataset seed;
6. method comparisons are stable enough under paired uncertainty to justify more compute;
7. sink and recoverable results remain separately interpretable; and
8. the full nine-species resource estimate fits the available scheduler budget.

If these gates fail, diagnose the failing dimension on the three-species panel rather than expanding
the same uncertainty to all nine species.

## Recommended execution order

1. Audit current artifacts and fix any unresolved launch-only blocker.
2. Add instrumentation and freeze a new diagnostic-capable runtime.
3. Run Stage 1 at the current 4k/default settings on all 24 public cells.
4. Pass return-blind acceptance, then analyze safe and yield performance.
5. Run the 4k/8k sentinel adequacy study and three-data-seed replication.
6. Run focused, method-specific hyperparameter sensitivities on the sentinel cells.
7. Freeze any justified new primary configuration and evaluate it on an untouched final test.
8. Decide whether to expand to all nine species.
