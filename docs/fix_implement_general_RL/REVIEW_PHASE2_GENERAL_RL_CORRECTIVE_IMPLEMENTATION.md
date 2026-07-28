# Review of the Phase 2 general-RL corrective implementation

## Verdict

**NOT READY for the corrected general-RL canary yet.**

The implementation made substantial and credible progress. RefPlan's model belief now affects
planning, OGSRL now trains a constrained policy, BA-MCTS now updates its ensemble belief inside
simulated histories, the work was isolated correctly, and the test/provenance discipline is much
better. However, four issues should be resolved before a canary is treated as validating the final
methods.

Do not inspect performance returns or submit the 1,152-row manifest while addressing these issues.
Do not interfere with the frozen PLUS/MOOR jobs.

## Finding 1 — Blocking: Delphic compatibility gate is too weak

The current gate accepts a world when the total-variation distance between its **marginal action
frequency** and the empirical marginal action frequency is at most 0.15. Matching
`P(a)` is not sufficient evidence that worlds reproduce the observable behavior/trajectory
distribution. A model can match the overall action histogram while assigning actions incorrectly
for every public history, observation, or state-action region.

This is particularly important here because all 20 worlds pass with very small marginal TV while
their counterfactual Q variance can be extremely large. Large Q variance is meaningful as Delphic
uncertainty only after observational compatibility has been established; otherwise it may reflect
arbitrary random-projection misspecification.

The report also needs to reconcile two statements:

- Section 4 says conditional behavior NLL is approximately 24 nats and unusable;
- the same section later says calibration reduced behavior NLL from approximately 25 to
  approximately 2.

Determine exactly which split, normalization, and likelihood each number describes.

### Required resolution

Before changing code, return a plan for a stronger observable-compatibility gate. It should assess
the held-out observable distribution conditionally, not only its marginal. At minimum consider:

1. integrated/marginalized held-out `P(a_t | public history_t)` relative to a latent-free reference;
2. held-out public next-observation likelihood or calibrated prediction error conditional on public
   history and action;
3. held-out public reward-surrogate likelihood/error where that channel is used;
4. per-action and per-public-state-region calibration, so global frequency matching cannot hide
   local incompatibility;
5. a joint score or explicit set of gates with thresholds fixed before returns;
6. a toy counterexample that matches `P(a)` but mismatches `P(a|history)` and must fail;
7. a toy example with observationally compatible worlds that disagree only counterfactually and
   must pass.

Full neural ELBO reproduction is not required. A lightweight approximation is acceptable, but the
observable-compatibility claim must be supported by more than an action histogram. If this cannot
be done defensibly with the current world representation, report that Delphic remains an
idea-level ambiguity penalty and propose the appropriate narrower name.

## Finding 2 — Blocking paper closeness: RefPlan still lacks the planning prior

The full-paper audit identified the conservative offline policy as one half of RefPlan's "doubly
Bayesian" construction. The correction restored deployment model-belief marginalization but still
samples/enumerates candidate plans without an offline policy prior. Disclosure makes the current
name honest (`RefPlan-inspired`), but it does not satisfy the user's goal of staying as close to the
paper as reasonably possible.

### Required resolution

Return a design for a lightweight public behavior/conservative policy prior trained from the same
4,000-transition offline split. It should:

- use only public observations/history and public actions;
- generate or weight candidate action sequences during planning;
- retain enough registered exploration that unsupported deterministic behavior does not eliminate
  all alternatives;
- avoid holdout fitting;
- keep the model posterior and the policy prior as distinct objects;
- include a toy test proving that changing the policy prior changes candidate-plan probabilities or
  selection while holding the model belief fixed;
- state precisely which parts remain computational adaptations of the paper.

If there is a strong reason not to implement this inexpensive prior, quantify the reason and return
for an explicit naming/scope decision. Do not silently make the omission permanent.

## Finding 3 — Blocking privacy verification: equality scanning does not exclude transformed leaks

The new recursive scan establishes that no stored scalar happens to equal `K_ref` or the private
safety threshold. It cannot detect transformed or normalized leakage such as `log(K)`, `K/2`, a
threshold-derived label, a private-family-dependent branch, or a derived feature.

### Required resolution

Add a **private-value intervention invariance** test:

1. hold the entire method-visible public dataset, public configuration, seeds, and method ID fixed;
2. change private `r`, private `K`, private safety thresholds, private family labels, and any
   privileged metadata behind the blocked interface;
3. refit and act with each hidden method;
4. assert byte-identical fitted public artifacts where deterministic and identical actions/planning
   diagnostics otherwise;
5. run the intervention one private field at a time so a failure identifies the leak;
6. retain the API/name blocking and relabel tests as complementary checks.

If the framework cannot construct this paired intervention without regenerating public data,
explain why and test the closest lower-level method-facing boundary directly.

## Finding 4 — Scientific decision: OGSRL should not be finalized with an inert safety cost

Keeping the safety-cost constraint present but constant is honest for a pipeline smoke test, but it
does not validate the final OGSRL-inspired baseline. The paper's method performs constrained policy
optimization with support/OOD and domain safety costs. In the benchmark, exact extinction
(`state == 0`) is the wrong observable event because it never occurs over the 25-step offline
episodes.

The final hidden method should retain the binding OOD guardian and use a **public, preregistered,
non-private low-abundance cost**. This is permissible because no performance returns have been read
and the degeneracy was discovered through a mechanism diagnostic.

### Required resolution: plan before implementation

Compare candidate public costs without inspecting returns. Candidate definitions may use only
public observation histories and a scale estimated from the training split. Consider continuous
costs before arbitrary binary thresholds, for example a bounded shortfall below a public
training-derived abundance scale. The returned plan must provide:

- the exact formula and range;
- why it expresses low-abundance risk rather than generic rarity;
- how its scale is estimated without holdout fitting or private `K`/safety information;
- prevalence/spread by population, action, family, and observation-noise level;
- whether it remains informative in both healthy and sink populations;
- a fixed safety budget or a data-derived rule specified before returns;
- tests showing the safety dual binds on a constructed case and is not automatically saturated;
- wording that this is a public proxy and carries no guarantee for the private safety objective;
- matched access: whether other methods consume the same public risk channel and in what role.

Do not select among candidate costs by policy return. Select by interpretability, non-degeneracy,
privacy, and stability only. If no defensible public cost works across the benchmark, retain OGSRL
as **OOD-guarded OGSRL-inspired** and explicitly remove claims that its domain-safety component was
instantiated.

## Finding 5 — Required engineering clarification: runtime accounting and thread allocation

The report measures approximately 125% CPU per process while estimating one allocated core per
task. Before launch, verify that the scheduler/cgroup permits this and that NumPy/BLAS thread counts
are explicitly controlled. Otherwise the projected 128-way wall time and allocated-core accounting
may be optimistic or may oversubscribe the node.

### Required resolution

- report `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and any NumPy backend;
- run deterministic 1-thread versus current-thread canaries for each method;
- compare fitted parameters, action outputs, BA-MCTS beliefs, hashes where applicable, wall time,
  and CPU utilization;
- choose and register one CPU/thread configuration;
- report aggregate task-hours, actual CPU-hours, allocated core-hours, and elapsed wall separately;
- retain BA-MCTS 256/depth-8 primary and 512/depth-12 only as a small sensitivity unless measured
  evidence requires another decision.

## Decisions accepted from the report

- Keep the matched 4,000-transition primary budget.
- Keep shared ensemble size 5 for RefPlan/OGSRL/BA-MCTS; use 7 only as a shared sensitivity.
- Keep BA-MCTS 256 simulations/depth 8 as primary.
- Keep Delphic world count 20 as the provisional primary value, subject to the strengthened
  compatibility gate.
- Keep MOPO archived and excluded.
- Keep all reader-facing labels explicit that these are paper-inspired adaptations.
- Keep the isolated worktree; do not merge into the dirty main tree yet.

## Required next response

This is a **plan-only reconciliation step**. Do not edit code yet.

Write `PHASE2B_GENERAL_RL_REMAINING_ISSUES_PLAN.md` containing:

1. response to each finding above, with agreement/disagreement and source/code evidence;
2. reconciliation of the Delphic NLL numbers;
3. proposed strengthened Delphic compatibility equations and thresholds;
4. proposed RefPlan public policy-prior implementation;
5. private-value intervention test design;
6. OGSRL public-risk candidate comparison and recommended preregistered definition;
7. thread/resource validation plan;
8. exact files/functions/tests to change;
9. return-blind acceptance gates;
10. rerun and snapshot scope;
11. revised GO/NO-GO recommendation.

Do not launch jobs, inspect performance returns, modify the prepared manifest, merge the isolated
branch, or touch the active PLUS/MOOR run. Stop after returning the plan for review.
