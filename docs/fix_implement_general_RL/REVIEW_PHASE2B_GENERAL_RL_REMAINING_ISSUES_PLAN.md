# Review of `PHASE2B_GENERAL_RL_REMAINING_ISSUES_PLAN.md`

## Verdict

**Approve the overall direction, but do not implement Phase 2C from the plan as currently written.**

The RefPlan prior, private-value intervention test, OGSRL public low-abundance cost, and thread
pinning are well motivated. The Delphic construction is not yet scientifically defensible, and the
OGSRL cost budget needs unit-consistent specification. Resolve the points below before editing code.

Do not launch jobs, inspect performance returns, merge the isolated worktree, alter the prepared
manifest, or disturb PLUS/MOOR.

## F1 — Blocking: the proposed Delphic construction manufactures the desired gate result

The proposed construction

`Q_w = Q_0 + g_w(s,a) * u(s,a) * kappa_amb(s)`

directly adds random, support-gated counterfactual perturbations to Q. Because `u(s,a)` is small on
observed support and large off support, this construction is designed to reduce `V_obs` and increase
`V_cf`. Passing `V_cf / V_obs >= 3` would therefore be partly tautological rather than evidence that
the fitted data admit multiple observationally compatible causal worlds.

More importantly, an arbitrarily perturbed Q-function need not be the value function of any
coherent candidate world with a behavior model, transition kernel, reward model, and latent
confounder. Delphic uncertainty is variation across values **induced by compatible worlds**; it is
not arbitrary Q disagreement placed off support.

### Required revision

Choose one of the following paths before implementation.

**Path A — coherent lightweight Delphic-inspired worlds.** Candidate variation must enter through
world-level generative components, not through a direct Q perturbation. Each world must define:

1. a public-history/latent behavior model;
2. a public next-observation transition model;
3. a public reward model or the common public reward mapping, with the distinction disclosed;
4. a latent variable or coupling that can influence both behavior and outcomes;
5. an observed-data likelihood or equivalent observable-distribution objective;
6. a policy-evaluation procedure deriving `Q_w` from that world's fitted dynamics/reward.

Different initializations or latent couplings may produce multiple worlds, but each retained world
must independently pass observable compatibility. Agreement on support and disagreement
counterfactually must emerge from the fitted worlds; it must not be hard-coded into Q.

A small discrete latent variable with EM/variational fitting or another lightweight marginal-
likelihood construction is acceptable. Full neural reproduction is not required.

**Path B — honest downgrade.** If Path A is not feasible within the current framework/deadline,
retain the existing value-disagreement penalty only under the reader-facing name:

`Ensemble value-disagreement pessimism (Delphic-motivated)`

Under Path B:

- do not call the candidates observationally compatible worlds;
- do not call their variance Delphic uncertainty;
- report that observed-support disagreement shows misspecification contamination;
- keep it as an idea-level proposal ingredient, not a paper-aligned Delphic baseline;
- do not engineer a ratio of three by support-gating random Q perturbations.

**Recommendation:** first return a concrete Path-A generative design and cost estimate. If it cannot
be specified coherently and tested without substantial new infrastructure, choose Path B now. A
scientifically honest downgrade is preferable to a construction optimized to pass its own gate.

## F2 — Blocking: Delphic thresholds are not fully preregistered

The proposed gates leave two values undefined:

- `tau_obs` in `V_obs <= tau_obs * scale^2`;
- the scaled positive floor in `V_cf > 0`.

The value scale also needs robust behavior when `median |Q_obs|` is zero or nearly zero. Specify all
units, floors, aggregation rules, and failure behavior before implementation or further diagnostic
measurement.

For Path A, thresholds should validate compatibility, not force every cell to exhibit large
counterfactual ambiguity. A compatible cell may legitimately have little ambiguity. Separate:

- **world validity:** observational compatibility and Bellman/generative coherence;
- **uncertainty presence:** magnitude and concentration of cross-world variance.

Failure to find large `V_cf` should normally mean "little detected Delphic ambiguity," not that an
otherwise compatible method is invalid. At `sigma_obs=0`, near-zero ambiguity is the expected
scientific result and should pass the validity check while reporting zero ambiguity; it should not
make the method fail merely because `V_cf > 0` is false.

## F3 — Blocking: OGSRL safety-budget units are ambiguous

The plan proposes a per-transition cost in `[0,1]` but defines the budget as "mean discounted cost"
and then clips it to `[0.02, 0.10]`. A discounted cumulative cost over 25 or 50 steps is not on the
same scale as a per-step cost, so the proposed cap can accidentally make the constraint nearly
impossible.

### Required revision

Define one explicit cost functional. The recommended form is the normalized discounted occupancy
cost

`C_H = ((1-gamma) / (1-gamma^H)) * sum_{t=0}^{H-1} gamma^t c(o_{t+1})`,

which remains in `[0,1]`. Define the primary budget as the training behavior policy's estimated
`E[C_H]` on complete training episodes. Since the measured value is already non-degenerate, do not
apply an unexplained floor/cap. If numerical clipping is necessary, preregister its sole numerical
purpose and show that it does not change any measured cell's budget.

State whether fitting uses horizon 25, deployment uses horizon 50, and how the same budget is made
comparable across those horizons. Prefer the normalized form precisely because it makes that scale
stable. Include uncertainty/error bars for the behavior estimate without using evaluation returns.

The public shortfall itself is approved provisionally:

`s_low = Q_0.20(positive training observations)`

`c(o') = max(0, (s_low-o')/s_low)`.

It must remain labelled a public low-abundance proxy with no guarantee for the private safety
objective.

## F4 — Required paper-closeness revision: RefPlan prior should govern complete candidate plans

Sampling only the first action from `pi_prior` and then reverting to uniform continuations is too
weak to represent a trajectory prior. During candidate generation, evaluate the public policy prior
at each simulated public history/state and sample every continuation from the registered
epsilon-mixture. Keep common random numbers and model-belief marginalization intact.

Specify whether the prior enters only through proposal sampling or also through an explicit
log-prior term. Use one primary mechanism, not an ambiguous "and/or":

- recommended primary: prior-guided sequence proposal at every horizon step;
- no additional log-prior score unless the paper-derived temperature and units are specified.

Add tests showing that the prior affects later actions as well as the first action.

## F5 — Accepted with minor requirements

The following may proceed after F1-F4 are resolved:

- paired private-value intervention invariance;
- common ensemble size 5 and matched 4,000-transition data;
- BA-MCTS 256/depth 8 primary;
- one thread and one allocated CPU per task, with deterministic parity measurements;
- MOPO archived/excluded;
- isolated worktree retained without merging.

For the privacy test, vary private fields at both the context-construction boundary and any lower
level global/config object accessible during `fit`, `observe`, or `act`. The test should fail if a
future method bypasses `MethodContext` and reads private configuration indirectly.

## Required next response

This remains a **plan-only scientific decision step**. Do not edit code yet.

Write `PHASE2B2_GENERAL_RL_FINAL_SCIENTIFIC_DECISIONS.md` containing:

1. agreement/disagreement with F1-F5;
2. a concrete coherent Path-A Delphic world model, fitting objective, Q derivation, tests, and
   compute estimate;
3. a direct comparison of Path A versus immediate Path B downgrade;
4. complete Delphic validity and uncertainty-reporting thresholds with no undefined symbols;
5. the unit-consistent OGSRL normalized discounted cost and budget definition;
6. the full-horizon RefPlan policy-prior proposal mechanism;
7. final file/function/test changes;
8. exact downgrade labels and report claims if any mechanism cannot pass;
9. revised implementation and canary runtime estimates;
10. a decision table requiring at most the genuine remaining user choices.

Do not run further empirical probes merely to tune thresholds. Existing return-blind measurements
may be used to explain the design, but all remaining constants must be justified independently of
comparative policy performance.
