# Algorithm Paper / Implementation Reflection Audit

Date: 2026-07-05

Scope: I read the seven local papers in `algorithms/`, the current
`29_6_Algorithm_Paper_Audit_Guide.tex`, and the nearby continuous-setting
handoff / implementation-plan documents. This folder does not contain the
runnable code repository, so this is an audit of the documented implementation
reflection, plus a list of code-level checks that still need the repo.

## Overall Verdict

The TeX file is useful and mostly honest that the benchmark is using adapted
methods rather than source-code reimplementations. However, if the current code
really follows the TeX exactly, there are several serious conflicts with either
the paper settings or the benchmark's own POMDP setting.

The biggest issue is not that the benchmark changes neural methods into
lightweight ecological variants. That is defensible. The problem is that some
sections still describe paper names too strongly after replacing the paper's
central mechanism:

- BA-MCTS becomes an online tree-search heuristic, not the paper's offline
  Continuous-BAMCP-plus-policy-iteration algorithm.
- Delphic-CQL becomes random-feature ensemble pessimism, not compatible-world
  hidden-confounding uncertainty unless the worlds are actually trained to
  factorize the observational distribution.
- PLUS becomes a 21-point capacity-only Ricker baseline, while the paper's model
  uncertainty is broader; the statement that the 21-candidate count is preserved
  is misleading.
- OGSRL keeps the right high-level idea, but the documented guardian feature
  space is too small to certify state-action support in this benchmark.
- The POMDP definition includes rewards in public history even though the reward
  is a near-invertible function of the true latent next abundance. If reward is
  used by the filter or policy, this leaks the hidden state.

## High-Severity Findings

### H1. Reward in public history can leak the true latent state

Document evidence:

- `29_6_Algorithm_Paper_Audit_Guide.tex` defines public history as
  `(o_0,a_0,R_0,o_1,...,o_t)`.
- The same file defines reward from true next abundance `s_{t+1}`.
- `29_6_Real_Ecology_Setting_Implementation_Plan.tex` correctly warns that
  realized reward must not be a filter/policy input because it is an invertible
  readout of `s_{t+1}` in yield mode after adding back known cost.

Why this matters:

This is a serious conflict with the POMDP setting. If the filter, policy, belief
features, or method inputs condition on `R_t`, the agent is no longer acting only
from noisy surveys. In yield mode,

`R_t + cost(a_t) = alpha * s_{t+1} / (s_{t+1} + K_base)`

is monotone and invertible in `s_{t+1}`. That would largely defeat the partial
observability experiment.

Recommendation:

Rewrite the audit guide so the belief/policy history is
`(o_0,a_0,o_1,...,a_{t-1},o_t)` plus known public controls/context. Treat `R_t`
as a logged training/evaluation target only. Add a code test: two rollouts with
identical `(o,a)` histories but different logged rewards must produce identical
beliefs and actions.

### H2. BA-MCTS is under-described as a light adaptation, but the paper's core training loop is gone

Paper setting:

The ICLR 2026 BA-MCTS paper frames offline MBRL as a pessimistic BAMDP, uses deep
ensembles as prior world-model samples, applies a posterior update over models,
uses Continuous BAMCP with double progressive widening / PUCT-style search, and
distills search results into actor/critic learning, primarily with SAC. The paper
explicitly says running search at every deployment state is too expensive, so the
offline policy is learned from search-generated segments.

TeX reflection:

The benchmark fits the shared ridge ensemble, then does decision-time tree search
with bucketed continuous states and a posterior residual update. There is no
documented Continuous BAMCP/DPW implementation, no policy/value distillation, and
the value leaf `V(s')` is not tied to the paper's learned critic.

Why this matters:

This is not just a simplified model class. It removes the paper's offline policy
improvement operator. The current method may be a useful BAMCP-style ecological
planner, but it should not be described as preserving BA-MCTS paper structure
without a strong caveat.

Recommendation:

Either rename it to "BA-MCTS-inspired online Bayes tree search" / "BAMCP-style
planner", or implement the paper-shaped loop: Continuous BAMCP search targets,
policy/value distillation, pessimistic one-step lookahead-Q penalty, and a
trained value function for leaves. At minimum, make the leaf-value approximation
explicit and test whether zero / poor leaves explain collapse.

### H3. Delphic-CQL does not establish compatible worlds

Paper setting:

Delphic Offline RL targets nonidentifiable hidden confounding. It defines
compatible worlds as latent generative models that induce the same observational
trajectory distribution, including a confounder distribution, behavior policy,
transition/reward or Q model, and counterfactual value predictions. Delphic
uncertainty is across-world variance in counterfactual value. The practical paper
trains variational latent world models and then adds a delphic penalty to offline
Q-learning/CQL.

TeX reflection:

The benchmark samples random projections of belief features, fits behavior
classifiers and linear Q-heads, and calls their value variance delphic
uncertainty.

Why this matters:

Random-feature worlds are not automatically compatible worlds. Unless each world
is trained/validated to explain the same observational distribution with a latent
confounder mechanism, the "delphic" uncertainty is closer to arbitrary ensemble
variance. That weakens the causal/confounding interpretation.

There is also a likely equation mismatch: the paper penalizes the Bellman target
for the sampled `(s,a)` with `u_d(s,a)`. The TeX penalizes the target with
`U(phi_t, a*_t)`, where `a*_t` is the current greedy action, not necessarily the
observed action being updated.

Recommendation:

Either implement actual compatible latent world models, or rename this baseline
to "Delphic-inspired random-world CQL". Fix the Bellman penalty to apply to the
action whose Q target is being updated, and use a CQL regularizer closer to
`logsumexp_a Q(s,a) - Q(s,a_data)` if the method is still called CQL.

### H4. PLUS candidate-set wording is misleading and the adaptation is much narrower than the paper

Paper setting:

PLUS handles state and model uncertainty through a finite prior over candidate
models and Bayesian updates. The paper's simulation text uses 21 candidates for
`r`, 11 for `K`, and 10 for noise, while the real hake example identifies 144
candidate models over `r`, `K`, process noise, and measurement noise. The paper
also explicitly warns that PLUS fails when the true model is not in or well
approximated by the candidate set, and says structural model diversity can be
added to the prior set.

TeX reflection:

The benchmark uses exactly 21 candidates, but now only over carrying capacity
`K`, and says the "21-candidate count is preserved."

Why this matters:

The TeX preserves the number 21 from the paper's `r` grid, not the paper's full
model-uncertainty construction. A K-only grid is a defensible benchmark baseline
if action-specific growth set-points are known, but it is not a faithful PLUS
model-uncertainty prior. It also cannot learn Allee, theta-logistic, or regime
structure unless those are included as candidates.

Recommendation:

Replace "21-candidate count is preserved" with "we use a deliberately reduced
21-point K-only PLUS-style prior." If PLUS is meant to be a stronger ecology
baseline, include candidate axes for `K`, observation/process noise, and possibly
structural family. If it is intentionally misspecified, say so in the table.

### H5. OGSRL's guardian feature space is too small for the benchmark support constraint

Paper setting:

OGSRL constrains policies in a guarded CMDP using a state-action support guardian
plus explicit safety-cost constraints. Its practical implementation uses scalable
support estimators such as KDE/kNN, but the guardian is still over the relevant
state-action space.

TeX reflection:

The documented guardian embeds only
`[log(1 + mean_abundance / K_ref), action_index/(A-1)]`.

Why this matters:

In this benchmark, support depends on population identity, observation noise,
belief uncertainty, public control accumulators (`rho`, `kappa`, `K_eff`), and
possibly reward mode/safety context. A guardian over only mean abundance and
action index can label unsupported controls as safe, or reject supported controls
because the public context was omitted. That cuts directly against OGSRL's core
state-action support guarantee.

Recommendation:

Build the guardian on the same public belief/context feature vector available to
the policy: belief moments/quantiles, population context, action, `rho`,
`kappa/K_ref`, `K_eff/K_ref`, and perhaps uncertainty width. Separately define
training-time discounted cumulative budgets and deployment-time one-step
feasibility thresholds; do not reuse one as the other.

## Medium-Severity Findings

### M1. MOPO is acceptable as MOPO-style, but not as MOPO proper

The paper's MOPO is a fully observed MDP algorithm that learns probabilistic
neural ensemble dynamics/reward models, uses an uncertainty-penalized reward, and
optimizes a policy using model-generated rollouts. The TeX uses ridge bootstrap
next-abundance models and particle MPC. This keeps "model uncertainty pessimism"
but removes synthetic rollout policy optimization and the paper's probabilistic
uncertainty estimator. This is reasonable for an inspectable ecological benchmark
if named "MOPO-style" or "MOPO-MPC"; do not imply paper-level guarantees.

### M2. RefPlan is a posterior-ensemble planner, not the paper's doubly Bayesian RefPlan

The RefPlan paper uses a prior offline policy, a VAE/RNN encoder for epistemic
belief over latent dynamics, a decoder dynamics/reward model, control-as-inference
trajectory posterior estimation, and test-time MPC that marginalizes latent
samples. The TeX replaces this with a bootstrap ensemble posterior and residual
likelihood update. This is a plausible benchmark simplification, but the TeX
should state that it is "RefPlan-inspired" and lacks the learned latent epistemic
encoder and prior-policy trajectory posterior.

### M3. MOOR-Ricker is a useful mechanistic baseline, but not MOOR's fishery pipeline

The MOOR paper learns a continuous fishery POMDP from catch/effort series, with
catch as both observation and reward, then discretizes the learned POMDP for a
solver. The benchmark instead fits a Ricker map to public belief means and plans
with particle MPC over 11 conservation actions. This is a fair "Ricker expert"
baseline, but the TeX should avoid implying it is MOOR's algorithm beyond the
mechanistic learn-then-plan spirit. Also report a pure mechanistic filter variant
if the current MOOR baseline benefits from the shared learned filter.

### M4. Log residual posterior updates need a zero-abundance rule

RefPlan and BA-MCTS update model posterior weights using
`log(o_{t+1}) - log(s_hat_{t+1})`. In this setting `s=0` is absorbing, so `o=0`
can occur. The TeX should specify either an epsilon floor or an emission model
with a point mass at zero. Otherwise near-collapse episodes can create NaNs or
infinite likelihood ratios.

### M5. The theta-logistic rate convention is missing from the audit guide

The implementation plan says Ricker/Allee/regime use `r = ln(lambda)`, while
theta-logistic uses `r = lambda - 1`. The audit guide gives a single
`r_eff` notation and transition family formulas without restating this
family-specific conversion. If the code uses the wrong column for theta, that is
a serious dynamics error; if the code is correct, the TeX should still state the
convention.

### M6. The negative-growth split is reasonable but should be labeled as a benchmark convention

Using `r_+` in the density-dependent term and `exp(r_-)` as unconditional
mortality avoids sign pathologies for negative real-data set-points. That is a
reasonable engineering/ecology convention, but it is not the standard Ricker or
Allee map. The TeX should explicitly mark this as a real-data sign convention.

### M7. Shared particle MPC wording says "continuous-action" although actions are finite

The benchmark actions are 11 named discrete interventions. The shared planner is
a continuous-state / particle-belief sequence sampler over finite action
sequences, not a continuous-action engine. This is minor but worth fixing for
clarity.

## Reasonableness of the TeX's Adaptation Reasons

Reasonable:

- Using public beliefs/particles is appropriate for the continuous noisy-survey
  POMDP, provided rewards and private states are excluded from belief/policy
  inputs.
- Using transparent linear/ridge models is defensible for a small ecological
  benchmark where auditability matters more than reproducing deep architectures.
- Using shared evaluation and shared MPC can make comparisons cleaner, as long as
  paper names are softened to "style" or "inspired" where core algorithms were
  replaced.
- Treating MOOR and PLUS as ecology-native mechanistic baselines is scientifically
  sensible.
- The real-data set-point `r`, cumulative `K`, direct translocation, and true-state
  reward are coherent with the local real-ecology specification.

Needs revision:

- The final summary table is too generous for BA-MCTS, Delphic-CQL, and PLUS.
  Those adaptations do not merely simplify implementation; they change the
  method's main object.
- The TeX should distinguish "benchmark adaptation for controlled comparison"
  from "paper-faithful implementation".
- It should include an explicit "not faithful to original paper in these ways"
  paragraph per heavily adapted method.
- It should add a code-verification checklist, because the document currently
  presents implementation behavior without the actual code in this folder.

## Code Repo Checks Needed

When the implementation repo is available, I would audit these first:

1. Confirm reward is never in filter, policy, behavior classifier, guardian, or
   belief-feature inputs.
2. Confirm private simulator state/hidden family/parameters are not used when
   building the belief cache or dynamics targets, except through public
   observation likelihoods.
3. Confirm theta cells use LGM `r = lambda - 1`, while Ricker/Allee/regime use
   `ln(lambda)`.
4. Confirm log-likelihood updates handle `o=0` and `s_hat=0`.
5. Inspect BA-MCTS leaves: whether `V(s')` is learned, zero, or an accidental
   placeholder.
6. Inspect Delphic training: whether worlds are genuinely compatible latent
   models or only random projections.
7. Inspect PLUS candidate construction and whether structural candidates are
   intentionally absent.
8. Inspect OGSRL guardian features and budget semantics.
9. Confirm MOOR is reported both with a pure mechanistic belief/filter and with
   any shared learned front-end, if both exist.
10. Check training/holdout split is by episode, not by transition.

## Bottom Line

I would not reject the benchmark design. I would reject any wording that claims
these are faithful implementations of the original algorithms. The scientifically
safe framing is:

> We implement audit-friendly benchmark variants inspired by MOPO, RefPlan,
> BA-MCTS, PLUS, Delphic ORL, OGSRL, and MOOR. The variants share a public-belief
> POMDP interface and a common evaluation protocol; therefore, results compare
> benchmark adaptation families rather than reproducing the original paper
> codebases.

Then explicitly mark BA-MCTS, Delphic-CQL, and PLUS as the most heavily adapted.
