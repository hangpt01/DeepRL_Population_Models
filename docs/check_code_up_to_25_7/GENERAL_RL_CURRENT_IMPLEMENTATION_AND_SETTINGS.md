# Current General-RL Methods: Implementation, Adaptations, and Experiment Settings

## Status and purpose

This document records the current hidden-demographics general-RL implementation as of
20 July 2026. It is intended to be the technical reference behind the supervisor-facing LaTeX
explanation.

The retained methods are:

1. `refplan` -- RefPlan-inspired posterior-adaptive planning;
2. `ogsrl` -- OGSRL-inspired guarded constrained policy optimization;
3. `bamcts` -- BA-MCTS-inspired Bayes-adaptive tree search;
4. `ensemble_value_disagreement_pessimism` -- bootstrap fitted-Q disagreement pessimism,
   motivated by Delphic pessimism at the idea level.

MOPO is archived and excluded. The historical `delphic` method ID is a deprecated read/CLI alias
only and is not used in the current experiment manifest.

The first three methods retain their paper-defining control or inference mechanisms in
benchmark-adapted form. The fourth method is intentionally **not** described as a Delphic
reproduction: it has no latent causal compatible-world construction and estimates ordinary
bootstrap fitted-value disagreement.

## Current run state

- A 64-row limited canary completed successfully on `m3h/m3h`.
- All 64 rows completed; no failures, cancellations, or retries.
- Return-blind structural acceptance passed for all rows.
- Performance outputs remain quarantined and unread.
- A matched 576-row full package using only `sigma_obs={0.1,0.2}` is prepared and validated but,
  at the time of this record, not yet submitted.

## Scientific question

The comparison asks whether general offline RL and planning methods can learn useful management
policies when ecological demographic structure is not disclosed. In hidden mode, a method does not
receive:

- intrinsic or action-specific demographic parameters;
- carrying capacities;
- the true demographic family label;
- the private ecological safety threshold;
- privileged latent population state during deployment;
- native mechanistic solver access.

It receives public logged transitions, management actions, noisy survey observations, public action
channels/costs, and public reward or proxy-risk quantities constructed without private parameters.

## Common benchmark setting

| Item | Current setting |
|---|---|
| Populations | 9 total: 7 recoverable and 2 demographic sinks |
| Demographic families | Ricker, Allee, theta-logistic, regime-switching |
| Observation noise | Primary matched full package: `sigma_obs={0.1,0.2}` |
| Reward modes | `safe` and `yield` |
| Management actions | 11 discrete actions |
| Offline data per cell | 4,000 transitions |
| Offline episode structure | 160 complete episodes x 25 transitions |
| Train/holdout | Episode-preserving 80/20: 128/32 episodes, approximately 3,200/800 rows |
| Evaluation | 5 seeds x 4 episodes x 50 steps = 20 episodes and 1,000 decisions per method-cell |
| Discount | 0.95 |
| Public dynamics ensemble | 5 ridge-linear bootstrap members, ridge `1e-3` |
| Threads/resources | One numerical thread and one allocated CPU per task |

The 576-row arithmetic is:

```text
9 populations x 4 families x 2 noise levels x 2 reward modes = 144 data cells
144 cells x 4 methods = 576 method rows
```

Within a cell, all four methods use the same public dataset path and content hash. Exact reuse of
the ecological experiment files succeeded for 140/144 cells. In the four Crab-eating fox/theta
cells, the ecological files contained 4,005 rows: 160 complete episodes plus five partial rows.
The general-RL package keeps the exact 160 complete ecological episodes and excludes the five
partial rows, preserving the registered 4,000-transition complete-episode design. Source and
package hashes are both recorded.

## At-a-glance comparison

| Method | Paper-level idea | Current uncertainty object | Current decision mechanism | Online adaptation |
|---|---|---|---|---|
| RefPlan-inspired | Reflect on deployment evidence, then plan while marginalizing model belief | Posterior over five public dynamics members | Prior-guided sequence proposals; posterior-marginalized pessimistic scoring | Model posterior updates after each public transition |
| OGSRL-inspired | Learn an OOD guardian and optimize a constrained policy in the guarded model | Support/OOD risk plus public low-abundance occupancy | Trained Lagrangian softmax actor with OOD and safety duals | Public belief updates; policy is fixed after offline training |
| BA-MCTS-inspired | Treat model uncertainty as a BAMDP and update belief inside search | Categorical model belief at every simulated tree history | 256-simulation, depth-8 Bayes-adaptive MCTS | Belief updates within each simulated history and from public observations |
| EVD pessimism | Penalize actions with uncertain long-run value; Delphic-motivated only | Variance across 20 episode-bootstrap conservative Q estimators | Greedy `mean(Q)-0.1*Var(Q)` | No online model update; fitted ensemble evaluated at current public features |

## 1. RefPlan-inspired

### Original paper idea

Reflect-then-Plan frames offline model-based planning through a doubly Bayesian lens:

1. a conservative offline policy supplies a prior over plans;
2. a belief over possible dynamics models is updated from deployment observations;
3. planning marginalizes the current dynamics belief rather than committing to one model;
4. uncertainty in predicted return is penalized;
5. MPPI-style probabilistic inference combines the prior and rollout returns.

The identity-defining mechanism is not merely estimating a posterior. The posterior must affect the
deployment plan.

### Current implementation

- Fits a five-member public observation-space dynamics ensemble from the 3,200-row training split.
- Fits a separate standardized multinomial public behavior-policy prior.
- Mixes that prior with 10% uniform exploration.
- Generates action-sequence proposals sequentially. At every planning depth, the prior is
  reevaluated on the simulated posterior-predictive public history/state.
- Preserves explicit first-action coverage so every action can still be evaluated.
- Maintains a normalized posterior over dynamics members.
- Updates that posterior from public transition likelihood after deployment observations.
- Scores every candidate sequence under every model member using common random numbers.
- Uses posterior-weighted mean predicted value minus a posterior-weighted return standard-deviation
  penalty.
- Replans after every real observation.

Current primary settings:

| Setting | Value |
|---|---:|
| Dynamics members | 5 |
| Planning horizon | 5 |
| Candidate sequences | 96 |
| Public belief particles | 32 |
| Prior uniform floor | 0.10 |
| Return pessimism coefficient | 0.5 |
| Discount | 0.95 |

### Fidelity and adaptation

Retained:

- separate policy prior and dynamics posterior;
- deployment posterior update;
- belief marginalization in planning;
- return-uncertainty pessimism;
- prior-guided trajectory proposal.

Adapted or approximated:

- linear public dynamics replace neural latent dynamics;
- a calibrated public behavior policy replaces a deep conservative offline-RL prior;
- discrete sequence search replaces continuous MPPI actions;
- reflected-score argmax approximates MPPI softmax weighting;
- short horizon and finite proposal count are computational choices.

Allowed claim: **RefPlan-inspired history-conditioned public-prior planner with a separate,
deployment-updated model posterior.** It is not an official reproduction.

### Improvements

- Replace the linear dynamics with a recurrent or latent history-conditioned model.
- Train a genuinely conservative offline-RL policy prior rather than behavior logistic regression.
- Implement MPPI softmax weighting and temperature sensitivity.
- Test horizons and sequence counts return-blindly.
- Calibrate dynamics posterior likelihood and return uncertainty.

## 2. OGSRL-inspired

### Original paper idea

Offline Guarded Safe RL separates three tasks:

1. learn a guardian defining the in-distribution state-action region;
2. restrict or guard the learned transition model outside supported regions;
3. run constrained policy optimization inside the guarded model, accounting for both OOD exposure
   and domain safety cost.

The original implementation uses KDE/kNN components and CPO, but the paper permits other
constrained optimizers. A one-step greedy action rule without trained constrained policy
optimization would not preserve the method's identity.

### Current implementation

- Fits the shared five-member public dynamics ensemble.
- Fits a public kNN support guardian.
- Trains a linear softmax actor, critic, and two Lagrange multipliers for 30 iterations.
- Uses one dual for OOD/support exposure and one for public low-abundance occupancy.
- The public low-abundance scale is the 20th percentile of positive training observations:

```text
s_low = Q_0.20(o_train > 0)
c(o_next) = clip((s_low-o_next)/s_low, 0, 1)
```

- Uses the normalized discounted 25-step occupancy

```text
C_25 = sum_t [gamma^t / sum_j gamma^j] c(o_(t+1)).
```

- The safety budget is exactly the mean `C_25` of complete training behavior episodes. It is not
  clipped or selected from performance.
- Actor training applies cost inside each sampled predictive trajectory and averages complete
  pathwise occupancies.
- Deployment uses 256 deterministic-common-random-number predictive paths for each forced first
  action, balanced over the five model members, with fitted public predictive residuals and actor
  continuation actions.
- Deployment compares expected pathwise `C_25` with the same training-derived budget.

Current primary settings:

| Setting | Value |
|---|---:|
| Dynamics members | 5 |
| Guardian | Public kNN support guardian |
| Actor training iterations | 30 |
| Actor rollout starts | 128 |
| Safety-cost horizon | 25 |
| Deployment predictive paths | 256 per forced first action |
| Discount | 0.95 |

### Fidelity and adaptation

Retained:

- explicit OOD guardian;
- guarded public model;
- multi-step constrained policy optimization;
- separate OOD and safety-cost constraints;
- learned policy rather than one-step greedy planning.

Adapted or approximated:

- Lagrangian policy-gradient actor replaces trust-region CPO;
- linear observation-space dynamics replace richer paper models;
- public kNN support replaces the exact guardian estimator/configuration;
- public observation-derived low-abundance cost replaces unavailable private safety cost;
- aggregate observation-space residual cannot separate ecological process and survey noise;
- 25-step receding risk is matched to logged episode length inside a 50-step evaluation.

Allowed claim: **OGSRL-inspired guarded constrained actor using public OOD and low-abundance
proxies.** It carries no guarantee for the private ecological safety objective.

### Improvements

- Use a latent state-space model that separately represents process and observation uncertainty.
- Port CPO or another trust-region constrained optimizer.
- Calibrate kNN/KDE support and OOD thresholds across populations.
- Test 128/256/512 predictive paths and actor iterations return-blindly.
- Study alternative public risk proxies without selecting them from policy returns.
- Report when safety or OOD constraints are slack rather than forcing every dual to bind.

## 3. BA-MCTS-inspired

### Original paper idea

Bayes-Adaptive MCTS treats offline model uncertainty as a Bayes-adaptive MDP. The planning state is
augmented with a belief over possible world models. After every simulated transition, the belief is
updated by that model's likelihood. MCTS therefore reasons about both control and information:
future observations may change which model is trusted.

The paper's continuous-state/action algorithm combines PUCT, double progressive widening, deep
ensembles, and an outer policy/value distillation loop. It explicitly explains why root sampling is
not sufficient in its DPW setting.

### Current implementation

- Uses the shared five-member public dynamics ensemble as a categorical model hypothesis set.
- Carries log model weights inside every simulated tree history.
- Samples a successor model from the current node belief.
- Updates member weights inside the tree using public transition likelihood after the simulated
  observation.
- Uses numerically stable log normalization and a defined zero-likelihood fallback.
- Uses discrete management actions, so action progressive widening is unnecessary.
- Aggregates continuous public observations into tree buckets.
- Runs 256 simulations to depth 8 at every real decision.
- Replans at every evaluation step.

The practical update is conceptually

```text
b_next(m) proportional to b(m) * p_m(o_next | history, action).
```

Current primary settings:

| Setting | Value |
|---|---:|
| Model members | 5 |
| Simulations per decision | 256 |
| Search depth | 8 |
| UCB coefficient | 1.25 |
| Discount | 0.95 |
| Registered sensitivity | 128/depth 5 and 512/depth 12 subsets |

### Fidelity and adaptation

Retained:

- BAMDP-style augmented model belief;
- in-tree likelihood-based belief update;
- model selection from the current simulated belief;
- uncertainty-aware tree search.

Adapted or omitted:

- ridge-linear public ensemble replaces deep world models;
- finite discrete actions remove action-DPW;
- observation bucketing replaces full continuous state-DPW machinery;
- no outer policy/value-network distillation loop;
- evaluation-time search is used directly;
- compute budget is much smaller than AlphaZero-scale planning.

Allowed claim: **BA-MCTS-inspired public ensemble tree search with genuine in-tree Bayesian model
belief updates.** It is an evaluation-time adaptation, not the full RL+Search system.

### Improvements

- Increase simulations/depth on a preregistered subset.
- Add learned policy/value priors and outer search distillation.
- Use richer probabilistic dynamics and calibrated member likelihoods.
- Improve continuous-observation tree handling.
- Report tree depth, node count, belief movement, and search disagreement for each cell.

## 4. Ensemble value-disagreement pessimism (Delphic-motivated)

### Original Delphic paper idea

Delphic Offline RL addresses nonidentifiable hidden confounding. It fits multiple latent causal
worlds that are observationally compatible with the same behavior data. It evaluates a policy in
each compatible world, defines Delphic uncertainty as cross-world Q variance, and subtracts a
pessimism penalty so the learned policy avoids decisions whose value depends strongly on
unidentifiable counterfactual assumptions.

The ecology dataset has a relevant confounding pattern because the data-collection behavior policy
can use privileged true abundance while hidden methods observe only noisy surveys. However, a true
Delphic implementation would still require coherent latent generative worlds fitted to the same
observable trajectory distribution.

### Current implementation

The current method deliberately does not claim to implement those latent worlds.

- Draws 20 episode-level bootstrap resamples of the same training split.
- Fits one independent linear conservative Q estimator per bootstrap.
- Uses 35 fitted-Q iterations, ridge regularization, and CQL-style zero pseudo-targets.
- Members differ only through their registered bootstrap data and seed.
- Adds no random perturbation directly to Q.
- Computes empirical ensemble mean and variance:

```text
Q_bar(o,a) = mean_m Q_m(o,a)
V_Q(o,a) = mean_m (Q_m(o,a)-Q_bar(o,a))^2
score(o,a) = Q_bar(o,a) - 0.1*V_Q(o,a).
```

- Chooses the greedy action under that pessimistic score.
- The registered penalty is operational but sparse: in the audited Amur/vulture holdouts it changed
  approximately 0.125--0.375% of actions compared with pure ensemble-mean Q.

Current primary settings:

| Setting | Value |
|---|---:|
| Bootstrap Q members | 20 |
| Fitted-Q iterations | 35 |
| CQL-style coefficient | 0.5 |
| Disagreement coefficient | 0.1 inverse-Q units |
| Discount | 0.95 |

### Fidelity and adaptation

Retained from the motivating Delphic idea:

- pessimism from disagreement in long-run action value;
- cross-estimator Q variance rather than only one-step transition variance.

Not retained:

- no latent confounder model;
- no observationally compatible causal worlds;
- no ELBO or trajectory-likelihood compatibility fitting;
- no valid claim that the variance is Delphic uncertainty.

Allowed claim: **episode-bootstrap conservative fitted-Q ensemble with empirical value-disagreement
pessimism; Delphic-motivated at the idea level only.**

### Improvements

- For a real Delphic adaptation, fit discrete-latent generative worlds with behavior, transition,
  and reward components under episode-level marginal likelihood or ELBO.
- Require each world to match held-out observable behavior and transition distributions.
- Derive Q only through each fitted world's transition/reward model.
- Separate world validity from the magnitude of detected ambiguity.
- Otherwise retain the honest EVD name and test nonlinear Q ensembles, bootstrap count, and
  disagreement scaling.

## Common hidden-demographics adaptations

| Benchmark requirement | Adaptation | Scientific consequence |
|---|---|---|
| `r`, `K`, family, and private threshold unavailable | Public observation/history features only | Methods learn structural uncertainty from data rather than privileged ecology |
| No exact latent state at deployment | Learned public observation filter/cache | Partial observability may dominate weak results |
| Discrete conservation actions | Finite 11-action planning/control | Continuous-action machinery is omitted |
| Offline data only | All models/policies fit from fixed 4,000-transition logs | Coverage and behavior-policy bias matter |
| Private safety objective | Public low-abundance and OOD proxies | Safe-mode results remain information-limited; no private guarantee |
| Finite evaluation | 50-step episodes | Planning horizons are approximations inside the longer task |
| Matched compute deadline | Linear models and compute-scaled planning | Paper identity can be retained while neural scale is not reproduced |

## Paper-closeness assessment

| Method | Defining mechanism retained? | Current closeness | Main reason it is not a reproduction |
|---|---|---|---|
| RefPlan-inspired | Yes | Core-preserving adaptation | Linear models/prior and argmax sequence search replace deep latent MPPI system |
| OGSRL-inspired | Yes | Core-preserving adaptation | Lagrangian linear actor and public proxy replace CPO/domain safety model |
| BA-MCTS-inspired | Yes | Core-preserving evaluation-time adaptation | No deep ensemble or outer policy/value distillation |
| EVD pessimism | Only the broad pessimistic-value-disagreement idea | Idea-level inspiration | No latent compatible worlds or confounding model |

## Verification completed

- 203/203 tests pass in the frozen Phase 2E runtime.
- Hidden privacy lifecycle is tested through `fit -> act -> observe -> act`.
- Private-field interventions leave fitted public artifacts and actions invariant.
- RefPlan posterior affects actions and the history-conditioned prior affects later proposal steps.
- OGSRL actor/duals train; pathwise cost is applied before predictive averaging; common-random-number
  and Jensen tests pass.
- BA-MCTS model belief moves inside simulated histories and concentrates on consistent members.
- EVD members use distinct episode bootstraps, empirical variance is exact, and a constructed case
  changes action under the registered penalty.
- One-thread numeric parity passes.
- The 64-row canary completed with 64/64 accepted validity receipts and no return fields opened.

## Matched full package and provenance

Package: `general_phase2e_full_sigma01_02_20260720_v1`.

| Artifact | SHA-256 |
|---|---|
| 576-row manifest | `1526ce08dcf1b1d148232c075d41dbd78f0cfa2d7119cff9e330f965b6451df4` |
| Registration | `15fd7aa1c3ddc460fd255e5d66518d49244f81cf3dbbd291a5bdfe52b376b37a` |
| 144-cell dataset registry | `474d1a65d2e5ec28c741f5b7ac5691be891712e753ee6a9a6590337d62f5f0c4` |
| Frozen code tree | `f615d363a525422abfb983f0eee7c433b009fff81a4d0ae4925bfdaba0f3aa72` |

Projected aggregate compute is 43.7052 core-hours. Receipt-based ideal elapsed estimates are about
2 h 44 min at concurrency 16, 1 h 22 min at concurrency 32, and 41 min at concurrency 64, excluding
queue and scheduler overhead.

## Interpretation guardrails

- Do not call any method an exact or official reproduction.
- Use `-inspired` for RefPlan, OGSRL, and BA-MCTS.
- Never call EVD's variance Delphic uncertainty or its members compatible worlds.
- Do not characterize safe-mode weakness as intrinsic method failure: the private safety objective
  is deliberately hidden.
- Do not attribute weak performance solely to the algorithm until 4,000-vs-8,000 data adequacy and
  key compute sensitivities are checked.
- Report individual methods, not only a cellwise "Best general" envelope.
- "Best general" is an oracle cellwise maximum, not one deployable policy.
- Keep trajectory plots illustrative rather than headline statistical evidence.

## Recommended supervisor-slide sequence

1. Hidden-demographics problem and common information boundary.
2. One-row comparison of the four uncertainty objects.
3. One slide per method: paper idea -> retained core -> ecology adaptation.
4. Shared settings and matched data grid.
5. Fidelity/adaptation table.
6. Verification and privacy gates.
7. Limitations and next improvements.
8. Results only after return-blind acceptance and explicit outcome authorization.

