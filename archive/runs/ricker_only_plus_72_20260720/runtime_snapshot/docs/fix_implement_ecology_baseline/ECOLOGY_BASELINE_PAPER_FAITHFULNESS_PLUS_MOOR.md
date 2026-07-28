# Paper-Faithfulness Problem: PLUS and MOOR Native Baselines

## Purpose and governing decision

The ecology baselines should follow their original papers as closely as possible. A
benchmark-specific change is acceptable only when the original fishery problem and
the present conservation benchmark are genuinely incompatible. Every such change
must be named, justified, and tested. Convenience, a shared helper implementation,
or reduced coding effort is not by itself a scientific justification for replacing a
paper's core algorithm.

The completed hidden-r,K experiment must remain frozen. Its results are valid for
the implementations that were actually run, but corrected paper-faithful baselines
would be new methods requiring new names, provenance, and experiments. Old results
must not be silently relabeled.

## Primary sources

### PLUS

- Memarzadeh & Boettiger, *Adaptive management of ecological systems under partial
  observability*: https://www.sciencedirect.com/science/article/pii/S0006320717316737
- Official supporting information:
  https://ars.els-cdn.com/content/image/1-s2.0-S0006320717316737-mmc1.pdf
- Archived authors' software, `pomdpplus` 0.2.0:
  https://zenodo.org/records/1161521

### MOOR

- Ju et al., *Model-based offline reinforcement learning for sustainable fishery
  management*: https://onlinelibrary.wiley.com/doi/10.1111/exsy.13324
- Author-hosted PDF:
  https://rdl.cecs.anu.edu.au/papers/ES23-Model%E2%80%90basedOfflineReinforcementLearningForSustainableFisheryManagement.pdf
- The paper states that its synthetic data and code are at:
  https://github.com/jujun622/moorfisherys

The MOOR paper contains its additional experiments in Appendices A and B; I did not
find a separate supplementary-methods document comparable to the PLUS supplement.

## Independent audit status

`AUDIT_ECOLOGY_BASELINE_PAPER_FAITHFULNESS_PLUS_MOOR.md` independently checked the
paper-facing claims and judged this note **sound with minor revisions**. That audit
read the full MOOR paper and archived PLUS software, but could not access the frozen
code-server snapshot. Consequently, the descriptions of the original algorithms are
independently source-checked, while the exact frozen hidden-mode mechanics remain
attributed to the code-server forensic reports pending direct replication.

## PLUS: what the original method requires

Published PLUS represents ecological uncertainty with a finite set of candidate
mechanistic POMDP models. Each candidate defines a transition model `T` and
observation model `O`, is solved separately, and supplies candidate-specific action
values. Deployment maintains both an abundance belief and a posterior over candidate
models. The candidate posterior is updated using the likelihood of each new
observation under that candidate, and decisions combine candidate-specific values
using the posterior weights.

The paper, supplement, and archived software use candidates that are actual ecological
models with different parameter values. Candidate grids vary selected parameters
rather than necessarily varying every ecological and noise parameter simultaneously.
For example, the archived tuna workflow varies growth rate and measurement noise
while holding some other quantities fixed. The supplement studies candidate-count
sensitivity from smaller to larger Ricker candidate sets, and the real application
uses on the order of one hundred candidate parameterizations. These are candidate
models, not offline observations and not separately trained black-box regressors. The
exact grid and count are design choices that must be reported.

The paper's demonstrated experiments generally place all candidates within one
structural family. Its discussion permits different structures in the candidate set,
but a Ricker/Allee/theta/regime candidate bank is therefore an extension of PLUS,
not an exact reproduction. This extension is defensible only if every candidate is
still a real mechanistic model and the closed-world assumption is disclosed.

## Code-server-reported problem in hidden `plus_native`

The code-server forensic report classifies hidden `plus_native` as four fitted,
action-conditional polynomial transition regressions followed by online Bayesian
weighting. The labels `Ricker`, `Allee`, `theta`, and `regime` choose polynomial basis
functions; they do not instantiate those ecological equations.

The implementation mechanics in this section are attributed to the code-server
forensic report on the frozen snapshot (base commit `5f9cf32a69d47e2d31b2ed6ee30ebbcfe84b8536`,
reported non-pyc tree digest
`cfe0df14bc1f2bfdcf83098f9fe252def9d24e303db0eeefb10509d7b46efab8`). The frozen
tree is not mounted in this local review environment, so these probes have not been
independently reproduced here. They should be replicated on the frozen tree before
the detailed mechanics are cited as independently established facts.

In particular, the implementation does not fit `r`, `K`, an Allee threshold, a theta
exponent, or regime-switching probabilities. The regime candidate is not a genuine
regime-switching model. There is one fitted regression object per label, and its model
coefficients remain fixed online. The implementation uses a shared public reward
surrogate and a tabular QMDP-style solution rather than the paper's mechanistic
candidates and candidate-specific SARSOP solutions.

This implementation preserves the high-level posterior model-averaging skeleton of
PLUS, but not the core candidate-model semantics. It should be described as:

> PLUS-inspired posterior model averaging over fitted polynomial transition templates.

It should not be called a reproduction, faithful adaptation, or direct evaluation of
published PLUS. The completed results can support claims about this implemented
PLUS-inspired baseline only.

## Required PLUS correction

A corrected baseline should:

1. Construct actual mechanistic candidates. For this benchmark these may include
   parameterized Ricker `(r,K)`, Allee `(r,K,C)`, theta-logistic `(r,K,theta)`, and
   regime-switching models with explicit latent-regime transition probabilities.
2. Represent parameter uncertainty within a family using a grid or posterior samples,
   rather than only one fitted object per family name.
3. Use sanitized, temporally ordered public histories to calculate a mechanistic
   likelihood or posterior. It must not access the true family, true `r,K`, private
   action tables, safety threshold, or evaluator state.
4. Initialize candidate weights either uniformly, as in the archived `pomdpplus`
   default, or from a historical-data likelihood/posterior. This is a legitimate
   preregistered variant, not itself a fidelity defect. The critical requirement is
   Bayesian evidence updating over mechanistic candidates.
5. Build candidate-specific transition and observation models, solve each candidate,
   and update candidate weights from public action-observation likelihoods as in PLUS
   Appendix C.
6. Keep candidate parameters fixed during deployment unless online parameter learning
   is explicitly introduced and labelled as an extension beyond PLUS.
7. Use SARSOP when feasible. A different POMDP planner is an acceptable computational
   adaptation only if it preserves belief-state planning and is disclosed and tested;
   QMDP is a substantive approximation, not a silent implementation detail.
   Original PLUS itself averages separately solved candidate values rather than solving
   an exact joint state-by-model Bayes-adaptive POMDP; the fidelity bar is therefore
   belief-state planning within candidates plus Bayesian model averaging, not an exact
   Bayes-adaptive optimum.
8. Report candidate count and a small candidate-count sensitivity analysis. No exact
   candidate count is required.

## MOOR recheck: what the original method actually does

MOOR genuinely uses offline historical data to learn one mechanistic fishery POMDP.
It is not a near-data-free parameter-known planner and it is not posterior averaging
over a candidate bank.

MOOR's overall procedure (Algorithm 1, expanded with Sections 4.3--4.5) is:

1. Receive a time-ordered catch series `c_t` and effort/action series `e_t`.
2. Impute missing effort values, while excluding missing catches from the loss.
3. Fit one mechanistic continuous POMDP. A Beverton-Holt model fits
   `(rho,K,B0,q)`; the misspecified Schaefer alternative fits `(r,K,B0,q)`.
4. Minimize expected trajectory-level catch SSE. For stochastic dynamics, the
   expectation is approximated by simulated latent biomass trajectories and optimized
   with automatic differentiation and stochastic L-BFGS.
5. Normalize catch and effort, use multiple random initializations, and retain the
   fit with the smallest SSE.
6. Discretize state, action, observation, transition, and reward models using Monte
   Carlo simulation.
7. Plan online in the learned POMDP with DESPOT while maintaining a belief over latent
   biomass.

The paper uses one 50-step effort/catch history per fit and three independently
generated datasets per environment for evaluation. This is evidence that MOOR learns
from offline data, not a universal data-budget prescription. The paper also finds that
wide effort variation is important for identifiability. Therefore, the benchmark must
report action coverage and must preserve episode order if it claims to use MOOR's
trajectory fitting method.

An important correction to the local implementation plan is that original MOOR makes
catch both the observation and reward. It does not use this benchmark's latent-state
conservation reward. A shared benchmark reward is a necessary objective adaptation
for fair evaluation, but MOOR must not be cited as provenance for that reward choice.
Concretely, `29_6_Real_Ecology_Setting_Implementation_Plan.tex` lines 298--299 state
that PLUS and MOOR both reward the latent state. More precisely, MOOR rewards catch
`c = q e B`, while archived PLUS uses harvest utility such as `min(x,h)`. Both are
state-and-action yield objectives, not the benchmark's abundance-conservation reward.

## Necessary versus unacceptable MOOR adaptations

Necessary adaptations include:

- replacing fishing effort with the benchmark's 11 conservation actions;
- replacing catch observations with public abundance surveys;
- replacing catch reward with the common benchmark reward and cost objective;
- mapping MOOR's mechanistic transition to the benchmark's population and action
  semantics;
- handling several episodes rather than one uninterrupted 50-year fishery history;
- choosing a tractable belief-state planner under the registered compute budget;
- keeping all hidden demographics, family labels, safety information, and evaluator
  state private.

These adaptations should preserve MOOR's defining core: fit one explicit mechanistic
latent-state model from ordered offline histories, construct its POMDP transition and
observation models, and plan under a belief in that learned model.

Unacceptable silent substitutions include:

- replacing the ecological equation with an unconstrained polynomial or generic ridge
  transition regression while retaining the name MOOR;
- fitting independent transitions while discarding trajectory order and latent-state
  propagation;
- using true `r,K`, family, state, or action-effect tables in hidden mode;
- calling an exact-table parameter-known planner original MOOR, since original MOOR
  estimates its model from data;
- using a point observation as the true state without a belief/filtering treatment;
- changing DESPOT to QMDP or MPC without reporting the planning approximation;
- pooling data in a way that gives a nominally single-population mechanistic model
  incompatible population scales without explicit contextualization.

Using one Ricker model for all four benchmark families can be a deliberate
misspecification experiment, analogous in spirit to MOOR's Schaefer-on-Beverton-Holt
study. It must be stated as such. It does not test whether a correctly specified MOOR
adaptation can fit Allee, theta, or regime dynamics.

## Code-server-reported problem in `moor_native`

According to the code-server forensic audit of the frozen snapshot, the overall
classification is **C**, with an important full-versus-hidden distinction. The exact
hidden-mode mechanics and dependence probes below are agent-reported and remain
pending independent reproduction on the frozen tree:

- **Hidden mode: C.** It is generic offline action-conditional regression plus online
  state filtering, not paper-faithful MOOR.
- **Full mode: D.** It is a parameter-known, table-built Ricker upper bound that fits
  only one scalar `K` by grid search. It is not original MOOR.

Hidden mode fits one separate ridge regression for each action:

```text
log(1 + o_next/s) = beta[a,0] + beta[a,1] log(1 + o/s).
```

Its learned quantities are `beta[a,0]`, `beta[a,1]`, and one residual standard
deviation per action. It does not fit `r` or `rho`, `K`, `B0`, catchability `q`, a
density-dependence parameter, or a mechanistic action effect. It reads observations,
next observations, and action identifiers, but does not use episode IDs, timestamps,
termination, population identity, or history in the dynamics loss.

The fitted dynamics treats transitions as exchangeable rows. Reversing complete
episodes or reversing rows within each episode left the policy unchanged. Thus the
implementation has neither MOOR's ordered latent-biomass propagation nor its
trajectory-level expected catch SSE. The 11 actions are independent regression
buckets rather than a mechanistic effort or action-response parameterization.

The implementation projects the regression onto 61 abundance bins, constructs
Gaussian transition kernels, and uses tabular value iteration. At deployment it
selects actions by averaging fully observable Q-values under the current abundance
belief, which is QMDP-style planning rather than DESPOT. It does perform an online
public action-observation belief update, but its coefficients, transition model,
reward table, and Q-values remain fixed.

The representative audited hidden cell contained 4,000 rows in 160 episodes. MOOR
used 3,200 rows from 128 episodes and held out 800 rows from 32 episodes. Episode
boundaries affected only the train/holdout split, not the regression objective.

Public-data dependence and privacy are genuine. Perturbing public next observations
changed fitted parameters, transitions, Q-values, and parts of the greedy policy.
Blocking private table loaders and relabelling private family metadata did not change
hidden fitted artifacts. These tests establish data dependence and no obvious private
table/family leakage; they do not establish MOOR paper faithfulness.

Full mode uses true configuration quantities and table-resolved action effects. It
fits a scalar `K`, but its evaluation filter is built from a separate exact-table
default-`K` solver while planning uses the later fitted-`K` solver. Therefore full
filtering and planning are not even based on the same model. This arm remains useful
as a parameter-known benchmark comparison, but it must not be called original MOOR.

## Paper-versus-implementation summary

| Component | Published MOOR | Hidden frozen implementation (code-server reported) |
|---|---|---|
| Dynamics | Explicit Beverton-Holt or Schaefer model | Per-action log-linear regression |
| Parameters | `rho/r, K, B0, q` | `beta[a,0], beta[a,1], sigma[a]` |
| Data structure | One ordered historical trajectory | Exchangeable rows from many episodes |
| Objective | Expected trajectory catch SSE | One-step ridge prediction loss |
| Optimizer | Monte Carlo stochastic L-BFGS, multiple starts | Closed-form ridge solve |
| State treatment | Latent biomass POMDP | Binned transformed observation scale |
| Discretization | Monte Carlo POMDP construction | Gaussian kernels on 61 bins |
| Planner | DESPOT | Value iteration plus belief-weighted Q-values |
| Actions | Mechanistic fishing effort/catchability | Eleven categorical regression buckets |

## Supported report wording

Subject to replication of the code-server findings on the frozen tree, the existing
sentence "MOOR-native fits one public model from sanitized transitions" is technically
true but scientifically too vague. A more precise attributed description is:

> The code-server forensic audit reports that hidden `moor_native` fits one
> action-conditional log-linear observation-transition
> model from 3,200 sanitized training transitions, constructs a reward table using the
> shared public surrogate, solves the resulting 61-state tabular model by value
> iteration, and selects actions using belief-weighted Q-values.

It is misleading to call the frozen implementation paper-faithful MOOR, a fitted
Ricker model, mechanistic MOOR, DESPOT planning, or ecological-parameter fitting. The
defensible label is **benchmark-native MOOR-inspired regression baseline**, provided
that the methodological departure is disclosed.

## Consequences for interpretation

The completed headline comparison is between the general learners and two
ecology-inspired approximations, not direct implementations of published PLUS and
MOOR. This does not invalidate the measured returns of the frozen implementations,
but it changes which scientific claim those returns support.

Corrected baselines require a new registered run. The current known-r,K/full results
remain scientifically useful as parameter-known comparison arms, but should not be
presented as reproductions of algorithms whose papers learn under uncertainty.

## Required next steps

1. Preserve the completed run and label its methods according to what they actually
   implement.
2. Independently reproduce the reported implementation probes against the frozen tree
   before citing the detailed hidden-mode mechanics as established facts.
3. Implement corrected PLUS and MOOR baselines from their mechanistic equations. For
   MOOR this requires interpretable ecological and action-response parameters,
   episode-ordered latent-state propagation, trajectory-level stochastic fitting with
   bounds and multiple starts, model-consistent POMDP construction, and DESPOT or a
   clearly declared belief-state planning approximation.
4. Preregister all unavoidable benchmark adaptations and planner approximations.
5. Add unit tests for parameter recovery, posterior/belief updates, private-information
   blocking, trajectory ordering, and action-coverage sensitivity.
6. Run corrected baselines under new method identifiers and compare them with the
   frozen results rather than overwriting them.
