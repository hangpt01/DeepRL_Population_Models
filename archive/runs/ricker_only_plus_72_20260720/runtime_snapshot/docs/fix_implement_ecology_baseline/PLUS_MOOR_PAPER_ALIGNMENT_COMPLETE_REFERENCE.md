# PLUS and MOOR in the Hidden-Demographics Benchmark

## Consolidated paper-alignment, implementation, and adaptation reference

This document consolidates and supersedes the overlapping discussion in:

- `ECOLOGY_BASELINE_PAPER_FAITHFULNESS_PLUS_MOOR.md`
- `REVIEW_PROVISIONAL_PAPER_ALIGNED_PLUS_MOOR_IMPLEMENTATION.md`
- `DECISIONS_FOR_CORRECTED_PLUS_MOOR_PLAN.md`

It is intended to remain a detailed scientific and implementation reference. It separates what the published methods do, what the completed frozen experiment actually ran, why a provisional correction was rejected, and what the approved corrected implementation must do.

## 1. Status and evidence boundary

Three generations must not be conflated.

| Generation | Status | What may be claimed |
|---|---|---|
| Frozen hidden-r,K experiment | Completed and scientifically interpretable for the methods actually run | Results compare general learners with PLUS-inspired and MOOR-inspired approximations, not paper-faithful PLUS and MOOR |
| Provisional mechanistic repair | Implemented and briefly launched, then cancelled | Scientifically void because two blocking model/objective mismatches were confirmed; do not report its results |
| Corrected design in this document | Approved target specification | Paper-aligned adaptation once implemented, tested, and reproduced on the frozen tree; not yet an independently verified completed implementation |

Accordingly, the phrase **current version** below means the approved corrected design. It must not be described as an exact reproduction of either paper. A defensible eventual label is **paper-aligned adaptation**: the defining inferential and decision-theoretic structure is restored, while unavoidable benchmark adaptations are named and tested.

The detailed mechanics of the frozen implementation were reported by the code-server agent from the inaccessible frozen tree. They should be replicated directly on that tree before final publication wording is frozen.

## 2. Bottom-line scientific conclusion

The old hidden baselines preserved superficial motifs from PLUS and MOOR but changed their model class and inference sufficiently that they were not reproductions:

- old hidden PLUS used posterior weighting over action-conditional polynomial regressions rather than a bank of mechanistic ecological POMDPs;
- old hidden MOOR used exchangeable one-step per-action ridge regressions rather than fitting one mechanistic population model to ordered trajectories;
- both were solved with tabular/QMDP-like approximations that did not preserve the original belief-state planning logic.

The corrected design restores the identity of each method:

- **PLUS** remains Bayesian online model discrimination over separately solved, fixed mechanistic candidate POMDPs;
- **MOOR** remains offline trajectory-level fitting of one mechanistic population POMDP followed by planning in that fitted model.

The corrected methods share the benchmark's data, action semantics, observation process, reward, horizon, and evaluation protocol. They differ in the same conceptual way as the papers: PLUS carries model uncertainty online; MOOR commits to one fitted model before deployment.

## 3. Published PLUS: invariant algorithmic core

The paper and archived `pomdpplus` implementation define PLUS through the following structure.

1. Construct a finite candidate bank of mechanistic POMDPs. Each candidate has a complete transition model, observation model, reward model, and ecological parameterization.
2. Solve every candidate POMDP independently. The original implementation uses SARSOP alpha-vectors.
3. Maintain a state belief within every candidate model and a posterior probability over candidates.
4. After each action and observation, update each within-model state belief and update the model posterior using the candidate's transition-observation evidence.
5. Keep candidate parameters fixed online. Learning means Bayesian model discrimination, not continuous online parameter fitting.
6. At decision time, combine the candidate-specific policy values using the current model posterior and select the maximizing action.

In compact notation, for candidate models $M_j$, candidate state beliefs $b_t^j$, and model probabilities $w_t^j$, the evidence update is

\[
w_{t+1}^j \propto w_t^j\,
p(o_{t+1}\mid b_t^j,a_t,M_j),
\qquad \sum_j w_{t+1}^j=1.
\]

The action is chosen from posterior-weighted candidate values, schematically

\[
a_t^* = \arg\max_a \sum_j w_t^j Q_j(b_t^j,a).
\]

A uniform initial candidate prior is the archived default and is not a fidelity problem. The important requirements are mechanistic candidates, model-specific evidence, Bayesian posterior updating, and candidate-specific POMDP solutions.

The published examples include parameter grids within ecological model families. A cross-family bank is compatible with the method but is an extension, not a mandatory property of the original experiment. Likewise, bootstrap-derived candidate parameterizations are a defensible adaptation if preregistered and disclosed.

## 4. Published MOOR: invariant algorithmic core

MOOR has a different inferential commitment.

1. Use an ordered historical effort/catch trajectory, preserving temporal dependence.
2. Fit one mechanistic biomass model, such as Beverton-Holt or Schaefer, with parameters including growth, carrying capacity, initial biomass, and catchability.
3. Minimize a trajectory-level squared-error objective. Latent biomass trajectories are propagated through the mechanistic model, with Monte Carlo integration over process uncertainty.
4. Normalize variables, enforce parameter bounds, use automatic differentiation and stochastic L-BFGS with multiple starts, and retain the minimum-training-loss solution.
5. Convert the fitted continuous model to a POMDP through Monte Carlo discretization.
6. Plan under partial observability; the paper uses DESPOT.

In the fishery paper, catch is simultaneously the observation and reward. Measurement noise is not independently sampled inside the trajectory SSE. The experimental use of 50 time points and three data sets is not a universal MOOR definition.

The defining identity is therefore: **one mechanistic model fitted offline to ordered trajectories, then used consistently for discretization, filtering, and planning**.

## 5. The conservation benchmark being solved

The benchmark is not a fishery. It has 11 conservation interventions, noisy abundance surveys, finite episodes, cumulative habitat-capacity effects, and a common conservation reward. To avoid leaking private demographics, population size is normalized by a public survey-derived scale $S$, such as the median positive survey:

\[
x_t=N_t/S,\qquad k_t=K_t/S,\qquad y_t=o_t/S.
\]

The public action mechanism category is known, but numerical ecological effects are private. Let

- $u_a$ be a translocation increment applied before growth;
- $d_a$ be a cumulative capacity increment;
- $r_a$ be a signed rate effect;
- $g_a=\max(r_a,0)$ and $h_a=\max(-r_a,0)$.

Then

\[
k_{t+1}=\operatorname{clip}(k_t+d_a,k_0,k_{\max}),
\qquad m_t=\max(x_t+u_a,0).
\]

The candidate mechanistic families are:

**Ricker**

\[
x_{t+1}=m_t\exp\!\left[g_a\left(1-\frac{m_t}{k_{t+1}}\right)-h_a\right].
\]

**Allee**

\[
x_{t+1}=m_t\exp\!\left[g_a\left(1-\frac{m_t}{k_{t+1}}\right)
\left(\frac{m_t}{C}-1\right)-h_a\right].
\]

**Theta-logistic**

\[
x_{t+1}=\left[m_t+g_a m_t
\left(1-\left(\frac{m_t}{k_{t+1}}\right)^\theta\right)\right]_+
\exp(-h_a).
\]

**Regime-switching**

The latent regime $z_t$ follows a candidate-specific transition matrix $\Pi$, and the active regime selects its mechanistic growth map. The same discrete regime law must be used during fitting, kernel construction, filtering, and planning.

Process noise is multiplicative lognormal. The normalized survey model is

\[
y_t\mid x_t\sim \operatorname{LogNormal}(\log x_t,\sigma_o^2),
\]

where $\sigma_o$ is public and fixed by the evaluation cell.

## 6. What the frozen experiment actually implemented

### 6.1 Old hidden `plus_native`

The code-server audit reported four action-conditional polynomial transition templates labelled Ricker, Allee, theta, and regime. These labels did not denote the ecological equations above: the templates had no ecological $r,K,C,\theta$, or regime transition matrix $\Pi$. They were fitted from the offline rows and combined using a Bayesian-weighting skeleton, then solved through a tabular/QMDP-like approximation.

This retained a broad PLUS idea, namely model averaging, but removed the paper's central object: a bank of actual mechanistic candidate POMDPs with candidate-specific observation evidence and separately solved policies.

Correct reporting label:

> PLUS-inspired posterior model averaging over action-conditional polynomial transition templates.

### 6.2 Old hidden `moor_native`

The code-server audit reported independent per-action ridge regressions of the form

\[
\log(1+o_{t+1}/S)=\beta_{a0}+\beta_{a1}\log(1+o_t/S).
\]

Rows were effectively exchangeable. There was no ordered latent trajectory propagation, no mechanistic growth/capacity/catchability parameterization, and no single fitted ecological POMDP. The fitted regressions were converted to a 61-bin Gaussian transition model and used with value iteration plus belief-weighted (Q) values.

Correct reporting label:

> MOOR-inspired offline per-action regression baseline.

### 6.3 Old full `moor_native`

The known-(r,K) arm reportedly used a parameter-known Ricker table and fitted only a scalar (K), with different solver/filter details from hidden mode. It remains a useful known-demographics comparison. It is not an implementation of the original MOOR fitting procedure and must not be used to invalidate the known-(r,K) result.

## 7. Why those old implementations were not paper-faithful

| Paper invariant | Old PLUS | Old MOOR | Consequence |
|---|---|---|---|
| Mechanistic ecological dynamics | Replaced by labelled polynomials | Replaced by one-step ridge regressions | Ecological inductive bias was not represented |
| Ordered trajectory likelihood/loss | Not the PLUS candidate evidence model | Rows treated exchangeably | MOOR's offline identification mechanism disappeared |
| Candidate-specific observation evidence | Incomplete/surrogate | Not applicable | PLUS posterior did not mean the paper's model posterior |
| Separate candidate POMDP solutions | QMDP-like approximation | Not applicable | PLUS policy aggregation changed meaning |
| One fitted mechanistic model | Not applicable | Absent | MOOR became generic offline regression |
| Belief-state planning | QMDP-like | Value iteration/belief-weighted Q | Information-gathering behavior was suppressed |
| Same model in fit/filter/planner | Not established | Not mechanistically meaningful | Internal consistency could not be claimed |

The problem is not simply that the original papers used SARSOP or DESPOT. Those solvers may be replaced with disclosure. The deeper problem was replacement of the model and inference structure that defines each algorithm.

## 8. The rejected provisional repair

The first mechanistic repair restored much of the intended structure but contained two blocking scientific errors.

### 8.1 MOOR observation-noise bias

The provisional objective sampled observation noise inside the SSE. For latent prediction (x), it minimized terms based on

\[
\mathbb{E}_{\nu}\left[(x\exp(\sigma_o\nu)-y)^2\right],
\]

instead of comparing the observed survey with its conditional mean. This changes the optimizer, not merely its Monte Carlo variance. At $\sigma_o=0.4$, the induced scale distortion is about 14.8 percent downward relative to the conditional-mean target.

### 8.2 Regime fit/deployment mismatch

The provisional regime fit averaged regime parameters $C$ and $\eta$ and used one mean-field transition, whereas deployment sampled discrete regimes under $\Pi$. Since nonlinear ecological maps do not satisfy $f(\mathbb{E}\phi)=\mathbb{E}f(\phi)$, the fitted process was not the process being planned with.

The launched run was therefore cancelled, its artifacts retained only for diagnosis, and its results declared scientifically void.

## 9. Corrected PLUS-adapted implementation

### 9.1 Registered identity

Method ID:

`plus_adapted_mechanistic_pbvi`

Display description:

> PLUS-adapted mechanistic candidate-POMDP baseline with fixed regime-persistence candidates, bootstrap parameter candidates, and PBVI.

### 9.2 Offline construction

1. Preserve episode order and boundaries; do not shuffle transitions into an exchangeable table.
2. Build a preregistered finite bank of mechanistic candidates using the benchmark equations.
3. Enforce action structural zeros from public mechanism categories.
4. Estimate continuous candidate parameters from the offline episodes. Bootstrap MAP fits may provide several fixed candidates per family.
5. For regime candidates, use a small preregistered grid of fixed $\Pi$ values. Conditional on each $\Pi$, fit continuous parameters by averaging over sampled discrete regime trajectories with common random numbers.
6. Freeze every candidate's parameters before deployment.
7. Construct each candidate's transition and observation kernels from that same model.
8. Solve each candidate independently using finite-horizon context-conditioned PBVI.

### 9.3 Online operation

For every candidate, maintain a state belief. Maintain a posterior over candidates. At each observation:

1. predict the state belief under the executed action;
2. calculate the candidate-specific survey evidence;
3. update the model posterior by Bayes' rule;
4. update the candidate state belief;
5. select the action maximizing posterior-weighted candidate PBVI values.

No continuous parameter is updated online. The online learning object is the posterior over fixed candidate POMDPs, exactly as required by PLUS.

### 9.4 Why this is paper-aligned

- the candidates are complete mechanistic POMDPs, not regression labels;
- each candidate has its own transition, observation, belief, and solved value function;
- model probabilities are updated from sequential evidence;
- actions use posterior-weighted candidate values;
- uncertainty over dynamics remains explicit during deployment.

PBVI replaces SARSOP and bootstrap/fixed-$\Pi$ candidate construction extends the original bank design. These facts prevent the label “exact reproduction,” but they do not remove PLUS's defining algorithmic identity.

## 10. Corrected MOOR-adapted implementation

### 10.1 Registered identity

Method ID:

`moor_adapted_ricker_misspec_pbvi`

Display description:

> MOOR-adapted mechanistic trajectory-fitting baseline with preregistered Ricker misspecification, conditional-mean survey SSE, and PBVI.

### 10.2 Offline trajectory fitting

Fit one Ricker model jointly to all complete training episodes. Each episode resets the latent initial abundance distribution and cumulative capacity to its initial state. Within-episode order is preserved.

For each deterministic optimization start:

1. sample common process-noise paths;
2. propagate the full latent abundance trajectory under the observed actions;
3. map latent abundance to the conditional mean survey,

\[
\widehat y_{e,t}=\mathbb{E}[y_{e,t}\mid x_{e,t}]
=x_{e,t}\exp(\sigma_o^2/2);
\]

4. minimize

\[
L(\vartheta)=\frac{1}{|\mathcal E|}
\sum_{e\in\mathcal E}\frac{1}{T_e}
\sum_{t=1}^{T_e}
\mathbb{E}_{\text{process paths}}
\left[(\widehat y_{e,t}(\vartheta)-y_{e,t})^2\right];
\]

5. use normalized variables, parameter bounds, automatic differentiation, L-BFGS, and deterministic multiple starts;
6. retain the parameter vector with minimum training objective.

Observation noise is integrated through the conditional mean and is never independently sampled inside the SSE. Holdout episodes are used for diagnostics, not optimizer selection.

The default primary objective contains no shrinkage, group, or complementarity regularizer. Any future regularization is a named sensitivity analysis.

### 10.3 Planning model

The fitted Ricker model is used unchanged to construct the transition kernel, observation kernel, belief filter, and finite-horizon context-conditioned PBVI planner. It is fitted to every true benchmark family without revealing the family label. Thus it is correctly specified for Ricker cells and deliberately misspecified for Allee, theta, and regime cells.

### 10.4 Why this is paper-aligned

- one mechanistic population model is fitted offline;
- complete ordered trajectories drive the loss;
- process uncertainty is integrated by Monte Carlo propagation;
- optimization uses bounded, normalized, multi-start gradient-based fitting;
- one fitted model is used consistently for inference and control;
- misspecification is explicit and preregistered, echoing the paper's model-misspecification study.

The observation/reward variables and planner differ from the fishery study, so this is a MOOR-aligned conservation adaptation, not a literal MOOR reproduction.

## 11. Required benchmark adaptations

These changes are necessary because retaining the fishery formulation literally would solve the wrong problem.

| Original paper setting | Conservation benchmark | Why adaptation is necessary |
|---|---|---|
| Fishing effort action | 11 interventions with rate, capacity, combined, or translocation mechanisms | The benchmark's management decisions are conservation interventions |
| Catch observation | Noisy abundance survey | Catch does not exist in the benchmark |
| Catch/harvest reward | Common conservation reward including abundance/persistence, cost, and safety terms | All methods must optimize the same evaluation objective |
| One historical fishery series | Multiple finite ordered episodes | Offline data are episodic and have real reset boundaries |
| Biomass and carrying-capacity scaling | Public survey-derived scale $S$ | True $K$ and demographics are private in hidden mode |
| Fishery effort-to-catch mechanism | Public qualitative intervention category, private effect magnitude | Gives ecological meaning without leaking hidden numerical parameters |
| Stationary fishery state | Capacity, previous survey, and timestep context | Habitat actions have cumulative effects and the horizon is finite |
| Paper's exact solver | PBVI | Original solvers are unavailable/incompatible with the benchmark implementation |

The first seven rows are problem-required adaptations. PBVI is instead a computational substitution and must be disclosed and validated.

## 12. Registered extensions and approximations

The following are defensible but are not required by the original papers:

- PLUS cross-family candidate banks;
- bootstrap MAP candidate parameterizations;
- a fixed grid of regime persistence matrices;
- common random numbers when integrating process or regime paths;
- MOOR's single Ricker model applied across all hidden true families;
- context-conditioned finite-horizon PBVI;
- any future regularization, if separately named and reported.

For first corrected runs, regime $\Pi$ is fixed per PLUS candidate rather than learned. The $\Pi$ grid and expected number of switches over 25 steps must be reported. A later extension may estimate $\Pi$ by particle or stochastic EM, but it must not be silently folded into the primary method.

## 13. Side-by-side old versus corrected comparison

| Dimension | Previous PLUS-inspired baseline | Corrected PLUS-adapted target | Previous MOOR-inspired baseline | Corrected MOOR-adapted target |
|---|---|---|---|---|
| Model | Polynomial transition templates | Mechanistic candidate POMDP bank | Per-action log-linear ridge | One fitted mechanistic Ricker POMDP |
| Data use | Offline rows fit templates | Episode-preserving candidate fitting plus online evidence | Exchangeable one-step pairs | Ordered complete-trajectory SSE |
| Uncertainty | Template weights | State belief per candidate plus Bayesian model posterior | Regression/kernel uncertainty only | Process uncertainty during one-model fitting and belief filtering |
| Online learning | Approximate template reweighting | Bayesian posterior over fixed candidates | None | None, matching MOOR's offline commitment |
| Regimes | Label without discrete $\Pi$ | Discrete candidate-specific $\Pi$, same law everywhere | Not represented | Deliberate one-Ricker misspecification |
| Planner | QMDP-like | Candidate-specific PBVI and posterior value averaging | Value iteration/belief-weighted Q | PBVI in the fitted model |
| Defensible name | PLUS-inspired | PLUS-adapted, paper-aligned after verification | MOOR-inspired | MOOR-adapted, paper-aligned after verification |

## 14. Verification gates before rerunning the full experiment

The corrected design is not complete until the code-server implementation passes the following checks.

### PLUS gates

- candidate kernels match the declared mechanistic equations;
- action structural zeros are enforced;
- every candidate is solved independently;
- posterior updates use sequential candidate-specific observation evidence;
- posterior weights normalize and respond correctly in synthetic identification tests;
- regime candidates use the same discrete $\Pi$ process in fit, kernel, filter, and planner;
- no hidden true family or parameter enters candidate construction;
- candidate count and $\Pi$-grid sensitivity are reported on the registered subset.

### MOOR gates

- episode boundaries and within-episode order are preserved;
- observation noise is not sampled inside SSE;
- the conditional mean includes $\exp(\sigma_o^2/2)$;
- only process paths are Monte Carlo averaged in the fitting objective;
- the same fitted model feeds kernel, filter, and planner;
- multi-start optimization is deterministic and the minimum training objective selects the fit;
- synthetic parameter-recovery and trajectory-prediction tests pass;
- true family information is absent from fitting.

### Shared gates

- PBVI performs belief-state backups rather than QMDP backups;
- capacity, previous survey, and timestep context are represented consistently;
- all methods use the same reward and action availability;
- no private demographic value enters normalization or priors;
- canary runs report wall time, memory, convergence, and seed reproducibility.

The full 288-cell sweep should not be authorized before canary and runtime review. Earlier provisional timing suggested roughly 31 minutes for one MOOR cell and more than three hours for an unfinished 16-candidate PLUS cell, implying a potentially very large CPU budget.

## 15. Reporting rules

For the completed frozen experiment:

- retain the results as evidence about the algorithms actually run;
- relabel the ecology baselines as PLUS-inspired and MOOR-inspired approximations;
- do not retroactively describe them as original PLUS or MOOR;
- do not characterize the known-(r,K) arm as wrong or merely an artifact.

For corrected experiments:

- use the registered method IDs and full display descriptions;
- state explicitly that PBVI substitutes for SARSOP/DESPOT;
- distinguish required benchmark adaptations from optional extensions;
- report candidate-bank construction, priors, posterior updates, $\Pi$ grid, fitting bounds, starts, Monte Carlo paths, and convergence;
- present model-fit and identification diagnostics alongside control outcomes;
- call the implementations paper-aligned adaptations only after the verification gates pass.

## 16. Diagnostic and sensitivity scope

Primary configurations and diagnostics from the same fit may cover the full 288 cells. Expensive candidate-count, regularization, uncertainty-interval, and regime-grid sensitivities should first use the registered 32-dynamics-cell subset:

- Amur tiger: recoverable and declining;
- Egyptian vulture: demographic sink;
- four true dynamics families;
- survey noise $\sigma_o\in\{0,0.1,0.2,0.4\}$.

Fits should be reused across reward modes wherever the model fit is reward-independent.

## 17. Claim language

Recommended:

> We implemented paper-aligned adaptations of PLUS and MOOR for the hidden-demographics conservation benchmark. PLUS retains Bayesian online discrimination over separately solved mechanistic candidate POMDPs. MOOR retains offline trajectory-level fitting of one mechanistic model followed by planning in that fitted POMDP. We adapted actions, observations, rewards, episode handling, and normalization to the conservation data, and used finite-horizon PBVI as a disclosed computational substitute for the papers' planners.

Avoid:

- “exact reproduction”;
- “the original PLUS/MOOR implementation”;
- “fully faithful” without qualifying adaptation and solver substitution;
- interpreting poor safe-mode control as intrinsic algorithm failure when the safety objective is private;
- comparing corrected results directly with frozen results without marking the baseline version change.

## 18. Primary sources and local specification

### PLUS

- Memarzadeh & Boettiger, *Resolving the measurement uncertainty paradox in ecological management*, Biological Conservation (2018): <https://www.sciencedirect.com/science/article/pii/S0006320717316737>
- Supplementary material: <https://ars.els-cdn.com/content/image/1-s2.0-S0006320717316737-mmc1.pdf>
- Archived `pomdpplus` source: <https://zenodo.org/records/1161521>

### MOOR

- Ju et al., *Model-based offline reinforcement learning for sustainable fishery management*, Expert Systems (2023): <https://onlinelibrary.wiley.com/doi/10.1111/exsy.13324>
- Author-hosted paper: <https://rdl.cecs.anu.edu.au/papers/ES23-Model%E2%80%90basedOfflineReinforcementLearningForSustainableFisheryManagement.pdf>

### Local benchmark specification

- `29_6_Algorithm_Paper_Audit_Guide.tex`, especially the mechanistic model definitions around lines 92--123;
- `29_6_Real_Ecology_Setting_Implementation_Plan.tex`, especially the hidden-mode data/action definitions around lines 172--205 and method adaptation around lines 271--282.

## 19. Final interpretation

The correction is not “give general RL the true Ricker model.” It is to give ecological baselines the mechanistic structure that defines their published algorithms while preserving hidden demographics. PLUS receives a finite, uncertain bank of plausible mechanistic models and learns which candidate is supported online. MOOR receives ordered offline trajectories and fits one deliberately restricted mechanistic model before deployment. Neither receives the true hidden family or true private parameters.

That is the fair comparison: common data and objective, explicit ecological priors for ecological baselines, black-box flexibility for general learners, and transparent accounting for every unavoidable adaptation.
