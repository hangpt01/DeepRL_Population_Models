# Paper-Faithful PLUS and MOOR Implementation Plan

**Status:** planning gate only; no implementation is authorized by this document.  
**Controlling specification:** `ECOLOGY_BASELINE_PAPER_FAITHFULNESS_PLUS_MOOR.md`.  
**Prepared:** 2026-07-17 (Australia/Melbourne).  
**Scope:** additive paper-faithful baselines; all existing methods and the completed
hidden-r/K experiment remain unchanged.

## 1. Executive summary

The completed experiment compared general methods with `plus_native` and
`moor_native` as they were implemented. Those numerical results remain valid and
must remain reproducible. The existing native methods are not to be edited, renamed,
or retroactively described as paper-faithful.

This plan adds two new method families. The initial registered targets are:

- `plus_faithful_pbvi`: mechanistic PLUS with finite-horizon, history-conditioned
  point-based value iteration. The `_pbvi` suffix discloses a **computational
  approximation** to the paper's SARSOP planning;
- `moor_faithful_ricker_misspec_despot`: the intended MOOR implementation when the
  APPL Online DESPOT solver is actually called at deployment;
- `moor_faithful_ricker_misspec_pbvi`: an explicit fallback if DESPOT fails its build,
  licensing, determinism, or runtime gate;
- `plus_faithful_sarsop`: a deferred optional target, enabled only if a future
  reward formulation preserves the registered public objective without an
  intractable context expansion and SARSOP is actually invoked for every candidate.

The first registered scientific run should be **hidden-only**. It will fit all
mechanistic parameters from sanitized, ordered public histories. Existing
`plus_native` and `moor_native` full-mode results remain separately named
parameter-known upper bounds. A new full arm for the faithful methods is deferred
until it has a distinct scientific question and unambiguous method name.

This revision resolves the external audit's three blocking findings before Stage 0:
PLUS uses PBVI rather than assuming SARSOP can cheaply represent the history/time
dependent reward; total sweep scope is selected from blinded measured canaries under
a pre-approved CPU-hour ceiling; and all major result directions are interpreted in
advance in Section 24.1.

The central implementation rule is separation of concerns:

1. one module owns normalized mechanistic equations;
2. one module fits or constructs fixed ecological parameter candidates from public
   episodes;
3. one module constructs model-consistent transition, observation, and reward
   interfaces;
4. planner adapters expose a common belief-state action-value interface;
5. each policy owns exactly the belief(s) used by its planner;
6. serialized artifacts establish data dependence, privacy, and reproducibility.

No polynomial basis may carry an ecological-family label. No hidden method may
receive an `EnvironmentConfig`, true family, true state, private table, private
safety threshold, evaluator sidecar, or path from which those values can be inferred.

Primary algorithm sources remain the controlling specification's archived PLUS
paper/software (`https://zenodo.org/records/1161521`) and the MOOR paper/code
(`https://onlinelibrary.wiley.com/doi/10.1111/exsy.13324`,
`https://github.com/jujun622/moorfisherys`). Implementation comments or existing
benchmark names are not accepted as paper-faithfulness evidence.

## 2. Verified current code architecture

### 2.1 Live tree

The live worktree was inspected before planning. `git status --short --branch`
reported `main...origin/main [ahead 3]`, many modified tracked files, and untracked
hidden-r/K implementation/run files. This plan assumes neither `HEAD` nor the dirty
worktree alone defines the completed experiment. Unrelated work must not be reverted.

Current routing and behavior:

- `src/real_ecology_benchmark/methods/__init__.py` registers both old native IDs and
  defines `NATIVE_METHODS`.
- `src/real_ecology_benchmark/methods/base.py` gives hidden policies a sanitized
  `MethodContext`, while full policies receive `EnvironmentConfig`.
- `src/real_ecology_benchmark/pipeline.py` loads the public dataset, fits/loads the
  shared public reward/risk surrogate, builds `MethodContext`, performs an
  episode-disjoint 80/20 train/holdout split, fits the policy, and evaluates it.
- `src/real_ecology_benchmark/dataset.py` already preserves `episode_id`, contiguous
  `timestep`, complete-episode terminal markers, public rewards/costs, opaque
  `pop_id`, and separate `terminated`/`truncated` flags. No dataset schema change is
  required for the mechanistic fit.
- `src/real_ecology_benchmark/collector.py` collects complete episodes to a 4,000-row
  target and records bounded overshoot.
- `src/real_ecology_benchmark/public_surrogate.py` fits the shared public reward and
  genuine-termination surrogate. It must remain the common benchmark reward source;
  it must not become the ecological transition model.
- `src/real_ecology_benchmark/types.py` excludes reward and private evaluator values
  from `PublicTransition`; online policy/filter updates therefore receive action and
  public observation history only.
- `src/real_ecology_benchmark/beliefs.py` contains the existing filters and caches.
  New faithful policies should own their candidate-specific beliefs rather than
  forcing them through the old single-grid cache.
- `src/real_ecology_benchmark/native_fit.py` fits action-conditional polynomial/log
  regressions and tabular QMDP-style solvers. It will not be reused by the faithful
  methods.
- `src/real_ecology_benchmark/native_solver.py` builds table/config-driven full-mode
  solvers. It will not be reused by hidden faithful methods.

### 2.2 Existing native mechanics that must remain reproducible

- Hidden `plus_native` creates four fitted polynomial transition templates, one per
  family label, initializes uniform weights, maintains one abundance belief per
  template, updates template weights from action-observation evidence, and combines
  tabular Q values.
- Hidden `moor_native` fits one action-conditional log-linear observation transition
  model and uses a fitted-grid QMDP-style policy.
- Full `plus_native` builds its candidate solvers from `EnvironmentConfig` and the
  ecological tables.
- Full `moor_native` grid-searches one scalar `K` but otherwise uses table/config
  dynamics; its pre-fit evaluation filter and post-fit planner are not the same
  fitted object.

These are historical facts for the frozen implementation, not components to copy
into the corrected baselines.

### 2.3 Current serialization and monitoring

Datasets, belief caches, learned proposals, and the public surrogate have `.npz`
save/load support. Policies do not currently have a general fitted-model
serialization contract. Native summaries retain fit diagnostics but not a complete,
reloadable candidate bank or optimizer state. The faithful implementation therefore
needs a versioned, no-pickle artifact schema and a policy artifact hook.

## 3. Frozen/live repository boundary

The completed run is immutable:

`real_ecology_runs/hidden_rk_comparison_20260716/`

Verified provenance:

- frozen source: `real_ecology_runs/hidden_rk_comparison_20260716/code/`;
- frozen at: `2026-07-16T02:46:59+10:00`;
- base commit: `5f9cf32a69d47e2d31b2ed6ee30ebbcfe84b8536`;
- registered rows: 1,440 per arm, 288 cells per arm;
- registered methods: `refplan`, `bamcts`, `ogsrl`, `plus_native`, `moor_native`;
- collection: 4,000 target rows with complete 25-step episodes;
- current snapshot hashes include:
  - `plus_native.py`: `957a7c838f8198f192582a60f17f5abf028f432810e39f8caa337dc9f6f0f020`;
  - `moor_native.py`: `06a3ecba6f2b12838bd5df4147ed0d4f29887f4adaaa9cd70c57c10ee7736f36`;
  - `native_fit.py`: `9c9a8e9eaf0e7651fad1c30221aec0fcc736cdada6d698e3ad8567a0929d0ce7`;
  - `native_solver.py`: `2e5d6fc93b7066a2acf9940b2f9ba8ad1e8ec667178ac1437886bf68f8e0c682`;
  - `pipeline.py`: `f4c5e39c83a05cbebcba8664ffba3df754f870dcbdbe561ac4d6871a3da139bd`.

Implementation must never write beneath that run root. A corrected run must use a
new root, frozen source snapshot, method IDs, manifests, registration, and provenance.
Old outputs must never be relabelled as faithful PLUS or MOOR.

## 4. PLUS paper components to preserve

The corrected PLUS baseline must preserve all of the following:

1. A finite bank of actual mechanistic ecological POMDPs.
2. Multiple fixed parameter candidates within every included family.
3. Candidate-specific transition and observation models.
4. A separate abundance/regime belief for each candidate, plus that candidate's
   deterministic fully observed capacity context.
5. Candidate-specific belief-state planning.
6. An initial candidate prior that is explicitly configured and serialized.
7. Online Bayesian candidate-weight updates using each candidate's predictive
   action-observation evidence.
8. Posterior-weighted candidate action values.
9. Candidate parameters, transitions, rewards, and solved policies fixed online.
10. Candidate-count sensitivity and posterior-convergence reporting.

The default candidate prior will be uniform, matching the archived `pomdpplus`
default. A public-history-informed prior is an optional preregistered variant, not a
silent replacement.

## 5. MOOR paper components to preserve

The corrected MOOR baseline must:

1. Fit one explicit controlled Ricker latent-state model.
2. Consume temporally ordered complete public episodes.
3. Propagate latent abundance and fitted cumulative-capacity state within episodes.
4. Reset a fitted latent initial-state distribution at each episode boundary.
5. Minimize trajectory-level public-survey SSE under Monte Carlo latent propagation.
6. Normalize variables using public data only.
7. use bounded differentiable parameters, automatic differentiation, stochastic
   L-BFGS, multiple deterministic starts, and minimum-loss selection;
8. fit interpretable growth, mortality, capacity, stocking, initial-state, and process
   noise parameters;
9. construct transition and observation kernels from the fitted model by Monte Carlo;
10. maintain a latent belief during deployment;
11. plan with DESPOT or an explicitly named tested belief-state approximation;
12. use the identical fitted model object for filtering and planning.

One preregistered Ricker model is used on every hidden cell. It is correctly specified
only when the private simulator family is Ricker and deliberately misspecified on
Allee, theta-logistic, and regime-switching cells. The method never receives that
private family label.

## 6. Necessary benchmark adaptations

Each departure is classified here and again where it enters the implementation.

| Departure | Classification | Reason |
|---|---|---|
| Fishing effort becomes one of 11 conservation action IDs | necessary benchmark adaptation | The benchmark has conservation interventions, not fishing effort. |
| Catch observation becomes noisy abundance survey | necessary benchmark adaptation | Abundance is the registered public observation. |
| Catch/yield reward becomes shared public benchmark reward surrogate | necessary benchmark adaptation | All methods must optimize the same registered objective without private safety access. |
| Multiple 25-step episodes replace one uninterrupted fishery history | necessary benchmark adaptation | The public offline dataset is episode structured. |
| Episode-specific initial abundance is represented by one fitted reset distribution | necessary benchmark adaptation | Collection deliberately varies initial abundance; a single global point `B0` is incompatible. |
| Public median survey abundance supplies scale instead of true `K` | necessary benchmark adaptation | Hidden mode forbids private population scale. |
| Four-family PLUS bank | optional extension | Published examples are principally same-family banks; the benchmark extension is closed-world structural uncertainty. |
| One Ricker MOOR fit on all four private simulator families | optional extension | It is a preregistered model-misspecification study analogous in spirit, but not identical, to the paper's misspecification analysis. |
| PBVI instead of SARSOP or DESPOT | computational approximation | It preserves belief-state planning but changes the solver. |
| QMDP/value iteration/MPC under a faithful name without suffix | unacceptable replacement | It hides a substantive planning change. |
| Polynomial regressions named Ricker/Allee/theta/regime | unacceptable replacement | Family labels must denote actual equations. |
| True table/config parameters in hidden fitting | unacceptable replacement | It violates the information regime. |

The public reward surrogate may be queried by simulated public histories. It may not
be used as a source of private safety labels or as a substitute for ecological
dynamics. No post-hoc public safety threshold is introduced.

## 7. Explicit normalized mechanistic state

Let `S` be the median positive survey observation in the training episodes. `S` is
public, serialized, and fixed. Define normalized abundance `x_t = N_t/S` and
normalized effective capacity `k_j,t = K_j,t/S` for candidate `j`. Within a fixed
candidate, capacity is a deterministic function of that candidate's fixed parameters,
reset value, and the public action history. It is therefore a **candidate-specific
fully observed control state**, not a random variable in that candidate's abundance
belief. Across PLUS candidates there is still capacity uncertainty because candidates
have different fixed `k0`, `d_a`, and `k_max` values.

Candidate hidden state is `z_j,t=x_t`, or `z_j,t=(x_t,g_t)` for a regime-switching
candidate. Fully observed planning context is

`h_j,t = (t, o_{t-1}/S, o_t/S, k_j,t)`.

The full action sequence need not be copied into a tabular state: `k_j,t` is its
sufficient deterministic summary under candidate `j`. PBVI and DESPOT propagate it
along each future action branch. A future SARSOP/MOMDP adapter may declare it as a
fully observed component, but the resulting reachable-state count must be measured;
removing capacity from the hidden belief does not by itself guarantee a fixed 9x
runtime reduction.

For each action `a`, fit nonnegative parameters:

- `g_a`: density-dependent positive growth strength;
- `h_a`: unconditional mortality/degradation strength;
- `d_a`: cumulative normalized capacity increment;
- `u_a`: normalized direct stocking/translocation increment.

Capacity evolves as

`k_{t+1} = clip(k_t + d_a, k_min, k_max)`.

Managed abundance before growth is `m_t = max(x_t + u_a, 0)`. Process innovation is
`epsilon_t ~ Normal(0, 1)` with fitted `sigma_p >= 0`. A group-sparsity penalty on
`(g_a, h_a, d_a, u_a)` and a complementarity penalty on `g_a*h_a` discourage one
sparsely covered action from simultaneously inventing every effect.

This action decomposition is a **necessary benchmark adaptation**. It is mechanistic
and interpretable, but it is not the fishing-effort/catchability parameterization in
the source papers.

## 8. Exact proposed PLUS candidate equations

All candidates use the common action and capacity update above. Candidate parameters
are fixed after construction.

### 8.1 Ricker

`x_{t+1} = m_t * exp(g_a * (1 - m_t/k_{t+1}) - h_a + sigma_p*epsilon_t)`.

### 8.2 Allee-Ricker

With fitted normalized Allee threshold `c > 0`:

`x_{t+1} = m_t * exp(g_a * (1 - m_t/k_{t+1}) * (m_t/c - 1) - h_a + sigma_p*epsilon_t)`.

The bounds enforce `0 < c < k_min` for each candidate.

### 8.3 Theta-logistic

With fitted `theta > 0`:

`q_t = max(0, m_t + g_a*m_t*(1 - (m_t/k_{t+1})**theta))`,

`x_{t+1} = q_t * exp(-h_a + sigma_p*epsilon_t)`.

The zero floor is part of the declared model, not an optimizer-only repair.

### 8.4 Regime-switching depensation

Let `g_t in {0,1}` and

`Pr(g_{t+1}=j | g_t=i) = Pi[i,j]`,

where each row of `Pi` is a fitted simplex with a persistence lower bound. Each regime
has fitted threshold `c_g` and multiplier `eta_g > 0`:

`x_{t+1} = m_t * exp(eta_{g_t}*g_a*(1 - m_t/k_{t+1})*(m_t/c_{g_t} - 1) - h_a + sigma_p*epsilon_t)`.

This is an actual hidden-regime ecological model. A threshold indicator in a
polynomial basis is prohibited.

## 9. Exact proposed MOOR controlled-Ricker equation

MOOR uses exactly the Ricker equation in Section 8.1, with one fitted parameter vector

`Theta = {mu_B0, sigma_B0, k0, k_max, sigma_p, (g_a,h_a,d_a,u_a) for a=0..10}`.

At each episode boundary:

`log x_{e,0} ~ Normal(mu_B0, sigma_B0**2)` and `k_{e,0}=k0`.

Within an episode:

`k_{e,t+1}=clip(k_{e,t}+d_{a_e,t}, k_min, k_max)`,

`m_{e,t}=max(x_{e,t}+u_{a_e,t},0)`,

`x_{e,t+1}=m_{e,t}*exp(g_{a_e,t}*(1-m_{e,t}/k_{e,t+1})-h_{a_e,t}+sigma_p*epsilon_e,t)`.

All four private simulator families receive this same Ricker learner. No branch may
inspect `cfg.kind`, family-bearing paths, or family metadata.

## 10. Observation and noise models

The public survey protocol is multiplicative log-normal:

`o_t/S = x_t * exp(eta_t)`, with `eta_t ~ Normal(0, sigma_o**2)`.

`sigma_o` is read from sanitized public dataset metadata/`MethodContext` and fixed,
because observation-noise level is part of the public experimental protocol. The
candidate fits `sigma_p` separately. A zero-abundance state emits exactly zero and is
absorbing unless the chosen action has fitted `u_a > 0`; this behavior must be decided
and tested before implementation because the current simulator treats zero as
absorbing before action effects.

Noise separation diagnostics must report:

- fitted `sigma_p` and public `sigma_o`;
- process-residual autocorrelation by episode;
- survey innovation calibration;
- synthetic recovery with one noise source held at zero;
- profile loss showing whether `sigma_p` is identifiable separately from `sigma_o`.

No private true-state residual may enter fitting or model selection.

## 11. Mechanistic 11-action parameterization

### 11.1 Sharing and regularization

The default parameterization fits all 11 action IDs jointly. It uses:

- bounded transforms for positivity and capacity ordering;
- weak shrinkage of each action vector toward a pooled public-data mean;
- group sparsity on action deviations;
- a complementarity penalty discouraging simultaneous large `g_a` and `h_a`;
- no use of action names, true effect tables, species tables, or filename tokens;
- action cost only in the shared reward, never as a dynamics label.

Regularization coefficients and bounds are configuration values and must be fixed by
synthetic calibration before any real hidden result is inspected.

### 11.2 Coverage and sparse actions

Every fit records per-action rows, episodes, initial-state range, and observation
range. An action is provisionally `sparse` when it appears in fewer than
`max(25, 5*p_a)` transitions, where `p_a` is the number of free action-effect
parameters. Sparse actions retain pooled/shrunken effects and receive an explicit
identifiability warning; they are not silently assigned table effects or zero effects.
The exact threshold is an approval item.

### 11.3 Identifiability

The following are mandatory diagnostics, not tuning gates:

- Hessian/Fisher condition estimate in unconstrained coordinates;
- parameter-profile or bootstrap intervals;
- pairwise correlation among `k0`, `g_a`, `d_a`, and `u_a`;
- action-specific effective sample sizes;
- sensitivity to start values and Monte Carlo seed banks;
- held-out trajectory survey error.

Weak identifiability is reported, not repaired using private tables.

## 12. PLUS candidate-grid and prior design

### 12.1 Candidate construction

For each of the four families:

1. fit a family-specific mechanistic MAP model from ordered training episodes using
   only the sanitized public history;
2. form a deterministic bounded proposal distribution from the local curvature and
   bootstrap variation across public episodes;
3. draw a seeded low-discrepancy set of parameter proposals;
4. rescore each proposal with a sequential candidate filter on training episodes;
5. retain diverse high-likelihood candidates, not duplicate numerical fits;
6. freeze every retained parameter vector before deployment.

Default proposal: four retained candidates per family, 16 total. Preregistered
candidate-count sensitivity: two, four, and eight candidates per family (8/16/32
total). Counts are deliberately smaller than the historical 144-model application;
the paper does not require 144, and runtime is first benchmarked in smoke tests.
More candidates are not automatically more faithful: uniqueness, public-history
support, posterior calibration, and solver convergence matter more than proximity to
the historical application's count. Sixteen is the preregistered headline default.
Eight and 32 are sensitivity arms on a fixed diagnostic subset; neither may replace
the headline count after real-data fitting or return inspection.

Candidate diversity requires a minimum standardized parameter distance and records
which proposal each retained candidate came from. If a family cannot produce two
distinct finite candidates, the method fails loudly for that cell; it does not clone
one candidate under multiple IDs.

**Registered implementation variant (2026-07-17, before any 4,000-row faithful
execution):** the initial implementation uses deterministic episode-bootstrap MAP
fits instead of the local-curvature/low-discrepancy proposal-and-rescore procedure in
steps 2--4 above. Candidate zero is the family MAP fit; each additional candidate is
an independently optimized MAP fit on a seeded complete-episode bootstrap resample.
Artifacts and manifests call this `episode_bootstrap_map_v1`. This is a posterior
approximation with within-family parameter uncertainty when more than one candidate
per form is run; it is not claimed to implement the original proposal-and-rescore
procedure. The 16-candidate default and diversity fail-loud rule are unchanged.

### 12.2 Initial prior

Default:

`w_j,0 = 1/M` for all `M` candidates.

Optional preregistered variant:

`w_j,0 proportional to exp(ell_j(H_holdout)/tau_prior)`,

where `ell_j` is sequential public action-observation log likelihood on complete
holdout episodes and `tau_prior` is fixed before the real sweep. The manifest must
name `uniform` or `public_history_likelihood`; no adaptive fallback is allowed.

The cross-family bank is an **optional extension** beyond the paper's principal
same-family examples and must be stated in every report and manifest.

## 13. PLUS belief and posterior updates

For candidate `j`, abundance/regime belief `b_j,t(z)` is initialized from its episode
reset distribution and the first public survey; candidate capacity `k_j,t` is updated
deterministically from the same action history. Given action `a_t`:

`bbar_j,t+1(z') = integral T_j(z'|z,a_t) b_j,t(z) dz`,

`L_j,t+1 = integral O_j(o_t+1|z') bbar_j,t+1(z') dz'`,

`b_j,t+1(z') = O_j(o_t+1|z') bbar_j,t+1(z') / L_j,t+1`,

`w_j,t+1 = w_j,t L_j,t+1 / sum_k w_k,t L_k,t+1`.

Use log-space normalization and a declared likelihood floor. Recovery from a fully
impossible observation must be deterministic and reported. Reward, evaluator state,
true abundance, true family, and private termination/safety labels are absent from
both updates.

Each candidate planner supplies `Q_j(b_j,t,a | h_t)`. PLUS selects

`a_t = argmax_a sum_j w_j,t Q_j(b_j,t,a | h_t)`.

Candidate model parameters, kernels, reward adapter, and solved planner objects remain
fixed online. Only candidate abundance beliefs, regime beliefs, deterministic
capacity context, and candidate weights update.

## 14. MOOR trajectory objective and optimizer

### 14.1 Episode-preserving objective

For normalized public surveys `y_e,t=o_e,t/S`, draw `M_fit` common-random-number
latent trajectories from the equations in Section 9. For start `s`, minimize

`J_s(Theta) = (1/M_fit) sum_m sum_e sum_t mask_e,t * (y_e,t - yhat_e,t,m(Theta))**2`

`             + lambda_group R_group + lambda_comp R_comp + lambda_bound R_bound`.

`yhat` is a simulated public survey from the propagated latent abundance and the
known public observation kernel. Each episode independently draws/reset its initial
latent abundance and resets `k` to fitted `k0`. `terminated` ends propagation;
`truncated` is an ordinary episode boundary, not an extinction target.

This is trajectory-level survey SSE. Independent transition regression is prohibited.
Reversing time within an episode must change the fit; permuting whole episodes must
not, apart from floating-point summation tolerance.

### 14.2 Optimization

Proposed dependency: CPU PyTorch automatic differentiation and `torch.optim.LBFGS`.
PyTorch is not currently installed and may only be added after dependency approval.
The fitted MOOR vector has approximately 49 free parameters, so dependency size alone
is a legitimate operational concern. PyTorch is recommended because it supplies both
reverse-mode automatic differentiation through Monte Carlo trajectory propagation and
an L-BFGS implementation with a closure interface. A smaller automatic-differentiation
stack may replace it only if it passes the identical gradient, optimizer, determinism,
and serialization tests. Hand-derived gradients and finite differences are not
automatic differentiation and would violate the controlling MOOR contract; they are
not faithful fallbacks.

Defaults to calibrate synthetically before registration:

- normalized `float64` optimization;
- eight deterministic starts drawn from a recorded seed sequence;
- 16 Monte Carlo paths per episode during fitting;
- fixed common random numbers within each L-BFGS closure;
- at most 100 accepted L-BFGS iterations per start;
- gradient-norm and relative-objective stopping checks;
- fail on NaN, bound saturation, or non-finite simulated trajectory;
- select the finite start with minimum training trajectory objective;
- holdout episode SSE is reporting-only, not start selection.

The exact starts, bounds, penalties, random-number hashes, objective traces, gradients,
and selected-start reason are serialized. Increasing starts or MC paths after viewing
real results is prohibited.

## 15. POMDP construction and shared reward

Each fitted/candidate continuous model is discretized by seeded Monte Carlo:

- abundance grid: configurable, provisionally 41 public-scale bins plus zero;
- capacity context grid: configurable, provisionally nine bins over fitted candidate
  bounds; it is fully observed within a candidate and is not part of its hidden belief;
- regime grid: one state except for regime-switching candidates, which use two;
- observation grid: configurable, provisionally 41 survey bins plus zero;
- transition samples: provisionally 256 per state/action;
- observation samples: provisionally 256 per next-state bin;
- interpolation: sparse barycentric mass with rows checked to sum to one.

The common public reward is a **necessary benchmark adaptation**. Candidate planners
evaluate the existing shared public surrogate on simulated public histories and
public action costs. Because that surrogate uses previous/current/following survey and
time, the planner interface must carry fully observed context `h_t`. It is not
acceptable to silently hard-code `t=0` or set previous observation equal to latent
state.

The initial PLUS implementation therefore uses finite-horizon, context-conditioned
PBVI. It constructs a seeded reachable belief/context graph from the current public
history and candidate model, backs up candidate-specific values for the registered
horizon, and samples reachable observation branches. It does **not** materialize the
full Cartesian product of abundance x capacity x previous-observation x time. DESPOT
receives the same history through its generative scenario callback.

Stationary SARSOP is deferred. A naive explicit encoding with 41 abundance bins, nine
capacity bins, 41 previous-observation bins, and 50 time indices would approach
760,000 state-context combinations before actions; that is a warning calculation,
not an unavoidable exact state count. `_sarsop` may be enabled only after a separate
formulation and measured solver probe show that the exact registered reward is
preserved tractably. Dropping previous observation or time from the reward is a new
scientific choice, not an implementation shortcut.

The same fitted/candidate model object must supply transition sampling, observation
likelihood, filtering, discretization, and planner simulation. No parallel
table-derived filter is permitted.

## 16. Planner and dependency decision

### 16.1 Verified environment state

No `pomdpsol`, SARSOP, DESPOT, APPL, POMDP Python package, PyTorch, JAX, SciPy,
Autograd, or related environment module was found. `pyproject.toml` currently depends
only on NumPy and PyYAML. The active interpreter has NumPy 1.23.5 and PyYAML 5.4.1,
while `pyproject.toml` declares NumPy >=1.24 and PyYAML >=6.0. This pre-existing
environment mismatch must be resolved by a fresh locked implementation/run
environment that satisfies the declared requirements; do not lower requirements or
alter the frozen environment merely to match the current shell. Nothing was installed
during planning.

### 16.2 SARSOP

The official APPL SARSOP repository
(`https://github.com/AdaCompNUS/sarsop`) is a C++/GNU-make project. It accepts POMDP or
POMDPX and writes a serialized policy file. Its repository documents an Apache-2.0
notice/license structure. Solver seeding is not documented in the inspected README,
so determinism must be established by repeated fixed-input policy-hash tests rather
than assumed.

Recommendation: do **not** use SARSOP for the initial headline. Keep its adapter as a
deferred stage after the PBVI experiment, and implement it only if the history/time
reward can be represented without changing the objective and a measured candidate
solve passes the approved total budget. If implemented, invoke it through a
subprocess adapter with immutable POMDPX and policy artifacts; do not vendor or patch
it silently.

### 16.3 DESPOT

The official APPL Online DESPOT repository
(`https://github.com/AdaCompNUS/despot`) is C++, builds with make or CMake, accepts
POMDPX or a direct C++ black-box model interface, and exposes runtime seed, timeout,
particle-count, and depth options. The repository contains Apache-2.0, GPL-2.0, zlib,
and a top-level license/attribution set, so the exact linked components require a
license review before integration. DESPOT is an online planner; the fitted model and
planner configuration can be serialized, but a reusable complete online search tree
must not be promised as a policy artifact.

Recommendation: use DESPOT for the headline MOOR method through a pinned C++ black-box
adapter if build, licensing, seed determinism, and per-action latency pass smoke gates.

### 16.4 Fallback

Implement finite-horizon, context-conditioned PBVI as the initial PLUS planner and
MOOR fallback. It must use explicit beliefs, stage-specific backups, sampled reachable
public-history contexts, and candidate-specific values. It must not collapse to a
fully observable MDP or one-step QMDP. Any PBVI result uses the `_pbvi` suffix and is
classified as a **computational approximation**. QMDP, fully observable value
iteration, and generic MPC are not acceptable fallbacks under the proposed faithful
IDs.

## 17. Proposed method IDs and metadata

| ID | Meaning | Eligible use |
|---|---|---|
| `plus_faithful_pbvi` | Mechanistic candidate PLUS with finite-horizon context-conditioned PBVI | Initial PLUS headline; explicit planning approximation |
| `plus_faithful_sarsop` | Same candidate bank with per-candidate SARSOP | Deferred; only after exact-reward feasibility and solver gates |
| `moor_faithful_ricker_misspec_despot` | Single fitted Ricker MOOR, DESPOT, misspecified outside Ricker cells | Preferred MOOR headline after solver gate |
| `moor_faithful_ricker_misspec_pbvi` | Same fitted model with PBVI | Explicit approximation/fallback |

Metadata must separately record `algorithm_lineage`, `planner`, `planner_version`,
`planner_binary_sha256`, `planner_seed_behavior`, `model_equation_version`,
`benchmark_adaptations`, and `extensions`. A method is not registered under `_sarsop`
or `_despot` unless runtime diagnostics prove that solver was invoked.

## 18. Full-versus-hidden routing recommendation

1. **Initial run:** hidden-only for all new faithful IDs. This directly tests learning
   from sanitized public histories and avoids conflating fidelity repair with a new
   parameter-known algorithm.
2. **PLUS full mode:** do not initially construct one. If true `r,K` are supplied,
   the candidate bank and prior need a separately preregistered uncertainty axis.
   A future method should be named `plus_faithful_knownrk_<planner>` and must state
   which nuisance parameters remain uncertain.
3. **MOOR with true `r,K`:** do not call it faithful MOOR. MOOR's defining operation
   is fitting the mechanistic model from historical data. A true-parameter method is
   a separately named parameter-known oracle/upper bound.
4. **Existing full methods:** retain `plus_native` and `moor_native` as frozen
   parameter-known comparison arms. Their behavior and results remain useful.
5. **Comparisons:** report new hidden faithful methods against (a) the frozen hidden
   inspired baselines, (b) general learners in the matched hidden regime, and (c)
   existing full parameter-known upper bounds as context. Do not present (c) as a
   same-method full-minus-hidden causal delta.

## 19. File-by-file implementation map

No file is to be changed until this plan is approved.

### 19.1 New production modules

- `src/real_ecology_benchmark/faithful_ecology.py`  
  Dataclasses, bounded parameter transforms, all four exact equations, action/capacity
  update, process and survey kernels. Must import no `realdata`, `actions`,
  `EnvironmentConfig`, evaluator, or private dataset type.

- `src/real_ecology_benchmark/faithful_fit.py`  
  Ordered-episode tensors, public normalization, MOOR trajectory SSE, family MAP
  fitting, multiple starts, posterior/bootstrap candidate construction, coverage and
  identifiability diagnostics.

- `src/real_ecology_benchmark/faithful_pomdp.py`  
  Monte Carlo discretizer, sparse kernels, candidate belief update, model evidence,
  public-history reward adapter, and model-consistency checks.

- `src/real_ecology_benchmark/faithful_artifacts.py`  
  Versioned `.npz`/JSON save-load, hashes, privacy scan, schema validation, and
  round-trip identity checks. No pickle.

- `src/real_ecology_benchmark/planners/base.py`  
  Common belief-state planner protocol and planner provenance.

- `src/real_ecology_benchmark/planners/pbvi.py`  
  Seeded finite-horizon PBVI approximation with serialized alpha vectors or equivalent
  finite-horizon policy representation.

- `src/real_ecology_benchmark/planners/sarsop.py`  
  **Deferred optional file, not part of the initial implementation.** If separately
  approved: POMDPX writer, subprocess invocation, timeout/error handling, policy
  parser, executable/version/hash capture, and actual-invocation diagnostics.

- `src/real_ecology_benchmark/planners/despot.py`  
  APPL Online adapter, deterministic seed plumbing, black-box model serialization,
  timeout handling, and actual-invocation diagnostics.

- `src/real_ecology_benchmark/methods/plus_faithful.py`  
  Candidate bank lifecycle, separate beliefs, posterior update, posterior-weighted
  candidate action values, diagnostics, and artifact hook.

- `src/real_ecology_benchmark/methods/moor_faithful.py`  
  Single-model fit, single consistent filter/planner belief, misspecification metadata,
  diagnostics, and artifact hook.

### 19.2 Existing production files changed additively

- `src/real_ecology_benchmark/config.py`  
  Add `FaithfulModelConfig`, `FaithfulFitConfig`, and `FaithfulPlannerConfig`; validate
  candidate counts, bounds, starts, MC budgets, solver paths, and exact method/planner
  compatibility. Do not add private fields to `MethodContext`.

- `src/real_ecology_benchmark/methods/__init__.py`  
  Register new IDs without changing existing mappings. Replace the single
  `NATIVE_METHODS` assumption with explicit routing capabilities while preserving its
  old values and behavior.

- `src/real_ecology_benchmark/methods/base.py`  
  Add a no-op versioned `save_fit_artifacts(output)` hook or protocol. Existing
  policies inherit the no-op unchanged.

- `src/real_ecology_benchmark/pipeline.py`  
  Route faithful methods only through hidden `MethodContext`; skip old pre-fit belief
  caches; fit before constructing candidate-specific filters; save faithful artifacts;
  record solver provenance. Existing method branches must remain behaviorally
  identical.

- `src/real_ecology_benchmark/manifest.py`  
  Permit new method/filter/planner combinations and emit faithful-specific registered
  fields. Existing manifest generation remains unchanged by default.

- `src/real_ecology_benchmark/cli.py`  
  Surface new validated config fields and fail on method/planner suffix mismatch.

- `pyproject.toml`  
  Add an opt-in `paper-faithful-fit` extra for the approved automatic-differentiation
  dependency. External DESPOT, and any later SARSOP binary, remain explicitly
  provisioned and hashed, not implicit PyPI dependencies. Create and record a lock that satisfies the
  existing NumPy/PyYAML minimums; do not opportunistically weaken those minimums.

### 19.3 Files explicitly not changed unless a failing interface test proves necessary

- `methods/plus_native.py`, `methods/moor_native.py`, `native_fit.py`,
  `native_solver.py`;
- `dataset.py`, `collector.py`, `types.py`, `public_surrogate.py`, `evaluator.py`;
- frozen run files and existing result/report files.

### 19.4 New configurations and scripts

- `configs/paper_faithful_hidden.yaml`;
- `configs/paper_faithful_hidden_smoke.yaml`;
- `scripts/make_paper_faithful_manifest.py`;
- `scripts/run_paper_faithful_acceptance.py`;
- `scripts/run_paper_faithful_solver_probe.py`;
- `scripts/slurm/run_paper_faithful_row.sh` only if existing row runner cannot carry
  the external solver environment without changing old behavior.

### 19.5 New tests

- `tests/real/test_faithful_equations.py`;
- `tests/real/test_plus_faithful.py`;
- `tests/real/test_moor_faithful.py`;
- `tests/real/test_faithful_privacy.py`;
- `tests/real/test_faithful_planners.py`;
- `tests/real/test_faithful_artifacts.py`;
- synthetic fixtures under `tests/fixtures/faithful_ecology/`.

## 20. Configuration and artifact schema

### 20.1 Configuration

Record at minimum:

- equation version and normalized scale rule;
- family set and candidate count per family;
- candidate prior type and temperature;
- parameter bounds and regularization coefficients;
- sparse-action threshold;
- train/holdout split seed and episode IDs;
- optimizer, starts, iterations, tolerances, MC paths, and random-number seeds;
- state/capacity/observation bins and MC discretization samples;
- planner name, path, time/depth/particle/belief-point budget, and seed;
- reward-surrogate version/hash;
- hidden-only routing and explicit adaptation/extension labels.

### 20.2 Fitted artifacts

Every row saves:

- `faithful_fit.npz`: normalized parameters, physical transformed parameters,
  initial-state distribution, noise parameters, action coverage, selected start;
- `faithful_fit.json`: schema version, equations, bounds, objective, diagnostics,
  public dataset hash, split episode IDs, seeds, dependency versions;
- PLUS only: `candidate_bank.npz/json`, candidate IDs/families/parameters, initial prior,
  historical likelihoods, diversity diagnostics, candidate hashes;
- `pomdp_model.npz/json`: grids and sparse transition/observation/reward hashes;
- one candidate model artifact per PLUS candidate;
- planner artifact: PBVI policy arrays or DESPOT model/config; a SARSOP policy XML is
  required only for a separately approved `_sarsop` method;
- `planner_provenance.json`: executable source revision, binary hash, command,
  deterministic seed support, stdout/stderr hashes, actual invocation count;
- evaluation posterior/belief diagnostics with bounded sampling frequency;
- `privacy_audit.json` and artifact-level forbidden-name scan.

All arrays use `allow_pickle=False`. Loading validates schema, dimensions, row sums,
parameter bounds, hashes, planner suffix, and public dataset identity.

### 20.3 Manifest/provenance

New manifests add `method_impl_version`, `planner`, `candidate_count`, `prior`,
`fit_budget_id`, `discretization_id`, and `equation_version`. A new run snapshot records
the dirty status, non-bytecode tree hash, config hash, external source revisions,
binary hashes, compiler/version, dependency lock, Slurm job IDs, and output counts.

## 21. Detailed test matrix

| # | Test | Exact acceptance condition |
|---:|---|---|
| 1 | Existing natives unchanged | Existing native unit tests pass; fixed-seed old-ID fixture actions/fit diagnostics match pre-change golden arrays; frozen tree hashes are untouched. |
| 2 | Equation correctness | Every family/action equation matches hand-computed noiseless values at low, equilibrium, and above-capacity states; all probability kernels normalize. |
| 3 | Synthetic parameter recovery | On identifiable synthetic Ricker histories, fitted `k0`, covered-action `g/h/d/u`, and `sigma_p` meet preregistered relative/absolute tolerances across fixed seeds. |
| 4 | PLUS posterior convergence | Data generated by one in-bank candidate drive its posterior above a fixed threshold and improve log score relative to uniform; out-of-bank case remains calibrated rather than forced to truth. |
| 5 | PLUS state belief | One-step hand Bayes update equals implementation; repeated observations normalize and move mass in the expected direction. |
| 6 | Uniform prior | `uniform` initializes every candidate exactly `1/M`, independent of fit likelihood ordering. |
| 7 | Fixed online parameters | Hash all candidate parameters/kernels/policies before and after deployment; hashes must be identical while beliefs/posterior change. |
| 8 | Candidate count | 2/4/8 candidates per family produce exactly 8/16/32 unique candidates and matching planner artifacts; invalid/duplicate banks fail loudly. |
| 9 | MOOR order sensitivity | Reverse time within each episode and verify objective/fit changes materially; permute complete episodes and verify equality within numerical tolerance. |
| 10 | Episode reset | A two-episode synthetic fixture proves latent `x0` and `k0` reset and no final state crosses the boundary. |
| 11 | Multiple starts | Selected artifact is the finite start with minimum registered training objective; failed starts remain recorded. |
| 12 | Action-effect recovery | Synthetic actions separately affecting growth, capacity, mortality, and stocking recover the correct active component and sign. |
| 13 | Sparse-action warning | Below-threshold actions emit structured warnings, shrink toward the pooled effect, and never read table defaults. |
| 14 | Noise separation | Synthetic process-only, observation-only, and mixed cases distinguish fitted `sigma_p` from fixed public `sigma_o` within preregistered tolerances. |
| 15 | Filter/planner consistency | Object/hash identity proves each transition/observation model used by filtering is the one supplied to its planner; no second exact solver exists. |
| 16 | No reward online | Spy policy/filter inputs verify `observe` and belief updates receive no reward; changing evaluation reward after fitting cannot alter posterior/belief updates. |
| 17 | Family relabel invariance | Relabel private family/config kind and family-bearing output paths while holding public data fixed; every hidden fit/planner artifact is byte-identical. |
| 18 | True-r/K/table independence | Corrupt species, lambda, action-effect tables and all private `EnvironmentConfig` demographic fields while holding public data fixed; hidden artifacts remain byte-identical. |
| 19 | Filename/manifest leakage | Forbidden tokens/values in filenames, manifests, config aliases, sidecars, and evaluator summaries cannot enter method constructors or serialized method artifacts. |
| 20 | Determinism/serialization | Same seed/input yields identical parameter/model/policy hashes; save-load produces identical beliefs, posterior, action values, and actions. |

Additional required tests:

- candidate likelihood and planner use the same observation convention;
- zero-state/stocking behavior is explicit and consistent;
- `terminated` versus `truncated` handling is correct;
- public-data perturbation changes fits and solver hashes;
- external solver timeout/failure never silently falls back to PBVI;
- `_sarsop`/`_despot` IDs fail unless invocation diagnostics are present;
- reward-context time/previous-observation handling is not hard-coded;
- candidate capacity is identical for identical action histories and fixed candidate
  parameters, carries no within-candidate belief variance, and differs only when
  candidate parameters differ;
- safe/yield reward surrogates change reward/planner artifacts but not fitted ecological
  dynamics when the public transitions are identical;
- the runtime-canary summarizer cannot load return, reward, action-quality, or private
  evaluator fields and deterministically selects complete-hidden/core/stop from only
  registered costs, limits, and `B_total`.

## 22. Smoke-test protocol

No full real sweep begins until all stages below pass.

1. **Equation smoke:** two actions, 15 abundance bins, five capacity bins, deterministic
   observations, one candidate per test family.
2. **Synthetic fit smoke:** 16 complete 25-step Ricker episodes with designed action
   coverage; recover known growth/capacity/stocking effects with two starts and four MC
   paths.
3. **PLUS bank smoke:** two distinct candidates per family, uniform prior, 10-step
   posterior update generated from an in-bank candidate.
4. **PBVI smoke:** solve tiny candidate POMDPs, round-trip artifacts, and verify an
   action changes when the belief changes.
5. **External solver probe:** one tiny model for each enabled external solver, repeated
   three times with the same seed/input. Record build revision, binary hash, output
   hash, latency, peak RSS, and license files. DESPOT is mandatory before a `_despot`
   ID is enabled. SARSOP is deferred and is not a blocker for the initial `_pbvi`
   PLUS method.
6. **One real hidden cell:** Amur tiger, private simulator Ricker, safe reward,
   `sigma=0.1`, 4,000 target rows, complete episodes. Fit and run one evaluation seed
   only. This is interface validation, not a headline result or tuning source.
7. **Misspecification smoke:** one Allee cell with the same MOOR Ricker implementation;
   verify the method never receives/reports the family except evaluator-side summary.
8. **Old-method regression:** rerun the small existing native fixtures and compare
   golden actions/diagnostics before authorizing a new experiment.
9. **Blinded runtime canary:** run both proposed headline methods on a fixed stratified
   set chosen before execution: one recoverable and one sink population, all four
   private simulator families, both safe/yield rewards, with `sigma=0.0` and `0.2`
   assigned symmetrically (16 scientific cells, 32 method rows). Collect only elapsed
   CPU/GPU time, peak RSS, artifact size,
   optimizer convergence status, and planner invocation counts for sizing. Return and
   policy-quality fields remain unopened until scope and budget are registered.

Every smoke writes to a new temporary or registered smoke root, never the frozen run.

## 23. Runtime and memory estimates

These are planning estimates, not budget facts, and must be replaced by measured
blinded-canary results:

- **MOOR fitting:** eight starts x 100 L-BFGS iterations x 16 MC paths across roughly
  128 training episodes. Expect tens of minutes to a few CPU hours per cell without
  vectorized autodiff; target peak memory below 4 GB. GPU is optional and must not
  change equations or seeds.
- **PLUS construction:** 16 episode-bootstrap MAP fits (four per family) are likely
  several times the MOOR fit cost. Cache dynamics candidates across safe/yield
  only when public transition data, split, candidate config, and fit hashes match.
- **SARSOP:** no initial-sweep estimate is asserted. The prior 60-second-per-candidate
  estimate applied to the unaugmented stationary model and is withdrawn for the
  history/time-aware reward. SARSOP remains deferred until a faithful formulation is
  measured.
- **DESPOT:** online cost repeats at every evaluation action. A one-second search with
  20 x 50 evaluation steps is about 17 minutes/row solely for planning. The final
  timeout must be set from pre-result smoke scaling, not outcome quality.
- **PBVI:** expected to be faster and easier to serialize, but belief points x states
  x actions x candidates can still reach several GB. Use sparse kernels and record
  peak RSS.

The proposed complete hidden scope is 288 scientific cells x two methods = 576 method rows,
excluding candidate-count sensitivity. From the blinded canary, compute per-method
CPU-hour and memory distributions and project

`C_complete = 288 * (mean_cpu_plus + mean_cpu_moor) * 1.5`,

where 1.5 is a preregistered scheduling/estimation contingency, not the previously
observed 1.38 ratio treated as a universal constant. Compare `C_complete` with a
PI-approved total ceiling `B_total` before opening canary returns:

- if `C_complete <= B_total` and memory limits pass, register the 288-cell complete
  hidden scope;
- otherwise register the balanced 144-cell core: all nine populations, all four
  families, safe and yield rewards, and only the endpoint noise levels `sigma=0.0`
  and `0.2`, for 288 method rows;
- if the core also exceeds `B_total`, stop and redesign compute. Do not choose a
  smaller scientific subset after inspecting returns.

Candidate-count sensitivity uses a separately preregistered fixed diagnostic subset
and cannot select the headline count based on return. Before either sweep, produce a
scaling table for state bins, candidate count, MC samples, solver time, artifact size,
and peak memory. Choose budgets on feasibility and convergence diagnostics, never on
which method wins. The audit's approximately 1,000 CPU-hour upper end is treated as a
risk hypothesis until the canary measures it.

## 24. Migration and provenance strategy

1. Leave old modules, IDs, configs, manifests, reports, and run trees untouched.
2. Add faithful methods under new IDs and a new implementation version.
3. Gate registration on complete unit, privacy, artifact, and solver tests.
4. Freeze a fresh source snapshot including approved dependency lock and external
   solver source/binary hashes.
5. Generate a new hidden-only manifest; do not reuse or mutate the 20260716 manifest.
6. Write a registration document containing equations, adaptations, extensions,
   candidate count/prior, fit budget, planner budget, and reporting language.
7. Run smoke and acceptance jobs under separate roots.
8. Only after acceptance, submit the new registered sweep.
9. Aggregate matched cells without overwriting historical summaries.
10. Report old methods as benchmark-native inspired/parameter-known baselines and new
    methods by their exact solver-bearing IDs.

### 24.1 Outcome interpretation pre-registration

All comparisons use matched cells and report each faithful method separately. The
cellwise `best general` envelope over RefPlan, BA-MCTS, and OGSRL may be secondary
context only; it is not one deployable policy.

- **Faithful ecological methods outperform general methods:** conclude that
  mechanistic inductive bias is strong under this public-data budget even without
  oracle demographics. Re-scope the frozen general-method advantage to the
  ecology-inspired approximations actually tested; do not defend the old ordering by
  changing candidate count, fit budget, or planner after seeing results.
- **General methods outperform faithful ecological methods:** conclude that the
  hidden-r/K advantage survives comparison with corrected mechanistic baselines under
  the registered adaptations and compute budget. Do not generalize beyond the tested
  candidate bank, Ricker misspecification, and planner approximation.
- **Mixed ordering by family, reward, noise, or recoverability:** report the
  interaction directly. In particular, strong Ricker and weak non-Ricker MOOR results
  are consistent with deliberate model misspecification, not proof that structure is
  generally useful or useless.
- **Safe-mode differences:** retain the pre-registered statement that safety is
  information-limited under a private safety objective when public termination and
  reward-tail signals are weak. A weak safe result is not intrinsic OGSRL, PLUS, or
  MOOR failure, and no public threshold is added post hoc.
- **Non-identifiable or failed fits:** report failure/coverage rates as outcomes. Do
  not replace them with table parameters, discard hard cells, or expand budgets after
  looking at returns.

Both primary directions and mixed/null results are publishable. The frozen report and
run remain read-only. A new report or external erratum should state explicitly that
the 20260716 natives were ecology-inspired approximations; it must not alter the
frozen PDF, outputs, or provenance.

## 25. Open scientific decisions requiring approval

The plan recommends defaults, but implementation should not silently settle these:

1. **Planner/dependency path:** approve `plus_faithful_pbvi` as the initial PLUS
   headline and DESPOT as the preferred MOOR planner, with
   `moor_faithful_ricker_misspec_pbvi` as its explicit fallback. SARSOP is deferred.
2. **License gate:** approve use of the exact pinned APPL components after their full
   license/notice files are reviewed; DESPOT's multi-license tree needs particular
   attention.
3. **Action model:** approve the four-component `(g,h,d,u)` action parameterization,
   capacity saturation, complementarity penalty, and behavior at zero abundance.
4. **Sparse-action rule:** approve `max(25,5*p_a)` or replace it before real fitting.
5. **PLUS bank:** approve 16 candidates by default, 8/16/32 sensitivity, uniform
   default prior, and the cross-family extension as a headline benchmark method.
   Candidate count is not changed after real fitting or return inspection.
6. **MOOR initial state:** approve a shared fitted log-normal reset distribution rather
   than one scalar `B0` across deliberately varied episode starts.
7. **Reward context:** approve finite-horizon context-conditioned PBVI for PLUS and
   explicit time/previous-survey context in PBVI/DESPOT reward callbacks; a stationary
   `t=0` shortcut is not acceptable.
8. **Autodiff dependency:** approve PyTorch CPU for stochastic L-BFGS or nominate an
   equivalent automatic-differentiation implementation.
9. **Compute ceiling:** set `B_total` before the blinded canary, then apply the fixed
   full-versus-balanced-core rule in Section 23 before real-result inspection.
10. **Full arm:** approve hidden-only initial execution and retention of existing full
    natives solely as separately labelled parameter-known upper bounds.
11. **Dependency environment:** approve a fresh lock satisfying NumPy >=1.24,
    PyYAML >=6.0, and the selected autodiff dependency; do not weaken requirements to
    reproduce the stale active shell.

## 26. Ordered implementation stages and acceptance criteria

### Stage 0: Approval and dependency probe

- Resolve all Section 25 decisions.
- Pin solver revisions and inspect complete licenses.
- Build only in an isolated approved environment.
- Acceptance: tiny repeated solver runs establish interface, seed behavior,
  serialization, runtime, memory, and binary hashes.

### Stage 1: Mechanistic equations and synthetic generator

- Implement normalized state/action/noise equations without pipeline integration.
- Acceptance: equation, kernel normalization, zero-state, and hand-calculation tests
  pass for all four forms.

### Stage 2: MOOR ordered trajectory fitter

- Implement episode resets, MC objective, bounded autodiff L-BFGS, starts, action
  diagnostics, and artifacts.
- Acceptance: order sensitivity, episode resets, synthetic recovery, noise separation,
  sparse-action warnings, minimum-start selection, and deterministic serialization.

### Stage 3: PLUS candidate construction

- Implement family MAP fits, proposal generation, candidate selection, fixed banks,
  priors, and artifacts.
- Acceptance: unique configurable candidate counts, public-data dependence,
  uniform-prior behavior, and no private/table dependence.

### Stage 4: Belief and reward-consistent POMDP construction

- Implement MC discretization, candidate evidence, public-history reward adapter, and
  consistent filter/planner objects.
- Acceptance: hand Bayes tests, same-model identity, reward-context tests, and
  transition/observation/reward normalization.

### Stage 5: PBVI planner integration

- Implement the explicit approximation and both `_pbvi` policies.
- Acceptance: belief-sensitive actions, separate candidate values, posterior-weighted
  PLUS action values, deterministic policy artifacts, and no QMDP fallback.

### Stage 6: External solver integration

- Add DESPOT only if Stage 0 passed. A SARSOP adapter is a deferred, non-blocking
  extension requiring its own exact-reward feasibility and measured-cost approval.
- Acceptance: actual-invocation markers, suffix enforcement, timeout failure behavior,
  repeatability, and comparison against exact tiny POMDP solutions where feasible.

### Stage 7: Pipeline, registry, manifest, and artifacts

- Register new IDs and add hidden-only routing without touching old algorithm files.
- Acceptance: all existing tests/goldens pass; new manifests carry complete provenance;
  hidden policies cannot receive `EnvironmentConfig`; frozen run hashes unchanged.

### Stage 8: Privacy and adversarial tests

- Run relabel, table-corruption, filename/config/manifest leakage, reward-input, and
  public-data perturbation batteries.
- Acceptance: private changes are byte-invariant, public history changes fitted
  artifacts, and every serialized method artifact passes the forbidden-name/value
  scan.

### Stage 9: Registered smoke

- Execute Section 22 under a new frozen smoke snapshot.
- Acceptance: finite fits/actions, complete artifacts, solver provenance, bounded
  runtime/memory, and no scientific-code changes after seeing returns. Run the blinded
  canary, set full versus balanced-core scope using Section 23, and register that scope
  before opening any canary return or policy-quality field.

### Stage 10: New experiment approval

- Produce a run-ready handoff with measured budgets and unresolved limitations.
- Stop for explicit approval before generating or submitting a full manifest.

## 27. Stop condition

Approval of this plan authorizes neither dependency installation nor implementation.
After review, implementation should begin at Stage 0 and stop again at the solver and
scientific-choice gate if any Section 25 item remains unresolved.
