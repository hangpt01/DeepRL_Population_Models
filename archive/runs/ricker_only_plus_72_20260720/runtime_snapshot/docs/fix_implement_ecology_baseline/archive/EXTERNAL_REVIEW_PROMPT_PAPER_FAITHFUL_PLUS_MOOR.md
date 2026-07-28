# External Paper-Faithfulness Review Prompt — corrected PLUS / MOOR baselines

**Purpose:** send this to the reviewing agent that authored
`ECOLOGY_BASELINE_PAPER_FAITHFULNESS_PLUS_MOOR.md`. That agent has the source papers but **no access
to the repository**, so everything it needs to judge fidelity is transcribed below directly from the
implementation. Paste the whole document.

---

## PROMPT BEGINS

You wrote `ECOLOGY_BASELINE_PAPER_FAITHFULNESS_PLUS_MOOR.md`, the controlling scientific
specification for this workstream. **Your diagnosis was correct and has now been confirmed directly in
code.** The previous `plus_native` / `moor_native` "mechanistic families" were literally polynomial
design matrices:

```
ricker → [1, x]              allee  → [1, x, x²]
theta  → [1, x, √x, x²]      regime → [1, x, x², 1(x<1)]
```

fitted by per-action ridge regression on `x = log1p(o/S)`, with no `r`, no `K`, no Allee threshold, no
theta exponent, and no regime transition matrix — and the fit never read `episode_id`, so transitions
were exchangeable rows with no trajectory propagation. Your "benchmark-native MOOR-inspired regression
baseline" label was accurate.

Corrected baselines have now been **implemented and smoke-tested**. You cannot see the code, so this
document transcribes the implemented mathematics and algorithms verbatim from the source. **Please
review the implementation as described here, against the primary sources** (Memarzadeh & Boettiger +
archived `pomdpplus` 0.2.0; Ju et al. + `moorfisherys`).

### What you do NOT need to assess

These were independently verified by a code-level audit and you cannot check them anyway — please
spend no effort on them:

- execution validity (Slurm accounting, exit codes, logs, hashes, artifact completeness);
- privacy/leak-freedom (the equation module imports only `numpy`/`dataclasses`/`hashlib`/`json` and has
  no route to species tables, action tables, `EnvironmentConfig`, evaluator state, or private
  sidecars; table-blocking probes pass; public-data perturbation changes fitted hashes);
- test/lint status (128 tests pass; Ruff clean).

**Only you can do the paper comparison. That is the entire ask.**

---

## 1. Notation and normalisation

All quantities are normalised by a **public** scale `S` = median strictly-positive survey observation
in the training episodes (fixed, serialized). Latent abundance `x = N/S`; capacity `k = K/S`.
There are 11 conservation actions. `σ_o` is the public survey log-noise, **read from the public
protocol metadata and held fixed (never fitted)**. `σ_p` is the fitted process scale.

Per action `a`, four **non-negative** fitted parameters:

- `g_a` — density-dependent growth strength
- `h_a` — unconditional mortality/degradation
- `d_a` — cumulative capacity increment
- `u_a` — direct stocking/translocation increment

Other fitted scalars: `μ_B0`, `σ_B0` (episode reset log-normal), `k_0` (reset capacity), `k_max`
(capacity ceiling), `σ_p`, plus form-specific `C` (depensation thresholds, 2), `θ` (theta exponent),
`η` (regime multipliers, 2), `Π` (2×2 regime transition matrix).

Enforced constraints (hard, at construction): `g,h,d,u ≥ 0`; `k_max ≥ k_0`; `0 < C < k_0`; `η > 0`;
**each row of `Π` is a probability simplex**; `σ_p ≥ 0`; `σ_B0, k_0, k_max, σ_o, S, θ > 0`.

## 2. Implemented state and action dynamics

```
capacity:   k_{t+1} = clip(k_t + d_a, k_0, k_max)          # monotone non-decreasing since d_a ≥ 0
managed:    m_t     = max(x_t + u_a, 0)                     # stocking applied BEFORE growth
```

## 3. The four implemented equations (verbatim)

```
Ricker:   x_{t+1} = m_t · exp( clip( g_a·(1 − m_t/k_{t+1}) − h_a , −40, 40 ) )

Allee:    x_{t+1} = m_t · exp( clip( g_a·(1 − m_t/k_{t+1})·(m_t/C₀ − 1) − h_a , −40, 40 ) )

Theta:    core    = max( 0 , m_t + g_a·m_t·(1 − (m_t/k_{t+1})^θ ) )
          x_{t+1} = core · exp( −h_a )

Regime:   with latent z_t ∈ {0,1}:
          x_{t+1} = m_t · exp( clip( η_{z_t}·g_a·(1 − m_t/k_{t+1})·(m_t/C_{z_t} − 1) − h_a , −40, 40 ) )
          z_{t+1} ~ Π[z_t, ·]      (2×2 row-stochastic, fitted)
```

**Zero handling (declared model behaviour, not an optimizer repair):**
`if x_t == 0 and u_a == 0 → x_{t+1} = 0`. Zero abundance is absorbing **unless** the chosen action has
fitted stocking `u_a > 0`.

**Process noise (multiplicative log-normal):**
`x_{t+1} ← mean · exp(σ_p · ε)`, `ε ~ N(0,1)`; forced to `0` when `mean ≤ 0`.

## 4. Implemented observation model

```
y = o / S ;   y | x  ~  LogNormal( log x , σ_o² )
p(y|x) = exp( −½ ((log y − log x)/σ_o)² ) / ( y · σ_o · √(2π) )
```

`y = 0` has likelihood 1 only for `x ≤ 0` and 0 otherwise; a zero abundance cannot emit a positive
survey. `σ_o` is fixed public protocol, never fitted.

## 5. Implemented MOOR fitting objective (ordered, trajectory-level)

For each **complete episode** `e`, latent state is reset and propagated in time order over `M` Monte
Carlo paths with **common random numbers** held fixed inside each L-BFGS closure:

```
reset:   k ← k_0
         x_{e,0} = exp( μ_B0 + σ_B0 · ξ_e )                       # per-episode log-normal reset
         ŷ_{e,0} = x_{e,0} · exp( σ_o · ν_{e,0} )
         loss   += mean_M( ( ŷ_{e,0} − o_{e,0}/S )² )

step t:  k ← min(k_max, k + d_{a}) ; m = clamp(x + u_a, min=0)
         mean   = <one of the four equations above>
         x      ← mean · exp( σ_p · η_{e,t} )                     # MC latent propagation
         ŷ      = x · exp( σ_o · ν_{e,t+1} )                      # simulated survey
         loss  += mean_M( ( ŷ − o_next/S )² )
         if terminated[t]: break                                   # genuine termination ends propagation
```

Objective returned:

```
J(Θ) = loss / count
     + shrinkage·mean( (A − Ā)² )                 # A = [g,h,d,u] per action, Ā = column mean
     + group·mean( ‖A_a − Ā‖₂ )                   # group sparsity on action deviations
     + complementarity·mean( g_a · h_a )          # discourages simultaneous growth+mortality
```

`truncated` (horizon end) is an ordinary episode boundary, **not** an extinction target.

**Optimizer:** PyTorch float64 autodiff, `torch.optim.LBFGS` with `strong_wolfe` line search;
`starts` deterministic starts (seeded `seed + 1009·start`); failed/non-finite starts are recorded as
`inf` rather than dropped; **selection = the finite start with the minimum training objective**.
Held-out episode survey SSE is computed but is **reporting-only and never used for selection**.
Diagnostics recorded: per-start objective traces, gradient norms, an L-BFGS curvature-condition ratio,
across-start parameter dispersion, per-action row/episode coverage, and sparse-action flags
(threshold `max(sparse_action_rows, 20)`).

## 6. Implemented PLUS candidate bank (`episode_bootstrap_map_v1`)

```
split episodes once → (history, holdout)
for form in {ricker, allee, theta, regime}:
    for j in range(candidates_per_form):        # currently 4 → 16 candidates total
        episodes = history                       if j == 0
        episodes = bootstrap_resample(history)   if j > 0    # episode-level, WITH replacement, same size
        fit = fit_mechanistic_model(form, episodes, ...)     # full §5 procedure, MAP
        enforce diversity: standardized L2 distance to every existing same-form candidate
                           must exceed minimum_candidate_distance, else FAIL LOUD
prior: uniform  w_j = 1/16   (default; softmax-of-loss variant exists but is not the registered default)
```

So **within-family parameter uncertainty is represented by a non-parametric episode-level bootstrap
over MAP fits** — candidate 0 on the full history, candidates 1..3 on bootstrap resamples. This
replaces the originally planned local-curvature / low-discrepancy proposal-and-rescore construction.
Candidates are **frozen** after construction.

## 7. Implemented PLUS deployment

```
per candidate j: its own belief b_j over (abundance bins × regime), its own POMDP, its own planner
act:      Q_j = planner_j.action_values(b_j)
          a*  = argmax_a  Σ_j w_j · Q_j(a)
observe:  (b_j', log L_j) = pomdp_j.update(b_j, a, o')       # L_j = candidate predictive evidence
          log w_j ← log w_j + log L_j ;  normalise in log-space
```

Candidate parameters, transition/observation kernels, reward adapters, and solved planners are
**fixed online**; only beliefs, the deterministic capacity context, and the candidate posterior move.
Reward, evaluator state, true abundance, true family, and private safety labels enter neither update.

## 8. Implemented planner

Seeded **finite-horizon, context-conditioned point-based value iteration (PBVI)**: backups over a
seeded graph of *reachable* belief points; capacity, previous survey, and timestep remain explicit
context at every point; **no stationary approximation and no QMDP reduction**. Method IDs are
`plus_faithful_pbvi` and `moor_faithful_ricker_misspec_pbvi`.

**SARSOP and DESPOT are deferred, not silently swapped.** Reasons: (a) no APPL binary exists in the
environment; (b) the shared benchmark reward depends on the *previous survey and the timestep*, so it
is not a stationary `R(s,a)` — making it Markov for an infinite-horizon solver would require
augmenting the state with a previous-observation bin and a time index. A solver-named ID cannot be
registered unless runtime diagnostics prove that solver was invoked.

## 9. Implemented MOOR method

One controlled-**Ricker** model (form = `ricker`) is fitted on **every** cell regardless of the private
simulator family (Ricker / Allee / theta / regime). The method never receives the family label.
The identical fitted model object supplies filtering **and** planning (no second table-built filter
exists). ID: `moor_faithful_ricker_misspec_pbvi`.

## 10. Departures currently claimed, with their labels

| Departure | Current label |
|---|---|
| Fishing effort → 11 conservation action IDs with `(g,h,d,u)` | necessary benchmark adaptation |
| Catch observation → noisy abundance survey | necessary benchmark adaptation |
| Catch/yield reward → shared public benchmark reward surrogate (MOOR **not** cited as provenance) | necessary benchmark adaptation |
| Many 25-step episodes → replace one 50-step fishery history | necessary benchmark adaptation |
| Per-episode fitted log-normal reset distribution instead of a single scalar `B0` | necessary benchmark adaptation |
| Public median survey supplies scale instead of true `K` | necessary benchmark adaptation |
| Cross-family PLUS bank (Ricker/Allee/theta/regime) | optional extension |
| One Ricker MOOR on all four private families | optional extension (preregistered misspecification study) |
| PBVI instead of SARSOP/DESPOT | computational approximation |
| `episode_bootstrap_map_v1` instead of proposal-and-rescore | registered variant |

## 11. Two implementation details we want your explicit opinion on

Found by reading the source; both are choices you could not have seen from any prior summary:

**(a) The survey noise is *simulated into the prediction*, not integrated out.** The objective forms
`ŷ = x·exp(σ_o·ν)` with a drawn `ν`, then takes squared error against the observed survey. So `J` is a
Monte Carlo expectation over **both** process and observation noise, rather than a marginal likelihood
or an SSE against the latent-mean prediction. This inflates the SSE by a σ_o-dependent constant and
adds MC variance. **Does published MOOR simulate observation noise into its predicted catch, or does
it compare against the expected catch?** If the latter, is this a fidelity defect or an innocuous
variance-inflating choice?

**(b) The regime candidate is fit under a mean-field approximation but deployed as a discrete 2-state
model.** During fitting, the regime is *not* sampled: the threshold and multiplier are
probability-weighted, `C̄ = (1−p)C₀ + p·C₁`, `η̄ = (1−p)η₀ + p·η₁`, with `p` propagated deterministically
through `Π`. At deployment the same candidate uses a **sampled discrete** `z ∈ {0,1}` with `Π`, and the
belief carries a regime dimension. **Is fitting a hidden-regime model by mean-field marginalisation
while deploying it as a discrete latent-regime POMDP acceptable, or does it break the regime
candidate's claim to be a genuine regime-switching model?**

## 12. Questions to answer

**PLUS**

1. Do the four implemented equations qualify as *actual mechanistic candidate POMDPs* in the paper's
   sense — i.e. is the polynomial-template defect genuinely repaired?
2. Is `episode_bootstrap_map_v1` (non-parametric episode bootstrap over MAP fits) an acceptable
   realisation of the paper's "candidate grid or posterior samples", or does bootstrap-MAP fail to
   represent within-family parameter uncertainty as PLUS requires?
3. Is 4 candidates/form (16 total) defensible against the paper's ~100-model real application, given a
   preregistered 8/16/32 sensitivity arm? Your spec says no exact count is required — please confirm
   or refine.
4. Does uniform prior + per-candidate belief + posterior-weighted candidate action values match PLUS's
   decision rule (Appendix C)?
5. Given SARSOP is unavailable and the reward is non-stationary, does this PBVI meet your stated
   fidelity bar — *"belief-state planning within candidates plus Bayesian model averaging, not an exact
   Bayes-adaptive optimum"*?

**MOOR**

6. Does the controlled Ricker with per-action `(g,h,d,u)` legitimately stand in for
   effort/catchability, or does it break MOOR's mechanistic semantics?
7. Is the §5 ordered-episode Monte Carlo **survey** SSE + autodiff L-BFGS + multiple deterministic
   starts + minimum-loss selection faithful to Algorithm 1 / §4.3–4.5?
8. Is a shared fitted log-normal per-episode reset distribution an acceptable adaptation of a single
   `B0` across deliberately varied episode starts?
9. Is one Ricker across all four private families a fair analogue of the paper's
   Schaefer-on-Beverton-Holt misspecification study?
10. Is replacing catch observation/reward with abundance survey + the shared benchmark reward correctly
    classified as a *necessary adaptation*, and correctly **not** attributed to MOOR as provenance?

**Both**

11. Answer §11(a) and §11(b).
12. Does anything above contradict the papers in a way the current labels fail to disclose?

**Classify every departure you identify** into exactly one of your own four categories — *necessary
benchmark adaptation*, *computational approximation*, *optional extension*, *unacceptable
replacement* — and flag any label in §10 you consider wrong.

## 13. Known-open — please do NOT report these as findings

- The accepted 160-row smoke used **one candidate per form**, so within-family uncertainty is not yet
  exercised. A 4,000-row, 16-candidate one-cell run is executing now.
- Profile intervals and the full residual/noise-calibration suite are incomplete.
- The blinded runtime canary has not run; no CPU ceiling is registered; **no headline sweep is
  authorized**.
- DESPOT/SARSOP absent; no `_despot`/`_sarsop` ID registered.
- The frozen 20260716 hidden-r/K run and the old native baselines are deliberately untouched and
  retained as separately-labelled comparison arms.

## 14. Requested output

Findings **ordered by severity**, each tagged **paper-faithfulness** or **scientific-design**, citing
the paper section and the specific equation/procedure above. Then state plainly:

> May these now be called **implementations of published PLUS and MOOR**, or is a qualified label still
> required — and if so, give the exact wording to use in the paper.

## PROMPT ENDS
