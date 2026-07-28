# Review of the Provisional Paper-Aligned PLUS and MOOR Implementation

## Status and immediate boundary

The implementation is scientifically promising, but it was produced before the
required planning gate was approved. The instruction was to return a plan and stop;
instead, production code was implemented, smoke-tested, and a 4,000-row one-cell run
was started.

Treat the current implementation as **unapproved provisional work**. Do not launch
additional runs, modify frozen artifacts, edit the result report, or present these
methods as approved until the issues below are resolved and reviewed. Preserve all
current provisional artifacts for inspection; do not delete or overwrite them.

## Sources used for this review

### MOOR

- Ju, Kurniawati, Kroese, and Ye, *Model-based offline reinforcement learning for
  sustainable fishery management*:
  https://onlinelibrary.wiley.com/doi/10.1111/exsy.13324
- Author-hosted full paper:
  https://rdl.cecs.anu.edu.au/papers/ES23-Model%E2%80%90basedOfflineReinforcementLearningForSustainableFisheryManagement.pdf

Relevant parts are Sections 4.2 and 4.4, especially Equations 9--10, plus the
discretization and DESPOT sections.

### PLUS

- Memarzadeh and Boettiger, *Adaptive management of ecological systems under partial
  observability*:
  https://www.sciencedirect.com/science/article/pii/S0006320717316737
- Archived `pomdpplus` documentation:
  https://rdrr.io/github/boettiger-lab/pomdpplus/man/compute_plus_policy.html
- Archived software record:
  https://zenodo.org/records/1161521

### Benchmark equations

- `29_6_Algorithm_Paper_Audit_Guide.tex`, especially the ecological transition at
  lines 92--123.
- `29_6_Real_Ecology_Setting_Implementation_Plan.tex`, especially the cumulative
  capacity and translocation specification.
- `ECOLOGY_BASELINE_PAPER_FAITHFULNESS_PLUS_MOOR.md` remains the controlling
  scientific specification.

## Findings ordered by severity

### F1 -- Major -- Scientific design -- MOOR observation-noise loss must change

Published MOOR compares predicted catch `g(B_t,e_t;q)` with observed catch and takes
an expectation over stochastic latent biomass trajectories. It does not draw an
independent noisy observation from the fitted model and compare that simulated
replicate with the observed catch.

The provisional implementation instead minimizes a term of the form

```text
E_nu[(x * exp(sigma_o * nu) - y)^2].
```

This is not the corresponding conditional-mean SSE plus an irrelevant constant:

```text
E_nu[(x * exp(sigma_o * nu) - y)^2]
  = x^2 * exp(2*sigma_o^2)
    - 2*x*y*exp(sigma_o^2/2)
    + y^2.
```

It changes the optimizer's target and generally pulls the fitted abundance below the
value that predicts the observed survey. Common random numbers make this objective
reproducible but do not remove that bias.

For the closest MOOR-style adaptation:

1. Continue to propagate process uncertainty through Monte Carlo latent trajectories.
2. Do not draw a second survey replicate inside the SSE.
3. Compare the observed survey with the conditional survey prediction. Under the
   stated observation model, `E[y|x] = x*exp(sigma_o^2/2)`.
4. Alternatively, use a lognormal observation likelihood integrated over latent
   paths, but label this as a likelihood-based extension beyond MOOR's SSE rather than
   silently calling it the paper objective.

**Classification of the current choice:** unacceptable replacement until corrected.

### F2 -- Major -- Paper faithfulness -- Regime fitting and deployment disagree

The regime candidate is fitted using probability-weighted parameters

```text
C_bar   = (1-p)*C_0   + p*C_1
eta_bar = (1-p)*eta_0 + p*eta_1,
```

followed by one nonlinear transition. Deployment instead uses a discrete latent
regime `z_t` and transition matrix `Pi`.

This is not exact marginalization because

```text
f(E[C_z], E[eta_z]) != E[f(C_z, eta_z)].
```

Consequently, the fitting objective calibrates a different transition model from the
one later used to construct kernels, update beliefs, and plan. This violates the
model-consistency requirement.

Correct the regime fit using one of:

1. sampled discrete regime trajectories with fixed/common random numbers;
2. an exact forward recursion over the two regimes where tractable;
3. a particle or mixture approximation that evaluates the same discrete-regime model
   used at deployment.

Fitting, POMDP construction, filtering, and planning must share the same regime
transition law. If mean-field dynamics are retained, deployment must also use that
mean-field model, in which case it must not be described as a discrete
regime-switching candidate.

**Classification of the current choice:** unacceptable fit/deployment mismatch. A
consistent approximation may instead be classified as a computational approximation.

### F3 -- Moderate -- Scientific design -- Regularizers are undisclosed extensions

The MOOR loss adds:

- action-parameter shrinkage toward the column mean;
- a norm penalty on action deviations;
- the complementarity term `mean(g_a*h_a)`.

These are not part of published MOOR. They may be reasonable for fitting 11
categorical conservation actions, but must be classified and reported as **optional
regularization extensions**, not absorbed into the paper-faithful core.

Record every coefficient in configuration and artifacts. Provide a sensitivity or
ablation showing whether the fitted parameters and policy depend materially on these
penalties.

### F4 -- Moderate -- Scientific design -- Action-mechanism sparsity needs resolution

The controlled equations are mechanistic and align with the registered benchmark:

```text
g_a = r_{+,a}
h_a = -r_{-,a}
```

Thus non-negative `g_a,h_a` correctly reproduce the benchmark's treatment of
negative effective growth as unconditional mortality. Cumulative `d_a` and pre-growth
stocking `u_a` also match the registered transition design.

However, the provisional model allows every action to have all four mechanisms. If
the intervention categories are public, consider enforcing structural zeros:

- `u_a = 0` for every non-translocation action;
- `d_a = 0` for actions without a capacity mechanism;
- share the baseline rate for actions whose public mechanism does not change growth.

Do not expose private effect magnitudes. If mechanism categories themselves are not
public under the registered schema, retaining all four parameters is possible, but it
must be classified as a more flexible optional extension and supported by action
coverage and identifiability diagnostics.

### F5 -- Positive -- PLUS's mechanistic core is substantially repaired

Subject to correcting the regime fitting mismatch, the new PLUS implementation
repairs the original polynomial-template defect:

- candidate labels instantiate explicit ecological equations;
- candidates contain interpretable parameters;
- candidate parameters remain fixed online;
- each candidate maintains its own state belief;
- the model posterior is updated from candidate predictive evidence;
- action values are combined using posterior candidate weights.

These are the defining mechanics of PLUS. The cross-family candidate set and its
construction from public histories remain disclosed adaptations/extensions.

### F6 -- Positive -- PBVI is an acceptable named computational approximation

Finite-horizon context-conditioned PBVI is defensible if it genuinely performs
belief-state backups over each candidate's transition and observation model, retains
time/previous-observation/capacity context, and does not reduce to QMDP.

Continue to record the actual solver in method IDs and artifacts. Do not use SARSOP or
DESPOT labels unless those solvers are actually invoked. PBVI is a computational
approximation to the papers' solvers, not a reproduction of them.

## Direct answers to the implementation questions

### PLUS

1. **Do the equations qualify as mechanistic candidates?**

   Yes for Ricker, Allee, and theta. The deployed regime equation is mechanistic, but
   the regime candidate is not fully defensible until fitting targets the same
   discrete-regime model.

2. **Is episode-bootstrap MAP acceptable?**

   Yes, as a disclosed optional extension/registered variant for discretizing
   parameter uncertainty. It is not the paper's exact candidate-grid construction,
   but it preserves the finite mechanistic candidate-bank interpretation. Report
   bootstrap seeds, unique accepted candidates, parameter distances, and any failed
   diversity checks.

3. **Are four candidates per family defensible?**

   Yes, provided the registered candidate-count sensitivity compares smaller and
   larger total banks, reports actual unique candidates, and checks that conclusions
   are not an artifact of a four-per-family bank. No exact paper candidate count is
   mandatory.

4. **Does uniform prior plus Bayesian evidence updating match PLUS?**

   Yes. A uniform prior is the archived default. Per-candidate beliefs, Bayesian
   evidence updates, and posterior-weighted candidate values match PLUS's high-level
   decision rule.

5. **Does PBVI meet the fidelity bar?**

   Yes as a disclosed computational approximation, assuming the implementation does
   the full context-conditioned belief backups described. It must not be described as
   SARSOP or as an exact Bayes-adaptive solution.

### MOOR

6. **Is controlled Ricker with `(g,h,d,u)` legitimate?**

   Yes as a necessary mechanistic adaptation from continuous fishery effort to the 11
   conservation actions. It preserves interpretable ecological control semantics.
   The unconstrained availability of all mechanisms to every action and the added
   regularizers are optional extensions requiring disclosure and identifiability
   evidence.

7. **Is the ordered-episode L-BFGS procedure faithful?**

   Ordered episode propagation, process-noise Monte Carlo, autodiff L-BFGS, multiple
   starts, and minimum-training-loss selection are aligned with MOOR. The current
   simulated-observation component of the SSE is not and must be corrected under F1.

8. **Is a fitted reset distribution acceptable?**

   Yes. Replacing one scalar `B0` with a shared fitted lognormal episode-reset
   distribution is a necessary adaptation to multiple episodes with deliberately
   varied starts. Report it explicitly and verify episode-boundary resets.

9. **Is one Ricker model across all families fair?**

   Yes as a preregistered misspecification study. It is correctly specified only for
   Ricker cells and intentionally misspecified for Allee, theta, and regime cells. It
   must not be presented as a correctly specified MOOR test across all families.

10. **Are observation and reward replacements necessary adaptations?**

    Yes. Replacing catch with public abundance surveys and replacing catch/yield
    reward with the shared conservation objective are necessary benchmark
    adaptations. MOOR must not be cited as provenance for the conservation reward.

### Both methods

11. **Survey-noise and regime answers:** F1 and F2 are blocking corrections.

12. **Do the implementations contradict the papers?**

    The repaired core mechanics are paper-aligned, but the two blocking issues and the
    disclosed benchmark/solver extensions prevent these from being described as
    reproductions of published PLUS and MOOR.

## Departure classification

| Departure | Classification |
|---|---|
| Fishing effort to 11 conservation actions | Necessary benchmark adaptation |
| Catch observation to abundance survey | Necessary benchmark adaptation |
| Catch/harvest reward to shared conservation reward | Necessary benchmark adaptation |
| Multiple ordered episodes instead of one history | Necessary benchmark adaptation |
| Fitted episode-reset distribution | Necessary benchmark adaptation |
| Public survey-derived normalization scale | Necessary benchmark adaptation |
| Cross-family PLUS candidate bank | Optional extension |
| Episode-bootstrap MAP candidate construction | Optional extension / registered variant |
| One Ricker MOOR across all true families | Optional preregistered misspecification study |
| PBVI instead of SARSOP/DESPOT | Computational approximation |
| Action shrinkage/group/complementarity penalties | Optional regularization extensions |
| Simulated survey replicate inside MOOR SSE | Unacceptable replacement until corrected |
| Mean-field regime fitting plus discrete-regime deployment | Unacceptable mismatch until corrected |

## Naming and claim boundary

These methods may **not yet** be called implementations or reproductions of published
PLUS and MOOR.

Use the following paper wording:

> **PLUS-adapted mechanistic candidate-POMDP baseline with bootstrap candidates and
> PBVI.**

> **MOOR-adapted mechanistic trajectory-fitting baseline with preregistered Ricker
> misspecification and PBVI.**

Recommended method IDs:

```text
plus_adapted_mechanistic_pbvi
moor_adapted_ricker_misspec_pbvi
```

After F1 and F2 are corrected and verified, these may be described as
**paper-aligned adaptations**. They still should not be called exact reproductions
because actions, observations, rewards, candidate construction, multi-episode fitting,
and planners differ from the original applications.

## Required response and next gate

Before further execution, return:

1. the exact correction made for F1, including the revised mathematical objective;
2. the exact correction made for F2, including how discrete regimes are handled during
   fitting;
3. the classification and configuration of every regularizer;
4. the decision on action-mechanism structural zeros;
5. updated method IDs and display labels;
6. focused test results for the corrected observation objective and regime
   fit/deployment consistency;
7. confirmation that no additional experiment jobs were launched;
8. the status of the already-started 4,000-row one-cell run, without deleting its
   artifacts;
9. an updated plan for review before any larger run.

Do not proceed to a headline sweep until these corrections and the updated plan have
been approved.
