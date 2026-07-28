# Decisions for the Corrected PLUS and MOOR Implementation Plan

## Status

The code-server response is accepted. The two blocking findings are confirmed, the
unauthorized job has been stopped without deleting its artifacts, and no further code
or execution is authorized yet.

The decisions below settle Sections 3.1--3.5 of the code-server response. The next
step is for the code side to return an updated implementation plan, feasibility check,
and revised cost estimate. Do not modify code, install dependencies, freeze a new
snapshot, or submit jobs until that plan is approved.

## Decision 1: regime fitting and transition matrix

### Selected option

Use **option (b)** for the first corrected run, strengthened into a small
preregistered candidate grid over fixed regime-transition matrices.

Do not learn `Pi` with gradient descent in the first corrected implementation. Each
regime candidate must instead have a fixed, candidate-specific, row-stochastic `Pi`
selected before results are observed. Fit the remaining continuous parameters using
discrete regime trajectories generated under that candidate's fixed `Pi`, with common
random numbers for reproducibility.

This is especially appropriate for PLUS: PLUS represents uncertainty through a finite
bank of fixed candidate models. Different registered `Pi` values can therefore be
represented by different mechanistic regime candidates rather than forcing one
gradient-based estimate of `Pi`.

### Requirements

1. Propose a small persistence grid for `Pi` in the updated plan. Do not choose it from
   evaluation returns or the private true regime matrix.
2. Prefer a transparent low-dimensional parameterization, such as a symmetric
   persistence parameter, unless asymmetric switching is scientifically necessary.
3. State how the grid relates to the 25-step horizon and how many expected regime
   switches each value implies.
4. Keep the grid configurable and serialized in candidate metadata.
5. Use the same discrete regime transition law in fitting, POMDP construction,
   filtering, and planning.
6. Report Monte Carlo path count and convergence/sensitivity to that path count.
7. Do not describe `Pi` as fitted. Describe it as a preregistered candidate parameter.

### Classification

- Fixed candidate-specific `Pi`: **optional candidate-bank design**, consistent with
  the original PLUS finite-candidate interpretation.
- Common-random-number simulation of the discrete regime process: **computational
  approximation** to the trajectory expectation.
- Not learning `Pi` in the first corrected run: a disclosed scope restriction, not a
  paper-faithfulness defect.

### Ranking of the other options

1. **Future preferred extension:** stochastic EM or particle-EM that learns `Pi` from
   a posterior over discrete regime paths and uses the same discrete model at
   deployment. This requires a separate convergence and runtime study.
2. **Acceptable future approximation:** differentiable mixture/particle fitting with
   model-consistency tests.
3. **Not selected:** Gumbel-softmax for the primary baseline. It adds relaxation bias
   and fit/deployment mismatch risk.
4. **Not selected:** consistent mean-field deployment, because it removes the genuine
   discrete regime-switching candidate.
5. **Not selected:** dropping the regime family. Structural uncertainty over the
   registered regime family is scientifically important.

The code-server statement that sampled discrete transitions give zero gradient with
respect to `Pi` is correct for naive pathwise autodiff. That is why `Pi` is fixed rather
than fitted in this first version.

## Decision 2: MOOR survey target

### Selected target

Use the conditional survey mean:

```text
y_hat = E[y | x] = x * exp(sigma_o^2 / 2).
```

The trajectory SSE must average only over latent process trajectories. Do not draw a
second survey-noise realization inside the prediction loss.

For process path `m`, the survey component should be of the form

```text
loss += (x_t^(m) * exp(sigma_o^2/2) - y_t)^2.
```

Average this over process paths, observed timesteps, and episodes using the registered
normalization. The deployment observation likelihood remains the stated lognormal
likelihood; this decision changes the fitting objective, not the observation filter.

### Paper interpretation

Published MOOR treats catch as a deterministic function of latent biomass and effort.
Its fitting expectation is over stochastic biomass trajectories; it does not add a
separate catch-measurement-noise draw inside the SSE. Therefore MOOR does not directly
solve the survey-noise problem present here.

Using `x` alone would be the most literal syntactic substitution for predicted catch,
but it would ignore the known non-unit mean of the benchmark's lognormal observation
model. Using the conditional survey mean preserves MOOR's substantive principle:
compare the observed quantity against its model-implied conditional prediction.

### Classification

This is a **necessary benchmark adaptation** caused by replacing deterministic catch
observations with explicitly noisy abundance surveys. It is not claimed as a component
of published MOOR. State it explicitly in the method description.

A future marginal-likelihood implementation may be studied as an **optional
statistical extension**, but it is not the primary corrected MOOR objective.

## Decision 3: diagnostic scope and computational budget

### Full 288-cell scale

Only the following are mandatory at full scale, after a successful runtime canary and
explicit CPU approval:

1. the single preregistered corrected PLUS configuration;
2. the single preregistered corrected MOOR configuration;
3. diagnostics obtainable from those same fits without refitting:
   - train and holdout trajectory loss;
   - action and episode coverage;
   - sparse-action and boundary warnings;
   - candidate parameter tables and diversity distances;
   - posterior normalization and effective candidate count;
   - residual summaries by observation-noise level;
   - fitted process-noise summaries;
   - planner convergence/coverage diagnostics;
   - filtering/planning model-hash consistency;
   - runtime and memory.

Do not run candidate-count sensitivity, regularizer ablations, profile intervals, or
bootstrap uncertainty suites over all 288 cells.

### Registered diagnostic subset

Use a subset chosen before corrected-method returns are inspected:

```text
populations: Amur tiger (recoverable but declining) and Egyptian vulture (sink)
families:    Ricker, Allee, theta, regime
sigma_o:     0, 0.1, 0.2, 0.4
```

This gives 32 population-family-noise dynamics cells. Reuse each fitted model/candidate
bank across reward modes wherever scientifically valid; model fitting is independent
of the reward mode. If policy sensitivity must be reported for both yield and safe
rewards, solve both from the same serialized fitted model rather than refitting.

The following belong on this registered subset:

1. PLUS candidate-count sensitivity, including smaller/default/larger banks;
2. regime `Pi`-grid and regime-path Monte Carlo sensitivity;
3. MOOR action-parameter identifiability checks;
4. any optional regularized-versus-unregularized MOOR comparison;
5. process/observation residual calibration plots;
6. parameter uncertainty intervals if they are retained as a claim.

### Priority order

1. Correctness and model-consistency tests.
2. Runtime canary and caching/reuse verification.
3. PLUS candidate-count sensitivity on the subset.
4. MOOR action identifiability and any necessary regularization ablation on the subset.
5. Residual/noise calibration from existing holdout predictions.
6. Profile/bootstrap parameter intervals, subset only and optional unless inferential
   parameter claims are made.

### Required runtime design

The updated plan must identify all reusable computation. In particular, candidate
fitting and MOOR model fitting should not be repeated merely because the reward mode
changes. Cache fitted mechanistic models under hashes of the public dataset, fitting
configuration, family/form hypothesis, candidate seed, and observation protocol.

No full sweep is authorized until the code side reports the canary runtime after the
F1/F2 corrections, expected cache reuse, requested CPU ceiling, and wall-clock plan.

## Decision 4: public action mechanisms and structural zeros

### Schema decision

Treat the **qualitative intervention mechanism category as public**. Action identity,
name, cost, and whether it is a rate, capacity, combined, or translocation intervention
describe the manager's available action. They are not private demographic response
magnitudes.

Keep the numerical action effects private in hidden mode. The method may know what
kind of intervention was selected, but it must fit the magnitude of its ecological
response from sanitized public histories.

### Required structural constraints

1. Only the translocation action may have a nonzero direct stocking parameter `u_a`.
2. Only actions whose public mechanism includes capacity may have nonzero `d_a`.
3. Rate-only actions must not invent capacity or direct-state effects.
4. Capacity-only actions should share the fitted baseline rate unless their public
   definition includes a rate component.
5. Combined actions may include both their declared rate and capacity mechanisms.
6. Exact response magnitudes, true `r,K`, private set-point tables, and private
   population-specific effects remain unavailable.

### Fidelity verdict

- Public categories plus structural zeros: **acceptable necessary benchmark
  adaptation** and the preferred primary specification. This is analogous to MOOR
  knowing that fishing effort acts through its declared catch mechanism while fitting
  unknown parameters.
- Private categories plus all four mechanisms free: still a mechanistic model, not a
  black-box regression, but it becomes a substantially more flexible **optional
  extension**. It would require strong action-coverage, identifiability, and
  parameter-recovery evidence and would support a weaker paper-alignment claim.

Use the public-category branch for the corrected primary methods.

## Decision 5: regularization after structural zeros

### Primary model

Remove the group-deviation, broad shrinkage, and `g_a*h_a` complementarity penalties
from the registered primary MOOR objective.

Fit one signed action-specific effective rate parameter `r_a` where the public action
mechanism permits a rate effect, then derive

```text
g_a = max(r_a, 0)
h_a = max(-r_a, 0).
```

This enforces the benchmark's positive-growth/unconditional-mortality split by
construction and makes the complementarity penalty unnecessary. Use a bounded,
optimizer-compatible parameterization and document treatment near `r_a = 0`.

Structural zeros for `d_a` and `u_a` remove the principal reason for group sparsity.
The primary fit should therefore use the mechanistic constraints and parameter bounds,
without the three additional regularizers.

### Optional regularized variant

If the unregularized constrained model fails preregistered identifiability or numerical
acceptance criteria, propose a weak hierarchical shrinkage variant as a separate
method/configuration. Do not silently activate it. Compare it with the unregularized
model on the registered diagnostic subset before selecting a primary configuration.

### Classification

- Structural constraints: **necessary benchmark adaptation**.
- Signed `r_a` with deterministic positive/negative split: part of the mechanistic
  benchmark model.
- Any later hierarchical shrinkage: **optional regularization extension** requiring
  disclosed coefficients and subset sensitivity.

## Required updates to names and descriptions

Retain the adopted provisional IDs:

```text
plus_adapted_mechanistic_pbvi
moor_adapted_ricker_misspec_pbvi
```

Use the display descriptions:

> PLUS-adapted mechanistic candidate-POMDP baseline with fixed regime-persistence
> candidates, bootstrap parameter candidates, and PBVI.

> MOOR-adapted mechanistic trajectory-fitting baseline with preregistered Ricker
> misspecification, conditional-mean survey SSE, and PBVI.

These are **paper-aligned adaptations**, not exact reproductions of published PLUS or
MOOR.

## Next required deliverable

Return an updated plan before writing code. It must include:

1. the proposed fixed `Pi` candidate grid and expected switches over 25 steps;
2. the exact corrected MOOR objective;
3. the signed-rate and structural-zero action parameterization;
4. the revised parameter count and identifiability analysis;
5. removal of the three regularizers from the primary objective;
6. any optional regularized fallback as a separate configuration;
7. the candidate/model caching design across reward modes;
8. the 32-cell diagnostic-subset manifest keys;
9. the candidate-count and regime-path sensitivity settings;
10. corrected canary runtime and full-sweep CPU projections;
11. exact tests for conditional-mean survey fitting and regime fit/deployment
    consistency;
12. confirmation that the scientifically void run remains preserved and excluded from
    all claims;
13. confirmation that no code, dependency, snapshot, or job changes occur before this
    updated plan is approved.

If any decision is not implementable, report the concrete mathematical or server
constraint and stop. Do not substitute another method without approval.
