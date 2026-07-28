# Planning Guide: Ricker-Only PLUS Under Cross-Family Ecological Stress Tests

## Purpose

Use this document to plan a corrected experiment in which both ecological baselines make the same
high-level mechanistic-family assumption:

- PLUS uses a finite bank containing **only Ricker candidate POMDPs**.
- MOOR continues to fit **one Ricker model**.
- The hidden benchmark environments continue to span Ricker, Allee, theta-logistic, and
  regime-switching dynamics.

This design answers the intended question:

> How robust are paper-aligned ecological baselines that assume Ricker dynamics when the hidden
> ecological process is Ricker or belongs to a different mechanistic family?

The existing cross-family PLUS configuration answers a different question---whether PLUS can use
online Bayesian evidence to discriminate among a bank that already contains Ricker, Allee, theta,
and regime candidates. Do not confuse or pool these two designs.

## Copy-paste starter prompt

```text
Plan a new Ricker-only PLUS ecological stress-test experiment. Read
PLAN_GUIDE_RICKER_ONLY_PLUS_STRESS_TEST.md completely, followed by
CHAT_HANDOFF_ECOLOGY_BASELINES_PLUS_MOOR.md and its ordered scientific-control files.

The intended scientific design is that PLUS and MOOR both assume the Ricker family. The hidden
environment must still span Ricker, Allee, theta-logistic, and regime-switching families so that
the three non-Ricker families are genuine out-of-family misspecification stress tests for both
methods. Do not include Allee, theta, or regime candidates in the new PLUS bank.

Treat the existing 16-candidate cross-family PLUS run as a separate frozen experiment. Do not edit,
overwrite, relabel, or silently reuse its runtime snapshot, manifests, artifacts, acceptance
record, or results. Before planning execution, verify current Slurm and artifact status without
opening quarantined comparative returns before their applicable acceptance gate passes.

Prepare a return-blind implementation and experiment plan covering: the new method/configuration
identifier; construction of a Ricker-only candidate bank; candidate count and parameter diversity;
offline episode use; frozen seeds; PBVI settings; new manifests and hashes; tests; acceptance;
resource estimates; and a small canary before any full run. Explain every departure from the
existing frozen runtime. Do not implement or submit jobs until the plan is approved.
```

## Scientific correction

### Existing frozen PLUS design

The frozen `plus_adapted_mechanistic_pbvi` configuration contains 16 candidates:

- four Ricker;
- four Allee;
- four theta-logistic; and
- four regime-switching candidates.

That is a registered cross-family extension of PLUS. Because every benchmark family is represented
in the bank, non-Ricker environment cells do not impose family-level misspecification on PLUS.
They test structural-model discrimination and finite-bank parameter adequacy instead.

### Required new PLUS design

The new PLUS bank must contain Ricker candidates only. Each candidate must still be a complete,
fixed mechanistic POMDP with its own:

- Ricker transition parameters;
- observation model;
- discretized transition and observation kernels;
- within-candidate state belief; and
- independently solved PBVI value representation.

PLUS must continue to update a Bayesian posterior over the fixed Ricker candidates using sequential
action-observation evidence and select actions from posterior-weighted candidate PBVI values.
Continuous parameters must not be updated online.

Allee thresholds, theta exponents, regime states, regime multipliers, and regime transition
matrices must not occur in the candidate bank or affect the Ricker-only fitting, kernels, filter,
planner, or action values.

## Candidate-bank planning requirements

The plan must explicitly preregister the following before any comparative returns are inspected:

1. **Candidate count.** Prefer retaining 16 total candidates for a clean comparison with the
   existing 16-candidate cross-family PLUS configuration, unless fit diversity, runtime, or a
   paper-aligned parameter grid justifies another count.
2. **Candidate construction.** Specify how distinct fixed Ricker parameterizations are obtained.
   Candidate sources may include a full-history deterministic fit, deterministic episode-bootstrap
   MAP fits, or a preregistered Ricker parameter grid. Do not select candidates using evaluation
   returns or the hidden true parameters.
3. **Candidate diversity.** Define return-blind checks for duplicate or nearly duplicate Ricker
   candidates, parameter coverage, finite objectives, and usable kernels.
4. **Prior.** Use a uniform initial candidate prior unless a different return-blind prior is
   scientifically justified and preregistered.
5. **Offline data.** Preserve complete episode order and boundaries. Keep the registered train and
   holdout split unless a new split is explicitly justified and frozen.
6. **Bank-size sensitivity.** Plan smaller/default/larger Ricker-only banks on a registered subset
   if computationally feasible. Do not use the sensitivity to select the primary configuration
   after viewing control returns.

The plan must state whether retaining 16 candidates means one full-history fit plus 15 deterministic
bootstrap/grid candidates or another preregistered construction. It must not mechanically convert
the current four-per-family allocation without checking that the resulting Ricker bank is diverse,
identifiable, and computationally defensible.

## Stress-test matrix

Retain the registered hidden-environment factors:

- populations: Amur tiger and Egyptian vulture for the diagnostic canary;
- true families: Ricker, Allee, theta-logistic, and regime-switching;
- observation noise: `0`, `0.1`, `0.2`, and `0.4`;
- primary reward mode: safe, unless a separately approved manifest changes it;
- inherited matched data budget: 4,000 transitions, labelled
  **data-adequacy-unverified**.

Under the corrected design, interpret family specification as follows:

| Hidden environment family | Ricker-only PLUS | Ricker-only MOOR |
|---|---|---|
| Ricker | family correctly specified | family correctly specified |
| Allee | deliberately misspecified | deliberately misspecified |
| Theta-logistic | deliberately misspecified | deliberately misspecified |
| Regime-switching | deliberately misspecified | deliberately misspecified |

The hidden family label and private demographic parameters must never enter fitting, candidate
construction, filtering, planning, or online updates.

## Fairness and interpretation

The two methods share a Ricker-family assumption but remain algorithmically different:

- PLUS represents parameter uncertainty with several fixed Ricker candidate POMDPs and performs
  online Bayesian discrimination among them.
- MOOR commits offline to one fitted Ricker model and does not maintain an online model posterior.

Therefore the experiment compares robustness of two different inference/control procedures under
the same family-level inductive bias. It does not make the methods identical.

Poor non-Ricker performance may reflect family misspecification, weak action-effect identification,
observation noise, finite candidate/path/PBVI budgets, or the unresolved 4,000-transition data
bottleneck. Do not attribute it solely to algorithm quality.

Performance differences between the existing cross-family PLUS and the proposed Ricker-only PLUS
may be reported as a separate **candidate-family coverage sensitivity**. They must not be presented
as if they were results from the same primary configuration.

## Provenance and naming

Do not modify the frozen cross-family runtime or its records:

- snapshot commit: `fd50c38c9fad9ca113c224826b9bebddedea67d0`;
- runtime digest: `f70ec7122263ec00da0a4950994dfe483ed29c71bc15e477007ca9ae0a6d2333`;
- existing fit manifest hash:
  `a8f39d83eb70a32be2c42fabecc56ca4af4d3d40edb97f87242934085fd7ee86`;
- existing safe-plan manifest hash:
  `f0322718a78d485827c39661e2f1ded4cbde1d8bccd4cd053f4d1fe93dec087d`.

The new design requires a new configuration identifier, new snapshot/digest if runtime behavior or
configuration semantics change, new manifests, and separate artifacts. A suitable provisional
identifier is:

`plus_adapted_ricker_only_pbvi`

Use a display label such as:

> Paper-aligned PLUS adaptation with a fixed Ricker-only candidate-POMDP bank and PBVI.

Do not call it exact PLUS. Do not relabel the existing cross-family artifacts with this identifier.

## Required implementation tests

Before a canary is launched, add or verify tests demonstrating that:

1. every PLUS candidate is Ricker and satisfies the registered Ricker equation;
2. changing inactive Allee, theta, or regime constants cannot change fitting, kernels, beliefs,
   PBVI values, or actions;
3. no hidden true-family label or private demographic parameter is read;
4. candidate parameters remain fixed during deployment;
5. every candidate is solved independently;
6. one normalized state belief is maintained per candidate;
7. the model posterior remains finite and normalized and responds to synthetic Ricker evidence;
8. posterior-weighted PBVI action selection is deterministic under frozen seeds;
9. candidate serialization includes construction provenance and hashes; and
10. old cross-family and new Ricker-only method routing cannot be confused.

## Acceptance and execution sequence

1. Verify the current Slurm and artifact state of all existing `adapt32-*` work without reading
   comparative returns prematurely.
2. Produce and approve a return-blind Ricker-only implementation/experiment plan.
3. Create a new frozen snapshot, runtime digest, manifests, and acceptance specification applicable
   to the emitted diagnostics.
4. Run unit, privacy, family-invariance, determinism, provenance, and small exact-POMDP tests.
5. Run a very small structural canary before the registered 32-cell diagnostic.
6. Run the 32-cell diagnostic only after the canary passes and resource use is approved.
7. Apply the registered return-blind acceptance gate before opening comparative returns.
8. Request explicit permission before interpreting accepted performance returns.
9. Do not authorize the full 288-cell sweep from this planning guide alone.

## Reporting language

Use:

> Both ecological baselines assumed Ricker dynamics. Ricker environments assessed in-family
> performance, whereas Allee, theta-logistic, and regime-switching environments provided
> preregistered out-of-family mechanistic-misspecification stress tests.

Also disclose that PLUS used a finite bank of Ricker parameterizations while MOOR used one fitted
Ricker model. Retain all safe-mode, finite-compute, PBVI-substitution, and data-adequacy caveats.

