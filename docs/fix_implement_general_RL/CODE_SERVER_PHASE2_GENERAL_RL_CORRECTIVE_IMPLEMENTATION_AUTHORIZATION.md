# Phase 2 authorization: correct and verify the hidden general-RL baselines

## Controlling inputs

Use these reports as the controlling audit and plan:

1. `PHASE1B_GENERAL_RL_DEEP_PAPER_VERIFICATION_AND_CORRECTIVE_PLAN.md`
2. `PHASE1_GENERAL_RL_BASELINE_PAPER_ALIGNMENT_AUDIT.md`

The full-paper findings in Phase 1B supersede the preliminary paper interpretations in Phase 1.
Local-code findings from both remain relevant.

## Approved scientific decisions

The following decisions resolve D1-D6.

| Decision | Approval |
|---|---|
| D1 Delphic | Retain it as **Delphic-inspired** and add it to the corrected hidden manifest, subject to the compatibility requirement below. |
| D2 BA-MCTS | Implement the paper's Eq.-4-style in-tree ensemble-belief update. Do not implement the outer policy/value distillation loop for this evaluation-time planner baseline. |
| D3 models | Lightweight benchmark-compatible models are acceptable. Preserve and disclose the shared-model limitation, and match common ensemble capacity across methods where they use the same model role. |
| D4 extra baselines | Do not add CQL or IQL to the retained experiment now. A cheap behavior-policy/behavior-cloning return floor may be prepared later as a contextual diagnostic, but it is not part of this implementation phase or the retained four. |
| D5 compute | Use compute-scaled primary values: BA-MCTS 256 simulations and depth 8; Delphic 20 worlds. Prepare, but do not launch, preregistered return-blind mechanism sensitivities at BA-MCTS `{128/5, 256/8, 512/12}` and Delphic world counts `{10,20,30}`. |
| D6 naming | Reader-facing names for all four must say **paper-inspired adaptation**. After the corrections, internal method IDs may remain stable for compatibility, but no text may imply an official reproduction. Delphic must explicitly retain the `-inspired` label. |

Keep the primary offline-data budget matched at **4,000 transitions**, episode-preserving, with the
existing 20% diagnostic holdout. Do not select data budgets or hyperparameters from performance
returns.

## Important refinement 1: Delphic compatibility is not optional

Phase 1B established that privileged behavior creates genuine hidden confounding in the hidden
benchmark. That makes Delphic scientifically relevant. It does **not** make arbitrary random
projection worlds observationally compatible.

Do not satisfy the Delphic task by only renaming the method and adding it to the manifest. Implement
the smallest defensible benchmark-compatible world-construction procedure that makes candidate
worlds reproduce the observed behavior distribution to a preregistered tolerance. This need not be
the paper's full neural ELBO, but it must include all of the following:

1. a fitted behavior-action likelihood conditional on public history/features and each latent-world
   hypothesis;
2. observed transition and public-reward likelihood or prediction diagnostics;
3. a rule that rejects, filters, weights, or refits worlds that are materially incompatible with
   held-out observed data;
4. a minimum surviving world-diversity requirement before cross-world Q variance is interpreted as
   Delphic uncertainty;
5. a constructed test where observationally incompatible worlds fail the gate;
6. a constructed test where compatible worlds agree observationally but disagree counterfactually,
   producing non-zero `Var_w Q_w`;
7. explicit disclosure that this is a lightweight compatibility approximation rather than the
   paper's ELBO-trained world model.

Preregister the numerical compatibility and diversity diagnostics before opening any performance
returns. If a defensible compatibility criterion cannot be implemented without a larger redesign,
stop on Delphic and report the blocker; do not silently retain arbitrary random worlds.

## Important refinement 2: shared-model fairness

Where RefPlan, OGSRL, and BA-MCTS use the same public ensemble as the world-model component, give
them the same training data, feature representation, ensemble size, bootstrap procedure,
regularization, and random-seed schedule unless a paper-defining mechanism requires otherwise.

The Phase 1B table proposed seven ensemble members only for RefPlan. Do not give RefPlan a larger
base ensemble while leaving otherwise comparable methods at five without scientific justification.
Choose one common primary ensemble size before canary execution. Seven is acceptable if used by all
applicable methods and its runtime is acceptable; otherwise retain five for all and register seven
as a shared sensitivity. Delphic's world count is a different algorithmic object and need not equal
the shared ensemble size.

## Implementation scope

Implement and verify Stages 1-9 from the Phase 1B plan, with the refinements below. Prepare Stages
10-12, but do not launch the full experiment.

### 1. Provenance isolation

- Do not switch branches in, reset, clean, or destructively manipulate the current dirty working
  tree.
- Do not make a commit that accidentally includes ecology-baseline work or unrelated files.
- Prefer an isolated git worktree based on `5f9cf32`, then import only the identified hidden-general
  paths/hunks and new general-baseline files.
- Record a path-by-path source provenance manifest and SHA-256 digest before correction and after
  correction.
- Confirm that active PLUS/MOOR jobs execute from their frozen copied code and are unaffected.

If isolated import cannot be done without losing or conflating changes, stop and report exact paths
and conflicts.

### 2. RefPlan

- Restore deployment belief updating and make the current belief materially affect every hidden
  planning call.
- Marginalize the model belief in candidate-plan scoring; do not merely compute a posterior
  diagnostic or select a single posterior-mode model.
- Retain the uncertainty penalty and document the discrete-action sequence-planning adaptation.
- Disclose that the paper's conservative prior policy is omitted unless a genuine prior policy is
  subsequently implemented.
- Add toy tests proving posterior sensitivity and behavioral distinction from MOPO.

### 3. OGSRL

- Run the existing Lagrangian constrained optimizer in hidden mode over the public guarded model and
  public reward/risk channels.
- Verify that actor, critic, and dual variables actually update and influence actions.
- Preserve the guardian/support restriction and deployment feasibility mask.
- Remove theoretical safety-guarantee language under the private hidden safety objective.
- Before accepting the implementation, measure the public risk-target prevalence without opening
  performance returns. If it is degenerate or the constraint cannot bind, stop and report rather
  than inventing a new safety target post hoc.

### 4. BA-MCTS

- Carry a categorical ensemble belief as part of each simulated node/history.
- Update it inside the tree using a numerically stable Eq.-4-style public transition likelihood and,
  where valid and observable, public reward likelihood.
- Do not use private state, private reward, `r`, `K`, or private safety quantities in the update.
- Define zero-likelihood handling, log-weight normalization, and belief reset/flooring explicitly.
- Sample or marginalize successor models from the updated node belief; do not fix one root model for
  the entire rollout.
- Discrete actions justify omitting action progressive widening. Document the chosen public-state
  bucketing/state-widening treatment.
- Add exact toy tests for posterior concentration, history-dependent actions, and distinction from
  RefPlan and root-sampling BAMCP.

### 5. Delphic-inspired

- Implement the compatibility requirements above.
- Preserve cross-world Q variance and pessimistic control.
- Add the corrected method to the hidden manifest only after compatibility, privacy, and mechanism
  tests pass.
- Document privileged-behavior confounding and show with a constructed test that the ambiguity
  channel vanishes or materially decreases when the relevant hidden-information gap is removed.

### 6. Privacy and mechanism tests

Implement the Phase 1B relabel-invariance, value-leakage, path/ordering, deterministic-seed, and
constructed-mechanism tests for all four methods. Test numeric payloads and derived arrays, not only
forbidden field names.

### 7. Return-blind adequacy checks

Implement diagnostics for action/episode coverage, holdout dynamics error, behavior likelihood,
guardian support and risk prevalence, ensemble diversity, BA-MCTS belief movement/tree statistics,
and Delphic compatibility/world diversity. These diagnostics must not read task returns or private
evaluation summaries.

### 8. Canary and runtime accounting

After unit tests pass, run only deterministic non-performance canaries needed to establish runtime,
memory, artifact integrity, and mechanism operation. Do not run the full experiment.

The Phase 1B phrase "1-2 CPU-days wall" conflates quantities and is not an actionable deadline
estimate. Report separately:

- aggregate actual CPU-hours;
- aggregate allocated core-hours;
- elapsed wall time at each proposed concurrency;
- queue delay separately from execution time;
- per-method fit and evaluation distributions, including tail estimates;
- peak CPUs and memory;
- expected completion time for a 2-hour, 4-hour, and 8-hour execution window.

Use measured canaries rather than only operation-count inference. Do not inspect performance
returns while benchmarking runtime.

## Acceptance gates before preparing a full manifest

All gates must pass:

1. isolated, reproducible corrected source snapshot and digest;
2. full existing tests plus new privacy and mechanism tests;
3. RefPlan hidden actions demonstrably depend on the deployment posterior;
4. OGSRL hidden ConOpt demonstrably trains and its constraint channel is non-degenerate;
5. BA-MCTS belief demonstrably updates inside simulated histories;
6. Delphic candidate worlds pass the preregistered observational-compatibility and diversity gates;
7. no private-information access in any hidden method;
8. matched 4,000-transition inputs and matched shared-model hyperparameters;
9. deterministic canaries and measured resource estimates;
10. no performance returns inspected and no interference with PLUS/MOOR jobs.

Failure of a defining-mechanism gate is a blocker, not a warning. Do not rename a failure away
unless explicitly returning for a new user decision.

## Required return

Write `PHASE2_GENERAL_RL_CORRECTIVE_IMPLEMENTATION_REPORT.md` containing:

1. exact files and commits/digests;
2. implementation account for each method;
3. paper mechanism versus adaptation matrix;
4. Delphic compatibility definition, thresholds, and evidence;
5. shared-model matching decision;
6. all test commands and results;
7. privacy and return-blindness evidence;
8. canary runtime/resource measurements;
9. remaining deviations and scientific limitations;
10. proposed frozen 4,000-transition manifests, not submitted;
11. GO/NO-GO verdict for a limited corrected general-RL canary;
12. any decisions still requiring approval.

Stop after the report. Do not submit the full general-RL experiment and do not inspect comparative
performance returns without separate explicit authorization.
