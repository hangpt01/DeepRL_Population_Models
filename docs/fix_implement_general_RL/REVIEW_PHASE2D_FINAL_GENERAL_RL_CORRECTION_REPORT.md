# Review of the Phase 2D final general-RL correction report

## Verdict

**Nearly ready, but do not launch yet.**

Phase 2D correctly resolves the previous blockers:

- the fourth method is now an honest episode-bootstrap fitted-Q ensemble with a neutral canonical
  ID and an explicit Delphic-motivated-only provenance statement;
- OGSRL's behavior budget, actor constraint, and deployment comparison now share a normalized
  25-step cost definition;
- RefPlan reevaluates a separate public policy prior along simulated proposal histories;
- lifecycle privacy and thread-parity coverage are substantially stronger;
- the manifest is unsubmitted and the work remains isolated.

Two algorithm-mechanism questions and the limited-canary registration remain before GO.

## F1 — Required mechanism check: is value-disagreement pessimism operationally active?

The registered score is

`mean_Q - 0.1 * variance_Q`.

The two return-blind probes report mean variances of approximately `0.007–0.050`, implying an
average penalty around `0.0007–0.005`. The report does not give the Q-value scale, action gaps, or
whether the registered penalty changes any action. A method called value-disagreement pessimism
must not silently reduce to ensemble-mean fitted Q because its penalty is numerically negligible.

### Required return-blind audit

Without changing `lambda_V`, report on the frozen train/holdout public states for the healthy and
sink probes:

1. distribution of `mean_Q`, `sqrt(variance_Q)`, `variance_Q`, and `lambda_V*variance_Q`;
2. distribution of the gap between the best and second-best ensemble-mean actions;
3. fraction of states where the registered pessimistic score selects a different action from
   `argmax mean_Q`;
4. fraction where the selected action changes across a small **preregistered mechanism sensitivity**
   `{lambda_V = 0, 0.1, 0.3, 1.0}`, with no policy returns read;
5. a constructed test where disagreement pessimism must change the selected action;
6. Q and penalty units after public reward normalization.

Do not choose a new coefficient solely to force an action-change percentage. If `0.1` is effectively
inert, return for a scientific decision among:

- retain it and rename the method `bootstrap conservative Q ensemble`;
- register a scale-normalized disagreement penalty fixed independently of returns;
- select one coefficient from paper/algorithmic provenance plus a preregistered sensitivity.

## F2 — Required consistency check: OGSRL expected cost under model uncertainty

The report says deployment propagates the public dynamics ensemble's **predictive mean** and then
computes low-abundance occupancy. For the convex shortfall cost,

`c(E[o'])` is generally not equal to `E[c(o')]`.

Cost of the predictive mean can underestimate low-abundance risk and may be inconsistent with actor
training if actor rollouts sample ensemble/process uncertainty.

### Required read-only trace

Document exactly, for behavior-budget fitting, actor training, and deployment:

- whether next observations are observed, sampled, member-specific, or ensemble means;
- whether cost is applied before or after averaging across particles/members;
- number and seeding of model/process/observation draws;
- whether actor training and deployment estimate the same expected normalized `C_25` object.

The scientifically preferred deployment statistic is a deterministic Monte Carlo/common-random-
number estimate of

`E_model,process,observation[C_H]`,

applying `c(o')` within each rollout before averaging. If the current code instead computes
`C_H` on one predictive-mean trajectory, return the exact file/function change and runtime impact
needed to make training and deployment uncertainty treatment consistent. Do not implement until
the audit is reviewed.

## F3 — Limited canary must be separately registered

The prepared 1,152-row manifest is the full corrected hidden arm, not a limited canary. A GO for a
limited canary must not be interpreted as permission to submit all 1,152 rows.

Prepare a plan, not a manifest yet, for a small method-balanced diagnostic containing:

- at least one healthy/recoverable population and one demographic sink;
- all four environment families;
- low and high observation-noise conditions, including a nonzero hidden-information condition;
- all four methods with matched public datasets, seeds, and 4,000-transition budget;
- enough cells to exercise RefPlan posterior/prior updates, OGSRL safety and OOD duals, BA-MCTS
  in-tree belief movement, and bootstrap-Q disagreement;
- structural/mechanism acceptance criteria fixed before returns;
- quarantined performance outputs until acceptance;
- exact runtime, memory, CPU, array-concurrency, scheduler, dependency, stop, and retry policy;
- a clear distinction between canary validity diagnostics and comparative performance analysis.

Use measured runner-level timing for complete fit/evaluate rows. The Phase 2D `77–78 task-hour`
figure is an extrapolation and must not be treated as a measured launch budget.

## Accepted without further redesign

- canonical fourth-method ID and honest label;
- episode-level bootstrap Q members with no direct output perturbation;
- matched 4,000-transition input budget;
- common ensemble size 5 for the model-based methods;
- BA-MCTS 256 simulations/depth 8;
- OGSRL public shortfall definition and unclipped train-behavior normalized budget;
- RefPlan history-conditioned proposal prior;
- privacy lifecycle structure;
- one thread/one CPU registration;
- MOPO exclusion and immutable historical artifacts;
- no general-RL performance returns inspected.

## Required next response

Perform an **audit and launch-plan step only**. Do not edit runtime code, regenerate manifests or
snapshots, launch jobs, inspect policy returns, merge the worktree, or touch PLUS/MOOR.

Write `PHASE2D2_GENERAL_RL_MECHANISM_AND_CANARY_AUDIT.md` containing:

1. response to F1 with the requested penalty/action-scale diagnostics;
2. response to F2 with a line-by-line uncertainty-treatment trace;
3. any required correction plan and expected runtime impact;
4. proposed limited-canary cell set and exact row count;
5. preregistered return-blind acceptance table;
6. measured complete-row runtime plan and scheduler resource request;
7. snapshot/manifest procedure after any approved correction;
8. final `GO`, `GO-CONDITIONAL`, or `NO-GO` recommendation.

Stop after the report. No recommendation in that report authorizes submission.
