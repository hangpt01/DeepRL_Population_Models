# Phase 2E authorization: OGSRL uncertainty correction and frozen canary package

## Controlling inputs

Read and follow:

1. `REVIEW_PHASE2D_FINAL_GENERAL_RL_CORRECTION_REPORT.md`
2. `PHASE2D2_GENERAL_RL_MECHANISM_AND_CANARY_AUDIT.md`
3. `PHASE2D_FINAL_GENERAL_RL_CORRECTION_REPORT.md`

The Phase 2D2 audit is accepted. It correctly identifies one remaining runtime-code blocker:
OGSRL deployment computes shortfall on predictive-mean trajectories rather than expected pathwise
shortfall under public predictive uncertainty.

This authorization permits the OGSRL correction, specified tests, acceptance tooling, and creation
of frozen but unsubmitted 4-row and 64-row canary packages. It does **not** authorize any Slurm job,
local performance evaluation, return inspection, or full-arm submission.

## Settled value-disagreement decision

Keep:

- canonical method ID `ensemble_value_disagreement_pessimism`;
- 20 episode-bootstrap conservative-Q members;
- `lambda_V=0.1`;
- score `mean_Q - lambda_V * variance_Q`;
- no support/ambiguity multiplier and no direct random output perturbation.

Do not change `lambda_V` from the observed action-change percentages. Register and disclose:

- its influence is operational but sparse on the two audited datasets;
- it changed approximately 0.125–0.375% of holdout actions relative to ensemble-mean Q;
- this is a mechanism observation, not evidence of performance.

Add the constructed two-action regression test from Phase 2D2. Do not make a minimum real-data
action-change percentage beyond nonzero on the already frozen Amur/vulture mechanism fixtures. Do
not inspect returns.

## Authorized OGSRL correction

Retain the settled public cost and behavior budget:

```text
s_low = Q_0.20(positive training observations)
c(o') = clip((s_low-o')/s_low, 0, 1)
C_H = sum_t [gamma^t / sum_j gamma^j] c(o_{t+1})
gamma = 0.95
H_cost = 25
safety_budget = mean over complete training behavior episodes of C_25
```

There is no budget floor/cap and no private-safety guarantee.

### Shared pathwise predictive-cost primitive

Refactor OGSRL so actor training and deployment call one tested primitive that:

1. begins from registered public-belief starts;
2. samples a `PublicDynamicsEnsemble` member according to the registered member distribution;
3. samples the fitted aggregate public predictive residual in log-observation space;
4. applies `c(o_next)` **inside each sampled trajectory before averaging**;
5. samples continuation actions from the trained public actor;
6. forms normalized pathwise `C_25`;
7. averages pathwise occupancies only after each trajectory's cost is complete.

Describe the estimand only as

`E_belief,bootstrap-member,public-predictive-residual,actor[C_25]`.

Do not claim the observation-space model separately identifies ecological process noise and survey
measurement noise.

### Deployment

- Register `ogsrl_deployment_rollouts=256`.
- For each candidate forced first action, use 256 public predictive paths.
- Use a balanced member schedule over the five ensemble members.
- Use registered Gaussian predictive-residual draws and categorical actor continuation draws.
- Use common random numbers across candidate first actions.
- Derive a local deterministic stream from immutable method seed, belief timestep, rollout index,
  and depth. Do not consume or mutate the actor's deployment RNG.
- Evaluate a fixed 25-step receding risk horizon at every decision, including near the end of the
  50-step evaluation episode. Document that this is a registered conservative receding-horizon
  public-risk proxy matched to the 25-step behavior-data episodes.
- Compare the resulting expected normalized `C_25` with the unchanged train-only behavior budget.

If fixed 25-step simulation is impossible near episode termination because the public model or
runner refuses post-horizon predictions, stop and report rather than silently shortening the
horizon or reusing the `C_25` budget for another functional.

### Tests

Add:

- actor and deployment call the same pathwise-cost helper;
- pathwise cost-before-average test;
- strict Jensen test on a distribution crossing `s_low`;
- all-zero/all-one and exact normalization tests;
- deterministic repeat and local-RNG nonmutation tests;
- common-random-number equality across candidate-action comparisons;
- balanced ensemble-member schedule test;
- constructed action-risk and dual-response test;
- unchanged budget equality to train-only mean `C_25`;
- no floor/cap regression test.

Run the targeted and full test suites with one thread.

## Accepted limited-canary design

Prepare a separate limited diagnostic with exactly 16 scenarios and 64 method rows:

- populations: Amur tiger and Egyptian vulture;
- families: `ricker`, `allee`, `theta`, `regime`;
- observation noise: `sigma_obs=0.0` and `0.4`;
- reward mode: `safe` only;
- methods: `refplan`, `ogsrl`, `bamcts`, `ensemble_value_disagreement_pessimism`;
- collection seed: 116;
- 4,000 transitions as 160 complete 25-step episodes;
- learned public filter and hidden demographics;
- evaluation seed: 9001;
- one 50-step evaluation episode per row;
- identical public dataset path/hash across methods within each scenario.

This is a mechanism and pipeline canary, not a comparative performance sample. Generated
performance fields must remain quarantined and unread until structural acceptance and separate human
authorization.

## Prepare two new manifests, submit neither

After code/tests pass and a new snapshot is frozen, create:

1. a **4-row timing-preflight manifest** containing one row per method for Egyptian vulture,
   `regime`, `sigma_obs=0.4`, safe reward;
2. the complete **64-row limited-canary manifest** above.

The four rows must be byte-identical to the corresponding four rows in the 64-row manifest except
for manifest-local index/metadata fields. Freeze and hash both. Do not alter the existing 1,152-row
prepared full-arm manifest; preserve it as prior unsubmitted history.

## Return-blind acceptance tooling

Implement or prepare acceptance tooling that can read only:

- source/snapshot/manifest/data hashes;
- dataset row/episode/action counts;
- fit completion and finite diagnostics;
- method mechanism diagnostics specified in Phase 2D2;
- action outputs required for mechanism invariants;
- timing, CPU, RSS, scheduler, dependency, and exit metadata;
- privacy/thread parity receipts.

It must be technically unable to deserialize or report policy reward, return, survival, ranking, or
cross-method performance fields. Add tests proving forbidden fields are inaccessible.

Use the Phase 2D2 acceptance table unchanged except where this authorization clarifies the OGSRL
estimand. Failure actions remain fixed and no algorithmic failure gets an automatic retry.

## Scheduler diagnosis, without submission

Do not assume `comp` is usable merely because it appeared in the plan. The earlier ecological run
observed severe `comp/normal` queue delay and immediate `m3h/m3h` placement.

After freezing the 4-row package, run only non-submitting scheduler probes such as `sbatch
--test-only` for eligible partitions/QOS choices. Compare at least:

- `comp/normal`;
- `m3h/m3h`, if the account and executable are eligible.

For the four-row timing preflight, prepare commands for:

- one CPU/task;
- 8 GiB/task;
- 2-hour task limit;
- all BLAS/OpenMP thread variables pinned to one;
- both serial `%1` and parallel `%4` variants;
- no requeue and no automatic retry.

Recommend the partition and concurrency from current queue estimates and scientific timing needs,
but submit nothing. Report queue delay separately from expected execution.

The 60-row remainder must remain separately gated after the four-row timing receipt. Do not encode
automatic submission of the remainder.

## Isolation constraints

- Work only in `/fs04/scratch2/ce25/general_rl_phase2_iso`.
- Do not merge into the dirty main worktree.
- Do not inspect performance returns.
- Do not launch local or Slurm evaluation jobs.
- Do not touch PLUS/MOOR jobs, runtime, manifests, acceptance files, or artifacts.
- Do not modify old frozen snapshots.
- Keep one thread/one allocated CPU registration.
- Keep matched 4,000-transition data, shared ensemble size 5, and BA-MCTS 256/depth 8.

## Required report

Write `PHASE2E_OGSRL_FIX_AND_CANARY_PACKAGE_REPORT.md` containing:

1. exact OGSRL code changes and mathematical estimand;
2. tests and full-suite results;
3. EVD regression test and unchanged coefficient evidence;
4. refreshed isolated snapshot and registration hashes;
5. 4-row and 64-row manifest paths, counts, and hashes;
6. return-blind acceptance tool and forbidden-field tests;
7. non-submitting scheduler diagnosis and exact prepared commands;
8. measured non-performance local mechanism timing, if available without running evaluation rows;
9. remaining limitations;
10. `GO-CONDITIONAL`, `NO-GO`, or `READY-FOR-4-ROW-PREFLIGHT` recommendation.

Stop after the report. No recommendation authorizes submission of even the four-row preflight.
