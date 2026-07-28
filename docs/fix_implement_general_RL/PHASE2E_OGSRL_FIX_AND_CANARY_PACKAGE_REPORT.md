# Phase 2E — OGSRL fix and frozen canary package report

## 1. Outcome and scope

Phase 2E was implemented only in the isolated worktree
`/fs04/scratch2/ce25/general_rl_phase2_iso` under
`CODE_SERVER_PHASE2E_OGSRL_FIX_AND_CANARY_PACKAGE.md`.

The remaining OGSRL blocker is corrected: hidden actor training and deployment
now use one shared pathwise public-predictive cost primitive. Deployment applies
low-abundance shortfall to sampled next observations inside each trajectory,
forms normalized `C_25` per completed path, and averages only those pathwise
occupancies. It no longer evaluates shortfall on an ensemble-mean trajectory.

The EVD constructed action-switch regression is present with
`lambda_V=0.1` unchanged. A new frozen package contains separate 4-row and
64-row manifests, both explicitly unsubmitted. A validity-only receipt reader
rejects outcome fields before JSON payload deserialization. Four scheduler
choices were checked only with `sbatch --test-only`; no scheduler job was
submitted and the resulting estimate IDs were absent from `squeue`.

No local or Slurm evaluator was run. No policy, survival, or ranking return was
read. The main dirty worktree was not modified, the old 1,152-row manifest was
not altered, the isolated worktree was not merged, and PLUS/MOOR was not
touched.

**Recommendation: READY-FOR-4-ROW-PREFLIGHT, pending separate human submission
authorization.** This report does not authorize submission of even the four
rows. The other 60 rows remain separately gated.

## 2. Exact OGSRL implementation

### Registered functional and estimand

The behavior-derived public proxy is unchanged:

```text
s_low = Q_0.20({positive train observations})
c(o_next) = clip((s_low-o_next)/s_low, 0, 1)
C_25 = sum(t=0..24) [0.95^t / sum(j=0..24) 0.95^j] c(o_(t+1))
safety_budget = mean over complete train behavior episodes of C_25.
```

There is no budget floor or cap. Deployment estimates

```text
E_belief,bootstrap-member,public-predictive-residual,actor[C_25].
```

This is a public observation-space predictive estimand. The fitted residual is
an aggregate predictive residual; the implementation does not claim to
separately identify ecological process noise and survey measurement noise, and
the public proxy gives no guarantee for the private safety objective.

### Code changes

- `PlannerConfig.ogsrl_deployment_rollouts=256` is now registered separately
  from the unchanged `ogsrl_cost_horizon=25`.
- `PublicDynamicsMember.sample_next_from_standard_normal` accepts an externally
  registered Gaussian innovation. Existing `sample_next` delegates to it. This
  preserves the fitted predictive distribution while allowing common random
  numbers across candidate actions.
- `OGSRLPolicy._public_pathwise_cost_rollout` is the single actor/deployment
  primitive. It samples member identity and the log-observation predictive
  residual, samples categorical continuation actions, and computes
  `_public_low_abundance_cost(following)` before any trajectory averaging.
- `_public_rollouts` calls that primitive and adds the actor's reward and OOD
  channels to the returned pathwise records. Its normalized safety returns and
  dual therefore remain pathwise `C_25` estimates.
- `_public_belief_action_risks` calls the same primitive once for every forced
  first action. It uses 256 systematic/stratified weighted starts from the
  current public belief and a balanced five-member schedule: each member occurs
  51 or 52 times at every depth.
- One immutable local `SeedSequence` is derived from method seed, belief
  timestep, rollout count, fixed horizon, rollout index, and depth-indexed
  matrices. The member schedule, standard-normal residual matrix, and actor
  uniform matrix are generated once and reused for all candidate first actions.
  Deployment neither reads nor mutates `policy.rng`.
- Deployment now always simulates exactly 25 public-model steps, including near
  the end of the 50-step evaluation episode. This is the registered conservative
  receding-horizon proxy matched to the 25-step behavior episodes; it is never
  silently shortened.
- OOD feasibility remains the exact belief-weighted public guardian statistic.
  The safety side is the mean of completed sampled `C_25` paths and is compared
  with the unchanged train-only behavior budget.
- `run_real_manifest_row.py` accepts and records the deployment-rollout field
  and, for Phase 2E package rows only, writes a dedicated validity receipt.

Primary source hashes:

| File | SHA-256 |
|---|---|
| `src/real_ecology_benchmark/methods/ogsrl.py` | `da965bc320f3cbfaf381a46aaa77a7eb0ef7c3b6b7b493bee02e338544e6993b` |
| `src/real_ecology_benchmark/public_models.py` | `f23e8aa566a2e92bd017a5bd25b8f93accb39debabe30337cf0788241ab92abb` |
| `src/real_ecology_benchmark/general_canary_acceptance.py` | `fd5500dd15a148315b61f5624f03026ae424be09a65163bd20d5c4757e1220db` |

## 3. Mechanism and full-suite verification

### Added/retained mechanism gates

The Phase 2E tests cover:

1. actor and deployment both calling `_public_pathwise_cost_rollout`;
2. a crossing distribution with `mean(c(draw))=0.5` and
   `c(mean(draw))=0.0`, proving strict Jensen separation and cost-before-average;
3. all-zero, all-one, `gamma=1`, and exact hand-normalization examples;
4. repeatable deployment risks and byte-identical policy RNG state before and
   after repeated deployment calculations;
5. exact equality of member, residual, and actor-uniform CRN matrices across all
   forced first-action comparisons;
6. per-depth balanced five-member schedules with maximum count difference one;
7. constructed action-risk response and trained safety/OOD dual movement;
8. behavior budget equality to the exact train-only complete-episode mean;
9. no floor/cap, because both `safety_budget` and
   `deployment_safety_limit` equal that unmodified empirical mean; and
10. existing privacy lifecycle and one-thread checks.

Final targeted command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python \
  -m pytest tests/real/test_general_phase2e_canary.py \
  tests/real/test_general_paper_mechanisms.py -q -p no:cacheprovider
```

Result: **31 passed**.

Final full command used the same environment and interpreter with
`python -m pytest tests/ -q -p no:cacheprovider`. Collection contained 203
tests. Result: **203/203 passed**, exit 0. Measured test-suite resources were:

- wall time: 49.10 s;
- process user/system CPU: 43.89 s / 1.93 s;
- peak RSS: 306,736 KiB (299.55 MiB);
- one allocated CPU and all three thread variables fixed to one.

Additional gates:

- `scripts/check_docs.py`: `docs-check: OK`;
- current-source forbidden compatible-world, Delphic-uncertainty,
  direct-Q-perturbation, and predictive-mean-shortfall search: no hits in the
  operative source/current reader-facing files;
- live source/config/script/test digest matches the frozen snapshot source
  digest exactly;
- frozen code-tree digest independently rechecked and matched;
- old 1,152-row manifest hash independently rechecked and unchanged.

## 4. EVD regression and unchanged coefficient

`test_registered_disagreement_penalty_can_change_selected_action` registers the
constructed example:

```text
mean_Q = [1.00, 0.99]
variance_Q = [0.20, 0.00]
lambda=0:   [1.00, 0.99] -> action 0
lambda=0.1: [0.98, 0.99] -> action 1
```

The canonical method ID remains
`ensemble_value_disagreement_pessimism`, there are still 20 episode-bootstrap
conservative-Q members, and `lambda_V` remains exactly 0.1. There is no
support/ambiguity multiplier or direct random Q perturbation. Documentation and
the new package registration disclose that influence was operational but sparse
on the two frozen Phase 2D2 fixtures: approximately 0.125--0.375% of holdout
actions changed relative to ensemble-mean Q. This is mechanism evidence, not
performance evidence, and no coefficient was selected from returns.

## 5. Frozen package, manifests, and hashes

Package root:

```text
real_ecology_runs/general_phase2e_canary_20260720_v1/
```

The snapshot contains `src`, `configs`, `scripts`, `tests`, public ecology data,
the dependency lock, and `pyproject.toml`. Outcome paths are under the separate
`quarantine/performance` directory. No outcome exists because no evaluator ran.

| Artifact | Rows/files | SHA-256 |
|---|---:|---|
| `manifests/timing_preflight_4_rows.csv` | 4 rows | `9ca38d6a88a773a54850e51fdb0a94078df2a8ca4039aba8ffbc561bf330b4a0` |
| `manifests/limited_canary_64_rows.csv` | 64 rows | `e7272ec72138dba167191c0f15d420effb18e84adb0e753065384a2900ee13d9` |
| `manifests/registration.json` | registration | `c4fa70ea39e7a8868defe4c21cca00a7fd374843cd6d0295f626d266032696d4` |
| live `src+configs+scripts+tests` | 152 files | `c0b7180b1789d880540e110071333f5dd81c85b532c0424666b12137c51a78d7` |
| frozen snapshot source subset | 152 files | `c0b7180b1789d880540e110071333f5dd81c85b532c0424666b12137c51a78d7` |
| complete frozen `code/` tree | 161 files | `5d6f73c7f9575b7c9ca21ea1390f48b58c14e0fb1f9268dd0b656df0aedf3743` |
| frozen canary config | 1 file | `96ab2dfa11a4ad65f2ad26a438ee1b34bcf04d9af8d636a0bd024c5606213733` |
| prepared scheduler-command file | 1 file | `fb8ca3236ce5c43e82cc44ef6e36a1d81d44fa74fac4d5c120c81efa7a6a72cf` |

The limited manifest contains exactly 16 scenarios:

```text
{Amur tiger, Egyptian vulture}
x {ricker, allee, theta, regime}
x {sigma_obs=0.0, sigma_obs=0.4}
x safe reward only.
```

Each scenario has one row for each of RefPlan, OGSRL, BA-MCTS, and EVD, giving
16 rows per method and 64 total. All register collection seed 116, 4,000 rows,
160 complete 25-step episodes, hidden demographics, learned public filtering,
evaluation seed 9001, one 50-step evaluation episode, ensemble size five, and
BA-MCTS 256/depth 8. The four preflight rows are the Egyptian vulture / regime /
`sigma_obs=0.4` projections. After excluding only manifest-local `index` and
`package_role`, the four rows match the corresponding limited-manifest rows
field for field.

The prior unsubmitted full-arm manifest remains:

```text
real_ecology_runs/general_corrected_prepared/manifest_general_hidden.csv
SHA-256 e71f67daf7cd048577418f6e074e5ea4997fd0a23ecac61fc1419e9e09ab5a66
rows 1,152; unchanged
```

## 6. Return-blind acceptance tooling

The frozen acceptance entry point is
`code/scripts/check_general_phase2e_canary.py`. It imports the strict reader in
`general_canary_acceptance.py` and accepts only files whose basename is exactly
`validity_receipt.json`; it refuses `summary.json`, `episodes.csv`, and every
other filename before opening the file.

The Phase 2E runner emits this dedicated receipt only for rows whose
`package_role` begins with `phase2e_`. Its schema can contain only:

- source, manifest, and dataset hashes plus row identity;
- dataset target/actual/overshoot rows, episodes, episode length, number of
  actions, and logged count per action;
- fit completion and finiteness of explicitly whitelisted mechanism diagnostics;
- selected RefPlan, OGSRL, BA-MCTS, or EVD mechanism fields;
- row/fit/evaluation timing and RSS metadata;
- exit/dependency state; and
- the three registered thread counts.

It does not copy evaluator outcome fields. Before calling `json.loads` on a
payload, the reader scans raw JSON key tokens and rejects keys containing
reward, return, survival, ranking/rank, performance, collapse, persistence,
unsafe, true-state, or economic field fragments. It also enforces an exact
top-level schema, a 2 MB structural size cap, finite floats, exit code zero,
`submitted:false`, and all thread counts equal to one.

Tests establish that:

- a non-receipt filename is rejected without calling `Path.read_bytes`;
- a receipt containing `operational_return` is rejected while a patched
  `json.loads` proves the payload object was never deserialized;
- non-whitelisted fit diagnostics are absent;
- valid structural receipts load; and
- generated package registrations remain unsubmitted and refuse overwrite.

This tooling does not make performance output safe to inspect. Those files
remain quarantined until structural acceptance and a separate human decision.

## 7. Scheduler diagnosis — test-only

The exact expanded commands are frozen at:

```text
real_ecology_runs/general_phase2e_canary_20260720_v1/
  provenance/prepared_scheduler_commands.txt
```

All four commands register:

```text
sbatch --test-only
--cpus-per-task=1 --mem=8G --time=02:00:00 --no-requeue
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
array 0-3 with either %1 or %4
frozen code/config/manifest paths and frozen snapshot hash
```

They differ only by these exact scheduler choices:

```text
--partition=comp --qos=normal --array=0-3%1
--partition=comp --qos=normal --array=0-3%4
--partition=m3h  --qos=m3h    --array=0-3%1
--partition=m3h  --qos=m3h    --array=0-3%4
```

Final `sbatch --test-only` results on 2026-07-20 were:

| Partition/QOS | Concurrency | Estimated eligible start | Probe result |
|---|---:|---|---|
| `comp/normal` | `%1` | 2026-07-25 11:34:07 | eligible, about five days of queue delay |
| `comp/normal` | `%4` | 2026-07-25 11:35:18 | eligible, about five days of queue delay |
| `m3h/m3h` | `%1` | 2026-07-20 00:23:39 | eligible/immediate |
| `m3h/m3h` | `%4` | 2026-07-20 00:23:40 | eligible/immediate |

Slurm printed estimate-only IDs `58396666`, `58396673`, `58396674`, and
`58396679`. A direct `squeue` check returned no rows for them, confirming no job
was submitted. Queue estimates are transient and must be rechecked at any later
authorization point.

**Prepared recommendation: `m3h/m3h`, serial `0-3%1`.** `%4` is currently
eligible, but serial execution avoids shared-dataset creation races and resource
contention and gives cleaner per-method complete-row timing. Queue delay must be
reported separately from execution time. No automatic retry/requeue is enabled,
and no command for automatic submission of the 60-row remainder exists.

## 8. Return-blind local mechanism timing

One full-budget Amur tiger / Ricker / `sigma_obs=0.2` mechanism probe was run
locally with 4,000 transitions, train/holdout split, the registered models,
one thread, and only three calls to each fitted policy. It did not invoke
`ContinuousEvaluator` and did not read returns.

Corrected OGSRL measurements:

| Quantity | Measurement |
|---|---:|
| Fit wall | 16.7911 s |
| Fit process CPU | 16.7077 s |
| Mean deployment action | 126.118 ms |
| p95 deployment action | 126.754 ms |
| Full four-method probe wall | 21.7102 s |
| Probe peak RSS | 161.855 MiB |

The earlier Phase 2D mechanism probe reported 412.75 ms mean OGSRL action time;
the pathwise-vectorized correction is therefore faster in this return-blind
probe. The budget remained exactly `0.09816464664090638`, `H=25`, the safety
dual moved to `0.938467022586942`, all 11 actions had at least 181 logged rows,
and 256 deployment paths were used. These are mechanism/data diagnostics, not
performance results.

This is not the required complete runner-row timing. The 4-row preflight still
must measure full `run_real_manifest_row.py` wall/CPU/RSS and scheduler elapsed
under quarantine before any decision on the other 60 rows.

## 9. Remaining limitations and gate

- The dynamics model is a low-capacity observation-space bootstrap ensemble.
  Its residual aggregates public predictive error and does not separately model
  ecological process and survey emission noise.
- The deployment expectation is a finite 256-path deterministic-CRN Monte Carlo
  approximation, not an analytic expectation.
- The public low-abundance proxy does not guarantee private ecological safety.
- EVD's registered penalty is operational but sparse and remains an idea-level
  Delphic-motivated bootstrap-Q baseline, not Delphic uncertainty.
- RefPlan and BA-MCTS remain transparent benchmark adaptations rather than full
  deep published systems.
- No complete fit/evaluate row has yet been timed. The local mechanism timing
  cannot set a complete-row launch budget.
- Dataset SHA-256 equality across the four methods in each scenario can only be
  certified after the shared canary dataset exists. The validity receipts are
  prepared to enforce it.
- Scheduler start estimates are ephemeral. The 8 GiB / 2 h request remains a
  conservative preflight request until runner-level receipts exist.
- Performance outputs, if later produced under authorization, remain unread in
  quarantine. The 60-row remainder and the old 1,152-row arm remain unauthorized.

## 10. Final recommendation

**READY-FOR-4-ROW-PREFLIGHT, but not authorized to submit.**

The Phase 2E runtime blocker is resolved, all 203 tests pass, the EVD coefficient
decision is unchanged and regression-tested, validity-only tooling is frozen,
both manifests are independently hashed and unsubmitted, and `m3h/m3h %1` is
the current scheduler recommendation. A human must separately authorize the
four-row preflight. After it runs, only return-blind validity/timing receipts may
be reviewed. The remaining 60 rows require a further explicit decision; nothing
in this report authorizes them or the 1,152-row full arm.
