# Phase 2D2 general-RL mechanism and limited-canary audit

## 1. Scope, method, and verdict

This is the requested return-blind audit of the isolated Phase 2D worktree. It
does not authorize a submission. No runtime code, prepared manifest, snapshot,
experiment output, or scheduler state was changed; no job was launched; no
policy, survival, or ranking return was read; and PLUS/MOOR was not touched.

The value-disagreement diagnostics below refit the two frozen Phase 2D probe
cells in memory with the registered settings: 4,000 transitions, 25-step
episodes, collection seed 116, episode split seed 117, 20 complete-episode
bootstrap Q members, member seeds `40116 + 1009*(m+1)`, `gamma=0.95`, and
`lambda_V=0.1`. The healthy/recoverable cell is Amur tiger, Ricker,
`sigma_obs=0.2`; the sink cell is Egyptian vulture with the same family and
noise. The fit used 3,200 train and 800 holdout public states. Only fitted Q,
variance, action, model-cost, source, and resource information was examined.

**Current verdict: NO-GO.** The registered value penalty is weak but demonstrably
active, so it is not an ensemble-mean-only implementation. RefPlan remains
acceptable. The blocking defect is OGSRL deployment: it applies the convex
shortfall cost after taking ensemble predictive means and therefore does not
estimate the same uncertainty-averaged cost used in actor training. In
addition, no complete `run_real_manifest_row.py` fit/evaluate row has yet been
timed. The OGSRL correction and a separately authorized four-row timing
preflight must precede any decision on the remaining 60 limited-canary rows.

## 2. F1 — operational effect of value-disagreement pessimism

### 2.1 Registered computation and units

`methods/ensemble_value_disagreement.py:108-110` computes the population
variance across the 20 independently fitted Q members, and lines 186-188 select

```text
argmax_a [mean_m Q_m(o,a) - 0.1 * Var_m(Q_m(o,a))].
```

The logged public reward is a dimensionless utility: normalized abundance
benefit `s/(s+K_ref)`, action cost, and any registered collapse penalty are in
the same reward unit. There is no additional standardization of
`dataset.rewards` before fitted-Q iteration. Thus Q is in discounted public
reward/utility units, variance is in squared Q units, `lambda_V=0.1` has
inverse-Q units, and `lambda_V*variance` is again in Q units. The numerical
coefficient was not changed in this audit.

### 2.2 Frozen-state distributions

Each entry below is `mean (SD); [min, p05, p25, p50, p75, p95, max]`. Q-related
distributions use all state-action pairs; the action-gap distribution uses one
value per state.

| Cell/split | `mean_Q` | `sqrt(var_Q)` |
|---|---|---|
| Amur train | -0.16549 (0.44219); [-2.85792, -1.11656, -0.29483, -0.03478, 0.09846, 0.30250, 1.01231] | 0.06856 (0.04972); [0.00568, 0.01908, 0.03560, 0.05398, 0.08354, 0.17631, 0.86229] |
| Amur holdout | -0.12889 (0.41580); [-2.34763, -1.07110, -0.21166, -0.01107, 0.10362, 0.30324, 0.91957] | 0.06579 (0.04863); [0.00670, 0.01824, 0.03396, 0.05139, 0.07995, 0.17103, 0.49676] |
| Vulture train | -1.45969 (0.90806); [-3.58851, -2.65089, -2.23652, -1.74373, -0.56244, -0.08458, 2.04294] | 0.19134 (0.07241); [0.04832, 0.10063, 0.14170, 0.17927, 0.22538, 0.32424, 0.80266] |
| Vulture holdout | -1.41228 (0.96575); [-3.63147, -2.69835, -2.26485, -1.67916, -0.45293, 0.00283, 2.20847] | 0.19399 (0.07741); [0.04737, 0.09812, 0.13942, 0.18006, 0.23175, 0.34133, 0.66512] |

| Cell/split | `var_Q` | `0.1*var_Q` | Best-minus-second `mean_Q` gap |
|---|---|---|---|
| Amur train | 0.007173 (0.013487); [0.000032, 0.000364, 0.001267, 0.002914, 0.006978, 0.031084, 0.743551] | 0.000717 (0.001349); [0.000003, 0.000036, 0.000127, 0.000291, 0.000698, 0.003108, 0.074355] | 0.074675 (0.072613); [0.000016, 0.003884, 0.018945, 0.049426, 0.106652, 0.224834, 0.430510] |
| Amur holdout | 0.006694 (0.011927); [0.000045, 0.000333, 0.001154, 0.002641, 0.006392, 0.029251, 0.246768] | 0.000669 (0.001193); [0.000004, 0.000033, 0.000115, 0.000264, 0.000639, 0.002925, 0.024677] | 0.076323 (0.070151); [0.000071, 0.002899, 0.019834, 0.051541, 0.119014, 0.224370, 0.418048] |
| Vulture train | 0.041854 (0.037075); [0.002335, 0.010125, 0.020078, 0.032137, 0.050795, 0.105131, 0.644264] | 0.004185 (0.003708); [0.000233, 0.001013, 0.002008, 0.003214, 0.005080, 0.010513, 0.064426] | 0.187463 (0.177556); [0.000009, 0.011642, 0.064450, 0.137390, 0.253497, 0.532166, 1.732133] |
| Vulture holdout | 0.043625 (0.039528); [0.002244, 0.009628, 0.019438, 0.032421, 0.053708, 0.116507, 0.442380] | 0.004363 (0.003953); [0.000224, 0.000963, 0.001944, 0.003243, 0.005371, 0.011651, 0.044238] | 0.207177 (0.204176); [0.000018, 0.011494, 0.064967, 0.146878, 0.277084, 0.597646, 1.596885] |

The median registered penalty is 0.00026--0.00324 Q units, while the median
best/second mean-Q gap is 0.0495--0.1469. The concern in the review is therefore
well founded: for most states the penalty is much smaller than the decisive
action gap.

### 2.3 Action sensitivity, without returns

Fractions are relative to `argmax mean_Q` at `lambda_V=0`.

| Cell/split | `lambda=0.1` | `lambda=0.3` | `lambda=1.0` | Any change over `{0.1,0.3,1.0}` |
|---|---:|---:|---:|---:|
| Amur train (3,200) | 0.0015625 (5 states) | 0.0034375 (11) | 0.0109375 (35) | 0.0109375 |
| Amur holdout (800) | 0.0012500 (1) | 0.0025000 (2) | 0.0125000 (10) | 0.0125000 |
| Vulture train (3,200) | 0.0031250 (10) | 0.0065625 (21) | 0.0181250 (58) | 0.0181250 |
| Vulture holdout (800) | 0.0037500 (3) | 0.0112500 (9) | 0.0175000 (14) | 0.0175000 |

The registered penalty changes at least one train and holdout decision in both
populations. It is therefore operationally active, not identically inert, but
its influence is sparse (0.125--0.375% at the registered coefficient). This is
a mechanism result, not evidence of performance. No coefficient should be
changed from these percentages. Under the existing honest name, retaining
`lambda_V=0.1` is scientifically defensible provided this weak activity is
reported rather than described as a strong conservative effect.

A constructed, unit-level example also establishes the selection mechanism:

```text
mean_Q = [1.00, 0.99], variance_Q = [0.20, 0.00]
lambda=0:   scores = [1.00, 0.99] -> action 0
lambda=0.1: scores = [0.98, 0.99] -> action 1
```

This test should be committed as a non-return mechanism regression test in a
future authorized correction, but it was not added during this audit.

## 3. F2 — OGSRL uncertainty-treatment trace

The registered bounded step cost and normalized occupancy are

```text
s_low = Q_0.20({positive train observations})
c(o') = clip((s_low-o')/s_low, 0, 1)
C_H = sum(t=0..H-1) [gamma^t / sum(j=0..H-1) gamma^j] c(o_(t+1)),
gamma=0.95, registered H=25.
```

This is algebraically the requested `(1-gamma)/(1-gamma^H)` form. The trace is:

| Stage and source | Next value and draws | Placement of cost and averaging | Actual estimand |
|---|---|---|---|
| Behavior budget, `ogsrl.py:416-455` | Observed `dataset.next_observations`; no new model, process, or observation draws. `s_low` and all costs use train data only. Dataset generation was seeded at 116, but the estimator itself has no RNG. | `c` is applied to every observed next observation at line 423; `C_25` is formed per complete 25-step episode at 426-435; 128 train-episode values are averaged at 437. No clipping/floor/cap is applied to the mean. | Empirical behavior-policy `mean_episode C_25` on the `[0,1]` scale. |
| Actor rollout starts, `ogsrl.py:404-409, 488-503` | Each iteration draws 128 starts from cached public posterior log-moments, then samples actor actions. One policy RNG, initialized from the registered method seed, drives row, start, action, member, and residual draws. | Starting uncertainty is sampled before rollout. | A Monte Carlo public-belief start distribution. |
| Actor transitions, `ogsrl.py:457-486`; `public_models.py:40-50,107-118` | At each of 25 steps and for each path, one of five bootstrap dynamics members is sampled uniformly, then one Gaussian fitted residual is sampled in log observation space. There is no separately identified future process-noise draw and observation-emission draw: the observation-space residual is an aggregate predictive innovation. | `c(following)` is evaluated inside every sampled path at line 481, before path averaging. `normalized_discounted_occupancy` is applied pathwise at lines 342-351; the batch is averaged at 361-363 and for the dual at 521-525. | For the 30 training iterations: 30 x 128 x 25 = 96,000 sampled transitions for the actor objective, plus the same number in RNG-restored holdout diagnostics. This approximates `E_start,member,residual,actor[C_25]`. |
| Deployment, `ogsrl.py:572-611` | No member, residual/process, or observation draws. Each of five member means is evaluated and immediately averaged by `dynamics.predict` (lines 101-105 of `public_models.py`). The current public belief particles are propagated deterministically. | At depth 0, `c` is applied after ensemble-member averaging (587-588). Later, it is applied to each candidate action's ensemble mean (591-598), then averaged over actor action probabilities; the next trajectory state is also action-probability averaged (599). Finally `C_H` is formed per belief particle and belief-weight averaged (602-605). | One predictive-mean trajectory per current belief particle and candidate first action: generally `C_H(c(E_member[o']))`, not `E_member,residual[C_H]`. `H=min(25,50-t)`, so it is not even the actor's `C_25` late in an episode. |

Consequently, current OGSRL **does not average low-abundance cost across
model/process predictive uncertainty at deployment**. The convex shortfall can
make cost of the predictive mean lower than expected cost. Actor training and
deployment share the bounded cost formula and normalized units, but they do not
estimate the same expected object. The observation-space dynamics also cannot
separately identify process and measurement noise; its member bootstrap plus
fitted residual is only a joint public predictive approximation.

### 3.1 Exact correction plan (not implemented)

The minimal correction is confined to
`src/real_ecology_benchmark/methods/ogsrl.py`, with a small registered planner
field and tests:

1. Factor one shared public predictive-cost rollout primitive used by both
   `_public_rollouts` and `_public_belief_action_risks`. It must call
   `PublicDynamicsEnsemble.sample_next`, evaluate
   `_public_low_abundance_cost(following)` on every sampled next value, form
   pathwise `C_25`, and average only afterward.
2. Register `ogsrl_deployment_rollouts=256`. For each candidate first action,
   use 256 weighted/stratified current-belief starts, a balanced schedule over
   the five model members, Gaussian fitted-residual draws, and categorical
   continuation-action draws. Use common random numbers across candidate first
   actions. Derive a local deterministic stream from the immutable method seed,
   belief timestep, rollout index, and depth; do not consume the actor's mutable
   deployment RNG.
3. Use exactly 25 model steps at deployment so the behavior budget, actor dual,
   and feasibility mask all compare the same normalized `C_25`. This removes
   the present late-episode `min(25,50-t)` mismatch. If reviewers instead want a
   remaining-horizon estimand, it requires separately registered
   timestep-specific behavior budgets; it must not silently compare `C_H` with
   a `C_25` budget.
4. Describe the resulting expectation precisely as
   `E_belief,bootstrap-member,public-predictive-residual,actor[C_25]`. A claim of
   separately resolved `E_process,observation` would require a larger latent
   transition plus observation-emission model and is not supported by the
   current observation-space ensemble.
5. Add deterministic-repeat, common-random-number, Jensen, pathwise-cost, exact
   normalization, and actor/deployment-shared-helper tests. The Jensen test
   should show `mean(c(draw)) >= c(mean(draw))` up to numerical tolerance and a
   strict gap in a constructed distribution crossing `s_low`.

No safety budget or coefficient is to be tuned from policy returns. The
behavior budget remains the existing unclipped train-only mean of observed
`C_25` values.

### 3.2 Expected runtime impact

The current deployment performs, per decision, approximately
`11 * (1 + 24*11) * 5 * P = 14,575*P` deterministic member-prediction rows,
where `P` is the number of belief particles. The proposed Monte Carlo performs
`11*25*256 = 70,400` sampled transition rows, independent of `P` after the 256
starts are selected. With the registered evaluation filter `P=256`, this is
70,400 sampled rows versus about 3.73 million simple deterministic member rows,
although RNG and grouping overhead differ. The likely action-time effect is
therefore comparable to or lower than the current measured probe mean of
412.75 ms, not a justified several-fold multiplier. That is an operation-count
estimate only. Exact wall/CPU/RSS impact must be measured through the complete
runner preflight in Section 6 after implementation; no launch budget is inferred
from the Phase 2D 77--78 task-hour extrapolation.

## 4. RefPlan confirmation

RefPlan requires no new correction from this review. The public behavior prior
is fitted as `policy.policy_prior` while the five-member dynamics belief is a
separate numeric `policy.posterior` (`methods/refplan.py:44-59`). Proposal
generation calls `policy_prior.probabilities(features)` inside every planning
depth (`public_models.py:199-217`), after advancing the posterior-weighted
simulated public state. The existing state-dependent test verifies one prior
call per horizon step and divergence at later proposal states. The model
posterior is used separately for predictive propagation and scoring.

## 5. Proposed limited canary — plan only

No manifest was generated. The proposed diagnostic has exactly **16 public
dataset cells and 64 method rows**:

| Dimension | Registered values |
|---|---|
| Population role | Amur tiger (recoverable) and Egyptian vulture (demographic sink) |
| Environment family | `ricker`, `allee`, `theta`, `regime` |
| Observation noise | `sigma_obs=0.0` (low) and `0.4` (high, nonzero hidden information) |
| Reward mode | `safe` only |
| Methods | `refplan`, `ogsrl`, `bamcts`, `ensemble_value_disagreement_pessimism` |
| Shared data | collection seed 116, 4,000 transitions, complete 25-step episodes, learned public filter, hidden r/K, identical dataset path and SHA-256 across the four methods in each scenario |
| Evaluation | seed 9001, one 50-step episode per row, matched across methods; generated performance fields quarantined and unread during validity review |

The Cartesian product is `2*4*2=16` scenarios and `16*4=64` rows. It is 5.6%
of the 1,152-row full hidden arm and is method-balanced. It is a mechanism and
pipeline canary, not a comparative performance sample.

## 6. Preregistered return-blind acceptance and resource plan

### 6.1 Acceptance table

| Gate | Fixed return-blind acceptance criterion | Failure action |
|---|---|---|
| Snapshot/data identity | All 64 rows use one new frozen code hash; each scenario's four rows have identical public dataset hash, seed, row count 4,000, episode structure, action channels, and filter configuration. No private artifact enters a method. | Stop; no retry as a scientific result. |
| General execution | Every fit/evaluation completes, all fitted artifacts/actions/diagnostics are finite, thread variables equal 1, and runner resource receipts exist. | Stop affected tranche. |
| Data adequacy | Every dataset contains 160 complete 25-step episodes and every one of 11 actions has at least 50 logged transitions. | Stop cell; do not alter behavior data or threshold. |
| EVD mechanism | Twenty distinct registered member seeds; complete-episode bootstrap membership differs; finite nonzero fitted-Q variance on train and holdout; exact empirical-variance identity; constructed two-action test changes action. Frozen Amur and vulture checks at `lambda=0.1` must continue to change at least one train and one holdout state, with no minimum percentage beyond nonzero. | Stop; do not tune `lambda_V`. |
| OGSRL definition | `gamma=.95`, `H=25`, bounded pathwise cost, behavior budget exactly the unclipped train-episode mean on `[0,1]`; actor and deployment call the shared uncertainty-averaged helper; deterministic CRN repeat is byte-identical; Jensen constructed test passes. | Stop; no budget/dual tuning. |
| OGSRL mechanism | Safety and OOD channels each have finite cross-action spread above `1e-9`; both duals are finite and at least one differs from initialization by more than `1e-6` in every cell. Direction/binding is recorded, not forced. | Stop method/cell and review; no performance-based change. |
| RefPlan | Prior and posterior remain distinct fitted objects; prior called exactly once at every proposal depth; constructed state-dependent continuation changes later proposal probabilities; posterior is normalized and moves by more than `1e-9` on at least one public transition per cell. | Stop method/cell. |
| BA-MCTS | 256 simulations, depth 8, five public model members; finite nontrivial tree; constructed member-consistent observation produces root-to-child model-belief KL greater than `1e-9`. | Stop method/cell. |
| Privacy/thread parity | Existing `fit -> act -> observe -> act` intervention suite and one-thread parity pass on the frozen snapshot. | Stop all rows. |
| Quarantine | Validity tooling is technically unable to read return/ranking/survival fields; performance files remain under a separate access-restricted quarantine root until a signed validity decision. | Stop review immediately on breach. |

Validity review may inspect only dataset identity/coverage, fit diagnostics,
mechanism invariants, actions needed by the invariant checks, exceptions,
hashes, timing, CPU, RSS, and scheduler metadata. Comparative analysis may
inspect rewards/returns, survival, ranks, or method comparisons only after all
validity gates pass and separate human authorization is recorded.

### 6.2 Complete-row timing preflight and exact scheduler request

There is presently **no measured complete runner-row timing** for these methods.
The Phase 2D probe times cover fit plus synthetic act calls and cannot satisfy
the review's requirement. After the OGSRL correction is approved, tested, and
snapshotted, the first authorized launch must therefore be only four complete
`run_real_manifest_row.py` rows: one row per method on Egyptian vulture,
`regime`, `sigma_obs=0.4`, safe reward, using the same 4,000-row dataset and the
one-episode evaluation above. Performance outputs remain quarantined.

Exact preflight request:

```text
scheduler: Slurm partition comp
array: 0-3%1 (serial, one method per task)
cpus-per-task: 1
mem: 8G
time: 02:00:00 per task
OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1
```

Record runner `manifest_row_seconds`, process CPU seconds, Slurm elapsed,
`MaxRSS`, allocated core-hours, and queue delay separately. `/usr/bin/time -v`
or equivalent may wrap the frozen runner, but the validity reader must redact
return fields. Acceptance requires every row to finish within 60 minutes and
peak below 6 GiB, leaving at least a twofold wall margin and 2 GiB memory
margin. These are resource stop rules, not algorithm thresholds.

Only after a human reviews the four-row validity/timing receipt may the
remaining 60 rows be considered. Their exact fixed request is Slurm `comp`,
`--array=<remaining-indices>%4`, `--cpus-per-task=1`, `--mem=8G`, and
`--time=02:00:00`, with the same thread pins. The remainder must not be
auto-submitted: if later authorized, it uses `afterok:<four-row-job-id>` plus a
separate recorded human GO. Concurrency is capped at four and launch must be
deferred if it would contend with PLUS/MOOR.

Stop the tranche on any failed mechanism/hash/privacy/resource gate, missing
dependency, nonzero exit, NaN, or quarantine breach. A row may be retried once
only for documented scheduler preemption or node/filesystem failure, using the
identical index, seed, data hash, and code snapshot. Algorithmic, mechanism,
timeout, OOM, or deterministic-reproduction failure gets no automatic retry.

## 7. Snapshot and manifest procedure after an approved correction

1. Implement only the reviewed OGSRL shared pathwise-cost Monte Carlo change
   and its tests in the isolated worktree; add the constructed EVD mechanism
   test without changing `lambda_V`.
2. Run the targeted mechanism/privacy/thread tests and the full one-thread test
   command. Run documentation and forbidden-claim checks. Read no returns.
3. Create a **new** Phase 2D2 canary snapshot and hash receipt. Do not overwrite
   historical snapshots and do not alter the existing 1,152-row prepared
   manifest.
4. Only after explicit authorization, generate a separate 64-row canary
   manifest from the exact Cartesian product in Section 5. Validate counts (16
   per method), uniqueness, hidden routing, 4,000-transition registration,
   seeds, and dataset-key equality; mark it `submitted:false`.
5. Freeze the manifest and code hashes before the timing preflight. Any source,
   config, test, or manifest change invalidates the receipt and returns to step
   2. Submission requires a later explicit authorization not contained here.

## 8. Final classification

| Method | Current classification | Audit disposition |
|---|---|---|
| Ensemble value-disagreement pessimism (Delphic-motivated) | Honest episode-bootstrap conservative fitted-Q ensemble with a weak but operational empirical-variance penalty; not Delphic uncertainty or compatible worlds. | Keep implementation and `lambda_V=0.1`; add the constructed regression test and disclose sparse action influence. |
| OGSRL-inspired | Guarded constrained observation-space model actor with a correctly normalized empirical behavior budget, but deployment evaluates predictive-mean cost rather than expected pathwise cost. | Blocking correction required before any runner canary. |
| RefPlan-inspired | History-conditioned public behavior-prior proposals at every depth, separate from the dynamics posterior. | Keep unchanged. |
| BA-MCTS-inspired | Public dynamics-ensemble tree search with in-tree member-belief updates at 256 simulations/depth 8. | Keep unchanged. |

**NO-GO for launch now.** A future recommendation can become
`GO-CONDITIONAL` for the remaining limited canary only after (a) the OGSRL
uncertainty-averaging correction and tests pass on a new frozen snapshot and
(b) the separately authorized four-row complete-runner preflight passes the
return-blind mechanism and resource gates. Neither condition nor this report
authorizes a job submission, and the 1,152-row arm remains out of scope.
