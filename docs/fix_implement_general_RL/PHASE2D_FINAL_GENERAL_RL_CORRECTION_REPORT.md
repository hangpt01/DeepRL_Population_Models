# Phase 2D — Final General-RL Correction Report

## 1. Outcome and scope

Phase 2D implements the controlling authorization
`CODE_SERVER_PHASE2D_FINAL_GENERAL_RL_CORRECTIONS.md` only in the isolated
worktree `/fs04/scratch2/ce25/general_rl_phase2_iso`.

The four registered hidden-method comparison is now:

1. RefPlan-inspired;
2. OGSRL-inspired;
3. BA-MCTS-inspired;
4. **Ensemble value-disagreement pessimism (Delphic-motivated)**, canonical ID
   `ensemble_value_disagreement_pessimism`.

The fourth method is an honest episode-bootstrap conservative fitted-Q
ensemble. It is not registered or described as a reproduction of Delphic's
latent causal construction. The old `delphic` ID remains only as a deprecated
read/CLI alias and is absent from the prepared manifest.

No scheduler job or experiment was launched. No performance, survival, or
ranking return was inspected. No threshold or coefficient was selected from
policy performance. The main dirty worktree, old frozen run snapshots, and
PLUS/MOOR code/jobs/artifacts were not touched. The corrected manifest remains
unsubmitted and the isolated worktree was not merged.

**Recommendation: GO for reviewer-authorized limited corrected canary only.**
This recommendation does not authorize submission. The full experiment remains
unauthorized.

## 2. Exact implementation changes

### Fourth method and public behavior model

- Added `src/real_ecology_benchmark/behavior_model.py` with the calibrated
  train-only multinomial public behavior model used by RefPlan and as a
  descriptive fourth-method diagnostic.
- Added
  `src/real_ecology_benchmark/methods/ensemble_value_disagreement.py` with
  `BootstrapQMember` and `EnsembleValueDisagreementPolicy`.
- Replaced `src/real_ecology_benchmark/methods/delphic.py` with a small
  deprecated import-path shim. It contains no direct Q construction or old
  policy class.
- Removed the operative `src/real_ecology_benchmark/delphic_compat.py` gate
  module.
- Updated `methods/__init__.py` with the canonical ID, exact reader label, and
  explicitly deprecated historical alias.
- Updated `training_monitor.py` to use the neutral feature augmenter and
  `refplan.py` to use the neutral public behavior module.

### OGSRL

- Added `normalized_discounted_occupancy` in `methods/ogsrl.py`.
- Changed `_fit_public_safety_scale` to estimate the unclipped complete-episode
  normalized behavior budget and its uncertainty.
- Added `_normalized_cost_returns`; `_rollout_training_metrics` uses it for the
  actor safety objective and dual statistic.
- Changed `_public_rollouts` and hidden actor training to the common registered
  cost horizon.
- Replaced `_public_belief_action_risks`' one-step comparison with a normalized
  receding-horizon public-model occupancy calculation.
- Replaced `PlannerConfig.ogsrl_rollout_horizon=6` plus budget floor/cap with
  `PlannerConfig.ogsrl_cost_horizon=25`. The retired horizon is no longer a
  tuning axis in `make_general_audit_manifest.py`.

### RefPlan

- Added `PublicParticlePlanner._proposal_features` and
  `_history_conditioned_prior_sequences` in `public_models.py`.
- Changed `plan_marginalized` to generate proposals sequentially from updated
  posterior-predictive public states, while retaining posterior-marginalized
  scoring and explicit first-action coverage.

### Privacy and tests

- Replaced the bounded fitted scalar digest in
  `tests/real/test_general_privacy.py` with complete deterministic traversal of
  all fitted numeric artifacts.
- Extended the intervention lifecycle to `fit -> act -> observe -> act`, and
  compares fitted artifacts, posterior, diagnostics, and both actions.
- Added lower-level provider blocks/sentinels for private tables, action
  resolution, native solver access, the public-action-channel provider, and
  `realdata.DATA_DIR`; also varied the high-level data-directory field at the
  context boundary.
- Replaced the old fourth-method gate tests with bootstrap membership,
  deterministic-fit, no-output-perturbation, and exact empirical-variance
  tests.
- Added normalized-cost invariants, exact examples, budget-equality, dual
  movement, action-risk response, and state-dependent later-action RefPlan
  tests.
- Updated synthetic smoke tests to the canonical fourth-method ID and neutral
  class.

### Operative scripts, documentation, and registration

- Updated `general_adequacy_probe.py`, `general_thread_parity.py`,
  `make_general_corrected_manifest.py`, `run_real_manifest_row.py`, and current
  generic manifest/report scripts to the canonical ID and `H_cost` field.
- Updated `README.md`,
  `docs/benchmark/04_algorithm_adaptations_and_claims.tex`, the hidden-parameter
  audit, protocol method list, and docs checker. Old search-budget audit files
  are marked superseded where their old OGSRL horizon language remains as
  history.
- Regenerated only the authorized unsubmitted
  `real_ecology_runs/general_corrected_prepared/manifest_general_hidden.csv`.
- Refreshed only `real_ecology_runs/general_corrected_prepared/code/`, its hash
  receipt, and `registration_general_corrected.json`. Older frozen run
  snapshots were not rewritten.
- Wrote new return-blind diagnostics under
  `real_ecology_runs/general_adequacy_phase2d/`; Phase 2C diagnostics remain
  unchanged history.

## 3. Fourth-method mathematical definition and naming

Let the complete training episodes be `E_1,...,E_n`. For member `m`, draw `n`
episode indices with replacement using the deterministic seed
`seed + 1009*(m+1)`, concatenate their complete transitions, and fit a separate
linear conservative Q estimator. No member sees another member's bootstrap.

At fitted-Q iteration `k`, member `m` uses

```text
y_m,t = r_t + gamma * (1-done_t) * max_a Q_m,k(o_{t+1},a).
```

For each action, the ridge solve includes the logged-action regression rows and
the registered CQL-style zero pseudo-target rows scaled by
`sqrt(cql_alpha)`, with `cql_alpha=0.5`. All members use the same public
features, public rewards, action set, `gamma=0.95`, 35 fitted-Q iterations, and
ridge rule. Their only differences are the registered episode bootstrap and
seed used to draw it. No random quantity is added to Q.

For fitted member values `Q_m(o,a)` and `M=20`:

```text
Q_bar(o,a) = (1/M) * sum_m Q_m(o,a)
V_Q(o,a)   = (1/M) * sum_m (Q_m(o,a)-Q_bar(o,a))^2
score(o,a) = Q_bar(o,a) - lambda_V * V_Q(o,a).
```

The registered `lambda_V=0.1` is retained without return tuning. Since `V_Q`
has squared Q units, `lambda_V` is explicitly registered in inverse-Q units;
the subtracted term is therefore on the Q scale.

The calibrated public behavior model is descriptive only. There is no support
or ambiguity multiplier, random Q perturbation, survivor filter, variance-ratio
target, or pass/fail uncertainty gate. Observed-action disagreement is reported
as potentially containing bootstrap estimation error and model
misspecification. Delphic motivated the idea of pessimism from cross-model
value disagreement; the implemented baseline does not reproduce Delphic's
latent causal/generative construction.

The two full-budget return-blind probes found nonzero fitted disagreement:

| Cell | Mean logged-action Q variance | Mean other-action Q variance | Mean bootstrap unique-episode fraction |
|---|---:|---:|---:|
| Amur tiger | 0.00774702 | 0.00711531 | 0.61875 |
| Egyptian vulture | 0.04952972 | 0.04108662 | 0.61875 |

These values have no pass/fail target and were not used to tune the method.

## 4. OGSRL normalized safety functional and evidence

The approved public proxy remains

```text
s_low = Q_0.20(positive training observations)
c(o') = clip((s_low-o')/s_low, 0, 1).
```

It is a public low-abundance proxy and gives no guarantee for the private safety
objective. The one registered functional is

```text
C_H(c_1:H) = ((1-gamma)/(1-gamma^H))
             * sum_{t=0}^{H-1} gamma^t c_{t+1},
gamma = 0.95, H_cost = 25.
```

The helper normalizes discount weights by their sum, which is algebraically
identical and stable as `gamma -> 1`. It returns 0 for all-zero costs, 1 for
all-one costs for every positive horizon, remains bounded, and matches exact
hand calculations.

The scientific budget is exactly the arithmetic mean of `C_25` over complete
train-only behavior episodes. There is no floor or cap. The actor/dual uses
normalized 25-step safety returns. Deployment forces each candidate first
action, propagates the public dynamics ensemble's predictive mean, follows the
learned public actor thereafter, and compares its normalized occupancy over
`min(25, 50-t)` positive remaining steps with the same budget.

Return-blind full-budget evidence (128 training episodes per cell):

| Cell | `s_low` | Budget `mean(C_25)` | SD | SE | Descriptive 95% interval | Final safety dual |
|---|---:|---:|---:|---:|---:|---:|
| Amur tiger | 36.15252445 | 0.0981646466 | 0.1672099600 | 0.0147794121 | [0.06919700, 0.12713229] | 0.93898003 |
| Egyptian vulture | 19.30537004 | 0.0545761645 | 0.0917769957 | 0.0081120170 | [0.03867661, 0.07047572] | 1.35700241 |

The Egyptian-vulture dual moves upward, consistent with an active modeled
constraint. The Amur dual moves downward from 1.0, indicating that this fitted
modeled constraint is comparatively slack rather than universally binding.
Both respond to the corrected cost, and the constructed action test proves that
changing only normalized low-abundance risk changes the feasible action. The
Amur direction is reported as observed; the budget was not modified to force
binding.

## 5. RefPlan sequential history-conditioned prior

For each proposal candidate, RefPlan now:

1. starts from the current public belief mean and public feature/history vector;
2. evaluates the calibrated prior and mixes it with epsilon 0.10 uniform;
3. samples the current-depth action, preserving explicit first-action coverage;
4. obtains every public ensemble member's deterministic next prediction and
   propagates their model-posterior-weighted predictive mean;
5. updates public proposal features, including previous/current observation and
   timestep;
6. re-evaluates the prior on that updated simulated public history;
7. repeats through the planning horizon; and
8. scores the resulting fixed sequence under every ensemble member with the
   existing common-random-number starts and posterior-marginalized reflected
   score.

The prior is proposal-only; no log-prior term was added. The fitted prior and
model posterior remain separate objects, and `observe` changes only the model
posterior. The new state-dependent test holds the root prior identical, verifies
that predictive histories diverge at the next depth, verifies one prior call per
depth, and observes different later proposed actions.

## 6. Privacy lifecycle design and evidence

The intervention test changes one private field at a time at the high-level
environment boundary and requires byte-identical sanitized contexts. During
the complete method lifecycle it blocks the known table/action/native providers,
the action-channel provider, and the data-directory global with sentinels. Each
method then runs:

```text
fit -> reset -> act -> observe(public transition) -> public-filter update -> act.
```

The comparison includes:

- a complete deterministic traversal/hash of every fitted numeric artifact;
- model posterior state after `observe` where applicable;
- final public diagnostics;
- both selected actions.

The complete artifact hash has no recursion-depth or array-length truncation.
Its documented exclusions are public input/runtime metadata rather than fitted
state: `method_context`, `model_cfg`, `planner_cfg`, runtime `rng`,
`training_history`, externally attached train/holdout dataset/cache views,
`last_diagnostics`, and `fit_diagnostics`. Posterior and diagnostics are hashed
separately at the lifecycle comparison point. The context test separately
varies `environment.data_dir` while holding the public action-channel schema
fixed.

All four methods pass every private-field intervention. The existing forbidden
API, no-private-artifact, deterministic fit, and structural relabel tests also
remain. This is verification hardening; no current leak was found.

## 7. Verification results

Targeted command:

```text
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest \
  tests/real/test_general_paper_mechanisms.py \
  tests/real/test_general_privacy.py \
  tests/synthetic/test_methods.py -q -p no:cacheprovider
```

Result: **30 passed**.

Full requested suite, using the repository's NumPy-compatible CPU-Torch
environment:

```text
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python \
  -m pytest tests/ -q -p no:cacheprovider
```

Result: **192 passed (100%)**.

Additional gates:

- `scripts/check_docs.py`: **OK**;
- forbidden current fourth-method claims/direct-perturbation search: **no hits**;
- live source versus prepared source snapshot: **byte-identical**;
- manifest/registration validator: **OK**, 1,152 rows, 288 per method,
  `submitted:false`;
- snapshot hash verification: **matched**;
- default-thread versus one-thread parity: recursive hashes, numeric bytes, and
  actions identical for all four methods; **passes:true**.

No policy return was part of any verification gate.

## 8. Runtime changes from Phase 2C

The full-budget return-blind probes use one thread and the same two frozen
public-data cells. Times are diagnostic wall times, not performance returns:

| Method/quantity | Phase 2C | Phase 2D | Change |
|---|---:|---:|---:|
| RefPlan mean act, Amur | 30.64 ms | 38.82 ms | 1.27x from sequential prior proposals |
| OGSRL fit, Amur | 4.58 s | 18.43 s | 4.03x from H=25 actor rollouts |
| OGSRL mean act, Amur | 55.66 ms | 412.75 ms | 7.42x from H=25 receding risk |
| Fourth-method fit, Amur | 13.33 s | 1.32 s | 0.10x with linear bootstrap Q members |
| Fourth-method mean act, Amur | 1.45 ms | 0.16 ms | 0.11x |
| Whole probe, Amur | 34.97 s | 46.95 s | 1.34x |
| Whole probe, vulture | 34.92 s | 47.52 s | 1.36x |

If the previous 57.5 task-hour arm estimate scaled like these aggregate probes,
the rough Phase 2D planning estimate would be approximately 77--78 task/core
hours at one allocated core per task. That is an extrapolation, not a measured
experiment budget. Actual CPU-hours, allocated core-hours, elapsed wall, and
queue delay must still be recorded separately. No scheduler run was made.

## 9. Refreshed registration, manifest, snapshot, and hashes

The new unsubmitted prepared manifest contains exactly 1,152 rows:

| Canonical ID | Rows |
|---|---:|
| `refplan` | 288 |
| `ogsrl` | 288 |
| `bamcts` | 288 |
| `ensemble_value_disagreement_pessimism` | 288 |

Key hashes:

| Artifact | SHA-256 |
|---|---|
| Prepared manifest | `e71f67daf7cd048577418f6e074e5ea4997fd0a23ecac61fc1419e9e09ab5a66` |
| Prepared code tree (60 included files) | `2df0b7232d885587b14849259a09d4d056c1bce9d69ec3343753eab9de2b29a0` |
| Registration JSON | `2b40bc11db0a930afd623d25ccd27bdf9c800dc715a9d2bf5df1a06d839ae5ee` |
| Amur return-blind diagnostic | `15a4a8ab424d7ad770277b0b4f4fab78490fc6bc3789641497be05010f1d9d4e` |
| Vulture return-blind diagnostic | `aa853cb1a60e21ca086b39208ac0a001ca99842ba2ac26ef6f5619c4e88cb831` |
| Thread parity diagnostic | `9ee7f6693ccdf19d119226d8c3a43b9913d203568a8730eb69c7f87c24c7d0e9` |

The registration's per-source digests were independently rechecked against the
live files. The prepared source snapshot was independently compared with the
live `src/` tree and selected scripts/tests. The manifest is explicitly
`submitted:false`. No old frozen manifest or result snapshot was changed.

## 10. Remaining deviations and allowed claims

| Method | Allowed current claim | Remaining paper deviation |
|---|---|---|
| Ensemble value-disagreement pessimism | Episode-bootstrap conservative Q ensemble using empirical fitted-value variance pessimistically; Delphic-motivated at idea level | No latent causal/generative construction, observable-likelihood fit, or paper uncertainty semantics |
| OGSRL-inspired | Guarded constrained public-model actor using normalized public low-abundance occupancy and OOD proxy | Linear actor/model/guardian; public proxy does not guarantee private safety; Amur modeled constraint was comparatively slack |
| RefPlan-inspired | History-conditioned public-prior proposals with separate model posterior and posterior-marginalized reflected scoring | Linear prior/model and fixed candidate sequences after proposal; no deep latent encoder or MPPI softmax |
| BA-MCTS-inspired | Public ensemble-model belief updated in tree with registered depth/simulation budget | Lightweight public dynamics ensemble and benchmark-specific tree approximation |

Thread pinning, matched 4,000-transition data, common public ensemble size five,
BA-MCTS 256/depth 8, MOPO exclusion, return-blind calibration, and separate
resource accounting remain unchanged.

**Final Phase 2D recommendation: GO for a limited corrected canary after human
review; NOT AUTHORIZED TO SUBMIT.** Stop here. Do not launch, merge, inspect
returns, or modify PLUS/MOOR.
