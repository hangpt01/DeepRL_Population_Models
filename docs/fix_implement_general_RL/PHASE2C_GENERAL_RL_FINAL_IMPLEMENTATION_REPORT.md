# Phase 2C — Final General-RL Implementation Report

Phase 2C implements the approved
`PHASE2B_GENERAL_RL_REMAINING_ISSUES_PLAN.md` for the four hidden general-RL
baselines in the isolated worktree
`/fs04/scratch2/ce25/general_rl_phase2_iso` at detached base `5f9cf32`.

No worktree was merged. The main dirty tree and running PLUS/MOOR jobs were not
touched. No scheduler job was launched, the prepared 1,152-row manifest was not
submitted or regenerated, and no performance/survival return or ranking was
read. All calibration and decisions below use only model-internal, return-blind
diagnostics.

## 1. Outcome

| Area | Phase 2C outcome |
|---|---|
| Delphic | Calibrated reference and G-D1..G-D4 implemented. Support-gated worlds pass the sink cell but fail healthy-cell G-D4 (`1.5166 < 3`). The reader-facing method is therefore honestly downgraded to **ensemble value-disagreement pessimism (Delphic-motivated)**; no validated-delphic-uncertainty or compatible-worlds claim remains. |
| OGSRL | Inert extinction risk replaced by the public bounded low-abundance shortfall. The train-only behavior budget is clipped to `[0.02, 0.10]`; the safety dual moves and the cost is non-degenerate in both full-budget cells. |
| RefPlan | A standardized train-only public logistic policy prior is a distinct fitted object from the model posterior. Candidate sequences use the prior with a registered `epsilon=0.10` uniform floor. |
| Privacy | One-private-field-at-a-time intervention invariance now checks byte-identical sanitized contexts, fitted numeric artifacts, and actions for all four methods. |
| Threads/resources | OpenMP/OpenBLAS/MKL are pinned to one thread and one allocated core. Default-vs-one-thread numeric bytes and actions are identical. Task-hours, actual CPU-hours, allocated core-hours, elapsed wall, and queue delay are registered separately. |

BA-MCTS's Phase 2 in-tree Eq.-4 belief update and the shared primary ensemble
size 5 remain unchanged. MOPO remains archived and excluded.

## 2. Delphic-motivated construction and gate

### Calibrated behavior channel

`fit_reference_behavior` now standardizes public train features and uses the
same registered optimizer as the world behavior heads: 400 iterations,
learning rate 0.3, and L2 `1e-2`. The returned reference object carries its
train-only mean and scale, preventing holdout refitting.

Full-4,000 held-out conditional NLL is internally consistent:

| Cell | Calibrated reference NLL | World NLL range | G-D1 |
|---|---:|---:|---|
| Amur tiger (healthy) | 2.18035 | 2.17537--2.18426 | pass |
| Egyptian vulture (sink) | 2.08027 | 2.07449--2.08302 | pass |

### G-D1..G-D4

- G-D1: each retained world's conditional behavior NLL is no more than
  `reference NLL + 0.25`.
- G-D2: maximum TV between predicted and empirical action distributions in
  public-observation quintiles is at most `0.20`.
- G-D3: observed-action cross-world variance is at most `0.05 * value_scale^2`.
- G-D4: counterfactual variance is positive and at least three times observed
  variance. At least three worlds must survive.

All worlds share one public-feature fitted-Q base. World-specific disagreement
is multiplied by posterior ambiguity and a squared normalized kNN
excess-distance support taper: it is zero inside the public support envelope
and reaches one only for clearly unsupported state-action pairs. The support
matrix is cached once per fixed feature set; this is an engineering
optimization with no threshold change.

| Cell | Region TV max | `V_obs` | `V_cf` | Ratio | Decision |
|---|---:|---:|---:|---:|---|
| Amur tiger | 0.16233 | 2.31891 | 3.51685 | **1.51660** | G-D4 fail |
| Egyptian vulture | 0.15411 | 92.41920 | 314.45256 | **3.40246** | G-D1..D4 pass |

Because the strengthened construction did not pass robustly in both healthy
and sink populations, the preregistered D-A fallback applies. Internal method
ID `delphic` is retained for manifest compatibility, but the registry and
reader-facing documentation label it **ensemble value-disagreement pessimism
(Delphic-motivated)**. The implementation does not call its variance validated
delphic uncertainty.

The new toy gates also pass: a world matching global `P(a)` while reversing
`P(a|public region)` fails G-D2; hand-built worlds identical on each logged
action and divergent only on a row-specific unobserved action pass G-D3/G-D4.

## 3. OGSRL public low-abundance constraint

The hidden safety cost is now

`c(o_next) = clip((s_low - o_next) / s_low, 0, 1)`,

where `s_low` is the 20th percentile of positive **training** observations.
The budget is the mean behavior-policy discounted episode cost on train,
clipped to `[0.02, 0.10]`. Deployment evaluates the same cost on public-model
predicted next observations. It never consumes private `K`, private safety
thresholds, family, or truth. This remains a public proxy and carries no
guarantee for the private safety objective.

| Cell | `s_low` | Cost prevalence | Cost mean | Budget | `lambda_safety` |
|---|---:|---:|---:|---:|---:|
| Amur tiger | 36.1525 | 0.18656 | 0.08255 | 0.10 | 1.99888 |
| Egyptian vulture | 19.3054 | 0.21438 | 0.06851 | 0.10 | 2.64152 |

Both dual values moved from initialization 1.0. The constructed mechanism test
also verifies the exact bounded costs `[1, 0.5, 0, 0]` at
`[0, 0.5*s_low, s_low, 2*s_low]` and verifies dual movement.

## 4. RefPlan public prior

Hidden RefPlan now fits a standardized multinomial-logistic public policy prior
on the training public belief features and actions (400 iterations, learning
rate 0.3, L2 `1e-2`). It stores `prior_weights` and `prior_scaler` separately
from the deployment dynamics `posterior`. Candidate first actions and
continuations are sampled from

`pi_tilde = 0.9 * pi_prior + 0.1 * Uniform`.

Every discrete first action is also retained as a coverage candidate. The toy
test fixes the belief and dynamics, swaps two hand-set priors, and confirms that
plan selection changes. This remains a linear public-policy and reflected-score
argmax approximation, not the paper's deep conservative prior or MPPI softmax.

## 5. Privacy verification

`test_private_value_intervention_invariance` reuses exactly one public dataset,
surrogate, belief cache, and seed. It changes one private field at a time:
`K_base`, `r_min`, `r_max`, `kind`, `safety_threshold`, `C_low`, `C_high`, both
regime thresholds, and population name. For each intervention it rebuilds the
sanitized `MethodContext`, then refits RefPlan, OGSRL, BA-MCTS, and the
Delphic-motivated method.

Every context is pickle-byte-identical, fitted numeric artifacts are
float-byte-identical, and actions are identical. The existing forbidden-name,
API-block, opaque-ID, deterministic-fit, relabel, and exact-equality checks are
retained as complementary gates.

## 6. Thread parity and resource accounting

Registered execution:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
allocated cores/task = 1
```

The local default-vs-one-thread canary compares fitted numeric vectors and
actions for every method. All numeric vectors are byte-identical (`max_abs=0`)
and all actions match. RefPlan, OGSRL, and BA-MCTS also have identical recursive
object hashes. The Delphic-motivated recursive object-graph hash differs because
that hash is alias/traversal-sensitive, while its extracted fitted numeric bytes
and action are identical; the registered parity decision is based on numeric
artifacts and actions.

Resource reports must keep these quantities separate:

- task-hours: sum of per-task wall duration;
- actual CPU-hours: sum of measured process CPU seconds / 3600;
- allocated core-hours: task wall duration times one allocated core / 3600;
- elapsed wall: schedule makespan at the realized concurrency;
- queue delay: reported separately, never folded into compute.

The retained-arm planning estimate remains conservative and is not a launch:

| Quantity | Registered estimate/status |
|---|---|
| Aggregate task-hours | 57.5 h (Phase 2 measured model; Phase 2C one-thread canaries were no slower) |
| Actual CPU-hours | not measured for the unlaunched arm; must be summed from process CPU time after execution |
| Allocated core-hours | approximately 57.5 h at one allocated core/task |
| Elapsed wall at 32 / 64 / 128 concurrent tasks | approximately 1.80 / 0.90 / 0.45 h, excluding queue delay |
| Queue delay | unknown and reported separately |

No scheduler run was performed in Phase 2C.

## 7. Verification

Targeted command:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest \
  tests/real/test_general_paper_mechanisms.py \
  tests/real/test_general_privacy.py -q
```

Result: **25 passed**.

Requested full command was run in the repository's documented NumPy-compatible
CPU-Torch environment:

```bash
cd /fs04/scratch2/ce25/general_rl_phase2_iso
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python \
  -m pytest tests/ -q
```

Result: **191 tests passed (100%)**. A first run in the general `pytorchrl`
environment produced 16 unrelated faithful-Torch failures because that Torch
binary was compiled for NumPy 1.x while the environment has NumPy 2.4.6; the
compatible repository environment (NumPy 2.2.6, Torch 2.13 CPU, pytest 9.1.1)
resolved those environmental failures without code changes.

Return-blind evidence:

- `real_ecology_runs/general_adequacy_phase2c/adequacy_Amur_tiger_ricker_sig0.2_n4000.json`
- `real_ecology_runs/general_adequacy_phase2c/adequacy_Egyptian_vulture_ricker_sig0.2_n4000.json`
- `real_ecology_runs/general_adequacy_phase2c/thread_parity.json`

## 8. Snapshot and decision

`real_ecology_runs/general_corrected_prepared/registration_general_corrected.json`
now registers the Phase 2C thresholds, honest Delphic-motivated label, RefPlan
prior, OGSRL cost/budget rule, one-thread resources, canary paths, and code
digests. The frozen `code/` copy and provenance hashes are refreshed to the
Phase 2C implementation. The manifest remains exactly 1,152 prepared rows and
`submitted:false`.

**Phase 2C implementation verdict: READY FOR REVIEW, NOT AUTHORIZED FOR
LAUNCH.** The plan is fully implemented, with its required honest Delphic
downgrade. Stop here for reviewer approval; do not submit the manifest, inspect
returns, merge the worktree, or launch an experiment.
