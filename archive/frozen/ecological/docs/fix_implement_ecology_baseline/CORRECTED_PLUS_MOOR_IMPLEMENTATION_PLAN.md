# Corrected PLUS/MOOR Adaptation Implementation Plan

**Status:** plan only, awaiting approval  
**Date:** 2026-07-17  
**Controlling decisions:** `DECISIONS_FOR_CORRECTED_PLUS_MOOR_PLAN.md`  
**Superseded execution:** `paper_faithful_one_cell_4000_20260717_v1` is scientifically void

## 0. Scope and claim boundary

This plan implements the five settled decisions without reopening them. The public method IDs will be:

```text
plus_adapted_mechanistic_pbvi
moor_adapted_ricker_misspec_pbvi
```

The display descriptions will be:

> PLUS-adapted mechanistic candidate-POMDP baseline with fixed regime-persistence
> candidates, bootstrap parameter candidates, and PBVI.

> MOOR-adapted mechanistic trajectory-fitting baseline with preregistered Ricker
> misspecification, conditional-mean survey SSE, and PBVI.

These are paper-aligned adaptations, not reproductions of published PLUS or MOOR. PBVI remains a
named computational approximation. No headline sweep is authorized by this plan.

## 1. Fixed regime-transition candidate grid

### Primary grid

Use a symmetric two-state persistence model, fixed separately for each regime candidate:

```text
Pi(p) = [[p, 1-p],
         [1-p, p]]

p in {0.80, 0.90, 0.97}.
```

The grid is preregistered and independent of private simulator parameters and evaluation returns.
**Convention (verified against the fitter, 2026-07-18):** a complete episode has 25 transition/action
steps, so the regime path uses 25 latent states `z_0..z_24` across those 25 transitions, giving **24
inter-state switch opportunities**. The realised expected switch count is therefore `24(1-p)`:

| Persistence `p` | Expected switches over a 25-transition episode (`24(1-p)`) |
|---:|---:|
| 0.80 | 4.80 |
| 0.90 | 2.40 |
| 0.97 | 0.72 |

(The earlier `25(1-p)` figures — 5.00 / 2.50 / 0.75 — over-counted by one: the objective draws a 25th
regime update `z_25` per episode but it is never used in a transition. See
`tests/real/test_phase2_verification.py::RegimeSwitchCountTests`, which empirically confirms `24(1-p)`.
The transition *process* is unchanged; only the reported number is corrected.)

`Pi` is never optimized. For each candidate it is a fixed candidate parameter, serialized in full
with its persistence value, grid ID, row-simplex validation, and hash. Common uniform random numbers
generate discrete regime paths. Because `Pi` is fixed, gradients are required only for continuous
parameters and do not pass through the discrete regime draws.

### One shared regime law

Introduce one canonical discrete regime-law object and version hash. At transition `t`, abundance is
updated under current regime `z_t`, then `z_{t+1}` is sampled from `Pi[z_t,:]`. The fitter, serialized
model, POMDP kernel builder, online filter, and planner must all call or serialize this same law. The
current probability-weighted mean-field path is removed; no fitting/deployment approximation mismatch
is permitted.

### Default bank size and cost control

The primary bank remains **16 total candidates**, so the persistence grid does not silently multiply
the default cost:

| Family | Candidates | Allocation |
|---|---:|---|
| Ricker | 4 | registered complete-episode bootstrap-MAP schedule |
| Allee | 4 | registered complete-episode bootstrap-MAP schedule |
| theta-logistic | 4 | registered complete-episode bootstrap-MAP schedule |
| regime | 4 | `p=0.80`: 1, `p=0.90`: 2, `p=0.97`: 1 |

The two `p=0.90` candidates use the full-history MAP and one seeded episode-bootstrap MAP; the outer
grid points use full-history MAP fits under their distinct fixed transition laws. Candidate seed,
episode multiset, `Pi` grid entry, parameter hash, and diversity distance are serialized. The tradeoff
is explicit: the default retains all three persistence values and the 16-candidate cost, but has less
within-`Pi` bootstrap resolution at the outer persistence values.

## 2. Corrected conditional-mean MOOR objective

Let `Y_{e,t}` be the public survey in registered normalized survey units and let
`X_{e,t}^{(m)}(theta)` be the latent abundance prediction on process path `m`. For public protocol
noise `sigma_o`, the primary data objective is

```text
L_data(theta) =
  1 / sum_e(T_e M) *
  sum_e sum_t sum_m
    [X_{e,t}^{(m)}(theta) * exp(sigma_o^2 / 2) - Y_{e,t}]^2.
```

The existing registered survey-scale normalization is applied to both prediction and observation
before this expression. Initial-state uncertainty and process-noise paths remain Monte Carlo with
fixed common random numbers. There is **no survey-noise draw** in the prediction and no survey random
bank. The random-bank hash covers only episode-reset and process draws.

This objective is used for training and ordered holdout trajectory SSE. The deployment observation
model remains the public lognormal likelihood; changing the fitting target does not change filtering.
The conditional-mean factor is a necessary benchmark adaptation for replacing MOOR's deterministic
catch observable with an explicitly noisy abundance survey.

For the registered primary configuration, the complete objective is exactly `L_data` plus only the
existing parameter bounds/transforms. Section 5 removes the three provisional penalties.

## 3. Signed rates, public channels, and structural zeros

### Public schema

Add `action_channels: tuple[str, ...]` to hidden-mode `MethodContext`. The only permitted channel
tokens are:

```text
none, rate, capacity, rate+capacity, state
```

The trusted loader reads only `channel` from `real_ecology_data/actions.csv`. Method-facing objects
continue to expose action ID and cost, but never expose `K_multiplier`, `dN_fraction`,
`lambda_source`, `name_mechanistic`, `name_original`, `interpretation`, or any other table column.
The channel map is population-independent and therefore carries no population-specific demographic
response information.

### Primary action parameterization

The public channel determines which parameters exist:

| Actions | Public channel | Fitted rate | Fitted capacity `d_a` | Fitted stocking `u_a` |
|---|---|---|---|---|
| `a0` | none | shared baseline | 0 | 0 |
| `a1-a4` | rate | action-specific | 0 | 0 |
| `a5-a6` | capacity | shared baseline | action-specific | 0 |
| `a7-a9` | rate+capacity | action-specific | action-specific | 0 |
| `a10` | state | shared baseline | 0 | action-specific |

Each permitted effective rate is one bounded signed parameter `r_a`. The transition derives

```text
g_a = max(r_a, 0),
h_a = max(-r_a, 0).
```

Use preregistered signed bounds `r_min=-1.5` and `r_max=2.0` per transition. The smooth raw transform is
`r_a = 0.25 + 1.75*tanh(q_a)`, followed by the exact max split. These bounds preserve the provisional
positive-growth bound and conservatively include the provisional mortality range; they are method
bounds, not private action-effect values. Do not use softplus, a nonzero deployed offset, or
simultaneous positive `g_a,h_a` in the primary model. The raw coordinate for exact zero is
`atanh(-1/7)`, approximately `-0.14384`. At exactly zero,
the autodiff implementation uses the symmetric Clarke subgradient
`dg/dr=1/2`, `dh/dr=-1/2`. Deterministic starts include values on both sides of zero but never rely on
an offset in the deployed model. Strong-Wolfe failures are recorded per start; all-start failure is a
cell failure, not permission to switch objectives. A focused near-zero synthetic test is mandatory.

### Privacy gates

Extend the forbidden-name guard with every non-channel action-table column above. Tests patch the full
action-table reader to raise after the channel-only context has been built, recursively inspect all
method-facing objects and saved artifacts, and assert that hidden methods can recover neither private
magnitudes nor descriptive strings. Only the eleven-element tuple of categorical channel values may
cross the boundary.

## 4. Parameter count and identifiability

The provisional action model had `11 actions x (g,h,d,u) = 44` free action-effect parameters. The
corrected primary has:

```text
rate:      1 shared baseline + 7 rate/combined action rates = 8
capacity:  a5-a9                                      = 5
stocking:  a10                                        = 1
total action-effect parameters                         = 14
```

This is an exact reduction from 44 to 14, or 68.2% fewer action-effect parameters (approximately
3.1-fold smaller). Family-level reset, process, capacity-scale, Allee, theta, and regime parameters
remain separately reported and are not hidden inside this count.

Identifiability diagnostics record per free action parameter: row count, episode count, observed-state
range, selected-start variation, boundary proximity, gradient norm, local condition estimate, and
holdout loss. Structural-zero parameters are absent from the optimizer rather than fitted and later
zeroed. Channel design rank and signed-rate left/right profile around zero are also reported.

Weak identification is an outcome. The fitter may not fill missing information from private tables,
expand bounds after looking at results, or activate regularization automatically.

No inferential parameter claims will be made. Parameters are model components and descriptive
diagnostics for the policy comparison, not confidence-qualified ecological estimates. Profile and
bootstrap intervals are therefore omitted from the required run. Adding inferential parameter claims
would require a separately approved subset-only uncertainty plan before those claims are made.

## 5. Primary objective removes all three regularizers

The registered primary configuration removes:

```text
action shrinkage              = 0 and absent from the objective
group-deviation penalty       = 0 and absent from the objective
g_a*h_a complementarity       = 0 and absent from the objective
```

Configuration and artifacts carry `regularization_variant=none_structural_v1`. Validation fails if a
primary config supplies a nonzero legacy coefficient. Signed rates make the complementarity penalty
unnecessary, while public-channel structural zeros remove the stated motivation for group sparsity.

## 6. Separate optional hierarchical fallback

The only proposed fallback is a separately named configuration,
`regularization_variant=hierarchical_weak_v1`. It adds standardized weak quadratic shrinkage of
action-specific signed rates and capacity effects toward their public-channel means:

```text
lambda_rate = 1e-3
lambda_capacity = 1e-3
lambda_stocking = 0
group penalty = 0
complementarity penalty = 0
```

It has a distinct config hash, fit-cache key, artifact label, manifest arm, and output root. It is not
silently activated for a failed cell. The primary diagnostic subset is first evaluated unregularized.
If fewer than two of eight starts are finite, the selected gradient norm exceeds `1e-3` outside a
`|r_a| <= 1e-4` Clarke-stationary neighborhood, the local condition estimate exceeds `1e8`, or more
than 25% of free effects lie within 1% of a bound, the cell is flagged. If more than 10% of the 32
registered dynamics cells are flagged, execution pauses and requests separate approval for the
hierarchical subset ablation. No fallback result can replace the primary post hoc.

## 7. Reward-independent fit cache and reward-specific plan cache

### Transition-field hash

Do not key fitted models on `dataset_sha256`, because it includes reward and has a 0% safe/yield cache
hit rate. Define `fit_transition_hash_v1` over exact dtype, shape, and contiguous bytes for only:

```text
observations, actions, next_observations,
episode_id, timestep, terminated, truncated
```

The complete fit key additionally hashes:

```text
equation/law schema versions; observation sigma and normalization protocol;
num_actions and action_channels; fit config and regularization variant;
form; fixed Pi/grid ID when applicable; candidate/bootstrap seed;
sampled episode multiset; parameter bounds; optimizer version.
```

Rewards, reward mode, action costs, full `dataset_sha256`, evaluator fields, and private sidecars are
excluded from dynamics-fit identity. The full public dataset hash remains provenance metadata but is
not a cache key. Atomic writes, a per-key lock, complete artifact-hash validation, and fail-loud
collision checks prevent concurrent safe/yield rows from producing or reading partial fits.

### Reuse boundary

The 144 completed hidden-run dynamics cells have byte-identical fit fields across safe/yield, so each
MOOR model and PLUS bank is fitted once per dynamics cell and loaded for the other reward mode. PBVI
planning is **not** reused across reward modes. Its key includes fitted-model hash, reward-surrogate
hash, reward mode, planner config, horizon/context convention, and seed.

Instrumentation records separate durations and cache statuses for collection, fit, POMDP build, PBVI
solve, evaluation, and serialization. Tests require equal safe/yield transition hashes, unequal full
dataset hashes, one fit invocation, two reward-specific planner invocations, and byte-identical loaded
model artifacts.

## 8. Registered 32-cell diagnostic-subset manifests

The immutable dynamics-cell key is:

```text
expose_rk=hidden
data_mode=real
population in {amur_tiger, egyptian_vulture}
family in {ricker, allee, theta, regime}
sigma_o in {0p0, 0p1, 0p2, 0p4}
```

This is `2 x 4 x 4 = 32` dynamics cells. `amur_tiger` is marked recoverable and
`egyptian_vulture` sink. A fit manifest contains 32 cells x 2 method IDs = 64 fit rows. A separate
plan/evaluation manifest contains 32 cells x 2 methods x 2 rewards (`safe`, `yield`) = 128 rows, each
referencing an immutable fit key and model hash. Reward mode never appears in the fit key.

Every row also records: action-channel schema hash, observation protocol, candidate allocation, Pi
grid ID, regime-path count, bootstrap seeds, regularization variant, fit budget, planner budget,
dataset target/actual/overshoot/episode count, cache source, and scope=`registered_diagnostic_subset`.

Sensitivity manifests are separate from the primary subset manifest and cannot overwrite its output.
No corrected-method return may be inspected before these manifests and their hashes are frozen.

## 9. Candidate-count and regime-path sensitivities

Use one-factor-at-a-time subset sensitivities around the 16-candidate/16-path primary; do not run a
costly full factorial.

### Candidate bank size

| Arm | Total | Ricker | Allee | theta | regime Pi allocation `(0.80,0.90,0.97)` |
|---|---:|---:|---:|---:|---|
| smaller | 12 | 3 | 3 | 3 | `(1,1,1)` |
| primary | 16 | 4 | 4 | 4 | `(1,2,1)` |
| larger | 32 | 8 | 8 | 8 | `(2,4,2)` |

All bank-size arms use 16 regime process paths and identical registered seed prefixes. The primary
remains 16 regardless of sensitivity outcomes.

### Regime-path Monte Carlo

At the primary 16-candidate allocation, compare `M_regime in {8,16,32}` common-random-number paths.
Non-regime fits remain at the registered 16 process paths. Report objective, parameter, holdout-loss,
kernel, posterior, action-value, runtime, and memory sensitivity. The primary remains 16 paths; no
setting is selected from returns.

The subset therefore has five PLUS configurations, not nine: primary `(16 candidates,16 paths)`, two
candidate-count deviations at 16 paths, and two regime-path deviations at 16 candidates. Fits are
reused across reward modes; planning still runs separately for safe and yield.

### Nested fit reuse within the suite

The 12- and 16-candidate banks are deterministic prefixes of the 32-candidate bank, with candidate
IDs, seeds, episode multisets, and regime-Pi schedule ordered so the allocations above hold at each
prefix. The subset therefore fits the 32-candidate bank once and forms the smaller banks by immutable
selection and prior renormalization; it does not refit their shared candidates. The path-count arms
reuse the 12 non-regime candidates and refit only the four regime candidates at 8 and 32 paths.

Under linear candidate/path scaling, five independent arms would cost approximately `5.875*F16` in
fitting per dynamics cell. Nested reuse costs approximately
`2*F16 + 0.125*F16 + 0.5*F16 = 2.625*F16`, a 55.3% fit-cost reduction. This arithmetic is a sizing
model, not a substitute for measured component timings. Planning remains five configurations x two
reward modes and receives no analogous fit-reuse credit.

## 10. Canary and CPU projections

### Preserved measured evidence

The void run measured MOOR at 30:56 for one 4,000-row cell and PLUS at more than 2:59:55 without
finishing 16 candidates. A mechanical no-reuse 288-cell extrapolation is approximately 149 CPU-hours
for MOOR plus more than 858 CPU-hours for PLUS, over 1,000 CPU-hours total. These are lower-bound
sizing observations from defective objectives, not forecasts for corrected code.

### Corrected canary sequence

After code and plan approval:

1. Run synthetic correctness tests locally; no real return is opened.
2. Run one Amur-tiger/Ricker/`sigma=0.1` 4,000-target dynamics cell for both methods and both rewards,
   fitting once and planning twice. Hard canary ceiling: 24 CPU-hours total.
3. If step 2 passes, run one Egyptian-vulture/regime/`sigma=0.4` dynamics cell with the same reuse
   pattern. Cumulative canary ceiling: 48 CPU-hours total.
4. Open only timing, memory, convergence, cache, and invocation fields until scope is registered.

The canary separately measures `F_m` (fit plus model serialization), `K_m` (POMDP/kernel build),
`P_{m,r}` (reward-specific PBVI solve), and `E_{m,r}` (evaluation) for method `m` and reward `r`.

For 144 dynamics cells and two rewards, project:

```text
C_no_cache = sum_m 144 * [2*F_m + 2*K_m + sum_r(P_mr + E_mr)]
C_cache_conservative = sum_m 144 * [F_m + 2*K_m + sum_r(P_mr + E_mr)]
C_cache_kernel_reuse = sum_m 144 * [F_m +   K_m + sum_r(P_mr + E_mr)]
```

Kernel build is conservatively kept reward-specific in the formula until the canary proves which
kernel components are reward-independent. The canary must time transition/observation kernel creation
separately, compare safe/yield kernel-array hashes, and verify that any shared kernel excludes reward
tables/callbacks. Use the kernel-reuse projection only if that gate passes; otherwise use the
conservative projection. Report both with a fixed 1.5 scheduling and estimation contingency. The
theoretical maximum saving from reward-mode fit reuse is one half of fit cost, not one half of total
runtime.

The 32-cell diagnostic subset uses the same formula with 32 replacing 144. No full sweep is submitted
unless a PI supplies `B_full` and `1.5*C_cache <= B_full`. If that inequality fails, only the registered
subset may run, it remains diagnostic rather than a headline replacement, and the absence of a
full-scale claim is reported plainly.

### Sensitivity-suite projection and ceiling

For sensitivity arm `a` in `{bank12, primary16, bank32, regime_paths8, regime_paths32}`, define
`K_a`, `P_{a,r}`, and `E_{a,r}` as measured kernel, reward-specific planning, and evaluation costs.
With the nested fit schedule in Section 9, define `F32` as the one 32-candidate fit and `Freg8_4`,
`Freg32_4` as the two four-candidate regime refits. Project the complete 32-cell suite both ways:

```text
C_suite_conservative = 32 * [
  F32 + Freg8_4 + Freg32_4
  + sum_a(2*K_a + sum_r(P_ar + E_ar))
]

C_suite_kernel_reuse = 32 * [
  F32 + Freg8_4 + Freg32_4
  + sum_a(K_a + sum_r(P_ar + E_ar))
]
```

The void run gives `F16 > 2.98 CPU-h` because PLUS was still fitting when canceled. Linear scaling
therefore puts the nested suite's fitting alone above approximately
`32*2.625*2.98 = 250 CPU-h`; with the fixed 1.5 contingency this is above 375 CPU-hours before
planning. The corrected objective may change this timing, so the corrected canary replaces the
numerical estimate, but not the requirement to price the full suite.

Set a hard requested sensitivity-suite ceiling `B_sensitivity=600 CPU-hours`, inclusive of the 1.5
contingency and all five arms, both reward-specific planners, evaluation, failed-task allowance, and
serialization. Submit the suite only when the applicable measured projection satisfies
`1.5*C_suite <= B_sensitivity`. If it does not, stop and return a scope decision; do not silently drop
cells, arms, reward modes, or diagnostics. This ceiling is separate from the 48-CPU-hour canary ceiling
and the still-required PI-supplied full-sweep ceiling `B_full`.

## 11. Exact acceptance tests

### Conditional-mean survey fitting

1. **Analytic scalar test at `sigma_o=0.4`:** generate observations from a known latent scale and fit
   only that scale. Across fixed seeds, relative recovery error must be <=3%; the test also computes
   the removed objective's expected `exp(-0.4^2)=0.8521` ratio and demonstrates its approximately
   14.8% downward target without using that objective in production.
2. **Ordered synthetic trajectory recovery at `sigma_o=0.4`:** fit known controlled-Ricker episodes
   with process paths and require median abundance-scale recovery within 5% over preregistered seeds,
   finite gradients, and no survey random-bank field.
3. **Zero-noise regression:** at `sigma_o=0`, `exp(sigma_o^2/2)=1`; corrected predictions and SSE must
   be bit-identical to the no-survey-draw deterministic objective, and fitted parameters must agree to
   `atol=1e-10`, `rtol=1e-10` under the same process bank.

### Regime fit/deployment consistency

4. The fitter, model artifact, POMDP, filter, and planner must expose the same `regime_law_hash` and
   fixed `Pi` hash for every regime candidate.
5. For fixed actions, initial state/regime, process bank, and regime-uniform bank, the fitting
   trajectory simulator and deployed model simulator must produce identical regime paths and latent
   transitions. No mean-field state or probability-weighted threshold may exist in the fit path.
6. Every Pi matrix must be row-stochastic, immutable, absent from optimizer coordinates, and survive
   save/load byte-identically.
7. Direct Monte Carlo transition frequencies and serialized POMDP kernels must agree within a
   preregistered binomial tolerance at each persistence grid point.

### Action, privacy, caching, and optimizer gates

8. Assert the exact 44-to-14 action-parameter count and every channel-implied structural zero.
9. Exercise a synthetic optimum at `r_a=0` and on both sides of zero; verify the symmetric Clarke
   subgradient, finite L-BFGS behavior, exact `g*h=0`, and no softplus/offset leakage.
10. Recursively prove that method inputs/artifacts expose channel tokens but none of the forbidden
    action-table columns or values.
11. Safe/yield transition-field hashes must match while full dataset hashes differ; fitting executes
    once, planning executes twice, and corrupted or partial cache entries fail loudly.
12. Candidate allocation, unique parameter hashes, diversity thresholds, fit/holdout episode resets,
    posterior normalization, and filtering/planning model hashes remain acceptance gates.

No test threshold may be changed after real corrected-method returns are inspected.

## 12. Void-run preservation and exclusion

`real_ecology_runs/paper_faithful_one_cell_4000_20260717_v1/` remains preserved. Its frozen code still
has registered snapshot hash `2aae03ba1a22b8c9a3e2c850aa31814253f02f1b9691d5899b5b93e4ea0b13f5`.
`VOID.md`, completed MOOR artifacts, partial PLUS cancellation log, public dataset, manifests, and
runtime records remain inspection-only. The run has no acceptance artifact and contributes no fitted
parameters, returns, comparisons, calibration choices, or scientific claims.

Corrected runs use a new root, new code snapshot, new equation/law schema versions, new method IDs,
and new fit/cache hashes. Nothing from the void fit cache may be loaded.

## 13. Plan-only confirmation

This deliverable changes documentation only. During this planning turn:

- no source or test code is modified;
- no configuration or dependency is modified or installed;
- no snapshot is frozen;
- no Slurm or other experiment job is submitted;
- the void run is neither edited nor deleted;
- the five controlling scientific decisions are not reopened.

The pre-plan digest over `src/`, `configs/`, and `scripts/` was
`9b42a10bdf04e87429d7d411c4e73e3728f9dec32cd024697ddcdcf63ab0cb9b`. The post-plan digest is the same
`9b42a10bdf04e87429d7d411c4e73e3728f9dec32cd024697ddcdcf63ab0cb9b`, satisfying the mechanical
plan-only gate.

The digest recipe, run from the repository root, is:

```bash
find src configs scripts -type f \
  ! -path '*/__pycache__/*' ! -name '*.pyc' ! -name '*.pyo' -print0 \
  | sort -z | xargs -0 sha256sum | sha256sum
```

After implementation approval, extend the snapshot verifier to accept repeated roots and encode this
recipe directly. That source change is deliberately not made during this plan-only gate.

Implementation begins only after explicit approval of this updated plan. A corrected canary and any
larger execution require their own later authorization and CPU ceiling.

## Implementation change map after approval

| Surface | Planned change |
|---|---|
| `config.py` | add channel-only context, Pi grid/allocation, signed-rate, regularization-variant, cache-schema, and sensitivity config validation |
| `realdata.py` | add a strict channel-only public loader; never return the mixed private action record to hidden methods |
| `privacy.py` | forbid all non-channel action-table columns/values and recursively audit method-facing artifacts |
| `faithful_ecology.py` | canonical fixed-Pi discrete regime law, signed-rate structural action representation, and law hash |
| `faithful_fit.py` | conditional-mean SSE, process-only random bank, fixed-Pi discrete paths, 14-parameter action decoder, zero-kink convention, no primary penalties, transition hash, fit serialization |
| `faithful_pomdp.py` | consume the canonical regime law and fitted-model hash; preserve reward-specific PBVI context |
| `faithful_artifacts.py` | serialize Pi/grid, law/cache hashes, channels schema, all timing components, regularization variant, candidate allocation, and diagnostics |
| `methods/plus_faithful.py` | expose `plus_adapted_mechanistic_pbvi`, load/reuse immutable banks, and retain per-candidate beliefs/evidence updates |
| `methods/moor_faithful.py` | expose `moor_adapted_ricker_misspec_pbvi`, load/reuse one constrained Ricker fit, and retain explicit misspecification label |
| `methods/__init__.py` | register only the adopted external IDs; old provisional IDs fail loudly in corrected manifests |
| `pipeline.py` | split fit and plan stages, implement atomic transition-keyed fit cache, instrument fit/kernel/plan/evaluation costs, and prohibit reward-keyed refitting |
| `make_paper_faithful_manifest.py` | generate separate 32-cell fit and 128-row reward-plan manifests plus isolated sensitivity manifests |
| acceptance and tests | add the exact Section 11 gates, cache/privacy checks, void-cache exclusion, canary blinding, CPU accounting, and a repeated-root code/config/script verifier |

Implementation order after approval is: schema/privacy tests; canonical equations and objectives;
synthetic correctness; fit/plan cache split; method renaming and artifacts; diagnostic manifests;
return-blind canary. Any failed blocking gate stops before real execution.
