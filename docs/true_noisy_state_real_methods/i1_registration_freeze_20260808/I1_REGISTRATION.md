# I1 registration freeze — method-level state-information pilot

**Registration ID:** `method_level_state_information_pilot_i1_20260808`  
**Status:** FROZEN FOR INDEPENDENT READ-ONLY AUDIT; EXECUTION NOT AUTHORIZED  
**Controlling plan:** `docs/true_noisy_state_real_methods/DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN_REV3_1.md`  
**Plan SHA-256:** `701b4509f1dc04b7885891be72e526420ac5558be56bdde2b9628789398fe959`  
**Confirmation audit:** `docs/true_noisy_state_real_methods/CLAUDE_REV3_1_CONFIRMATION_AUDIT.md`  
**Audit SHA-256:** `0f448db89109186aa4bbd64bc000da45ae662386dea79e658bcd802f4880cd17`  
**Registration root:** `docs/true_noisy_state_real_methods/i1_registration_freeze_20260808`  
**Proposed future execution root:** `outputs/method_level_state_information_pilot_20260808_v1` — frozen as a name only; not created or authorized

This registration freezes the design and qualitative predictions. It does not authorize
truth extraction, derived data, adapters, tests, regression, rebaseline, scientific
execution, or Slurm work.

## 1. Experiment identity

Scientific question:

> For the accepted ecological and general-RL implementations, how much performance is
> lost when exact abundance is replaced by noisy surveys in out-of-family Allee
> environments, and is that loss consistent with each method's treatment of hidden
> abundance?

The primary estimand is end-to-end method robustness to hidden abundance. It is a
method-bundle stress test, not a causal state-representation effect. Transition/value
fits, residual noise, model-posterior sharpness, objectives, planners, safety calibration,
compute and policy class may differ.

A result may be labelled **frozen-fit/state-input-only** only under the byte-identity rule
in Section 5. Otherwise it is end-to-end bundle evidence. Cross-method absolute returns
are architecture-bundle comparisons.

Registered cells:

- Amur tiger × Allee × sigma 0.2 — general-method activity/stress cell and OGSRL anchor.
- Crab-eating fox × Allee × sigma 0.2 — ecological-method activity cell.

Registered methods:

- adapted PLUS: `plus_adapted_ricker_only_pbvi`;
- adapted MOOR: `moor_adapted_ricker_misspec_pbvi`;
- RefPlan-inspired: `refplan`;
- OGSRL-inspired: `ogsrl`;
- BA-MCTS-inspired: `bamcts`;
- EVD pessimism: `ensemble_value_disagreement_pessimism`.

Arm T supplies current exact abundance through a method-specific, context-preserving
contract. Arm O supplies accepted noisy surveys and accepted method logic. Sigma 0.2 is
screening only. Stage C/sigma 0.1 is absent and separately unauthorized.

## 2. Common controls and accepted identities

- Accepted safe evaluator with collapse penalty P=10.
- Eleven registered actions from `configs/ecology/action_effects_long.csv`, file SHA-256
  `b0944a1e2f00460c9754c4697eb1c2407391e92572c4fca0f713b63985048f14`;
  accepted logical action hash `b591e63937a3eefe5570de5adf2d479b83b3783e5e83dde864eecbbeaee9a673`.
- Environment hash `c7079a34155c482ed449359c72b7089ac04dbc64f148776347f54a75066b4b06`.
- Evaluator hash `125ec6073b203e34dc10d7182082e9439d577c9fa819953d280b3401128136bb`.
- Reward hash `356d126d23fedc69e3632b923b6e8a9a071896813f7ddf9f5ad509823a8b33d8`.
- Real-data identity hash `f293ed77f79208446fb04df3498b68941f15fb7df0979e8b8d3362ef8ed546d1`.
- Horizon 50; discount 0.95; 20 paired evaluation identities in this exact order:
  `7001, 7002, 7003, 7004, 7051, 7052, 7053, 7054, 7101, 7102, 7103, 7104,
  7151, 7152, 7153, 7154, 7201, 7202, 7203, 7204`.
- Accepted master seed 116 and surrogate split seed 20116. No collection, fit, planning,
  split or evaluation identity may change. I2 must inventory all embedded method-specific
  seeds before execution; a missing identity is a stop.
- Identical row ordering, splits, action effects, evaluator, true Allee distribution,
  reward, evaluation pairing and non-information hyperparameters.

Accepted table: `results/accepted/MATCHED_P10_144_METHOD_CELLS.csv`,
SHA-256 `7431318803e468c13ff3008acdc530c0d9f9a919cc29a6fe04951fc8a2cadc1c`.
Receipt: `results/accepted/MATCHED_P10_144_RECEIPT.json`,
SHA-256 `a198c70f34ed0191a6580c49d52ab675ea7b6768f87076994775b223624d3163`.

The exact public/truth paths, hashes and bindings are frozen in
`I1_SOURCE_SCHEMA.json` (SHA-256
`4de0b45f347576d82465c9590c9caf98646cecb2d3c3d4afac07cfca414c20bc`).
Both cells contain 4,000 rows arranged as 160 complete 25-transition episodes.

Accepted and frozen namespaces are immutable: never modify, overwrite, append, rescore,
or place a new file beneath them.

## 3. Truth schema, units and private-information boundary

The source archive has exactly 13 keys:
`C, entry, initially_unsafe, metadata_json, next_regime, next_states, r_base,
r_eff_true, regime, reward_true, safety_penalty_applied, states, theta`.

Only `states` is generally allowlisted. `next_states` is conditionally allowlisted
only as the same row's offline one-step supervised target for a primary end-to-end method
that requires refitting. It is unavailable to every frozen-fit diagnostic and must never
reach deployment, action selection, an online adapter, a planning root, a later-row
decision or another runtime path. Claude verified
`next_states[t] == states[t+1]` within episodes at zero tolerance.

`metadata_json` is binding-only: a dedicated, separately authorized external extractor
may read only `public_dataset_sha256`; it must not enter a derived artifact. Every other
source key is forbidden. Methods and training pipelines may never open original
`truth.npz` directly. Arm O may never access a derived artifact.

The future derived overlay schema is exact:

- state-only/frozen/runtime overlay: exactly `states`;
- one-step fitting overlay: exactly `states` and `next_states`.

Already-public action/context/identity/reward fields remain in immutable `public.npz`
and join by exact row order. Any missing, unexpected, reordered, nonfinite, unit-mismatched
or binding-mismatched value aborts.

Truth states and public observations share raw abundance units. Audited
`mean(observation/state)` is 1.02142 for tiger and 1.01972 for fox; the theoretical
log-normal multiplier at sigma 0.2 is 1.02020. General methods therefore use raw abundance.
PLUS/MOOR grids remain in latent model units: direct point-mass assignment must convert
through each fitted model's `survey_scale`. Raw truth must never be snapped directly to
a latent grid. The selected latent bin times `survey_scale` must match the registered
raw-state discretisation.

The private-cache override near `pipeline.py:432` is registered only for this diagnostic,
the two cells and Arm T. Any future overlay must be created by an external stage outside
frozen tracks, in a separately authorized private namespace, with source/derived hashes
and row alignment. Retain it read-only and access-restricted through independent results
audit and final acceptance; destroy only under separate user authorization while keeping
non-sensitive receipts. This I1 performs no override and creates no overlay.

## 4. Arm T boundary and I2 gates

Any future I2 must fail closed unless it proves:

- each method's required public observation/action/timestep history is preserved;
- only current true abundance reaches runtime action selection;
- method code cannot open original `truth.npz`;
- Arm O cannot access a derived overlay;
- hidden family, thresholds, reward, reward components, demographic parameters, latent
  regimes, evaluator constants and future information are absent;
- no sigma or observation-scale value is collapsed to zero/epsilon;
- no all-zero observation-likelihood fallback is entered;
- PLUS/MOOR use direct point-mass belief assignment after `survey_scale` conversion;
- RefPlan's adapter remains unresolved until independently audited in I2;
- OGSRL recalibrates its exact-state safety budget and reports the safety-axis change;
- every model-fitting, residual-noise, posterior-sharpness, reward, policy and safety
  component change is receipted.

## 5. Frozen residual_sigma interpretation rule

> `residual_sigma` is a transition/model-fitting quantity and a possible mediator of the
> end-to-end information intervention. If Arm T and Arm O do not use byte-identical fitted
> artifacts with identical `residual_sigma`, their return difference includes a
> model-fit/model-posterior change and must not be interpreted as a pure
> state-representation effect.

Before returns are opened, record every fitted-artifact hash and every available
`residual_sigma` for every method and arm.

The **frozen-fit/state-input-only** label requires all four:

1. fitted-artifact hashes are byte-identical across arms;
2. `residual_sigma` values are bit-identical;
3. transition/value/reward fits and policy parameters are unchanged;
4. only the registered state-information adapter differs.

Any non-identical artifact hash or `residual_sigma` automatically triggers:

`MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY`

There is no magnitude threshold and no post-hoc “small enough” exception. Refit reports
must show raw Arm T/O values, absolute difference, ratio when the Arm O denominator is
positive, and posterior entropy/sharpness changes where available. RefPlan and BA-MCTS
must state when exact-state refitting sharpens posteriors through lower residual noise.
The primary end-to-end comparison remains reportable but cannot identify state
representation causally. A mechanism claim requires a separate matched-fit registration.
This rule is fixed before I2 and cannot change after returns open.

## 6. Metrics and synthesis

Primary within each method and cell:

`paired_raw_loss = mean_return_true_state - mean_return_noisy_survey`

The sampling unit is one intact paired evaluation identity. The point estimate is the
arithmetic mean of the 20 ordered paired return differences in raw return units.

The 95% interval is a paired nonparametric percentile bootstrap: resample the 20 intact
T/O pairs with replacement 100,000 times using NumPy Generator PCG64 seed 20260808; use
the 0.025 and 0.975 quantiles with `numpy.quantile(method="linear")`. Never resample arms
independently or across cells.

Also report collapse-entry rate, first-collapse time, unsafe-occupancy steps, discounted
safety-penalty contribution, minimum abundance, time below threshold, action diversity,
paired-decision differences, action costs, adaptive headroom, runtime and failures.

Normalized loss is secondary:
`paired_raw_loss / max(abs(mean_return_true_state), 1.0)`.
If `abs(mean_return_true_state) < 1.0`, report `NA_DENOMINATOR_GUARD`; never rank by
the ratio.

Do not pool or average tiger and fox. Cross-cell synthesis is limited to sign, within-cell
ordering, activity classification and interpretation category. EVD is reported separately
because it uses raw logged rewards while the other five use the learned surrogate.

## 7. Activity and validity gates

Tiger is the general-method activity/stress cell; fox is the ecological-method activity
cell. Accepted Arm O pre-classifications are frozen in `I1_REGISTRATION.json` with all
12 entropy/headroom values and episode-file hashes. Key classifications:

- tiger PLUS/MOOR: inactive/non-discriminating, constant action 10;
- tiger OGSRL: active with positive headroom;
- fox PLUS/MOOR: active with positive headroom;
- fox OGSRL: inactive/non-discriminating;
- the remaining general methods are decision-active, with headroom labels recorded
  exactly in the JSON.

Any single constant deployed action is non-discriminating. A state-representation
interpretation requires decision activity and meaningful positive adaptive headroom.

Mandatory gates include exact adapter leakage, 4,000 rows/160×25, behavior horizon 25,
no episode shorter than 25, accepted Arm O per-episode/action/event parity, accepted CPU
architecture, source-only hashes, return reconstruction within absolute tolerance
`1e-9`, paired identities, no pooling and no Stage C inference.

## 8. Qualitative predictions

Predictions were sealed before new returns in `SEALED_PREDICTIONS.json`, SHA-256
`82a7fcb394c80e673ede04c5692db839821b06f23f752076e8cf34bb29da50c3`.

They freeze the plan's tiered outcomes: possible tiger ecological non-discrimination,
fox ecological activity, RefPlan as decisive control, observation-aware preprocessing
interpretation, calibrated latent-state motivation, transition/reward/planner redirection,
bundle-only interpretation after any model-fit change, and no causal state-representation
claim from this pilot.

## 9. Provenance and compute

All commands require `PYTHONDONTWRITEBYTECODE=1` and `LC_ALL=C`.

Historical hashes:

- ecological, 149 files: `951365d7874a417d7e66b14538dc275a9f325ac4643df8ea1af4d1d24877fb01`;
- general, 162 files: `614524d7b058418a3ff3f370e7e5b57582213c2c518766d8d88f1ec093e7a35e`.

Source-only hashes and complete 52/55-file lists are frozen in
`I1_REGISTRATION.json`:

- ecological: `2b3b8ae6d2f8ff5ffb17c4885ded9e8f1f6b3c0cb662f393186fe4b4706a884e`;
- general: `f90cea6f28dcacb910b5e036bf9e09958715d00a2418fd0856a3d5a12856bdbd`.

Existing-fit planning estimate: 18,587.53 seconds/5.163 core-hours per cell-arm and
74,350.12 seconds/20.65 core-hours for two cells × two arms. This excludes adapter work,
truth extraction, exact-state general-method rebuilds, PLUS refitting/replanning, queue
and I/O. PLUS dominates at about 4.25 hours per existing-fit method-cell; the nearest
cold eight-candidate estimate is 2.41 hours per affected cell. Reassess after I2. Any PLUS
fit/planning change stops scheduling until measured, audited and separately authorized.

## 10. Stop rules

Stop on any leakage; source/dataset/accepted/fitted/track hash mismatch; row-alignment
failure; unexpected schema key; short episode; unapproved model/refit/reward/policy/seed
change; Arm O parity failure; zero-noise or all-zero-likelihood path; lost history context;
inability to receipt frozen-fit versus end-to-end arms; CPU mismatch; return reconstruction
error above `1e-9`; or tracked source/config/test change.

## 11. Authorization status

| Stage/action | Status |
|---|---|
| I0 | Complete and independently verified |
| Revision 3.1 | Independently verified |
| I1 | Authorized and completed for documentation freeze only |
| I2 | Not authorized |
| Truth extraction/derived Arm T data | Not authorized; none created |
| Adapters/tests | Not authorized |
| Regression/rebaseline | Not authorized |
| Stage B | Not authorized |
| Stage C | Not authorized and absent |
| Slurm jobs | None submitted |
| Accepted outputs/frozen tracks | Untouched |
| Source code/tests/configurations | Untouched |

Stop after I1 and wait for independent read-only audit.
