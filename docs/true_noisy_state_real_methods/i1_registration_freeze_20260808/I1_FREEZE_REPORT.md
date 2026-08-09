# I1 registration freeze report

Date: 2026-08-08 (Australia/Melbourne)  
Registration ID: `method_level_state_information_pilot_i1_20260808`  
Mode: documentation-only registration freeze  
Scientific execution: not authorized and not performed

## 1. Executive result

The I1 registration is complete. The experiment identity, two cells, six methods, two
information arms, end-to-end estimand, secondary frozen-fit eligibility, information
boundary, complete truth schema, state units, metrics, activity/validity gates, qualitative
predictions, compute assumptions, stop rules and authorization boundary are frozen.

No truth arrays were opened by I1, extracted, copied or written. No derived Arm T artifact,
adapter, test, execution configuration, experiment execution namespace, regression,
rebaseline, evaluation or Slurm job was created or run.

## 2. Created registration artifacts

| File | SHA-256 before checksum-manifest creation |
|---|---|
| `I1_REGISTRATION.md` | `d877fbcf74db7b2a0005d74e19f168e52de9b60ee3df613646e3017d21acb4f3` |
| `I1_REGISTRATION.json` | `d4a9b99af333ffc26bc681a7283447680230ad010f0d23b7c496f4edb66e28ff` |
| `SEALED_PREDICTIONS.json` | `82a7fcb394c80e673ede04c5692db839821b06f23f752076e8cf34bb29da50c3` |
| `I1_SOURCE_SCHEMA.json` | `4de0b45f347576d82465c9590c9caf98646cecb2d3c3d4afac07cfca414c20bc` |
| `I1_HASHES.sha256` | Created last; standard checks cover the other five files and a normalized self-check covers the manifest itself. |
| `I1_FREEZE_REPORT.md` | Covered by `I1_HASHES.sha256` after this report was finalized. |

## 3. Frozen scientific and information-boundary decisions

- Primary estimand: paired end-to-end method robustness to hidden abundance.
- Primary metric: `mean_return_true_state - mean_return_noisy_survey` within method/cell.
- Sampling unit: the 20 intact paired evaluation identities.
- Interval: 100,000-resample paired percentile bootstrap, PCG64 seed 20260808, 95%
  interval, linear 0.025/0.975 quantiles.
- No fox/tiger pooling; only sign, within-cell ordering, activity class and interpretation
  category may be synthesized across cells.
- EVD remains separate because it uses raw logged rewards.
- Sigma 0.2 is screening only; Stage C is absent.
- The 13-key truth schema is frozen and complete. Only `states` and conditional
  offline-only `next_states` are scientific inputs to a future external extractor.
- `metadata_json` is binding-only and may not enter a derived artifact.
- Methods and training pipelines may never open original `truth.npz`.
- Runtime access to `next_states`, later rows or future realized information is forbidden.
- Truth/public state units are raw abundance. PLUS/MOOR must divide by fitted
  `survey_scale` before point-mass latent-grid assignment.
- The future derived overlay schemas are exactly `states`, or `states` plus
  offline-only `next_states`; already-public context remains in immutable `public.npz`.

## 4. Frozen residual_sigma rule

The full rule appears in both human and machine-readable registrations. Any non-identical
fitted-artifact hash or `residual_sigma` triggers:

`MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY`

No magnitude threshold or post-hoc “small enough” exception exists. The
**frozen-fit/state-input-only** label requires byte-identical fitted artifacts,
bit-identical `residual_sigma`, unchanged transition/value/reward fits and policy
parameters, and only the registered state adapter differing.

## 5. Sealed predictions

Eight qualitative predictions are sealed in `SEALED_PREDICTIONS.json`, SHA-256
`82a7fcb394c80e673ede04c5692db839821b06f23f752076e8cf34bb29da50c3`.
They contain no invented numerical result and cover tiger ecological non-discrimination,
fox ecological activity, RefPlan's decisive-control role, observation-aware preprocessing,
latent-state motivation, transition/reward/planner redirection, bundle-only labelling
after fit changes, and the prohibition on causal state-representation claims.

## 6. Validation results

| Validation | Result |
|---|---|
| Controlling plan SHA-256 | **MATCH** — `701b4509f1dc04b7885891be72e526420ac5558be56bdde2b9628789398fe959` |
| Confirmation audit SHA-256 | **UNCHANGED** — `0f448db89109186aa4bbd64bc000da45ae662386dea79e658bcd802f4880cd17` |
| Revision 3.1 changelog | **UNCHANGED** — `cb44f16b089d8ac4ba50ae02abca72c2fe33a28f3e3cddce899b4c863181fb37` |
| Revision 3 and changelog | **UNCHANGED** — `164a536b…3290`, `5e1d4da1…6f7d` |
| I0 checksum verification | **3/3 OK** |
| JSON parsing | **PASS** for all three JSON files |
| Truth schema | **13 exact keys / 13 classified fields** |
| Source-only file lists | **52 ecological / 55 general** |
| Sealed predictions | **8**, all qualitative |
| Activity evidence | **12 method/cell labels** with exact episode paths and hashes |
| Truth/public paths and file hashes | **MATCH** for both cells |
| Accepted table/receipt and action/CPU/dependency hashes | **MATCH** |
| Historical ecological/general hashes | **MATCH** — `951365d7…fb01`, `614524d7…e35b` |
| Source-only ecological/general hashes | **MATCH** — `2b3b8ae6…884e`, `f90cea6f…dbd` |
| `git diff --exit-code` and `git diff HEAD --exit-code` | **exit 0** |
| Future execution root | **ABSENT** — not created |
| Scientific arrays/datasets written | **NONE** |
| Code/test/config changes | **NONE** |
| Jobs/tests/regression/evaluations | **NONE** |

The final untracked set consists only of ten pre-existing planning/audit files under
`docs/true_noisy_state_real_methods/` and the six new I1 registration files. No untracked
path exists under `src/`, `configs/`, `tests/`, `results/`, `provenance/` or the
proposed future execution root.

`I1_HASHES.sha256` uses five standard SHA-256 entries plus a reproducible
`SELF-NORMALIZED-SHA256` comment for itself, calculated after replacing its recorded
self-digest with 64 zeroes. This avoids an impossible literal self-referential digest
while covering all six required files.

## 7. Compute and stops

The existing-fit estimate is 18,587.53 seconds per cell-arm and 74,350.12 seconds
(20.65 core-hours) for two cells × two arms. It excludes adapter development, truth
extraction, general Arm T rebuilds and PLUS refitting/replanning. PLUS is the expected
wall bottleneck at about 4.25 hours per existing-fit task; the nearest cold eight-candidate
fit estimate is 2.41 hours per affected cell. Compute must be reassessed after I2.

All registered leakage, hash, alignment, short-episode, unapproved-refit, parity,
zero-noise/all-zero-likelihood, context-preservation, architecture, return-reconstruction
and arm-classification failures are hard stops.

## 8. Authorization status

- I0: complete and independently verified.
- Revision 3.1: independently verified.
- I1: authorized and completed for documentation freeze only.
- I2: not authorized.
- Truth extraction and derived Arm T artifacts: not authorized.
- Adapters/tests: not authorized.
- Regression/rebaseline: not authorized.
- Stage B and Stage C: not authorized.
- Slurm jobs: none submitted.
- Accepted outputs and frozen tracks: untouched.

No I2 work or scientific execution may begin without a separate authorization after an
independent read-only audit of this freeze.

PASS — I1 REGISTRATION FROZEN; READY FOR CLAUDE READ-ONLY AUDIT
