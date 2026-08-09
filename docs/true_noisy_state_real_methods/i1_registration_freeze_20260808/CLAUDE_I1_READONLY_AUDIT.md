# Claude independent read-only audit of the frozen I1 registration

Date: 2026-08-08 (Australia/Melbourne)
Auditor: Claude (independent read-only pass)
Registration: `method_level_state_information_pilot_i1_20260808`
Controlling plan: `DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN_REV3_1.md` (`701b4509…e959`, verified unchanged)
Controlling audit: `CLAUDE_REV3_1_CONFIRMATION_AUDIT.md` (`0f448db89109186aa4bbd64bc000da45ae662386dea79e658bcd802f4880cd17`, verified unchanged)
Environment: every command run with `PYTHONDONTWRITEBYTECODE=1` and `LC_ALL=C`; no bytecode generated

Collision check: `CLAUDE_I1_READONLY_AUDIT.md` did not exist before this pass. No existing file was overwritten.

---

## 1. Executive verdict

**I1 is complete, internally consistent and independently verified. Every factual claim I tested reproduces exactly. The one carry-forward condition from my Revision 3.1 audit is closed. I1 is ready for a bounded I2 authorization.**

Six conclusions:

1. **Integrity is clean.** All six expected hashes match. All three JSON files are valid. The manifest covers all six artifacts via five standard entries plus a normalized self-check, and I independently reproduced that self-check procedure from scratch. Revision 3.1, both prior audits and all I0 artifacts are unchanged; `git diff --exit-code` returns 0; both frozen tracks are unchanged on both recipes.

2. **The Markdown and JSON registrations agree on every field I checked — no mismatches.** I compared all seventeen required categories. Where the JSON carries more detail (bootstrap parameters, 52/55-file source lists, twelve activity rows, six gate blocks), the Markdown summarises it without contradiction.

3. **The `residual_sigma` rule is now fully frozen — this closes my Revision 3.1 carry-forward.** It appears identically in both registrations, with the automatic label `MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY`, `numeric_threshold: null`, `post_hoc_small_enough_exception: false`, four bit-identity conditions for the frozen-fit label, and five required per-member reporting items. This was the single item I flagged as a rejection criterion for this audit; it is satisfied in full.

4. **The source schema is exact.** It declares 13 keys; I verified the set is *identical* to the real `truth.npz` archive, with no duplicates, and that every declared dtype and shape is accurate. Two fields are allowlisted (`states`; `next_states` offline-only), `metadata_json` is binding-only, and the remaining ten are individually named as forbidden with `permitted_consumer: none`. The two derived overlay schemas are exact-key, and `state_only_overlay` explicitly forbids `next_states`, which structurally blocks it from every frozen-fit diagnostic.

5. **Substantive facts verify, not just hashes.** The 20 frozen evaluation identities are the real accepted ones (block seeds 7001/7051/7101/7151/7201 × 4 episodes, confirmed against both the accepted config and the accepted episode records). All twelve activity entropy/headroom values reproduce to my independently computed figures. The action-table hash, five accepted shared hashes, six per-method compute means, both source-only file lists (52 and 55 paths, matched against disk) and both truth/public bindings all reproduce exactly.

6. **Nothing was created.** No truth array was opened by I1, no derived overlay, no execution namespace (`outputs/method_level_state_information_pilot_20260808_v1` does not exist), no adapter, test, config or job. No file under `/fs04/scratch2/ce25/general_rl_phase2_iso` or `outputs/` was written today.

Three residual ambiguities remain (§9). All are I2 design items, none is a registration defect, and the sharpest one — a byte-identical reward surrogate consuming a changed input distribution — is confined to methods that are already labelled bundle-only, so it cannot corrupt the primary inference.

---

## 2. Integrity table

### 2.1 Expected hashes

| Artifact | Expected | Recomputed | Result |
|---|---|---|---|
| `I1_REGISTRATION.md` | `d877fbcf…b4f3` | `d877fbcf74db7b2a0005d74e19f168e52de9b60ee3df613646e3017d21acb4f3` | **MATCH** |
| `I1_REGISTRATION.json` | `d4a9b99a…e28ff` | `d4a9b99af333ffc26bc681a7283447680230ad010f0d23b7c496f4edb66e28ff` | **MATCH** |
| `SEALED_PREDICTIONS.json` | `82a7fcb3…50c3` | `82a7fcb394c80e673ede04c5692db839821b06f23f752076e8cf34bb29da50c3` | **MATCH** |
| `I1_SOURCE_SCHEMA.json` | `4de0b45f…c20bc` | `4de0b45f347576d82465c9590c9caf98646cecb2d3c3d4afac07cfca414c20bc` | **MATCH** |
| `I1_HASHES.sha256` | `5c483370…3b27` | `5c483370c35eebab21348a8b1a24e84e5f1cdc63b8a4f9f029251e37cff13b27` | **MATCH** |
| `I1_FREEZE_REPORT.md` | `e335019a…01a3` | `e335019a6db7e4bebb02923d1f44e0f554fa22ac831c8cba5f65c9e9ea2201a3` | **MATCH** |

### 2.2 JSON validity and manifest coverage

All three JSON files parse cleanly. `I1_REGISTRATION.json` has 18 top-level keys, `I1_SOURCE_SCHEMA.json` 13, `SEALED_PREDICTIONS.json` 6.

`sha256sum -c I1_HASHES.sha256` **from the repository root** (the manifest uses repo-root-relative paths, matching the I0 convention) → **5/5 OK**. **All six I1 artifacts are covered**: five by standard entries, and `I1_HASHES.sha256` itself by the normalized self-check.

### 2.3 Independent verification of the normalized self-check

The freeze report (lines 100–103) states the manifest's self-digest is computed "after replacing its recorded self-digest with 64 zeroes". I implemented that from the description alone:

| Quantity | Value | Result |
|---|---|---|
| Recorded `SELF-NORMALIZED-SHA256` | `5e26c67cc614ad67fb0a8f0cd590a0ec2bdcda5def1d0df99d4c9177fde08a1a` | — |
| Recomputed (digest → 64 × `0`, then SHA-256 whole file) | `5e26c67cc614ad67fb0a8f0cd590a0ec2bdcda5def1d0df99d4c9177fde08a1a` | **MATCH** |
| Occurrences of the digest string in the file | 1 | **unambiguous replacement** |

The procedure is well-defined, reproducible from the written description, and solves the self-reference problem correctly.

### 2.4 Surrounding state

| Check | Result |
|---|---|
| `DRAFT_…_REV3_1.md` | `701b4509…e959` — **unchanged** |
| `REV3_1_CHANGELOG.md` | `cb44f16b…fb37` — **unchanged** |
| `DRAFT_…_REV3.md` / `REV3_CHANGELOG.md` | `164a536b…3290` / `5e1d4da1…6f7d` — **unchanged** |
| `CLAUDE_REV3_1_CONFIRMATION_AUDIT.md` | `0f448db8…cd17` — **unchanged; matches the value I1 cites** |
| `CLAUDE_REV3_READONLY_AUDIT.md` | `3139e575…67cb` — **unchanged** |
| `CLAUDE_I0_READONLY_AUDIT.md` | `3e9ec9e6…2ffc` — **unchanged** |
| `I0_HASHES.sha256` | **3/3 OK** |
| `git diff --exit-code` | **exit 0 — CLEAN** |
| `git diff HEAD --exit-code` | **exit 0 — CLEAN** |
| HEAD / branch | `77cd38ad…` / `e1-phase1-parity` — unchanged, no branch switch |

**Untracked files (reported separately) — 16**, exactly as the freeze report states (ten pre-existing planning/audit files plus the six new I1 files). None under `src/`, `configs/`, `tests/`, `results/`, `provenance/` or any execution root. This audit adds a seventeenth.

### 2.5 Frozen tracks

| Recipe | Track | Files | Hash | Result |
|---|---|---:|---|---|
| Historical | ecological | 149 | `951365d7…fb01` | **unchanged** |
| Historical | general | 162 | `614524d7…e35b` | **unchanged** |
| Source-only | ecological | 52 | `2b3b8ae6…a884e` | **unchanged** |
| Source-only | general | 55 | `f90cea6f…6bdbd` | **unchanged** |

The JSON's complete 52- and 55-path file lists were compared against disk: **both match exactly**.

---

## 3. Markdown / JSON consistency table

| # | Field | `I1_REGISTRATION.md` | `I1_REGISTRATION.json` | Verdict |
|---|---|---|---|---|
| 1 | Scientific question and estimand | §1, question verbatim; "end-to-end method robustness … method-bundle stress test, not a causal state-representation effect" | `scientific_identity.question`, `.primary_estimand`, `.architecture_bundle_limitation`, `.narrower_diagnostic_rule` | **AGREE** |
| 2 | Cells | tiger (general activity/OGSRL anchor), fox (ecological activity) | `cells[2]` with same ids and roles | **AGREE** |
| 3 | Methods | six, accepted registration ids | `methods[6]`, same ids, plus per-method frozen-fit eligibility | **AGREE** |
| 4 | Arms | Arm T context-preserving exact abundance; Arm O accepted noisy + accepted logic | `arms.T` / `arms.O`, `zero_noise_likelihood:false` | **AGREE** |
| 5 | Information boundaries | §3 + §4 | `truth_information_boundary` (19 keys) | **AGREE** |
| 6 | Evaluator | accepted safe evaluator, hash `125ec607…` | `accepted_shared_hashes.evaluator` identical | **AGREE** |
| 7 | Reward | safe mode, P=10, hash `356d126d…` | `reward_mode`, `collapse_penalty:10`, same hash | **AGREE** |
| 8 | Actions | 11 actions, file `b0944a1e…`, logical `b591e639…` | `action_count:11`, `action_table` same two hashes | **AGREE** |
| 9 | Horizon / discount | 50 / 0.95 | `evaluation_horizon:50`, `discount:0.95` | **AGREE** |
| 10 | Data identities | 20 ids listed; seed 116; split seed 20116; 4,000 rows / 160×25 | identical 20-id list in three places; same seeds; same row/episode counts | **AGREE** |
| 11 | Primary / secondary metrics | paired raw loss; normalized secondary with `max(abs(...),1.0)` guard | `metrics.primary` (+ full interval spec), `metrics.normalized_secondary` | **AGREE** |
| 12 | Activity gates | tiger/fox roles; five key classifications; constant action ⇒ non-discriminating | `activity_gate` with all 12 rows, entropies, headrooms, labels, episode hashes | **AGREE** |
| 13 | Parity / provenance gates | §7 list | `gates[6]` (G0–G5), `architecture_gate`, `provenance` | **AGREE** |
| 14 | Stopping rules | §10, thirteen conditions in prose | `stopping_rules[13]`, one-to-one | **AGREE** |
| 15 | Compute limits | 18,587.53 s / 74,350.12 s / 20.65 core-h; exclusions; PLUS 4.25 h; cold-fit 2.41 h | `compute` with identical figures and `excludes[5]` | **AGREE** |
| 16 | Authorization status | §11 table, 13 rows | `authorization`, 13 keys | **AGREE** |
| 17 | `residual_sigma` rule | §5 | `residual_sigma_interpretation_rule` (12 keys) | **AGREE** (§5 below) |

**Mismatches found: none.** No wording difference alters scientific meaning. The Markdown is a faithful human-readable projection of the JSON; the JSON is strictly the superset.

### Independently verified factual claims

| Claim | Source of truth | Result |
|---|---|---|
| Action table `b0944a1e2f00460c9754c4697eb1c2407391e92572c4fca0f713b63985048f14` | recomputed from `configs/ecology/action_effects_long.csv` | **MATCH** |
| Five accepted shared hashes (actions, environment, evaluator, reward, realdata) | I0 accepted receipt manifest | **5/5 MATCH** |
| Six per-method compute means; 18,587.53; 74,350.12; 20.65 core-h | I0 manifest + recomputation | **MATCH** |
| 20 evaluation identities | `configs/tracks/general/general_phase2e_full_sigma01_02.yaml` (`seeds: [7001, 7051, 7101, 7151, 7201]`, `episodes_per_seed: 4`) **and** the accepted tiger OGSRL `episodes.csv` `seed` column | **EXACT MATCH — these are the real accepted identities** |
| All 12 activity entropy + headroom values | my own recomputation from column 25 and the fixed-action screen | **12/12 MATCH** |
| Both truth/public paths, hashes, bindings, row counts | recomputed | **MATCH** |
| Source-only 52/55 file lists | compared against disk | **MATCH** |

---

## 4. Source-schema audit

`I1_SOURCE_SCHEMA.json` declares `exact_source_key_set` with 13 entries and a `fields` array of 13 entries.

**Set identity:** the declared key set is **identical** to the real `truth.npz` archive key set in both cells, with **no duplicates, no missing key and no extra key**. Verified by set comparison against the live archive.

**Per-key verification** (name / classification / permitted use / prohibited consumers / derived eligibility). Every declared `dtype` and `shape` was also checked against the archive and is accurate.

| Key | Classification | Permitted consumer | Derived-artifact eligible | dtype/shape | Verdict |
|---|---|---|---|---|---|
| `states` | allowlisted | methods **only via derived overlay**; direct `truth.npz` access prohibited | **true** | float64 [4000] | **CORRECT** |
| `next_states` | allowlisted conditionally | offline fitter via overlay only; never methods directly, deployment, action selection or planning roots | only in `one_step_fit` overlay | float64 [4000] | **CORRECT** |
| `metadata_json` | binding metadata only | external extractor may read **only** `public_dataset_sha256` | **false** | `<U1363` 0-d scalar | **CORRECT** |
| `C` | forbidden | none | false | float64 [4000] | **CORRECT** |
| `theta` | forbidden | none | false | float64 [4000] | **CORRECT** |
| `r_base` | forbidden | none | false | float64 [4000] | **CORRECT** |
| `r_eff_true` | forbidden | none | false | float64 [4000] | **CORRECT** |
| `regime` | forbidden | none | false | int8 [4000] | **CORRECT** |
| `next_regime` | forbidden | none | false | int8 [4000] | **CORRECT** |
| `reward_true` | forbidden | none | false | float64 [4000] | **CORRECT** |
| `entry` | forbidden | none | false | bool [4000] | **CORRECT** |
| `initially_unsafe` | forbidden | none | false | bool [4000] | **CORRECT** |
| `safety_penalty_applied` | forbidden | none | false | bool [4000] | **CORRECT** |

**Required confirmations:**

| Requirement | Verdict |
|---|---|
| Only `states` and restricted offline `next_states` scientifically allowlisted | **CONFIRMED** — the other ten carry `permitted_consumer: "none"`, `permitted_stage: "none in this pilot"` |
| `metadata_json` is binding-only | **CONFIRMED** — read limited to `public_dataset_sha256`; may not enter a derived artifact |
| All remaining fields explicitly prohibited | **CONFIRMED** — individually named, not left to a catch-all |
| Methods cannot access original `truth.npz` | **CONFIRMED** — schema per-key, `truth_information_boundary.method_direct_truth_access:false`, fail-closed req 9, gate G5, stop rule 1 |
| `next_states` cannot reach runtime action selection | **CONFIRMED** — `next_states_runtime_access:false`, `one_step_fit_overlay.runtime_access:false`, MD §3, gate G5, stop rules |
| Frozen-fit diagnostics cannot receive `next_states` | **CONFIRMED, structurally** — `state_only_overlay.exact_keys == ["states"]` and its `forbidden` clause names `next_states`; `next_states_frozen_fit_access:false`; fail-closed req 8 |
| Row alignment, finiteness, units, public binding fail closed | **CONFIRMED** — `fail_closed_requirements` items 3, 4, 5, 6 respectively, plus items 1, 2, 7, 8, 9, 10 |

The `evidence_boundary` block correctly records `arrays_opened_by_i1: false` and cites my Revision 3.1 audit plus `dataset.py` / `collector.py` declarations as the schema source — consistent with what I independently observe.

---

## 5. Residual-sigma audit

**This closes the single carry-forward condition from my Revision 3.1 confirmation audit.**

| Requirement | Markdown §5 | JSON `residual_sigma_interpretation_rule` | Verdict |
|---|---|---|---|
| Fitted hashes and `residual_sigma` recorded **before returns open** | "Before returns are opened, record every fitted-artifact hash and every available `residual_sigma` for every method and arm." | `before_returns_opened[2]`, both explicit | **MET** |
| Frozen-fit status requires byte-identical artifacts **and** bit-identical residuals | four numbered conditions: (1) byte-identical hashes, (2) bit-identical `residual_sigma`, (3) unchanged transition/value/reward fits and policy parameters, (4) only the registered adapter differs | `frozen_fit_label_requirements[4]`, same four | **MET** |
| Any difference automatically triggers `MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY` | verbatim, as a display block | `automatic_label` verbatim; `trigger: "any non-identical artifact hash or residual_sigma value"` | **MET — string identical in both** |
| No magnitude threshold, no post-hoc exception | "There is no magnitude threshold and no post-hoc 'small enough' exception." | `numeric_threshold: null`; `post_hoc_small_enough_exception: false`; `decision_type: "report-and-labelling rule; no arbitrary numerical threshold"` | **MET** |
| Per-member raw values, differences, ratios required | "raw Arm T/O values, absolute difference, ratio when the Arm O denominator is positive, and posterior entropy/sharpness changes" | `refit_reporting[5]`: Arm T raw per member, Arm O raw per member, absolute difference, ratio when denominator positive, posterior entropy/sharpness change | **MET** |
| RefPlan / BA-MCTS posterior-sharpening reported | "RefPlan and BA-MCTS must state when exact-state refitting sharpens posteriors through lower residual noise." | `refplan_bamcts_statement`, same content | **MET** |
| No causal state-representation claim from a changed-fit contrast | "The primary end-to-end comparison remains reportable but cannot identify state representation causally. A mechanism claim requires a separate matched-fit registration." | `causal_boundary`, same content | **MET** |
| Consistent across Markdown and JSON | — | — | **CONSISTENT — no divergence in any element** |

Additionally `immutable_after: "fixed before I2 and cannot change after returns are opened"`, and sealed prediction **P7** independently restates the rule and its label, so it is bound into the sealed set as well. This is a genuinely tamper-resistant construction: the rule is stated in three places that are all hash-frozen.

---

## 6. Sealed-prediction audit

**Sealed identity: `SEALED_PREDICTIONS.json` = `82a7fcb394c80e673ede04c5692db839821b06f23f752076e8cf34bb29da50c3`** — recomputed and matching, and cited identically in `I1_REGISTRATION.md` §8, `I1_REGISTRATION.json` (`sealed_predictions.sha256` and `controlling_documents.sealed_predictions.sha256`) and `I1_HASHES.sha256`.

| Requirement | Verdict |
|---|---|
| Created before I2 or results | **CONFIRMED** — `sealed_before_returns_opened: true`; file mtime 14:25 is the earliest of the six I1 artifacts; no execution namespace, derived data or result exists anywhere |
| Qualitative predictions only | **CONFIRMED** — eight statements, **zero numeric values** in any statement; `numerical_results_invented: false` |
| Matches Revision 3.1's tiered interpretation rules | **CONFIRMED** — `tiered_outcomes` maps one-to-one onto Rev3.1 §9's five tiers (ecological advantage / RefPlan sufficient / broader belief limitation / non-state limitation / non-discriminating) |
| Includes tiger's possible ecological non-discrimination | **CONFIRMED** — **P1**, with the rule that a constant-policy zero loss is not evidence of robust inference |
| Includes fox's ecological activity role | **CONFIRMED** — **P2** |
| Treats RefPlan as the decisive control | **CONFIRMED** — **P3**, "must not be grouped mechanically with the other general methods" |
| Redirects poor true-state performance to transition/reward/planner limitations | **CONFIRMED** — **P6** |
| Restricts changed-fit results to end-to-end bundle interpretation | **CONFIRMED** — **P7**, carrying the exact automatic label |
| Prohibits a causal state-representation claim | **CONFIRMED** — **P8**, requiring a separately registered matched-fit experiment |
| No undisclosed numerical prediction derived after result access | **CONFIRMED** — no numbers; and no returns exist to have been accessed |

`reporting_constraints` additionally re-binds the five synthesis rules (primary paired raw loss, no pooling, limited cross-cell synthesis, EVD separate, sigma 0.2 screening only). The sealed set is well-formed and cannot be silently widened after results without changing the hash.

---

## 7. Arm T boundary audit

| Requirement | Location | Verdict |
|---|---|---|
| Context-preserving method-specific adapters | MD §4 bullet 1; `context_preservation_required:true`; G5 | **MET** |
| Current state only at runtime | MD §4 bullet 2; `next_states_runtime_access:false`; G5 | **MET** |
| No zero-noise / zero-observation-scale construction | MD §4 bullets 6–7; `sigma_or_observation_scale_zero_prohibited:true`; `all_zero_likelihood_path_prohibited:true`; stop rule 8 | **MET** |
| Direct point-mass belief assignment for PLUS/MOOR | MD §4 bullet 8; `plus_moor_direct_belief_assignment:true` | **MET** |
| Raw-to-latent conversion through `survey_scale` | MD §3; `state_units.plus_moor`; `plus_moor_unit_contract.required_conversion` = `latent_state = raw_true_abundance / fitted_model.survey_scale`, with a **reverse check** (`selected_latent_bin * survey_scale` must match the registered raw-state discretisation) and an explicit prohibition on snapping raw truth to the latent grid | **MET — stronger than required** |
| No hidden family, threshold, reward, parameters or future information | MD §4 bullet 5; schema per-key; G5; stop rules | **MET** |
| Derived data isolated from Arm O | `arm_o_derived_access:false`; fail-closed req 10; G5 | **MET** |
| No method access to original truth archives | schema per-key; `method_direct_truth_access:false`; fail-closed req 9 | **MET** |
| OGSRL safety-budget treatment declared | MD §4 bullet 10; `ogsrl_safety_budget: "recalculate on exact abundance for primary Arm T and report safety-axis shift"` | **MET** |
| Model-fit changes reported | MD §4 bullet 11; §5 rule; `refit_reporting` | **MET** |
| Leakage tests before scientific execution | `fail_closed_i2_leakage_tests_required:true`; G5; I2 unauthorized until audited | **MET** |

RefPlan is correctly held open: `refplan_adapter_status: "unresolved I2 design"` and `frozen_fit_diagnostic: "optional secondary only; eligibility unresolved until context-preservation and byte-identity tests pass"`.

### Remaining ambiguous information routes

Three routes an I2 implementer could get wrong while still passing every literal check. None is a registration defect; all are I2 design conditions.

**(a) A byte-identical reward surrogate consuming a changed input distribution.** The frozen-fit label's four conditions are all *artifact-identity* tests. The accepted public surrogate predicts from `(previous, current, following, action, timestep, pop_id)`. Under a general-method Arm T those inputs become abundance-space values produced by a rebuilt dynamics ensemble, while the surrogate artifact stays byte-identical. An implementer could therefore preserve the surrogate hash, pass conditions 1–4 as written, and never receipt the input-distribution shift. Two mitigating facts: all four general methods are already `not eligible for primary Arm T` frozen-fit status, so this cannot corrupt the primary inference; and for PLUS/MOOR — the only frozen-fit-eligible methods — I confirmed the surrogate is fed `belief.previous_observation` / `current_observation` (preserved observations) and grid abundances times `survey_scale`, so their inputs stay in-distribution and their frozen-fit claim is safe. **Recommendation:** require I2 to receipt surrogate *input* distribution per arm alongside the artifact hash.

**(b) Per-method overlay assignment is derivable but not tabulated.** Which of the two derived schemas each method receives is inferable (PLUS/MOOR are frozen-fit ⇒ `next_states_frozen_fit_access:false` ⇒ `state_only_overlay`; the four general methods refit ⇒ `one_step_fit_overlay`) but never stated as a six-row table. The dangerous direction is already blocked by fail-closed requirement 8. **Recommendation:** I2 produces the explicit six-row mapping as its first deliverable.

**(c) Arm T collapses the `sd` feature to exactly zero.** With a point-mass state, `public_features` returns `sd = q10 = q50 = q90 = mean`. This is well-defined and consistent between the offline view and runtime (both exact-state), so it is not a leak and not a train/deploy mismatch — but it makes the arm trivially identifiable from features alone. Worth an explicit note in the I2 component-change table so it is not later mistaken for an implementation error.

---

## 8. Metrics and decision-rules audit

| Requirement | Location | Verdict |
|---|---|---|
| Paired raw true-minus-noisy loss is primary | MD §6; `metrics.primary.formula` | **MET** |
| Episode identity is the paired unit | "The sampling unit is one intact paired evaluation identity"; `sampling_unit`; bootstrap resamples intact T/O pairs | **MET** |
| Collapse, timing, unsafe occupancy, exact penalty contribution required | MD §6; `metrics.required[14]` — collapse-entry rate, first-collapse time, unsafe-occupancy steps, discounted safety-penalty contribution, minimum abundance, time below threshold, action diversity, paired-decision differences, action costs, headroom, runtime, failures | **MET** |
| Normalized loss secondary and guarded | `max(abs(...), 1.0)` denominator; `NA_DENOMINATOR_GUARD`; `ranking_permitted:false` | **MET** |
| Fox and tiger never pooled | MD §6; `cross_species_pooling:false`; `cross_cell_synthesis_only[4]`; bootstrap rule "no cross-cell resampling" | **MET** |
| EVD reported separately | MD §6; `evd_separate_reporting:true`; sealed constraint 4 | **MET** |
| Constant policies non-discriminating | MD §7; `activity_gate.new_arm_rule`; G4 | **MET** |
| Accepted Arm O parity per episode, action and event | MD §7; G2 (tiger anchor) and G3 (every Stage B Arm O rerun) | **MET** |
| Sigma 0.2 screening only | MD §1; `scope: "sigma 0.2 screening only"` | **MET** |
| Stage C absent and separately authorized | MD §1; `stage_c.included:false`, `.authorized:false` | **MET** |

The interval specification is fully reproducible and was not required by any prior audit — a genuine improvement: paired nonparametric percentile bootstrap, 100,000 resamples, NumPy Generator PCG64 seed 20260808, quantiles 0.025/0.975, `numpy.quantile(method="linear")`, intact T/O pairs only, no cross-cell resampling. This removes analyst discretion from the interval.

Gates G0–G5 are all present with concrete requirements, plus an `architecture_gate` pinning Xeon Platinum 8452Y (`xenon-8452Y`, profile `db6dc5a6…`) with `unavailable_action: "stop; do not invent tolerance or rebaseline"`.

---

## 9. Remaining blockers

**Blocking: none.**

**Non-blocking — carry into the I2 authorization:**

1. **Surrogate input-distribution receipting** (§7a). Require I2 to record, per arm and method, the input distribution fed to the frozen public surrogate, not only its artifact hash. Without this, "byte-identical artifact" can be reported while the artifact is used out of distribution.
2. **Explicit per-method overlay mapping** (§7b). Require the six-row table (method → `state_only_overlay` or `one_step_fit_overlay`) as an I2 deliverable, so overlay assignment is registered rather than inferred.
3. **Document the zero-`sd` feature consequence** (§7c) in the I2 component-change table.
4. **Cosmetic:** the freeze report's artifact table labels the five hashes "SHA-256 before checksum-manifest creation". They are simply the files' hashes — creating the manifest does not alter them, as my recomputation confirms. Reword at the next opportunity; no action required.

For the record, the two open items I carried forward from my Revision 3.1 audit are both now closed: the `residual_sigma` interpretation rule is fully frozen (§5), and the registration freezes the complete 13-key source schema, the derived-artifact schemas, both truth paths with hashes and bindings, the unit convention, the source-only recipe with full file lists, and the retention/destruction policy (§4, §2.5).

---

## 10. Smallest recommended I2 authorization

I1 **is** sufficiently complete to authorize a bounded I2 design-and-unit-test stage.

> **Authorize I2 Increment A — adapter design and unit tests against synthetic fixtures only.**

| Capability | Authorized? | Note |
|---|---|---|
| Design adapters (written design + six-row overlay mapping + component-change table) | **YES** | first deliverable |
| Implement adapters in a **new external module** outside `src/tracks/**` | **YES** | frozen tracks stay byte-identical; verified by source-only hash before and after |
| Use **synthetic fixtures only** | **YES** | a synthetic 13-key archive with the registered dtypes/shapes exercises every leakage, schema, alignment, finiteness, binding and unit test at full coverage with zero private-data exposure |
| Read truth **metadata** (key set, dtypes, shapes, `public_dataset_sha256`) | **YES** | schema/binding verification only — the level already exercised by this audit and by Revision 3.1 |
| Construct a temporary allowlisted fixture **from real truth** | **NO** | defer to Increment B; synthetic fixtures give equivalent test coverage, so real-truth exposure buys nothing at this stage |
| Run unit tests (existing suite + new leakage tests) | **YES** | under `PYTHONDONTWRITEBYTECODE=1`, `LC_ALL=C`; no scientific evaluation |
| Touch real truth **arrays** | **NO** | requires Increment B |
| Run scientific evaluations, regression, rebaseline, Slurm jobs | **NO** | I3+ and separately authorized |

Increment A must stop and return an independent-audit artifact showing: frozen-track source-only hashes unchanged; `git diff --exit-code` clean; the new module confined outside `src/tracks/**`; every G5 leakage assertion implemented and demonstrated to **fail closed** on a deliberately malformed synthetic fixture; the six-row overlay mapping; and the surrogate input-distribution receipting design.

Only after that audit should **Increment B** (real-truth extraction dry-run producing a hashed derived overlay in a separately authorized private namespace, still with no scientific evaluation) be considered.

**Authorization status of this audit.** Performed: read-only inspection of all six I1 artifacts, the controlling plan, the controlling audit and all I0/prior-audit artifacts; hash recomputation and independent reproduction of the normalized self-check; JSON validation and field-by-field cross-checking; read-only Git metadata queries (`diff --exit-code`, `status`, `rev-parse`) with no branch change, checkout, fetch or commit; schema-level verification of `truth.npz` (key set, dtypes, shapes) within the boundary already established by the Revision 3.1 audit record; read-only reads of accepted public datasets, accepted episode files, accepted config and repository source. Created exactly one file: this report. **No truth array values were extracted, copied or written; no derived dataset created; no adapter implemented; no I2 work begun; no regression, rebaseline, evaluation or Slurm job run; no network access.** Verified side-effect free: both frozen tracks unchanged on both recipes, `git diff --exit-code` returns 0, and all six I1 hashes plus every controlling-document hash still verify after all probes.

---

## Verdict

**PASS — I1 VERIFIED; READY FOR BOUNDED I2 AUTHORIZATION**
