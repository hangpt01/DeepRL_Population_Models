# Claude final read-only confirmation audit of Revision 3.1

Date: 2026-08-08 (Australia/Melbourne)
Auditor: Claude (independent read-only confirmation pass)
Audited: `DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN_REV3_1.md`, `REV3_1_CHANGELOG.md`, against `CLAUDE_REV3_READONLY_AUDIT.md` and `DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN_REV3.md`
Environment: every command run with `PYTHONDONTWRITEBYTECODE=1` and `LC_ALL=C`; no bytecode generated

Collision check: `CLAUDE_REV3_1_CONFIRMATION_AUDIT.md` did not exist before this pass. No existing file was overwritten.

---

## 1. Integrity result

### 1.1 Expected Revision 3.1 hashes

| File | Expected | Computed | Result |
|---|---|---|---|
| `DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN_REV3_1.md` | `701b4509…e959` | `701b4509f1dc04b7885891be72e526420ac5558be56bdde2b9628789398fe959` | **MATCH** |
| `REV3_1_CHANGELOG.md` | `cb44f16b…fb37` | `cb44f16b089d8ac4ba50ae02abca72c2fe33a28f3e3cddce899b4c863181fb37` | **MATCH** |

### 1.2 Revision 3 and its changelog byte-identical

| File | Value | Result |
|---|---|---|
| `…_REV3.md` | `164a536bb9a0afc4d78b42189ac119a089677bf6b83129d739ecb7b7ba103290` | **unchanged** |
| `REV3_CHANGELOG.md` | `5e1d4da1a97b5b8e4b8607b0888fb7d915e4c176520e82ce9ed45933ef266f7d` | **unchanged** |

Both match the values the Revision 3.1 changelog records at lines 5–6 as "before and after". Revision 3 was copied, not edited.

### 1.3 I0 and prior Claude audit artifacts

`sha256sum -c I0_HASHES.sha256` → **3/3 OK** (`I0_RECONNAISSANCE_REPORT.md`, `I0_READONLY_MANIFEST.json`, Revision 2 plan).

| Audit artifact | SHA-256 | Result |
|---|---|---|
| `CLAUDE_I0_READONLY_AUDIT.md` | `3e9ec9e68f705598b194c27487fb8ce1e92191e990fc001273a2b2db282b2ffc` | **unchanged** |
| `CLAUDE_REV3_READONLY_AUDIT.md` | `3139e575859badd11c4b4ef8cc3d2d058fe312598e622831aa11d6795f6167cb` | **unchanged; matches the value the Rev3.1 changelog cites at line 8** |

### 1.4 Git

| Check | Result |
|---|---|
| `git diff --exit-code` | **exit 0 — CLEAN** |
| `git diff HEAD --exit-code` | **exit 0 — CLEAN** |
| HEAD | `77cd38adb11970d56ce4c96bf84de15fac3108ac` (unchanged) |
| Branch | `e1-phase1-parity` (unchanged; no branch switch) |

**Untracked files (reported separately)** — nine, all planning/audit documents under `docs/true_noisy_state_real_methods/`; none under `src/`, `configs/`, `tests/`, `results/` or `provenance/`:

```
?? .../CLAUDE_REV3_READONLY_AUDIT.md
?? .../DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN_REV3.md
?? .../DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN_REV3_1.md
?? .../REV3_1_CHANGELOG.md
?? .../REV3_CHANGELOG.md
?? .../i0_.../CLAUDE_I0_READONLY_AUDIT.md
?? .../i0_.../I0_HASHES.sha256
?? .../i0_.../I0_READONLY_MANIFEST.json
?? .../i0_.../I0_RECONNAISSANCE_REPORT.md
```

This audit adds a tenth (`CLAUDE_REV3_1_CONFIRMATION_AUDIT.md`).

### 1.5 Frozen tracks and accepted artifacts

| Recipe | Track | Files | Hash | Result |
|---|---|---:|---|---|
| Historical (`LC_ALL=C`, all files) | ecological | 149 | `951365d7…fb01` | **unchanged** |
| Historical | general | 162 | `614524d7…e35b` | **unchanged** |
| Source-only | ecological | 52 | `2b3b8ae6d2f8ff5ffb17c4885ded9e8f1f6b3c0cb662f393186fe4b4706a884e` | **unchanged; matches Rev3.1** |
| Source-only | general | 55 | `f90cea6f28dcacb910b5e036bf9e09958715d00a2418fd0856a3d5a12856bdbd` | **unchanged; matches Rev3.1** |

Accepted artifacts all unchanged: `MATCHED_P10_144_METHOD_CELLS.csv` `74313188…`, `MATCHED_P10_144_RECEIPT.json` `a198c70f…`, `constant_reward_screen.csv` `a09d21d2…`, `m1_m15_comparison.csv` `8ac75e8b…`, `requirements.txt` `e3da5657…`, `pyproject.toml` `032bb9db…`, `provenance/frozen_tracks.sha256` `319cd428…`.

### 1.6 Change containment

The Revision 3 → 3.1 diff is **6 hunks, 123 changed lines** (+95 net). Every deletion was inspected: the status header, the four targeted passages, one authorization-ledger row and one closing paragraph. Nothing else was removed. The regions between hunks were compared directly at correct offsets and are **byte-identical**. **No Revision 3 safeguard was silently dropped** — spot-checks confirm retention of the sigma-0.2 screening status, separate Stage C authorization, no-pooling rule, EVD separate-objective rule (lines 629–630, 654–655), PLUS/MOOR point-mass assignment, `survey_scale` conversion, zero/epsilon and all-zero-likelihood prohibitions, bytecode/`git diff` safeguards, 160 × 25 checks, and the `OracleStateFilter` prohibition.

**Integrity result: PASS.**

---

## 2. Four-amendment compliance table

Assessed against Revision 3.1 directly, not against the changelog.

| # | Requirement (from my Rev3 audit) | Changelog | Rev3.1 location | Independent verdict |
|---|---|---|---|---|
| **1** | Correct the activity-record explanation | APPLIED | G4, lines 545–551 | **CORRECT** — all five required statements present; gate explicitly unchanged; independently re-verified (§2 below) |
| **2** | Name, path, hash and bind the exact-state source | APPLIED | §5 item 1, lines 225–244 | **CORRECT** — every path, hash, row count, binding and noise check independently reproduced (§3 below) |
| **3** | Strict truth-field allowlist | APPLIED | §5 item 1, lines 246–281; I2 lines 756–767 | **CORRECT** — allowlist complete against the real 13-key archive; all prohibitions present; fail-closed checks specified (§4 below) |
| **4** | Register and isolate the private-truth override | APPLIED | §5 item 1, lines 283–303; I2 756–769; ledger 867–881 | **CORRECT** — all ten required elements present (§5 below) |

All four `APPLIED` statuses are confirmed. No amendment is partially applied. The changelog's line-number citations resolve correctly.

### Amendment 1 — activity-record correction, independently re-verified

Revision 3.1 lines 545–551 state all five required elements:

| Required statement | Present | Independent re-verification |
|---|---|---|
| The same accepted `episodes.csv` files were inspected | yes — "these same accepted files and hashes" | all eight hashes in the G4 table match the accepted artifacts |
| The earlier audit displayed only the first 12 of 61 columns | yes — "displayed only the first 12 of their 61 columns" | column count confirmed = **61** |
| `action_entropy` is at column 25 | yes | confirmed at **index 25** in all eight general files |
| All eight I0 entropy values reproduce | yes | **confirmed exactly**: tiger 1.69514 / 1.13894 / 1.14487 / 1.19685; fox 1.97367 / 0.00000 / 1.49684 / 0.33010 |
| I0 activity table and G4 labels are verified | yes | fox OGSRL `0/20` decision-active count also reproduces |

**No stale "different files" explanation remains in Revision 3.1.** The phrase survives only in two legitimate places: Revision 3 line 471 (correctly unedited — Revision 3 is superseded, not revised) and Revision 3.1 changelog lines 40–41, where it is explicitly labelled "Old Revision 3 wording" in a before/after block. Both are correct provenance practice, not stale text.

Revision 3.1 also correctly preserves the operative consequence: "This provenance correction does not change the decision-activity gate."

---

## 3. Truth-path and hash verification

Independently recomputed for both registered cells.

| Check | Amur tiger × Allee × σ0.2 | Crab-eating fox × Allee × σ0.2 |
|---|---|---|
| `truth.npz` absolute path exists | **YES** | **YES** |
| `truth.npz` SHA-256 claimed | `1658f587cc2b1144362bd90182efce1bd33b65d35c47c1e9dae4320d705b8668` | `4149e293d70a59b000a3b947ce663fb3de282fb404cea857f02be00718a467e5` |
| Recomputed | identical | identical |
| Result | **MATCH** | **MATCH** |
| Paired `public.npz` SHA-256 claimed | `9e9c3a6d…f771c` | `7d1b2fc8…87e19` |
| Recomputed | **MATCH** | **MATCH** |
| Rows (`states` / `next_states` / public `observations`) | 4,000 / 4,000 / 4,000 | 4,000 / 4,000 / 4,000 |
| 4,000-row alignment | **CONFIRMED** | **CONFIRMED** |
| Embedded `public_dataset_sha256` | `7e71172af4bc95b2c31291414af53371c1ea94393e01675d269b2b7c360324c9` | `688d580f1e47a61dcad9b8a6cb96f1d62835bfbd0b835554cf31c8ab8bdd0c13` |
| Binding matches truth metadata **and** accepted public `dataset_sha256` | **MATCH** | **MATCH** |
| Claimed `sd[log(observation/state)]` | 0.19942 | 0.19655 |
| Recomputed | 0.19942 | 0.19655 |
| Result | **MATCH** (to 5 dp) | **MATCH** (to 5 dp) |

Both absolute paths in Revision 3.1's table are exact and resolve. Every hash, row count, binding and noise statistic reproduces.

**`public.npz` is correctly described.** Revision 3.1 line 228 states it "contains no true-abundance field: its 13 keys are public-only." Independently confirmed: exactly **13 keys**, and a search for any `true`/`state`/`abundance`/`latent` key returns **none**.

**No future row or evaluator-private field is exposed at action time.** Revision 3.1 lines 255–259 restrict `next_states` to the same row's offline supervised target and state that "No later row, future realized trajectory information or future state may be exposed to a decision"; I2 line 766–767 asserts this separately from the general leakage test.

One additional structural check I ran, which strengthens the case for that prohibition: **`next_states[t]` is exactly equal to `states[t+1]` within every episode** (verified with zero tolerance in both cells). So `next_states` *is* literally a future row's current state. Revision 3.1's decision to (a) permit it only as an offline one-step target paired with the same row's `states`, and (b) bar it from every runtime path, is exactly the right boundary — and the separate I2 assertion is necessary rather than redundant.

---

## 4. Allowlist and leakage verdict

### 4.1 Permitted fields

Revision 3.1 lines 248–253 permit exactly two scientific source fields:

- `states` — current true abundance;
- `next_states` — **only** for methods whose primary end-to-end Arm T requires offline model refitting, and **only** as the supervised one-step next-state target, paired only with the same row's `states` and already-public action.

This matches the required scope precisely.

### 4.2 Completeness against the real archive

I enumerated the actual `truth.npz` schema (identical in both cells): **13 keys** — `C`, `entry`, `initially_unsafe`, `metadata_json`, `next_regime`, `next_states`, `r_base`, `r_eff_true`, `regime`, `reward_true`, `safety_penalty_applied`, `states`, `theta`.

| Key | Disposition in Rev3.1 | Covered? |
|---|---|---|
| `states` | allowlisted | ✔ |
| `next_states` | conditionally allowlisted, offline target only | ✔ |
| `metadata_json` | may inspect **only** `public_dataset_sha256`; no metadata content may enter the derived artifact or method-visible process | ✔ |
| `C`, `theta`, `r_base`, `r_eff_true` | named and forbidden ("hidden Ricker/Allee parameters") | ✔ |
| `regime`, `next_regime` | named and forbidden ("latent regime identity") | ✔ |
| `reward_true` | named and forbidden ("true/evaluator reward; reward components") | ✔ |
| `entry`, `initially_unsafe`, `safety_penalty_applied` | named and forbidden | ✔ |

**Every key in the real archive is explicitly accounted for.** No field is left to the catch-all alone, though the catch-all ("every field not explicitly allowlisted") is also present.

### 4.3 Required prohibitions

| Prohibited category | Rev3.1 | Verdict |
|---|---|---|
| Hidden family / `kind` | line 261 | **MET** |
| Safety threshold (and MVP threshold) | line 262 | **MET** |
| True/evaluator reward and reward components | line 262 | **MET** |
| Hidden demographic parameters (`r_base`, `r_eff_true`, `C`, `K`, `theta`) | lines 262–263 | **MET** |
| Regime identity | lines 263–264 | **MET** |
| Future environmental randomness | line 265 | **MET** |
| Private evaluator constants | lines 265–266 | **MET** |
| Every non-allowlisted field | lines 266–267 | **MET** |
| Any part of `metadata_json.environment` | line 266 | **MET** |

### 4.4 Runtime and structural guarantees

| Requirement | Rev3.1 | Verdict |
|---|---|---|
| `next_states` never available during deployment or action selection | lines 256–257; I2 766–767 | **MET** |
| Later trajectory rows cannot leak into the current decision | lines 258–259; I2 766–767 | **MET** |
| Frozen-fit diagnostics do not receive `next_states` | line 257 | **MET** (wording nit below) |
| Schema fails closed | lines 271–276 — registered source schema, abort on unexpected/missing key | **MET** |
| Row count fails closed | line 276 — "exactly 4,000 finite rows" | **MET** |
| Ordering fails closed | lines 276–277 — "exact public row order" | **MET** |
| Units fail closed | line 277 — "verified current/next-state units" | **MET** |
| Finiteness fails closed | line 277 — "no NaN or nonfinite values" | **MET** |
| Public-dataset binding fails closed | line 278 — "exact `public_dataset_sha256` match" | **MET** |
| Direct-access boundary | lines 246–248 — methods and training pipelines must never open `truth.npz`; the external extractor is the sole reader; I2 assertion at 756–757 | **MET** |

**Wording nit (non-blocking).** Line 257 bars `next_states` from "a frozen-fit Arm T diagnostic **that does not require refitting**". Since a frozen-fit diagnostic by definition performs no refit, the qualifier is redundant and could be misread as leaving room for a "frozen-fit diagnostic that does require refitting" — a contradiction in terms. The operative intent is unambiguous from lines 252–253 and 268–270, and I2 line 766 independently bars the runtime paths. Recommend simplifying to "or any frozen-fit Arm T diagnostic" at I1, but this does not affect the information boundary.

**Allowlist and leakage verdict: SAFE.** The boundary is complete against the actual archive schema, fails closed on every structural dimension, and separates offline supervision from runtime decisions correctly.

---

## 5. Private-invariant override verdict

I confirmed the cited invariant is real: `src/tracks/general/real_ecology_benchmark/pipeline.py:432` reads *"Oracle beliefs require truth and are evaluator-only; never cache them for training."*

| Required element | Rev3.1 | Verdict |
|---|---|---|
| Explicitly acknowledges the `pipeline.py:432` invariant | lines 283–284 | **MET** — quoted in substance and located |
| Override limited to this preregistered diagnostic | lines 285–286 | **MET** |
| Limited to the two pilot cells and exact-state arms | line 286 | **MET** |
| Requires a new derived allowlisted artifact in a later authorized namespace | line 292–293 | **MET** |
| Prohibits changes to accepted datasets, truth files, public files, caches, frozen tracks | lines 290–291 | **MET** |
| Prevents methods from opening the original `truth.npz` | lines 294–295; also 246–248; I2 756–757 | **MET** |
| Prevents Arm O from accessing the derived artifact | line 296; I2 761 | **MET** |
| Requires source/derived hashes and row-level alignment | line 293 | **MET** |
| Retains the original invariant everywhere else | lines 301–302; I2 768–769 | **MET** |
| Requires leakage tests before execution | line 303; I2 753–772 | **MET** |

All ten elements present. The override is performed by "a dedicated external extraction/adapter stage outside `src/tracks/**`" (lines 288–289), which preserves the frozen-track prohibition, and a retention/destruction policy is required in I1 before construction (lines 298–299).

Correctly, **Revision 3.1 performs no override and creates no derived artifact** — it registers a future, separately authorized boundary only. Confirmed: no derived artifact exists on disk, and the authorization ledger adds "Truth-data extraction or derived artifact creation | **Not authorized**".

**Private-invariant override verdict: CORRECTLY REGISTERED AND ISOLATED.**

---

## 6. Disposition of the three remaining I1 requirements

| # | Requirement | Where stated in Rev3.1 | Correctly assigned to I1? | Blocker? |
|---|---|---|---|---|
| 1 | Register the complete source-archive schema | lines 271–272, hard-gated: "no extractor may be implemented before that registration" | **Yes** | **No** |
| 2 | Verify state units | lines 244, 277 | **Yes** | **No** |
| 3 | Preregister the `residual_sigma` interpretation rule | **not in the plan document** — only in changelog lines 173–175 | Partly | **No**, but see below |

**1 — Source-archive schema.** Correctly assigned and properly gated: extraction cannot begin before the schema is frozen. For I1's convenience, the archive is identical in both cells and contains exactly 13 keys: `C`, `entry`, `initially_unsafe`, `metadata_json`, `next_regime`, `next_states`, `r_base`, `r_eff_true`, `regime`, `reward_true`, `safety_penalty_applied`, `states`, `theta`. Registering this is a transcription task, not a research question. Not a blocker.

**2 — State units.** Correctly assigned, and the answer is already determinate from read-only evidence: `mean(observation/state)` = 1.02142 (tiger) and 1.01972 (fox) against the theoretical `exp(σ²/2)` = 1.02020, and `next_observations/next_states` gives 1.02131 / 1.01921. Truth `states` are therefore in **the same raw survey units as `observations`**, with no scale offset — so the general methods, which operate in observation units, need no conversion, while PLUS/MOOR still require the `survey_scale` conversion to latent units that Revision 3.1 already mandates at lines 207–208 and 685–687. This is a routine I1 verification with a known expected answer. Not a blocker.

**3 — `residual_sigma` interpretation rule.** This one is genuinely absent from the controlling document. Revision 3.1 requires *recording and reporting* per-member `residual_sigma` and posterior sharpness (lines 316–317 and I2 line 772), but nowhere preregisters a threshold or an explicit report-only decision. The Revision 3.1 changelog (lines 173–175) states it "remains an I1 registration requirement", but the changelog is not the controlling plan, and an I1 executor working from `…_REV3_1.md` alone would not encounter it.

Assessment: **not a blocker for the freeze**, for three reasons. I classified it non-blocking in my Revision 3 audit. It is a deliverable *of* I1 (a registration content item), not a precondition *for* I1. And the I1 registration output must itself be independently audited before I2, which is the natural and sufficient checkpoint. It must, however, be carried forward as an explicit acceptance condition — see §7.

**None of the three requires another plan revision before I1.**

---

## 7. Smallest recommended next authorization

> Authorize an **independently audited I1 registration freeze only** — no truth extraction, no derived artifact, no adapter, no namespace writes beyond the registration root itself, no code changes, no regression, no execution.

Two conditions the I1 registration must satisfy, to be checked by the I1 audit and treated as rejection criteria:

1. **The registration must contain a preregistered `residual_sigma` / model-posterior-sharpness interpretation rule** — either a numeric shift threshold above which the information-axis interpretation is qualified, or an explicit "report-only, no stop" declaration. This is the one item recorded in the Revision 3.1 changelog but absent from the plan text, and preregistering it is precisely what prevents post-hoc rationalisation once results open.
2. **The registration must freeze**: the complete 13-key source-archive schema; the derived-artifact schema; both absolute `truth.npz` paths with the two verified hashes and embedded bindings; the unit convention (truth `states` in raw survey units, `survey_scale` conversion required for PLUS/MOOR only); the source-only track recipe, file list and hashes; and the derived-artifact retention/destruction policy.

Optional tidy-up at the same time: simplify line 257 to "or any frozen-fit Arm T diagnostic" (§4.4 nit).

I2, truth extraction, derived-artifact construction, adapter implementation, regression, rebaseline, Stage B and Stage C remain outside scope and unauthorized regardless of this verdict.

**Authorization status of this audit.** Performed: read-only inspection of Revision 3.1, its changelog, Revision 3, both prior Claude audits and the I0 artifacts; hash recomputation; read-only Git metadata queries (`diff --exit-code`, `status`, `rev-parse`) with no branch change, checkout, fetch or commit; read-only reads of both `truth.npz` files, both accepted `public.npz` files, the eight accepted general episode files, and repository source. Created exactly one file: this report. **No truth-state dataset was extracted, derived, copied or written**; no plan, changelog, audit, source file, test, configuration or registration was modified; no namespace created; no adapter implemented; no evaluation, regression, rebaseline or Slurm job run; no network access. Verified side-effect free: both frozen-track hashes and both source-only hashes unchanged after all probes, `git diff --exit-code` still returns 0, and all prior artifact hashes still verify.

---

## Verdict

**PASS — REVISION 3.1 READY FOR I1 REGISTRATION FREEZE**
