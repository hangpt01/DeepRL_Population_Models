# Claude final independent read-only audit of Revision 3

Date: 2026-08-08 (Australia/Melbourne)
Auditor: Claude (independent read-only pass)
Audited: `DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN_REV3.md`, `REV3_CHANGELOG.md`, and the prior I0 / Claude-audit artifacts
Environment: all commands run with `PYTHONDONTWRITEBYTECODE=1` and `LC_ALL=C`; no bytecode generated

Collision check: `CLAUDE_REV3_READONLY_AUDIT.md` did not exist before this pass. No existing file was overwritten.

---

## 1. Executive verdict

**Revision 3 is a faithful, high-quality application of A1–A10. All ten amendments are present, and nine are fully correct. One material specification gap prevents an I1 registration freeze.**

Five conclusions:

1. **Integrity is clean.** Both expected hashes match exactly. Revision 2, the entire I0 directory, my prior audit, both frozen tracks, all accepted artifacts, `requirements.txt` and `pyproject.toml` are unchanged. `git diff --exit-code` and `git diff HEAD --exit-code` both return 0. Only six untracked planning artifacts exist, all inside `docs/true_noisy_state_real_methods/`.

2. **The scientific framing is now correct and, in one respect, better than my own audit.** Revision 3 draws all seven required distinctions cleanly, no longer claims OGSRL/BA-MCTS/EVD escape sigma-dependent preprocessing, and explicitly refuses to equate the upstream `BeliefState` with PLUS/MOOR beliefs. Section 3's closing paragraph is an accurate, well-sourced statement of the corrected position.

3. **I must correct my own prior audit.** In `CLAUDE_I0_READONLY_AUDIT.md` §8 I reported a "reproducibility gap" — that the accepted general-track `episodes.csv` files contain no action column and I0's activity entropies could not be reproduced. **That was my error.** Those files have 61 columns including `action_entropy` at column 25; my script printed only the first 12 and I concluded wrongly. Revision 3's G4 provenance note is right, and I have now reproduced all eight I0 entropy figures exactly (tiger 1.69514 / 1.13894 / 1.14487 / 1.19685; fox 1.97367 / 0.00000 / 1.49684 / 0.33010). I0's activity table is fully verified. One detail in Revision 3 and the changelog is still inaccurate: they state I inspected "different"/"shorter" files. The hashes are identical to the eight files Revision 3 lists — same files, misread by me.

4. **The one blocking gap: Arm T's true-abundance source is unspecified.** Revision 3 §5 item 1 requires a derived exact-state view whose "current/next state fields come from the pinned true-abundance fields". The accepted `public.npz` contains **no true-abundance field** — its 13 keys are all public. The true states are in a separate artifact, `truth.npz`, which Revision 3 **never names, never paths, never hashes, and never bounds**. I located and verified it: it exists for both pilot cells, is row-aligned (4,000 rows), is cryptographically linked to the accepted dataset via `public_dataset_sha256`, and reproduces σ = 0.2 exactly (`sd[log(obs/state)]` = 0.19942 tiger, 0.19655 fox). So Arm T **is feasible** — but the file is a concentrated leakage bundle containing every single item Revision 3 forbids: the hidden family label (`kind = "allee"`), the private safety threshold (25.0 / 10.25), evaluator-only reward (`reward_true`), realized future process draws (`next_states`), and hidden `r`/`C`/`theta`/`regime`. An I2 implementer following Revision 3 literally would open exactly that file with no registered field allowlist.

5. **Everything else passes.** All twelve gate requirements, all seven provenance safeguards and all four compute separations are satisfied. Revision 3 explicitly states the 20.65 core-hour figure excludes adapter development and refit work.

The required fix is small and documentation-only: name the artifact, hash it, register a field-level allowlist, forbid `next_states` from reaching the online adapter, and record the deliberate override of the existing `pipeline.py:432` invariant. That is a Revision 3.1 amendment, not a redesign.

---

## 2. Integrity verification

### 2.1 Expected Revision 3 hashes

| File | Expected | Computed | Result |
|---|---|---|---|
| `DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN_REV3.md` | `164a536b…3290` | `164a536bb9a0afc4d78b42189ac119a089677bf6b83129d739ecb7b7ba103290` | **MATCH** |
| `REV3_CHANGELOG.md` | `5e1d4da1…6f7d` | `5e1d4da1a97b5b8e4b8607b0888fb7d915e4c176520e82ce9ed45933ef266f7d` | **MATCH** |

### 2.2 Prior artifacts unchanged

`sha256sum -c I0_HASHES.sha256` → all three **OK**:

- `I0_RECONNAISSANCE_REPORT.md` OK
- `I0_READONLY_MANIFEST.json` OK
- `DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN.md` (Revision 2) OK — **Revision 2 was not edited**; Revision 3 is a separate file

`CLAUDE_I0_READONLY_AUDIT.md` = `3e9ec9e68f705598b194c27487fb8ce1e92191e990fc001273a2b2db282b2ffc`, which **matches the value the changelog records at line 9**. My prior audit is unmodified.

### 2.3 Git state

| Check | Result |
|---|---|
| `git diff --exit-code` | **exit 0 — clean** |
| `git diff HEAD --exit-code` | **exit 0 — clean** |
| HEAD | `77cd38adb11970d56ce4c96bf84de15fac3108ac` (unchanged) |
| Branch | `e1-phase1-parity` (unchanged; no branch switch) |

**Untracked files (reported separately, as required)** — six, all planning artifacts, none under `src/`, `configs/`, `tests/`, `results/` or `provenance/`:

```
?? docs/true_noisy_state_real_methods/DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN_REV3.md
?? docs/true_noisy_state_real_methods/REV3_CHANGELOG.md
?? docs/true_noisy_state_real_methods/i0_.../CLAUDE_I0_READONLY_AUDIT.md
?? docs/true_noisy_state_real_methods/i0_.../I0_HASHES.sha256
?? docs/true_noisy_state_real_methods/i0_.../I0_READONLY_MANIFEST.json
?? docs/true_noisy_state_real_methods/i0_.../I0_RECONNAISSANCE_REPORT.md
```

(This audit adds a seventh: `CLAUDE_REV3_READONLY_AUDIT.md`.)

### 2.4 Frozen tracks — historical and source-only

| Recipe | Track | Files | Hash | Result |
|---|---|---:|---|---|
| Historical (`LC_ALL=C`, all files) | ecological | 149 | `951365d7…fb01` | **matches I0 and Rev3 line 341** |
| Historical | general | 162 | `614524d7…e35b` | **matches I0 and Rev3 line 342** |
| Source-only (excl. `__pycache__`, `.pytest_cache`, `.pyc`, `.pyo`) | ecological | 52 | `2b3b8ae6d2f8ff5ffb17c4885ded9e8f1f6b3c0cb662f393186fe4b4706a884e` | **matches Rev3 line 346** |
| Source-only | general | 55 | `f90cea6f28dcacb910b5e036bf9e09958715d00a2418fd0856a3d5a12856bdbd` | **matches Rev3 line 347** |

Bytecode count independently confirmed: **204** `.pyc` files across the two tracks, exactly as Revision 3 line 343 states. Both source-only hashes reproduce under Revision 3's stated exclusion set and under the narrower `__pycache__`-only exclusion — the two agree, confirming no stray `.pyc`/`.pyo` outside `__pycache__`.

### 2.5 Accepted artifacts unmodified

| Artifact | Hash | Result |
|---|---|---|
| `results/accepted/MATCHED_P10_144_METHOD_CELLS.csv` | `74313188…dc1c` | unchanged |
| `results/accepted/MATCHED_P10_144_RECEIPT.json` | `a198c70f…3163` | unchanged; **corrected path resolves** |
| `results/followups/reward_screen/constant_reward_screen.csv` | `a09d21d2…ed14` | unchanged |
| `results/diagnostic_replay/m1_m15_comparison.csv` | `8ac7e15b…47b9` (`8ac7…`) | unchanged |
| `requirements.txt` | `e3da5657…a2ea` | unchanged; **corrected path resolves** |
| `pyproject.toml` | `032bb9db…8fca` | unchanged |
| `provenance/frozen_tracks.sha256` | `319cd428…6ffc` | unchanged |

**No accepted artifact was modified.** Both A10 path corrections resolve to real files carrying the originally recorded content hashes.

---

## 3. A1–A10 compliance table

Assessed against Revision 3 directly, not against the changelog.

| # | Requirement | Changelog says | Rev3 location | Independent verdict |
|---|---|---|---|---|
| **A1** | Rewrite the sigma record-verification row to `PARTLY CONTRADICTED` with sources | APPLIED | §2, line 70 | **CORRECT** — full text present; `pipeline.py:246-259`, `beliefs.py:597-599`, `beliefs.py:625`, ESS 1.00, log-RMSE 0.196–0.202 all retained and all independently re-verified |
| **A2** | Replace the OGSRL/BA-MCTS/EVD role cells and add the discriminating-property footnote | APPLIED | §3, lines 94–96 (cells), 101–114 (footnote) | **CORRECT** — all three cells rewritten verbatim in substance; footnote present and expanded; `faithful_pomdp.py:186-207` cited |
| **A3** | Replace the motivating-hypothesis bullet | APPLIED | §1, lines 51–53 | **CORRECT** — exact substantive replacement |
| **A4** | Replace the Arm O sigma sentence | APPLIED | §5 Arm O, lines 276–278 | **CORRECT** — exact substantive replacement |
| **A5** | Insert the five-point mandatory Arm T adapter subsection | APPLIED | §5, lines 223–249; component table 257–264 | **INCOMPLETE** — items 1, 3, 4, 5 fully correct; **item 2's dataset decision is registered but its true-abundance source is unspecified** (see §5.2) |
| **A6** | Make G1 executable; stop claiming the nonexistent fix | APPLIED | §7 G1, lines 375–399 | **CORRECT** — and stronger than requested: adds explicit 160×25, horizon-25, guard-non-entry and one-transition-margin checks |
| **A7** | Delete the fix-delta rebaseline branch | APPLIED | §7 G2, lines 417–420 | **CORRECT** — branch removed; also removed from G3 (line 435) and the architecture-unavailable fallback replaced with a hard stop (411–412) |
| **A8** | Correct the live-fix provenance row | APPLIED | §2, line 76 | **CORRECT** — exact substantive replacement; the single "branch-isolated" occurrence is inside the corrective sentence |
| **A9** | Close Q3/Q4, add Q7/Q8 | APPLIED | §14, lines 803–819 | **CORRECT** — Q3 and Q4 closed with evidence; Q7 and Q8 added **and answered** as registered decisions |
| **A10** | Fix two paths, amend the hash recipe, cite the action-entropy source | APPLIED | §2 lines 78–80; §6.1 lines 314–349; §7 G4 lines 455–474 | **CORRECT** — and it corrected *my* error (see §3.1) |

**Stale-text sweep.** Zero occurrences remain of `does not use sigma`, `no sigma use`, `do not use it`, `EXPLAINED FIX DELTA`, `REBASELINE REQUIRED`, or `receive the constant in context`. No Revision 2 statement survives that contradicts an applied amendment.

**Changelog accuracy.** Every `APPLIED` status is independently confirmed. The changelog's own line-number citations resolve correctly. One application note is factually wrong (§3.1). The changelog does not overstate: it claims no amendment was partially applied, and on my reading nine are complete and one (A5) is complete in structure but incomplete in specification — a distinction the changelog did not draw.

### 3.1 Correction to my own prior audit (A10)

My prior audit stated:

> "The per-episode action counts and entropies in I0 §7 for the four *general* methods cannot be reproduced from the artifacts I0 lists: the accepted `episodes.csv` files are episode-level summaries with no action column."

**This was wrong.** Verified now:

- All eight accepted general-track `episodes.csv` files have **61 columns**, with `action_entropy` at **column 25** — exactly as Revision 3 G4 states.
- Their SHA-256 values match Revision 3's table and I0's manifest exactly.
- All eight I0 entropy means reproduce to 5 decimal places, and the fox OGSRL `0/20` activity count reproduces (`action_entropy > 0` in 0 of 20 episodes).

I0's activity table is therefore **fully verified**, and the "reproducibility gap" I recorded should be struck. Revision 3's corrective action is right.

One residual inaccuracy to fix for the record: Revision 3 line 471 and changelog line 181 state that Claude inspected "shorter, distinct source-repository summaries" that are "different files". They were the **same files** — the ecological `episodes.csv` files also carry 61 columns and `action_entropy`. The cause was a display truncation in my script, not a different artifact. The corrective content stands; only the explanation needs a one-line fix.

---

## 4. Scientific-framing audit

Revision 3 distinguishes all seven required concepts. Verified against the text:

| Concept | Where | Verdict |
|---|---|---|
| Raw noisy observations | §3 line 111 ("receipt of noisy observations"); §5 Arm O 275–278 | **distinct** |
| Sigma-dependent preprocessing | §2 line 70; §3 lines 94–96, 101–107 | **distinct** |
| `PublicObservationFilter` outputs | §3 lines 104–107 (memoryless, deletes action, uniform weights, ESS 1.00, log-RMSE 0.196–0.202) | **distinct** |
| Recursive calibrated latent-abundance belief | §3 lines 102–103 (`faithful_pomdp.py:186-207`); §9 lines 592–594 | **distinct** |
| Model/value/support uncertainty | §1 lines 51–53; §3 uncertainty column; RefPlan "over **model identity**, not over abundance" (line 108) | **distinct** |
| End-to-end information-regime effect | §1 lines 38–41; §5 lines 266–271; §5 line 286 | **distinct** |
| Narrower frozen-fit diagnostic | §1 lines 43–45; §5 lines 268–270; §5 lines 287–289 | **distinct** |

**Required negative check 1 — does Rev3 still say OGSRL/BA-MCTS/EVD receive no sigma-dependent preprocessing?** **No.** Line 101 states the opposite outright: "All six methods' inputs depend on `observation_noise_sigma`." Each of the three role cells now names the sigma pathway. Zero stale occurrences of the old phrasing.

**Required negative check 2 — does Rev3 imply the upstream `BeliefState` is equivalent to PLUS/MOOR beliefs?** **No.** Lines 101–108 make the non-equivalence the organising point: "The discriminating property is **not** sigma consumption but the kind of state estimate maintained." Lines 110–114 enumerate five separately-tracked properties (noisy observations; upstream sigma-dependent `BeliefState`; temporal/action history; actual denoising; calibrated latent-abundance belief). §9's tiered interpretation rules preserve the distinction operationally: line 586 warns the result "is not evidence that all sigma-dependent preprocessing is equivalent", and lines 592–594 name "sigma-scaled preprocessing without denoising" as the falsifiable mechanism.

The framing is accurate and, on the RefPlan model-identity point, more precise than Revision 2.

---

## 5. Arm T audit

### 5.1 Ten required properties

| # | Requirement | Rev3 location | Verdict |
|---|---|---|---|
| 1 | Prohibits sigma / observation-scale collapse | 200–202; 246–247; I2 assertion 683–684; stop rule 612–613 | **MET** (four independent places) |
| 2 | Prohibits the all-zero-likelihood route | 202–204; negative test 690–691; stop rule 612–613 | **MET** — I2 requires a fail-closed negative test |
| 3 | Preserves observation/action history for general methods | 213–221; item 2 at 231–234; table "Preserve" cells 261–264; I2 assertion 676 | **MET** — `OracleStateFilter.set_true_state` prohibited by name with the zeroed-context reason |
| 4 | Requires raw→latent conversion via `survey_scale` | 207–208; I2 assertion 685–687 | **MET** |
| 5 | Defines direct point-mass assignment for PLUS/MOOR | 208–210; table 259–260; I2 687 | **MET** — includes "no double Bayesian update" |
| 6 | Treats RefPlan's adapter as unresolved, not assumed | 268–270 ("optional, explicitly secondary … after context-preservation tests"); §14 Q1 open at 801–802, 821–822 | **MET** |
| 7 | Acknowledges OGSRL safety-budget calibration | item 3 at 235–240; table 262; I2 679–680 | **MET** — recalibration chosen, safety-axis shift declared reportable |
| 8 | Acknowledges residual-noise / posterior-sharpness change | item 4 at 241–244; table columns 261–263; I2 681–682 | **MET** |
| 9 | Labels the primary comparison an end-to-end method bundle | 266–271; §1 38–41; §8.3 531–532 | **MET** |
| 10 | Frozen-fit diagnostic only where artifact identity is demonstrated | 268–270; 288–289; item 5 at 245–249 | **MET** — 39-file PLUS / 9-file MOOR hash proof required in I2 |

All ten are satisfied. The component-change table (lines 257–264) is a genuine improvement on what A5 asked for: eleven columns covering current state, history, offline view, dynamics, residual noise, posterior sharpness, safety budget, reward surrogate, policy training and online updating, for all six methods.

### 5.2 The blocking gap — Arm T's true-abundance source is unspecified

Revision 3 line 228 requires that general methods "receive a derived exact-state view whose current/next state fields come from the **pinned true-abundance fields**".

**The accepted `public.npz` has no true-abundance field.** Its 13 keys are `action_costs, actions, costs, dones, episode_id, metadata_json, next_observations, observations, pop_ids, rewards, terminated, timestep, truncated`. A search for any `true`/`state`/`latent`/`abundance` key returns nothing.

The true states live in a separate artifact that Revision 3 **never mentions**. I located and verified it:

```
<GEN>/quarantine/private/regime_hidden/reward_safe/<species>/allee/sigma_0p2/truth.npz
```

| Property | Tiger | Fox |
|---|---|---|
| SHA-256 | `1658f587cc2b1144362bd90182efce1bd33b65d35c47c1e9dae4320d705b8668` | `4149e293d70a59b000a3b947ce663fb3de282fb404cea857f02be00718a467e5` |
| Rows | 4,000 | 4,000 |
| `public_dataset_sha256` link | `7e71172a…0324c9` — **matches accepted** | `688d580f…8bbdd0c13` — **matches accepted** |
| `sd[log(obs/state)]` | 0.19942 | 0.19655 |

The σ recovery (0.199 / 0.197 against the registered 0.200) confirms these are the true states behind the accepted observations, and the embedded `public_dataset_sha256` binds them cryptographically to the exact accepted datasets. **Arm T is therefore feasible** — the design is sound and the data exists. But `truth.npz` appears **zero times** in Revision 3.

This matters because of what else is in the file.

### 5.3 Remaining leakage routes

`truth.npz` is a single artifact containing **every category Revision 3 prohibits**:

| Prohibited category (Rev3 lines 195–197, 219–221) | Present in `truth.npz`? | Field |
|---|---|---|
| Hidden family label | **YES** | `metadata_json.environment.kind = "allee"` |
| Private safety threshold | **YES** | `environment.safety_threshold` = 25.0 (tiger) / 10.25 (fox); also `mvp_threshold`, `safety_fraction`, `safety_penalty_mode` |
| Evaluator-only reward constants | **YES** | `reward_true` (4,000 distinct values — the true objective); `safety_penalty_applied` |
| Evaluator-only information beyond current abundance | **YES** | `next_states`, `entry`, `initially_unsafe`, `regime`, `next_regime` |
| Future randomness | **YES** | `next_states` is the realized next true state, i.e. the realized future process draw |
| Hidden `r`, `C`, `K`, `theta` | **YES** | `r_base`, `r_eff_true`, `C` (Allee depensation threshold, 160 unique), `theta`, `environment.K_base`, `allee_C_frac_*`, `r_base_*`, `regime_threshold_*` |

Two further consequences:

1. **Revision 3's design deliberately overrides an existing in-code invariant.** `pipeline.py:432` reads: *"Oracle beliefs require truth and are evaluator-only; never cache them for training."* Revision 3's primary general-method Arm T requires precisely a truth-derived training view. That is legitimate — it is the experiment — but it inverts a documented safety rule and must be registered as a conscious, bounded override rather than silently contradicted.

2. **`next_states` is dual-use.** For the *offline* dataset view it is the legitimate transition target (the exact-state analogue of `next_observations`). For the *online* adapter it is future information. Revision 3's component table correctly specifies online delivery as current abundance only (lines 259–264), but nowhere forbids `next_states` from reaching the online path. I2 needs an explicit assertion.

**No other leakage route was found.** The PLUS/MOOR direct-assignment convention, the context-preserving general adapters, the prohibition on `OracleStateFilter`, and the sigma/observation-scale prohibitions are all correctly bounded. The `costs` field already present in the accepted public dataset is pre-existing and arm-independent.

### 5.4 Required Arm T amendments

Add to §5 item 1 (documentation only):

> The exact-abundance source is `<GEN>/quarantine/private/regime_hidden/reward_safe/<species>/allee/sigma_0p2/truth.npz`, SHA-256 `1658f587cc2b1144362bd90182efce1bd33b65d35c47c1e9dae4320d705b8668` (tiger) and `4149e293d70a59b000a3b947ce663fb3de282fb404cea857f02be00718a467e5` (fox). Both carry `public_dataset_sha256` matching the accepted datasets; I1 must verify that binding before use.
>
> **Field allowlist.** The Arm T adapter may read **only** `states` and `next_states`. Reading `r_base`, `r_eff_true`, `C`, `theta`, `regime`, `next_regime`, `entry`, `reward_true`, `initially_unsafe`, `safety_penalty_applied` or any part of `metadata_json.environment` is prohibited. `metadata_json` may be opened solely to verify `public_dataset_sha256`. I2 must add a test that fails closed if any other field is dereferenced.
>
> **`next_states` scope.** `next_states` may be used only to build the offline transition target. It must never reach an online adapter, deployment feature or planning root. I2 must assert this separately from the general leakage test.
>
> **Registered invariant override.** `pipeline.py:432` states that truth must never be cached for training. Arm T deliberately overrides this for the derived exact-state view only, outside `src/tracks/**`, under the field allowlist above. Record the override explicitly in the I1 registration.

---

## 6. Gate audit

| Requirement | Rev3 location | Verdict |
|---|---|---|
| The nonexistent OGSRL fix is not claimed as available | G1 377–384; G0 367; G2 417–420; §14 Q3 805–807 | **MET** — stated four times; G0 requires recording that it "is not present locally and must not be claimed" |
| The revised gate uses the verified 160 × 25 datasets and horizon 25 | G0 357–360; G1 386–392 | **MET** — G1 verifies both datasets at 160 × 25, `ogsrl_cost_horizon` = 25, and that `25 < 25` is false |
| Stops if a future dataset contains a shorter episode | G1 392, 394–398 | **MET** — names the exactly-one-transition margin and requires a stop before OGSRL fitting until a separately reviewed fix with the three named tests exists |
| Accepted Arm O parity per episode, action sequence, event count | G2 403–408; G3 433–435 | **MET** — three-level comparison at G2 and per-rerun comparison at G3 |
| Fox is the ecological activity cell | 151; 606; §9 592–594 | **MET** |
| Tiger is the general-method activity cell | 150 | **MET** — also the OGSRL regression anchor |
| Constant-policy methods cannot be credited with state robustness | G4 439–441, 451–453; §9 606 | **MET** — "A zero true-to-noisy loss from a constant policy is not evidence"; non-discriminating label mandated; the reference constants (tiger 10, fox 1) match my verified fixed-action screen |
| Sigma 0.2 is screening only | 157–158; 171 | **MET** |
| Stage C remains separately authorized and conditional | 160–171; 726–729; 784; 812–813 | **MET** — four independent statements, including the both-cells non-discrimination stop rule |
| Raw paired loss is primary within a cell | §8.1 490–497; 516 | **MET** |
| Fox and tiger losses are not pooled | §8.1 499–503 | **MET** — "Do not pool, average or rank raw losses across species"; cross-cell synthesis restricted to preregistered qualitative features |
| EVD cannot determine the main decision rule | §8.4 552–553; §9 577–578 | **MET** — "cannot break a tie"; "reported separately … and cannot determine this rule" |

**All twelve met.** Three gate strengthenings beyond what A6/A7 required are worth noting: G2 now stops outright if the pinned 8452Y architecture is unavailable rather than permitting a new baseline (411–412); G3's inherited fix-delta exception is removed (435); and G1 adds the margin-of-one-transition check that my audit raised only as commentary.

---

## 7. Provenance audit

| Requirement | Rev3 location | Verdict |
|---|---|---|
| `PYTHONDONTWRITEBYTECODE=1` | §6.1 320; G0 370 | **MET** — mandated for "every future reconnaissance command, test, adapter check, regression and scientific task" |
| `LC_ALL=C` | §6.1 321, 338; G0 370 | **MET** |
| Source-only hashes excluding generated bytecode | §6.1 328, 343–347 | **MET** — exclusion set and both hashes recorded; **independently reproduced** (§2.4) |
| Historical full-track hashes retained for provenance only | §6.1 332–333, 341–343 | **MET** — "never use them as the sole integrity check" |
| `git diff --exit-code` | §6.1 326; G0 371 | **MET** |
| Explicit untracked-file reporting | §6.1 326; G0 371 | **MET** |
| Corrected manifest paths without changing verified content hashes | §2 lines 78–79; G0 368–369 | **MET** — both corrected paths carry the original hashes, and both resolve (§2.5) |

**All seven met.** §6.1 adds two safeguards beyond A10: treat accepted directories read-only where practical (327), and never treat an import-generated bytecode change as evidence that source changed (330–331). Line 349 requires I1 to freeze the exact source-only recipe, file list and hashes — the right place for it.

---

## 8. Compute audit

| Separation required | Rev3 treatment | Verdict |
|---|---|---|
| Frozen-fit PLUS | 630–631: eight frozen fits with runtime-belief-only intervention, ≈ 4.25 h per method-cell wall dominance | **SEPARATED** |
| PLUS requiring refitting/rebuilding | 638–640: cold-fit contingency ≈ 2.41 h per affected cell (2.15–2.53), explicitly flagged as not measuring replanning or adapter work | **SEPARATED** |
| Known planning compute | 623–627: 18,588 s ≈ 5.16 core-h per cell-arm; 20.65 core-h per sigma; 41.31 core-h both sigmas | **SEPARATED** |
| Unknown adapter/refit costs | 635–643: excluded and named | **SEPARATED** |

**Required negative check — does Rev3 claim 20.65 core-hours covers unmeasured refitting or development?** **No.** Line 635 states the figure "**is an existing-fit execution estimate only**", and lines 636–638 enumerate what it excludes: adapter design, implementation and audit; all unmeasured exact-state rebuild work for the general methods; and any PLUS refitting or replanning. Lines 640–643 add a conditional stop: if any PLUS fit or planning artifact would change, stop, measure the actual path, revise, and re-authorize.

Arithmetic re-verified: 15,314.56 + 1,940.64 + 52.42 + 430.15 + 847.26 + 2.5 = 18,587.53 s; ×4 = 74,350.12 s = 20.6528 core-h; ×2 = 41.31 core-h. All consistent.

One observation, not a defect: the per-task means underlying 20.65 core-hours were measured on the accepted **noisy** path. Arm T rebuilds dynamics ensembles, retrains the OGSRL actor/guardian and refits EVD's 20 Q members, so the general-method Arm T tasks will not match their Arm O receipts. Revision 3 already excludes this as "unmeasured exact-state rebuild work", so the estimate is not overclaimed — but I1 should expect the general-method Arm T column to be the least predictable part of the budget.

---

## 9. Remaining blockers

**Blocking (must be fixed before an I1 registration freeze):**

1. **Arm T's true-abundance source is unspecified.** Revision 3 requires "pinned true-abundance fields" that do not exist in `public.npz`. The real source is `truth.npz`, which Revision 3 never names, paths or hashes — and which contains every prohibited category: hidden family (`kind = "allee"`), private safety threshold, `reward_true`, realized future draws (`next_states`), and hidden `r`/`C`/`theta`/`regime`. Fix: adopt the §5.4 amendment — name and hash the artifact, impose a two-field allowlist (`states`, `next_states`), scope `next_states` to offline use only, and register the deliberate override of `pipeline.py:432`. This is documentation-only and does not change the design.

**Non-blocking (fix for the record):**

2. **The "different files" explanation is inaccurate.** Revision 3 line 471 and changelog line 181 attribute my prior error to inspecting "different"/"shorter" files. The hashes are identical to the eight files Revision 3 lists; all twelve accepted episode files (ecological and general) carry 61 columns and `action_entropy`. Replace with: "Claude's prior audit truncated the column listing and reported the column absent; the files are the same and the column is present at index 25." My own audit's §8 "reproducibility gap" paragraph should be treated as withdrawn — **I0's activity table is fully verified** (§3.1).

3. **`residual_sigma` recording is required but has no acceptance criterion.** §5 item 4 and I2 line 681 require recording per-member `residual_sigma` and posterior sharpness by arm, but nothing states what magnitude of shift would invalidate the information-axis interpretation. I1 should preregister a threshold or an explicit "report-only, no stop" decision, so the confound cannot be rationalised after results open.

4. **A5's "belief-only" option was silently dropped.** My A5 item 2 offered two options; Revision 3 chose a third (same rows, replaced state fields). That is a defensible and arguably better choice, but the component table's "Change: derived exact-state view" for four methods means this is *not* the belief-only arm whose unreachability argument G1 relies on. The G1 reasoning still holds — episode boundaries and row identities are preserved, so `ogsrl.py:563` remains unreachable — but Revision 3 should say plainly that the offline *view* changes while the episode *structure* does not, so a later reader does not mistake "no regeneration" for "no offline change".

**Explicitly not blockers:** the missing OGSRL short-episode fix (correctly handled and made irrelevant by the no-regeneration decision); the frozen-track bytecode fragility (correctly superseded by source-only hashes); and the two manifest path errors (corrected, with content hashes preserved).

---

## 10. Recommended next authorization

**Readiness assessment:**

| Stage | Ready? | Reason |
|---|---|---|
| I1 registration freeze only | **Not yet** | Blocker 1: freezing a registration that says "pinned true-abundance fields" without naming `truth.npz`, hashing it, or bounding its fields would freeze an underspecified — and leakage-exposed — contract. The fix is one subsection. |
| I2 adapter design | **No** | I2 must not precede a corrected, frozen I1 registration. |
| Neither | — | Too harsh: nine of ten amendments are fully correct, the framing is sound, all gates and provenance safeguards pass, and the one gap is documentation-only with the data already verified to exist. |

**Smallest next authorization boundary:**

> Authorize a **Revision 3.1 documentation-only amendment**, with no code, no registration namespace, no adapter and no execution. Scope limited to:
>
> 1. §5 item 1 — insert the `truth.npz` path, both SHA-256 values, the `public_dataset_sha256` verification step, the `states`/`next_states` field allowlist, the offline-only scope for `next_states`, and the registered `pipeline.py:432` override (text supplied in §5.4);
> 2. §7 G4 line 471 and changelog line 181 — correct the "different files" explanation (text supplied in §9.2);
> 3. §5 item 4 / §11 I2 — add a preregistered `residual_sigma` shift threshold or an explicit report-only decision;
> 4. §5 item 1 — state that the offline dataset *view* changes while row identities and episode structure do not.
>
> Then a short read-only confirmation pass on the four edits. If they verify, Revision 3.1 is ready for **I1 registration freeze only** — I1 still requiring its own explicit authorization.

I1, I2, regression, rebaseline, Stage B and Stage C remain unauthorized regardless of this verdict.

**Authorization status of this audit.** Performed: read-only inspection of Revision 3, the changelog, the I0 artifacts and my prior audit; hash recomputation; read-only Git metadata queries (`diff --exit-code`, `status`, `rev-parse`) with no branch change, checkout, fetch or commit; read-only reads of accepted datasets, `truth.npz`, accepted episode files and result tables; read-only source inspection. All commands ran under `PYTHONDONTWRITEBYTECODE=1` and `LC_ALL=C`; both frozen-track hashes were unchanged afterwards and `git diff --exit-code` still returns 0. Created exactly one file: this report. No plan, changelog, code, configuration, test, registration, accepted output or frozen track was modified; no adapter implemented; no evaluation, regression, rebaseline or Slurm job run; no network access.

---

## Verdict

**REVISE — REVISION 3 REQUIRES SPECIFIED CHANGES**
