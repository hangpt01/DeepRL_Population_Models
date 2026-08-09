# Claude confirmation audit — I1 zero-SD rule addendum

Date: 2026-08-08 (Australia/Melbourne)
Auditor: Claude (read-only confirmation pass)
Environment: every command run with `LC_ALL=C` and `PYTHONDONTWRITEBYTECODE=1`; no bytecode written, no truth array or dataset opened, no repository mutation of any kind.

Collision check: `CLAUDE_I1_ZERO_SD_ADDENDUM_CONFIRMATION_AUDIT.md` did not exist. No file was overwritten.

---

## 1. Executive verdict

**The zero-SD rule itself is verified, faithful to the controlling memo, and ready. One factual disclosure in the freeze report is wrong and must be corrected before it is relied upon.**

Five conclusions:

1. **All integrity checks pass.** Four addendum artifacts hash as recorded; the normalized manifest self-check reproduces from first principles and uses the same convention as the original I1 manifest; all six original I1 artifacts, Revision 3.1, both prior Claude audits and the memo are byte-identical; sealed predictions are unamended; no tracked file is modified or staged.

2. **Memo-to-addendum fidelity is complete.** Every one of the ten required freezes is present in both the Markdown and the JSON, and the two agree on all thirteen normative categories with **zero mismatches**. Critically, the addendum does **not** claim the SD is bit-exactly zero before assignment — it states the opposite explicitly and forbids `sd == 0.0` as a primary rule.

3. **Branch semantics, RefPlan disclosure and the RNG invariant are all correctly frozen**, including both halves of the RNG fact and the requirement that future tests check returned-value exactness *and* call/state parity.

4. **The addendum closes the exact blocker Codex reported.** Fitted scale, near-zero rule, offline/runtime parity and receipt requirements are each pinned to an exact, non-discretionary value. An I2 Increment A implementer would not have to invent any scientific or preprocessing rule.

5. **The disclosed git bus error is real, reproducible, and mischaracterised.** I reproduced it independently: `git ls-tree -r HEAD` dies with **exit 135 = 128 + 7 (SIGBUS, core dumped)** on 3/3 attempts. It is **not transient**, it is **not in packed content**, and it is **not unrelated** — it is localized to the `src`, `provenance` and `archive` subtrees at HEAD, and specifically inside `src/tracks/general`, one of the two frozen tracks. The report's *safety conclusion* is nonetheless correct: the protected-scope checks completed and returned clean, and SIGBUS fails loudly rather than producing a false pass. Content-level integrity is fully established by git-independent SHA-256.

Because the rule content is sound but a frozen artifact carries an incorrect integrity statement about a frozen track, the verdict is REVISE with a single, tightly scoped correction (§9). It does not block I2 Increment A on the merits, and §10 specifies that boundary.

---

## 2. Integrity

### 2.1 Addendum artifacts and memo

| Artifact | SHA-256 |
|---|---|
| `I1_ZERO_SD_RULE_ADDENDUM.md` | `5b56bfe79971df5fd9f2c3915d897a452c95d481e1fae5b53ea978941637f8e2` |
| `I1_ZERO_SD_RULE_ADDENDUM.json` | `3439d33c4a10c68e1c420576b86868c7e60d5a52494ddb6d3655f0f503267d54` |
| `I1_ZERO_SD_RULE_ADDENDUM_HASHES.sha256` | `299dfb2ef4aa7286ba598b5ed963ef06dfcb318b1ef5e60aa5f17691bd8c4d5c` |
| `I1_ZERO_SD_RULE_ADDENDUM_REPORT.md` | `cbeda75c157a93bf9eee1d27b90b7c41c4c4071d471f84b731a3073c4af10ceb` |
| `CLAUDE_ZERO_SD_RULE_MEMO.md` | `2cbd27fbe39bdd5bd415f74eca1e80f84a74cfb85778263831088f0ae18792c4` |

`sha256sum -c I1_ZERO_SD_RULE_ADDENDUM_HASHES.sha256` from the repository root → **3/3 OK**. All four artifacts are covered: three standard entries plus the normalized self-check.

The memo hash `2cbd27fb…` is cited identically in the Markdown header (line 7), the JSON `controlling_documents.claude_zero_sd_rule_memo.sha256`, and the report table — and matches my independent computation.

### 2.2 Normalized manifest self-check, verified independently

Implemented from the written convention alone (replace the recorded digest with 64 zeroes, hash the whole file):

| Quantity | Value | Result |
|---|---|---|
| Recorded `SELF-NORMALIZED-SHA256` | `da93739705821eed6f196d83573cfe942c810e156a15f5a068b4d3a0db5ffa2e` | — |
| Recomputed | `da93739705821eed6f196d83573cfe942c810e156a15f5a068b4d3a0db5ffa2e` | **MATCH** |
| Occurrences of the digest in the file | 1 | **unambiguous** |
| Same convention as original `I1_HASHES.sha256`? | re-verified that manifest too | **MATCH — identical convention** |

### 2.3 Controlling and original artifacts unchanged

| Artifact | Expected | Result |
|---|---|---|
| `DRAFT_…_REV3_1.md` | `701b4509…fe959` | **MATCH** |
| `CLAUDE_I1_READONLY_AUDIT.md` | `bebf0d14…8471a` | **MATCH** |
| `CLAUDE_REV3_1_CONFIRMATION_AUDIT.md` | `0f448db8…0cd17` | unchanged |
| `CLAUDE_REV3_READONLY_AUDIT.md` | `3139e575…6167cb` | unchanged |
| Original `I1_HASHES.sha256` manifest | — | **5/5 OK** |
| `I1_REGISTRATION.md` / `.json` | `d877fbcf…` / `d4a9b99a…` | unchanged |
| `I1_SOURCE_SCHEMA.json` | `4de0b45f…` | unchanged |
| `I1_FREEZE_REPORT.md` | `e335019a…` | unchanged |
| **`SEALED_PREDICTIONS.json`** | `82a7fcb3…50c3` | **unchanged — not amended** |

Every hash the addendum JSON cites for the six original artifacts was independently recomputed and matches.

### 2.4 Frozen tracks (git-independent)

| Recipe | Track | Files | Hash | Result |
|---|---|---:|---|---|
| Historical | ecological / general | 149 / 162 | `951365d7…` / `614524d7…` | unchanged |
| Source-only | ecological | 52 | `2b3b8ae6d2f8ff5ffb17c4885ded9e8f1f6b3c0cb662f393186fe4b4706a884e` | unchanged |
| Source-only | general | 55 | `f90cea6f28dcacb910b5e036bf9e09958715d00a2418fd0856a3d5a12856bdbd` | unchanged |
| Declaration | `provenance/frozen_tracks.sha256` | — | `319cd42884da74cbdd54b228ecf4fbb11688e40293d376a333bf1b3be6ffcaaf` | unchanged |

### 2.5 No protected tracked file modified or staged

| Check | Result |
|---|---|
| `git diff --exit-code` | **exit 0** (5/5 attempts) |
| `git diff-files --quiet` (worktree vs index) | **exit 0** |
| `git diff-index --cached --quiet HEAD` (index vs HEAD) | **exit 0 — nothing staged** |
| Per-scope, worktree and index: `src`, `configs`, `tests`, `docs/true_noisy_state_real_methods`, `results`, `provenance` | **all exit 0** |
| `git status --short --untracked-files=all` | 22 lines, **all `??`** — no tracked modification |
| Untracked total | **22** — matches the report's stated final count |

---

## 3. Memo-to-addendum fidelity

Markdown and JSON assessed independently against `CLAUDE_ZERO_SD_RULE_MEMO.md`.

| # | Required freeze | Markdown | JSON | Verdict |
|---|---|---|---|---|
| 1 | Direct assignment of column 1 to literal `0.0` | A4 code block; A5 "emit literal float64 `0.0`" | `primary_direct_assignment.assignments.column_1_sd: 0.0` | **CORRECT** |
| 2 | Columns 2–4 exact copies of column 0 | A4 `features[:, 2:5] = features[:, 0][:, None]`; A2 "bit-identical" | `columns_2_through_4: "exact copies of column 0"`; `arm_t_exact_aliases.required_equality: "bit-identical"` | **CORRECT** |
| 3 | Prohibition on computed-moment residue as primary path | A4 "must not compute weighted moments and then detect or repair their residue" | `weighted_moment_construction_then_residue_detection_permitted: false` | **CORRECT** |
| 4 | Column order preserved | A2 "registered feature order must be preserved"; A5 "retain every registered column in its registered position" | `feature_schema.preserve_order: true` | **CORRECT** |
| 5 | No deletion, merging, jitter, artificial variance | A2 "not be removed, merged, jittered, reordered"; A4 "no artificial noise or epsilon variance" | `drop_merge_jitter_reorder_or_silent_reconstruction_permitted: false`; `artificial_noise_or_epsilon_variance_permitted: false` | **CORRECT** |
| 6 | `std <= 1e-8` as fail-closed guard only | A4 "only a fail-closed validation guard … not the construction mechanism" | `validation_guard.role: "fail-closed validation only; not feature construction"` | **CORRECT** |
| 7 | Explicit constant masks, offsets, scale `1.0` | A5 mask + float64 mean offset + literal `1.0` | `explicit_constant_mask_required: true`; `masked_offset`; `masked_scale: 1.0` | **CORRECT** |
| 8 | Offline/runtime transformation parity | A5 parity paragraph; A5 "one shared transformation implementation" | `parity_invariants` (8 fields); `runtime_statistics_recomputed: false` | **CORRECT** |
| 9 | Explicit alias and rank-collapse receipting | A5 "record the alias group `[0, 2, 3, 4]`, state-block rank, and design-matrix condition number" | `alias_and_rank_collapse_receipted: true`; `arm_t_exact_aliases`; ranks 5 → 1 | **CORRECT** |
| 10 | No modification of Arm O preprocessing or `src/tracks/**` | A4 explicit prohibition, naming `behavior_model.py`, `public_surrogate.py` | `implementation_location_rule` — four separate `false` flags | **CORRECT** |

**Required negative check — does it wrongly claim bit-exact zero before assignment?** **No.** The addendum states the opposite, twice and unambiguously:

- Markdown A2: *"column 1 may be literal `0.0` or may contain floating-point reduction residue as large as approximately `4.441e-16`. Exact-equality detection such as `sd == 0.0` is therefore forbidden as the primary rule."*
- JSON: `arm_t_sd_exactly_zero_for_every_input: false`, `arm_t_sd_max_observed_residue_approximately: 4.441e-16`, `exact_equality_detection_as_primary_rule_permitted: false`.

This is the single most important thing the addendum had to get right, and it is correct in both representations. The memo's other measured facts also transfer accurately: Arm O already-constant columns `[5, 9]`, Arm T additional constant column `[1]`, rank 5 → 1, and all five existing floors (1e-8, 1e-8, 1e-6, 0.02, 0.03) with `existing_division_by_zero_or_nan_path: false`.

---

## 4. Branch semantics

| Requirement | Location | Verdict |
|---|---|---|
| Primary end-to-end Arm T distinguished | MD A5; JSON `end_to_end_preprocessing` | **CORRECT** |
| RefPlan optional frozen-fit secondary distinguished | MD A6; JSON `frozen_fit_secondary.scope`, `primary_estimand: false` | **CORRECT** |
| PLUS/MOOR branch vacuous | MD A3, A6; JSON `method_scope.plus_moor_rule`, `frozen_fit_secondary.vacuous_for` | **CORRECT** |
| Unavailable to OGSRL/BA-MCTS/EVD | MD A6; JSON `frozen_fit_secondary.unavailable_to` | **CORRECT** |
| Changed preprocessing or learned artifact forces the exact label | MD A5 verbatim `MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY`; JSON `required_label` string-identical | **CORRECT** |
| Label prevents frozen-fit and causal claims | MD A5 "must not be described as frozen-fit, state-input-only, or a causal state-representation contrast"; JSON `causal_or_frozen_fit_label_permitted_after_any_artifact_change: false` | **CORRECT** |

The trigger set is broader than the minimum and correctly so: `artifact_changes_triggering_bundle_only` lists preprocessor, feature fit, dynamics fit, reward surrogate, `residual_sigma`, policy parameters, and "any other learned artifact". This closes the surrogate-input loophole direction I raised in the I1 audit by making *any* learned-artifact change sufficient.

---

## 5. RefPlan disclosure

| Required element | Location | Verdict |
|---|---|---|
| Variable SD during prior fitting | MD A7 "fitted with a variable SD feature"; JSON `prior_fit_sd: "variable"` | **PRESENT** |
| Root SD hard-set to zero during planning | MD A7; JSON `planner_root_sd: "hard-assigned to zero"` | **PRESENT** |
| ≈ `−2.7018` tiger, `−2.7348` fox | MD A7; JSON `standardized_sd_offset_approximately` and `frozen_fit_secondary.approximately_standardized_sd` | **PRESENT — matches my measured values exactly** |
| Arm T may remove the pre-existing mismatch | MD A7 "Arm T may remove the mismatch"; JSON `interpretation` | **PRESENT** |
| Resulting limitation on interpreting the contrast | MD A7 "cannot automatically be attributed solely to improved abundance information"; JSON same wording | **PRESENT** |
| Both-arm receipt and report obligation | MD A7; JSON `refplan_mismatch_disclosure.receipt_both_arms` / `report_both_arms` | **PRESENT** |
| Frozen-fit branch OOD extrapolation receipted | MD A7 final sentence; JSON `refplan_ood_extrapolation_receipt_required: true` | **PRESENT** |

Complete and faithful to memo §2 and §4.

---

## 6. RNG rule

| Requirement | Location | Verdict |
|---|---|---|
| `rng.normal(loc, 0.0)` returns `loc` exactly | MD A8; JSON `rng_normal_zero_scale.returned_value: "loc exactly"` | **RECORDED** |
| It still advances RNG state | MD A8; JSON `rng_state_advances: true` | **RECORDED** |
| Future tests check returned-value exactness | JSON `rng_invariant.future_tests_required[0]`; MD A10 item 12 | **REQUIRED** |
| Future tests check RNG call/state parity | JSON `future_tests_required[1]`,`[2]` (advancement and call count); MD A8, A10 item 12 | **REQUIRED** |
| Receipt carries both parities | JSON receipt fields `rng_call_count_parity`, `rng_state_advancement_parity` | **REQUIRED** |

Both halves of the fact are recorded, and both are pushed into the mandatory test set and the receipt. No RNG implementation is authorized (`"No RNG implementation is authorized by this addendum"`).

---

## 7. Machine-readable completeness — Markdown / JSON agreement

| # | Normative category | Markdown | JSON | Mismatch? |
|---|---|---|---|---|
| 1 | dtype and construction mode | A4 `numpy.float64`, "assign, do not detect" | `feature_schema.dtype`, `primary_direct_assignment.mode` | none |
| 2 | Constant mask | A5 explicit mask | `explicit_constant_mask_required`, `constant_mask_criterion` | none |
| 3 | Alias map and rank diagnostic | A5 `[0, 2, 3, 4]`, rank, condition number | `arm_t_exact_aliases`, ranks 5/1, `alias_and_rank_collapse_receipted` | none |
| 4 | Offsets and scales | A5 float64 mean offset, literal `1.0` | `masked_offset`, `masked_scale: 1.0`, `existing_1e_8_floor_used_as_assigned_sd_scale: false` | none |
| 5 | Guard threshold | A4/A5 `std <= 1e-8`, inclusive; runtime `1e-6` | `constant_threshold: 1e-8`, `threshold_comparison: "<="`, `runtime_constant_tolerance: 1e-6`, `relative_scaling_term: null` | none |
| 6 | Preprocessing hashes | A5 artifact SHA-256 fit + runtime | `parity_invariants.hash_identity`; receipt hash fields | none |
| 7 | Fit/runtime parity | A5 paragraph | `parity_invariants` (8 keys) | none |
| 8 | RefPlan offset receipts | A7 | `refplan_mismatch_disclosure`, `frozen_fit_secondary.approximately_standardized_sd` | none |
| 9 | RNG parity | A8 | `rng_invariant` | none |
| 10 | Per-method eligibility | A3, A6 | `method_scope`, `frozen_fit_secondary.vacuous_for` / `unavailable_to` | none |
| 11 | Permitted / forbidden actions | A11 | `permitted_actions` (2), `forbidden_actions` (11) | none |
| 12 | Sealed predictions unchanged | A1, A7 | `sealed_predictions.unchanged: true`, `p7_already_covers_refitted_artifact_changes: true` | none |
| 13 | I2 remains unauthorized | A1, A12 | `controlling_status.i2_authorized: false` + 4 further `false` flags | none |

**Total mismatches: 0.** Neither representation was treated as overriding the other; each was read on its own terms and then compared. The JSON is a strict superset in detail (receipt field list, negative-test list, fail-closed list) and contradicts the Markdown nowhere.

One cosmetic observation, not a mismatch: Markdown A2 renders the ten columns as a 1-based numbered list while the prose and JSON use zero-based indices. The addendum disambiguates explicitly — *"Indices are zero-based in implementation and receipts"* — and JSON `feature_schema.indexing: "zero-based"` with `affected_column.index: 1` confirms it. No ambiguity survives, but a future revision could render the list as `0.`–`9.` to remove the visual mismatch.

---

## 8. Scope, sufficiency, and the git bus error

### 8.1 Does it close Codex's blocker?

| Blocker element | Closed? | Exact frozen value |
|---|---|---|
| Fitted scale | **Yes** | `1.0` literal for masked columns; the existing `1e-8` floor explicitly forbidden as the assigned-SD scale |
| Near-zero rule | **Yes** | `std <= 1e-8`, inclusive, guard-only; construction is direct assignment |
| Offline/runtime parity | **Yes** | eight named invariants, single shared implementation, artifact SHA-256 identity, `runtime_statistics_recomputed: false` |
| Receipt requirements | **Yes** | ~40 mandatory fields; memo §8 schema declared controlling; `parity_result = PASS` gate |

**An I2 Increment A implementer would not have to invent a scientific or preprocessing rule.** Every decision point I could identify has a registered value: dtype, construction mode, both thresholds, the comparison operator, offset definition, scale, the emitted constant, serialization mode, tolerance space, finite check, alias handling, rank reporting, label text, method eligibility, and the full test and receipt sets. The combined Revision 3.1 + I1 + addendum stack is sufficient.

### 8.2 Independent assessment of the git bus error

I reproduced it. It is **deterministic, not transient**:

| Probe | Result |
|---|---|
| `git ls-tree -r --name-only HEAD` | **exit 135 = 128 + 7 (SIGBUS), "Bus error (core dumped)", 3/3 attempts** |
| `git ls-tree -r HEAD -- src` | **exit 135** |
| `git cat-file --batch-check --batch-all-objects` | **exit 135** |
| `git rev-list --objects --all` | returns only the 7 commit OIDs; the tree/blob walk aborts |
| `git ls-tree HEAD` (non-recursive) | exit 0, 27 entries |
| `git cat-file -p HEAD` | exit 0, correct commit `77cd38ad…` |
| `git ls-files` | exit 0, 3,769 files |
| `git diff --exit-code` | **exit 0, 5/5** |
| `git status --short -uall` | completes, 22 lines |

**Localization** — recursive walk of each top-level path at HEAD:

| Path | Result |
|---|---|
| `src/tracks/ecological` | exit 0 — 52 files |
| `configs` | exit 0 — 72 |
| `tests` | exit 0 — 54 |
| `docs` | exit 0 — 221 |
| `results` | exit 0 — 23 |
| **`src` (specifically `src/tracks/general`)** | **exit 135** |
| **`provenance`** | **exit 135** |
| **`archive`** | **exit 135** |

Descending by OID: `src` → `src/tracks` → `src/tracks/general` all read fine non-recursively; `git ls-tree -r 87ea148d…` (the `general` subtree) faults. So the damaged object lies below `src/tracks/general` in HEAD's history.

**Nature.** The object store holds **4,365 loose objects** and a pack containing only **4** objects (3,466-byte packfile). A stray zero-byte `.git/objects/1e/tmp_obj_j26OYJ` dated today 16:13 is present, and `git count-objects -v` reports it as garbage — evidence of an interrupted object write. `stat -f` on the working directory persistently returns **"Network is down"**, and I independently hit an `EIO` write failure on this same filesystem earlier today. SIGBUS on an mmap'd loose object whose backing store is truncated or unavailable is the textbook symptom.

**Three corrections to the report's disclosure** (report §5, lines 96–102):

| Report wording | Finding |
|---|---|
| "transient Git reader condition" | **Incorrect** — deterministic, 3/3 and on every deep-walk variant |
| "reading unrelated historical **packed** content" | **Incorrect** — the pack holds 4 objects; the fault is in **loose**-object reading |
| "**unrelated** historical … content" | **Incorrect** — it affects `src/tracks/general` (a **frozen track**), `provenance`, and `archive` |

**Assessment against the three required questions:**

1. **Does it affect any controlling or protected tracked content?** *The git object-store history of protected content: yes* — `src/tracks/general` and `provenance`. *The controlling or protected content itself: no.* Working-tree content is intact and verified git-independently: source-only general track `f90cea6f…`, ecological `2b3b8ae6…`, `provenance/frozen_tracks.sha256` `319cd428…`, all four addendum artifacts, all six original I1 artifacts, Revision 3.1 and both prior audits.

2. **Are targeted hashes and status checks sufficient to establish integrity?** **Yes**, for two independent reasons. First, the primary evidence is content SHA-256 computed directly from the working tree, which does not involve git at all. Second, the git checks that *did* complete are logically sound: `git diff-index --cached --quiet HEAD -- src/tracks/general` returns 0 because tree-diff prunes on OID equality without descending into the faulting objects — and OID equality is content equality. Crucially, **SIGBUS fails loudly (exit 135) and can never produce a false "clean"**, so a completed check is a valid check. I re-ran the affected scopes specifically: `src/tracks/general` and `provenance` return exit 0 for both worktree-vs-index and index-vs-HEAD.

3. **Does it create an I2 blocker?** **Not for Increment A**, which uses synthetic fixtures in an external module and needs no git history. It **is** a live operational risk to register: `git fsck`, `git gc`, `git clone`, `git archive`, and any full `git diff HEAD` over `src`/`provenance` will crash, so any future stage that intends to *prove frozen-track immutability via git history* has no working path and must use the source-only SHA-256 recipe instead. That substitution is already the registered primary method (`I1_REGISTRATION.md` §9, Revision 3.1 §6.1), so the registered provenance chain remains intact.

I performed **no** repair: no `gc`, `repack`, `fsck --lost-found`, `prune`, `checkout`, `reset`, or any other mutation. The stray temp object was left in place.

---

## 9. Required correction

**One correction, to `I1_ZERO_SD_RULE_ADDENDUM_REPORT.md` §5 only.** The zero-SD rule, the Markdown addendum, the JSON addendum and the manifest need **no change**.

Replace the sentence beginning *"Full `git diff --exit-code` attempts encountered a repository-level Git bus error while reading unrelated historical packed content."* and its "transient" closing sentence with:

> A repository-level Git bus error (SIGBUS, exit 135) occurs deterministically on any
> recursive object walk — `git ls-tree -r HEAD`, `git cat-file --batch-all-objects`,
> `git rev-list --objects --all`. It is reproducible, not transient. The object store
> holds 4,365 loose objects and a 4-object pack, so the fault is in loose-object reading,
> not packed content. Recursive walks fault for `src` (specifically `src/tracks/general`),
> `provenance` and `archive`; they succeed for `src/tracks/ecological`, `configs`,
> `tests`, `docs` and `results`. A stray zero-byte `.git/objects/1e/tmp_obj_j26OYJ` and a
> persistent `stat -f` "Network is down" on this filesystem indicate an interrupted write
> on a network mount rather than logical repository corruption.
>
> This does not affect the integrity conclusion. SIGBUS terminates the process and can
> never yield a false "clean", so every check that completed is valid: `git diff
> --exit-code`, `git diff-files --quiet` and `git diff-index --cached --quiet HEAD`
> return 0, including when restricted to `src/tracks/general` and `provenance`, because
> tree-diff prunes on OID equality without descending into the faulting objects. All
> content integrity is independently established by working-tree SHA-256, which does not
> use Git.
>
> Registered consequence: `git fsck`, `git gc`, `git clone`, `git archive` and full
> `git diff HEAD` over `src`/`provenance` are expected to fail. Frozen-track immutability
> must be proven with the registered source-only SHA-256 recipe, never via Git history.
> No repair, repack, garbage collection or prune is authorized under this addendum.

Add the same as a `known_repository_condition` block in `I1_ZERO_SD_RULE_ADDENDUM.json`, and re-issue `I1_ZERO_SD_RULE_ADDENDUM_HASHES.sha256`. Because the report and JSON hashes change, the manifest must be regenerated with the same normalized self-check convention. The six original I1 artifacts, the memo, Revision 3.1 and the sealed predictions all remain untouched.

**Sealed predictions:** unchanged and unaffected. `SEALED_PREDICTIONS.json` is byte-identical at `82a7fcb3…50c3`; P7 already governs changed fitted artifacts, and nothing in this correction is a scientific claim.

---

## 10. Smallest recommended next authorization

The correction in §9 is documentation-only and independent of the scientific rule, so it can be applied in parallel with — not before — the following boundary.

> **Authorize I2 Increment A: isolated external adapter module, synthetic fixtures only.**

| Capability | Authorized? |
|---|---|
| Design and implement the degenerate-feature transformation in a **new external module outside `src/tracks/**`** | **YES** |
| **Synthetic fixtures only**, reproducing the ten-column float64 schema, including fixtures whose reduction-based construction leaves nonzero residue | **YES** |
| Read truth **metadata / binding only** (key set, dtype, shape, `public_dataset_sha256`) | **YES** |
| Run unit tests, including all 11 negative fail-closed cases and the 13 assertion groups in MD A10 / JSON `required_i2_tests` | **YES** |
| Emit the §A9 receipt from synthetic inputs with `parity_result = PASS` | **YES** |
| Open real truth arrays or construct any real or derived dataset | **NO** |
| Fit models, run regression, rebaseline, produce scientific output, create an execution namespace, submit Slurm work | **NO** |
| Edit `src/tracks/**`, `behavior_model.py`, `public_surrogate.py`, Arm O preprocessing, or any accepted artifact | **NO** |
| `git gc` / `repack` / `fsck --lost-found` / `prune` / any repository repair | **NO** — separate authorization, and only after a filesystem-health assessment |

Increment A must stop and return an independent-audit artifact showing: source-only frozen-track hashes unchanged (git-independent); `git diff-files` / `git diff-index --cached` clean for all protected scopes; the new module confined outside `src/tracks/**`; every negative test demonstrated to fail closed; and a synthetic receipt whose column-1 transformed value is exactly `0.0`, whose columns 2–4 are bit-exact aliases of column 0, and whose `parity_result` is `PASS`.

**Authorization status of this audit.** Performed: read-only inspection and hashing of the four addendum artifacts, the memo, all six original I1 artifacts, Revision 3.1 and both prior Claude audits; independent reproduction of the normalized manifest self-check; JSON validation; read-only Git metadata and object-store diagnostics with no branch change, checkout, fetch, commit, gc, repack, prune or repair. Created exactly one file: this report. **No truth array, dataset or derived data was opened, extracted or created; no adapter, preprocessor, code, test or configuration was written; no model fitted; no evaluation, regression or rebaseline run; no I2 namespace created; no Slurm job submitted; no network access.** Re-verified after all probes: both frozen-track source-only hashes unchanged, all four addendum hashes unchanged, original I1 manifest 5/5 OK, addendum manifest 3/3 OK, `git diff --exit-code` returns 0, 22 untracked paths all documentation.

---

## Verdict

**REVISE — ZERO-SD ADDENDUM REQUIRES SPECIFIED DOCUMENTATION CHANGES**
