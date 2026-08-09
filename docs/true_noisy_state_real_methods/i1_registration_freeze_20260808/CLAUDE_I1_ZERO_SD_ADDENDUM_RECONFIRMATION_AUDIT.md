# Claude reconfirmation audit — corrected I1 zero-SD provenance documentation

Date: 2026-08-08 (Australia/Melbourne)
Auditor: Claude (read-only reconfirmation pass)
Controlling prior audit: `CLAUDE_I1_ZERO_SD_ADDENDUM_CONFIRMATION_AUDIT.md`, SHA-256 `494bd27fb7d6a1b0e81271daab0300622ffcfa201d6f7a1293fc689e467e6765`, verdict `REVISE — ZERO-SD ADDENDUM REQUIRES SPECIFIED DOCUMENTATION CHANGES`
Environment: every command run with `LC_ALL=C` and `PYTHONDONTWRITEBYTECODE=1`; no bytecode written, no truth array or dataset opened, no Git object touched, no repair of any kind.

Collision check: `CLAUDE_I1_ZERO_SD_ADDENDUM_RECONFIRMATION_AUDIT.md` did not exist. No file was overwritten.

**Git restriction compliance.** No broad `git status`, no full `git ls-tree`, no broad `git diff HEAD`, no `fsck`, `gc`, `repack`, `clone` or `archive` was run. Only narrowly scoped `git diff-files --quiet -- <path>`, `git diff-index --cached --quiet HEAD -- <path>`, `git ls-files -m`, and `git ls-files --others` were used — all previously demonstrated to complete. `.git/objects/1e/tmp_obj_j26OYJ` was listed by `ls` only and neither read, deleted, renamed nor repaired. No SIGBUS, EIO or network-mount error occurred during this pass.

---

## 1. Executive verdict

**The correction is complete, exact, and confined to provenance disclosure. The sole reason for the prior `REVISE` is fully resolved, and no scientific rule changed.**

Six findings:

1. All four reported hashes match to the full 64 hex characters, including the normalized manifest self-hash, which I reproduced from first principles.
2. The normative Markdown `I1_ZERO_SD_RULE_ADDENDUM.md` is **byte-identical** to its pre-correction value `5b56bfe7…37f8e2`. The rule text was not touched.
3. All three inaccurate descriptors — "transient", "packed content", "unrelated" — now appear **only inside explicitly negating sentences** that refute them. No residual assertion survives.
4. All ten sub-items required of the corrected disclosure are present in both the Markdown report and the JSON, including the explicit labelling of the interrupted-write/network-mount explanation as a **diagnostic hypothesis, not proven causation**.
5. The JSON gained exactly **one** new top-level key (`known_repository_condition`) and retained all 24 pre-existing keys. Across the same thirteen normative categories used in the prior audit, the corrected JSON and the unchanged Markdown agree with **zero mismatches**.
6. Regenerating the checksum manifest was mathematically necessary and was done correctly, preserving the original three-entry-plus-self structure and the same normalization convention.

---

## 2. Complete SHA-256 values

| Artifact | Full SHA-256 | Reported | Result |
|---|---|---|---|
| `I1_ZERO_SD_RULE_ADDENDUM.md` | `5b56bfe79971df5fd9f2c3915d897a452c95d481e1fae5b53ea978941637f8e2` | `5b56bfe7…37f8e2` | **MATCH — UNCHANGED** |
| `I1_ZERO_SD_RULE_ADDENDUM.json` | `c39f58c4a44ec1951e777d6d19fba6fd605a1fd66a20fcedc1cadb31f0a699d8` | `c39f58c4…a699d8` | **MATCH — corrected** |
| `I1_ZERO_SD_RULE_ADDENDUM_REPORT.md` | `e15849eb21b369e5b1e16e541f2faf8281e1567b5851ae29af699d219013d932` | `e15849eb…13d932` | **MATCH — corrected** |
| `I1_ZERO_SD_RULE_ADDENDUM_HASHES.sha256` (normalized self-hash) | `d8ea5e028b88a8a528aca3b178ef9bde6083d33637a6292517885bbed0bc6762` | `d8ea5e02…bc6762` | **MATCH — regenerated** |
| `I1_ZERO_SD_RULE_ADDENDUM_HASHES.sha256` (literal file hash) | `cf688111978643911ac0e8952e9bbe77392da907abaefa950243b3c7a5852e41` | — | recorded for completeness |

Modification times corroborate the scope: the Markdown is unchanged at 16:44; the JSON, report and manifest are all 19:04.

---

## 3. Verification results

### 3.1 Standard checksum entries

`sha256sum -c I1_ZERO_SD_RULE_ADDENDUM_HASHES.sha256` from the repository root → **3/3 OK** (Markdown, JSON, report). The manifest's Markdown entry carries the unchanged `5b56bfe7…37f8e2`, providing an independent second attestation that the rule file was not modified.

### 3.2 Normalized manifest self-check, from first principles

Implemented directly from the convention (replace the recorded digest with 64 zeroes, hash the entire file), without reusing any prior script logic:

| Quantity | Value | Result |
|---|---|---|
| Recorded `SELF-NORMALIZED-SHA256` | `d8ea5e028b88a8a528aca3b178ef9bde6083d33637a6292517885bbed0bc6762` | — |
| Recomputed | `d8ea5e028b88a8a528aca3b178ef9bde6083d33637a6292517885bbed0bc6762` | **MATCH** |
| Occurrences of the digest in the file | 1 | **unambiguous replacement** |

### 3.3 Original I1 artifacts

`sha256sum -c I1_HASHES.sha256` → **5/5 OK**: `I1_REGISTRATION.md`, `I1_REGISTRATION.json`, `SEALED_PREDICTIONS.json`, `I1_SOURCE_SCHEMA.json`, `I1_FREEZE_REPORT.md`. The original manifest itself is unchanged at `5c483370…3b27`.

### 3.4 Controlling documents, memo, prior audits, sealed predictions

| Artifact | SHA-256 | Result |
|---|---|---|
| `DRAFT_…_REV3_1.md` | `701b4509f1dc04b7885891be72e526420ac5558be56bdde2b9628789398fe959` | unchanged |
| `CLAUDE_ZERO_SD_RULE_MEMO.md` | `2cbd27fbe39bdd5bd415f74eca1e80f84a74cfb85778263831088f0ae18792c4` | unchanged |
| `CLAUDE_I1_READONLY_AUDIT.md` | `bebf0d14b327b159fde9db30fc079d878765f25b10ab0f3529a7a7a01698471a` | unchanged |
| `CLAUDE_I1_ZERO_SD_ADDENDUM_CONFIRMATION_AUDIT.md` | `494bd27fb7d6a1b0e81271daab0300622ffcfa201d6f7a1293fc689e467e6765` | unchanged |
| `CLAUDE_REV3_1_CONFIRMATION_AUDIT.md` | `0f448db89109186aa4bbd64bc000da45ae662386dea79e658bcd802f4880cd17` | unchanged |
| `CLAUDE_REV3_READONLY_AUDIT.md` | `3139e575859badd11c4b4ef8cc3d2d058fe312598e622831aa11d6795f6167cb` | unchanged |
| **`SEALED_PREDICTIONS.json`** | `82a7fcb394c80e673ede04c5692db839821b06f23f752076e8cf34bb29da50c3` | **unchanged — not amended** |

The corrected report's controlling-input table (lines 25–30) still cites Revision 3.1, the prior I1 audit, the memo and the sealed predictions at these exact values.

### 3.5 Registered source-only frozen-track hashes

| Track | Files | SHA-256 | Result |
|---|---:|---|---|
| ecological | 52 | `2b3b8ae6d2f8ff5ffb17c4885ded9e8f1f6b3c0cb662f393186fe4b4706a884e` | unchanged |
| general | 55 | `f90cea6f28dcacb910b5e036bf9e09958715d00a2418fd0856a3d5a12856bdbd` | unchanged |
| `provenance/frozen_tracks.sha256` | — | `319cd42884da74cbdd54b228ecf4fbb11688e40293d376a333bf1b3be6ffcaaf` | unchanged |

### 3.6 Narrow Git checks

| Scope | worktree vs index | index vs HEAD |
|---|---:|---:|
| `docs/true_noisy_state_real_methods` | 0 | 0 |
| `src/tracks/ecological` | 0 | 0 |
| **`src/tracks/general`** | **0** | **0** |
| `configs` | 0 | 0 |
| `tests` | 0 | 0 |
| `results` | 0 | 0 |
| **`provenance`** | **0** | **0** |

`git ls-files -m -- docs/true_noisy_state_real_methods` → **0 modified tracked files**. `git ls-files --others` in the I1 directory lists 13 paths, all documentation. Every check completed; none returned SIGBUS, EIO or a mount error. The two scopes whose *history* traversal faults (`src/tracks/general`, `provenance`) still verify cleanly here, because tree-diff prunes on OID equality without descending into the faulting objects.

---

## 4. Correction compared directly with §9 of the prior audit

### 4.1 Inaccurate descriptors removed

| Descriptor | Report occurrences | JSON occurrences | Context | Verdict |
|---|---:|---:|---|---|
| "transient" | 1 | 0 | line 109: *"It is reproducible, **not transient**."* | **REMOVED as an assertion** |
| "packed content" | 1 | 0 | line 111: *"the fault is in loose-object reading, **not packed content**."* | **REMOVED as an assertion** |
| "unrelated" | 1 | 1 | report line 133: *"must **not** be described as unrelated"*; JSON `protected_scope_effect`: *"must **not** be characterized as unrelated"* | **REMOVED as an assertion** |

Every surviving occurrence is inside a sentence that explicitly negates the term. No residual claim that the fault is transient, packed, or unrelated to protected scopes exists anywhere in either artifact.

### 4.2 Required content now recorded

| # | Required by §9 / task 4 | Report | JSON | Verdict |
|---|---|---|---|---|
| 1 | Deterministic SIGBUS / exit 135 in three audited full-tree attempts | "SIGBUS, exit 135 … occurs deterministically"; three commands named | `audited_probe`: `command: "git ls-tree -r HEAD"`, `attempts: 3`, `failures: 3`, `exit_code: 135`, `signal: "SIGBUS"` | **RECORDED** |
| 2 | Loose-object rather than packed-content traversal | "4,365 loose objects and a 4-object pack … not packed content" | `loose_objects_approximately: 4365`, `packed_objects_approximately: 4`, `traversal_association` | **RECORDED** |
| 3 | Affected scopes incl. `src/tracks/general`, `provenance`, `archive` | line 111–112 | `failing_recursive_scopes: ["src","src/tracks/general","provenance","archive"]` | **RECORDED** |
| 4 | Clean scopes incl. `src/tracks/ecological`, `configs`, `tests`, `docs`, `results` | line 112–113 | `clean_recursive_scopes` — exactly those five | **RECORDED** |
| 5 | Working-tree integrity primarily via registered source-only SHA-256 | "independently established by working-tree SHA-256, which does not use Git" | `working_tree_content_integrity.primary_method`, with all three hashes embedded | **RECORDED** |
| 6 | Successful narrow checks remain valid | "SIGBUS terminates the process and can never yield a false 'clean', so every check that completed is valid" | `completed_check_rule`; `false_success_possible_after_signal_termination: false`; `targeted_cached_tree_comparison` | **RECORDED** |
| 7 | Full Git-history traversal unavailable as an integrity gate | "must not be used as I2 integrity gates"; "never via Git history" | `full_git_history_traversal_available: false`; `..._safe_as_current_integrity_gate: false`; 7-item `unavailable_or_unsafe_operations` | **RECORDED** |
| 8 | Git repair outside I2, separate authorization | "Repository repair is outside I2 and requires separate authorization following a filesystem-health assessment" | `integrity_workflow.repository_repair`; `scientific_effect.i2_repair_authorized: false` | **RECORDED** |
| 9 | Interrupted-write / network-mount labelled a hypothesis | "a supported diagnostic hypothesis, not a proven causal conclusion" | `hypothesis_status: "supported diagnostic hypothesis; not a proven causal conclusion"` | **RECORDED** |
| 10 | Protected-scope effect stated plainly | "The Git-history fault touches a protected frozen-track scope" | `protected_scope_effect` names `src/tracks/general` as a protected frozen-track scope | **RECORDED** |

Two additions beyond what §9 required, both improvements: the JSON records `temporary_object_action: "do not delete, rename, move, repair, or inspect contents under this authorization"`, which hard-freezes the stray temp object against well-meaning cleanup; and the report states plainly that the condition "does not invalidate the zero-SD rule or the sealed predictions."

---

## 5. No scientific rule changed

The normative Markdown is byte-identical, so by construction no rule expressed there changed. The risk was drift in the **corrected JSON**. I therefore re-ran the prior audit's thirteen-category comparison, reading each representation on its own terms.

| # | Category | Corrected JSON | Unchanged Markdown | Verdict |
|---|---|---|---|---|
| 1 | dtype and construction mode | `numpy.float64`; `mode: "assign, do not detect"` | A4 float64; direct assignment | **AGREE** |
| 2 | Constant mask | `explicit_constant_mask_required: true`; `"fitted std <= 1e-8"` | A5 explicit mask, same criterion | **AGREE** |
| 3 | Alias map and rank diagnostic | aliases `[2,3,4]` from col 0, `bit-identical`; rank 5 → 1 | A2/A5 alias group `[0, 2, 3, 4]`, rank, condition number | **AGREE** |
| 4 | Offsets and scales | `masked_scale: 1.0`; `existing_1e_8_floor_used_as_assigned_sd_scale: false` | A5 float64 mean offset, literal `1.0`, floor forbidden as scale | **AGREE** |
| 5 | Guard threshold | `1e-8`, `<=`, runtime `1e-6`, `relative_scaling_term: null` | A4/A5 identical | **AGREE** |
| 6 | Preprocessing hashes | `hash_identity: "SHA-256 recorded at fit and reverified at runtime"` | A5 artifact SHA-256 fit + runtime | **AGREE** |
| 7 | Fit/runtime parity | 8 invariants; `runtime_statistics_recomputed: false` | A5 parity paragraph, single implementation | **AGREE** |
| 8 | RefPlan offset receipts | `{tiger: -2.7018, fox: -2.7348}` in both `measured_facts` and `frozen_fit_secondary` | A7 same two values | **AGREE** |
| 9 | RNG parity | `returned_value: "loc exactly"`, `rng_state_advances: true`, 3 required tests | A8 identical, incl. call count | **AGREE** |
| 10 | Per-method eligibility | 4 affected / 2 unaffected / 3 unavailable | A3, A6 identical | **AGREE** |
| 11 | Permitted and forbidden actions | 2 permitted / 11 forbidden | A11 identical | **AGREE** |
| 12 | Sealed predictions unchanged | `unchanged: true`; hash `82a7fcb3…50c3` | A1, A7 | **AGREE** |
| 13 | I2 remains unauthorized | `i2_authorized: false`, `truth_extraction_authorized: false` (+3 more) | A1, A12 | **AGREE** |

**Mismatches: 0.**

*Method note.* My automated check initially flagged category 9. On inspection this was an artefact of my own substring test: the Markdown wraps the phrase as "still advancing\nthe RNG state", so the literal string was absent while the content was identical. Verified manually against Markdown A8 — both representations record that `rng.normal(loc, 0.0)` returns `loc` exactly *and* advances RNG state, and both require exact-value, advancement and call-count tests. No defect.

Additional spot checks, all passing: the label string `MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY` is byte-identical in both; `arm_t_sd_exactly_zero_for_every_input: false` with residue `4.441e-16` and `exact_equality_detection_as_primary_rule_permitted: false` are retained, so the corrected JSON still does **not** claim bit-exact zero; and all four `implementation_location_rule` prohibitions on editing `src/tracks/**`, `behavior_model.py`, `public_surrogate.py` and Arm O preprocessing remain `false`.

**Structural check:** the JSON has 25 top-level keys — the original 24 plus `known_repository_condition`. Nothing was removed or renamed. The report's §4 "Frozen rule summary" is substantively unchanged and still lists all ten rule bullets, including the 4.441e-16 residue, the guard-only role of `std <= 1e-8`, scale `1.0`, the alias group, and the RefPlan offsets.

**Arm T, branch semantics, RefPlan, RNG, receipt and sealed-prediction rules: all unchanged.**

---

## 6. Manifest regeneration was necessary and correct

Necessity is arithmetic: the manifest contains the SHA-256 of the JSON and the report. Both files changed (`3439d33c…` → `c39f58c4…` and `cbeda75c…` → `e15849eb…`), so the manifest's own bytes necessarily changed, and its normalized self-digest necessarily changed with them.

Correctness:

| Property | Result |
|---|---|
| Structure preserved | 3 standard entries + 1 normalized self-line, identical to the pre-correction layout |
| Paths | repo-root-relative, unchanged in form |
| Markdown entry | carries the unchanged `5b56bfe7…37f8e2` and verifies **OK** |
| JSON and report entries | carry the new values and verify **OK** |
| Self-digest | recomputed independently → **MATCH**; single occurrence |
| Convention | identical to the original `I1_HASHES.sha256`, which I re-verified in the same pass |

No stale entry, no orphaned reference, no omitted artifact.

---

## 7. Does the correction resolve the sole reason for `REVISE`?

**Yes, completely.**

The prior `REVISE` rested on exactly one defect: the freeze report's characterisation of the Git fault as "transient", "packed", and "unrelated", the third of which could have led a future reader to believe Git-history verification of a frozen track remained available. §9 specified the replacement text; the correction adopts it in substance, extends it to the JSON as a structured `known_repository_condition` block, and adds the hypothesis-labelling and temp-object-freeze clauses.

Everything the prior audit already passed remains passing: the rule content, the thirteen-category agreement, the sufficiency of the frozen constants for I2 Increment A, and all integrity verification. No new defect was introduced. No further documentation change is required.

---

## 8. Smallest safe next authorization

> **Authorize I2 Increment A only.**

| Capability | Authorized? |
|---|---|
| Isolated external adapter module **outside `src/tracks/**`** | **YES** |
| **Synthetic fixtures only**, including fixtures whose reduction-based construction leaves nonzero residue | **YES** |
| Truth **metadata and binding checks only** (key set, dtype, shape, `public_dataset_sha256`) | **YES** |
| Unit tests: the 13 assertion groups and 11 negative fail-closed cases in MD A10 / JSON `required_i2_tests` | **YES** |
| Emit the A9 receipt from synthetic inputs with `parity_result = PASS` | **YES** |
| Real truth arrays or accepted datasets | **NO** |
| Fitting, evaluation, regression, rebaseline, scientific output, Slurm work | **NO** |
| Editing `src/tracks/**`, `behavior_model.py`, `public_surrogate.py`, Arm O preprocessing, or any accepted artifact | **NO** |
| Git repair: `fsck`, `gc`, `repack`, `prune`, clone, archive, or any action on `.git/objects/1e/tmp_obj_j26OYJ` | **NO** — separate authorization after a filesystem-health assessment |

**Integrity must be established through the registered filesystem SHA-256 recipes and narrowly scoped Git checks that complete successfully — never through broad Git-history traversal.** Increment A must stop and return an independent-audit artifact showing: both source-only frozen-track hashes unchanged; narrow `git diff-files` / `git diff-index --cached` clean for every protected scope; the new module confined outside `src/tracks/**`; every negative test demonstrated to fail closed; and a synthetic receipt whose column-1 transformed value is exactly `0.0` and whose columns 2–4 are bit-exact aliases of column 0.

**Authorization status of this audit.** Performed: read-only hashing and inspection of the four addendum artifacts, all six original I1 artifacts, Revision 3.1, the zero-SD memo, and all three prior Claude audits; independent first-principles reproduction of the normalized manifest self-check; JSON validation and thirteen-category comparison; registered source-only frozen-track recomputation; narrowly scoped Git checks only. Created exactly one file: this report. **No plan, registration, addendum, checksum, report, audit, memo, code, configuration, test, result, accepted output, frozen track or Git object was modified; no truth array or dataset opened; no I2 work, evaluation, fitting, rebaselining or Slurm execution performed; no repository repair attempted; no network access.** Re-verified after all probes: all four addendum hashes, the original 5/5 manifest, both frozen-track source-only hashes, and every controlling-document hash reproduce unchanged; the stray temp object remains untouched at 0 bytes, `Aug 8 16:13`.

---

## Verdict

**PASS — CORRECTION VERIFIED; READY FOR BOUNDED I2 INCREMENT A**
