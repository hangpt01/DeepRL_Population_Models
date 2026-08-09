# Claude read-only code, contract and provenance audit — I2 Increment A

Date: 2026-08-08 (Australia/Melbourne)
Auditor: Claude (read-only code audit)
Namespace: `docs/true_noisy_state_real_methods/i2_increment_a_exact_state_adapters_20260808/`
Environment: every command run with `LC_ALL=C` and `PYTHONDONTWRITEBYTECODE=1`; all execution from a fresh temporary directory outside the repository.

Collision check: `CLAUDE_I2A_READONLY_CODE_AUDIT.md` did not exist. No file was overwritten.

**Git restriction compliance.** No broad `git status`, full `git ls-tree`, broad `git diff HEAD`, `fsck`, `gc`, `repack`, clone or archive was run. Only narrowly scoped `git diff-files --quiet -- <path>` and `git diff-index --cached --quiet HEAD -- <path>` were used; all completed. `.git/objects/1e/tmp_obj_j26OYJ` was never accessed. No SIGBUS, EIO or mount error occurred during this pass.

**Temporary-directory disclosure.** One directory was created outside the repository at `/tmp/claude-17635/.../scratchpad/i2a_audit_AVuoeE`, used for all probe execution and both test reruns, and is the only path this audit created besides this report. Zero cache or bytecode artifacts were produced anywhere in the repository (verified before and after).

---

## 1. Executive verdict

**I2A is high-quality, stayed strictly inside the synthetic boundary, and closes the interface-design questions. One specified fail-closed requirement is not met, so it needs a small, precisely scoped code change before I2B.**

Seven findings:

1. **Integrity is exact.** All 14 namespace files hash as recorded; 13 standard entries plus the normalized self-check cover every file; the self-hash `ed6f5de5…f1fec5` reproduces from first principles. No I2A file exists outside the namespace, and no protected content changed.
2. **The information boundary is genuinely fail-closed.** I probed all 13 registered fields: every one is denied in I2A, including `states` and `next_states`, which are represented as future capabilities but never granted. All 10 forbidden fields are rejected by exact name. `metadata_json` requires both the correct capability *and* the purpose `public_hash_binding`. The receipt cannot claim clean metadata-only access after a scientific request — denials are logged with `allowed: False` and no payload is ever counted.
3. **The exact-state adapter holds under adversarial probing.** Column 1 is bit-exact positive zero; **−0.0 is rejected**; 1e-18/1e-12/1e-9 jitter is rejected; a 1-ulp alias drift is rejected; sigma and observation-scale collapse are rejected at 0 and 1e-15; context columns 6–9 and both histories are preserved; the caller's array is not mutated; strided `.view(np.uint64)` comparisons work correctly on non-contiguous column slices.
4. **The RNG contract is verified decisively.** Beyond equal returned values, I confirmed that `scale=0.0` and `scale=0.25` consume identical generator state by comparing a *subsequent* draw — the strong test the brief asked for.
5. **32/32 tests pass** under both `unittest` and `pytest -p no:cacheprovider`, rerun independently from the temp directory, leaving zero repository artifacts.
6. **One defect (REVISE-level).** `classify_interpretation` grants `FROZEN-FIT/STATE-INPUT-ONLY` to RefPlan and PLUS/MOOR when **both** artifact snapshots contain unpopulated placeholders — `""`, `None`, `"unknown"`, `"TBD"`. Task 6 explicitly requires that unknown or missing artifact status **fails closed rather than defaulting to eligible**. It currently defaults to eligible.
7. **Two lower-severity observations:** the in-adapter context check is tautological (it compares an object with itself), and a NaN in a mandatory RefPlan disclosure field propagates into a receipt that is not strict-JSON parseable.

None of these is an information-boundary or scientific-data violation. No real array, dataset, or accepted artifact was touched by I2A or by this audit.

---

## 2. Namespace and integrity

### 2.1 The 14 pre-audit files, independently hashed

| # | File | SHA-256 |
|---:|---|---|
| 1 | `I2A_COMPLETION_REPORT.md` | `b7b3f8724c3a3dadee3f60800dcfad4ddf5436f668be44b1a80a5af2a473e3c1` |
| 2 | `I2A_HASHES.sha256` | `69705b5208fe9c6d378a7dbeb2b0fd4cf004b0d587e98d3bc99f378195147fd5` |
| 3 | `INFORMATION_ACCESS_RECEIPT.json` | `46e83a14a49c9f10aeb2f920d69111236287fde51a9aee1d479d4b1fd6dfd5aa` |
| 4 | `METHOD_CONTRACTS.json` | `5df3a6f409b38a8342bf16d83af94b0068517eb43db5e33aefb6dffafd5352b2` |
| 5 | `METHOD_OVERLAY_MATRIX.md` | `e3d1090230eab9d1111965d7ce3006cce15030d56c8721a51a946e181ac91a0b` |
| 6 | `PREPROCESSING_RECEIPT_SCHEMA.json` | `64f5d07f673c798689c5dec1f59a4aa0640f366f1ee1943e6908d60455e07165` |
| 7 | `README.md` | `b681a060b95eff7f0c5d1df76630169a5891a33bf116b8e5272e542105a7d6e0` |
| 8 | `SYNTHETIC_PREPROCESSING_RECEIPT.json` | `13316dec8249bb591c5fc24479cff4be92bd02c50b12034141a12153baa86c8b` |
| 9 | `TEST_REPORT.md` | `774a472230916b8b530b5ce5c10e4d5c50ed53e9b3fb79e8aceb2d37a453fc2f` |
| 10 | `__init__.py` | `20584740ebd1c120131c1ca71d8b9d6812790f28ca010bc46de209181dff9345` |
| 11 | `adapter_interfaces.py` | `5d0f2f7c14a6a21fc3ab3eade0bbdca2233e046a37aa3fd6b4486cb5f01afeed` |
| 12 | `information_boundary.py` | `6aeee51006a4722d0e02183386fa997b3628977039ed25a82443d110895f4c7a` |
| 13 | `synthetic_fixtures.py` | `9b2b94063d176cbd1ef789627a1f3c438bba0d767c9203de76caaa1856d1343f` |
| 14 | `test_i2a_contracts.py` | `379fac530c505521bf6b7a636e6217847aff34d78686649c825b1af7d4726966` |

`find` reports exactly **14 files** and **one** directory (flat, no subdirectories). **This audit report was not among them** — it did not exist before this pass.

### 2.2 Checksum coverage and self-check

`sha256sum -c I2A_HASHES.sha256` → **13/13 OK**. The 14th file, the manifest itself, is covered by the normalized self-check:

| Quantity | Value | Result |
|---|---|---|
| Recorded `SELF-NORMALIZED-SHA256` | `ed6f5de55d3abf35feda6bad3bcfd14235211a0040b67423e967801cbaf1fec5` | — |
| Recomputed from first principles (digest → 64 zeroes, hash whole file) | `ed6f5de55d3abf35feda6bad3bcfd14235211a0040b67423e967801cbaf1fec5` | **MATCH** |
| Digest occurrences | 1 | **unambiguous** |
| Matches reported `ed6f5de5…f1fec5` | — | **MATCH** |

Convention is identical to the I1 and zero-SD-addendum manifests. **All 14 files are checksum-covered**, as Codex reported.

### 2.3 Controlling documents

| Artifact | Result |
|---|---|
| Revision 3.1 plan | `701b4509…fe959` — unchanged |
| `I1_HASHES.sha256` manifest | **5/5 OK** (registration, schema, sealed predictions, freeze report) |
| `I1_ZERO_SD_RULE_ADDENDUM_HASHES.sha256` manifest | **3/3 OK** |
| `CLAUDE_I1_READONLY_AUDIT.md` | `bebf0d14…8471a` — unchanged |
| `CLAUDE_ZERO_SD_RULE_MEMO.md` | `2cbd27fb…8792c4` — unchanged |
| `CLAUDE_I1_ZERO_SD_ADDENDUM_CONFIRMATION_AUDIT.md` | `494bd27f…e6765` — unchanged |
| `CLAUDE_I1_ZERO_SD_ADDENDUM_RECONFIRMATION_AUDIT.md` | `c5ae30c2…9eef0` — unchanged |
| Sealed predictions | `82a7fcb3…50c3` — **unchanged, not amended** |
| Frozen track, ecological (52 files) | `2b3b8ae6…a884e` — unchanged |
| Frozen track, general (55 files) | `f90cea6f…6bdbd` — unchanged |

**Reconfirmation verdict confirmed** by direct read: `PASS — CORRECTION VERIFIED; READY FOR BOUNDED I2 INCREMENT A`.

### 2.4 No protected file modified; no stray I2A output

Narrow scoped checks, all exit 0 (worktree-vs-index and index-vs-HEAD): `src`, `configs`, `tests`, `results`, `provenance`, `docs/true_noisy_state_real_methods`. No I2A-named file exists anywhere outside the namespace. Zero `__pycache__`, `.pytest_cache`, `.pyc` or coverage artifacts in the namespace or repository, before or after my test reruns.

---

## 3. Static side-effect audit

Every executable file inspected line by line. Complete operation inventory:

| Module | Operation | Line | In scope? |
|---|---|---|---|
| `adapter_interfaces.py` | `np.random.default_rng(seed)` | 546 | **Yes** — seeded local Generator, no global RNG mutation |
| `adapter_interfaces.py` | *no filesystem, network, subprocess, or env operation at all* | — | **Yes** |
| `information_boundary.py` | `Path(path)` from caller argument | 300 | **Yes** — no discovery |
| `information_boundary.py` | `zipfile.ZipFile(supplied_path, "r")` | 309 | **Yes** — read-only |
| `information_boundary.py` | `archive.open(member, "r")` header read | 315 | **Yes** — header bytes only |
| `information_boundary.py` | `archive.read("metadata_json.npy")` | 330 | **Yes** — allowlisted member only |
| `information_boundary.py` | `np.load(BytesIO, allow_pickle=False)` | 331 | **Yes** — pickle disabled, in-memory |
| `synthetic_fixtures.py` | `np.savez(path, **archive)` | 79 | **Yes** — **the only write**, to a caller-supplied path |

Confirmed **absent** across all modules: `subprocess`, `os.system`, `popen`, `socket`, `urllib`, `requests`, `shutil`, `rmtree`, `os.remove/unlink`, `chdir`, `os.environ` mutation, `importlib`, `__import__`, `exec`, `eval`, `pickle`, `allow_pickle=True`, `glob`, `walk`, `listdir`, `iterdir`, any `git`/`slurm`/`sbatch`/`srun` reference, and any global-configuration mutation.

Against the required checklist, production adapter code:

- **does not discover repository datasets by path** — no glob/walk/listdir; every path is a caller argument;
- **contains no accepted-output or truth-array paths** — grep for `fs04`, `scratch2`, `quarantine`, `real_ecology_runs`, `public.npz`, `/home/` returns nothing in production code; the single `truth.npz` string in the test file is a *negative* fixture asserting rejection;
- **does not load real `states`, `next_states`, public arrays or datasets** — the only payload read anywhere is `metadata_json.npy`;
- **does not write outside a caller-supplied destination** — one write, one caller-supplied path;
- **does not submit jobs, call external services, mutate Git, or alter global configuration**;
- **does not import or monkey-patch `src/tracks/**`** — imports are limited to `numpy` and the stdlib (`copy`, `hashlib`, `io`, `json`, `zipfile`, `dataclasses`, `pathlib`, `typing`, `collections`);
- **does not silently fall back** — 69 explicit `raise` statements across the two production modules; only two `try` blocks exist, both narrowly wrapping JSON parsing and both re-raising as `BoundaryViolation … from exc`. No bare `except`, no `pass` fallback, no permissive `.get(..., True)` default.

---

## 4. Information boundary

`REGISTERED_KEY_ORDER` matches the real 13-key `truth.npz` schema exactly, and `FORBIDDEN_FIELDS` contains exactly the 10 non-allowlisted keys.

I exercised the gate against **all 13 registered fields**:

| Field | Result |
|---|---|
| `C`, `entry`, `initially_unsafe`, `next_regime`, `r_base`, `r_eff_true`, `regime`, `reward_true`, `safety_penalty_applied`, `theta` | **denied — "forbidden private field requested: `<exact name>`"** |
| `metadata_json` | denied unless capability `i2a_metadata_binding` **and** purpose `public_hash_binding` |
| `states` | **denied — "array access is not authorized in I2A"** |
| `next_states` | **denied** — runtime purposes give the specific future-information reason; all other purposes hit the I2A denial |

| Required property | Verdict |
|---|---|
| Metadata-only binding inspection cannot materialize scientific arrays | **CONFIRMED** — the inspector reads NPY *headers* (shape/dtype via `read_array_header`) and only the `metadata_json` payload; it returns headers plus the binding, never an array |
| Real state access is not exercised in I2A | **CONFIRMED** — every `states`/`next_states` request raises |
| `states` and `next_states` are separate future capabilities | **CONFIRMED** — `CAPABILITY_FUTURE_OFFLINE_STATES` and `CAPABILITY_FUTURE_OFFLINE_NEXT_STATES`, the latter carrying `offline_fitting_only: True`, both `authorized_in_i2a: False` |
| Runtime/deployment `next_states` access is structurally fail-closed | **CONFIRMED** — denied for `runtime`, `deployment`, `action_selection`, `planning_root`, and denied again by the catch-all |
| Every forbidden private field rejected by exact registered name | **CONFIRMED** — 10/10 |
| Missing/extra/wrong-dtype/wrong-unit/non-finite/misaligned/reordered/mis-bound inputs fail closed | **CONFIRMED** — `_validate_key_order` is an order-sensitive tuple comparison; per-field shape and dtype checks; `np.isfinite` on numeric fields; single-row-alignment-token requirement; unit must be `raw_abundance`; binding must match; the NPZ inspector additionally requires exact member names *and order* |
| Receipts cannot falsely claim metadata-only access after a scientific request | **CONFIRMED** — three denied requests were logged with `allowed: False`, `scientific_payload_loaded_fields` stayed `[]`, and `boundary_result` is derived from the three instrumentation counters rather than asserted |
| No exception path or alternate API bypasses the allowlist | **CONFIRMED** — neither public entry point returns scientific arrays; `validate_synthetic_archive` returns a summary dict only |

**Design observations, not defects.** (a) NPY *header* inspection of scientific members is not routed through `AccessController.request`; it is consistent with the I1 schema, which already registers every field's dtype and shape, so it discloses nothing new — but I2B should route it through the gate for uniform receipting. (b) `SyntheticProvenance.validate()` checks *declared* flags; a caller could supply real arrays with a false declaration. That is an unavoidable trust boundary for a synthetic-fixture stage and is correctly surfaced in the receipt rather than hidden.

---

## 5. Exact-state feature adapter

Verified from the implementation and then adversarially, not from tests.

| Requirement | Implementation | Probe result |
|---|---|---|
| Column 1 assigned literal `0.0` | `output[:, 1] = np.float64(0.0)` (line 163); validated by `values[:,1].view(np.uint64) == 0` | **PASS** — all rows bit-exact |
| Columns 2–4 exact copies of column 0 | `output[:, 2:5] = output[:, 0][:, None]` (line 164) | **PASS** — bit-exact |
| Primary path computes no weighted moments | no mean/variance/quantile computation anywhere in `adapt` | **CONFIRMED** by reading |
| `sd == 0.0` not used as detection | detection is by *assignment*; `std <= 1e-8` appears only in validators | **CONFIRMED** |
| `std <= 1e-8` validation-only | `validate_constructed_features` line 94, `passes_constant_validation_guard`, mask fitting | **CONFIRMED** |
| No epsilon variance, jitter, deletion, merging, reordering | shape/order enforced; jitter rejected | **PASS** — 1e-18, 1e-12, 1e-9 all rejected |
| dtype and shape enforced | `_require_float64` + `(n,10)` check | **PASS** |
| Masks/offsets/scale `1.0`/alias map/rank receipts agree | fitted mask `(F,T,F,F,F,T,F,F,F,T)` = columns 1, 5, 9; `scales[1]` bit-exact `1.0`; alias map `{2:0,3:0,4:0}`; rank 1 | **PASS** — matches the registered expectation exactly (col 1 new; cols 5 and 9 pre-existing) |
| Offline/runtime use the same registered rule | one `PreprocessorArtifact.transform` for both | **PASS**, and see round-trip below |
| History and context columns 7–9 preserved | only columns 0–5 written on a copy | **PASS** — columns 6–9 bit-identical; caller array unmutated |
| Wholesale OracleStateFilter-style history reset rejected | `validate_context_preservation` detects zeroed context | **PASS** (test 08 exercises it) |
| `sigma → 0` and observation-scale-to-zero rejected | `MIN_LIKELIHOOD_SCALE = 1e-12` guards both | **PASS** at 0.0 and 1e-15 for both parameters |

**Numerical-representation hazards, all checked:**

- **Signed zero** — `−0.0` has bit pattern `0x8000…`, so the `view(np.uint64) == 0` test rejects it. Probed: **rejected**, with the message "assigned SD must be literal positive float64 zero". This is a genuinely strong check.
- **Non-contiguous views** — `values[:, 1]` is strided; I confirmed same-itemsize `.view(np.uint64)` is valid on strided arrays and returns correct bit patterns.
- **Alias drift** — a 1-ulp perturbation of a single element in column 2 is **rejected**.
- **NaN / non-finite** — rejected up-front by `_require_float64`.
- **Mutable aliasing** — `adapt` copies the base array; columns 2–4 are assigned by value from column 0 (not a broadcast view), so later mutation of column 0 cannot silently propagate.
- **Degenerate corner** — if all abundances were zero, `log1p(0)=0` makes the whole state block zero and `matrix_rank` returns 0, failing the rank-one check. This fails *closed*, which is the safe direction, but I2B should register the intended handling of an all-extinct batch.

**Finding 5-A (low severity).** Inside `adapt`, `validate_context_preservation` is called with `action_history_before` and `action_history_after` bound to the *same object*, and with `before=base`/`after=output` where `output` is a copy whose columns 6–9 were never written. Both comparisons are therefore tautological and cannot fail. The guarantee still holds *structurally* (copy-then-write-0:6), and the function is genuinely exercised by test 08 with independent arrays — but the in-adapter call provides no regression protection, and the receipt's `context_parity: "PASS"` and identical before/after hashes are true by construction rather than measured. A future integrated adapter will not be a pure copy, so I2B must capture genuinely independent before/after snapshots.

---

## 6. Per-method contracts

`METHOD_CONTRACTS.json` records all six methods with ten fields each, plus seven `common_forbidden_shortcuts` and `scientific_artifacts_constructed: false`. Every element required by the brief is present:

| Requirement | Verdict |
|---|---|
| Consumed state interface | present for all six |
| Preserved history/context | present; PLUS/MOOR list action history and timing, the four general methods list previous/current observation, timestep, action and observation history |
| Point-mass behaviour where applicable | `direct_point_mass_belief: true` for PLUS/MOOR only |
| Survey-scale conversion where applicable | explicit formula `latent_state = raw_abundance / fitted_model.survey_scale`, then registered nearest latent bin |
| Preprocessing/refitting consequences | present for all six |
| Potentially changed learned artifacts | `possible_later_changes` per method |
| Frozen-fit eligibility | present |
| End-to-end-only conditions | present |
| Forbidden shortcuts | per method plus the common list |

Specific checks:

- **PLUS/MOOR do not use the public-feature zero-SD branch** — both record `public_zero_sd_preprocessing: "not consumed; branch is vacuous"` and `preprocessing: "not applicable"`. **Correct**, and consistent with my earlier finding that neither file references `public_features`.
- **Point-mass/bin conversion fully specified without inventing a boundary rule** — `adapt_point_mass_belief` requires a strictly increasing latent grid, a matching registered raw grid, and verifies `grid * survey_scale` is **bit-identical** to the registered raw grid before selecting `argmin(|grid − latent|)`. It invents no new tie-break or boundary convention; it defers to the registered discretisation and proves the mapping. Method membership is restricted to the two ecological ids.
- **RefPlan separates primary end-to-end from optional frozen-fit** — recorded in the contract, the overlay matrix and the eligibility logic.
- **RefPlan frozen eligibility requires byte-identical Arm O preprocessing** — enforced in code (`arm_o.preprocessor_sha256 == arm_t.preprocessor_sha256`) *after* the all-fields equality gate.
- **OGSRL records dataset-derived safety calibration** — `safety_budget: "dataset-derived; exact-state recalibration is a safety-axis change and a belief-only replacement cannot isolate state representation"`.
- **BA-MCTS and RefPlan record `residual_sigma` and posterior changes** — both carry `residual_sigma: true` and `model_posterior_sharpness: true`.
- **EVD marked for separate objective/reporting treatment** — `objective_status: "uses unmatched raw logged rewards; must be reported separately"`, with `pool with surrogate-reward methods` listed as forbidden.

**Prose-versus-enforcement check.** The one contract statement that *requires* code enforcement — `frozen_fit_eligibility: unavailable` for OGSRL, BA-MCTS and EVD — **is** enforced: `classify_interpretation` returns `eligible = False` for any method outside the RefPlan/PLUS/MOOR branches, and an unknown method name also falls to `False`. Verified by probe. The remaining contract fields are declarative design records appropriate to a design increment.

---

## 7. Interpretation gates — one defect

**Change detection is exhaustive and correct.** I enumerated all 9 `LearnedArtifactSnapshot` fields × 5 methods = **45 combinations**, changing one field at a time. In every case the result was `frozen_fit_eligible: False`, `causal_state_representation_label_permitted: False`, and exactly the label:

`MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY`

`residual_sigma_hex` tuple value changes and tuple *length* changes are both detected. No changed-artifact combination can retain a frozen-fit label. Per-method gating behaves correctly when nothing changed: RefPlan eligible on preprocessor identity; PLUS/MOOR eligible only with `direct_point_mass_replacement=True` (and `False` without it); OGSRL, BA-MCTS, EVD and any unknown method always ineligible.

### Finding 7-A — **fail-open on unknown or unpopulated artifact status (REVISE)**

Task 6 requires: *"unknown or missing artifact status fails closed rather than defaulting to eligible."* It does not.

`classify_interpretation` decides solely on **equality** between the two snapshots. It never validates that the fields are populated or well-formed. Probed with both snapshots set to placeholder values:

| Both snapshots contain | `frozen_fit_eligible` | Label emitted |
|---|---|---|
| `""` (empty strings) | **True** | `FROZEN-FIT/STATE-INPUT-ONLY` |
| `None` | **True** | `FROZEN-FIT/STATE-INPUT-ONLY` |
| `"unknown"` | **True** | `FROZEN-FIT/STATE-INPUT-ONLY` |
| `"TBD"` | **True** | `FROZEN-FIT/STATE-INPUT-ONLY` |

The same holds for PLUS/MOOR with `direct_point_mass_replacement=True`. This is realistic rather than contrived: a method with no ensemble would plausibly be given `ensemble_sha256=""`, and an integration that has not yet wired a hash source would naturally leave fields empty or `None`. The gate would then certify a frozen-fit, state-input-only claim on the basis of two equally empty records — precisely the "compliant-looking but unverified" failure mode the whole registration chain exists to prevent.

**Required change (small and local).** Before the equality comparison, validate each snapshot: every `*_sha256` field must be a populated 64-character lowercase hex string; `residual_sigma_hex` must be a non-empty tuple of parseable float hex strings; reject `None`, empty, and non-hex placeholders with `ContractViolation`. Add negative tests for `""`, `None`, `"unknown"`, and a short/malformed digest, for both the RefPlan and PLUS/MOOR branches.

### Finding 7-B — tautological parity inside the receipt builder (low severity)

`build_preprocessing_receipt` computes `transformed_fit = artifact.transform(values)` and `transformed_runtime = artifact.transform(values.copy())`, then reports `fit_runtime_parity_result: "PASS"`. Calling the same function twice on equal inputs demonstrates determinism, not fit/runtime parity across a serialize-and-reload boundary. I performed the genuine round trip myself — serialized via `to_bytes()`, reconstructed the artifact from the JSON hex payload, and re-transformed — and it **passes**: reloaded `sha256` matches and the transformed output is bit-identical. So the property holds; only the *evidence recorded in the receipt* is weaker than it appears. I2B should make the receipt's parity claim rest on a reload.

---

## 8. RefPlan receipt

`REQUIRED_REFPLAN_DISCLOSURE_FIELDS` contains all seven mandatory fields. Probed by removing each in turn:

| Omitted field | Result |
|---|---|
| `fit_time_sd_mean` | **rejected** |
| `fit_time_sd_std` | **rejected** |
| `planner_root_sd` | **rejected** |
| `arm_o_standardized_offset` | **rejected** |
| `arm_t_standardized_offset` | **rejected** |
| `species` | **rejected** |
| `arm_t_removes_preexisting_mismatch` | **rejected** |
| unregistered species (`bottlenose_dolphin`) | **rejected** |

The builder injects `tiger_reference_offset: -2.7018`, `fox_reference_offset: -2.7348`, the species-matched reference, an interpretation warning, and `disclosure_complete: True`. `build_preprocessing_receipt` refuses to emit unless `disclosure_complete` is set, and the emitted receipt carries both references plus the Arm O and Arm T offsets. All required fields are therefore mandatory, and the tiger/fox references are present at the registered values.

**Finding 8-A — no value sanity checking (low severity).** NaN offsets, a wrong-signed tiger offset (`+2.7018`), and a `planner_root_sd` inconsistent with the "hard-zero root" disclosure are all **accepted**. Strictly, the registered contract requires these values to be *disclosed and recorded*, not validated, so this is not a contract breach. But NaN has a concrete downstream consequence I verified: the emitted receipt serializes with a bare `NaN` token and **fails strict JSON reparse**, producing a non-interoperable provenance artifact. Recommend requiring finiteness on the four numeric disclosure fields, and — since the registered disclosure states the planner root SD is hard-zero — asserting `planner_root_sd == 0.0` or recording an explicit deviation reason.

---

## 9. RNG contract

Independently verified with synthetic execution:

| Property | Result |
|---|---|
| `rng.normal(loc, 0.0)` returns `loc` exactly | **PASS** — bit-exact via `view(np.uint64)` on a 3-element vector |
| RNG state advances | **PASS** — `bit_generator.state` differs before/after |
| Paired interfaces preserve call count | **PASS** — both recorders report 1 call |
| Paired interfaces preserve ordering | **PASS** — identical ordinals |
| **Decisive subsequent-draw test** | **PASS** — two generators seeded identically, one called with `scale=0.0` and one with `scale=0.25`, then each drawing 4 further values: the subsequent draws are **identical**, proving equal RNG consumption rather than merely equal immediate outputs |

This is the strong form the brief required. `RecordingRNG.state` deep-copies the bit-generator state, so comparisons cannot be defeated by aliasing.

---

## 10. Test-quality audit and independent rerun

**Rerun results.** From the temporary directory with `PYTHONDONTWRITEBYTECODE=1`:

- `python -m unittest discover` → **Ran 32 tests … OK** (0 failures, 0 errors, 0 skipped), 0.034 s
- `python -m pytest -q -p no:cacheprovider --rootdir=<temp>` → **32 passed**

Both agree with Codex's reported 32/32. Repository artifacts created: **zero** — verified `__pycache__`, `.pytest_cache`, `*.pyc` count is 0 in the namespace before and after, the namespace still holds exactly 14 files, and the I2A manifest still verifies 13/13.

**Coverage.** 32 tests, 80 assertions, four classes covering features (10), preprocessing (6), the information boundary (7) and methods/interpretation (9). Test names are ordinal-prefixed but the suite is **not order-dependent**: `unittest discover` and `pytest` execute in different orders (unittest ran the `MethodAndInterpretation` class before `Preprocessing`) and both pass.

**Strength assessment.**

| Concern | Finding |
|---|---|
| Tests that merely restate implementation constants | Test 31 checks that the receipt schema contains registered keys — largely a restatement, but appropriate as a schema-conformance check. The substantive tests assert *behaviour*. |
| Assertions that could pass despite a broken adapter | Test 07 (history/context) is weakened by Finding 5-A: because `adapt` copies and never writes columns 6–9, this assertion cannot fail for the current implementation. Test 08 compensates by exercising the validator with genuinely zeroed context. Test 13's parity assertion inherits Finding 7-B. |
| Missing negative/adversarial cases | **Yes, one material gap**: no test covers unknown/unpopulated artifact status in `classify_interpretation` (Finding 7-A). Also absent: negative zero on column 1 (the code handles it correctly — I verified — but nothing pins the behaviour), and a serialize/reload parity test. |
| Order dependence | None — confirmed by two runners with different orders. |
| Platform-sensitive assumptions | Bit-level `view(np.uint64)` comparisons assume IEEE-754 float64 and consistent endianness. Acceptable for the pinned x86_64 profile; worth an explicit note in I2B. `float.hex()` round-tripping is exact and portable. |
| Tests accidentally touching real repository data | **None.** No real path appears in any test. The single `"truth.npz"` occurrence is a negative-provenance fixture asserting rejection. `write_synthetic_npz` is called only with a `tempfile` path. |
| Fixtures possibly derived from real arrays | **No.** All fixtures are closed-form (`arange`, `linspace`, constants). The synthetic binding is the deliberately non-hash literal `"synthetic_public_dataset_sha256_" + "a"*64`, which cannot be confused with the real bindings `7e71172a…` or `688d580f…`. `test_02` derives the reduction residue analytically from constants rather than from data. |

---

## 11. Receipt consistency

| Cross-check | Result |
|---|---|
| Test count: completion report, test report, README, my two reruns | **32** everywhere — consistent |
| File count: manifest (13 + self), `find` (14), Codex's claim (14) | consistent |
| `test_i2a_contracts.py` hash cited in the completion report | matches the manifest and my computation |
| Constant threshold `1e-8`, comparison `<=`, runtime tolerance `1e-6` | identical in code, receipt schema, synthetic receipt, addendum and memo |
| Constant mask `[F,T,F,F,F,T,F,F,F,T]` (columns 1, 5, 9) | synthetic receipt matches my independently fitted artifact |
| `design_rank: 1`, `collinear_groups: [[0,2,3,4]]`, `maximum_absolute_assigned_sd: 0.0` | consistent with code and my probes |
| Label string | byte-identical across code, receipts, overlay matrix, contracts |
| RefPlan offsets `tiger −2.7018`, `fox −2.7348` | consistent across contracts, matrix, code, receipt, addendum, memo |
| Access receipt: `scientific_payload_loaded_fields: []`, `real_scientific_arrays_materialized: 0`, `real_metadata_artifacts_opened: 0`, `real_public_artifacts_opened: 0`, `boundary_result: PASS` | consistent with the code path and with my probes |
| Method eligibility: contracts vs overlay matrix vs `classify_interpretation` | consistent (PLUS/MOOR conditional; RefPlan secondary-only; other three unavailable) |
| Authorization status across README, completion report, contracts | consistent — integration, real data, and later stages all "not authorized and not performed" |

**Only disagreement found:** the completion report states the suite ran in 0.020 s while my rerun took 0.034 s. This is machine timing variance, not a substantive discrepancy. No disagreement in file counts, hashes, test counts, accessed metadata, schema rules, feature rules, eligibility, or authorization status.

---

## 12. Scope verdict and next boundary

| Question | Answer |
|---|---|
| Stayed within synthetic adapter work? | **Yes.** Zero real arrays, datasets or accepted artifacts opened; the only write is to a caller-supplied temp path; no integration with `src/tracks/**`. |
| Closes the interface-design questions without real data? | **Yes** — feature construction, preprocessing artifact and parity, point-mass/survey-scale conversion, the access gate, per-method contracts, the overlay matrix and the receipt schema are all concretely specified and exercised. |
| Safe to preserve as an audited implementation? | **Yes**, after Finding 7-A is fixed. Nothing must be discarded. |
| Ready for the next stage? | **After the 7-A fix**, yes — for design work only. |

### Recommended next increment

The four candidate activities must stay separated, and only the first should be authorized next:

1. **Design/registration work — recommend as the next single bounded increment (I2B-Design).** Fix Finding 7-A (fail-closed artifact-status validation plus negative tests); optionally address 5-A, 7-B and 8-A; and register the per-method overlay assignment, the integration-point specification for each of the six methods, and the Arm O/Arm T artifact-capture plan. **Still synthetic fixtures only; still no real arrays; still no integration.**
2. **External adapter integration** — not next. Must be preceded by an audit of the I2B-Design output confirming the eligibility gate fails closed and the integration points do not require editing `src/tracks/**`.
3. **Allowlisted real-state extraction** — not next. Must be preceded by an audit of the integration increment, and requires the separate truth-extraction authorization that I1 has withheld.
4. **Regression or parity testing** — after extraction, and gated on G0–G2.
5. **Scientific execution** — last, and separately authorized.

**Audit that must precede the increment after next:** an independent read-only audit of the I2B-Design deliverable, specifically re-probing `classify_interpretation` with unpopulated, malformed and partially-populated snapshots, and confirming the integration specification introduces no new information route.

**Authorization status of this audit.** Performed: read-only inspection and hashing of all 14 namespace files, all I1 and zero-SD artifacts, Revision 3.1 and all prior Claude audits; independent reproduction of the normalized manifest self-check; static side-effect analysis; adversarial synthetic probing of the adapter, boundary, eligibility gate, RefPlan disclosure and RNG contract; two independent reruns of the 32-test suite from a temporary directory outside the repository. Created exactly one file inside the repository: this report. **No real truth or public array, dataset or accepted artifact was opened; no method run; no model fitted; no adapter integrated; no execution namespace created; no policy evaluated; no rebaseline; no Slurm job; no Git object touched; no repository cache or bytecode artifact produced; no network access.** The single temporary directory `/tmp/claude-17635/.../scratchpad/i2a_audit_AVuoeE` is outside the repository; I verified its exact path before use and left it in the session scratchpad rather than deleting it, so this report's probe evidence remains inspectable — it contains only synthetic probe output and is not in the repository.

Post-audit re-verification: I2A manifest 13/13 OK, namespace still 14 files, I1 manifest 5/5 OK, both frozen-track source-only hashes unchanged, all narrow protected-scope Git checks exit 0.

---

## Verdict

**REVISE — I2A REQUIRES SPECIFIED CODE OR CONTRACT CHANGES**
