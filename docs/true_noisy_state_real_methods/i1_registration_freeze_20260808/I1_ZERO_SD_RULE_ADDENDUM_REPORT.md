# I1 zero-SD rule addendum freeze report

Date: 2026-08-08 (Australia/Melbourne)  
Registration: `method_level_state_information_pilot_i1_20260808`  
Mode: immutable documentation-only addendum  
Environment: `PYTHONDONTWRITEBYTECODE=1`, `LC_ALL=C`

## 1. Executive result

Claude's zero-standard-deviation memo defines an implementable rule, and that rule is
now frozen in additive human-readable and machine-readable registration artifacts. The
addendum distinguishes direct feature construction, validation, primary end-to-end
preprocessing, and RefPlan's optional frozen-fit secondary diagnostic. It also freezes
the RefPlan mismatch disclosure, RNG invariant, minimum future receipt schema, and
synthetic-only I2 test requirements.

No existing I1 artifact, controlling plan, audit, memo, accepted output, source, test,
configuration, result, or frozen track was edited. Sealed predictions remain unchanged.
This addendum authorizes no I2 work.

## 2. Controlling input verification

| Input | SHA-256 | Result |
|---|---|---|
| Revision 3.1 plan | `701b4509f1dc04b7885891be72e526420ac5558be56bdde2b9628789398fe959` | MATCH |
| `CLAUDE_I1_READONLY_AUDIT.md` | `bebf0d14b327b159fde9db30fc079d878765f25b10ab0f3529a7a7a01698471a` | MATCH |
| `CLAUDE_ZERO_SD_RULE_MEMO.md` | `2cbd27fbe39bdd5bd415f74eca1e80f84a74cfb85778263831088f0ae18792c4` | RECORDED AND USED |
| `I1_REGISTRATION.md` | `d877fbcf74db7b2a0005d74e19f168e52de9b60ee3df613646e3017d21acb4f3` | MATCH |
| `I1_REGISTRATION.json` | `d4a9b99af333ffc26bc681a7283447680230ad010f0d23b7c496f4edb66e28ff` | MATCH |
| `SEALED_PREDICTIONS.json` | `82a7fcb394c80e673ede04c5692db839821b06f23f752076e8cf34bb29da50c3` | MATCH; UNCHANGED |
| `I1_SOURCE_SCHEMA.json` | `4de0b45f347576d82465c9590c9caf98646cecb2d3c3d4afac07cfca414c20bc` | MATCH |
| `I1_HASHES.sha256` | `5c483370c35eebab21348a8b1a24e84e5f1cdc63b8a4f9f029251e37cff13b27` | MATCH |
| `I1_FREEZE_REPORT.md` | `e335019a6db7e4bebb02923d1f44e0f554fa22ac831c8cba5f65c9e9ea2201a3` | MATCH |

The original I1 checksum manifest verified 5/5 standard entries before creation. All
listed controlling inputs were rehashed after creation and remained byte-identical.

## 3. Addendum artifacts

Exactly four new files were created in the frozen I1 directory:

- `I1_ZERO_SD_RULE_ADDENDUM.md`
- `I1_ZERO_SD_RULE_ADDENDUM.json`
- `I1_ZERO_SD_RULE_ADDENDUM_HASHES.sha256`
- `I1_ZERO_SD_RULE_ADDENDUM_REPORT.md`

The checksum manifest has standard SHA-256 entries for the Markdown addendum, JSON
addendum, and this report. It covers itself with a normalized self-check: replace the
single recorded digest on the `SELF-NORMALIZED-SHA256` line with 64 zeroes and hash the
entire manifest. This is the same reproducible convention used by the original I1
manifest.

## 4. Frozen rule summary

- `public_features` column 1 is the affected SD feature; exact-zero detection is not a
  valid primary rule because reduction residue can reach approximately `4.441e-16`.
- The external Arm T constructor uses float64 and directly assigns column 1 to literal
  `0.0` and columns 2–4 as bit-exact copies of column 0.
- `std <= 1e-8` is an inclusive fail-closed validation guard, never the primary feature
  construction mechanism.
- End-to-end Arm T stores an explicit constant mask, float64 mean offset, scale `1.0`,
  and emits literal `0.0`; offline and runtime use one serialized transformation.
- Columns remain ordered and present. The alias group `[0,2,3,4]`, rank collapse,
  condition diagnostic, offsets, scales, masks and artifact hashes must be receipted.
- Any changed preprocessing or learned artifact requires
  `MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY`.
- PLUS and MOOR do not consume the public feature, so the rule is vacuous for their
  direct point-mass belief implementations.
- Only RefPlan's optional secondary diagnostic may use the frozen-fit branch, which must
  reuse the Arm O preprocessing artifact byte-for-byte.
- RefPlan's pre-existing variable-fit/hard-zero-runtime mismatch and approximate tiger
  and fox offsets (`-2.7018`, `-2.7348`) must be disclosed for both arms.
- Synthetic future I2 tests must verify that `rng.normal(loc, 0.0)` returns `loc`
  exactly while preserving intended paired RNG advancement and call count.

## 5. Integrity validation

Pre-correction identities for the three authorized files were recorded before editing:

| Authorized file | Pre-correction SHA-256 |
|---|---|
| `I1_ZERO_SD_RULE_ADDENDUM.json` | `3439d33c4a10c68e1c420576b86868c7e60d5a52494ddb6d3655f0f503267d54` |
| `I1_ZERO_SD_RULE_ADDENDUM_REPORT.md` | `cbeda75c157a93bf9eee1d27b90b7c41c4c4071d471f84b731a3073c4af10ceb` |
| `I1_ZERO_SD_RULE_ADDENDUM_HASHES.sha256` | `299dfb2ef4aa7286ba598b5ed963ef06dfcb318b1ef5e60aa5f17691bd8c4d5c` |

| Check | Result |
|---|---|
| Addendum JSON | PASS — parsed by `python -m json.tool` with bytecode disabled |
| Original I1 hashes | PASS — unchanged before and after creation |
| Revision 3.1 and both Claude controls | PASS — unchanged |
| Historical ecological track | PASS — 149 files; `951365d7874a417d7e66b14538dc275a9f325ac4643df8ea1af4d1d24877fb01` |
| Historical general track | PASS — 162 files; `614524d7b058418a3ff3f370e7e5b57582213c2c518766d8d88f1ec093e7a35e` |
| Source-only ecological track | PASS — 52 files; `2b3b8ae6d2f8ff5ffb17c4885ded9e8f1f6b3c0cb662f393186fe4b4706a884e` |
| Source-only general track | PASS — 55 files; `f90cea6f28dcacb910b5e036bf9e09958715d00a2418fd0856a3d5a12856bdbd` |
| Source-only exclusions | PASS — `__pycache__`, `.pytest_cache`, `*.pyc`, and `*.pyo` excluded |
| Protected target scopes | PASS — no tracked change under controlling docs, `src`, `configs`, or `tests`; index has no staged changes |
| Git untracked report | PASS — 20 documentation paths only: 18 pre-existing plus the two addendum files present before report/manifest creation; final count is 22 after all four files |
| Scientific arrays or datasets written | NONE |
| Execution namespace created | NONE |
| Code, test, or configuration changed | NONE |
| Tests, fits, regressions, rebaselines, or evaluations run | NONE |
| Slurm jobs submitted | NONE |

The final ordinary `git status --short --untracked-files=all` completed and reported only
documentation paths.

A repository-level Git bus error (SIGBUS, exit 135) occurs deterministically on any
recursive object walk — `git ls-tree -r HEAD`, `git cat-file --batch-all-objects`,
`git rev-list --objects --all`. It is reproducible, not transient. The object store
holds 4,365 loose objects and a 4-object pack, so the fault is in loose-object reading,
not packed content. Recursive walks fault for `src` (specifically `src/tracks/general`),
`provenance` and `archive`; they succeed for `src/tracks/ecological`, `configs`,
`tests`, `docs` and `results`. A stray zero-byte `.git/objects/1e/tmp_obj_j26OYJ` and a
persistent `stat -f` "Network is down" on this filesystem indicate an interrupted write
on a network mount rather than logical repository corruption.

The interrupted-write/flaky-network-mount explanation is a supported diagnostic
hypothesis, not a proven causal conclusion. The earlier EIO encountered while writing
Claude's zero-SD memo is additional supporting evidence.

This does not affect the integrity conclusion. SIGBUS terminates the process and can
never yield a false "clean", so every check that completed is valid: `git diff
--exit-code`, `git diff-files --quiet` and `git diff-index --cached --quiet HEAD`
return 0, including when restricted to `src/tracks/general` and `provenance`, because
tree-diff prunes on OID equality without descending into the faulting objects. All
content integrity is independently established by working-tree SHA-256, which does not
use Git.

The targeted cached tree comparison for `src/tracks/general` completed successfully
through OID/tree equality without descending into the faulting objects. The registered
ecological, general, provenance, and other controlling source hashes reproduced in
Claude's audit. The Git-history fault touches a protected frozen-track scope and must not
be described as unrelated.

Registered consequence: `git fsck`, `git gc`, `git clone`, `git archive` and full
`git diff HEAD` over `src`/`provenance` are expected to fail. Frozen-track immutability
must be proven with the registered source-only SHA-256 recipe, never via Git history.
No repair, repack, garbage collection or prune is authorized under this addendum.
Full `git ls-tree`, archive/clone operations, and broad Git diffs over affected scopes
must likewise not be used as I2 integrity gates until separately repaired. Future stages
must use the registered source-only SHA-256 recipes and only narrowly scoped checks that
complete successfully. Repository repair is outside I2 and requires separate
authorization following a filesystem-health assessment.

This repository condition does not invalidate the zero-SD rule or the sealed
predictions.

## 6. Authorization status

- Original I1 registration: frozen, independently audited, and unchanged.
- Zero-SD addendum: frozen as documentation only.
- Sealed predictions: unchanged.
- I2 and I2 Increment A: not authorized by this addendum.
- Real truth-array access, extraction, and derived Arm T data: not authorized and not
  performed.
- Adapter, preprocessing, source, test, and configuration implementation: not authorized
  and not performed.
- Regression, rebaseline, Stage B, Stage C, and scientific evaluation: not authorized
  and not performed.
- Slurm jobs: none submitted.
- Accepted outputs and frozen tracks: untouched.

PASS — I1 ZERO-SD PROVENANCE CORRECTION COMPLETE; READY FOR CLAUDE RECONFIRMATION
