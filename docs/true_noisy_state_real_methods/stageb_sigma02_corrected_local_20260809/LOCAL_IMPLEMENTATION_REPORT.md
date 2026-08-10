# Local corrected Stage B implementation report

Date: 2026-08-10 (Australia/Melbourne)

Status: **LOCAL ARM64 DEVELOPMENT TEST — NOT SCIENTIFIC EVIDENCE**

## Repository and source gate

- Physical repository: `/Users/hphu0007/Temporary_Code/DeepRL_Population_Models`.
- Development branch: `corrected-stageb-sigma02-local`.
- Controlling starting commit: `e44c5931f97f73f4805ac128bc18e904ca68469f`, parent
  `77cd38adb11970d56ce4c96bf84de15fac3108ac`, fetched from
  `origin/e1-phase1-parity`.
- Local `main` remained at `3291eefbe64b5ab19019120bb0ba34a0a9586a53`.
- Revision 3.1 SHA-256: `701b4509f1dc04b7885891be72e526420ac5558be56bdde2b9628789398fe959`.
- I2A corrective audit SHA-256:
  `e62682d3756a39215578c77317b995f88416c9cbea2b581b6f69d4f665541a87`.
- Frozen ecological/general source-only hashes reproduced as `2b3b8ae6…a884e` and
  `f90cea6f…6bdbd` before and after implementation.
- `provenance/frozen_tracks.sha256` remains `319cd428…caaf`.
- The independent external audit remained unchanged at `b93f9644…d9e67`.
- No applicable `AGENTS.md` exists.

Post-commit clean-clone validation found that the original synthetic registration fixture had
captured the pre-commit `HEAD` (`e44c5931…`) as a literal. The bounded correction replaces that
literal with a fail-closed runtime `git rev-parse HEAD` plus commit-object verification. It does
not change the production registration validator or weaken its equality requirement.

The quarantined directories
`stageb_sigma02_corrected_20260809/` and
`stageb_sigma02_corrected_recovery1_20260809/` were enumerated only to confirm their
presence. Their file contents were not opened, imported, copied, executed, tested,
modified, moved or deleted.

## Local environment

- Host: Apple Silicon M4 Pro, Darwin ARM64 (`arm64`).
- macOS system Python: 3.9.6; not used for implementation tests.
- Isolated interpreter: Homebrew Python 3.10.20 in ignored `.venv-stageb-local`.
- Test dependencies: NumPy 2.2.6, PyYAML 6.0.3, pytest 9.1.1 and Ruff 0.12.12.
- CPU-only. MPS/GPU was not used.

Homebrew installation performed the requested network installation of Python 3.10 and
its dependencies. Homebrew's automatic cleanup also removed an unneeded Homebrew
`ripgrep` formula. This was an environment-side effect outside the repository; repository
content was unaffected.

## Implementation created

All changes are confined to
`docs/true_noisy_state_real_methods/stageb_sigma02_corrected_local_20260809/`:

- `registration.py`: exact six-part freeze, HMAC-sealed non-serializable process capability,
  complete-episode return sink and direct-construction rejection;
- `artifacts.py`: strict method/component fitted-state schemas, registered expected hashes,
  fresh-object semantic parity, cell/source/consumer-bound matched surrogate, complete
  ecological policy/interface identity and definitionally absent EVD surrogate handling;
- `diagnostics.py`: registered member counts and identities, phase-correct Arm-specific
  transition evidence, complete 51-point realized BA-MCTS posterior, 50-step RefPlan
  dispersion and activity receipts rebuilt from their complete action sequences;
- `evidence.py`: evaluator-only per-step evidence, RNG continuity and exact return/component/
  collapse reconstruction;
- `boundary.py`: recursively typed public allowlists, closed belief attributes/diagnostics,
  current-state-only runtime view, context/RNG preservation and a strict task receipt that
  leaves paired O/T interface identity pending until Arm T exists;
- `publication.py`: unique staging, descendant-only paths, full-tree coverage, exclusive
  no-retry lock and final-operation atomic rename;
- `orchestration.py`: immutable canonical Arm O receipts checked against frozen task/code
  hashes and referenced published evidence; strict dispatch to registered diagnostics and
  boundary validators; mandatory published real-object bytes, component coverage and
  independently replayed action/prediction parity; Ed25519-authenticated cross-process gate
  loading, sealed Arm T process capability and identity-validating finalizer;
- `real_artifacts.py`: allowlisted lossless serialization of complete real frozen fitted
  instance graphs, fresh-object reconstruction and real prediction/action plus post-call parity;
- registration templates, closed formal schemas, four Slurm role templates plus dependency-chain
  template, normalized-self-covering source/test manifest, and 131 synthetic/adversarial tests.

No file under `src/tracks/**`, accepted result/configuration namespace, existing audit,
failed attempt, slide or remote was modified.

## Tests and results

All commands used `PYTHONDONTWRITEBYTECODE=1`; inherited tests additionally used `LC_ALL=C`
and cache-free external temporary roots.

| Suite | Runner | Result |
|---|---|---:|
| audited I2A | pytest | 38/38 passed |
| audited I2A | unittest | 38/38 passed |
| audited fast-track | pytest | 15/15 passed |
| audited fast-track | unittest | 15/15 passed |
| corrected local suite | pytest | 131/131 passed |
| corrected package | Ruff 0.12.12 | clean |

The corrected run emitted NumPy runtime warnings from the frozen general-track behavior-model
matrix product while exercising the tiny RefPlan parity fixture on ARM64. The gate still
reconstructed identical finite action and post-call state receipts and all tests passed. The
warning is local synthetic-fixture behavior, not scientific evidence; Xeon/accepted-artifact
parity remains mandatory and fail-closed.

The corrected suite reproduces every first-pass independent probe: forged registration and
Arm-T tokens, arbitrary return mappings, placeholder/history-only fitted state, altered
parameters under unchanged expected hashes, cross-cell/source surrogate rebinding, nested
evaluator fields and private belief diagnostics, incomplete/mismatched posterior and action
series, O/T RNG/innovation mismatch, non-boolean/reverting collapse evidence, publication
path traversal/symlinks/extras and failure at every prepublication fsync. It also covers the
original malformed registration, residual, exact-return, context, sigma/Oracle, activity,
M3 parity and Slurm dependency cases.

The second corrective pass additionally covers wrong controlling source/configuration hashes,
cross-cell dataset-hash reuse, int32 exact-state coercion, contradictory ecological conversion
indices, complete real frozen dynamics reconstruction/prediction, referenced published Arm O
evidence, and authenticated cross-process gate receipt tampering. Synthetic fitted instances of
all six actual frozen policy classes now reload into fresh objects and execute their real `act`
methods with output, RNG/state, and shared-object-identity parity. The Arm O gate dereferences,
reloads, and validates every method/component artifact in the atomically published task tree.

The third corrective pass removes the last permissive orchestration fixtures. The complete
Arm O gate now rejects arbitrary `synthetic_probe`/unrecognized diagnostic maps and accepts
only semantically rebuilt transition, activity and method-specific receipts plus the strict
information-boundary receipt. Every task bundle must publish the actual frozen policy graph,
its operation input and parity receipt, and an exact registration/cell/public-view/logical-
component binding. The gate reloads the real class and replays parity itself; missing evidence,
CanonicalArtifact-only stand-ins and well-formed but fabricated parity claims all fail closed.
The test fixture uses tiny instances of all six actual frozen policy classes rather than a
canonical stand-in at the Arm O gate.

The post-commit corrective pass adds direct regressions for current-`HEAD` binding, rejection
of the stale pre-commit SHA, unavailable or failed Git, malformed or additional Git output,
and a formatted 40-hex value that does not identify a commit. The tests derive identity from
the repository being tested, so an ordinary future commit does not require another embedded
SHA edit.

The artifact-contract V3 pass is source-backed by fit-only probes of all six methods in both
cells. It removes the fabricated requirements for distinct general-method preprocessing and
feature-transformation objects, PBVI alpha vectors, and a MOOR candidate prior. It replaces
the single generic fixture with an exact component-keyed mapping derived from each serialized
native feature order, fixes the registered row count at three, requires exact float64 bytes,
and separately binds the fixture manifest. Component hashes remain the bundle identity and
the complete real-object graph reload/replay gate is unchanged. The schema now rejects V2
artifact evidence. `ARTIFACT_CONTRACT_V3.md` contains the full applicability and migration
contract; fit-only receipts are sealed outside the repository development clone.

`SOURCE_TEST_HASHES.sha256` covers every other candidate file with a standard entry and
covers itself with `SELF-NORMALIZED-SHA256`, calculated after replacing the one recorded
self-digest with 64 zeroes. The corrected suite independently reproduces both coverage and
the normalized self-check.

## ARM64-unavailable checks and unresolved deployment gates

The following were deliberately not attempted locally:

- real accepted public/derived dataset binding and hashes;
- loading or fitting accepted-data Ricker/PBVI/general-method artifacts;
- real matched surrogate reconstruction or accepted-historical-surrogate parity;
- access to evaluator-private truth arrays;
- paper-faithful PyTorch fit/runtime behavior;
- accepted per-episode/action/event Arm O parity;
- Xeon 8452Y floating-point, BLAS, RNG, runtime and artifact-byte equivalence;
- Slurm parsing/submission or any M3 contact;
- scientific evaluation or return generation.

M3 must confirm that every real frozen-track fitted object is externally accessible and can
be serialized by `real_artifacts.py` without unsupported state or parity failure. The
real interpreter-dispatch driver, actual output roots and task-local accepted input bindings
remain deployment configuration, not Mac evidence. Issued registration and Arm-T objects
remain process-local and non-serializable; the implemented Ed25519 receipt verifier
revalidates the embedded registration and completed gate and then issues a new local token.
M3 must provision an owner-only 32-byte signing seed to the gate and its distinct public key
to Arm T, then audit the driver wiring. Any inaccessible fitted state or
reload/action-parity failure is a hard gate, not permission to use training history or
invent an identity.

The external `STAGEB_DRIVER` that dispatches the Slurm role templates remains an explicit M3
deployment input. Its absence is disclosed and does not weaken the locally tested signed
receipt loader/verifier; the final M3 audit must inspect the concrete driver wiring.

## Review status

The bounded driver pass adds the tracked four-role `driver.py`, mandatory registration-time
driver-byte binding, source-backed EVD transition inapplicability, atomic per-task publication,
and synthetic/adversarial driver coverage. The corrected suite now collects 192 tests. The
driver has no fit path; it loads and revalidates V4 fit-only publications and keeps Arm T
unreachable until the authenticated Arm O gate has been verified with a public key.

The local source-like infrastructure is complete and synthetically tested. It is ready for
an ordinary driver commit and clean-clone M3 fit-only validation, but it is not ready for
scientific execution until the replacement registration, keys and no-submit dependency package
are prospectively frozen and independently audited.
