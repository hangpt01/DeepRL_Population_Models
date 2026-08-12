# Parity eligibility and failure-diagnostic correction

This change is a prospective, fail-closed correction to the accepted-parity evidence
contract. It does not alter a scientific method, hyperparameter, evaluator, reward, seed,
horizon, fitted object, frozen track, or any prior scientific/evidence namespace.

An accepted-parity baseline can gate success only when its versioned binding identifies the
producer registration and `DRIVER_INPUTS`, controlling commit and normalized source manifest,
driver, RNG-v2 contract and effective noise, interpreter and NumPy runtime, fitted object,
component set and artifact plan, evaluator/reward/configuration, evaluation identities,
horizon, discount, and an explicit comparison contract. The 2026-08-08 fast-track canaries
lack those execution bindings and are classified `LEGACY_INELIGIBLE`; they are preserved but
cannot be silently upgraded or used to authorize Stage B success.

Eligible evidence retains exact comparison for registered identity/non-numeric fields and an
absolute tolerance of `1e-9` for the explicitly registered numeric-field set. Reference-header,
episode-count, missing-field, invalid-number, and non-finite-number discrepancies fail closed.
Relative deltas use `abs(observed-reference) / max(abs(observed), abs(reference))`, with the
all-zero case defined as zero; non-finite operands are invalid.

Before raising for baseline ineligibility, task-binding mismatch, or parity mismatch, Arm O now
atomically publishes a checksum-covered, collision-safe failure diagnostic under the dedicated
`arm-o-failure-diagnostics` namespace. This namespace is not an accepted scientific output
namespace: its documents say `FAIL`, `accepted: false`, `qualifies_as_task_result: false`, and
`automatic_retry_permitted: false`; they contain no success receipt. Raw truth-bearing values
are never emitted. Field names, canonical value hashes, and permitted absolute/relative delta
magnitudes provide durable diagnosis without exposing truth trajectories, runtime
`next_states`, thresholds, or returns. Artifact-only staging remains failed-attempt provenance.

The driver-input schema is v4, its registration binding is v3, the combined receipt schema is
v3, and the accepted-parity pass receipt is v2. Older driver-input documents cannot satisfy the
new mandatory baseline binding. The corrected local test count is 346.

The obsolete array `59027305` and its downstream jobs are historical failed-attempt evidence.
The downstream gate `59027306`, Arm T `59027307`, and finalizer `59027308` were terminalized
without retry or scientific submission after their dependencies became permanently
unsatisfiable; their separate checksum-covered terminalization namespace is preserved outside
the repository.
