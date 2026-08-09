# I2 Increment A — external exact-state adapter contracts

Status: bounded implementation complete only after the accompanying synthetic test and
integrity reports pass. This namespace is external to `src/tracks/**` and is deliberately
not integrated with any accepted method.

## Scope

This package implements only caller-supplied interfaces, fail-closed contracts, and
synthetic fixtures for the registered exact-state design. It does not discover or open
scientific datasets. The archive inspector accepts a path explicitly and reads NumPy
headers plus the allowlisted `metadata_json.public_dataset_sha256` binding only; it has no
API that returns `states`, `next_states`, or another private payload.

The feature adapter copies a supplied ten-column float64 public-feature matrix, replaces
only columns 0–5, directly assigns column 1 to literal positive `0.0`, makes columns 2–4
bit-exact aliases of column 0, and verifies columns 6–9 plus caller-supplied action and
observation histories remain unchanged using independently captured before/after
snapshots, including interface shape, dtype, and feature order. Positive registered sigma and observation scale
are mandatory; this is not an `OracleStateFilter` wrapper and has no zero-likelihood
route.

The preprocessing implementation is synthetic-only. It records explicit constant masks,
float64 offsets, literal scale `1.0` for masked features, the alias map and rank collapse.
Its runtime transform reloads a distinct object across a canonical strict-JSON
serialization boundary, validates artifact/hash/schema/mask/offset/scale/alias/rank
parity, and emits literal zero for masked columns. No existing Arm O preprocessing is
edited.

Frozen-fit eligibility for adapted PLUS, adapted MOOR, and RefPlan fails closed unless
both arms supply all eight registered learned-artifact identities as exactly 64
lowercase hexadecimal SHA-256 values and a nonempty tuple of canonical finite float64
`residual_sigma` hex values. Identity equality is considered only after validation and
does not replace the method-specific direct-point-mass or explicit RefPlan-secondary
condition. RefPlan mismatch disclosures use strict JSON and enforce the registered
species offsets and internally consistent mismatch-removal semantics.

## Files

- `information_boundary.py`: 13-key schema, capabilities, access instrumentation, and
  header/metadata-only NPZ inspection.
- `adapter_interfaces.py`: exact feature adapter, preprocessing artifact, PLUS/MOOR
  point-mass adapter, interpretation labels, RefPlan disclosure, and RNG recorder.
- `METHOD_CONTRACTS.json`: immutable six-method overlay contracts.
- `synthetic_fixtures.py`: deterministic values generated independently of real arrays.
- `test_i2a_contracts.py`: synthetic-only unit and contract tests.
- `PREPROCESSING_RECEIPT_SCHEMA.json`: mandatory future receipt schema.
- `METHOD_OVERLAY_MATRIX.md`, receipts, test report, completion report, and checksum
  manifest: audit artifacts.

## Authorization boundary

No real state, next-state, public dataset, accepted cache, model, environment, evaluator,
or private field is read. No model, reward surrogate, normalizer, ensemble, policy, or
safety calibration is scientifically fitted. No method integration, regression,
rebaseline, Stage B/C evaluation, Slurm work, or Git repair is authorized or present.

Tests must be run from a fresh external temporary directory, with the namespace on
`PYTHONPATH`, bytecode and pytest cache disabled, and `I2A_TEST_TMPDIR` set to that exact
external directory. Both runners are required:

```text
cd /tmp/<fresh-i2a-directory>
PYTHONPATH=<absolute-i2a-namespace> I2A_TEST_TMPDIR=$PWD PYTHONDONTWRITEBYTECODE=1 LC_ALL=C \
  python -m unittest discover -s <absolute-i2a-namespace> -p 'test_i2a_contracts.py' -v
PYTHONPATH=<absolute-i2a-namespace> I2A_TEST_TMPDIR=$PWD PYTHONDONTWRITEBYTECODE=1 LC_ALL=C \
  python -m pytest -q -p no:cacheprovider --rootdir=$PWD \
  <absolute-i2a-namespace>/test_i2a_contracts.py
```
