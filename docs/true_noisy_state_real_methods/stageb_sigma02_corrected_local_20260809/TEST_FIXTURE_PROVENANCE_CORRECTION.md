# Corrected Stage B test-fixture provenance correction

This tracked correction supersedes only the interpretation of the corrected Stage B test-suite
receipts at commit `c86f9cee797fdb4d39674cac85ee0047060a4947`. It does not rewrite or delete any
historical report, log, receipt, registration, key, fitted artifact, or evidence namespace.

## What the historical result established

The reported 227/227 whole-directory runs were factually accurate. Pytest collected
`tests/test_driver.py` before `tests/test_registration.py`; importing `test_driver.py` globally
replaced `tests.conftest.complete_bundle` with a V2 wrapper. The same committed source therefore
passed as a whole directory while `test_registration.py` failed 13 tests when selected alone.
Those historical runs established health only under that import order. They did not establish
isolated registration-test health or order independence.

The wrapper also replaced every synthetic task's `evaluation_identity_sha256` with the driver's
constant. That made the synthetic manifest incapable of representing the real stale-identity defect
identified by the DRIVER_INPUTS blocker audit.

## Bounded correction

- `tests/conftest.py` now constructs the interpreter-bound V2 registration fixture directly,
  including the committed driver hash and complete registered interpreter bindings.
- The normal task rows receive the canonical synthetic evaluation identity when they are created.
  No test module silently rewrites task identities after construction.
- `tests/test_driver.py` no longer imports or replaces `conftest.complete_bundle`.
- `tests/test_driver.py` also no longer imports or mutates `test_orchestration_slurm`. The V2-only
  synthetic receipt adaptations now run as reversible, test-scoped fixtures from `conftest.py`, so
  the orchestration module also passes when selected alone.
- Registration tests prove importing `test_driver` preserves both the identity and behavior of the
  shared factory, statically reject assignment to that factory from any test module, and prove a
  deliberately stale evaluation identity remains stale until the production runtime driver rejects
  it.

## Scientific and evidence impact

This is test scaffolding and provenance only. Production `registration.py`, `driver.py`, the
DRIVER_INPUTS contract, `src/tracks/**`, accepted inputs, fitted objects, replay recipes, and
interpreter bindings are unchanged. Registration freeze/runtime enforcement remains in force.
The successful 36 fit/replay receipts do not import the test package and remain valid pre-execution
integration evidence. No scientific evaluator was constructed, no private or truth data was
accessed, no scientific job was submitted, and no return was calculated.

The earlier reports and their 227/227 statements remain preserved. Their corrected interpretation
is: **accurate whole-directory results from an order-dependent suite, not proof of isolated module
health**.

## Required verification

The correction is accepted only if fresh processes pass `test_registration.py` alone,
`test_driver.py` alone, both explicit module orders, the complete corrected directory, the I2A and
fast-track suites, lint/format, strict JSON, shell syntax, normalized source-manifest coverage,
frozen-track hashes, and the audit reproduction script without its former isolated/full-suite
discrepancy.

The verified corrected selection matrix is:

| Fresh-process selection | Result |
|---|---:|
| `test_registration.py` alone | 31/31 passed |
| `test_driver.py` alone | 27/27 passed |
| driver then registration | 58/58 passed |
| registration then driver | 58/58 passed |
| every corrected `test_*.py` module alone | 10/10 modules passed |
| complete corrected directory | 231/231 passed |
| I2A | 38/38 passed |
| fast-track | 15/15 passed |

Ruff lint and format, strict JSON 11/11, shell syntax 5/5, normalized source-manifest coverage,
and both frozen-track hashes also pass. Re-running the audit reproduction script now yields green
whole-directory, isolated-registration, and driver-plus-registration selections; its previous
selection discrepancy is absent.
