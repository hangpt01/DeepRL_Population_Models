# DRIVER_INPUTS and registration-contract correction

This change makes the corrected Stage B runtime inputs a prospective, registration-bound
contract. `canonical_plan.py` is the single evaluation-identity and artifact-plan recipe used by
both registration freeze and driver execution. Freeze now rejects a stale evaluation identity or
an artifact plan that cannot be reconstructed from the registered sealed fit-probe sources.

`driver_inputs.py` deterministically produces and validates canonical
`corrected_stageb_driver_inputs_v2` bytes. The frozen registration contains the complete descriptor,
its exact canonical SHA-256, and the hashes of both the producer and JSON schema. The descriptor
uses the stable `registration_id`; this avoids a circular hash while the driver still requires byte
equality with the descriptor embedded in the exact frozen registration capability.

The input capability is split three ways:

- the policy view contains only the repository, 12 frozen fit probes, and two public datasets;
- evaluator-only inputs contain 12 ordered `(cell, method)` accepted-parity CSV bindings;
- gate-only inputs contain the inherited and corrected test receipts.

Arm T is registered for the exact `current_abundance` field only. Runtime `next_states`, refitting,
truth-archive policy access, and unregistered descriptor keys are rejected. Prerequisite staging
creates an owner-only output root containing exactly `DRIVER_INPUTS.json`; unknown top-level files,
pre-existing role namespaces, symlinks, and retries fail closed.

The fixture-correction commit immediately preceding this implementation changed
`CORRECTED_TEST_COUNT` from 227 to 231 as its sole production change. This implementation adds 18
real-file and adversarial DRIVER_INPUTS cases, so the complete corrected suite contains 249 tests
and necessarily updates that gate constant from 231 to 249. Historical 227/227 evidence and later
231/231 fixture-correction evidence remain preserved and are not reinterpreted as 249-test runs.

No frozen track source, scientific fitted artifact, prior evidence namespace, evaluator result, or
return is changed by this implementation.
