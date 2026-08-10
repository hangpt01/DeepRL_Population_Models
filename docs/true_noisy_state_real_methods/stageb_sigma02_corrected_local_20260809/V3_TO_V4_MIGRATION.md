# Artifact Contract V3 to V4 migration

V3 remains immutable historical evidence. V4 is a forward-only schema migration.

| Area | V3 | V4 |
|---|---|---|
| Ecological cache | lossy scalar rate/capacity-style projection | complete candidate-wise `MechanisticModel` projection, schema `corrected_stageb_ricker_fit_cache_v2` |
| Ecological scale | ecological `residual_sigma` with an apparent shared floor | exact unfloored `process_scale`, cache-bound and byte-identical |
| General scale | `residual_sigma >= 0.02` | unchanged, explicitly scoped to general learned dynamics |
| Diagnostics | one apparent global scale rule | method-scoped V2 receipt with ecological `below_general_floor` and null applicable floor |
| Artifact evidence | `corrected_stageb_artifact_bundle_evidence_v3` | `corrected_stageb_artifact_bundle_evidence_v4`; mutual rejection |
| Fit probe | external V3 probe receipts | one tracked V4 probe with corrected names, actor projection, call signature, source-first diagnostic, and atomic publication |

There is no in-place receipt upgrade. No V3 cache, process-scale component, diagnostic,
artifact-evidence receipt, or probe result may be relabelled V4. Future deployment must
regenerate all twelve probe receipts from the committed V4 script and independently bind
real accepted fitted objects.

Unchanged protections include component-specific deterministic fixtures, matched Arm-O
surrogate identity, PLUS/MOOR shared O/T fitted-policy identity, complete real-object
serialization/reload/replay, runtime current-state-only access, and the prohibition on truth,
future state, hidden evaluator information, and returns during fit probing.
