# Artifact Contract V4 implementation report

Date: 2026-08-10 (Australia/Melbourne)

Development base: `e7c99b3a1bb95bc2d631d42b9349fc666d244d1a` on
`corrected-stageb-sigma02-local`.

Development clone:
`/fs04/scratch2/ce25/DeepRL_Population_Models_artifact_contract_v4_dev_20260810`.

Development evidence root:
`/fs04/scratch2/ce25/stageb_artifact_contract_v4_dev_20260810`.

## Implementation outcome

The canonical ecological fitted cache now preserves every registered field of the real
`MechanisticModel` as candidate-wise float64 data and binds the real parameter hashes. The
ecological scale component and diagnostics preserve exact nonnegative process scales without
the general-method floor. General learned dynamics retain their 0.02 residual-sigma floor.
Artifact evidence and analysis schemas are versioned forward and mutually reject V3/V4.

The tracked V4 fit probe has twelve closed task identities, projects caller-supplied fitted
objects only, writes a source diagnostic before validation, validates complete canonical and
real-object evidence, and publishes atomically. Synthetic adapters and actual frozen source
class fixtures cover every method/cell path. No evaluator, truth archive, runtime next state,
or return path is used.

## Verification status

The final counts, lint results, manifest identities, and frozen-track hashes are recorded in
`V4_TEST_REPORT.md`. `V4_CHANGED_FILES.txt` is the exact tracked change inventory. The
normalized candidate source/test manifest covers the complete package.

No file under `src/tracks/**` was changed. No fit, scientific evaluator, return, registration
freeze, key, Slurm job, driver execution, commit, or push occurred. All V1--V3 clones,
reports, receipts, failed task directories, and probe scripts remain untouched. All V3 probe
receipts are excluded from V4 evidence.
