# General-track replay diagnostic contract

This bounded diagnostic revision explains, but does not repair or relax, the independent
general-track fitted-object replay failure observed after commit
`8c068af4561c9fc4cd325740f279eaa0c6dc98c4`. Ecological tasks 0, 1, 6 and 7 had reproduced;
general tasks 2–5 and 8–11 had reproduced their object bytes, operation inputs and selected
action/output but not the producer's post-call state digest. No evaluator or return ran.

`FROZEN_REPLAY_DIAGNOSTIC.json` uses schema
`corrected_stageb_frozen_replay_diagnostic_v1`. It is a separate required fit-probe artifact,
written with fsync before the task directory's atomic publication and covered by
`PUBLICATION_SUCCESS.json`. It records producer-time pre/post graph digests for both the live
fitted object and a fresh reload, both exact normalized post-call graphs, canonical per-path
digests, RNG-state digests, operation arguments and hashes, selected action/output evidence,
and object-reference topology.

Two comparisons are intentionally distinct:

- strict parity retains the existing post-call graph, including traversal-assigned object
  reference labels, and remains mandatory;
- the diagnostic-only graph replaces object identifiers with structural definition paths so
  an auditor can distinguish label renumbering from a substantive state change.

The only established post-call normalization remains the two already registered wall-clock
fields (`CandidatePOMDP.kernel_build_seconds` and `PointBasedPlanner.elapsed_seconds`). This
revision adds no volatile field, tolerance, omission, wildcard or special-case acceptance.
Large first-difference values are represented by canonical byte count and SHA-256; small values
are also included directly. The exact normalized post-call graphs and their complete canonical
path-digest maps let a separate process locate and compare the producer path. They live in the
separate bounded diagnostic artifact rather than inflating the ordinary parity receipt.

The independent diagnostic entry point reloads the checksum-bound object, reuses the exact
published operation input, performs one replay, and emits
`corrected_stageb_cross_process_replay_comparison_v1`. Classification of the first difference
must follow evidence and source inspection. This implementation does not classify any path as
volatile and does not implement the eventual correction.

The corrected suite contains 202 tests after adding ten replay-diagnostic regressions. They
cover a representative RefPlan replay, exact difference paths and values, RNG divergence,
object-reference renumbering, continued strict parity, unchanged ecological receipts, atomic
checksum coverage, and the evaluator/truth/next-state/return boundary.
