# acceptance_v2_frozen_runtime_observable_20260719

**Amendment time:** 2026-07-19T14:16:11+10:00  
**Status:** preregistration amendment made before submission and before inspection of any performance return.  
**Reason:** the original acceptance table requires diagnostics that the frozen runtime does not emit.

The original table in `PHASE2_PREREGISTRATION_AND_BUDGET_PACKAGE.md` remains unchanged and is the
archived record. This amendment does not claim that unavailable evidence passed. It defines a limited,
return-blind structural acceptance procedure for snapshot `fd50c38c` only.

Registered run label: **Registered 32-cell hidden-demographics ecological-baseline diagnostic at the
inherited matched 4,000-transition budget, with limited frozen-runtime structural acceptance.**

## Original-to-amended criterion mapping

| # | Original criterion | v2 classification | Amendment |
|---|---|---|---|
| 1 | Privacy & provenance | EVALUABLE GATE | Exact snapshot/manifest identity, emitted privacy audit, registered implementation IDs, and planner provenance are gated. |
| 2 | Finite objectives | EVALUABLE GATE | Every selected fitted objective must exist and be finite. |
| 3 | Optimizer termination | NOT EVALUABLE | Per-start L-BFGS termination status is not emitted. |
| 4 | Scale-aware convergence | DESCRIPTIVE ONLY | Objective traces and raw selected-gradient norms are reported; projected/bound-aware gradients and a registered scaled tolerance are unavailable. |
| 5 | Parameter-boundary occupancy | NOT EVALUABLE | Bound-active coordinates and boundary occupancy are not emitted; epsilon was not numerically fixed. |
| 6 | Multi-start agreement | DESCRIPTIVE ONLY | Finite start counts/objectives are reported, but the agreement tolerance was not registered. |
| 7 | Holdout trajectory error | DESCRIPTIVE ONLY | Holdout normalized survey SSE is reported; a comparable train error and ratio threshold are unavailable. Non-finite values are reported, not silently passed. |
| 8 | PLUS candidate failures & diversity | DESCRIPTIVE ONLY | Candidate counts/forms/parameter hashes are reported. The runtime enforces its construction minimum, but no separate acceptance warning threshold was fully registered. |
| 9 | PLUS posterior normalization | NOT EVALUABLE | Rollout posterior history is not emitted outside forbidden evaluation summaries. |
| 10 | PBVI finite values & legal actions | EVALUABLE GATE + NOT EVALUABLE remainder | Emitted action values must be finite and their argmax legal; run-specific deterministic repeat is NOT EVALUABLE. |
| 11 | Fit/kernel/filter/planner consistency | EVALUABLE GATE | Receipt, fitted-model, POMDP, planner, transition, regime-law, and parameter hashes are cross-checked where emitted. |
| 12 | Missing / failed cells | EVALUABLE GATE | All 64 fit receipts, 64 plan artifact sets, and successful Slurm array task exits are required. |
| 13 | Runtime / memory limits | EVALUABLE GATE | Slurm allocated core-hours, states, exit codes, elapsed time, and MaxRSS are checked against approved limits. |

## Frozen-runtime observable gates

Pre-run gates require the registered runtime identity, both exact manifest hashes, the 166-test suite,
privacy/routing tests, exhaustive 64-pair positional alignment, correct runner dispatch, 128-row
no-execution dry-run, unique outputs, and activated fit-receipt/cache-hit gates for every plan row.

Per-fit and per-plan gates use only the frozen manifests, fit receipts, `faithful_fit.json`,
`planner_provenance.json`, `privacy_audit.json`, candidate/POMDP metadata and arrays,
`pbvi_policy_diagnostics.npz`, public dataset metadata needed for the complete-episode allowance, and
Slurm accounting. `summary.json` and all return/comparison/ranking fields are forbidden.

The output decision is exactly one of:

- `PASS_LIMITED_STRUCTURAL_ACCEPTANCE`
- `FAIL_STRUCTURAL_ACCEPTANCE`
- `INCOMPLETE`

It is never called full scientific acceptance.

## Limitations and deferred instrumentation

Optimizer convergence, boundary occupancy, run-specific posterior behaviour, and run-specific PBVI
repeatability are not fully observable and cannot be validated at 32-cell scale by this run. The run
does not authorize the full 288-cell experiment. Before that experiment, add and validate per-start
optimizer status, projected/bound-aware gradients, boundary occupancy, train/common-holdout
diagnostics, return-independent PLUS posterior diagnostics, run-specific PBVI repeatability, and fully
registered candidate-diversity measures.

The inherited 4,000-transition budget remains unvalidated, safe-mode results remain
information-limited, and 4k-vs-8k adequacy remains pending.
