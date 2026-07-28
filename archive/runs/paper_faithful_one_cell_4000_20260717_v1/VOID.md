# VOID — Do Not Use Scientific Outputs

Date voided: 2026-07-17T14:00:27+10:00

This registered one-cell smoke is preserved for forensic inspection and runtime
measurement, but **all scientific outputs are void**. Do not report, aggregate,
compare, relabel, resume, or reuse its fitted parameters, policy behavior, returns, or
diagnostics as experimental evidence.

## Cancellation record

- Slurm array `58358631` was canceled at the user's direction after an external
  source-level audit rejected the implementation.
- Task `58358631_0` (PLUS) was terminated after `02:59:55` wall time and
  `02:58:48` TotalCPU. It did not finish and produced no completed PLUS evaluation or
  fit-artifact tree.
- Task `58358631_1` (MOOR) completed before cancellation in `00:30:56`, using
  `00:30:38.973` TotalCPU and 546,256 KiB peak batch RSS.
- Dependent return-blind acceptance job `58358925` was canceled before execution.
  No `acceptance.json` exists for this run.
- No files from the completed MOOR row, partial PLUS task, shared dataset, frozen code,
  logs, manifests, or provenance were deleted.

## Blocking scientific defects

1. **Survey-objective bias (F1).** The shared mechanistic fitting objective sampled a
   second lognormal survey-noise draw inside squared error. That objective is not
   conditional-mean survey SSE plus a parameter-independent constant and biases fitted
   abundance downward. The completed MOOR fit is affected; the unfinished PLUS fits
   used the same objective.
2. **Regime fit/deployment mismatch (F2).** The regime candidate was fitted with a
   probability-weighted mean-field transition but deployed as a discrete switching
   model. Fitting, kernel construction, filtering, and planning therefore did not share
   one transition law.

Both corrections change the fitting objective or transition law. Consequently, even a
completed run under this snapshot would remain scientifically void, and its precise
timing is not assumed to transfer to a corrected implementation.

## Runtime evidence only

The sole retained operational evidence is the lower-bound compute measurement:

| Task | Measured result |
|---|---:|
| MOOR adapted baseline, one 4,000-row cell | completed in 30:56 |
| PLUS adapted baseline, 16 candidates, one 4,000-row cell | exceeded 2:59:55 and did not finish |

A mechanical 288-cell extrapolation is approximately 149 CPU-hours for MOOR and more
than 858 CPU-hours for PLUS, or over 1,000 CPU-hours total. This is a sizing warning,
not a runtime forecast for corrected code. A PI-approved CPU ceiling and subset rule
are required before proposing any sweep.

## Required gate before resubmission

Follow
`docs/fix_implement_ecology_baseline/CLAUDE_DIRECTIVE_FIX_BEFORE_RESUBMIT_PLUS_MOOR.md`:
correct F1 and F2, classify/ablate the optional regularizers, update the plan and method
names, answer the required review items, and obtain explicit approval of the updated
plan and Section 25 decisions. No further job is authorized from this run root.
