# Ricker-only PLUS 72-cell launch plan

This package is a new experiment. It does not modify, relabel, or reuse the results or
acceptance record of `plus_adapted_mechanistic_pbvi`. The only reused objects are individually
hash-verified Ricker fit-cache entries; the new experiment always produces its own receipts,
planner artifacts, evaluation outputs, and acceptance record.

## Registered design

- Method: `plus_adapted_ricker_only_pbvi`.
- Cells: nine populations by four hidden environment families by sigma 0.1/0.2 = 72.
- Bank: one full-history Ricker MAP fit plus seven deterministic episode-bootstrap Ricker MAP
  fits, seed root 51116, uniform prior 1/8.
- Offline data and split: 4,000 complete-episode transitions; inherited deterministic collection,
  0.8 episode-level fit fraction, preserved episode order and boundaries.
- PBVI: 41 state bins, 9 capacity bins, 41 observation bins, 256 transition samples, 256
  observation samples, 32 belief points, 7 observation branches, horizon 5, discount 0.95.

The hidden family is used only by data generation and evaluation. The policy receives a
`MethodContext`, which has no family label or private demographic parameters.

## Departures from frozen cross-family runtime

The ecological equations, optimizer, data collection and splitting, candidate-fit cache identity,
POMDP construction, Bayesian update, PBVI solver, and evaluator are unchanged. The new snapshot
only adds: the distinct method route; permission for the registered one-form configuration; the
Ricker-only construction label; strict eight-candidate method validation; launch dispatch and
   full-cache-identity locking; the new config/manifests; and return-blind acceptance. The old method route is
explicitly pinned back to all four families, so it cannot inherit the new one-form configuration.
No old manifest, artifact, result, or acceptance record is changed.

## Launch shape and dependencies

1. Materialize the 64 hash-verified reusable references (48 unique cache objects) into the new
   cache. Cache identity remains content-derived; mismatches are ignored by the scientific fit
   lookup and fail the reuse audit.
2. Run a separate reduced-budget, non-comparative **structural smoke test**. It uses eight Ricker
   candidates but only 160 transitions, two optimizer starts, 12 iterations, four Monte Carlo
   paths, smaller discretization/PBVI budgets, and a separate run root. Those reductions—not the
   production 14--17 minute candidate runtime—make a 30-minute fit limit plausible. It tests
   routing, candidate cardinality, privacy, serialization, cache gating, and acceptance only; it
   provides no production-runtime evidence. One smoke worker runs fit then plan on success, with a
   01:00:00 limit; one 00:05:00 smoke acceptance job uses `afterany` on that worker.
3. Submit one cell-worker array `0-71%72`, one CPU and 4 GB per element, 05:30:00. Element `i`
   invokes the registered fit runner for fit row `i`. Shell `errexit` is the exact cell-local
   success dependency: the same allocation invokes plan row `i` only if fit row `i` exits zero.
   A failed fit therefore suppresses only its own plan and cannot block any other cell. There is
   never a second overlapping Slurm array, so the aggregate allocation cannot exceed 72 PLUS tasks.
4. Submit one 1-CPU, 4-GB, 00:20:00 final acceptance job with `afterany:<cell_worker_array_id>`.
   It therefore runs after successes, failures, timeouts, or deadline cancellations and records
   `INCOMPLETE` whenever required artifacts are absent.

Peak production PLUS allocation is therefore 72 CPUs and 288 GB. Fit and plan work can overlap
across cells, but each cell occupies one allocation throughout and the array throttle is 72. The
structural smoke peaks at one CPU and 4 GB before production begins. The
production maximum requested allocation is 396.33 core-hours; including every canary time limit,
the absolute maximum is 397.42 core-hours. Operational controls warn at 300, hold pending work at
330, and stop before 400. No requeue or retry is permitted.

## Timing and monitoring

The return-blind production estimate is 4.2--5.3 hours when the immediate queue gate predicts a
start within ten minutes. Cell-local pipelining can improve median completion but does not reduce
the conservative maximum-cell estimate. The structural smoke is budgeted separately and is not
used to calibrate this production estimate.

The six-hour production clock starts at `T0`, the earliest Slurm-recorded actual `Start` transition
of any production cell-worker array element. Queue time before `T0` is reported separately. Check
at `T0`, `T0+2h`, `T0+4h`, `T0+5h`, and `T0+5h45m`. At `T0+5h45m`, hold every still-pending array
element so no new scientific work starts. At exactly `T0+6h`, cancel **both pending and running**
cell-worker elements; there is no grace period. Slurm termination preserves artifacts already
written. The `afterany` acceptance job then runs outside the scientific-execution clock and records
the completed structural state, normally `INCOMPLETE` if anything was cancelled. Report only
states, completion/failure counts, elapsed time, memory, and allocated core-hours. Comparative
returns remain sealed even after structural acceptance until the user separately authorizes them.

## Cache-lock identity

The lock is no longer based on transition-data hash alone. Before fitting, launch tooling derives
all eight exact scientific cache keys and acquires their locks in sorted order. Each lock digest
includes: frozen runtime digest, full configuration-file SHA-256, full public-dataset SHA-256,
transition-data hash, candidate index, candidate seed, and the fitted-model cache key. That cache
key itself includes equation/fit schema, model and fit settings, observation protocol and scale,
action channels, episode multiset and holdout split, candidate ID/seed, optimizer runtime, and
effective Monte Carlo budget. Any mismatch yields a different lock and the scientific cache loader
still independently verifies the cache payload and array hash before declaring a hit.
