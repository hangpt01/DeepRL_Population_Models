# VOID Paper-Faithful 4,000-Target One-Cell Smoke Handoff

> **Scientific status: VOID.** See `VOID.md`. Preserve this root for forensic and
> runtime inspection only; do not use any scientific output.

## Registered execution

- Root: `real_ecology_runs/paper_faithful_one_cell_4000_20260717_v1/`
- Slurm array: `58358631` (`0` PLUS, `1` MOOR)
- Return-blind acceptance job: `58358925`, dependency `afterok:58358631`.
- Final status: task 0 (PLUS) canceled after 02:59:55; task 1 (MOOR) completed in
  00:30:56; dependent acceptance job canceled before execution.
- Scope: one Amur tiger / private-Ricker / safe / sigma=0.1 cell, hidden r/K,
  4,000 target public transitions with complete episodes. This is not a headline result.
- PLUS: 16 `episode_bootstrap_map_v1` candidates, four per registered form.
- MOOR: one fitted Ricker model; no candidate prior.
- Fit budget: 8 starts, 100 L-BFGS iterations, 16 Monte Carlo paths.
- PBVI budget: 41 abundance bins, 9 capacity bins, 41 observation bins,
  32 belief points, 7 observation branches, horizon 5.
- Evaluation: seed 7001, four episodes, horizon 50.

## Frozen provenance

- Source commit: `5f9cf32a69d47e2d31b2ed6ee30ebbcfe84b8536` plus the frozen dirty tree.
- Snapshot tree SHA-256:
  `2aae03ba1a22b8c9a3e2c850aa31814253f02f1b9691d5899b5b93e4ea0b13f5`
  over 120 included files.
- Config SHA-256:
  `c9f4c3bfb80f870cba99f321c765a407e4d2283e669be28dd36c1c79276aa50a`.
- Manifest SHA-256:
  `c442e8968b8a81ae3e317492a84b43986d2c27587b45050a2dfd28d11938dfa0`.
- Dependency-lock SHA-256:
  `b696ac987bdf5abec3726a3f5c0484b69893375af850e1ca96785999f623a380`.
- Solver probe SHA-256:
  `8af8b60dbc26375a8ae2e0f522948a7b8502ae123bbf5619c5e204996b5a15fa`.
- External DESPOT/SARSOP solvers remain disabled; PBVI is the registered planner.

## Requested resources

- Partition `comp`; array `0-1%2`.
- Two CPUs and 24 GiB RAM per task.
- 48-hour wall-time ceiling per task.
- Frozen working directory and command paths are recorded in `submission.json` and
  independently visible through `scontrol show job 58358631`.

## Measured runtime before cancellation

- MOOR: `COMPLETED`, exit `0:0`, elapsed `00:30:56`, TotalCPU `00:30:38.973`,
  peak batch RSS 546,256 KiB.
- PLUS: `CANCELLED`, elapsed `02:59:55`, TotalCPU `02:58:48`, peak observed RSS
  549,860 KiB. The 16-candidate fit did not finish.
- Acceptance `58358925`: canceled at `00:00:00`; `acceptance.json` was not generated.
- Mechanical 288-cell sizing lower bound: about 149 CPU-hours MOOR plus more than
  858 CPU-hours PLUS, over 1,000 CPU-hours total. This does not forecast corrected code.

The run was stopped because the audited survey objective and regime candidate law must
change. The measured timing is retained only as a compute warning.

## Audit commands

```bash
squeue -j 58358631
squeue -j 58358925
sacct -j 58358631 \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,TotalCPU,MaxRSS,AllocCPUS -P
sacct -j 58358925 \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,TotalCPU,MaxRSS,AllocCPUS -P

.venv-paper-faithful/bin/python \
  scripts/hash_paper_faithful_snapshot.py \
  real_ecology_runs/paper_faithful_one_cell_4000_20260717_v1/code \
  --expected 2aae03ba1a22b8c9a3e2c850aa31814253f02f1b9691d5899b5b93e4ea0b13f5

.venv-paper-faithful/bin/python \
  real_ecology_runs/paper_faithful_one_cell_4000_20260717_v1/code/scripts/run_paper_faithful_acceptance.py \
  --root real_ecology_runs/paper_faithful_one_cell_4000_20260717_v1 \
  --expected-rows 2 --target-rows 4000 --require-bootstrap-map
```

The acceptance command is return-blind, but it must **not** be run for this void partial
run. It remains above only to document the originally registered gate.
