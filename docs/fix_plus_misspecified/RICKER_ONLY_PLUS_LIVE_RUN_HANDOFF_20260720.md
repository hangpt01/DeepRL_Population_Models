# Ricker-only PLUS ecological stress-test: live-run handoff

Last updated: **2026-07-20 05:59:11 AEST**.

This is the controlling handoff for continuing the currently running Ricker-only PLUS
experiment in a new chat. Treat performance returns as sealed. Status checks must remain
return-blind unless the structural acceptance gate passes and the user separately authorizes
opening comparative results.

## Immediate state

- Production Slurm array: **58396952** (`ricker_plus72_prod`).
- Return-blind acceptance job: **58397033** (`ricker_plus72_accept`), submitted with
  `afterany:58396952`.
- Submission time: `2026-07-20 00:53:41 AEST`.
- Production `T0` (first actual task start): **2026-07-20 00:53:50 AEST**.
- All 72 tasks had started by `00:54:21 AEST`.
- Return-blind checkpoint at `05:59:11 AEST`: **26 COMPLETED, 46 RUNNING, 0 failed**;
  72/72 atomic fit receipts and 26/72 atomic plan-completion receipts; approximately
  **352.97 allocated core-hours**. Acceptance remained pending on the array.
- No performance return was opened during this work.
- No production, general-RL, or other user job has been cancelled.

The user explicitly said: **before holding or cancelling any of these jobs, tell the user
first**. Do not cancel automatically. This instruction applies even at a registered threshold
or deadline. At the last checkpoint there were no pending PLUS elements to hold; all unfinished
elements were already running.

## Scientific design

- Method: `plus_adapted_ricker_only_pbvi` only. Do not run MOOR in this experiment.
- Candidate bank: eight Ricker candidate POMDPs per cell:
  - one full-history Ricker candidate;
  - seven deterministic Ricker episode-bootstrap candidates;
  - uniform prior `1/8`.
- Hidden environments: Ricker (in-family control), Allee, theta-logistic, and
  regime-switching (three out-of-family stress tests).
- Nine populations.
- Observation-noise levels: `0.1` and `0.2`.
- Total cells: `9 populations x 4 families x 2 noise levels = 72`.
- Hidden family labels and private demographic parameters are not exposed to PLUS.
- Offline data: 4,000 complete-episode transitions with the frozen episode-level fit/holdout
  split and seed schedule.
- PBVI production settings are frozen in the registered config: 41 state bins, 9 capacity
  bins, 41 observation bins, 256 transition samples, 256 observation samples, 32 belief
  points, 7 observation branches, horizon 5, discount 0.95.

The old 16-candidate cross-family PLUS experiment is separate and immutable. Do not edit,
overwrite, relabel, or silently reuse its runtime snapshot, manifests, evaluation artifacts,
acceptance record, or results. This Ricker-only experiment reuses only individually hash-verified
Ricker fit-cache objects, not old PLUS results.

## Frozen identity

Durable branch:

`snapshot/ricker-only-plus-72-20260719`

Frozen package:

- Package-identity/head commit:
  `a69f3b0cbf61a9c8802f506979bd0e454125ec02`
- Refrozen routing-content commit:
  `9546be7a3d4a8859a9f82b9b02b8139205109113`
- Scientific runtime digest:
  `7628816c85a39d49373892a0e72d1751de2f0a32dd003b93a54ca1f3e151982c`
- Launcher-overlay digest:
  `f3906f4c7a9c26f26034d000a9ec9881d2d00f97f5058c57f163210ab3eeffdb`
- Production config SHA-256:
  `b871668bae1c971795267bc402c3450e1fe6a628b4e8fee91b1300cfd73fbc26`
- Fit manifest SHA-256:
  `cb478f860b5c523407f4f11f3731904285d2a23164103ed667180ff56b9abdea`
- Plan manifest SHA-256:
  `398780b8267af1b53f4092994eb001caa335537910905190dc58acad23595cc2`
- Acceptance specification SHA-256:
  `eb7ad8dabe89646429fedb126538675beb21cc6e962a8ad54cf03a590a838555`
- Freeze receipt SHA-256:
  `3285a9cc512576fe6648558ecafa5d5715de6fddd91678fc33e2bbf641286dd4`
- Launch-package identity SHA-256:
  `4cecee192dedc5d11f538ee8083531b23b9634f06b4f6db3cc5a6c30b539e7ac`

`apply_row_config` is executable manifest-routing code under `scripts/`, so it is not included
in the scientific digest over `src/real_ecology_benchmark`. It is included in the launcher-overlay
digest. The routing fix therefore honestly changed and refroze the executable overlay while the
scientific `src` digest and scientific configuration remained unchanged.

## Routing fix and verification

The initial smoke failed because `regime_path_count=not_applicable` was converted with `int()` for
the Ricker-only route. The refrozen fix ensures:

1. Ricker-only rows accept the literal `not_applicable` marker.
2. The Ricker-only route never reads, converts, or applies `regime_path_count`.
3. Cross-family PLUS still requires an applicable integer regime-path count.
4. Missing, noninteger, `not_applicable`, and disallowed integer values still fail for the
   cross-family route.
5. All 72 fit plus 72 plan rows resolve through the actual router in dry-run mode.

Verification completed before production:

- focused routing tests: `10 passed`;
- full suite: `176 passed`;
- Ruff and `git diff --check`: passed;
- durable 144-row dry run: passed;
- `regime_path_count_consumed=false`;
- no fit/plan path collisions;
- `return_fields_opened=false`.

Structural-smoke rerun acceptance was
`PASS_LIMITED_STRUCTURAL_ACCEPTANCE`, with one fit and one plan complete, no failures, no warnings,
no partial files, and `return_fields_opened=false`.

Smoke rerun job IDs were `58396761` and `58396770`; acceptance was `58396775`. Two identical smoke
allocations occurred because the first `sbatch` client response was blank although Slurm accepted
the job. They ran sequentially and acceptance verified the final atomic receipts and hashes. No
production duplication occurred.

## Filesystem locations

Repository/workspace:

`/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models`

Durable frozen runtime clone actually used by production:

`/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/ricker_only_plus_72_20260720/runtime_snapshot_routing_fix`

Production root:

`/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/ricker_only_plus_72_20260720/production`

Key frozen files relative to the runtime clone:

- `configs/paper_faithful_hidden_ricker_only_plus_v1.yaml`
- `experiments/ricker_only_plus_72/manifests/ricker_only_plus_fit_72.csv`
- `experiments/ricker_only_plus_72/manifests/ricker_only_plus_plan_72.csv`
- `experiments/ricker_only_plus_72/manifests/verified_reuse_64.csv`
- `experiments/ricker_only_plus_72/acceptance_spec.json`
- `experiments/ricker_only_plus_72/FREEZE_RECEIPT.json`
- `experiments/ricker_only_plus_72/LAUNCH_PACKAGE_IDENTITY.json`
- `scripts/run_real_manifest_row.py`
- `scripts/run_ricker_only_fit_locked.py`
- `scripts/run_ricker_only_plan_with_receipt.py`
- `scripts/run_ricker_only_plus_acceptance.py`
- `scripts/slurm/run_ricker_only_plus_cell.sh`
- `src/real_ecology_benchmark/methods/plus_faithful.py`
- `src/real_ecology_benchmark/planners/pbvi.py`
- `src/real_ecology_benchmark/evaluator.py`

Production outputs are separated into `datasets/`, evaluator-only `private/`, `fit_cache/`,
`fit_receipts/`, `evaluation/`, `completion_receipts/`, and `logs/`. Do not read
`evaluation/**/summary.json`, `evaluation/**/episodes.csv`, or fields such as
`operational_return`, `true_return`, survival returns, rankings, or comparative summaries before
the applicable gate and separate user authorization.

## Launch shape and resource controls

- Partition/QOS/account: `m3h` / `m3h` / `ce25`.
- Array: `0-71%72`.
- Each array element runs its own fit and, only after that fit succeeds, its aligned plan/evaluation
  in the same allocation. One failed fit blocks only its own plan.
- One CPU and 4 GB per element; OMP, MKL, OpenBLAS and NumExpr threads fixed to one; Torch
  intra-op/inter-op fixed to one.
- No requeue and no automatic retries.
- Per-element Slurm limit: `05:30:00`.
- Peak PLUS allocation: 72 CPUs and 288 GB.
- Warning threshold: 300 allocated core-hours.
- Pending-work hold threshold: 330 allocated core-hours.
- Hard ceiling: 400 allocated core-hours.
- The six-hour scientific clock runs from `T0=00:53:50 AEST`; six hours is
  `06:53:50 AEST`. The per-task 5.5-hour Slurm limits should terminate any stragglers by roughly
  `06:24 AEST`, before the six-hour point.
- Final acceptance has one CPU, 4 GB, 20 minutes and `afterany:58396952`.

Both the 300-hour warning and 330-hour threshold have been crossed. Because every unfinished PLUS
task was already running, there was no pending PLUS work to hold. The registered time limits and
already completed tasks still protected the 400-hour ceiling at the last checkpoint. Before any
future hold or cancellation, report the intended action to the user and wait for direction.

## Safe return-blind status checks

Use Slurm accounting and atomic receipt counts only. For example:

```bash
sacct -j 58396952,58397033 \
  --format=JobIDRaw,JobName,State,ExitCode,Submit,Start,End,ElapsedRaw,AllocCPUS,MaxRSS \
  -n -P

find /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/ricker_only_plus_72_20260720/production/fit_receipts \
  -type f -name fit_receipt.json -print

find /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/ricker_only_plus_72_20260720/production/completion_receipts \
  -type f -name plan_completion.json -print
```

Do not infer completion merely from ordinary artifact files. Acceptance requires atomic fit and
plan completion receipts, exact artifact sets and verified hashes, and rejects temporary or partial
files.

When the array ends, job `58397033` should run automatically and atomically write:

`/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/ricker_only_plus_72_20260720/production/acceptance.json`

Permitted decisions are `PASS_LIMITED_STRUCTURAL_ACCEPTANCE`,
`PASS_WITH_WARNING_STRUCTURAL_ACCEPTANCE`, `FAIL_STRUCTURAL_ACCEPTANCE`, or `INCOMPLETE`.
Confirm `return_fields_opened=false`. The acceptance program records its decision in JSON but does
not necessarily exit nonzero for FAIL/INCOMPLETE, so downstream work must inspect the acceptance
decision explicitly; `afterok` alone is not an adequate gate.

## Proposed trajectory follow-up (not yet implemented or submitted)

The user wants a later, additional job that records trajectories for visual comparison with the
general-RL runs. No such Ricker-only trajectory job has been implemented or submitted yet.

The current evaluator records timestep values internally but saves only episode-level rows and
summaries. The repository's existing `scripts/capture_trajectories.py` is **not suitable** for this
experiment: it targets an older synthetic benchmark, refits several methods, uses a different cell
grid/configuration, and prints reward summaries.

A correct follow-up should be a new overlay and separate output root. It should:

1. run only `plus_adapted_ricker_only_pbvi`;
2. require a passing production `acceptance.json` decision, not merely a successful Slurm exit;
3. verify the frozen config, manifest, runtime, fit-cache, fit-receipt and plan-artifact hashes;
4. load/reconstruct the frozen PLUS policy without refitting identical candidates;
5. use an explicit immutable trajectory manifest;
6. use evaluation seeds and horizons paired exactly with the intended general-RL comparison;
7. save per-step true abundance, public observation, chosen action, belief estimate/interval,
   candidate posterior weights, termination/truncation flags, cell identity and seed;
8. write atomically to a new root such as
   `real_ecology_runs/ricker_only_plus_72_20260720/trajectory_export_v1`;
9. leave the frozen runtime snapshot and current production artifacts unchanged;
10. keep trajectory outputs sealed until structural acceptance passes and the user explicitly
    authorizes comparative inspection.

Before implementing that job, identify the exact general-RL trajectory cells, episode count, seeds,
horizon, action encoding, and artifact location so the comparison is genuinely paired. Creating
trajectories may access evaluator-only true abundance, so it is not merely a return-blind structural
check.

## Suggested first message in the new chat

> Read `docs/fix_plus_misspecified/RICKER_ONLY_PLUS_LIVE_RUN_HANDOFF_20260720.md` completely and
> continue from its latest return-blind checkpoint. First check Slurm jobs 58396952 and 58397033
> plus atomic fit/plan receipt counts. Do not open evaluation summaries, episode returns or
> comparative outputs. Do not hold or cancel any job without telling me first. After production
> acceptance, help design the separate paired trajectory-export overlay for comparison with the
> general-RL runs; do not modify the frozen runtime or production artifacts.
