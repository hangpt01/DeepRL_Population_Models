# Phase 2F matched full general-RL package report

Date: 2026-07-20  
Package: `real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1`

## Outcome

The new matched full general-RL package is prepared and frozen but **not
submitted**. Its registration has both `submitted:false` and
`authorized_to_submit:false`. No scheduler command without `--test-only` was
executed.

The manifest contains exactly 576 rows:

- 9 populations;
- `ricker`, `allee`, `theta`, and `regime`;
- `sigma_obs` exactly 0.1 and 0.2;
- `safe` and `yield` reward modes;
- RefPlan, OGSRL, BA-MCTS, and ensemble value-disagreement pessimism;
- 144 rows per method and four methods per dataset cell.

Every row registers 4,000 transitions, 160 complete 25-step episodes, the
episode-preserving 80/20 split, evaluation seeds 7001/7051/7101/7151/7201,
four episodes per seed, a 50-step evaluation horizon, collection seed 116, and
one thread. Runtime source, tests, method implementations, planner settings,
method seed derivations, and Phase 2E hyperparameters are unchanged. The only
new runtime configuration difference from the canary is the explicitly
requested 20-episode evaluation schedule and its output path.

## Frozen artifacts and hashes

| Artifact | SHA-256 |
|---|---|
| 576-row manifest | `1526ce08dcf1b1d148232c075d41dbd78f0cfa2d7119cff9e330f965b6451df4` |
| 144-cell dataset registry | `474d1a65d2e5ec28c741f5b7ac5691be891712e753ee6a9a6590337d62f5f0c4` |
| registration | `15fd7aa1c3ddc460fd255e5d66518d49244f81cf3dbbd291a5bdfe52b376b37a` |
| complete frozen code tree (164 files) | `f615d363a525422abfb983f0eee7c433b009fff81a4d0ae4925bfdaba0f3aa72` |
| unchanged Phase 2E runtime/source/tests (132 files) | `44ddb4cd99f7bd4e28a20f7143249f6653791dfc39008fb88d0210a13532c08f` |
| full-run config | `4e878a9ec7d528af0bfb4948e78036dc864f0d2b158a2fdf5559b24798b1222b` |
| prepared launch commands | `22e0e5a73f30f94f7f3177b593d445ac0b28e83675da72447346894e030fb0a4` |
| receipt-only resource projection | `09a8cb427bc583503212c555b11a7f47e359c7dc74332e0960d83714c9a2c739` |
| scheduler test-only record | `41065da717bd139d66c23f7398f5e8efc6e115a1bebb820457d219ab99f2db27` |

The old 1,152-row manifest remains unchanged at SHA-256
`e71f67daf7cd048577418f6e074e5ea4997fd0a23ecac61fc1419e9e09ab5a66`.

## Ecological dataset matching

The comparison source was
`hidden_rk_comparison_20260716/outputs/hidden`. All requested 144 cells exist.

Exact byte-for-byte public-dataset and action-array reuse succeeded for 140
cells. Matching private calibration companions were copied mechanically without
deserializing their values.

Exact whole-file reuse is impossible for these four cells:

| Reward | Population | Family | sigma |
|---|---|---|---:|
| safe | Crab-eating fox | theta | 0.1 |
| safe | Crab-eating fox | theta | 0.2 |
| yield | Crab-eating fox | theta | 0.1 |
| yield | Crab-eating fox | theta | 0.2 |

Each ecological source has 4,005 rows and 163 episode IDs: 160 complete
25-step episodes plus three partial episodes of lengths 1, 1, and 3. Whole-file
reuse would violate the required 4,000-row/160-complete-episode design. For
these four cells only, the package contains the exact 160 complete ecological
episodes and omits the five partial-episode rows. Their package dataset hashes
therefore differ from the ecological whole-file hashes; both source and package
hashes are disclosed in the registry.

Independent dataset validation confirmed:

- 144 registered cells and four manifest rows sharing each cell path;
- every package dataset has exactly 4,000 rows and 160 episodes of length 25;
- collection seed 116 and 11 action channels everywhere;
- minimum logged support of 68 transitions for every action;
- public schema/privacy checks and every registered content hash pass.

## Verification

The final frozen code tree passed all 203 tests with
`PYTHONPATH=src` and `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.
This includes mechanism tests, episode-preserving split tests, the
fit-to-act-to-observe-to-act intervention-invariance privacy lifecycle, and the
strict return-blind validity reader.

The strict reader from the new snapshot was also run against all 64 completed
Phase 2E canary validity receipts. It accepted all 64, found all four methods,
and opened no outcome file or forbidden outcome field.

## Receipt-only resource projection

For each canary receipt, the 20-episode projection is

`manifest_row_seconds + 19 * evaluation_seconds`.

This keeps observed non-evaluation time fixed and assumes evaluation time is
linear in episode count. It is a return-blind planning estimate, not a runtime
guarantee for the broader population set.

| Method | Mean projected task time | 144-task total |
|---|---:|---:|
| RefPlan | 39.36 s | 1.5744 task-hours |
| OGSRL | 321.14 s | 12.8455 task-hours |
| BA-MCTS | 730.47 s | 29.2187 task-hours |
| Ensemble value-disagreement pessimism | 1.67 s | 0.0666 task-hours |

Total projected CPU/core time is **43.7052 core-hours**. Ideal work-conserving
elapsed times, excluding queue delay and scheduler overhead, are:

| Concurrency | Projected elapsed |
|---:|---:|
| 16 | 2.7316 h (2 h 43 m 54 s) |
| 32 | 1.3658 h (1 h 21 m 57 s) |
| 64 | 0.6829 h (40 m 58 s) |

## Current scheduler probes

All six commands used `sbatch --test-only`, one CPU, 8 GiB, a two-hour limit,
one-thread exports, and `--no-requeue`. The estimate IDs were subsequently
queried together with `squeue`; no rows existed, confirming no jobs were
created.

| Partition/QOS | Concurrency | Estimate-only ID | Estimated start |
|---|---:|---:|---|
| comp/normal | 16 | 58397171 | 2026-07-25 12:00:32 AEST |
| comp/normal | 32 | 58397174 | 2026-07-25 12:00:35 AEST |
| comp/normal | 64 | 58397175 | 2026-07-25 12:00:37 AEST |
| m3h/m3h | 16 | 58397172 | 2026-07-20 01:31:33 AEST |
| m3h/m3h | 32 | 58397173 | 2026-07-20 01:31:35 AEST |
| m3h/m3h | 64 | 58397176 | 2026-07-20 01:31:41 AEST |

These estimates are transient and do not authorize launch.

## Launch preparation and quarantine

`provenance/prepared_launch_commands.txt` contains separate sections for the
six non-submitting probes and six real launch templates at concurrency 16, 32,
and 64 on both eligible partition/QOS pairs. The real templates are prominently
marked **DO NOT RUN WITHOUT EXPLICIT APPROVAL**.

All future summaries, episode files, rewards, returns, survival metrics,
rankings, and performance outputs resolve beneath the package's `quarantine`
root. No quarantined canary return or comparative outcome was opened while
preparing this package. PLUS/MOOR files and jobs were not modified or
interacted with.

## Verdict

**PREPARED, VALIDATED, AND NOT SUBMITTED.** A new explicit launch authorization
is still required.
