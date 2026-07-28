# Scheduler diagnosis and lower-resource launch alternatives

**Date:** 2026-07-19 Australia/Melbourne. **No jobs submitted. No return fields opened.**

## Exact source of the current estimate

```text
sbatch --test-only --job-name=adapt32-fit --partition=comp --array=0-63%100 --cpus-per-task=2 --mem=4G --time=08:00:00 --no-requeue --output=/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/real_ecology_runs/adapted_32cell_diagnostic_pending/logs/adapt32_fit_%A_%a.out --error=/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/real_ecology_runs/adapted_32cell_diagnostic_pending/logs/adapt32_fit_%A_%a.err --export=ALL,OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,MKL_NUM_THREADS=2 /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/real_ecology_runs/adapted_32cell_diagnostic_pending/code/scripts/phase2_launch.py execute-row --stage fit --fit-manifest /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/real_ecology_runs/adapted_32cell_diagnostic_pending/manifests/diagnostic_fit_64.csv --plan-manifest /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/real_ecology_runs/adapted_32cell_diagnostic_pending/manifests/diagnostic_plan_safe_64.csv --derived-root /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/real_ecology_runs/adapted_32cell_diagnostic_pending/launch/derived_plan_rows --run-root /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/real_ecology_runs/adapted_32cell_diagnostic_pending --code-root /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/real_ecology_runs/adapted_32cell_diagnostic_pending/code --python-bin /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python

exit 0
sbatch: Job 58391396 to start at 2026-07-21T16:31:24 a using 2 processors on nodes m3j001 in partition comp
```

Resolved association: user `hphung`, default account `ce25`, normal QOS; no reservation. There is no
current pending reason because `--test-only` does not retain a job and `squeue -u hphung` is empty.

At diagnosis time `comp` reported 3,890 allocated / 1,866 idle / 36 unavailable / 5,792 total CPUs.
Many idle nodes had hundreds of GB free memory. The account had zero jobs. Association limits were 500
jobs and 1,000 submitted jobs; normal QOS allowed 1,000 jobs and 256 CPUs/user. The request's 128-CPU
peak is within those limits.

Fair-share evidence: `hphung` fair-share factor 0.283418, LevelFS 0.5. Scheduler is multifactor
backfill with weights age=1000, fair-share=1000, job-size=1000, partition=1000, QOS=5000; normal-QOS
jobs show a 500 QOS component. No exact `sprio` row exists for a test-only job. Existing comp jobs have
higher priority and the same partition/QOS components.

Active reservations cover cryosparc/sexton nodes only, none in `comp` or `m3h`. Outages were m3j000
(36 comp CPUs), m3d127 (peach), and m3s119 (peach). The primary controller was down and the backup was
active; no evidence tied that failover to the stable partition-specific estimates.

## Test-only comparison

Slurm reports only a first-task start for `--test-only`; final times below combine that start with the
registered measured central critical path and a requested-time-limit bound. They are projections, not
Slurm guarantees.

| Shape | Partition/QOS | First-start estimate | Queue delay | Central execution after first start | Requested-limit bound | Peak CPUs | Central core-h | Maximum core-h | Waves |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A mixed, 2 CPU, 8h/3h | comp/normal | 2026-07-21 16:33 | ~50 h | ~5.75 h | 11 h | 128 | ~385 | 1,408 | 1 |
| B mixed, 1 CPU, 8h/3h | comp/normal | 2026-07-21 16:33 | ~50 h | ~5.75 h | 11 h | 64 | ~192 | 704 | 1 |
| C split, 1 CPU, k=32 | comp/normal | 2026-07-21 16:34 | ~50 h | ~5.75 h | 9 h | 64 | ~192 | 352 | 1 |
| D split, 1 CPU, k=16 | comp/normal | 2026-07-21 16:34 | ~50 h | ~9.65 h | 15 h | 64 | ~192 | 352 | 2 |
| D split, 1 CPU, k=8 | comp/normal | 2026-07-21 16:34 | ~50 h | ~17.45 h | 27 h | 32 | ~192 | 352 | 4 |
| C split, 1 CPU, k=32 | m3h/m3h | immediate (~14:46) | ~0 | ~5.75 h | 9 h | 64 | ~192 | 352 | 1 |
| D split, 1 CPU, k=16 | m3h/m3h | immediate (~14:54) | ~0 | ~9.65 h | 15 h | 64 | ~192 | 352 | 2 |
| D split, 1 CPU, k=8 | m3h/m3h | immediate (~14:54) | ~0 | ~17.45 h | 27 h | 32 | ~192 | 352 | 4 |

Changing `comp` from 2 to 1 CPU, 8 h to 30 min, or concurrency 32 to 8 did not move the predicted
start. Therefore CPUs, memory, wall time, job-count limits, and account/QOS CPU limits are not the
primary cause. The evidence supports partition-specific priority/backfill placement on `comp`.

Other permitted tests: `comp/rtq` remained 2026-07-21; `comp/irq` worsened to 2026-07-25;
`short/shortq` predicted 2026-07-20 02:33 and has a 30-minute maximum, unsuitable for PLUS;
`desktop/desktopq` was rejected with `QOSMaxSubmitJobPerUserLimit`. `m3h/m3h` was allowed by the user
association, had 108-124 idle CPUs during probes, and all four split-stage shapes tested immediate.

## One-thread equivalence and split-overlay evidence

- Frozen one-thread suite: 166 tests + 14 subtests, zero failures/errors; Torch intra/inter-op = 1/1.
- Launch/acceptance tests: 7 passed under the same one-thread environment.
- Lightweight deterministic 1-vs-2-thread probe: identical objective (absolute difference 0),
  parameters (max difference 0; identical model hash), kernels (max difference 0; identical hash),
  PBVI action values (max difference 0; identical hash), and selected PBVI action (0 in both).
- Split dry runs: 128 commands, exact union of all original fit/plan rows, zero omissions/duplicates,
  original indices and output paths retained, one thread throughout, no returns opened.

Launch-only commit: `313ee9cf40f226145106739443df6a05390281df`; tree
`0cfdc5abd0b6effc2bf0ad707123b046c35931be`; overlay patch digest
`e39112fe0baf03a8294914fdde1312051cfd1e21f0dba3e7b70a6b4b26c9cfc1`.

Frozen identities remain runtime `f70ec7122263ec00da0a4950994dfe483ed29c71bc15e477007ca9ae0a6d2333`,
fit manifest `a8f39d83eb70a32be2c42fabecc56ca4af4d3d40edb97f87242934085fd7ee86`,
plan manifest `f0322718a78d485827c39661e2f1ded4cbde1d8bccd4cd053f4d1fe93dec087d`.

## Recommendation

Current A on `comp`: **NO-GO**. Recommended candidate: **GO only after separate explicit launch
authorization** for C on `m3h/m3h`, one CPU/thread, method-specific arrays, concurrency 32. Its queue
probe was immediate and central post-start completion is ~5.75 h; the 9 h requested-limit bound means
the existing 8 h hard stop may still truncate tail tasks. No jobs were submitted by this diagnosis.
