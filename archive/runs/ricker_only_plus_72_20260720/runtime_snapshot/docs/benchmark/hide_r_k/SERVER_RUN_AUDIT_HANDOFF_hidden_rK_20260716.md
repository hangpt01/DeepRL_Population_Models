# Server Run Audit Handoff: Hidden-r/K Matched Experiment (2026-07-16)

## 1. Audit request

This handoff is for Claude to audit the **execution process and result completeness** of the
matched current-semantics `expose_rk=full` versus `expose_rk=hidden` experiment. Do not trust the
success statement below without checking the frozen snapshot, manifests, Slurm accounting, logs,
resolved per-row manifests, and output counts.

The experiment has finished. No rerun is requested by this document, and no source or result file
should be modified during the audit.

## 2. Run identity

- Run root:
  `real_ecology_runs/hidden_rk_comparison_20260716/`
- Frozen code/config/data snapshot:
  `real_ecology_runs/hidden_rk_comparison_20260716/code/`
- Snapshot time: `2026-07-16T02:46:59+10:00`
- Git HEAD recorded at freeze: `5f9cf32a69d47e2d31b2ed6ee30ebbcfe84b8536`
- The implementation was an uncommitted audited worktree. The complete contemporaneous
  `git status --short` is preserved in
  [PROVENANCE.txt](../../../real_ecology_runs/hidden_rk_comparison_20260716/PROVENANCE.txt).
  Jobs ran from the frozen snapshot, not the subsequently editable live tree.
- Registered run shape and budgets:
  [registration.json](../../../real_ecology_runs/hidden_rk_comparison_20260716/manifests/registration.json)
- Recorded parent job IDs:
  [job_ids.txt](../../../real_ecology_runs/hidden_rk_comparison_20260716/job_ids.txt)

## 3. Jobs that ran

| Purpose | Slurm job | Array/concurrency | Resources per task | Result | Observed task interval |
|---|---:|---:|---|---|---|
| Current-semantics parameter-known arm, `expose_rk=full` | `58326885` | `0-287%112` | `comp`, 1 CPU, 8 GiB, 4 h | 288/288 `COMPLETED`, 288/288 exit `0:0` | `02:49:03` to `03:56:12` |
| Corrected hidden-demographics arm, `expose_rk=hidden` | `58326886` | `0-287%128` | `comp`, 1 CPU, 8 GiB, 4 h | 288/288 `COMPLETED`, 288/288 exit `0:0` | `02:50:04` to `03:32:35` |
| Completeness and matched comparison analysis | `58326887` | single task, `afterany:58326886:58326885` | `comp`, 1 CPU, 24 GiB, 2 h | `COMPLETED`, exit `0:0`, elapsed 60 s | `03:56:16` to `03:57:16` |

Times are Slurm's local timestamps on `2026-07-16` (`+10:00` at snapshot time). Full-array task
elapsed times ranged from 538 to 1,790 seconds. Hidden-array tasks ranged from 371 to 1,107
seconds.

The row wrapper was
[run_block.sh](../../../real_ecology_runs/hidden_rk_comparison_20260716/run_block.sh). One array
task ran one complete ecological cell: five contiguous method rows (`BS=5`). This prevented two
tasks from racing to create the same cell dataset.

### Actual submission commands

The jobs were submitted directly with these effective commands from the repository root:

```bash
sbatch --parsable --job-name=hidden-rk-hidden --array=0-287%128 \
  --export=ALL,ROOT=$RUN/code,CONFIG=$RUN/code/configs/hidden_rk.yaml,OUTPUT_ROOT=$RUN/outputs/hidden,BS=5 \
  $RUN/run_block.sh $RUN/manifests/manifest_hidden.csv

sbatch --parsable --job-name=hidden-rk-full --array=0-287%112 \
  --export=ALL,ROOT=$RUN/code,CONFIG=$RUN/code/configs/hidden_rk.yaml,OUTPUT_ROOT=$RUN/outputs/full,BS=5 \
  $RUN/run_block.sh $RUN/manifests/manifest_full.csv

sbatch --parsable --dependency=afterany:58326886:58326885 \
  --export=ALL,ROOT=$RUN/code,RUN=$RUN $RUN/run_analysis.sh
```

Here `$RUN` is the absolute path to
`real_ecology_runs/hidden_rk_comparison_20260716`.

The checked-in run-local [submit_sweep.sh](../../../real_ecology_runs/hidden_rk_comparison_20260716/submit_sweep.sh)
expresses the same jobs and resources, but it was **not** the command used for the final submission;
the three direct `sbatch` calls above were used.

## 4. Registered scientific matrix

Each arm contains:

- 9 populations;
- 4 private simulator families: `ricker`, `allee`, `theta`, `regime`;
- observation noise `sigma` in `{0.0, 0.1, 0.2, 0.4}`;
- reward modes `safe` and `yield`;
- methods `refplan`, `bamcts`, `ogsrl`, `moor_native`, `plus_native`;
- filters `learned` for the three general methods and `native_discrete` for both native methods.

Thus each arm has `9 * 4 * 4 * 2 * 5 = 1,440` method rows across 288 ecological cells.

Both arms used the **same frozen config**:
[hidden_rk.yaml](../../../real_ecology_runs/hidden_rk_comparison_20260716/code/configs/hidden_rk.yaml).
The manifest row overrides only `expose_rk` for this arm-level contrast. Load-bearing settings are:

- seed `116`;
- safe collapse penalty `P=5.0`, occupancy mode;
- 4,000 target transitions, 25-step complete episodes;
- planner horizon 5, 96 sequences, 32 particles;
- evaluation seeds `7001, 7051, 7101, 7151, 7201`;
- 4 evaluation episodes per seed, horizon 50;
- NumPy backend, one BLAS thread.

Manifest/config SHA-256 values at audit time:

```text
ac7e5e0aa4ff9cdfea8d5f43d49355b61cbfa86086f9db451f994cbc8e481f08  manifest_full.csv
f38ffdc8ab6918578bb33808c4d7e40f1aac41df67211c1eaa6a6e1184d0b476  manifest_hidden.csv
f88d29641d518b03522c65a89fde193b695476fcd8f8ce482ccdad4e4957277b  registration.json
c5791237960bc998acfafacf39a2f77c9a24827aa412c8b2c0ebb4ff36e93e00  code/configs/hidden_rk.yaml
```

## 5. Pre-submit checks actually run

Before submission:

- all 9 tests in `tests.real.test_hidden_rk` passed;
- the frozen runner and manifest module passed `py_compile`;
- all run shell scripts passed `bash -n`;
- `git diff --check` passed;
- `sbatch --test-only` accepted the array resource request;
- both manifests were checked for 1,440 unique scientific rows, identical scientific key sets,
  correct regime labels, and contiguous five-row cell blocks;
- the frozen config was checked for `P=5.0`, `target_rows=4000`, and `episode_length=25`.

This launch-time check was the focused 9-test hidden-r/K suite, not a fresh rerun of the entire
114-test repository suite. The full suite had passed during the preceding implementation audit.

## 6. Completion evidence

The dependent analysis wrote
[completion.json](../../../real_ecology_runs/hidden_rk_comparison_20260716/analysis/completion.json),
which records:

```json
{
  "complete": true,
  "expected_per_arm": 1440,
  "summary_counts": {"full": 1440, "hidden": 1440}
}
```

Independent filesystem recount at handoff creation found, for **each** arm:

- 1,440 readable `summary.json` files;
- 1,440 `manifest_row_resolved.json` files;
- 1,440 `episodes.csv` files;
- 288 summaries per method;
- 720 summaries per reward mode;
- 360 summaries per family;
- 360 summaries per observation-noise value;
- all full summaries labelled `expose_rk=full` and all hidden summaries labelled
  `expose_rk=hidden`.

### Complete-episode row accounting

In each arm, 1,400 method summaries report exactly 4,000 rows. Forty summaries report 4,005 rows,
five rows of bounded complete-episode overshoot. Those forty summaries are five methods over the
same eight ecological cells: Crab-eating fox / theta, both reward modes, all four noise values.
They report 163 episodes instead of 160. This is within the registered `0 <= overshoot < 25`
contract and is identical across full and hidden.

## 7. Results and evidence locations

### Inputs and provenance

- Full manifest:
  [manifest_full.csv](../../../real_ecology_runs/hidden_rk_comparison_20260716/manifests/manifest_full.csv)
- Hidden manifest:
  [manifest_hidden.csv](../../../real_ecology_runs/hidden_rk_comparison_20260716/manifests/manifest_hidden.csv)
- Frozen source/config/data: `real_ecology_runs/hidden_rk_comparison_20260716/code/`
- Provenance and dirty-worktree record:
  [PROVENANCE.txt](../../../real_ecology_runs/hidden_rk_comparison_20260716/PROVENANCE.txt)

### Per-row outputs

- Full outputs: `real_ecology_runs/hidden_rk_comparison_20260716/outputs/full/`
- Hidden outputs: `real_ecology_runs/hidden_rk_comparison_20260716/outputs/hidden/`

Each output tree includes public datasets, private evaluator sidecars, calibration records,
evaluation episodes, method summaries, training histories, and resolved per-row manifests.

### Aggregate outputs

- Completion gate:
  [completion.json](../../../real_ecology_runs/hidden_rk_comparison_20260716/analysis/completion.json)
- Compact matched comparison:
  [comparison_summary.json](../../../real_ecology_runs/hidden_rk_comparison_20260716/analysis/comparison_summary.json)
- Per-cell hidden-versus-full values:
  [paired_hidden_vs_full.json](../../../real_ecology_runs/hidden_rk_comparison_20260716/analysis/paired_hidden_vs_full.json)
- Headline general-method comparison against `moor_native`:
  `analysis/aggregate_headline_vs_moor.json`
- Contextual comparison against both native methods, with `plus_native` treated as closed-world:
  `analysis/aggregate_context_vs_both_natives.json`

## 8. Slurm log location and scan result

Because the final jobs were submitted directly from the repository root, the Slurm logs are at
the **repository root**, not under the run's `logs/` directory:

```text
hidden-rk-full_58326885_<array-index>.out
hidden-rk-full_58326885_<array-index>.err
hidden-rk-hidden_58326886_<array-index>.out
hidden-rk-hidden_58326886_<array-index>.err
hidden-rk-analysis_58326887.out
hidden-rk-analysis_58326887.err
```

There are 288 stdout and 288 stderr files for each arm. All 576 array stderr files are zero bytes.
The analysis stderr is zero bytes. A scan found no `ROW ... FAILED`, Python traceback, `FATAL:`, or
`incomplete sweep` marker.

This misplaced-log detail is an organizational defect in the launch, but it does not change the
frozen code, manifests, outputs, or scheduler completion result. An audit should not infer that
`real_ecology_runs/hidden_rk_comparison_20260716/logs/` contains the run logs.

## 9. Suggested independent audit checks

1. Re-run Slurm accounting:

   ```bash
   sacct -X -j 58326885,58326886,58326887 \
     --format=JobID,JobName,State,ExitCode,Elapsed,Start,End -n -P
   ```

2. Recompute the four SHA-256 values in Section 4.
3. Confirm each manifest has 1,440 unique rows and the two scientific key sets match after
   excluding `index` and `expose_rk`.
4. Confirm each arm has exactly 1,440 summaries, resolved manifests, and episode CSVs.
5. Confirm regime labels, per-method counts, reward/family/noise coverage, and row-budget bounds.
6. Scan the repository-root Slurm logs using the filename patterns in Section 8.
7. Sample resolved row manifests and verify they agree with the summary, public dataset metadata,
   frozen config, and arm path.
8. Re-run `run_analysis.sh` to a separate audit location or independently reconstruct its matched
   keys; do not overwrite the recorded analysis artifacts.
9. Audit that the full and hidden arms differ only through the manifest `expose_rk` route and its
   intended downstream behavior, not through planner/evaluator budgets.
10. Keep result interpretation separate from execution validity. In particular, safe mode remains
    pre-registered as information-limited when public safety channels are weak; this run's clean
    execution does not by itself validate a causal interpretation of every performance difference.

## 10. Current execution verdict to challenge

The recorded evidence supports: **both 1,440-row arms and the dependent analysis ran to completion
without detected row failures, missing outputs, invalid JSON, regime-label errors, or budget-bound
violations.** Claude should either reproduce that conclusion from the evidence above or document
the exact contradictory file, task, row, or invariant.
