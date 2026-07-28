# Codex Phase 116 Current Run / Audit Handoff

Last verified from `squeue -u hphung`: 2026-06-13.

This file describes the code/scripts and Slurm jobs that are currently active for
Phase 116. It intentionally replaces the older "full final benchmark chain"
description: the current queue is a reduced one-day calibration/tuning run plus a
freeze-only selector. The full final benchmark is not auto-attached to this
current run.

## Current Status

Active experiment:

- Phase: calibration tuning only.
- Methods currently running: `mopo` and `refplan`.
- Environments: all six Phase 116 scenario cells:
  - `allee_ricker_pomdp_116`
  - `theta_logistic_pomdp_116`
  - `regime_switch_pomdp_116`
  - `allee_ricker_pomdp_116_10a`
  - `theta_logistic_pomdp_116_10a`
  - `regime_switch_pomdp_116_10a`
- Reward mode: `collapse_sensitive`.
- Tuning seeds: `1160`, `1161`, `1162`.
- Dataset size: `50000` transitions per env/action/seed scenario.
- Eval during tuning: 50 episodes, horizon 50.
- Shared datasets: one dataset per env/action/seed scenario, reused by MOPO and RefPlan.
- WandB: online, project `deeprl_population_models`, group `stress_pomdp_116_tune_1day`.
- GPU request per array task: one L40S GPU, 8 CPUs, 64 GB RAM.
- Wall time per array task: 8 hours.
- Array chunking: each Slurm array task runs two manifest rows via `ROWS_PER_TASK=2`.
- Concurrency: at most four array tasks at a time via `%4`, matching the user's available GPU-slot target.

Active Slurm jobs at the last queue check:

```text
56433936_[66-107%4]  stress11  PD  (JobArrayTaskLimit)
56433936_59          stress11  R   1:52:24  m3g114
56433936_63          stress11  R   0:43:49  m3g102
56433936_64          stress11  R   0:36:42  m3g105
56433936_65          stress11  R   0:28:35  m3g112
56433939             stress11  PD  (Dependency)
```

`56433939` is a freeze-only selector depending on successful completion of
`56433936`. It selects the best one-day MOPO/RefPlan configs and writes frozen
config files; it does not submit final benchmark jobs.

## What Is Not Currently Running

The following are not part of the active queue:

- Final benchmark seeds `7001--7005`.
- Final 100-episode evaluation.
- `base` reward-mode ablation.
- PLUS and MOOR-Ricker final evaluations.
- Final report aggregation.
- The earlier 1296-row matched-budget tuning chain.
- The earlier after-tune script that would submit final datasets/methods/report jobs.

Those pieces still exist in scripts/manifests as planned infrastructure, but the
current job chain was reduced at the user's request to fit roughly one day and
to avoid the previous 5--7 day calibration sweep.

## Current Manifest

Current tuning manifest:

```text
claude_build/outputs/manifests/stress116_tune_methods_1day.tsv
```

Manifest summary:

- Total rows: 216.
- MOPO rows: 108.
- RefPlan rows: 108.
- Six env/action cells, 36 rows each.
- Three tuning seeds, 72 rows each.
- Twelve config tags total: six MOPO configs and six RefPlan configs.
- `run_tag=tune116`.
- `data_n=50000`.
- `eval_episodes=50`.
- `eval_horizon=50`.

Current shared dataset manifest:

```text
claude_build/outputs/manifests/stress116_tune_datasets.tsv
```

Dataset manifest summary:

- Total rows: 18.
- Six env/action cells x three tuning seeds.
- All rows use `data_n=50000`.
- Dataset paths are under:

```text
claude_build/outputs/stress_pomdp_116/shared_datasets/tune116/
```

The current worker uses these `.npz` files through the manifest `dataset_path`
column and runs with `REQUIRE_DATASET=true`.

## Current Tuning Configs

MOPO config tags in the one-day manifest:

| Config tag | Planner horizon | Rollouts | Pessimism lambda | Epochs |
|---|---:|---:|---:|---:|
| `mopo_h5_r20_lam0p5_ep50` | 5 | 20 | 0.5 | 50 |
| `mopo_h5_r20_lam1p0_ep100` | 5 | 20 | 1.0 | 100 |
| `mopo_h5_r25_lam0p5_ep50` | 5 | 25 | 0.5 | 50 |
| `mopo_h5_r25_lam1p0_ep50` | 5 | 25 | 1.0 | 50 |
| `mopo_h5_r25_lam2p0_ep50` | 5 | 25 | 2.0 | 50 |
| `mopo_h5_r50_lam0p5_ep100` | 5 | 50 | 0.5 | 100 |

RefPlan config tags in the one-day manifest:

| Config tag | Horizon | Sequences | Latent samples | Kappa | Uncertainty penalty |
|---|---:|---:|---:|---:|---:|
| `ref_h10_n256_l8_k10p0_u0p1` | 10 | 256 | 8 | 10.0 | 0.10 |
| `ref_h10_n512_l12_k5p0_u0p1` | 10 | 512 | 12 | 5.0 | 0.10 |
| `ref_h8_n1024_l12_k5p0_u0p05` | 8 | 1024 | 12 | 5.0 | 0.05 |
| `ref_h8_n256_l16_k5p0_u0p1` | 8 | 256 | 16 | 5.0 | 0.10 |
| `ref_h8_n256_l8_k2p0_u0p05` | 8 | 256 | 8 | 2.0 | 0.05 |
| `ref_h8_n512_l12_k10p0_u0p2` | 8 | 512 | 12 | 10.0 | 0.20 |

Note for audit: the manifest contains both string forms `0.1` and `0.10` for
RefPlan uncertainty penalty, but these are numerically equivalent Hydra values.

## Gate Status

All six control-gap gate JSONs exist and passed. The gate was run with 20
episodes, horizon 50, MPC horizon 4. Allee and regime use hard thresholds
`reward_gap >= 1.0` and `collapse_gap >= 0.05`; theta is diagnostic with relaxed
thresholds `reward_gap >= 0.5` and `collapse_gap >= 0.0`.

| Env/action cell | Passed | Reward gap | Collapse gap |
|---|---:|---:|---:|
| `allee_ricker_pomdp_116` | true | 2.4912 | 0.1000 |
| `theta_logistic_pomdp_116` | true | 0.6416 | 0.0000 |
| `regime_switch_pomdp_116` | true | 2.8734 | 0.1000 |
| `allee_ricker_pomdp_116_10a` | true | 5.8645 | 0.2500 |
| `theta_logistic_pomdp_116_10a` | true | 2.7926 | 0.0500 |
| `regime_switch_pomdp_116_10a` | true | 7.0536 | 0.3000 |

Gate JSONs are under:

```text
claude_build/outputs/stress_pomdp_116/*_collapse_sensitive_seed116/control_gap/control_gap_summary.json
```

## Submitted Current Jobs

The active reduced tuning job is:

```bash
sbatch --parsable \
  --export=ALL,BASE=/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/claude_build,WB_MODE=online,WB_PROJECT=deeprl_population_models,WB_GROUP=stress_pomdp_116_tune_1day,REQUIRE_GATE_PASS=true,REQUIRE_DATASET=true,ROWS_PER_TASK=2 \
  --array=0-107%4 \
  claude_build/scripts/slurm/stress_pomdp_116_manifest_worker.sh \
  outputs/manifests/stress116_tune_methods_1day.tsv
```

Job ID: `56433936`.

The dependent freeze-only selector is:

```bash
sbatch --parsable --dependency=afterok:56433936 \
  --export=ALL,BASE=/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/claude_build,WB_MODE=online,WB_PROJECT=deeprl_population_models \
  claude_build/scripts/slurm/stress_pomdp_116_freeze_only.sh
```

Job ID: `56433939`.

Expected freeze-only outputs:

```text
claude_build/outputs/stress_pomdp_116/frozen/phase116_frozen_1day.env
claude_build/outputs/stress_pomdp_116/frozen/phase116_frozen_1day.json
claude_build/outputs/stress_pomdp_116/frozen/phase116_tuning_scores_1day.csv
```

## Files Claude Should Audit

Core Phase 116 environment/action implementation:

- `claude_build/src/environments/ricker_env.py`
- `claude_build/src/environments/__init__.py`
- `claude_build/config/env/allee_ricker_pomdp_116.yaml`
- `claude_build/config/env/theta_logistic_pomdp_116.yaml`
- `claude_build/config/env/regime_switch_pomdp_116.yaml`
- `claude_build/config/env/allee_ricker_pomdp_116_10a.yaml`
- `claude_build/config/env/theta_logistic_pomdp_116_10a.yaml`
- `claude_build/config/env/regime_switch_pomdp_116_10a.yaml`

MOPO vectorization / planner path:

- `claude_build/src/models/ensemble.py`
- `claude_build/src/planners/pessimistic_planner.py`
- `claude_build/config/planner/pessimistic.yaml`
- `claude_build/tests/test_pessimistic_planner_vectorized.py`

Phase 116 Slurm and manifest automation:

- `claude_build/scripts/slurm/stress_pomdp_116.sh`
- `claude_build/scripts/slurm/stress_pomdp_116_manifest_worker.sh`
- `claude_build/scripts/slurm/stress_pomdp_116_freeze_only.sh`
- `claude_build/scripts/experiments/make_stress116_manifest.py`
- `claude_build/scripts/analysis/select_stress116_frozen.py`
- `claude_build/scripts/analysis/stress116_control_gap.py`
- `claude_build/scripts/analysis/aggregate_stress116_phase116.py`

Planned but not active in the current queue:

- `claude_build/scripts/slurm/stress_pomdp_116_after_tune.sh`
- `claude_build/scripts/slurm/stress_pomdp_116_report.sh`
- `claude_build/outputs/manifests/stress116_tune_methods.tsv`
- `claude_build/outputs/manifests/stress116_final_methods_template.tsv`

## Important Fix History

These fixes are already in the scripts used by the current jobs:

1. Slurm base-directory detection was fixed so jobs submitted from the repo root
   do not treat the Slurm spool directory as the project base.
2. Hydra struct overrides for live danger-zone metrics were fixed by using
   `+eval.live_danger_low` and `+eval.live_danger_high`.
3. TSV generation now uses Unix line endings. The original CRLF output caused
   Bash associative-array keys like `refplan_uncertainty_penalty\r`.
4. The manifest worker now parses TSV rows using Python `csv.DictReader`, so the
   empty `method` field in dataset rows no longer shifts columns. This fixed the
   earlier accidental 50-transition toy dataset generation.
5. Chunked array recursion now calls
   `$BASE/scripts/slurm/stress_pomdp_116_manifest_worker.sh`, not a script path
   relative to the Slurm spool directory.
6. The long 1296-row tuning chain was canceled and replaced with the current
   216-row one-day tuning manifest using `--mopo-config-limit`.
7. The current dependent job is freeze-only, so it will not unexpectedly launch
   the full final benchmark after tuning.

Canceled superseded jobs from the earlier chains:

- `56397705`, `56397707`
- `56414065`, `56414069`
- `56417264`, `56417275`

## Audit Questions For Claude

Please check the current code/scripts against the active run, especially:

1. Does `stress116_tune_methods_1day.tsv` faithfully encode the intended
   one-day tuning matrix?
2. Does `stress_pomdp_116_manifest_worker.sh` correctly map each chunked array
   task to two manifest rows and preserve the shared dataset path?
3. Does the worker enforce gate pass and dataset existence before running each
   method row?
4. Does MOPO actually use the vectorized pessimistic planner by default?
5. Are hidden variables still excluded from agent observations in all six envs?
6. Does `stress_pomdp_116_freeze_only.sh` only freeze configs and avoid launching
   final benchmark jobs?
7. Are WandB settings online and grouped as `stress_pomdp_116_tune_1day` without
   exposing the local API key?

