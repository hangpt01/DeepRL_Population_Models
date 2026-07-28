# Codex Audit Of Claude's 136/137 Final-Run Plan

Date: 2026-06-14.

Source files reviewed:

- `docs/codex_claude_conversation_md/136_claude_ask_codex_modify.md`
- `docs/codex_claude_conversation_md/137_claude_final_benchmark_halfday_config.md`
- `claude_build/scripts/slurm/stress_pomdp_116_after_tune.sh`
- `claude_build/scripts/experiments/make_stress116_manifest.py`
- `claude_build/scripts/slurm/stress_pomdp_116_manifest_worker.sh`
- `claude_build/scripts/slurm/stress_pomdp_116_report.sh`
- `claude_build/scripts/analysis/aggregate_stress116_phase116.py`

## Bottom Line

I mostly agree with Claude's plan: the tuning/freeze phase is complete, the
final comparison still needs to run, and the half-day final-run configuration is
basically legitimate. The final phase should be launched only after fixing one
real manifest-label bug and one report-text mismatch.

I would not present Claude's runtime reasoning as written. The half-day plan is
safe, but the completed tuning metadata shows MOPO is fast and RefPlan is the
slow method, not the other way around.

## Verified As Correct

### Tuning and freeze are complete

`squeue -u hphung` showed no active jobs at the check time. The frozen files
exist:

- `claude_build/outputs/stress_pomdp_116/frozen/phase116_frozen_1day.env`
- `claude_build/outputs/stress_pomdp_116/frozen/phase116_frozen_1day.json`
- `claude_build/outputs/stress_pomdp_116/frozen/phase116_tuning_scores_1day.csv`

The exact one-day tuning manifest has 216 rows, and all 216 matched tuning rows
have status `0` metadata.

### The current final-run script is configured for the half-day plan

`claude_build/scripts/slurm/stress_pomdp_116_after_tune.sh` now defaults to:

- `TUNE_MANIFEST=outputs/manifests/stress116_tune_methods_1day.tsv`
- `FROZEN_ENV=outputs/stress_pomdp_116/frozen/phase116_frozen_1day.env`
- `FROZEN_JSON=outputs/stress_pomdp_116/frozen/phase116_frozen_1day.json`
- `FINAL_EVAL_EPISODES=50`
- `FINAL_REWARD_MODES=collapse_sensitive,base`
- `FINAL_QOS=rtq`
- `FINAL_DATASET_CONCURRENCY=6`
- `FINAL_METHOD_CONCURRENCY=6`

It submits dataset -> method -> report with `afterok` dependencies.

### The generated final matrix is the intended size

Dry-run manifest generation produced:

- Dataset manifest: 60 rows.
- Method manifest: 240 rows.
- Methods: 60 each for `mopo`, `refplan`, `plus`, and `moor_ricker`.
- Held-out seeds: `7001,7002,7003,7004,7005`.
- Reward modes: `collapse_sensitive` and `base`.
- Eval: 50 episodes, horizon 50.
- Dataset size: 75,000 transitions.

### The frozen hyperparameters do propagate into final rows

The frozen config currently is:

```text
MOPO_EPOCHS=50
MOPO_PLANNER_HORIZON=5
MOPO_ROLLOUTS=25
MOPO_LAMBDA=0.5
REFPLAN_ENSEMBLE=15
REFPLAN_HORIZON=8
REFPLAN_SEQUENCES=512
REFPLAN_LATENTS=12
REFPLAN_KAPPA=10.0
REFPLAN_UNC_PENALTY=0.2
```

The dry-run final manifests carry those parameter values.

### Gate checks should not block final rows

All six gate summaries exist and passed:

| Env/action cell | Reward gap | Collapse gap |
|---|---:|---:|
| `allee_ricker_pomdp_116` | 2.4912 | 0.1000 |
| `theta_logistic_pomdp_116` | 0.6416 | 0.0000 |
| `regime_switch_pomdp_116` | 2.8734 | 0.1000 |
| `allee_ricker_pomdp_116_10a` | 5.8645 | 0.2500 |
| `theta_logistic_pomdp_116_10a` | 2.7926 | 0.0500 |
| `regime_switch_pomdp_116_10a` | 7.0536 | 0.3000 |

`stress_pomdp_116_manifest_worker.sh` checks the collapse-sensitive gate summary
by env key, so the same gate summaries cover final `base` rows too.

### `rtq` submission syntax is accepted

I ran `sbatch --test-only --qos=rtq --array=0-0%1 ...` against the manifest
worker. Slurm accepted the request and returned a hypothetical scheduled job.

## Disagreements / Required Fixes Before Launch

### 1. Final manifest `config_tag` is wrong for frozen learned methods

This is the only issue I consider a real pre-launch bug.

The final manifest rows correctly carry the frozen parameter columns, but
`make_stress116_manifest.py` computes `config_tag` from `BASE_METHOD_CFG` instead
of the loaded frozen config when `phase` is not a tuning phase. Dry-run evidence:

```text
mopo    config_tag=mopo_h5_r50_lam1p0_ep50       actual frozen: h5 r25 lambda0.5 ep50
refplan config_tag=ref_h10_n512_l12_k5p0_u0p10   actual frozen: h8 n512 l12 k10.0 u0.2
```

This likely would not change the executed hyperparameters, because the worker
uses the explicit manifest columns. But it does affect output directory names,
report labels, and auditability. It is too confusing for a final benchmark.

Recommended fix:

- In `_method_rows`, when constructing the tag for non-tuning rows, compute the
  tag from `{**BASE_METHOD_CFG, **args.frozen_cfg, **cfg}`.
- Then regenerate the final manifests.

### 2. The LaTeX report text still says final rows use 100 eval episodes

`aggregate_stress116_phase116.py` has hardcoded text:

```text
Final rows use 100 evaluation episodes...
```

The half-day plan uses `FINAL_EVAL_EPISODES=50`, and the dry-run manifest also
has `eval_episodes=50`. This report text should be changed to read the value
from the manifest or say "the evaluation episode count recorded in the
manifest."

Recommended fix:

- Do not hardcode 100 in `_write_tex`.
- Pull unique `eval_episodes`, `eval_horizon`, and `data_n` values from the
  aggregate/seed data or pass them through to the writer.

### 3. The score CSV name is inconsistent

Claude says the freeze writes:

```text
phase116_tuning_scores_1day.csv
```

The existing freeze-only job did write that file. However,
`stress_pomdp_116_after_tune.sh` currently re-runs the selector and writes:

```text
outputs/stress_pomdp_116/frozen/phase116_tuning_scores.csv
```

This is not fatal, because the frozen env/json names are still `_1day`. But it
is inconsistent with the half-day doc and with the reduced-run naming.

Recommended fix:

- Change the after-tune selector output to
  `outputs/stress_pomdp_116/frozen/phase116_tuning_scores_1day.csv`.

## Disagreements / Caveats That Are Not Blockers

### 4. Runtime reasoning is probably wrong, but conservative

Claude's plan says MOPO is the expensive method and estimates 60 MOPO final runs
at about 50 minutes each.

The completed one-day tuning metadata says:

| Method | Rows | Mean min | Median min | P90 min | Max min |
|---|---:|---:|---:|---:|---:|
| MOPO | 108 | 3.86 | 4.03 | 5.58 | 7.83 |
| RefPlan | 108 | 51.84 | 45.72 | 105.45 | 112.33 |

So in the current code path, RefPlan was the slow method, not MOPO. The
half-day estimate is still safe because final uses only one frozen RefPlan
configuration, not the slowest tuning grid over all RefPlan configs. But the
explanation in Claude's doc should not be treated as accurate performance
diagnosis.

### 5. The report job requests a GPU even though aggregation is CPU work

`stress_pomdp_116_report.sh` has `#SBATCH --gres=gpu:L40S:1`. It will work, but
it unnecessarily consumes a GPU slot for a lightweight aggregation job.

This is not a correctness blocker. If queue pressure matters, change the report
job to CPU-only.

### 6. Direct `BASE=$PWD bash scripts/slurm/stress_pomdp_116_after_tune.sh` is okay but unusual

Claude's manual launch command runs the after-tune script directly instead of
submitting it as a Slurm job. That is acceptable because the script mostly
generates manifests and submits Slurm arrays. It assumes the login shell has
`module`/Conda access. If that is unreliable, submit the after-tune script with
`sbatch` instead.

## Overall Recommendation

I agree with the plan after these small fixes:

1. Fix final `config_tag` generation from frozen configs.
2. Fix the LaTeX report's hardcoded "100 evaluation episodes" text.
3. Align the after-tune tuning-score CSV name with `_1day`.
4. Then launch the final comparison with the half-day settings.

I would keep Claude's scientific plan intact: held-out seeds `7001--7005`, both
reward modes, all four methods, shared scenario datasets, frozen learned-method
configs, and seed-level aggregation.

