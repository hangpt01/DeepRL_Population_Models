# Codex Chat Handoff: Phase 116 And Current Experiment Flow

Last updated: 2026-06-13.

Use this file to move to another chat without losing the current coding and
experiment context. The active experiment flow should not be interrupted unless
the user explicitly asks to cancel, resubmit, or change jobs.

## Repository And Working Area

- Repo: `/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models`
- Active project subdir: `claude_build`
- Main current topic: Phase 116 decision-relevant POMDP stress benchmark for
  population-management RL.
- Main active status file for Claude audit:
  `docs/codex_claude_conversation_md/codex_116_full_benchmark_run_status.md`
- Paper/slide-style experiment-setting file:
  `docs/phase116_current_experiment_setting.md`

## High-Level Story Of This Chat

The chat began with a narrow selected-method benchmark request: compare RefPlan
and MOPO against related ecology baselines, specifically BioConserv18 PLUS and
ExpertSys23 MOOR-Ricker, rather than running every implemented method.

The early benchmark used the 5-action Ricker environment with seed 42,
`DATA_N=25000`, `EVAL_EPISODES=50`, `EVAL_HORIZON=50`, WandB online, shared
offline dataset paths, and concurrent GPU jobs. It also explored whether a
Schaefer/logistic true-environment stress setting existed. A compact LaTeX
report was produced and then discussed in detail.

The discussion then moved into interpreting the report:

- Why the Schaefer/logistic equation looked similar to Ricker.
- What `r_eff`, `K_eff`, `r_base`, and `K_base` mean.
- Why both Ricker and Schaefer can share fixed points in the low-abundance
  regime.
- How cumulative reward is computed over horizon 50.
- What final abundance, minimum abundance, and catastrophic rate mean.
- What RefPlan's latent VAE ensemble and uncertainty-aware planning settings
  mean.
- Why final/min abundance values can be non-integers: they are means over
  episode-level integer abundance bins.
- Why MOPO was much slower than RefPlan in the early runs.
- How uncertainty and policy-entropy tables should be read.

After that, the user asked to implement the POMDP stress-model task from
`docs/codex_stress_models_pomdp_task.md`. Three hidden-variable stress
environments were added and tested:

- Allee-threshold stress.
- Theta-logistic stress.
- Regime-switch stress.

The design keeps hidden variables out of the agent observation. Hidden
parameters/regime values are sampled internally and only exposed through
diagnostics/info, not through the policy observation or offline dataset features.

Claude audited the first implementation and found the mechanics mostly correct
but the generated datasets were degenerate. The main fixes were:

- Correct the collapse-rate denominator in tests.
- Retune environment/collection settings to avoid all-collapse data.
- Add required metrics and hidden-bucket diagnostics.
- Clarify collapse-sensitive reward behavior.
- Keep POMDP hidden variables out of observations.

Later Claude found that the first Phase 116 gate passed mechanically but was not
decision-relevant because actions had too little authority over hidden-threshold
crossing. The fix was to add real direct management authority through action
effects:

- `harvest_fraction`
- `stocking_delta`

The action is applied before growth as a managed abundance, conceptually:

```text
s_managed = s * (1 - harvest_fraction) + stocking_delta
```

This action-authority model was mirrored in the true environments and in the
Ricker-assumption transitions used by PLUS and MOOR, so the comparison is about
hidden dynamics/model-family misspecification rather than different action
physics. Allee and regime are treated as hard-gated decision-relevant settings;
theta is diagnostic/relaxed.

## Main Code/Experiment Changes Made

Phase 116 environments and configs:

- Added/updated 5-action Phase 116 stress configs:
  - `claude_build/config/env/allee_ricker_pomdp_116.yaml`
  - `claude_build/config/env/theta_logistic_pomdp_116.yaml`
  - `claude_build/config/env/regime_switch_pomdp_116.yaml`
- Added 10-action Phase 116 stress configs:
  - `claude_build/config/env/allee_ricker_pomdp_116_10a.yaml`
  - `claude_build/config/env/theta_logistic_pomdp_116_10a.yaml`
  - `claude_build/config/env/regime_switch_pomdp_116_10a.yaml`
- Registered the new env IDs in `claude_build/src/environments/__init__.py`.
- Implemented dynamics/action handling in
  `claude_build/src/environments/ricker_env.py`.

MOPO planner speed work:

- Added batched ensemble prediction:
  `claude_build/src/models/ensemble.py`
- Added vectorized pessimistic planner rollout path:
  `claude_build/src/planners/pessimistic_planner.py`
- Set vectorized planner behavior in:
  `claude_build/config/planner/pessimistic.yaml`
- Added numerical identity tests:
  `claude_build/tests/test_pessimistic_planner_vectorized.py`

Phase 116 automation:

- `claude_build/scripts/slurm/stress_pomdp_116.sh`
- `claude_build/scripts/slurm/stress_pomdp_116_manifest_worker.sh`
- `claude_build/scripts/slurm/stress_pomdp_116_freeze_only.sh`
- `claude_build/scripts/experiments/make_stress116_manifest.py`
- `claude_build/scripts/analysis/select_stress116_frozen.py`
- `claude_build/scripts/analysis/stress116_control_gap.py`
- `claude_build/scripts/analysis/aggregate_stress116_phase116.py`

Important Slurm/script fixes already applied:

- Fixed Slurm `BASE` detection so jobs launched from repo root do not use the
  Slurm spool path as the project base.
- Fixed Hydra live danger-zone overrides by using
  `+eval.live_danger_low` and `+eval.live_danger_high`.
- Fixed TSV line endings so CRLF headers do not produce Bash keys with `\r`.
- Replaced fragile Bash row parsing with Python `csv.DictReader` so empty TSV
  fields do not shift columns.
- Fixed chunk recursion to call the worker through `$BASE/scripts/slurm/...`
  instead of a path relative to `/var/spool`.
- Added a freeze-only job so the current reduced tuning run will not
  accidentally launch the full final benchmark.

## Current Active Experiment

The current active run is not the full final benchmark. It is a reduced one-day
tuning run for MOPO and RefPlan only, followed by freeze-only config selection.

Active tuning manifest:

```text
claude_build/outputs/manifests/stress116_tune_methods_1day.tsv
```

Current matrix:

- Rows: 216.
- Methods: `mopo`, `refplan`.
- Env/action cells:
  - `allee_ricker_pomdp_116`
  - `theta_logistic_pomdp_116`
  - `regime_switch_pomdp_116`
  - `allee_ricker_pomdp_116_10a`
  - `theta_logistic_pomdp_116_10a`
  - `regime_switch_pomdp_116_10a`
- Reward mode: `collapse_sensitive`.
- Tuning seeds: `1160`, `1161`, `1162`.
- Dataset size: 50,000 transitions.
- Eval during tuning: 50 episodes, horizon 50.
- WandB: online.
- WandB project: `deeprl_population_models`.
- WandB group: `stress_pomdp_116_tune_1day`.
- Slurm concurrency: array `%4`.
- Rows per task: `ROWS_PER_TASK=2`.
- Per-task request: one L40S GPU, 8 CPUs, 64 GB RAM, 8-hour wall time.

Shared dataset manifest:

```text
claude_build/outputs/manifests/stress116_tune_datasets.tsv
```

Shared dataset location:

```text
claude_build/outputs/stress_pomdp_116/shared_datasets/tune116/
```

There should be 18 shared `.npz` datasets: 6 env/action cells x 3 tuning seeds.
These are reused by MOPO and RefPlan.

## Current Slurm Queue Snapshot

Last checked with `squeue -u hphung` on 2026-06-13:

```text
56433936_[71-107%4]  stress11  PD  (JobArrayTaskLimit)
56433939             stress11  PD  (Dependency)
56433936_70          stress11  R   0:09:11  m3g102
56433936_69          stress11  R   0:11:13  m3g114
56433936_65          stress11  R   1:20:07  m3g112
56433936_64          stress11  R   1:28:14  m3g105
```

Interpretation:

- `56433936` is the active reduced one-day tuning array.
- Pending tasks are waiting because of the `%4` array concurrency cap, not
  because of a dependency problem.
- `56433939` is waiting on `afterok:56433936`.
- `56433939` only selects frozen configs; it does not submit final benchmark
  jobs.

Do not cancel or resubmit these jobs unless the user asks.

## Current Gate Results

All six control-gap gates passed. Gate seed is 116, 20 episodes, horizon 50,
MPC horizon 4.

| Env/action cell | Reward gap | Collapse gap | Gate status |
|---|---:|---:|---|
| `allee_ricker_pomdp_116` | 2.4912 | 0.1000 | pass |
| `theta_logistic_pomdp_116` | 0.6416 | 0.0000 | diagnostic pass |
| `regime_switch_pomdp_116` | 2.8734 | 0.1000 | pass |
| `allee_ricker_pomdp_116_10a` | 5.8645 | 0.2500 | pass |
| `theta_logistic_pomdp_116_10a` | 2.7926 | 0.0500 | diagnostic pass |
| `regime_switch_pomdp_116_10a` | 7.0536 | 0.3000 | pass |

Gate JSONs:

```text
claude_build/outputs/stress_pomdp_116/*_collapse_sensitive_seed116/control_gap/control_gap_summary.json
```

## Current One-Day Tuning Configs

MOPO configs:

| Config tag | Horizon | Rollouts | Lambda | Epochs |
|---|---:|---:|---:|---:|
| `mopo_h5_r20_lam0p5_ep50` | 5 | 20 | 0.5 | 50 |
| `mopo_h5_r20_lam1p0_ep100` | 5 | 20 | 1.0 | 100 |
| `mopo_h5_r25_lam0p5_ep50` | 5 | 25 | 0.5 | 50 |
| `mopo_h5_r25_lam1p0_ep50` | 5 | 25 | 1.0 | 50 |
| `mopo_h5_r25_lam2p0_ep50` | 5 | 25 | 2.0 | 50 |
| `mopo_h5_r50_lam0p5_ep100` | 5 | 50 | 0.5 | 100 |

RefPlan configs:

| Config tag | Horizon | Sequences | Latents | Kappa | Unc. penalty |
|---|---:|---:|---:|---:|---:|
| `ref_h10_n256_l8_k10p0_u0p1` | 10 | 256 | 8 | 10.0 | 0.10 |
| `ref_h10_n512_l12_k5p0_u0p1` | 10 | 512 | 12 | 5.0 | 0.10 |
| `ref_h8_n1024_l12_k5p0_u0p05` | 8 | 1024 | 12 | 5.0 | 0.05 |
| `ref_h8_n256_l16_k5p0_u0p1` | 8 | 256 | 16 | 5.0 | 0.10 |
| `ref_h8_n256_l8_k2p0_u0p05` | 8 | 256 | 8 | 2.0 | 0.05 |
| `ref_h8_n512_l12_k10p0_u0p2` | 8 | 512 | 12 | 10.0 | 0.20 |

## What Is Not Running Yet

These are planned but not active in the current Slurm queue:

- Final benchmark seeds `7001--7005`.
- Final 100-episode evaluation.
- `base` reward-mode ablation.
- Final PLUS and MOOR-Ricker evaluation.
- Final report aggregation.
- Full final LaTeX report for Phase 116.
- The earlier long 1296-row matched-budget tuning chain.
- The after-tune script that would automatically launch final jobs.

The current run is a reduced tuning/freeze pass only.

## Important Docs Produced In This Chat

- `docs/quick_selected_methods_report.tex`
  - Earlier selected-method benchmark report.
- `docs/phase116_current_experiment_setting.md`
  - Paper/slide-style description of the current Phase 116 experiment setting.
  - The user explicitly asked that this file not contain repo-operational
    details.
- `docs/codex_claude_conversation_md/codex_116_full_benchmark_run_status.md`
  - Operational status/audit handoff for Claude about the current running jobs.
- This file:
  - `docs/codex_claude_conversation_md/codex_chat_handoff_phase116_future_tasks.md`

## Suggested Next Steps In A New Chat

If the new chat is about auditing the current run:

1. Read `docs/codex_claude_conversation_md/codex_116_full_benchmark_run_status.md`.
2. Verify `squeue -u hphung`.
3. Check whether job `56433936` has completed.
4. If complete, check whether job `56433939` produced:
   - `claude_build/outputs/stress_pomdp_116/frozen/phase116_frozen_1day.env`
   - `claude_build/outputs/stress_pomdp_116/frozen/phase116_frozen_1day.json`
   - `claude_build/outputs/stress_pomdp_116/frozen/phase116_tuning_scores_1day.csv`
5. Audit the scripts listed in the status file against the active manifest.

If the new chat is about running the final benchmark:

1. Do not tune on final seeds.
2. Use the frozen configs selected from the one-day tuning run.
3. Decide explicitly whether to run the full original final matrix or another
   reduced matrix.
4. Include PLUS and MOOR-Ricker in the final comparison.
5. Include both `collapse_sensitive` and `base` if the user wants the planned
   primary-plus-ablation design.
6. Preserve shared datasets per scenario and identical eval episode seeds across
   methods.

If the new chat is about writing slides/docs:

1. Use `docs/phase116_current_experiment_setting.md` for paper-style experiment
   details.
2. Avoid putting repo paths, job IDs, and Slurm implementation details into the
   slide-facing document unless the user asks for an audit/engineering slide.
3. Use the operational status file only for implementation/audit context.

## Cautions For Future Work

- Do not overwrite or simplify the POMDP contract: hidden variables must stay
  out of agent observations and offline dataset features.
- Do not accidentally launch the full final benchmark from the current
  freeze-only chain.
- Be careful with Bash TSV parsing; empty fields caused a previous serious
  bug.
- Be careful with Slurm path detection; the spool directory caused previous
  failures.
- Theta is diagnostic/relaxed; Allee and regime are the hard decision-relevant
  gate settings.
- The current one-day tuning run is a compromise requested by the user after the
  full matched-budget grid was estimated at 5--7 days.
- WandB is intentionally online for the current run; do not expose or print the
  API key.

