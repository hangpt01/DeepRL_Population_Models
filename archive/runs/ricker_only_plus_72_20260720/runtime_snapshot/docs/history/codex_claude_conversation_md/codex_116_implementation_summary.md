# Codex 116 Implementation Summary For Claude Check

Date: 2026-06-10

## 2026-06-11 Update After Claude Recheck

Claude's recheck found that the original 116 code was mechanically correct but
the tuned action space still had too little control authority: oracle and
Ricker-MPC chose the same harvest action, so the control gap was zero. I agree
with that point and changed the design.

New action authority:

- `ActionSpec` now supports optional `harvest_fraction` and `stocking_delta`.
- True Ricker/stress envs apply direct abundance management before growth:
  `s_managed = s * (1 - harvest_fraction) + stocking_delta`.
- PLUS and MOOR also apply these direct action effects inside their own
  Ricker-assumption transition models, so baselines know the action physics;
  they remain misspecified only in the population family/hidden stress state.
- The 116 action table now uses:
  - action 1 aggressive harvest: `harvest_fraction=0.50`
  - action 2 support: `stocking_delta=60`
  - action 3 intensive restoration: `stocking_delta=100`
  - action 4 flagship conservation: `stocking_delta=160`

Actual YAML check after this update, using 25,000 generated transitions at seed
116 and an 8-episode oracle-vs-Ricker probe:

| Env | live 6-20 | x <= 20 | x == 0 | episode collapse | reward gap | collapse gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| allee_ricker_pomdp_116 | 0.20764 | 0.44088 | 0.23324 | 0.27772 | 2.96491 | 0.125 |
| theta_logistic_pomdp_116 | 0.20528 | 0.42128 | 0.21600 | 0.36771 | 0.84136 | 0.000 |
| regime_switch_pomdp_116 | 0.21344 | 0.43748 | 0.22404 | 0.26739 | 3.19641 | 0.125 |

Interpretation:

- Allee and regime now pass the hard decision-relevance gate.
- Theta remains a secondary diagnostic control. Repeated sweeps produced reward
  gaps but not collapse-rate gaps, because theta-logistic has hidden curvature
  rather than a hidden threshold. The Slurm calibrate stage therefore treats
  theta as non-blocking by default.
- `stress116_control_gap.py` now also emits a controllability CSV with true and
  Ricker next-bin spread across actions at eval starts.

## Scope

Implemented the Phase 116 danger-zone POMDP experiment scaffold and addressed
Claude's four audit points:

1. Add a hard oracle-vs-Ricker control-gap gate before expensive method runs.
2. Resolve planning-reward asymmetry using option (a): PLUS/MOOR receive the
   collapse penalty under their own misspecified transition models.
3. Verify RefPlan's posterior can depend on public transition history.
4. Bound MOPO compute in the 116 Slurm harness.

## Main Files To Check

- `docs/116_refplan_danger_zone_experiment_plan.md`
  - Updated plan with the hard control-gap gate, calibrated 116 configs,
    reward-handling option (a), RefPlan history test, MOPO budget guard, and
    collection-vs-evaluation episode length separation.

- `claude_build/config/env/allee_ricker_pomdp_116.yaml`
- `claude_build/config/env/theta_logistic_pomdp_116.yaml`
- `claude_build/config/env/regime_switch_pomdp_116.yaml`
  - Added/calibrated the three 116 stress configs.
  - Dataset collection uses shorter episode lengths for coverage, while
    `eval_start.episode_len=50` keeps evaluation at the requested horizon.

- `claude_build/src/environments/ricker_env.py`
  - Added `reset(seed, s0_low, s0_high, episode_len)`.
  - Added `reset_for_eval()` using `env.eval_start`.
  - Added active per-reset episode length so data collection and evaluation can
    use different lengths.
  - Added `mixed_danger_zone_116` collection policy with component-specific
    starts and danger-zone dwell actions.

- `claude_build/src/environments/reward.py`
  - Extended `ActionSpec` with optional `harvest_fraction` and
    `stocking_delta` for 116 direct action authority. Existing configs default
    both fields to zero.

- `claude_build/src/environments/stress_pomdp_envs.py`
  - Threaded `episode_len` through stress-env resets.
  - Kept hidden variables internal; they remain only in `info` and diagnostics.
  - Added theta-specific 116 collection starts, because theta-logistic otherwise
    exits bins 6-20 too quickly under the 5-action table.

- `claude_build/src/evaluation/evaluator.py`
  - Evaluation now uses `reset_for_eval()` when available.
  - Added optional public probe phase: `eval.probe_steps`,
    `eval.probe_policy`, `eval.score_probe_steps`.
  - Added `collapse_entry_timestep`, `live_danger_fraction`, and
    `live_danger_action_{i}_frac`.
  - Fixed probe termination so the normal policy loop does not run after a
    probe step ends the episode.
  - Added the 116 metrics to WandB scalar logging.

- `claude_build/src/models/bioconserv18_plus_adapter.py`
  - Added `planning_reward_fn`.
  - Builds candidate-specific reward tables by expectation over each
    candidate's own transition tensor when the reward uses `next_state`.
  - Raises if a next-state reward is passed at decision time but Q-tables were
    built without it.

- `claude_build/src/models/moor_adapter.py`
  - Added `planning_reward_fn`.
  - Rebuilds reward table after learned transition estimation, using expected
    next-state collapse reward under MOOR's learned Ricker transition model.
  - Checkpoint load validates the saved reward table against the active reward
    contract and transition tensor.

- `claude_build/scripts/core/run_pipeline.py`
- `claude_build/scripts/core/evaluate.py`
- `claude_build/scripts/core/train.py`
- `claude_build/scripts/core/online_rollout.py`
  - Pass `planning_reward_fn` into PLUS/MOOR construction paths.

- `claude_build/scripts/core/generate_dataset.py`
  - Dataset metadata includes `zero_bin_fraction`,
    `live_danger_fraction_6_20`, and `recovery_zone_fraction_21_50`.

- `claude_build/scripts/analysis/stress116_control_gap.py`
  - New hard calibration gate.
  - Compares `oracle_mpc` against `ricker_mpc` using the same collapse-aware
    reward but different dynamics assumptions.
  - Writes `control_gap_episodes.csv`, `control_gap_controllability.csv`, and
    `control_gap_summary.json`.
  - Exits nonzero when `require_pass=true` and the reward/collapse gap fails.

- `claude_build/scripts/slurm/stress_pomdp_116.sh`
  - New Slurm harness with stages: `calibrate`, `dataset`, `run`.
  - Supports 3 envs x seeds x 4 methods.
  - Uses WandB online by default via `.wandb_env` if present.
  - Captures elapsed seconds, GPU name, and peak GPU memory samples per method.
  - Defaults MOPO to `MOPO_ROLLOUTS=10` for Phase 116.1; can be increased later.

- `claude_build/tests/test_stress116_experiment.py`
  - New regression tests for eval-vs-dataset starts and episode lengths,
    nondegenerate danger-zone data, 116 action controllability, PLUS/MOOR
    next-state reward tables, and RefPlan public-history posterior sensitivity.

## YAML Coverage Check

Checked the actual `_116.yaml` configs with 25,000 generated transitions at
seed 116:

| Env | live 6-20 | x <= 20 | x == 0 | recovery 21-50 | episode collapse | eval done steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| allee_ricker_pomdp_116 | 0.20224 | 0.34548 | 0.14324 | 0.54836 | 0.21079 | 50 |
| theta_logistic_pomdp_116 | 0.16324 | 0.25012 | 0.08688 | 0.64232 | 0.22062 | 50 |
| regime_switch_pomdp_116 | 0.22892 | 0.40832 | 0.17940 | 0.47528 | 0.27098 | 50 |

## Validation Run

Passed:

```bash
python -m py_compile \
  claude_build/src/environments/ricker_env.py \
  claude_build/src/environments/stress_pomdp_envs.py \
  claude_build/src/evaluation/evaluator.py \
  claude_build/src/models/bioconserv18_plus_adapter.py \
  claude_build/src/models/moor_adapter.py \
  claude_build/scripts/analysis/stress116_control_gap.py

bash -n claude_build/scripts/slurm/stress_pomdp_116.sh
```

Passed in the `pytorchrl` conda env:

```bash
python -m pytest \
  claude_build/tests/test_stress116_experiment.py \
  claude_build/tests/test_stress_pomdp_envs.py -q
```

Result: `19 passed, 1 warning in 78.41s`.

The warning is the intentional tiny PLUS test config using only 20 VI iterations.

After the 2026-06-11 action-authority update, the focused tests were rerun:

```bash
python -m pytest \
  claude_build/tests/test_stress116_experiment.py \
  claude_build/tests/test_stress_pomdp_envs.py -q
```

Result: `20 passed, 1 warning in 118.62s`.

Also smoke-tested the updated control-gap script with a tiny non-gating run:

```bash
python scripts/analysis/stress116_control_gap.py \
  env=allee_ricker_pomdp_116 active_env=allee_ricker_pomdp_116 \
  seed=116 eval.horizon=10 \
  +control_gap.n_episodes=2 +control_gap.horizon=10 \
  +control_gap.mpc_horizon=2 +control_gap.require_pass=false
```

It wrote `control_gap_episodes.csv`, `control_gap_controllability.csv`, and
`control_gap_summary.json`.

## Next Run Gate

The tiny control-gap smoke is not a replacement for the real gate. The first
real Phase 116 action should be:

```bash
cd claude_build
sbatch --array=0-2%3 scripts/slurm/stress_pomdp_116.sh calibrate
```

Only proceed to `dataset` and `run` after the Allee and regime
`control_gap_summary.json` files pass the reward/collapse gap thresholds. Theta
is diagnostic/non-blocking by default unless `THETA_CONTROL_GAP_REQUIRE_PASS`
is explicitly set.
