# Experiment Plan 116: Danger-Zone POMDP Stress Benchmark

## Purpose

The current `stress_pomdp_smoke` benchmark verified that the new POMDP stress
environments run end-to-end, but it did not create a strong RefPlan advantage.
Claude's audit of the report identified the main reason: after retuning away
from all-collapse dynamics, the Allee and regime-switch settings became too
gentle. Most evaluation episodes stayed near carrying capacity, hidden
thresholds rarely mattered, and fixed-family Ricker baselines were already
near-optimal. Theta-logistic remained decision-relevant, but MOPO won there
rather than RefPlan.

This plan defines a separate setting named `stress116_danger_zone`. It is not a
replacement for the current smoke setting. It is a follow-up benchmark designed
to test whether a learned, uncertainty-aware, history-conditioned method can
beat fixed-family ecology baselines when hidden stress variables actually affect
management decisions.

## Core Hypothesis

RefPlan should only have a chance to win when all of the following are true:

- The true dynamics are outside the Ricker model family used by PLUS and
  MOOR-Ricker.
- Episodes operate near a hidden threshold or hidden regime boundary.
- Bad actions can cause collapse within the planning horizon.
- The hidden variable is inferable from recent abundance-action history.
- The offline dataset contains enough live danger-zone transitions for tabular
  estimates to be meaningful.
- RefPlan receives a non-smoke planning budget.

The defensible benchmark claim should be:

> Learned uncertainty-aware methods beat fixed-family ecology baselines under
> decision-relevant misspecification.

The stronger claim:

> RefPlan beats MOPO.

requires a specifically hidden-state-inferable, longer-horizon setting where
belief over the latent threshold/regime matters more than one-step pessimistic
value estimation.

## New Setting Tag

Use `116` everywhere in paths and logging so this run is clearly separated from
the current smoke setting.

- Plan file: `docs/116_refplan_danger_zone_experiment_plan.md`
- Output root: `claude_build/outputs/stress_pomdp_116/`
- Slurm logs: `claude_build/logs/stress_pomdp_116_*`
- WandB group: `stress_pomdp_116_danger_zone`
- Optional Slurm script: `claude_build/scripts/slurm/stress_pomdp_116.sh`
- Optional config suffix: `_116`, for example
  `allee_ricker_pomdp_116.yaml`

## Environments

Keep the POMDP contract unchanged:

- Agent observation remains only the abundance bin `x_t`.
- Hidden variables `r_base`, `C`, `theta`, and `z_t` stay internal.
- Hidden variables may appear only in `info`, diagnostics, and post-hoc
  summaries.
- Saved datasets must not contain hidden-variable arrays.

Primary environments:

| Environment | Role in setting 116 |
| --- | --- |
| `allee_ricker_pomdp` | Main hidden-threshold stress test. |
| `regime_switch_pomdp` | Main hidden-regime stress test. |
| `theta_logistic_pomdp` | Misspecification control where MOPO already looked strong. |

Primary reward mode:

- `collapse_sensitive`

Secondary reward mode:

- `base`, only as an ablation after the collapse-sensitive setting works.

## Required Code Additions

### 0. Hard Decision-Relevance Gate

Before any method run, Phase 116.0 must verify that the environment has a real
control gap:

- `oracle_mpc`: receding-horizon controller that simulates the true hidden
  stress environment by copying the current env state, so it knows the hidden
  `C`, `theta`, or regime.
- `ricker_mpc`: receding-horizon controller using a compensatory Ricker model
  at the public abundance-bin midpoint. It receives the same collapse-aware
  reward but the wrong dynamics.

Gate:

```text
G_oracle - G_ricker >= 1.0
collapse_rate_ricker - collapse_rate_oracle >= 0.05
```

For Allee and regime-switch stress, this is a hard gate. If it fails, reject
that candidate environment/config before Phase 116.1.

For theta-logistic, the gate is diagnostic by default: theta has hidden
curvature rather than a hidden collapse threshold, and sweeps showed that it
can produce reward gaps without a binary collapse-rate gap. It remains a
secondary misspecification control, not a blocker for the threshold/regime
stress benchmark.

The implemented script is:

```text
claude_build/scripts/analysis/stress116_control_gap.py
```

### 1. Separate Dataset And Evaluation Start Distributions

The current environment uses `env.dataset.s0_low` and `env.dataset.s0_high` in
`reset()`, so training and evaluation start from the same distribution. Setting
116 should support a separate evaluation start distribution.

Add one of these small interfaces:

- Preferred: extend `reset(seed=None, s0_low=None, s0_high=None)`.
- Alternative: add environment attributes/setters that the evaluator can apply
  before evaluation.

Required config keys:

```yaml
env:
  dataset:
    s0_low: 60.0
    s0_high: 300.0
    episode_len: 25
  eval_start:
    s0_low: 80.0
    s0_high: 250.0
    episode_len: 50
```

The evaluator should use `env.eval_start` if present. Dataset generation should
continue to use `env.dataset`.

### 2. New Collection Policy: `mixed_danger_zone_116`

The current `mixed_danger_zone` policy produced only about 2-4 percent live
danger-zone coverage in bins 6-20. Setting 116 should intentionally keep more
transitions near the hidden threshold without making the dataset all bin 0.

Proposed episode mixture:

| Component | Weight | Behavior |
| --- | ---: | --- |
| random_low_start | 0.25 | Uniform actions, low/mid initial abundance. |
| harvest_probe | 0.30 | Prefer harvest until `x <= 25`, then switch to do-nothing/support dwell. |
| rescue_dwell | 0.25 | If `x <= 25`, dwell with no-op/support/restoration; otherwise mild harvest/do-nothing. |
| threshold_probe | 0.20 | Alternate action 0/1 early to reveal hidden threshold response, then act randomly. |

Setting 116 gives actions direct abundance authority in addition to
`delta_r`/`delta_K`: action 1 removes a fraction of abundance, while actions
2/3/4 add stocking/translocation pulses. This fixes the prior dead-end where
all actions landed in almost the same next bin and the hidden threshold was not
a decision.

Calibration targets:

- `fraction(6 <= x <= 20)`: 0.15 to 0.25
- `fraction(x <= 20)`: 0.25 to 0.45
- `fraction(x == 0)`: below 0.30
- episode-level collapse entry during data collection: 0.10 to 0.35

The coverage checks must be episode-level where appropriate. Do not divide
episode collapse counts by transition count.

### 3. Stronger But Nondegenerate Stress Configs

Do not jump directly to the original high-growth priors that caused near-total
collapse. Use a calibration sweep first.

Calibrated candidate configs:

| Env | Candidate parameters |
| --- | --- |
| Allee | `r_base in U(0.12,0.30)`, `C in U(90,150)`, collection `s0 in [80,280]`, collection `episode_len=25`, eval `episode_len=50` |
| Theta | `r_base in U(0.18,0.40)`, `theta in U(3,6)`, collection `s0 in [60,200]`, collection `episode_len=20`, eval `episode_len=50`; secondary diagnostic/control |
| Regime | `r_base in U(0.12,0.30)`, safe `C=90`, harsh `C=180`, harsh growth multiplier `0.65`, stay probability `0.90`, collection `episode_len=30`, eval `episode_len=50` |

Calibrated 5-action authority:

| Action | Direct effect |
| --- | --- |
| 0 Do nothing | no direct abundance change |
| 1 Aggressive harvest | `harvest_fraction=0.50` |
| 2 Disease/predator control | `stocking_delta=60` |
| 3 Intensive restoration | `stocking_delta=100` |
| 4 Flagship conservation | `stocking_delta=160` |

These shorter collection episodes are only for dataset coverage. Evaluation
episodes still use `eval.horizon=50` and `env.eval_start.episode_len=50`.

Collapse-sensitive reward:

- Keep the one-time collapse-entry penalty semantics.
- Test `collapse_penalty in {10, 20}` during calibration.
- Use `collapse_penalty=20` for the main 116 run only if the naive/Ricker
  policies still over-harvest into collapse.

### 4. Hidden-State-Inferable Evaluation Variant

Add an optional evaluation warm-up/probing mode if the plain danger-zone start
does not separate RefPlan from MOPO.

Proposed option:

```yaml
eval:
  probe_steps: 5
  probe_policy: noop_or_light_harvest
  score_probe_steps: false
```

During probe steps, all methods receive the same public history
`(x_t, a_t, x_{t+1})`, but no hidden variables. Scoring begins after the probe.
This tests whether methods can use public history to infer the latent threshold
or regime before making high-stakes decisions.

Use this only as a secondary variant. The primary setting should first try
ordinary agent-controlled evaluation from the danger-zone start distribution.

RefPlan code check:

- `RefPlanAdapter` conditions on the full public `(x_t,a_t,x_{t+1})` history
  through `TabularDynamicsEnsemble.posterior_from_history`.
- The regression test
  `claude_build/tests/test_stress116_experiment.py::test_refplan_posterior_depends_on_public_transition_history`
  verifies that two public histories ending in different observed transitions
  produce different posterior mass over latent transition members.
- No hidden variable is added to the RefPlan observation.

## Methods

Compare only the selected four methods:

- `RefPlan`
- `MOPO`
- `PLUS`
- `MOOR-Ricker`

Keep PLUS and MOOR-Ricker intentionally misspecified:

- PLUS uses its compensatory/Ricker-style internal candidate models.
- MOOR uses `dynamics_model=ricker` and learns `[r,K]`.
- Neither baseline receives hidden variables.

Reward handling uses Claude's recommended option (a):

- PLUS and MOOR-Ricker do see the collapse penalty.
- They apply that penalty under their own wrong compensatory/Ricker transition
  model.
- Therefore a learned-method win is attributable to dynamics misspecification
  and hidden-threshold modeling, not to the baselines being blind to the reward.

## Hyperparameters

### RefPlan 116 Budget

The smoke run used a very small RefPlan search budget. Setting 116 should use a
larger budget before judging RefPlan.

Primary RefPlan config:

```text
model.ensemble_size=15
model.planning_horizon=10
model.num_sequences=512
model.latent_samples=12
model.kappa=5.0
model.uncertainty_penalty=0.10
```

Small tuning grid on calibration seeds:

```text
planning_horizon in {8, 10, 12}
num_sequences in {256, 512, 1024}
latent_samples in {8, 12, 16}
kappa in {2.0, 5.0, 10.0}
uncertainty_penalty in {0.05, 0.10, 0.20}
```

Freeze one RefPlan config before final test seeds.

### MOPO 116 Budget

Use a fair non-smoke MOPO budget:

```text
model.ensemble_size=5
model.frame_stack_len=2
model.bootstrap_ratio=0.8
model.num_epochs=50
planner.pessimism_lambda=1.0
planner.horizon=5
planner.num_rollouts=10 for Phase 116.1
planner.num_rollouts=25 only after the control-gap gate passes
```

Optional tuning grid:

```text
planner.horizon in {5, 8}
planner.num_rollouts in {25, 50}
planner.pessimism_lambda in {0.5, 1.0, 2.0}
model.num_epochs in {50, 100}
```

Compute guard:

- Phase 116.1 defaults to 20 evaluation episodes and 10 MOPO rollouts.
- Phase 116.2 defaults to 50 evaluation episodes and at most 25 MOPO rollouts.
- Do not launch Phase 116.3 until per-method wall-clock from Phase 116.2 is
  reviewed.
- Record GPU memory/utilization with the wrapper in the Slurm script.

### PLUS

```text
model.num_candidate_models=21
model.discount=0.95
model.transition_n_grid=2000
model.use_env_transition=false
model.likelihood_floor=1e-12
```

### MOOR-Ricker

```text
model.dynamics_model=ricker
model.learn_params=[r,K]
model.n_restarts=30
model.lbfgs_max_iter=20
model.transition_n_grid=2000
model.discount=0.95
```

## Run Phases

### Phase 116.0: Calibration Without Expensive Methods

Purpose: choose nondegenerate danger-zone parameters.

Run cheap rollouts under:

- random policy
- harvest-heavy policy
- rescue policy
- fixed Ricker-like greedy policy if available

Candidate environments:

- Allee collapse-sensitive (hard gate)
- Regime collapse-sensitive (hard gate)
- Theta collapse-sensitive (diagnostic control)

Use:

```text
seeds = 1160, 1161, 1162
DATA_N = 25000
eval episode_len = 50
```

Pass criteria:

- live danger-zone coverage 15-25 percent
- zero-bin fraction below 30 percent
- collapse-entry rate under harvest-heavy policy at least 30 percent
- collapse-entry rate under rescue policy below harvest-heavy rate
- oracle-vs-Ricker control gap passes for Allee and regime:
  - `G_oracle - G_ricker >= 1.0`
  - `collapse_rate_ricker - collapse_rate_oracle >= 0.05`
- theta-logistic reports the control-gap diagnostics but does not block
  Phase 116.1 by default
- hidden variables are absent from saved dataset arrays

### Phase 116.1: One-Seed Smoke

Purpose: verify the new harness and tuned configs before spending more compute.

Run:

```text
primary envs = allee_ricker_pomdp_116, regime_switch_pomdp_116
secondary diagnostic env = theta_logistic_pomdp_116
reward_mode = collapse_sensitive
methods = mopo, refplan, plus, moor_ricker
seed = 116
DATA_N = 25000
EVAL_EPISODES = 20
EVAL_HORIZON = 50
WANDB_MODE = online
WANDB_GROUP = stress_pomdp_116_danger_zone_smoke
MOPO_ROLLOUTS = 10
```

Expected pass criteria:

- all 12 method jobs complete
- all methods write CSV and JSON
- MOPO, RefPlan, and MOOR-Ricker share the same dataset path per scenario
- WandB runs sync successfully
- result table includes runtime, GPU type, and GPU memory if wrapper is enabled

### Phase 116.2: Tuned Short Benchmark

Purpose: decide whether the 116 setting creates the intended learned-method
advantage.

Run:

```text
primary envs = allee_ricker_pomdp_116, regime_switch_pomdp_116
secondary diagnostic env = theta_logistic_pomdp_116
reward_mode = collapse_sensitive
methods = mopo, refplan, plus, moor_ricker
seeds = 1161, 1162, 1163
DATA_N = 50000
EVAL_EPISODES = 50
EVAL_HORIZON = 50
WANDB_MODE = online
WANDB_GROUP = stress_pomdp_116_danger_zone
MOPO_ROLLOUTS = 25
```

Primary pass/fail question:

- Does at least one learned method beat both fixed-family baselines in at least
  both hard-gated primary environments?

RefPlan-specific question:

- Does RefPlan beat MOPO in at least one hidden-state-inferable environment
  after receiving the larger planning budget?

### Phase 116.3: Final Test If Phase 116.2 Works

Only run this after configs are frozen.

```text
envs = selected successful 116 environments
reward_mode = collapse_sensitive
methods = mopo, refplan, plus, moor_ricker
seeds = 7001, 7002, 7003, 7004, 7005
DATA_N = 50000 or 75000
EVAL_EPISODES = 100
EVAL_HORIZON = 50
WANDB_MODE = online
WANDB_GROUP = stress_pomdp_116_final
```

Do not use final seeds to tune parameters.
Before launching this phase, check Phase 116.2 MOPO runtime. If MOPO wall-clock
is too high, reduce final `EVAL_EPISODES` or reserve faster GPUs/longer walls.

## Metrics To Report

Keep existing metrics:

- cumulative reward mean/std and per-episode min/max
- final abundance mean/std
- minimum abundance mean/std
- catastrophic low-abundance rate
- collapse-entry rate
- uncertainty mean
- saved `policy_entropy` field
- runtime
- GPU type
- peak GPU memory if available

Add 116-specific diagnostics:

- live danger-zone visit fraction during evaluation: `6 <= x <= 20`
- action distribution inside live danger zone
- mean collapse-entry timestep, conditional on collapse
- hidden-bucket performance:
  - Allee: by `C_low`, `C_mid`, `C_high`
  - Theta: by `theta_low`, `theta_mid`, `theta_high`
  - Regime: by majority safe/harsh
- learned-method advantage over best fixed-family baseline:
  `G_method - max(G_PLUS, G_MOOR)`
- recovery-zone dataset coverage: `21 <= x <= 50`, to check that models see
  enough recovery-to-K dynamics after starting below K.

For final comparisons, aggregate at the seed level. Do not treat individual
episodes as independent trained agents.

## GPU And Runtime Logging

Add a lightweight wrapper around method execution:

```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu \
  --format=csv -l 5 > "$OUT_ROOT/gpu_${METHOD}_${SEED}.csv" &
GPU_MON_PID=$!
trap 'kill $GPU_MON_PID 2>/dev/null || true' EXIT
```

Summarize:

- GPU name
- peak `memory.used`
- mean/max utilization if available
- wall-clock runtime from the script timer
- Slurm elapsed time from `sacct`

## Proposed Slurm Structure

Use separate dataset and method arrays, as in the smoke run.

Dataset stage:

```text
3 envs x 1 reward mode x seeds
```

Method stage:

```text
3 envs x 1 reward mode x seeds x 4 methods
```

Use `%4` concurrency initially. Increase only after the first smoke passes.

Example command shape:

```bash
sbatch --array=0-8%4 \
  --export=ALL,WB_MODE=online,WB_PROJECT=deeprl_population_models,WB_GROUP=stress_pomdp_116_danger_zone \
  claude_build/scripts/slurm/stress_pomdp_116.sh dataset

sbatch --dependency=afterok:<DATASET_JOB_ID> --array=0-35%4 \
  --export=ALL,WB_MODE=online,WB_PROJECT=deeprl_population_models,WB_GROUP=stress_pomdp_116_danger_zone \
  claude_build/scripts/slurm/stress_pomdp_116.sh run
```

## Interpretation Rules

Do not claim RefPlan is best from a one-seed or 10-episode run.

A successful 116 result should show at least one of:

- learned methods beat both fixed-family baselines in the danger-zone setting;
- MOPO remains strongest, confirming that model-free-family learning helps but
  RefPlan is not yet competitive;
- RefPlan improves specifically when the hidden variable is inferable from
  history and the planning budget is increased.

If PLUS and MOOR-Ricker tie exactly again, treat them as one effective
deterministic Ricker-family policy for interpretation.

If all methods remain near-identical, the setting is still not
decision-relevant enough and should be rejected rather than overinterpreted.

## Deliverables

After Phase 116.1 or 116.2, produce:

- compact CSV/Markdown result table sorted by cumulative reward
- `.tex` report under `docs/latex_files_reports/` or `docs/`
- WandB group link/name
- dataset coverage table
- runtime/GPU-memory table
- short explanation of whether setting 116 actually fixed the weaknesses of
  the current smoke setting
