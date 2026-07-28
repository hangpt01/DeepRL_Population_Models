# Plan: Hardened Experiment Protocol For Ecology Baseline Comparison

## Scope

This plan compares the methods currently implemented in `claude_build` under the updated Ricker adaptive-management formulation:

- `PLUS`
- `MOPO`
- `MOReL`
- `COMBO`
- `CQL`
- `IQL`
- `BAMCTS-Tabular`
- `ROMI-Tabular`
- `RefPlan-Tabular`
- `MOOR`

The primary benchmark is the 5-action `ricker` environment. The 10-action `ricker_full` environment is a secondary robustness benchmark. The legacy `hmMDP` baseline is excluded from all phases because its fixed 4-action MOMDP action space is incompatible with both the 5-action and 10-action environments.

The final comparison must use disjoint tuning and test seeds. Any seed used to pick hyperparameters is forbidden in the final table.

## Required Code And Harness Fixes Before Expensive Runs

### Model Safety Fixes

- Add `_reward_table_ready: bool = False` to `bamcts`, `romi`, and `refplan`.
- Set `_reward_table_ready=True` only after reward-table initialization in `fit_offline()` and after a valid `load()`.
- Do not set `_reward_table_ready` in `update()`.
- Make `select_action()` require both fitted dynamics and a ready reward table.
- Add a regression test for `update()` before `fit_offline()` followed by `select_action(reward_fn=None)`: it must fail instead of planning with an all-zero reward table.

### Script Defaults And Tunable Knobs

- Change `ROMI_ENSEMBLE_ETA` default from `0.0` to `0.1` in:
  - `claude_build/scripts/slurm_general_mbrl_baselines.sh`
  - `claude_build/scripts/run_general_mbrl_single.sh`
- Expose BAMCTS belief bucketing in scripts:
  - `BAMCTS_TREE_DECIMALS="${BAMCTS_TREE_DECIMALS:-2}"`
  - pass `model.tree_belief_decimals=$BAMCTS_TREE_DECIMALS`
- Add a BAMCTS subtree-reuse test with production smoothing:
  - `prior_count=0.05`
  - depth at least 3
  - assert at least one non-root node is revisited.

### Dataset Reproducibility

Add a generate-once dataset path for every `(seed, action_set, data_n, collection_policy)` tuple:

- Add `claude_build/scripts/generate_dataset.py` or equivalent manifest action.
- Save datasets under a deterministic path such as:
  - `outputs/datasets/{action_tag}/{policy}/n{data_n}/seed{seed}.npz`
- Add `dataset_path` support to `scripts/run_pipeline.py`, mirroring `scripts/train.py`:
  - if `dataset_path` exists, load it;
  - validate action range and reward contract before training;
  - otherwise generate and save it.
- Every method in the same seed/action/data/policy cell must consume the exact same `.npz`.

This is required for fairness. Same seed is not enough if each method writes its own fresh dataset independently.

### Manifest Runner

Add a manifest-based runner instead of hand-launching hundreds of one-off commands.

Required files:

- `claude_build/scripts/experiments/make_experiment_manifest.py`
- `claude_build/scripts/experiments/run_manifest_row.py`
- `claude_build/scripts/slurm/slurm_experiment_manifest.sh`
- `claude_build/scripts/slurm/slurm_experiment_manifest_cpu.sh`
- `claude_build/scripts/analysis/aggregate_experiments.py`

Manifest schema:

```text
row_id  run_id  phase  method  seed  action_set  data_n  collection_policy  dataset_path  config_tag  device  time_limit  wandb_group  output_dir  overrides
```

Rules:

- `run_id` is the output directory name and must be globally unique.
- `config_tag` is a short human-readable hyperparameter slug.
- `overrides` is a single shell-safe Hydra override string, parsed with `shlex.split()` by `run_manifest_row.py`.
- `RUN_NAME=$run_id`, not just `{method}_{action_set}_seed{seed}`.
- Output directories include `run_id` so tuning rows cannot overwrite each other.
- W&B:
  - `WANDB_RUN_GROUP=$wandb_group`
  - `WANDB_NAME=$run_id`

Runner requirements:

- `--dry-run`: print every resolved command without executing.
- `--row-index N`: execute exactly one manifest row.
- `--start-at 1`: useful after row 0 succeeds.
- Before array submission, run row 0 synchronously. Submit the SLURM array only if row 0 exits successfully.
- Phase launchers wait for their worker arrays with `sbatch --wait`, so launcher completion means phase completion.
- The full-ladder wrapper `scripts/slurm/run_all_phases.sh` chains launchers with `--dependency=afterok`.
- Method worker rows use generic GPU requests:
  - `#SBATCH --gres=gpu:1`
  - `#SBATCH --array=0-N%4`
- Dataset-only worker rows use a CPU manifest worker.
- Runtime limit is 24 hours per launcher and worker row, matching the approved 1-day job budget.
- Do not pin A100 or L40S in the first intensive run. Queue flexibility is more important while debugging.

### Evaluation Metrics

Keep the existing evaluator metrics and add:

- `min_abundance`: minimum visited discrete state bin in the episode.
- `catastrophic_low_abundance`: `1` if `min_abundance <= 5`, otherwise `0`.

This defines the ecological safety tie-breaker operationally. With 100 bins of width 10, bin `<=5` means the trajectory touched abundance below roughly 60 individuals.

### Aggregation And Statistical Tests

`aggregate_experiments.py` should read all CSV/JSON outputs for a phase and emit:

- `outputs/summaries/{phase}_summary.csv`
- `outputs/summaries/{phase}_summary.json`
- optional plots under `outputs/summaries/{phase}_plots/`

Primary statistical unit: seed-level mean cumulative reward. Do not treat individual evaluation episodes as independent trained agents.

For final method comparisons:

- Report mean, standard deviation, median, and 95% bootstrap CI over seed-level means.
- Use paired seed-level permutation/sign tests for pairwise comparisons against the best method and against each default baseline.
- Apply Holm correction within each family of pairwise comparisons.
- Episode-level summaries may be reported as descriptive diagnostics only.

## Seed Pools

Use disjoint seed pools:

- Tuning pool:
  - `101`
  - `102`
  - `103`
- Final test pool:
  - `7001`
  - `7002`
  - `7003`
  - `7004`
  - `7005`
- Sensitivity pool:
  - `7001`
  - `7002`
  - `7003`

The final test pool is never used to select hyperparameters. Sensitivity reuses test-pool seeds only after model/config selection is frozen.

## Phase 0: Smoke Test

Purpose: catch runtime/config errors before expensive sweeps.

Settings:

- `ACTION_SET=5`
- `DATA_N=10000`
- `collection_policy=random`
- `SEED=101`
- `EVAL_EPISODES=5`
- `EVAL_HORIZON=50`
- `COMBO_STEPS=1000`
- `CQL_STEPS=1000`
- `IQL_STEPS=1000`
- `ENSEMBLE_EPOCHS=2`
- `PLAN_HORIZON=3`
- `PLAN_ROLLOUTS=3`
- `BAMCTS_SIMS=16`
- `REFPLAN_SEQUENCES=16`
- `REFPLAN_LATENTS=2`

Methods:

```text
plus mopo morel combo cql iql bamcts romi refplan
```

Pass criteria:

- All methods exit successfully.
- All methods write CSV and JSON outputs.
- Dataset path is shared across methods for the smoke seed.
- The smoke manifest intentionally includes a separate 10k dataset row at `outputs/datasets/5a/random/n10000/seed101.npz`; Phase 0.5 later creates the 75k datasets for full experiments.
- W&B run names and output directories are unique.
- The hmMDP exclusion is visible and not silent.

Estimated compute:

- 1 smoke dataset row plus 10 method rows.
- Usually less than one 24-hour SLURM wave with `%4`, but stop immediately if any smoke row fails.

## Phase 0.5: Dataset Pre-Generation

Purpose: remove dataset-sampling confounds before method comparison.

Generate and validate datasets for all planned cells:

- Phase 1 and Phase 2/3:
  - action set `5`
  - policy `random`
  - `DATA_N=75000`
  - seeds `101 102 103`
- Phase 4:
  - action set `5`
  - policy `random`
  - `DATA_N=75000`
  - seeds `7001 7002 7003 7004 7005`
- Phase 5 one-axis sensitivities:
  - data sizes `25000`, `150000` at action set `5`, policy `random`, seeds `7001 7002 7003`
  - action set `10`, `DATA_N=75000`, policy `random`, seeds `7001 7002 7003`
  - policy `suboptimal_heuristic`, action set `5`, `DATA_N=75000`, seeds `7001 7002 7003`

Validation:

- action ids are in range for the env;
- all expected actions appear for random collection with enough samples;
- stored rewards match `env.reward(state, action, next_state)`;
- metadata records action set, data size, policy, seed, and reward-contract version.

Estimated compute:

- CPU/light GPU only.
- Run before submitting Phase 1.

## Phase 1: Default / Original-Inspired Matched Comparison

Purpose: establish a fair default baseline before tuning.

Settings:

- `ACTION_SET=5`
- `DATA_N=75000`
- `collection_policy=random`
- seeds `101 102 103`
- `EVAL_EPISODES=200`
- `EVAL_HORIZON=50`
- `eval.discount=0.95`
- `terminate_on_extinction=false`

Default configs:

| Method | Config |
| --- | --- |
| PLUS | `num_candidate_models=21`, `discount=0.95`, `transition_n_grid=2000`, `likelihood_floor=1e-12` |
| MOPO | `ensemble_size=5`, `frame_stack_len=2`, `bootstrap_ratio=0.8`, `num_epochs=50`, `planner.pessimism_lambda=1.0`, `planner.horizon=5`, `planner.num_rollouts=50` |
| MOReL | same ensemble as MOPO, `uncertainty_threshold=1.0`, `halt_penalty=50`, `planner.horizon=5`, `planner.num_rollouts=50` |
| COMBO | `conservative_beta=1.0`, `rollout_length=1`, `real_ratio=0.5`, `n_steps=50000`, dynamics epochs `50` |
| CQL | `alpha=1.0`, `gamma=0.99`, `learning_rate=6.25e-5`, `n_steps=50000` |
| IQL | `expectile=0.7`, `beta=3.0`, `gamma=0.95`, `learning_rate=3e-4`, `n_steps=50000` |
| BAMCTS | `ensemble_size=15`, `mcts_simulations=128`, `mcts_depth=5`, `exploration_c=1.25`, `pessimism_lambda=0.10`, `tree_belief_decimals=2` |
| ROMI | `ensemble_size=15`, `xi_bins=2`, `robust_eta=0.75`, `uncertainty_count_scale=10`, `adaptive_weight_power=1.0`, `ensemble_eta=0.1` |
| RefPlan | `ensemble_size=15`, `planning_horizon=4`, `num_sequences=128`, `latent_samples=8`, `kappa=5.0`, `uncertainty_penalty=0.10`, `terminal_value_weight=1.0` |
| MOOR | `dynamics_model=ricker`, `learn_params=[r,K]`, `n_restarts=30`, `lbfgs_max_iter=20`, `discount=0.95`, `transition_n_grid=2000`, `likelihood_floor=1e-12` (misspecification ablation: `dynamics_model=schaefer`) |

Estimated compute:

- 30 GPU rows.
- Good first full-scale checkpoint before hyperparameter search.

## Phase 2: Coarse Hyperparameter Sweep

Purpose: produce a shortlist, not choose final configs.

Settings:

- `ACTION_SET=5`
- `DATA_N=75000`
- `collection_policy=random`
- seed `101` only
- `EVAL_EPISODES=50`
- `EVAL_HORIZON=50`

Selection after Phase 2:

- Keep top 4 configs per method by mean cumulative reward.
- Reject a config from the shortlist if it has clearly pathological ecology diagnostics:
  - very low mean abundance relative to competitors;
  - high catastrophic-low-abundance frequency;
  - unstable runtime or repeated failures.
- Do not select final hyperparameters from this single-seed phase.

The grids below are candidate lists, not Cartesian products.

### PLUS Candidates

1. `M=21, discount=0.95, grid=2000, floor=1e-12`
2. `M=11, discount=0.95, grid=2000, floor=1e-12`
3. `M=31, discount=0.95, grid=2000, floor=1e-12`
4. `M=41, discount=0.95, grid=2000, floor=1e-12`
5. `M=21, discount=0.90, grid=2000, floor=1e-12`
6. `M=21, discount=0.99, grid=2000, floor=1e-12`
7. `M=21, discount=0.95, grid=2000, floor=1e-9`
8. `M=21, discount=0.95, grid=500, floor=1e-12`
9. `M=21, discount=0.95, grid=1000, floor=1e-12`

### MOPO Candidates

1. `lambda=1.0, horizon=5, rollouts=50`
2. `lambda=0.0, horizon=5, rollouts=50`
3. `lambda=0.5, horizon=5, rollouts=50`
4. `lambda=2.0, horizon=5, rollouts=50`
5. `lambda=5.0, horizon=5, rollouts=50`
6. `lambda=1.0, horizon=3, rollouts=50`
7. `lambda=1.0, horizon=10, rollouts=50`
8. `lambda=1.0, horizon=5, rollouts=100`
9. `lambda=1.0, horizon=10, rollouts=100`

### MOReL Candidates

1. `threshold=1.0, halt=50, horizon=5, rollouts=50`
2. `threshold=0.5, halt=50, horizon=5, rollouts=50`
3. `threshold=2.0, halt=50, horizon=5, rollouts=50`
4. `threshold=1.0, halt=10, horizon=5, rollouts=50`
5. `threshold=1.0, halt=100, horizon=5, rollouts=50`
6. `threshold=1.0, halt=50, horizon=3, rollouts=50`
7. `threshold=1.0, halt=50, horizon=10, rollouts=50`
8. `threshold=1.0, halt=50, horizon=5, rollouts=100`
9. `threshold=2.0, halt=100, horizon=5, rollouts=50`

### COMBO Candidates

1. `beta=1.0, rollout_length=1, real_ratio=0.5`
2. `beta=0.0, rollout_length=1, real_ratio=0.5`
3. `beta=0.5, rollout_length=1, real_ratio=0.5`
4. `beta=2.0, rollout_length=1, real_ratio=0.5`
5. `beta=5.0, rollout_length=1, real_ratio=0.5`
6. `beta=1.0, rollout_length=3, real_ratio=0.5`
7. `beta=1.0, rollout_length=5, real_ratio=0.5`
8. `beta=1.0, rollout_length=1, real_ratio=0.7`
9. `beta=2.0, rollout_length=3, real_ratio=0.7`

### CQL Candidates

1. `alpha=1.0, gamma=0.99, lr=6.25e-5`
2. `alpha=0.1, gamma=0.95, lr=6.25e-5`
3. `alpha=0.5, gamma=0.95, lr=6.25e-5`
4. `alpha=1.0, gamma=0.95, lr=6.25e-5`
5. `alpha=2.0, gamma=0.95, lr=6.25e-5`
6. `alpha=4.0, gamma=0.95, lr=6.25e-5`
7. `alpha=1.0, gamma=0.95, lr=3e-5`
8. `alpha=1.0, gamma=0.95, lr=1e-4`
9. `alpha=1.0, gamma=0.99, lr=1e-4`

### IQL Candidates

1. `expectile=0.7, beta=3, gamma=0.95, lr=3e-4`
2. `expectile=0.5, beta=3, gamma=0.95, lr=3e-4`
3. `expectile=0.9, beta=3, gamma=0.95, lr=3e-4`
4. `expectile=0.7, beta=1, gamma=0.95, lr=3e-4`
5. `expectile=0.7, beta=10, gamma=0.95, lr=3e-4`
6. `expectile=0.7, beta=3, gamma=0.99, lr=3e-4`
7. `expectile=0.7, beta=3, gamma=0.95, lr=1e-4`
8. `expectile=0.7, beta=3, gamma=0.95, lr=6e-4`
9. `expectile=0.9, beta=10, gamma=0.95, lr=3e-4`

### BAMCTS Candidates

1. `M=15, sims=128, depth=5, c=1.25, lambda=0.10, decimals=2`
2. `M=15, sims=64, depth=5, c=1.25, lambda=0.10, decimals=2`
3. `M=15, sims=256, depth=5, c=1.25, lambda=0.10, decimals=2`
4. `M=15, sims=128, depth=3, c=1.25, lambda=0.10, decimals=2`
5. `M=15, sims=128, depth=8, c=1.25, lambda=0.10, decimals=2`
6. `M=15, sims=128, depth=5, c=1.25, lambda=0.00, decimals=2`
7. `M=15, sims=128, depth=5, c=1.25, lambda=0.25, decimals=2`
8. `M=15, sims=128, depth=5, c=1.25, lambda=0.10, decimals=1`
9. `M=15, sims=128, depth=5, c=0.50, lambda=0.10, decimals=2`
10. `M=15, sims=128, depth=5, c=2.50, lambda=0.10, decimals=2`
11. `M=5, sims=128, depth=5, c=1.25, lambda=0.10, decimals=2`
12. `M=31, sims=128, depth=5, c=1.25, lambda=0.10, decimals=1`

### ROMI Candidates

1. `M=15, xi=2, eta=0.75, count_scale=10, power=1.0, ensemble_eta=0.1`
2. `M=15, xi=1, eta=0.75, count_scale=10, power=1.0, ensemble_eta=0.1`
3. `M=15, xi=5, eta=0.75, count_scale=10, power=1.0, ensemble_eta=0.1`
4. `M=15, xi=2, eta=0.50, count_scale=10, power=1.0, ensemble_eta=0.1`
5. `M=15, xi=2, eta=1.00, count_scale=10, power=1.0, ensemble_eta=0.1`
6. `M=15, xi=2, eta=0.75, count_scale=3, power=1.0, ensemble_eta=0.1`
7. `M=15, xi=2, eta=0.75, count_scale=30, power=1.0, ensemble_eta=0.1`
8. `M=15, xi=2, eta=0.75, count_scale=10, power=1.0, ensemble_eta=0.0`
9. `M=15, xi=2, eta=0.75, count_scale=10, power=1.0, ensemble_eta=0.25`
10. `M=15, xi=2, eta=0.75, count_scale=10, power=0.5, ensemble_eta=0.1`
11. `M=15, xi=2, eta=0.75, count_scale=10, power=2.0, ensemble_eta=0.1`
12. `M=31, xi=2, eta=0.75, count_scale=10, power=1.0, ensemble_eta=0.1`

### RefPlan Candidates

1. `M=15, H=4, N=128, latents=8, kappa=5, penalty=0.1, terminal=1.0`
2. `M=15, H=2, N=128, latents=8, kappa=5, penalty=0.1, terminal=1.0`
3. `M=15, H=8, N=128, latents=8, kappa=5, penalty=0.1, terminal=1.0`
4. `M=15, H=4, N=64, latents=8, kappa=5, penalty=0.1, terminal=1.0`
5. `M=15, H=4, N=256, latents=8, kappa=5, penalty=0.1, terminal=1.0`
6. `M=15, H=4, N=128, latents=4, kappa=5, penalty=0.1, terminal=1.0`
7. `M=15, H=4, N=128, latents=16, kappa=5, penalty=0.1, terminal=1.0`
8. `M=15, H=4, N=128, latents=8, kappa=1, penalty=0.1, terminal=1.0`
9. `M=15, H=4, N=128, latents=8, kappa=10, penalty=0.1, terminal=1.0`
10. `M=15, H=4, N=128, latents=8, kappa=5, penalty=0.0, terminal=1.0`
11. `M=15, H=4, N=128, latents=8, kappa=5, penalty=0.5, terminal=1.0`
12. `M=15, H=4, N=128, latents=8, kappa=5, penalty=0.1, terminal=0.5`
13. `M=15, H=4, N=128, latents=8, kappa=5, penalty=0.1, terminal=2.0`

Estimated compute:

- 90-110 GPU rows depending on final manifest expansion.
- This phase is intentionally noisy and is used only to shortlist top 4 configs per method.

## Phase 3: Multi-Seed Tuning Confirmation

Purpose: select final tuned configs without relying on a single lucky seed.

Settings:

- Use top 4 configs per method from Phase 2.
- seeds `101 102 103`
- `ACTION_SET=5`
- `DATA_N=75000`
- `collection_policy=random`
- `EVAL_EPISODES=200`
- `EVAL_HORIZON=50`

Selection rule:

1. Primary: highest mean seed-level cumulative reward across the three tuning seeds.
2. Tie-breaker 1: lower seed-level standard deviation.
3. Tie-breaker 2: higher mean abundance.
4. Tie-breaker 3: lower mean catastrophic-low-abundance frequency.
5. Manual override is allowed only for failed/unstable runs and must be recorded in the summary JSON.

Keep exactly one tuned config per method for Phase 4.

Estimated compute:

- `9 methods * 4 configs * 3 seeds = 108` GPU rows.
- Stop after aggregation and inspect before Phase 4.

## Phase 4: Final Matched Comparison

Purpose: final fair table.

Settings:

- configs per method:
  - default/original-inspired config;
  - best tuned config selected in Phase 3.
- seeds `7001 7002 7003 7004 7005`
- `ACTION_SET=5`
- `DATA_N=75000`
- `collection_policy=random`
- `EVAL_EPISODES=200`
- `EVAL_HORIZON=50`

Report:

- mean cumulative reward
- standard deviation across seed-level means
- median seed-level mean cumulative reward
- 95% bootstrap CI over seed-level means
- mean abundance
- final abundance
- minimum abundance
- catastrophic-low-abundance frequency
- state coverage
- policy entropy
- uncertainty mean/max
- number of environment steps
- wall-clock runtime
- training time
- evaluation time
- W&B run URL if available

Estimated compute:

- `9 methods * 2 configs * 5 seeds = 90` GPU rows.
- This is the only phase used for final performance claims on the primary benchmark.

## Phase 5: One-Axis Robustness And Sensitivity

Purpose: test whether final conclusions are stable under ecological setting changes.

Use only the frozen Phase 3 tuned config and the default config for each method. Do not retune in Phase 5.

Seeds:

- `7001 7002 7003`

Evaluation:

- `EVAL_EPISODES=200`
- `EVAL_HORIZON=50`

Run one axis at a time, not a full Cartesian product.

Conditions:

1. Data-size sensitivity:
   - `ACTION_SET=5`
   - `collection_policy=random`
   - `DATA_N=25000`
   - `DATA_N=150000`
   - compare against Phase 4 `DATA_N=75000`
2. Action-space sensitivity:
   - `ACTION_SET=10`
   - `DATA_N=75000`
   - `collection_policy=random`
3. Dataset-policy sensitivity:
   - `ACTION_SET=5`
   - `DATA_N=75000`
   - `collection_policy=suboptimal_heuristic`
   - compare against Phase 4 random policy.

Estimated compute:

- Additional rows: `4 conditions * 9 methods * 2 configs * 3 seeds = 216`.
- This is expensive but still bounded because axes are not crossed.

## Fairness Rules

- All methods use the same reward contract and action set in a given row.
- All methods use the same pre-generated offline dataset for a given `(seed, action_set, data_n, collection_policy)`.
- Hyperparameter tuning uses only seeds `101 102 103`.
- Final primary claims use only seeds `7001 7002 7003 7004 7005`.
- Model-free, model-based, tabular planning, and PLUS methods are not forced to equal compute budgets because their computational structures differ substantially.
- Runtime is reported separately so readers can judge performance versus compute cost.
- Planning-heavy decision budgets are logged:
  - MOPO/MOReL: `num_actions * num_rollouts * horizon`
  - BAMCTS: `mcts_simulations * mcts_depth`
  - RefPlan: `num_sequences * latent_samples * planning_horizon`
- Hyperparameter grids are comparable but not identical. The grid size for every method is reported because methods have different numbers of meaningful knobs.

## Stop Gates

Do not launch the next phase until the previous phase passes its gate.

| Gate | Requirement |
| --- | --- |
| After code fixes | targeted tests and py_compile pass |
| After Phase 0 | all methods produce outputs; row 0 dry-run and sync-run workflow works |
| After Phase 0.5 | all needed datasets exist and pass reward/action validation |
| After Phase 1 | default comparison summary generated with no output collisions |
| After Phase 2 | top 4 configs per method selected and recorded |
| After Phase 3 | exactly one tuned config per method frozen |
| After Phase 4 | final primary table and statistical tests generated |
| After Phase 5 | sensitivity table generated without changing selected configs |

## Acceptance Criteria

Code/harness readiness:

- `pytest -q claude_build/tests/test_general_mbrl_baselines.py`
- targeted tests for dataset loading, reward-table readiness, and BAMCTS production-prior tree reuse
- `python -m py_compile` for modified scripts
- manifest dry-run prints all commands
- sync row-0 launch succeeds before array submission

Experiment readiness:

- manifests can be generated for Phases 0 through 5;
- no run names collide;
- every row has a dataset path;
- every dataset path is shared across methods in the same experimental cell;
- W&B group/name identify phase, action set, data size, method, config, and seed;
- method worker rows request generic `#SBATCH --gres=gpu:1`;
- final aggregation emits summary CSV/JSON and seed-level statistical tests.

## Recommended Execution Order

1. Implement the small code/harness fixes.
2. Run tests and Phase 0 smoke.
3. Generate all Phase 1 through Phase 4 datasets.
4. Run Phase 1 defaults.
5. Run Phase 2 coarse sweep.
6. Aggregate Phase 2 and select top 4 per method.
7. Run Phase 3 tuning confirmation.
8. Freeze tuned configs.
9. Run Phase 4 final comparison.
10. Run Phase 5 only after the Phase 4 table is stable.
