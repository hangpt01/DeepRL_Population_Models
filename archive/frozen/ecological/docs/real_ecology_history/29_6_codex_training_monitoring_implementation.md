# Codex Implementation Note: Training/Holdout Monitoring

Date: 2026-07-04

Scope: `discrete_action_cont_obser/real_ecology_cont_obser/`.

## What Changed

Added local-first training diagnostics with optional Weights & Biases logging.

New config block:

```yaml
training:
  enabled: true
  holdout_fraction: 0.2
  plot: true
  wandb: false
```

The block is now explicit in:

- `configs/real_default.yaml`
- `configs/real_experiment.yaml`
- `configs/real_probe.yaml`
- `configs/real_smoke.yaml`

## Saved Artifacts

Each method row now saves the following under the row output directory, for example:

```text
evaluation/<population>/<family>/sigma_<x>/backend_<backend>/reward_<mode>/<method>/<filter>/
```

Files:

- `training_history.csv`
- `training_history.json`
- `training_history.png` when `training.plot: true` and matplotlib is available

Summary fields added to `summary.json`:

- `training_split`
- `training_history_rows`
- `training_history_csv`
- `training_history_json`
- `training_history_plot` when written
- `training_plot_status`
- `training_wandb_status`
- final metric aliases like `training_train_dynamics_log_mse`, `training_holdout_dynamics_log_mse`

## Holdout Split

The split is episode-disjoint, not row-random. The policy fit receives the training split only, and the holdout split is used for validation diagnostics.

No private state is used. Metrics are computed from public observations, rewards, actions, and cached public-belief summaries.

## Method-Specific Curves

Iterative methods now log per-iteration curves:

- Delphic-CQL logs `q_bellman_mse` and `mean_delphic_uncertainty` for train and holdout during CQL fitting.
- OGSRL logs `surrogate_loss`, `objective_mean`, `reward_return`, `safety_cost`, `ood_cost`, and dual variables for train and holdout during actor fitting.

Closed-form/grid/model-fit methods still log final train/holdout diagnostics:

- MOPO / RefPlan / BA-MCTS / OGSRL dynamics: `dynamics_log_mse`, `dynamics_rmse`
- MOOR: Ricker fit prediction metrics
- Delphic final Q Bellman MSE
- PLUS: numeric fit diagnostics such as `candidate_models`

## W&B

W&B is opt-in through config:

```yaml
training:
  wandb: true
  wandb_project: real-ecology-cont-obser
  wandb_entity: null
  wandb_group: null
  wandb_run_name: null
```

If `wandb` is unavailable or disabled, the run still writes local artifacts and records the status in `training_wandb_status`.

## Verification

Config parse:

```text
configs/real_default.yaml True 0.2 False
configs/real_experiment.yaml True 0.2 False
configs/real_probe.yaml True 0.2 False
configs/real_smoke.yaml True 0.2 False
```

Unit tests:

```text
PYTHONPATH=src python -m unittest discover -s tests -v
Ran 25 tests in 3.901s
OK
```

Tiny iterative-method smoke:

```text
delphic 147 True disabled
ogsrl 429 True disabled
```

CLI smoke:

```text
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml --backend numpy
status: ok
training_history_rows: 5
training_plot_status: ok
```

## Audit Notes

This is a protocol-affecting change when `training.enabled=true` and `holdout_fraction>0`: policies fit on the training split rather than the full dataset. That is intentional for a real holdout validation curve. Set `training.holdout_fraction: 0.0` to recover full-dataset fitting while still writing local fit diagnostics.
