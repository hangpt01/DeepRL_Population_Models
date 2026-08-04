# Configs, manifests, and hyperparameters

## Three layers and precedence

1. `config.load_config()` parses YAML into nested dataclasses. A compact real
   environment is expanded through `real_environment()`.
2. `cli._config()` applies data-mode rebuild, then population/family rebuild,
   then sigma, reward, expose regime, backend, and transitions overrides.
   `real_environment_like()` refreshes population-derived K/N0/r/safety while
   carrying explicit non-source settings.
3. Manifest runners load a YAML then `apply_row_config()` overlays the selected
   row, assigns cell-specific dataset/output paths, and stamps the resolved row
   (`scripts/general/run_real_manifest_row.py`).

The manifest is authoritative for an accepted run. For example the general YAML
has sigma 0.4, while every accepted manifest row supplies 0.1 or 0.2. Trusting
the YAML alone reconstructs the wrong cell.

The general accepted manifest has 576 rows and columns for cell identity,
method/filter, data/evaluation budgets and method-specific planner budgets.
The ecological PLUS/MOOR manifests have 24 rows each and additionally freeze
candidate construction, fit-cache receipts/hashes, PBVI discretization,
blinding, and matched-general indices.

## Load-bearing values

| field | default / accepted | defined → used | risk |
|---|---|---|---|
| `seed` | 116 / 116 | `BenchmarkConfig` → collection and all offsets | changes dataset/fit |
| dataset transitions/length | 2000/25; accepted 4000/25 | `DatasetConfig` → `collect_dataset` | coverage |
| filter particles/proposal/ESS | 256/learned/.5; faithful outer 64/internal | `FilterConfig` → beliefs | estimator cost/quality |
| dynamics ensemble/ridge | 5/1e-3 | `ModelConfig` → public dynamics fits for RefPlan/MOPO/BA-MCTS | does not control EVD’s hard-coded 20-member Q ensemble |
| native bins/VI | 51/250 | `ModelConfig` → native solvers | discretization |
| planner H/sequences/particles/gamma/pessimism | 5/96/32/.95/.5 | `PlannerConfig` → MPC/public planners | objective/search |
| BA-MCTS depth/sims | defaults 5/128; accepted 8/256 | manifest → BA-MCTS | unequal compute |
| OGSRL horizon/rollouts/quantile | 25/256/.20 | general config → OGSRL | safety proxy |
| eval seeds/episodes/H/gamma | five seeds×4/50/.95 | `EvaluationConfig` → evaluator | headline sample |
| training/holdout | true/.2 | `TrainingConfig` → split/diagnostics | fit data |
| P/reward/safety | 10/safe/occupancy accepted | env/reward | headline scale |
| expose/sigma/process noise | hidden/.1-.2/0 | env/filter | information/randomness |
| backend/device/strict | numpy/0/true | backend resolver | numerics/failure |
| faithful fit | 8 starts,100 iterations,16 paths,80% history | `FaithfulFitConfig` → LBFGS | adapted fits |
| faithful planner | b41/c9/o41, 256 samples,32 points,7 branches,H5 | `FaithfulPlannerConfig` → PBVI | approximation |

Load-bearing accepted values include every cell identity, dataset hash/path,
reward/safety, method/filter, model/planner budget, evaluation seeds, and
faithful cache key. Inert for accepted general rows are native solver fields and
faithful fields; synthetic controls and W&B are also inert. `training.plot`
changes artifacts, not scores. A YAML sigma shadowed by every row is inert.

## Exact accepted row reproduction

General row 0:

```bash
PYTHONPATH=src/tracks/general python scripts/general/run_real_manifest_row.py \
  experiments/accepted_general/manifests/full_general_sigma01_02_576_rows.csv 0 \
  --config experiments/accepted_general/configs/general_phase2e_full_sigma01_02.yaml \
  --output-root "$DEEPRL_GENERAL_OUTPUT_ROOT" \
  --dataset-root "$DEEPRL_GENERAL_DATA_ROOT"
```

This is RefPlan, hidden, Egyptian vulture, Ricker, sigma .1, safe P=10,
collection seed 116, 4,000 rows, learned filter, evaluation seeds recorded in
the row. Exact reproduction additionally requires the pinned public/private
dataset paths and environment expected by the runner.

An accepted ecological PLUS row must use `PYTHONPATH=src/tracks/ecological`,
`experiments/accepted_p10/configs/plus_ricker_only_p10.yaml`,
`plus_p10_plan_24.csv`, and a cache-hit-required fit receipt. The ecological
runner/preparation scripts own that cache-reuse contract.

Historical ecological results with P=5 are not comparable: reward subtracts P
for every occupancy penalty step. The accepted receipt explicitly records
`historical_p5_mixed_into_matched_tables=false`; accepted rows carry
`collapse_penalty=10.0`.
