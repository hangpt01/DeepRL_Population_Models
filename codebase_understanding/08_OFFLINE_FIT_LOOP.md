# Offline fit loop

This is the repository’s analogue of a training loop. It is not an online
policy-improvement loop: no environment interaction occurs inside the common
`policy.fit(dataset, cache)` call.

## Real control flow

```text
run_method(method,cfg,filter,regenerate):
  activate resolve_backend_for_workload(...)
  dataset = ensure_dataset(...)                   # cache or collect once
  if hides_rk:
      surrogate = load-by-public-hash or fit(seed+20000)
      context = sanitized MethodContext
  factory = make_filter_factory(...)
  cache = load-by-filename or build(seed+30000)   # except faithful
  if faithful:
      faithful_fit owns ordered 80/20 episode split
  else:
      train,holdout = split_train_holdout(seed+50000)
  policy = METHODS[method](context-or-env, seed=seed+40000)
  attach_training_history(...)
  diagnostics = policy.fit(train, train_cache)    # sole common "training" call
  record_final_fit_metrics(...)                   # reporting only
  rows = ContinuousEvaluator(...).run(policy)
  save episodes, summary, beliefs, fit/training artifacts
```

This matches `src/tracks/general/real_ecology_benchmark/pipeline.py`,
`run_method()` and `build_method()`.

For faithful methods, the complete command-to-optimizer chain is
`real_ecology_benchmark.__main__` → `cli.main()` → `cli.cmd_run()` →
`pipeline.run_method()` → `build_method()` →
`MOORFaithfulRickerPBVIPolicy.fit()` or `PLUSFaithfulPBVIPolicy.fit()` →
`faithful_fit.fit_mechanistic_model()` → `torch.optim.LBFGS(...)` →
`optimizer.step(closure)` → `_trajectory_objective()` →
`objective.backward()`. For general methods there may be no gradient optimizer:
the same chain ends in ridge solves, fitted-Q iterations, model construction, or
linear actor updates inside the selected `fit()`.

## Commands

A smoke cell from scratch:

```bash
PYTHONPATH=src/tracks/general python -m real_ecology_benchmark smoke \
  --config configs/tracks/general/real_smoke.yaml
```

One accepted general row (requires the external pinned dataset roots selected by
the runner):

```bash
PYTHONPATH=src/tracks/general python scripts/general/run_real_manifest_row.py \
  experiments/accepted_general/manifests/full_general_sigma01_02_576_rows.csv 0 \
  --config experiments/accepted_general/configs/general_phase2e_full_sigma01_02.yaml \
  --output-root /path/to/general-output --dataset-root /path/to/pinned-data
```

Row 0 is Egyptian vulture/Ricker/sigma .1/RefPlan, as verified in the manifest.
For direct package execution use:

```bash
PYTHONPATH=src/tracks/general python -m real_ecology_benchmark run \
  --config experiments/accepted_general/configs/general_phase2e_full_sigma01_02.yaml \
  --population "Egyptian vulture" --environment ricker --sigma 0.1 \
  --reward-mode safe --expose-rk hidden --method refplan --filter learned
```

The latter reproduces semantics only if its configured dataset paths resolve to
the authoritative files; the runner additionally stamps the resolved row.

## What “one step” means

For closed-form ridge fits, one training step is the whole `fit()`. EVD performs
35 fitted-Q Bellman iterations per bootstrap member
(`methods/ensemble_value_disagreement.py`, `fit_iterations=35`). OGSRL performs
`train_iterations` linear actor updates in `ogsrl.py` (constructor default;
inspect the selected track/summary for the deployed count). Native solvers run
up to 250 VI iterations (`ModelConfig.native_vi_iterations`). PBVI performs
finite-horizon backwards backups per action request over a reachable belief
graph (`planners/pbvi.py:action_values()`), rather than an offline gradient
epoch. Faithful models run eight LBFGS starts, each with at most 100 iterations.

`TrainingHistory` logs generic scalar rows during fit and final train/holdout
prediction or Bellman MSE. Nothing in `pipeline.py` uses these values for model
selection; they are reporting artifacts. Faithful fit objective selects the
lowest finite LBFGS start internally (`faithful_fit.py`).

## Failure and silence modes

- belief caches are loaded without hash/shape/config compatibility validation;
- a surrogate hash mismatch triggers a silent refit, but status is recorded as
  `fitted`;
- `policy.fit()` exceptions are **not** swallowed by `pipeline.py`; they abort
  the row;
- backend fallback is allowed only if `compute.strict=false` and is recorded by
  `Backend.to_dict()`;
- empty `fit_diagnostics` is legal for base/no-op implementations;
- evaluator action exceptions are swallowed into action 0, independently of fit;
- faithful fit-cache misses may recompute unless the accepted runner’s receipt
  gate requires hits.
