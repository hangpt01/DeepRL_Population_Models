# One benchmark cell, end to end

## Actual lifecycle

```mermaid
flowchart TD
  A[manifest row + base YAML] --> B[BenchmarkConfig / real_environment_like]
  B --> C[ContinuousEcologyEnv]
  C --> D[collect_dataset: privileged scripted policy]
  D --> E[public .npz]
  D --> F[private evaluator-only .npz]
  E --> G[hidden: fit/load public reward-risk surrogate]
  G --> H[MethodContext + filter factory]
  E --> H
  H --> I[belief cache]
  E --> J[episode-disjoint train/holdout split]
  I --> J
  J --> K[one policy.fit(dataset, beliefs)]
  K --> L[ContinuousEvaluator.run: 20 fresh online episodes]
  F -. diagnostics only .-> M[summary extras]
  L --> N[episodes.csv + summary.json]
  N --> O[accepted 144-cell CSV + receipt]
```

The authors’ implemented purpose is to compare belief-space planners and
general offline methods under hidden demographic parameters/model structure.
Evidence is the real-grid construction in
`src/tracks/general/real_ecology_benchmark/manifest.py:make_manifest()`, the
hidden `MethodContext` boundary in `config.py`, and the method registry in
`methods/__init__.py`. The setting is offline fitted model/Q/planner
construction followed by online simulator evaluation: `pipeline.build_method()`
calls `policy.fit()` once, then `pipeline.run_method()` calls
`ContinuousEvaluator.run()`. Repository searches find no replay-buffer, target
network, actor/critic-network, or policy-gradient framework; the only PyTorch
optimizer is `torch.optim.LBFGS` in `faithful_fit.py:fit_mechanistic_model()`.

## Entrypoints and object order

`python -m real_ecology_benchmark run ...` enters
`cli.main()` → `cli.cmd_run()` → `pipeline.run_method()` (`__main__.py` calls
`cli.main`). Evaluation is deliberately not a separate script: `run_method()`
fits and then constructs `ContinuousEvaluator` in the same cell process.

Object order in `run_method()` is:

1. workload-specific `Backend` (`backend.resolve_backend_for_workload()`);
2. cached/new `TrajectoryDataset` plus private sidecar
   (`pipeline.ensure_dataset()`, `collector.collect_dataset()`);
3. hidden `PublicRewardRiskSurrogate` and `MethodContext`;
4. filter factory (`make_filter_factory()`);
5. `BeliefCache`/`PublicBeliefCache`;
6. train and diagnostic holdout datasets (`training_monitor.split_train_holdout()`);
7. policy from `METHODS[method]`, then its sole `fit()`;
8. `ContinuousEvaluator`, fresh environments, episode rows;
9. summary, offline beliefs, training history, and method fit artifacts.

Outputs are under
`evaluation.output_dir/data_<mode>/backend_<name>/regime_<expose_rk>/reward_<mode>/<method>/<filter>`;
this exact namespace is built by `pipeline._reward_mode_output_root()` and
`run_method()` lines 506–508. The accepted manifest runners additionally make
the outer directory population/family/sigma-specific.

## Runner scripts

`scripts/general/run_real_manifest_row.py` is the load-bearing general row
runner: `read_manifest_row()` selects a row, `apply_row_config()` overlays it,
and `main()` calls the package pipeline. The ecological counterpart under
`scripts/ecological` produced the adapted accepted rows. The Slurm wrappers
submit those row runners; `make_*manifest.py` files generate routine grids.
`scripts/diagnostics/replay/run_diagnostic_replay.py` and
`run_tier_b_replay.py` reconstruct accepted rows and enforce parity;
follow-up scripts test hypotheses rather than produce the headline table.

The final accepted CSV is assembled by
`scripts/build_three_species_p10_merged_report.py` from frozen general and
ecological artifacts, not by `manifest.aggregate_summaries()` alone.

