# Maintainer reading order

## Mental model

This repository is an **offline, model-based POMDP benchmark**, not a deep-RL
training framework. `collector.collect_dataset()` simulates a fixed public
action/observation/reward dataset and a private truth sidecar from
`ContinuousEcologyEnv`; the behavior policy may use true state
(`src/tracks/general/real_ecology_benchmark/collector.py`, `collect_dataset()`).
Each registered policy is constructed once and receives one call to
`policy.fit(dataset, beliefs)` in `pipeline.build_method()`. Evaluation then
creates fresh simulator episodes in `ContinuousEvaluator.run()`; these online
rollouts score the already fitted policy and do not update it
(`src/tracks/general/real_ecology_benchmark/evaluator.py`). The hidden regime
replaces `EnvironmentConfig` with the sanitized `MethodContext` and uses a
public-observation belief cache (`config.py`, `MethodContext`;
`pipeline.py`, `_hidden_method_context()`). The scientific object is the
unbounded continuous population map in `envs.py`, with action-set-point growth,
cumulative capacity, noisy surveys, and true-next-state rewards. There are two
frozen packages with the same import name; accepted ecological PLUS/MOOR and
accepted general baselines came from different tracks. The headline table is
`results/accepted/MATCHED_P10_144_METHOD_CELLS.csv`: 144 fitted-method cells,
not 144 independent datasets or training seeds.

## Dependency map and recommended order

```text
01 big picture
  → 02 two tracks / layout
  → 03 file+function map
  → 04 environment & ecological model
  → 05 offline dataset pipeline
  → 06 beliefs & information regime
  → 07 methods & planning objectives
  → 08 offline fit loop
  → 09 evaluation protocol
  → 10 metrics & result tables
  → 11 configs & manifests
  → 12 reproducibility, caches, provenance
  → 13 verification & diagnostics
  → 14 scientific validity risks
  → 15 practical code-reading plan
```

For a 3–4 hour pass, read 01, 02, 04–10, 14, then execute 15. Read 03, 11–13
as reference/deeper dives. After 01 you can trace a cell; after 02 you can select
the correct import tree; after 04–06 you know the scientific and information
boundaries; after 07–10 you know what is fitted, planned, and measured; after
11–13 you can reproduce and audit artifacts; after 14 you know what the results
do and do not establish.

| Order | File | What it explains / what you should understand afterward | Pass |
|---:|---|---|---|
| 0 | `00_README_READING_ORDER.md` | mental model, dependency order, vocabulary | essential |
| 1 | `01_BIG_PICTURE_FLOW.md` | one cell from manifest through accepted result | essential |
| 2 | `02_TWO_TRACKS_AND_REPO_LAYOUT.md` | safe imports, complete track diff, ownership | essential |
| 3 | `03_FILE_FUNCTION_ROLE_MAP.md` | module/function call map and modification risk | deep reference |
| 4 | `04_ENVIRONMENT_AND_ECOLOGICAL_MODEL.md` | four maps, controls, actions, observations, rewards | essential |
| 5 | `05_OFFLINE_DATASET_PIPELINE.md` | privileged collection, schemas, splits and dataset caches | essential |
| 6 | `06_BELIEFS_AND_INFORMATION_REGIME.md` | hidden/public/private boundary, filters, surrogate | essential |
| 7 | `07_METHODS_AND_PLANNING_OBJECTIVES.md` | fitted objects, deployed selectors, real objectives | essential |
| 8 | `08_OFFLINE_FIT_LOOP.md` | command-to-fit/optimizer path and seed offsets | essential |
| 9 | `09_EVALUATION_PROTOCOL.md` | fitted object/cache to fresh episodes and metrics | essential |
| 10 | `10_METRICS_LOGGING_AND_RESULT_TABLES.md` | metric definitions, variance scope, result artifacts | essential |
| 11 | `11_CONFIGS_MANIFESTS_HYPERPARAMETERS.md` | three config layers and accepted overrides | deep dive |
| 12 | `12_REPRODUCIBILITY_SEEDS_CACHES_PROVENANCE.md` | seed tree, cache identities, external data and hashes | deep dive |
| 13 | `13_VERIFICATION_AND_DIAGNOSTIC_LAYER.md` | Make targets, M1–M15, replay and follow-ups | deep dive |
| 14 | `14_SCIENTIFIC_VALIDITY_RISKS.md` | repo-specific threats to the scientific conclusion | essential |
| 15 | `15_RECOMMENDED_CODE_READING_PLAN.md` | timed executable audit plan | essential/action |

## Code-defined glossary

- **cell**: one population × true dynamics family × survey sigma × reward mode ×
  method × information regime. These columns are materialized by
  `manifest.make_manifest()` and the accepted manifest runners.
- **track**: one frozen `src/tracks/{ecological,general}/real_ecology_benchmark`
  import tree. Both expose the same package name.
- **manifest row**: a CSV experiment specification consumed by
  `scripts/general/run_real_manifest_row.py:apply_row_config()` (or its
  ecological counterpart).
- **accepted**: an artifact copied under `results/accepted` and hash-described
  by `MATCHED_P10_144_RECEIPT.json`; it does not mean statistically replicated.
- **parity gate**: comparison of seven replayed summary fields at tolerance
  `1e-9`, enforced by `scripts/verify_accepted_cell.py` and
  `scripts/diagnostics/replay/run_diagnostic_replay.py:enforce_acceptance()`.
- **receipt**: JSON recording inputs, hashes, coverage, gates, and/or cache reuse;
  examples are under `results/followups` and `provenance/diagnostic_replay`.
- **filter**: an object implementing reset/update of `BeliefState`
  (`types.py`, `BeliefFilter`; `pipeline.py`, `make_filter_factory()`).
- **belief cache**: transition-aligned fitted filter features and mean states
  (`beliefs.py`, `BeliefCache`, `PublicBeliefCache`).
- **surrogate**: hidden-regime public-data models of logged reward and genuine
  termination (`public_surrogate.py`, `PublicRewardRiskSurrogate`).
- **`expose_rk`**: `hidden` with a sanitized context, or `full` with demographic
  config/control fields (`config.py`, `hides_rk()`).
- **hidden/full**: information regimes enforced in `pipeline.make_filter_factory()`
  and `methods/base.py:BasePolicy.__init__()`.
- **operational/true return**: discounted sums of `StepResult.reward` and private
  `evaluator_info["reward_true"]`; identical for set-point real cells because
  `envs.py:step()` assigns `reward_true = reward`.
- **headroom**: diagnostic difference between a ceiling/reference and a method,
  not a core evaluator field; see `src/diagnostics/replay_analysis/metrics.py`.
- **reference/constant-action control**: non-learning comparison policies in
  `scripts/diagnostics/replay/{a0_baseline,constant_action_sweep}.py`, not entries
  in `methods.METHODS`.
- **faithful/adapted/native/inspired**: implementation labels, not equivalence
  proofs. “Faithful/adapted” uses mechanistic fitting and PBVI
  (`faithful_fit.py`, `planners/pbvi.py`); “native” uses tabular ecology solvers
  (`native_fit.py`, `native_solver.py`); “inspired/motivated” denotes repository
  adaptations named in `methods/__init__.py:METHOD_LABELS`.
