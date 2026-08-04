# General repository handoff for a new chat

This file is a compact orientation, not a substitute for source inspection. For
the detailed, evidence-backed map, begin with
`codebase_understanding/00_README_READING_ORDER.md`. Treat executable source,
configs, manifests, receipts, and artifacts as authoritative when they disagree
with prose.

## 1. Repository identity

This repository preserves a reproducible ecological offline-RL benchmark for
population management under partial observability and uncertain demographic
dynamics. The scientific environment is a continuous population process with
four supported demographic families, discrete management actions, noisy
lognormal surveys, action-dependent controls, and reward based on the true next
state (`README.md`; `src/tracks/general/real_ecology_benchmark/envs.py`,
`ContinuousEcologyEnv` and `transition_value()`;
`observation.py`, `LogNormalObservationModel`; `reward.py`,
`ContinuousReward`).

The main setting is offline, model-based RL/planning in a POMDP: a fixed
simulator-generated trajectory dataset is collected first, a policy/model or
planner is fitted once, and the fitted policy is evaluated in fresh online
simulator episodes without policy-parameter updates
(`collector.py:collect_dataset()`; `pipeline.py:build_method()` and
`run_method()`; `evaluator.py:ContinuousEvaluator.run()`). It is not a generic
deep-RL actor/critic framework. Some methods use fitted linear models, ridge
systems, fitted-Q iterations, MCTS, MPC, value iteration, or PBVI. The faithful
mechanistic fit is the gradient-optimized exception and uses PyTorch LBFGS
(`faithful_fit.py:fit_mechanistic_model()`).

The implementation language is Python. Core dependencies are NumPy and PyYAML;
paper-faithful mechanistic fitting optionally uses PyTorch, analysis uses pandas
and Matplotlib, and CuPy can be selected as a numerical backend
(`pyproject.toml`; `requirements.txt`; `backend.py:resolve_backend_for_workload()`;
`training_monitor.py`). The repository does not wrap the environment in Gym or
use a standard external RL dataset library: environment and trajectory schemas
are repository-native (`envs.py:ContinuousEcologyEnv`;
`dataset.py:TrajectoryDataset` and `PrivateTrajectoryData`).

Configuration is nested YAML parsed into dataclasses, then optionally overridden
by CLI arguments and finally by manifest rows (`config.py:load_config()`;
`cli.py:_config()`; `scripts/general/run_real_manifest_row.py:apply_row_config()`).
Runs write NPZ caches/data, CSV episode and training logs, JSON summaries and
receipts, and method-specific fitted artifacts. W&B support exists but is off in
accepted configs; there is no TensorBoard integration
(`pipeline.py:run_method()`; `training_monitor.py:save_training_artifacts()` and
`_log_wandb()`; `faithful_artifacts.py`). There is no generic neural-network
checkpoint loader: fitted state is normally in memory or reconstructed from
dataset, belief, surrogate, or faithful-fit caches
(`pipeline.py`; `faithful_artifacts.py`).

There are two frozen packages with the same import name:
`src/tracks/ecological/real_ecology_benchmark` and
`src/tracks/general/real_ecology_benchmark`. Exactly one track must be placed on
`PYTHONPATH`. Accepted PLUS/MOOR and accepted general-method cells came from
different tracks (`README.md`; `Makefile`, targets `test-ecological` and
`test-general`; `codebase_understanding/02_TWO_TRACKS_AND_REPO_LAYOUT.md`).

## 2. Current understanding status

### Big-picture experiment flow

The traced cell flow is:

```text
base YAML + manifest row
  -> BenchmarkConfig
  -> load or collect public/private trajectory data
  -> hidden public reward/risk surrogate when applicable
  -> filter factory and transition-aligned belief cache
  -> episode-disjoint training/holdout data
  -> construct policy and call policy.fit(...) once
  -> evaluate fitted policy in fresh simulator episodes
  -> episodes.csv + summary.json + fit/training artifacts
  -> accepted tables/receipts where applicable
```

The orchestration is in `pipeline.py:run_method()`, with dataset creation in
`pipeline.ensure_dataset()` and `collector.collect_dataset()`, method creation in
`pipeline.build_method()`, and scoring in
`evaluator.py:ContinuousEvaluator.run()` and `summarize()`.

### Main training entry point

Direct execution is `python -m real_ecology_benchmark run ...`, which follows
`__main__.py` -> `cli.main()` -> `cli.cmd_run()` -> `pipeline.run_method()` ->
`pipeline.build_method()` -> the selected policy class's `fit()` method. The
accepted general batch path begins at
`scripts/general/run_real_manifest_row.py:main()`, which selects and overlays a
manifest row before calling the same package pipeline. The ecological track has
the corresponding runner under `scripts/ecological/`.

There is no single optimizer step shared by all methods. For faithful MOOR/PLUS,
the optimizer trace is `MOORFaithfulRickerPBVIPolicy.fit()` or
`PLUSFaithfulPBVIPolicy.fit()` ->
`faithful_fit.fit_mechanistic_model()` -> `torch.optim.LBFGS.step(closure)` ->
`_trajectory_objective()` -> `backward()`. General methods instead terminate in
method-specific ridge solves, fitted-Q iterations, model construction, planning,
or linear actor updates (`methods/ensemble_value_disagreement.py`,
`methods/ogsrl.py`, `dynamics.py`, and `public_models.py`). See
`codebase_understanding/08_OFFLINE_FIT_LOOP.md` for the complete map.

### Main evaluation entry point

Evaluation is part of `pipeline.run_method()`, not a separate generic CLI.
`ContinuousEvaluator.run(policy)` resets a fresh environment, filter, and policy
RNG for every evaluation episode; calls `policy.act()`; steps
`ContinuousEcologyEnv`; passes only a `PublicTransition` back to the policy; and
records public returns plus evaluator-only private-truth diagnostics. It does not
call `fit()` or update policy parameters (`evaluator.py:ContinuousEvaluator.run()`;
`types.py:PublicTransition`). `ContinuousEvaluator.summarize()` computes means
and sample standard deviations, and `save()` writes `episodes.csv` and
`summary.json`. See `codebase_understanding/09_EVALUATION_PROTOCOL.md`.

### Data and dataset flow

`collector.MixedDangerZonePolicy` generates fixed simulator trajectories; the
collector is privileged because `collect_dataset()` passes the true environment
state to its action rule. `dataset.py` separates `TrajectoryDataset` public
fields from `PrivateTrajectoryData` truth fields. Public data include noisy
observations, actions, truth-derived logged rewards, next observations,
termination/truncation markers, episode/timestep identifiers, costs, opaque
population IDs, and the action-cost table. Full-mode data add public demographic
control fields; hidden mode withholds them
(`dataset.py:PUBLIC_FIELDS`, `PUBLIC_SANITIZED_FIELDS`,
`PUBLIC_CONTROL_FIELDS`, and `PRIVATE_FIELDS`; `collector.py:collect_dataset()`).

Generic training/holdout separation is episode-disjoint
(`training_monitor.py:split_train_holdout()`); faithful methods own a separate
ordered episode split in `faithful_fit.py`. Hidden belief features are built from
public observation history by `beliefs.cache_public_beliefs()`; full-mode caches
can include model/config-derived belief features through
`beliefs.cache_dataset_beliefs()`. The exact feature representation is
`types.py:BeliefState.public_features()` or `features()`.

### Model, world-model, policy, and critic components

The selected method is looked up through `methods/__init__.py:METHODS`. The
accepted general methods are RefPlan, OGSRL, BA-MCTS, and ensemble value
disagreement (EVD); accepted ecological methods are faithful/adapted MOOR and a
Ricker-only PLUS variant (`results/accepted/MATCHED_P10_144_METHOD_CELLS.csv`;
the two track-specific `methods/__init__.py` registries).

- Public/full learned dynamics are implemented in `public_models.py` and
  `dynamics.py`; behavior-prior fitting is in `behavior_model.py`.
- General planning/action selection is implemented by
  `methods/refplan.py:RefPlanPolicy`, `methods/bamcts.py:BAMCTSPolicy`,
  `methods/ogsrl.py:OGSRLPolicy`, and
  `methods/ensemble_value_disagreement.py:EnsembleValueDisagreementPolicy`.
- Legacy/native model and value components are in `native_fit.py`,
  `native_solver.py`, and `discretize.py`.
- Faithful mechanistic candidates and their fitted parameters are in
  `faithful_ecology.py`, `faithful_fit.py`, and `faithful_pomdp.py`; PBVI is in
  `planners/pbvi.py:PointBasedPlanner`.
- There is no generic neural actor/critic pair. EVD's fitted Q ensemble is the
  closest critic-like component; OGSRL fits its own constrained linear actor
  (`methods/ensemble_value_disagreement.py`; `methods/ogsrl.py`).

Method objectives are not uniform. Five accepted methods plan using
`PublicRewardRiskSurrogate.predict()`, while EVD fits logged rewards directly;
all are scored using the exact environment reward
(`public_surrogate.py`; `public_models.py`; `methods/bamcts.py`;
`methods/ogsrl.py`; `faithful_pomdp.py`; EVD's `_fit_member()`;
`envs.py:step()`).

### Metrics and logging flow

`ContinuousEvaluator.run()` produces one row per evaluation episode, including
discounted operational/true return, collapse entry, unsafe/MVP occupancy,
persistence, action cost, population extrema, filter diagnostics, action
diagnostics, and fallback counts. `summarize()` creates mean and sample-SD
fields. `pipeline.run_method()` adds timings, memory, fit diagnostics, cache
status, surrogate diagnostics, and training artifacts. Definitions and known
interpretation traps are catalogued in
`codebase_understanding/10_METRICS_LOGGING_AND_RESULT_TABLES.md`.

The headline copied artifact is
`results/accepted/MATCHED_P10_144_METHOD_CELLS.csv`, assembled by
`scripts/build_three_species_p10_merged_report.py`. Its within-cell SD covers 20
evaluation episodes for one dataset and one fitted policy, not independent
dataset or fit replications. Acceptance/provenance details are in
`results/accepted/MATCHED_P10_144_RECEIPT.json` and `provenance/`.

### Config and hyperparameter flow

Precedence is YAML dataclasses -> CLI overrides -> manifest-row overlays
(`config.py`, `cli.py`, and the applicable row runner). For accepted runs, the
manifest row is load-bearing and may override misleading YAML defaults. Common
configuration groups cover data, environment, filter, model, planner,
evaluation, training, and compute settings (`config.py:BenchmarkConfig` and its
nested dataclasses). Some method defaults are not routed through these configs:
EVD hard-codes its 20-member ensemble and fitted-Q settings, and OGSRL hard-codes
several safety/OOD/training values because `pipeline.build_method()` passes only
the seed for those constructors
(`methods/ensemble_value_disagreement.py:EnsembleValueDisagreementPolicy.__init__()`;
`methods/ogsrl.py:OGSRLPolicy.__init__()`; `pipeline.py:build_method()`).

### Reproducibility, seeding, and fitted-state flow

The base seed fans out into collection, surrogate split, belief-cache streams,
policy construction, and train/holdout splitting; evaluation uses configured
block seeds plus deterministic environment/filter/policy offsets
(`pipeline.py:run_method()`; `evaluator.py:ContinuousEvaluator.run()`;
`envs.py:ContinuousEcologyEnv._spawn_rngs()`). Dataset, belief, public-surrogate,
and faithful-fit caches have different identities and validation strength.
Faithful accepted replays can require registered cache hits and
`recomputed_fits=0` (`faithful_fit.py`; `faithful_artifacts.py`;
`scripts/diagnostics/replay/run_diagnostic_replay.py:enforce_acceptance()`).
Source/input/artifact hashes and replay receipts live under `provenance/`.

Exact accepted reproduction is not guaranteed by Git contents alone: external
trajectory NPZs, fitted caches/receipts, and their expected scratch layout may
be required (`REPRODUCE.md`; `configs/paths.example.yaml`;
`provenance/dataset_hashes.csv`). See
`codebase_understanding/12_REPRODUCIBILITY_SEEDS_CACHES_PROVENANCE.md`.

### Known scientific-validity risks

The repository-specific risks already identified include:

- mismatch between fitted public reward/risk surrogates and exact evaluation
  reward (`public_surrogate.py`; accepted method call sites listed above);
- truth-derived logged rewards and a privileged true-state behavior policy
  (`envs.py:step()`; `collector.py:collect_dataset()` and
  `MixedDangerZonePolicy.act()`);
- a tuned collection collapse band and one collection dataset/fit seed for each
  accepted cell (`collector.py:calibration_summary()`; accepted manifests);
- infeasible safety recovery for Egyptian vulture under the available dynamics
  (`configs/ecology/populations.csv`; `scripts/verify_constants.py`);
- family degeneracy when effective growth rates are clamped nonpositive and
  accepted deterministic population dynamics with zero process noise
  (`envs.py:transition_value()`; accepted configs/manifests);
- silent evaluator fallback to action 0 after specified action errors, with
  fallback counts absent from the copied headline table
  (`evaluator.py:ContinuousEvaluator.run()`; accepted CSV schema);
- incomplete cache validation and possible stale/cross-cell reuse
  (`pipeline.py:ensure_dataset()` and belief-cache loading);
- unequal planning/fitting compute and hard-coded, unreachable method
  hyperparameters (`methods/bamcts.py`, EVD, OGSRL, and
  `pipeline.py:build_method()`);
- accepted Ricker-only PLUS not testing four-family posterior uncertainty
  (`src/tracks/ecological/real_ecology_benchmark/methods/plus_faithful.py`;
  `experiments/accepted_p10/manifests/plus_p10_plan_24.csv`);
- only episode-level, not dataset/fit-level, uncertainty in the headline table
  (`evaluator.py:summarize()`; accepted manifests);
- different frozen source tracks and short planning horizons relative to the
  evaluation horizon (`src/tracks/{ecological,general}`; config/manifests).

Risk severity, evidence, mitigations, and open checks are in
`codebase_understanding/14_SCIENTIFIC_VALIDITY_RISKS.md`. These are threats to
interpretation, not automatic proof that a result is invalid.

## 3. Important operating rules for future work

1. Select exactly one package track on `PYTHONPATH`; never infer equivalence
   between same-named files across tracks (`README.md` and
   `codebase_understanding/02_TWO_TRACKS_AND_REPO_LAYOUT.md`).
2. Start with `codebase_understanding/00_README_READING_ORDER.md`, then use the
   numbered documents as a map. Verify any load-bearing claim in source,
   manifests, configs, results, or provenance before relying on it.
3. Distinguish offline fitting from online simulator evaluation. Evaluation
   rollouts are fresh, but they are not field validation
   (`pipeline.py:run_method()`; `evaluator.py:ContinuousEvaluator.run()`).
4. Distinguish public policy inputs from private evaluator-only truth
   (`dataset.py`; `config.py:MethodContext`; `types.py:PublicTransition`).
5. Treat accepted manifest rows, dataset hashes, fit-cache receipts, source-track
   hashes, and the selected backend as part of the experiment identity
   (`scripts/*/run_real_manifest_row.py`; `provenance/`; accepted receipt).
6. Do not equate similarly named PLUS/MOOR implementations: legacy, native, and
   faithful versions have different models and planners (track-specific
   `methods/plus*.py` and `methods/moor*.py`).
7. Do not describe general `methods/delphic.py` as a distinct Delphic method; in
   the general track it is an alias for EVD (`methods/delphic.py` and
   `methods/__init__.py`).
8. If evidence is absent, write `Unclear`, name the likely files/artifacts, and
   give a concrete verification command or inspection procedure.

## 4. Known unclear or externally dependent points

Unclear:
Whether all accepted cells had zero evaluator action fallbacks.

Likely relevant files:
External accepted `summary.json` or `episodes.csv` files;
`src/tracks/general/real_ecology_benchmark/evaluator.py`;
`results/accepted/MATCHED_P10_144_METHOD_CELLS.csv` (which omits this field).

How to verify:
Locate every accepted external summary using its manifest/receipt path and
aggregate `fallback_count_mean` and the episode-level `fallback_count`; require
zero for each accepted row.

Unclear:
Representative public reward/risk surrogate errors across all accepted cells.

Likely relevant files:
External accepted `summary.json` files; public trajectory NPZs named by
`provenance/dataset_hashes.csv`; `public_surrogate.py`.

How to verify:
Aggregate `public_surrogate_diagnostics` and evaluator-only safety-stratified
diagnostics from all summaries, or refit the documented surrogate from every
pinned public dataset without using private fields for fitting.

Unclear:
Whether collection and evaluation episode seeds ever collide in the frozen
accepted data.

Likely relevant files:
External private/public dataset metadata, accepted manifests, `collector.py`,
and `evaluator.py`.

How to verify:
Recover or reconstruct the collector's per-episode RNG draws from base seed 116
and compare them with every realized evaluation seed. The public dataset itself
does not record collection environment seeds.

Unclear:
Whether a checkout without the original scratch project can reproduce every
accepted row exactly.

Likely relevant files:
`REPRODUCE.md`, `configs/paths.example.yaml`, `provenance/dataset_hashes.csv`,
faithful fit-cache receipts, and external dataset/cache roots.

How to verify:
Resolve every receipt path and hash, then run `make verify` and the applicable
general/ecological cell replay in the pinned Python/hardware profile. Missing
external NPZ or fit-cache artifacts means exact full-table reproduction is not
established.

## 5. Fast reading order

For a new broad question, read:

1. `codebase_understanding/00_README_READING_ORDER.md`
2. `codebase_understanding/01_BIG_PICTURE_FLOW.md`
3. `codebase_understanding/02_TWO_TRACKS_AND_REPO_LAYOUT.md`
4. the relevant topic document among `04` through `14`
5. the cited source functions, configs, manifests, and artifacts

Use `codebase_understanding/03_FILE_FUNCTION_ROLE_MAP.md` as the symbol index and
`15_RECOMMENDED_CODE_READING_PLAN.md` for a deeper executable audit.
