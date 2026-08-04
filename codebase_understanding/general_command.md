# Command: document the DeepRL_Population_Models codebase

You are helping me take maintainer-level control of this research codebase. Much of it was
generated or heavily edited by AI agents across many phases, and the repository now carries a
large amount of historical documentation that may or may not match the code.

I do not want a vague summary. I want a technically grounded explanation based on the actual
files, functions, classes, configs, CSV tables, manifests, tests, and execution paths that exist
in **this** repository.

Write the output into:

```text
codebase_understanding/
```

Use Markdown. Use equations, Mermaid diagrams, or LaTeX only where they genuinely clarify the
ecological dynamics, the reward channels, the belief/filter flow, a method's planning objective,
or the offline-fit vs evaluation difference. Do not create visuals for style.

Target reader: comfortable with Python and NumPy, familiar with RL concepts, wants working
control of this repository in 3–4 hours. Do not assume they know ecology, POMDPs in detail, or
this project's history.

---

## What this repository actually is — read this before you start

Grounding facts, which you must verify rather than assume, but which should shape your reading
order:

* **Setting: offline, model-based, belief-space (POMDP) planning/RL.** A fixed dataset is
  simulated once by a scripted privileged behavior policy; every method performs a one-shot
  offline `fit()`; policies are then rolled out online **for evaluation only**. There is no
  gradient policy-improvement loop, no replay buffer, no actor/critic, no target networks.
  PyTorch appears in essentially one place: gradient-based fitting of a mechanistic model in
  `faithful_fit.py`.
* **Environment:** `ContinuousEcologyEnv` in
  `src/tracks/general/real_ecology_benchmark/envs.py` — a continuous-state, unbounded,
  single-population conservation POMDP with four hidden dynamics families (Ricker, Allee,
  theta-logistic, regime-switching), set-point growth rate `r` and cumulative carrying capacity
  `K`.
* **Data source:** real ecological parameter tables in `configs/ecology/`
  (`species.csv` — 9 populations; `actions.csv` — an 11-action management menu with per-step
  costs; `action_effects_long.csv` — the precomputed per-(species, action) lookup the env reads).
  Trajectories are **simulated** from those parameters, not observed field data.
* **Two frozen import tracks:** `src/tracks/ecological/` and `src/tracks/general/` are two
  byte-preserved copies of the same package name `real_ecology_benchmark`. They must never be
  on `PYTHONPATH` simultaneously. Accepted PLUS/MOOR cells came from one tree; accepted
  RefPlan/OGSRL/BA-MCTS/EVD cells from the other.
* **Scientific claim under test:** whether a *new method that maintains uncertainty over the
  population-model family* is justified — i.e. do adapted ecological POMDP planners (PLUS, MOOR)
  outperform general offline RL methods (RefPlan, OGSRL, BA-MCTS, EVD) on a benchmark where
  demographic parameters and the model family are hidden. The headline artifact is the
  144-method-cell table `results/accepted/MATCHED_P10_144_METHOD_CELLS.csv`.
* **Reading orientation (claims, not ground truth):**
  `docs/current_checking_flow/00_START_HERE.md` states the current interpretation, including
  several negative findings. Treat it as a set of hypotheses to check against code.

**Do not read `REPO_AUDIT.md` in full — it is ~3 MB.** Grep it if needed. `docs/` contains
dozens of phase reports; sample them, do not read them exhaustively, and never cite a doc as
evidence of what the code does. Code is the authority; docs are claims.

---

## Which track to read

Read **`src/tracks/general/real_ecology_benchmark/` as the primary path**, because it carries
the general-RL methods that dominate the accepted 576-row manifest. Then run:

```bash
diff -rq --exclude=__pycache__ src/tracks/ecological src/tracks/general
```

and document **every divergence** between the two trees, file by file, with a one-line
explanation of what changed and why it matters scientifically. This diff is a required
deliverable, not an optional extra.

---

# Required output structure

```text
codebase_understanding/
  00_README_READING_ORDER.md
  01_BIG_PICTURE_FLOW.md
  02_TWO_TRACKS_AND_REPO_LAYOUT.md
  03_FILE_FUNCTION_ROLE_MAP.md
  04_ENVIRONMENT_AND_ECOLOGICAL_MODEL.md
  05_OFFLINE_DATASET_PIPELINE.md
  06_BELIEFS_AND_INFORMATION_REGIME.md
  07_METHODS_AND_PLANNING_OBJECTIVES.md
  08_OFFLINE_FIT_LOOP.md
  09_EVALUATION_PROTOCOL.md
  10_METRICS_LOGGING_AND_RESULT_TABLES.md
  11_CONFIGS_MANIFESTS_HYPERPARAMETERS.md
  12_REPRODUCIBILITY_SEEDS_CACHES_PROVENANCE.md
  13_VERIFICATION_AND_DIAGNOSTIC_LAYER.md
  14_SCIENTIFIC_VALIDITY_RISKS.md
  15_RECOMMENDED_CODE_READING_PLAN.md
```

Optional extras only if they are linked from and explained by a Markdown file above:

```text
  diagrams/
    cell_lifecycle.mmd
    belief_and_information_flow.mmd
    reward_channels.mmd
  latex/
    dynamics_families.tex
    method_objectives.tex
```

Do not generate files disconnected from the main explanation.

---

# 00_README_READING_ORDER.md

* Recommended reading order and what each file explains.
* What the reader should understand after each file.
* Which files are essential for a 3–4 hour pass; which are deeper dives.
* A 5–10 sentence mental model of the whole repository, stating explicitly that this is an
  offline model-based POMDP benchmark, not a deep-RL training framework.
* A dependency map:

```text
big picture
  → two tracks / layout
  → file+function map
  → environment & ecological model
  → offline dataset pipeline
  → beliefs & information regime
  → methods & planning objectives
  → offline fit loop
  → evaluation protocol
  → metrics & result tables
  → configs & manifests
  → reproducibility, caches, provenance
  → verification & diagnostics
  → scientific validity risks
```

* A short glossary of this project's vocabulary, defined from code, not from docs: *cell*,
  *track*, *manifest row*, *accepted*, *parity gate*, *receipt*, *filter*, *belief cache*,
  *surrogate*, `expose_rk`, *hidden vs full*, *operational vs true return*, *headroom*,
  *reference/constant-action control*, *faithful/adapted/native/inspired method*.

---

# 01_BIG_PICTURE_FLOW.md

Explain the lifecycle of **one benchmark cell** end to end. A cell is
(population × dynamics family × observation noise σ × reward_mode × method × information regime).

Trace it concretely through:

```text
manifest row / YAML config
→ EnvironmentConfig built from configs/ecology CSVs
→ ContinuousEcologyEnv
→ collect_dataset (scripted behavior policy) → public .npz + private .npz
→ [hidden regime only] public reward/termination surrogate fit
→ belief filter factory → belief cache
→ train/holdout split
→ policy.fit(dataset, beliefs)   ← the "training" step
→ ContinuousEvaluator.run(policy) → 20 online episodes
→ episodes.csv + summary.json
→ aggregation into the accepted 144-cell table + receipt
```

Answer, with file and function references:

* What is the main purpose of this codebase, in the authors' own terms?
* Precisely how would you classify the RL setting, and what evidence in the code supports that
  classification? Explicitly note what is absent (no replay buffer, no policy gradient, no
  target networks) so the reader is not looking for it.
* What is the entry point to run one cell? (`real_ecology_benchmark.cli:cmd_run` →
  `pipeline.run_method`; also `python -m real_ecology_benchmark`.)
* What is the entry point for evaluation, and why it is not a separate script.
* What are the runner scripts under `scripts/general/` and `scripts/diagnostics/` for, and which
  are the load-bearing ones for reproducing accepted results?
* What key objects are constructed during a run, in order?
* Where do results land on disk, and how does `_reward_mode_output_root` namespace them?

Include one Mermaid diagram of the actual cell lifecycle above.

---

# 02_TWO_TRACKS_AND_REPO_LAYOUT.md

This section is project-specific and required.

* Explain the two frozen tracks, why the duplicate package name is deliberate, and what breaks
  if both are on `PYTHONPATH`.
* Explain `src/tracks/real_ecology_data -> ../../configs/ecology` and why it is load-bearing.
* Give the complete diff between the two trees (see "Which track to read"), classified as:
  files only in one track; files that differ; and what the difference is
  (e.g. `delphic.py` vs `ensemble_value_disagreement.py`, `behavior_model.py`,
  `general_canary_acceptance.py`, and any divergence in `config.py`, `manifest.py`,
  `faithful_fit.py`, `methods/ogsrl.py`, `methods/bamcts.py`, `methods/plus_faithful.py`).
* For each accepted result, state which track produced it and how a reader can tell.
* Map the top-level directories: `src/`, `scripts/`, `configs/`, `experiments/`, `results/`,
  `provenance/`, `docs/`, `tests/`, `archive/`, `.verification/`, `.review/`. Say which are
  scientific inputs, which are outputs, which are history.
* Identify what is genuinely dead or superseded (e.g. the `delphic` alias, synthetic-only code
  paths, archived scripts) and say so plainly.

---

# 03_FILE_FUNCTION_ROLE_MAP.md

File-by-file map of every important module. For each:

```text
File:
Role:
Main classes:
Main functions:
Inputs:
Outputs:
Called by:
Calls into:
Affects: environment? dataset? beliefs? methods? evaluation? configs? provenance?
Risk level if modified:
Why this file matters:
```

Cover at minimum, in `src/tracks/general/real_ecology_benchmark/`:

`cli.py`, `pipeline.py`, `config.py`, `envs.py`, `actions.py`, `controls.py`, `realdata.py`,
`dummydata.py`, `reward.py`, `observation.py`, `collector.py`, `dataset.py`, `beliefs.py`,
`public_surrogate.py`, `public_models.py`, `behavior_model.py`, `dynamics.py`, `planning.py`,
`planners/pbvi.py`, `native_fit.py`, `native_solver.py`, `discretize.py`, `faithful_fit.py`,
`faithful_ecology.py`, `faithful_pomdp.py`, `faithful_artifacts.py`, `evaluator.py`, `gate.py`,
`manifest.py`, `training_monitor.py`, `telemetry.py`, `backend.py`, `privacy.py`, `types.py`,
and every file in `methods/`.

Also cover: `src/diagnostics/replay_analysis/*`, `scripts/verify_*.py`,
`scripts/diagnostics/replay/*`, `scripts/diagnostics/followups/*`, `scripts/general/*` (group
the routine manifest builders; detail the runners), and the `tests/` layout.

Then explicitly flag:

* files that look important but are unreachable from any entry point;
* logic duplicated across `plus.py` / `plus_native.py` / `plus_faithful.py` (same for MOOR) and
  what actually distinguishes them;
* abstractions that add indirection without adding behavior;
* code whose ownership boundary is unclear between `pipeline.py`, `manifest.py`, and the
  `scripts/` runners.

---

# 04_ENVIRONMENT_AND_ECOLOGICAL_MODEL.md

The environment is the scientific object in this project. Treat it as the most important
section after the big picture.

Explain:

* `EnvironmentConfig` — every field that changes dynamics, reward, or safety, and where each is
  populated from (`real_environment`, `real_environment_like`, `species.csv`,
  `action_effects_long.csv`).
* The four dynamics families in `transition_value()`. Write each map as an equation matched to
  the code, including:
  * the `r_pos = max(r_eff, 0)` / `r_mort = min(r_eff, 0)` split and why it exists;
  * `_safe_mul_exp` overflow/underflow guards;
  * absorbing state at `s = 0`;
  * regime switching and `regime_persistence`.
* The set-point `r` / cumulative `K` control scheme in `controls.py`: `rho`, `kappa`, `K_eff`,
  clipping to `[K_base, K_max]` and `[r_min, r_max]`, and which action is the only direct-state
  action (`a10` translocation, `dN = 0.10 * N0`).
* The 11-action menu from `configs/ecology/actions.csv`: channel, `lambda_source`,
  `K_multiplier`, `dN_fraction`, `cost_step`. State plainly that harvest actions have negative
  cost (revenue) and that costs come from a separate cost-portal source than the demographic
  parameters.
* The λ → r conversions: `r_Ricker = ln(λ)` for ricker/allee/regime; `r_LGM = λ − 1` for theta.
* The observation model `LogNormalObservationModel` and what σ ∈ {0.1, 0.2} means in abundance
  terms.
* **The three reward channels** in `reward.py`, which is a known source of confusion:
  `operational()`, `true()`, `state_reward()`. Say exactly which one the environment logs into
  the dataset for real cells, which the evaluator accumulates, and what `effective_collapse_penalty`
  does under `reward_mode ∈ {yield, safe}`.
* `safety_penalty_indicator` — `occupancy` vs `crossing`, and why the default changed.
* Per-population `K_ref`, `s_safe`, `N0`, `mvp_threshold`, and the depletion-aware
  `default_safety_fraction` tiering. State the actual values for the three benchmark species and
  warn that there is no global default.
* Termination vs truncation: absorbing `s = 0`, `horizon = 50` for evaluation,
  `episode_length = 25` for collection.
* `process_noise_sigma` — confirm its accepted value and state what a zero value implies about
  the stochasticity of the benchmark.

Give a runnable snippet that instantiates one real cell and prints the resolved action table,
`K_ref`, `s_safe`, `r_min/r_max`, and a one-step transition under each family.

---

# 05_OFFLINE_DATASET_PIPELINE.md

Classify the setup explicitly at the top: **offline RL over a simulator-generated dataset, with
online rollout used only for evaluation.**

Describe:

```text
Data generating policy:
Public dataset fields:
Private (evaluator-only) fields:
Episodes / transitions per cell:
Observation representation:
Action representation:
Reward source (logged vs recomputed):
Terminated vs truncated:
Train / holdout split:
Belief cache:
Caching, locking, and cell validation:
Seeding:
```

Answer:

* What is `MixedDangerZonePolicy` and its four mixture components? Why is it privileged (acts on
  true state, not observation), and what does that imply about the behavior policy's
  realizability?
* What are `CollectorProfile`, `DEFAULT_PROFILE`, `REAL_DEFAULT_PROFILE`, and the per-(family,
  action-count) overrides in `PROFILES`? What is the `[0.15, 0.24]` healthy-start collapse band
  and what does calibrating to it do to the data distribution?
* Exactly which fields are public (`TrajectoryDataset`) and which are private
  (`PrivateTrajectoryData`)? Which private fields would leak truth if a method touched them?
* Rewards are **logged from the environment using the true next state** — so what information
  about the hidden state is implicitly in the public reward channel? Discuss the metadata note
  `truth_derived_public_signal`.
* How does `ensure_dataset` validate a cached dataset against the requested cell
  (`_validate_dataset_cell`), and what mismatches would it *not* catch?
* How does the file-lock path work and what happens on a stale lock?
* How does `split_train_holdout` split (episode-disjoint, seed offset), and is the holdout used
  for anything that affects results, or only for diagnostics?
* Where is the train/eval boundary? Evaluation regenerates fresh episodes from
  `evaluation.seeds`; verify whether those seeds can collide with collection seeds, and say so
  either way.

Give a concrete snippet that loads a public `.npz` and prints: transitions, episodes,
observation stats, action frequencies, reward statistics, terminated/truncated counts, and
`dataset_sha256`.

---

# 06_BELIEFS_AND_INFORMATION_REGIME.md

Project-specific and required. This is where "what does the method know" is decided.

Explain:

* The `expose_rk` regime: `hidden` vs `full`. What `hides_rk` gates, and why the accepted
  results are `hidden`.
* `MethodContext` — the sanitized view a hidden-regime policy receives instead of
  `EnvironmentConfig`. Enumerate exactly what it exposes and what it withholds. Show where
  `BasePolicy.__init__` refuses an `EnvironmentConfig` in hidden mode.
* The filter ladder in `make_filter_factory`: `raw`, `learned`, `reference`, `ricker`,
  `true_family`, `native_discrete`, `oracle`, `faithful_internal`. For each: what it assumes,
  and which are forbidden in hidden mode.
* `ParticleFilter`, `PublicObservationFilter`, `DiscreteGridFilter`, `OracleStateFilter`,
  `LearnedLinearProposal`, `MechanisticProposal`, `ReferenceProposal`.
* Why `OracleStateFilter` raises if used as a training input, and how it is used as an
  evaluator-only ablation in `run_oracle_state_ablation`.
* The belief caches (`BeliefCache`, `PublicBeliefCache`), what features they store, how the
  cache key is constructed, and how a stale cache could silently be reused.
* `PublicRewardRiskSurrogate` — what it is fitted on, its episode-level fit/holdout split, what
  it predicts, its reported diagnostics, and `evaluator_only_surrogate_diagnostics`.
* **The critical point:** which methods plan against the surrogate and which use logged rewards.
  Verify this against the code for all six methods and state it explicitly, because the
  planner's objective and the evaluator's scoring reward may not be the same function.

Include a Mermaid diagram of information flow showing the public / sanitized / private boundary.

---

# 07_METHODS_AND_PLANNING_OBJECTIVES.md

Replaces the generic "model implementation" section. There are no policy/critic networks here —
there are fitted models and planners.

For each entry in `METHODS`, provide:

```text
Method key / class / file:
Lineage: adapted | inspired | motivated | native | faithful — and what that word means here
What is fitted offline:
What the fitted object represents (dynamics? Q? posterior? candidate bank?):
Planner used at action-selection time:
Objective actually optimized (write the equation from the code):
Reward channel it plans against (logged / surrogate / mechanistic):
Uncertainty mechanism (posterior, ensemble variance, pessimism penalty, OOD guardian, none):
Information it consumes (MethodContext fields, belief features):
Hyperparameters that matter and where they come from:
Compute cost, per `timings` in summary.json:
Used in the accepted results? Which track?
```

Cover: `mopo`, `refplan`, `bamcts`, `moor`, `plus`, `moor_native`, `plus_native`,
`ensemble_value_disagreement_pessimism` (and the deprecated `delphic` alias), `ogsrl`,
`moor_adapted_ricker_misspec_pbvi`, `plus_adapted_mechanistic_pbvi`.

Then answer:

* Which model chooses actions at evaluation time for each method?
* Which fitted objects are used only during `fit()` and discarded?
* Where are the actual PyTorch gradients in this repo, and what do they fit?
  (`faithful_fit.py` — describe the objective, the reparameterizations, the custom
  `SymmetricPositive` autograd function, and the hierarchical penalties.)
* What is `PUBLIC_FORM_CANDIDATES` and how does PLUS's candidate posterior over families relate
  to the paper's four-family design?
* Are gradients flowing where expected in `faithful_fit.py`, and is anything detached that
  should not be?
* Are any registered methods never exercised by an accepted manifest row?

Write the objectives as equations only where they match the code. Keep the notation consistent
with `04` (`s`, `s'`, `o`, `a`, `r_eff`, `K_eff`, `s_safe`, `K_ref`).

---

# 08_OFFLINE_FIT_LOOP.md

This is this repo's analogue of a training loop. Say so, and say what it is not.

Trace `pipeline.run_method` line by line:

```text
Backend resolution (numpy / cupy, strict mode):
Dataset construction or cache load:
Surrogate fit or load (hidden regime):
MethodContext construction:
Filter factory:
Belief cache construction or load:
Train/holdout split:
Policy construction (METHODS[method](...)):
policy.fit(dataset, cache):
Training-history attachment and final fit metrics:
Evaluator construction:
Timing and telemetry capture:
Artifact writing:
```

Answer:

* What single command runs one cell end to end? Give the exact invocation, including
  `PYTHONPATH` selection.
* What is "one training step" in this repo, and why the question is partly ill-posed here.
* Which methods have an iterative fit (e.g. EVD's `fit_iterations`, PBVI sweeps, VI iterations)
  and which are closed-form/one-shot? Give the actual iteration counts and where they are
  configured.
* Which seeds are derived where (`cfg.seed + 20_000 / 30_000 / 40_000` and the split offset) and
  what each controls.
* What does `attach_training_history` / `TrainingHistory` actually record, and is any of it used
  for selection or only for reporting?
* What can silently go wrong: stale caches reused across cells, surrogate hash mismatch falling
  through to a refit, a `fit()` exception swallowed downstream, backend fallback changing
  numerics, `fit_diagnostics` being empty for some methods.

Include pseudocode matching the real control flow of `run_method`, not a generic RL loop.

---

# 09_EVALUATION_PROTOCOL.md

Trace `ContinuousEvaluator.run` step by step.

```text
Entry point:
Episode construction (block seeds × episodes_per_seed):
Environment reset semantics (deterministic s0 = N0):
Filter reset and policy reset:
Action selection and the fallback path:
Step, observe, belief update:
Return accumulation (operational vs true, discounting):
Per-episode row construction:
Aggregation in summarize():
Where results are written:
```

Answer:

* How many episodes, at what horizon, at what discount, and where those are configured
  (`evaluation.seeds`, `episodes_per_seed`, `horizon`, `discount`; note `min(eval horizon, env
  horizon)`).
* Is the policy deterministic at evaluation? Is exploration disabled? Is there any per-episode
  randomness left, and where does it come from?
* Are evaluation seeds disjoint from collection seeds and belief-cache seeds? Show the arithmetic.
* Is the environment used at evaluation the same object/config as the one used to collect data?
  What would it mean scientifically if it is?
* What does the `except (FloatingPointError, ValueError, RuntimeError): action = 0` fallback do
  to a method's score, how often does it fire (`fallback_count`), and is a silent degradation to
  do-nothing distinguishable from a genuine do-nothing policy?
* Which metrics use `evaluator_info` (private truth)? List them and confirm none leak into the
  policy path.
* How does `run_oracle_state_ablation` differ, and why is it an upper bound rather than a result?
* What does `gate.py` (`run_decision_gate`) evaluate, and how does the clairvoyant
  `ExactEpisodeProposal` controller function as a ceiling?

Provide an explicit **offline-fit vs online-evaluation comparison table**: data source,
information available, reward channel, randomness, seeds, what is written.

---

# 10_METRICS_LOGGING_AND_RESULT_TABLES.md

For each metric in `evaluator.py` and `manifest.py`:

```text
Metric:
Where computed:
Definition (from code):
Uses private truth?
Aggregation (mean/std, ddof):
Where it lands (episodes.csv / summary.json / accepted CSV):
Used for any selection or claim?
Known interpretation trap:
```

Cover at least: `operational_return`, `true_return`, `collapse_entry(_ies)`,
`collapse_entry_timestep`, `unsafe_fraction`, `mvp_fraction`, `mvp_breach`, `persistence`,
`economic_cost`, `min_true_state`, `final_true_state`, `filter_rmse`, `filter_log_rmse`,
`filter_coverage90`, `filter_ess_fraction`, `filter_unsafe_brier`, `action_entropy`,
`method_uncertainty_mean`, `danger_action_*_fraction`, `fallback_count`, and the timing/RSS
fields.

Then answer:

* Which metric is the headline scientific result, and in which column of
  `results/accepted/MATCHED_P10_144_METHOD_CELLS.csv`?
* What is the variance actually being reported by `*_std`? State explicitly that within-cell SD
  is 20 episodes under **one** dataset and **one** fitted policy — not variation across datasets,
  seeds of collection, or fits — and therefore what it cannot support.
* Is `operational_return` equal to `true_return` for real cells? Prove it from `envs.step`.
* How does `aggregate_summaries` in `manifest.py` compare methods, and what does the
  `--baseline METHOD:FILTER` "beats-both" comparison do?
* Where are the final result artifacts: `results/accepted/`, `results/followups/{S2,H12,H14,S6,
  reward_screen}/`, `results/diagnostic_replay/`, and the receipt JSONs. Explain what a receipt
  contains and what it guarantees.
* Is there any logger (TensorBoard/W&B)? Check `training.wandb` / `training.plot` and report
  what is actually wired versus configured-but-unused.
* Are constant-action / reference controls reported anywhere, and what is the documented rule
  about treating them as a method? (See `scripts/diagnostics/replay/constant_action_sweep.py`
  and `a0_baseline.py`.)

---

# 11_CONFIGS_MANIFESTS_HYPERPARAMETERS.md

This project has **three** configuration layers. Explain all three and how they interact:

1. YAML configs (`configs/*.yaml`, `configs/tracks/*`, `experiments/*/configs/*.yaml`) parsed by
   `config.load_config` into `BenchmarkConfig`.
2. CLI overrides in `cli._config` (`--population`, `--environment`, `--sigma`, `--reward-mode`,
   `--expose-rk`, `--data-mode`, `--backend`, ...) — document the precedence order and the
   rebuild-vs-override behavior of `real_environment_like`.
3. **Manifest rows** — the actual experiment specification.
   `experiments/accepted_general/manifests/full_general_sigma01_02_576_rows.csv` (576 rows) and
   `experiments/accepted_p10/manifests/{moor,plus}_p10_plan_24.csv`. Document their columns,
   how a row overrides the base YAML, and which runner consumes them.

Note explicitly where the base YAML and the manifest disagree (e.g. a config-level
`observation_noise_sigma` that every manifest row overrides) so the reader does not trust the
YAML alone.

For each important hyperparameter give:

```text
Name / dataclass field:
Default:
Defined where:
Used where:
Effect:
Risk if changed:
Appears in accepted manifests as:
```

Cover at minimum: `seed`; `dataset.transitions`, `episode_length`; `filter.particles`,
`proposal`, `ess_fraction`; `model.ensemble_size`, `ridge`, `native_state_bins`,
`native_vi_iterations`; `planner.horizon`, `sequences`, `particles`, `discount`, `pessimism`,
`bamcts_depth`, `bamcts_simulations`, `ogsrl_cost_horizon`, `ogsrl_deployment_rollouts`,
`ogsrl_low_abundance_quantile`; `evaluation.seeds`, `episodes_per_seed`, `horizon`, `discount`;
`training.enabled`, `holdout_fraction`; `environment.collapse_penalty` (P=10 vs the superseded
P=5), `reward_mode`, `safety_penalty_mode`, `expose_rk`, `observation_noise_sigma`,
`process_noise_sigma`; `compute.backend`, `device`, `strict`; and the `faithful.fit.*` settings.

Answer:

* Which config values are genuinely load-bearing and which are inert for accepted cells?
* Where are conflicting or shadowed values?
* **What exact config + manifest row + command reproduces one accepted cell?** Give it verbatim.
* Flag the P=5 vs P=10 trap: historical ecological returns used `collapse_penalty=5` and are not
  comparable to the accepted table.

---

# 12_REPRODUCIBILITY_SEEDS_CACHES_PROVENANCE.md

Replaces the generic seeds/checkpoints section — there are no model checkpoints here; there are
**caches, hashes, and receipts**.

Explain:

```text
Seed derivation tree (cfg.seed and every offset):
RNG streams inside the env (SeedSequence.spawn: parameters/regime/process/observation/initial):
Where NumPy generators are created and reset:
Torch seeding in faithful_fit.py:
Backend determinism (numpy vs cupy; BLAS/threading; the accepted 8452Y Slurm profile):
Cache layers: dataset .npz, belief cache, learned filter, public surrogate, faithful fit cache:
Cache key construction and staleness detection for each:
What is saved per run (episodes.csv, summary.json, offline_beliefs.npz, fit artifacts):
Provenance: provenance/*.sha256, dataset_hashes.csv, frozen_snapshots.json, receipts:
```

Answer:

* Can a single accepted cell be reproduced bit-for-bit, and what external inputs are required
  (`SCRATCH_PROJECT`, `DEEPRL_GENERAL_DATA_ROOT`, fit caches, private trajectories)?
* What is the `recomputed_fits = 0` requirement and why does it exist?
* What is the seven-field `1e-9` parity gate, which fields, and where is it enforced?
* What cannot be reproduced from this repository alone, and what is the minimum external data
  needed? Be specific and honest about it.
* Is anything nondeterministic in principle (thread counts, BLAS, cupy fallback, dict ordering)?

List the exact commands to: run the full verification, run one accepted ecological cell, run one
accepted general cell, run the unit tests for each track, and run a smoke cell from scratch.

---

# 13_VERIFICATION_AND_DIAGNOSTIC_LAYER.md

Project-specific and required. A large fraction of this repository is instrumentation, not
experiment, and a reader who does not understand that will misread the repo.

Explain:

* The `Makefile` targets and exactly what each proves: `verify-self-tests`, `verify-constants`,
  `verify-integrity`, `verify-negative-gate`, `verify-cell`, `verify-cell-general`,
  `verify-standalone-s2`, `test-ecological`, `test-general`.
* `src/diagnostics/replay_analysis/` — the M1–M15 metric battery. For each metric: what it
  measures and which hypothesis it tests. Explain `constants.py` as a registered, audit-sourced
  constants file, including the documented inert `r_lgm` discrepancy for the vulture, and
  `schema_report` / graceful `{"available": False}` degradation.
* `scripts/diagnostics/replay/` — instrumented replay of accepted policies, the parity gate, and
  the constant-action / a0 reference sweeps.
* `scripts/diagnostics/followups/` — S2 (oracle regret / VPI), H12, H14, S6, and the reward
  screen. Say what each was designed to answer.
* The independent review artifacts: `REVIEW.md`, `S2_ORACLE_REVIEW.md`, `FIXES.md`,
  `VERIFY_REPORT.md`, `.review/`. Summarize which findings were fixed and which remain open —
  and check the fix claims against current code rather than trusting `FIXES.md`.
* `tests/` — what the test suites actually assert per track, and which are structural
  (shapes/schemas) versus scientific (equation identities, gate behavior).

State clearly which of these are the checks a maintainer should run before trusting any number.

---

# 14_SCIENTIFIC_VALIDITY_RISKS.md

Code-level risks that could change the research conclusion. Risk table:

```text
Risk:
Severity:
Affected files/functions:
Why it matters for the claim:
How to verify (concrete command or check):
Suggested fix:
Status (open / mitigated / already documented where):
```

Investigate at minimum the following project-specific risks. Verify each against code — confirm,
refute, or mark unresolved. Do not restate them as findings without checking.

* **Objective mismatch.** Methods that plan against the learned surrogate are scored on the
  logged true-next-state reward. Determine which methods, quantify the surrogate's reported fit
  quality, and state whether any safety-related conclusion can be attributed to a method's
  reasoning.
* **Truth in the public reward.** Rewards are computed from the true next state and logged into
  the public dataset. Determine what a method could in principle invert from the reward channel.
* **Privileged behavior policy.** The collector acts on true state; the offline data is therefore
  not from a realizable observational policy. What does this do to offline-RL assumptions
  (coverage, behavior-policy estimation in `behavior_model.py`)?
* **Collector calibration to a target collapse band.** The behavior mixture is tuned per family
  to land in `[0.15, 0.24]`. Is the data distribution then a tuned quantity that co-varies with
  the result?
* **Feasibility of a benchmark cell.** Check whether any species/threshold combination admits no
  policy that clears `s_safe` (the Egyptian vulture claim: `s_safe = 81.25`, reachable abundance
  far below). If a cell is infeasible, every method scores identically there — determine whether
  such cells are included in aggregate rankings.
* **Family degeneracy.** If a chosen action drives `r_eff ≤ 0`, `r_pos = 0` zeroes the
  density-dependent term — the only place the four families differ. Verify this in
  `transition_value` and determine whether it makes the family-uncertainty question
  unanswerable for the deployed policies.
* **`process_noise_sigma = 0`.** Confirm the accepted value and analyze what a deterministic
  transition map means for "uncertainty" claims and for the S2 oracle's privilege.
* **Silent fallback to action 0.** Does `fallback_count > 0` in any accepted cell, and does the
  aggregation surface it?
* **Cache reuse across cells.** Can a dataset, belief cache, learned filter, or surrogate from a
  different cell be reused? Test `_validate_dataset_cell` coverage and the cache-key construction
  for gaps.
* **Unequal compute across methods.** Extract per-method `fit_seconds` / `row_seconds` from
  summaries or the docs and state whether the comparison is compute-matched.
* **Sample size and statistics.** 20 episodes, one dataset, one fit per cell, `std(ddof=1)`
  across episodes only. Are any comparative claims supported at that resolution? Are confidence
  intervals or paired tests computed anywhere?
* **S2 oracle findings.** `S2_ORACLE_REVIEW.md` reports (a) the VPI baseline optimizes over five
  hand-picked candidates rather than admissible common policies, and (b) a sign inversion in the
  stored `mean_regret`. Check the current state of `scripts/diagnostics/followups/run_s2.py`
  against those claims and report what is fixed.
* **Two tracks, one results table.** Accepted PLUS/MOOR and general-method cells come from
  different frozen trees. Verify that the shared evaluator, reward constants, action table, and
  seeds are genuinely identical across tracks — this is the load-bearing assumption of the
  matched 144-cell comparison. Diff the relevant files and report any divergence.
* **Deprecated aliases and naming.** `delphic` → `ensemble_value_disagreement_pessimism`.
  Confirm no artifact or table attributes Delphic-style latent-confounder uncertainty to a
  method that only bootstraps a ridge Q ensemble.
* Also check the standard items where they apply: leakage between fit and evaluation data,
  holdout influencing anything selected, normalization statistics fitted on all data,
  terminal-state bootstrapping, discounting consistency between planner and evaluator, and tests
  that only exercise happy paths.

Close with:

* the places where a bug would most plausibly invalidate the headline comparison;
* which risks must be resolved before trusting any number;
* which are lower priority;
* the three or four smallest checks a maintainer should run first, with exact commands.

No "top-10 files" list.

---

# 15_RECOMMENDED_CODE_READING_PLAN.md

A practical 3–4 hour plan, in timed blocks, using real paths, functions, and commands from this
repository. Suggested shape — adjust to what you actually found:

```text
0:00–0:15  Orientation: two tracks, repo layout, one cell's lifecycle
0:15–0:45  The environment: envs.py, controls.py, actions.py, reward.py + the ecology CSVs
0:45–1:10  The offline dataset: collector.py, dataset.py, ensure_dataset
1:10–1:40  Beliefs and the information regime: beliefs.py, MethodContext, public_surrogate.py
1:40–2:20  Methods: two in depth (one adapted, one general) + the METHODS registry
2:20–2:45  pipeline.run_method end to end
2:45–3:10  evaluator.py and the accepted results table
3:10–3:35  Configs, manifests, provenance, receipts
3:35–4:00  Validity risks + run the fast verification checks
```

For each block give:

```text
Files to read (exact paths):
Functions/classes to inspect:
Questions to answer:
Expected understanding after this block:
Verification command or snippet to run:
```

Every block must end with something the reader can execute — a `make` target, a `python -c`
snippet, a `grep`, or a test invocation — that confirms understanding rather than assuming it.

---

# Grounding requirements

Every claim must be grounded in code. When you assert something, cite:

* file path (repo-relative), and the track if the file exists in both;
* function or class name, with line numbers where useful;
* config field, manifest column, or CSV column where applicable;
* the command or script that exercises it.

Use the form: ``In `src/tracks/general/real_ecology_benchmark/pipeline.py`, `run_method()` does X.``

Where the two tracks differ, say which track you are describing.

If something is unclear, do not guess. Write:

```text
Unclear: ...
Files inspected:
Competing explanations:
How to resolve:
```

Where documentation under `docs/` contradicts the code, report **both** and say which one you
verified. Do not silently adopt the doc's version. Several docs are explicitly marked historical
or superseded; check dates and the decision log before relying on any of them.

---

# Quality bar

Prioritize:

* execution flow of one cell, end to end;
* the environment and reward semantics, which are the scientific object here;
* the information boundary (public / sanitized / private) and who sees what;
* the offline-fit versus online-evaluation distinction;
* what each metric means and what it cannot support;
* configs, manifests, caches, and provenance;
* scientific-validity risks specific to this benchmark.

Avoid:

* generic RL/deep-RL explanations not tied to this repository;
* looking for machinery that is not here (replay buffers, target networks, policy gradients) —
  note its absence once and move on;
* folder summaries without function-level detail;
* citing `docs/` as evidence of code behavior;
* reproducing the historical narrative in `docs/` instead of reading the code;
* long prose without code references;
* "top 10 files/functions" lists.

The result should read in order from `00_README_READING_ORDER.md` through
`15_RECOMMENDED_CODE_READING_PLAN.md`, and should be good enough that a maintainer could use it
to audit — not merely navigate — this codebase.
