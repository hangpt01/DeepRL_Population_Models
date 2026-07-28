# Server Context — Motivation Experiment, Native-Baseline Gate

**Date:** 2026-07-11
**Author:** code-aware agent (server side)
**Responds to:** `docs/benchmark/SERVER_HANDOFF_motivation_experiment.md`
**Status:** read-only inspection complete. **No code, config, or run has been touched.**
**Revision 2 (2026-07-11):** corrections from `SERVER_AUDIT_motivation_native_baselines_gate.md`
folded in. Three of them are load-bearing and are marked **[AUDIT]** below: the `K_eff` lattice
is population-scaled and my "25-unit" claim was wrong (§3); the aggregator keys on `filter`, so
renaming the baselines is *not* enough to make the headline table compute (§5c); and
`make_manifest` cannot express per-method filter routing (§5g). Read this revision, not the first.

This document answers the §4 gate, maps every §5–§6 requirement onto real file paths,
lists the places where the handoff's assumed mechanics differ from the code as built, and
proposes an implementation plan. It stops short of implementing anything.

---

## 1. Gate result — **BUILD REQUIRED**

**There is no native discrete-POMDP PLUS/MOOR in the benchmark package.** The planning-side
expectation in §4 is correct. But the picture is more useful than a flat "no": a *working
native discrete-POMDP solver stack already exists in this repository*, just not wired to the
benchmark. Details below.

### 1a. `PLUS-GPU` is not a solver

The `PLUS-GPU` string the handoff asked about resolves to a **compute-device routing
constant**, not a solver:

- [backend.py:34](../../src/real_ecology_benchmark/backend.py#L34) — `PLUS_GPU_WORKLOADS = frozenset({"method:plus", "oracle_ablation:plus"})`
- [backend.py:139](../../src/real_ecology_benchmark/backend.py#L139) — used inside `_cupy_scope_for_workload()`, which returns the scope string `"plus_mechanistic_transition"` for those workloads.

Its whole meaning: *PLUS rows on set-point-cumulative cells are allowed to request the CuPy
backend, because PLUS's hot path (the candidate-bank mechanistic transition inside
`MechanisticProposal`) is the one kernel that has a GPU implementation.* Everything else in
PLUS — planner, posterior weights, I/O — stays on CPU NumPy. There is no alpha-vector, no
belief grid, no value iteration behind that name.

### 1b. No POMDP solver code in the benchmark package, and none installed

Grep over `src/`, `configs/`, `scripts/` for `pomdp|pbvi|sarsop|despot|qmdp|value.?iteration|discretiz|native.*solver`:

- The only hits in `src/real_ecology_benchmark/` are the word "POMDP" in four **docstrings** — [envs.py:1](../../src/real_ecology_benchmark/envs.py#L1), [evaluator.py:1](../../src/real_ecology_benchmark/evaluator.py#L1), [types.py:35](../../src/real_ecology_benchmark/types.py#L35), [\_\_init\_\_.py:4](../../src/real_ecology_benchmark/__init__.py#L4). Zero solver code.
- No `discretiz*` anywhere in the package. Nothing bins state, observation, or belief.
- Installed solver packages: `pomdp_py`, `pomdpy`, `pypomdp`, `sarsop`, `julia` — **all missing** (`ModuleNotFoundError`).
- The package's entire third-party surface is `numpy`, `PyYAML`, `hashlib`. No torch, no d3rlpy, no cupy on the login node. Confirmed against [pyproject.toml](../../pyproject.toml) (`dependencies = ["numpy>=1.24", "PyYAML>=6.0"]`) and an import scan of `src/`. **A NumPy QMDP/belief-grid solver adds no new dependency.** That is a point in favour of the handoff's "start with QMDP" recommendation.

### 1c. But a real native discrete-POMDP stack *does* exist — in `baseline_original/`

This is the part the planning side could not see. The vendored AAAI21 baseline
(the "2-state n-action adaptive-management solver", the 16-model MOMDP) ships a complete,
compiled, native POMDP toolchain:

| Asset | Path | State |
|---|---|---|
| **SARSOP solver binary** | [baseline_original/sarsop/src/pomdpsol](../../baseline_original/sarsop/src/pomdpsol) | **compiled and executable** — ELF 64-bit x86-64, 17 MB, built 2026-05-16 |
| POMDPX model writer | `baseline_original/python_port/src/building hmMDP/generate_pomdpx.py` (`write_hmMDPx`) | present |
| SARSOP driver | `baseline_original/python_port/src/solving hmMDP/main.py` (`_resolve_sarsop_binary`) | present |
| Alpha-vector reader / evaluator | `baseline_original/python_port/src/simulations/read_policyx.py` | present |
| Discrete `value_iteration` + Bellman backup | `baseline_original/python_port/src/utils/mdp_tools.py:67,81` | present |
| End-to-end benchmark driver | `baseline_original/python_port/scripts/run_python_benchmarks.py` | present, has been run (see `results/potoroo/`) |

**Why this does not close the gate.** This stack solves the *published* hmMDP problem — its
own state/action/observation spaces, its own gross-yield objective. It is not importable
from `real_ecology_benchmark` (it is not a package; two of its source directories have
spaces in their names), it knows nothing about our 11 real actions, our 9 populations, our
`(rho, kappa, K_eff)` public controls, or our conservation reward. Nothing in
[methods/\_\_init\_\_.py](../../src/real_ecology_benchmark/methods/__init__.py) references it.

**Why it still matters.** It de-risks the build in two ways: (1) it is prior art in this
repo for the exact discretize→POMDPX→solve→alpha-vector loop, and (2) if belief-grid VI
turns out to be too weak on the finer grids, we have a **compiled SARSOP already on disk** —
the upgrade path in §4 of the handoff ("add PBVI/SARSOP later if a dependency is available")
is available *today*, via a POMDPX export, with no install.

### Gate verdict

> Native discrete-POMDP PLUS/MOOR: **does not exist — build required.**
> `PLUS-GPU` is a CuPy routing flag, not a solver.
> No POMDP solver library is installed.
> A compiled SARSOP + POMDPX writer exists in `baseline_original/` and is a reusable asset for a later solver upgrade, but is bound to the published hmMDP problem, not to our env.

---

## 2. Configuring / running native PLUS + MOOR

Not applicable — nothing to configure. Superseded by §3 and §6.

---

## 3. Build-required: what has to be added

Proposed new modules (names are proposals, not yet written):

| File | Contents |
|---|---|
| `src/real_ecology_benchmark/discretize.py` | **new.** Per-population state grid `0 … K_max` + collapse bin + overflow bin; observation grid; discretized log-normal emission matrix; the `kappa → K_eff` public lattice. Resolution is a config field so coarse-vs-fine is a flag, not an edit. |
| `src/real_ecology_benchmark/methods/native_solver.py` | **new.** Shared machinery: builds `T[a, k, s, s']`, `O[s', o]`, `R[s, a]` from a mechanistic model; QMDP / belief-grid value iteration; discrete belief update on `(a, o)` only. |
| `src/real_ecology_benchmark/methods/plus_native.py` | **new.** `PLUSNativePolicy` — candidate Ricker models over `K`, discretized, evidence/posterior update over candidates, belief planning. |
| `src/real_ecology_benchmark/methods/moor_native.py` | **new.** `MOORNativePolicy` — least-squares Ricker fit from the offline `(o, a, o')` data, discretize once, solve once. |
| `src/real_ecology_benchmark/beliefs.py` | **modify.** Add `DiscreteGridFilter` implementing the `BeliefFilter` protocol (see §5b for why this is required rather than optional). |
| `src/real_ecology_benchmark/pipeline.py` | **modify.** One new branch in `make_filter_factory` for the native belief route. |
| `src/real_ecology_benchmark/methods/__init__.py` | **modify.** Two new `METHODS` entries. |
| `src/real_ecology_benchmark/manifest.py` | **modify.** Parameterize the hardcoded `{"plus","moor"}` baseline set (see §5c — this is a correctness bug for this run, not a cosmetic one). |
| `src/real_ecology_benchmark/cli.py` | **modify.** Method/filter `choices=[...]` lists; optionally expose `make_manifest`'s existing filter arguments. |
| `src/real_ecology_benchmark/collector.py` | **modify.** Emit a `dataset_sha256` into dataset metadata (acceptance test §7.3). |
| `tests/real/test_native_baselines.py` | **new.** |

### The native model is a MOMDP, not a flat POMDP — and that is a large simplification

This is the single most important structural fact for the build, and it is not visible from
the docs side. In the real set-point setting:

- **`rho` is memoryless.** `accumulator_decay_r = 1.0`, so `rho_next = (1 - 1.0)·rho + delta_r[a] = delta_r[a]` ([beliefs.py:52](../../src/real_ecology_benchmark/beliefs.py#L52), [controls.py](../../src/real_ecology_benchmark/controls.py) docstring). And `r_base_low = r_base_high = 0.0` for every real cell, so `r_eff = clip(delta_r[a], r_min, r_max)` is a **pure function of the current action**. `rho` does not need to be in the POMDP state at all.
- **`kappa`/`K_eff` are public and observed.** They are carried in `public_info` and in the dataset as `PUBLIC_CONTROL_FIELDS` ([dataset.py:23](../../src/real_ecology_benchmark/dataset.py#L23)). `kappa_next = kappa + delta_K[a]`; `K_eff = clip(K_base + kappa, K_min, K_max)`, saturating at `K_max = 2·K_base`. So `K_eff` is a small, finite, **fully-observed** public dimension.
- **Only the *environment's* abundance `s` is hidden** — plus the per-episode hidden biology (`C` for allee, `theta` for theta-logistic, `regime`), which is exactly the structural uncertainty the naive Ricker solver is *supposed* to be blind to. (This says nothing about a *method's* internal model uncertainty; PLUS-native's posterior over candidate `K` models is a separate, additional hidden dimension it maintains itself — see the corollary below.)

So the native model is a **mixed-observability MDP**: belief over `s` only, conditioned on an
observed `K_eff` level. `T` is indexed `[a, k_level, s, s']` and the belief update never has to
integrate over `K_eff`. This is the same MOMDP shape as the AAAI21 baseline, and it keeps the
belief-grid solve tractable at the grid sizes §4 asks for.

**[AUDIT] The `K_eff` lattice is population-scaled — do not hardcode 25.** My first revision read
`delta_K ∈ {0, +25, +75}` off the *Amur tiger* action table and wrongly generalized it to an
absolute 25-unit lattice. The capacity channel is a **multiplier**, not an absolute step:
[real_ecology_data/actions.csv](../../real_ecology_data/actions.csv) gives `K_multiplier` 1.1 (a5, a7)
and 1.3 (a6, a8, a9), which `action_effects_long.csv` materializes per population as
`dK_step = {0.1, 0.3} · K_base`. Verified across all nine populations via `resolve_actions(cfg)`:

| Population | `K_base` | `delta_K` | `delta_K / K_base` |
|---|---|---|---|
| Egyptian vulture, Jaguar | 325 | 0, 32.5, 97.5 | 0, 0.1, 0.3 |
| Amur tiger, Puerto Rican parrot | 250 | 0, 25, 75 | 0, 0.1, 0.3 |
| Iberian lynx | 165 | 0, 16.5, 49.5 | 0, 0.1, 0.3 |
| Asian elephant | 120 | 0, 12, 36 | 0, 0.1, 0.3 |
| Crab-eating fox | 41 | 0, 4.1, 12.3 | 0, 0.1, 0.3 |
| Bottlenose dolphin | 35 | 0, 3.5, 10.5 | 0, 0.1, 0.3 |
| Spotted turtle | 31 | 0, 3.1, 9.3 | 0, 0.1, 0.3 |

A hardcoded 25 would be wrong for seven of the nine. **Build the lattice normalized in units of
`K_base`:** `kappa/K_base` on multiples of 0.1, saturating at 1.0 (since `K_max = 2·K_base`), so
`K_eff/K_base ∈ {1.0, 1.1, …, 2.0}` — **11 levels for every population**, derived from
`resolve_actions(cfg)` rather than from a literal. The same scaling applies to the translocation
action `a10`, whose `dN` is 10% of `N0`, not a fixed +20. Discretizer tests must include at least
one non-250 population (Bottlenose dolphin `K_base=35` and Spotted turtle `K_base=31` are the
tightest cases).

**Corollary for PLUS-native's candidate bank — REVISION 3 (2026-07-11): the original corollary here was WRONG, and it cost a build cycle. Recorded in full because the reasoning error is instructive.**

*What I originally wrote:* "the structural quantity a Ricker-form baseline is uncertain about here is **`K`, not `r`** — `r` is action-determined and known ... PLUS-native should keep the same candidate semantics [as adapted PLUS], discretized. A candidate bank spanning *dynamics families* would ... quietly hand PLUS the answer and destroy the point of the experiment."

*Why that is wrong.* The claim about what is **public** is correct (`r` is action-determined, `K_base` comes from `species.csv`). But it does not follow that `K` is what PLUS should be *uncertain* about, because **`K` carries almost no signal in this setting**. The Ricker exponent is `r_pos · (1 − s/K_eff)` with `r_pos = max(r_eff, 0)`, and **6 of the 11 real actions have `r_setpoint ≤ 0`** (a0, a1, a2, a5, a6, a10) — for those, `r_pos = 0` and **`K` cancels out of the dynamics entirely**. For the 5 positive-`r` actions, doubling `K` moves the one-step prediction by 1–3 individuals out of 150. Measured consequence when this was built: PLUS-native's model posterior never left uniform (`[0.143]×7` at every step), all candidates shared one greedy policy, and **PLUS-native produced byte-identical episodes to MOOR-native** (return 5.6558, identical trajectories). MOOR's own fit was degenerate too (`fit_loss = 0.0` exactly, `K_hat` on a tie-break). Within the Ricker form, this env has *nothing left to be uncertain about* — so "average over Ricker parameters" is mathematically the same method as "fit one Ricker".

*The correct reading, and what is now implemented.* The handoff §4 says "**candidate mechanistic models** + posterior/evidence update" — a bank of *models*, which is what the adaptive-management literature and the AAAI21 16-model MOMDP actually do. So:

- **MOOR-native** = the single-form baseline: least-squares Ricker fit, discretize, solve. It carries the "commits to one mechanistic form" handicap.
- **PLUS-native** = a posterior over the **four mechanistic forms** (Ricker / Allee / theta / regime), each discretized and solved independently, with the model posterior updated from `(a, o)` evidence only.

This does **not** hand PLUS the answer, and the fear that it would was the second error in the original corollary. Verified across all 9 populations × both reward modes: the four forms disagree about the optimal action on **3–64% of states** (Iberian lynx 245/385; Spotted turtle is the one cell where they fully agree), so model form genuinely matters — but enumerating the forms does not solve the problem, because the nuisance biology (`C`, `theta`) is continuous and hidden, the regime is latent, and the belief must be discretized. Measured after the fix, PLUS-native and MOOR-native are now distinct methods with the expected ordering (Iberian lynx/allee: MOOR 8.60 vs PLUS 9.93; regime: 8.82 vs 10.46).

This also makes the experiment *more* informative rather than less: if PLUS-native still fails with the true form inside its hypothesis class, that is a much stronger result for the paper than beating a baseline that was never allowed to consider the right model.

**Implementation note:** the regime candidate has a genuinely hidden binary mode, so the native solver's hidden state is `(abundance, regime)` — regime-major flat index `z = g·num_states + s`, with the regime evolving under the cell's `regime_persistence` Markov kernel. Pinning the regime at 0 instead would have been an implementation handicap of exactly the kind §4 forbids. The other three forms have `num_regimes = 1` and collapse to the abundance grid.

---

## 4. Exact code paths

Everything the handoff refers to, located.

### Method registry
- [methods/\_\_init\_\_.py:12-20](../../src/real_ecology_benchmark/methods/__init__.py#L12-L20) — `METHODS = {"mopo":…, "refplan":…, "bamcts":…, "moor":…, "plus":…, "delphic":…, "ogsrl":…}`. The single source of truth; `build_method` looks methods up here by string ([pipeline.py:200](../../src/real_ecology_benchmark/pipeline.py#L200)).
- [methods/base.py:15](../../src/real_ecology_benchmark/methods/base.py#L15) — `BasePolicy`. The lifecycle the handoff calls `reset/act/update/log` is actually **`fit(dataset, beliefs) → reset(seed) → act(belief, observation) → observe(belief, action, PublicTransition)`**, plus `log_training(...)`. Constructor signature: `(env_cfg, model_cfg, planner_cfg, seed=…)`.
- [types.py:139](../../src/real_ecology_benchmark/types.py#L139) — `BeliefPolicy` protocol (the contract the evaluator relies on).

### Manifest / sweep config
- [manifest.py:15-120](../../src/real_ecology_benchmark/manifest.py#L15-L120) — `make_manifest(...)`. It **already accepts** `methods`, `populations`, `families`, `sigmas`, `reward_modes` as arguments; the `data_mode="real"` branch (lines 65–111) writes exactly the reward-mode × population × family × σ × method × filter grid §5 asks for.
- [manifest.py:51,95](../../src/real_ecology_benchmark/manifest.py#L95) — the extra `filter="ricker"` rows are appended **only** for methods literally named `"plus"` and `"moor"`. This is the "adapted PLUS/MOOR" the handoff contrasts against.
- [cli.py:153](../../src/real_ecology_benchmark/cli.py#L153) — `cmd_manifest` exposes **only** `--output` and `--data-mode`. The five filter arguments above are unreachable from the CLI. Restricting the sweep to 5 methods + 2 reward modes therefore needs either a CLI change or a script that calls `make_manifest` directly.
- [scripts/make_real_experiment_manifests.py:30](../../scripts/make_real_experiment_manifests.py#L30) — the real launcher's own `METHODS` tuple, plus `FIELDS` (adds a `job_kind` column) and per-method sharding (`plus` is split out as the slow method, lines 131–132). This is the script the last full run actually used.
- [cli.py:210,215](../../src/real_ecology_benchmark/cli.py#L210) — `--method` `choices=[...]` hardcoded twice; [cli.py:211](../../src/real_ecology_benchmark/cli.py#L211) — `--filter` `choices=["learned","reference","raw","ricker","true_family"]`.

### Filter / policy routing
- [pipeline.py:145-186](../../src/real_ecology_benchmark/pipeline.py#L145-L186) — `make_filter_factory(cfg, dataset, mode)`. **This is the routing switch.** Modes: `reference`, `raw`, `learned`, `ricker`, `true_family`, `oracle`. Each returns `(factory, proposal)` where `factory()` builds a `ParticleFilter`/`RawObservationFilter`/`OracleStateFilter`.
- [pipeline.py:239-322](../../src/real_ecology_benchmark/pipeline.py#L239-L322) — `run_method(method, cfg, filter_mode, regenerate)`. The full row: resolve backend → `ensure_dataset` → `make_filter_factory` → belief cache (`{dataset}.{filter}.beliefs.npz`) → train/holdout split → `build_method` → `ContinuousEvaluator` → save. **`filter_mode` is the only per-row routing knob that exists.**
- [beliefs.py](../../src/real_ecology_benchmark/beliefs.py) — `ParticleFilter`, `RawObservationFilter`, `OracleStateFilter`, `LearnedLinearProposal`, `MechanisticProposal`, `ReferenceProposal`, `BeliefCache`, `cache_dataset_beliefs`.

### Evaluator
- [evaluator.py:20-222](../../src/real_ecology_benchmark/evaluator.py#L20-L222) — `ContinuousEvaluator.run(policy)`. Per episode: `make_env` → `env.reset(seed)` → `filt = self.filter_factory()` → `belief = filt.reset(obs, seed+10_000)` → `policy.reset(seed+20_000)` → loop { `policy.act(belief, observation)` → `env.step(action)` → `policy.observe(belief, action, PublicTransition(...))` → `belief = filt.update(belief, action, obs)` }.
- **Reward-leakage guard is structural and already holds.** `PublicTransition` ([types.py:29-41](../../src/real_ecology_benchmark/types.py#L29-L41)) carries `observation, done, truncated, public_info` and **no reward field**. A native policy physically cannot see the realized reward online. Nothing to add; just do not read `result.reward` (it isn't there) and do not train the belief on `dataset.rewards`.
- [evaluator.py:96-107](../../src/real_ecology_benchmark/evaluator.py#L96-L107) — the `try/except` around `policy.act` swallows `FloatingPointError/ValueError/RuntimeError` into `action = 0` + `fallback_count += 1`. **A native solver that silently degenerates will look like a working policy that chose "Do Nothing" 50 times.** `fallback_count` must be checked in the smoke test (see §7).

### Metrics output
- [evaluator.py:164-220](../../src/real_ecology_benchmark/evaluator.py#L164-L220) — the per-episode row. Contains the entire reward-agnostic battery §6 asks for: `operational_return`, `true_return`, `collapse_entry`, `unsafe_fraction`, `mvp_fraction`, `mvp_breach`, `economic_cost`, `min_true_state`, `final_true_state`, `persistence`, plus filter diagnostics (`filter_rmse`, `filter_log_rmse`, `filter_coverage90`, `filter_ess_fraction`, `filter_unsafe_brier`), plus the cell keys (`population`, `environment`, `sigma_obs`, `reward_mode`, `filter`, `num_actions`, backend metadata).
- [evaluator.py:224-254](../../src/real_ecology_benchmark/evaluator.py#L224-L254) `summarize`, [evaluator.py:256-274](../../src/real_ecology_benchmark/evaluator.py#L256-L274) `save` → writes `episodes.csv` + `summary.json` per row directory.
- [manifest.py:123-280](../../src/real_ecology_benchmark/manifest.py#L123-L280) — `aggregate_summaries`: groups by `(backend, reward_mode, model, filter)`, computes per-seed means, `beats_both_cells`, `beats_both_rate`, and the recoverable/sink split via `realdata.recoverable_population_names()`. **See §5c — this function has a hardcoded assumption that will silently mis-score this experiment.**
- `scripts/extract_report_tables.py`, `scripts/summarize_real_outputs.py` — downstream table builders, both with their own hardcoded `METHODS` lists and `("plus","moor")` baseline sets.

### Reward-mode config
- [config.py:117](../../src/real_ecology_benchmark/config.py#L117) — `EnvironmentConfig.reward_mode: str = "safe"`; validated against `{"yield","safe","observed","belief_expected"}` at [config.py:190](../../src/real_ecology_benchmark/config.py#L190).
- [reward.py:98-109](../../src/real_ecology_benchmark/reward.py#L98-L109) — `effective_collapse_penalty(cfg)`: **`reward_mode="yield"` forces `P = 0` automatically.** "Yield P0" therefore needs no extra config — just `reward_mode: yield`.
- [reward.py:112-117](../../src/real_ecology_benchmark/reward.py#L112-L117) — `build_reward(cfg)`, the single constructor every method uses.
- [config.py:94](../../src/real_ecology_benchmark/config.py#L94) — `collapse_penalty: float = 10.0`. **Not 5.** See §5d.
- [pipeline.py:35-56](../../src/real_ecology_benchmark/pipeline.py#L35-L56) — `_reward_mode_output_root`: outputs are namespaced `…/data_{mode}/backend_{name}/reward_{reward_mode}/{method}/{filter}/`, so a safe run cannot clobber a yield run. Separate agent per mode is already enforced by construction.

### Offline dataset loading
- [dataset.py:13-30](../../src/real_ecology_benchmark/dataset.py#L13-L30) — `PUBLIC_FIELDS` = `observations, actions, rewards, next_observations, dones, episode_id, timestep`; `PUBLIC_CONTROL_FIELDS` = `rho, kappa, K_eff, next_rho, next_kappa, next_K_eff`. `PRIVATE_FIELDS` (truth) live in a separate sidecar file.
- [dataset.py:176-188](../../src/real_ecology_benchmark/dataset.py#L176-L188) — `load_public`; [dataset.py:219](../../src/real_ecology_benchmark/dataset.py#L219) — `assert_public_schema` asserts truth never leaks into the public file.
- [pipeline.py:96-142](../../src/real_ecology_benchmark/pipeline.py#L96-L142) — `ensure_dataset`: caches per-cell, takes a lock file, and calls `_validate_dataset_cell` ([pipeline.py:59](../../src/real_ecology_benchmark/pipeline.py#L59)) which **hard-fails** if the cached dataset's `(kind, population, num_actions, sigma, reward_mode, collapse_penalty, …)` differ from the requested cell.
- [scripts/run_real_manifest_row.py:43-62](../../scripts/run_real_manifest_row.py#L43-L62) — `apply_row_config`: the per-cell dataset path is `{root}/datasets/reward_{mode}/{pop}/{family}/sigma_{σ}/public.npz`. **All methods in a cell already read the exact same file** — the "same offline dataset" guarantee holds by construction. What is missing is the *audit trail*: dataset metadata records an `action_table_hash` ([collector.py:336](../../src/real_ecology_benchmark/collector.py#L336)) but **no content hash of the dataset itself**. Acceptance test §7.3 needs a `dataset_sha256` added.

---

## 5. Conflicts between the handoff and the code

Six real ones. (a)–(c) change the implementation; (d) changes the run config; (e)–(f) are hazards the handoff could not have known about.

### (a) There is no `solver=` axis. The routing knob is `filter=`.
§5 says native baselines should get "a `solver=native_discrete_pomdp` route, **not** a `filter` value." But `filter` is not merely a nominal label — it is the actual per-row routing dimension threaded through the entire stack: manifest column → `run_method(filter_mode)` → `make_filter_factory` → belief-cache filename → `ContinuousEvaluator.filter_label` → output path → the aggregation group key. Adding a parallel `solver` axis means touching the manifest schema, both row runners, the output-path convention, and every aggregation/table script — for zero behavioural gain.

**Recommendation (adopting the handoff's own "map them onto your code" clause):** implement the native belief as a **new filter mode `native_discrete`**, and the solvers as **methods `plus_native` / `moor_native`**. The pairing `(method=plus_native, filter=native_discrete)` expresses precisely what §5 wants — "their own discretized belief, not the shared particle filter" — with a one-branch diff. The intent is preserved; only the name of the knob changes. Flagging it, per §9.5, rather than silently substituting.

### (b) The evaluator **always** builds and steps a filter. A policy cannot opt out.
[evaluator.py:42-48](../../src/real_ecology_benchmark/evaluator.py#L42-L48) constructs `filt = self.filter_factory()` unconditionally, and [evaluator.py:77-95](../../src/real_ecology_benchmark/evaluator.py#L77-L95) computes `filter_rmse`, `filter_coverage90`, `filter_ess_fraction`, `filter_unsafe_brier` from `belief.states` / `belief.weights` on every step. So "native baselines do not use the shared particle filter" cannot be implemented by simply ignoring the belief argument: the shared PF would still be constructed and stepped, and the native rows' `filter_*` metrics would describe **a filter the native method never consulted** — a quietly false number in the results table.

**Recommendation:** implement `DiscreteGridFilter` as a real `BeliefFilter` (`reset`/`update`), returning a `BeliefState` whose `states` are the grid centres and whose `log_weights` are the discrete belief mass. `BeliefState.weights/ess/mean_state()` ([types.py:64-82](../../src/real_ecology_benchmark/types.py#L64-L82)) and the evaluator's quantile code are generic over `(states, weights)` and will work unchanged. The native policy then reads the belief it is handed — which *is* its own discretized belief — and `filter_rmse`/`coverage90`/`unsafe_brier` become honest measurements of the native belief's quality. This is strictly better for the paper: it lets us *show* the discretized belief is worse-calibrated, instead of asserting it.

### (c) **[AUDIT]** `aggregate_summaries` will not compute the headline table — and the fix is bigger than renaming the baselines.
Two independent defects stack here. **The second one is the highest-risk reporting bug in this plan.**

**(c-i) Hardcoded baseline names.** [manifest.py:212-218](../../src/real_ecology_benchmark/manifest.py#L212-L218):
```python
if method in {"plus", "moor"}:
    continue                      # treat as baseline, not challenger
...
baseline = max(seed_means[plus_key], seed_means[moor_key])   # compare against ADAPTED plus/moor
```
With methods named `plus_native`/`moor_native`, they fall through the `continue`, are classified as **challengers**, and get compared against `plus`/`moor` — which are not in this run's method set at all. The same hardcoding recurs in `scripts/extract_report_tables.py:78,88,127`.

**(c-ii) `filter` is part of the comparison key — so a cross-filter comparison is structurally impossible.** [manifest.py:197-202](../../src/real_ecology_benchmark/manifest.py#L197-L202) builds the per-seed key as
`(backend, reward_mode, population, environment, num_actions, sigma_obs, filter, block_seed, model)`,
and [manifest.py:206](../../src/real_ecology_benchmark/manifest.py#L206) defines `scenarios = key[:7]` — **which includes `filter` at index 6.** The baseline lookup at [manifest.py:217-218](../../src/real_ecology_benchmark/manifest.py#L217-L218) is `(*scenario, seed, "plus")`, i.e. *same filter as the challenger*.

This is precisely the comparison the motivation run needs to make: general methods at `filter=learned` versus native baselines at `filter=native_discrete`. Under the current keying they **never share a scenario**, so every baseline lookup misses and `beats_both` yields nothing — the headline table comes out **empty**. Note this means my revision-1 fix (just parameterize the baseline names) would have left the table empty for a *different* reason, and the failure is silent: no exception, no warning, just zero rows.

**Fix:** the paired comparison must key on
`(backend, reward_mode, population, environment, num_actions, sigma_obs, block_seed)` — dropping `filter` from the scenario — and take the filter per *role*, e.g. `aggregate_summaries(root, output, challenger_filter="learned", baselines=(("plus_native","native_discrete"), ("moor_native","native_discrete")))`. Keep the existing same-filter path intact for the previously-published adapted-baseline reports; add the cross-filter path alongside it rather than mutating it.

### (d) "Safe P5, the locked operating point" is **not** what the code defaults to.
`collapse_penalty` defaults to **10.0** ([config.py:94](../../src/real_ecology_benchmark/config.py#L94)), and [configs/real_experiment.yaml](../../configs/real_experiment.yaml) does not set it — so a run launched from the existing config would use P=10, not P=5. The P=5 operating point was selected empirically by the 2026-07-05 `collapse_penalty ∈ {2,5,10,20}` grid, but that result was **never written back into the config defaults**. The motivation run must set `collapse_penalty: 5.0` explicitly in its config. (Yield P=0 is automatic and needs nothing — see §4.)

### (e) `σ = 0` breaks a discretized log-normal emission unless special-cased.
[observation.py](../../src/real_ecology_benchmark/observation.py): at `sigma == 0` the observation is **exact** (`result = values.copy()`), and `log_prob` is a degenerate point mass — `-inf` for every state that is not exactly `o`. A naive discretized emission matrix `O[s', o]` built by evaluating the log-normal density at `sigma=0` will be all-zero/all-`-inf`, the belief update will produce a NaN or an all-zero row, `policy.act` will raise, and the evaluator will silently swallow it into `action = 0` + `fallback_count` (see §4, evaluator). The σ=0 column of the sweep — a quarter of the run — would look like a working "Do Nothing" policy.

**Fix:** special-case `σ = 0` in the discretizer as a deterministic emission (`belief = indicator(bin(o))`), and assert `fallback_count == 0` in the smoke test.

### (g) **[AUDIT]** `make_manifest` cannot express per-method filter routing.
Revision 1 of this document said `make_manifest` "already takes all these arguments; only the CLI/launcher wrapper needs writing." **That was an overclaim, and it is wrong on the axis that matters.** It parameterizes `methods`, `populations`, `families`, `sigmas`, `reward_modes` — but the *filter* axis is a hardcoded literal:

- [manifest.py:78-90](../../src/real_ecology_benchmark/manifest.py#L78-L90) — `for filter_mode in ("learned", "raw")` for **every** method;
- [manifest.py:94-109](../../src/real_ecology_benchmark/manifest.py#L94-L109) — the extra `filter="ricker"` rows, appended only for the literal names `"plus"`/`"moor"`.

So `make_manifest(methods=[…, "plus_native", "moor_native"], data_mode="real")` would emit `learned` **and** `raw` rows for the native baselines and **zero** `native_discrete` rows — not the requested grid, and the native methods would be routed through the shared particle filter, which is the exact thing §4 forbids.

**Fix:** add per-method filter routing — `{refplan, bamcts, ogsrl} → learned`, `{plus_native, moor_native} → native_discrete`, no `raw` rows unless explicitly requested as an ablation. Either extend `make_manifest` with a `method_filters: dict[str, tuple[str, ...]]` argument or write a dedicated motivation manifest generator. `scripts/make_real_experiment_manifests.py` carries its own hardcoded `METHODS` tuple (line 30) and cell-row logic (lines 78–127) and must be updated or bypassed too.

### (h) **[AUDIT]** Build the tabular reward/transition through the existing primitives, not by hand.
The native `T`/`R` tables must mirror repo semantics exactly, or the baseline becomes a straw man by accident:
- Translocation `a10` adds `stocking_delta` **before** growth ([envs.py:212-217](../../src/real_ecology_benchmark/envs.py#L212-L217)), and that delta is 10% of `N0` — population-scaled, like `delta_K`.
- Set-point `r` and `delta_K` are population- *and* family-specific; always source them from `resolve_actions(cfg)` ([actions.py:204-220](../../src/real_ecology_benchmark/actions.py#L204-L220)), never from literals.
- The safety penalty is `safety_penalty_indicator(cfg, previous_state, next_state, crossing)` ([reward.py:75-95](../../src/real_ecology_benchmark/reward.py#L75-L95)). Real cells default to **`occupancy`** (penalty depends on `s'` alone), but `crossing` mode depends on **both** `s` and `s'` — so an `R[s,a]` table folded over `s'` must respect whichever mode the cell configures. Do not assume occupancy.

### (f) Minor: "the current `filter=ricker` particle-MPC adaptation".
Accurate but incomplete. `filter=ricker` is one *ablation* row for `plus`/`moor` ([manifest.py:95](../../src/real_ecology_benchmark/manifest.py#L95)); the *primary* `plus`/`moor` rows run with `filter=learned` and `filter=raw` like everyone else. The "adapted PLUS/MOOR" being contrasted against is the whole `plus`/`moor` method family (which always uses `MechanisticProposal("ricker") + ParticleMPC` internally regardless of the filter column — see [plus.py:35-38](../../src/real_ecology_benchmark/methods/plus.py#L35-L38), [moor.py:83-90](../../src/real_ecology_benchmark/methods/moor.py#L83-L90)). No action needed; noted so the acceptance test in §7.2 ("at least as good as their adapted `filter=ricker` version") compares against the right rows.

### Non-conflicts (checked, and the handoff is right)
- **Eval protocol** 5 seeds × 4 episodes × horizon 50 — matches `EvaluationConfig` defaults ([config.py:473-477](../../src/real_ecology_benchmark/config.py#L473-L477)) and `real_experiment.yaml`. No change.
- **Grid `0 … 2·K_base = K_max`** — verified: `K_max == 2·K_base` for all 9 populations (K_base ranges 31 → 325, so grid size does need to adapt per population, exactly as §4 says).
- **11 actions, real costs, both reward modes on our env** — all present and unchanged by this build.
- **Separate agent per reward mode** — already enforced by the output-path namespacing ([pipeline.py:35](../../src/real_ecology_benchmark/pipeline.py#L35)).

---

## 6. Implementation plan (not yet executed)

**Step 0 — freeze.** Snapshot into the new run directory (`real_ecology_runs/motivation_native_baselines_20260711/code/`) before launching, per the working convention from the P_safe run: the spooled Slurm scripts must point at a frozen tree, not a live-edited one. **[AUDIT]** Snapshot the whole context the P_safe run froze, not just `src/`: `src/real_ecology_benchmark/`, `configs/`, the `scripts/` used by the manifest/Slurm/aggregation path, `real_ecology_data/`, and the run's docs + manifests.

**Step 1 — `discretize.py`.** Per-population state grid (`n_bins` config field; collapse bin at `s ≈ 0`; overflow bin above `K_max`), observation grid, emission matrix with the **σ=0 special case** (§5e), and the `kappa → K_eff` public lattice — **normalized in units of `K_base` (multiples of 0.1, saturating at 1.0 → 11 levels), derived from `resolve_actions(cfg)`, never a hardcoded 25** (§3 [AUDIT]). Pure NumPy. Unit-test that the emission rows sum to 1 for every σ ∈ {0, 0.1, 0.2, 0.4}, **and test at least one non-250 population** (Bottlenose dolphin `K_base=35`, Spotted turtle `K_base=31`).

**Step 2 — `native_solver.py`.** Given a mechanistic model, build `T[a, k, s, s']` (Ricker form, `r_eff = clip(delta_r[a])`, `K_eff` from the public lattice), `O[s', o]`, and `R[s, a] = Σ_{s'} T · r(s', a)` — the reward is a function of the **next** state ([reward.py:52](../../src/real_ecology_benchmark/reward.py#L52)), so it must be folded through `T`. **[AUDIT]** Source the action effects from `resolve_actions(cfg)`, apply `a10`'s stocking delta before growth, and use `safety_penalty_indicator(cfg, …)` for the penalty term — including the fact that `crossing` mode depends on both `s` and `s'` (§5h). Do not hand-code a simplified Ricker reward table. Then QMDP (MDP value iteration + belief-weighted Q) as the first cut, behind a `solver="qmdp"|"belief_vi"` switch. Discrete belief update on `(a, o)` only — never on reward. Note that while `PublicTransition` makes online reward leakage impossible, the *offline* dataset does carry public `rewards` ([dataset.py:13-21](../../src/real_ecology_benchmark/dataset.py#L13-L21)) — keep the native model/evidence fits on `(o, a, o')` only.

**Step 3 — `DiscreteGridFilter`** in `beliefs.py` + the `native_discrete` branch in `make_filter_factory` (§5b).

**Step 4 — `MOORNativePolicy`.** Least-squares Ricker `(r, K)` fit from the offline `(o, a, o')` public data — mirroring the existing grid search at [moor.py:62-97](../../src/real_ecology_benchmark/methods/moor.py#L62-L97) but fitting on **raw observations** rather than shared-PF filtered states (that PF is exactly what native must not use) — then discretize once and solve once in `fit()`.

**Step 5 — `PLUSNativePolicy`.** **[REVISION 3]** Candidates are the **four mechanistic forms** (Ricker / Allee / theta / regime), each discretized and solved independently — *not* a bank over Ricker `K`, which is unidentifiable here and collapses PLUS onto MOOR (see the §3 corollary). Per-candidate discrete belief bank + evidence/posterior update on `(a, o)` only; action = argmax of posterior-weighted candidate Q. The regime candidate carries a hidden binary mode, so its hidden state is `(abundance, regime)`.

**Step 6 — registry + CLI.** Two `METHODS` entries; extend the two `--method` choice lists and the `--filter` choice list in `cli.py`.

**Step 7 — aggregation. [AUDIT] This is the step that decides whether the headline table exists at all.** Add a **cross-filter** paired comparison: drop `filter` from the scenario key and take it per role (`challenger_filter="learned"` vs `baseline_filter="native_discrete"`), *and* parameterize the baseline names. Doing only the latter leaves the table silently empty (§5c-ii). Leave the existing same-filter path untouched for the published adapted-baseline reports. Mirror the change in `scripts/extract_report_tables.py`.

**Step 8 — dataset hash.** Add `dataset_sha256` to dataset metadata in `collector.py`; surface it in `summary.json`. (Acceptance §7.3.)

**Step 9 — run config + manifest. [AUDIT]** New `configs/motivation_native.yaml` (`collapse_penalty: 5.0` — §5d) **plus a separate `configs/motivation_native_smoke.yaml`** (small dataset, 1 seed × 1 episode, coarse grid — the `run` CLI has no evaluation-override flags, so the smoke protocol has to live in a config). Then a manifest generator with **explicit per-method filter routing** (§5g): `make_manifest`'s filter axis is hardcoded and cannot express the general/native split as it stands.

**Step 10 — tests, then smoke, then sweep.** §7 below, then the acceptance battery, then launch.

**Compute note.** The native solvers are pure-NumPy tabular value iteration — CPU-bound, no GPU path. Per the P_safe-run lesson, they belong on the **CPU queue**; do not let them sit behind the GPU fair-share queue.

---

## 7. First smoke test after implementation

The single highest-information cell, chosen because it is the one place where the naive
baseline's Ricker assumption is **correct** — so any failure here is an implementation bug,
not the phenomenon we are trying to measure (this is acceptance test §7.2):

> **Cell:** `population = "Amur tiger"`, `family = ricker`, `σ = 0`, `reward_mode = safe (P=5)`.
> **Method:** `moor_native` (single fit, single solve — the simplest native path). `plus_native` second.
> **Protocol:** 1 seed × 1 episode × horizon 50, coarse grid — **[AUDIT]** carried by `configs/motivation_native_smoke.yaml`, because the `run` CLI exposes only `--method`/`--filter`/`--regenerate` ([cli.py:209-213](../../src/real_ecology_benchmark/cli.py#L209-L213)) and has no flags for seeds, episodes, horizon, dataset size, or grid resolution. A command without that config would silently run the *full* evaluation protocol.

```
PYTHONPATH=src python -m real_ecology_benchmark.cli run \
  --config configs/motivation_native_smoke.yaml \
  --method moor_native --filter native_discrete \
  --population "Amur tiger" --environment ricker --sigma 0 --reward-mode safe
```

**Hard pass conditions — mechanical, all four, or the pipeline is broken:**
1. **`fallback_count == 0`.** Non-negotiable. Anything above zero means `policy.act` is throwing and the evaluator is silently substituting "Do Nothing" (§4/§5e). A run can look completely healthy with a broken solver otherwise.
2. **No NaN/inf** in the belief weights, the emission matrix, or any `episodes.csv` field; every emitted action is a valid id; the emission matrix is neither all-zero nor all-uniform (the σ=0 discretizer failure signature).
3. **`filter_rmse` finite and `filter_coverage90` sane at σ=0** — confirms `DiscreteGridFilter` is producing a real belief and the σ=0 special case works.
4. **The action histogram is not a point mass on `a0`.** This one is justified for *this specific cell*, not generic: Amur tiger's `a0` has `r_setpoint = -0.0458`, so do-nothing is a monotone decline into the safety floor — an all-`a0` policy cannot be optimal under safe mode, and would indicate a degenerate solve.

**Diagnostics, not gates. [AUDIT]** `action_entropy > 0` was a hard condition in revision 1; it is unsound as one. A *constant* policy can be legitimately optimal here (e.g. always `a3`/`a4`, both of which have `r_setpoint > 0` for Amur tiger), and that has zero entropy. Record it; do not fail on it.

**Then, as a separate acceptance test — not the first smoke gate — [AUDIT]:** `operational_return(moor_native) ≥ operational_return(moor, filter=ricker)` on the same cell **under the same full evaluation protocol** (handoff §7.2, the "not crippled" proof). Comparing a 1-episode smoke run against a 20-episode baseline row proves nothing, so this belongs after the smoke passes, at full protocol.

Then repeat at `σ = 0.4` (same cell) before trusting any noisy cell, and once at
`family = allee` to confirm the native path *runs* on a misspecified family (it is expected to
score badly there — that is the result, not a bug).

---

## Summary for the PI

- **Gate: build required.** `PLUS-GPU` is a CuPy device-routing flag ([backend.py:34](../../src/real_ecology_benchmark/backend.py#L34)), not a solver; there is no discretizer, belief grid, or POMDP solver anywhere in the benchmark package, and no POMDP library is installed.
- **Unexpected asset:** a **compiled SARSOP binary + POMDPX writer** already sits in `baseline_original/` from the AAAI21 port. It is bound to the published problem, not our env, so it does not close the gate — but it makes the "add SARSOP later" upgrade path in §4 available with no new dependency.
- **The native model is a MOMDP**, not a flat POMDP: `r` is action-determined, `K_eff` is public and observed, and only the environment's abundance is hidden. This makes the build materially cheaper than §4 assumes, and it fixes what PLUS-native's candidate bank must span (**K**, not r).
- **Things in the handoff that need adjusting before code is written:** the routing knob is `filter=`, not `solver=` (§5a); the evaluator cannot be opted out of, so the native belief must be a real `BeliefFilter` (§5b); the aggregator will not compute the headline table (§5c); `make_manifest` cannot express the general/native filter split (§5g); and **P=5 is not the code default — it is 10** (§5d).
- **Corrections from the codex audit, now folded in.** Three were load-bearing. (1) **My `K_eff` lattice was wrong:** I generalized `{0, 25, 75}` from Amur tiger, but capacity effects are *multipliers* — `dK = {0.1, 0.3}·K_base` — so a hardcoded 25 breaks seven of the nine populations. Use a `K_base`-normalized lattice (§3). (2) **The aggregator keys on `filter` at index 6 of a 7-element scenario tuple**, so `filter=learned` challengers and `filter=native_discrete` baselines can never be paired; renaming the baselines — my original fix — would have left the headline table silently empty for a different reason (§5c-ii). (3) **`make_manifest`'s filter axis is a hardcoded literal**, so my claim that it "already takes all these arguments" was an overclaim on the one axis this run depends on (§5g). The audit also correctly tightened the smoke test (no CLI eval overrides exist; `action_entropy` is not a sound gate) and the freeze scope.
- **No code, config, or run has been touched** — by either agent.
