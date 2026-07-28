# Server Audit - Native-Baseline Gate Context

**Date:** 2026-07-11
**Audits:** `docs/benchmark/SERVER_CONTEXT_motivation_native_baselines_gate.md`
**Intended reader:** Claude / next implementation agent
**Scope:** read-only code inspection plus this audit document. No code, config, or run was changed.

## Bottom Line

Claude's main gate result is correct: **native discrete-POMDP PLUS/MOOR does not exist in `src/real_ecology_benchmark`; build required.** `PLUS-GPU` is backend routing, not solver code, and the vendored SARSOP/POMDPX stack under `baseline_original/` is real but not wired to this benchmark.

Do **not** implement straight from Claude's file without the corrections below. The largest implementation risks are:

1. the manifest cannot currently express the requested method/filter routing;
2. the aggregator cannot compare `filter=learned` general methods against `filter=native_discrete` baselines;
3. the public `K_eff` lattice is per-population scaled, not an absolute 25-unit lattice.

## Confirmed Accurate

### Gate verdict

Correct. The active package has no native PLUS/MOOR solver registry entries:

- `src/real_ecology_benchmark/methods/__init__.py:12-20` registers only `mopo`, `refplan`, `bamcts`, `moor`, `plus`, `delphic`, `ogsrl`.
- `src/real_ecology_benchmark/backend.py:28-34` defines GPU-supported workload labels, including `PLUS_GPU_WORKLOADS = {"method:plus", "oracle_ablation:plus"}`.
- `src/real_ecology_benchmark/backend.py:136-145` maps PLUS rows to the `"plus_mechanistic_transition"` CuPy acceleration scope, not to a POMDP solver.

The active package only depends on NumPy/PyYAML in `pyproject.toml:13`. A direct import check found no installed `pomdp_py`, `pomdpy`, `pypomdp`, `sarsop`, or `julia`.

### SARSOP asset

Correct. There is a compiled solver asset:

- `baseline_original/sarsop/src/pomdpsol` exists, executable, ELF x86-64, about 17 MB.
- `baseline_original/python_port/src/building hmMDP/generate_pomdpx.py:11-84` writes the historical hmMDP POMDPX format.
- `baseline_original/python_port/src/solving hmMDP/main.py:18-53` resolves and calls the SARSOP binary.
- `baseline_original/python_port/src/simulations/read_policyx.py:20-53` reads policyx alpha vectors and updates beliefs.

Claude is also right that this does **not** close the gate: that stack targets the published hmMDP/MOMDP problem, not this repo's 11-action real-ecology environment, reward modes, public controls, or current evaluator.

### Routing and evaluator shape

Correct that `filter` is currently the routing dimension:

- `src/real_ecology_benchmark/pipeline.py:145-186` dispatches filter modes in `make_filter_factory`.
- `src/real_ecology_benchmark/pipeline.py:239-322` threads `filter_mode` through dataset loading, belief cache path, evaluator, and output path.
- `src/real_ecology_benchmark/evaluator.py:42-49` always constructs a filter before policy evaluation.
- `src/real_ecology_benchmark/evaluator.py:77-95` computes filter diagnostics from the returned `BeliefState`.

So implementing native belief as a real `BeliefFilter` is the right direction.

### P5 conflict

Correct. The code default is still `collapse_penalty = 10.0` in `src/real_ecology_benchmark/config.py:92-94`, and `configs/real_experiment.yaml` does not override it. Historical P_safe artifacts picked 5, but that was not made the live default. A motivation config must set `environment.collapse_penalty: 5.0` explicitly.

## Corrections Required Before Coding

### 1. Existing manifest generation does not produce the requested run grid

Claude says `make_manifest(...)` already accepts the needed axes, but the current real branch always writes:

- `filter in ("learned", "raw")` for **every** method at `src/real_ecology_benchmark/manifest.py:78-90`;
- extra `filter="ricker"` rows only for literal methods `"plus"` and `"moor"` at `manifest.py:94-109`.

If you call `make_manifest(methods=["refplan","bamcts","ogsrl","plus_native","moor_native"], data_mode="real")`, it will create `learned` and `raw` rows for `plus_native`/`moor_native`, and it will not create `native_discrete` rows. That is not the handoff's requested grid.

Required implementation change:

- add a dedicated motivation manifest generator or extend `make_manifest` with per-method filter routing, e.g.
  - `refplan`, `bamcts`, `ogsrl` -> `filter=learned`;
  - `plus_native`, `moor_native` -> `filter=native_discrete`;
  - no `raw` rows unless explicitly requested for an ablation.

The standalone launcher script also has its own hardcoded method/filter logic in `scripts/make_real_experiment_manifests.py:30` and `_cell_rows` at lines 78-127, so update or bypass that script too.

### 2. Parameterizing the baseline method names is not enough for aggregation

Claude correctly found the hardcoded baseline names in `aggregate_summaries`, but the proposed fix is incomplete.

Current per-seed comparison key includes `filter`:

- `src/real_ecology_benchmark/manifest.py:197-202` builds keys as `(backend, reward_mode, population, environment, num_actions, sigma_obs, filter, block_seed, model)`.
- `manifest.py:206` defines scenarios as `key[:7]`, so `filter` is part of the scenario.
- `manifest.py:217-218` looks up `"plus"` and `"moor"` under the **same filter** as the challenger.

In the motivation run, general methods will be `filter=learned`, while native baselines should be `filter=native_discrete`. They will never share the same scenario key. Even after changing baseline names to `("plus_native","moor_native")`, the comparison will still be empty unless the filter dimension is handled explicitly.

Required implementation change:

- add a motivation-specific paired comparison that maps challenger rows and baseline rows by `(backend, reward_mode, population, environment, num_actions, sigma_obs, block_seed)` while allowing different filters;
- or add aggregation arguments such as `challenger_filter="learned"` and `baseline_filter="native_discrete"`;
- keep the old same-filter comparison for the previous adapted-baseline reports.

This is the highest-risk reporting bug in Claude's plan.

### 3. `K_eff` is not on an absolute 25-unit lattice

Claude's "25-unit lattice" is only true for populations with `K_base = 250` (Amur tiger and Puerto Rican parrot). The live action table scales capacity effects by population:

- `real_ecology_data/actions.csv:7-11` defines capacity multipliers `1.1` and `1.3`.
- `real_ecology_data/action_effects_long.csv:7-10` shows Egyptian vulture `dK_step` values `32.5` and `97.5` for `K_base=325`.
- `real_ecology_data/action_effects_long.csv:18-21` shows bottlenose dolphin `dK_step` values `3.5` and `10.5` for `K_base=35`.
- `real_ecology_data/action_effects_long.csv:29-32` shows Amur tiger `25.0` and `75.0` for `K_base=250`.
- `real_ecology_data/species.csv:2-10` confirms `K_max = 2 * K_base` for all nine populations.

Required implementation change:

- derive the public `K_eff`/`kappa` grid from `resolve_actions(cfg)` or `real_ecology_data/action_effects_long.csv`, not from a hardcoded 25;
- a normalized lattice in units of `K_base` is probably the cleanest representation: `K_eff/K_base` spans `1.0 ... 2.0` with increments generated by action effects (`0.1`, `0.3`, saturation);
- include non-250 populations in the discretizer tests, e.g. Bottlenose dolphin or Spotted turtle.

### 4. PLUS-native has hidden candidate state in addition to abundance

Claude's "only abundance is hidden" simplification is good for MOOR-native's single assumed model, but incomplete for PLUS-native. If PLUS-native keeps a candidate bank over `K`, its internal belief includes:

- discrete abundance belief;
- posterior/evidence over candidate `K` models;
- optionally a per-candidate abundance belief bank, analogous to `PLUSPolicy.bank` in `src/real_ecology_benchmark/methods/plus.py:62-163`.

`kappa`/`K_eff` is public and should not be hidden, but the candidate parameter for PLUS-native is hidden model uncertainty. Do not collapse PLUS-native into a single abundance-only filter unless you intentionally give up the PLUS posterior mechanism.

### 5. The smoke command does not actually enforce the stated smoke protocol

Claude's smoke text says "1 seed x 1 episode x horizon 50, coarse grid", but the example CLI command has no overrides for seeds, episodes, horizon, dataset size, or grid resolution. The current CLI exposes no evaluation override flags:

- `src/real_ecology_benchmark/cli.py:209-213` only exposes `--method`, `--filter`, and `--regenerate` for `run`.

Required implementation change:

- create a separate `configs/motivation_native_smoke.yaml`, or add CLI overrides for evaluation/dataset/native-grid settings;
- make the smoke command point at the smoke config explicitly.

Also downgrade `action_entropy > 0` from a hard pass condition to a diagnostic unless you first prove the target cell's optimal policy should be non-degenerate. `fallback_count == 0`, finite filter metrics, no NaN/inf rows, and a valid action id every step should be hard smoke checks. The adapted-baseline comparison is better as a follow-up acceptance test with the same evaluation protocol, not the first smoke gate.

### 6. Reward and transition details need to be more explicit in the implementation plan

Claude correctly notes reward is on next state, but the tabular native solver must exactly mirror several repo details:

- Direct translocation action `a10` adds `stocking_delta` before growth: see `src/real_ecology_benchmark/envs.py:212-217` and `actions.py:128-135`.
- Real set-point actions are population/family-specific from `resolve_actions(cfg)` at `actions.py:204-220`.
- Reward uses `safety_penalty_indicator(cfg, previous_state, next_state, crossing)` in `reward.py:75-95`, with default real mode `occupancy` but an available `crossing` mode.
- If the solver constructs `R[s,a]`, it should fold over `s'` and use the same penalty semantics. For `occupancy`, penalty depends on `s'`; for `crossing`, it depends on both `s` and `s'`.

In other words, do not hand-code a simplified Ricker reward table. Build it through the same action specs, safety indicator semantics, and next-state transition table.

## Additional Notes

### `sigma=0` hazard is real, but phrase it as a discretizer issue

The existing continuous observation model already special-cases zero noise:

- `src/real_ecology_benchmark/observation.py:23-27` samples exact observations at `sigma=0`.
- `observation.py:42-47` gives a point-mass log probability with tolerance.
- `src/real_ecology_benchmark/beliefs.py:460-466` and `beliefs.py:520-531` handle exact-observation particle filtering.

The new native discretizer still needs its own deterministic emission/binning logic. The failure mode is not in the existing observation model; it is in a naive grid emission matrix that has no bin containing the exact observation.

### Reward leakage wording

Claude is right that `PublicTransition` has no reward field (`src/real_ecology_benchmark/types.py:29-41`) and online `policy.observe(...)` cannot receive realized reward. But offline datasets do contain public `rewards` (`dataset.py:13-21`). For the native baselines, keep belief/model evidence updates on `(o, a, o')` only, and use the configured reward model for planning unless the experiment explicitly allows using logged rewards as public offline reward labels.

### Freeze/snapshot step

Claude's "freeze `src/` into the run directory" matches the previous run convention, but do not implement it as an ad hoc copy of only `src/`. The frozen P_safe run included code plus real data/config context. For motivation runs, snapshot at least:

- `src/real_ecology_benchmark/`;
- `configs/`;
- `scripts/` used by manifests/Slurm/aggregation;
- `real_ecology_data/`;
- the new docs and manifest files.

## Recommended Revised Implementation Order

1. Add `DiscreteGridFilter` and native discretizer primitives, deriving action effects from `resolve_actions(cfg)`.
2. Add `native_solver.py` with exact transition/reward table semantics, including `sigma=0` emission tests and non-250 population tests.
3. Add `moor_native` and `plus_native` method classes. Keep MOOR's fitted model single-Ricker; keep PLUS's posterior over candidate `K` models.
4. Register methods and add `native_discrete` routing in `make_filter_factory`.
5. Add a motivation-specific manifest generator with explicit per-method filters.
6. Add a motivation-specific aggregation path that compares `learned` general rows against `native_discrete` baseline rows.
7. Add `dataset_sha256` metadata/logging.
8. Add `configs/motivation_native.yaml` with `collapse_penalty: 5.0` and a separate smoke config.
9. Run the first smoke: one Ricker cell, `sigma=0`, `moor_native`, `filter=native_discrete`, smoke config. Hard checks: `fallback_count == 0`, finite metrics, finite belief weights, valid actions, no all-zero/all-uniform emission bug.
10. Then run the adapted-baseline comparison on the same cell/protocol before launching the full sweep.

## Final Audit Verdict

Use Claude's gate document as a strong discovery pass, not as a final implementation spec. The build-required verdict, PLUS-GPU interpretation, SARSOP asset discovery, P5 config conflict, evaluator-filter requirement, and `sigma=0` warning are all valuable. The next agent must fix the manifest/aggregation/filter comparison design and the per-population `K_eff` lattice before writing solver code.
