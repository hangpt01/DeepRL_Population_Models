# Spec: Canonical Documentation Bundle (`docs/benchmark/`)

Date: 2026-07-10
Status: implementation-ready. This document is **self-contained** — you do not need
to read any prior audit or plan to execute it.
Repo root: `/fs04/scratch2/ce25/Claude_DeepRL_Population_Models`

---

## 1. Goal

Produce a small canonical documentation bundle that explains the benchmark **from
zero context**: problem setting, real-ecology data, state/action/reward design,
algorithms and their adaptations, implementation map, experiment protocol, recent
results, and limitations.

After this work, `docs/benchmark/` is the **only** place that describes current
truth. Historical material stays as provenance but is never cited as current.

## 2. Non-negotiable ground rules

### 2.1 Source-of-truth hierarchy (obey while drafting)

1. **Code and CSVs are authoritative, always.** `src/real_ecology_benchmark/*.py`,
   `real_ecology_data/*.csv`.
2. **`docs/architecture.md` is current** (it carries the axes glossary).
3. **Semi-authoritative — re-verify every number and name against code:**
   `docs/29_6_Real_Ecological_Data_Actions_and_Costs.tex`,
   `docs/29_6_REAL_ECOLOGY_IMPLEMENTATION_ALIGNMENT_GUIDE.md`,
   `docs/29_6_Real_Ecology_Setting_Implementation_Plan.tex`.
   **These predate the `data_mode` split, the `setpoint_cumulative` rename, and the
   P-safe decision.** Expect stale names and possibly stale numbers.
4. **Never cited as current:** anything in `docs/history/` or
   `docs/real_ecology_history/`. Mine them for content; attribute nothing.

### 2.2 One source of truth per fact

Never restate the same table in both a `.tex` and a `.md`. The `.md` links to the
`.tex`. This is what stops `01`/`05` and `04`/`08` drifting apart.

### 2.3 Banned terms in the bundle

`Tier-2`, `Tier-3`, `Tier-4`, `claude_build`, `discrete_action_cont_obser`,
"standalone repo", and `real_setpoint` — the last permitted **only** in a single
sentence noting it is a legacy config alias for `setpoint_cumulative`.

In prose, translate the code identifiers:
- `tier2_one_step` → "one-step synthetic control"
- `cumulative_capped` → "cumulative-control synthetic"

### 2.4 Anti-drift footer

Every bundle document ends with:

```
Verified against: <files touched> @ <commit-sha>
```

---

## 3. The single most important requirement

**`04` must state, in its first paragraph and not a footnote, that these are
linear/mechanistic re-implementations of algorithmic *ideas* — not the published
deep-RL systems.**

This is not stylistic. It is what a reviewer attacks first. The verified evidence:

- `pyproject.toml` declares `dependencies = ["numpy>=1.24", "PyYAML>=6.0"]`. There
  is **no `torch`, no `jax`, no `d3rlpy`** anywhere in `src/`.
- `src/real_ecology_benchmark/dynamics.py` implements `LinearDynamicsMember` /
  `ContinuousDynamicsEnsemble`, fitted by **ridge-regularised least squares**
  (`np.linalg.solve(X.T @ X + ridge*I, ...)`) over per-action linear/quadratic
  features.
- `src/real_ecology_benchmark/methods/value.py` (`fit_mechanistic_q`) fits a
  **linear Q by ridge least squares**.
- Planning is **particle MPC** over that learned model
  (`src/real_ecology_benchmark/planning.py`).

Therefore the difference from the literature is **model class**, not
implementation detail. Phrases like "not a byte-for-byte reimplementation" are
wrong and must not appear — the gap is architectural.

Two further disclosures, both verified in code:

- **PLUS** under set-point mode spans candidate **`K`** rather than `r`, because
  `r` is known from the action's set-point (`methods/plus.py:26-37`).
- **MOOR** under set-point mode fits **`K` only**, treating `rho` as the known `r`
  (`methods/moor.py:37-42`).

These are setting-driven adaptations that change **what each baseline is uncertain
about**. They are defensible, but must be disclosed because they materially affect
the comparison.

**Genuine strength to state:** PLUS and MOOR are **closest to original paper
intent**, because their originals are already mechanistic. Say so.

---

## 4. Target layout

```
docs/benchmark/
  README.md                                   reading order; glossary; term-translation
                                              table; source-of-truth hierarchy (§2.1);
                                              one-source-of-truth rule (§2.2); scope (§6)
  preamble.tex                                shared LaTeX preamble so each .tex builds standalone
  01_problem_setting_and_design.tex           POMDP + SCIENTIFIC rationale
  02_real_ecology_data_actions_and_costs.tex  all six CSVs + provenance/citations
  03_state_action_reward_reference.tex        real | dummy | synthetic
  04_algorithm_adaptations_and_claims.tex     per-method, descriptive + disclosure
  05_implementation_and_code_map.md           ENGINEERING rationale + code map
  06_experiment_protocol_and_reproducibility.md
  07_recent_real_ecology_results.md
  08_limitations_tracker.md                   living; aggregate only
```

`.tex` for paper-bound artifacts carrying math (01–04). `.md` for living
repo-facing docs (README, 05–08).

---

## 5. Per-file specification

### `README.md`
Reading order; complete glossary; old-term translation table; the source-of-truth
hierarchy (§2.1); the one-source-of-truth rule (§2.2); the scope statement (§6);
and "what claims this benchmark makes".

**Glossary must define:** `data_mode` (real | dummy | synthetic); `control_mode`
(`setpoint_cumulative` canonical; `tier2_one_step` / `cumulative_capped` = legacy
code identifiers, synthetic only); `reward_mode` (yield | safe); `rho` (public
growth set-point); `kappa` (public capacity accumulator); `K_eff`;
`s_safe` / `safety_threshold`; `mvp_threshold`; collapse; occupancy vs crossing
penalty; public vs evaluator-only (private) info; gate; manifest; cell / row;
recoverable vs sink populations.

**State up front, because it is historically the most confusing fact:**
`control_mode="setpoint_cumulative"` is shared by **real *and* dummy**. The data
source is `data_mode`, a separate axis.

### `01_problem_setting_and_design.tex` — **scientific** rationale
Self-contained POMDP formulation: continuous latent abundance `s ≥ 0` with exact
extinction terminal; log-normal observation model; hidden episode parameters and
optional switching regime; public/private data boundary; dataset format;
evaluation setting; and a real / dummy / synthetic mode table.

Must include a **design-rationale section**: why continuous state; why set-point
`r`; why cumulative `K`; why the occupancy (rather than crossing) safety penalty;
why `collapse_penalty = 5`.

*Sources:* `src/real_ecology_benchmark/{config,envs,controls,dataset,types,observation}.py`;
`docs/architecture.md`; `docs/29_6_Real_Ecology_Setting_Implementation_Plan.tex`
(re-verify); `real_ecology_runs/psafe_overnight_20260705/analysis/DECISION_psafe.md`
(for the `P=5` rationale).

**Do not** derive anything from the discretized-state or synthetic-only history.

### `02_real_ecology_data_actions_and_costs.tex`
Cover **all six CSVs** in `real_ecology_data/` — not just the three the env reads:

```
species.csv              species_lambda.csv       actions.csv
action_effects_long.csv  cost_sources.csv         cost_anchors_portal.csv
```

For a *real-data* claim, the **provenance of the numbers is the crux**: which
papers/portals the λ values came from, how costs were anchored. Include citations.
Also: the population table (9 populations), the 11-action table, set-point `r`,
cumulative `K`, the translocation action (`a10`, the only non-zero
`stocking_delta`), cost scaling, and data caveats/limitations.

*Sources:* `real_ecology_data/*.csv` and its `README.md`;
`src/real_ecology_benchmark/{realdata,actions}.py`;
`docs/29_6_Real_Ecological_Data_Actions_and_Costs.tex` (re-verify).

### `03_state_action_reward_reference.tex`
Canonical reference: state variables; action semantics; reward modes (`yield`,
`safe`); safety thresholds and penalties; MVP / collapse / unsafe-occupancy
metrics; public reward vs evaluator-only truth; and the differences across real,
dummy, and synthetic.

*Sources:* `src/real_ecology_benchmark/{reward,envs,evaluator,gate}.py`;
`docs/29_6_REAL_ECOLOGY_IMPLEMENTATION_ALIGNMENT_GUIDE.md` (re-verify);
`docs/real_ecology_history/reward_state_dependent_change_proposal.md` (mine only).

### `04_algorithm_adaptations_and_claims.tex` — the critical file
Open with the model-class statement from §3. Then, **per method**, a fixed
template:

1. Original paper's idea (3–5 lines).
2. **Original paper's architecture vs. this implementation** (model class,
   optimiser, capacity) — *mandatory field*.
3. What this code implements.
4. What was adapted or simplified, and why.
5. What inputs it may access (public vs evaluator-only).
6. How it acts at evaluation time.
7. What we may and may not claim.

Include this identifier mapping table — the CLI and configs use the bare names, so
without it a reader hits `--method mopo` and re-forms the exact assumption this
file exists to prevent:

| Code identifier | Reader-facing name | Original paper | Model class here |
| --- | --- | --- | --- |
| `mopo` | MOPO-style | Yu et al., MOPO | ridge-linear ensemble + pessimistic particle MPC |
| `refplan` | RefPlan-inspired | RefPlan | posterior ensemble planner |
| `bamcts` | BA-MCTS-inspired | BA-MCTS | finite-action belief tree |
| `plus` | PLUS | BioConserv18 PLUS | mechanistic Ricker bank (candidate `K`) |
| `moor` | MOOR | ExpertSys23 MOOR | single misspecified Ricker (fits `K`) |
| `delphic` | Delphic-CQL-inspired | Delphic offline RL | linear/ridge compatible-world adaptation |
| `ogsrl` | OGSRL-inspired | OGSRL | guardian/objective adaptation |

**Verify every "original paper" cell** against
`docs/real_ecology_history/29_6_algorithm_method_notes.tex` before publishing. Do
not take the citations in this spec on trust.

This file is **descriptive and per-method**. Aggregate limitations belong in `08`.

*Sources:* `src/real_ecology_benchmark/methods/*.py`;
`src/real_ecology_benchmark/{planning,beliefs,dynamics,pipeline}.py`;
`docs/real_ecology_history/29_6_algorithm_method_notes.tex`;
`docs/real_ecology_history/AUDIT_algorithm_paper_implementation.md`;
`docs/methods.md`.

### `05_implementation_and_code_map.md` — **engineering** rationale
Two halves, and it must not repeat `01`'s scientific rationale.

**(a) Engineering rationale:** why `data_mode` is orthogonal to `control_mode`;
why the public/private data boundary exists; why the dataset-cache validation keys
are what they are; why the compute-backend workload matrix refuses to mislabel a
NumPy run as GPU.

**(b) Code map** — concept → file:
config loading (`config.py`) · real data loader (`realdata.py`) · dummy data
(`dummydata.py`) · actions (`actions.py`) · public controls (`controls.py`) ·
env transition (`envs.py`) · reward (`reward.py`) · dataset public/private split
(`dataset.py`, `collector.py`) · observation model (`observation.py`) · belief
filters (`beliefs.py`) · learned dynamics (`dynamics.py`) · planner
(`planning.py`) · methods (`methods/*.py`) · pipeline & CLI
(`pipeline.py`, `cli.py`) · evaluation & gate (`evaluator.py`, `gate.py`) ·
manifests (`manifest.py`, `scripts/`, `scripts/slurm/`) · compute backend
(`backend.py`) · tests (`tests/{real,dummy,synthetic}/`).

### `06_experiment_protocol_and_reproducibility.md`
Concrete run protocol from the **repo root**:

```bash
PYTHONPATH=src python -m unittest discover -s tests      # 93 tests
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/dummy_setpoint_smoke.yaml
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/synthetic_default.yaml
```

Cover: the `generate → run → gate → aggregate` flow; `--data-mode` override; the
config inventory (`real_*`, `dummy_setpoint_*`, `synthetic_*`,
`cumulative_controls_12h`); manifest generation and Slurm rows; output layout and
its `backend_*/reward_*/data_*` namespacing; seeds; and the provenance bundle
location `real_ecology_runs/psafe_overnight_20260705/`.

*Sources:* `docs/experiment_protocol.md`, `docs/reproducibility.md`, `Makefile`,
root `README.md`, `scripts/`.

### `07_recent_real_ecology_results.md`
Summarise the P-safe experiment: the chosen penalty and **why** (`DECISION_psafe.md`);
headline method comparison; learned-vs-raw filter ablation; and links to
`REAL_ECOLOGY_EXPERIMENT_RESULTS.pdf` plus key figures.

**These honest findings are mandatory** — a reviewer will otherwise find them first:
- Most real populations are **robust to collapse** under the current setting; the
  Amur tiger is the main population reaching the collapse band. This is a
  negative/robustness result and must not be buried.
- The **two demographic sinks** (`r_max_ricker ≤ 0`; see
  `realdata.sink_population_names()`) are excluded from the headline collapse
  metric and reported separately. State this as a methodological choice, with its
  justification.

*Sources (all exist, verified):*
`real_ecology_runs/psafe_overnight_20260705/analysis/` →
`PAPER_RESULT_PACKAGE.md`, `RESULTS_SUMMARY.md`, `DECISION_psafe.md`,
`P5_control_review.md`, `learned_vs_raw_p5.md`, `all_metrics.csv`, `rollup.csv`,
`convergence_note.txt`, `figures/`, `report_figures/`,
`REAL_ECOLOGY_EXPERIMENT_RESULTS.{tex,pdf}`, `trajectory_demo_20260705/`.

### `08_limitations_tracker.md`
Living, reviewer-facing. **Aggregate only — no per-method restatement**; link to
`04`. Cover: method-adaptation limits (the model-class gap); data caveats; the
robustness caveat; the sink-population exclusion; and what future work would
strengthen the claims.

---

## 6. Scope decision (make it once, state it in the README)

`07` covers the **real-ecology** setting. Decide and record whether the bundle also
summarises the **synthetic** stress-benchmark results. A supervisor will ask.

This is the same decision as the disposition of `22_6_Continuous_Observation_*.tex`
in §7 — those two files *are* the synthetic reports. Either add a one-paragraph
scope statement in the README pointing at them, or give `07` a short synthetic
section.

---

## 7. Dispositions for every loose file at `docs/` root

There are **eleven**. Each needs a destination. Do not leave any as competing
current truth.

| File | Role | Disposition |
| --- | --- | --- |
| `architecture.md` | axes glossary → bundle README / `01` | fold, then pointer stub or `history/` |
| `methods.md` | 31 lines → `04` | fold, then retire |
| `experiment_protocol.md` | → `06` | fold, then retire |
| `reproducibility.md` | → `06` | fold, then retire |
| `29_6_Real_Ecology_Setting_Implementation_Plan.tex` | source for `01` / `05` | → `history/` after distilling |
| `29_6_Real_Ecological_Data_Actions_and_Costs.tex` | source for `02` | → `history/` |
| `29_6_REAL_ECOLOGY_IMPLEMENTATION_ALIGNMENT_GUIDE.md` | source for `03` / `05` | → `history/` |
| `22_6_Continuous_Observation_New_Baselines.tex` | synthetic-benchmark report | → `history/` or `references/` (see §6) |
| `22_6_Continuous_Observation_Experiment_Results.tex` | synthetic-benchmark report | → same |
| `AUDIT_documentation_bundle_plan_2026-07-10.md` | process doc | → `history/` |
| `SPEC_documentation_bundle_2026-07-10.md` (this file) | process doc | → `history/` when done |

`docs/README.md` currently asserts "Current reference material lives in this
directory." That becomes false — **update it** to point at `docs/benchmark/` first.
Also update the root `README.md`.

Untouched: `docs/history/`, `docs/real_ecology_history/`, `docs/references/`,
`docs/claude_build_report/`, `docs/plots_selected/`.

---

## 8. Tooling

Add **both** targets — a linter is not a compiler:

- **`make docs`** — compiles every `.tex` in `docs/benchmark/`. They must build
  standalone via the shared `preamble.tex`. An uncompilable `.tex` in the repo is
  worse than none.
- **`make docs-check`** — lints: banned terms (§2.3), broken relative paths, and
  missing `Verified against:` footers.

---

## 9. Implementation order (one commit each)

1. Skeleton: create `docs/benchmark/`, `README.md`, `preamble.tex`, and the two
   `make` targets. No prose yet.
2. Write `01`–`04` (`.tex`), verifying every number against code and CSVs.
3. Write `05`–`08` (`.md`).
4. Dispositions (§7): fold and move the eleven root files; update `docs/README.md`
   and root `README.md`.
5. Verify (§10).

---

## 10. Acceptance checks

- `docs/` root contains **only** `README.md`, `benchmark/`, and the existing
  subdirectories.
- `make docs` compiles every `.tex`; `make docs-check` passes.
- Banned-term sweep returns nothing but the single legacy-alias sentence:
  ```bash
  rg -n "Tier-?[234]|real_setpoint|claude_build|discrete_action_cont_obser|standalone repo" docs/benchmark/
  ```
- Every bundle document ends with `Verified against: <files> @ <commit-sha>`.
- No table appears in both a `.tex` and a `.md`.
- `04` states the model-class distinction in its **first paragraph**.
- The code still passes: `PYTHONPATH=src python -m unittest discover -s tests` → 93 OK.
- Constants quoted in the docs match code: `NUM_REAL_ACTIONS == 11`,
  `NUM_REAL_POPULATIONS == 9`, seven methods, nine populations.

---

## 11. Do NOT

- Do not modify anything under `real_ecology_runs/psafe_overnight_20260705/` — it is
  a frozen provenance bundle.
- Do not cite `docs/history/` or `docs/real_ecology_history/` as current truth.
- Do not copy numbers from the `29_6_*` specs without re-verifying against code —
  they predate the `data_mode` split, the `setpoint_cumulative` rename, and the
  P-safe decision.
- Do not restate the same table in a `.tex` and a `.md`.
- Do not describe the methods as reproductions of the published deep-RL systems, or
  use the phrase "byte-for-byte". The gap is **model class**.
- Do not leave `docs/architecture.md`, `methods.md`, `experiment_protocol.md`, or
  `reproducibility.md` at `docs/` root as a second "current" explanation.
