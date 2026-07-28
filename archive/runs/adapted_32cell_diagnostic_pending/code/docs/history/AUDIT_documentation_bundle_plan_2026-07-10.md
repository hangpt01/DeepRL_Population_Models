# Audit: Canonical Documentation Bundle Plan

Auditor: Claude
Date: 2026-07-10
Audited: codex's proposed `docs/current_benchmark/` documentation plan
Checked against: live code in `src/real_ecology_benchmark/`, `real_ecology_data/*.csv`,
`pyproject.toml`, and `real_ecology_runs/psafe_overnight_20260705/analysis/`.

---

## Verdict

**The plan is right in principle and should proceed.** The diagnosis is correct:
the docs are organised but not self-contained, and a new canonical reader-facing
layer beats more shuffling of historical files.

Two things must change before implementation:

1. **The algorithm-adaptation framing is dangerously understated.** The codebase
   has **no deep-learning dependency at all** (`dependencies = ["numpy>=1.24",
   "PyYAML>=6.0"]`). `dynamics.py` is a **ridge-regularised linear ensemble**;
   `methods/value.py` fits a **linear Q by least squares**. So MOPO, Delphic-CQL,
   OGSRL, RefPlan and BA-MCTS here are *not* the deep-RL originals — they are
   linear/mechanistic analogues in the same algorithmic family. Codex's phrase
   "not byte-for-byte reimplementations" is far too soft; the difference is
   **model class**, not bytes. This is the single highest-stakes item in the whole
   bundle and it is what a reviewer will attack first. (Finding F1.)

2. **The plan creates a second source of truth.** It says "do not delete old docs,"
   but `docs/architecture.md`, `methods.md`, `experiment_protocol.md`, and
   `reproducibility.md` sit at `docs/` root as *current reference*. Adding a new
   canonical set without retiring or redirecting those recreates exactly the drift
   we just spent three refactors cleaning up. (Finding F2.)

Everything else is refinement.

---

## What checks out (verified, no action needed)

- **All ten cited source documents exist**, including
  `docs/real_ecology_history/29_6_algorithm_method_notes.tex` and
  `AUDIT_algorithm_paper_implementation.md`.
- **All cited result artifacts exist**: `PAPER_RESULT_PACKAGE.md`,
  `RESULTS_SUMMARY.md`, `REAL_ECOLOGY_EXPERIMENT_RESULTS.{tex,pdf}`,
  `report_figures/` **and** `figures/`. (I initially suspected the
  `report_figures/` path was wrong — it is not.)
- **`docs/references/` exists** as the plan assumes.
- The seven methods named in the plan match the code exactly: `mopo`, `refplan`,
  `bamcts`, `plus`, `moor`, `delphic`, `ogsrl`.

### The screenshot files: nothing was lost

Four of the seven files in the user's screenshot —
`29_6_Algorithm_Paper_Audit_Guide.tex`, `Baselines_State_Action_Reward_Reference.tex`,
`CHAT_HANDOFF_real_ecology_results_and_algorithm.md`,
`LIMITATIONS_tracker_real_ecology.md` — **do not exist in this repo and were never
tracked in git history**. They are a template of the *kind* of docs wanted, not
files lost during cleanup. Confirmed via `git log --all --diff-filter=D`.

Two of them imply doc types the plan should adopt (see F4).

---

## Answers to the plan's seven audit questions

### Q1 — Is the file set sufficient for a zero-context reader?

**Nearly. Four gaps:**

- **No living limitations tracker.** Doc `05` is a claims-boundary *snapshot*; the
  screenshot's `LIMITATIONS_tracker_real_ecology.md` is a *living list* that gets
  updated as limitations are found and closed. These serve different purposes.
  Add `09_limitations_tracker.md`, or make explicit that `05` is that tracker.
- **No design rationale.** The user asked for an *implementation plan*; codex
  substituted an *implementation map* (a code-reading guide). Both are useful, but
  neither answers **"why is the setting built this way?"** — why set-point `r`, why
  cumulative `K`, why occupancy rather than crossing penalty, why `collapse_penalty
  = 5`. That is the first question a supervisor asks. Distil it from
  `docs/29_6_Real_Ecology_Setting_Implementation_Plan.tex` and
  `analysis/DECISION_psafe.md` into a rationale section of `01` (or its own doc).
- **Data provenance is under-specified.** `02` says "source CSV files," but
  `real_ecology_data/` has six: `species.csv`, `species_lambda.csv`, `actions.csv`,
  `action_effects_long.csv`, `cost_sources.csv`, `cost_anchors_portal.csv`. For a
  *real-data* claim, the provenance of the numbers (which papers/portals λ came
  from, how costs were anchored) is the crux. `02` must cover all six **with
  citations**, not just the three the env reads.
- **Glossary lives only inside the README.** Acceptable, but it must be complete
  (see Q5) and it must state the historically most confusing fact up front:
  `control_mode="setpoint_cumulative"` is shared by **real *and* dummy**; the data
  source is `data_mode`.

### Q2 — Which existing docs are authoritative?

Enforce this hierarchy, and say so in the bundle README:

1. **Code and CSVs are authoritative, always.** `config.py`, `envs.py`,
   `controls.py`, `actions.py`, `reward.py`, `realdata.py`, `dummydata.py`,
   `methods/*`, and `real_ecology_data/*.csv`.
2. **`docs/architecture.md`** — currently accurate (it carries the axes glossary).
3. **Semi-authoritative, must be re-verified line by line:**
   `29_6_Real_Ecological_Data_Actions_and_Costs.tex`,
   `29_6_REAL_ECOLOGY_IMPLEMENTATION_ALIGNMENT_GUIDE.md`,
   `29_6_Real_Ecology_Setting_Implementation_Plan.tex`. **These were written before
   the `data_mode` split, the `setpoint_cumulative` rename, and the P-safe
   decision.** Expect stale names (`real_setpoint`, "Tier-4") and possibly stale
   numbers.
4. **Never cite as current:** anything in `docs/history/` or
   `docs/real_ecology_history/`. Mine them for content; attribute nothing.

`docs/methods.md` is only **31 lines** (architecture 84, experiment_protocol 29,
reproducibility 23 — 167 lines total). These are too thin to be sources for `04`.
Write `04` from the **code** plus `29_6_algorithm_method_notes.tex`.

### Q3 — Are any proposed docs redundant?

- **`04` vs `05` overlap.** `04`'s per-method "what was adapted/simplified" is the
  same content as `05`'s "which are simplified." Enforce a clean split:
  **`04` = descriptive** (what each method *is* and does in this code);
  **`05` = normative** (what we may and may not claim, plus limitations).
- **`07` vs the existing `experiment_protocol.md` + `reproducibility.md`** — pure
  duplication. Same for **`04` vs `methods.md`** and **`01`/`06` vs
  `architecture.md`**. See F2: these must be retired or reduced to pointers.

### Q4 — Missing experiment artifacts?

`08`'s source list should add: `DECISION_psafe.md` (the *why* behind P=5),
`P5_control_review.md`, `rollup.csv`, `all_metrics.csv`, `convergence_note.txt`,
`manifests/`, and `trajectory_demo_20260705/`.

Two **honest findings** must appear in `08` or a reviewer will find them first:

- Most real populations are **robust to collapse** under the current setting (only
  the Amur tiger reaches the collapse band; others hold λ ≥ 0.86 under `a2`). This
  is a negative/robustness result and must not be buried.
- The **two demographic sinks** (`r_max_ricker ≤ 0`) are excluded from the headline
  collapse metric and reported separately — `realdata.sink_population_names()`.
  State this explicitly as a methodological choice, with its justification.

Also decide: does the bundle mention the **synthetic stress benchmark** result
(baselines failing under collapse-sensitive settings)? A supervisor will ask "what
about the synthetic results?" Either include a short section or say explicitly that
the bundle scopes to real ecology.

### Q5 — Which old terms must be banned or translated?

**Ban outright in the new bundle:** `Tier-2`, `Tier-3`, `Tier-4`, `claude_build`,
`discrete_action_cont_obser`, "standalone repo", and `real_setpoint` (permitted
only in one sentence describing the legacy alias).

**In prose, translate the code identifiers:**
`tier2_one_step` → "one-step synthetic control"; `cumulative_capped` →
"cumulative-control synthetic".

**Must be defined in the glossary:** `data_mode` (real | dummy | synthetic);
`control_mode` (`setpoint_cumulative` canonical; the two synthetic identifiers);
`reward_mode` (yield | safe); `rho` (public growth set-point); `kappa` (public
capacity accumulator); `K_eff`; `s_safe`/`safety_threshold`; MVP threshold;
collapse; occupancy vs crossing penalty; public vs evaluator-only (private) info;
gate; manifest; cell/row; recoverable vs sink populations.

### Q6 — Are the algorithm adaptation claims accurate relative to code? **(F1)**

**No — and this is the most important finding of the audit.** What the code
actually does:

- **`pyproject.toml`: `dependencies = ["numpy>=1.24", "PyYAML>=6.0"]`.** There is no
  `torch`, no `jax`, no `d3rlpy` anywhere in `src/`.
- **`dynamics.py`**: `LinearDynamicsMember` / `ContinuousDynamicsEnsemble`, fitted by
  ridge-regularised least squares (`np.linalg.solve(X.T @ X + ridge*I, ...)`) over
  per-action linear/quadratic features.
- **`methods/value.py`**: `fit_mechanistic_q` fits **linear Q-weights** by ridge
  least squares.
- Planning is **particle MPC** over that learned model.

Consequences that `04` and `05` must state plainly:

- **MOPO** in the literature is a *neural* probabilistic ensemble plus SAC. Here it
  is a ridge-linear ensemble plus particle MPC. Same *idea* (uncertainty-penalised
  model-based rollouts); different **model class**.
- **Delphic-CQL / OGSRL / BA-MCTS / RefPlan** are likewise linear/mechanistic
  analogues, not the deep implementations.
- **PLUS**, under set-point mode, spans candidate **`K`** rather than `r`, because
  `r` is known from the action set-point (`methods/plus.py:26-37`).
- **MOOR**, under set-point mode, fits **`K` only**, treating `rho` as the known `r`
  (`methods/moor.py:37-42`).

The last two are *setting-driven* adaptations that change **what each baseline is
uncertain about** — defensible, but they must be disclosed, because they materially
affect the comparison.

**Required change:** give `04` a mandatory per-method field —
*"Original paper's architecture vs. this implementation (model class, optimiser,
capacity)."* And replace `05`'s soft framing with something like:

> These are **re-implementations of the algorithmic ideas in a linear/mechanistic
> model class**, not reproductions of the original deep-RL architectures. Results
> should be read as a comparison of *strategies* (pessimism, belief-conditioned
> planning, conservative value estimation) under a shared ecological model, not as
> a benchmark of the published deep-RL systems.

Which methods are closest to paper intent? **PLUS and MOOR** — they are ecological
baselines whose originals are already mechanistic, so the adaptation gap is small.
Say so; it is a genuine strength.

### Q7 — `.tex`, `.md`, or mixed?

**Mixed, with a rule.**

- **LaTeX** for artifacts destined for the paper/thesis, which carry math and will be
  lifted into the manuscript: `01` (problem setting), `02` (data spec), `03`
  (state/action/reward reference), and probably `05` (claims boundary).
- **Markdown** for repo-facing living docs that must stay in sync with code and read
  well on GitHub: `README`, `06` (implementation map), `07` (protocol), `08`
  (results summary), `09` (limitations tracker).

**The rule: one source of truth per fact.** Never restate the same table in both a
`.tex` and a `.md`; the `.md` links to the `.tex`.

Two practical requirements the plan omits:
- Each `.tex` must compile standalone (shared `preamble.tex` or its own preamble),
  otherwise they rot silently.
- Add a `make docs` target that compiles them. An uncompilable `.tex` in the repo is
  worse than no `.tex`.

---

## Findings not covered by the plan's questions

### F2 — Retire or redirect the old "live" docs (structural, must fix)

The plan's "do not delete old docs" is correct for *history* but wrong for the four
current-reference docs at `docs/` root. Once the bundle exists:

- Fold `experiment_protocol.md` + `reproducibility.md` into `07`, then replace both
  with a one-line pointer (or move to `docs/history/`).
- Fold `methods.md` into `04`; retire it.
- `architecture.md` currently holds the axes glossary — **move** that content into
  the bundle README/`01` rather than duplicating it, then reduce or retire the file.

Also update `docs/README.md`, which today asserts "Current reference material lives
in this directory." Creating `docs/current_benchmark/` inside `docs/` makes that
statement false.

### F3 — The folder name will age badly

`docs/current_benchmark/` has the same defect we just fixed in `configs/default.yaml`
(the file named "default" was not the default). "Current" is a moving target and will
be wrong the moment a next-generation setting appears. Prefer **`docs/benchmark/`**
or `docs/guide/`. Keep the `01_`–`08_` numeric prefixes — those are good.

### F4 — Adopt the two useful doc types the screenshot implies

- `LIMITATIONS_tracker_real_ecology.md` → a **living** limitations tracker (`09`),
  distinct from `05`'s claims snapshot.
- `Baselines_State_Action_Reward_Reference.tex` → already covered by `03` ✓.
- `CHAT_HANDOFF_*` is a process doc — correctly excluded from a canonical set.

### F5 — Build in an anti-drift mechanism

These docs have already drifted once (the "Tier" vocabulary and `real_setpoint`
survived in prose long after the code moved on). Cheap defences:

- End every bundle doc with a footer: `Verified against: <files> @ <commit-sha>`.
- Consider a tiny test asserting that constants quoted in the docs match code —
  e.g. `NUM_REAL_ACTIONS == 11`, `NUM_REAL_POPULATIONS == 9`. A doc that states a
  number the code contradicts is worse than a doc that omits it.

---

## Corrected file set

```
docs/benchmark/                       (renamed from current_benchmark)
  README.md                  reading order + glossary + source-of-truth hierarchy
  01_problem_setting.tex     POMDP, obs model, public/private boundary,
                             modes table, + DESIGN RATIONALE section
  02_real_ecology_data.tex   all six CSVs, provenance + citations, caveats
  03_state_action_reward.tex state vars, action semantics, reward modes, metrics
  04_algorithms.tex          per-method; MANDATORY "original architecture vs here"
  05_claims_boundary.tex     what we may/may not claim (see F1 wording)
  06_implementation_map.md   concept -> file guide
  07_experiment_protocol.md  run commands, matrix, gate, aggregate, outputs
  08_results_report.md       P-safe, P=5 rationale, ablation, robustness +
                             sink caveat, links to PDF/figures
  09_limitations_tracker.md  living list (NEW)
```

## Bottom line

Approve the plan, with these changes:

1. **F1 (critical):** rewrite the adaptation framing around **model class** — no
   neural nets anywhere; ridge-linear dynamics and linear Q. Add the mandatory
   per-method architecture field. Name PLUS/MOOR as the closest to paper intent.
2. **F2:** specify retirement/redirect of `architecture.md`, `methods.md`,
   `experiment_protocol.md`, `reproducibility.md`, and fix `docs/README.md`.
3. **Q1 gaps:** add the design-rationale section, cover all six CSVs with
   provenance/citations, add the living limitations tracker.
4. **Q4:** include the honest robustness finding and the sink-population exclusion.
5. **F3:** rename the folder to `docs/benchmark/`.
6. **Q7:** mixed `.tex`/`.md` with one-source-of-truth, standalone-compilable
   `.tex`, and a `make docs` target.
7. **F5:** `Verified against: … @ <commit>` footers.

The instinct — "the code is clean enough; the docs need one new canonical layer
rather than more shuffling" — is correct. The bundle's value stands or falls on
getting F1 right.

---
---

# Addendum — Audit of codex's Revised Plan

Date: 2026-07-10
Reviewed: codex's revision, which states it "accepts Claude's F1 and F2 fully."

## Verdict on the revision

**It does — substantively, not nominally. Approve to implement**, with seven
tightenings. Items **A1–A3 are blocking**; A4–A7 are cheap and prevent known
failure modes.

### What the revision got right (no further action)

- **F1 accepted with accurate framing.** "MOPO-style: ridge-linear ensemble +
  pessimistic particle MPC, not neural MOPO/SAC" is exactly the model-class
  distinction the audit demanded. The `*-inspired` naming convention
  (RefPlan-inspired, BA-MCTS-inspired, Delphic-CQL-inspired) does real defensive
  work in a paper.
- **PLUS/MOOR disclosure** — closest to paper intent *because* their originals are
  mechanistic; PLUS varies candidate `K`, MOOR fits `K` only. Correct and honest.
- **`05` merges implementation plan + code map**, resolving the plan-vs-map gap.
- **`04` (`.tex`, paper-bound) vs `08` (`.md`, living tracker)** correctly resolves
  the snapshot-vs-tracker distinction from F4.
- F3 (`docs/benchmark/`), F5 (anti-drift footer), Q1 design rationale, Q4 honest
  findings, Q5 translation table — all accepted.

---

## A1 (BLOCKING) — Disposition all ten loose docs, not four

`docs/` root currently holds **ten** loose files. The revision names four
(`architecture.md`, `methods.md`, `experiment_protocol.md`, `reproducibility.md`).
The remaining six are left as competing current truth — **the exact F2 problem the
revision claims to fix, relocated.**

| File | Role | Disposition |
| --- | --- | --- |
| `architecture.md` | axes glossary → bundle README/01 | fold, then pointer stub or `history/` |
| `methods.md` | 31 lines → `04` | fold, then retire |
| `experiment_protocol.md` | → `06` | fold, then retire |
| `reproducibility.md` | → `06` | fold, then retire |
| `29_6_Real_Ecology_Setting_Implementation_Plan.tex` | source for `01`/`05` | → `history/` after distilling |
| `29_6_Real_Ecological_Data_Actions_and_Costs.tex` | source for `02` | → `history/` |
| `29_6_REAL_ECOLOGY_IMPLEMENTATION_ALIGNMENT_GUIDE.md` | source for `03`/`05` | → `history/` |
| `22_6_Continuous_Observation_New_Baselines.tex` | synthetic-benchmark report | → `history/` or `references/` (see A7) |
| `22_6_Continuous_Observation_Experiment_Results.tex` | synthetic-benchmark report | → same |
| `AUDIT_documentation_bundle_plan_2026-07-10.md` | this file; process doc | → `history/` when the bundle lands |

Acceptance: after the work, `docs/` root contains **only** `README.md`,
`benchmark/`, and the existing subdirectories (`history/`, `real_ecology_history/`,
`references/`, `claude_build_report/`, `plots_selected/`).

## A2 (BLOCKING) — "Claims" appears in two filenames

`04_algorithm_adaptations_and_claims.tex` **and**
`08_limitations_and_claims_boundary.md`. This is the Q3 redundancy resurfacing
under new names — a reader will not know which is authoritative.

Enforce a hard split:
- **`04`** — per-method, **descriptive** + adaptation disclosure. Paper-bound.
- **`08`** — rename to **`08_limitations_tracker.md`**: aggregate limitations, data
  caveats, robustness caveats, future work. **No per-method restatement**; link to
  `04` instead.

## A3 (BLOCKING) — Design rationale is duplicated across `01` and `05`

`01` carries "why set-point `r` / cumulative `K` / occupancy / `P=5`"; `05` carries
"why it was implemented this way." Draw the boundary explicitly:

- **`01` = scientific rationale.** Why this problem formulation; why the action
  semantics are set-point `r` and cumulative `K`; why occupancy rather than crossing
  penalty; why `collapse_penalty = 5`.
- **`05` = engineering rationale + code map.** Why `data_mode` is orthogonal to
  `control_mode`; why the public/private data boundary exists; where each concept
  lives in the code.

## A4 — `make docs-check` is a linter, not a compiler

The revision took the lint (banned terms, stale paths) but dropped the compile.
Keep both:
- **`make docs`** — compiles the `.tex` files. Give them a shared `preamble.tex` so
  each builds standalone. An uncompilable `.tex` in the repo is worse than none.
- **`make docs-check`** — the lint: banned terms (`Tier-2/3/4`, `real_setpoint`,
  `claude_build`, `discrete_action_cont_obser`, "standalone repo"), broken relative
  paths, and missing `Verified against:` footers.

## A5 — The source-of-truth hierarchy is missing (Q2)

This is an **operational rule while drafting**, not decoration. The `29_6_*` specs
**predate** the `data_mode` split, the `setpoint_cumulative` rename, and the P-safe
decision; expect stale names and possibly stale numbers. Put this in the bundle
README and obey it while writing:

> Code and CSVs are authoritative, always. `architecture.md` is current. The
> `29_6_*` specs are **pre-refactor** and every number and name taken from them must
> be re-verified against code. Nothing in `history/` or `real_ecology_history/` is
> ever cited as current.

## A6 — State the "one source of truth per fact" rule

Given the `01`/`05` and `04`/`08` overlaps, this rule is what stops them drifting
apart. **Never restate the same table in a `.tex` and a `.md`** — the `.md` links to
the `.tex`. Add it to the bundle README next to the hierarchy.

## A7 — The scope decision is still unmade

`07` is titled "recent **real ecology** results." Does the bundle mention the
synthetic stress-benchmark results (baselines failing under collapse-sensitive
settings)? A supervisor will ask.

Note this is the **same decision** as the `22_6_*` disposition in A1 — those two
files *are* the synthetic reports. Resolve once: either add a one-paragraph scope
statement in the bundle README pointing at them, or give `07` a short synthetic
section.

---

## Concrete addition to `04`: an identifier mapping table

The CLI and configs still use the bare names (`--method mopo`). Without an explicit
mapping, a reader hits `mopo` in `configs/` and re-forms the precise assumption `04`
exists to prevent. Include:

| Code identifier | Reader-facing name | Original paper | Model class here |
| --- | --- | --- | --- |
| `mopo` | MOPO-style | Yu et al., MOPO | ridge-linear ensemble + pessimistic particle MPC |
| `refplan` | RefPlan-inspired | RefPlan | posterior ensemble planner |
| `bamcts` | BA-MCTS-inspired | BA-MCTS | finite-action belief tree |
| `plus` | PLUS | BioConserv18 PLUS | mechanistic Ricker bank (candidate `K`) |
| `moor` | MOOR | ExpertSys23 MOOR | single misspecified Ricker (fits `K`) |
| `delphic` | Delphic-CQL-inspired | Delphic offline RL | linear/ridge compatible-world adaptation |
| `ogsrl` | OGSRL-inspired | OGSRL | guardian/objective adaptation |

Verify each "original paper" cell against `docs/real_ecology_history/29_6_algorithm_method_notes.tex`
before publishing — do not take this table's citations on trust.

---

## Corrected final file set

```
docs/benchmark/
  README.md                              reading order, glossary, term-translation
                                         table, source-of-truth hierarchy (A5),
                                         one-source-of-truth rule (A6), scope (A7)
  01_problem_setting_and_design.tex      POMDP + SCIENTIFIC rationale (A3)
  02_real_ecology_data_actions_and_costs.tex   all six CSVs + provenance/citations
  03_state_action_reward_reference.tex   real | dummy | synthetic
  04_algorithm_adaptations_and_claims.tex  per-method, descriptive + disclosure
                                         + identifier mapping table
  05_implementation_and_code_map.md      ENGINEERING rationale + code map (A3)
  06_experiment_protocol_and_reproducibility.md
  07_recent_real_ecology_results.md      incl. robustness + sink caveats
  08_limitations_tracker.md              renamed (A2); living, no per-method restatement
  preamble.tex                           so the .tex compile standalone (A4)
```

## Acceptance checks

- `docs/` root contains only `README.md`, `benchmark/`, and the subdirectories (A1).
- `make docs` compiles every `.tex`; `make docs-check` passes (A4).
- `rg -n "Tier-?[234]|real_setpoint|claude_build|discrete_action_cont_obser|standalone repo" docs/benchmark/`
  returns nothing but the single legacy-alias sentence.
- Every bundle doc ends with `Verified against: <files> @ <commit-sha>`.
- No table appears in both a `.tex` and a `.md` (A6).
- `04` states the model-class distinction in its first paragraph, not a footnote.

## Bottom line

The revision accepts the two findings that mattered. Implement it with A1–A3
resolved up front and A4–A7 folded in. The bundle still stands or falls on `04`
saying, plainly and early, that these are **linear/mechanistic re-implementations of
algorithmic ideas — not the published deep-RL systems**.
