# Server Handoff — Motivation Experiment (for the code-aware Claude/Codex)

**Who you are.** You are the agent that built and ran this real-ecology benchmark
(`real_ecology_runs/`, `discrete_action_cont_obser/`, `code/real_ecology_data`, the shared
particle filter, particle-MPC planner, the 7 methods, the evaluator/manifest). You know the
code and the prior results. **This file gives you a NEW task plus the design decisions that were
made off-server (in a planning session with the PI) — they are settled; map them onto your code.**
Where this file says "confirm in your code," you have the authority I lack.

**What I (the planning side) do not have.** I could not read the codebase — it lives on this
server, not in the docs folder. So treat all module/path references as *descriptions to locate*,
not verified facts. Companion docs (on the PI's machine, transferable if useful):
`MOTIVATION_EXPERIMENT_implementation_prep.md` (fuller [VERIFY] list), `SCOPE_AND_CONTRIBUTIONS.pdf`,
`FINAL_baseline_selection.xlsx`, `LIMITATIONS_tracker_real_ecology.md`.

---

## 1. The job, in one paragraph

Run the **motivation experiment** for the paper: **general offline model-based RL** (run on the
continuous state via the particle filter) versus the **ecological baselines run in their
NAIVE/native form** (discretize + discrete-POMDP solve — *not* the current `filter=ricker`
particle-MPC adaptation), on the real-ecology, unknown-model-form setting, in **both reward modes
(safe P5 and yield P0)**. Purpose: motivate the problem — show that in this continuous, noisy,
structurally-uncertain *conservation* setting, naive ecological methods struggle while general
offline MBRL does better, and expose where the general methods *still* fail (safety/sinks/Allee),
which motivates the proposed method.

## 2. Why (paper framing — so your implementation choices serve the point)

The paper positions the problem as **offline RL for conservation of real populations under three
escalating layers of uncertainty**: (1) state (survey noise) — baselines already do this; (2)
parameter; (3) **structural / model-form** uncertainty — the true dynamics may be Ricker, Allee
(tipping point), theta-logistic, or regime-switching, which differ *qualitatively*. Plus a
**conservation objective** (persistence/abundance − cost − collapse penalty), not the extractive
harvest-yield objective of the ecological baselines. The motivation experiment must make the
structural-uncertainty and conservation angles visible — that is where naive, single-form,
discretized ecological solvers should break.

## 3. Run set — 5 methods (exactly how each runs)

| Method | Side | How to run |
|---|---|---|
| **RefPlan** | general MBRL | existing implementation, `filter=learned` |
| **BA-MCTS** | general MBRL | existing implementation, `filter=learned` |
| **OGSRL** | general MBRL | existing implementation, `filter=learned` |
| **PLUS** | ecological | **NAIVE/native** discretize + discrete-POMDP solve (NEW path, see §4) |
| **MOOR/Ju** | ecological | **NAIVE/native** discretize + discrete-POMDP solve (NEW path, see §4) |

**Exclude** MOPO and Delphic from this run. Per-cell agents (this is NOT the pooled-agent
experiment).

## 4. The naive/native ecological baselines — the one real build

**Definition (settled).** "Naive/native" = paper-native **solver mechanics**: discretize the
latent state / observation / belief and solve a **discrete POMDP**. It runs on **our
real-ecology environment** — keep the 11 named actions, the real costs, and both reward modes.
It does **NOT** mean:
- the current `filter=ricker` + particle-MPC adaptation (that is the *adapted* PLUS/MOOR), nor
- reverting to the published single-harvest-knob / gross-yield problem, nor
- a deliberately crippled method.

So: native PLUS/MOOR *solver style* on *our* env. The intended disadvantage is **structural**
(discretization + mechanistic-form assumption under unknown Allee/theta/regime dynamics), not an
implementation handicap.

**Gate check first (do this before building).** Search the repo for an existing native solver:
`grep`/`rg` for `PLUS-GPU`, `pomdp`, `pbvi`, `sarsop`, `despot`, `qmdp`, `value.?iteration`,
`discretiz`, `native.*solver`, and `class .*(PLUS|MOOR)`. Also check installed solvers
(`pomdp_py`, `pomdpy`, `pypomdp`, `sarsop`). Report: does a native discrete-POMDP PLUS/MOOR
already exist (e.g. is "PLUS-GPU" actually a native solver), or is this build-required?
*Planning-side expectation (unverified): build-required, and no POMDP solver is installed — but
you can confirm.*

**If build-required:**
- **Discretizer** for `s`, `o`, and a discrete belief support. Per-population grid `0 … 2·K_base`
  (= `K_max`) plus an overflow bin and a collapse/`s≈0` bin. Real scales are `K = 31–325`, so grid
  size adapts per population. Run a coarse-vs-fine resolution sensitivity check.
- **MOOR/Ju native:** least-squares POMDP model fit from the offline `(o,a,o')` data, discretize,
  solve.
- **PLUS native:** candidate mechanistic models + posterior/evidence update, belief planning over
  the discretized belief, solve.
- **Solver:** start with **QMDP / belief-grid value iteration** (cheapest faithful first cut);
  add PBVI/SARSOP later if a dependency is available.
- **Interface:** give the native policies the same `reset/act/update/log` shape as the existing
  methods so the evaluator and metrics logger handle them unchanged.

**Fairness guardrails (keep the motivation honest, not a straw man):** native baselines (i) fit
their model from the *same* offline dataset the general methods use (log the dataset hash per
cell); (ii) run on the *same* env, seeds, and reward-agnostic battery; (iii) use a *reasonable*
discretization (report resolution; verify finer grids don't flip the story). They do **not** use
the shared particle filter — they run their own discretized belief. Document this asymmetry
explicitly; it is the point.

## 5. Run configuration

- Methods: `{RefPlan, BA-MCTS, OGSRL, PLUS_native, MOOR_native}` (add the two native variants to
  the method registry; route them to the native path, away from the shared particle filter/MPC).
- Filter/observability: general methods `filter=learned`; native ecological use their own
  discretized belief (a `solver=native_discrete_pomdp` route, not a `filter` value).
- Factors: all 9 populations, 4 families (Ricker/Allee/theta/regime), σ ∈ {0, 0.1, 0.2, 0.4}.
- **Reward modes: BOTH.** (a) **safe mode, P5** (the locked operating point); (b) **yield mode,
  P0** (baseline-matched — the ecological baselines are natively yield/harvest agents). Train a
  **separate agent per reward mode**. **Do NOT compare raw returns across modes** (objectives
  differ) — score both on the common reward-agnostic battery and report side by side.
- Eval protocol: keep identical to the prior audited run — 5 seeds × 4 episodes × horizon 50 — for
  comparability.
- New run dir, e.g. `real_ecology_runs/motivation_native_baselines_<date>/`.
- Reward-leakage guard must hold for the native baselines (belief updates on `(o,a)` only; realized
  reward never an input).

## 6. Reporting / metrics

- Reuse the reward-agnostic battery: control/true return, quasi-extinction (collapse) probability,
  unsafe fraction, min/final abundance, persistence, economic cost.
- **Report per-σ (do NOT pool)** — noise-robustness is part of the motivation.
- **Report per-dynamics-family** and the **recoverable-vs-sink split** (sinks reported separately,
  per the prior report).
- Headline figures for the motivation: general-vs-native-ecological **return and collapse by σ and
  by dynamics family**. The family breakdown is where naive Ricker-form solvers should visibly
  degrade on Allee/theta/regime cells — that is the money plot.
- Also produce the safe-vs-yield comparison on the battery (expected: yield≈safe on recoverable
  persistence, but yield-mode exposes the safety gap on sinks/Allee where a yield-only agent
  rationally liquidates the stock).

## 7. Acceptance tests

- [ ] Gate resolved: native solver exists (configure) or not (build), with evidence.
- [ ] Native PLUS/MOOR produce a sane policy on a **Ricker cell** (where their assumption is
      correct) at least as good as their adapted `filter=ricker` version — proves the native
      pipeline works and isn't crippled.
- [ ] Same offline dataset feeds general and native-ecological methods (dataset hash logged per cell).
- [ ] Both reward modes run (safe P5 + yield P0), separate agent per mode, identical battery.
- [ ] Per-σ, per-family, recoverable/sink metrics emitted in the existing schema.
- [ ] Discretization-resolution sanity check: coarse vs. fine does not flip the headline.
- [ ] Reward-leakage guard holds for the native baselines.
- [ ] Run completes with 0 failures; native metrics match the general-method schema.

## 8. Decisions already made (do not re-litigate)

- **Baseline selection.** This run uses RefPlan, BA-MCTS, OGSRL (general) + PLUS, MOOR (native
  ecological). The *fuller* final set (adding MOBILE, Nicol 2024/Management Science, Chadès
  2025/PNAS; citing GPDP and MOPO) is for a **later** run, not this one. See
  `FINAL_baseline_selection.xlsx`.
- **Naive = native solver mechanics on our env** (§4), confirmed with the correction that it is
  NOT the published single-knob problem.
- **Both reward modes**, separate agents, battery-only cross-mode comparison.
- **Motivation purpose:** show general > naive-ecological AND where general still fails — do not
  optimize this run to make general look maximally good; the remaining failure is the paper's hook.

## 9. First actions (ordered)

1. Run the §4 gate grep; report whether a native solver exists. **This decides build vs. configure.**
2. Confirm the method-registry / manifest mechanics for adding `PLUS_native`, `MOOR_native` and
   restricting the sweep to the 5 methods and both reward modes.
3. If build-required: implement discretizer + QMDP/belief-grid + native MOOR fit + native PLUS
   posterior, behind the existing method interface.
4. Wire the manifest (§5), run a 1-cell smoke test (Ricker, σ=0, both modes), check the §7 tests,
   then launch the full sweep.
5. Report back: gate result, exact file paths touched, and any point in §4–§6 that conflicts with
   how the code actually works.
