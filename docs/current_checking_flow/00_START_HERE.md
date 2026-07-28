# START HERE — orientation, reading order, and file index

Last updated: 27 July 2026. Read this first if you have lost the thread.

---

## 1. What this project is, in six lines

You are comparing **adapted ecological POMDP planners** (PLUS, MOOR) against
**general offline RL methods** (RefPlan, OGSRL, BA-MCTS, EVD) on a conservation
benchmark where demographic parameters and the true population-model family are
hidden. Three species, four hidden families, two observation-noise levels, six
methods, one common evaluator — 144 accepted method-cells.

**The eventual goal:** decide whether a *new method for uncertainty over the
population-model family* (Ricker vs Allee vs theta-logistic vs regime-switching)
is scientifically justified. Everything else is instrumentation for that
decision.

---

## 2. Where things stand, 27 July

**The accepted result is unchanged and remains valid.** No number in
`MATCHED_P10_144_METHOD_CELLS.csv` has been altered. What changed is the
*interpretation*, on five points:

1. **Only one of three species can support a method claim.** Measured headroom
   against constant-action baselines: fox positive, tiger zero in 5 of 8 cells,
   vulture ≈ −4.5 (every method loses to doing nothing).
2. **Egyptian vulture is provably infeasible.** No policy in the 11-action set
   can exceed abundance 42.94 against a safety threshold of 81.25. Not a hard
   problem — an impossible one.
3. **No method optimises the reward it is scored on.** Five of six plan against
   a learned linear surrogate with no safety-threshold term (R² 0.30–0.73);
   only EVD uses raw logged rewards.
4. **The family degeneracy is self-inflicted.** PLUS and MOOR deploy actions
   with `r ≤ 0`, which zeroes the growth term — the only place the four families
   differ. They don't fail to distinguish families; their chosen action erases
   the distinction.
5. **Compute is unequal by 26× per cell** (measured), and PLUS gets 6,126× EVD's
   per-task time.

**Diagnostic replay COMPLETE (27 July, 22:09 AEST); M1–M15 pipeline run after.**
Tier A (10) + Tier B (12) + A6 checkpoints: 24/24 parity receipts PASS at 0.0,
`recomputed_fits=0`, accepted CSV unchanged. Load-bearing new result: PLUS's
posterior-averaging is decision-active on all four fox cells (switch 0.14–0.71)
but its effect vs MOOR is **family-dependent** — harmful on Allee, helpful on
regime/theta, neutral on Ricker; nets ≈0. So the extra machinery buys no
consistent advantage over a single misspecified model (not "uniformly harmful" —
an earlier over-generalization now corrected). Full scorecard in
`NEXT_WORK_QUEUE.md` §1-OUTCOME; pipeline outputs in
`from AI agent in code server/analysis_out/`. Next: the S2 spec (thresholds
confirmed).

---

## 3. Reading order, by situation

### "I've forgotten everything and have 15 minutes"
1. This file, §1–§2.
2. `MY_RESEARCH_QUESTIONS.md` — your own questions and what has answered them.
3. `UNCERTAINTY_TAXONOMY_AND_STAGED_DIAGNOSTIC_PLAN.md` §0 only (the executive
   summary — about 6 pages).

### "What exactly is the accepted result?"
1. `CHAT_HANDOFF_...P10_ECOLOGICAL_BASELINES.md` — the controlling document for
   provenance, chronology, metric definitions and reporting rules.
2. `merged_results_...tex` — the tables, rankings and interpretation.
   **Presentation only; the CSV on the server controls the science.**

### "What does method X actually do, and how does it differ from its paper?"
- PLUS / MOOR → `PLUS_MOOR_PAPER_ALIGNED_IMPLEMENTATION_EXPLANATION.md`, then
  `PLUS_MOOR_RICKER_STRESS_TEST_SUPERVISOR_EXPLANATION.tex`
- RefPlan / OGSRL / BA-MCTS / EVD →
  `GENERAL_RL_CURRENT_IMPLEMENTATION_AND_SETTINGS.md`, then
  `GENERAL_RL_PAPER_IDEAS_AND_ECOLOGY_ADAPTATIONS.tex`
- The original papers → `ecological baseline papers/`, `general RL papers/`

### "What is actually true in the code, versus what we assumed?"
1. `READONLY_CODE_AUDIT_INFO_SYMMETRY_AND_METRICS.md` (audit 1) — reward
   constants, information symmetry, environment structure, artifact availability.
2. `READONLY_EXTRACTION_PASS_2.md` (audit 2) — action tables, which reward each
   method optimises, surrogate fidelity, measured compute.

**Read these before trusting any claim about the implementation.** Both
overturned things that had been assumed for weeks.

### "What do we do next, and why?"
1. `UNCERTAINTY_TAXONOMY_AND_STAGED_DIAGNOSTIC_PLAN.md` §9 (staged plan), §10
   (decisions you must make), §11 (what is still open).
2. `DIAGNOSTIC_REPLAY_RERUN_SPEC.md` — the currently-running experiment.
3. `REWARD_REDESIGN_MEMO.md` — the reward axis, with a near-free screening step.

### "I have a supervisor meeting"
`MY_RESEARCH_QUESTIONS.md` → this file §2 → taxonomy §0. The five interpretation
changes in §2 above are the things a supervisor will react to.

---

## 4. File index

### Controlling documents — do not edit casually
| File | What it is | Authority |
|---|---|---|
| `CHAT_HANDOFF_...P10_ECOLOGICAL_BASELINES.md` | Provenance, chronology, metric interpretation, reporting guardrails | **2nd** — controls interpretation |
| `merged_results_...tex` / `.pdf` | Derived presentation report: tables, rankings, interpretation | **3rd** — presentation only. **PDF is stale vs the TeX; recompile before use** |

*(1st authority is `MATCHED_P10_144_METHOD_CELLS.csv` + receipt, on the server,
SHA `7431318…`. Never mix in the superseded `OPTION_A_MATCHED_144` table.)*

### Method documentation — algorithms valid, old status sections historical
| File | Covers |
|---|---|
| `PLUS_MOOR_PAPER_ALIGNED_IMPLEMENTATION_EXPLANATION.md` | PLUS/MOOR: paper ideas, why old versions were "inspired", what the corrected adaptations do |
| `PLUS_MOOR_RICKER_STRESS_TEST_SUPERVISOR_EXPLANATION.tex` / `.pdf` | Same, supervisor-facing, with the mechanistic equations and the four-family design |
| `GENERAL_RL_CURRENT_IMPLEMENTATION_AND_SETTINGS.md` | The four general methods: implementations, settings, fidelity to papers |
| `GENERAL_RL_PAPER_IDEAS_AND_ECOLOGY_ADAPTATIONS.tex` | Same, supervisor-facing, with fidelity/adaptation tables |
| `THREE_SPECIES_PERFORMANCE_AND_ADEQUACY_PLAN.md` | The original staged plan. Stage 0 and the safe part of Stage 1 are done; later stages are proposals |

### Audits — ground truth about the code
| File | Settles |
|---|---|
| `READONLY_CODE_AUDIT_INFO_SYMMETRY_AND_METRICS.md` | `s_safe`/`K_ref` per species, no information asymmetry, `process_noise = 0`, family degeneracy mechanism, candidate diversity, what is/isn't stored |
| `READONLY_EXTRACTION_PASS_2.md` | Full action tables, which reward each method optimises, surrogate R² and bias, measured compute, deployed actions, candidate unanimity |

### Analysis and plans — the live working documents
| File | What it is |
|---|---|
| `UNCERTAINTY_TAXONOMY_AND_STAGED_DIAGNOSTIC_PLAN.md` | **The main analysis.** Uncertainty taxonomy + by-method matrix, decision-discrimination, the staged experiment plan, hypothesis table, open items. ~100 KB — use the section list below |
| `DIAGNOSTIC_REPLAY_RERUN_SPEC.md` | Spec for the instrumented replay currently running: scope, parity gate, log schema, metrics M1–M15 |
| `REWARD_REDESIGN_MEMO.md` | Why the reward is a first-order problem, candidate objectives, and a minutes-long screening test before any expensive re-plan |
| `CHAT_HANDOFF_03_FOLLOWUPS_AND_REPO.md` | **CURRENT chat-to-chat handoff.** Post-fleet results, provisional S2/screening/H14, in-flight H12/S6 reruns + repo cleanup, new-docs index, pending decisions. **Start a new chat from this one** |
| `CHAT_HANDOFF_02_ANALYSIS_PHASE.md` | Prior handoff (fleet phase); superseded on live state by handoff 03, conventions still valid |
| `MY_RESEARCH_QUESTIONS.md` | Your own questions and hypotheses, with status |
| `NEXT_WORK_QUEUE.md` | What to prepare before results land. **§1 contains predictions sealed 27 July, before the fleet reported — do not edit that section** |
| `analysis/` | Python pipeline turning replay logs into M1-M15 tables and figures. Built and unit-tested before the fleet reported; `python3 run_analysis.py --self-test` |
| `00_START_HERE.md` | This file |

### Section map for the big analysis file
`UNCERTAINTY_TAXONOMY_AND_STAGED_DIAGNOSTIC_PLAN.md`:

| § | Contents |
|---|---|
| 0 | **Executive summary** — the findings, the corrections, the S1 result, the vulture infeasibility proof |
| 1 | Uncertainty taxonomy (U1–U13) and the uncertainty-by-method matrix |
| 2 | Is the task decision-discriminating? Diagnostics, controls, cell ranking |
| 3 | True-state control: the 2×2 factorial and what each arm isolates |
| 4 | Observation and state uncertainty; what would show they are good POMDP solvers |
| 5 | Transition uncertainty within Ricker; what PLUS's posterior really represents |
| 6 | Family uncertainty: the shared-equilibrium argument and the gate |
| 7 | Data and tuning adequacy; the minimal sensitivity study |
| 8 | Reward sensitivity (see also the reward memo) |
| 9 | Staged plan: motivation, hypotheses H1–H15, stages S0–S10, criteria |
| 10 | **Decisions you must make** |
| 11 | What is still open, and its status board |

### Papers
`ecological baseline papers/` — BioConserv18 (PLUS) + supplement, ExpertSys23 (MOOR).
`general RL papers/` — ICLR24 Delphic, ICLR26 BA-MCTS, ICML25 RefPlan,
NeurIPS25 OGSRL, and ICML23 MOBILE (**context only — not one of the six
implemented methods**).

---

## 5. Chronology, so version confusion doesn't recur

| When | What happened |
|---|---|
| ~20 July | General-RL 576-row package and the Ricker-only PLUS design frozen |
| 23 July | P=10 correction executed: PLUS/MOOR **re-planned** (not re-scored) to match the general methods' penalty |
| 24 July | 144-cell matched comparison accepted; merged report written |
| 25 July | Read-only code-to-document audit; handoff and merged TeX updated; supporting docs marked historical |
| 26 July | Audit 1 (information symmetry, metrics) and audit 2 (actions, surrogate, compute); S1 constant-action sweep; instrumented replay begun |
| 27 July | Controlling documents updated with the reference-policy findings; reward memo written |
| 27 July (pm) | A6-PLUS checkpoint passed; 11-run fleet launched and completed. Tier A + Tier B + checkpoints = 24/24 parity 0.0. M1–M15 pipeline run after: PLUS averaging is family-dependent (harmful on Allee, helpful on regime/theta, nets ≈0), not uniformly harmful; item 19 closed; M12/OGSRL prediction missed. Outcomes in `NEXT_WORK_QUEUE.md` §1-OUTCOME |

---

## 6. Things that are easy to get wrong

- **P=5 vs P=10.** Historical ecological returns used P=5. They are not
  comparable and must never enter the P=10 table.
- **"Adapted" vs "inspired" vs "motivated."** PLUS/MOOR are *adapted*;
  RefPlan/OGSRL/BA-MCTS are *inspired*; EVD is *Delphic-motivated only* and its
  variance must never be called Delphic uncertainty.
- **Within-cell SD is not uncertainty.** It is 20 episodes under one dataset and
  one fitted policy — not variation across datasets or fits.
- **Equal returns are not algorithmic equivalence**, and a constant policy is
  not evidence that uncertainty was handled.
- **Zero collapse-entry is not safety** when the population starts unsafe.
- **The planner's reward is not the evaluator's reward** (audit 2). Do not
  credit any method with safety reasoning based on these returns.
- **Constant-action baselines are reference controls**, never a seventh method
  in the ranking tables.
- **`K_ref` and `s_safe` are species-specific**: 250/25.0, 41/10.25, 325/81.25.
  There is no global 500/25.
- **The merged PDF is stale** relative to its TeX. Recompile before showing it.
