# Phase 2 — Data-Budget & Hyperparameter-Adequacy Plan (preregistered)

**Date:** 2026-07-18
**Status:** PLAN ONLY. No runtime code modified, no data collected, no jobs submitted, no returns
inspected, frozen 32-cell manifests unchanged. 32-cell submission **paused** pending this plan's approval.

The 4,000-transition budget was inherited from the earlier benchmark; it has **not** been shown
sufficient for the corrected mechanistic PLUS/MOOR. Finite objectives ≠ adequacy.

---

## 1. Current effective-sample-size audit (frozen datasets; no returns read)

Measured from the completed hidden-r/K frozen public datasets (same collector/seed/4,000-target as the
32-cell config; transitions are deterministic given cell+seed). Public action **channel** map:
`a0=none; a1–a4=rate; a5,a6=capacity; a7–a9=rate+capacity; a10=state(translocation)`.

At 4,000 transitions every cell has **160 complete episodes** (128 fit / 32 holdout under the 0.8 split;
crab_eating_fox 163). **No zero-count and no sparse (<25-row) actions in any audited cell.** Per-action
coverage is highly non-uniform, and the identification bottleneck is **capacity actions in the sink
(Egyptian vulture)**:

| Cell | min rows / min episodes across RATE (a1-4,7-9) | across CAPACITY (a5-9) | STOCK (a10) |
|---|---|---|---|
| amur_tiger / ricker / σ0.1 | 181 / 86 | 253 / 111 | 325 / 139 |
| amur_tiger / regime / σ0.4 | 151 / 81 | 194 / 86 | 354 / 142 |
| egyptian_vulture / regime / σ0.4 | 139 / 61 | **68 / 35** | 519 / 155 |
| egyptian_vulture / ricker / σ0.1 | 139 / 61 | **68 / 35** | 519 / 155 |
| crab_eating_fox / theta / σ0.1 | 132 / 69 | 145 / 69 | 147 / 73 |

**Interpretation.** Total row count (4,000) is misleading: each fitted **action** coordinate is
identified only by the transitions/episodes exercising that action. The weakest fitted coordinate is
**Egyptian-vulture capacity (`a5`/`a6`): ≈68 transitions in ≈35–36 distinct episodes** — the
behaviour policy rarely applies pure-capacity actions to a declining sink. Rate and translocation
coordinates are better covered (≥130 rows / ≥60 episodes). Shared reset/process/capacity-scale scalars
are identified by all 4,000 transitions. **Per-parameter identification, not row count, governs
adequacy** — and it is the a5/a6 capacity effects in sinks that are marginal at 4,000.

*(This is a coverage audit only; no `operational_return`/`true_return`/`summary.json` was read.)*

## 2. Consequential-hyperparameter inventory

Provenance codes: **P** paper-derived · **B** benchmark adaptation · **C** computational approximation.
"Selected before returns" is **Yes** for all — every value is a registered default fixed prior to any
corrected return.

### PLUS (`plus_adapted_mechanistic_pbvi`)
| Hyperparameter | Value | Prov. | Prior sensitivity evidence |
|---|---|---|---|
| total candidates / per family | 16 / 4 | B/P | none yet (registered 8/16/32 sweep pending) |
| bootstrap candidate construction | `episode_bootstrap_map_fixed_pi_v2` (cand 0 = full history; 1–3 = episode bootstrap) | B | none |
| fixed-`Π` grid | (0.80, 0.90, 0.97) | B/P | none |
| process paths / regime paths | 16 / 16 | C | none (16/32/64 sweep pending) |
| optimizer starts / iterations | 8 / 100 (L-BFGS, lr 0.5, tol_grad 1e-7, tol_change 1e-9) | P/C | none |
| abundance grid (state_bins) | 41 (+0 state) | C | none |
| capacity / observation bins | 9 / 41 | C | none |
| transition / observation MC samples | 256 / 256 | C | none |
| PBVI horizon | 5 | C | none |
| PBVI belief points / obs branches | 32 / 7 | C | none |
| PBVI expansion / backups / tol / seed | reachable-graph forward sample; full backward sweep to horizon; deterministic; seeded | C | validated **exact** on the tiny POMDP test (action + optimal value) |

### MOOR (`moor_adapted_ricker_misspec_pbvi`)
| Hyperparameter | Value | Prov. | Prior sensitivity |
|---|---|---|---|
| fitted parameter count | 5 scalars + **14** action-effect (8 rate incl. baseline, 5 capacity, 1 stocking) + Allee/theta/regime = inactive constants (single Ricker) | B | mask test proves inactive params cannot affect outputs |
| process paths | 16 | C | none |
| starts / iterations / tolerances | 8 / 100 / (1e-7, 1e-9) | P/C | none |
| abundance grid / obs discretization | 41 / 41 | C | none |
| PBVI settings | horizon 5, belief 32, branches 7 | C | none |

**Gap:** almost every computational hyperparameter has **no prior sensitivity evidence** — which is
exactly what §6 registers.

## 3. Nested offline-data sensitivity (design)

Complete episodes, **nested prefixes** — collect once at the largest budget per cell, then take strict
episode-prefix subsets (same ordering/seed) for smaller budgets. Never arbitrary row truncation.

- **Budgets:** 1,000 / 2,000 / 4,000 / 8,000 target transitions (**16,000 optional**, if runtime permits).
- **Adequacy subset (difficult conditions, smaller than the 32-cell run):**
  populations **Amur tiger, Egyptian vulture**; families **Ricker, regime** (the mechanistically hardest;
  all-four is affordable only at ~2×, see below); **σ_o ∈ {0.1, 0.4}**; **PLUS + MOOR**;
  **fitting + planning diagnostics only, no comparative returns.**
- Dynamics cells = 2 × 2 × 2 = **8**.

### Exact job / CPU arithmetic

- **Collection jobs = 8** (one per cell at the max budget; smaller budgets are prefixes → no extra
  collection).
- **Fit jobs = 8 cells × 4 budgets × 2 methods = 64.**
- **Plan-diagnostic jobs = 64** (one per fit; PBVI action/value stability on a *frozen* public
  belief/context set — **not** a full 5-seed×4-episode evaluation, so cheap).

CPU (fit ∝ transitions; measured PLUS ≈ 1 h / 1,000 transitions, MOOR ≈ 0.085 h / 1,000; plan-diagnostic
≈ 0.15 h/config avg):

| | PLUS fits | MOOR fits | plan-diag | total |
|---|---:|---:|---:|---:|
| central | 8·(1+2+4+8)=120 | 8·1.28=10 | 64·0.15≈10 | **≈ 140 CPU-h** |
| upper (1.5×) | | | | **≈ 210 CPU-h** |
| + 16,000 (optional) | +8·16=128 | +11 | +5 | **≈ 420 CPU-h (needs separate approval)** |

**Proposed adequacy-study ceiling: 200 CPU-h** (without 16,000). All-four-families would roughly double
fits (16 cells) → defer unless the 2-family result is ambiguous.

## 4. Return-blind adequacy outcomes (metric definitions)

None of these read `true_return`, `operational_return`, survival return, or any comparative ranking.

1. action/episode coverage per fitted coordinate (§1 method);
2. train and holdout **normalized trajectory SSE** (the fit objective on each fold);
3. **holdout/train error ratio**;
4. **multi-start objective agreement** (spread & # finite starts);
5. **projected gradient** on free (non-bound-active) coords + optimizer status (scale-aware, §5);
6. **parameter-boundary occupancy** (fraction of free params within ε of a bound);
7. **parameter stability under nested budgets & bootstrap** — standardized L2 distance of the fitted
   parameter vector between adjacent budgets, and across bootstrap resamples at fixed budget;
8. **fitted transition-kernel distance** — mean total-variation between the discretized kernels at
   4,000 vs 8,000 on a **frozen public (state, action) grid**;
9. **PLUS candidate diversity** (pairwise standardized distance; collapse/duplication) and
   **posterior-identification** (synthetic in-bank recovery);
10. **PBVI action agreement** — fraction of a **frozen public belief/context set** where argmax action
    matches between budgets;
11. **PBVI value stability** on that set (relative change);
12. runtime & memory.

Synthetic parameter-recovery may be reported as a *separate* diagnostic (the learner never sees truth),
but **no tuning to any one evaluation cell**.

## 5. Preregistered sufficiency thresholds (fixed now, before observing)

Decision on whether **4,000** is adequate. Thresholds are proposed and justified **before** the
diagnostics are computed.

| Test | ADEQUATE | ADEQUATE-WITH-WARNING | INADEQUATE |
|---|---|---|---|
| Per-parameter coverage | every fitted action coord ≥ **40 transitions AND ≥ 20 episodes** | ≥ 20 transitions AND ≥ 10 episodes | any < 20 transitions **or** < 10 episodes |
| Holdout-SSE improvement 4k→8k | < **10 %** relative | 10–25 % | > **25 %** |
| Fitted-kernel change 4k→8k (mean TV on frozen grid) | < **0.05** | 0.05–0.15 | > **0.15** |
| PBVI action-agreement 4k→8k (frozen belief set) | ≥ **95 %** | 85–95 % | < **85 %** |
| Parameter-boundary / weak-ID rate | < 25 % free params bound-active | 25–40 % | (occupancy alone never sole INADEQUATE) |
| PLUS candidate collapse | all forms ≥ 2 distinct candidates | — | any form < 2 distinct (collapse) |
| Optimizer failure rate | ≥ 2 finite starts, 0 all-fail | some 1-finite-start | any all-start failure |

**Overall rule (per cell, then aggregated):**
- **4,000 ADEQUATE** iff coverage PASS **and** holdout-improvement < 10 % **and** kernel-change < 0.05
  **and** action-agreement ≥ 95 % — for **all** difficult cells.
- **ADEQUATE-WITH-WARNING** if the difficult cells pass but a subset (anticipated: Egyptian capacity)
  sits in WARNING bands → keep 4,000 primary but **label those cells data-limited** in every report.
- **INADEQUATE** if any difficult cell hits an INADEQUATE band → the **primary budget must increase**
  (to the smallest budget clearing all ADEQUATE bands, expected 8,000) **or** the 4,000-transition
  result is **explicitly labelled data-limited**.

Justification of the material thresholds: 10 % holdout / 0.05 kernel-TV / 95 % action-agreement are the
points below which a further doubling of data does not change the fitted model or its induced policy in
a decision-relevant way; they are conventional "practically converged" margins and are set here without
reference to any performance outcome.

## 6. Focused hyperparameter sensitivities (OFAT around the primary)

One component at a time around (16 candidates, 16 paths, 8 starts, PBVI 32/H5, grid 41):

| Sensitivity | Settings | Needs refit? | Reuse |
|---|---|---|---|
| PLUS candidate bank | 8 / **16** / 32 | **Yes** (different bank) | none |
| process paths | 16 / **16** / 32 / 64 | **Yes** (different MC) | none |
| optimizer starts | **8** / 16 | **Yes** | none |
| PBVI belief points / backups | **32** / 64 (+larger horizon budget) | **No** | **reuses the fitted model** — plan-only |
| abundance grid | **41** / 61 | No refit, **re-discretize + re-plan** | reuses the fit; rebuilds POMDP |

Run the sensitivity study on a **single frozen budget** (the primary, once §5 sets it) and a **single
frozen difficult cell pair** (Amur/ricker/σ0.1 + Egyptian/regime/σ0.4) to bound cost; PBVI/grid arms are
plan-only (cheap). Report the same §4 diagnostics; **no return tuning.**

## 7. Comparison-fairness constraints

- **If the final primary budget is increased above 4,000:** every method in any *headline* comparison
  must receive the **same** permitted budget. The existing general-RL results (trained on 4,000) are
  **not directly matched** — either **rerun the general learners at the new budget**, or present the
  larger-budget ecological result strictly as a **separate data-sensitivity / oracle** comparison,
  never as a matched headline.
- **If 4,000 remains primary:** still run and report the larger-budget (8,000) ecological sensitivity,
  so that any weak ecological-baseline performance cannot be attributed, unexamined, to data scarcity.

## 8. Recommendation

**Do NOT proceed with the 32-cell 4,000-transition canary unchanged yet.** Reasons: (a) adequacy of
4,000 for the corrected mechanistic fits is **untested** — finite objectives in the earlier canary do
not establish it; (b) the coverage audit shows a real identification bottleneck (**Egyptian-vulture
capacity ≈68 rows / 35 episodes**) that could be data-limited; (c) the nested-budget adequacy study is
**cheaper (~140 CPU-h) and more informative** than committing 32 cells to a possibly-inadequate budget.

**Proposed sequence:** approve and run the **8-cell nested-budget adequacy study** (§3) under the
preregistered thresholds (§5) and hyperparameter sensitivities (§6), return-blind; then set the primary
offline budget from its outcome; **then** run the 32-cell diagnostic canary at the validated budget with
matched-budget fairness (§7). If you prefer to proceed at 4,000 in parallel, the 32-cell result must be
labelled **provisional / data-adequacy-unverified** until the study reports.

**Return-blind throughout. Nothing was run; no returns inspected; frozen manifests and runtime code
unchanged.**
