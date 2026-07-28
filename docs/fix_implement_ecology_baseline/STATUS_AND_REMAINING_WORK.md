# PLUS/MOOR paper-aligned baselines — status & remaining work

**Updated:** 2026-07-18
**One-line status:** corrected code implemented and audited (approved); return-blind runtime canary
run and completed; **no scientific experiment has been run yet** and the canary's own acceptance gate
was never executed.

This file is the single entry point for the current state. The full back-and-forth that produced it
is archived under `archive/` and is not needed for day-to-day work.

---

## 1. What this workstream is

The original `plus_native` / `moor_native` "mechanistic" baselines were found to be **polynomial
regressions wearing ecological labels** (`ricker→[1,x]`, `allee→[1,x,x²]`, …), with no `r`, `K`, Allee
threshold, theta exponent, or regime transition matrix, and no trajectory propagation. They are
therefore *ecology-inspired approximations*, not implementations of published PLUS/MOOR — a fact that
narrows what the completed hidden-r/K headline (`hidden_rk_comparison_20260716`) can claim.

This workstream builds **genuinely mechanistic, paper-aligned** replacements as **new** methods,
leaving the old ones and all completed runs untouched.

- New method IDs: **`plus_adapted_mechanistic_pbvi`**, **`moor_adapted_ricker_misspec_pbvi`**.
- These are **paper-aligned adaptations, never reproductions** (actions, observations, reward, planner,
  and multi-episode fitting all differ from the source fishery papers).

## 2. Controlling documents (read these; the rest is archive)

| File | Role |
|---|---|
| `ECOLOGY_BASELINE_PAPER_FAITHFULNESS_PLUS_MOOR.md` | External controlling scientific spec (the problem statement). |
| `REVIEW_PROVISIONAL_PAPER_ALIGNED_PLUS_MOOR_IMPLEMENTATION.md` | External paper reviewer's findings (F1–F6). |
| `DECISIONS_FOR_CORRECTED_PLUS_MOOR_PLAN.md` | The five settled scientific decisions. **Do not reopen.** |
| `CORRECTED_PLUS_MOOR_IMPLEMENTATION_PLAN.md` | The approved implementation plan actually built. |
| `paper_faithful_fit_requirements.lock` | Pinned CPU-PyTorch environment for the fitter. |

Source of truth for the **equations as built**: `src/real_ecology_benchmark/faithful_ecology.py`
(mechanistic forms), `faithful_fit.py` (objective/fitter), `planners/pbvi.py` (planner).

## 3. What is DONE and verified

Independently audited (`archive/CLAUDE_AUDIT_CORRECTED_PLUS_MOOR_IMPLEMENTATION.md`) — verdict
**approve**:

- Four genuinely mechanistic equations (real Allee `C`, theta exponent, 2×2 regime simplex `Π`);
  no polynomial-label defect.
- **F1 fixed** — MOOR objective uses conditional survey mean `x·exp(σ_o²/2)`, no survey-noise draw.
- **F2 fixed** — regime is fit and deployed under one **fixed discrete** `Π` (mean-field path removed);
  the zero-gradient trap is avoided because `Π` is a constant, not a fitted parameter.
- 44 → 14 action parameters via public `channel`-driven structural zeros + signed `r_a`.
- Three regularizers removed from the primary objective (separate `hierarchical_weak_v1` fallback).
- Reward-independent fit cache keyed on transition fields only; PBVI still per reward mode.
- Only the adapted IDs are registered; old natives and the frozen 20260716 run untouched.
- **154 tests pass; Ruff clean** (both re-run independently).

## 4. Runtime — MEASURED (corrected code, from the canary)

| Stage | PLUS | MOOR |
|---|---|---|
| fit / dynamics cell | **3.7–4.1 h** | **~0.3 h** |
| plan / (cell × reward) | **~1.4 h** | **~0.1 h** |

Projected **288-cell headline sweep ≈ 1,100–1,150 CPU-h** (PLUS dominates). **32-cell diagnostic
subset ≈ 250 CPU-h** — one wave inside the 256-CPU limit, ~4 h wall.

## 5. REMAINING WORK / OPEN PROBLEMS (ordered)

### Blocking before any result is trusted

1. **The canary has no acceptance gate.** `acceptance.json` / `completion.json` are **absent** from
   `real_ecology_runs/adapted_plus_moor_corrected_canary_20260717_v1/`. All 8 rows exited 0 with
   complete receipts, but `scripts/run_paper_faithful_acceptance.py` (or the adapted equivalent) was
   never run, so finite-fit / model-consistency / privacy validity is **unconfirmed by the project's
   own check**. Run it before treating the canary as passed.

2. **Return-blind discipline is at risk.** The canary's plan rows wrote `operational_return` /
   `true_return` to disk (expected for plan-stage rows). Pre-registration requires that **no one
   inspects those values** until the subset scope/manifest is frozen. Freeze the subset manifest and
   hashes *before* opening any corrected return.

### Known code-version issues (minor; from the implementation audit)

3. **Silent plan deviation (M1).** The approved plan text specified a *symmetric* `r_a = r_max·tanh(q)`
   transform; the code uses an *asymmetric* bounded transform on `[-1.5, 2.0]`, validation-pinned. It
   is arguably better (contains the true `r∈[-0.4753,0.0707]` with headroom) but is **undocumented
   drift from approved text** — record it in the plan.

4. **Diagnostic subset has no CPU ceiling (M2).** `requested_cpu_ceiling_hours` is set for
   `sensitivity` (600) and `canary` (48) but **empty** for the 64 `diagnostic_fit` / 128
   `diagnostic_plan` rows — the largest block. Give it a ceiling.

5. **Sensitivity-suite cost stated but not gated (from plan audit F1).** Five PLUS configs on 32 cells;
   projected but only ceiling-tagged (600 CPU-h), not gated the way the full sweep is.

### Not yet run (the actual science)

6. **32-cell diagnostic subset** — manifests generated (64 fit / 128 plan rows) but **not submitted**.
7. **288-cell headline sweep** — **not authorized, not run.** Needs a PI CPU budget on the measured
   ~1,150 CPU-h.

### Decisions still owned by the PI

8. **Subset vs. full sweep** — pick scope and supply a CPU ceiling `B_full`. The plan's gate is
   `1.5·C_cache ≤ B_full`; if it fails, only the registered subset runs (diagnostic, not a headline).
9. **Whether inferential parameter claims will be made** — if not (recommended for a first run), the
   omitted profile/bootstrap intervals are legitimately skipped; state it.

## 6. Recommended next steps

1. Run the canary acceptance check → confirm the canary is actually valid (cheap; unblocks item 1).
2. Record M1; add the M2 subset ceiling (doc/config edits, no rerun).
3. PI: choose subset vs sweep and supply the CPU ceiling.
4. Freeze the chosen manifest + hashes (return-blind), then launch at full 256-CPU capacity.
5. Only after that, open returns and analyze.

## 7. What must NOT happen (pre-registration guardrails)

- Do not inspect corrected returns before the run scope is frozen.
- Do not reopen the five decisions in `DECISIONS_FOR_CORRECTED_PLUS_MOOR_PLAN.md`.
- Do not relabel `plus_native`/`moor_native` or the frozen 20260716 run as paper-faithful.
- Do not present the canary as a scientific result — it is a runtime/interface measurement only.
- The void run `paper_faithful_one_cell_4000_20260717_v1/` stays preserved and excluded from all
  claims.
