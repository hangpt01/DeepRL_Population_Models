# Code Review — Codex "latest-doc alignment" changes

Date: 2026-07-05
Reviewer: Claude (Opus 4.8)
Scope reviewed: `discrete_action_cont_obser/real_ecology_cont_obser/`

Sources of truth used for this review:

- `docs/29_6_Real_Ecology_Setting_Implementation_Plan.tex` (E5/E6/E6')
- `docs/29_6_Real_Ecological_Data_Actions_and_Costs.tex`
- `docs/29_6_REAL_ECOLOGY_IMPLEMENTATION_ALIGNMENT_GUIDE.md`
- `docs/HANDOFF_real_ecology_data_setting.md`

Codex touched: `realdata.py`, `config.py`, `controls.py`, `types.py`,
`methods/ogsrl.py`, and added `tests/test_real_ecology.py::TestRealDataAndConfigContract`.

Note: `real_ecology_cont_obser/` is **untracked in git**, so there is no diff to
review against — findings below are from reading the current files.

---

## Verdict

The five changes Codex reported are real, land where it says, and are consistent
with the specs. Tests pass (31) and the smoke run is `ok`. The work is a correct
*doc-alignment* pass. The important caveats are (a) one substantive science
interaction that the raised safety floor **exposes** and Codex left unresolved,
and (b) two spec items that are still not implemented. Codex's own honesty about
the pooled-agent mismatch is accurate.

---

## What is correct and aligned

1. **Data loader (`realdata.py`).** Prefers `revised_cost_action_table_v2/` and
   falls back to the vendored `revised_cost_action_table/`; validates 9
   populations, 11 actions, every `(population, action)` row, required columns,
   and that every set-point sits inside the per-population/per-family data caps
   (`load_effects`, fail-loud). Matches guide §4/§11 and handoff §Core-Spec.

2. **No silent clipping of valid rates.** `private_r_eff` (real_setpoint branch)
   now **raises** when `rho` is outside the data caps instead of clamping, with
   `clip` kept only as a post-validation numerical guard. Matches guide §6 step 3
   ("the clip is a guard; it should not silently change valid table values").

3. **Reward-leakage surface removed (`types.py`).** `PublicTransition` no longer
   carries `reward`. Verified end-to-end:
   - `evaluator.py:110` constructs `PublicTransition` with only
     `observation/done/truncated/public_info`;
   - realized reward is still logged separately (`operational_return += … result.reward`
     at `evaluator.py:117`) — logged for evaluation, never handed to `observe`;
   - no `methods/*.py` reads `result.reward`; the only `dataset.rewards` uses are
     offline Bellman targets in `delphic.py` (allowed by the spec).
   This is a *structural* guarantee (a method literally cannot receive reward
   online), which is stronger than the behavioral test the handoff asked for.

4. **Set-point initial control (`controls.py`).** `initial_controls` seeds `rho`
   with the `a0` set-point rate for `real_setpoint`, so reset starts at the real
   baseline regime (test `test_initial_public_rate_is_baseline_action` covers the
   Egyptian vulture, whose baseline `r<0`).

5. **Benefit uses `K_base`, fixed per episode.** `real_environment` sets
   `K_ref = pop.K_base` (`config.py:187`); reward benefit
   `alpha·s'/(s'+K_ref)` therefore tracks `K_base`, not `K_eff` (spec E6, plan
   line 205). Confirmed by `test_benefit_half_at_capacity`.

6. **OGSRL guardian feature space (Task C).** `KNNGuardian.features` now uses
   `log1p(s/K_ref)`, an unsafe indicator, a safety-distance term, an action
   one-hot, and population-scaled public controls (`rho`, `kappa/K`, `K_eff/K`)
   — no `reward_mode`, no private state. Matches Task C exactly; the hard-coded
   `500` is gone.

7. **Depletion-aware safety fraction (`config.default_safety_fraction`).** The
   tiered rule (0.10 healthy → 0.20 → 0.25 small/depleted) is **explicitly
   endorsed** by the spec (plan E6, lines 313–321), so this is aligned, not
   invented. The old absolute floor `50` is gone.

---

## Issues to resolve

### A. HEADLINE — the raised floor makes the `safe` penalty dead for depleted starts

This is the one finding that is a real, not cosmetic, problem, and it is caused
by the interaction of two of Codex's own changes.

- The depletion-aware floor now sets `s_safe(Egyptian vulture) = 0.25 · 325 =
  81.25`, while `N0 = 41`. Codex's **own test asserts** `s_safe > N0`
  (`test_depletion_aware_safety_defaults`, lines 225–230), i.e. the vulture
  **starts below the safety floor**.
- The `safe` penalty is **crossing-only**:
  `1[s_t > s_safe ∧ s_{t+1} ≤ s_safe]`. If the trajectory begins below `s_safe`
  and stays there, the antecedent `s_t > s_safe` is never true, so the penalty
  **never fires** — the `safe` mode collapses to `yield` for exactly the
  depleted/sink populations it is supposed to protect (vulture, and by the same
  logic the bottlenose-dolphin sink).
- Plan E6 (lines 326–334) flags precisely this and says **"resolve this before
  calibration."** Codex raised the floor (which widens the hole) but left the
  crossing-vs-occupancy decision unmade.

Recommendation: make the penalty an explicit design choice — downward-crossing
`1[cross]`, below-safe occupancy `1[s_t ≤ s_safe]`, or a mixture — before any
`safe`-mode calibration or run. Until then, `safe`-mode headline numbers for the
sinks/depleted cells are not meaningful.

### B. Absolute minimum-viable-population floor not implemented

Plan E6 (lines 322–325) requires reporting an absolute MVP floor
(`N_mvp ≈ 20–50`) **alongside** the relative `c_safe(p)·K_base` floor. Only the
relative fraction exists; no MVP floor is computed or reported. Reporting-level,
but spec-required for the write-up.

### C. Magic thresholds + one population untested

`default_safety_fraction` hard-codes cutoffs (`K_base ≤ 75`, `≤ 120`,
depletion `< 0.20`, `< 0.50`). These are reasonable and deterministic but are
**Codex's calibration choices, not spec values** — record them as a chosen
convention that must be revisited when `P_safe` is calibrated.
`test_depletion_aware_safety_defaults` pins expected fractions for **8 of the 9**
populations; **Iberian lynx** is missing from the expectation dict, so its
fraction is unverified. Add it.

### D. Stale Tier-3 defaults linger in `BeliefState.features`

`types.py:84` still defaults `K_ref=500.0, safety_threshold=50.0` — the exact
absolutes the spec calls "nonsensical." All real callers pass `env_cfg` values,
so this is dormant, but it is a latent trap. Prefer removing the defaults (make
them required) so a future caller can't silently reintroduce the `50` floor.

### E. Behavioral leakage regression (handoff Task B items 2–4) not added

Codex added a *structural* test (`PublicTransition` has no `reward` field). That
does guarantee `observe()` cannot receive reward. The stronger behavioral test
the handoff requested — "two rollouts with identical `(o,a,public)` but different
rewards produce identical belief/action" — was not added. Given the structural
guarantee this is low priority, but note the handoff item is only partially met.

### F. Data table provenance — confirm before any run (handoff Task D)

There is **no `revised_cost_action_table_v2/`** in this checkout, so the loader
uses `revised_cost_action_table/`. The alignment guide (§1) asserts v2 is "the
current table folder in this workspace" — that is not true here. A sibling
`discrete_action_cont_obser/real_ecology_data/` exists with the same file names.
Before any experiment, confirm with the user which folder is authoritative and
that the vendored table is the intended latest; regenerate datasets/caches if it
changes. Not a code defect — a run-blocker.

### G. Pooled-agent architecture still missing (Codex disclosed this correctly)

Guide §5 mandates **one pooled agent over all nine populations** with population
identity observed; recoverable-vs-sink is a reporting stratifier, not a training
split. The package is still shaped around single-population cells: `K_base`,
`K_ref`, `s_safe`, and the OGSRL guardian are all per-config-cell. Making it
truly pooled needs the dataset schema, filter context, method features, and
runners changed together. Codex's caveat is accurate; this is the largest
remaining gap and should be the next planned unit of work.

---

## Suggested next actions (in order)

1. Resolve the crossing-vs-occupancy safety-penalty decision (Issue A) — this
   blocks meaningful `safe`-mode results and is spec-mandated "before
   calibration."
2. Add the absolute MVP floor reporting (Issue B).
3. Add Iberian lynx to the safety-fraction test; annotate the tier thresholds as
   a calibration convention (Issue C).
4. Confirm the authoritative data table with the user (Issue F).
5. Scope the pooled-agent migration as a separate, deliberate change (Issue G).
