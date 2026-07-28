# Phase 1 Audit Report — Paper-Aligned PLUS / MOOR

**Protocol:** `CODE_SERVER_AUDIT_AND_FINISH_PAPER_ALIGNED_PLUS_MOOR.md`, Phase 1 (inspect & report only)
**Date:** 2026-07-18
**Auditor:** Claude (code-server), established from live code/tests/jobs — not from prior messages.
**No code was edited, no job launched, no artifact modified.**

> **Reference-availability finding (report before matrices):** the named controlling reference
> `PLUS_MOOR_PAPER_ALIGNMENT_COMPLETE_REFERENCE.md` **does not exist anywhere in the repository.** I
> audited against the in-folder controlling documents that encode the same requirements —
> `ECOLOGY_BASELINE_PAPER_FAITHFULNESS_PLUS_MOOR.md`,
> `REVIEW_PROVISIONAL_PAPER_ALIGNED_PLUS_MOOR_IMPLEMENTATION.md`,
> `DECISIONS_FOR_CORRECTED_PLUS_MOOR_PLAN.md` — plus the self-contained requirement matrices in the
> Phase 1 protocol itself. **Provide the named reference or confirm these are the controlling set.**

---

## A. Repository and execution identity

1. **Repo:** `/fs04/scratch2/ce25/Claude_DeepRL_Population_Models`
2. **Branch / commit:** `main` @ `5f9cf32a69d47e2d31b2ed6ee30ebbcfe84b8536`. **The corrected PLUS/MOOR
   code is uncommitted** — it lives in the dirty working tree, not in any commit. (No fresh code
   snapshot has been frozen for it.)
3. **Relevant dirty files:** `src/real_ecology_benchmark/{faithful_ecology,faithful_fit,faithful_pomdp,faithful_artifacts,config,privacy,pipeline,cli,realdata}.py`, `methods/{plus_faithful,moor_faithful,__init__}.py`, `planners/*`, corrected configs/scripts/tests.
4. **Method implementation files:**
   - PLUS: `methods/plus_faithful.py` → `faithful_fit.py` (bank/fit) → `faithful_pomdp.py` (POMDP/belief) → `planners/pbvi.py`; equations in `faithful_ecology.py`.
   - MOOR: `methods/moor_faithful.py` → same fit/pomdp/planner stack; one Ricker candidate.
5. **Config / registry:** `config.py::FaithfulModelConfig/FaithfulFitConfig/FaithfulPlannerConfig`; registry `methods/__init__.py:26-27` maps **only** `plus_adapted_mechanistic_pbvi`, `moor_adapted_ricker_misspec_pbvi`; `FAITHFUL_METHODS` at `:31-33`.
6. **Tests exercising each method:** `tests/real/test_corrected_adapted_mechanistic.py` (17), `test_moor_faithful.py`, `test_plus_faithful.py`, `test_faithful_equations.py`, `test_faithful_planners.py`, `test_faithful_privacy.py`, `test_faithful_artifacts.py` — **32 adapted/faithful tests; 154 total, all pass** (re-run).
7. **Jobs:** **0 active.** Corrected canary array `58372348/58372349/58372352` — **8/8 COMPLETED, exit 0:0**. Superseded earlier canary attempt (`58371927`) cancelled. Void provisional run had jobs cancelled/preserved.
8. **Artifact directories & provenance:**
   - `real_ecology_runs/paper_faithful_one_cell_4000_20260717_v1/` — **scientifically VOID** (has `VOID.md`); provisional pre-correction implementation; excluded from all claims. **Do not reuse.**
   - `real_ecology_runs/adapted_plus_moor_corrected_canary_20260717_v1/` — corrected canary; complete receipts; **but NO `acceptance.json`/`completion.json`** (see H).
   - `paper_faithful_smoke_20260717_v{1..4}/` — earlier provisional smokes, `SUPERSEDED.md`.

## B. Overall classification

**Both `plus_adapted_mechanistic_pbvi` and `moor_adapted_ricker_misspec_pbvi`: "Implemented but not
sufficiently verified."**

Justification: the code is complete and the two invalidating blockers are genuinely fixed (§E,
verified in source, not from comments). But the verification set is incomplete — the corrected
canary's **acceptance gate was never run**, there is **no PBVI-vs-exact-POMDP numerical validation**,
and **real-data parameter recovery is unverified at scale**. Neither method is "Complete and supported
by evidence" until those close. Neither is routed to old code (§C/§D leak rows PASS).

## C. PLUS component matrix

| Requirement | Status | Evidence |
|---|---|---|
| Registered mechanistic Ricker/Allee/theta/regime equations | **PASS** | `faithful_ecology.py::noiseless_next` (real `depensation_thresholds`, `theta_exponent`, `regime_matrix`); `test_faithful_equations.py` |
| Every candidate a complete POMDP (transition/obs/reward/context) | **PASS** | `faithful_pomdp.py::CandidatePOMDP` (`initial_belief`, `update`→`log_evidence`, `expected_public_reward`) |
| Public channel structural zeros | **PASS** | `faithful_fit.py::_action_layout:236`; `test_real_channel_map_has_exact_14_parameters_and_structural_zeros` |
| Candidate params fixed during deployment | **PASS** | `plus_faithful.py::observe` updates only posterior/beliefs; `test_fixed_pi_has_no_optimizer_coordinates` |
| Offline fit preserves episode order & resets | **PASS** | `_ordered_indices`; `test_ordered_objective_uses_within_episode_order` + `_is_invariant_to_whole_episode_permutation` |
| Per-candidate state belief | **PASS** | `plus_faithful.py` `internal_beliefs` list, one per POMDP |
| Each candidate solved independently | **PASS** | per-candidate `PointBasedPlanner` list |
| Online weights use sequential transition-obs evidence | **PASS** | `plus_faithful.py::observe` `log w += log_evidence` |
| Posterior normalizes & stays stable | **PASS** | log-space normalize with floor; `LIKELIHOOD_FLOOR` |
| Action = argmax posterior-weighted candidate PBVI values | **PASS** | `plus_faithful.py::act` `posterior @ candidate_scores` |
| Regime candidates: discrete latent regime, candidate-specific fixed `Pi` | **PASS** | `fixed_regime_matrix(p)`; `test_registered_regime_candidate_allocations_are_exact_nested_prefixes` |
| Same regime law in fit/kernel/filter/planner | **PASS** | `regime_law_hash`; `test_fitter_and_deployed_one_step_transitions_are_identical`, `test_pomdp_regime_transition_frequency_matches_fixed_pi` |
| Bootstrap construction deterministic & documented | **PASS** | `episode_bootstrap_map_fixed_pi_v2`; seeded; `build_candidate_bank` |
| PBVI does belief-state backups, not QMDP | **PARTIAL** | `pbvi.py` reachable-belief backups, explicit "no QMDP reduction"; **but not validated against an exact POMDP solution** (see G/H) |
| True hidden family/params cannot leak into candidates | **PASS** | `faithful_ecology.py`/`faithful_fit.py` import only numpy/config/dataset — no `realdata`/`resolve_actions`/`NativeSolver`; `test_faithful_privacy.py` |

Also: **candidate count** 16 = ricker 4 / allee 4 / theta 4 / regime 4; **prior** uniform; **`Pi` grid**
`{0.80, 0.90, 0.97}` → expected switches over 25 steps `25(1−p)` = 5.00 / 2.50 / 0.75, regime
allocation (1,2,1); **belief** discrete abundance×regime grid; **PBVI** finite-horizon reachable-belief
backups, seeded. **Old polynomial `plus_native` is NOT reachable** from the new ID (separate module, no import).

## D. MOOR component matrix

| Requirement | Status | Evidence |
|---|---|---|
| One Ricker fitted jointly to complete ordered episodes | **PASS** | `faithful_fit.py::fit_mechanistic_model`, `_trajectory_objective` over `ordered` |
| Episode boundaries reset abundance & capacity | **PASS** | `_trajectory_objective:444-445` resets `capacity`, `latent` per episode; `test_ordered_*` |
| Public channel structural zeros | **PASS** | `_action_layout`; 14-param count test |
| Objective propagates complete latent trajectories | **PASS** | per-episode step loop `:451-` |
| MC averaging in SSE uses process noise only | **PASS** (Blocker 1) | `:456` `latent = mean·exp(process_scale·process_noise)`; no survey draw |
| Observation noise NOT sampled in SSE | **PASS** | `test_ricker_objective_has_no_sampled_survey_noise_channel` |
| Predicted survey = `x·exp(σ_o²/2)` | **PASS** | `conditional_survey_mean_factor`; `:449,458` |
| Normalized, bounded, autodiff L-BFGS | **PASS** | `torch.optim.LBFGS` strong-wolfe; bounded transforms |
| Deterministic multiple starts | **PASS** | `fit` loops `starts`, seeded `_initial_raw` |
| Minimum training objective selects fit | **PASS** | `selected = min(finite, key=objectives)` |
| Holdout diagnostic-only | **PASS** | `holdout_normalized_survey_sse` reporting field; not in selection |
| Primary objective has no shrinkage/group/complementarity | **PASS** | `regularization_variant=none_structural_v1`; validation rejects nonzero legacy coeff |
| Selected model used unchanged in kernel/filter/planner | **PASS** | one `FittedModel` → POMDP/planner; `regime_law_hash` identity |
| Same Ricker across true families, no family label | **PASS** | `moor_faithful.py` always form=ricker; no `cfg.kind` read |
| PBVI belief-state backups, not QMDP | **PARTIAL** | as PLUS |
| True hidden params cannot leak into fit/norm/priors | **PASS** | channel-only context; `test_faithful_privacy.py` |

Also: **fitted vector** = reset(μ,σ), initial/ceiling capacity, process_scale, 14 action effects (8
signed rate + 5 capacity + 1 stocking), + Allee/theta thresholds where applicable; **`Pi` fixed**,
not estimated; **process paths** 16, CRN fixed per closure; **starts** multiple, min-obj selection,
non-finite→`inf` recorded; **grid** public-scale abundance bins + Gaussian obs kernel. **Old ridge/
scalar-`K` `moor_native` NOT reachable** from the new ID.

## E. Blocker recheck

**Blocker 1 (MOOR observation objective): FIXED.** `faithful_fit.py:449,458` compute
`mean((latent · survey_mean_factor − target)²)` with `survey_mean_factor = exp(σ_o²/2)`; the random
bank is `(initial, process, regime_uniforms)` — **no survey-noise bank exists**. Tests that would fail
on reintroduction: `test_ricker_objective_has_no_sampled_survey_noise_channel`,
`test_conditional_survey_mean_recovers_latent_scale_without_noise_draw`,
`test_zero_noise_fit_is_bit_deterministic_under_same_process_bank`.

**Blocker 2 (regime fit/deployment consistency): FIXED.** Fitter draws a **discrete** regime from CRN
uniforms under a **constant** `Pi`: `:446` initial `(u ≥ 0.5)`, `:461-462` transition
`(u > Pi[z,0])`; `Pi = torch.tensor(fixed_regime_matrix(p))` — a constant, not a `Parameter`, so no
gradient flows to `Pi` (the trap is avoided by construction). No `regime_prob` mean-field path remains
in `faithful_fit.py`. Tests: `test_fitter_and_deployed_one_step_transitions_are_identical`,
`test_pomdp_regime_transition_frequency_matches_fixed_pi`, `test_canonical_regime_law_uses_current_state_then_switches`, `test_fixed_pi_has_no_optimizer_coordinates`.

## F. Shared-framework audit

Both methods share: sanitized `MethodContext` (channel-only public metadata); one registered 80/20
episode split; public median-survey normalization; identical observation model (`exp(σ_o²/2)` mean,
lognormal likelihood); the shared public reward surrogate per reward mode; identical horizon/context
(previous survey + timestep + capacity); identical evaluation seeds/scenarios; identical artifact
schema/provenance fields; and the same fit-cache (`fit_transition_hash_v1`, reward-excluded).
**Intentional method-specific differences:** PLUS = 16-candidate cross-form bank + online model
posterior; MOOR = single misspecified Ricker, no posterior. Both are disclosed adaptations/extensions.

## G. Test & canary evidence

154 tests pass (re-run, `.venv-paper-faithful`); Ruff clean incl. without `E402,E731` on corrected
modules. Coverage of the protocol's 12 required areas:

| # | Area | Present? |
|---|---|---|
| 1 | mechanistic one-step equations | ✅ |
| 2 | episode reset & ordering | ✅ (both order halves) |
| 3 | structural zeros | ✅ |
| 4 | no private-info leakage | ✅ (privacy + relabel) |
| 5 | PLUS posterior norm & synthetic identification | ✅ partial (normalization ✅; in-bank identification ✅) |
| 6 | PLUS candidate value aggregation | ✅ |
| 7 | MOOR conditional-mean SSE | ✅ |
| 8 | MOOR synthetic param/trajectory recovery | ✅ (σ=0.4 scalar + ordered) |
| 9 | regime fit/kernel/filter consistency | ✅ |
| 10 | **PBVI vs small exactly solvable POMDP** | ❌ **NOT TESTED** — `test_faithful_planners.py` has no exact-solution reference |
| 11 | deterministic repeatability | ✅ |
| 12 | provenance & method-ID routing | ✅ (`test_only_adopted_external_method_ids_are_registered`) |

**Canary:** ran, 8/8 exit 0, complete receipts for 2 cells × 2 methods × 2 rewards. **No acceptance
run** — the acceptance script (`scripts/run_paper_faithful_acceptance.py`) was never executed, so the
canary is unvalidated by the project's own gate. Measured cost: PLUS fit 3.7–4.1 h/cell, MOOR fit
~0.3 h; PLUS plan ~1.4 h/(cell×reward), MOOR plan ~0.1 h.

## H. Remaining-work plan (severity-ordered)

**Blocking experiment validity**
- **H1 — Canary acceptance never run.** Why: without it, finite-fit/model-consistency/privacy validity
  of the corrected canary is unconfirmed. Files: `scripts/run_paper_faithful_acceptance.py` (or adapted
  equivalent) vs `real_ecology_runs/adapted_plus_moor_corrected_canary_20260717_v1/`. Fix: run it,
  produce `acceptance.json`. Tests: none new. Runtime: minutes. Depends on: nothing.
- **H2 — Return-blind discipline at risk.** Plan-stage returns (`operational_return`/`true_return`) are
  on disk. Freeze the subset manifest+hashes **before** any return is inspected. Fix: procedural.

**Required verification missing**
- **H3 — No PBVI-vs-exact-POMDP validation (G10).** Why: PBVI's numerical correctness (belief backups,
  not just non-QMDP structure) is unverified. Files: `planners/pbvi.py`, new `tests/real/test_faithful_planners.py`.
  Fix: add a tiny exactly-solvable POMDP and assert PBVI ≈ exact value/policy within tolerance.
  Runtime: seconds. Depends on: nothing.
- **H4 — Real-data parameter recovery/identifiability unverified at scale.** Only synthetic recovery is
  tested; no evidence the 14-parameter fit is identifiable on real 4,000-row cells. Fix: read the
  canary fit diagnostics (coverage, condition, holdout) once acceptance runs.

**Performance / runtime risk**
- **H5 — Full sweep ≈ 1,150 CPU-h** (PLUS dominates). Subset ≈ 250 CPU-h (one wave ≤256 CPU). The
  32-cell diagnostic subset carries **no CPU ceiling** (`requested_cpu_ceiling_hours` empty for
  fit/plan rows). Fix: set a subset ceiling; PI budget for the sweep.

**Documentation / naming only**
- **H6 — Undocumented deviation from approved plan text (M1):** code uses an asymmetric signed-rate
  transform on `[-1.5, 2.0]`; the approved plan specified symmetric `r_max·tanh(q)`. Benign/arguably
  better, validation-pinned — but record it.
- **H7 — Controlling reference `PLUS_MOOR_PAPER_ALIGNMENT_COMPLETE_REFERENCE.md` is missing.** Provide
  it or designate the in-folder docs as controlling.
- **H8 — Corrected code is uncommitted / no frozen snapshot.** Before any registered run, freeze a code
  snapshot (as done for the void run) so the sweep has immutable provenance.

**Design decisions pending (PI)**
- Subset vs. full sweep + CPU ceiling `B_full`; whether inferential parameter claims will be made
  (governs whether profile/bootstrap intervals are required).

---

## Phase 1 recommendation

### `READY FOR LIMITED CANARIES ONLY`

The scientific implementation is sound and both invalidating blockers are genuinely fixed, so no
scientific code change is required to proceed to limited/verification runs. But it is **not** ready to
propose a full experiment: close **H1** (run the canary acceptance gate), **H3** (add the
PBVI-vs-exact test), and **H8** (freeze a snapshot); resolve **H5/H6/H7** and the PI budget decision.
The full ecological-baseline experiment requires separate approval after that evidence is reviewed.

**Stopping here per protocol — no code edited, no job launched, no Phase 2 work performed.**
