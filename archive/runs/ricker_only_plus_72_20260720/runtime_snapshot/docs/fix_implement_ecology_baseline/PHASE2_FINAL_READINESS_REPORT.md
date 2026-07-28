# Phase 2 — Final Readiness Report (paper-aligned PLUS / MOOR)

**Date:** 2026-07-18
**Controlling reference:** `PLUS_MOOR_PAPER_ALIGNMENT_COMPLETE_REFERENCE.md`
**Decision applied:** Option A — retain the approved asymmetric `r_a` parameterization.
**Blindness preserved:** no performance return (`operational_return`/`true_return`) was inspected at
any point.

---

## H. Recommendation (up front)

### `READY FOR REGISTERED 32-CELL ECOLOGICAL CANARY`

All verification gates in reference §14 pass, both invalidating blockers remain fixed, PBVI is
validated against an exact POMDP, MOOR's single-Ricker isolation is proven, privacy/routing hold, and
the existing canary — proven to run byte-identical runtime code — passes structural acceptance. The
one non-code prerequisite before launch is registering a CPU ceiling for the 32-cell subset (a
manifest/config item, not a correctness blocker). The full 288-cell sweep still requires separate
budget approval after the 32-cell canary.

## A. Reconciled PLUS / MOOR matrices (vs the controlling reference)

All rows PASS. The only Phase-1 PARTIAL — PBVI-vs-QMDP — is upgraded to **PASS** by the new exact test.

| Reference §14 gate | PLUS | MOOR | Evidence |
|---|---|---|---|
| Candidate kernels = declared mechanistic equations (§5) | PASS | PASS (Ricker) | `faithful_ecology.py::noiseless_next`; `test_faithful_equations` |
| Action structural zeros from public channels | PASS | PASS | `_action_layout`; `test_structural_zeros_from_public_channels` |
| Episode order + reset boundaries | PASS | PASS | `test_ordered_objective_*` |
| Conditional-mean survey SSE, process-only MC, no obs-noise draw | — | PASS | `faithful_fit.py:449,458`; `test_ricker_objective_has_no_sampled_survey_noise_channel` |
| Each candidate solved independently; posterior from sequential evidence | PASS | n/a | `plus_faithful.py::observe/act` |
| Same discrete `Π` in fit/kernel/filter/planner | PASS | n/a | `test_fitter_and_deployed_one_step_transitions_are_identical` |
| Same fitted model in kernel/filter/planner | PASS | PASS | `regime_law_hash` / single `FittedModel` |
| Multi-start deterministic, min-train selection, holdout diagnostic-only | PASS | PASS | `fit_mechanistic_model` |
| No primary shrinkage/group/complementarity | PASS | PASS | `regularization_variant=none_structural_v1` |
| **PBVI belief-state backups, not QMDP** | **PASS** | **PASS** | **`test_pbvi_matches_exact_and_diverges_from_qmdp…`** |
| No hidden family/private parameter leakage | PASS | PASS | `test_faithful_privacy`; imports (numpy/config/dataset only) |
| Old `plus_native`/`moor_native` unreachable from adapted IDs | PASS | PASS | registry maps only adapted IDs; no `native_*`/`realdata`/`resolve_actions` imports |

**Both blockers remain fixed** (re-verified in source): B1 conditional-mean/process-only
(`faithful_fit.py:449,458`); B2 discrete fixed-`Π`, no mean-field (`:446,461-462`, `Π` a constant).

## B. `r_a` resolution (Option A retained)

`r_a = 0.25 + 1.75·tanh(q)`, `r_a ∈ [-1.5, 2.0]`, then `g_a=max(r_a,0)`, `h_a=max(-r_a,0)`. This is the
**approved bounded optimizer parameterization** (matches `CORRECTED…PLAN.md` §3 and is compatible with
reference §5, which fixes only the max-split). It is documented as a bounded optimizer coordinate;
`q=0 → r=0.25` is the **raw-coordinate centre, not a statistical prior** (no prior distribution is
imposed; the fitter uses explicit deterministic multi-starts). New tests
(`SignedRateTransformTests`): transform vs formula, asymptotic bounds `→(-1.5, 2.0)`, `q=0→0.25`,
Clarke sub-gradients (`0 / 0.5 / 1`), exact max-split with `g·h=0`, and channel-driven structural
zeros (14 free action params).

## C. PBVI-vs-exact POMDP test (results)

Built a deterministic 4-state "probe-to-reveal" POMDP (types × {normal, revealed}) with
**action-independent** observations — the same observation interface the real `CandidatePOMDP`
exposes — where active information gathering strictly helps. The test drives the **real
`PointBasedPlanner.action_values`** (the identical backup used by PLUS and MOOR). At an uncertain root
belief:

- **Exact optimum = probe** (gather information first); **QMDP = commit** (blind) — QMDP **diverges**;
- **PBVI = probe = exact** action → PBVI is doing belief-state backups, **not** QMDP;
- **Supported claim (exact wording):** *In the registered exact-POMDP test, PBVI selected the exact
  optimal information-gathering action and matched that action's exact value within 1e-5. Values for
  suboptimal actions remained approximate, as expected for point-based planning.* (No general value
  exactness is claimed.) A second test confirms PBVI commits correctly when the type is known.

## D. Test results (exact commands + output)

```
PYTHONPATH=src .venv-paper-faithful/bin/python -m pytest -q
    → 166 passed, 14 subtests passed  (was 154; +the new verification suite)
.venv-paper-faithful/bin/ruff check … (corrected src + new test, no repo-wide ignores)
    → All checks passed!
```

Focused scientific gates re-run and passing: conditional-mean SSE & no-sampled-survey (B1); discrete
regime fit/kernel/filter/planner consistency (B2); PLUS posterior normalization & synthetic
identification; candidate-specific PBVI aggregation; episode order/reset; structural zeros;
`test_faithful_privacy` (table-independence, family relabel, forbidden-name scan); adopted-ID-only
routing; deterministic repeatability; **PBVI-vs-exact** (new); **MOOR mask** (new); **regime count**
(new).

## E. Frozen implementation identity

- **Runtime code digest** (content-only, `src/real_ecology_benchmark/*.py`):
  `f70ec7122263ec00da0a4950994dfe483ed29c71bc15e477007ca9ae0a6d2333`.
- **Full snapshot digest** (`src` + `configs`, 71 files, via `hash_paper_faithful_snapshot.py`):
  `0599e53e2d8257dc2513cda07bb4847037fde3799363b8ef995f43b091e34c69`.
- **This phase changed only tests and documentation** — no runtime `src/` file was modified (verified:
  no `src/**.py` mtime today; new file is `tests/real/test_phase2_verification.py`; docs are the
  regime-count correction + reports).
- **Remaining dirty files / git note:** the corrected PLUS/MOOR implementation is uncommitted and
  **intertwined** in shared files (`config.py`, `pipeline.py`, `methods/__init__.py`, …) with the
  larger uncommitted hidden-r/K benchmark tree (~96 changed/untracked entries). A git commit
  "containing only PLUS/MOOR" is therefore not cleanly separable. The **authoritative freeze is the
  digest above** (the mechanism the canary records and this project uses); I did **not** create a
  git commit bundling unrelated work. If you want a durable git anchor, say so and I will branch from
  `main` and commit a labelled snapshot of the full working state.

## F. Canary validation (no returns inspected)

The runtime code the existing canary ran is **byte-identical** to the current frozen runtime — proven:
canary-frozen runtime digest `f70ec712…` **==** live `f70ec712…`, and the faithful configs are
identical. Per the protocol ("tests/documentation-only changes do not require a fresh performance
canary"), the **existing canary validates the frozen implementation**; no fresh canary is required.

Ran the pre-existing structural acceptance checker (predates the canary, does not read returns):

- **`acceptance.json`: passed = True**, 8 rows, `return_fields_opened = False`.
- Structure: 4 MOOR rows (pomdp_artifact_count = 1, single Ricker) + 4 PLUS rows (= 16 candidates); all
  `finite_objectives = True`.
- **Convergence** (fit diagnostics, not returns): all candidate objectives finite; MOOR selected
  gradient-norms [0.26, 2.15, 4.04], PLUS (64 candidates) [1.6e-3, 0.08, 16.8] — finite but not all
  near-zero, i.e. some candidates sit at bounds / are weakly identified on scarce data. This is a
  reportable fit-quality observation, expected for bootstrap candidates, and does not affect
  structural acceptance.
- **Runtime / CPU / memory** (sacct, 2 CPU/task): PLUS fit 4:05:25, peak RSS ~554 MB; MOOR fit 20:08,
  ~552 MB; PLUS plan ~1:25/cell, ~240 MB. All well under 4 GB. **Projected 32-cell subset ≈ 250
  CPU-h; full 288 ≈ 1,150 CPU-h.**
- Reproducibility: deterministic-repeatability unit test passes; the frozen snapshot digest is
  reproducible via the published recipe.

## G. Remaining findings and limitations

- **Task 3 — regime switch-count convention corrected.** A complete episode has **25 transition steps**
  → 25 regime states `z_0..z_24` → **24 inter-state switch opportunities**. Realised expected switches
  = **24(1-p) = 4.80 / 2.40 / 0.72**, not the previously reported `25(1-p)` (the objective draws a 25th
  update `z_25` that is never used). Corrected in `CORRECTED…PLAN.md`; empirically asserted by
  `RegimeSwitchCountTests`. Process unchanged.
- **Task 4 — MOOR parameter vector.** For `moor_adapted_ricker_misspec_pbvi` (form=ricker) the fitted
  vector is: `reset_mean, reset_scale, initial_capacity, capacity_ceiling, process_scale` + `1 baseline
  + rate-action` signed rates + capacity increments + stocking. **Allee `C`, theta exponent, regime
  multipliers, and `Π` are set to fixed constant defaults, consume no fitted coordinate, and are never
  read by the ricker branch** (`num_regimes=1`, no regime sampling). New `MOORRickerMaskTests` proves
  that perturbing all of them leaves one-step dynamics, the discretized transition kernels, and PBVI
  action-values byte-identical — they cannot influence MOOR fitting, kernel, filter, or planning.
- **Task 5 — known-r,K mapping (report only; not implemented).** From the simulator equations and
  private table: in set-point mode **population intrinsic `r_base = 0` (unused)** — it maps to **no**
  fitted coordinate, so revealing "population r" is uninformative for the `r_a`. The **per-action
  effective rate is the action set-point `ρ(a)`** (a private action-effect), which maps directly to the
  fitted **`r_a`**; these are action effects, not a single population r. For capacity:
  `K_eff = clip(K_base + κ, K_base, K_max)`, so **true `K_base` ↔ `k_0`** (baseline) and **true `K_max`
  ↔ `k_max`** (ceiling) on scale `S` (`k = K/S`) — **two distinct true quantities**; revealing "K" fixes
  both only if both `K_base` and `K_max` are given, and per-action `d_a` still corresponds to the
  private capacity increments. **Consequence for the later ablation:** a clean **known-K-only** mode is
  implementable (fix `k_0, k_max` from the true `K_base/K_max` on scale `S`; keep `r_a, d_a` fitted). A
  clean **known-r** mode is *not* cleanly separable — the benchmark has no intrinsic population r
  distinct from the per-action set-points, so revealing "r" is equivalent to revealing action effects.
  This is a switch on *which coordinates are fixed-from-truth vs fitted*, behind a new method ID.
- **Limitations / disclosed approximations:** PBVI is a disclosed computational substitute for
  SARSOP/DESPOT (value-approximate on suboptimal actions); some PLUS bootstrap candidates are weakly
  identified on scarce data (finite but non-zero gradients); the 32-cell subset needs a registered CPU
  ceiling before launch; runtime measurements are from the (validated) existing canary.

---

**Stopped per protocol. No 32-cell subset or full sweep launched; no performance returns inspected.**
