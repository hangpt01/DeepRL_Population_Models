# Phase 2 — Provenance, Preregistration & Budget Package

**Date:** 2026-07-18
**Status:** package for approval. **No jobs submitted. No performance returns inspected.**
**Controlling reference:** `PLUS_MOOR_PAPER_ALIGNMENT_COMPLETE_REFERENCE.md`.

---

## 1. Corrected PBVI claim wording

Adopted verbatim (in `PHASE2_FINAL_READINESS_REPORT.md` §C and the test comment):

> In the registered exact-POMDP test, PBVI selected the exact optimal information-gathering action and
> matched that action's exact value within 1e-5. Values for suboptimal actions remained approximate,
> as expected for point-based planning.

No general value-exactness is claimed.

## 2. Durable recoverable snapshot

| Field | Value |
|---|---|
| Branch | `snapshot/paper-faithful-plus-moor-20260718` |
| Commit | `fd50c38c9fad9ca113c224826b9bebddedea67d0` |
| Label | full benchmark-state **source** snapshot (returns excluded) |
| Files in snapshot | **460** (src 52, tests 24, configs 19, scripts 44, docs 205, `real_ecology_data` 7, `baseline_original` 103, build files) |
| Runtime code digest | `f70ec7122263ec00da0a4950994dfe483ed29c71bc15e477007ca9ae0a6d2333` |
| Full snapshot digest (src+configs) | `0599e53e2d8257dc2513cda07bb4847037fde3799363b8ef995f43b091e34c69` |
| File manifest (path + per-file sha256 + bytes) | `docs/fix_implement_ecology_baseline/SNAPSHOT_FILE_MANIFEST_20260718.json` |
| Manifest sha256 | `6d4f269b51426d3a49a2b5beaa8b0e302d6b3a21dfa30be07e282737bcea820d` |

- **`main` history and working tree are unchanged** — the snapshot is a separate branch; the amend that
  stripped returns was done in an isolated git worktree, never touching `main`.
- **Shared hidden-r/K files are included** (`config.py`, `pipeline.py`, `methods/__init__.py`, etc.);
  this is not a PLUS/MOOR-only commit.
- **Result returns excluded:** the 586 pre-existing `real_ecology_runs/` files that `main`'s HEAD
  already tracked were removed from the snapshot; the manifest asserts 0 return/`summary.json`/`.npz`
  files.
- **Remaining untracked / external (not in snapshot, by design):** `real_ecology_runs/` (all run
  outputs incl. returns and the pending manifests); `.venv-paper-faithful/` (external CPU-PyTorch env,
  reproducible from `paper_faithful_fit_requirements.lock`, which *is* in the snapshot). Nothing else
  required to reproduce the implementation is external.
- **Recoverability:** `git checkout fd50c38c` (or `git worktree add … fd50c38c`) reconstructs the
  complete source; the manifest independently verifies every file's content hash.

## 3. Registered 32-cell manifest and hashes

**Prepared, not submitted.** Location: `real_ecology_runs/adapted_32cell_diagnostic_pending/manifests/`.

Matrix: {Amur tiger, Egyptian vulture} × {ricker, allee, theta, regime} × σ∈{0, 0.1, 0.2, 0.4} = **32
dynamics cells**; corrected `plus_adapted_mechanistic_pbvi` + `moor_adapted_ricker_misspec_pbvi`;
**one preregistered primary reward mode = `safe`** (conservation-relevant; matches the prior
hidden-r/K comparison arm). Frozen 80/20 episode split and frozen fit/candidate/PBVI/eval seeds are
inherited from the frozen runtime config.

| Manifest | Rows | sha256 |
|---|---:|---|
| `diagnostic_fit_64.csv` | 64 | `a8f39d83eb70a32be2c42fabecc56ca4af4d3d40edb97f87242934085fd7ee86` |
| `diagnostic_plan_safe_64.csv` | 64 | `f0322718a78d485827c39661e2f1ded4cbde1d8bccd4cd053f4d1fe93dec087d` |

### Exact job-count arithmetic

- **Fit jobs = 64** = 32 dynamics cells × 2 methods. **Fitting is reward-independent** (keyed on the
  transition-field hash), so each cell×method is fit **once**.
- **Plan+evaluation jobs = 64** = 32 cells × 2 methods × 1 reward mode (`safe`). Planning/evaluation
  is reward-dependent (PBVI reward is the shared surrogate for that mode).
- **Total = 128 jobs** for the primary run.
- **Reuse:** if the second reward mode (`yield`) is added later it reuses all **64 fits** unchanged and
  adds only **64** plan+eval jobs (0 new fits). No fit is repeated across methods or reward modes.

## 4. Registered CPU ceiling (from measured canary runtime)

Measured (validated canary, 2 CPU/task): PLUS fit ≈ **4.0 h/cell** (peak RSS ~554 MB), MOOR fit ≈
**0.34 h**, PLUS plan+eval ≈ **1.42 h/(cell×reward)**, MOOR plan+eval ≈ **0.17 h**.

**CPU-hour projection (32 cells, safe only):**

| | Fit (64 jobs) | Plan+eval (64 jobs) | Total |
|---|---:|---:|---:|
| central | 32·4.0 + 32·0.34 = **138.9** | 32·1.42 + 32·0.17 = **50.8** | **≈ 190 CPU-h** |
| lower (~0.85×) | | | **≈ 161 CPU-h** |
| upper (1.5× contingency) | | | **≈ 285 CPU-h** |

**Provisional hard ceiling = 300 CPU-h** — the arithmetic shows this is **sufficient** (upper ≈ 285 <
300).

Proposed Slurm limits:
- **Per-job:** 2 CPU; `--mem=4G` (≈7× peak RSS headroom); wall-time **`--time=08:00:00`** for fit rows
  (~2× the 4 h max), **`--time=03:00:00`** for plan rows.
- **Array concurrency:** `%100` (≤ 200 concurrent CPUs, within the 256 cap; leaves headroom).
- **Cumulative submission ceiling:** 300 CPU-h, tracked from `sacct` `TotalCPU`. **Stopping behaviour:**
  at **≥ 80 % (240 CPU-h)** consumed, hold all further array submissions and require re-approval before
  continuing; a per-job `--time` breach fails that one cell (see §5), not the array.
- **Do not launch until this ceiling is explicitly approved.**

## 5. Preregistered scientific acceptance criteria

The pre-existing **structural** checker (privacy status, finite objectives, overshoot bound,
return-blindness) remains required and is insufficient alone. These are **validity/diagnostic** criteria
— **not** performance-return thresholds. **Weak identification is classified separately from numerical
failure.** No universal raw-gradient threshold is used; convergence is scale-aware and bound-aware.

| # | Criterion | Measure | PASS | PASS-WITH-WARNING | FAIL | Failure stops |
|---|---|---|---|---|---|---|
| 1 | Privacy & provenance | forbidden-name scan; table-independence; family relabel; recorded digests match freeze | all pass | — | any private name reachable / digest mismatch | **whole array** |
| 2 | Finite objectives | selected training objective finite | finite | — | non-finite/NaN | one cell |
| 3 | Optimizer termination | recorded L-BFGS status per start | ≥1 start "converged" | all starts max-iter but finite min-obj | all starts line-search-fail / NaN | one cell·method |
| 4 | Scale-aware convergence | relative objective improvement over last 5 iters of the selected start (scale-free); **projected** grad on **free** (non-bound-active) coords as secondary | rel-Δobj < 1e-4 **or** proj-grad below preregistered scaled tol | plateaued but proj-grad above tol with bound-active coords (**weak-ID**, flagged) | non-finite gradient | warn (weak-ID) / one cell if non-finite |
| 5 | Parameter-boundary occupancy | fraction of free params within ε of a bound | < 25 % | 25–50 % (weak-ID flag) | — (occupancy alone is never a FAIL) | warn |
| 6 | Multi-start agreement | # finite starts; min-obj reproduced by ≥2 starts | ≥2 finite, agree | only 1 finite start | 0 finite starts | one cell·method |
| 7 | Holdout trajectory error | normalized holdout survey SSE | finite, comparable to train | finite but ≫ train (overfit/weak-ID flag) | non-finite | warn / one cell if non-finite |
| 8 | PLUS candidate failures & diversity | # candidates built; pairwise standardized distance | all forms yield ≥2 distinct candidates | some near-duplicate (flag) | a form yields < 2 distinct candidates | one cell·method (PLUS) |
| 9 | PLUS posterior normalization | posterior sums to 1, finite over the rollout | yes | — | NaN / not normalized | one cell·method (PLUS) |
| 10 | PBVI finite values & legal actions | `action_values` finite; argmax ∈ [0,A); deterministic on repeat | yes | — | non-finite / illegal / non-deterministic | one cell·method |
| 11 | Fit/kernel/filter/planner consistency | `regime_law_hash` + fitted-`model_hash` identical across stages | identical | — | mismatch | one cell·method |
| 12 | Missing / failed cells | complete receipts for every registered row | all present | — | any missing / errored | **whole array** (completion gate) |
| 13 | Runtime / memory limits | per-job wall & RSS; cumulative CPU-h | within limits | ≥80 % ceiling → pause & re-approve | wall/RSS exceeded | pause array (ceiling) / one cell (wall) |

## 6. Scientific status of this run

Registered name: **"Registered 32-cell hidden-demographics ecological-baseline diagnostic canary."**

It is **not**: the full 288-cell experiment; a known-r,K vs hidden-r,K ablation; evidence that general
RL is superior; or a performance-tuned method-selection run. Its purpose is to establish that the
corrected mechanistic PLUS/MOOR **fit, discretize, filter, and plan validly** at 32-cell scale under
the preregistered validity criteria — **without inspecting comparative returns.**

## 7. Revised known-r,K proposal (report only — not implemented)

Grounded in the simulator (set-point mode: intrinsic `r_base = 0` and unused; the per-action effective
rate is the private set-point `ρ(a)`; `K_eff = clip(K_base+κ, K_base, K_max)`, so `K_base ↔ k_0`,
`K_max ↔ k_max` on scale `S`). Three separately named future alternatives:

- **A. Known-capacity ablation.** Reveal `K_base` and `K_max` → fix `k_0, k_max` from truth on scale
  `S`; keep the action rates `r_a` and capacity increments `d_a` hidden/fitted. Clean; reveals only the
  population capacity scale, no action effects, no family. Implementable as one switch on which
  coordinates are fixed-from-truth.
- **B. Known-action-effects oracle.** Reveal the per-action set-points `r_a` (and optionally `d_a`) →
  **explicitly labelled as revealing private intervention effects**, *not* population `r`. Stronger
  information leak; must never be called "known population r."
- **C. Modified intrinsic-r benchmark.** Introduce an *active* population intrinsic `r` distinct from
  action effects (currently `r_base = 0`, inert). This **changes the environment/generative model** and
  is therefore not the current benchmark; it would need its own registration and re-collection.

**Recommendation.** What was previously described to supervisors as "known r,K" corresponds to revealing
**capacity + the per-action effective rates** — i.e. **A combined with B**, where the "r" is the
per-action set-points (action effects), *not* an intrinsic population r. To honour that framing without
relabelling action effects as intrinsic r: run **A (known-capacity)** as the clean, defensible
demographics-known upper bound, and if the rate information is also required, add **B** with the honest
label "known action set-points/effects." **Do not use C** — it is a different environment, and no
intrinsic population r exists in the current benchmark to reveal.

## 8. Final recommendation

### GO for the "Registered 32-cell hidden-demographics ecological-baseline diagnostic canary" — conditional on two explicit approvals

All Phase-2 verification is complete and reproduced (166 tests; PBVI validated vs exact; blockers fixed;
privacy/routing/consistency proven; structural acceptance passed; durable snapshot + manifest frozen).
The remaining gates are **your explicit approvals**, per your own instruction not to launch without
them:

1. **Approve the 300 CPU-h ceiling** and the §4 Slurm limits.
2. **Approve the §5 acceptance table** as the preregistered validity gate.

On approval, the next action is to submit the frozen 64-fit + 64-plan manifests under those limits,
return-blind, and produce `acceptance.json` before any comparative return is read. The **full 288-cell
sweep remains separately gated** (≈1,150 CPU-h) after the 32-cell diagnostic.

**Stopped per protocol — no jobs submitted, no performance returns inspected.**
