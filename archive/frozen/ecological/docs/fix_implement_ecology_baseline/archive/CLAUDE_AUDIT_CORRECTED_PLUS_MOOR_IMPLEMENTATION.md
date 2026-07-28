# Audit — corrected adapted PLUS/MOOR implementation

**Reviewer:** Claude (independent; verified in the live tree, not from the handoff)
**Date:** 2026-07-17
**Contract:** `CORRECTED_PLUS_MOOR_IMPLEMENTATION_PLAN.md` (approved) + `DECISIONS_FOR_CORRECTED_PLUS_MOOR_PLAN.md`
**Status:** read-only. Nothing modified.

---

## 0. Verdict

**Approve.** Both blocking defects (F1 survey-noise objective, F2 regime fit/deployment mismatch) are
genuinely fixed — I verified the corrected mathematics in code, not from the summary. Every one of my
prior findings is closed, several better than requested. Every mechanical claim reproduces exactly.
Three minor items below, none blocking.

## 1. The two blocking defects — genuinely fixed

**F1 — conditional-mean survey objective.** `faithful_fit.py:32-37` defines
`conditional_survey_mean_factor(sigma) = exp(sigma²/2)`, applied at the loss (`lines 449, 458`) as
`mean((latent * survey_mean_factor − target)²)`. **The survey random bank is gone**: the bank is now
`(initial, process, regime_uniforms)` (`line 219-223`) — no second survey draw exists to be sampled.
Exactly Decision 2.

**F2 — regime fit/deployment mismatch.** The mean-field path is **eliminated from the fitter**:
`regime_prob` no longer appears anywhere in `faithful_fit.py`. The objective now draws a **discrete**
regime from common random numbers under a **fixed** Π:

```
line 446:  regime = (regime_uniforms[:, ep, 0] >= 0.5).to(int64)          # initial
line 462:  regime = (regime_uniforms[:, ep, step+1] > probability_zero)   # transition
```

This is structurally identical to deployment's `(draws > regime_matrix[z, 0])`, and Π is built via
`torch.tensor(fixed_regime_matrix(persistence))` — **a constant, not a `Parameter`** — so the
zero-gradient trap is avoided by construction rather than worked around. Fitting and deployment now
share one law.

*(The two surviving `regime_prob` hits are legitimate and correctly scoped:
`faithful_pomdp.py:116,137` samples discrete regimes at deployment, and `types.py:106` is a belief
summary feature. Neither is a mean-field fit approximation.)*

## 2. My prior findings — all closed

| Prior finding | Status |
|---|---|
| **Cache key trap** (`dataset_sha256` includes rewards → 0 % hit rate) | ✅ `fit_cache_policy = fit_transition_hash_v1_reward_excluded`; test `test_fit_cache_is_reward_independent_and_transition_sensitive`. Also correctly keeps `test_reward_mode_is_dataset_cache_key` — the *dataset* cache **is** reward-keyed while the *fit* cache is not. That distinction is subtle and right. |
| **Channel-only exposure** | ✅ strict channel loader; `action_channel_schema_hash` in manifests; `test_real_channel_map_has_exact_14_parameters_and_structural_zeros` |
| **`r_a = 0` kink** | ✅ `test_signed_rate_zero_uses_symmetric_clarke_subgradients` |
| **Π grid multiplying the bank** | ✅ held at 16; `candidate_allocation = 4;4;4;1,2,1` |
| **F1 (mine): sensitivity suite un-costed** | ✅ `requested_cpu_ceiling_hours = 600` for sensitivity, 48 for canary |
| **F2 (mine): `r_max` unstated** | ✅ **better than asked** — `signed_rate_min/max = [-1.5, 2.0]` is not merely stated but **validation-pinned**: *"corrected primary requires signed rate bounds [-1.5,2.0]"* |
| **Minor: multi-root digest recipe** | ✅ the verifier now accepts `roots [roots ...]` |

## 3. Mechanical claims — all reproduce

| Claim | Verified |
|---|---|
| 154 tests pass | ✅ re-ran `pytest -q` independently: 154 passed, exit 0 |
| Digest `6456fdf…` over `src/ configs/ scripts/` | ✅ **reproduces exactly** — 114 files, `matched: true`, exit 0 |
| Manifests 64 / 128 / 320 / 8 | ✅ generated all four independently: **64 / 128 / 320 / 8**, exact |
| No Slurm jobs | ✅ `squeue` empty |
| Void run untouched | ✅ verifier `matched: true` on `2aae03ba…` |
| Only adapted IDs registered | ✅ `METHODS` has `plus_adapted_mechanistic_pbvi`, `moor_adapted_ricker_misspec_pbvi`; no provisional keys |
| Ruff ignores aren't fitter exemptions | ✅ ran Ruff **without** `--ignore E402,E731` on all six corrected modules: **All checks passed** |

Manifest arithmetic is internally consistent: 32 cells × 2 methods = 64 fit; × 2 rewards = 128 plan;
5 PLUS configs × 32 × 2 = 320 sensitivity; 2 cells × 2 methods × 2 rewards = 8 canary.

## 4. Metadata — checked, not assumed

Applying the lesson from earlier rounds (my numeric checks held while *descriptive* fields slipped
past twice), I inspected an actual manifest row rather than the handoff's description. Every declared
field is present and correct:

```
observation_protocol        = lognormal_conditional_mean_v1      ← the F1 fix, named
fit_cache_policy            = fit_transition_hash_v1_reward_excluded
regime_persistence_grid     = 0.80;0.90;0.97
candidate_allocation        = 4;4;4;1,2,1
candidate_construction      = episode_bootstrap_map_fixed_pi_v2
regularization_variant      = none_structural_v1
return_blinding             = timing_convergence_cache_only_until_scope_frozen
headline_sweep_authorized   = False
```

The claimed acceptance suite exists in full — I located all of it (initially missed because it lives
in the new `test_corrected_adapted_mechanistic.py`): σ=0.4 scalar and ordered-trajectory recovery,
zero-noise bit-determinism, no-sampled-survey-channel, canonical regime law, fixed-Π
no-optimizer-coordinates, Π immutability round-trip, exact 44→14, Clarke gradients, cache
reward-independence, corruption fail-loud, nested sensitivity keys, allocation prefixes.

## 5. Findings (all minor)

### M1 — undocumented deviation from the approved plan text

The approved plan §3 specified `r_a = r_max·tanh(q_a)` — **symmetric**. The implementation uses an
**asymmetric** bounded transform on `[-1.5, 2.0]` (arctanh-based inverse). This is benign and arguably
better: the range comfortably contains the simulator's true rates (Amur tiger `r ∈ [-0.4753, 0.0707]`)
with headroom on both sides, and it is validation-pinned against drift. **But it is a silent deviation
from approved text**, which is precisely the failure mode this workstream exists to prevent. Record
the change and its justification in the plan.

### M2 — the primary diagnostic subset carries no CPU ceiling

`requested_cpu_ceiling_hours` is populated only for `sensitivity` (600) and `canary` (48). The
`diagnostic_fit` and `diagnostic_plan` rows are **empty**. Yet the 32-cell subset is the largest block
of work — PLUS measured **>3 h/cell**, so 32 cells is plausibly 100+ CPU-h of fitting alone, plus 128
planning rows. Execution is gated on approval and measured projections, so this is not a defect — but
the asymmetry is odd: the biggest block is the only un-ceilinged one. Give it a ceiling.

### M3 — scope reminder, not a defect

No corrected canary has run, so **all runtime and memory for the corrected code are unmeasured**. The
≥1,000 CPU-h figure derives from the *void, defective* run and is a lower-bound sizing observation
only. Nothing about corrected feasibility is established yet.

## 6. Scope of this audit

**Verified:** the corrected F1/F2 mathematics in source; mean-field removal tree-wide; Π constancy and
optimizer-coordinate absence; registry; the digest; all four manifest counts; manifest metadata; test
count by re-run; existence of the named acceptance tests; Ruff without ignores; job/void state.

**Not verified:** that the acceptance tests are individually *correct* (I confirmed they exist and pass,
not that each assertion is well-posed); numerical correctness of PBVI backups against an exact
reference; parameter recovery on real data at scale; corrected runtime. These remain open and must not
be inferred from this approval.

## 7. Recommendation

**Approve the implementation.** Record M1 in the plan and give the diagnostic subset a ceiling (M2).
The next legitimate step is the **return-blind canary** under its 48 CPU-h cumulative ceiling — that is
the first evidence about corrected runtime, and the gate for everything after it. No subset,
sensitivity suite, or headline sweep is authorized until the canary reports and the PI supplies a
CPU budget.
