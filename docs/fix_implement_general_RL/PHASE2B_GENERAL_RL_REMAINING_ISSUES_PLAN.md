# Phase 2B — Remaining-Issues Reconciliation Plan (Hidden General-RL Baselines)

Plan-only response to `REVIEW_PHASE2_GENERAL_RL_CORRECTIVE_IMPLEMENTATION.md`. **No code was
edited, no jobs launched, no performance returns inspected, the isolated worktree was not merged,
and the active PLUS/MOOR run was not touched.** The read-only measurements below were run against
the Phase-2 worktree code and touch only model-internal quantities (behavior likelihood, Q-variance
decomposition, cost prevalence/spread, timing) — never `operational_return`/`true_return`/survival/
rankings.

I accept the review's verdict: **NOT READY for the corrected canary.** The measurements below turn
one of the reviewer's concerns from "possible" to **confirmed** (Delphic's cross-world variance is
misspecification-dominated), so this plan is more conservative than the review on Delphic.

---

## 1. Response to each finding (agree/disagree + evidence)

| Finding | Position | Evidence |
|---|---|---|
| **1. Delphic gate too weak (marginal TV)** | **Agree, and stronger: confirmed failing.** Marginal-TV compatibility is insufficient, and a measurement shows the cross-world Q variance is **not** concentrated on counterfactuals. | Full-4000, obs-vs-counterfactual raw Q variance: Amur `ratio cf/obs = 1.06`, Egyptian vulture `2.42` (should be ≫1). A naive "public-pinned + random perturbation" variant gives `~1.0` too. So current worlds are misspecification-dominated. |
| **2. RefPlan lacks the offline policy prior** | **Agree.** Marginalization is fixed but the "doubly Bayesian" prior policy is still absent. | `refplan.py` hidden `act` uses `PublicParticlePlanner._sequences` (random/enumerated), no π_prior. A calibrated public behavior policy is cheap (standardized logistic, held-out NLL 2.13 measured). |
| **3. Equality scan misses transformed leaks** | **Agree.** Exact-equality scanning cannot catch `log K`, `K/2`, family-branching, threshold-derived labels. | `test_general_privacy.py::test_no_fitted_value_equals_private_scale_or_threshold` only tests exact equality. |
| **4. OGSRL should not finalize with an inert safety cost; use a public low-abundance cost (option C)** | **Agree; recommend C.** A public shortfall cost is non-degenerate and binds at low abundance. | Candidate `c(o')=max(0,(s_low−o')/s_low)`, `s_low=`20th pct of positive training obs: data prevalence 0.18–0.21; action-cost spread up to **0.673** at `0.5·s_low` (argmax = aggressive harvest a2, argmin = translocation a10); zero at ≥1.5·s_low; works for healthy (Amur) and sink (vulture). |
| **5. Runtime/thread accounting** | **Agree; resolved to reliable.** | Threads were **unset** (OpenBLAS dynamic). BA-MCTS act time is identical at `OPENBLAS=unset` (599 ms) vs `=1` (584 ms) → **Python-loop-bound, not BLAS-bound**; pinning threads=1 gives clean 1-core/task accounting with negligible slowdown. numpy = scipy-openblas 0.3.29, NO_AFFINITY. |

---

## 2. Reconciliation of the Delphic NLL numbers

The two figures in the Phase-2 report §4 describe **different models on the same held-out split**:

| Quantity | Model | Fitter | Held-out conditional NLL |
|---|---|---|---|
| "≈24 nats, unusable" | latent-free **reference** (`fit_reference_behavior`, `delphic_compat.py`) | **uncalibrated**: unstandardized features, 200 iters, lr 0.15, ridge 1e-3 | measured **27.5** |
| "≈2 after calibration" | per-world behavior head (`delphic.py:_fit_world`) | **calibrated**: standardized features, 400 iters, lr 0.3, L2 1e-2 | measured worlds **1.86–1.92** |
| (new, for the fix) | **standardized** latent-free reference | calibrated (same recipe as worlds) | measured **2.13** |

Root cause of the apparent contradiction: I calibrated the *world* behavior heads but **left
`fit_reference_behavior` uncalibrated**, and the gate switched to marginal-TV so the reference NLL
became a stale diagnostic. **Fix:** calibrate the reference identically (standardize) so conditional
NLL is apples-to-apples; against the calibrated reference (2.13) the worlds (1.86–1.92) are
conditionally compatible — i.e. the *behavior* channel is fine; the **value** channel is the real
problem (§3).

---

## 3. Strengthened Delphic observable-compatibility gate (equations + thresholds)

**Design principle:** cross-world Q variance is valid delphic uncertainty **only** if worlds
(a) reproduce the observable behavior conditionally and per region, (b) **agree on observed-support
value**, and (c) disagree **primarily on counterfactuals**. All quantities are held-out; thresholds
are preregistered **before** any return. Let `W` = worlds, `H` = held-out transitions, `Q_w(s,a)` the
world Q, `a^o` the observed action.

**Gate G-D1 — conditional behavior compatibility.** With a *calibrated* latent-free reference
`π_ref` (standardized logistic on train, eval on holdout):
`NLL_w = −mean_H log π_w(a^o|s)`, require `NLL_w ≤ NLL_ref + τ_nll`, **τ_nll = 0.25 nats**.
(Measured: ref 2.13, worlds 1.86–1.92 → passes.)

**Gate G-D2 — per-region behavior calibration.** Bucket holdout by observation quintile `b`; for each
world and bucket, `TV_{w,b} = ½Σ_a |P̄_w(a|b) − P̂(a|b)|`; require `max_{w,b} TV_{w,b} ≤ τ_region`,
**τ_region = 0.20** (evaluated on the full-4000 holdout where each bucket has ≥100 samples; the
0.46 seen on the tiny fixture was sampling noise from ~10/bucket). Catches the "matches `P(a)` but
not `P(a|history)`" counterexample.

**Gate G-D3 — observed-support value agreement.** `V_obs = mean_H Var_w Q_w(s,a^o)`; require the
worlds to agree where data pins them, i.e. `V_obs ≤ τ_obs · scale²` with `scale` a public
value-scale (median |observed Q|). **This is the gate the current construction FAILS.**

**Gate G-D4 — counterfactual disagreement + ratio.** `V_cf = mean over (s, a≠a^o) Var_w Q_w(s,a)`;
require `V_cf ≥ φ · V_obs` with **φ = 3.0** (delphic variance must be ≥3× larger off-support than
on-support) **and** `V_cf > 0` scaled. Vanishes to fail at σ_obs=0 (already demonstrated).

**Gate passes** iff G-D1 ∧ G-D2 ∧ G-D3 ∧ G-D4 hold for ≥ `MIN_SURVIVING_WORLDS = 3`.

**Toy tests (required):** (6) a world with a deliberately history-shuffled behavior head that
matches `P(a)` but not `P(a|history)` → fails G-D2; (7) two hand-built worlds with identical
observed-support Q but latent-driven divergence only on an unobserved action → pass G-D3, satisfy
G-D4.

**Construction change needed to pass G-D3/G-D4 (this is the crux).** The current worlds add the
latent as extra regressors in the Q head, injecting variance **everywhere** (measured
`ratio ≈ 1`). To make worlds agree on-support and diverge off-support, refit the Q head so the
latent's influence is **support-gated**: fit a shared public-feature Q base `Q0` (identical across
worlds → agreement on-support), then add a per-world counterfactual perturbation
`Δ_w(s,a) = g_w(s,a) · u(s,a) · κ_amb(s)` where `u(s,a) = 1 − support(s,a)` is the public kNN-guardian
support (small in-data, large OOD), `κ_amb(s)` the posterior ambiguity, `g_w` a per-world random
sign/scale. This concentrates divergence on low-support counterfactuals by construction.

**Honest fallback (Decision D-A).** The measurement shows the current representation does **not**
pass G-D3/G-D4. If, after the support-gated construction, the full-4000 `ratio ≥ φ` does not hold
robustly across healthy **and** sink populations, **downgrade the name**: report Delphic as an
**"ensemble value-disagreement pessimism (Delphic-motivated)"** idea-level baseline, drop the
"observationally-compatible worlds / delphic uncertainty" claim, and keep it clearly separated from
the paper. Do not present misspecification variance as delphic uncertainty.

---

## 4. RefPlan public policy-prior design

Add a lightweight **public conservative behavior policy** `π_prior(a | public features)` and use it
in planning, keeping it a **distinct object** from the model posterior.

- **Fit:** calibrated multinomial-logistic on the **train** split's public belief features →
  actions (standardized; 400 iters; L2 1e-2) — the same fitter already validated (held-out NLL
  2.13). **No holdout fitting.** Stored on the policy as `self.prior_weights`, `self.prior_scaler`.
- **Exploration floor:** mix with uniform, `π̃ = (1−ε)π_prior + ε·Uniform`, **ε = 0.10 (registered)**,
  so unsupported deterministic behavior cannot eliminate all alternatives.
- **Use in planning:** in `plan_marginalized`, replace the uniform candidate-sequence sampler with
  **π̃-sampled first actions** (and π̃-biased continuations), and/or weight each candidate's reflected
  score by its prior log-prob `Σ_t log π̃(a_t|s_t)` with a registered temperature. Recommended
  minimal form: sample the first action of each candidate sequence from π̃ at the current belief
  mean; keep the model-posterior marginalization unchanged.
- **Prohibited inputs:** public only; no r/K/family/safety.
- **Toy test:** hold the model belief fixed; swap `π_prior` for a different fitted prior (or a
  hand-set one favoring a different action) and assert the candidate-plan first-action distribution
  / selected action changes — proving the prior is a live, separate object.
- **Remaining computational adaptations to state:** π_prior is a linear public policy, not the
  paper's conservative deep offline-RL policy; MPPI softmax weighting is approximated by
  reflected-score argmax. RefPlan stays **RefPlan-inspired** but is materially closer.

---

## 5. Private-value intervention-invariance test design

Exact-equality scanning is replaced/augmented by a **paired private-intervention** test that needs
**no data regeneration** (the method-facing boundary is `(MethodContext, dataset, cache)`, and the
`MethodContext` is provably built from public data only — `pipeline.py:_hidden_method_context` reads
`num_actions`, `action_costs`, `action_channels`, `observation_noise_sigma`, `horizon`,
`observation_scale`(=median obs from surrogate), `pop_id`, `reward_mode` — none private).

Design (`tests/real/test_general_privacy.py::test_private_value_intervention_invariance`):
1. Collect the public dataset + private sidecar **once**; fit surrogate; build `context_A`, `cache`.
2. For each private field in `{K_base, r_min, r_max, kind(family), safety_threshold, C_low/C_high,
   regime_thresholds, population-name}` **one at a time**: build `env_B = replace(env_A, field=…)`
   and rebuild `context_B = _hidden_method_context(cfg_B, dataset, surrogate)` **reusing the same
   dataset+surrogate** (no regeneration).
3. Assert `context_B == context_A` field-by-field (the public context is invariant to the private
   intervention — this catches transformed leaks like `log K`, `K/2`, a family label).
4. Fit each hidden method (`refplan/ogsrl/bamcts/delphic`) with `context_A` vs `context_B` on the
   same `dataset/cache/seed`; assert **byte-identical fitted arrays** and identical `act`/planning
   diagnostics.
5. Keep the API-block, forbidden-name, relabel, and (now-secondary) equality-scan tests as
   complementary checks.

**Why this catches transformed leaks the scan cannot:** the fit is a pure function of
`(context, dataset, cache)`; if the context is byte-identical under a one-field private change and
the fit is byte-identical, then **no transform of that private field** can have entered the method.
The one field varied at a time localizes any failure. (If a future change made a method read
`env_cfg` in hidden mode, step 4 would diverge and fail.)

---

## 6. OGSRL public-risk candidate comparison + recommended definition

Selection criteria (per review): interpretability, non-degeneracy, privacy, stability — **not**
policy return.

| Candidate | Formula (public only) | Non-degenerate? | Action-discriminating? | Verdict |
|---|---|---|---|---|
| Extinction event | `1[o'==0]` | **No** (prevalence 0.0, constant) | No | reject (current inert channel) |
| Binary low-abundance | `1[o' < s_low]` | Yes (~20%) | Yes but discontinuous | weaker (threshold brittleness) |
| **Bounded shortfall (recommended)** | `c(o') = max(0, (s_low − o')/s_low)`, `s_low = Q_{0.20}(o_train>0)` | **Yes** (prev 0.18–0.21; mean 0.06–0.08) | **Yes** (spread 0.67 at 0.5·s_low; 0 at ≥1.5·s_low) | **select** |
| Relative log-depth | `max(0, log(s_low/o'))` | Yes | Yes but unbounded | reject (unbounded, less stable) |

**Recommended preregistered definition.** `s_low = 20th percentile of positive observations on the
training split` (public; **not** private K or safety threshold; no holdout fitting). Per-transition
public safety cost `c(o') = max(0, (s_low − o')/s_low) ∈ [0,1]`, evaluated on the **model-predicted /
belief next observation** at deployment (as the existing risk channel is). It expresses
**low-abundance risk** (proximity to zero), not generic rarity, because it is a monotone shortfall
below a low-abundance scale, zero for healthy abundance.

- **Scale estimation:** training quantile only; recomputed per cell; independent of private
  `K`/`safety_threshold` (measured `s_low` = 21–140 across pops vs private `K` = 31–325 → different).
- **Prevalence/spread (measured, return-blind):** Amur/vulture/dolphin/jaguar data prevalence
  0.18–0.21 at q0.20; action spread up to 0.67 at low abundance, 0 at healthy — informative in both
  **healthy and sink** populations.
- **Budget (preregistered rule, before returns):** `safety_budget = mean discounted c under the
  behavior policy on the training split` (data-derived), floored at 0.02 and capped at 0.10; register
  the exact value per cell. This guarantees the dual is neither auto-satisfied (budget 0) nor
  auto-saturated (budget 1).
- **Binding evidence required:** a constructed low-abundance belief where an aggressive-harvest
  action exceeds budget and the trained OGSRL policy avoids it, with `lambda_safety` moving off its
  initialization (measured spread 0.67 ⇒ it will bind).
- **Matched access:** the shared **public surrogate reward already encodes** the collapse penalty for
  all methods; this new *cost channel* is an OGSRL constraint input only. RefPlan/BA-MCTS/Delphic do
  **not** consume it as a constraint (they are unconstrained planners) — matched in that all methods
  see the same public reward, and only OGSRL adds the guarded safety constraint (its defining role).
- **Wording:** this is a **public proxy** for low-abundance risk and **carries no guarantee** for the
  private safety objective; the theoretical safety claim stays removed.

If, at implementation, the dual does not bind on the constructed case for some population, retain
**OOD-guarded OGSRL-inspired** and drop the domain-safety-cost claim (do not tune the cost by
return).

---

## 7. Thread / resource validation plan

- **Register:** `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, numpy backend
  scipy-openblas 0.3.29 (NO_AFFINITY); 1 allocated core per task.
- **Justification (measured):** BA-MCTS act (the 85%-of-compute driver) is Python-loop-bound —
  599 ms (threads unset) vs 584 ms (threads=1); OGSRL/Delphic fits use only small linear solves.
  Pinning threads=1 costs ≈0 and makes 1-core/task accounting **reliable** (slightly conservative).
- **Validation to run before canary:** deterministic 1-thread vs default-thread canary per method
  comparing fitted arrays, actions, BA-MCTS beliefs (hashes), wall, and CPU%; confirm identical
  numerics (pure-numpy ⇒ thread count must not change results) and record wall.
- **Accounting to report separately:** aggregate task-hours, actual CPU-hours, allocated core-hours,
  elapsed wall at each concurrency, queue delay separately. Keep BA-MCTS 256/8 primary; 512/12 only
  as a small preregistered sensitivity.

---

## 8. Exact files / functions / tests to change (Phase 2C implementation, not now)

| Area | File · function | Change |
|---|---|---|
| Delphic reference calib. | `delphic_compat.py::fit_reference_behavior` | standardize features; 400/lr0.3/L2 1e-2 (match worlds) |
| Delphic gate | `delphic_compat.py::gate_worlds` (+ `CompatibilityReport`) | add G-D1..G-D4; replace marginal-TV-only decision; store per-region TV, `V_obs`, `V_cf`, ratio |
| Delphic construction | `delphic.py::_fit_world` | support-gated counterfactual perturbation (shared public Q base + `u(s,a)·κ_amb` divergence) |
| Delphic wiring/name | `delphic.py::fit`, `methods/__init__` label, `docs/benchmark/04_…tex` | apply gate; downgrade name if D-A fails |
| RefPlan prior | `refplan.py` (fit: add `prior_weights`; `plan_marginalized` call), `public_models.py::plan_marginalized` (accept π̃ sampler) | fit + ε-floor + prior-sampled candidates |
| OGSRL cost | `ogsrl.py` (`_public_low_abundance_cost`, use in `_public_rollouts` safety channel + `_public_belief_action_risks`), `config.py` (register `s_low_quantile`, budget rule) | replace inert risk with shortfall cost |
| Privacy | `tests/real/test_general_privacy.py` | add `test_private_value_intervention_invariance` |
| Mechanism | `tests/real/test_general_paper_mechanisms.py` | add Delphic G-D2/G-D3/G-D4 toy tests, RefPlan prior toy test, OGSRL dual-binds-on-low-abundance test |
| Probe | `scripts/general_adequacy_probe.py` | emit `V_obs/V_cf/ratio`, per-region TV, low-abundance cost prevalence/spread, `lambda_safety` |

MOPO untouched; the shared public surrogate reward untouched.

---

## 9. Return-blind acceptance gates (Phase 2C)

1. Delphic reference calibrated; NLL numbers reconciled and consistent in one table.
2. Delphic gate G-D1..G-D4 implemented; **G-D3/G-D4 pass on full-4000 for healthy + sink** (ratio
   ≥ 3) **or** name downgraded per D-A (no misspecification-as-delphic claim).
3. Delphic toy counterexample (matches `P(a)`, not `P(a|history)`) **fails**; compatible-but-
   counterfactual toy **passes**.
4. RefPlan prior is a distinct object; toy test shows swapping the prior changes plan selection with
   the model belief fixed; ε-floor keeps alternatives.
5. OGSRL public low-abundance cost non-degenerate (prevalence 0.05–0.30) and **dual binds** on a
   constructed low-abundance case; `lambda_safety` moves; budget preregistered.
6. Private-value intervention invariance passes one-field-at-a-time for all four methods.
7. Threads pinned=1; 1-thread vs default numerics identical; resources reported separately.
8. Full suite green; matched 4,000 budget and shared ensemble 5 unchanged; **no returns inspected**.

Failure of gate 2 (Delphic) or 5 (OGSRL binding) is a blocker resolved **only** by the honest
downgrade/relabel path, never by tuning to returns.

---

## 10. Rerun and snapshot scope

- All work continues in the **isolated worktree** `/fs04/scratch2/ce25/general_rl_phase2_iso`
  (base `5f9cf32`); **do not merge**; main dirty tree and PLUS/MOOR untouched.
- Re-run: full `pytest tests/`; the two return-blind adequacy canaries (Amur healthy, Egyptian
  vulture sink) plus a thread-parity canary; regenerate `post_correction_sha256` + registration.
- The **1,152-row manifest stays prepared and unsubmitted**; regenerate only if the Delphic
  name/inclusion changes.
- New pre/post digests + an updated `registration_general_corrected.json` (add Delphic gate
  thresholds, RefPlan ε, OGSRL `s_low_quantile`/budget, thread config).

---

## 11. Revised GO/NO-GO

**NO-GO for the corrected canary until Phase 2C lands.** Ranked:

1. **Delphic (blocking, downgraded from Phase-2 "pass").** Measurement confirms cross-world Q
   variance is misspecification-dominated (`ratio ≈ 1`, not ≫1); the marginal-TV gate does not
   establish value-channel compatibility. Requires the support-gated construction + G-D3/G-D4 gate,
   **or** an honest name downgrade. This is the decisive open item.
2. **OGSRL (blocking for the *final* method; OK for a smoke).** Adopt the public low-abundance
   shortfall cost (option C); the inert extinction channel must not be presented as the instantiated
   domain-safety component.
3. **RefPlan (paper-closeness).** Add the public policy prior; otherwise it stays honestly
   `-inspired` but not as close as the user wants.
4. **Privacy (verification).** Add the private-value intervention test.
5. **Runtime (engineering).** Pin threads=1; accounting is then reliable — **resolved in plan**.

RefPlan (marginalization) and BA-MCTS (in-tree belief) mechanism corrections from Phase 2 **stand**
and were not weakened by this review. The one decision that could change scope materially is
**D-A: whether Delphic can be made genuinely compatible or must be renamed** — I recommend
attempting the support-gated construction first, with the downgrade path preregistered so a negative
result is reported, not tuned away.

**Stop.** Awaiting approval of this plan (especially the Delphic construction-vs-downgrade decision
and the OGSRL option-C cost) before any Phase 2C code change. No jobs launched, no returns inspected,
worktree not merged, PLUS/MOOR untouched.
