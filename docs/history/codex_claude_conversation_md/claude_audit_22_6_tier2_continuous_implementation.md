# Claude's code audit — Tier-2 continuous benchmark (`discrete_action_cont_obser/`)

Date: 2026-06-20. Auditor: Claude (Opus 4.8).

Scope: the standalone repo Codex implemented at `discrete_action_cont_obser/` against (a) the locked design `docs/22_6_Continuous_Observation_New_Baselines.tex`, (b) the code-level handoff `docs/stress_pomdp_116_continuous_successor_handoff.md`, (c) the agreed plan of record `docs/planning/codex_stress_pomdp_22_6_tier2_continuous_implementation_plan.md`, and (d) the resolved Codex/Claude decisions (`codex_22_6_response...md`, `claude_concurrence_codex_tier2_response.md`).

## Verification method

I read all 24 source modules (~3,960 LOC), the 4 test files, configs, scripts, and docs; ran the suite (**14/14 pass**); and ran dynamic checks: σ_obs=0 filter exactness, filter RMSE vs σ_obs, no-clipping, truth-access grep, and an end-to-end smoke. Evidence is cited as `file:line`.

## Bottom line

**This is a high-quality, faithful implementation — substantially better than a scaffold.** The locked physics, the leakage discipline, the three-controller gate, the per-candidate PLUS bank, and the structural faithfulness of Delphic/OGSRL are all genuinely present, not stubbed. The standalone constraint is honored (zero `claude_build` imports). Tests pass and the pipeline runs end to end.

The caveats are about **scientific validity of the eventual numbers, not about whether the code is the right code.** Five findings are worth acting on before the 75k matrix runs; none are fatal. Critically, **no scientific result exists yet** — "full 75k matrix not executed" means every headline number is still ahead, and the validation that matters (calibration band, gate pass/fail on real cells, Delphic-uncertainty-responds-to-confounding) is not yet in the test suite.

---

## What is faithful and correct (verified)

### Physics layer — excellent
- **Continuous, unbounded, no `s_max`.** Growth is computed in log-domain with an overflow *guard* (raises, never silently clips) and a non-negativity floor only ([envs.py:21-31](discrete_action_cont_obser/src/tier2_benchmark/envs.py#L21-L31), [envs.py:152-154](discrete_action_cont_obser/src/tier2_benchmark/envs.py#L152-L154)). `test_no_discrete_geometry` asserts the env has no `num_states`/`s_max` attribute ([test_environment.py:70-73](discrete_action_cont_obser/tests/test_environment.py#L70-L73)).
- **`C_safe` collision resolved** (Decision #1): the benchmark floor is `safety_threshold=50`; the regime model's biological 90/180 are separate, hardcoded in dynamics ([envs.py:144-145](discrete_action_cont_obser/src/tier2_benchmark/envs.py#L144-L145), [config.py:27](discrete_action_cont_obser/src/tier2_benchmark/config.py#L27)).
- **`s=0` absorbing + terminal, no stocking revival** (Decision #3): `transition_value` returns 0 at `state==0` *before* applying authority, so stocking cannot revive a dead stock; `terminated = state_next==0`; `step` on a done env raises ([envs.py:123](discrete_action_cont_obser/src/tier2_benchmark/envs.py#L123), [envs.py:193](discrete_action_cont_obser/src/tier2_benchmark/envs.py#L193), [envs.py:157](discrete_action_cont_obser/src/tier2_benchmark/envs.py#L157)). `test_exact_extinction_is_terminal` covers it.
- **Floor-only collapse with entry latch** (Decision #2): first healthy→at/below crossing latches once for the penalty; dynamics continue; only exact zero terminates ([envs.py:179-185](discrete_action_cont_obser/src/tier2_benchmark/envs.py#L179-L185)). Consistent `<=` semantics (Decision #5).
- **Reward from `o_t`, penalty from latent** (design Eq. 5): operational reward uses `observation_prev` and the latent entry bit; `R_true` from latent in a private channel ([envs.py:186-187](discrete_action_cont_obser/src/tier2_benchmark/envs.py#L186-L187), [reward.py:23-33](discrete_action_cont_obser/src/tier2_benchmark/reward.py#L23-L33)).
- **5 independent RNG streams** for CRN pairing ([envs.py:65-68](discrete_action_cont_obser/src/tier2_benchmark/envs.py#L65-L68)).
- **Log-normal observation kernel is correct including the Jacobian** `−log(o)`, plus exact branches for σ=0, `s=0,o=0` (point mass), and the impossible `s=0,o>0`/`s>0,o=0` cases ([observation.py:32-55](discrete_action_cont_obser/src/tier2_benchmark/observation.py#L32-L55)). `test_exact_and_zero_atoms` covers the atoms.
- **Action tables match Tier-1 exactly** (5- and 10-action), with a provenance hash ([actions.py:30-49](discrete_action_cont_obser/src/tier2_benchmark/actions.py#L30-L49)); authority-before-growth verified numerically by `test_ricker_authority_before_growth`.

### Data + filter — excellent, with active leakage guard
- **Public/private split with an enforced schema guard:** `assert_public_schema` raises if any of `{states, r_base, C, theta, regime, reward_true, …}` appears in the public artifact ([dataset.py:169-177](discrete_action_cont_obser/src/tier2_benchmark/dataset.py#L169-L177)); episode-contiguous, done-terminated validation; episode-level (not transition-level) splits ([dataset.py:51-69](discrete_action_cont_obser/src/tier2_benchmark/dataset.py#L51-L69), [dataset.py:78-87](discrete_action_cont_obser/src/tier2_benchmark/dataset.py#L78-L87)).
- **Privileged behavior locked** (Decision #6, foundational for Delphic): the collector acts on true `s` ([collector.py:41-65](discrete_action_cont_obser/src/tier2_benchmark/collector.py#L41-L65), [collector.py:48](discrete_action_cont_obser/src/tier2_benchmark/collector.py#L48)), and **the reward-entry-bit is disclosed** in metadata ("collapse entry penalty only", [collector.py:141](discrete_action_cont_obser/src/tier2_benchmark/collector.py#L141)).
- **Known-emission PF with the 3-rung proposal ladder** I recommended: `ReferenceProposal` (weak, no sim eqns), `LearnedLinearProposal` (public-data), `MechanisticProposal` (Ricker/true-family); plus `RawObservationFilter` and `OracleStateFilter` ablations ([beliefs.py:28-310](discrete_action_cont_obser/src/tier2_benchmark/beliefs.py#L28-L310)). ESS-triggered systematic resampling + context rejuvenation for static latents ([beliefs.py:239-252](discrete_action_cont_obser/src/tier2_benchmark/beliefs.py#L239-L252)).
- **σ_obs=0 exact** (verified: RMSE 0.000) and **RMSE rises monotonically with noise** — I measured mean filter RMSE 0.0 → 16.9 → 22.3 → 35.5 for σ∈{0,0.1,0.2,0.4} (true-family PF), and end-to-end coverage90≈0.89 vs nominal 0.90. The Phase-3 exit criterion holds.
- **Offline-only belief caching** (my agreed lock-in): beliefs are cached per (dataset, filter) and reused; the oracle filter explicitly raises if used as a training input ([pipeline.py:168-170](discrete_action_cont_obser/src/tier2_benchmark/pipeline.py#L168-L170), [pipeline.py:171-183](discrete_action_cont_obser/src/tier2_benchmark/pipeline.py#L171-L183)).

### Methods — faithful continuous adaptations
- **PLUS: Rao-Blackwellized per-candidate filter bank** (the P2 lock-in Codex accepted) — one belief per candidate, each propagated by its own Ricker proposal, weighted by the known emission likelihood, candidate posterior updated by per-candidate marginal evidence; continuous fitted Q, Ricker-only candidates ([plus.py:60-119](discrete_action_cont_obser/src/tier2_benchmark/methods/plus.py#L60-L119)).
- **Delphic-CQL has the faithful structure** (not "just an ensemble"): compatible worlds varying latent dim/scale + bootstrap, per-world behavior model and behavior Q, counterfactual Q, `u_Δ = Var_w`, Delphic Bellman penalty on a linear CQL head ([delphic.py:28-151](discrete_action_cont_obser/src/tier2_benchmark/methods/delphic.py#L28-L151)). (Quality caveat below.)
- **OGSRL** has a k-NN guardian with held-out quantile threshold, a learned simulator, separate safety + OOD costs, and **dual ascent on both budgets** ([ogsrl.py:23-212](discrete_action_cont_obser/src/tier2_benchmark/methods/ogsrl.py#L23-L212)). (Two deviations below.)
- **MOPO/RefPlan/BA-MCTS** are continuous: bootstrap ensemble + pessimistic particle MPC; member-posterior reflect-then-plan; belief-rooted tree with **progressive aggregation finer near the safety boundary** ([mopo.py](discrete_action_cont_obser/src/tier2_benchmark/methods/mopo.py), [refplan.py](discrete_action_cont_obser/src/tier2_benchmark/methods/refplan.py), [bamcts.py:54-58](discrete_action_cont_obser/src/tier2_benchmark/methods/bamcts.py#L54-L58)).
- **All predict latent and compose with the known emission** (not trained on `o_{t+1}`); the shared `ParticleMPC` **samples future observations** to integrate obs noise and computes entry on latent particles ([planning.py:97-120](discrete_action_cont_obser/src/tier2_benchmark/planning.py#L97-L120)). Continuous fitted-Q, no tabular VI ([value.py](discrete_action_cont_obser/src/tier2_benchmark/methods/value.py)).
- **Verified no method reads truth**: grep of `methods/` for `evaluator_info`/`reward_true`/dataset `next_states` finds only local model-rollout variables.

### Gate, evaluator, matrix — faithful
- **Three-controller gate**: belief_oracle (true family+prior, noisy obs), belief_ricker (misspecified, same obs), clairvoyant (realized params, diagnostic only); hard gate compares the two information-matched controllers, reusing the shared `ParticleMPC`; action-disagreement reported ([gate.py:51-155](discrete_action_cont_obser/src/tier2_benchmark/gate.py#L51-L155)). The manifest runner **requires the gate only for allee/regime at σ≤0.2** — exactly Decision #16 ([run_manifest_row.py:43-50](discrete_action_cont_obser/scripts/run_manifest_row.py#L43-L50)).
- **Evaluator** separates observed vs private metrics, computes collapse/RMSE from latent truth, uses **real action entropy** (Tier-1's mislabeled `policy_entropy` fixed), reports filter coverage/ESS/unsafe-Brier, **filter & planner seconds**, and fallback counts; danger fractions use physical units, not bins ([evaluator.py:71-181](discrete_action_cont_obser/src/tier2_benchmark/evaluator.py#L71-L181)). No `visited/num_states` coverage.
- **Paired, non-overlapping held-out seeds** `[7001,7051,7101,7151,7201]` spaced by 50 → 50 episodes each with no overlap, fixing the Tier-1 overlapping-seed flaw ([config.py:84](discrete_action_cont_obser/src/tier2_benchmark/config.py#L84)).
- **336-row manifest** (3 env × 2 act × 4 σ × 7 methods × {learned,raw}) confirmed by running `make_manifest` ([manifest.py:13-39](discrete_action_cont_obser/src/tier2_benchmark/manifest.py#L13-L39)); aggregation does paired `max(PLUS,MOOR)` deltas with a bootstrap CI on beats-both rate ([manifest.py:66-123](discrete_action_cont_obser/src/tier2_benchmark/manifest.py#L66-L123)). `full.yaml` is the real config (75k, 50 ep, 1024 particles, ensemble 15).

---

## Findings (severity-ranked)

### 1. [Medium] P1 lock-in is runnable but NOT in the headline manifest
Our concurrence locked: *the method-appropriate-filter PLUS/MOOR ablation must run on the headline cells.* The pipeline supports `ricker`/`true_family` filter modes ([pipeline.py:118-123](discrete_action_cont_obser/src/tier2_benchmark/pipeline.py#L118-L123)) and the CLI exposes them, but `make_manifest` only generates `{learned, raw}` rows ([manifest.py:21](discrete_action_cont_obser/src/tier2_benchmark/manifest.py#L21)). Meanwhile **MOOR consumes the shared *learned* filter for both fitting and acting** ([moor.py:43-47](discrete_action_cont_obser/src/tier2_benchmark/methods/moor.py#L43-L47), [moor.py:70](discrete_action_cont_obser/src/tier2_benchmark/methods/moor.py#L70)). So the default headline runs "MOOR-on-a-learned-belief," and the comparison a reviewer will demand — MOOR/PLUS with their own mechanistic filter — is absent from the 336. (PLUS is less exposed: it runs its RB-Ricker bank and uses the shared belief only to seed.) **Fix:** add `ricker`/`true_family` filter rows for PLUS and MOOR to the headline manifest, or explicitly predeclare `raw` as the mechanistic backstop. Low effort; the machinery exists.

### 2. [Medium] Leakage *surface* in the policy `observe()` hook
`policy.observe(belief, action, result)` passes the full `StepResult`, whose `evaluator_info` carries latent truth ([evaluator.py:102](discrete_action_cont_obser/src/tier2_benchmark/evaluator.py#L102), [types.py:106](discrete_action_cont_obser/src/tier2_benchmark/types.py#L106)). No implemented method reads it (verified), and the *dataset* is guarded — but the *eval-time policy interface* is not. A future method (or a careless edit) could read `result.evaluator_info["state"]` and silently cheat. **Fix:** pass a sanitized public result (observation, reward, done, public_info) to `observe()`; keep `evaluator_info` inside the evaluator. Defense-in-depth.

### 3. [Medium] Delphic's "compatible worlds" are random-feature linear models
The structure is faithful, but each world differs only by a **random `tanh` projection** of the *same* belief features, fit by linear ridge ([delphic.py:82-106](discrete_action_cont_obser/src/tier2_benchmark/methods/delphic.py#L82-L106)). The worlds are not constrained to be *equally data-compatible but differently confounded*, so `u_Δ` may track random-projection/epistemic variance rather than genuine confounding-driven value spread. The plan's own exit criterion — *u_Δ responds to a confounding-strength control, not merely bootstrap size* — is **untested**. **Fix before headline Delphic claims:** validate `u_Δ` against the σ_obs sweep, which (given the privileged-`s` collector) *is* the confounding-strength knob: at σ=0 the learner sees the confounder so `u_Δ` should be near-zero, and it should grow with σ. If `u_Δ` is flat across σ, the worlds aren't capturing Delphic uncertainty.

### 4. [Medium] OGSRL guardian is advisory at deployment; safety uses the posterior mean
Two deviations from the agreed OGSRL spec:
- **Guardian not enforced at act time.** Training applies a soft OOD cost via `λ_ood`, but `act()` picks `argmax(policy)` and only *logs* OOD — it never masks/over­rides an OOD action, and there is no method-level hard fallback ([ogsrl.py:214-226](discrete_action_cont_obser/src/tier2_benchmark/methods/ogsrl.py#L214-L226)). The paper's guardian is hard containment `G={(s,a):g≤τ}`. (The evaluator has a generic `try/except → action 0` fallback, but that is not OGSRL's guarded fallback.)
- **Safety/OOD computed from the posterior mean, not the belief.** Rollouts start from `beliefs.mean_states` and `act()` uses `belief.mean_state()` ([ogsrl.py:163](discrete_action_cont_obser/src/tier2_benchmark/methods/ogsrl.py#L163), [ogsrl.py:215-219](discrete_action_cont_obser/src/tier2_benchmark/methods/ogsrl.py#L215-L219)). We agreed safety should be the *posterior probability of entry/occupancy*, so a mean above the floor can hide substantial collapse mass. **Fix:** enforce the guardian at deployment (mask/fallback, report fallback rate) and compute safety/OOD from belief particles.

### 5. [Low] The shared learned dynamics model is weak and handicaps the learned methods
`ContinuousDynamicsEnsemble` is ridge on `[1, x, x², action-onehot]` where **the action enters only as an intercept** ([dynamics.py:14-17](discrete_action_cont_obser/src/tier2_benchmark/dynamics.py#L14-L17)) — it cannot represent action×abundance interactions (harvest/stocking effects vary with state). Every learned method (MOPO/RefPlan/BA-MCTS/Delphic/OGSRL) inherits this ceiling, which could bias the headline *against* the learned methods — the charitable direction, but still a validity concern for a "can learned beat mechanistic" claim. The action authority itself is applied exactly in the env; the issue is the *model's* expressiveness. **Fix for headline:** add action×x interaction features or a small MLP member.

### 6. [Low] Smaller items
- **MOOR uses the cheaper LS-on-posterior-means fit** (grid search, [moor.py:41-51](discrete_action_cont_obser/src/tier2_benchmark/methods/moor.py#L41-L51)), not the state-space EM the plan preferred. Acceptable (the plan declared LS the cheaper variant) — just label it as such.
- **`fit_mechanistic_q` fits on a geomspace support set** up to ~max(0.995-quantile, 1000) ([value.py:28-33](discrete_action_cont_obser/src/tier2_benchmark/methods/value.py#L28-L33)). It's a fitting support (adaptively ranged), not a dynamics ceiling, but there's no convergence/range ablation (the plan asked for one if any grid is retained).
- **Process-noise semantics differ for theta** (additive `noise*managed` vs in-exponent for the others, [envs.py:142](discrete_action_cont_obser/src/tier2_benchmark/envs.py#L142)). Irrelevant at the base σ_p=0; matters only under process-noise stress.

### 7. [Info] Validation depth and repo state
- The 14 tests are smoke + key invariants (the *right* invariants: σ=0 exact, no-leak, no-discrete-geometry, authority-before-growth, extinction-terminal). **Absent**: a calibration-band `[0.15,0.24]` assertion on a real dataset, RMSE-monotonic-in-σ as a test, the Delphic-confounding-response check, a gate pass/fail check on a real cell, and any end-to-end beats-baseline smoke. These are the Phase exit criteria and should exist before the headline run is trusted.
- The folder is **untracked git** (`?? discrete_action_cont_obser/`) — expected, since you'll create the GitHub repo. Suggest `git init` + first commit + a minimal CI that runs `make test`.

---

## Cross-check against the plan-of-record decisions

| Decision / lock-in | Status |
|---|---|
| #1 `safety_threshold=50`, reserve C | ✅ |
| #2 entry latch, no snap-to-zero | ✅ |
| #3 `s=0` terminal, no revival/revenue | ✅ |
| #4 incident collapse on healthy starts | ✅ ([collector.py:184-188](discrete_action_cont_obser/src/tier2_benchmark/collector.py#L184-L188)) |
| #5 consistent `<=` | ✅ |
| #6 privileged behavior (foundational) | ✅ |
| #7 shared filter feeds methods | ✅ for learned methods/MOOR; PLUS uses RB bank seeded from it |
| #8 filter uses known emission, learned transition | ✅ |
| #9 `m` unidentifiable; state RMSE only | ✅ (PLUS r-posterior + regime kept as method diagnostics) |
| #10 no internal grid/VI primary | ✅ (continuous fitted-Q) |
| #11 Delphic-CQL headline | ⚠️ structure ✅, mechanism needs validation (Finding 3) |
| #12 OGSRL GMB-CPO-discrete | ⚠️ dual ✅, guardian-enforcement + posterior-safety missing (Finding 4) |
| #14 3-controller gate | ✅ |
| #16 gate at σ≤0.2 | ✅ |
| #18 one 75k dataset/cell, 5 held-out blocks | ✅ |
| #19 private sidecar, separate loader | ✅ |
| Reward-entry-bit disclosed | ✅ |
| Offline-only belief caching | ✅ |
| Oracle-state ceiling ablation | ✅ ([pipeline.py:193-210](discrete_action_cont_obser/src/tier2_benchmark/pipeline.py#L193-L210)) |
| Filter RMSE as covariate | ✅ |
| P1: method-appropriate filter on headline | ❌ runnable but not in manifest (Finding 1) |

---

## Recommended next steps before the 75k matrix

1. **Add the calibration + science-validation tests** (band `[0.15,0.24]`, RMSE-monotone-in-σ, Delphic `u_Δ` vs σ, a gate pass/fail on one real allee cell). Cheap, and they convert "14 smoke tests pass" into "the benchmark behaves as designed."
2. **Run the gate matrix on real cells** and confirm allee/regime pass at σ≤0.2 — the headline study is gated on this and it has not been exercised at full size.
3. **Fix Findings 1, 2, 4** (manifest filter rows for PLUS/MOOR; sanitize the `observe()` result; enforce the OGSRL guardian + posterior-based safety). These are small and they directly protect headline validity.
4. **Strengthen the dynamics model** (Finding 5) or, at minimum, report dynamics RMSE per cell so the learned-method ceiling is visible.
5. **Then** stage: 1-cell pilot → belief matrix → raw ablation → oracle-state ceiling, with the predeclared wall/CPU-GPU budgets.

Overall: the implementation faithfully realizes the agreed plan and our resolved decisions. The remaining work is validation and four targeted fixes — not redesign. No code was modified during this audit.
