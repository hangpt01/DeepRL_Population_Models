# Tier-3 final implementation plan (cumulative ecological controls)

Date: 2026-06-27. Status: **converged plan; implementation is now in progress behind
`control_mode=cumulative_capped`.**
Authoritative source spec: `29_6_Improve_Continuous_Observations.tex` (C0–C8) and
`29_6_tier2_continuous_state_implementation_handoff.md` (current Tier-2 state).
This document supersedes the back-and-forth review files (now deleted) and is the single
spec Codex implements from and Claude audits against.

**Scope rule:** modify code only inside `discrete_action_cont_obser/`. Do not touch
`claude_build/`, the old discrete repo, or parent-level files.

---

## 0. Framing (settled, do not relitigate)

Tier-3 is a **deliberate model-only redesign**, not a bug fix. The executed Tier-2 code *did*
have direct stock authority (`harvest_fraction` up to 0.50, `stocking_delta` up to 160 in
`actions.py`, applied in `transition_value`). Tier-3 **removes** those structure-agnostic
channels and routes all management through persistent, capped `(r,K)` controls, so that acting
well *requires* inferring the hidden structure. Consequence: **Tier-3 results are not directly
comparable to Tier-2** without staged ablations; say so wherever the two are placed side by side.

Three non-negotiable correctness invariants carried from the review:

- **I1 (leakage).** Public to methods: `{rho_t, kappa_t, K_eff_t, action id, observation,
  public reward, done/truncated, timestep}`. `K_eff_t = clip(500 + kappa_t, K_min, K_max)` is
  public *only because* `K_base=500` is fixed/public. **Never expose `r_eff`** (it = `clip(r_base
  + rho)` and `r_base` is hidden → leaks `r_base`; near the theta cap it also leaks `theta`).
  Private/evaluator-only: `s, r_base, C, theta, regime, r_eff_true`, theta-specific cap effects,
  process noise. The tex wording "`(r_eff,K_eff)` known exactly" is **wrong**; only `rho/kappa`
  and `K_eff` are known to the agent.
- **I2 (rollout state).** Cumulative controls break the Markov-in-`(s,context,regime)`
  assumption: `r_eff_t/K_eff_t` depend on the *sum* of prior rollout actions. The rollout state
  must be augmented to **`(s, context, regime, rho, kappa)`** everywhere multi-step lookahead
  happens.
- **I3 (reward channels).** Belief-expected reward (C2) is planner/training-side only. Realized
  operational reward and evaluator-only true reward stay separate channels. `K_ref=500` stays
  **fixed**; do not track `K_eff`.

---

## 1. What changes vs. the current code (verified file map)

| Concern | Current state (verified) | Tier-3 change |
|---|---|---|
| Action effect on `(r,K)` | one-step, non-persisted; `envs.py:128-129` | cumulative `rho,kappa` accumulators + caps |
| Direct `s` authority | present: `harvest_fraction`, `stocking_delta` (`actions.py:11-49`, `envs.py:110-111,125`) | **removed** in new tables (`=0`) |
| Caps | none | `r_max=0.55` (ricker/allee/regime), `r_max(θ)=1.6/θ` (theta), `r_min=-0.15`, `K_min=500`, `K_max=1500` |
| Theta prior | `r_base~U(0.18,0.40)`, `θ~U(3,6)` → `rθ` up to 2.4 (`config.py:139`) | draw θ then `r_base~U(0.12, min(0.40, 1.2/θ))` |
| Rollout state | `(s, context, regime)` only — `planning.py:47-117`, `bamcts.py:60`, `ogsrl.py:124`, `dynamics.py:14` | augment with `(rho, kappa)` via shared abstraction |
| Belief particles | already carry `contexts`, `regimes` (`types.py:45-53`) | give `contexts` structured `(r_base,C,θ)` semantics + priors (C1) |
| Reward | fixed `K_ref=500`, observed-`o` operational + true (`envs.py:186-187`) | keep fixed `K_ref`; add belief-expected mode (C2) |

Most learners (MOPO/RefPlan/BA-MCTS/Delphic/OGSRL) consume controls generically **once** the
shared rollout-state and `BeliefState.features` carry `(rho,kappa,K_eff)`. The **mechanistic**
predictors (`plus.py`, `moor.py`, `value.py`, `gate.py`) plus the planner's internal transition
(`planning.py`, `dynamics.py`) need hand-written cumulative-aware transition logic.

---

## 2. Implementation phases (ordered; each phase gates the next)

### Phase 0 — Spec/flag scaffolding (no behavior change)
- Add `control_mode ∈ {tier2_one_step, cumulative_capped}` to `EnvironmentConfig`; default
  `tier2_one_step`. All Tier-3 behavior lives behind `cumulative_capped`.
- Document the I1 public/private boundary in `docs/architecture.md` and `types.py` docstrings.

### Phase 1 — C0 environment + frozen Tier-2 (the setting change)
- Under `cumulative_capped`: add persistent `rho,kappa` to env state (reset to 0 each episode);
  replace per-step `r_eff/K_eff` with the cumulative update (accumulate Δr,ΔK → clip to caps).
  Implement leaky accumulators now with default `d_r=d_K=0` (so decay is a later config flip).
- New 5/10 action tables with `harvest_fraction=0`, `stocking_delta=0`, Δr/ΔK and costs initially
  from the tex; harvest = negative Δr + negative cost (revenue), floored at `r_min`. If the tex
  costs fail GATE C0-econ, recalibrate the costs before proceeding.
- Theta prior stabilization (Phase-1 table above), under `cumulative_capped` only.
- **Golden regression test** pinning `tier2_one_step`: fix seed/kind/action-sequence/noise,
  assert exact arrays for observations, rewards, private states, entry flags, action-table hash.
  Keep the `tier2_one_step` private-dict schema **unchanged** (do not emit `rho/kappa` there).
- **GATE C0-econ (acceptance, not assumption):** in C0 smoke, numerically verify the redesigned
  economics before proceeding: (a) restoration has an *interior* optimum — sweep a fixed-`kappa`
  policy, show return rises then flattens below `K_max`; (b) a no-investment regulator does **not**
  weakly dominate the conservation actions. If either fails, recalibrate action magnitudes/costs
  here, before C1/C2. The first implementation found the tex restoration costs dominated after
  direct stocking was removed, so the Tier-3 restoration/conservation costs in `actions.py` are
  lowered and pinned by `tests/test_tier3_controls.py`.

### Phase 2 — Augmented rollout-state plumbing (I2; hard gate)
- Build **one** shared rollout-state object `(s, context, regime, rho, kappa)` with a single
  `advance(action)` that does `rho,kappa ← accumulate; r_eff,K_eff ← clip; s' ← f_model`.
- Refactor `planning.score_sequences`, `bamcts._simulate`, `ogsrl._rollouts`, and the dynamics
  `_design`/`sample_next` signature to consume it. Extend dataset rows + `PublicTransition.public_info`
  + `BeliefState` + cached features with current/next `(rho,kappa,K_eff)`. Version caches so old
  Tier-2 artifacts can't be silently reused.
- **GATE Phase-2 (hard):** test that a constant-action simulated rollout reproduces the env's own
  cumulative `(rho,kappa)` trajectory exactly. No planner output is trusted until this passes.

### Phase 3 — Minimal method compatibility
- Learned dynamics features condition on public controls; planner/BA-MCTS/OGSRL/RefPlan/MOPO
  proposal calls carry controls. PLUS/MOOR/value/gate get explicit cumulative model-only
  transitions. Keep per-method *algorithm* changes (C3–C6) disabled until C8 passes.

### Phase 4 — C8 recalibration + gates
- Re-derive collector profiles for all six family/action cells (old profiles won't transfer:
  no stocking + `r_min<0`).
- Require **healthy-start** incident collapse in `[0.15, 0.24]`.
- **Dataset-coverage check (new):** confirm the dataset actually contains harvest-driven *delayed*
  collapse trajectories — otherwise learners can't see the over-exploitation downside. Not just
  the aggregate collapse band.
- Decision gate hard at `σ_obs ≤ 0.2`; `σ_obs = 0.4` reported as a diagnostic gate.
- Report collapse **split by start condition** (above/below hidden `C`); pooled collapse is not
  interpretable under a no-stocking design.

### Phase 5 — C1 + C2 (the Tier-A levers)
- C1: joint `(s, r_base, C, θ, z)` belief filter with per-cell priors (theta's constrained
  prior), hidden params fixed per episode while controls evolve, Rao–Blackwellize where possible,
  ESS/depletion diagnostics, no leakage through public feature summaries. Add `oracle_state`,
  `oracle_params`, `true_family` ablation switches.
- C2: belief-expected reward mode for planning/training; `P·Pr(collapse)` from the belief's
  predictive next-state distribution; realized + true reward stay separate (I3).

### Phase 6 — C3–C6 (staged, not factorial)
- C3 MLP dynamics member (composed with exact emission). C4 Delphic model-based on shared MPC
  (validate `u_Δ` monotone in `σ_obs`, →0 at `σ=0`). C5 OGSRL cumulative-vs-per-step budget
  recalibration per `σ_obs`. C6 BA-MCTS fitted/safety-aware leaves.

### Open gaps from the C0 audit (converged 2026-06-27; close before any Phase-4 ranking)
Status: C0/Phase-2 foundation merged and audited (`29_6_claude_audit_tier3_C0_implementation.md`).
Claude + Codex converged on three follow-ups; none block merged work, all gate the *next* phase's
numbers.
- **G1 (Medium) — control-blind learned filter proposal.** `LearnedLinearProposal` and
  `ReferenceProposal` in `beliefs.py` accept `rho/kappa` but ignore them, so the belief
  *prediction* step is blind to cumulative effort (worst at high `σ_obs`). The dynamics ensemble
  and mechanistic proposal are controls-aware; only the filter proposal is not. **Resolution:**
  the C1 joint-param filter (Phase 5) **must** condition on public controls; until then,
  non-control-aware proposal modes are marked legacy/debug and **excluded from Tier-3 ranking
  manifests** (do not let the blind path survive as a comparison artifact). **Implemented
  2026-06-27:** `ReferenceProposal` now applies post-action public controls to its drift, and
  `LearnedLinearProposal` uses a versioned 8-column cumulative design
  `(1, log_state, context[0:3], next_rho, next_kappa/K_ref, next_K_eff/K_ref)` under
  `control_mode=cumulative_capped`; old 5-column caches are not reused across the mode boundary.
- **G2 (Low–Med) — `environment_with_kind_defaults` is not `control_mode`-aware.** It still
  applies Tier-2 family priors, so a Tier-3 cell built through it gets theta floor `0.18` (spec
  drift vs `0.12`; not a stability bug — `0.18×6=1.08<1.2`) and, for true ricker cells,
  `r_base≈0.95` colliding with `r_max=0.55` (permanent clipping). Latent **manifest/config
  foot-gun**, not an active headline failure (headline families are allee/theta/regime; mechanistic
  predictors inherit the active cell's range). **Resolution:** make kind-defaults
  `control_mode`-aware (Tier-3 theta floor `0.12`; Tier-3 ricker `r_base ≤ ~0.30`) before Phase-4.
  **Implemented 2026-06-27:** `environment_with_kind_defaults` preserves the Tier-2 priors for
  `tier2_one_step` and uses Tier-3-safe priors for `cumulative_capped`.
- **G3 (note) — economics gate is C0-smoke only.** The interior-optimum test is ricker/open-loop;
  the per-family closed-loop economics acceptance is a Phase-4 deliverable, not yet cleared for
  allee/theta/regime.

### Ablation knobs (keep the architecture ablation-possible)
`control_mode`, `reward_mode ∈ {observed, belief_expected}`,
`filter.proposal ∈ {learned, joint_param, oracle_state, oracle_params, true_family}`,
`model.backend ∈ {ridge, mlp}`, `delphic.mode ∈ {cql, model_based}`,
`bamcts.leaf_value ∈ {zero, fitted}`, `decay ∈ {0, on}`.

---

## 3. Validation bar — do not read method rankings until all pass

1. `tier2_one_step` passes the golden regression test.
2. GATE C0-econ passes (interior restoration optimum; conservation not dominated).
3. GATE Phase-2 passes (constant-action rollout reproduces env `rho/kappa`).
4. `cumulative_capped` C0+C8 re-enters the healthy-start collapse band `[0.15,0.24]` **and** the
   dataset covers harvest-driven collapse.
5. Hard decision gate passes at `σ_obs ≤ 0.2`; `σ_obs=0.4` gate is reported first.
6. C0-audit gaps closed: G1 (filter conditions on controls, or blind proposals excluded from the
   ranking manifest) and G2 (`control_mode`-aware kind-defaults) — see Open gaps above.

First meaningful positive result: paired-seed true-return improvement (with paired-bootstrap CIs,
Tier D) and unchanged-or-lower collapse for C0+C1+C2 over C0-only learners, plus ≥2/6 high-noise
cells beating same-setting `max(PLUS, MOOR)`. **Conditional honesty:** if the `σ_obs=0.4` gate
says the redesigned cell is not decision-relevant, a null high-noise result is the correct,
reportable answer — do not force beats-both above 0.

---

## 4. Claude's audit checklist (how I'll review Codex's code)

Per phase, I will check:

- **Leakage (I1):** grep that no method/dataset/feature path reads `r_eff`, `r_base`, `s`, `C`,
  `theta`, `regime` from the public side; confirm `BeliefState.features` and public_info expose
  only `{rho,kappa,K_eff}` controls. Confirm the existing no-leak tests
  (`tests/test_dataset_filter.py`, `test_evaluator_gate.py`) still pass and are extended to the
  new control fields.
- **Rollout state (I2):** read `planning.py`, `bamcts.py`, `ogsrl.py`, `dynamics.py` and verify
  each threads `(rho,kappa)` and updates them per simulated action — not a constant. Confirm the
  Phase-2 constant-action reproduction test exists and asserts equality with the env trajectory.
- **Tier-2 freeze:** golden test present, asserts exact arrays, and `tier2_one_step` private
  schema is unchanged (no `rho/kappa` leakage into the old path).
- **Reward (I3):** `reward.py` keeps `K_ref` fixed; belief-expected path never touches true `s`;
  evaluator logs realized + true separately.
- **Economics gate:** the C0-smoke interior-optimum / not-dominated check is an actual runnable
  assertion, not prose.
- **Calibration:** collapse band `[0.15,0.24]` on healthy starts, collapse split by start
  condition, and the harvest-driven-collapse coverage check are all present.
- **Caps & theta prior:** `r_max=0.55`, `r_max(θ)=1.6/θ`, `r_min=-0.15`, `K∈[500,1500]` applied;
  theta prior draws θ-then-`r_base` with headroom.
- **Tests:** every claimed gate has a test; method smoke runs under `cumulative_capped`; ablation
  switches are exercised.

---

## 5. One-paragraph summary for the implementer

Freeze Tier-2 behind `control_mode=tier2_one_step` with a golden test. Implement C0 model-only
cumulative-capped controls (remove direct stocking/harvest, add `rho/kappa` accumulators + caps +
theta-prior fix) behind the flag, and **prove the new reward economics has an interior optimum in
C0 smoke before going further.** Then land the shared augmented rollout-state `(s,context,regime,
rho,kappa)` abstraction and prove a constant-action rollout reproduces the env — this is a hard
gate before any planner is trusted. Recalibrate (C8) to the `[0.15,0.24]` healthy-start band with
harvest-collapse coverage, pass the gate at `σ≤0.2`, then add C1 (joint-param filter) and C2
(belief-expected reward), then C3–C6 staged. Keep the leakage boundary (`rho,kappa,K_eff` public;
`r_eff` never public) and the fixed `K_ref` throughout. Stop and report honestly if `σ=0.4` is
not decision-relevant after the redesign.
