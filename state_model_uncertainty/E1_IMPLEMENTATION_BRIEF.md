# E1 implementation brief — the observability and model gaps on Ricker environments

**Revision 4, 4 August 2026.** Revision 1 was withdrawn after `AUDIT_IMPLEMENTATION_BRIEF.md`
verified three of its load-bearing source claims to be false; Revision 4 removes the
factorial design after the Phase 0 degeneracy finding (§1). What changed is recorded in §9. Controlling specification remains `RESEARCH_PLAN_05_STATE_AND_MODEL_UNCERTAINTY.md`
§7.1. Source references are against `github.com/hangpt01/DeepRL_Population_Models` @
`3291eef`, re-verified by direct reading.

---

## 1. What the experiment asks, in plain terms

A conservation manager decides each year, for 50 years, how to intervene on a population.
Two things can be wrong with what they know:

- **Do they see the population accurately?** Either the true count, or a noisy survey.
- **Do they know how the population behaves?** Either the registered true dynamics, or
  dynamics inferred from 4,000 historical observations.

| | Sees the true count | Sees a noisy survey |
|---|---|---|
| **Knows the dynamics** | **A1** — reference | **A3** |
| **Learned the dynamics from the 4k log** | **A2** | **A4** |

**The question:** which handicap costs more return — not seeing clearly, or not knowing
the dynamics?

> ### ⚠ REVISED 4 Aug, Phase 0 finding — A1−A3 is DEGENERATE
> Process noise is zero, so dynamics are deterministic. An agent in **A3** knows the
> registered model, knows the action it chose, and knows `N0` — so it propagates the exact
> state forward and **surveys carry no information it does not already have.**
> `A1 − A3 = 0` in exact arithmetic; any non-zero value is barycentric grid diffusion plus
> PBVI approximation, and given fox's small action margins it could still exceed 0.10 and
> look like a result.
>
> **The observability gap is therefore measured in the fitted row: `A2 − A4`.** There the
> agent's model is wrong, its predicted trajectory drifts, and observations genuinely
> correct that drift.
>
> | Contrast | Status |
> |---|---|
> | **A2 − A4** | **Observability gap — PRIMARY.** Valid: model is misspecified, so surveys inform |
> | **A1 − A2** | **Model gap — PRIMARY.** Shares the A2 corner, so the two are comparable |
> | A3 − A4 | Model gap at noisy state — secondary |
> | **A1 − A3** | **Numerical diagnostic, not a result.** Should read ≈ 0; its measured value is a direct read on grid + solver error (U11) |
>
> **This is no longer a 2×2 factorial.** One margin is structurally zero, so the
> interaction collapses to approximately the observability effect itself. Report three
> informative arms plus one degenerate cell — do not present an interaction term as though
> the factorial were intact.
>
> **Scope consequence, to be carried into the write-up:** on this benchmark, observability
> only matters *through* model error. State uncertainty and model uncertainty are not
> separable here.

**A4 is not "the realistic setting."** All four arms plan the private true reward, which no
benchmark method does. A4 is the fitted/noisy corner *under E1's oracle objective*, and is
**not** comparable to accepted PLUS.

---

## 2. Scope

- **Cells:** fox × Ricker × σ∈{0.1, 0.2}; tiger × Ricker × σ∈{0.1, 0.2}.
- **Ricker only** — Allee/theta/regime carry an episode-persistent latent that would need a
  belief over (abundance × parameter). Vulture excluded (infeasible).
- Tiger is a control expected to show near-zero adaptive headroom.

---

## 3. Phases

| Phase | Owner | Gate |
|---|---|---|
| **0. Scope and report** | Codex | **Hard stop.** Ends the turn; requires explicit human `PROCEED PHASE 1` |
| **1. Registered-kernel parity gate** | Codex | **Blocking.** No planning code until parity passes |
| **2. Build** | Codex | — |
| **3. Validate** | Codex | Must pass before Phase 5 |
| **4. Seal predictions** | **Hang** | Must precede Phase 6 |
| **5. Independent audit** | Claude | Must pass before Phase 6 |
| **6. Run** | Codex | — |
| **7. Aggregate and read** | Codex → Hang | — |

**Budget intent: ~3 days, full scope** (decision, 4 Aug) — **authorised incrementally, one
phase at a time.** The window was extended from 1–2 days on the implementer's post-source
estimate of 2–3 focused days plus ~½ day aggregation. All four cells and all probes are
retained as the target. Compute was never the constraint (25–40 core-h, minutes when
sharded); the cost is the parity gate, the diagnostic POMDP and policy, and V1–V8.

**Phase 1 is authorised on its own.** Phase 2 is not. Phase 1 is a few hours' work, it must
happen first in any scenario, and it is the step that reveals whether the registered-model
construction is viable at all — so the remaining estimate should be *measured* after it
rather than accepted in advance. On completion the implementer reports actual elapsed time
and re-estimates, split by: Phase 2 implementation · essential validation, fox only · tiger
extension · QMDP/N2 · aggregation and independent audit. Authorisation for Phase 2 follows
that report.

**If the re-estimate exceeds the window**, the preferred cut is **fox only, retaining every
numerical-correctness probe** (parity, 41-vs-61, budget doubling, the A1−A3 red flag).
Deferring QMDP is the next cut after that, and its cost is specific: **N2 goes unanswered**,
so no claim can be made that belief planning rather than filtering is the lever — and
§7.1's gate loses the input it uses to decide whether E2 should target the planner or the
model axis. Never cut parity, the red flag, discretisation or budget sensitivity: those are
what stand between this experiment and another clean-running wrong number.

Rejected alternatives, recorded so the choice is auditable: *fox only with all probes*
(~1.5–2 days, loses the tiger control — which S1 has largely already supplied, zero
headroom in 5/8 tiger cells); *full scope with probes deferred* (rejected by both reviewer
and implementer — without QMDP, the 41-vs-61 bin probe and budget doubling, `A1−A3` is an
unexplained discrepancy rather than a calibrated bound on solver error, and the primary
contrasts could not trigger the E2 gate); *a 1–2 day partial* (validation content would be
determined by the clock rather than by preregistered priority).

**Registered red-flag tolerance (implementer, accepted 4 Aug):** `|A1−A3| ≥ 0.05` in either
fox cell makes every other contrast unsafe to interpret — half the practical margin for an
estimand that is exactly zero. At `|A1−A3| ≥ 0.10` the result is categorically
uninterpretable and must be re-baselined. Because fox action-value margins are ~0.0005,
action changes do not scale smoothly with the diagnostic, so a contrast **cannot** be
repaired by subtracting it.

**Phase 0: CLOSED, 4 Aug.** All answers accepted across two rounds. It caught two defects
before any code existed — the `reset_log_scale` sentinel is read by `initial_belief()`
(`faithful_pomdp.py:165-166`) and underflows to a silent uniform prior, and the
registered-row observability contrast is degenerate (§1), which removed the factorial
design. **Phase 1 is authorised.**

Registered conventions from Phase 0, to be honoured and recorded in the receipt: the
registered POMDP overrides `initial_belief()` with a delta on the bin nearest
`N0 / survey_scale`, and the base implementation is unreachable from A1/A2/A3 (A4 alone
uses it, with the fitted model's genuine positive reset scale); separate base/override
initial-belief counters assert this; `reset_log_scale = eps` survives **only** to satisfy
the frozen dataclass validator, affects `parameter_hash()` and provenance alone, and is
disclosed as a non-semantic field.

---

## 4. The changes, with verified locations

All paths relative to `src/tracks/ecological/real_ecology_benchmark/`.

### 4.0 What is confirmed reusable

`PointBasedPlanner.__init__` stores an injected model object with no `isinstance` check and
calls only `predict`, `update`, `expected_public_reward`, `representative_observations`
(`planners/pbvi.py:25-45, 57-110`). **A diagnostic subclass of `CandidatePOMDP` can change
reward and observation semantics without editing the planner.** That constraint holds.

**But not everything is model-level.** See 4.2.

### 4.1 Registered-parameter Ricker model (A1, A3) — and its parity gate

> **Do not use `default_model()` (`faithful_ecology.py:286-309`).** It is documented "used
> by equation/planner tests, never real fitting", with hard-coded `growth=0.2`. Revision 1
> cited it as a constructor path; that was wrong.

The mapping is **not** a field copy and must be specified before it is written:

- the action table supplies **signed rate set-points**, cumulative capacity increments and
  stocking (`actions.py:128-168`), while `MechanisticModel` expects **separate
  `growth`/`mortality` arrays** (`faithful_ecology.py:66-88`). The signed-rate split is
  `g = max(r,0)`, `h = max(-r,0)`;
- there is **no registered `survey_scale`** in `EnvironmentConfig`; in fitted models it
  comes from the public surrogate (`config.py:58-120, 227-253`;
  `faithful_fit.py:552-570`). Its source for the registered model must be declared;
- `MechanisticModel` **rejects a zero reset scale** (`faithful_ecology.py:124-145`), so a
  deterministic registered `N0` cannot be expressed by filling the dataclass naively;
- raw vs normalised units must be stated explicitly.

> ### ⚠ BLOCKING GATE — registered-kernel parity
> A mis-mapped model will still be row-stochastic, hash cleanly, plan without error and
> return plausible values. Part of A1−A2 would then be **constructor mismatch reported as a
> fitted-model gap**. This is the most likely way E1 produces a confident wrong number.
>
> **Before any planning code is written**, for every E1 cell, every action, all capacity
> grid values, and a preregistered abundance set including `N0`, `s_safe`, grid midpoints
> and action-induced states:
> - compare `MechanisticModel.noiseless_next()` / `next_capacity()` against
>   `ContinuousEcologyEnv.transition_value()` and the public control update, **in raw
>   abundance units**, at tight float64 tolerance;
> - compare a fixed-action 50-step rollout from `N0`, process noise off;
> - record source rows, transforms, unit convention, reset convention and a parity digest
>   in `E1_RECEIPT.json`;
> - **fail closed on any mismatch.** Do not proceed.

### 4.2 True-state arms (A1, A2) — a policy-level intervention, not a likelihood change

> **Revision 1 was wrong here, and this was its worst error.** It treated true-state
> delivery as a change to the observation likelihood. It is not.

`moor_faithful.act()` builds `self.internal_belief` from the `observation` argument, and
`observe()` updates it with `result.observation` — **the noisy survey**
(`methods/moor_faithful.py:90-113`). The evaluator passes `reset.observation` /
`result.observation` to the policy; true state lives only in `evaluator_info["state"]`
(`evaluator.py:96-157`). Changing the likelihood would snap **the survey**, not the truth,
and A1/A2 would silently measure something else.

**Required construction — direct belief assignment, in LATENT units, via the external
belief.**

> ### ⚠ UNIT ERROR — the grid is in latent units, not raw abundance
> `CandidatePOMDP.abundance_grid` is in **latent** units: raw abundance is recovered as
> `state_abundances() * model.survey_scale` (`faithful_pomdp.py:216, 241`), and
> `observation_likelihood` divides the raw observation by `survey_scale`
> (`faithful_ecology.py:236`). For a **fitted** model, `survey_scale` is
> `context.observation_scale` — the median positive observation, ≈ **41.45** for fox
> (`faithful_fit.py:447-454, 552`).
>
> **Snapping raw `evaluator_info["state"]` onto that grid is therefore wrong for A2 by a
> factor of ~41.** The belief would sit at or beyond the top grid bin while the arm still
> ran cleanly and returned plausible numbers. Snap
> `evaluator_info["state"] / model.survey_scale`, and assert
> `selected_bin_value * model.survey_scale` equals the registered discretisation of the raw
> truth.

> ### ⚠ ROUTE — the policy never receives `evaluator_info`
> `PublicTransition` **deliberately excludes** it — the docstring states it "carries only
> the observation/action-public context history" (`types.py:25-40`). And `policy.observe()`
> is called *before* the evaluator constructs the next oracle belief
> (`evaluator.py:107, 143`). So "set the belief after every transition from
> `evaluator_info`" is **not implementable through the policy interface.**

**The implementable route, using machinery that already exists:**

1. Run the evaluator with **`OracleStateFilter`**. The evaluator already branches on it and
   encodes truth into the external `BeliefState` via `set_true_state`
   (`evaluator.py:44-48, 144-146`; `beliefs.py:751`).
2. The diagnostic policy's **`act(belief, observation)`** reads `belief.mean_state()` —
   raw truth — converts to latent by dividing by `model.survey_scale`, and **resets its
   internal belief to a delta on the nearest latent bin**, immediately before every
   decision.
3. **`observe()` is a no-op** for internal filtering. `CandidatePOMDP.update()` is never
   called in A1/A2.

Note the diagnostic wrapper must **not** inherit `_require_public_filter()`
(`methods/moor_faithful.py:88`), which raises when the filter is not the public one.

The assignment therefore happens **at every `act()`**, not "after every transition" — a
distinction that matters because the two are not the same point in the loop.

This is deliberate and must be described as such: the true-state arm is an **experimental
information intervention**, not Bayesian conditioning. It also dissolves a problem the
previous revision had — a snapped delta fed through `update()` does **not** guarantee
positive evidence, because `_deposit_samples` spreads each draw barycentrically over two
bins (`faithful_pomdp.py:85-105`) while the simulator evolves from the true continuous
state and the POMDP from the previously snapped grid state, so the truth's bin can fall
outside the kernel row's support → evidence zero → `posterior = predicted`
(`:186-207`). Bypassing the **base Bayesian** update removes the failure mode entirely.

**Assertions required at every step:** the belief's argmax bin `b` satisfies
`grid[b] * survey_scale` == registered discretisation of raw `belief.mean_state()`; belief
max > 1 − 1e-9.

> ### ⚠ "update() is never called" is wrong — it needs three separate counters
> `PointBasedPlanner._condition()` **necessarily** calls `self.model.update()` during
> lookahead (`planners/pbvi.py:42-46`), so a blanket "zero update calls" assertion would
> reject a correct implementation — or push someone toward an impossible one. Separate them:
>
> | Counter | Expected in A1/A2 |
> |---|---|
> | Runtime policy filtering of `result.observation` | **zero** — `observe()` is a no-op |
> | Base `CandidatePOMDP.update()` (Bayesian conditioning on a likelihood) | **zero** |
> | Diagnostic **identity-conditioning** override, called by PBVI lookahead | **> 0, expected** — returns a delta on the branch bin |
>
> The diagnostic subclass must therefore **override `update()`** so that during forward
> planning it returns the branch-bin delta rather than performing Bayesian conditioning.

**These arms are "true state, binned."** Residual discretisation error survives, as
taxonomy §3.2 warns. The 41-vs-61 bin probe measures it.

**Forward lookahead must match.** For A1/A2 the planner must also assume perfect future
observation, or it plans "true now, noisy later". In the diagnostic POMDP, override
`representative_observations()` to return the **predictive support bins with their
predictive probabilities**, and the conditioning step to return a delta on the branch bin.

> **Revision 1 claimed uniform branch weighting is exact under deterministic Ricker
> dynamics. That is false.** Zero process noise makes all Monte Carlo samples equal, but
> barycentric deposit still splits that mean across two bins, so a point-mass belief
> predicts a **two-bin** distribution. Seven uniform quantile branches
> (`faithful_pomdp.py:231-242`) approximate it in increments of 1/7. And A2's fitted
> `process_scale` is a **learned parameter** that need not be zero even though the
> simulator's is (`faithful_fit.py:531-570`).

**Branch-count decision, to be registered:** return the predictive support with its
predictive weights, and **record the realised branch count and support size per arm**.

> **"Exact" applies to the branch weights only — not to the PBVI solution.**
> `_reachable_graph` selects each successor by index and **discards the weights entirely**
> (`planners/pbvi.py:66-73`); only the backup uses them (`:91-101`). With 32 belief points
> and a support that may be large — especially A2, whose fitted `process_scale` need not be
> zero — exact weights do not make the solve exact. Do not describe anything beyond the
> weights as exact. Do not pad to 7 and do not claim
the 7-branch budget is held fixed across all arms — it is not, and the asymmetry must be
reported. The 7-branch quadrature applies to the noisy arms A3/A4 only, where it is the
frozen setting.

### 4.3 True-reward objective (all four arms)

`expected_public_reward()` (`faithful_pomdp.py:209-229`) calls
`self.context.surrogate.predict(...)`. Add an explicit true-reward branch selected by an
explicit flag, recorded in the receipt.

> ### ⚠ TRAP — do not implement by setting `surrogate = None`
> The function returns exactly `-action_costs[action]` when the surrogate is `None`
> (`faithful_pomdp.py:218-220`). That is a **cost-only planner**: it runs, raises nothing,
> and returns plausible numbers. Verified correct in Revision 1 and re-confirmed.

**Call the source reward implementation, do not duplicate the formula.** `reward.py:29-63,
75-117` defines `alpha·x'/(x'+K_ref) − cost − penalty` with the penalty mode selected
separately. Duplicating it risks divergence on `alpha`, the `<= s_safe` boundary, penalty
mode, and terminal-state handling. Declare the penalty mode and `alpha` in the receipt.

**Registration:** `K_ref` and `s_safe` are private and `MethodContext` exposes neither —
that absence is the information-symmetry guarantee (taxonomy §0.2 Challenge C). E1 arms are
oracle/analysis constructs and are permitted the privilege, but it must be recorded, and
**E1 values must never be tabulated alongside accepted method rows.**

### 4.4 Fitted arms (A2, A4)

Reuse the **frozen MOOR fitting path unchanged** (16 process paths, 8 deterministic L-BFGS
starts, max 100 iterations, minimum-training-loss selection). Declare the exact fit config
digest, cache key and path, fit seed value, and whether a cache hit is required.

**Fit once per cell; store; reuse byte-identically in A2 and A4.** Refitting between them
puts fit noise inside the observability contrast. Define "kernel hash" explicitly —
model-parameter hash, transition-table hash, or serialised artifact hash — and use one
definition throughout.

**Note for interpretation:** one fit per cell means the fitted arms carry no fit-seed
variance. The 20 evaluation seeds quantify evaluation variation only. Any fitted-arm
conclusion is conditional on this single fit and this single collection log.

---

## 5. Command for Codex — implementation

```text
Implement experiment E1 in DeepRL_Population_Models. Controlling spec:
RESEARCH_PLAN_05_STATE_AND_MODEL_UNCERTAINTY.md section 7.1. Execution detail, verified
source references and traps: E1_IMPLEMENTATION_BRIEF.md sections 4.0-4.4. Read both fully
before writing any code.

HARD CONSTRAINTS
- Do NOT modify src/tracks/** . Frozen, byte-identical to two accepted codebases. All E1
  code goes in a new module under scripts/diagnostics/, importing from the ecological track.
- Do NOT modify planners/pbvi.py.
- Do NOT modify the accepted CSV or any archived result.
- Work on a branch. Do not push to main.
- Flag every deviation from this command. Never silently adapt.

================ PHASE 0 - SCOPE, THEN END YOUR TURN ================
Answer the following in writing. Then END THE TURN. Do NOT write or modify any code. Wait
for the user to reply with the exact string PROCEED PHASE 1.
Any answer that is negative, uncertain, or that you could not verify against source is a
STOP - say so plainly rather than proposing a workaround.

  (a0) RESET PRIOR AND THE SENTINEL. reset_log_scale is NOT non-semantic - initial_belief()
      reads it (faithful_pomdp.py:165-166) and eps underflows the Gaussian to all-zeros,
      after which _normalize returns a UNIFORM belief. State: the registered POMDP's
      initial_belief() override (an explicit delta on the bin nearest N0/survey_scale, used
      by A1 and A3); that the base initial_belief() is never reached by A1/A2/A3; that A4
      uses the fitted model's genuine positive reset scale; and that the sentinel's only
      remaining effect is on parameter_hash()/provenance, which must be disclosed in the
      receipt as a non-semantic field.

  (a) REGISTERED MODEL MAPPING. Give the exact construction of a registered-parameter
      Ricker MechanisticModel: the signed-rate to growth/mortality split, the source of
      survey_scale (there is no registered field for it - config.py:58-120,
      faithful_fit.py:552-570), the raw-vs-normalised unit convention, and how a
      deterministic registered N0 is represented given that MechanisticModel rejects a zero
      reset scale (faithful_ecology.py:124-145). Do NOT use default_model()
      (faithful_ecology.py:286-309) - it is a test dummy.
  (b) TRUE-STATE ROUTE AND UNITS. PublicTransition deliberately excludes evaluator_info
      (types.py:25-40) and policy.observe() runs before the next oracle belief is built
      (evaluator.py:107,143), so evaluator_info is NOT reachable from the policy. Confirm
      the route is: run with OracleStateFilter; the diagnostic policy reads
      belief.mean_state() inside act(); it converts to LATENT units by dividing by
      model.survey_scale; it resets its internal belief to a delta on the nearest latent
      bin BEFORE every decision; observe() is a no-op; runtime filtering calls and BASE
      Bayesian CandidatePOMDP.update() calls are zero, while the diagnostic
      identity-conditioning override IS called by PBVI lookahead and must have a positive
      call count.
      State explicitly that abundance_grid is in latent units (faithful_pomdp.py:216,241)
      and that for a fitted model survey_scale = context.observation_scale (~41.45 for fox,
      faithful_fit.py:447-454,552), so snapping RAW truth would be wrong by that factor.
      Confirm the diagnostic wrapper does not inherit _require_public_filter()
      (methods/moor_faithful.py:88).
  (c) FORWARD LOOKAHEAD. State how representative_observations() will return exact
      predictive support and weights for A1/A2, and the realised branch count you expect.
      Confirm you understand that uniform 1/branch weighting is NOT exact here, because
      _deposit_samples spreads each draw barycentrically over two bins
      (faithful_pomdp.py:85-105).
  (d) TRUE REWARD. Confirm you will call the source reward implementation (reward.py:29-63,
      75-117) rather than duplicate the formula, state the penalty mode and alpha, and
      confirm you will NOT set surrogate=None (faithful_pomdp.py:218-220 returns
      action-cost only).
  (e) FIT ARTIFACT. State the fit config digest, cache key and path, fit seed value, and
      your definition of "kernel hash".
  (f) DEGENERACY ACKNOWLEDGEMENT. Confirm you understand that A1-A3 is excluded from the
      decision rule and is collected only as a numerical diagnostic expected to read ~0,
      and that the primary contrasts are A2-A4 (observability) against A1-A2 (model),
      sharing the A2 corner. Confirm you will NOT report an interaction term as a factorial
      quantity. State what magnitude of A1-A3 you would consider a red flag for every other
      contrast.

  (g) Your build estimate, now that you have read the code.

================ PHASE 1 - REGISTERED-KERNEL PARITY GATE ================
Blocking. No planning or experiment code until this passes.
For every E1 cell, every action, all capacity-grid values, and a preregistered abundance
set including N0, s_safe, grid midpoints and action-induced states:
  - compare MechanisticModel.noiseless_next() and next_capacity() against
    ContinuousEcologyEnv.transition_value() and the public control update, IN RAW ABUNDANCE
    UNITS, at tight float64 tolerance;
  - compare a fixed-action 50-step rollout from N0 with process noise off;
  - assert exact action-cost and action-table hashes;
  - OBSERVATION PARITY: the environment observes raw abundance as state * LogNormal(0,sigma)
    (observation.py:19) while the candidate likelihood first divides by survey_scale
    (faithful_ecology.py:236). For the registered raw-unit model the source-supported
    convention is survey_scale = 1.0 and observation_scale = sigma_o. Compare the candidate
    likelihood against LogNormalObservationModel.log_prob() across grid states, BOTH sigma
    values, zero, and representative off-grid observations. NOTE THE TYPES:
    observation_likelihood() returns a DENSITY, log_prob() returns a LOG-density. Compare
    log(observation_likelihood) against log_prob wherever finite, and test the zero-support
    cases explicitly; or normalise both vectors and compare the resulting posterior weights.
    Do not compare them directly. A wrong survey_scale passes every transition test while
    corrupting A3;
  - RESET PARITY: test the deterministic N0 initial prior explicitly against the simulator
    reset;
  - write the parity digest, source rows, transforms, unit and reset conventions to
    E1_RECEIPT.json.
FAIL CLOSED on any mismatch and report. Do not proceed.

================ PHASE 2 - BUILD ================
  1. Registered-parameter Ricker model per your Phase 0 (a), parity-gated by Phase 1.
  2. Diagnostic policy wrapper for A1/A2. Run with OracleStateFilter. Inside act(), read
     belief.mean_state(), divide by model.survey_scale to get latent units, and set the
     internal belief to a delta on the nearest LATENT bin - before every decision.
     observe() is a no-op. Runtime filtering calls and BASE Bayesian
     CandidatePOMDP.update() calls are zero; the diagnostic identity-conditioning override
     is called by PBVI lookahead (pbvi.py:42-46) and must have a positive call count.
     Do not inherit _require_public_filter().
  3. Diagnostic CandidatePOMDP subclass overriding representative_observations() to return
     exact predictive support and weights for A1/A2, with the conditioning step returning a
     delta on the branch bin. Record realised branch counts.
  4. Explicit true-reward branch calling the source reward implementation, selected by an
     explicit flag recorded in the receipt.
  5. Fitted arms: frozen MOOR path unchanged; fit once per cell; store; reuse
     byte-identically in A2 and A4.

ARMS
  A1 registered params + true state    A3 registered params + noisy survey
  A2 fitted params  + true state       A4 fitted params  + noisy survey
HELD FIXED across all four: PBVI belief points, horizon, abundance bins; planning objective
(true reward, same penalty mode and alpha); 11-action table; dataset hash, episode-
preserving 80/20 split, 4000-transition budget for fitted arms; the 20 registered
evaluation seeds; true-state-based evaluation reward; P=10; horizon 50; gamma=0.95; the
evaluator.
NOT held fixed, and this must be reported: observation branch count differs between the
true-state arms (exact support) and the noisy arms (7-branch quadrature). This is correct -
there is no observation uncertainty to quadrature in a true-state arm - but it is an
asymmetry and must appear in the receipt and the write-up.

SEEDING - CORRECTNESS REQUIREMENT
Kernel construction and planning use a planning RNG seed FIXED and INDEPENDENT of the 20
evaluation seeds; one preregistered common planning seed per cell across all four arms.
Sharding distributes evaluation rollouts only. Record the planning seed separately in the
receipt. Rationale: S2's policies were seed-specific and shared episode draws with
evaluation; that channel must not reopen.

================ PHASE 3 - VALIDATE, OPEN NO RETURNS ================
  V1. Small-POMDP exact-solution probe, target 1e-5.
  V2. A1/A2, at EVERY step, full episode, every cell: assert
      grid[argmax_bin] * model.survey_scale equals the registered discretisation of the RAW
      truth (belief.mean_state()) - the multiplication back to raw units is the point, since
      the grid is latent. Assert belief max > 1 - 1e-9. Assert CandidatePOMDP.update() call
      count is zero. Report model.survey_scale per arm so a unit error is visible in the log.
      For update() use THREE counters, not one: runtime policy filtering (expect 0), base
      CandidatePOMDP.update() Bayesian conditioning (expect 0), and the diagnostic
      identity-conditioning override called by PBVI lookahead (expect > 0). A blanket
      zero-update assertion is wrong - pbvi.py:42-46 must call update() to plan.
  V3. Objective parity: compare the planner's reward against the source reward
      implementation across ALL actions, on both sides of the s_safe boundary, at zero and
      positive abundance. Not one hand-computed value.
  V4. Kernel reuse: assert A2 and A4 used the identical stored kernel hash.
  V5. Zero-evidence counters: report the number of times evidence <= LIKELIHOOD_FLOOR and
      the number of fallback-to-uniform initial beliefs, per arm. Expect zero in A1/A2 by
      construction; a non-zero count in A3/A4 must be reported, not suppressed.
  V6. Discretisation probe: 41 vs 61 abundance bins, same arm, cell and seeds.
  V7. Budget probe on the noisy arms AND on A2: horizon 5->15 and 32->256 belief points.
      A2 is included because its exact-support branch set may be large while the reachable
      graph ignores weights, so 32 belief points may bind hardest there. Report support
      size and support coverage per arm.
  V8. QMDP comparator on the identical CandidatePOMDP. NOTE: no QMDP implementation exists
      over CandidatePOMDP; native_solver.py:108-186 uses a different grid/model/reward
      route and is NOT a valid comparator. Implement QMDP over the same POMDP object or
      report that N2 cannot be answered.
  Report all eight. Do not proceed to Phase 6 until the Phase 5 audit passes.

================ PHASE 6 - RUN (after audit and sealed predictions) ================
  4 arms x 4 cells: fox x ricker x {0.1,0.2}, tiger x ricker x {0.1,0.2}.
  Shard by evaluation seed: 320 array tasks, under the 500-job ceiling.
  Pin --constraint=EPYC9534 on comp. One architecture only.

================ PHASE 7 - OUTPUTS ================
  Per (cell, arm) value with paired 95% intervals across the 20 seeds. State the CI method
  (paired-t or paired bootstrap with the number of resamples) and the sampling unit
  explicitly; pair on seed.
  Contrasts A2-A4 (observability), A1-A2 (model) and A3-A4 (model at noisy state), each
  with paired intervals. Cell-wise. No pooling, no single headline.
  A1-A3 reported SEPARATELY and labelled "numerical diagnostic - zero by construction".
  The algebraic identity A4-A2-A3+A1 may be listed ONLY if labelled "diagnostic identity,
  not a factorial estimand". Do not present it as an interaction result.
  Holdout predictive error for the fitted arms.
  Per-arm compute: planner seconds and model evaluations.
  All probe outputs (V1, V5-V8) as named artifacts.
  The sealed prediction reproduced verbatim alongside the outcome.
  E1_RECEIPT.json: arm definitions, parity digest, objective flag, penalty mode and alpha,
  kernel hashes, fit seed and config digest, planning seed (separate from evaluation
  seeds), realised branch counts per arm, planner budget, architecture, code digest, and an
  explicit note that E1 arms received private K_ref/s_safe and are NOT benchmark methods.
```

---

## 6. Command for Claude — independent audit

```text
Independent read-only audit of the E1 implementation in DeepRL_Population_Models. You did
NOT write this code. Find what is wrong before any scientific number is trusted. Modify
nothing. Write one report.

Controlling documents: RESEARCH_PLAN_05_STATE_AND_MODEL_UNCERTAINTY.md section 7.1 and
E1_IMPLEMENTATION_BRIEF.md sections 4.0-4.4.

CONTEXT: an independent line-review of this project's S2 oracle previously found a headline
result was a bug - the code reproduced its archived numbers to 1e-15 and was still wrong,
because THE ERROR WAS IN WHAT THE CODE MEANT, NOT WHETHER IT RAN. A first version of this
implementation brief was also withdrawn after three of its source claims were falsified by
direct inspection. Assume the same class of error is present. Passing tests is not evidence.

CHECK 1 - REGISTERED-KERNEL PARITY. Highest priority: this is the most likely route to a
confident wrong number, and it survives every other check.
  Do NOT trust the implementer's parity receipt. INDEPENDENTLY recompute it: compare
  MechanisticModel.noiseless_next()/next_capacity() against
  ContinuousEcologyEnv.transition_value() and the public control update, in raw abundance
  units, over every action, capacity grid value and a broad abundance set including N0 and
  s_safe. Verify the signed-rate split, the survey_scale source, the unit convention and
  the reset representation. A mis-mapped model is row-stochastic, hashes cleanly and plans
  without error; part of A1-A2 would then be constructor mismatch reported as a
  fitted-model gap.

CHECK 2 - ARE THE TRUE-STATE ARMS TRUE-STATE?
  (a) UNITS FIRST. abundance_grid is in LATENT units (faithful_pomdp.py:216,241) and for a
      fitted model survey_scale = context.observation_scale (~41.45 for fox,
      faithful_fit.py:447-454,552). Verify empirically, running A1 AND A2 for a full
      episode, that grid[argmax_bin] * model.survey_scale equals the registered
      discretisation of raw truth AT EVERY STEP. A raw-onto-latent snap would put A2's
      belief at or beyond the top bin while returning plausible values - check the bin
      index distribution, not just that a point mass exists.
  (a2) Verify the route is OracleStateFilter -> BeliefState -> belief.mean_state() read
      inside act(), not an attempt to reach evaluator_info (which PublicTransition excludes,
      types.py:25-40).
  (b) Verify the update-call split, using three counters rather than one: runtime policy
      filtering = 0; base CandidatePOMDP.update() Bayesian conditioning = 0; diagnostic
      identity-conditioning override during PBVI lookahead > 0 (pbvi.py:42-46 must call
      update() to plan at all). If BASE update ever fires, check whether the zero-evidence
      fallback posterior=predicted (faithful_pomdp.py:186-207) or the fallback-to-uniform
      initial belief (faithful_pomdp.py:18-23, 161-179) fired.
  (c) Verify the FORWARD lookahead assumes perfect future observation. Inspect
      pbvi.py action_values() and the overridden representative_observations(). A planner
      that sees truth now but anticipates noisy futures measures a different quantity.
  (d) Verify the returned branch weights are the predictive probabilities, not uniform
      1/branch_count. Uniform is NOT exact here - _deposit_samples spreads each draw
      barycentrically over two bins.

CHECK 3 - OBJECTIVE
  Trace the objective actually executed - do not trust the flag name. Verify it calls the
  source reward implementation rather than a duplicated formula, and independently compare
  planner reward against reward.py across ALL actions, both sides of the s_safe boundary,
  and at zero abundance. Confirm surrogate=None (cost-only, faithful_pomdp.py:218-220) is
  used nowhere.

CHECK 4 - HELD-FIXED AND PROVENANCE
  - Diff src/tracks/** against the frozen trees. Verify the 107 provenance hashes and
    ledger SHA. Confirm no accepted artifact was touched.
  - Diff the receipt against the spec's held-fixed list item by item: action table, PBVI
    budget, gamma, horizon, evaluator, dataset hash, split, cell scope, penalty mode, alpha.
  - Verify A2 and A4 used the identical stored kernel, and that the fit used the declared
    frozen MOOR config, data split and cache key.
  - Verify the planning RNG is genuinely independent of the 20 evaluation seeds. If a shard
    index or evaluation seed reaches kernel construction or belief-point selection, the S2
    foreknowledge channel has reopened. Trace it.
  - Verify the receipt records the private-constants privilege, realised branch counts and
    the parity digest.

CHECK 5 - PROBES AND STATISTICS
  - Is the QMDP comparator over the IDENTICAL CandidatePOMDP? native_solver.py:108-186 uses
    a different grid/model/reward route and is not valid. If no valid comparator exists,
    N2 is unanswered - say so.
  - Do the budget (horizon 5->15, 32->256) and discretisation (41 vs 61) probes use the
    same arm, cell and seeds?
  - Is the CI method stated, is pairing genuinely on seed, and is the sampling unit correct
    given episodes_per_seed (evaluator.py:31-40)?
  - Are A2-A4, A1-A2 and A3-A4 each reported with paired intervals, cell-wise, no pooling?
  - Is A1-A3 labelled a numerical diagnostic rather than an observability gap, and is the
    algebraic identity A4-A2-A3+A1 either omitted or labelled as a diagnostic identity
    rather than a factorial estimand? Flag any presentation of it as an interaction result.
  - Was the sealed prediction timestamped BEFORE the run, and were returns unopened until
    after this audit?

CHECK 6 - NUMERICAL SANITY
  Any NaN, all-zero likelihood, fallback-to-uniform belief, or non-zero zero-evidence
  counter? If A1 < A3 or A1 < A2, that is NOT automatically a bug - A1 is a matched
  full-state PBVI reference, not a mathematical upper bound, and PBVI is approximate.
  Report it as a diagnostic and say which explanation you favour.

CHECK 7 - ADVERSARIAL
  Name the single most likely way these numbers are wrong in a way every check above would
  pass. Then look for it specifically.

VERDICT: TRUST / TRUST WITH CAVEATS / DO NOT TRUST, findings ranked
BLOCKER / SHOULD-FIX / NICE-TO-HAVE with file and line evidence. Say NOT VERIFIABLE where
you cannot check. Do not interpret the scientific returns - your verdict is whether they
MAY be opened.
```

---

## 7. Decision rule — fixed before any return is opened

**Primary contrast:** **A2−A4** (observability, under the fitted model) against **A1−A2**
(model, at true state), sharing the **A2** arm. A1−A3 is excluded from the decision — it is
degenerate (§1) and is reported only as a numerical diagnostic. **No factorial interaction
is estimated.**

**Margin test — corrected.** A positive gap is claimed only when its **paired 95% lower
confidence bound exceeds 0.10** return units. A gap is declared *not distinguished* when
its interval lies entirely within **[−0.10, +0.10]**. Anything else is *indeterminate* and
is reported as such.

> Revision 1 said "interval excludes 0.10", which is wrong: an interval of [0.01, 0.05]
> excludes 0.10 while demonstrating the opposite of the claim.

Margin value 0.10 is the project's existing preregistration (`DECISION_LOG.md` D-006).

**No-reversal requirement:** the model-gap ordering must also hold at noisy state, i.e.
A3−A4 must agree in sign and rough magnitude with A1−A2. The observability axis has only
one valid measurement (A2−A4), so no reversal check is available for it — state that
limitation rather than implying one.

**Row precedence**, applied in order:

| # | Condition | Action |
|---|---|---|
| 1 | Neither gap's lower bound exceeds 0.10 in any cell | **Do not build E2.** Fix the benchmark or write the methodological result |
| 2 | Model gap flips sign between A1−A2 and A3−A4 | **No dominance claim** — the model effect is not stable across state conditions. Report both; proceed to E2 only if A2−A4's lower bound exceeds 0.10 |
| 3 | A2−A4 lower bound > 0.10 and exceeds A1−A2, model gap consistent across rows | **Proceed to E2** |
| 3b | A1−A3 approaches 0.10 | **Override.** Numerical error is the size of the effect — re-baseline at finer discretisation before any claim |
| 4 | A1−A2 lower bound > 0.10 and exceeds A2−A4, consistent across rows | **Redirect the contribution to the model axis** |
| 5 | Otherwise | Indeterminate. Report and decide with the probe results and the A1−A3 diagnostic in hand |

**Across cells:** the two fox cells (σ = 0.1, 0.2) are the evidence; tiger is a control. If
the two fox cells disagree on which gap dominates, the result is **indeterminate** — report
both, do not average, and do not let one cell carry the gate. Tiger cannot veto a fox
result, and a tiger result cannot substitute for a fox one.

---

## 8. Things that must not happen

- E1 values tabulated next to accepted method rows. **A4 is not accepted PLUS.**
- Any claim ranking family uncertainty against state or model uncertainty.
- The fitted-model gap called "parameter estimation cost" — the fit comes through the
  **noisy** training log.
- A single headline number pooled across cells.
- Returns opened before Phase 3 validation, Phase 4 sealed predictions and Phase 5 audit.
- Any modification to `src/tracks/**` or accepted artifacts.
- Proceeding past Phase 0 or Phase 1 without an explicit human go-ahead.

---

## 9. What changed from Revision 1, and why

`AUDIT_IMPLEMENTATION_BRIEF.md` inspected the repository directly and falsified three
load-bearing claims. All three were verified independently before this revision.

| Revision 1 claim | Status | Consequence |
|---|---|---|
| True-state delivery is a model-level likelihood change | **FALSE** | `moor_faithful` filters `result.observation`; the evaluator passes only the survey. Revision 1 would have snapped **the noisy survey** and A1/A2 would have measured the wrong thing while running cleanly. **Replaced by direct belief assignment (§4.2)** |
| Snap-to-nearest-bin makes evidence strictly positive | **FALSE** | Barycentric deposit means the true next state's bin can lie outside the kernel row's support → evidence zero → `posterior = predicted`. **Dissolved by never calling the base Bayesian update** |
| Uniform branch weighting is exact under deterministic Ricker | **FALSE** | A point-mass belief predicts a **two-bin** distribution; A2's fitted `process_scale` need not be zero. **Replaced by exact predictive support and weights (§4.2)** |
| `faithful_ecology.py:291` is a constructor path | **Misleading** | It is `default_model()`, a documented test dummy. **Removed; the real mapping is specified as an open Phase 0 question (§4.1)** |
| Margin = "interval excludes 0.10" | **Logic error** | Replaced by lower bound > 0.10 (§7) |
| Phase 0 "report before proceeding" | **Not a gate** | A single autonomous turn can print and continue. **Now ends the turn and requires explicit `PROCEED PHASE 1`** |
| Build ≈ 2–4 hours | **Withdrawn** | Estimated from reading, not building, and wrong twice in opposite directions. Phase 0 (f) now produces it |

| Snap the delta at the bin nearest **raw** truth | **UNIT ERROR** | `abundance_grid` is **latent**; for a fitted model `survey_scale = context.observation_scale` (≈41.45 fox). A2's belief would sit at or beyond the top bin while returning plausible values. **Now: snap `truth / survey_scale`, assert `grid[b]*survey_scale` == raw truth** (§4.2) |
| Route truth via `evaluator_info` at the policy | **NOT REACHABLE** | `PublicTransition` deliberately excludes it (`types.py:25-40`) and `observe()` runs before the next oracle belief (`evaluator.py:107,143`). **Now: `OracleStateFilter` → `BeliefState` → `belief.mean_state()` read inside `act()`** (§4.2) |
| "Exact predictive support" implies an exact solve | **Overstated** | `_reachable_graph` discards branch weights entirely (`planners/pbvi.py:66-73`); only the backup uses them. With 32 belief points the solve is not exact. **Now: "exact" qualifies the weights only; budget probe extended to A2** |

**Revision 3 (same day)** corrected the three rows above after a further audit, and added
observation-model and reset parity to the Phase 1 gate: the environment observes
`state × LogNormal(0,σ)` (`observation.py:19`) while the candidate likelihood divides by
`survey_scale` (`faithful_ecology.py:236`), so a wrong `survey_scale` passes every
transition-parity test while corrupting A3.

The `surrogate = None` cost-only trap was the one Revision 1 claim that verified as
correct, and it is retained.

**Also added on the audit's evidence:** the registered-kernel parity gate (§4.1) as a
blocking precondition; QMDP flagged as having no valid existing comparator; explicit CI
method and sampling unit; zero-evidence counters; per-arm compute reporting; row precedence
and an across-cell rule (§7); and the audit-coverage gaps its §2.3 table identified.
