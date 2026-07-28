# Handoff to Codex — the r,K-leak error and the full discussion so far

**Purpose.** Continue an ongoing discussion (previously with another assistant) about an
**experimental-setup error** in the real-ecology offline-RL benchmark: the ecological parameters
that were meant to create *uncertainty* are actually exposed to the agents, so the experiment does
not test what it was designed to test. This file gives you (Codex) the reading order, the error,
the conversation to date, where we landed, and what's open.

**This is a BRAINSTORMING session, not an implementation request.** You have access to **this docs
folder only — no code**. Reason from the design docs; do **not** propose code diffs. The goal is to
think the problem, the fix design, and the framing through *before* anything is handed to the
separate AI agent on the code server. Any claim that can only be settled by looking at the code:
flag it and defer it to that later step.

---

## 0. Read these files first (in this order) to get the whole context

1. **`SERVER_FIX_BRIEF_hide_rK.md`** — PRIMARY. The culmination of this discussion: the problem,
   the root cause (§0), the public/private schema, the fix, acceptance tests, and deliverables.
   Read this first; the rest is supporting context.
2. **`29_6_Real_Ecology_Setting_Implementation_Plan.tex`** — the design/spec doc where the leak
   originates. Look at **E1** and **E8** (they declare r_eff, K_eff, K_base, caps, λ as *known
   observed inputs*).
3. **`29_6_Algorithm_Paper_Audit_Guide.tex`** — method equations, the offline tuple
   `(o,a,R,o',d,ρ,κ,K_eff,…)`, the feature vector with "public-control terms", and the dynamics
   families (Ricker/Allee/θ/regime) with the r₊/r₋ split.
4. **`SERVER_RESULTS_motivation_native_run.md`** — this week's run + the "informational asymmetry /
   natives are near-oracle" diagnosis.
5. **`SERVER_METHOD_ADAPTATIONS_motivation_native.md`** — what `moor_native` / `plus_native` are and
   what they read.
6. **`SERVER_REVIEW_standalone_report_readiness.md`** — the finding that `plus_native` resolves the
   true form in ~50 steps (form-oracle).
7. **`SERVER_RESULTS_general_method_audit.md`** — budget audit (deeper search hurts → model-limited).
8. **`09_motivation_native_results.tex`** — the results write-up.
9. **`SCOPE_AND_CONTRIBUTIONS.tex`** — the intended motivation/contribution (what the experiment was
   *supposed* to show).
10. **`Baselines_State_Action_Reward_Reference.tex`** — original PLUS / Ju(MOOR) settings.
11. Background if needed: `CHAT_HANDOFF_real_ecology_results_and_algorithm.md`,
    `LIMITATIONS_tracker_real_ecology.md`, and last week's deck
    `06_07_Real_Ecology_Experiment_Results_private.pptx`.

*(**You have ONLY this docs folder — no code access.** The actual code lives on a separate server;
code-level verification (git blame, grep) and the fix itself are a **later step for the server
agent**, not part of this session. Treat any code-level claim here as *to be confirmed later*.)*

---

## 1. The error, in one paragraph

The benchmark was meant to test whether general offline model-based RL can beat **naive ecological
baselines under uncertainty** (unknown demographics / unknown model form). But the environment
**publishes the ecological parameters** — per-action growth rate `r(a)/r_eff` and carrying capacity
`K` — as **observed inputs the agents use**. So there is essentially **no parameter uncertainty**:
the native ecological solvers read `r, K` and plug them into a near-correct Ricker model → they
become **near-oracle**, which is why they dominate general MBRL (the "inverted" result). It is a
**structure-known** experiment, not the hidden-parameter one intended.

## 2. The conversation so far (the sequence of realizations)

1. **Why did the result invert?** This week the native ecological baselines beat general MBRL
   860/864; last week the general learners had "won." We reconciled it: the general-learner return
   was ~unchanged (7.74→7.70 on recoverable safe); the **ecological baseline jumped** (6.28→8.52)
   because last week used the *adapted* particle-MPC PLUS/MOOR while this week uses their *native*
   discretize-and-solve form. **The baseline got stronger; the learners didn't get worse.**
2. **Digging into "why native is so strong"** led to the real cause: the env exposes `r, K`, so the
   native solver has a near-exact model. Both weeks used the same public-table env, so **neither week
   tested the intended uncertainty** — last week just had a weaker ecological opponent.
3. **`plus_native` is a form-oracle, not naive.** It carries a posterior over the 4 candidate forms
   and resolves the true one in ~50 steps → it effectively "knows" the model. So the *structural*
   uncertainty is also compromised, separate from the r,K parameter leak.
4. **Root cause found.** It is a **design/spec error faithfully implemented**, not a coding bug: the
   implementation plan (E1/E8) declared `r_eff, K_eff, K_base, caps, λ` as *known observed inputs*,
   conflating **"known population identity" with "known demographics."** The code did what the doc
   said. It went uncaught because there was a *reward*-leakage guard but **no parameter-leakage
   guard**, and no public/private schema was derived from the motivation.
5. **Refinement (important).** `r, K` being *in the action/species tables* is **fine and necessary
   — for the simulator**. The leak is the **methods reading them as inputs** (E8: *"the methods only
   need to read the (already known) (r_eff, K_eff)"*). And the **Ricker form prior** of
   `moor_native` is **not** a leak — it is a legitimate ecological modeling choice and the genuine
   **structural-uncertainty axis** (misspecified on 3 of 4 families).
6. **Two distinct advantages** the ecological side gets: (a) exact **parameters** (r, K); (b) a
   near-correct **functional form** (Ricker). General RL gets neither in usable form — it learns a
   near-blackbox model. General RL's real handicap is the **form burden**, not parameter access.
7. **Partial uncertainty does exist:** C (Allee threshold), θ, and regime z **are** hidden — so the
   setting is not unreasonable, but that residual uncertainty was second-order because the dominant
   dynamics (r, K + near-Ricker) were handed over.

## 3. Where we landed — the fix and the framing

- **The fix (see `SERVER_FIX_BRIEF_hide_rK.md`):** hide `r, K` (and all aliases: ρ, κ, K_eff/base/
  max/ref, r_min/max/base, λ, dK_step, dN) from **all methods**, keeping them only in the private
  evaluator sidecar for the simulator; sanitize the public dataset methods may load; fix indirect
  leaks (`K_ref = K_base`, `s_safe = c_safe·K_base`); strengthen the reward guard; and make the
  native solvers **estimate** r, K from data (`fit_from_public_data()`), not read the tables.
- **Structural axis (§6.7 of the brief):** never leak `true_family` (or aliases); `moor_native`
  stays single-Ricker (misspecified on Allee/θ/regime); `plus_native` is reported *separately* as a
  closed-world model-selection solver, not the naive headline; do **not** redesign its candidate set
  in this first fix.
- **Framing / reframe:** two regimes — **structure-known** (data-rich; current run, keep as a
  labelled comparison arm) vs **structure-unknown** (hide r, K; the intended uncertainty test, and
  the clean motivation). Neither prior week did the structure-unknown run.
- **Acceptance tests:** table-independence (corrupt the tables → all method training/fit/features
  byte-identical) and family-relabel (relabel `true_family` → artifacts unchanged).
- **Root-cause requirement (§0 of the brief):** the code-server agent must first trace and report
  where the exposure was decided (doc line + git blame), whether spec-vs-code, and a one-line
  prevention rule.

## 4. Open questions / what to help with next

- From the **design docs** (not the code), trace where r,K exposure is specified (implementation
  plan E1/E8; the feature-vector spec in the algorithm guide) and sanity-check that the diagnosis
  holds. Mark anything that would need **code-level confirmation later** (the server agent will do
  git blame / grep) — don't assert it now.
- Pressure-test the **fix design** in `SERVER_FIX_BRIEF_hide_rK.md`: is the public/private schema
  airtight? any remaining side-channel? is the `fit_from_public_data()` contract sufficient?
- Advise on the **structure-unknown** design decisions still open (brief §9): population identity
  known-label vs hidden; `expose_rk ∈ {hidden, noisy, full}`; whether/when to build an "open-world
  PLUS" (candidate set with distractors / truth sometimes absent).
- Sanity-check the **framing**: is "structure-known vs structure-unknown regimes" the right honest
  narrative, and is the current run correctly positioned as a *data-rich comparison arm* (a valid
  result) rather than a discarded artifact?

## 5. How to continue (your role, Codex) — brainstorming only

This is a **brainstorming session from the docs — no code access, no implementation.** Read §0's
files, then confirm you understand (i) the error (r,K leaked to methods; form also compromised via
plus_native), (ii) the root cause (spec E1/E8 faithfully implemented; conflated identity with
demographics), and (iii) the fix + framing above. Then help the user **think it through before
anything is handed to the server agent**:

- pressure-test the diagnosis and the fix design — is the public/private schema airtight? any
  remaining side-channel? is the parameter-vs-form distinction right?
- stress-test the **framing** — structure-known vs structure-unknown regimes; is the current run
  fairly positioned as a valid "data-rich" comparison arm rather than a discarded artifact?
- work through the open design decisions (§4 and brief §9): population identity known-label vs
  hidden; `expose_rk ∈ {hidden, noisy, full}`; whether/when to build an "open-world PLUS".

Where a claim can only be settled by reading the code, **say so and defer it** — the code
verification and the actual change are a later, separate step for the server agent.
