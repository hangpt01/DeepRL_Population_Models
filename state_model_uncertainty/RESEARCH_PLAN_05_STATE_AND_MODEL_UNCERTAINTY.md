# Research plan 05 — state and model uncertainty, and where the contribution lives

Written 4 August 2026 (Australia/Melbourne). **This is the current plan.** It supersedes
the forward-looking sections of `CHAT_HANDOFF_04_STATE_AND_NEXT.md` §7 and
`NEXT_WORK_QUEUE.md` §2–§8. Everything those files record as *results* still stands and
is cited below.

The accepted 144-cell benchmark result is frozen and untouched
(`MATCHED_P10_144_METHOD_CELLS.csv`, SHA `7431318803e468c13ff3008acdc530c0d9f9a919cc29a6fe04951fc8a2cadc1c`).
No accepted number has changed at any point in this programme.

---

## 0. How to audit this document

Every factual claim below carries a pointer to the file and section that establishes it.
An auditing agent with read access to this folder should be able to check each one
without running anything. Evidence tags follow
`UNCERTAINTY_TAXONOMY_AND_STAGED_DIAGNOSTIC_PLAN.md` §0:
**[E]** accepted table · **[D]** derived analytically · **[A]/[A2]** read-only code audit ·
**[E3]** measured on the accepted evaluator · **[H]** hypothesis, untested.

Reading order for an auditor:

1. This file.
2. `NEXT_WORK_QUEUE.md` §1 (sealed predictions) and §1-OUTCOME (results record).
3. `DECISION_LOG.md` D-001 → D-011; **D-009 and D-011 are the load-bearing ones.**
4. `UNCERTAINTY_TAXONOMY_AND_STAGED_DIAGNOSTIC_PLAN.md` §0 (executive summary),
   §1.2 (the uncertainty-by-method matrix — **the single most important table for the
   new framing**), §3.5 (the 2×2 factorial), §9.3 (hypothesis table), §11 (open items).
5. `in new repo/S2_ORACLE_REVIEW.md` — the retraction in full detail.
6. `S6_THREE_CELL_RESULTS_AND_THETA_DIAGNOSIS.md` — the replication verdict.
7. `S4_STATE_MODEL_FACTORIAL_SPEC.md` and `S5_MODEL_CLASS_VS_PLANNER_SPEC.md` — the two
   specs that survive the reframing.

**Three things an auditor should specifically check for and flag if found:**

- Any reintroduction of the retracted fox VPI **0.128**, or of the claim "family
  uncertainty is real / the motivation survives". Both are withdrawn (§4 below).
- Any claim that PLUS beats MOOR, or that family-averaging helps, stated without the
  seed-stability caveat. S6 refuted the ordering (§3.6).
- Any P=5 return quoted in a P=10 context, or a constant-action policy presented as a
  seventh method. Both are standing guardrails
  (`CHAT_HANDOFF_04_STATE_AND_NEXT.md` §8).

---

## 1. The reframing — what the project is now about

### 1.1 What it used to be about

> Build a method for **uncertainty over the population-model family**
> (Ricker / Allee / theta-logistic / regime-switching).

That framing is **retired.** Not because it was refuted outright, but because the cheap
gating experiment came back negative and the more interesting question was sitting
next to it the whole time.

### 1.2 What it is about now

> Under partial observation and a limited offline budget, **who handles state and model
> uncertainty, at what cost — and can a general offline-RL method be given the state
> uncertainty it currently lacks, so that it matches the adapted ecological planners
> without their mechanistic prior?**

The contribution is the second clause. **PLUS keeps state uncertainty and
transition-model uncertainty in separate objects** — a belief over latent abundance per
candidate, crossed with a posterior over the candidate bank — and combines them only at
the decision. Every general method in the comparison either represents one of the two, or
fuses both into a single observation-space residual. **The proposal is to keep that
factorisation and drop the mechanistic prior**, replacing eight hand-specified Ricker
candidates with a posterior over learned latent transition models. The hypothesis to be tested is that this **architecture** — explicit latent state plus an
explicit model posterior — rather than the Ricker equation, is what the ecological
advantage consists of. It is a *bundle* claim, not a proof that separation per se is the
mechanism (§5.4), and any result is **limited to this benchmark** until replicated
elsewhere (§6, §10). See §5.4–§5.6.

**The motivating result comes first, and it is an experiment, not an assertion.** Before
proposing anything, measure whether deployment-state uncertainty and transition-model
uncertainty actually cost return on this benchmark, and which dominates — under one
planner, one objective, and one convention, so the two numbers are comparable to each
other. That is **E1** (§7.1), scoped to Ricker environments for reasons given there.

**What E1 does not do**, and this is a deliberate narrowing made on 4 August: it does not
rank these against *family* uncertainty. The family result stands separately (§4) and is
measured under a different information structure; putting them in one table would repeat
the error that produced the retracted 0.128.

### 1.3 Why this is better supported than the old framing

This is not a pivot away from a failure. It is the hypothesis that was **on the original
list and never run**:

> `MY_RESEARCH_QUESTIONS.md` §2, hypothesis 6 — *"State or fitted-transition uncertainty
> matters more than family uncertainty."* Status: **OPEN. This is the 2×2 factorial (S4).
> Not run.**

Three weeks of work established that hypothesis 5 (family uncertainty) is a dead end on
this benchmark. Hypothesis 6 — the one that was never tested — is where the contribution
lives. That is a clean, honest narrative, and it is the story the supervisor report tells.

**One precision about hypothesis 6's own wording.** It says state/transition uncertainty
"matters *more than* family uncertainty". E1 as scoped does **not** test the comparative
clause — it tests the first half, that state and transition-model uncertainty matter at
all and which of the two dominates. The comparative clause would need family crossed with
state information, i.e. a belief over (state × family), which is out of budget (§7.1,
§8). Hypothesis 6 is therefore **partially** addressed, and the write-up must say which
half.

---

## 2. The hypotheses, restated for the new framing

Old hypothesis numbering (H1–H15) is preserved where it still applies so the audit trail
survives; new ones are **N1–N7**.

### 2.1 The new hypothesis set

| # | Hypothesis | Status | Evidence / where it gets tested |
|---|---|---|---|
| **N1** | **The deployment-observability gap and the fitted-Ricker model gap are each non-zero, and one is larger** — both measured under one planner, one objective, one convention, sharing the A2 arm | **OPEN — this is the gate** | **E1** (§7.1), Ricker environments only |
| **N1a** | Observability gap non-zero: **A2−A4** lower bound exceeds the margin | OPEN | E1, **fitted-model row only** — the registered row is degenerate |
| **N1b** | Model gap non-zero: **A1−A2** lower bound exceeds the margin | OPEN | E1, true-state column |
| **N1c** | The model gap is consistent across state conditions: A1−A2 and A3−A4 agree in sign | OPEN | E1, secondary check |
| — | **A1−A3 is a numerical diagnostic, not a hypothesis.** It is zero by construction (deterministic dynamics + known model + known `N0` ⇒ surveys uninformative); its measured value reads grid and solver error | — | E1 diagnostic output |
| **N2** | Belief planning, not merely filtering, is what captures it: PBVI > QMDP on an identical model | **OPEN** | **E1**, same solve. Taxonomy §4.5, §3.7 |
| **N3** | **The gap is real: most general methods have no observation model.** OGSRL, BA-MCTS and EVD do not consume `σ_o` at all; only PLUS, MOOR and RefPlan model observations | **ESTABLISHED [A2]** | `READONLY_EXTRACTION_PASS_2.md`; taxonomy §1.2 (U1 row), §9.3 H15, §11.B item 8 |
| **N4** | **An explicit latent-state + model-posterior architecture beats observation-space fusion.** Note: a *bundle* claim — model class, state representation and observation likelihood co-vary together. It does **not** isolate factorisation (§5.4) | **OPEN. The accepted returns do not support it descriptively** (§5.5) | **E2**, P1 vs P4 |
| **N4′** | **It transfers off the mechanistic prior:** the same architecture over *learned* transition models matches PLUS, under a matched objective and a preregistered margin | **OPEN — this is the contribution** | **E2**, `S5_MODEL_CLASS_VS_PLANNER_SPEC.md` NEW cells |
| **N5** | **It is not the Ricker equation:** a flexible non-mechanistic latent model inside that factorisation closes the gap too | **OPEN** | **E2** = S5 NEW cell B |
| **N6** | It does so at materially lower compute than PLUS (15,314 s/cell) | **OPEN — measurable for free in E2/E3** | Compute figures: taxonomy §0.2 [A2] |
| **N7** | The result survives independent collection and fit seeds | **OPEN — mandatory, not optional** | **E3**, using the S6 protocol. S6 proved single-seed fox claims are worthless (§3.6) |

> **Withdrawn, 4 Aug — recorded, not deleted.** An earlier N1 claimed the three
> uncertainties could be *ranked against each other on one common scale*, with family
> last. That claim is **retracted before it was tested.** The family evidence (full-state
> VPI ≈ 0, cross-family regret 0.180–0.594) is computed under a different information
> structure and a different planner convention from anything E1 produces; putting them in
> one table would repeat the error that produced the retracted 0.128. The family result
> stands **separately and on its own terms** (§4). See §7.1, "What E1 can and cannot
> establish".
>
> An earlier **N1c** claimed a factorial interaction `A4−A2−A3+A1 ≠ 0`. Withdrawn: with
> `A1−A3 ≈ 0` by construction the interaction reduces algebraically to approximately the
> observability effect itself, so it is **not an independent estimand.** It may be reported
> as a labelled diagnostic identity, never as a factorial result.
>
> An earlier N4 claimed **separation itself** was the mechanism. Also withdrawn:
> `P(x,j|h) = P(j|h)·P(x|j,h)`, so factorised and joint are the same object and the claim
> is vacuous as stated. N4 above is the testable replacement (§5.4).

### 2.2 The original seven, and where they now sit

From `MY_RESEARCH_QUESTIONS.md` §2:

| # | Original hypothesis | Status now | Role in the new framing |
|---|---|---|---|
| 1 | Ecological model structure is a strong inductive bias | **OPEN** | **Absorbed into N4/N5.** No longer tested on its own — N5 is the sharper version |
| 2 | General methods insufficiently tuned → **under-resourced** (26×; PLUS 6,126× EVD) | Measured, untested at parity | **Becomes an asset (N6)**, not a threat: if the prototype wins at low compute, the asymmetry argues *for* the contribution |
| 3 | 4,000 transitions insufficient | OPEN (= same claim as #1) | **Dropped.** Only matters for the framing being abandoned (§8) |
| 4 | Reward produces a dominant/constant policy | **CONFIRMED, strongly** [E3] | **Background constraint.** Determines which cells can support any claim (§6) |
| 5 | Family uncertainty doesn't change the optimal policy | **LARGELY ANSWERED** | **Closed and retired** (§4). Becomes one paragraph of motivation-clearing in the write-up |
| **6** | **State or fitted-transition uncertainty matters more than family uncertainty** | **OPEN — never run; E1 addresses the first half only** | **This is now the thesis.** N1 tests that state and model uncertainty matter and which dominates. The *comparison to family* is out of scope (§1.3, §7.1) |
| 7 | Planner approximation controls performance | **PARTLY** | **Confound to control**, not a hypothesis. PBVI horizon 5 on a 50-step task must be held identical across the comparison, and budget-doubling reported |

---

## 3. What has been done — the evidence trail

Chronological, each item with its controlling file. All of it is interpretation and
diagnostics built on top of the frozen accepted result; no accepted number changed.

### 3.1 The accepted benchmark (before this programme)

3 species (Crab-eating fox, Amur tiger, Egyptian vulture) × 4 hidden families × 2 noise
levels × 6 methods = **144 accepted method-cells**. True-state reward with a hard safety
penalty P=10, horizon 50, γ=0.95, 20 evaluation seeds.
Source: `CHAT_HANDOFF_THREE_SPECIES_MATCHED_P10_ECOLOGICAL_BASELINES.md`,
`merged_results_three_species_matched_general_and_ecological.tex`.

### 3.2 Two read-only code audits (26 July)

`READONLY_CODE_AUDIT_INFO_SYMMETRY_AND_METRICS.md` and
`READONLY_EXTRACTION_PASS_2.md`. What they established, with the items that matter most
to the new framing marked **★**:

- **★ `σ_o` is NOT USED by OGSRL, BA-MCTS or EVD** [A2]. Only PLUS, MOOR and RefPlan
  model observations at all. *(This is N3 — the gap the contribution fills.)*
- **★ No method optimises the reward it is scored on.** Five of six plan against a shared
  learned linear surrogate with no safety-threshold term (R² 0.30–0.73); EVD alone trains
  on raw rewards [A2]. Taxonomy §1.3 item 0.
- **★ Measured compute:** ecological 26× general per cell; PLUS 15,314 s/cell vs EVD
  2.5 s = **6,126×** [A2]. Taxonomy §0.2.
- No information asymmetry: `MethodContext` exposes no `s_safe` and no `K_ref` — the
  comparison is fair on that axis (H11 resolved).
- `process_noise_sigma = 0.0` — **there is no aleatoric uncertainty in the benchmark**
  (U3 absent). PLUS/MOOR spend 16 Monte-Carlo paths modelling stochasticity that does not
  exist.
- The four families collapse to the same map whenever the deployed action has `r ≤ 0`
  (`r_pos = max(r_eff,0)`), so family identity is switched off by the policy's own action
  choice in 2 of 3 species [A].
- Reward constants confirmed species-specific: tiger `K_ref` 250 / `s_safe` 25; fox 41 /
  10.25; vulture 325 / 81.25.

### 3.3 S1 — the constant-action sweep (26 July) [E3]

11 constant actions × 24 cells × 20 seeds, 264 rows, evaluator-only, minutes of compute.
Taxonomy §0.3–§0.4.

| Species | Headroom over best constant | Consequence |
|---|---|---|
| Crab-eating fox | **positive** | The only species where any method beats a constant |
| Amur tiger | **zero in 5 of 8 cells** | Mostly non-discriminating |
| Egyptian vulture | **≈ −4.5** | **Every method loses to doing nothing** |

Also: Egyptian vulture is **provably infeasible** [D] — `sup_π x_t = 42.94` against
`s_safe = 81.25`, at any horizon (taxonomy §0.5). PLUS and MOOR deploy a strictly
dominated action (`a5`) in all eight vulture cells.

**This is the cheapest experiment in the programme and it reframed everything.** It
should have been in the original design.

### 3.4 The diagnostic replay fleet and the M1–M15 pipeline (27 July)

Predictions sealed *before* results (`NEXT_WORK_QUEUE.md` §1), outcomes recorded
separately (§1-OUTCOME). 24/24 parity receipts PASS at exactly 0.0; `recomputed_fits = 0`;
accepted CSV verified unchanged.

Key findings:

- PLUS's posterior-averaging is **decision-active** on all four fox cells
  (`switch_vs_MAP` 0.141 / 0.709 / 0.222 / 0.675) but **family-dependent in sign** —
  harmful on Allee (−0.644), helpful on regime (+0.449) and theta (+0.174), neutral on
  Ricker. **Nets ≈ 0.** (D-002; open item 19 closed.)
- The cleanest single demonstration: on tiger/theta the candidates are *never* unanimous
  and disagree enormously (centred 2.657), yet `switch_vs_MAP = 0.000` and return SD = 0
  — because `r_pos = 0` deletes the very term they disagree about. **Inference fully
  engaged; decision untouched.**
- A sealed prediction **missed and was recorded, not buried** (D-003): OGSRL's formal
  constraint never binds anywhere; its kNN *guardian* carries safety and fires on tiger
  (override 0.995 / 0.929), not fox.

### 3.5 Followups (28 July) — S2, H12, H14, reward screen

`NEXT_WORK_QUEUE.md` §1-OUTCOME "FOLLOWUPS FINAL". All PASS parity.

- **H12 confirmed:** the fox advantage is **scale-fragile** — PLUS fox return moves up to
  −1.005 under an `observation_scale` perturbation; tiger invariant. The fitted method's
  fox edge is entangled with abundance-scale calibration.
- **H14 refuted:** the surrogate methods are *not* "safe by accident" — surrogate error is
  large even on visited *safe* tiger states (+3.0 to +4.6).
- **Reward screen:** a graded distance-to-threshold penalty is representable (surrogate R²
  jumps, vulture 0.30 → 0.92–0.96) and moves the tiger and vulture winners.
  `REWARD_REDESIGN_MEMO.md`.
- **S2 as first reported: retracted the next day.** See §4.

### 3.6 S6 — fox replication (30 July) [D-011]

3 of 4 fox cells complete (ricker, allee, regime; 3×3 collection × fit seeds).
`S6_THREE_CELL_RESULTS_AND_THETA_DIAGNOSIS.md`.

- **PLUS − MOOR flips sign in all three cells.** No seed-robust ordering between the two
  ecological methods exists.
- Beating the best constant survives robustly only for **MOOR on ricker** and **both on
  allee**; **on regime, nothing robustly beats the constant.**
- A CI excluding zero does **not** imply seedwise stability — ricker PLUS-minus-constant
  has a positive aggregate interval yet flips sign in at least one combo.
- Theta: 8/9 combos failed on an **OGSRL behaviour-budget bug** (its cost-horizon code
  assumes 25-step episodes; the new seeds produced valid 1–2 step ecological collapses).
  Diagnosed; a behaviour-preserving, branch-isolated fix plus regression tests is written
  up in that file. The accepted theta cell is unaffected.

**Consequence for the new framing: any method claim on this benchmark must be
seed-replicated to be worth anything.** N7 is mandatory, not optional.

### 3.7 Infrastructure

Clean repo `DeepRL_Population_Models` at `/home/hphung/ce25_scratch2/`, built
non-destructively from two frozen tracks, independently reviewed, fixed and re-verified
(an accepted cell reproduces bit-for-bit; 36/36 self-tests; negative-gate test exits
non-zero). D-007. Audit map: `repo_cleaning/REPO_AUDIT.md`, `in new repo/VERIFY_REPORT.md`.

**Two tracks, same package name.** `src/tracks/ecological/…` and `src/tracks/general/…`
are both `real_ecology_benchmark`, each byte-identical to one of two separately-frozen
accepted codebases. **Import one at a time.** They share env/evaluator/reward
byte-for-byte. This matters for E2: the prototype must import the *general* track's
model classes and the *ecological* track's PBVI, which is exactly the boundary the
two-track layout was designed to keep clean.

**Cluster constraint — CHANGED, 4 August 2026.** Earlier handoffs record that a 365-day
`SPEC_NODES` reservation locked the Xeon-8452Y nodes that produced the accepted results
and that this account was denied them. **A capacity snapshot on 4 Aug shows that is no
longer true at the configuration layer** (§9): `m3h` has `AllowAccounts=ALL`, user
`hphung` holds QoS `m3h`, the Xeon-8452Y nodes `m3h100-101` had 72 idle CPUs, and **no
active `SPEC_NODES` reservation covers them** — the four active reservations are
`cryosparc-general`, `sexton`, `CryoemFacility` and `august`, none naming account `ce25`
or nodes `m3h100-101`.

**This is a configuration-layer observation, not a verified capability.** Before treating
accepted-comparison work as unblocked, run a **parity smoke test**: reproduce one accepted
cell (A6-MOOR is the standard anchor, per D-007) on `m3h` with
`--constraint=xenon-8452Y` and confirm `max_abs_diff = 0.0`. Until that passes, keep
treating 1e-9 accepted comparison as unproven. If it passes, the S9 reward axis and any
accepted-baseline arm become available again, which they were not a week ago.

---

## 4. What is retracted — do not reintroduce

**The fox VPI 0.128 is withdrawn.** It was a baseline bug, not a value of information.
`DECISION_LOG.md` D-009; `in new repo/S2_ORACLE_REVIEW.md`;
`SUPERVISOR_BRIEFING_S2_RETRACTION.md`.

The VPI baseline maximised over only **five hand-picked policies** (four family oracles
plus a majority vote), not a genuine family-agnostic optimum. An independent line-review
constructed a simple admissible policy — play the action all four oracles agree on at
t=0, read the exact next abundance (which separates the four families by at least
0.018 across all 20 seeds), then follow the matching oracle — and it reaches
**11.839420699045172** against the perfect-information value **11.839420699045174**.
So 0.128 was *candidate-set shortfall*, not value of information.

**Corrected full-state VPI ≈ 0** (0.0 at 321 bins; −1.8e-15 at 41 bins). The reviewer
reproduced the archived numbers to 1e-15 *and then* broke them with the counterexample,
which is why the correction is trusted.

Secondary findings from the same review, also to be respected:

- The stored `mean_regret` column had **the opposite sign** from the stated definition.
  Corrected fox Ricker row is **positive**: Allee 0.358, theta 0.180, regime 0.594.
- The "exact oracle" is exact only for the discretised/interpolated model; transferred
  policies can occasionally beat the nominal own-family oracle on coarse grids. Call it a
  grid/interpolation oracle, not an unqualified optimum.
- Grid convergence converged the *restricted candidate metric*, not genuine VPI; visited-
  action flip fractions remain 7.9% (fox) and 3.0% (tiger) from 161 to 321 bins, so value
  convergence is not policy stability.

**S2 used no benchmark methods** — it is oracle/planning-only backward induction, so
VPI ≈ 0 is a property of the problem, not an artefact of any method.

**The one honest caveat, which the new framing renders moot:** S2 is a *full-state*
computation, so VPI ≈ 0 rules out the value of family knowledge only under exact
observation. The noisy-observation VPI was never computed. Under the old framing that was
the decisive open number. **Under the new framing it is not, and we are not going to
compute it** (§8).

---

## 5. The gap the contribution fills — and the evidence it is real

This section exists because it is the strongest part of the new story and it is already
established, at zero further cost.

### 5.1 Three of six methods do not model observation noise

From the uncertainty-by-method matrix, taxonomy §1.2, U1 (latent state) row [A2]:

| Method | Latent state / observation model |
|---|---|
| PLUS (adapted) | **E,U** — one belief per candidate; **uses σ_o** |
| MOOR (adapted) | **E,U** — one belief; **uses σ_o** |
| RefPlan-inspired | C — 32 public belief particles; **uses σ_o** via a log-normal observation model |
| OGSRL-inspired | **I — `σ_o` NOT USED**; public belief update with no observation model |
| BA-MCTS-inspired | **I — `σ_o` NOT USED**; tree state is a bucketed observation |
| EVD | **I — `σ_o` NOT USED** |

And the U10 (family) row is **I** for all six — no method represents family uncertainty
at all, which is why the accepted 144 cells could never speak to it as a capability
(taxonomy §1.4).

### 5.2 The taxonomy already flagged this as the likely real mechanism

Taxonomy §1.3, conflation 2, written 25 July and tagged **[H]**:

> *"General methods regress in observation space, so U3 and U1 are fused into one
> residual… Adapted PLUS/MOOR separate them by construction. **This, not the Ricker
> equation, may be the real inductive-bias advantage.**"*

That hypothesis is now the thesis. It has been sitting in the analysis for six weeks,
untested.

### 5.3 The comparison is currently confounded three ways

Taxonomy §0.2 Challenge B. PLUS/MOOR differ from all four general methods simultaneously
in (i) model class, (ii) state representation, (iii) planner — plus a 26× compute gap.
**No accepted cell isolates any one of them.** E2 controls some of these rivals — planner, objective, risk and support treatment are held fixed — and estimates the remaining contrast as a *bundle*. It does **not** isolate a single axis; see §5.4.

### 5.4 The mechanism, stated precisely — it is the *factorisation*, not the observation model

The sharper version of the claim, and the one the plan now runs on:

**PLUS's architecture keeps the two uncertainties separate and combines them only at the
decision.** It maintains a belief over latent state *per candidate* (`b_t^j`) and a
posterior over the candidate bank (`w_t^j`), then acts on
`argmax_a Σ_j w_t^j Q_j(b_t^j, a)`. State uncertainty and transition-model uncertainty are
represented in different objects and never fused.

Every general method in the comparison either represents only one of the two, or fuses
them into a single residual. **The proposal is to keep PLUS's factorisation and drop its
mechanistic prior** — belief over latent state with a correct observation model, crossed
with a posterior over *learned* transition models rather than eight hand-specified Ricker
candidates.

> **The claim, as it can actually be tested.** Under a 4,000-transition offline budget on
> this partially observed conservation POMDP, an **explicit latent-state and
> model-posterior architecture** outperforms observation-space fusion — and a general
> method given that architecture matches the adapted ecological planners without the
> mechanistic prior, at a fraction of the decision-time compute.
>
> **Scope:** this is an architecture-*bundle* claim on *this benchmark*. It does not
> isolate factorisation as the causal component (§5.4 precision 1), and it does not
> license a general statement about partially observed offline RL (§6, §10).

**Two precisions, both forced by review and both limiting.**

1. **"Factorised beats fused" is vacuous as stated**, because `P(x,j|h) = P(j|h)·P(x|j,h)`
   — the factorised and joint representations are the *same object*. What is not vacuous:
   PLUS's **planning** rule is `argmax_a Σ_j w_j Q_j(b_j,a)`, where each `Q_j` is solved
   inside candidate *j* alone, so it never values an action for how it will move `w`; and
   the general methods fuse state and process uncertainty into an **observation-space
   residual**, which is a different model class rather than an approximation of the joint.
   The testable claim is therefore *explicit latent-state and model-posterior representation
   versus observation-space fusion* — and that is an **architecture-bundle** comparison, in
   which model class, state representation and observation likelihood co-vary together. It
   does **not** isolate factorisation as a mechanism, and must not be described as if it does.
2. **E1 does not test this claim.** E1 measures whether the two uncertainties cost return
   at all, with models *fixed* within each arm — there is no posterior-representation
   contrast in it. Only **E2** attacks the representation claim. E1 is the gate that decides
   whether E2 is worth building, not a partial test of it.

### 5.5 The architecture 2×2 — already populated by the accepted methods

Reading the U1 / U5 / U6 / U7 rows of the by-method matrix (taxonomy §1.2) as a factorial
over *which uncertainties are represented separately*:

| | Represents state uncertainty (U1) | Represents transition-model uncertainty (U5/U6) | Method |
|---|---|---|---|
| **Both, separately** | **E,U** — belief per candidate | **E,U** — 8 Ricker MAPs, uniform prior | **PLUS** |
| **State only** | **E,U** — one belief | **I** — point commitment is *defining* (U7) | **MOOR** |
| **Model only** | **I** — tree state is a bucketed observation, σ_o unused | **E,U** — categorical belief updated inside the tree | **BA-MCTS** |
| Conflated / partial | **C** — 32 *public* particles, observation-space regression | E,U — 5-member posterior, observation space | RefPlan |
| Model, not updated | **I** — σ_o unused | E — 5 members, actor **fixed** after training | OGSRL |
| Neither | **I** — σ_o unused | **C** — fused into `Q` variance | EVD |

Taxonomy §1.3 item 1 states the key contrast outright: *"BA-MCTS has model belief but no
state belief… Comparing it with PLUS on a partially observed task therefore compares two
different uncertainties."* **That is the axis the proposal is built on, and three of the
four corners already have accepted returns.**

**The full read-off across all 24 accepted cells, and its verdict: the ordering does not
hold.** This table is **descriptive only** — it is not evidence for the architecture claim,
and win counts are not causal.

| Species | "Both" (PLUS) | "State only" (MOOR) | "Model only" (BA-MCTS) | Verdict |
|---|---|---|---|---|
| **Tiger**, all 8 cells | 4.224 / 4.264 | **identical to PLUS in all 8** | 1.105 down to −0.782 | PLUS's model posterior buys **exactly nothing** over MOOR. And 5 of 8 tiger cells have zero headroom, so the ranking is intervention-cost calibration, not uncertainty handling |
| **Fox**, 8 cells | wins 3 (both regime, Ricker/0.1) | **wins 3** (Ricker/0.2, both Allee) | **wins 2** (both theta) | A **3/3/2 split**. "State only" beats "both" as often as the reverse. MOOR beats PLUS by 0.317 and 0.644 on Allee; PLUS wins both regime cells. Family-dependent reversal, not an architecture ordering |
| **Vulture**, all 8 cells | ties MOOR | ties PLUS | worst | **RefPlan — the "conflated" architecture — wins all 8.** Unusable as evidence either way: the task is infeasible and every method loses to a constant |

Source: `merged_results_three_species_matched_general_and_ecological.tex` (tiger block
ll. 72–87, fox ll. 96–111, vulture ll. 120–135), independently tabulated in
`ADVERSARIAL_REVIEW_PLAN_05.md` §6A.

**The honest conclusion: the accepted returns do not visibly track the architecture.**
Tiger shows the model posterior contributing nothing; fox splits three ways; vulture
inverts the ordering entirely. The axis is also confounded throughout — model class,
planner, risk attitude, support treatment and compute all vary alongside it — so these
cells could not identify the mechanism even if the ordering had held. And every fox number
carries the S6 caveat: those orderings do not survive reseeding (§3.6).

This is why E1 and E2 exist. **The architecture 2×2 motivates a controlled experiment; it
is not itself evidence.** Any write-up that presents this table as support for the claim
is overstating it.

### 5.6 A counterargument, considered and answered

*"RefPlan already models observation noise and it is the worst method on fox — so an
observation model cannot be the lever."*

RefPlan is **not a clean instance of the architecture**. The matrix tags its U1 as **C
(conflated)**, not **E,U**: its 32 belief particles are *public* and it regresses in
**observation space**, so U1 and U3 are fused into a single residual (taxonomy §1.3
item 2) and its "belief" is not over latent abundance. It also differs from PLUS on model
class (ridge-linear black box), planner (sequence search, not belief-state PBVI), and
risk attitude (λ_ref = 0.5 pessimism, which taxonomy §1.3 item 4 notes is a *cost* in a
benchmark where the safety penalty is mostly slack).

So RefPlan's failure is consistent with the claim rather than against it: it is what a
black-box transition model with an observation-noise likelihood bolted on looks like,
which is precisely what the proposal is *not*. **The counterexample to watch is fox-theta
BA-MCTS (§5.5), not RefPlan.**

---

## 6. The problem the new framing must confront honestly

**The benchmark can barely support any method claim, and the new framing needs it to.**

- Only fox has positive headroom over a constant action (§3.3).
- Fox's minimum PBVI top-1/top-2 action-value margin is **0.0005** — ~100× smaller than
  tiger, ~350× smaller than vulture (taxonomy §0.1 item 3).
- The fox advantage moves by up to **−1.0** under an abundance-scale perturbation (H12).
- And S6 showed the fox orderings **do not survive independent seeds** (§3.6).

So "our method beats PLUS on fox" is, on current evidence, a claim the benchmark cannot
carry at a single seed. Three legitimate responses, and the plan should pick deliberately:

| Response | What it costs | Comment |
|---|---|---|
| **(i) Measure the prize first** | E1 | **Recommended.** Note the observability gap is measurable only in the **fitted** row (A2−A4), which does carry fitting-seed fragility; the registered row is degenerate. Neither row is immune to the 0.0005 figure — that is a PBVI *action-value* margin — which is what the budget and discretisation probes are for |
| (ii) Make the benchmark discriminate | S9 graded reward, ~1 day build + ~60 core-h/variant | The reward screen already showed a graded penalty is representable and **converts vulture from an inert cell into a damage-mitigation problem with a gradient** (`REWARD_REDESIGN_MEMO.md`; taxonomy §8.2). This is the one dropped experiment worth **re-promoting** if E1 says the cells are too inert |
| (iii) Always seed-replicate | E3, S6 protocol, ~230 core-h per 3×3 cell | Mandatory for any headline number. Non-negotiable after S6. Note the wall clock is set by **shard length (~10 h per combo)**, not by core-hours — 36 shards run in ~10 h however many cores are free |

**This is also the honest thing to tell the supervisor**, and it is a contribution in its
own right: *any* benchmark cell should ship with its constant-action baseline and its
oracle upper bound before a method is run on it.

---

## 7. The next experiments

Three, gated. Each can stop the programme. **E1 is authorised at ~3 days, full scope
(4 Aug), extended from the original 1–2 day window on the implementer's post-source
estimate. E2 does not fit any near-term window, and
that is the point of running E1 first.**

### 7.1 E1 — the observability and model gaps on Ricker environments (THE GATE)

**Question.** *Does deployment-state uncertainty cost more or less return than
transition-model uncertainty?* Both measured under one planner, one objective, and one
information convention, sharing the A2 arm, so the two numbers are directly comparable.

> **This is not a 2×2 factorial.** One cell of the design is degenerate: with deterministic
> dynamics, a known registered model and known `N0`, an agent observing noisily can still
> propagate the exact state, so `A1−A3 = 0` by construction. The observability gap is
> therefore measured **only** in the fitted row (`A2−A4`), and no factorial interaction is
> estimated. Three informative contrasts plus one degenerate cell. Full reasoning:
> `E1_IMPLEMENTATION_BRIEF.md` §1.

**Scope decision, 4 Aug: Ricker environments only.** The reason is technical and
load-bearing, so it is recorded here rather than in a footnote.

`C` (Allee threshold) and `θ` (theta exponent) are drawn **once per episode in `reset()`
and persist** (`READONLY_EXTRACTION_PASS_2.md`, item 10); the regime family carries an
initial latent regime plus Markov switching under `Π`. A genuinely non-clairvoyant oracle
for those families therefore cannot be built by averaging over the parameter prior at each
transition — that describes a *different* simulator in which the parameter is redrawn
every step, destroying persistence. The faithful object is a belief over
`(abundance, C)` / `(abundance, θ)` / `(abundance, regime)`, i.e. an **augmented state
space** and a new solver. That does not fit the budget.

**Ricker has no episode-level draw at all.** Its map is `x' = m·exp{g(1−m/k)}` — no `C`,
no `θ`, no regime — and `process_noise_sigma = 0.0`. So on Ricker environments the
clairvoyance problem does not arise, one PBVI code path covers every arm, and no
augmentation is needed. This is the only version of E1 that is both correct and affordable.

#### The design — 2×2, one planner, one objective

| | **True state** | **Noisy state (σ_o)** |
|---|---|---|
| **Registered true Ricker parameters** | **A1** | **A3** |
| **Ricker fitted from the 4k log** | **A2** | **A4** |

**Held identical across all four arms:** the PBVI planner and its full budget (32 belief
points, horizon 5, 41 abundance bins; **observation branches — see the exception below**); the planning objective
(**the true reward**, not the learned public surrogate); the 11-action table; the dataset
hash, episode-preserving 80/20 split and 4,000-transition budget for both fitted arms; the
20 registered evaluation seeds; true-state-based reward; P = 10; horizon 50; γ = 0.95; the
evaluator. **Exactly one factor varies in each adjacent simple-effect comparison** (A1↔A3,
A2↔A4, A1↔A2, A3↔A4). A1↔A4 changes both axes and is reported as a total, never as a
single-factor effect.

**The fitted arms use the frozen MOOR fitting path unchanged** — one mechanistic Ricker
model fitted jointly to complete ordered episodes, process-only Monte Carlo with 16 paths,
8 deterministic L-BFGS starts at max 100 iterations, minimum-training-loss selection
(`PLUS_MOOR_PAPER_ALIGNED_IMPLEMENTATION_EXPLANATION.md`, "Frozen MOOR Configuration").
Reusing it rather than writing a new fitter both defines the treatment and removes build
risk. **One fitted kernel per cell, stored and reused byte-identically in A2 and A4** — the
model must not be refitted between the two, or the observability contrast picks up fit
noise. Record the fit seed and the kernel hash in the receipt.

**True-state arms use the same PBVI with an identity observation model** (Variant A,
taxonomy §3.3). This removes the second solver entirely — `backward_oracle()` is **not**
used in E1.

**The convention must be stated precisely, because two natural readings are wrong.**

First, passing a point-mass *current* belief into a PBVI that still anticipates **noisy
future observations** measures "true state now, noisy later". The true-state arms must
assume perfect future observation in the lookahead too.

Second — and this is where an earlier draft of this section was wrong — **true state is
not delivered by changing the observation likelihood.** The faithful policy filters
`result.observation` itself, and `PublicTransition` deliberately excludes
`evaluator_info`. A1/A2 require a **diagnostic policy wrapper** that runs under
`OracleStateFilter`, reads `belief.mean_state()` inside `act()`, converts it to **latent
units** (the abundance grid is latent; divide by `model.survey_scale`), and sets the
internal belief directly to a delta on the nearest bin before every decision.
Runtime filtering and **base Bayesian** `CandidatePOMDP.update()` calls are zero in these
arms; PBVI lookahead calls an **identity-conditioning override**, which is expected and
must have a positive call count (`planners/pbvi.py:42-46` must call `update()` to plan).

**Consequence for the held-fixed list:** the true-state arms use the exact predictive
support rather than the frozen 7-branch quadrature, so **branch count is not held fixed
across arms**. That is correct — there is no observation uncertainty to quadrature in a
true-state arm — but it is an asymmetry and must be reported. Full construction, units and
assertions: `E1_IMPLEMENTATION_BRIEF.md` §4.2.

**A1 is not an upper bound.** More information upper-bounds the value of an *optimal*
policy under nested policy classes; it does not make one approximate horizon-5 PBVI run a
bound on another. Call A1 the **matched full-state PBVI reference**. If A3 ever exceeds
it, that is a solver/approximation diagnostic, not a logical impossibility — and it is
informative, so report it rather than treating it as a bug.

#### Quantities reported

| Label | Contrast | What it actually measures |
|---|---|---|
| **Deployment-observability gap** | **A2−A4 only** (fitted params) | Return lost to acting on noisy surveys instead of the true count. **A1−A3 is degenerate and excluded** — with deterministic dynamics and a known model the agent propagates the exact state, so surveys carry no information; see brief §1 |
| **Fitted-from-noisy-log gap** | A1−A2 (true state), A3−A4 (noisy) | Return lost to learning the Ricker parameters from the existing 4k log |
| ~~Interaction~~ | A4−A2−A3+A1 | **Collapses.** With A1−A3 ≈ 0 the interaction reduces to approximately the observability effect itself. Report it as a diagnostic, not as a factorial term |
| **Total** | A1−A4 | Registered/true-state reference to fitted-Ricker/noisy-deployment, **under E1's true-reward convention** — not a single-factor effect, and **not** the accepted PLUS setting |
| Context | A1 − V(best constant action) | Total value of adaptive control. `V(best constant)` is **already measured** by S1, so this is free. If ≈ 0 the cell is inert |

**The decision rule for N1, preregistered.**
but the observability axis has only **one** valid measurement, so N1 is decided as follows,
fixed before any return is opened:

1. **Primary:** **A2−A4** (observability, fitted model) against **A1−A2** (model, true
   state). These share the **A2** corner. A1−A3 is **excluded from the decision** — it is
   zero by construction (deterministic dynamics + known model ⇒ surveys uninformative) and
   serves only as a numerical diagnostic on grid and solver error.
2. **Consistency requirement:** the model gap must agree in sign across rows (A1−A2 vs
   A3−A4). The observability axis has **one** valid measurement, so no reversal check
   exists for it — state that limitation rather than implying one.
3. **Margin — stated as a bound, not an exclusion.** A positive gap is claimed only when
   its paired 95% **lower confidence bound exceeds 0.10** return units. A gap is *not
   distinguished* when its interval lies entirely within **[−0.10, +0.10]**. Anything else
   is *indeterminate*. (An earlier draft said "interval excludes 0.10", which is wrong —
   an interval of [0.01, 0.05] excludes 0.10 while showing the opposite of the claim.)
   The 0.10 value is the project's existing preregistration, `DECISION_LOG.md` D-006
   (δ_R = δ_VPI = 0.10, justified there against S1's measured fox learning gain of
   +0.23…+1.56). State the CI method — paired-t or paired bootstrap — and pair on seed.

**Estimand labels are deliberately operational, not causal.** In particular the second row
is *not* "parameter estimation cost": the fit is taken through the existing **noisy**
training log, so training-side state uncertainty is inside it. Taxonomy §3.4 warns that
observability has two loci — training and deployment — and that conflating them is a
known trap. E1 manipulates the deployment locus only; the training locus is held at
"noisy" throughout and must be named as part of the estimand.

These are **comparable conditional interventions under a fixed convention, not an additive
partition of uncertainty.** Report the full conditional table; do not collapse it to a
single ranking sentence, and do not report the algebraic interaction as a factorial
estimand — with `A1−A3 ≈ 0` it is not independent of the observability effect.

#### Also collected in the same solve

- **PBVI vs QMDP on the identical model** — this is **N2**, and it is part of E1, not an
  optional extra. If QMDP ≈ PBVI, belief *planning* is not the lever even where the
  observability gap is large, and E2 should aim at the filtering/model axis instead.
- **Budget sensitivity on the noisy arms**: PBVI at the frozen budget (horizon 5 / 32
  beliefs) and at horizon 15 / 256 beliefs. **A correctness guard, not a nicety** — PBVI is
  approximate. Because `A1−A3` is zero by construction, **its measured magnitude is a
  direct read on grid plus solver error** — if it approaches the 0.10 margin, numerical
  noise alone is the size of the effect being hunted, and every other contrast is suspect.
  Apply the enlarged budget to A2 and A4 as well.
- **Discretisation check**: 41 vs 61 abundance bins on one cell (S4 spec §6).
- **Compute actually consumed per arm** — model evaluations and planner seconds, not just
  hyperparameters. Reporting this is *not* the same as matching it; where arms differ,
  say so rather than implying parity.

#### Scope

**Fox × Ricker × {0.1, 0.2}** — the only species with measured headroom (§3.3), and the
one cell family where S6 found a stable result (MOOR above the constant).
**Tiger × Ricker × {0.1, 0.2}** as contrast, expected near-zero `A1 − V(best constant)`.
**Vulture excluded** — provably infeasible (§3.3), and it must not enter any ranking.

Four cells × four arms = 16 planner runs, plus the QMDP, budget and discretisation probes.

#### Outputs and statistical contract

Fixed **before any return is opened**:

- per (cell, arm) value with **paired confidence intervals across the 20 seeds**, paired on
  seed (S4 spec §7);
- the contrasts A2−A4, A1−A2, A3−A4 with paired intervals per cell; **A1−A3 reported
  separately and labelled a numerical diagnostic**; the algebraic identity
  `A4−A2−A3+A1` may be listed only if labelled "diagnostic identity, not a factorial
  estimand";
- **cell-wise reporting** — no pooling across cells, and no single headline number;
- a **preregistered practical margin** below which a gap is reported as "not distinguished"
  rather than as a result;
- holdout predictive error for the fitted arms, so a model that wins on return is also
  shown to fit;
- `E1_RECEIPT.json` with arm definitions, kernel provenance, planner budget, objective,
  seeds, and the planning-RNG seed **recorded separately from the evaluation seeds**;
- the sealed prediction (below) reproduced verbatim alongside the outcome.

#### Registration decisions to make before running

1. **Fix the objective for all four arms: the true reward.** Consequence — **A4 is not the
   accepted PLUS number**, because accepted PLUS planned the learned surrogate. Report the
   A4-vs-accepted-PLUS gap separately as the price of the objective mismatch (taxonomy §10
   item 1, the standing **[Blocking]** decision); never merge the two.
2. **Separate the planning RNG from the evaluation seeds.** Kernel construction and
   planning must be frozen independently of the 20 rollout seeds. Sharding may distribute
   *evaluation* work only. S2's policies were seed-specific and thereby shared the episode
   draws (`in new repo/S2_ORACLE_REVIEW.md`, "Per-episode draws"); that channel must not
   reopen here.
3. **Measure discretisation error two ways** — "continuous-kernel PBVI" is not a defined
   object, so the earlier phrasing is replaced by: (i) one-step transition parity of the
   registered kernel against the continuous simulator (the Phase 1 gate), and (ii) a
   41-vs-61-bin policy-sensitivity probe. Together these bound U11 without requiring an
   undefined planner.
4. **Seal the predicted ordering before the jobs report** — which of the two gaps you
   expect to be larger, and by roughly how much, in the `NEXT_WORK_QUEUE.md` §1 style. A
   prediction recorded afterwards is worth nothing, and this discipline has already caught
   two real errors in this project.

#### What E1 can and cannot establish

**Can:** which of deployment-state uncertainty and finite-data Ricker model fitting costs
more return, on Ricker environments, under a fixed PBVI convention; whether they interact;
whether belief planning beats a plug-in point estimate (N2); and whether each cell has any
adaptive headroom at all.

**Cannot, and the write-up must say so:**

- anything about **population-model-family** uncertainty. E1 runs on Ricker environments
  only. The family question stands separately on the corrected S2 result — full-state
  VPI ≈ 0, cross-family regret 0.180–0.594 — measured under a *different* convention, and
  **must not be tabulated on the same scale**;
- anything about family information under **noisy** observation, which remains unmeasured
  and is a stated limitation (§8);
- a clean "parameter estimation" cost, for the training-locus reason above;
- anything about the non-Ricker environments, whose oracle arms need the augmented state
  space this scope decision declines to build.

#### Why it is affordable — partially VERIFIED against source, 4 Aug

Checked against `github.com/hangpt01/DeepRL_Population_Models` @ `3291eef`:

- **The ecological-track planner is located**:
  `src/tracks/ecological/real_ecology_benchmark/planners/pbvi.py`, with
  `faithful_pomdp.py` alongside it. Note this is **not** under `methods/` — that path,
  cited in `CHAT_HANDOFF_04_STATE_AND_NEXT.md` §6, is wrong.
- **`run_s2.py` already plans the true reward, not the surrogate** — line 404 computes
  `xn/(xn + K_ref) − cost − penalty` directly, and the string `surrogate` appears 0 times
  in that file. So a true-reward objective is demonstrably implementable in this codebase.
- **PARTLY VERIFIED — the planner is reusable, but not everything is model-level.** `PointBasedPlanner.__init__`
  takes an injected `CandidatePOMDP` (`planners/pbvi.py:25-40`), and that object supplies
  the transition kernel, observation likelihood, reward and observation branches
  (`faithful_pomdp.py:42-66`). The 121-line planner is **not** modified. **But the
  true-state arms are a policy-level intervention, not a model-level one** — `moor_faithful`
  filters `result.observation` itself (`methods/moor_faithful.py:90-113`) and the evaluator
  passes only the survey to the policy (`evaluator.py:96-157`). See
  `E1_IMPLEMENTATION_BRIEF.md` §4.2.

**The three changes, now concrete:**

| # | Change | Where | Difficulty |
|---|---|---|---|
| 1 | Registered-parameter Ricker kernel for A1/A3 | construct a `MechanisticModel` from the registered parameters. **Not a field copy** — the action table gives *signed rate set-points* while the model wants separate growth/mortality arrays, there is no registered `survey_scale`, and a zero reset scale is rejected | **Fiddly, and the top source of silent error** — brief §4.1 |
| 2 | True reward instead of the surrogate | `expected_public_reward()` (`faithful_pomdp.py:209-229`) — the next-state abundances are already in scope as `following`, so the true reward is ~15 lines | **Easy, with a trap — see below** |
| 3 | True state for A1/A2 | **Not a likelihood change, and not via `evaluator_info`** (which `PublicTransition` excludes). Run under `OracleStateFilter`; the diagnostic policy reads `belief.mean_state()` inside `act()`, divides by `model.survey_scale` for **latent** units, and sets the internal belief to a delta on the nearest latent bin before every decision. `observe()` is a no-op; runtime filtering and base Bayesian `update()` calls are zero, while PBVI lookahead calls an **identity-conditioning override** with a positive call count | **The real work.** Units and the three update counters: brief §4.2 |
| 4 | **Registered-kernel parity gate** | compare `MechanisticModel.noiseless_next()` against `ContinuousEcologyEnv.transition_value()` in raw units before any planning | **Blocking precondition** — brief §4.1 |

**Trap on change 2, worth writing into the implementation brief.** `expected_public_reward`
falls back to `-action_costs[action]` when `context.surrogate is None`
(`faithful_pomdp.py:218-220`). Anyone implementing "true reward" by nulling the surrogate
gets a **cost-only planner** and a silently wrong arm that will still run and still produce
plausible numbers. The true-reward path must be an explicit branch, and the receipt must
record which objective was used.

**A correction — an earlier draft of this section claimed uniform branch weighting is exact
here. It is not.** `representative_observations` weights branches uniformly
(`1/branch_count`, `faithful_pomdp.py:242`) regardless of the predictive. Zero process
noise makes the Monte Carlo samples equal, but `_deposit_samples` still spreads that mean
**barycentrically across two neighbouring bins** (`faithful_pomdp.py:85-105`), so a
point-mass belief predicts a **two-bin** distribution and seven uniform quantiles
approximate it in increments of 1/7. A2 is worse: its fitted `process_scale` is a *learned
parameter* and need not be zero even though the simulator's is
(`faithful_fit.py:531-570`). **The true-state arms therefore require exact predictive
support and weights**, not the frozen 7-branch quadrature — which means the branch count is
*not* held fixed across arms, and that asymmetry must be reported. See
`E1_IMPLEMENTATION_BRIEF.md` §4.2.

**New registration requirement, surfaced by change 2.** The true reward needs `K_ref` and
`s_safe`, and `MethodContext` deliberately exposes **neither** — that absence is the
information-symmetry guarantee (taxonomy §0.2 Challenge C, H11 resolved). So E1's arms must
be given private constants that no benchmark method receives. That is legitimate for an
oracle/analysis construct measuring problem structure, but it must be **registered
explicitly**, and **E1 arms must never be tabulated alongside accepted method rows as if
they were methods.** Add this to the receipt.

**No build estimate is given here.** Two were given earlier, in opposite directions, both
from reading rather than building; both were withdrawn. Phase 0 of
`E1_IMPLEMENTATION_BRIEF.md` produces the estimate.

**The changes carry two verified silent-failure modes**, both of which run cleanly and
return plausible numbers: implementing the true reward by nulling the surrogate yields a
cost-only planner (`faithful_pomdp.py:218-220`), and snapping **raw** truth onto the
**latent** abundance grid mis-scales A2 by `survey_scale` (≈ 41 for fox). See
`E1_IMPLEMENTATION_BRIEF.md` §4.2–§4.3.

#### Cost, and the sharding trap

**No build estimate.** Two earlier figures (1–1.5 days, then 2–4 hours) were both
withdrawn — each was produced by reading the code rather than building against it. The
execution plan requires Phase 0 to produce the estimate after the implementer has read the
source. What is clear is that **validation, not writing, dominates**: proving the
true-state arms are genuinely true-state in the right units at every step, and that the
registered kernel matches the simulator. Full execution plan and the two agent commands:
`E1_IMPLEMENTATION_BRIEF.md`.

**Compute ≈ 25–40 core-h.** Against the 4 Aug capacity snapshot (§9) that is nominally
minutes — but only with enough shards. The measured relationship is

```
elapsed ≈ core-hours / min(500, runnable shard count)
```

E1 has **4 arms × 4 cells = 16 natural shards**. At ~2 core-h each that uses 16 of 500
slots and takes ~2 hours. **Shard by evaluation seed** — 4 × 4 × 20 = **320 shards**, under
the 500-job ceiling — and wall clock drops to roughly 10–15 minutes. Aggregate afterwards.
Subject to registration decision 2: sharding distributes evaluation only, never planning.

The S6 array is the cautionary precedent: those tasks ran ~10 h each on one core, so 36
shards took 10 h regardless of how many cores sat idle.

#### Architecture pinning

Pin to **one** architecture per experiment; never mix microarchitectures within a
comparison.

| Work | Pin | Why |
|---|---|---|
| E1, and E2 | `--constraint=EPYC9534` on `comp` | 609 idle CPUs at snapshot, homogeneous modern pool. None of these are accepted-comparison quantities |
| Anything compared to an **accepted** number (parity smoke test, the A4-vs-PLUS objective gap, S9) | `--constraint=xenon-8452Y` on `m3h` | The architecture that produced the accepted results. 72 idle CPUs at snapshot |
| E3 new-seed replication | one architecture, any | S6 used `ccemmp`; internal consistency is what matters |

#### Gates

| Outcome | Reading | Action |
|---|---|---|
Read every row using the preregistered decision rule above — primary corner effects, plus
the no-reversal requirement, against the 0.10 margin.

| Outcome | Reading | Action |
|---|---|---|
| **A2−A4 > A1−A2**, A2−A4 lower bound above margin, model gap consistent across rows | Deployment observability is the binding constraint | **Proceed to E2.** This is the result the contribution needs |
| **A1−A2 > A2−A4**, A1−A2 lower bound above margin, consistent across rows | Learning the dynamics from this log binds; filtering does not | **Redirect the contribution to the model axis.** Adding belief to a general method will not help |
| **Model gap flips sign between A1−A2 and A3−A4** | The model effect is not stable across state conditions | **No dominance claim.** Report both and treat the result as indeterminate |
| **Both below the 0.10 margin** | Nothing to win by handling either better here | **Do not build E2.** Fix the benchmark (S9 graded reward, §6(ii)) or write the methodological result |
| `A1 − V(best constant)` ≈ 0 in a cell | Inert cell, regardless of method | Retire it from the comparison |
| **PBVI ≈ QMDP** | Belief *planning* is not the lever even if A2−A4 is large | Aim E2 at filtering/model quality, not the planner |
| **A3 rises materially with planner budget** | The frozen horizon-5 budget was the binding limit | Re-baseline before claiming anything; the gap was partly U11 |
| **A1−A3 approaches the 0.10 margin** | Grid + solver error alone is the size of the effect being hunted | **Treat every other contrast as suspect.** Re-baseline at finer discretisation before claiming anything |

### 7.2 E2 — the prototype (the contribution itself)

**Only if E1 passes.** This is `S5_MODEL_CLASS_VS_PLANNER_SPEC.md`, reframed: those NEW
cells were specified as *diagnostics* to decide what the paper is about. Under the new
framing **they are the method being proposed**, and the grid is organised by **which
uncertainties are represented separately**, not by model class alone.

#### The design grid

| Arm | State uncertainty | Transition-model uncertainty | Planner | Status |
|---|---|---|---|---|
| PLUS | belief per candidate | 8 **mechanistic** Ricker MAPs | PBVI | accepted |
| MOOR | one belief | **none** (point commitment) | PBVI | accepted |
| BA-MCTS | **none** | categorical belief in tree | MCTS | accepted |
| **P1 — the proposal** | belief + log-normal observation model | posterior over **learned** latent transition models | PBVI | **new** |
| **P2 — ablation: state only** | belief + observation model | single learned model (point commitment) | PBVI | **new** |
| **P3 — ablation: model only** | plug-in point estimate | posterior over learned models | PBVI | **new** |
| **P4 — the competitor** | **fused** — observation-space residual, no explicit latent state | ensemble in observation space | PBVI over the observation state | **new — must be specified before build (below)** |

> **P4 is not yet an executable arm.** "Observation-space residual, no explicit latent
> state" plus "PBVI" does not say what the planner operates on. Before E2 is built, state
> explicitly: the **observation is treated as the Markov state**, so P4 is point-mass
> planning over the observation with an ensemble transition model fitted in observation
> space and no observation likelihood. If instead P4 keeps a latent variable, define it —
> and note that doing so weakens the contrast, because the contrast *is* the presence of an
> explicit latent state. Until this is written down, whether P1 and P4 truly share a
> planner is **not established**.

**P1 vs P4 is the claim; P2 and P3 are the ablations.**

- **P1 vs P4** → **N4**: explicit latent-state + model-posterior representation versus
  observation-space fusion. **This is an architecture-*bundle* comparison** — latent state,
  observation likelihood and model class co-vary together, because an exact joint belief is
  algebraically identical to `w_j b_j(x)` (§5.4) and so cannot serve as the comparator.
  **It must be reported as a bundle result, not as isolating factorisation.**
- **P1 vs P2, P1 vs P3** → complementarity: does representing both uncertainties beat
  dropping one? Useful, but **these do not test separation** and must not be labelled as if
  they do.
- **Preregister the P1-vs-P4 margin too**, not only the P1-vs-PLUS one. A bundle contrast
  without a decision threshold is not a test.
- **P1 vs PLUS** → **N4′**: does it survive dropping the mechanistic prior? **Requires
  rerunning PLUS under P1's planning objective** — accepted PLUS planned the learned
  surrogate, so an unmatched comparison changes objective and model source together.
  Preregister a **non-inferiority margin** for what "matches PLUS" means; grid values can
  converge while actions remain unstable (`in new repo/S2_ORACLE_REVIEW.md`, grid
  convergence — 7.9% fox action flips from 161→321 bins with value converged).
- **N5**: a flexible non-mechanistic transition family inside P1 versus a ridge-linear one.
- **N6**: decision-time seconds *and model evaluations* reported next to every return.
  **Match compute where arms differ in ensemble size, or state plainly that they differ** —
  reporting compute is not matching it.
- **Match risk attitude and OOD/support treatment across P1–P4**, or report them as part of
  the bundle. Taxonomy §1.3 items 4–5 name both as live rival explanations for the accepted
  ordering; leaving them free reintroduces the confounds E2 exists to remove.
- Uncertainty-blind control (plan in one randomly drawn model) → whether the posterior
  earns its keep at all, complementing the replay finding that PLUS's averaging nets ≈ 0
  on fox.

**Validation before returns are opened:** implement an explicit joint belief over
(candidate, abundance-bin) and check it reproduces the `w_j b_j` representation to
numerical tolerance under identical backups. §5.4's algebra says it must. If it does not,
an implementation or approximation effect exists and has to be named *before* E2's returns
are interpreted.

**On P3 and BA-MCTS:** P3 is *not* "the same architecture as BA-MCTS with the planner held
fixed" — P3 uses a plug-in state estimate inside PBVI, whereas BA-MCTS carries a
categorical model belief inside MCTS. P3 is a useful ablation in its own right; it is not a
controlled version of the fox-theta result, and the earlier claim that it was is withdrawn.

**Held fixed:** the PBVI budget identical across model classes (32 beliefs, horizon 5,
41 bins, 7 observation branches — E2 only; see §7.1 for E1's branch-count exception), the same 4k log and 80/20 split, the 20 registered
seeds, true-state reward, P=10, horizon 50, γ=0.95 (taxonomy §3.6); plus the planning
objective, risk functional, and support/OOD treatment across P1–P4.

**Also validate the new PBVI path** on the existing small-POMDP exact-solution probe
(target 1e-5), which PBVI already matched once
(`PLUS_MOOR_PAPER_ALIGNED_IMPLEMENTATION_EXPLANATION.md`, "Verification Completed").

**Scope limit on any E2 claim.** Whatever E2 finds holds *on this benchmark*: one species
with real headroom, a 0.0005 decision margin, an advantage that moves by up to −1.0 under
abundance-scale perturbation, and orderings that do not reseed (§6). A result here does
**not** license a general claim about partially observed offline RL. Either repair the
benchmark first (§6(ii)) or state the scope limit explicitly in every claim.

**Cost.** Build **~1 week** — this is the real bottleneck and it does not fit the 1–2 day
window. Compute ~20–40 core-h, wall ≈ ½–1 day.

### 7.3 E3 — the honest comparison

**PRECOMMITTED — not conditional on E2's outcome.** E3 runs on whatever E2 produces,
whichever direction it points. Making replication conditional on "a candidate result"
would precommit to replicating only favourable findings, and S6 already showed that
aggregate CIs can hide seedwise sign reversals (§3.6). **E2's first returns are exploratory
by definition; only E3-replicated numbers are reportable.** Declare this before E2's
returns are opened.

S6 protocol applied to the prototype: 3×3
independent collection × fit seeds, ordering stability across combos (not just paired
CIs — S6 showed a CI excluding zero can still flip sign per combo), and the
`Var_data / Var_fit / Var_eval` decomposition.

Precondition: land the OGSRL behaviour-budget fix from
`S6_THREE_CELL_RESULTS_AND_THETA_DIAGNOSIS.md` §"Proposed behavior-preserving fix", with
its three regression tests, before any new-seed run touches OGSRL.

**Cost.** ~230 core-h per 3×3 cell (priced in S6). **Wall ≈ 10–11 h regardless of core
count**, because each combo is a single ~10 h serial task (measured: array `58615270`
tasks ran 09:39–11:04 each). The 500-slot ceiling means several cells can run
*simultaneously* — so 3–4 cells cost the same wall clock as one. Launch them together.

### 7.4 Discipline that applies to all three

Non-negotiable, and it is what has kept this project honest:

1. **Seal predictions before results.** `NEXT_WORK_QUEUE.md` §1 is the template. E1's
   predictions must be written and dated before the job reports.
2. **Independent read-only review of any new research code before its numbers are
   trusted.** This is what caught the S2 bug. E2's new PBVI/latent-model path must get one.
3. **Every re-run carries a parity or validation anchor.**
4. **Never modify the accepted CSV or the frozen `src/tracks/**`.**
5. Pin one architecture per experiment; do not mix.

---

## 8. What is being dropped, and why

Being explicit about this is the point of the exercise — the previous plan was long
because it was written when this could still have been a family-uncertainty method paper.

| Dropped | Was | Why it goes |
|---|---|---|
| **Noisy-observation VPI (both parts)** | "the master question", 2–4 day build | **Dropped on time grounds, and the consequence is accepted rather than argued away.** It needs a belief over (state × family) — an augmented state space and a new solver, the same obstacle that forced E1 to Ricker-only (§7.1). **The consequence: no claim comparing family uncertainty to state or model uncertainty can be made.** That is a real limitation, not a caveat. The family result stands alone: full-state VPI ≈ 0, cross-family regret 0.180–0.594, both under exact observation, with the noisy-observation value **unmeasured and named as such** in every write-up. If it must be answered later, it can be added without invalidating E1 |
| **S7 (4k → 8k, compute parity)** | H2 / H3, 1–2 day build + 50–150 core-h | Tests "ecological structure helps under data scarcity" — the framing being abandoned. The compute asymmetry is more useful *as an asset* for N6 than as a hypothesis to test |
| **S6 5×5** | optional robustness, 1,000+ core-h | Tightens a fragility result already established well enough to act on. **Note: no longer expensive** — at 500 concurrent slots and ~10 h per combo, 100 combos is still ~10 h wall. It is dropped on *value*, not cost. Reconsider only if a reviewer demands a 5×5 grid |
| **S8 minimal sensitivity** | H2, ½ day + 10–20 core-h | Mostly cut. **The one knob that survives** is PBVI horizon 5 → 15, and it is absorbed into E1 as a correctness guard rather than run as its own study |
| **S6 theta re-run** | completes the 4th fox cell, ~90 core-h | **Re-promoted to "run it alongside E1."** The fix and its three regression tests are already written up; 8 combos × ~10 h run *concurrently* with E1 without competing for the 500-slot ceiling. It changes no conclusion but closes a record, and E3 needs the same OGSRL fix landed anyway — so this doubles as the fix's first live test |
| **S9 graded reward** | H10, 1 day + ~60 core-h/variant | **Conditionally re-promoted.** Not needed if E1 shows live headroom; becomes the *first* priority if E1 shows the cells are too inert (§6(ii)). Note its main blocker — no access to the accepted-result architecture — **appears to have lifted** (§3.7), pending the parity smoke test |
| **GPU port of PLUS** | — | Not worth it. PLUS is ~50–500× off its own flop bound, which indicates Python-loop overhead, not a need for FLOPS; the tensors (≈378 states) are far too small for a GPU to beat cached CPU BLAS; and float32 error (~1e-6 on values ~10) is the same order as fox's smallest decision margins (1.8e-5), so a port would risk flipping the argmaxes it is meant to accelerate. Build time, not compute, is the critical path anyway |

---

## 9. Cluster capacity — snapshot, 4 August 2026

Taken by the code-server agent; read-only, no jobs submitted or modified.

| | |
|---|---|
| **Concurrency ceiling** | **500 running jobs** (`MaxJobs`), 1,000 submitted (`MaxSubmitJobs`). `GrpTRES` and `MaxCPUs` unset — so for single-core array elements the effective limit is **500 cores** |
| **Idle capacity, `comp`** | ~2,250 idle CPUs. Best homogeneous pools: **AMD EPYC 9534 (609 idle)**, Intel Xeon Gold 6338 / Ice Lake (651), Cascade Lake (361), Genoa (252), Skylake (142) |
| **Xeon-8452Y (`m3h`)** | **72 idle CPUs, and the account is no longer excluded** — `AllowAccounts=ALL`, user holds QoS `m3h`, and no active `SPEC_NODES` reservation covers `m3h100-101`. See §3.7 — verify with a parity smoke test before relying on it |
| **Active reservations** | `cryosparc-general`, `sexton`, `CryoemFacility`, `august`. **None names account `ce25`**, none covers `m3h100-101` |
| **Queue** | 0 running, 0 pending jobs for this account. Fairshare 0.245 |
| **GPUs** | Access exists (`gpu` partition) but only ~2 × L40S realistically schedulable against **622 pending GPU tasks** partition-wide. `m3h` nodes carry H100s. Irrelevant given §8 — recorded for completeness |
| **Throughput** | 25 core-h → **5–15 min**; 250 core-h → **35–60 min** — *provided there are ≥500 runnable shards*. Otherwise `elapsed ≈ core-hours / min(500, shard count)` |
| **Recommended pin** | `--constraint=EPYC9534` on `comp` for new work; `--constraint=xenon-8452Y` on `m3h` where parity to accepted matters |

**What this changes about the plan.**

1. **Compute is effectively free; build time is the entire critical path.** E1's 25–40
   core-h is minutes of machine time against a day of writing and validating code.
   Every scheduling decision in §7 should be read in that light.
2. **The binding constraint is shard granularity, not cores.** See §7.1 — E1 must be
   sharded by seed (320 shards) or it will use 16 of 500 available slots.
3. **The accepted-architecture blocker appears to have lifted**, which re-opens S9 and any
   accepted-baseline arm — pending the parity smoke test in §3.7.
4. **S6 theta and E1 can run concurrently** without competing for the ceiling (§8).
5. The last 14 days of `sacct` **independently corroborate the S6 theta diagnosis**: in
   array `58615270`, tasks 27–31 and 33–35 FAILED while 32 COMPLETED — exactly the 8/9
   pattern and exactly the task that had its short episode assigned to holdout, as
   recorded in `S6_THREE_CELL_RESULTS_AND_THETA_DIAGNOSIS.md`.

---

## 9b. Review split — three agents, three jobs

The repo is public at `github.com/hangpt01/DeepRL_Population_Models` (HEAD `3291eef`,
4 Aug 2026), so source review no longer requires cluster access. That changes who does
what:

| Agent | Access | Job |
|---|---|---|
| **Cowork (this chat)** | folder + cloned repo | Task B (feasibility against source) — partly done, see §7.1; then write E1 |
| **Codex, work mode** | folder only | **Adversarial review only** (Task E). Its value is independent judgement, not extra reach — it has exactly the access this chat has. Asking it to re-check arithmetic wastes it |
| **Code-server chat** | cluster | Task C (parity smoke test), datasets, fit caches, launching jobs. Command issued only after the plan is approved |

The independent-review discipline is what caught the S2 bug, and it worked because the
reviewer was asked to *falsify* — it reproduced the archived numbers to 1e-15 and only
then broke them. A reviewer asked "is this right?" will mostly say yes. The command below
is written to prevent that.

### Review commands

Review prompts are **not stored in this plan** — they are issued directly and their
outputs land in this folder. Reports produced so far:

- `ADVERSARIAL_REVIEW_PLAN_05.md` — first adversarial pass on Plan 05 (verdict: PROCEED
  WITH CHANGES). Found E1's rows non-commensurable and E2 lacking a fused-both comparator.
- `REVIEW_OF_PROPOSED_REVISION.md` — second pass on `PROPOSED_REVISION_E1_2X2.md`
  (verdict: APPLY WITH CHANGES). Found that episode-persistent `C`/`θ`/regime cannot be
  marginalised into an abundance-only kernel, and recommended the Ricker-only design now
  in §7.1.

`PROPOSED_REVISION_E1_2X2.md` is **superseded**: it proposed an all-family 3×2, and what
was applied is the Ricker-only 2×2 the second review recommended instead. It is retained
as the record of that decision.

## 10. Auditor checklist

For an agent verifying this document against the folder:

- [ ] §3.3 headroom figures match `UNCERTAINTY_TAXONOMY_AND_STAGED_DIAGNOSTIC_PLAN.md`
      §0.3–§0.4.
- [ ] §3.4 `switch_vs_MAP` values (0.141 / 0.709 / 0.222 / 0.675) and the PLUS−MOOR fox
      deltas match `NEXT_WORK_QUEUE.md` §1-OUTCOME and `DECISION_LOG.md` D-002.
- [ ] §3.6 ordering-stability verdicts match
      `S6_THREE_CELL_RESULTS_AND_THETA_DIAGNOSIS.md` — in particular that PLUS−MOOR flips
      in **all three** complete cells.
- [ ] §4 retraction figures (11.839420699045172 vs …174; separation 0.018; VPI 0.0 at 321
      bins; regret Allee 0.358 / theta 0.180 / regime 0.594 **positive**) match
      `in new repo/S2_ORACLE_REVIEW.md` and `DECISION_LOG.md` D-009.
- [ ] §5.1 U1 row matches `UNCERTAINTY_TAXONOMY_AND_STAGED_DIAGNOSTIC_PLAN.md` §1.2 and
      `READONLY_EXTRACTION_PASS_2.md`.
- [ ] §7.1 is scoped to **Ricker environments only**, and the reason given (C and θ drawn
      per episode in `reset()` and persistent; regime switches under Π) matches
      `READONLY_EXTRACTION_PASS_2.md` item 10 and the family maps in taxonomy §0.1.
- [ ] §7.1 uses **one planner** (PBVI, point-mass belief for true-state arms) and does not
      reintroduce `backward_oracle()`.
- [ ] §7.1 labels the model-axis gap as **fitted-from-noisy-log**, not "parameter
      estimation", and says training-side state uncertainty is inside it.
- [ ] §7.1 states that **A4 plans the true reward and is therefore NOT the accepted PLUS
      number**.
- [ ] §7.1 includes PBVI-vs-QMDP (N2), paired CIs, a preregistered practical margin, and
      cell-wise reporting — none of these were dropped when the section was rewritten.
- [ ] **No claim anywhere ranks family uncertainty against state or model uncertainty.**
      The withdrawal is recorded at §2.1, and §7.1 "What E1 can and cannot establish",
      §8 and §10 all carry the limitation.
- [ ] §7.2 grid matches `S5_MODEL_CLASS_VS_PLANNER_SPEC.md` §1–§2, **plus P4**, and
      P1-vs-P4 is described as an **architecture-bundle** comparison, not a factorisation
      test.
- [ ] §7.3 states E3 is **precommitted**, not conditional on E2's outcome.
- [ ] No occurrence anywhere of "VPI 0.128" presented as a live result, or of "family
      uncertainty is real".
- [ ] Compute figures (15,314 s/cell PLUS; 2.5 s EVD; 6,126×; 26× per cell) match
      taxonomy §0.2.

**Known limitations of this document.**

1. **The E1 build estimate is withdrawn, not resolved.** Two figures were given and both
   retracted; Phase 0 of the brief produces the estimate. What *was* verified 4 Aug against
   `github.com/hangpt01/DeepRL_Population_Models` @ `3291eef`. The planner takes an
   injected `CandidatePOMDP`, so **the planner is untouched** — but *not* every change is
   model-level. The true-state arms are a **policy-level** intervention (`moor_faithful`
   filters the survey itself; `PublicTransition` excludes `evaluator_info`). Two
   silent-failure modes are verified and specified in `E1_IMPLEMENTATION_BRIEF.md` §4.2–4.3
   (cost-only surrogate fallback; raw-onto-latent unit error), and a **registered-kernel
   parity gate** is now a blocking precondition (§4.1). **No build estimate stands — Phase 0
   produces it.**
2. **Scope limits carried deliberately, not by oversight** — recorded here so no downstream
   document quietly drops them:
   - E1 runs on **Ricker environments only**; nothing it produces speaks to Allee, theta
     or regime environments, whose oracle arms need an augmented state space (§7.1).
   - **No family-versus-state comparison is made anywhere.** The family result is
     full-state only and the noisy-observation value is unmeasured (§8).
   - The "fitted-model gap" includes **training-side** state uncertainty, because the fit
     comes from the noisy 4k log. It is not parameter-estimation cost (§7.1).
   - E2's P1-vs-P4 is an **architecture-bundle** comparison, not a test isolating
     factorisation (§5.4, §7.2).
   - Any E2 result is **benchmark-limited** — one species, 0.0005 margin, orderings that
     do not reseed (§6). It does not license a general claim about partially observed
     offline RL.
3. **The PLUS loop-bound claim (§8, GPU row) is inferred** from the frozen configuration
   in `PLUS_MOOR_PAPER_ALIGNED_IMPLEMENTATION_EXPLANATION.md`, not from reading
   `plus_faithful.py` or `pbvi.py`. A one-hour profile would settle it and should precede
   any optimisation work.
4. **The Xeon-8452Y unblocking (§3.7) is a configuration-layer reading**, not a
   demonstrated capability. The parity smoke test settles it.
5. **The stale-record annotations** identified in `ADVERSARIAL_REVIEW_PLAN_05.md` §10 are
   accepted but **not yet applied** — `CHAT_HANDOFF_04` §2/§6/§7/§8, `NEXT_WORK_QUEUE.md`
   §2–§8, `MY_RESEARCH_QUESTIONS.md`, taxonomy §11 and
   `SUPERVISOR_MEETING_SUMMARY_30JUL.md` still read as current instructions. They are to
   be annotated as superseded, never rewritten.
