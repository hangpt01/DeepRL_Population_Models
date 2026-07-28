# Next work queue — what to prepare before the results land

Written 27 July 2026, while the A6-PLUS checkpoint and the 11-run fleet are
pending.

The organising principle: **do the work whose value decays when results
arrive.** Everything in §1 becomes worthless the moment you see the fleet
output. Everything in §2–§6 is merely useful, and can wait.

---

## 1. SEALED — predictions registered before seeing any fleet result

> **Timestamp: 27 July 2026, before job 58548480 (A6-PLUS) reported.**
> **Do not edit this section after results arrive.** Add a dated
> "outcome" column below it instead. A prediction recorded after the fact is
> worth nothing, which is the whole point of writing it now.

This project already runs return-blind acceptance. This is the same discipline
applied to *interpretation*: if the fox results confirm the PLUS-mechanism story,
it matters enormously whether that story was written down first.

### 1.1 A6-PLUS (Egyptian vulture × Ricker × σ 0.1)

| Metric | Prediction | If wrong |
|---|---|---|
| Parity, 7 fields | Pass at ≤ 1e-9 | Instrumentation bug — same-hardware MOOR baseline already passed 0.0, so architecture is excluded |
| M7 `frac(r_pos = 0)` | **exactly 1.000** | The infeasibility proof (§0.5 of the taxonomy) is wrong, or `r_pos` is mis-logged. Stop everything and re-derive |
| M3 candidate unanimity | **1.00** at every step | Candidates disagree despite growth parameters being inert — would mean disagreement enters through mortality, which is informative |
| M2 `switch_vs_MAP` | **0.000** | Would contradict the stored `pbvi_argmax_actions` (`5;5;5;5;5;5;5;5`) |
| M2 `switch_vs_uniform` | **0.000** | As above |
| M15 deployed = myopic oracle | **0%** overlap, matching A6-MOOR | — |
| M6 penalty component | **−184.611** exactly | Arithmetic error somewhere |

**M1 (posterior movement) — genuinely uncertain, and I am registering both
branches:**

- *Branch A (I lean here, ~65%):* the posterior **does** move and concentrates,
  because candidates differ in mortality and 50 log-normal observations at
  σ = 0.1 is a lot of evidence. Expect `max_j w^j > 0.5` by roughly step 20 and
  entropy falling well below `ln 8 = 2.079`.
- *Branch B (~35%):* the posterior stays near uniform because the candidates'
  predicted trajectories are too close to discriminate at this noise level.

**Either branch, combined with `M2 = 0`, is the same headline result:
inference succeeds and decisions are unaffected.** Branch A is the stronger
version — it shows the machinery works and still changes nothing.

### 1.2 Fox cells (A1 Ricker 0.2, A2 Allee 0.2, A3 regime 0.2, A4 theta 0.2)

| Metric | Prediction |
|---|---|
| M7 `frac(r_pos = 0)` | **< 1.0 in all four** — fox is the only species deploying a positive-`r` action (`a1`, `r = +0.2098`) |
| M2 `switch_vs_MAP` in **A2** | **> 0** — the stored `pbvi_argmax_actions` for fox/Allee/0.2 is `7;1;2;2;2;6;2;1`, so the MAP candidate (a7) differs from the deployed modal action (a2) |
| Direction of that switch | **Harmful.** MOOR (single fitted model) scores 11.611 against PLUS's 10.967 in this cell. Where posterior averaging moves the decision, I predict it moves it the wrong way |
| M5 `centred / raw` ratio | **≪ 1** — most cross-candidate disagreement is action-independent and cancels in the argmax |
| M4 margins in **A1** | Concentrated near zero; a large fraction of decisions below ε = 1e-3 (stored `pbvi_action_margin_min` = 0.001) |
| M3 unanimity | **< 1.0 in all four**, lowest in A2 (stored single-belief value 0.50) |

### 1.3 A5 (Amur tiger × theta-logistic × σ 0.1) — the sharpest test

| Metric | Prediction |
|---|---|
| M7 | **1.000** — deployed `a10` carries `r = −0.0458 < 0` |
| M2 both switch fractions | **0.000** |
| M3 unanimity | **< 1.0** — stored single-belief value is 0.62 |
| Return SD | **0.000** despite θ being redrawn every episode, because `r_pos = 0` makes θ unreachable |

**This is the cleanest available demonstration of the central claim:** candidates
that *disagree about the action* still produce a constant policy with zero
variance, because the deployed action deletes the term they disagree about.

### 1.4 Tier B (general methods)

| Metric | Prediction |
|---|---|
| M11, EVD on tiger | The recovery actions rank **low under `Q̄` itself**, not only after the pessimism penalty ⇒ the collapse is the conservative fitted Q (and 2.5 s of compute), **not** the disagreement mechanism |
| M12, OGSRL | Constraint **binds frequently on fox** (`s_low` 31.90 vs realised min N 21.297) and **rarely on tiger** (`s_low` 35.91 vs realised min N 79–109) |
| M13, surrogate error on trajectory | **Small on tiger and fox** — the visited states stay far from `s_safe`, so the surrogate's danger under-pricing never binds. This is the "safe by accident" prediction |
| M13, surrogate error on vulture | **Large and negative** — every visited state is in the under-predicted safe-row stratum |

### 1.5 What would genuinely surprise me

Recording these matters as much as the predictions:

1. `M7 ≠ 1.000` on A5 or A6 — would break the family-degeneracy argument.
2. `M2 > 0` on A5 — would mean posterior averaging *does* act on tiger and the
   constant policy has a different explanation.
3. A large `centred` disagreement in §1.2 — would mean candidates disagree in a
   decision-relevant way and PLUS's mechanism is real but mis-tuned.
4. Parity failure on A6-PLUS after the MOOR baseline passed 0.0 on the same
   hardware.

Any of these overturns a conclusion currently written into the taxonomy
document, and should be treated as a finding rather than a bug until proven
otherwise.

---

## 1-OUTCOME. Results, recorded after the fleet reported

> **Dated 27 July 2026, 22:09:55 AEST — after Tier A array `58569801` and Tier B
> array `58577745` completed.** The §1 predictions above are sealed and were NOT
> edited. This section is the paired outcome column. Source: the four files in
> `from AI agent in code server/` (final `DIAGNOSTIC_REPLAY_RECEIPT.json`,
> `SUMMARY.md`, `manifest.json`, `HANDOFF_CONTINUATION.md`).

**Integrity.** 24/24 parity receipts PASS at exactly 0.0 on all seven fields;
`recomputed_fits = 0`; accepted `MATCHED_P10_144_METHOD_CELLS.csv`
(`7431318803e468…`) verified unchanged; Tier B dataset hashes matched the
accepted rows (12/12); M14 schema separation held (belief columns only for
RefPlan). Tier B imported the separate general-method source, correct provenance.

**Scorecard against §1.1–§1.4:**

| Sealed prediction | Outcome |
|---|---|
| §1.1 A6-PLUS: parity, M7=1.000, M3=1.00, M2=0, M15 0%, M6 −184.611 | **All confirmed** (checkpoint, 27 Jul) |
| §1.1 M1 (Branch A, ~65%) posterior concentrates | **Confirmed, stronger** — TV 0.834, ‖Δw‖₁ 1.75, w>0.5 by step ~1 |
| §1.2 A2 (fox/Allee) `switch_vs_MAP` > 0, **harmful** | **Confirmed** — switch **0.709**; PLUS 10.967 < MOOR 11.611 |
| §1.2 A2 centred/raw ≪ 1 | Confirmed: raw 0.896, centred 0.105, ratio 0.117 |
| §1.2 A1 margin mass below 1e-3 | Confirmed: PLUS 5.3% (min 1.8e-5), MOOR 4.0% |
| §1.3 A5 return SD = 0.000 despite θ redraw | Confirmed: 9.1e-16 → 0.000 |
| §1.4 EVD tiger collapse = fitted Q + compute, not disagreement | Supported: λ_V=0 switch B2 = 0.004 (near-inert pessimism) |
| §1.4 **M12 OGSRL binds on fox, rarely on tiger** | **MISSED — inverted.** Formal constraint never binds anywhere (bind_fraction 0.0, slack + in B1–B3); the *guardian* carries safety and fires on **tiger** (override 0.995 / 0.929), not fox (0.0). Predicted s_low figures were also off (fox 15.15 vs predicted 31.90) |
| §1.4 M13 "safe by accident" on trajectory | **Not reported** in this batch (Tier-B trajectory M13 not included) |

**Load-bearing finding — open item 19 closed, then refined by the M1–M15 pipeline
(27 Jul, later run).** PLUS's posterior-averaging is **decision-active on all four
fox cells** (`switch_vs_MAP`: A1 0.141, A2 0.709, A3 0.222, A4 0.675), but its sign
versus MOOR's single-model commitment is **family-dependent**:

| Fox cell | family | PLUS | MOOR | PLUS − MOOR |
|---|---|---:|---:|---:|
| A1 | ricker (true) | 10.384 | 10.386 | −0.001 (tie) |
| A2 | allee | 10.967 | 11.611 | **−0.644 (worse)** |
| A3 | regime | 10.591 | 10.142 | **+0.449 (better)** |
| A4 | theta | 10.311 | 10.137 | **+0.174 (better)** |

Harmful only on Allee, helpful on regime and theta, neutral on the true Ricker
family; it roughly **cancels across fox** (Σ ≈ −0.02). The sealed §1.2 "averaging
moves the decision the wrong way" holds on A2 but *fails* on A3/A4. The honest
headline is therefore not "averaging is harmful" but "the extra PLUS machinery buys
no consistent, bankable advantage over a single misspecified model — sometimes
better, sometimes worse, netting ≈0." That still argues against the
family-uncertainty motivation, for a subtler reason. (My first write-up over-
generalized the A2 loss; corrected here.)

**Cleanest A5 confirmation:** tiger/theta candidates are *never* unanimous
(M3 = 0.000) and disagree enormously (centred 2.657), yet `switch_vs_MAP = 0.000`
and return SD = 0 — because `r_pos = 0` deletes the very term they disagree about.
Inference machinery fully engaged; decision untouched.

**To understand, not bury:** the M12/OGSRL miss was wrong on mechanism (guardian,
not the Lagrangian constraint, acts) and on species (tiger, not fox). Likely
explains why OGSRL "wins" the tiger cells (B3 headroom +0.49, the only general
method beating the best constant there). Not a §1.5 surprise; a Tier-B diagnostic.

**Pipeline run (27 Jul, later):** the tested `analysis/` pipeline was run on the
server (self-test 32/32; criterion-4 24/24 at 1e-9). It supplied the fox table
above and closed M3/A5-M2. **Still open:** Tier-B trajectory M13 is *not
recoverable* from these logs — `surrogate_reward_t` was never logged for
RefPlan/OGSRL/BA-MCTS, and EVD uses raw rewards by design. The "safe by accident"
test (H14) needs a re-instrumented general-method run.

---

## 2. Write the S2 specification — the decisive experiment for your core question

**Why this one first among the specs.** S2 answers the question the whole
project exists to answer: *do the four families induce different optimal
policies at all?* It requires **no learning and no fitted policies**, so it is
immune to the 0.0005-margin fragility that makes every fox result shaky. If
off-diagonal regret is ~0, the family-uncertainty motivation collapses
regardless of anything the replay shows.

Contents to specify:

- Per-family oracle policies `π*_F` (plan in the true family with true parameters).
- The **cross-family regret matrix** `R[F,G] = V(π*_G in G) − V(π*_F in G)`, and
  in particular the `R[Ricker, G]` row — the penalty PLUS and MOOR actually pay.
- **VPI** = `V(family-aware oracle) − V(best single family-agnostic policy)`,
  an upper bound on what any family-uncertainty method could achieve.
- Offline **identifiability ratios**: between-family prediction difference ÷
  process-noise scale on the visited support. Note `process_noise_sigma = 0`, so
  this needs re-framing — with deterministic dynamics, distinguishability is
  limited by observation noise alone, which likely makes families *easier* to
  tell apart than I assumed. **Worth thinking through before specifying.**
- Scope: fox cells (the only species with headroom), plus one tiger cell as a
  contrast.
- The four-part gate G1–G4 from taxonomy §6.6.

Effort: one working session. Cheap to run afterwards.

---

## 3. Build the analysis scaffolding now, against the known schema

The log schema is fully specified in `DIAGNOSTIC_REPLAY_RERUN_SPEC.md` §1.5, so
**the analysis code can be written before the data exists.** This is the most
concretely front-loadable item here.

What to build:

1. A loader for `logs/{cell}__{method}.npz` + `.csv.gz` (note: not parquet — no
   pyarrow in the venv).
2. Metric functions M1–M15, each with a unit test on synthetic input where the
   answer is known by construction.
3. The comparison tables: one row per (cell, method), columns M1–M15.
4. Four figures worth having ready:
   - **Headroom plot** — each method against its best constant action, per cell.
     This is the figure that carries the S1 story.
   - **Posterior entropy trajectories** with the action-switch fraction overlaid
     — the "inference succeeds, decisions unaffected" figure.
   - **Raw vs centred candidate disagreement** — the §2.2 mechanism figure.
   - **Return decomposition** into utility / cost / penalty, stacked, per method
     per species — the figure that shows the penalty is a constant offset.
5. A validation harness that reproduces the accepted `return_mean` from the
   per-step logs (this is acceptance criterion 4 — worth having independently).

Effort: one to two sessions. Highest practical payoff, because it converts the
fleet output into conclusions the same day it lands rather than a week later.

---

## 4. Specify S5 — the decisive test of the ecological-advantage claim

The single highest-value remaining experiment, and the one with the longest
implementation lead time, so specifying it early buys the most.

**The question:** is the ecological advantage the *Ricker equation*, or is it
*having a latent state with a correct observation model*? Currently PLUS/MOOR
differ from the general methods on three axes at once. S5 removes two of them.

**The design:** fit a flexible, non-mechanistic transition model inside a latent
state-space model with the same log-normal observation law, then plan with the
**same PBVI**. Compare against PLUS (8 Ricker candidates) and MOOR (1 Ricker
model) with the planner held fixed.

- If the flexible latent model ≈ PLUS → the advantage is latent-state modelling,
  not ecology. That reframes the paper, and generalises well beyond conservation.
- If PLUS ≫ flexible latent → the mechanistic prior is doing real work.

Also specify the paired direction: an *observation-space* variant of MOOR, to
confirm the axis from the other side.

Effort: one session to specify; substantial to implement.

---

## 5. Specify S6 — fit and collection-seed replication on fox

Promoted from optional to load-bearing. Fox is the only species that can support
any method claim, and its action-value margin is 0.0005. If its ordering doesn't
survive independent seeds, no method claim survives anywhere in the benchmark.

Specify: 3–5 independent collection seeds and independent fit seeds on the four
fox cells; the variance decomposition `Var_data`, `Var_fit`, `Var_eval`; and
paired confidence intervals for method differences.

One subtlety worth noting: with `process_noise_sigma = 0`, `Var_eval` comes only
from observation noise acting through the policy, plus the per-episode `C`, `θ`
and regime draws. So `Var_data` and `Var_fit` are the *only* meaningful variance
sources — which makes this study cleaner to interpret than usual.

Effort: one session to specify.

---

## 6. Lower priority, but real

- **Supervisor narrative skeleton.** Structure and figure list can be drafted
  now; the PLUS-mechanism section needs the fleet. Lead with the constant-action
  result — it changes what every other number means.
- **Decision log.** This project has strong provenance culture for artifacts but
  none for *decisions*. A short running log — what was decided, when, on what
  evidence, and what would reverse it — would have caught the P=5/P=10 mismatch
  earlier.
- **Related-work check.** Has anyone published the "constant-action baseline
  reveals the benchmark is non-discriminating" result for ecological RL? If not,
  that is a contribution in its own right. Worth 30 minutes of searching.

---

## 7. Do NOT do these yet

- Interpret or write up the PLUS mechanism — needs M1/M2/M3/M5.
- Finalise the supervisor summary — the part they will push hardest on is
  exactly the part still running.
- Any reward re-plan — run the minutes-long screening test in
  `REWARD_REDESIGN_MEMO.md` §4 first.
- Expand to more species — the three-species panel has not yet earned it.

---

## 8. Suggested order for this session

1. **§1 is already done** by virtue of being written above. Do not touch it again.
2. **§3 analysis scaffolding** — most practical payoff, converts the fleet output
   into conclusions immediately.
3. **§2 S2 spec** — most decisive for the core question, and the
   `process_noise = 0` wrinkle needs thinking through rather than typing.
4. **§4 S5 spec** if capacity remains.

If you only have appetite for one: **§3**, because the fleet lands in hours and
you will otherwise spend the next session writing loaders instead of thinking
about results.
