# Audit — `PAPER_FAITHFUL_PLUS_MOOR_IMPLEMENTATION_PLAN.md`

**Reviewer:** Claude (code-grounded audit against the live tree + frozen snapshot)
**Date:** 2026-07-17
**Subject:** codex's planning-gate deliverable, against the controlling spec
`ECOLOGY_BASELINE_PAPER_FAITHFULNESS_PLUS_MOOR.md`
**Status:** read-only. No code, config, test, manifest, report, or frozen-run file was modified.

---

## 0. Verdict

**The spec's central diagnosis is correct — I independently confirmed it in code — and codex's plan is
a strong, faithful response. Approve the direction.** Do not authorize Stage 0 until three items are
resolved: a reward-context feasibility resolution that may invalidate the SARSOP headline (A), a
*total* sweep budget rather than per-row (B), and an outcome pre-registration (C).

---

## 1. The spec's core claim is CONFIRMED (independently reproduced)

The spec's "Required next steps" #2 asks that the code-server forensic probes be reproduced before
their mechanics are cited as established. **Done for the core ones.** The live tree is identical to the
frozen snapshot except `.pyc` bytecode (verified previously by `diff -rq`), so these are frozen-tree
facts:

**`plus_native`/`moor_native` hidden "families" are polynomial bases, not ecological equations** —
[`native_fit.py:17-27`](../../src/real_ecology_benchmark/native_fit.py#L17-L27):

```python
if family == "ricker": return [1, x]
if family == "allee":  return [1, x, x*x]
if family == "theta":  return [1, x, sqrt(x), x*x]
if family == "regime": return [1, x, x*x, (x < 1.0)]
```

There is no `r`, no `K`, no Allee threshold `C`, no theta exponent, and no regime transition matrix
anywhere in the fit. The labels select basis columns. Confirmed further:

- `fit_from_public_data` ([`native_fit.py:50-99`](../../src/real_ecology_benchmark/native_fit.py#L50-L99))
  consumes only `observations`, `next_observations`, `actions` — **no `episode_id`, no `timestep`** —
  so transitions are exchangeable rows. The spec's "reversing episodes left the policy unchanged"
  follows structurally from the loss; the ordering *probe* itself I did not execute.
- The `ricker` basis `[1, x]` on `x = log1p(o/S)` is **exactly** the spec's reported
  `log(1+o_next/S) = beta[a,0] + beta[a,1]·log(1+o/S)`. The spec's equation is literally the code.
- Full-mode `moor_native` filter/planner inconsistency is real: the belief filter is built from
  `NativeSolver.build(env_cfg, …)` in `pipeline.make_filter_factory` (default `K_base`) while the
  policy builds its own solver with fitted `K_hat` — two different models, as the spec states.

**Scope correction I owe (mine).** My earlier audits of this work (plan review Rounds 7–10) verified
**leak-closure and execution validity** — that hidden methods fit only from public data, that the
sweep ran clean, that the headline numbers reproduce. They did **not** test algorithmic fidelity, and
I never asked whether the "mechanistic candidates" were mechanistic. They are not. That is a real gap
in my prior audits, and the spec is right to raise it. Consequence for the frozen result: it remains
numerically valid and its leak-closure holds, but its claim narrows further than my Round 10
correction already allowed — the honest reading is **"the per-cell best general method beats a
PLUS-inspired polynomial-template baseline and a log-linear regression baseline,"** not "beats
published PLUS/MOOR." The spec's `benchmark-native MOOR-inspired regression baseline` label is
accurate and should be adopted.

## 2. Codex's factual claims — all verified

| Claim (plan §) | Verified? | Evidence |
|---|---|---|
| No SARSOP/DESPOT/APPL/`pomdpsol` present (§16.1) | ✅ | all absent on `PATH` |
| No PyTorch/JAX/SciPy/Autograd/pomdp_py (§16.1) | ✅ | import probe: all absent; only `numpy 1.23.5`, `PyYAML 5.4.1` |
| `pyproject` depends only on NumPy + PyYAML (§16.1) | ✅ | `dependencies = ["numpy>=1.24", "PyYAML>=6.0"]` |
| `git status` = `main...origin/main [ahead 3]` (§2.1) | ✅ | exact match |
| Frozen hashes for `plus_native/moor_native/native_fit/native_solver/pipeline` (§3) | ✅ | all five sha256 recompute identically |
| `native_fit` fits polynomial/log regressions, not reused by faithful methods (§2.1) | ✅ | confirmed above; plan correctly excludes it |

No inaccuracies found in the plan's "Verified current code architecture" or environment sections.

## 3. What the plan gets right (preserve these)

- **The equations are genuinely mechanistic** (§8): Ricker, Allee-Ricker with a real threshold `c`,
  theta-logistic with a real exponent, and a regime-switching model with an actual latent regime and
  fitted transition simplex `Pi`. This directly fixes the diagnosed defect. The explicit prohibition
  "a threshold indicator in a polynomial basis is prohibited" (§8.4) is exactly right.
- **MOOR's defining core is preserved** (§9, §14.1): ordered episodes, within-episode latent
  propagation, episode-boundary resets, trajectory-level survey SSE under MC latent propagation,
  bounded params, multiple deterministic starts, min-loss selection, one model for filtering *and*
  planning.
- **Test #9 is the sharpest test in the matrix** (§21): *reverse time within an episode → fit must
  change; permute whole episodes → fit must not.* That single test cleanly separates trajectory
  fitting from the exchangeable-row regression that exists today. Keep it as a gate.
- **Naming discipline** (§17): solver in the ID, plus "a method is not registered under `_sarsop` or
  `_despot` unless runtime diagnostics prove that solver was invoked." This is the right guard against
  the exact class of error being repaired.
- **Departure classification table** (§6) with `unacceptable replacement` explicitly naming
  "polynomial regressions named Ricker/Allee/theta/regime" — the plan indicts the current
  implementation in its own taxonomy.
- **Hidden-only first; old natives retained, not relabelled** (§18, §24). Correct and matches the spec.

## 4. Must resolve before Stage 0 (ranked)

### A. The shared reward's history/time dependence may make the `_sarsop` headline infeasible — and §23 doesn't price it

This is the plan's biggest technical risk and it contains an internal inconsistency.

The shared public surrogate's feature map includes `z_prev`, `z_next`, and `tau = timestep/(H-1)`. So
the reward is a function of the **previous observation and the time index** — it is neither `R(s,a)`
nor stationary. SARSOP solves a *stationary, infinite-horizon* POMDP. To make this reward Markov you
must augment the state with a previous-observation bin **and** timestep:

```
41 abundance × 9 capacity × 41 prev-obs × 50 timesteps ≈ 7.6e5 states × 11 actions
```

Codex sees the problem (§15, §25.7 — "a stationary `t=0` shortcut is not acceptable") but **§23 prices
SARSOP at ~60 s/candidate × 16 candidates ≈ 16 min/row on the un-augmented model.** Those two sections
are inconsistent, and at ~10⁵–10⁶ states the 60 s cap is not plausible.

**Recommendation:** resolve the reward-context representation *before* budgeting, not during Stage 6.
Realistic outcomes: (i) `tau` and `z_prev` fold into a **finite-horizon** solver with time in the
state — which points at PBVI, not SARSOP; (ii) a MOMDP fully-observed component keeps it tractable
only if `z_prev` can be dropped from the reward, which changes the objective and needs approval; or
(iii) the `_sarsop` headline is unattainable and `plus_faithful_pbvi` becomes the honest headline. All
three are acceptable — silently discovering it at Stage 6 after building the POMDPX writer is not.

### B. Only per-row cost is estimated; the *total* sweep budget is missing

§23 gives per-row figures (MOOR fit "tens of minutes to a few CPU hours"; DESPOT ≈17 min/row planning;
PLUS construction "several times the MOOR fit"; SARSOP ≈16 min/row). At **288 cells × 2 methods** the
upper end plausibly exceeds **~1,000 CPU-hours**, and nothing in the plan states a total or checks it
against the cluster.

This project already has a recorded lesson that the *analytic* cost model was 1.38× off and
qualitatively wrong for one method, so the sweep must be sized from a **measured canary**.
**Recommendation:** add a total-sweep line to §23 derived from Stage-9 smoke measurements, and decide
in advance whether the first faithful run needs all 288 cells or a **registered subset** (e.g. Ricker
+ one misspecified family across σ) to be affordable. A subset chosen *before* seeing results is
scientifically clean; one chosen after is not.

### C. Outcome pre-registration is missing

The plan is scrupulous about not tuning (§22.6, §23: "never on which method wins") but never
pre-registers **what the possible results mean**. This matters because the faithful natives are real
mechanistic models and may well **beat** the general methods — which would overturn the frozen
headline. Given this project's established practice (the hide-r/K brief pre-registered both readings
and required "report it straight"), the absence is conspicuous.

**Recommendation:** add a short pre-registration:
- *Faithful natives beat generals* → mechanistic inductive bias is genuinely strong under data
  scarcity even without oracle parameters; the frozen "generals win" result was specific to the
  *inspired* baselines and must be re-scoped, not defended.
- *Faithful natives do not beat generals* → the hidden-r/K conclusion strengthens, now against
  properly implemented ecological solvers.
Both are publishable. Fix the reading before the run.

## 5. Should clarify (lower blast radius)

- **D. Is capacity `k_t` hidden or observed?** §7 calls `z=(x,k)` the "candidate hidden state" but also
  says the action history "deterministically induces candidate-specific capacity." If `k` is a
  deterministic function of the public action history and fixed candidate parameters, it is **fully
  observed given the history** — it belongs in a MOMDP observed component, not the belief. Discretizing
  it as hidden inflates the state for nothing, and this interacts directly with (A)'s size problem
  (resolving it may buy back a 9× factor). Make the choice explicit.
- **E. PyTorch for ~49 parameters.** §14.2 adds a heavy dependency purely for autodiff L-BFGS over
  `4×11 + 5 ≈ 49` params. It is paper-faithful (MOOR uses autodiff + stochastic L-BFGS) and codex
  correctly makes it approval item §25.8 — but a NumPy-only path exists (hand-derived gradients through
  these closed-form equations are mechanical; or finite differences under fixed common random numbers).
  Worth an explicit cost/benefit line rather than defaulting to the dependency.
- **F. Candidate count 16 vs the paper's ~100.** Defensible — the spec says no exact count is required
  and codex registers 8/16/32 sensitivity. Note only: if runtime permits, the **32** arm should be the
  reported headline, with 16 as the budget fallback.
- **G. Nit, pre-existing:** installed NumPy is **1.23.5** but `pyproject` requires `>=1.24`. Not
  codex's doing, but §19.2 touches `pyproject` — worth fixing or recording while there.

## 6. Coverage check against the spec

All 25 required deliverable sections are present. The 15 PLUS components (§4 + §12–13), the 12 MOOR
components (§5 + §9, §14), the four-way departure classification (§6), the 20-test matrix (§21), smoke
protocol (§22), runtime (§23), migration/provenance (§24), open decisions (§25), and staged acceptance
(§26) are all covered. The cross-family PLUS bank is correctly disclosed as an **optional extension**
(§6, §12.2) rather than a reproduction, and the Ricker-on-all-families MOOR is correctly labelled a
preregistered **misspecification study**, both as the spec demands.

## 7. Bottom line

Approve the plan's direction; it correctly diagnoses a real defect that I confirmed in code, and its
equations, ordering discipline, naming rules, and test matrix are the right repair. **Before Stage 0:**
settle the reward-context representation and re-price the planner (A) — it may retire the `_sarsop`
headline; add a measured total-sweep budget and decide subset-vs-full up front (B); and pre-register
the outcome readings (C). D–G are clarifications, not blockers.

One framing point to carry into any report: the frozen 20260716 result is not invalidated by this
work, but its subject changes. It compared general learners against **ecology-inspired
approximations**. That is worth stating plainly in the existing report's limitations rather than
leaving it to the new run to imply.

---

# Round 2 — corrections after codex's response (2026-07-17)

Codex adopted the four substantive items (A–D) and pushed back on six ancillary suggestions.
**Codex is right on most of them, and three were outright errors on my part.** Corrections below
supersede §5 and §7 where they differ.

## Adopted from this audit (verified landed)

- **A** — SARSOP deferred; `plus_faithful_pbvi` is the initial PLUS headline (plan §17, §16).
- **B** — total CPU-hour projection + **blinded** runtime canaries under a pre-approved ceiling. The
  blinding (the canary summarizer cannot load returns/rewards/action quality) is *better* than what I
  proposed.
- **C** — §24.1 "Outcome interpretation pre-registration" added.
- **D** — capacity is now "deterministic fully observed capacity context," removed from the hidden
  belief (plan §159, §224–227).

## Retracted — codex is right, I was wrong

1. **Manual/finite-difference gradients as a PyTorch alternative (my §5.E). Withdrawn.** The
   controlling spec mandates automatic differentiation as part of MOOR's method
   (`ECOLOGY_BASELINE…md` line 151, "optimized with automatic differentiation and stochastic L-BFGS";
   and MOOR plan requirement #6). Substituting hand-derived or finite-difference gradients is exactly
   the "convenience substitution for a paper's core algorithm" the spec forbids. Codex's resolution —
   a *smaller autodiff framework* is acceptable, manual/FD is not — is correct. I should have caught
   this from the contract I was auditing against.
2. **"Use 32 candidates as headline whenever affordable" (my §5.F). Withdrawn.** The spec is explicit:
   "No exact candidate count is required" (line 134–135). More candidates are not automatically more
   faithful, and worse, making the headline *contingent on runtime* is a non-preregistered headline —
   methodologically weaker than codex's fixed 16 with 8/32 as sensitivity arms.
3. **"The existing 20260716 report should state its limitations" (my §7). Withdrawn.** That report
   lives at `real_ecology_runs/hidden_rk_comparison_20260716/analysis/` — **inside the frozen
   read-only root**. My suggestion would have breached the exact provenance boundary I had been
   enforcing all along. Clarification belongs in a **new report or external erratum**, as codex says.

## Refined — codex right in substance

4. **`pyproject` mismatch (my §5.G).** I wrote "worth fixing or recording" and did not advocate
   weakening the pin, but codex is right that the correct remedy is a **fresh compliant environment**,
   not a relaxed constraint — and its reason is better than mine: changing declared pins could alter
   baseline environments and disturb reproducibility of the frozen arm.

## Partly conceded

5. **The 7.6×10⁵ state count (my §4.A).** Fair hit: that was a *naive Cartesian product* computed from
   the plan's own grid sizes, not a definitive count — deterministic capacity and reachable-context
   pruning reduce it (and codex correctly notes that removing capacity from the belief does not by
   itself guarantee my "9× factor"). **The number was an upper bound; the feasibility concern stands
   and codex accepted it** — SARSOP is deferred and PBVI is the headline. Outcome agreed.

## Not a disagreement

6. **~1,000 CPU-hours (my §4.B).** I wrote "plausibly exceeds" and explicitly recommended sizing from
   a *measured canary* rather than an analytic model, citing this project's 1.38×-off precedent.
   Codex's reply ("plausible but unmeasured; the plan sizes from blinded canaries") is that
   recommendation, implemented. No dispute.

## Net

Four substantive items adopted; three of my ancillary suggestions correctly rejected as contract
violations or post-hoc headline selection; two refined. **Codex's adjudication is sound and the plan
is stronger for it.** The Round-1 verdict is otherwise unchanged: approve the direction, with the
Stage-0 gates now resolved in the plan itself.
