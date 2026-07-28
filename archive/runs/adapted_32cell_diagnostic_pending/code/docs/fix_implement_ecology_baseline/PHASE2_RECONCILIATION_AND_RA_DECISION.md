# Phase 2 (cont.) — Reference reconciliation + `r_a` decision

**Date:** 2026-07-18
**Reference now available:** `PLUS_MOOR_PAPER_ALIGNMENT_COMPLETE_REFERENCE.md` (read in full).
**Status:** Step 2 (reconcile) done; **halted at the `r_a` decision** — but not for the reason the
Phase-2 instruction assumed. **No code edited, no job launched, no return read.**

---

## ⚠️ Correction I owe first (this reframes the whole `r_a` question)

The Phase-2 instruction says to change the code to *"the approved symmetric `r_max·tanh(q)`."*
**There is no approved symmetric parameterization.** The approved corrected plan actually specifies
the **asymmetric** form:

> `CORRECTED_PLUS_MOOR_IMPLEMENTATION_PLAN.md` §3, lines 147-148:
> *"Use preregistered signed bounds `r_min=-1.5` and `r_max=2.0` per transition. The smooth raw
> transform is `r_a = 0.25 + 1.75*tanh(q_a)`, followed by the exact max split."*

The implementation matches that verbatim (`faithful_fit.py:352`). **My Phase-1 report was wrong** when
it called this a "silent deviation from approved symmetric text" — I misread the plan. The
"symmetric was approved" premise the Phase-2 instruction is built on traces back to that error of
mine. I'm flagging it before any code change so the decision is made on the real record, not my
mistake.

**Against the now-controlling reference**, `r_a`'s parameterization is **undetermined**: reference §5
(line 99) requires only `g_a=max(r_a,0)`, `h_a=max(-r_a,0)` — which the code implements exactly — and
says nothing about the transform or bounds. So:

- vs the **reference**: `r_a` **PASSES** (produces the required max-split); the transform is a free choice.
- vs the **approved plan**: the code **matches exactly** (asymmetric `0.25+1.75·tanh`, `[-1.5, 2.0]`).
- "symmetric approved": **does not exist** in any controlling document.

Therefore this is **not** a code-vs-spec discrepancy to be corrected. It is a live question of whether
to **change the approved asymmetric design to a symmetric one** — a legitimate scientific preference
you raised, but a *design change requiring re-approval*, not a fix. **The `r_max` value / symmetric
form is genuinely undetermined by the controlling reference, so per the instruction's own rule I stop
on this point and return the choices below.**

## A. Reconciliation of the matrices against the controlling reference

Re-ran both matrices against `PLUS_MOOR_PAPER_ALIGNMENT_COMPLETE_REFERENCE.md`. **No verdict changed
from Phase 1** — the reference's requirements (§5 equations, §9 PLUS construction/online, §10 MOOR
fitting, §14 gates) match the implemented design. Confirmations:

- **Reference §5 equations** (Ricker/Allee/theta/regime, `g/h` max-split, `k`/`m` updates, lognormal
  survey) ↔ `faithful_ecology.py` — **PASS**, exact.
- **Reference §9.2 PLUS** (episode order; preregistered mechanistic bank; structural zeros; bootstrap-MAP
  candidates; fixed-Π grid fit under CRN discrete regimes; freeze; per-candidate kernels; independent
  PBVI) ↔ `plus_faithful.py`/`faithful_fit.py`/`faithful_pomdp.py` — **PASS** (candidate count 16 = 4/family, uniform prior, Π∈{0.80,0.90,0.97}).
- **Reference §10.2 MOOR** (one Ricker to ordered episodes; per-episode reset; conditional mean
  `x·exp(σ_o²/2)`; process-only MC; bounded autodiff L-BFGS multi-start; min-train selection; holdout
  diagnostic-only; no primary regularizers) ↔ `faithful_fit.py` — **PASS**.
- **Reference §9.2/§10.3 consistency** (same regime law / same fitted model in fit, kernel, filter,
  planner) — **PASS** (`regime_law_hash`; single `FittedModel`).
- **Reference §14 "PBVI belief-state, not QMDP"** — **PARTIAL**: `planners/pbvi.py` does belief-state
  backups (`action_values` backs up `max_a[ immediate + γ·Σ_o w·V(child belief) ]` over reachable
  child *beliefs* — not a QMDP per-state average), but is **not yet validated against an exact POMDP**
  (Step 3 test still to build).
- **Both blockers remain fixed** (re-verified in source): MOOR conditional-mean/process-only
  (`faithful_fit.py:449,458`); discrete fixed-Π regime, no mean-field (`:446,461-462`, `Pi` a constant).
- **Old paths unreachable:** `plus_faithful.py`/`moor_faithful.py`/`faithful_*` import no
  `native_fit`/`native_solver`/`resolve_actions`/`realdata`; registry maps only the adapted IDs.

New requirements the reference adds that the code does **not** yet satisfy: none that fail — only the
**PBVI-vs-exact validation** (already an open verification item) and the **known-r,K reporting**
(clarification only; see end).

## B. `r_a` resolution — DECISION REQUIRED (I stop here)

`g_a=max(r_a,0)`, `h_a=max(-r_a,0)` is correct and reference-compliant. The open question is only the
`r_a` transform/range. The env's set-point rate caps are small (≈`[-0.48, +0.07]`), but the **fitted**
`r_a` is a free effective per-step log-rate not bounded by those, so any sensible range must cover
plausible Ricker growth exponents (~1–2), not just the env caps.

| Option | Transform / range | `q=0` maps to | Change needed | Fresh canary? |
|---|---|---|---|---|
| **A — keep approved asymmetric** | `0.25 + 1.75·tanh(q)`, `[-1.5, 2.0]` | `r = +0.25` (mild growth) | **none** (matches plan+code) | no |
| **B1 — symmetric, `r_max=1.5`** | `1.5·tanh(q)`, `[-1.5, 1.5]` | `r = 0` (neutral) | code + config + init | **yes** |
| **B2 — symmetric, `r_max=2.0`** | `2.0·tanh(q)`, `[-2.0, 2.0]` | `r = 0` (neutral) | code + config + init | **yes** |

**Consequences on the axes you named:**
- *Feasible space:* A allows more growth (+2.0) than mortality (−1.5); B1 symmetric-tight; B2 symmetric-wide (more mortality headroom).
- *`q=0` value / implicit prior:* A biases the raw origin to mild positive growth; B centres it at neutral. (Note: the fitter uses **explicit deterministic multi-starts** at set rate values, not `q=0`, so `q=0` is the origin/prior, not the initialization.)
- *Optimization geometry:* symmetric tanh has a symmetric gradient profile about 0; the asymmetric shift skews curvature slightly toward the wider (growth) side.
- *Boundary behaviour / selected model:* different feasible bounds can select different optima only when the fit pushes a rate toward a bound; with the env truth well inside all three, this bites only for fast-growth/fast-collapse cells.
- *Structural zeros:* unaffected in all options (channel-driven, orthogonal).
- *Known-r,K mode:* N/A now (hidden-only); see clarification below.

**My recommendation (offered, not taken):** **Option A** is lowest-risk and fully reference-compliant
— the code already matches the approved plan, no re-canary, and the max-split (the only thing the
reference constrains) is exact. Choose **B** only if you specifically want the neutral-`q=0` symmetric
prior; it is scientifically defensible but is a **change to the approved design** (needs re-approval)
and **invalidates the existing canary as final validation** (per your own rule — a fresh blinded
canary from the new snapshot would be required).

**I will not change `r_a` or freeze anything until you pick A / B1 / B2** (or specify another `r_max`).

## C–F. Deferred until the `r_a` decision

- **C. PBVI-vs-exact test** — designed but not built this pass. The real `PointBasedPlanner.action_values`
  (`pbvi.py:77`) will be exercised on a tiny hand-specified `CandidatePOMDP` with a known exact
  finite-horizon belief value, including an information-gathering case where QMDP diverges. It is
  `r_a`-independent, so I can build it in parallel — but I held it to keep this pass focused on the
  decision that gates everything downstream.
- **D. Focused + full tests** — several (`g/h` split, gradients near `q=0`, range endpoints, structural
  zeros) must be written against the **final** `r_a` form, so they wait on the decision.
- **E. Freeze snapshot** — blocked: cannot freeze a "final" implementation with `r_a` unresolved.
- **F. Fresh canary / acceptance** — blocked on E; and if B is chosen, the existing canary cannot be
  the final validation.

## Known-r,K clarification (report-only, do NOT implement)

Fitted parameters and their demographic correspondence: **`k_0`/`k_max`** (normalized reset/ceiling
capacity) ↔ population **K** (on the public survey scale `S`, so `K ≈ k·S`); the per-action **`r_a`**
(→ `g_a` growth / `h_a` mortality) ↔ the population/action **growth rate `r`**; `d_a` capacity
increments, `u_a` stocking. The same fit code path could support the planned ablation by *source of
these values*: (i) **hidden r,K** — fit `k_0,k_max,r_a` from public data (current); (ii) **known r,K
only** — inject the true `K` (→ fix `k_0,k_max` from the private table) and true growth (→ fix `r_a`)
while all other private info stays hidden and everything else is still fitted; (iii) **known-model
oracle** — a separately-labelled method that takes the full true mechanistic model. All three are one
switch on *which parameters are fixed-from-truth vs fitted*, gated behind a new method ID. **Not
implemented; reported per instruction.**

## H. Recommendation

### `NOT READY: a scientific decision is required` (the `r_a` parameterization)

The implementation matches the approved plan and passes the controlling reference on every checked
gate; the only blocker is the `r_a` transform decision (A / B1 / B2), which is genuinely yours because
the reference is agnostic and the "symmetric approved" premise was my Phase-1 error. Once you decide, I
will in one pass: implement (if B) → add the `r_a` tests → build the PBVI-vs-exact test → run focused +
full suites → freeze a clean snapshot → run a fresh blinded canary + the pre-existing acceptance
checker → report C–F and the readiness call.

**Stopped per protocol. No 32-cell subset or full sweep launched; no returns inspected.**
