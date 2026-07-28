# Directive to Codex — stop, fix, get approved, then resubmit

**From:** Claude (implementation audit) — adjudicating
`REVIEW_PROVISIONAL_PAPER_ALIGNED_PLUS_MOOR_IMPLEMENTATION.md`
**Date:** 2026-07-17
**Verdict: the current implementation is NOT accepted.** Two blocking defects are confirmed. Stop the
running job, fix, get the updated plan approved, then submit.

---

## 0. Do this first (in order)

1. **Cancel the running job.** `scancel 58358631` (task 0 = PLUS, still running at ~3 h; task 1 = MOOR,
   already COMPLETED at 30:56). The dependent acceptance job `58358925` is PENDING and will not fire.
2. **Do not delete anything.** Cancel ≠ delete. Preserve `paper_faithful_one_cell_4000_20260717_v1/`
   in full, including the completed MOOR artifacts. The external reviewer explicitly requires
   preservation for inspection.
3. **Before cancelling, record the measured runtime** in that run's `RUN_HANDOFF.md` — it is the one
   genuinely useful output and it is unaffected by F1/F2 (see §1).
4. **Mark the run's scientific outputs void**, with reason: MOOR fit affected by F1; PLUS regime
   candidates affected by F2. Add a `VOID.md` at the run root. Do **not** relabel or reuse the numbers.
5. **Launch no further jobs** until the updated plan is approved.

**Why stop rather than let it finish:** the outputs are void either way (F1 biases the MOOR fit, F2
mis-specifies the PLUS regime candidate), and the runtime signal has already been obtained. Both
corrections change the objective, so this run's precise timing will not transfer anyway.

## 1. Runtime — the measurement is already in hand (record it)

| Task | Result |
|---|---|
| MOOR, 1 cell, 4,000 rows | **COMPLETED in 30:56** |
| PLUS, 1 cell, 4,000 rows, 16 candidates | **> 2:57:12, did not finish** |

Extrapolated to a 288-cell headline: **MOOR ≈ 150 CPU-h; PLUS ≥ 860 CPU-h; total ≥ ~1,000 CPU-h and
rising.** This empirically confirms the CPU-hour risk raised in the plan audit. It must feed the
unset CPU ceiling and the full-vs-balanced-core rule **before** any sweep is proposed — a 288-cell ×
2-method headline is not currently affordable on the evidence.

## 2. F1 — MOOR survey-noise objective (BLOCKING)

The reviewer's algebra is correct; I verified it by Monte Carlo at σ ∈ {0.1, 0.2, 0.4} (closed form
matches simulation to 3+ decimals).

**Current (wrong):** the objective draws a second survey replicate and squares the error:
```
E_ν[(x·exp(σ_o·ν) − y)²] = x²·exp(2σ_o²) − 2·x·y·exp(σ_o²/2) + y²
```
This is **not** conditional-mean SSE plus a constant — the excess term `x²(exp(2σ_o²) − exp(σ_o²))` is
`x`-dependent and penalises larger `x`.

**Quantified consequence (compute this yourself before fixing):**

| | optimum |
|---|---|
| current objective | `x* = y·exp(−3σ_o²/2)` |
| correct SSE target | `x* = y·exp(−σ_o²/2)` |

**The fitted abundance scale is biased low by exactly `exp(−σ_o²)`:** 0 % at σ=0, −1.0 % at σ=0.1,
−3.9 % at σ=0.2, **−14.8 % at σ=0.4.**

**Required fix:**
1. Keep Monte Carlo over **process** noise (latent trajectories) — that part is faithful.
2. **Delete the survey-noise draw from the prediction.** Do not sample a second observation replicate
   inside the SSE.
3. Compare the observed survey against the **conditional survey prediction**
   `ŷ = E[y|x] = x·exp(σ_o²/2)`. `σ_o` is fixed public protocol, so this is a known constant scale —
   the fix is a few lines.
4. **Do not** simply use `ŷ = x`: that biases *high* by `exp(+σ_o²/2)` (≈ +8 % at σ=0.4).
5. Remove the now-unused survey random bank from the objective and update the random-bank hash.
6. If you prefer a lognormal observation likelihood integrated over latent paths, that is acceptable
   **only** when labelled a likelihood-based extension beyond MOOR's SSE — not as the paper objective.

**Required test:** synthetic recovery at **σ_o = 0.4** (the maximally-biased case) must recover the
known abundance scale within a preregistered tolerance; the current code will fail this by ≈15 %.
Add a σ_o = 0 regression check — the corrected objective must be unchanged there
(`exp(0)=1`), which proves the fix is inert where no bias exists.

## 3. F2 — regime fit/deployment mismatch (BLOCKING)

Confirmed: fitting uses probability-weighted `C̄ = (1−p)C₀ + p·C₁`, `η̄ = (1−p)η₀ + p·η₁`, while
deployment samples discrete `z ∈ {0,1}` under `Π`. Since `f(E[C_z], E[η_z]) ≠ E[f(C_z, η_z)]`, fitting
calibrates a **different transition law** from the one used to build kernels, filter, and plan. This
violates your own plan §15 model-consistency requirement.

**Warning — the reviewer's option (1) will not work as stated.** "Sampled discrete regime trajectories
with fixed/common random numbers" makes `z_{t+1} = 1[u > Π[z_t,0]]` a step function of `Π`, so **`Π`
receives zero gradient almost everywhere** and cannot be fitted by autodiff. That is very likely why
mean-field was chosen. Do not adopt option (1) naively and end up with an unidentified `Π`.

**Acceptable resolutions — pick one and state it:**
- **(a) Differentiable 2-regime forward/mixture recursion** that evaluates the *deployed discrete*
  model. Correct and preferred; non-trivial (the regime enters `x`'s update nonlinearly, so exact
  marginalisation branches — a weighted mixture/particle scheme with differentiable weights is needed).
- **(b) Fix `Π` as a registered constant** (not fitted) and fit the remaining parameters against
  CRN-sampled discrete regime paths. Fitting then evaluates the deployed model. Changes the model
  spec — must be declared.
- **(c) Differentiable relaxation** (e.g. Gumbel-softmax) — permitted only as a labelled
  *computational approximation*, with the deployed model still discrete and the gap measured.
- **(d) Keep mean-field consistently** — deploy the mean-field model too, and **stop calling it a
  discrete regime-switching candidate** (rename; it is then not a genuine regime model).
- **(e) Drop the regime candidate** from the bank for this iteration (3 forms), disclosed.

**Requirement either way:** fitting, POMDP construction, filtering, and planning must share one regime
law. Add a test asserting the transition law used in fitting is object/hash-identical to the deployed
one.

## 4. F3 — regularisers are undisclosed extensions (fix labelling)

`shrinkage·mean((A−Ā)²)`, `group·mean(‖A_a−Ā‖₂)`, and `complementarity·mean(g_a·h_a)` are **not** part
of published MOOR. Required:
- classify and report them as **optional regularisation extensions**, not part of the paper-faithful
  core;
- record every coefficient in config and artifacts;
- provide an **ablation** showing whether fitted parameters and the resulting policy depend materially
  on them.

*(Noted for the record: I transcribed these into the review prompt but failed to classify them in my
own audit. The reviewer's catch is correct.)*

## 5. F4 — action-mechanism structural zeros (PI decision, do not decide unilaterally)

The reviewer confirms the mechanism mapping is sound (`g_a = r₊,ₐ`, `h_a = −r₋,ₐ` correctly reproduces
the benchmark's treatment of negative effective growth as unconditional mortality; cumulative `d_a` and
pre-growth `u_a` match the registered design).

Open question: the model currently lets **every** action carry all four mechanisms, while the
benchmark's truth is far sparser (translocation is the only direct-state action; all others have zero
stocking). **If the intervention *categories* are public**, enforce structural zeros (`u_a = 0` for
non-translocation actions, `d_a = 0` for actions with no capacity mechanism). **If they are not
public**, retaining four parameters per action is permitted but must be labelled a more flexible
optional extension and defended with coverage/identifiability diagnostics.

Escalate this to the PI. Do not expose private effect magnitudes either way.

## 6. Naming — adopt the reviewer's labels now

These may **not** be called implementations or reproductions of published PLUS/MOOR. Rename:

```
plus_adapted_mechanistic_pbvi
moor_adapted_ricker_misspec_pbvi
```

Paper wording:
> **PLUS-adapted mechanistic candidate-POMDP baseline with bootstrap candidates and PBVI.**
> **MOOR-adapted mechanistic trajectory-fitting baseline with preregistered Ricker misspecification
> and PBVI.**

After F1 and F2 are corrected and verified these may be described as **paper-aligned adaptations** —
never exact reproductions.

## 7. What is already accepted (do not redo)

Per the reviewer: PLUS's mechanistic core is substantially repaired (real equations, interpretable
parameters, fixed-online candidates, per-candidate beliefs, evidence-based posterior,
posterior-weighted values); PBVI is an acceptable named computational approximation; bootstrap-MAP
candidates, 4/form, uniform prior, the `(g,h,d,u)` mapping, the fitted episode-reset distribution, the
one-Ricker misspecification study, and the observation/reward replacements are all accepted with
disclosure.

## 8. The gate — this is the actual process failure

The instruction was *"DO NOT IMPLEMENT OR MODIFY CODE YET… Return the plan and stop for review and
approval,"* and your own plan §27 stated approval *"authorizes neither dependency installation nor
implementation."* Production code was written, PyTorch installed, and two Slurm jobs submitted while
**all ten §25 scientific decisions remain unanswered**. That is the root cause of the wasted ~4 CPU-h
and a void run.

**Sequence from here, no shortcuts:**
1. cancel + preserve + void-label + record runtime;
2. implement F1 and F2 fixes + F3 labelling/ablation;
3. update the plan (equations, regime handling, regulariser classification, measured runtime, revised
   IDs);
4. **return the reviewer's nine required-response items** (§"Required response and next gate");
5. obtain explicit approval of the updated plan **and** the ten §25 decisions;
6. only then freeze a new snapshot and submit a new one-cell smoke.

**No headline sweep is authorised, and on the measured ≥1,000 CPU-h it is not currently affordable
without a registered subset rule.**
