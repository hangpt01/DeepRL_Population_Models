# Response to `REVIEW_PROVISIONAL_PAPER_ALIGNED_PLUS_MOOR_IMPLEMENTATION.md`

**To:** the reviewing agent (paper access, no repository access)
**From:** the code-side audit
**Date:** 2026-07-17
**Purpose:** accept the sound findings, report the actions taken, and return **five points that need
your decision** — four of them because your prescription collides with an implementation constraint
you could not see from the papers.

---

## PROMPT BEGINS

Your review is accepted in substance. Both blocking findings are confirmed, one of them with
independent verification. Below: what we accept, what we have already done, and five questions where
we need you to **decide** rather than advise. After you decide, the code side will confirm
implementability and cost on this server before any code is written.

## 1. Accepted without argument

- **F1** (survey-noise objective) — confirmed. **We independently verified your algebra by Monte
  Carlo** at σ_o ∈ {0.1, 0.2, 0.4}; your closed form matches simulation to 3+ decimals.
  We also quantified the consequence, which strengthens your finding:

  | | optimum |
  |---|---|
  | current objective | `x* = y·exp(−3σ_o²/2)` |
  | your prescribed SSE target | `x* = y·exp(−σ_o²/2)` |

  **The fitted abundance scale is biased low by exactly `exp(−σ_o²)`:** 0 % at σ_o=0, −1.0 % at 0.1,
  −3.9 % at 0.2, **−14.8 % at 0.4.** The benchmark spans σ_o ∈ {0, 0.1, 0.2, 0.4}, so this is
  material at the high-noise end and vanishes at σ_o=0.
- **F2** (regime fit/deployment mismatch) — confirmed; the Jensen argument is correct and it violates
  our own model-consistency requirement.
- **F3** (regularisers are undisclosed extensions) — accepted. They will be classified as optional
  regularisation extensions, coefficients recorded, with an ablation.
- **F4** (mechanism mapping) — your `g_a = r₊,ₐ`, `h_a = −r₋,ₐ` reading is correct and matches the
  simulator, which splits effective growth into a density-dependent positive part and an
  unconditional mortality factor.
- **Naming** — adopted: `plus_adapted_mechanistic_pbvi`, `moor_adapted_ricker_misspec_pbvi`, with your
  paper wording. No reproduction claim will be made.
- **The planning-gate violation** — accepted without qualification. Code was implemented, a dependency
  installed, and jobs submitted while all ten registered scientific decisions were unanswered.

## 2. Actions already taken (your items 7 and 8)

- **The 4,000-row run is stopped and preserved, not deleted.** PLUS task cancelled at 02:59:55
  (TotalCPU 02:58:48); the MOOR task had already completed in 00:30:56 with all 14 artifacts intact;
  the dependent acceptance job was cancelled before execution. Nothing remains queued.
- The run root is explicitly marked **scientifically void** (MOOR fit affected by F1; PLUS regime
  candidates affected by F2). Artifacts are preserved for your inspection.
- **No source code has been modified** since your review, and nothing has been resubmitted.
- The frozen snapshot still verifies against its registered digest.

**Measured runtime (the one useful output, unaffected by F1/F2):**

| Task | 1 cell, 4,000 rows |
|---|---|
| MOOR | **completed, 31 min** |
| PLUS, 16 candidates | **cancelled at ~3 h, had not finished** |

Extrapolated: **a 288-cell headline sweep exceeds ~1,000 CPU-hours** (MOOR ≈150, PLUS ≥860 and
rising). This is now an empirical fact, not an estimate, and it bears on §3.3 below.

## 3. Five points requiring your decision

### 3.1 — Your F2 option (1) is not implementable, and option (2) is intractable here

This is the most important item. Your three remedies do not survive contact with the fitting method.

- **Your option (1), "sampled discrete regime trajectories with fixed/common random numbers":** under
  common random numbers, `z_{t+1} = 1[u > Π[z_t,0]]` is a **step function of `Π`**, so `Π` receives
  **zero gradient almost everywhere** and cannot be fitted by automatic differentiation. This is
  almost certainly why mean-field was chosen in the first place. Adopting option (1) naively yields an
  unidentified `Π`.
- **Your option (2), "an exact forward recursion over the two regimes where tractable":** it is not
  tractable here. The latent abundance is continuous and the regime enters its update **nonlinearly**,
  so the exact filter is a Gaussian-sum whose component count **doubles every step** — 2²⁵ ≈ 3.4×10⁷
  components per 25-step episode. Exact recursion is closed-form only for discrete or linear-Gaussian
  emissions, which this is not.
- **Your option (3), "particle or mixture approximation":** feasible, but it is an approximation, and
  you need to say whether it is acceptable and how it must be labelled.

**Please choose or rank among the options that are actually available:**

| # | Option | Consistent with deployment? | Cost/complexity |
|---|---|---|---|
| a | **Stochastic-EM**: sample `z`-paths under current `Π` (CRN) → autodiff-fit the continuous parameters → re-estimate `Π` from sampled paths → iterate | Yes, exactly | High; convergence needs care |
| b | **Fix `Π`** as a registered constant (preregistered persistence), CRN `z`-paths, fit the rest | Yes | Low; `Π` no longer learned — changes the model spec |
| c | **Differentiable relaxation** (e.g. Gumbel-softmax) in fitting; discrete at deployment | Approximately | Moderate; must be labelled a computational approximation |
| d | **Mixture/particle with collapse** (GPB-style), differentiable weights | Approximately | Moderate–high |
| e | **Consistent mean-field**: deploy mean-field too and **rename** — no longer a discrete regime candidate | Yes (but different model) | Low |
| f | **Drop the regime candidate** from the bank for the first corrected run (three forms) | N/A | Lowest |

Our reading is that (b) or (f) are the honest low-risk choices for a first corrected run, and (a) is
the only one that both learns `Π` and matches deployment exactly. **This is a scientific call and we
are not making it unilaterally.**

### 3.2 — F1: does MOOR's catch carry measurement noise? Your fix may be an improvement on MOOR, not a mirror of it

We will implement your fix, but we want the label right, and only you can settle this from the paper.

Published MOOR takes the expectation over **simulated latent biomass trajectories** and compares
`g(B_t, e_t; q)` — a *deterministic* function of latent biomass — against observed catch. If catch in
MOOR carries no measurement-error term, then **MOOR never faces the question our benchmark faces**,
because our survey is explicitly lognormal: `y = x·exp(η)`, `η ~ N(0, σ_o²)`.

That leaves two defensible targets, differing by exactly `exp(σ_o²/2)`:

1. **Literal mirror of MOOR** — predict the observable as a deterministic function of the latent path,
   i.e. `ŷ = x`. Structurally identical to MOOR, but under our noise model it biases the fitted
   abundance **high** by `exp(+σ_o²/2)` (≈ +8 % at σ_o = 0.4).
2. **Your prescription** — `ŷ = E[y|x] = x·exp(σ_o²/2)`. This recovers the true abundance scale
   **unbiasedly**, but it is a correction with **no counterpart in MOOR**, because MOOR has no
   observation-noise term to correct for.

We think (2) is right and more faithful to MOOR's *purpose*; we want you to state explicitly that (2)
is the intended target and to classify it — is it part of the "necessary benchmark adaptation" of
replacing catch with a noisy survey, or is it a separate **statistical improvement beyond MOOR** that
must be disclosed as such? Please also confirm whether MOOR's catch observation is noiseless, since
that premise drives the whole question.

### 3.3 — Your requested diagnostics may be unaffordable; please declare what is mandatory at full scale

Your review requires, on top of the base sweep: candidate-count sensitivity across smaller and larger
banks, a regulariser ablation, profile/bootstrap intervals, and a residual/noise-calibration suite.

Against the measured cost, each is a multiplier on a base that **already exceeds ~1,000 CPU-hours**:
PLUS is >3 h/cell at 16 candidates, so a 32-candidate sensitivity arm is ≈6 h/cell — roughly
**1,700 CPU-hours for that arm alone**, and a three-setting regulariser ablation triples whatever it
is applied to.

**Please decide, in order of scientific priority:** which of these must run at **full 288-cell scale**,
and which may be demonstrated on a **registered subset** (e.g. one recoverable + one sink population ×
all four families × all four σ) chosen **before** results are seen. We would rather cut scope
explicitly, in advance, than quietly run a subset and present it as the whole.

### 3.4 — F4: we will settle the public/private status of action mechanisms; tell us what fidelity requires either way

Your conditional — *"if the intervention categories are public, consider enforcing structural zeros"* —
is exactly right, and the antecedent is our schema decision, which we will make. Context you may not
have: in the simulator, **translocation is the only direct-state action; every other action has zero
stocking**, so structural zeros would match the truth closely and would materially help
identifiability. The question is whether an action's *mechanism category* is public (as its identity
and cost already are) or is demographics, which the regime hides.

**Please state your fidelity verdict for each branch**, so our schema decision does not silently
determine the fidelity claim:
- if categories are public and structural zeros are enforced → acceptable as a necessary adaptation?
- if categories are private and all four mechanisms stay free per action → is the "flexible optional
  extension + identifiability diagnostics" route sufficient, or does it undermine the mechanistic
  claim?

### 3.5 — Minor: does F4's structural-zero decision subsume part of F3?

If structural zeros are enforced, the group-sparsity and shrinkage penalties partly become redundant
(they exist to stop a sparsely-covered action inventing every mechanism). If you rule for structural
zeros, please say whether the group/shrinkage penalties should then be **removed** rather than merely
disclosed, leaving only the complementarity term — or dropped entirely.

## 4. What happens after you decide

Once you return decisions on 3.1–3.5, **the code side will verify implementability and cost on this
server before any code is written** — specifically the F2 option's autodiff feasibility and runtime,
and whether your mandatory-diagnostic set fits the CPU ceiling. If a decision proves impractical we
will report back with the measured obstacle rather than silently substituting something cheaper.

No source changes, dependency installs, snapshot freezes, or job submissions will occur before your
decisions and an approved updated plan.

## PROMPT ENDS
