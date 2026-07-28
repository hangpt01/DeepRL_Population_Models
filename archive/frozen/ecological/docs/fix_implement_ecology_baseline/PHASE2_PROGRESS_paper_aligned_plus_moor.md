# Phase 2 Progress — Paper-Aligned PLUS / MOOR (controlled verification pass)

**Date:** 2026-07-18
**Status:** Steps 1 and 3 complete; **halted at the Step 2 gate** — the controlling reference is not
in the repository. No code edited, no job launched, no return value read.

---

## BLOCKER (must resolve to continue): controlling reference is missing

`PLUS_MOOR_PAPER_ALIGNMENT_COMPLETE_REFERENCE.md` was named as *"now the controlling scientific
reference"* with the instruction to *"place/read it in the repository"* and *"Do not assume the other
local documents are identical."* **It is not present anywhere in the repo, and no readable content was
attached.** I therefore cannot perform Step 2 (re-run the matrices *against it*) or Step 5 (*apply only
corrections supported by the controlling reference*) without either fabricating its content or
substituting the other docs — both of which the instruction forbids.

**Action needed from you:** add the file to `docs/fix_implement_ecology_baseline/` (or paste its
content). Everything below is the reference-independent work I could complete honestly in the meantime.

---

## Step 1 — Blindness & provenance (COMPLETE)

- **Returns not read.** I inspected only code, config, manifests, and job metadata — never
  `operational_return` / `true_return` / any `summary.json` body.
- **Provenance:**
  - commit `5f9cf32a69d47e2d31b2ed6ee30ebbcfe84b8536` (`main`); **corrected code is uncommitted** in the
    working tree (+1,876 / −180 across 42 files).
  - code digest over `src/ configs/ scripts/` (no pyc): `6df2126bb07da8a1c955f64c1a93d7d194074f29b8b5fa3c021b54514a6298e2`.
  - artifacts: void run `paper_faithful_one_cell_4000_20260717_v1/` (`VOID.md`, excluded); corrected
    canary `adapted_plus_moor_corrected_canary_20260717_v1/` (complete receipts, **no acceptance.json**).
  - jobs: **0 active.**
- **Acceptance-criteria pre-existence — determined:** the acceptance checker
  `scripts/run_paper_faithful_acceptance.py` (mtime **2026-07-17 15:02**) predates the canary submission
  (**2026-07-17 23:32**) by ~8 h. It contains **only structural validity gates** — `privacy.status ==
  "passed"`, `finite_objectives` (objective not None and `< inf`), `0 ≤ overshoot_rows <
  episode_length` — and **no performance thresholds; it does not read returns.**
- **Classification:** because the only acceptance criteria are structural, results-blind, and genuinely
  pre-existing, the existing canary is **confirmatory for validity** (finite fits / privacy / budget)
  but **exploratory for any performance claim** (no performance criteria exist — appropriate at canary
  stage). Running the checker to emit `acceptance.json` would **not** create post-hoc thresholds and
  would **not** break blindness. It is therefore available as a blindness-safe step once ordering is
  unblocked (Step 6, first branch). I have **not** run it, to respect the required order.

## Step 3 — `r_a` transformation reconciliation (RESOLVED: documentation + equivalence test)

**Approved plan text** (`CORRECTED_PLUS_MOOR_IMPLEMENTATION_PLAN.md` §3):
`r_a = r_max·tanh(q_a)` (symmetric), then `g_a = max(r_a,0)`, `h_a = max(-r_a,0)`.

**Exact implementation** (`faithful_fit.py`):
- `:352` `signed_free = 0.25 + 1.75 * torch.tanh(raw[...])` → `r_a ∈ (−1.5, 2.0)`, centred at `+0.25`.
- decode: `growth = _symmetric_positive(signed_rates)`, `mortality = _symmetric_positive(-signed_rates)`.
- `:304-306` `_symmetric_positive` ≡ `torch.clamp(min=0.0)` ≡ `max(·, 0)`.

**Is the g/h derivation equivalent to `g=max(r,0), h=max(-r,0)`? YES — verbatim.** For any given `r_a`,
`(g_a, h_a)` is exactly the approved split. Numerical examples:

| `r_a` | approved (g,h) | implemented (g,h) |
|---:|---|---|
| −0.8 | (0, 0.8) | (0, 0.8) |
| 0 | (0, 0) | (0, 0) |
| +1.5 | (1.5, 0) | (1.5, 0) |

**The only deviation is the reparameterization of the free variable `r_a`**, on two axes:
1. **Feasible range** — implemented `[−1.5, 2.0]` (asymmetric, more growth headroom) vs approved
   symmetric `[−r_max, r_max]`.
2. **Prior/center** — `q=0 → r=+0.25` (mild positive-growth start) vs approved `q=0 → r=0`.

**Impact:**
- *parameter meaning* — unchanged (`r_a` = signed effective per-step rate; `g`/`h` its positive/negative parts);
- *transition dynamics* — identical for a given `r_a` (the exponent uses `g,h` the same way);
- *fitting* — differs only in reachable range and start; the simulator's true rates
  (`r ∈ [−0.4753, 0.0707]` for Amur tiger; all populations' caps fall well inside `[−1.5, 2.0]`) are
  **not truncated**, so no truth is excluded;
- *structural zeros* — unaffected (channel-driven, orthogonal to the rate transform);
- *known-r,K mode* — **N/A**: both faithful methods hard-require `expose_rk='hidden'`
  (`plus_faithful.py:28-29`, `moor_faithful.py:28-29`); no known-r,K mode exists for them;
- *interpretation* — unchanged.

**Resolution (Step 3 branch: "mathematically equivalent → correct documentation + add equivalence
test").** The scientifically meaningful transformation (the max-split) **is** the approved formula, so
this is not a scientific defect. Pending your approval to write code (blocked with the rest of Step 5
by the missing reference), I will: (a) correct the plan text to state the implemented
`r_a = 0.25 + 1.75·tanh(q)` and the asymmetric `[−1.5, 2.0]` bound; (b) add a test asserting
`g=max(r,0), h=max(-r,0)` across sampled `r`. **Flagging for your awareness:** the asymmetric range +
`+0.25` center is a benign but real deviation from the approved *symmetric* text; if you want strict
adherence, the one-line revert to `signed_free = r_max·tanh(q)` is trivial. This is a documentation /
minor-bound choice, **not** a design-decision-request-level blocker.

## Steps 2, 4, 5, 6 — NOT STARTED (correctly, per ordering)

- **Step 2 (reconcile against the reference):** blocked — reference absent.
- **Step 4 (PBVI-vs-exact-POMDP test):** not started. It is reference-independent and unambiguously
  needed, but the required order places it after Step 2, and the reference could (per your own warning)
  alter planner requirements. Ready to build immediately once unblocked.
- **Step 5 (apply corrections + freeze snapshot):** blocked — "apply only corrections supported by the
  controlling reference"; also cannot freeze a clean snapshot while the reference reconciliation is
  pending.
- **Step 6 (limited verification canary):** the blindness-safe first branch (run the pre-existing
  structural acceptance checker on the existing canary) is available; deferred to respect ordering.

---

## Return items

- **A. Matrices vs the reference** — cannot produce; reference absent. (Phase 1 matrices vs the
  in-folder docs stand: both methods "Implemented but not sufficiently verified"; both blockers fixed;
  old `plus_native`/`moor_native` unreachable from the new IDs — re-confirmed by import inspection.)
- **B. `r_a` reconciliation** — done above: g/h split is verbatim-equivalent; only a benign
  reparameterization/bound differs → documentation fix + equivalence test.
- **C. PBVI exact-test** — not built (ordering; reference gate).
- **D. Test commands/results** — none run this pass beyond the read-only inspection; the last full run
  was 154 passing (Phase 1).
- **E. Frozen snapshot/commit** — none; corrected code remains uncommitted pending reference-authorized
  completion.
- **F. Canary provenance/acceptance** — canary valid-looking (8/8 exit 0, complete receipts) but
  **unaccepted**; structural acceptance is pre-existing and blindness-safe to run when unblocked.
- **G. Remaining limitations** — missing controlling reference (blocking Step 2/5); PBVI numerics
  unvalidated; real-data identifiability unverified; corrected code uncommitted/unfrozen; asymmetric
  `r_a` bound documented-deviation.
- **H. Recommendation:**

### `NOT READY: verification remains`

Specifically blocked on: (1) **the controlling reference** — provide it so Step 2/5 can proceed
honestly; (2) the **PBVI-vs-exact test** (Step 4); (3) the **`r_a` doc-fix + equivalence test**
(Step 3, ready); (4) a **frozen snapshot** (Step 5); then (5) the blindness-safe **acceptance run** on
the existing canary or a fresh blinded canary (Step 6).

**Stopped per protocol.** No 32-cell subset or full sweep launched; no returns inspected.
