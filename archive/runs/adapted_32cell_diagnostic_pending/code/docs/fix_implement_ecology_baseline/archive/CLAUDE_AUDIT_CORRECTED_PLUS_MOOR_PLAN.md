# Audit — `CORRECTED_PLUS_MOOR_IMPLEMENTATION_PLAN.md`

**Reviewer:** Claude (independent; verified against the live tree, the void snapshot, and the scheduler)
**Date:** 2026-07-17
**Status:** read-only. Nothing modified.

---

## 0. Verdict

**Approve.** All five controlling decisions are implemented faithfully, all four server-side findings
are incorporated correctly, the plan-only gate holds, and every arithmetic claim I could check is
exact. One material gap (an un-costed sensitivity suite) and three minor items below — none blocking.

---

## 1. Mechanical gate — verified

| Claim | Verified |
|---|---|
| No jobs exist | ✅ `squeue` empty |
| Void snapshot unchanged | ✅ verifier returns `matched: true`, `2aae03ba…` |
| No code/config/script touched | ✅ nothing under `src/ configs/ scripts/` modified in the last 6 h |
| Void run preserved, excluded | ✅ intact; plan forbids loading its fit cache |

**Not verifiable:** the `9b42a10…` digest over `src/ + configs/ + scripts/` is published **without a
recipe**, and the existing `hash_paper_faithful_snapshot.py` takes a single root, so I cannot reproduce
a three-directory digest. This is the same class as the earlier F3 finding, already fixed once for
snapshots. **Minor:** either extend the verifier to accept multiple roots, or publish the recipe. I
confirmed plan-only by independent means (mtimes, scheduler, snapshot hash), so the claim stands —
just not by the digest it cites.

## 2. Decisions — faithfully implemented

- **D1 (regime):** symmetric `Π(p) = [[p,1−p],[1−p,p]]`, `p ∈ {0.80, 0.90, 0.97}`, preregistered, never
  optimized, serialized with hash. One canonical regime law shared by fitter/model/POMDP/filter/planner;
  **mean-field path removed**. Expected switches `25(1−p)` = 5.00 / 2.50 / 0.75 — **arithmetic correct**,
  and the 24-opportunity alternative (4.80 / 2.40 / 0.72) is stated to pre-empt a convention dispute.
  Good.
- **D2 (objective):** `L_data = (1/ΣₑTₑM) Σ [X^(m)·exp(σ_o²/2) − Y]²`, MC over reset+process only, **no
  survey draw, no survey random bank**, deployment likelihood unchanged. Exactly Decision 2.
- **D3 (scope/cache):** transition-only fit key, reward-specific planning, 32-cell subset, separate fit
  (64-row) and plan (128-row) manifests. Correct.
- **D4 (channels):** `action_channels` in `MethodContext`, channel-only loader, structural zeros exactly
  matching `actions.csv`.
- **D5 (signed rate):** three regularizers **absent from the objective** (not merely zeroed); validation
  fails on a nonzero legacy coefficient; fallback is a separately-named variant with preregistered
  trigger criteria and no post-hoc substitution.

## 3. My four server findings — all incorporated

1. **Cache key trap** — the plan states outright: *"Do not key fitted models on `dataset_sha256`,
   because it includes reward and has a 0% safe/yield cache hit rate,"* and defines
   `fit_transition_hash_v1` over exactly the seven transition fields. It also carries my second nuance:
   **"PBVI planning is not reused across reward modes,"** with the fit/plan cost split instrumented and
   tested (one fit invocation, two planner invocations).
2. **Channel-only exposure** — only `channel` crosses the boundary; `K_multiplier`, `dN_fraction`,
   `lambda_source`, `name_mechanistic`, `name_original`, `interpretation` are explicitly forbidden and
   guarded by a recursive privacy test. The population-independence of the channel map is stated.
3. **`r_a = 0` kink** — handled concretely: bounded `r_a = r_max·tanh(q_a)`, exact max split, **symmetric
   Clarke subgradient (`dg/dr = ½`, `dh/dr = −½`)**, starts on both sides of zero, per-start Strong-Wolfe
   failures recorded, all-start failure is a cell failure (not licence to switch objectives), plus a
   mandatory near-zero synthetic test. Better than what I asked for.
4. **Π grid cost** — the grid does **not** multiply the bank: 16 total retained, regime allocated
   `(1,2,1)` across the three persistence values, with the tradeoff stated honestly ("less within-`Π`
   bootstrap resolution at the outer persistence values").

## 4. Arithmetic — independently checked, all exact

| Claim | Check |
|---|---|
| 44 → 14 action parameters | ✅ rate 1 shared + 7 (a1–a4, a7–a9) = 8; capacity a5–a9 = 5; stocking a10 = 1 → **14** |
| 68.2 % reduction, ≈3.1-fold | ✅ 1 − 14/44 = 0.6818; 44/14 = 3.14 |
| Expected switches 25(1−p) | ✅ 5.00 / 2.50 / 0.75; 24-convention 4.80 / 2.40 / 0.72 |
| `exp(−0.4²) = 0.8521` → ≈14.8 % | ✅ e^−0.16 = 0.85214; 1 − 0.85214 = 14.79 % |
| Bank 16 = 4×4, regime (1,2,1) = 4 | ✅ |
| Sensitivity 12 = 3+3+3+3; 32 = 8+8+8+8 | ✅ |
| "five PLUS configurations, not nine" | ✅ primary + 2 bank + 2 path deviations |
| 32×2 = 64 fit rows; 32×2×2 = 128 plan rows | ✅ |
| `C_cache` saves exactly `F_m` per dynamics cell | ✅ formulas are consistent |

## 5. Trap I suspected — does not exist

I expected the candidate **diversity gate** to break for regime candidates differing only by `Π`. It
does not: `_model_vector` already concatenates `regime_matrix.reshape(-1)`, so candidates at different
persistence values are separated by construction, and the two same-`Π` (`p=0.90`) candidates are the
ones the gate genuinely tests. No action needed.

## 6. Findings

### F1 — Material — the subset sensitivity suite is not costed

§9 registers **five** PLUS configurations on the 32-cell subset (primary 16/16; banks 12 and 32; regime
paths 8 and 32). §10 projects only the **primary** subset cost ("the same formula with 32 replacing
144"). Nothing projects the suite.

This matters: the "larger" arm at **32 candidates is ≈2× the primary per-cell cost**, and PLUS already
measured **>3 h/cell at 16 candidates**. A rough sum across the five arms is on the order of
**several hundred CPU-hours** — plausibly comparable to the primary subset itself. And §10's budget
gate (`1.5·C_cache ≤ B_full`) governs only the **full sweep**; **no gate covers the subset suite.**

**Required before implementation:** project the sensitivity-suite cost with the same `F/K/P/E`
decomposition, and put it under an explicit ceiling. Note the suite is fit-dominated (bank size changes
fitting, not planning), so cache reuse helps it more than it helps the primary — worth quantifying
rather than assuming.

### F2 — Minor — `r_max` is unstated

§3 specifies `r_a = r_max·tanh(q_a)` but never gives `r_max`. It is a preregistered bound that defines
the feasible set, so it must carry a value **before** any result is seen. State it in the plan and
config.

### F3 — Minor/positive — kernel build conservatively treated as reward-specific

`2·K_m` appears in **both** projections. Transition and observation kernels are almost certainly
reward-independent (only the reward table varies by mode), so this leaves real saving on the table. The
plan is explicit that the canary will resolve which components are reward-independent — that is the
right call, and conservative projections are the correct error direction. Flagging only so the canary
actually measures it.

### F4 — Minor — profile intervals absent

Decision 3 makes them "subset only and optional **unless inferential parameter claims are made**." The
plan omits them, which is fine — but it should **state explicitly that no inferential parameter claims
will be made**, since that premise is what makes the omission legitimate.

## 7. Practices worth keeping

- Both switch-count conventions stated up front, pre-empting a reporting dispute.
- Regularizers **absent from the objective**, not zeroed — an unfakeable form of the claim.
- Fallback trigger criteria fully preregistered (finite-start count, gradient norm outside a Clarke
  neighbourhood, condition number, bound proximity, >10 % flagged cells → pause and request approval).
- "No test threshold may be changed after real corrected-method returns are inspected."
- "Nothing from the void fit cache may be loaded" — closes a contamination route I had not raised.
- Canary is return-blind and ceilinged (24 CPU-h, 48 cumulative), and the full sweep is gated on a PI
  budget rather than assumed.

## 8. Recommendation

**Approve for implementation**, conditional on F1 (cost the sensitivity suite and put a ceiling on it)
and F2 (state `r_max`) being added to the plan — both are edits, not redesigns. F3/F4 are notes.

The five decisions are settled and correctly encoded; the implementation map is concrete; the tests are
the right ones. The remaining risk is budget, not fidelity.
