# Results — General-Method Search-Budget Audit

**Date:** 2026-07-12
**Run:** `real_ecology_runs/general_audit_20260712/`
**Question:** is the motivation experiment's negative result about **search budget** (the general methods were under-tuned) or about **the model** (the environment hands the natives an exact model)?

**Answer: MODEL-LIMITED. More search budget does not close the gap. The negative result stands.**

---

## 1. The verdict (pre-registered before the numbers existed)

72 safe-mode cells, each with **complete 24-config coverage**. Each general method gets the best of its as-run config **and all 24 tuned configs**.

| | |
|---|---|
| as-run general beats native | **2 / 72 = 2.8 %** |
| **tuned** general beats native (best of 24 configs) | **3 / 72 = 4.2 %** |
| mean gap, as-run | **−1.216** |
| mean gap, best-of-24 | **−0.962** |
| mean gain from extra budget | +0.253 |
| max gain from extra budget | +1.958 |

Quadrupling the search budget moves the win rate from 2.8 % to **4.2 %** and closes only **21 %** of the mean gap. The pre-registered threshold for the search-budget explanation was ≥ 50 %.

> **The general methods are not simply under-tuned.** This closes claim C6 from the audit request.

---

## 2. The mechanism — and it is the interesting part

### Depth: the robust effect

**Mean return by search depth**, marginalising over the other axes — on **recoverable populations** (so the two demographic sinks, whose safe-mode occupancy penalties dominate the absolute numbers, cannot drive the result):

| method | shallow | mid | deep |
|---|---|---|---|
| `refplan` | `h5` **7.024** | `h10` 6.220 | `h20` **5.646** |
| `bamcts` | `d5` **6.848** | `d10` 6.490 | `d20` **6.176** |
| `ogsrl` | `roll6` **7.209** | `roll12` 7.187 | `roll24` **7.113** |

**Monotone: deeper search is strictly worse, for all three methods.** For OGSRL the *as-run* rollout (6) is the best of all — extra depth only hurts.

This is the fingerprint of a **wrong model**: each extra rollout step compounds model error, so a deeper plan is a more *confidently wrong* plan. A search-limited agent improves monotonically with depth. These do the exact opposite.

### Breadth: where the (small) gains come from

| method | as-run | best tuned | gain | cells where tuning HURT |
|---|---|---|---|---|
| `refplan` | −6.021 | −5.255 | +0.766 | 2/72 |
| `bamcts` | −6.300 | −5.636 | +0.665 | 3/72 |
| `ogsrl` | −5.044 | −4.797 | +0.247 | 7/72 |
| *native baseline (best of 2)* | | **−2.740** | | |

The best configs pair **shallow depth with wide search** — `h5_seq256` (256 sequences, not 96) and `d5_sims256` (256 simulations, not 128).

### **[CORRECTED]** Pessimism is a wash — an earlier version of this doc overclaimed

A previous revision said *"every winner has `pessimism = 0`"* and attributed part of the gain to dropping pessimism. **That was wrong**, and it depended on which statistic you use:

| statistic | `refplan` | `bamcts` |
|---|---|---|
| best config **by mean return** | `h5_seq256_`**`pess0p5`** | `d5_sims256_`**`pess0p5`** |
| **mode** of the per-cell argmax | `h5_seq256_`**`pess0`** (35/72) | `d5_sims256_`**`pess0`** (33/72) |
| pessimism **marginal** (all cells) | pess0 −6.493 vs pess0.5 −6.509 | pess0 −6.858 vs pess0.5 −6.868 |

The two statistics disagree, and the marginal difference is **0.01–0.11 on returns of ~6** — noise. **Pessimism does not matter here.** Only the depth effect and the breadth effect are real. (Caught by codex's audit; the depth claim, which is the load-bearing one, survives unchanged.)

**Even at its best tuning, the general side sits ~2.1–2.9 below the native baseline (−2.740).** The budget was never the binding constraint.

---

## 3. Why this is trustworthy

- **G1 (no-op proof).** Surfacing `bamcts_depth`, `bamcts_simulations`, `ogsrl_rollout_horizon` as config fields is **provably inert**: at defaults, both methods reproduce the completed motivation run **bit-for-bit** (6/6, exact float equality). Without this, "we tuned them and it didn't help" would have been vacuous — the knob previously did not reach BA-MCTS or OGSRL at all.
- **G2 (dataset identity).** All 144 dataset hashes match the motivation run. Both sides provably consumed identical offline data; nothing was regenerated.
- **Canary.** 24 configs → **24 distinct output directories**, proving the `config_tag` path fix; without it, every config for a cell would have silently overwritten the others.
- **Coverage gate.** The verdict uses only cells with **all 24 configs present** (see §4).

---

## 4. A contamination I caught, and which way it cut

The first analysis reported **141 cells** and a tuned rate of 2.8 %. That was **wrong**. It pooled in 69 `yield` cells left over from the first sweep, which I cancelled mid-flight when the window shrank. Those cells have **partial config coverage**, so their "best tuned general" was a max over a *truncated* config set — which **understates** the tuned general.

**That bias ran in favour of my own conclusion.** The clean figure is **4.2 %**, not 2.8 % — i.e. tuning helps *more* than the contaminated number suggested. The conclusion survives the correction, but the correction had to be made, and it is recorded here rather than quietly absorbed.

`analyze_budget.py` now enforces the coverage gate explicitly and drops partial cells with a printed count.

Related: one summary was left **truncated** by the `scancel` (valid file, missing `config_tag`). My "already done" check tested `path.exists()` rather than "parses and is complete", so it was skipped rather than re-run. Fixed and the row re-run. **Existence is not completeness.**

---

## 5. Scope and limits — stated plainly

- **72 cells, safe mode only.** The window shrank from 6–8 h to 3–4 h mid-run, so the cell grid was halved (yield dropped) **before** resubmission, with the reason recorded (plan §R6). The **config grid was kept whole**, including BA-MCTS's most generous corner — the configuration most likely to overturn this very result. Cutting *that* to save time would have been suppressing the falsifying evidence.
- **Reduced power**, not reduced validity: all 9 populations, all 4 families, both σ endpoints retained. The yield mode is the safest axis to lose because the motivation run showed `refplan`/`bamcts` at a beats-both rate of exactly 0.000 in *both* modes at every σ.
- The pre-registered reading was fixed before any numbers existed and is unchanged.

---

## 6. What this means

Two explanations entered this run. One survives.

- ~~(A) the general methods are under-tuned → the headline is about search budget~~ — **rejected**. 4× the budget buys 1.4 points of win rate and 21 % of the gap.
- **(B) the general methods are model-limited** → **supported**, and now with a mechanism: deeper search *hurts*, which only happens when the model is wrong.

This strengthens, rather than rescues, the motivation run's finding: in this environment the "naive" ecological baselines are handed a near-exact model (`r` from the public action table, `K_base` from `species.csv`) while the general methods must learn one from 4,000 transitions. The gap is informational, not computational.

**The open PI decision in `SERVER_RESULTS_motivation_native_run.md` §4 is unchanged — except that option (c), "first check whether the generals are under-tuned", is now closed. They are not.**
