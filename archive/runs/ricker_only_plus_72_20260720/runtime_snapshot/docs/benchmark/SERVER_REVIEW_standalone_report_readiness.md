# Review — Is the report standalone? And a finding that changes its central claim

**Reviewer:** Claude · **Date:** 2026-07-12
**Question asked:** *"I want the report to be enough as a document for someone without the code, covering setting, algorithm adaptation, results/analysis, and potential improvements. Does the .tex satisfy this?"*

---

## 1. Direct answer: **No. The `.tex` does not yet satisfy the requirement.**

codex correctly diagnosed the three missing sections — Benchmark Setting, Methods and Adaptations, Why the General Methods Fail — and then **wrote them into a separate Markdown file instead of into the report.**

Current `09_motivation_native_results.tex` section list (unchanged):

```
Question and Answer · Run Scope and Integrity · What Plots Are Available ·
Headline Returns · Training Diagnostics · Danger-Zone Actions ·
What the Plots Support · Potential Next Step · If Per-Timestep Figures Are Needed
```

**Still no Setting. Still no Methods/Adaptations. Still no failure mechanism.**

So the deliverable does not meet the brief, for three reasons:

1. **Wrong container.** The requirement is a *standalone report*. The missing content now lives in `SERVER_METHOD_ADAPTATIONS_motivation_native.md`, which is (a) not the report, (b) Markdown, so it cannot be `\input` into the PDF, and (c) a second file — which is precisely the split the brief was trying to eliminate.
2. **A reader with no code still cannot follow the argument.** They will not know what `refplan`/`bamcts`/`ogsrl`/`plus_native`/`moor_native` *are*, what the 11 actions or the safety threshold are, or why "the general methods learn a `ContinuousDynamicsEnsemble`" is the crux of the failure.
3. **The single most important sentence for the paper is in neither document** — see §2.

**The content codex wrote is good and accurate.** The problem is where it lives, not what it says. The fix is to port it into the `.tex` as three new sections before *Headline Returns*.

---

## 2. **NEW FINDING — `plus_native` is a form-ORACLE, not a naive baseline. Neither document says so.**

The adaptations file is careful to warn that *"naive is a dangerous word"* because the natives receive public `r`/`K` tables. **It does not go far enough.** I tested `plus_native`'s posterior over its four mechanistic candidates after one 50-step episode (Iberian lynx, σ=0.2, safe):

| true family | P(ricker) | P(allee) | P(theta) | P(regime) | identifies truth? |
|---|---|---|---|---|---|
| ricker | **0.999** | 0.000 | 0.000 | 0.001 | ✔ |
| allee | 0.000 | **0.901** | 0.001 | 0.098 | ✔ |
| theta | 0.000 | 0.021 | **0.917** | 0.062 | ✔ |
| regime | 0.000 | 0.097 | 0.000 | **0.903** | ✔ |

**`plus_native` recovers the true mechanistic form, from data, inside a single episode, in every family.** It is not structurally uncertain in any meaningful sense — it *resolves* the structural uncertainty and then plans with the true form.

This matters because the headline statistic compares the general methods against `max(plus_native, moor_native)`. **The "naive ecological baseline" the general methods lose to is, in practice, a solver that (i) reads exact per-action growth from the public action table, (ii) reads `K_base` from the species table, and (iii) identifies the true dynamics form.** That is close to an oracle.

### I caused this, and I should own it

The 4-form bank was **my** design decision (approved by the PI) after I found the original `K`-bank was unidentifiable — it made `plus_native` produce byte-identical episodes to `moor_native`. So the situation is:

- **`K` bank** → unidentifiable → PLUS degenerates onto MOOR (no method).
- **form bank** → *fully* identifiable → PLUS becomes a form-oracle (not naive).

**There is no configuration of `plus_native` in this environment that is simultaneously non-degenerate and genuinely structurally uncertain**, because the only structural axis (functional form) is fully resolvable from ~50 steps of data. That is a finding *about the benchmark*, and it is more damaging to the paper's premise than anything in the results doc so far.

---

## 3. **The good news: the negative result SURVIVES against the genuinely naive baseline**

The obvious worry is that the whole headline is an artifact of the form-oracle. It is not. `moor_native` **is** genuinely naive — a single fitted Ricker, misspecified on allee/theta/regime. Comparing the general methods against it *alone* (recoverable, safe, best-of-3 general):

| family | general | **moor_native** (naive) | gap | general wins | *plus_native (oracle)* | *gap* |
|---|---|---|---|---|---|---|
| ricker | 7.749 | 8.386 | **−0.636** | 3/28 | *8.393* | *−0.644* |
| allee | 7.762 | 8.544 | **−0.782** | 1/28 | *8.894* | *−1.132* |
| theta | 7.828 | 8.777 | **−0.950** | 1/28 | *8.931* | *−1.103* |
| regime | 7.462 | 8.370 | **−0.908** | 1/28 | *8.697* | *−1.235* |
| **all** | **7.700** | **8.519** | **−0.819** | **6/112 (5.4 %)** | *8.729* | *−1.029* |

**The general methods lose to the genuinely naive single-Ricker solver by −0.819, winning only 6 of 112 cells.** The form-oracle widens the gap (−0.819 → −1.029), but it does **not** create it.

### And the clean comparison is a *stronger* result

Look at the `moor_native` column by family: the gap is **−0.636 on Ricker** (where its assumption is *correct*) and **−0.782 / −0.950 / −0.908** on Allee / theta / regime (where its assumption is *wrong*).

> **A misspecified Ricker solver beats general offline MBRL by MORE on the families where its own assumption is false.**

That is the cleanest possible statement of the paper's inversion, and it is now free of the form-oracle confound. Model misspecification costs the native solver *less* than learned-model error costs the general methods. This should be the headline sentence.

---

## 4. What the report must change

1. **Stop calling the comparison baseline "naive."** With `plus_native` in the `max()`, it is not. Either:
   - **(preferred)** report the headline against **`moor_native` alone** as the naive baseline (−0.819, 6/112) and report `plus_native` **separately** as a *model-identifying* ecological solver — a strictly stronger, differently-motivated baseline; or
   - keep `max(plus, moor)` but relabel it honestly as *"best ecological solver, including one that identifies the true model form."*
2. **State that `plus_native` recovers the true form** (posterior 0.90–0.999). This is a benchmark finding, not a footnote: the paper's structural-uncertainty layer is *resolvable* by a simple mechanistic learner, which is why it does not create the intended handicap.
3. **Lead with the misspecification inversion** (§3) — it is the strongest, cleanest evidence and it survives every confound raised so far.

---

## 5. Verdict and recommended fix

| requirement | met? |
|---|---|
| results / analysis | **yes** |
| potential improvements | **yes** (though see my earlier addendum: the primary next step should be the model-source ablation, not the trajectory figure) |
| benchmark setting, for a reader with no code | **no** — not in the `.tex` |
| algorithm adaptations, and why the algorithms fail | **no** — not in the `.tex` (good content, wrong file) |
| **standalone** | **NO** |

**Recommended fix — one file, three new sections, ported from codex's Markdown into the `.tex` before *Headline Returns*:**

1. **Benchmark Setting** — hidden abundance, log-normal survey noise (σ ∈ {0, 0.1, 0.2, 0.4}), 11 real management actions with costs, per-population safety threshold, safe (P=5) vs yield (P=0) reward modes, 4,000-transition offline dataset, 5 seeds × 4 episodes × horizon 50.
2. **Methods and Adaptations** — what each of the five methods is; that the three general methods fit a `ContinuousDynamicsEnsemble` from logged data while the natives read public mechanistic tables; the filter-vs-planner distinction; **and the new §2 finding that `plus_native` identifies the true form.**
3. **Why the General Methods Fail Here** — learned-model error compounds over rollouts (deeper search *hurts*: h5 > h10 > h20, d5 > d10 > d20, roll6 > roll12 > roll24), while the native planning model is near-exact. Informational, not computational.

Then the existing results/plots/next-step sections stand as they are.

I can do this port and rebuild the PDF on request. It is a documentation change only — no new compute, no new claims beyond §2/§3 above, both of which are already measured.
