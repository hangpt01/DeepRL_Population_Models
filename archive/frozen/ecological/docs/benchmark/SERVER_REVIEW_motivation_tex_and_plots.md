# Review — `09_motivation_native_results.tex` + plotting script + figures

**Reviewer:** Claude · **Date:** 2026-07-12
**Under review:** `docs/benchmark/09_motivation_native_results.tex`, `scripts/plot_motivation_native_results.py`, `docs/benchmark/figures/motivation_native/`

**Verdict: accept, with one substantive correction and two fixes.** The TeX is honest, the numbers are right, and it does not oversell. But it declares the PI's **most-requested figure impossible when the repo already contains the tooling to make it**, and one figure silently changes population scope.

---

## 1. What I verified (all pass)

Every quantitative claim in the TeX was re-derived independently from the run artifacts:

| claim in the TeX | verified |
|---|---|
| natives beat/tie general in **860/864** paired cells | ✔ (4 general wins) |
| `refplan` 0/288, `bamcts` 0/288, `ogsrl` 4/288 | ✔ |
| family gaps −0.670 / −1.133 / −1.107 / −1.235 | ✔ exact |
| σ gaps (fig. 2) | ✔ exact: −0.698, −1.119, −1.118, −1.210 (n=28 each) |
| 1440 main + 1152 resolution summaries, 0 flips, 8/8 acceptance | ✔ |
| budget audit closes only **21 %** of the gap; deeper rollouts hurt all three | ✔ |
| `training_history.csv` on every row (1440), 1152 PNGs | ✔ |
| `refplan`/`bamcts` log `dynamics_log_mse` only; `ogsrl` also logs `surrogate_loss` | ✔ |

**Crucially, the TeX does *not* repeat the pessimism overclaim** that codex caught in the results doc. It correctly attributes the mechanism to depth/breadth only. Good.

The framing is right: it states plainly that this is a **negative result for the original motivation**, and does not try to rescue it. That is what the PI asked for.

---

## 2. **SUBSTANTIVE — "actions over time" is declared unavailable. It is not.**

The TeX (§"What Plots Are Available" and §"If Per-Timestep Figures Are Needed") says the actions-over-time figure "cannot be derived from the completed main-run artifacts alone" and would need a bespoke "trace-capture pass".

**The first half is true. The second half understates the situation, and it matters, because this was the PI's headline request ("actions over time (all methods in 1 figure)").**

The repo **already has both halves of that pipeline**:

| script | what it does |
|---|---|
| `scripts/capture_trajectories.py` | re-fits each method on one cell and saves **per-step abundance, action, and reward** (`{method}_action`, `{method}_reward`, lines 34–61, 94–95) |
| `scripts/plot_trajectories.py` | already plots **per-method action-frequency heatmaps, median action id, and immediate reward over time** (docstring lines 6–14) |

So "actions over time, all methods in one figure" is **not a new capability — it is an existing one that was never pointed at the native methods.**

### What actually blocks it (two small edits)

1. `capture_trajectories.py:27` — `METHODS = ["mopo", "refplan", "bamcts", "moor", "plus", "delphic", "ogsrl"]`. **The natives are absent.** It captures the *adapted* PLUS/MOOR, not `plus_native`/`moor_native`.
2. `capture_trajectories.py:81` — `make_filter_factory(cfg, dataset, "learned")` is **hardcoded**. The natives require `filter="native_discrete"`, and `run_method` now *hard-rejects* the wrong pairing (added this session). So as written, adding the natives to the list would raise rather than silently mis-run — good, but it must be routed per method.

### Cost: negligible

Per-row runtimes on this cell grid are `moor_native` 10 s, `plus_native` 10 s, `refplan` 96 s, `bamcts` 430 s, `ogsrl` 705 s → **~21 min of CPU for all 5 methods on one cell.** Three or four illustrative cells is **~1–1.5 CPU-hours**, i.e. minutes of wall time. Reusing the frozen datasets means no regeneration.

### Recommendation

Wire the two natives into `capture_trajectories.py` with per-method filter routing, capture 3–4 cells (suggest: **Amur tiger/ricker** where the native assumption is correct, **Iberian lynx/allee** where the mechanistic forms disagree most, and **Egyptian vulture/theta** as the sink), and produce the all-methods action-over-time panel. Label it **illustrative**, as the TeX already correctly insists — the audited statistics remain the 1440-row sweep.

This turns a "we can't show you that" into the paper's most legible figure. It is the single highest-value item in this review.

---

## 3. **FIX — the danger-zone figure changes population scope without saying so**

`plot_danger_actions` (script line 204–209) filters on `reward_mode == "safe"` **only**. Every other figure in the document filters to **recoverable populations** (`plot_headline_returns`, line 97).

So the danger-zone heatmap silently pools the **two demographic sinks** with the seven recoverables. Measured contribution:

| group | safe episodes | in the danger band | share of the figure |
|---|---|---|---|
| recoverable | 11 200 | 7 099 (63.4 %) | 81.6 % |
| **sink** | 3 200 | 1 600 (50.0 %) | **18.4 %** |

I had expected the figure to be sink-*dominated*; it is not — **63 % of recoverable episodes genuinely enter the danger band**, which is itself a useful fact the TeX under-sells. But an 18 % sink admixture is still an undeclared scope change, and the sinks are reported separately everywhere else in this project by design.

**Fix:** either add the `RECOVERABLE` filter for consistency, or split the heatmap into two panels (recoverable | sink). The second is better — the sinks' danger-band behaviour is genuinely different and worth showing.

**Secondary:** the mean is **unweighted across episodes** (line 209–210), so an episode with 1 danger step counts as much as one with 40. A `danger_steps`-weighted mean would be more representative. Minor, but easy.

---

## 4. **FIX — an under-sold finding that helps the paper**

The TeX says (§Danger-Zone Actions):

> "Among recoverable populations, collapse entry is essentially zero ... The safety pressure mainly activates on the two demographic sinks."

That is true **about collapse** (`s ≤ s_safe`) but it reads as though recoverables never come under safety pressure at all. They do: **63.4 % of recoverable safe-mode episodes enter the danger band** (`s_safe < s ≤ 4·s_safe`). The distinction matters — the policies are *repeatedly* near the boundary and *avoid* collapsing, which is a stronger and more interesting statement than "nothing happens".

Recommend separating the two sentences: collapse ≈ 0 on recoverables, **but** the danger band is entered in ~2/3 of episodes, so the action-composition figure is measuring real decisions under real pressure — not an empty denominator.

---

## 5. Minor notes

- `plot_danger_actions` re-normalises `arr / arr.sum()` (line 209). The evaluator already emits fractions summing to 1 when `danger_steps > 0`, so this is idempotent — harmless, but it hides that the guard on line 208 (`arr.sum() > 0`) is what actually excludes never-in-danger episodes. Worth a comment; the guard is correct and load-bearing.
- The training figure is honestly captioned ("not a convergence trace for RefPlan/BA-MCTS"). That is exactly right — those two log only final fit diagnostics, and the caption resists the temptation to imply otherwise.
- `action_labels()` reads from the **frozen snapshot** (`run/code/real_ecology_data/actions.csv`), not the live tree. Correct — figures stay tied to what actually ran.
- Verification footer cites the right artifacts and the frozen commit. Good practice.

---

## 6. Answer to the PI's actual question: *do the plots support the motivation?*

**No — and the TeX says so, correctly and without hedging.** For the record, my independent read of the same artifacts:

- **Returns by family (fig. 1) is the money plot, and it lands the wrong way up.** The native baselines lead on *every* family, and their lead is **largest on Allee/theta/regime** — precisely the families where the naive Ricker-form solver was supposed to break. The intended structural-uncertainty penalty is not merely absent; it runs backwards.
- **Returns by σ (fig. 2)** shows the gap is negative at every noise level, so partial observability is not the mechanism either.
- **Training diagnostics** rule out the boring explanations (no divergence, no missing logs, finite holdout error) but cannot rescue the story.
- **The danger-zone figure** is a genuine safety diagnostic, but with collapse ≈ 0 on recoverables it cannot carry the conservation argument.

The supported reading is the one the TeX gives: **a well-specified discretized ecological solver is a very strong baseline in this environment, and general offline MBRL does not dominate it.** Combined with the search-budget audit (deeper search makes the general methods *worse*), the failure is **model-limited**, and the environment hands the natives a near-exact model via the public action and species tables.

---

## 7. Recommended actions, in priority order

1. **Wire the natives into `capture_trajectories.py`** (+ per-method filter routing) and produce the actions-over-time / reward-over-time panel. ~1 CPU-hour. **This is the PI's top request and it is 90 % built already.**
2. **Scope-fix the danger-zone figure** — recoverable-only, or split recoverable | sink.
3. **Reword §Danger-Zone Actions** to separate "collapse ≈ 0" from "danger band entered in 63 % of episodes".
4. (Optional) weight the danger composition by `danger_steps`.

Nothing in the TeX needs retracting. The document is a fair, well-evidenced account of a negative result, which is what was asked for.

---

# Addendum — review of codex's patches + the "Potential Next Step" section

**Date:** 2026-07-12 (second pass)

## A. codex's patches: all three verified, all correct

| my finding | codex's patch | verified |
|---|---|---|
| danger heatmap silently pooled sinks | now **split into recoverable / sink panels** (`plot_danger_actions`, lines 208, 218–219, 276) | ✔ |
| TeX under-sold the danger-band finding | now states **7,099 / 11,200** recoverable safe-mode episodes enter the band | ✔ (matches my count exactly) |
| "actions over time impossible" was overstated | now says *not available from the completed artifacts, but feasible* — and **cites `capture_trajectories.py` + `plot_trajectories.py`** and the `native_discrete` routing that is missing | ✔ |
| figures | regenerated 21:53 | ✔ |

Nothing to add. The results package is now accurate.

## B. The "Potential Next Step" section is well-written — but it proposes a **figure**, not an **experiment**

The section opens: *"The most meaningful next step is a small paired trajectory diagnostic."*

**I disagree, and the disagreement matters.** A trajectory panel is qualitative mechanism illustration. It is explicitly (and correctly) labelled as unable to change any headline statistic. **It therefore cannot change the PI's decision between (a) reframe and (b) redesign.** It is a nice figure. It is not the most meaningful next step.

### The most meaningful next step is to test the central causal claim, which is currently only INDIRECTLY supported

Both results documents rest on one causal claim:

> *The natives win because the environment hands them a near-exact model — `r_eff` from the public action table, `K_base` from `species.csv` — while the general methods must learn dynamics from 4,000 transitions. The gap is **informational**, not computational.*

The evidence for it is all **indirect**: planning depth was ruled out (~4 % effect); more search budget doesn't help (21 % of the gap); deeper search actively *hurts* (model-error fingerprint). Every one of those is consistent with the claim — **none of them tests it.**

**The direct test:** give the general methods the *same mechanistic model the natives get*, and see whether the gap closes.

- If it **closes** → the deficit really is the learned model. Informational asymmetry **confirmed**, and option **(b)** (redesign the environment so solvers no longer receive a near-exact public model) is *known to work* before anyone invests in it.
- If it **does not close** → the central diagnostic claim in both results docs is **wrong**, option (b) would not rescue the motivation either, and option **(a)** is forced.

**This is the only cheap experiment that can change the PI's decision.** The trajectory plot cannot.

### Honest correction to my own first instinct: this is NOT a free config flip

I initially assumed `filter=ricker` / `filter=true_family` (already in `make_filter_factory`, pipeline.py:176–181) would do it. **They will not.** I checked:

- `build_method` constructs policies as `METHODS[method](cfg.environment, cfg.model, cfg.planner, seed=…)` — **the filter's proposal is never handed to the policy.**
- All three general methods **plan with their own learned ensemble**: `refplan.py:40` (+ `_MemberProposal`, line 72), `bamcts.py:49`, `ogsrl.py:300` — each calls `ContinuousDynamicsEnsemble.fit(...)`.

So `filter=*` changes **state estimation only**, not the planning model. Running `filter=ricker` on the generals would answer a *different, weaker* question ("does better filtering close the gap?"), and quietly mislabelling it as the model ablation would be a serious error.

**What it actually needs:** a planning-model switch on the three general policies — e.g. a `model.dynamics_source ∈ {learned, ricker, true_family}` knob that swaps the `ContinuousDynamicsEnsemble` for a `MechanisticProposal` inside `fit()`. That is a contained change (the `MechanisticProposal` class already exists and is exactly what the natives and the adapted PLUS/MOOR use), and it must pass the same **G1-style no-op proof** the search-budget audit used: at `dynamics_source=learned` the methods must reproduce the completed run **bit-for-bit**.

### Cost

3 methods × 72 safe cells × (96 + 430 + 705) s ≈ **25 CPU-h per model source**; two sources (`ricker`, `true_family`) ≈ **50 CPU-h → ~15 min at 240 cores**. On all 288 cells it is still under 1 hour of wall time. This is *cheaper than the trajectory capture debate is long*.

### Recommended framing for the TeX

Keep the trajectory diagnostic — it is a good figure and the PI asked for it. But **demote it from "the most meaningful next step" to "an illustrative companion"**, and promote the model-source ablation to the primary next step, because it is the one that:

1. tests the paper's central diagnostic claim rather than assuming it;
2. is capable of *falsifying* our own conclusion; and
3. determines which of the two PI options is even viable.

## C. Suggested edit to §Potential Next Step

Replace *"The most meaningful next step is a small paired trajectory diagnostic"* with a two-tier next step:

1. **Primary (decisive, ~50 CPU-h):** model-source ablation. Give `refplan`/`bamcts`/`ogsrl` the mechanistic model (`ricker`, then `true_family`) instead of the learned ensemble, gated by a bit-identical no-op proof at `learned`. This is the direct test of the informational-asymmetry claim, and it can falsify it.
2. **Secondary (illustrative, ~1 CPU-h):** the paired trajectory diagnostic exactly as codex describes it — three pre-registered cells, all five methods in one action-over-time / reward-over-time panel, labelled qualitative.

Everything else in the section — the three pre-registered cells, the "qualitative mechanism evidence only" caveat, and the closing paragraph on the (a)/(b) decision — is right and should stay.
