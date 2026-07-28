# Continuous-state setting: why it's a *future* study, and the Tier-1 step to do now

Design note + plan, written after the 5-method stress-116 benchmark (MOPO, RefPlan,
BA-MCTS, PLUS, MOOR). Motivating question: BA-MCTS came out the **weak** learned
method (beats-both-baselines 0.47 vs RefPlan 1.0, MOPO 0.97). RefPlan and BA-MCTS are
*continuous-control* papers; ours discretizes the state into bins. Is BA-MCTS weak
because the **discretized setting** disadvantages it — and would a **continuous-state**
setting (observe the model state, optionally with multiplicative observation error) be
more faithful and give better results?

Short version: **do a cheap resolution ablation (Tier 1) now; defer true continuous
(Tier 2/3) to a future, separate study.** Reasons below.

---

## 1. Refine the premise — what is and isn't discretized

The current POMDP hides **parameters**, not state: intrinsic growth `r_base`, Allee
threshold `C`, curvature `θ`, regime `z_t`. The abundance bin `x_t = floor(s_t/10)` is
observed **exactly** (no observation noise). Consequences:

- The **Bayes-adaptive / belief-over-models machinery is already exercised** — that's
  exactly the hidden `r_base`/`C`/`θ`. A hidden-parameter MDP ≈ a Bayes-Adaptive MDP,
  which is *literally what BA-MCTS targets*. So the methods are **not** out of their
  element on the belief axis.
- What is coarse is **state resolution** (bins of width 10 on a 0–1000 abundance
  scale) and there is **no state-observation noise**. The missing axis is **state
  observability**, not belief.

So "continuous would fix BA-MCTS" conflates two different things. BA-MCTS's weakness in
our results looks like **over-conservatism** (pessimism penalty + shallow 128-sim/
depth-5 search → it sits at high abundance ~46 and leaves harvest reward on the table),
which is a *planning* issue, not a *state-representation* issue. Continuous state is
not obviously the cure.

---

## 2. Why continuous is **not a good fit for this project** (but is for a future one)

This project's headline question is **"can general model-based offline RL beat
ecology-specific mechanistic baselines (PLUS, MOOR) under model misspecification?"**
Continuous state undermines that for four reasons:

1. **It breaks the apples-to-apples comparison.** PLUS (finite candidate-`r_base`
   models + value iteration) and MOOR (least-squares fit → Monte-Carlo *discretized*
   MDP) are finite-model methods *by construction*. A continuous-observation setting
   has no native discrete index for them; they'd each need a bolted-on state filter,
   which changes the method. The whole point of the benchmark is the **shared,
   identical setting** for all five — continuous removes it.

2. **The belief machinery is already tested.** As in §1, the methods' Bayes-adaptive
   core operates on the hidden parameters today. Continuous state adds a *different*
   challenge (state estimation under noise), not a fairer test of what these methods
   were built for.

3. **Faithful continuous = weeks of work, uncertain payoff.** Doing it *properly* means
   the papers' actual stacks — RefPlan's VAE-latent dynamics + MPPI, BA-MCTS's neural
   ensemble + double progressive widening. Offline model-learning in continuous space
   from limited data is also harder and less stable than clean count-based tabular
   posteriors. High effort, and the result may be *worse* in absolute reward (see §5).

4. **It doesn't isolate the variable.** If we jump straight to continuous+neural, a
   change in BA-MCTS could be the state representation, the neural model, the planner,
   or the noise — all at once. Not a clean experiment.

**It is a good fit for a *future, standalone* study** — a general-methods-only
"robustness to observation uncertainty" paper, where the absence of the ecology
baselines is fine because the question is different (which learner handles state
uncertainty best, not "do learners beat baselines"). Keep continuous for that.

---

## 3. Tier 1 — resolution ablation (DO NOW): is "BA-MCTS weak" a discretization artifact?

**Hypothesis:** if finer state resolution barely moves BA-MCTS toward RefPlan/MOPO,
the weakness is real (planning, not discretization). If BA-MCTS closes the gap as bins
get finer, the discrete benchmark under-sold it.

**Key advantage: this keeps all five methods comparable.** Finer bins are *still
tabular*, so PLUS/MOOR run unchanged — the head-to-head survives. This is the efficient
proxy for "more continuous" without leaving the setting.

### Design
- **Methods:** all five (mopo, refplan, bamcts, plus, moor_ricker) — keep the
  comparison; focus the *analysis* on how the general methods change.
- **Cells:** the threshold-sensitive ones, where resolution matters most — `allee`
  and `theta` at 5a and 10a (4 cells). Add `regime` if budget allows.
- **Reward mode:** `collapse_sensitive` (the decision-relevant one). `base` optional.
- **Seeds:** held-out 7001–7005, matched to the main run.
- **Resolutions:** current `bin_width=10` (already have it) + **one finer**, e.g.
  `bin_width=2` (a 5× refinement — enough to see the trend). Add `bin_width=5` only if
  the trend is ambiguous.

### The one tricky part — keep reward and thresholds on the **same physical scale**
`max_abundance=1000` stays fixed; only the binning changes. To keep rewards comparable
across resolutions, everything keyed off the *bin index* must rescale so it still
measures the same *abundance*:

| Quantity | Rule (so it tracks abundance, not bin count) | bw=10 (now) | bw=5 | bw=2 |
|---|---|---|---|---|
| `num_states` | `max_abundance / bin_width` | 100 | 200 | 500 |
| `reward.x_max` | `= num_states` → `R = α·x/x_max ≈ α·s/1000` | 100 | 200 | 500 |
| `reward.collapse_threshold_bin` | `≈ 50 / bin_width` (keep the s≈50–60 boundary) | 5 | 10 | 25 |
| `eval.live_danger_low / high` | `60/bw`, `200/bw` | 6 / 20 | 12 / 40 | 30 / 100 |
| `eval_start`, collection start ranges | already in **absolute abundance** → **unchanged** | — | — | — |

If `x_max` is *not* rescaled, rewards at different resolutions are on different scales
and the comparison is meaningless — this is the failure mode to watch.

### Mechanics (reuse the existing flow)
1. New env configs `*_116_bwN.yaml` (and `_10a`) with the rescaled `num_states`,
   `bin_width`, `reward.x_max`, `reward.collapse_threshold_bin`.
2. **Regenerate datasets** at each resolution (binning differs) — cheap, CPU, reuse
   `make_stress116_manifest.py --kind dataset` + the CPU worker.
3. **Re-run the control-gap gate** per resolution, or run the ablation with
   `REQUIRE_GATE_PASS=false` (it's an ablation, not the headline; decision-relevance
   should hold or strengthen at finer resolution).
4. Methods via the existing split: tabular (refplan/bamcts/plus/moor) → `comp` CPU,
   MOPO → `gpu:1`. Same manifest/worker machinery as the BA-MCTS add-on.
5. Aggregate with `aggregate_stress116_phase116.py`; compare each method's reward and
   the per-method beats-baselines rate **across bin widths**.

### Compute note
Tabular cost grows with `num_states`: transition tensors are `[M,S,A,S]`, value
iteration and the per-decision mixture/sampling are `O(S)`–`O(S²)`. At `bw=2`
(`S=500`) expect roughly **2–5× per-run** vs now — RefPlan/BA-MCTS may hit a few hours
each on CPU. Keep the cell/resolution count small, or trim RefPlan's
`num_sequences`/`latent_samples` for the ablation and disclose it. MOPO's categorical
head just widens to `S` outputs (modest). All still CPU/GPU-feasible in ~half a day if
scoped to 1 finer resolution × 4 cells × general methods.

### Read-out
- Does BA-MCTS's reward rise toward RefPlan/MOPO as bins get finer? → discretization
  artifact (re-frame the "weak" finding).
- Does it stay flat? → BA-MCTS is genuinely over-conservative; the main result stands,
  *strengthened* (robust to resolution).
- Either outcome is a clean, publishable robustness check.

---

## 4. Tier 2 — continuous observation + multiplicative noise (LATER)

The "give me the model state, with observation error multiplied" idea, scoped as a
**future general-methods-only robustness study**:

- **Env:** expose `o_t = s_t · η`, `η ~ LogNormal(0, σ)` (multiplicative survey error —
  ecologically the right noise model). `s_continuous` already exists in `info`, so this
  is a thin observation-wrapper change.
- **Methods:** each needs a **1-D Bayesian state filter** — maintain a belief over the
  current bin/state from the noisy `o_t` instead of observing it exactly — then feed
  that belief into the existing planner. This restores genuine **state** uncertainty,
  the regime where belief methods *should* differentiate.
- **Sweep `σ`** → degradation curves. Prediction worth testing: RefPlan/BA-MCTS
  (belief-based) degrade **more gracefully** than point-estimate methods as `σ` grows.
- **No PLUS/MOOR head-to-head** (they'd need the same filter) — fine, because the
  question is now "which learner handles observation uncertainty best," not "do learners
  beat baselines." That's the standalone story.

## 5. Will continuous be "better"? Intuition (not a promise)

Two opposing forces:
- **Pro:** thresholds are continuous (Allee `C≈150`, collapse near `s≈50–60`) but seen
  at ±10 resolution. Finer/continuous state → better threshold localization → less
  over-conservatism. This could **specifically help BA-MCTS's reward gap** (it's the
  one leaving harvest value unused). So *exact* finer observation → modest uniform gain,
  maybe a bit more for BA-MCTS; ranking probably stable.
- **Con:** multiplicative observation **noise removes information** — large `σ` is
  strictly harder than clean bins. And continuous offline model-learning is harder and
  less stable than count-based tabular posteriors.

**Net:** don't expect "continuous → better numbers." Exact finer observation gives a
small uniform improvement; observation *noise* likely *lowers* absolute reward. The
scientific value is the **comparison** (does resolution change the ranking? whose
robustness curve is flattest?), not higher absolute scores.

---

## 6. Recommendation

1. **Now:** Tier 1 resolution ablation — 1 finer `bin_width` (2), threshold-sensitive
   cells, all 5 methods, collapse_sensitive, held-out seeds. Cheap (no new method code,
   reuse datasets/flow), keeps the comparison intact, and directly answers the
   load-bearing question: *is BA-MCTS weak because of discretization, or for real?*
2. **Then decide:** the Tier-1 result tells you whether Tier 2 (continuous + obs noise,
   general-methods-only robustness study) is worth the larger effort.
3. **Tier 3** (native neural RefPlan/BA-MCTS) — out of scope for this project; a
   separate continuous-control study.

Scope reminder: all of this lives in `claude_build/` (the running repo), not
`DeepRL_Population_Models/` (the clean repo). Do not touch `.wandb_env`.
