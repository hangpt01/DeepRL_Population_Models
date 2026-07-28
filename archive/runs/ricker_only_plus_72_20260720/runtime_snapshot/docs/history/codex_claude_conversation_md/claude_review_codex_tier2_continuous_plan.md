# Claude's review of Codex's Tier-2 continuous-state implementation plan

Date: 2026-06-20.

Reviewed:

- `docs/planning/codex_stress_pomdp_22_6_tier2_continuous_implementation_plan.md` (Codex's plan)
- `docs/22_6_Continuous_Observation_New_Baselines.tex` (the design)
- `docs/stress_pomdp_116_continuous_successor_handoff.md` (code-level handoff)
- `docs/discrete_action_continuous_obser_state/` (Delphic ICLR24, OGSRL NeurIPS25, Confounded POMDP ICML24)
- Repo registries: `claude_build/src/environments/__init__.py`, `claude_build/src/models/__init__.py`

This is planning-only commentary. No code is authorized or implied. I have not modified anything.

---

## Bottom line

Codex's plan is strong and I would build from it. Its biggest value is **not** the module list — it is the set of semantic edge-cases it caught that the design `.tex` left contradictory, and the upgrade of the decision-relevance gate. I would adopt ~80% of it as written.

My disagreements are concentrated in three places, all about **scope and the one load-bearing design knot**:

1. The "**one shared learned latent filter for every method**" is the central scientific risk, and I think Codex's resolution (train a learned latent state-space proposal from observations) is simultaneously the *most* complex option and the one that most undermines the "mechanistic vs learned" contrast. The emission kernel is **known** (σ_obs is in the metadata), so a much simpler known-emission particle filter is available and should be the starting point. The "shared vs faithful-baseline" tension should be settled before anything else.
2. Codex pushes for **paper-faithful Delphic-CQL and full GMB-CPO-discrete first**. Given this project's compute history (GPU fair-share stalls, multi-day runs, CPU/GPU splits — see the Stress-116 benchmark notes), I'd invert that: runnable adaptation first, paper-faithful upgrade only if the contribution needs it.
3. The plan **underweights eval-time cost**. The filter now runs *inside the per-step decision loop* for every method, on top of particle MPC. That multiplier, not training, is the likely wall-clock bottleneck.

Plus one genuinely additive observation Codex missed: **σ_obs is itself the confounding-strength knob for Delphic** (given a privileged-state behaviour policy), which makes the noise sweep do double duty and turns Decision #6 from "open" into "foundational."

---

## Where Codex is clearly right (adopt as-is)

These are the catches that justify having run the plan at all. I verified each against the design `.tex` and the handoff and agree:

- **`C_safe` naming collision (Decision #1).** The design's reward Eq. (5) safety floor `C_safe` collides with the regime model's `C_safe=90` *safe-regime Allee threshold*, and theta has no biological `C` at all, and Tier-1's physical collapse threshold was ~50. This is a latent bug. Renaming the benchmark floor `s_safe_threshold` and reserving `C` for biology is mandatory. Agree.
- **`s=0` + stocking/harvest-revenue paradox (Decision #3).** The design says `s=0` is absorbing *and* defines stocking `δ(a)` (which would revive a dead population) *and* negative-cost harvest (which would mint revenue forever from an extinct stock). These cannot all hold. Codex's resolution — exact `s=0` ends the episode, keep the reward on the entering transition, mask padding — is correct. This is the sharpest catch in the document.
- **Floor-only dynamics vs Tier-1 snap-to-zero (Risk table / Decision #2).** Tier-1 *snapped* a sub-threshold bin to absorbing zero; the Tier-2 design says "non-negativity floor only." These are different physics. Latching the first healthy→unsafe entry for the one-time penalty while continuing true dynamics (absorbing reserved for exact `s=0`) is the right reconciliation. Agree.
- **Three-controller decision gate (Decision #14).** This is Codex's best *additive* contribution. The design `.tex` gate compares a true-state oracle against a noisy Ricker MPC, which confounds *information advantage* with *model advantage* — it could pass purely on clairvoyance. Splitting into (1) belief oracle, (2) belief Ricker MPC [both observation-matched, hard gate], and (3) clairvoyant oracle [diagnostic upper bound only] cleanly isolates structural misspecification under observation error. Strongly agree. (One caveat below.)
- **Predict latent next-state, then compose with the known emission — never fit a Gaussian on `o_{t+1}` directly.** The design `.tex` says methods predict "next observation (or latent)"; Codex correctly removes the ambiguity. Fitting directly on `o_{t+1}` conflates measurement noise with process/model uncertainty and would corrupt exactly the MOPO/Delphic pessimism penalties. Agree.
- **Delphic is not "an ensemble over m."** The design `.tex` describes Delphic loosely enough that it degenerates into ordinary epistemic ensemble disagreement, which is *not* delphic uncertainty. Codex is right that paper-faithful Delphic needs compatible latent-confounder worlds + a behaviour-policy model + a behaviour-value model, with `u_Δ = Var_w Q^{π}_w` as a Bellman penalty. (I disagree on *sequencing*, not on the diagnosis — see below.)
- **No internal grid/VI as the primary planner.** Reintroducing fixed grids would make "no bins" cosmetic and smuggle back an undeclared `s_max`. Continuous sampled MPC / fitted value is the right call. Agree.

---

## Substantive disagreements and refinements

### 1. The "shared filter" is the whole ballgame — and the design and Codex are in quiet tension

This is the decision everything else hangs on. Three facts collide:

- The design `.tex` §filter says the particle filter uses "**the learned dynamics ensemble** as the proposal." That is *a* method's model, not a shared one.
- Decision #7 wants "**one** learned public-data state filter checkpoint, frozen, fed to every method."
- Decisions #8–#9 forbid the filter from using the true equations and say `m` is an unidentifiable latent.

Codex resolves this by inventing a new entity — a *model-agnostic learned latent transition proposal trained from public trajectories*. That is internally consistent, but it has two costs Codex underplays:

- **It is the most complex option on the board.** Training a latent state-space model from observations alone, with no true `s`, is a deep variational SSM / DVBF-style problem and is itself a research artifact whose identifiability is exactly the concern raised in Decision #9. Making it "shared infrastructure that must be solid before Phase 4" front-loads the hardest unsupervised-learning problem in the project.
- **It hands the mechanistic baselines (PLUS, MOOR) a learned dynamics front-end**, which partly defeats the "mechanistic vs learned" contrast that is the entire point of beating PLUS/MOOR. Codex flags this in its risk table but then keeps the shared filter anyway.

**What Codex underuses: the emission kernel is KNOWN.** σ_obs is in the dataset metadata and the action authority (`h, δ`) is deterministic and known. So a bootstrap particle filter needs *only* a transition proposal — the hard part (the likelihood) is exact. That opens a much cheaper ladder:

- **Rung 0 (start here):** known-emission bootstrap PF with a *weakly-informative* transition (e.g. log-space random walk with calibrated volatility, or persistence). No learning, no global ceiling. At σ_obs=0 it reconstructs `s` exactly (o=s), giving a clean validation target.
- **Rung 1:** let the transition proposal be *each method's own* dynamics model — for PLUS/MOOR that means a **Ricker** particle filter (which is the *faithful* thing: PLUS/MOOR are state-space Ricker models, so a Ricker PF is their natural state estimator), and for learned methods their own learned ensemble.
- **Rung 2 (Codex's option):** one shared learned latent proposal, reported as a *common-reference ablation*, not the primary.

**My recommendation:** redefine "shared" as **shared emission + shared initial prior + shared feature interface + shared RNG/seed pairing**, with a *method-appropriate transition proposal*, and carry the single learned shared filter as one reported ablation. This preserves cross-method comparability where it matters (interface, priors, seeds) without forcing one learned dynamics assumption into every method. It also lets PLUS/MOOR stay genuinely mechanistic. This contradicts Decision #7 as written, which is exactly why it should be settled first.

### 2. PLUS specifically may be misrepresented by a single shared state filter

PLUS's mechanism is a posterior over candidate Ricker-`r` models. Different candidates imply *different state trajectories*, so the statistically correct object is a **Rao-Blackwellized / per-candidate bank of state filters** (one state belief per candidate `k`, weighted by the candidate posterior), not one shared `p(s|h)` with PLUS's model weights bolted on top. Forcing PLUS onto a single shared state filter and a separate candidate posterior is a defensible *approximation*, but it is not "PLUS adapted faithfully." If beating PLUS is the headline claim, PLUS deserves its faithful (marginalized-PF) form, or the approximation must be disclosed as part of the baseline definition. This reinforces recommendation #1.

### 3. Filter quality becomes a global confound — make it a measured covariate, not an invisible front-end

Whatever the filter is, every downstream method is *capped* by it: dynamics models are trained on filtered pseudo-states (an errors-in-variables / EM coupling), and planners roll out from the filter's belief. At σ_obs=0.4 the filter RMSE may **dominate and compress the between-method gaps**, so a null result could mean "methods are equal" or "the shared filter threw away the signal everyone needed." The plan reports filter RMSE but treats it as one metric among many. I'd elevate it:

- Report filter RMSE as a **covariate** alongside every return delta.
- Add an **oracle-filter ablation** (feed methods the true `s`) at one cell to upper-bound what *any* method could achieve given perfect state estimation. This separates "method is weak" from "filter is the ceiling" — the single most likely way the Tier-2 story goes ambiguous.

### 4. σ_obs *is* the confounding-strength knob for Delphic (Codex missed this; it's a free win)

This is the most useful thing I can add. Work through what the confounder actually is:

- The Tier-1 collector (`mixed_danger_zone_116`) acts on **true abundance `s_t`** (privileged), via abundance thresholds. The learner sees only `o_t`.
- At **σ_obs = 0**, `o_t = s_t`: the learner *sees the confounder*. And because the sub-policies condition only on `s` (so `a ⊥ m | s`), the hidden parameters `m` are *not* action confounders either — they are just unobserved transition heterogeneity. **Confounding ≈ 0.**
- At **σ_obs > 0**, `s_t` becomes hidden and is exactly a classic action-outcome confounder (behaviour acted on `s_t`; outcome `s_{t+1}` depends on `s_t`). **Confounding grows with σ_obs.**

So the noise sweep `{0, 0.1, 0.2, 0.4}` *is* a confounding-strength sweep, and Delphic's predicted signature is a **monotone increase in its advantage with σ_obs**. That is a clean, falsifiable prediction and a strong framing for the paper, and it means Codex's "inject a confounding-strength control" exit criterion for Delphic is already built into the design — it's σ_obs.

**Two consequences:**

- **Decision #6 (privileged-state behaviour policy) is not "open" — it is foundational.** If the collector acts only on `o_t`/public history, there is no `s`-confounding and Delphic collapses to an ordinary epistemic ensemble at *every* noise level. Lock it to "privileged true-`s` behaviour" or the Delphic contribution evaporates. I'd promote this above Decision #1 in priority.
- It also means at σ_obs=0 you should *expect* Delphic ≈ MOPO. That's a feature (a built-in negative control), not a failure — state it as such so a reviewer doesn't read it as Delphic underperforming.

### 5. theta should be reframed as a negative control, not merely "diagnostic/non-blocking"

Both docs keep theta non-blocking (Decision #15). But the Tier-1 gate gaps for theta were already marginal (reward gap 0.64 at 5a, just over the 0.5 diagnostic bar; see the handoff gate table), and **observation noise will wash out theta's subtle curvature signal further** — at σ_obs=0.4 theta's structural misspecification may simply *not be decision-relevant*. Rather than carry it as a limping headline cell, frame theta explicitly as a **negative control**: a cell where misspecification is *not* expected to be decision-relevant, so the *prediction* is that learners should **not** beat PLUS/MOOR. If they do — especially as noise rises — that's a red flag for overfitting/leakage, not a win. This turns a weak cell into an informative one.

### 6. Eval-time compute is the real risk, and "cached belief trajectories" is half-wrong

Codex's compute row counts ~336 method/cell/ablation *fits* and worries about training. But the dominant cost is **decision-time**: every method now runs a **particle-filter update inside the per-step loop**, *then* particle MPC, for 5 seeds × 50 episodes × 50 steps × 7 methods × (6 cells × 4 σ_obs) = a large constant on top of planners that were *already* the Tier-1 bottleneck (RefPlan ~50 min/run; MOPO full-budget eval ~2.8 h on an L40S per the benchmark notes). BA-MCTS with progressive widening + particle beliefs + per-node fitted value is the prime suspect for blowing the wall clock.

On caching specifically — Codex lists "cached belief trajectories" as a mitigation. **Split it:**

- **Offline-dataset beliefs: cache once, big win.** The logged actions are fixed, so the belief trajectory over the training data is method-independent (given the shared/frozen filter). Compute it once per (cell, σ_obs) and reuse for every method's training. Valid.
- **Eval-time beliefs: cannot be cached across methods.** Each method takes *different* actions, so the (obs, action) stream — and therefore the belief — diverges at the first step. Caching across methods here is infeasible. (It only helps for deterministic re-runs of the *same* method.)

Practical recommendations: predeclare a **hard per-run wall budget**; reuse the **CPU/GPU split that already rescued the Tier-1 final run** (CPU-bound methods off the GPU queue); and consider trimming held-out to fewer episodes for the expensive tree method, or running BA-MCTS on a reduced cell subset, *declared up front*.

### 7. The gate's "belief oracle" is now a real POMDP solver — reuse method machinery

The design `.tex` gate oracle was a cheap deep-copy-the-env exhaustive MPC with true state. Codex's "belief oracle" (knows the true dynamics family + prior, sees only noisy obs, does belief-space MPC) is scientifically better but is **substantially more code** — it's essentially a competent true-model belief planner. To avoid building a second planning stack just for the gate, **plug the true model into the same particle-MPC used by the methods**. That keeps the gate comparable to the methods and bounds the extra work. Worth making explicit in Phase 2.

### 8. Two smaller items

- **The reward leaks exactly one bit of latent truth, by design.** The `−20` penalty is computed on `(s_t, s_{t+1})` (latent), and the agent observes `R_t`, so it can infer "I just crossed the floor." This is realistic (a population crash *is* observable) and should be **kept**, but documented as the single channel where latent state enters the learner's input — otherwise it looks like a leak in code review. Codex notes this; I'd just make it a one-line explicit statement in the information contract.
- **The Tier-2 method set is a deliberate subset.** The repo registers `combo`, `cql`, `iql`, `romi`, `hmmdp_baseline` beyond the seven methods in the plan. Neither the design nor Codex ports them to Tier-2. That's reasonable, but state it as a scoping decision so it's not mistaken for an oversight.

---

## On the open decisions: where I differ from Codex's defaults

I agree with Codex's recommended defaults on Decisions **1, 2, 3, 4, 5, 8, 12, 13, 14, 17, 19, 21, 22**. Deltas:

- **#6 (behaviour-policy information):** agree with "privileged true-`s`," but **upgrade its status** — it is foundational for Delphic (see §4), not a Phase-2 nicety. Lock first.
- **#7 (meaning of "shared filter"):** **disagree with one-frozen-learned-filter-for-all as the primary.** Recommend shared emission/prior/interface + method-appropriate transition, learned-shared filter as an ablation (see §1–§2).
- **#9 (interpret `m`?):** agree it's largely unidentifiable, but "don't evaluate parameter recovery" is too broad — **PLUS's `r`-posterior concentration and the regime `z`-posterior *are* meaningful, evaluable diagnostics** (Codex's own Phase-5 exit criterion relies on PLUS concentrating). Keep state RMSE as the headline; keep method-specific parameter diagnostics where the parameter is actually estimated.
- **#11 / #12 (Delphic/OGSRL headline variant):** agree paper-faithful is the *goal*, **disagree on sequencing** — ship the runnable adaptation first, upgrade if warranted (see scope note below).
- **#15 (theta):** agree non-blocking, but **reframe as a negative control** (see §5) rather than a limping headline cell.
- **#16 (must every noise level pass the gate?):** agree Allee/regime pass at σ_obs ∈ {0, 0.1, 0.2} and 0.4 may be a declared stress regime — and note this dovetails with §4: at high σ_obs the *interesting* result is the confounding effect, not whether the misspecification gate still clears.

Codex's "lock Decisions 1–16 and 18–19 before any code" is right. I'd just reorder the very front to: **#6 → #7 → #1/#3** because #6/#7 reshape the most modules.

---

## Scope and sequencing

Codex's 8-phase order is sound and I'd keep it, with two changes:

1. **Build the shared particle-MPC + continuous dynamics ensemble as the first thing in Phase 4, and have *both* the learned methods *and* PLUS/MOOR plan through it.** Codex already implies this; making it the explicit Phase-4 entry point means Phase 5 (MOOR/PLUS) and the Phase-2 gate oracle all reuse one planning stack.
2. **Delphic and OGSRL: MVP-first.** Implement the design-`.tex`-level adaptations (Delphic-MOPO-style penalty; OGSRL Lagrangian with the posterior-probability safety cost Codex correctly specifies) to get a *complete, runnable* pipeline end-to-end, then upgrade to Delphic-CQL / GMB-CPO-discrete *only if* the simpler version shows signal and the contribution needs the faithful form. Codex inverts this (faithful first). For a research timeline against this project's compute reality, end-to-end-first de-risks far more than faithfulness-first. Label the MVP versions honestly as adaptations.

Net: I'd characterize Codex's plan as **correct but over-scoped for a first pass**. The faithful Delphic-CQL (10–15 worlds × behaviour model × behaviour-value model × delphic Bellman penalty) plus full GMB-CPO-discrete plus a learned shared latent SSM is, realistically, several months of engineering before a single headline number exists. An MVP spine produces the σ_obs sweep — the actual Tier-2 result — far sooner.

---

## Recommended go-ahead

I'd give the plan a **conditional go**, in this order:

1. **Settle Decisions #6 and #7 first** (privileged behaviour; meaning of "shared filter") — they reshape the most modules and they decide whether Delphic and the mechanistic baselines are meaningful at all.
2. Adopt Codex's edge-case resolutions (#1, #2, #3) and the three-controller gate verbatim.
3. Start the filter at **Rung 0/1** (known-emission PF, simple or own-model transition); treat the learned shared latent proposal as an ablation, not a prerequisite.
4. Add the **oracle-filter ablation** and **filter-RMSE-as-covariate** so a compressed result is interpretable.
5. Build the **shared particle-MPC spine** before any single method; reuse it for methods, baselines, and the gate oracle.
6. **MVP Delphic/OGSRL first**, faithful upgrade later; predeclare the eval-time wall budget and the CPU/GPU split before the full matrix.
7. Treat **theta as a negative control** and expect **Delphic ≈ MOPO at σ_obs=0** — both are built-in sanity checks, not failures.

Everything here is commentary on the plan. No code, configs, or data have been touched, per the standing instruction to stop at planning.
