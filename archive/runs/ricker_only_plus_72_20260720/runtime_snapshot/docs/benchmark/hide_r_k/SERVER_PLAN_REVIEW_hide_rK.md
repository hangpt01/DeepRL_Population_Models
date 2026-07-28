# Review of `SERVER_IMPLEMENTATION_PLAN_hide_rK.md`

**Reviewer:** Claude (code-grounded audit against the live tree)
**Date:** 2026-07-16
**Subject:** the plan-only deliverable produced against `SERVER_FIX_BRIEF_hide_rK.md`
**Status:** read-only. No code changed, no tests run. This is a review comment for the
implementation agent (codex) to act on **before** it starts editing.

---

## 0. Verdict

**The plan is sound, faithful to the brief, and — where I could check it — code-accurate.**
Approve the *direction*. Do **not** start the full 30-file refactor in §8 yet: three things need to
be tightened first (a pre-flight identifiability probe, an explicit `full`-mode byte-identity
invariant, and a proper design section for the learned reward/risk model), and three PI questions
need answers. Details below, ranked by how much they can sink the experiment.

I verified the load-bearing claims against the tree rather than trusting the summary:

| Plan claim | Verified? | Evidence |
|---|---|---|
| Root cause = spec/design error faithfully routed to code, later amplified by native solvers | ✅ | Matches brief §0 working hypothesis; confirmed in code below. |
| Git provenance `f62054f0 / 320fa633 / b873314e / 5f9cf32a`, all Hang Phung, 07-09/07-15 | ✅ (with one nuance, see F) | `git log -1` on each hash resolves with those authors/dates. |
| `BasePolicy` receives the full `EnvironmentConfig` | ✅ | [methods/base.py:25](../../../src/real_ecology_benchmark/methods/base.py#L25) `self.env_cfg = env_cfg`. |
| `plus_native.fit` ignores the dataset, builds solvers from tables | ✅ | [methods/plus_native.py:37-38](../../../src/real_ecology_benchmark/methods/plus_native.py#L37-L38) `del dataset, beliefs`. |
| `moor_native` fits only `K`, reads growth from the table | ✅ | [methods/moor_native.py:24-56](../../../src/real_ecology_benchmark/methods/moor_native.py#L24-L56): K-grid search, but growth comes from `predict_ricker_next(self.env_cfg,…)` → `resolve_actions`. |
| Leak chain `assumption_config → real_environment_like → tables` | ✅ | [native_solver.py:35-61](../../../src/real_ecology_benchmark/native_solver.py#L35-L61), [native_solver.py:120-136](../../../src/real_ecology_benchmark/native_solver.py#L120-L136). |
| `safety_threshold = c_safe·K_base`, `K_ref = K_base` | ✅ | [config.py:279-284](../../../src/real_ecology_benchmark/config.py#L279-L284). |
| Public dataset has **no** `pop_id` and **no** `cost` | ✅ | [dataset.py:14-31](../../../src/real_ecology_benchmark/dataset.py#L14-L31). Brief §6.2 requires adding both. |
| Native belief filter is built from exact tables **before** the policy is fit | ✅ | [pipeline.py:185-193](../../../src/real_ecology_benchmark/pipeline.py#L185-L193) (filter `NativeSolver.build`) precedes [pipeline.py:228](../../../src/real_ecology_benchmark/pipeline.py#L228) (policy) and [pipeline.py:242](../../../src/real_ecology_benchmark/pipeline.py#L242) (`policy.fit`). |

The six conflicts the plan flags in its §7.1 are all real. Good work — that section is the most
valuable part of the deliverable.

---

## 1. Must fix before implementation (ranked)

### A. Add a **Phase-0 identifiability pre-flight** — cheapest thing that can save the whole build

This is my biggest concern and the plan's biggest omission. The plan files identifiability under
"requires execution" (§7.3) and then proceeds straight to the full build + smoke. That inverts the
risk. If `r` and especially `K` are **not** recoverable from the 4000 noisy transitions, the native
fit is garbage, the natives lose for the *wrong* reason (bad estimator, not fair uncertainty), and
the result is inconclusive — after you've spent the whole refactor.

There is a concrete structural reason to worry, already documented in the codebase: **6 of the 11
actions have `r_setpoint ≤ 0`, which zeroes `r_pos` and cancels `K` out of the density-dependent
exponent entirely** ([plus_native.py:6-9](../../../src/real_ecology_benchmark/methods/plus_native.py#L6-L9)).
So `K` is weakly identified by construction on more than half the action space.

**Action:** before building anything, write a standalone script that fits Ricker `(r(a), K)` by
least squares from a *frozen* public dataset (one recoverable cell, one sink cell, σ ∈ {0, 0.2}) and
reports the estimate vs the private truth and a profile-likelihood / CI width for `K`. This mirrors
the `SERVER_HANDOFF_hidden_dynamics_experiment.md` Phase-0 philosophy ("a cheap falsification before
you build"). Pre-register the reading:
- **identifiable** → build proceeds as planned;
- **not identifiable** → the experiment is about *estimation under weak identification*, which is a
  legitimate but *different* framing — flag it to the PI, because it changes what the headline number
  means and may need a larger transition budget as a control arm.

~1 CPU-h. Insert it as step 0 of §8.

### B. Pin the `full`-mode byte-identity invariant — the guardrail most likely to be silently broken

Brief §7 requires `full` to reproduce the current run, and the parent hidden-dynamics handoff
(guardrail 4c) requires the unhidden path to reproduce the completed motivation run **bit-for-bit**.
Your plan routes even `full` through the *new* `MethodContext`, regime-namespaced cache keys, and
relabeled outputs (§2.3, §3). Every one of those is a chance to perturb the `full` numbers — a
changed cache key writes a new file, a reconstructed config object changes float formatting in a
serialized hash, a re-ordered dict changes an RNG-seed derivation, etc.

The plan has a "full-mode no-op snapshot" test (§6.4), which is the right instinct, but the plan
states it as a test, not as a **contract**. Make it a contract and state it precisely:

> In `full` mode, every method-facing input, every RNG draw, and every fitted artifact must be
> byte-identical to the pre-change code. The `full` `MethodContext` is a transparent pass-through of
> the current `EnvironmentConfig`; it changes labels/namespacing **only**.

Then gate implementation on it: run the no-op snapshot immediately after the `MethodContext` split
(§8 step 2), before touching any hidden logic. If `full` moves at that step, stop — the refactor is
not inert.

### C. The learned reward/risk model is a major sub-project, not a table row

Brief §6.4 forbids the reward decomposition / benefit term / `K_ref` / collapse flag from entering
any planner; only the scalar `R_t` may survive, as a label. Your §7.1.5 correctly notices this forces
a *learned* method-side reward for every planner — but the plan then treats it as one line in the
change map (§3, `reward.py` row) and one PI bullet (§7.2.3). It is the single largest behavioral
change in the whole plan and it can dominate the result: today the general planners *plan* by rolling
learned dynamics and scoring with `build_reward` (which needs `K_ref` + `safety_threshold` + true
benefit); replacing that scorer with a fitted reward model changes what these methods *are*.

Give it its own design section answering, concretely:
- **What does the reward model see?** `(o, a, o')` public tuples → `R`? If it takes `o'`, and `o'`
  is a noisy observation of `s'`, it is implicitly re-learning `α·s'/(s'+K_ref)` from data. That's
  fair and correct, but it means the *quality* of this fit now heavily gates the general-vs-native
  comparison — say so.
- **One shared reward model, or per-method?** A shared, frozen, public-data reward model keeps the
  comparison clean (every planner scores rollouts with the same learned reward). Recommend that.
- **OGSRL risk without `s_safe`:** brief §6.3 option (b) says learn risk from public
  reward/termination tails. Spell out the concrete signal (e.g. terminal `done` + reward-drop events)
  so the guardian is reproducible, not hand-wavy.

Escalate the "shared learned reward + how OGSRL represents risk" as a **sharp** PI decision, not the
soft "approve the approach" bullet currently in §7.2.3.

---

## 2. Should fix (real gaps, lower blast radius)

### D. `C_low / C_high / regime_threshold_*` are also `K_base`-scaled and live in `env_cfg`

The plan's K-leak list centers on `rho/kappa/K_eff/K_ref/s_safe`. But the Allee and regime **form
thresholds** are scaled to `K_base` too — [config.py:294-299](../../../src/real_ecology_benchmark/config.py#L294-L299)
(`C_low = c_low_frac·K_base`, `regime_threshold_low = reg_low_frac·K_base`, …) — and they sit in the
same `EnvironmentConfig` every policy receives. Any hidden-mode Allee/theta/regime native fit that
seeds its threshold from `cfg.C_low` / `cfg.regime_threshold_*` recovers the `K_base` scale through
the back door. Your §5.2 correctly says to *fit* "candidate-specific nuisance parameters needed by
Allee, theta or regime forms" — add the explicit prohibition: **those thresholds must be estimated
from data, never initialized from `cfg.C_low`/`cfg.regime_threshold_*`.** Add them to the forbidden
name list checked by the No-r/K-access test (§6.1) and the table-relabel test.

### E. Make the native **filter** solver and the native **policy** solver share one fitted model

There are two `NativeSolver`s in play per native method: the belief **filter's** solver (built in
[pipeline.py:185-193](../../../src/real_ecology_benchmark/pipeline.py#L185-L193)) and the **policy's**
own solver (`self.solver = NativeSolver.build(...)` in
[moor_native.py:48](../../../src/real_ecology_benchmark/methods/moor_native.py#L48)). Today both come
from the exact tables so they're automatically consistent. In hidden mode both must come from the
**same fitted model and the same grid**, or the belief (`log_weights` over hidden states) and the
policy (`action_values` over that grid) silently desynchronize — and `_require_native_belief`'s grid
check ([moor_native.py:19-22](../../../src/real_ecology_benchmark/methods/moor_native.py#L19-L22)) will
either trip or, worse, pass on mismatched semantics. State the invariant: **one fitted model per
native cell, threaded into both the filter factory and the policy.** This is the concrete meaning of
your "fit native models before native filters" (§5.3) and it should be called out as a hard
consistency requirement, because the current pipeline builds the filter first.

### F. Soften the git-blame provenance claim for `f62054f0`

`f62054f0` resolves, same author/day — but its commit message is *"Removing claude_build, still
duplicated in real_ecology_cont_obser…"*, i.e. a file-move/refactor. `git blame` attributes moved
lines to the move commit, so "this commit **decided** E1/E8" overstates it. Reword to "blame
surfaces at `f62054f0` (a 07-09 refactor by the same author); the decision predates or coincides with
it." Doesn't change the verdict (spec-error faithfully implemented), just the precision of the claim.
Deliverable-quality nit, since brief §0 asks for exact provenance.

### G. `cost` is a **required** schema addition, not optional; be explicit about the dependency

The plan lists adding public `cost` to the dataset (§3) and putting costs in `MethodContext` (§2.2),
but frames it softly. Make the dependency explicit: methods currently get action cost *only* through
the rich table-derived `ActionSpec`. In hidden mode that object is withheld, so **without adding
`cost` to the public schema (+ a `collector.py` change to write it), every planner loses the cost
side of the reward and the comparison breaks.** `pop_id`, by contrast, is genuinely optional and
under per-cell training is a constant (see PI question 1) — keep them separate in the plan; they are
not the same decision.

---

## 3. Endorsements (keep these exactly as planned)

- **Regime split `expose_rk ∈ {full, hidden}` behind one interface, `full` preserved, regime
  recorded in configs/manifests/caches/outputs/summaries** — faithful to brief §5.5. Correct.
- **`MethodContext` as a separately-constructed sanitized view** rather than ad-hoc field deletion —
  this is the right architecture: it enforces the boundary structurally (a method literally cannot
  reach `K_base` because it never holds the config), which is far more robust than grepping for
  aliases. Keep it.
- **PLUS-native stays the four closed-world forms, no open-world, no explicit r/K candidate axis**
  (§5.4) — exactly brief §6.7 / §9.3. Correct, and resist scope creep here.
- **Environment truth untouched; eval protocol (5×4×horizon 50) locked; simulator keeps the private
  tables** — consistent with brief §5.3/§6.6 and hidden-dynamics guardrail 1. Correct.
- **Table-corruption test + family-relabel test as byte-identity gates** (§6.2, §6.3) — these are the
  strongest possible leak proofs. Keep them, and extend the forbidden-name set per item D.

---

## 4. PI questions to escalate (answer before building)

The plan lists these; I'm confirming they are genuinely PI-level and sharpening two:

1. **Population identity under per-cell training.** Brief recommends known-label/unknown-demographics,
   but the current runner trains one cell per population/family, so `pop_id` is a *constant* token per
   run — informationally empty. Options: (i) keep the constant token (harmless, matches brief letter);
   (ii) drop `pop_id` entirely for now; (iii) expand scope to pooled multi-population training (makes
   the token meaningful, but is a much larger change and arguably a separate experiment). **My read:
   (ii) or (i) for this fix; defer pooling.** Confirm.
2. **Safety information.** Keep `c_safe·K_base` fully evaluator-only and have OGSRL learn risk from
   public tails (brief §6.3 option b, plan's recommendation), *or* define an absolute public threshold
   independent of `K_base` (option a). This is coupled to item C — decide them together.
3. **Learned reward/risk model design** (item C) — approve a single shared public-data reward model
   and the concrete OGSRL risk signal.

Already settled by the brief (do **not** reopen): `full`/`hidden` both live behind one interface;
`hidden` is the default; no open-world PLUS; no `noisy` variant in this fix.

---

## 5. Suggested sequencing change (recommendation, not a blocker)

The brief's headline is **five** methods (`refplan`, `bamcts`, `ogsrl`, `moor_native`,
`plus_native`); the schema fix applies to all, but `mopo`/`delphic`/adapted-`moor`/`plus` are not on
the critical path to the publishable number. Your §8 does everything at once across ~30 files. I'd
stage it:

1. Phase-0 identifiability probe (item A).
2. `expose_rk` split + `MethodContext` + `full`-mode byte-identity gate (item B).
3. Sanitized dataset (+`cost`), private sidecar, learned reward/risk model (item C).
4. Native public-data fit + hidden grid + filter/policy consistency (item E), **five headline
   methods only**, through the table-corruption + family-relabel gates.
5. Two smoke cells (Amur tiger/Ricker/σ0 and Iberian lynx/Allee/σ0.2), then the sweep.
6. **Then** fold in `mopo`/`delphic`/adapted methods as a second pass.

This gets the headline result behind fewer moving parts and keeps the causal story clean.

---

## 6. Bottom line

Ship the plan **after**: (A) inserting the identifiability pre-flight, (B) elevating `full`
byte-identity from test to contract, (C) giving the learned reward/risk model its own design section
and a sharp PI decision — plus the smaller fixes D–G. The conflict analysis (§7.1) and the
`MethodContext` architecture are the plan's real strengths; preserve them. Everything I could verify
in the live tree checked out.

---

# Round 2 — Resolution after codex's rebuttal (2026-07-16)

Codex responded. Its rebuttal is correct on the merits of A, B, C, and E; I concede all four
framings. One correction runs the other way — codex's rebuttal cites a **source tree that does not
exist in this repo** — and one new cross-check reinforces C. Converged position below; this section
supersedes §1–§2 where they differ.

## R0. Correction back to codex — you cited a non-existent tree

Your rebuttal references `discrete_action_cont_obser/src/ecobench/data/collector.py:294` and
`discrete_action_cont_obser/src/ecobench/methods/plus_native.py:39`. **That path is not in this
repo.** The live package is `src/real_ecology_benchmark/` (shipped as `ecorl-real-ecology` in
[pyproject.toml:6,19](../../../pyproject.toml#L6)); `ecobench` / `discrete_action_cont_obser` appears
only under `docs/history/` — it is the removed/duplicated tree from commit `f62054f0`. Your own
*plan* used bare filenames that map correctly to `src/real_ecology_benchmark/`; only the *rebuttal*
drifted to the old path.

Your **substance survives** — I re-verified both claims in the live tree
([collector.py:294-295](../../../src/real_ecology_benchmark/collector.py#L294-L295),
[plus_native.py:39-50](../../../src/real_ecology_benchmark/methods/plus_native.py#L39-L50)) — but
**re-anchor every path/line to `src/real_ecology_benchmark/` before implementing.** Do not edit
`ecobench` paths; they are not on disk.

## R1. Item-by-item resolution

**A — conceded: Phase 0 is an estimator-design probe, not a success gate.**
You're right. The brief explicitly treats non-identifiability as *part of the real hidden-information
uncertainty, to be reported* ([SERVER_FIX_BRIEF_hide_rK.md §7, "Identifiability sanity"](SERVER_FIX_BRIEF_hide_rK.md#L291)),
not grounds to reject the regime. My "natives lose for the wrong reason" and "saves the build"
framing overstated it, and my "~1 CPU-h" was an unsupported number — both withdrawn.
**Converged:** run a cheap pre-flight as a *design aid* that (i) falsifies an obviously broken
estimator and (ii) characterizes `K`/`r` recoverability to *report*, not to gate. The mandated
**4000-transition budget stays unchanged** in the benchmark. Weak identifiability is a finding, not a
failure.

**B — conceded on two counts.**
(1) I was factually wrong that the plan routes `full` through `MethodContext`; your §2.2 scopes
`MethodContext` to hidden mode and `full` keeps the current `EnvironmentConfig` path. Withdrawn.
(2) Universal byte-identity is the wrong contract, because the brief *requires* new regime labels in
configs/manifests/output dirs/caches/summaries ([brief §7, "Regime split intact"](SERVER_FIX_BRIEF_hide_rK.md#L300))
— those artifacts must change. **Converged contract:** under fixed seeds, `full` mode preserves
method-facing inputs, RNG behavior, solver parameters, chosen actions, and metrics; byte-identity
checks apply only to canonical payload arrays / explicitly-unchanged artifacts, **excluding** regime
labels, paths, and cache namespaces. It remains a **contract gated as a step** (run the equivalence
proof right after the `MethodContext` split, before any hidden logic), which we both agree on.

**C — conceded and strengthened: my example labels were illustrative and one was wrong.**
You're right that `done` cannot be a collapse label: `done = result.done or forced_end`, and
`forced_end = (t == episode_length - 1)`
([collector.py:294-295](../../../src/real_ecology_benchmark/collector.py#L294-L295)) — horizon
truncation is conflated with true termination. And the genuine collapse signal (`entry` /
`entered_safety_region`) is a **private** field
([dataset.py:33-44](../../../src/real_ecology_benchmark/dataset.py#L33-L44)), so it is off-limits to the
method-side reward/risk model. That makes C *harder*, not softer: the public data has no clean
collapse label. **Converged:** the dedicated reward/risk design (still a blocker, still PI-approved)
must specify censoring/truncation handling (horizon vs termination) and the exact permitted public
labels; my `done` / "reward-drop" suggestions were examples, now retracted as under-specified.

**E — conceded: "one fitted model" is MOOR-only; PLUS needs one fitted solver per form.**
Correct. `plus_native` builds a solver *and* a belief bank per candidate form
([plus_native.py:39-50](../../../src/real_ecology_benchmark/methods/plus_native.py#L39-L50),
bank at [66-69](../../../src/real_ecology_benchmark/methods/plus_native.py#L66-L69)). **Converged
invariant:** MOOR — one fitted solver shared by filter and policy; PLUS — one fitted solver *per
existing form candidate*, each used consistently for that candidate's likelihood update and action
values; and in **both**, the pipeline's exact-table `native_discrete` filter
([pipeline.py:185-193](../../../src/real_ecology_benchmark/pipeline.py#L185-L193)) must be removed /
neutralized / replaced in hidden mode so no table-built solver survives.

**Sequencing guardrail — accepted (yours strengthens mine).**
Staging the five headline methods first is fine **only if** every not-yet-converted method **fails
loudly or is unavailable** in hidden mode — it must never silently retain full `EnvironmentConfig`
access. Bake that in as a hard assertion at policy construction (hidden mode + unconverted method →
raise), so a partial rollout cannot leak by omission.

**D, F, G — agreed by both sides**, no change.

## R2. Net converged action list (this is the authoritative to-do)

1. **Re-anchor all paths to `src/real_ecology_benchmark/`** (R0). Discard `ecobench` references.
2. **Phase 0**: identifiability *probe* (design aid, not a gate); budget stays 4000; report weak
   identifiability as a finding.
3. **`expose_rk` split + hidden-only `MethodContext` + `full`-mode equivalence contract** (numerical/
   behavioral under fixed seed, not universal byte-identity), gated as a step.
4. **Reward/risk design section + PI decision**: define censoring/truncation and permitted public
   labels precisely; the private `entry`/`reward_true`/`initially_unsafe` fields are off-limits.
5. **Native fit**: MOOR one shared fitted solver; PLUS one fitted solver per form; neutralize the
   table-built `native_discrete` filter in hidden mode.
6. **Hard fail-loud guard**: hidden mode + unconverted method ⇒ raise (no silent full-config access).
7. **D**: add `C_low`/`C_high`/`regime_threshold_*` to the forbidden K-derived name set.
8. **F/G**: soften the `f62054f0` provenance wording; treat `cost` as a mandatory schema addition,
   `pop_id` as optional.

PI questions unchanged (population identity under per-cell training; safety threshold; shared learned
reward/risk approach). Everything else in codex's plan stands.

---

# Round 3 — accuracy correction + final converged status (2026-07-16)

**Correction (mine to make): the plan itself was cleaner than an earlier chat summary implied.**
Verified against the plan text:

- The plan contains **no `ecobench` / removed-tree paths** (grep: none). That error was only in codex's
  Round-2 rebuttal, not the plan.
- The plan **already** fits **one parameterization per form** for `plus_native`
  ([plan §5.4](SERVER_IMPLEMENTATION_PLAN_hide_rK.md#L230)); the "one fitted model" slip was this
  review's over-generalization (item E), never the plan's.
- The plan imposes **no universal byte-identity on full mode** — byte-identity is scoped to the
  *hidden* table-corruption / family-relabel invariance tests
  ([plan §6.2-6.3](SERVER_IMPLEMENTATION_PLAN_hide_rK.md#L253)); full mode gets a fixed-seed **no-op
  snapshot** ([plan §6.4](SERVER_IMPLEMENTATION_PLAN_hide_rK.md#L272)).
- "reward-drop" was this review's example label (item C), not a plan proposal.

So the plan revision is **additive / sharpening, not error-removal.** The nine edits below fold the
converged resolution into the plan; none of them fix a pre-existing plan defect except where noted.

## Plan edits to fold in (authoritative)

1. Add **Phase 0** as a *non-gating* identifiability probe; keep the 4000-transition budget.
2. **Sharpen** the full-mode no-op snapshot into an explicit numerical/behavioral equivalence
   contract (method-facing inputs, RNG, solver params, actions, metrics under fixed seed).
3. Add a dedicated **reward/risk design section** with explicit censoring/truncation rules — and
   **correct** the OGSRL "public reward/termination evidence" wording, since `done` conflates true
   termination with horizon truncation
   ([collector.py:294-295](../../../src/real_ecology_benchmark/collector.py#L294-L295)); the private
   `entry`/`reward_true`/`initially_unsafe` fields are off-limits. *(This one is a real wording fix.)*
4. State **MOOR and PLUS solver/filter consistency separately** (MOOR: one shared fitted solver;
   PLUS: one fitted solver per form); neutralize the table-built `native_discrete` filter in hidden.
5. Add the hidden-mode **fail-loud guard** for unconverted methods (raise, never silent full-config).
6. Explicitly forbid **`C_low` / `C_high` / `regime_threshold_*`** in method-facing inputs (item D).
7. **Soften** the `f62054f0` provenance wording (file move ≠ decision origin).
8. State that **action `cost` is mandatory**, `pop_id` optional.
9. **Re-anchor** the file-map's bare filenames explicitly to `src/real_ecology_benchmark/`.

## Final converged status (agreed with codex)

1. **Revise the plan** to fold in the nine edits above. *(Gate 1 — cheap, no PI needed.)*
2. **PI-independent scaffolding may then proceed on explicit approval:** `expose_rk` split, hidden
   `MethodContext`, regime labeling, full-mode equivalence gate, forbidden-name set (D), fail-loud
   guard, and **mandatory action-`cost` schema support** (cost is PI-independent).
3. **Scientific components remain blocked on the three PI decisions:** population identity (incl. the
   *optional* `pop_id` half of the schema), public safety threshold, and the shared learned
   reward/risk approach. Do not finalize these until the PI answers.

---

# Round 4 — audit of the PI-updated plan (2026-07-16)

The revised plan **faithfully folds in all three PI decisions and all nine Round-3 edits.** Spot-checked:
PI-1 opaque `pop_id` + pooling-allowed/per-cell-first (§2.3.1, §5.1, §8.2.1); PI-2 fully-private
K-derived safety incl. `entry`/`initially_unsafe` (§2.3.2, §8.2.2); PI-3 one shared frozen
reward/risk surrogate with the exact allowed/forbidden input lists (§5). The nine edits are all
present (Phase 0 non-gating §7.8/§9.1; full-equivalence contract §7.4; reward/risk section + `done`
confound fix §5.4; MOOR/PLUS consistency split §6.3/§6.4; fail-loud guard §7.1/§9; `C_low`/`C_high`/
`regime_threshold_*` forbidden §7.1; softened `f62054f0` provenance §1.3; cost-mandatory/`pop_id`
§8.2; file-map re-anchored to `src/real_ecology_benchmark/` §3). The new env claim is code-accurate:
the simulator already computes `terminated = state_next == 0.0` vs `truncated` separately at
[envs.py:344-346](../../../src/real_ecology_benchmark/envs.py#L344-L346).

## One substantive finding: the public risk target barely ever fires

§5.4 fits OGSRL's risk model to the genuine-termination event `y_term = 1[terminated]`, i.e. exact
extinction `state_next == 0`. I measured how often that event actually occurs in the frozen private
sidecars:

- **40 cells sampled** (all 4 families, σ ∈ {0…0.4}, both reward modes): genuine extinction =
  **3 / 160,005 transitions (0.0019%)**; **39 of 40 cells had zero.**
- The single seeded cell (amur_tiger/ricker/σ0/safe): **0 / 4000** extinctions.
- By contrast the *private* safety-crossing event `entry` averages **0.72%** (~380× more common), and
  `initially_unsafe` ≈ 24% (many endangered pops start below `s_safe`).

**Consequence.** The §5.4 logistic risk model will hit its own `constant_single_class` fallback in
~all cells → `risk_hat ≈ (0+1)/(n+2) ≈ 2.5e-4`, a flat constant with no discriminative power. So
OGSRL's public "risk-aware" interface is effectively **inert on this environment**. This is not a
plan defect — the fallback is correctly and honestly specified, and §5.4 already flags that the target
is extinction, not the `c_safe·K_base` objective — but it means the mechanism the plan relies on to
give OGSRL public safety awareness produces no signal, *by construction of the PI's private-safety
decision*: the only public event that tracks the safe objective (`entry`) is private, and the public
env event (`terminated`) essentially never happens within horizon 50.

**This does not reopen the PI decision** — it's the honest consequence of it. But three cheap actions
before the run:

1. **Add `terminated` prevalence to the Phase-0 probe** (one line per cell). If it's ~0 as measured,
   you know up front the risk logistic is degenerate — don't discover it in aggregation.
2. **Pre-register the expectation for OGSRL in hidden mode:** its public risk model is a near-constant;
   in *safe* mode, safety pressure reaches methods almost entirely through the **reward surrogate**
   (§5.3 ridge on `R_t`, which carries the collapse penalty in safe mode), not through `risk_hat`.
   Report it that way so a flat OGSRL safety result isn't misread as "OGSRL can't learn safety" when
   the truth is "there is no public extinction signal to learn from."
3. **PI awareness note (no action required):** if the paper wanted methods to *learn* safety-relevant
   risk from public data, this env doesn't support it under the settled private-safety rule. A public
   *observation-based* low-abundance event (distinct from the exact `s_safe` threshold) would be a
   different, later safety-definition choice — explicitly out of scope for this repair.

## Verdict

The plan is **approval-ready.** Finding above is a *measure-and-document* item, not a redesign: fold
action 1 into Phase 0 and action 2 into the OGSRL reporting contract, and the plan can go to
implementation on the staging already agreed (revise ✓ → PI-independent scaffolding + cost → then the
PI-settled scientific core).

---

# Round 5 — corrections after codex's complete sweep (2026-07-16)

Codex ran the full sweep and was right on three counts; I independently reproduced the numbers and
found the same. This section **supersedes the specific figures in Round 4**; the qualitative
conclusion (public extinction risk is near-degenerate) stands and is if anything firmer.

**Complete-sweep numbers (verified, not sampled).** Round 4's 40-cell figure used an undocumented
stride sample — a fair critique. Corrected: codex's complete 288-cell sweep = **24 extinctions in
1,152,040 transitions (0.00208%), 280/288 cells zero, all in crab_eating_fox/theta.** I independently
reran over a separate complete run (psafe_overnight, 721 cells / 2.88M transitions) and got the
**identical rates** — extinction **0.00208%**, `entry` **0.671%**, `initially_unsafe` **27.19%** —
which makes the finding robust across runs. Use these figures, not Round 4's sample.

**Fallback constant correction.** With the planned 80/20 episode split the single-class fallback fits
on ~3200 rows, so `risk_hat ≈ (0+1)/(3200+2) ≈ 3.12e-4`, not the 2.5e-4 I wrote (that used the full
4000). Conclusion unchanged — a flat, uninformative constant.

**Reward-surrogate is an expectation, not a guarantee.** Round 4 action 2 said safe-mode safety
pressure "reaches methods through the reward surrogate." Codex is right to weaken this: public `R_t`
*contains* the safe-mode penalty, but the ridge model may not recover a useful observation-based
association. **Revised action: Phase 0 must measure reward-surrogate holdout RMSE/MAE per cell
alongside `terminated` prevalence** — do not assume the reward channel carries safety signal; verify
it.

**OGSRL — resolved: keep it as a headline method.** My earlier "is OGSRL still worth carrying"
question is withdrawn. The brief §6.6 locks the five headline methods for the controlled comparison;
demoting OGSRL would break comparability with the motivation run. Correct handling is to **keep it and
report honestly** — disclose `risk_fallback`, `terminated` prevalence, and reward-surrogate quality —
so a flat OGSRL safety result reads as "no public risk signal exists on this env," not "OGSRL failed."

**New finding (codex's, verified): 4,005 ≠ 4,000 transitions in some cells.** The registered budget is
"4000 transitions/cell" (brief §6.6, a *keep-unchanged* invariant), but episode-preserving collection
overshoots: **8/288 cells (codex) — 20/721 in the run I checked — hold 4005 transitions, and they are
exactly the crab_eating_fox/theta cells.** These are the *same* cells that show genuine early
termination — i.e. the overshoot is a direct symptom of early-terminating episodes, since the
collector adds one more complete episode to reach ~4000 and lands at 4005. This is a pre-existing
data-integrity issue independent of hide-r/K, but it matters here because (a) the brief asserts 4000
as an invariant and (b) the full-mode equivalence gate (§7.4) compares array projections, which
varying row counts break. **Add to Phase 0 and resolve before implementation:** decide truncate-to-4000
vs register 4005-as-actual vs documented exception, apply it identically to `full` and `hidden`, and
record it.

## Verdict (unchanged)

Approval-ready. The two Phase-0 additions (reward-surrogate holdout quality; the 4000/4005 budget
resolution) and the OGSRL honest-reporting contract are measure-and-document items, not redesigns.
Everything else in the plan stands as audited in Round 4.

---

# Round 6 — final scientific decisions folded in; confirmation (2026-07-16)

The PI settled the last three items (transition budget, safe-mode reporting, known-problem
diagnostics) and codex folded them into the plan. Verified faithful:

- **Budget** (plan §2.3 / §7 / §9): "4000 target transitions per cell, preserving complete episodes,"
  `target_rows`/`actual_rows`/`overshoot_rows`/`episode_count` recorded, assertion
  `4000 <= actual_rows <= 4024`, **and** the general invariant `0 <= overshoot_rows < episode_length`.
  I verified the bound is **provably tight, not merely observed**: the collector adds at most one
  complete episode past target ([collector.py:275-327](../../../src/real_ecology_benchmark/collector.py#L275-L327))
  and `episode_length = 25` ([config.py:442](../../../src/real_ecology_benchmark/config.py#L442)), so
  worst-case overshoot = 24 → max = 4024 exactly. Tying it to `episode_length` (not a hardcoded 4024)
  makes the assertion robust to config changes. Correct.
- **Safe-mode** (§2.3): pre-registered information-limited interpretation; no post-hoc threshold, no
  retune, no demotion, no result removal. Correct.
- **Known-problem diagnostics** (§5.3, §5.4, §7.8): near-constant extinction risk documented as an
  expected property; Phase-0 termination prevalence + fallback activation; evaluator-only reward-error
  stratified by a new private `safety_penalty_applied` field; public low-reward-tail (fit-fold 10th
  pct) reward error — all explicitly **non-tuning** (cannot touch fitting, hyperparameters, selection,
  inputs or actions). This cleanly operationalizes the "does the reward channel carry safety signal?"
  question as a report-only diagnostic. Correct.

**Retraction (mine).** Round 5 called the 721-cell run an "independent replication." Codex is right
that it is not: the psafe_overnight penalty sweep duplicates the same transitions across P∈{2,5,10,20}
(extinction is penalty-invariant, so the copies are byte-identical draws), and it overlaps the
motivation datasets. It confirms the prevalence *arithmetic* is stable; it is **not** independent
evidence. The plan now states this correctly (§5.4). The finding itself — extinction ≈ 0.002%, risk
channel inert by construction — is unaffected; it rests on the complete 288-cell sweep.

## Final status

**All scientific gates are closed.** The plan is decision-complete and approval-ready. Remaining path
is purely execution on the agreed staging: plan revised ✓ → PI-independent scaffolding + mandatory
action-cost → PI-settled scientific core, with Phase 0 (non-gating) run first. No open audit items.

---

# Round 7 — implementation audit (2026-07-16)

Codex implemented the full plan (~1500 LOC across 38 files + `native_fit.py`, `public_surrogate.py`,
`public_models.py`, `privacy.py`). I audited it against the live tree and by re-running. **Verdict:
sound — the leak is closed, full mode is preserved, gates are real. Approve.** One finding turned out
to be pre-existing (not codex's), detailed below.

## Verified (evidence, not self-report)

- **Leak closure is structural, not just tested.** `native_fit.fit_from_public_data` uses only
  `dataset.{observations,next_observations,actions}` + `MethodContext` scalars — it imports nothing
  that can read a table ([native_fit.py:50-99](../../../src/real_ecology_benchmark/native_fit.py#L50-L99)).
  The `MethodContext` boundary is enforced: hidden policies carry no `env_cfg` (asserted in the leak
  test), and constructing a method with `EnvironmentConfig` in hidden mode fails loud.
- **The table-independence test is robust despite incomplete `resolve_actions` patching.**
  `resolve_actions` → `real_action_table` → `realdata.effects_for/actions_for`
  ([actions.py:149-150](../../../src/real_ecology_benchmark/actions.py#L149-L150)), and the test
  patches `realdata.pops_for/effects_for/actions_for` + `NativeSolver.build` to raise — a true
  backstop that catches table access via any import path. (Minor hardening suggestion below.)
- **Gates are non-vacuous:** table-independence (patch-and-raise over every method), family-relabel
  byte-identity of fits/actions/diagnostics, non-tuning surrogate boundary (flip
  `safety_penalty_applied` → coefficients unchanged, diagnostics change), the 4000≤actual≤4024 bound,
  and the fail-loud guard. The forbidden-name set in `privacy.py` is comprehensive (all r/K aliases,
  `C_low`/`C_high`, `regime_threshold_*`, `kind`/`family`/`true_family`/`population`/`data_dir`).
- **114 tests pass** (re-ran `make test` independently, exit 0).
- **Full mode preserved relative to HEAD.** Regenerated Amur-tiger/Ricker/σ0/safe full-mode
  (seed 116, 4000×25) and compared to the frozen motivation dataset: **obs, next_obs, dones, all
  controls EXACT**; only `rewards` differ (434/4000 rows by exactly 5.0). Isolated in a throwaway
  HEAD worktree: **HEAD (`5f9cf32`, pre-codex) produces the identical 434-row difference** — so
  codex's change reproduces HEAD's full mode faithfully.
- **Phase 0 consistent** with the pre-registered finding: 0 terminations → `constant_single_class`
  (1/3202); and the evaluator-only stratified diagnostic shows reward-surrogate RMSE ≈ 5.0 in the
  penalty stratum (vs ~1.6 without) — the surrogate barely predicts the safe-mode penalty, confirming
  safe mode is information-limited, reported honestly as non-tuning.

## Finding (pre-existing, NOT codex's — flag for PI)

The full-mode safe-mode **reward labels already diverged from the frozen 07-11 motivation datasets at
commit `5f9cf32`** ("Add the known r,K model — Before hiding them"): 434/4000 rows (~11%) have the
collapse penalty toggled, trajectories otherwise identical. This means a `full` re-run today does NOT
bit-reproduce the reward labels of the run that produced the 860/864 motivation result — the brief's
guardrail 4c is technically broken by the *baseline*, before hide-r/K. **Action for PI/codex:** before
leaning on `full` as "the motivation reproduction," confirm whether the motivation results should be
regenerated on `5f9cf32`+ semantics, or whether the frozen 07-11 numbers stand. Orthogonal to the
hide-r/K work, but it affects how the `full` comparison arm is described.

## Minor hardening (optional)

The leak test's direct `resolve_actions` patches cover only `actions.` and `methods.moor.`; other
modules hold their own imported reference. It's currently safe because the `realdata.*` patches are
the real backstop — but a future refactor that reads a table without going through `realdata`
(e.g. a hardcoded constant) would slip past. Consider asserting the backstop explicitly, or patching
`resolve_actions` in every importing module, so the guarantee doesn't rest on an implementation detail.

## Verdict

Implementation is **approved**. Leak closed and proven; full mode preserved; Phase 0 confirms the
pre-registered information-limited safe mode. The only open item is the PI clarification on the
pre-existing `full`-arm reward-label shift — a baseline bookkeeping question, not a defect in this work.

---

# Round 8 — run audit of the matched sweep (2026-07-16)

Audited `real_ecology_runs/hidden_rk_comparison_20260716/` (jobs 58326885/886/887) per the run
handoff's explicit request to verify, not trust. **Execution verdict: PASS.** Reproduced codex's
completion claim from the evidence; no contradictory file/task/row/invariant found.

- **Completeness:** each arm has exactly 1,440 `summary.json`, `manifest_row_resolved.json`, and
  `episodes.csv`. `sacct` rollup: 288 full + 288 hidden + 1 analysis, **all `COMPLETED` exit `0:0`**.
  `completion.json` = complete, 1440/1440 both arms.
- **Provenance:** all four recorded SHA-256 (both manifests, registration.json, config) recompute
  identically.
- **Validity (the crux):** the two manifests are **identical on every scientific key except
  `expose_rk`** (1,440 unique rows each; populations/families/σ/reward-modes/methods/filters match).
  Single shared frozen config → planner (h5/96/32) and eval (5 seeds × 4 ep × h50) budgets are
  identical across arms. The comparison is apples-to-apples.
- **Leak held at run level:** hidden `public.npz` carries **zero** control fields; full `public.npz`
  retains `rho/kappa/K_eff/next_*`. Policy `fit_diagnostics` on the hidden side = `{expose_rk_hidden,
  fit_loss, fitted_form_ricker}` — no r/K.
- **Logs:** 577 stdout + 577 stderr at repo root (288×2 + analysis); **all stderr zero-byte**, no
  `FAILED`/traceback/`FATAL`/`incomplete sweep` markers.
- **Overshoot:** exactly 40 summaries at 4,005 rows per arm = Crab-eating fox/theta × 8 cells × 5
  methods, identical across arms, within `0 <= overshoot < 25`. Matches the earlier prevalence finding.
- **Regime took effect (execution sanity, not interpretation):** paired hidden−full is negative for
  the general methods (e.g. bamcts/safe mean −0.61; another model −3.86) — hidden underperforms full,
  the expected direction when the r/K advantage is removed.

**One flag investigated and cleared.** A recursive name scan found `population` and `safety_threshold`
in all 2,880 hidden `summary.json`. These are **evaluator reporting fields** (the summary is the
evaluator's terminal output, which needs population/family/σ for per-cell reporting and the threshold
for `unsafe_fraction`/`mvp` metrics). No method reads the summary; the method-facing artifacts
(`public.npz`, policy diagnostics, belief cache, surrogate — the latter two tested clean) do not
contain them. **Not a leak** — the `privacy` name-guard is scoped to method artifacts, not evaluator
reports, which is correct. (Optional tidiness: the summary need not echo `safety_threshold`, but it is
harmless.)

The execution is clean and valid. Result *interpretation* remains separate, and safe mode stays
pre-registered as information-limited — this run's clean execution does not by itself validate any
causal reading of a given performance gap.

---

# Round 9 — report audit (2026-07-16)

Audited the TeX report + figures in `.../analysis/` (`REAL_ECOLOGY_HIDDEN_RK_EXPERIMENT_RESULTS.tex`,
generator `make_hidden_rk_report.py`). **Verdict: PASS — numbers reproduce exactly, framing is
correct, nothing hand-copied.**

- **Headline reproduces to the decimal.** Recomputed independently from the raw `summary.json`
  (`operational_return_mean`, best-general = max(refplan,bamcts,ogsrl), safe mode, 144 cells):
  full best-general − MOOR-native = **−1.060**, hidden = **+2.666**, hidden wins **125/144** — exact
  match to codex. (Full wins only 6/144 — under known r,K the native is near-oracle, the expected
  shape.) 288 matched cells per arm, **0 unmatched**, confirmed.
- **Termination count reproduces exactly:** 12 events / 576,020 safe transitions, 4 cells
  (crab_eating_fox/theta × σ, 3 each), constant-risk fallback in **140/144**.
- **No scientific code changed:** `diff -rq` frozen-snapshot vs live `src/` shows only `.pyc`
  bytecode differences; all `.py` identical.
- **Artifacts present & valid:** 11 figures in both PNG and PDF, all non-empty, all 11
  `\includegraphics` refs resolve; PDF 17 pages; TeX log has **0 errors, 0 overfull/underfull, 0
  undefined refs, 0 warnings** (codex's compile claim verified).
- **Framing/constraints honored:** old result kept as "upper-bound ablation… not discarded"
  (respectful); 721-cell wording corrected to "not an independent replication"; information-limited
  safety caveat present; trajectory panels labeled illustrative (8×); target/actual/overshoot rows
  reported. The pre-existing 07-11 reward-label difference is correctly neutralized — both arms use
  current semantics and the 07-11 result is labeled "historical only," so the matched contrast is
  uncontaminated.
- **Trajectory rollouts** (Slurm 58328792): 8 tasks COMPLETED exit 0, visualization-only.

No defects found. The report is publication-quality and its central claim — general offline MBRL
flips from losing under known r,K to winning once r/K are hidden — is independently verified.

---

# Round 10 — correction: codex counter-audited and was right (2026-07-16)

Codex reviewed Round 9 and caught a real defect I missed, plus three interpretive over-reaches in my
summary. **I concede all four.** Round 9's "no defects found" and my "premise is recovered" framing are
retracted here.

**Defect I missed (now fixed).** The report's setting table said **36 populations / 28 recoverable /
8 sinks**. The manifests contain **9 / 7 / 2** — the larger numbers are population×family combinations
(9×4, 7×4, 2×4). I verified the *result* tables reproduced but never checked the descriptive matrix
counts, and asserted "no defects / nothing hand-copied" too broadly — the counts were hand-entered
metadata. Confirmed the fix: the report now derives them via `\ReportPopulationCount{}` etc. from the
manifest, and 36/28/8 are gone. My independent manifest recount agrees: 7 recoverable (Amur tiger,
Spotted turtle, Asian elephant, Puerto Rican parrot, Iberian lynx, Jaguar, Crab-eating fox), 2 sinks
(Egyptian vulture, Bottlenose dolphin).

**Three interpretive claims retracted (my chat summary, not the report — the report never made them):**

1. *"The old negative result was an artifact of the r/K leak."* Too strong. The known-parameter
   result is **valid for the public-demographics / structure-known regime**; it answered a different
   question, it was not spurious. And — as I myself flagged in Round 6/7 — the historical 07-11 run
   used different reward labels, so its outcome can't be attributed solely to `expose_rk` without a
   like-for-like regeneration. Correct framing: the leak meant the old setting didn't test the
   *intended uncertainty* question, not that its result was meaningless.
2. *"The paper's premise is recovered."* Over-reach on two counts codex is right about: (a) "best
   general" is a **cellwise max** over RefPlan/BA-MCTS/OGSRL — an envelope, not a single deployable
   policy; and (b) safe mode is pre-registered and empirically **information-limited**. Defensible
   statement: **hiding r/K restores a fair test of the intended setting and reverses the aggregate
   safe-mode best-general-vs-native ordering** (−1.060 → +2.666; envelope wins 6→125/144). The swing
   is robust (same envelope both arms); the absolute "generals win / premise recovered" is
   envelope- and reward-mode-dependent and should not be stated flatly.
3. *"Nothing hand-copied / no defects."* Accurate for the generated result tables, wrong for the
   prose metadata (the population counts). Scope error on my part.

**What stands:** the numerical headline audit (−1.060 / +2.666 / 125-144 / 288 matched / 12
terminations) reproduced exactly and is unaffected. The report, after codex's fix, is factually
correct. My error was overstating scope ("no defects") and promoting a comparative result into an
absolute claim. Codex's `SERVER_REPORT_AUDIT_RESPONSE_hide_rK.md` is the correct adjudication.
