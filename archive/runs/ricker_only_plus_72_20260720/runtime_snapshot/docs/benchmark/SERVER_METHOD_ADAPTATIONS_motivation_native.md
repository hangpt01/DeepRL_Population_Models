# Method Adaptations — Motivation Native-Baseline Experiment

**Date:** 2026-07-12  
**Scope:** explanation-only companion to
`09_motivation_native_results.tex`,
`SERVER_RESULTS_motivation_native_run.md`, and
`SERVER_RESULTS_general_method_audit.md`.

## Short Answer

The ecological baselines in the motivation experiment are **native ecological
baselines for this benchmark**, but they are **not exact reproductions of the
published original PLUS/MOOR workflows**.

They do get the important "naive ecological solver" treatment the experiment
needed:

- their own discretized belief filter (`filter=native_discrete`),
- a tabular ecological transition/reward model,
- value iteration / QMDP-style action values,
- no shared particle filter,
- no learned `ContinuousDynamicsEnsemble`,
- no online reward leakage.

But they are not paper-faithful originals:

- they are not the vendored SARSOP/POMDPX hmMDP pipeline from `baseline_original/`,
- they are not the full published PLUS candidate-model prior,
- they are not the original fishery MOOR workflow,
- they are benchmark-native adaptations to the real-ecology continuous-state,
  noisy-observation setting.

The most important caveat is stronger: in this benchmark, the "naive" native
baselines are not weak. They receive public action-specific growth set-points
and public species carrying-capacity information, so their mechanistic planning
model is close to exact. The completed experiment therefore compares:

> learned approximate model + short-horizon/general planner
> versus
> public mechanistic ecological model + tabular native solver.

The natives should be strong under that comparison, and the results show that
they are.

## Why This File Exists

The results TeX explains the outcome. This file explains the method-adaptation
semantics behind that outcome, especially the distinction between:

1. the **adapted** ecological methods already present in the benchmark
   (`plus`, `moor`), and
2. the **native** ecological baselines built for the motivation experiment
   (`plus_native`, `moor_native`).

That distinction matters because the motivation question was not whether a
particle-MPC adaptation called PLUS/MOOR is competitive. It was whether general
offline MBRL can beat genuinely native, simple ecological solvers under noisy,
partially observed, structurally uncertain conservation dynamics.

## Benchmark Setting Being Tested

The motivation experiment uses the real-ecology benchmark:

- **Populations:** 9 real conservation populations.
- **Dynamics families:** Ricker, Allee, theta-logistic, and regime-switching.
- **Observation:** abundance is hidden and observed through noisy observations;
  the sweep uses four observation-noise levels.
- **Actions:** 11 discrete management actions from the real action table.
- **Public controls:** actions expose growth set-points and carrying-capacity
  effects through public control variables.
- **Reward modes:** both yield-style and safety/collapse-aware reward modes are
  evaluated.
- **Offline data:** all methods in a cell consume the same logged public dataset.
- **Evaluation:** policies are evaluated on paired cells with identical datasets;
  native and general methods are compared across filters.

The crucial structural feature is that, in the real set-point setting, much of
the ecological transition is publicly determined by the action/species tables:

- action-specific growth set-points are public,
- the species carrying-capacity scale is public,
- capacity changes are public cumulative controls,
- only abundance and, for regime-switching, regime mode remain hidden.

This makes the native problem closer to a MOMDP than a fully opaque POMDP.

## General Offline MBRL Adaptations

The general methods in this benchmark are not full reproductions of their
published deep-RL systems. They are lightweight, inspectable, NumPy-based
adaptations:

| Code identifier | Reader-facing label | What it is here |
|---|---|---|
| `refplan` | RefPlan-inspired | posterior ensemble planner over learned linear dynamics |
| `bamcts` | BA-MCTS-inspired | belief-rooted finite-action tree search over learned linear dynamics |
| `ogsrl` | OGSRL-inspired | guarded offline policy with learned linear dynamics, kNN support, and safety constraints |

All three learn a `ContinuousDynamicsEnsemble` from the offline dataset. This is
the important shared limitation: even when they plan more deeply or with more
search, their rollouts are only as good as the learned model they fit from
logged transitions.

The search-budget audit found that increasing search did not close the gap.
Deeper rollouts made all three general methods worse on recoverable populations,
which is the expected fingerprint of model error compounding over longer
rollouts. This is why the results are diagnosed as **model-limited**, not simply
under-tuned.

## Existing Adapted Ecological Methods: `plus` and `moor`

Before the motivation-native work, the benchmark already had methods named
`plus` and `moor`. These are **adapted particle-MPC ecological methods**, not
native discrete solvers.

### `plus`

`plus` is a mechanistic PLUS-style adaptation:

- it keeps a bank of Ricker-style mechanistic candidates,
- under real set-point mode the candidate axis is primarily carrying-capacity
  scale,
- it updates candidate weights from public evidence,
- it plans with particle MPC.

It is close in spirit to PLUS because PLUS is already mechanistic, but it is not
the full published PLUS system. It does not implement the original candidate
prior or solver stack.

### `moor`

`moor` is a mechanistic MOOR-style adaptation:

- it fits a single Ricker proposal,
- under set-point mode, growth is mostly action-determined and the fit mainly
  identifies carrying-capacity scale,
- it plans with particle MPC under that fitted proposal.

It is close in spirit to an interpretable ecological expert baseline, but it is
not the original fishery MOOR workflow.

### Why these were not enough

The original motivation asked for native ecological baselines. The adapted
`plus`/`moor` methods still participate in the benchmark's continuous
particle-MPC infrastructure. They do not answer the stricter question:

> what if the ecological baseline is allowed to be a native discretized
> ecological solver with its own belief grid and tabular plan?

That stricter question motivated `plus_native` and `moor_native`.

## Native Ecological Baselines Built for the Motivation Experiment

The motivation experiment added two native methods:

| Code identifier | Meaning |
|---|---|
| `moor_native` | native single-Ricker ecological solver |
| `plus_native` | native posterior over mechanistic ecological forms |

Both are routed through:

```text
method in {plus_native, moor_native}
filter = native_discrete
```

This routing is not just a label. The evaluator always constructs and steps a
belief filter and reports filter metrics. Therefore the native baselines needed
a real `DiscreteGridFilter`, not an ignored belief argument or a shared particle
filter pretending to be native.

## `moor_native`

`moor_native` is the simpler native ecological baseline.

It does:

- fit a single Ricker-form model from public offline observations/actions,
- search over carrying-capacity scale,
- build a discrete abundance grid,
- tabulate transition and reward,
- solve by value iteration/QMDP-style action values,
- act from its native discrete belief.

It intentionally remains Ricker even when the true family is Allee,
theta-logistic, or regime-switching. That is the intended "naive ecological"
misspecification.

However, it also receives the benchmark's public ecological structure:

- the action table's growth set-points,
- the species carrying-capacity scale,
- public cumulative capacity controls.

So it is naive in functional form, but not blind about the public management
mechanism.

## `plus_native`

`plus_native` is the native PLUS-style baseline.

It does:

- maintain a posterior over four mechanistic forms:
  Ricker, Allee, theta-logistic, and regime-switching,
- discretize and solve each candidate independently,
- keep a candidate-specific belief bank,
- update model evidence from action/observation evidence only,
- choose actions by posterior-weighted candidate action values.

The candidate bank is over **mechanistic forms**, not over Ricker parameters.
This was a deliberate correction. In this setting, action-specific growth and
capacity information are public enough that a Ricker parameter bank is weakly
identified and can collapse onto the MOOR behavior. The form axis is the actual
structural-uncertainty axis in the benchmark.

The regime candidate carries a hidden binary regime mode, so its hidden state is
`(abundance, regime)` rather than abundance alone.

Important caveat: in this benchmark, `plus_native` is better described as a
**model-identifying ecological solver** than as a naive baseline. In a
one-episode diagnostic on Iberian lynx, sigma 0.2, safe mode, it put posterior
mass 0.999 / 0.901 / 0.917 / 0.903 on the true form for Ricker / Allee /
theta-logistic / regime-switching cells. In other words, the candidate-bank
axis is not merely present; it is quickly resolved from the data.

This does not invalidate the negative result, but it changes the cleanest
framing. The best-native comparison includes a strong form-identifying solver.
The genuinely naive comparison is against `moor_native` alone.

## Did the Ecological Baselines Get Their Naive Original Setting?

The honest answer is **yes in the benchmark-native sense, no in the
paper-faithful reproduction sense**.

### Yes: they got a native naive ecological treatment

They were not forced through the shared learned particle filter. They were not
implemented as general learned dynamics models. They were given the native
machinery the motivation question required:

- a discretized state/belief representation,
- tabular ecological transition and reward tables,
- mechanistic assumptions instead of learned neural/linear dynamics,
- a solver-like value-iteration path,
- model evidence updates from public observations only.

For `moor_native`, the naive single-Ricker assumption is preserved. For
`plus_native`, the finite mechanistic candidate-bank idea is preserved.

### No: they are not the exact original published settings

They are not the original published solvers or workflows:

- `moor_native` is not the original MOOR fishery pipeline.
- `plus_native` is not the full published PLUS candidate prior.
- neither native method uses the vendored SARSOP/POMDPX hmMDP solver stack.
- both are rewritten for this benchmark's continuous abundance, 11-action,
  real-data, noisy-observation setting.

So the report should not say "paper-faithful PLUS/MOOR." It should say
"benchmark-native ecological baselines" or "native discrete ecological
adaptations."

### Also: "naive" is now a dangerous word

The motivation expected naive ecological baselines to be structurally
disadvantaged. But in this benchmark, the native ecological baselines are
strong because the public tables give them most of the dynamics structure they
need. Their misspecification is mainly the functional form, and the experiment
shows that this misspecification is second-order for action ranking.

The phrase "naive ecological baseline" is therefore technically true only if it
means "simple mechanistic solver with restricted model class." It is misleading
if it implies "weak baseline" or "uninformed baseline."

## Why the General Methods Fail Here

The general methods fail for a mechanism that is now fairly clear:

1. They must learn dynamics from the offline dataset.
2. Their learned dynamics model is approximate.
3. More search does not fix a wrong model.
4. Deeper rollouts make the approximation error compound.
5. The native solvers read public ecological structure directly and therefore
   plan with a much stronger model.

This explains the surprising result:

- natives win in 860/864 paired comparisons,
- the single-Ricker `moor_native` baseline alone beats the best general method
  by 0.819 mean return on recoverable safe cells,
- the general methods beat `moor_native` in only 6/112 recoverable safe cells,
- the gap is largest on Allee/theta/regime families,
- observation noise does not reverse the ordering,
- increasing search budget only closes a small part of the gap.

The original paper motivation was:

> structural/model-form uncertainty should break naive ecological solvers,
> so general offline MBRL should win.

The experiment found the opposite:

> public ecological structure makes native solvers strong,
> and learned general models are the weaker side of the comparison.

The sharpest version of the inversion is:

> A misspecified single-Ricker solver beats general offline MBRL by more on the
> families where its own assumption is false.

## What Claims Are Safe

Safe claims:

- This is a negative result for the original motivation.
- The ecological baselines are benchmark-native discrete ecological solvers.
- The native baselines are not paper-faithful reproductions.
- The native baselines receive strong public ecological information.
- `plus_native` should be described as a form-identifying ecological solver, not
  as a purely naive baseline.
- The negative result survives against the genuinely naive `moor_native`
  single-Ricker baseline.
- The general methods are model-limited in this setting.
- Search-budget increases do not rescue the general methods.

Unsafe claims:

- "PLUS/MOOR were reproduced exactly."
- "The native baselines are weak naive baselines."
- "`plus_native` is a naive structurally uncertain solver."
- "Changing `filter=ricker` gives general methods the native model."
- "The negative result proves general offline MBRL cannot work for ecology."
- "The result is just a tuning failure."

## Potential Improvement / Next Experiment

The most meaningful next experiment is not another broad tuning sweep. It is a
planning-model source ablation:

```text
model.dynamics_source in {learned, ricker, true_family}
```

The key test is to give `refplan`, `bamcts`, and `ogsrl` the same mechanistic
planning model source that the native baselines effectively exploit, while
leaving their planning machinery intact.

This cannot be done by only setting:

```text
filter = ricker
filter = true_family
```

Those settings change state estimation. They do not change the planning model,
because the general methods still fit and plan with their learned
`ContinuousDynamicsEnsemble`.

The model-source ablation would directly test the causal diagnosis:

- If the general methods close the gap with mechanistic planning dynamics, the
  informational-asymmetry diagnosis is confirmed.
- If they still do not close the gap, the explanation shifts toward solver
  structure or policy class, and simply redesigning the public ecological
  information may not recover the original motivation.

After that, trajectory/action-over-time plots remain useful as qualitative
mechanism figures, but they should not be treated as the primary decision
experiment.

## Recommended Wording for the Paper

Use:

> We compare general offline MBRL adaptations against benchmark-native
> ecological baselines implemented as discrete mechanistic planners.

Use:

> The ecological baselines are intentionally simple and structurally restricted,
> but in this benchmark they receive strong public action/species information,
> making them substantially stronger than the word "naive" might suggest.

Avoid:

> We reproduce the original PLUS and MOOR algorithms.

Avoid:

> General offline MBRL fails against naive ecological baselines.

Better:

> In this benchmark, general offline MBRL fails to outperform native
> mechanistic ecological solvers, largely because the benchmark exposes
> enough public ecological structure for those solvers to plan with a
> near-exact model.

Best for the clean naive-baseline comparison:

> A misspecified single-Ricker native solver outperforms general offline MBRL
> even on the non-Ricker dynamics families where its own model class is wrong.
