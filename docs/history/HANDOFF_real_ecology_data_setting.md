# Handoff: Real-Ecology Continuous-Observation Benchmark

Date: 2026-07-05

Audience: a fresh Codex/Claude chat that will continue implementing and cleaning
the real-ecology benchmark from the new real-ecology tex specifications.

This replaces the old handoff. Treat this file as the current authoritative
handoff for the real-ecology setting. The current setting is already implemented
as a self-contained subpackage:

`discrete_action_cont_obser/real_ecology_cont_obser/`

## Scope Rule

Modify only files under:

`discrete_action_cont_obser/`

Do not edit `claude_build/`, old discrete code, or parent-level files unless the
user explicitly approves it.

Prefer edits inside:

`discrete_action_cont_obser/real_ecology_cont_obser/`

The parent-level docs under `discrete_action_cont_obser/docs/` are source specs
and handoff material; edit them only when the user asks for a doc/handoff update.

## Source Of Truth

Read these first, in this order:

1. `discrete_action_cont_obser/docs/29_6_Real_Ecological_Data_Actions_and_Costs.tex`
2. `discrete_action_cont_obser/docs/29_6_Real_Ecology_Setting_Implementation_Plan.tex`
3. `discrete_action_cont_obser/real_ecology_cont_obser/docs/29_6_algorithm_method_notes.tex`
4. `discrete_action_cont_obser/real_ecology_cont_obser/docs/29_6_algorithm_audit_reasonable_todo.md`
5. `discrete_action_cont_obser/real_ecology_cont_obser/docs/29_6_algorithm_audit_unreasonable_points.md`

The two `.tex` specs are the mathematical source of truth. The report files are
not source of truth, but they record the latest audit convergence.

## Current Package State

The real-ecology package exists under:

`discrete_action_cont_obser/real_ecology_cont_obser/`

Important files:

- `src/real_ecology_benchmark/realdata.py`: loads the real ecology CSV tables.
- `src/real_ecology_benchmark/actions.py`: resolves the 11 real actions.
- `src/real_ecology_benchmark/config.py`: real environment config, `reward_mode`,
  compute backend config, training monitor config.
- `src/real_ecology_benchmark/envs.py`: real set-point dynamics.
- `src/real_ecology_benchmark/reward.py`: state-dependent reward.
- `src/real_ecology_benchmark/beliefs.py`: particle filter and vectorized real
  transition helper.
- `src/real_ecology_benchmark/methods/`: seven methods.
- `src/real_ecology_benchmark/pipeline.py`: dataset/filter/method/evaluator flow.
- `src/real_ecology_benchmark/training_monitor.py`: local and optional W&B
  training diagnostics.
- `src/real_ecology_benchmark/backend.py`: NumPy/CuPy backend selector and
  backend metadata.
- `scripts/make_real_experiment_manifests.py`: real experiment manifest builder.
- `scripts/slurm/`: Slurm row/gate/aggregate wrappers.

Current authoritative data folder:

`discrete_action_cont_obser/real_ecology_data/`

It contains:

- `actions.csv`
- `species.csv`
- `species_lambda.csv`
- `action_effects_long.csv`
- `cost_sources.csv`
- `cost_anchors_portal.csv`
- `README.md`

The redundant package-local `revised_cost_action_table/` copy has been removed.
Before any new experiment, confirm generated metadata reports
`discrete_action_cont_obser/real_ecology_data/` as `data_table` and rerun the
data/action tests after any table edit.

## Core Spec To Preserve

### Real Actions And Dynamics

The 11-action real setting is:

- `a0`: do nothing.
- `a1`: sustainable harvest, rate set-point from doubled adult mortality.
- `a2`: aggressive harvest, rate set-point from doubled adult + juvenile mortality.
- `a3`: predator/disease control, rate set-point from halved adult mortality.
- `a4`: breeding/recruitment support, rate set-point from halved adult + juvenile
  mortality.
- `a5`: moderate restoration, cumulative `+10% K_base`.
- `a6`: intensive restoration, cumulative `+30% K_base`.
- `a7`: light integrated conservation, `a3 + a5`.
- `a8`: adaptive conservation trial, `a3 + a6`.
- `a9`: flagship conservation, `a4 + a6`.
- `a10`: translocation, direct `+10% N0` before growth.

Per episode, choose one of nine real populations. Population identity is public
in the default setting. Initial state is deterministic:

`s0 = N0(population)`

Growth-rate actions are set-point regimes:

`r_eff_t = clip(r(lambda_{a_t}(population)), r_min(population), r_max(population))`

Capacity is cumulative:

`kappa_t = kappa_{t-1} + DeltaK(a_t)`

`K_eff_t = clip(K_base(population) + kappa_t, K_base(population), K_max(population))`

Translocation is direct:

`s_t <- s_t + DeltaN(a_t)` for `a10` only.

Then:

`s_{t+1} = f_family(s_t; r_eff_t, K_eff_t, hidden family parameters)`

Family-specific rate convention:

- Ricker / Allee / regime use `r = ln(lambda)`.
- Theta-logistic uses `r = lambda - 1`.

Keep the negative-growth numerical convention unless the user explicitly asks to
change the science: `r_pos = max(r_eff, 0)` drives density terms and
`r_mort = min(r_eff, 0)` is applied as unconditional mortality. This prevents
negative-`r` sign explosions in Ricker/Allee maps. Document it as a benchmark
convention, not the standard negative-rate Ricker/Allee map.

### Reward

The updated reward spec is state-dependent:

`R_t^(m) = alpha * s_{t+1}/(s_{t+1} + K_ref(population)) - cost(a_t) - P_m * I[crossing into unsafe]`

where:

- Benefit uses true `s_{t+1}`, never noisy observation `o_t`.
- `K_ref(population) = K_base(population)` and is fixed per episode. It must not
  track `K_eff`.
- `s_safe(population)` is per population. Do not use the old absolute floor
  `50`.
- Collapse/crossing indicator is:
  `I[s_t > s_safe(population) and s_{t+1} <= s_safe(population)]`
  unless the user explicitly chooses a below-safe occupancy penalty.
- `reward_mode="yield"` means `P=0`.
- `reward_mode="safe"` means `P=P_safe>0`.
- The two modes share identical benefit and cost. They differ only in `P`.
- A separate agent is trained per `reward_mode`.
- Raw returns across reward modes are not directly comparable. Use the
  reward-agnostic evaluation battery for comparison.

Critical leakage rule:

Because in yield mode `R_t + cost(a_t)` is a near-invertible function of true
`s_{t+1}`, realized reward must not be a belief-filter or online policy input.
It is a logged training/evaluation target only.

The belief/policy history is:

`(o_0, a_0, o_1, ..., a_{t-1}, o_t)` plus public population/control context.

It is not:

`(o_0, a_0, R_0, o_1, ..., R_{t-1}, o_t)`.

Offline Bellman targets may use `dataset.rewards`; that is not itself leakage.
The leak is using realized reward as online observation/belief/policy input.

### Evaluation Protocol

Evaluate every trained policy on a reward-agnostic battery:

- persistence / final abundance,
- collapse probability,
- minimum abundance,
- fraction of steps with `s <= s_safe`,
- economic cost,
- operational return under the training reward mode,
- true-return diagnostics,
- filter error and coverage,
- runtime and memory telemetry.

Cross reward mode with:

- recoverable vs sink populations,
- family / Allee stress setting,
- `sigma_obs` sweep.

Headline should be on the collapse-aware `safe` mode, with `yield` reported as
the baseline-style objective.

## Existing Audits And Converged Decisions

The latest algorithm-note audit is:

`real_ecology_cont_obser/docs/AUDIT_algorithm_paper_implementation.md`

The accepted/disputed split is recorded in:

- `real_ecology_cont_obser/docs/29_6_algorithm_audit_reasonable_todo.md`
- `real_ecology_cont_obser/docs/29_6_algorithm_audit_unreasonable_points.md`

Convergence:

- Do not put a code-verification checklist into `29_6_algorithm_method_notes.tex`.
- The reward-in-history issue is a documentation bug and leak risk, not proven
  live filter leakage. Fix the note and add a regression test.
- Full paper-faithful MOPO/RefPlan/BA-MCTS/Delphic/PLUS implementations are not
  required. These are benchmark adaptations and should be named honestly.
- `reward_mode` should not be an OGSRL support-feature by default.
- PLUS does not need broader paper axes unless the user wants a stronger new
  baseline; fix wording to say K-only reduced prior.
- MOPO/RefPlan deviations are framing notes, not correctness blockers.

## Immediate Implementation Tasks For The Next Chat

### Task A: Update The Algorithm Note

File:

`real_ecology_cont_obser/docs/29_6_algorithm_method_notes.tex`

Make it match the accepted audit:

1. Remove reward from public belief/history notation.
2. Add early framing that the seven methods are audit-friendly benchmark
   adaptations, not source-code-faithful reproductions.
3. Soften method names in prose:
   - `MOPO-style`
   - `RefPlan-inspired`
   - `BA-MCTS-inspired online Bayes tree search`
   - `MOOR-Ricker expert baseline`
   - `PLUS-style K-only Ricker prior`
   - `Delphic-inspired random-world CQL`
   - `OGSRL-style guarded offline policy`
4. For each method, add a compact "main deviations from the paper" paragraph.
5. Fix shared planner wording: finite-action, continuous-state, particle-belief
   sequence sampler, not a continuous-action engine.
6. Document zero-abundance handling for log-residual posterior updates:
   epsilon floors for RefPlan/BA-MCTS and point-mass zero observation likelihood
   where applicable.
7. State the theta rate convention next to the action equations.
8. Mark the `r_pos/r_mort` split as the benchmark's exploitation-mortality
   convention.
9. Fix PLUS wording: it preserves a 21-count grid only as a reduced K-only prior,
   not the paper's broader uncertainty structure.
10. Replace "compatible worlds" language in Delphic unless compatibility is
    actually trained/validated. Use "random latent worlds" or
    "Delphic-inspired worlds".
11. Make the Delphic target equation match code or change code and tests to
    match the equation. Current code subtracts the uncertainty column for the
    observed action in the Bellman target.
12. Clarify BA-MCTS has zero/depth leaf value unless a fitted leaf is added.
13. Distinguish OGSRL training-time discounted budgets from deployment one-step
    feasibility thresholds.
14. Clarify whether MOOR uses the shared learned filter front-end in reported
    rows.

Compile after edits:

```bash
cd discrete_action_cont_obser/real_ecology_cont_obser
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp docs/29_6_algorithm_method_notes.tex
```

### Task B: Add Reward Non-Leak Regression Tests

Add tests proving:

1. `reward_mode="yield"` and `reward_mode="safe"` use identical benefit/cost and
   differ only in collapse penalty.
2. Belief/filter updates do not consume `result.reward`.
3. Online policy posterior updates do not consume `result.reward`.
4. Two online rollouts or policy-observe calls with identical
   `(observation, action, public_info)` but different rewards produce identical
   belief/posterior/action state for methods where `observe` is implemented.
5. Offline Bellman-target use of `dataset.rewards` remains allowed and should
   not be falsely flagged as leakage.

Likely test files:

- `real_ecology_cont_obser/tests/test_reward_state_dependent.py`
- `real_ecology_cont_obser/tests/test_real_ecology.py`
- or a new focused `test_reward_leakage_guard.py`.

### Task C: Expand OGSRL Guardian Features

Current issue:

The KNN guardian embeds only state/action and hard-codes a scale like `500`.
That is too weak for the real benchmark.

Improve it so the support guard uses public context closer to the policy's
feature space:

- `log1p(s / K_ref)`,
- unsafe indicator or safety distance,
- action id / one-hot action,
- population-scaled public controls such as `rho`, `kappa/K_ref`, `K_eff/K_ref`,
- uncertainty/safety features when available from belief features,
- no `reward_mode` by default.

Do not use private true state labels, hidden family parameters, hidden noise, or
future observations.

Update the OGSRL section of `29_6_algorithm_method_notes.tex` after code changes.

### Task D: Validate Data Table Source Before Running Experiments

Before any run:

1. Use `discrete_action_cont_obser/real_ecology_data/` as the source of truth.
2. Confirm generated metadata records that folder as `data_table`.
3. Regenerate datasets/caches after any data-table change. Existing Slurm
   outputs may be stale.

### Task E: Run Tests And Smoke

From:

`discrete_action_cont_obser/real_ecology_cont_obser/`

Run:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
```

If GPU/CuPy is touched, also test both backend paths:

```bash
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml --backend numpy
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml --backend cupy --backend-strict false
```

Use strict CuPy only on a node where CuPy and CUDA are confirmed available.

## Experiment/Run Cautions

The user had Slurm jobs and outputs from earlier data/code states. Treat all
existing experiment outputs as potentially stale if:

- the real action/cost table changed,
- reward semantics changed,
- the algorithm implementation changed,
- backend selection changed,
- or training-monitor/output-path code changed.

Do not read rankings from stale outputs. Regenerate data, gates, manifests, and
summaries after source-table or reward changes.

Backend fairness rule:

- A ranking run should use one declared backend.
- Every summary must record requested/effective backend.
- CPU and GPU rows should not be pooled silently.

PLUS runtime note:

- PLUS used to be very slow.
- Recent vectorization reportedly made PLUS much faster on CPU.
- Do not assume GPU is necessary or faster until a timing probe confirms it.
- Do not reduce PLUS's 21 candidates unless the user explicitly approves,
  because 21 is intentionally preserved as a baseline count.

## Suggested Next-Chat Prompt

Paste this to the next chat:

```text
We are in /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models.

Scope: modify only inside discrete_action_cont_obser/. The active package is
discrete_action_cont_obser/real_ecology_cont_obser/.

First read:
1. discrete_action_cont_obser/docs/HANDOFF_real_ecology_data_setting.md
2. discrete_action_cont_obser/docs/29_6_Real_Ecological_Data_Actions_and_Costs.tex
3. discrete_action_cont_obser/docs/29_6_Real_Ecology_Setting_Implementation_Plan.tex
4. discrete_action_cont_obser/real_ecology_cont_obser/docs/29_6_algorithm_method_notes.tex
5. discrete_action_cont_obser/real_ecology_cont_obser/docs/29_6_algorithm_audit_reasonable_todo.md
6. discrete_action_cont_obser/real_ecology_cont_obser/docs/29_6_algorithm_audit_unreasonable_points.md

Then implement the immediate handoff tasks:
- update the algorithm method note to match the accepted audit,
- add reward non-leak regression tests,
- improve OGSRL's support guardian feature space if it is still the simple
  state/action KNN guard,
- validate the active real ecology data table before any run,
- run unit tests and smoke.

Do not launch experiments unless I explicitly ask.
```
