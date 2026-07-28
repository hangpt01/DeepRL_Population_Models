# Real-Ecology Implementation Alignment Guide

Purpose: help an AI coding agent implement the real-ecology setting by mapping
the existing dummy/Tier-2/Tier-3 algorithm code onto the two current real-ecology
specifications:

- `29_6_Real_Ecological_Data_Actions_and_Costs.tex`
- `29_6_Real_Ecology_Setting_Implementation_Plan.tex`

This file is a coding bridge, not a replacement for those TeX files. Use it as
the first implementation checklist after reading the TeX specs.

## 1. Authority Order

When documents disagree, use this priority:

1. `29_6_Real_Ecology_Setting_Implementation_Plan.tex`
2. `29_6_Real_Ecological_Data_Actions_and_Costs.tex`
3. `CHAT_HANDOFF_real_ecology_continuous_pre_discrete.md`
4. Earlier Tier-2/Tier-3 docs and result reports
5. Table-folder README files

Important known mismatch:

- The current table folder in this workspace is `real_ecology_data/`.
- Older notes may still say `revised_cost_action_table/` or
  `revised_cost_action_table_v2/`; those package-local copies are not the source
  of truth in this checkout.
- Implement the loader with a configurable table path and default it to
  `real_ecology_data/`.
- The table README reward sketch is stale if it says reward uses `o`. The current
  real-ecology reward uses true latent `s`.

## 2. What Carries Over From The Previous Dummy Code

Keep the existing architecture where possible:

- continuous latent abundance POMDP
- noisy log-normal observation model
- four dynamics families: Ricker, Allee-Ricker, theta-logistic, regime-switching
- shared particle filter
- shared ridge dynamics ensemble and particle MPC backbone
- existing method wrappers: MOPO-style, RefPlan-inspired, BA-MCTS-inspired,
  MOOR-Ricker, PLUS-style, Delphic-inspired CQL, OGSRL-style
- unified evaluator, seed protocol, logging layout, training/holdout monitoring

Do not rewrite the baselines into full paper-faithful implementations unless the
experiment is explicitly redesigned. The goal here is to port the existing
adapted benchmark to real ecological inputs.

## 3. What Must Change From Dummy/Tier-3 Code

Replace dummy constants and hard-coded action tables with real table values.

Core changes:

- population-specific `N0`, `K_base`, `K_max`, rate caps, and lambda profile
- 11 real actions, not the old 5/10 illustrative tables
- set-point growth rate `r_eff`, not cumulative `rho` growth-rate accumulation
- cumulative habitat capacity `kappa`
- translocation action `a10` acts directly on current state
- per-step real-cost index from `cost_step`
- reward on true `s_{t+1}`, not noisy `o_t`
- two separately trained reward modes: `yield` and `safe`
- reporting split by recoverable vs sink populations

## 4. Data Loader Contract

Load from `real_ecology_data/`:

- `actions.csv`
- `species.csv`
- `species_lambda.csv`
- `action_effects_long.csv`
- `cost_sources.csv`
- `cost_anchors_portal.csv`

Use `action_effects_long.csv` as the environment lookup table keyed by
`(population, action)`. It should provide:

- lambda or lambda source
- `r_setpoint_ricker`
- `r_setpoint_lgm`
- `dK_step`
- `dN`
- `cost_step`
- rate clip bounds

Implementation rule:

- Ricker, Allee-Ricker, and regime-switching use Ricker rates:
  `r = ln(lambda)`.
- Theta-logistic uses LGM rates:
  `r = lambda - 1`.

The loader should fail loudly if a required population/action row is missing.

## 5. Episode Reset Contract

On reset:

- choose one population `p`
- set `s0 = N0(p)`
- set `kappa = 0`
- set `K_base = K_ref = K_base(p)`
- set `K_max = K_max(p)`; current default is `2 * K_base`
- load `r_min`, `r_base`, `r_max` for the selected map family convention
- draw hidden structural variables as before: Allee threshold `C`,
  theta-logistic `theta`, regime state/parameters
- expose population identity and public population constants to the agent by
  default
- keep true `s` and hidden structure private

Do not train nine separate agents by default. Train one pooled agent over all
populations with population identity/context observed. Use recoverable vs sink as
a reporting stratifier, not a default training split.

## 6. Environment Step Contract

For action `a_t` in population `p`:

1. Read the `(p, a_t)` row from `action_effects_long.csv`.
2. Set growth rate as a maintained action regime:
   `r_eff = r_setpoint(p, a_t)`.
3. Assert `r_eff` is within the population's rate caps. The clip is a guard; it
   should not silently change valid table values.
4. Update cumulative capacity:
   `kappa <- kappa + dK_step(p, a_t)`.
5. Set:
   `K_eff = clip(K_base + kappa, K_base, K_max)`.
6. If `a_t == a10`, apply translocation before growth:
   `s <- max(s + dN(p, a10), 0)`.
7. Propagate abundance through the selected dynamics family using
   `(s, r_eff, K_eff)` and hidden family variables.
8. Apply process noise if the previous benchmark code already does so; keep the
   same noise semantics unless the TeX spec says otherwise.
9. Generate observation:
   `o_{t+1} = s_{t+1} * eta`, `eta ~ LogNormal(0, sigma_obs^2)`.
10. Compute reward from true state, not observation.

Negative growth convention:

- If the previous real-ecology implementation uses the split
  `r_plus = max(r_eff, 0)` and `r_minus = min(r_eff, 0)`, keep it.
- This treats negative set-points as unconditional mortality and avoids putting
  negative `r` inside density-dependence terms.
- Mark it as a benchmark convention, not a standard Ricker identity.

## 7. Reward Contract

Use one reward formula with a mode-dependent penalty weight:

```text
R_t =
    alpha * s_{t+1} / (s_{t+1} + K_base(p))
  - cost_step(a_t)
  - P_mode * 1[s_t > s_safe(p) and s_{t+1} <= s_safe(p)]
```

Modes:

- `yield`: `P_mode = 0`
- `safe`: `P_mode = P_safe > 0`

Rules:

- Train a separate agent for each reward mode.
- Do not compare raw returns across reward modes.
- Compare both modes using reward-agnostic metrics.
- `P_safe` is a penalty weight, not a probability.
- Consider the crossing-vs-occupancy choice explicitly. The current TeX formula
  charges only downward crossing; this can under-penalize populations that start
  below the safety floor or remain below it.

Reward leakage guard:

- The simulator may log `R_t`.
- Value-learning targets may use `R_t`.
- The particle filter and policy input must not condition on realized reward.
- Belief updates must use observation/action history, not reward history.

## 8. Public And Private State Boundary

Public / allowed for agent and method inputs:

- noisy observations `o_t`
- actions
- population identity/context
- `N0`, `K_base`, `K_max`
- action lambda profile and rate caps
- public cumulative controls such as `kappa`, `K_eff`, current selected
  `r_eff`
- belief features computed from public filtering

Private / simulator-only:

- true latent abundance `s_t`
- hidden Allee threshold `C`
- hidden theta `theta`
- hidden regime state/parameters
- true model family during method inference, except when evaluating a known-family
  oracle or a deliberately mechanistic baseline

Logged but not policy/filter input:

- realized reward
- true return
- collapse indicators
- true abundance metrics

## 9. Method Adaptation Rules

Do minimal interface migration first:

- Methods should read new belief/context features.
- Methods should read new `cost_step`.
- Methods should plan under set-point `r_eff`, cumulative `K_eff`, and
  translocation.
- Methods should not receive private simulator state.
- Methods should not receive reward as an observation channel.

Naming/framing:

- Use `MOPO-style` if the code is ridge-ensemble plus particle MPC rather than
  neural synthetic-rollout MOPO.
- Use `RefPlan-inspired` if the code uses ensemble posterior weights rather than
  the paper's VAE epistemic encoder.
- Use `BA-MCTS-inspired` or `BAMCP-style planner` if the code does decision-time
  tree search rather than the paper's Continuous-BAMCP policy iteration.
- Use `PLUS-style K-only prior` if the candidate set is 21 capacity candidates
  only.
- Use `Delphic-inspired random-world CQL` unless compatible latent worlds are
  actually trained and validated.
- Use `MOOR-Ricker` for the mechanistic Ricker baseline.

This naming does not block implementation; it prevents overclaiming later.

## 10. Offline Dataset Contract

The offline dataset should contain public transition data and known public
context, for example:

- `o_t`
- `a_t`
- `R_t` as target/logged reward
- `o_{t+1}`
- done flag
- population id
- public `r_eff`, `kappa`, `K_eff`
- reward mode used for collection/training if needed for bookkeeping

Do not include true `s_t`, hidden family parameters, or private hidden state in
method training inputs except for supervised simulator diagnostics that are
explicitly marked private/oracle.

Split train/validation by whole episode, not by transition.

The `~75k` transition buffer is a benchmark convention from prior deep offline RL
experiments, not inherited from PLUS or MOOR.

## 11. Acceptance Tests

Data and loader tests:

- All 9 populations load.
- All 11 actions load.
- Every `(population, action)` pair exists in `action_effects_long.csv`.
- Loader returns the exact `r_setpoint`, `dK_step`, `dN`, and `cost_step` from the
  table.
- Ricker/Allee/regime use Ricker rates; theta uses LGM rates.

Environment tests:

- `a0` leaves `kappa` unchanged and uses baseline `r`.
- `a5` and `a7` add `0.10 * K_base` to `kappa`.
- `a6`, `a8`, and `a9` add `0.30 * K_base` to `kappa`.
- `a10` adds `0.10 * N0` to current state before growth.
- `K_eff` never exceeds `K_max`.
- valid table rates never require silent clipping.
- `s = 0` remains absorbing if that was the previous environment convention.

Reward tests:

- Reward uses `s_{t+1}`, never `o_t`.
- `yield` mode sets `P_mode = 0`.
- `safe` mode sets `P_mode > 0`.
- Benefit equals approximately `alpha / 2` when `s_{t+1} = K_base`.
- Common evaluation metrics are computed independently of reward mode.

Leakage tests:

- Particle filter update does not use reward.
- Policy features do not include reward history.
- Two runs with identical observation/action histories but different logged
  reward values produce identical beliefs and actions.

Training/evaluation tests:

- training and validation are split by episode
- per-action behavior-policy coverage is logged
- training loss or planning diagnostics are logged for every method
- local metric history is saved even if cloud logging is unavailable
- results are stratified by recoverable vs sink populations

## 12. Reporting Contract

Always report:

- reward mode
- observation noise level
- population or population group
- recoverable vs sink status
- collapse probability
- final abundance
- minimum abundance
- fraction of steps below safety floor
- economic cost
- filter error diagnostics
- fallback rate for methods with fallbacks
- runtime

Do not pool sink and recoverable populations blindly. Egyptian vulture and
bottlenose dolphin are demographic sinks under vital-rate actions; only repeated
translocation can add individuals.

## 13. Suggested Implementation Order

1. Add/configure table loader.
2. Replace hard-coded dummy constants with population/action lookup values.
3. Implement reset with population context.
4. Implement set-point `r_eff`.
5. Implement cumulative `K_eff`.
6. Implement translocation `a10`.
7. Implement true-state reward and reward modes.
8. Add reward-leakage guard tests.
9. Update particle filter context and method feature construction.
10. Update shared MPC rollout model.
11. Update method wrappers only as needed for the new interface.
12. Recalibrate behavior policy and `P_safe`.
13. Add training/holdout logging.
14. Run acceptance tests before large experiment sweeps.

## 14. Minimal First Pass

If time is limited, do this first:

- data loader
- real action table lookup
- set-point `r_eff`
- cumulative `K_eff`
- translocation
- true-state reward with `yield` and `safe`
- no-reward-in-policy/filter test
- one smoke run per method

Then do recalibration and full sweeps.
