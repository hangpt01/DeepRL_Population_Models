# Codex Task: Add POMDP Stress Population Models

## Goal

Implement and test three additional stress population models for the existing offline RL population-management benchmark:

1. Critical depensation / Allee
2. Theta-logistic
3. Hidden regime-switching mixture

Do not implement delayed density dependence.

The benchmark should remain a POMDP. Agents should observe only the current abundance bin, while hidden variables such as threshold, curvature, regime, and growth rate remain unobserved.

## Current Setting To Preserve

Keep the existing benchmark structure where possible.

- State observation: current abundance bin only, `x_t`
- Abundance binning: `N = 100`, `w = 10`
- Maximum abundance: `1000`
- Carrying capacity base: `Kbase = 500`
- Episode horizon: `50`
- Actions: existing five management actions
- Methods: `PLUS`, `MOOR-Ricker`, `MOPO`, `RefPlan`

Keep the current action effects:

- `r_eff = r_base + delta_r(action)`
- `K_eff = Kbase + delta_K(action)`

Clip all next abundances to `[0, 1000]`.

## POMDP Observation Rule

The agent observation must remain:

- `observation = x_t`

Do not expose hidden variables to the policy or planner observation.

Hidden variables may be stored internally and used for diagnostics, but not as agent inputs:

- `r_base`
- `C`
- `theta`
- `regime z_t`

## New Environment IDs

Add these environment identifiers:

- `allee_ricker_pomdp`
- `theta_logistic_pomdp`
- `regime_switch_pomdp`

## Model 1: Critical Depensation / Allee

Dynamics:

- `s_next = s * exp(r_eff * (1 - s / K_eff) * (s / C - 1))`

Defaults:

- `r_base ~ Uniform(1.00, 1.30)`
- `C ~ Uniform(125, 175)`

POMDP detail:

- `C` is sampled at episode reset.
- `C` is hidden from the agent.

Purpose:

This creates a hidden collapse threshold. Below `C`, growth becomes negative. Fixed compensatory baselines should wrongly expect recovery where the true environment may collapse.

## Model 2: Theta-Logistic

Dynamics:

- `s_next = s + r_eff * s * (1 - (s / K_eff)^theta)`

Defaults:

- `r_base ~ Uniform(1.60, 2.10)`
- `theta ~ Uniform(3.0, 5.0)`

POMDP detail:

- `theta` is sampled at episode reset.
- `theta` is hidden from the agent.

Purpose:

This keeps scalar abundance dynamics but changes density-dependence curvature. The hidden `theta` value means the same observed abundance-action pair can imply different transition risks across episodes.

## Model 3: Hidden Regime-Switching Mixture

Use a hidden two-regime Allee-style model.

Hidden regime:

- `z_t in {safe, harsh}`
- `P(z_{t+1} = z_t) = 0.90`
- `P(z_{t+1} != z_t) = 0.10`

Regime parameters:

- `safe`: `C = 100`, `growth_multiplier = 1.00`
- `harsh`: `C = 200`, `growth_multiplier = 0.75`

Dynamics:

- `s_next = s * exp(growth_multiplier(z_t) * r_eff * (1 - s / K_eff) * (s / C(z_t) - 1))`

Defaults:

- `r_base ~ Uniform(1.00, 1.30)`
- `z_0 ~ Bernoulli(0.5)`

POMDP detail:

- `z_t` is hidden from the agent.
- The process is Markov in the full hidden state, but partially observed from `x_t`.

Purpose:

This tests hidden environmental regime shifts without using delayed density dependence.

## Reward Modes

Implement two reward modes for every new stress environment.

### base

Use the existing reward unchanged:

- `reward = x_t / 100 - cost(action)`

### collapse_sensitive

Use the existing reward plus collapse handling:

- `reward = x_t / 100 - cost(action)`
- If `x_next <= 5`, subtract `10` and enter an absorbing collapse state.
- Once collapsed, keep the population at bin zero for the rest of the episode.

## Data Collection Policy

Add a shared `mixed_danger_zone` collection policy.

Episode mixture:

- `50%`: uniform random actions
- `30%`: harvest-pressure policy
- `20%`: rescue policy

Harvest-pressure policy:

- Choose action `1` with probability `0.7`
- Otherwise choose uniformly from other actions

Rescue policy:

- If `x_t <= 20`, choose action `3` or `4` equally
- Otherwise choose action `1` with probability `0.5`
- Otherwise choose uniformly

Dataset size:

- `25,000` transitions per scenario

## Baseline Handling

Keep fixed-family baselines intentionally misspecified.

- `MOOR-Ricker` continues fitting first-order compensatory Ricker dynamics.
- `PLUS` uses internal compensatory transition models, not the true stress transition.
- `MOPO` and `RefPlan` learn from the generated offline dataset unchanged.
- Do not give `PLUS` or `MOOR-Ricker` access to hidden `C`, `theta`, or `z_t`.

## Tests

Add environment sanity tests.

Allee:

- Below `C`, no-action growth should decline.
- Above `C`, no-action growth should recover toward `K`.

Theta-logistic:

- `theta != 1` should produce different mid-abundance updates from standard logistic.
- Hidden `theta` should be sampled once and retained for the episode.

Regime-switching:

- Changing hidden `z_t` while holding `s_t` fixed should change `s_next`.
- Regime transition frequency should approximately match the configured `0.90` stay probability.

POMDP checks:

- Agent observation contains only `x_t`.
- Hidden variables are stored internally but not returned as observation.
- Dataset metadata may record hidden variables for diagnostics.
- Training inputs must not include hidden variables unless used only for post-hoc diagnostics.

Dataset coverage checks:

- At least `15%` of transitions have `x <= 20`.
- At least `5%` of transitions have `x <= 10`.
- In `collapse_sensitive` mode, collection collapse rate is between `2%` and `25%`.

## Smoke Benchmark

Run:

- `3 environments x 2 reward modes x 4 methods`
- Dataset seed: `42`
- Evaluation episodes: `10`
- MOPO reduced budget: `num_rollouts = 1`, `horizon = 3`

## Main Benchmark

Run:

- `3 environments x 2 reward modes x 4 methods`
- Dataset seeds: `42`, `52`, `62`
- Evaluation episodes: `50`
- Episode seeds: `42...91`

Use the same disclosed reduced MOPO budget across all scenarios unless vectorized MOPO planning already exists.

## Metrics To Report

Report standard metrics:

- Mean cumulative reward
- Final abundance
- Minimum abundance
- Catastrophic rate using `min bin <= 5`
- Collapse-entry rate for `collapse_sensitive` mode
- Mean epistemic uncertainty
- Mean policy entropy
- Runtime

Report POMDP diagnostics after evaluation only:

- Performance by hidden regime for `regime_switch_pomdp`
- Performance by hidden `C` bucket for `allee_ricker_pomdp`
- Performance by hidden `theta` bucket for `theta_logistic_pomdp`

Do not expose diagnostic hidden variables to agents during training or planning.

## Acceptance Criteria

The implementation is successful when:

- All three environments can generate datasets.
- All four methods run on all `3 x 2` scenarios.
- Observations remain POMDP-style with hidden variables unobserved.
- Results include all required metrics.
- At least one scenario makes fixed-family baselines fail on reward or collapse rate.
- At least one learned method beats both fixed-family baselines on reward or collapse rate in at least `2 of 3` seeds.

## Suggested Implementation Order

1. Locate current Ricker and Schaefer environment implementations.
2. Add the three new POMDP stress environment modes.
3. Add hidden per-episode or per-step variables without exposing them in observations.
4. Add `reward_mode`.
5. Add `mixed_danger_zone` dataset collection.
6. Add dynamics and POMDP sanity tests.
7. Add smoke benchmark config.
8. Run smoke benchmark and inspect coverage and collapse rates.
9. Run full benchmark after smoke passes.