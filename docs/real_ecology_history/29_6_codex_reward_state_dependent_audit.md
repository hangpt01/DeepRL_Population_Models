# 29_6 Codex Audit: State-Dependent Reward Implementation

> Superseded: the issues in this audit were fixed after Claude hit the session
> limit. See `29_6_codex_reward_state_dependent_fix_rebuttal.md` for the current
> post-fix verification. Claude's later response also correctly reframed two
> severity points: `P_safe` default tuning is an E9 calibration item rather than
> a reward-correctness blocker, and reward-mode output namespacing is an extra
> guard on top of the runner's per-cell output-directory contract.

Date: 2026-07-03

Scope: audit of `discrete_action_cont_obser/real_ecology_cont_obser/` after the state-dependent reward / `reward_mode` implementation, against:

- `discrete_action_cont_obser/docs/29_6_Real_Ecological_Data_Actions_and_Costs.tex`
- `discrete_action_cont_obser/docs/29_6_Real_Ecology_Setting_Implementation_Plan.tex`
- `real_ecology_cont_obser/docs/reward_state_dependent_change_proposal.md`

## Overall Verdict

Mostly implemented, but not fully faithful yet.

The core reward equation is now correct for fresh real-ecology runs, and the tests pass. At the time of this audit, I would still block experiment use until the cache-validation issue was fixed, and I would not trust safe-mode calibration until the default penalty at least passed the simple healthy-episode smoke check.

1. Cached datasets are not keyed or validated by `reward_mode`.
2. `P_safe=5` fails the simple default earliest-crossing threshold; full per-cell
   `P_safe` calibration is deferred to E9 and is not a reward-equation bug.

## Findings

### High: reward-mode cache validation is missing

`_validate_dataset_cell` omits `reward_mode`, so a `safe` dataset can be silently reused for a `yield` run. That is wrong because `dataset.rewards` differ on collapse crossings.

Evidence:

- `src/real_ecology_benchmark/pipeline.py:29` validates only:
  - `kind`
  - `population`
  - `num_actions`
  - `control_mode`
  - `observation_noise_sigma`
  - `process_noise_sigma`
  - `safety_threshold`
  - `K_base`
  - `r_base_low`
  - `r_base_high`

Probe result:

```text
cached_reward_mode safe
requested_reward_mode yield
validator_result PASS_BUT_SHOULD_REJECT
ensure_reused_mode safe
has_crossings 3
mean_reward -0.164005
```

Required fix: add at least `reward_mode` to the cached dataset validator. I would also include `collapse_penalty`, `alpha`, and the action-table hash because they affect rewards.

### Medium footgun: shared default outputs can overwrite safe/yield results

`run_method` writes to:

```text
evaluation.output_dir / method / filter
```

Evidence:

- `src/real_ecology_benchmark/pipeline.py:195`

There is no `reward_mode` in that default path. If the CLI runs `safe` and
`yield` with the same config/output root, one can overwrite the other unless an
external runner makes the output directory mode-specific.

Framing correction: this is not uniquely a `reward_mode` bug. The inherited path
also does not include population, family, or `sigma_obs`; the intended design is
that the manifest runner gives each cell its own `evaluation.output_dir`. Adding
`reward_<mode>` is useful cheap insurance for the new axis, but it does not make
the path fully cell-unique.

### Medium: aggregate still pools raw returns by model

The spec says raw returns should not be compared/pool across reward modes. Most paired comparison logic now includes `reward_mode`, but `model_return_mean` still pools only by model.

Evidence:

- `src/real_ecology_benchmark/manifest.py:96`

Required fix: either remove `model_return_mean` or key it by `reward_mode` and preferably full cell dimensions.

### Calibration caveat: `P_safe` default fails the smoke threshold

The spec requires refitting `P_safe` so one collapse outweighs a healthy episode at the new return scale.

Evidence:

- `src/real_ecology_benchmark/config.py:55` sets `collapse_penalty = 5.0`.
- With default evaluation horizon 50 and discount 0.95, a healthy half-benefit episode is:

```text
collapse_penalty 5.0
default_eval_horizon 50 discount 0.95
discounted_half_benefit_return 9.23055
P_exceeds_healthy_episode False
```

So `P_safe=5` does not satisfy the simple default half-benefit, earliest-crossing
criterion.

Framing correction: this is a default sanity failure, not proof that the reward
equation is wrong. No single fixed `P_safe` can be fully calibrated for every
population, family, horizon, timing of crossing, and cost path. The full per-cell
safe-mode calibration remains the E9 experiment-phase task.

## Reward Correctness Checklist

### PASS: benefit is on true `s_{t+1}` for real cells

Evidence:

- `src/real_ecology_benchmark/envs.py:308`
- `src/real_ecology_benchmark/reward.py:48`

Probe with observation noise `sigma=0.4`:

```text
safe P 5.0 reward 0.433166470007 true 0.433166470007 spec 0.433166470007 next_state 191.046598 obs0 212.45096
yield P 0.0 reward 0.433166470007 true 0.433166470007 spec 0.433166470007 next_state 191.046598 obs0 212.45096
```

### PASS: `K_ref = K_base(p)` fixed

Evidence:

- `src/real_ecology_benchmark/config.py:149`

### PASS: `s_safe = 0.1 * K_base(p)`

Evidence:

- `src/real_ecology_benchmark/config.py:152`
- `src/real_ecology_benchmark/config.py:153`

### PASS: collapse indicator is true-state crossing

Evidence:

- `src/real_ecology_benchmark/envs.py:301`

### PASS with calibration caveat: `yield` forces `P=0`, `safe` uses `P>0`

Evidence:

- `src/real_ecology_benchmark/reward.py:71`

Crossing probe:

```text
safe P 5.0 entered True reward -4.912010560037 true -4.912010560037
yield P 0.0 entered True reward 0.087989439963 true 0.087989439963
yield_minus_safe 5.0
```

### PASS: observation/policy input still uses observations

Evidence:

- `src/real_ecology_benchmark/evaluator.py:96` passes `observation` into `policy.act`.
- True state remains in evaluator/private paths, not policy input.

## Experiment Protocol Checklist

### PARTIAL: separate agent per `reward_mode`

The manifest now has a `reward_mode` axis.

Evidence:

- `src/real_ecology_benchmark/manifest.py:35`

Manifest probe:

```text
rows 4608
header index,reward_mode,population,recoverable,environment,num_actions,sigma_obs,method,filter
first 0,safe,Egyptian vulture,False,ricker,11,0.0,mopo,learned
contains_yield True
```

However, the cache/output issues above still make this unsafe operationally.

### PASS: reward-agnostic evaluator battery exists

Evidence:

- `src/real_ecology_benchmark/evaluator.py:175`

Metrics now include:

- `economic_cost`
- `min_true_state`
- `final_true_state`
- `persistence`
- `collapse_entry`
- `unsafe_fraction`

### PASS: planner/method internal rewards use predicted next state

Evidence:

- `src/real_ecology_benchmark/planning.py:110`
- `src/real_ecology_benchmark/methods/value.py:49`
- `src/real_ecology_benchmark/methods/bamcts.py:89`
- `src/real_ecology_benchmark/methods/ogsrl.py:160`

Tiny all-method sanity run completed for all 7 methods under both modes:

```text
('safe', 'mopo', 1.396033)
('safe', 'refplan', 1.504786)
('safe', 'bamcts', 1.277622)
('safe', 'plus', 1.241353)
('safe', 'moor', 1.15366)
('safe', 'delphic', 0.061409)
('safe', 'ogsrl', -2.770188)
('yield', 'mopo', 1.396033)
('yield', 'refplan', 1.504786)
('yield', 'bamcts', 1.864289)
('yield', 'plus', 1.241353)
('yield', 'moor', 1.15366)
('yield', 'delphic', 0.061409)
('yield', 'ogsrl', 1.302343)
```

## Tests Run

Command:

```bash
cd discrete_action_cont_obser/real_ecology_cont_obser
PYTHONPATH=src python -m unittest discover -s tests -v
```

Result:

```text
test_exploitable_population_reaches_band (test_real_ecology.TestE10CollapseBand) ... ok
test_robust_populations_below_band_is_consistent (test_real_ecology.TestE10CollapseBand) ... ok
test_sinks_reported_separately (test_real_ecology.TestE10CollapseBand) ... ok
test_abundance_term_half_at_Kbase (test_real_ecology.TestE10EnvReproducesData) ... ok
test_capacity_accumulates (test_real_ecology.TestE10EnvReproducesData) ... ok
test_eval_reset_starts_at_N0 (test_real_ecology.TestE10EnvReproducesData) ... ok
test_r_eff_within_caps_and_K_within_Kmax (test_real_ecology.TestE10EnvReproducesData) ... ok
test_setpoint_dk_cost_reproduce_table (test_real_ecology.TestE10EnvReproducesData) ... ok
test_translocation_adds_dN_before_growth (test_real_ecology.TestE10EnvReproducesData) ... ok
test_public_schema_guard (test_real_ecology.TestE10Leakage) ... ok
test_battery_agnostic_to_reward_mode (test_real_ecology.TestRewardStateDependent) ... ok
test_benefit_half_at_capacity (test_real_ecology.TestRewardStateDependent) ... ok
test_reward_mode_penalty_switch (test_real_ecology.TestRewardStateDependent) ... ok
test_reward_tracks_next_state_not_observation (test_real_ecology.TestRewardStateDependent) ... ok

----------------------------------------------------------------------
Ran 14 tests in 2.901s

OK
```

Smoke also passed:

```text
"status": "ok"
"reward_mode": "safe"
"operational_return_mean": 2.3199519649330393
"true_return_mean": 2.3199519649330393
```

## Report Accuracy

Mostly accurate:

- The report status says implemented.
- 14 tests pass.
- A+B code paths are present.
- The proposed reward/planner changes are implemented.

Overstated or incomplete:

- "reward_mode never pooled" is false for `model_return_mean`.
- The report does not mention the reward-mode cache validation hole.
- The report does not prove `P_safe` calibration.

Existing package hygiene issue still present:

- `real_ecology_cont_obser/data/` is ignored by repo-level `.gitignore`.
- `git check-ignore` still reports:

```text
discrete_action_cont_obser/.gitignore:10:data/ discrete_action_cont_obser/real_ecology_cont_obser/data/actions.csv
discrete_action_cont_obser/.gitignore:10:data/ discrete_action_cont_obser/real_ecology_cont_obser/data/species.csv
```

This is not part of the reward delta, but it is still a reproducibility trap.
