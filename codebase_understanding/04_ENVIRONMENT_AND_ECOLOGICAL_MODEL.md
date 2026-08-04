# Environment and ecological model

## Configuration and inputs

`src/tracks/general/real_ecology_benchmark/config.py:EnvironmentConfig` is the
complete simulator configuration. Dynamics fields are `kind`, `N0`, `K_base`,
`K_min/K_max`, `C_low/C_high`, `theta_low/theta_high`,
`regime_persistence`, regime thresholds/multiplier, `r_min/r_max`, cumulative
control decays, and process noise. Observation/safety/reward fields are
`observation_noise_sigma`, `safety_threshold/fraction`,
`safety_penalty_mode`, `mvp_threshold`, `collapse_penalty`, `alpha`, `K_ref`,
`reward_mode`, and `horizon`. Start-state fields are `initial_log_sigma` and
`low_start_probability`; real evaluation defaults make both zero.

`real_environment()` fills population values from `species.csv` via
`realdata.pops_for()` and action effects from `action_effects_long.csv` via
`actions.real_action_table()`. `real_environment_like()` rebuilds a cell while
preserving allowed overrides. `action_effects_long.csv`, not `actions.csv`, is
the executable per-species/action lookup (`realdata.load_effects()`).

## Dynamics equations

In `envs.py:ContinuousEcologyEnv.transition_value()`, let
`x = max(s + dN_a, 0)`, `K' = clip(K_base + kappa', K_min, K_max)`,
`r+ = max(r_eff,0)`, and `r- = min(r_eff,0)`. For set-point cells:

\[
\rho'=(1-\text{decay}_r)\rho+r_a,\quad
\kappa'=(1-\text{decay}_K)\kappa+\Delta K_a,\quad
r_\text{eff}=\operatorname{clip}(\rho',r_{\min},r_{\max}).
\]

`controls.advance_public_controls()` implements these equations; real defaults
`decay_r=1`, `decay_K=0` make rho the current action set point and kappa a
running capacity increment.

The four maps are:

\[
\begin{aligned}
\text{Ricker: }s'&=x\exp[r_+(1-x/K')+\epsilon]\exp(r_-),\\
\text{Allee: }s'&=x\exp[r_+(1-x/K')(x/C-1)+\epsilon]\exp(r_-),\\
\theta\text{-logistic: }s'&=\max(0,x+r_+x(1-(x/K')^\theta)+\epsilon x)\exp(r_-),\\
\text{regime: }s'&=x\exp[m_zr_+(1-x/K')(x/T_z-1)+\epsilon]\exp(r_-).
\end{aligned}
\]

These match the branch code in `transition_value()`. Splitting positive growth
from negative mortality prevents a negative rate times a negative
density-dependent factor from becoming explosive. Consequently, when
`r_eff <= 0`, `r+ = 0` removes the only family-specific density term; all four
maps reduce to managed abundance times `exp(r-)` (theta reaches the same result
after its core), a major validity issue.

`_safe_mul_exp()` computes in log space, raises on float overflow, returns zero
below float-tiny, and preserves zero as absorbing. Regime state flips after a
step when a regime RNG draw exceeds `regime_persistence`
(`envs.py:step()`).

## Actions and parameter conversion

`configs/ecology/actions.csv` defines:

| IDs | channel | lambda source / K multiplier / dN | cost |
|---|---|---|---:|
| a0 | none | a0 / 1.0 / 0 | 0 |
| a1,a2 | rate | a1,a2 / 1.0 / 0 | -0.05, -0.10 |
| a3,a4 | rate | a3,a4 / 1.0 / 0 | 0.25, 0.50 |
| a5,a6 | capacity | a0 / 1.1,1.3 / 0 | 0.1875, 0.50 |
| a7,a8,a9 | rate+capacity | a3,a3,a4 / 1.1,1.3,1.3 / 0 | 0.4375,0.75,1.0 |
| a10 | state | a0 / 1.0 / `0.1*N0` | 0.3125 |

Harvest costs are negative because they represent revenue. Cost provenance is a
separate portal table (`configs/ecology/cost_sources.csv`,
`cost_anchors_portal.csv`) from demographic `species_lambda.csv`; the executable
join is `action_effects_long.csv`. `realdata.py` validates
`r_Ricker=ln(lambda)` for Ricker/Allee/regime and `r_LGM=lambda-1` for theta.
Only a10 changes state directly (`actions.real_action_table()`, `dN`).

## Observation and reward

`observation.py:LogNormalObservationModel.sample()` implements
`o=s*LogNormal(0,sigma)`, so sigma is a log-scale standard deviation, not a
10%/20% additive error; `E[o|s]=s exp(sigma²/2)`. Accepted sigma values are 0.1
and 0.2.

`reward.py:ContinuousReward` exposes three channels:

- `operational(o,a,I)` uses an observation;
- `true(s,a,I)` uses a state;
- `state_reward(s',a,I)` uses true next state.

Real set-point cells call only `state_reward()` and assign that value to both
public `reward` and private `reward_true` in `envs.py:step()`. Thus operational
and true returns are exactly equal. The formula is
`alpha*s'/(s'+K_ref) - cost(a) - P*I`.
`effective_collapse_penalty()` sets P=0 for `yield`, otherwise the configured
P (10 in accepted rows). `safety_penalty_indicator()` uses every below-threshold
next state in `occupancy` mode, or a downward crossing in `crossing` mode; real
defaults use occupancy because a population may start unsafe.

## Accepted population scales

From `species.csv` and `config.default_safety_fraction()`:

| population | N0 | K_ref | K_max | safety fraction / s_safe | MVP |
|---|---:|---:|---:|---:|---:|
| Amur tiger | 200 | 250 | 500 | 0.10 / 25 | 50 |
| Crab-eating fox | 41 | 41 | 82 | 0.25 / 10.25 | 50 |
| Egyptian vulture | 41 | 325 | 650 | 0.25 / 81.25 | 50 |

There is no safe global threshold: tiering depends on K, N0, and N0/K in
`default_safety_fraction()`. Collection forces episode boundaries at 25 steps;
evaluation runs 50 (`collector.collect_dataset()`, `evaluator.run()`). True
extinction `s'=0` terminates; horizon creates truncation. Accepted configs and
manifests leave `process_noise_sigma=0`, so the transition map is deterministic
given the episode parameters and regime path; survey noise and regime switching
remain stochastic. Initial-state behavior differs by phase. Evaluation resets
to exactly `s0=N0` because real environment defaults set
`low_start_probability=0` and `initial_log_sigma=0`
(`envs.py:_sample_initial()`). Collection deliberately bypasses that start:
`collector.collect_dataset()` defaults to `start_low_probability=0.45` and
`start_log_sigma=0.15`, samples with `real_initial_state()`, and passes the
result as `state_override`. The offline dataset must not be described as
starting deterministically at N0.

## Runnable inspection

```bash
PYTHONPATH=src/tracks/general python - <<'PY'
from real_ecology_benchmark.config import real_environment
from real_ecology_benchmark.envs import make_env
from real_ecology_benchmark.actions import resolve_actions
for family in ("ricker","allee","theta","regime"):
    cfg = real_environment("Amur tiger", family, observation_noise_sigma=0.1)
    env = make_env(cfg); reset = env.reset(7001)
    print(family, cfg.K_ref, cfg.safety_threshold, (cfg.r_min,cfg.r_max))
    print([(a.id,a.delta_r,a.delta_K,a.stocking_delta,a.cost)
           for a in resolve_actions(cfg)])
    print(env.transition_value(env.state, env.actions[0], env._r_base,
          env._C, env._theta, env._regime, 0, env._rho, env._kappa))
PY
```
