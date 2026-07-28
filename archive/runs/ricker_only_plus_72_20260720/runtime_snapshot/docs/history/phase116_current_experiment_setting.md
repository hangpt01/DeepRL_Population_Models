# Phase 116 Experiment Setting: Decision-Relevant Hidden Misspecification

## Summary

This experiment evaluates whether learned offline decision methods can outperform
fixed-family ecological baselines when the true population dynamics contain
hidden structure that is not representable by the baselines' assumed Ricker
family.

The benchmark is a partially observed adaptive-management problem. The manager
observes only the current abundance bin, while the true environment may contain
a hidden Allee threshold, a hidden density-dependence curvature parameter, or a
hidden environmental regime. The goal is to test decision-making under
model-family misspecification, not under fully observed parameter uncertainty.

## Main Hypotheses

Primary hypothesis:

- A learned method should beat both fixed-family ecological baselines when the
  true dynamics are decision-relevantly misspecified and when action choices can
  move the population across hidden danger thresholds.

Secondary hypothesis:

- RefPlan may outperform MOPO in settings where recent public history contains
  information about the hidden threshold or hidden regime, because RefPlan
  explicitly conditions planning on public transition history.

The current run is a reduced tuning run for learned methods. It is used to
select one MOPO configuration and one RefPlan configuration before final
evaluation. The final benchmark should be run separately on held-out seeds.

## Compared Method Families

The full planned comparison uses four methods:

| Method | Role | Model assumption |
| --- | --- | --- |
| MOPO | learned offline model-based RL | learns dynamics from offline data |
| RefPlan | learned history-conditioned planning | learns tabular latent transition ensemble and updates posterior from public history |
| BioConserv18 PLUS | fixed-family ecology baseline | compensatory/Ricker-style internal models |
| ExpertSys23 MOOR-Ricker | fixed-family ecology baseline | fits misspecified Ricker dynamics |

The current tuning run includes only MOPO and RefPlan. PLUS and MOOR-Ricker are
reserved for the frozen final comparison.

## Observation Model

The problem is a POMDP. The agent observes only the discretized abundance:

```text
o_t = x_t
```

where `x_t` is an abundance bin. Hidden variables are never included in the
agent observation or training inputs.

Hidden variables:

| Environment | Hidden variable |
| --- | --- |
| Allee-Ricker | Allee threshold `C` |
| Theta-logistic | curvature parameter `theta` |
| Regime-switch | regime `z_t in {safe, harsh}` |
| All environments | per-episode base growth rate `r_base` |

Hidden variables are used only by the true simulator and by post-hoc diagnostic
summaries.

## State Discretization

| Quantity | Value |
| --- | ---: |
| Number of abundance bins | 100 |
| Bin width | 10 individuals |
| Maximum abundance | 1000 |
| Base carrying capacity | 500 |

The continuous abundance `s_t` is converted to a discrete bin `x_t`. Actions are
chosen from the discrete action set, and transitions are clipped to the valid
abundance range `[0, 1000]`.

## Reward

The primary reward is collapse-sensitive:

```text
R(x_t, a_t, x_{t+1}) = x_t / 100 - cost(a_t) - collapse_penalty
```

where the collapse penalty is applied when the next bin enters the collapse
region:

```text
x_{t+1} <= 5
```

Current collapse-sensitive settings:

| Quantity | Value |
| --- | ---: |
| collapse threshold | bin `<= 5` |
| collapse penalty | 20 |
| post-collapse state | absorbing bin zero |

A base-reward ablation is planned for the final benchmark:

```text
R_base(x_t, a_t) = x_t / 100 - cost(a_t)
```

## Stress Environments

### Allee-Ricker POMDP

The Allee-Ricker environment introduces a hidden critical threshold. Below the
threshold, population growth can become negative even when a compensatory Ricker
model would predict recovery.

Dynamics:

```text
s_{t+1} = s_t * exp(r_eff(a_t) * (1 - s_t / K_eff(a_t)) * (s_t / C - 1))
```

Parameters:

| Parameter | Value |
| --- | --- |
| `r_base` | Uniform `[0.12, 0.30]` |
| hidden threshold `C` | Uniform `[90, 150]` |
| base carrying capacity `K_base` | 500 |
| dataset initial abundance | Uniform `[80, 280]` |
| evaluation initial abundance | Uniform `[80, 250]` |
| dataset episode length | 25 |
| evaluation horizon | 50 |

Scientific purpose:

- Tests whether methods avoid hidden-threshold collapse.
- Fixed-family Ricker baselines are intentionally misspecified because they do
  not represent critical depensation.

### Hidden Regime-Switch POMDP

The regime-switching environment uses an unobserved two-regime Allee-like
process. The same observed abundance may imply different risks depending on
whether the hidden regime is safe or harsh.

Dynamics:

```text
s_{t+1} = s_t * exp(g(z_t) * r_eff(a_t)
                    * (1 - s_t / K_eff(a_t))
                    * (s_t / C(z_t) - 1))
```

Hidden regimes:

| Regime | threshold `C` | growth multiplier `g(z)` |
| --- | ---: | ---: |
| safe | 90 | 1.00 |
| harsh | 180 | 0.65 |

Regime transition:

```text
P(z_{t+1} = z_t) = 0.90
```

Other parameters:

| Parameter | Value |
| --- | --- |
| `r_base` | Uniform `[0.12, 0.30]` |
| base carrying capacity `K_base` | 500 |
| dataset initial abundance | Uniform `[80, 280]` |
| evaluation initial abundance | Uniform `[80, 250]` |
| dataset episode length | 30 |
| evaluation horizon | 50 |

Scientific purpose:

- Tests whether methods can act safely when environmental risk changes through
  an unobserved persistent regime.

### Theta-Logistic POMDP

The theta-logistic environment changes the curvature of density dependence
through a hidden parameter. It is a misspecification stress test without a
direct hidden Allee threshold.

Dynamics:

```text
s_{t+1} = s_t + r_eff(a_t) * s_t * (1 - (s_t / K_eff(a_t))^theta)
```

Parameters:

| Parameter | Value |
| --- | --- |
| `r_base` | Uniform `[0.18, 0.40]` |
| hidden curvature `theta` | Uniform `[3, 6]` |
| base carrying capacity `K_base` | 500 |
| dataset initial abundance | Uniform `[60, 200]` |
| evaluation initial abundance | Uniform `[80, 250]` |
| dataset episode length | 20 |
| evaluation horizon | 50 |

Scientific purpose:

- Tests hidden density-dependence curvature.
- Interpreted as a secondary diagnostic environment because it is not primarily
  a hidden-threshold collapse setting.

## Action Sets

Two management action spaces are evaluated.

### Five-Action Setting

| id | Action | `delta_r` | `delta_K` | cost | direct harvest | stocking |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Do nothing | 0.00 | 0 | 0.00 | 0.00 | 0 |
| 1 | Aggressive harvest | -0.02 | 0 | -0.40 | 0.50 | 0 |
| 2 | Predator/disease control | 0.02 | 0 | 0.20 | 0.00 | 60 |
| 3 | Intensive restoration | 0.02 | 200 | 0.30 | 0.00 | 100 |
| 4 | Flagship conservation | 0.04 | 200 | 0.60 | 0.00 | 160 |

### Ten-Action Setting

| id | Action | `delta_r` | `delta_K` | cost | direct harvest | stocking |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Do nothing | 0.00 | 0 | 0.00 | 0.00 | 0 |
| 1 | Sustainable harvest | -0.01 | 0 | -0.20 | 0.25 | 0 |
| 2 | Aggressive harvest | -0.02 | 0 | -0.40 | 0.50 | 0 |
| 3 | Predator/disease control | 0.01 | 0 | 0.20 | 0.00 | 40 |
| 4 | Breeding/recruitment support | 0.02 | 0 | 0.40 | 0.00 | 60 |
| 5 | Moderate restoration | 0.00 | 100 | 0.15 | 0.00 | 80 |
| 6 | Intensive restoration | 0.00 | 200 | 0.30 | 0.00 | 100 |
| 7 | Integrated conservation, light | 0.01 | 100 | 0.35 | 0.00 | 80 |
| 8 | Adaptive conservation trial | 0.01 | 200 | 0.50 | 0.00 | 120 |
| 9 | Flagship conservation programme | 0.02 | 200 | 0.60 | 0.00 | 160 |

The direct harvest and stocking terms are important. They make the action space
decision-relevant near hidden thresholds: some actions push abundance downward,
while restoration/stocking actions can rescue a population from the danger
zone.

## Offline Data Collection

Offline datasets are shared across methods within each environment, action set,
reward mode, and seed.

Current tuning datasets:

| Quantity | Value |
| --- | --- |
| dataset seeds | `1160`, `1161`, `1162` |
| transitions per dataset | 50,000 |
| reward mode | collapse-sensitive |
| collection policy | mixed danger-zone policy |
| number of dataset cells | 6 environment/action cells x 3 seeds = 18 |

The collection policy intentionally visits low-abundance danger regions without
collapsing every episode. It mixes random low starts, harvest-pressure behavior,
rescue/dwell behavior, and threshold-probing behavior.

Average dataset coverage across the three tuning seeds:

| Environment/action setting | live danger `6<=x<=20` | low abundance `x<=20` | zero bin `x=0` | recovery zone `21<=x<=50` | next state `<=5` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Allee, 5 actions | 0.196 | 0.418 | 0.222 | 0.488 | 0.233 |
| Allee, 10 actions | 0.171 | 0.367 | 0.196 | 0.510 | 0.205 |
| Regime-switch, 5 actions | 0.193 | 0.428 | 0.235 | 0.474 | 0.244 |
| Regime-switch, 10 actions | 0.165 | 0.366 | 0.201 | 0.503 | 0.209 |
| Theta-logistic, 5 actions | 0.200 | 0.400 | 0.200 | 0.518 | 0.218 |
| Theta-logistic, 10 actions | 0.187 | 0.341 | 0.153 | 0.546 | 0.168 |

Interpretation:

- The datasets contain substantial low-abundance and danger-zone coverage.
- The zero-bin fraction is nontrivial but not dominant.
- The data are therefore intended to be stressful without being degenerate.

## Decision-Relevance Gate

Before learned-method evaluation, each environment/action setting is checked by
an oracle-vs-misspecified-control gate.

Controllers:

- Oracle MPC: plans using the true hidden environment.
- Ricker MPC: plans using a compensatory Ricker model, while receiving the same
  collapse-sensitive reward.

Hard gate for Allee and regime-switching:

```text
G_oracle - G_Ricker >= 1.0
collapse_rate_Ricker - collapse_rate_oracle >= 0.05
```

Theta-logistic is interpreted diagnostically because it stresses hidden
curvature rather than a direct hidden collapse threshold.

Gate outcomes:

| Environment/action setting | reward gap | collapse gap | interpretation |
| --- | ---: | ---: | --- |
| Allee, 5 actions | 2.491 | 0.10 | passed |
| Allee, 10 actions | 5.865 | 0.25 | passed |
| Regime-switch, 5 actions | 2.873 | 0.10 | passed |
| Regime-switch, 10 actions | 7.054 | 0.30 | passed |
| Theta-logistic, 5 actions | 0.642 | 0.00 | diagnostic pass |
| Theta-logistic, 10 actions | 2.793 | 0.05 | diagnostic pass |

These results indicate that the hidden dynamics are decision-relevant: a
controller with the true hidden model can outperform a misspecified Ricker
controller.

## Current Tuning Protocol

The current run is a reduced tuning protocol designed to finish on roughly a
one-day timescale.

| Quantity | Value |
| --- | --- |
| methods tuned | MOPO and RefPlan |
| environments | 3 stress models |
| action sets | 5-action and 10-action |
| reward mode | collapse-sensitive |
| seeds | `1160`, `1161`, `1162` |
| evaluation episodes | 50 |
| evaluation horizon | 50 |
| configurations per method per scenario | 6 |
| total tuning runs | 216 |

The tuning run selects one MOPO configuration and one RefPlan configuration.
It should not be used as a final statistical comparison against PLUS or
MOOR-Ricker.

### MOPO Tuning Configurations

Fixed MOPO components:

- categorical bootstrapped dynamics ensemble
- ensemble size 5
- frame stack length 2
- bootstrap ratio 0.8
- pessimistic model-based planner

Tuned configurations:

| Label | planning horizon | rollouts | pessimism weight | epochs |
| --- | ---: | ---: | ---: | ---: |
| M1 | 5 | 25 | 1.0 | 50 |
| M2 | 5 | 20 | 0.5 | 50 |
| M3 | 5 | 20 | 1.0 | 100 |
| M4 | 5 | 25 | 0.5 | 50 |
| M5 | 5 | 25 | 2.0 | 50 |
| M6 | 5 | 50 | 0.5 | 100 |

### RefPlan Tuning Configurations

Fixed RefPlan components:

- tabular transition ensemble
- ensemble size 15
- history-conditioned posterior over latent transition members
- planning from public abundance-action history only

Tuned configurations:

| Label | horizon | sequences | latent samples | kappa | uncertainty penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 | 10 | 512 | 12 | 5.0 | 0.10 |
| R2 | 8 | 256 | 8 | 2.0 | 0.05 |
| R3 | 8 | 256 | 16 | 5.0 | 0.10 |
| R4 | 8 | 512 | 12 | 10.0 | 0.20 |
| R5 | 8 | 1024 | 12 | 5.0 | 0.05 |
| R6 | 10 | 256 | 8 | 10.0 | 0.10 |

## Planned Final Evaluation

After tuning, one MOPO configuration and one RefPlan configuration should be
frozen. The final evaluation should then compare four methods:

```text
MOPO, RefPlan, PLUS, MOOR-Ricker
```

Planned final settings:

| Quantity | Value |
| --- | --- |
| final seeds | `7001`, `7002`, `7003`, `7004`, `7005` |
| reward modes | collapse-sensitive primary; base reward ablation |
| dataset size | 75,000 transitions per scenario |
| evaluation episodes | 100 |
| evaluation horizon | 50 |
| statistical unit | seed-level mean, not individual episodes |

Planned final cells:

```text
6 environment/action cells
x 2 reward modes
x 5 final seeds
x 4 methods
```

The final seeds are disjoint from tuning seeds.

## Metrics

Primary performance metric:

- discounted cumulative reward

Ecological safety and health metrics:

- final abundance
- minimum abundance
- catastrophic low-abundance rate, defined by minimum bin `<= 5`
- collapse-entry rate
- collapse-entry timestep, conditional on collapse

Model and policy diagnostics:

- epistemic uncertainty
- policy entropy
- state coverage
- live danger-zone fraction
- action distribution inside the live danger zone

Hidden-variable diagnostics for final analysis only:

- Allee: performance by hidden threshold bucket
- Regime-switch: performance by majority hidden regime
- Theta-logistic: performance by hidden theta bucket

Runtime diagnostics:

- wall-clock runtime
- GPU type
- peak GPU memory

Final learned-advantage metric:

```text
advantage(method) = G_method - max(G_PLUS, G_MOOR)
```

## Interpretation Rules

The current tuning run can support statements about which MOPO/RefPlan
configuration looked best on calibration seeds. It cannot support final claims
about learned methods beating ecology baselines.

The final benchmark should be interpreted using seed-level statistics:

- Do not treat episodes as independent trained agents.
- Do not over-claim from one seed or one environment.
- A learned-method win should be assessed against the stronger of PLUS and
  MOOR-Ricker in the same environment/action/reward/seed cell.

Possible outcomes:

- If MOPO or RefPlan beats both fixed-family baselines across most
  decision-relevant cells, this supports the hidden-misspecification advantage
  claim.
- If MOPO beats RefPlan consistently, the result suggests learned model-based
  pessimism is more effective than the current tabular RefPlan adaptation.
- If RefPlan improves specifically in Allee or regime-switch settings, this
  supports the value of history-conditioned hidden-state inference.
- If all methods are similar, the benchmark may still be insufficiently
  discriminative despite passing the oracle-vs-Ricker gate.

## Slide Outline

1. Research motivation:
   - fixed-family ecology baselines can fail under hidden model-family
     misspecification.
2. POMDP formulation:
   - abundance bin observed; threshold, curvature, regime hidden.
3. Stress environments:
   - Allee threshold, theta-logistic curvature, hidden regime switching.
4. Management actions:
   - 5-action and 10-action spaces with harvest and restoration authority.
5. Decision-relevance gate:
   - oracle MPC vs misspecified Ricker MPC; all six cells pass.
6. Offline datasets:
   - 50k transitions per tuning cell; danger-zone coverage around 16--20%.
7. Tuning protocol:
   - MOPO vs RefPlan, 6 configurations each, seeds 1160--1162.
8. Planned final comparison:
   - frozen learned methods vs PLUS and MOOR-Ricker on seeds 7001--7005.
9. Metrics:
   - cumulative reward, collapse-entry, abundance health, hidden-bucket
     diagnostics, runtime.
10. Claim discipline:
   - tuning selects configs; final seed-level benchmark supports claims.
