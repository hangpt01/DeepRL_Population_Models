# P5 control review

Scope: `collapse_penalty=5`, `reward_mode=safe`, CPU/numpy results. General learners use `filter=learned`; ecological baselines are `PLUS-ricker` and `MOOR-ricker`.


This table answers whether general learners beat the two ecological baselines on matched population/family/noise cells. Higher `true_return_mean` is better.


## Family check

| Scope | Cells | Best general mean | Best baseline mean | Delta | Wins vs best baseline |
|---|---:|---:|---:|---:|---:|
| All 9 populations | 144 | -3.929 | -4.964 | +1.035 | 102/144 (70.8%) |
| Recoverable only | 112 | 7.736 | 6.275 | +1.461 | 96/112 (85.7%) |
| Sinks only | 32 | -44.757 | -44.301 | -0.456 | 6/32 (18.8%) |

## Single-method check

| Method | Mean return | Delta vs PLUS | Delta vs MOOR | Beats both | Recoverable beats both | Mean unsafe | Mean collapse |
|---|---:|---:|---:|---:|---:|---:|---:|
| ogsrl | -5.064 | +0.163 | +0.259 | 81/144 (56.2%) | 79/112 (70.5%) | 0.113 | 0.014 |
| mopo | -5.451 | -0.224 | -0.128 | 34/144 (23.6%) | 34/112 (30.4%) | 0.121 | 0.036 |
| bamcts | -5.693 | -0.466 | -0.370 | 68/144 (47.2%) | 64/112 (57.1%) | 0.126 | 0.047 |
| refplan | -5.308 | -0.081 | +0.015 | 38/144 (26.4%) | 37/112 (33.0%) | 0.121 | 0.034 |
| delphic | -9.765 | -4.538 | -4.442 | 33/144 (22.9%) | 33/112 (29.5%) | 0.185 | 0.188 |

## Population check

| Population | Best general mean | Best baseline mean | Delta | Wins |
|---|---:|---:|---:|---:|
| Amur tiger | 4.495 | 3.421 | +1.074 | 15/16 |
| Asian elephant | 7.390 | 6.750 | +0.640 | 14/16 |
| Bottlenose dolphin | 2.646 | 3.286 | -0.639 | 6/16 |
| Crab-eating fox | 10.320 | 9.850 | +0.470 | 16/16 |
| Egyptian vulture | -92.160 | -91.887 | -0.273 | 0/16 |
| Iberian lynx | 8.232 | 8.071 | +0.161 | 10/16 |
| Jaguar | 10.034 | 9.569 | +0.465 | 14/16 |
| Puerto Rican parrot | 5.551 | -1.308 | +6.858 | 16/16 |
| Spotted turtle | 8.130 | 7.573 | +0.558 | 11/16 |

## Learned-vs-raw control ablation at P5

The P5 raw-filter ablation is kept separate from the P_safe decision. It shows a clean null control result: learned filtering does not materially improve return or unsafe fraction versus raw observations.

| Method | learned true_return | raw true_return | Delta learned-raw | unsafe learned/raw |
|---|---:|---:|---:|---:|
| bamcts | -5.69 | -5.37 | -0.32 | 0.126/0.126 |
| delphic | -9.76 | -9.10 | -0.67 | 0.185/0.199 |
| moor | -4.50 | -4.69 | +0.18 | 0.112/0.115 |
| mopo | -5.45 | -5.42 | -0.03 | 0.121/0.122 |
| ogsrl | -5.06 | -4.49 | -0.58 | 0.113/0.112 |
| plus | -5.23 | -5.73 | +0.50 | 0.129/0.136 |
| refplan | -5.31 | -5.29 | -0.02 | 0.121/0.121 |

Interpretation: general learners, especially OGSRL, beat the ecological baselines on recoverable cells. The two sink populations remain a separate harder regime. Learned filtering does not buy control-return gains here; that does not test state-estimation accuracy because raw observations have no latent-state estimate.

