# Recent Real Ecology Results

This file summarizes the frozen P-safe run at
`../../real_ecology_runs/psafe_overnight_20260705/`. The full report is
`../../real_ecology_runs/psafe_overnight_20260705/analysis/REAL_ECOLOGY_EXPERIMENT_RESULTS.pdf`.

## Run Scope

The run is a CPU/NumPy real-ecology experiment over 9 populations, 4 dynamics
families, 4 observation-noise values, 7 methods, safe/yield reward settings where
configured, and filter fronts for learned, raw, and mechanistic baselines. The
analysis table `all_metrics.csv` has 7,488 rows: the four-penalty safe grid plus
the P5 raw-filter ablation and P10 yield rows.

## P-Safe Decision

The safe-mode experiment selected `collapse_penalty = 5`. The decision subset
uses general learners with the learned filter and PLUS/MOOR with the Ricker
filter. P5 was the largest tested penalty that kept the recoverable decision
populations' mean safe return non-negative while reducing collapse entries
relative to P2.

| P | Amur tiger return | Puerto Rican parrot return | Interpretation |
| --- | ---: | ---: | --- |
| 2 | 2.84 | 2.57 | weaker safety pressure |
| 5 | 2.17 | 0.62 | selected average-case lock |
| 10 | 0.85 | -2.29 | parrot already negative |
| 20 | -2.51 | -8.25 | over-penalized |

This is an average-case choice. Individual cells can still be negative at P5.

## Headline Comparison at P5 Safe

Best general learner versus the best ecological baseline on matched
population/family/noise cells:

| Scope | Cells | Best general | Best baseline | Delta | Wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| All populations | 144 | -3.93 | -4.96 | +1.04 | 102/144 |
| Recoverable only | 112 | 7.74 | 6.28 | +1.46 | 96/112 |
| Sinks only | 32 | -44.76 | -44.30 | -0.46 | 6/32 |

OGSRL is the strongest single general method in this run: it beats both
ecological baselines on 79 of 112 recoverable cells, with mean collapse 0.014.

## Robustness and Sink Findings

Most real populations are robust to collapse under the current setting. The Amur
tiger is the main population reaching the collapse band. That is an important
negative result: the benchmark is not a universal collapse generator for all
real populations.

The two demographic sinks, Egyptian vulture and bottlenose dolphin, are excluded
from the headline recoverable-population collapse metric and reported separately.
The exclusion is methodological: their strongest recovery action has
`r_max_ricker <= 0`, so vital-rate actions cannot make them grow in the Ricker
conversion. Only translocation adds individuals directly.

At P5 safe, the sinks are qualitatively different. Egyptian vulture is always
below its relative safety floor and penalty-dominated. Bottlenose dolphin is
mostly above its relative safety floor but below the absolute MVP diagnostic.

## Learned vs Raw Filter Ablation

The P5 raw-filter ablation found no material control-return advantage from the
learned filter over raw observations. Even at `sigma_obs = 0.4`, the mean
learned-minus-raw return difference is about +0.06 over methods/populations/
families. This is a control-return finding, not a state-estimation claim; raw
observations do not provide a latent-state estimate.

## Key Artifacts

- `../../real_ecology_runs/psafe_overnight_20260705/analysis/PAPER_RESULT_PACKAGE.md`
- `../../real_ecology_runs/psafe_overnight_20260705/analysis/RESULTS_SUMMARY.md`
- `../../real_ecology_runs/psafe_overnight_20260705/analysis/DECISION_psafe.md`
- `../../real_ecology_runs/psafe_overnight_20260705/analysis/P5_control_review.md`
- `../../real_ecology_runs/psafe_overnight_20260705/analysis/learned_vs_raw_p5.md`
- `../../real_ecology_runs/psafe_overnight_20260705/analysis/all_metrics.csv`
- `../../real_ecology_runs/psafe_overnight_20260705/analysis/rollup.csv`
- `../../real_ecology_runs/psafe_overnight_20260705/analysis/report_figures/`

Verified against: real_ecology_runs/psafe_overnight_20260705/analysis/PAPER_RESULT_PACKAGE.md, real_ecology_runs/psafe_overnight_20260705/analysis/RESULTS_SUMMARY.md, real_ecology_runs/psafe_overnight_20260705/analysis/DECISION_psafe.md, real_ecology_runs/psafe_overnight_20260705/analysis/P5_control_review.md, real_ecology_runs/psafe_overnight_20260705/analysis/learned_vs_raw_p5.md, real_ecology_runs/psafe_overnight_20260705/analysis/all_metrics.csv, src/real_ecology_benchmark/realdata.py @ be96c36
