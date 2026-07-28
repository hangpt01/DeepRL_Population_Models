# Limitations Tracker

This living tracker records aggregate limitations. Per-method implementation
details and claim boundaries live in `04_algorithm_adaptations_and_claims.tex`.

## Model-Class Limitation

The algorithms are benchmark-native linear/mechanistic adaptations. The result
tables compare strategy families under a shared ecological model class; they do
not benchmark the published deep-RL architectures. This is the main claims
boundary for external readers.

Status: open by design. It should be stated in every paper-facing summary.

## Real-Data Limitation

The real ecology tables combine population/action growth-rate set-points with a
relative cost index. They are transparent and auditable, but they are not a full
empirical transition model for each species. Cost scaling preserves ordering and
spacing rather than dollars.

Status: open. Strengthen with additional species-specific validation if the
paper claims empirical ecological accuracy beyond benchmark construction.

## Robustness Limitation

Most real populations are robust to collapse under the current setting. The Amur
tiger is the main population reaching the collapse band, while several
recoverable populations remain far from collapse. This limits claims about broad
collapse-sensitive behavior in the current real-data grid.

Status: open. Future work could add calibrated stress variants or report
collapse-sensitive subsets explicitly.

## Sink-Population Limitation

Egyptian vulture and bottlenose dolphin are demographic sinks under the Ricker
conversion. They are excluded from the headline recoverable-population collapse
metric and reported separately. This is methodologically justified but must stay
visible.

Status: open. Future work could add a separate sink-rescue analysis focused on
translocation and action preference.

## Filter-Ablation Limitation

The P5 learned-vs-raw ablation found no material control-return advantage for the
learned filter. This does not prove the learned filter is useless for latent
state estimation, but it blocks a headline control-return claim about filtering.

Status: open. Add state-estimation-specific analysis if the filter becomes a
paper claim.

## Synthetic Scope Limitation

The current results document focuses on the real-ecology P-safe run. Synthetic
stress-benchmark reports remain archived as provenance and are not integrated
into the current headline.

Status: open. Decide whether the manuscript needs a short synthetic-results
appendix.

Verified against: docs/benchmark/04_algorithm_adaptations_and_claims.tex, real_ecology_runs/psafe_overnight_20260705/analysis/PAPER_RESULT_PACKAGE.md, real_ecology_runs/psafe_overnight_20260705/analysis/RESULTS_SUMMARY.md, real_ecology_runs/psafe_overnight_20260705/analysis/learned_vs_raw_p5.md, src/real_ecology_benchmark/realdata.py @ be96c36
