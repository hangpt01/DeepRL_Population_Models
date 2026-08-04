# Metrics, logging, and result tables

## Episode metrics

All definitions below are in
`src/tracks/general/real_ecology_benchmark/evaluator.py:ContinuousEvaluator.run()`.
Every numeric summary gets mean and sample SD in `summarize()`.

| metric | definition; private truth? | interpretation trap / destination |
|---|---|---|
| `operational_return` | discounted `StepResult.reward`; reward is truth-derived in real cells | headline `operational_return_mean`; episodes/summary/accepted CSV |
| `true_return` | discounted private `reward_true`; yes | identical to operational for real set-point cells |
| `collapse_entry`, `collapse_entries`, `collapse_entry_timestep` | first downward crossing of `s_safe`; yes | `envs.py:step()` latches `_entry_latched` after the first crossing and clears it only on reset. `collapse_entries` is therefore 0/1, identical to `collapse_entry`, not a count of recovery–recollapse cycles; a cell starting unsafe has zero entries. |
| `unsafe_fraction` | fraction of states including reset at/below `s_safe`; yes | denominator is steps+1 |
| `mvp_fraction`, `mvp_breach` | occupancy/any state <= fixed MVP=50; yes | MVP can exceed species K |
| `persistence` | final state above `s_safe`; yes | not merely non-extinction |
| `economic_cost` | undiscounted sum of action costs | negative harvest cost is revenue |
| `min_true_state`, `final_true_state` | extrema/end state; yes | accepted names min population |
| `filter_rmse`, `filter_log_rmse` | belief mean versus truth | evaluates filter, not policy reward |
| `filter_coverage90` | truth inside weighted particle 5–95% interval | particle approximation |
| `filter_ess_fraction` | mean ESS/particles | public filter uniform weights can look ideal |
| `filter_unsafe_brier` | belief unsafe probability versus truth indicator | hidden public filter lacks true threshold |
| `action_entropy` | entropy of per-episode empirical action counts | `action_entropy_mean` is populated only in the 48 ecological rows (24 PLUS, 24 MOOR); all 96 general rows are blank. The same 48/96 asymmetry affects `constant_episode_policy`, PBVI action diagnostics, episode paths and episode hashes, so these columns are not cross-track comparable without external general episode artifacts. |
| `method_uncertainty_mean` | first scalar diagnostic key containing uncertainty/entropy | non-comparable across methods |
| `danger_action_i_fraction` | action i fraction when previous state is `(s_safe,4*s_safe]` | all zero if no danger steps |
| `fallback_count` | caught/invalid actions plus hard-fallback diagnostic | not surfaced in accepted CSV |
| filter/planner seconds | per-episode wall time | machine/load dependent |

`pipeline.run_method()` adds dataset/surrogate/cache/init/fit/evaluation/save/row
seconds and peak RSS to `summary.json`. The manifest runner adds total row time
and RSS.

## Headline artifact and variance

`results/accepted/MATCHED_P10_144_METHOD_CELLS.csv` uses
`operational_return_mean` as the reward result and
`operational_return_sd` for within-cell spread. Each SD is across 20 episodes
under **one fixed dataset and one fitted policy**, not across collection
datasets, fit seeds, or independent replications. It cannot estimate
dataset/fitting variability or support population-level uncertainty claims.
The file contains no confidence intervals or paired hypothesis tests.

For real cells `envs.py:step()` computes `reward=state_reward(s')` then
`reward_true=reward`; evaluator sums both with the same gamma, proving equality.

`manifest.aggregate_summaries()` groups means by backend/regime/reward/model,
then computes block-seed paired differences. With repeated
`--baseline METHOD:FILTER`, a challenger “beats both” when its block-seed mean
exceeds the maximum of the specified baseline means; it bootstraps a rate over
recoverable cells. That generic aggregator is not itself the builder of the
copied 144-row table.

## Artifacts and logging

- `results/accepted`: matched P=10 table and receipt;
- `results/followups/{S2,H12,H14,S6,reward_screen}`: focused analyses;
- `results/diagnostic_replay`: M1–M15 and schema tables;
- `provenance/**_RECEIPT.json`: input/output hashes, gates, coverage and cache
  facts appropriate to each job.

A receipt guarantees only the checks it records; it is not a proof of model
validity. The accepted receipt guarantees matched shared-file hashes, dataset
hash per physical cell, seeds/horizon/discount, coverage and P=10 exclusion of
historical P=5.

There is no TensorBoard integration. `training_monitor.save_training_artifacts()`
writes CSV/JSON and optionally Matplotlib PNG. W&B is genuinely wired in
`_log_wandb()` but accepted configs set `training.wandb=false`; plot is disabled
for general accepted config and enabled in ecological overlays.

Constant actions and a0 are reported by
`scripts/diagnostics/replay/constant_action_sweep.py` and `a0_baseline.py`.
They are reference controls, not learning methods and not registry entries; the
accepted CSV’s blank `constant_episode_policy` fields for most rows reinforce
that separation.
