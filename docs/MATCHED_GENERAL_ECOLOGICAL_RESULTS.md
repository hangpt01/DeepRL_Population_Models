# Matched Phase 2E general-RL and ecological baseline results

## Slide-ready general-RL presentation summary

**Presentation status (2026-07-20):** the report includes the completed safe-mode ecological baseline `plus_adapted_ricker_only_pbvi` alongside the four general methods, plus a separately authorized 16-cell corrected-MOOR slice (`moor_adapted_ricker_misspec_pbvi`). The late ecological trajectory capture (`58399693`) and its aggregator (`58399718`) were cancelled, so ecological comparison is available for frozen episode/cell metrics but not for ecological population/action/reward-over-time curves.

**Post-outcome trust audit:** the complete audit is incorporated in this file under “Why the experiment looks this way.” Corrected MOOR and Ricker-only PLUS execute identical constant policies on their entire 16-cell overlap, so that slice cannot distinguish the algorithms. OGSRL results are outcomes of the deployed actor + guardian + fallback composite; 18.89% of its registered decisions carry the evaluator's conflated fallback flag, concentrated in sink cells. RefPlan and BA-MCTS pass the behavioral audit cleanly; EVD remains usable with its existing “Delphic-motivated, not full Delphic” limitation.

### Experimental scope for the slide

```text
9 species × 4 demographic families × sigma_obs {0.1, 0.2}
× reward {safe, yield} × 4 general methods = 576 method-cells
```

Each method-cell contains 20 paired 50-step evaluation episodes. The methods are RefPlan, OGSRL, BA-MCTS, and ensemble value-disagreement pessimism (EVD, Delphic-motivated).

### Headline general-method results

| Method | Recoverable safe return | Recoverable yield return | Overall collapse rate | Overall unsafe fraction | Overall persistence |
|---|---:|---:|---:|---:|---:|
| RefPlan | 5.082 | 6.201 | 0.130 | 0.163 | 0.825 |
| OGSRL | **6.654** | 6.649 | **0.041** | **0.125** | **0.869** |
| BA-MCTS | 5.332 | **7.322** | 0.141 | 0.182 | 0.778 |
| EVD pessimism | 4.755 | 7.266 | 0.282 | 0.220 | 0.613 |

### Matched safe-mode comparison with the ecological baseline

The primary comparison below uses the **69 completed cells with byte-identical public datasets on both sides**. It excludes the timed-out Iberian lynx/Allee/σ=0.2 ecological cell and the two Crab-eating fox/Theta cells whose ecological dataset hashes differ from the registered general datasets.

| Method | Recoverable safe return (53 cells) | Recoverable collapse | Recoverable unsafe | Recoverable persistence | Sink safe return (16 cells) | Sink unsafe | Sink persistence |
|---|---:|---:|---:|---:|---:|---:|---:|
| RefPlan | 4.913 | 0.008 | 0.001 | 0.999 | -92.318 | 0.500 | 0.500 |
| OGSRL | **6.496** | 0.016 | 0.001 | 0.998 | -92.718 | 0.500 | 0.500 |
| BA-MCTS | 5.133 | **0.001** | **0.000** | **0.999** | -94.172 | 0.500 | 0.500 |
| EVD pessimism | 4.523 | 0.153 | 0.040 | 0.847 | -124.490 | 0.813 | 0.000 |
| Ricker-only PLUS | 5.102 | 0.034 | 0.025 | 0.966 | **-46.247** | 0.500 | 0.500 |

`Ricker-only PLUS` means `plus_adapted_ricker_only_pbvi`; it is not `plus_native`. No MOOR arm and no ecological yield arm are present in this 72-cell experiment. A separate corrected MOOR diagnostic contributes a scoped 16-cell matched comparison below; it must not be treated as a full 9-population result.

### Main presentation findings

1. **The deployed OGSRL actor/guardian/fallback composite gives the highest recoverable safe return, including against Ricker-only PLUS.** BA-MCTS has the lowest realized collapse/unsafe rates on the strict matched subset. OGSRL's return must not be attributed to its trained actor alone because the frozen evaluator records fallback events without separating guardian infeasibility from execution fallbacks.
2. **BA-MCTS gives the highest recoverable yield return.** Its safety outcomes are weaker than OGSRL, especially in vulnerable yield-mode cells.
3. **EVD pessimism is not uniformly conservative in realized ecology.** It is competitive in yield return but has the largest collapse and unsafe rates and the lowest persistence among the four methods.
4. **Sink species must be presented separately.** The pooled sink safe returns are strongly negative, driven especially by Egyptian vulture. A zero collapse-entry rate does not imply safety when a sink begins below the safety boundary.
5. **Method-specific results matter.** Do not summarize these results as a single “best general method”; the leading method changes between safe return, yield return, realized safety, and sink return. Ricker-only PLUS has substantially less-negative sink return, while the deployed OGSRL composite leads recoverable safe return.

### Three validated trajectory examples for slides

The plots below use the same 20 paired episodes and overlay all four general methods. Each has population, categorical action, and signed immediate reward over 50 timesteps.

**Amur tiger — recoverable, Ricker, sigma=0.2, safe**

![Amur tiger general-RL trajectories](../general_only_trajectory_results/figures/cell_05_amur_tiger_ricker_sigma_0p2_safe.png)

**Puerto Rican parrot — recoverable but vulnerable, Ricker, sigma=0.2, safe**

![Puerto Rican parrot general-RL trajectories](../general_only_trajectory_results/figures/cell_21_puerto_rican_parrot_ricker_sigma_0p2_safe.png)

**Egyptian vulture — demographic sink, Ricker, sigma=0.2, safe**

![Egyptian vulture general-RL trajectories](../general_only_trajectory_results/figures/cell_13_egyptian_vulture_ricker_sigma_0p2_safe.png)

All 48 validated figures for the three selected species are in:

```text
analysis/general_only_trajectory_results/figures/
```

The corresponding mergeable per-step values are in `general_only_trajectories_long.csv.gz`; no plot must be digitized to recover its data.

### Claims to avoid on the slide

- Do not claim a full-scope comparison with corrected MOOR, `plus_native`, or ecological yield. Corrected MOOR is included only for its separately completed 16-cell matched slice; the only 9-population ecological result is safe-mode `plus_adapted_ricker_only_pbvi`.
- Do not call EVD pessimism a full Delphic-compatible-world method.
- Do not pool recoverable and sink safe returns without showing the strata separately.
- Do not interpret action IDs as continuous magnitudes; actions are categorical.
- Do not interpret `collapse_entry=0` for a sink as evidence that it stayed safe.

## Purpose and scope

This file reports the completed general-RL experiment and the currently available matched safe-mode ecological baseline. It reports and points to results for:

- RefPlan (`refplan`)
- OGSRL (`ogsrl`)
- BA-MCTS (`bamcts`)
- Ensemble value-disagreement pessimism, Delphic-motivated (`ensemble_value_disagreement_pessimism`)
- Ricker-only adapted PLUS (`plus_adapted_ricker_only_pbvi`)
- Corrected adapted MOOR (`moor_adapted_ricker_misspec_pbvi`), limited to the separately authorized 16-cell matched slice

The 72-cell Ricker-only ecological experiment has no MOOR, `plus_native`, or yield-mode arm. Do not use older MOOR/PLUS outputs from `hidden_rk_comparison_20260716` as substitutes. The corrected paper-aligned MOOR diagnostic is recorded separately below and is included only on its exact 16-cell overlap.

## Corrected MOOR diagnostic: location and current gate

The corrected MOOR method is `moor_adapted_ricker_misspec_pbvi`. Its authoritative outputs are in the separate frozen diagnostic:

```text
/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/adapted_32cell_diagnostic_pending/
```

The per-cell MOOR evaluation artifacts are below `evaluation/` under directories containing `moor_adapted_ricker_misspec_pbvi/`; the fit provenance is below `fit_receipts/` and `fit_cache/`. Do not substitute files from `hidden_rk_comparison_20260716`.

This is a **partial, safe-only diagnostic**, not a full match to the 576-row general-RL experiment. It contains 32 MOOR cells:

```text
2 populations (Amur tiger, Egyptian vulture)
× 4 hidden families (Ricker, Allee, theta-logistic, regime-switching)
× sigma_obs {0, 0.1, 0.2, 0.4}
× safe reward
```

Only 16 cells overlap this report's registered general-RL comparison domain: the same two populations, all four families, and `sigma_obs` in `{0.1, 0.2}`, safe reward only. MOOR is Ricker-form committed, so Ricker is its in-family control and the other three hidden families are out-of-family misspecification tests.

Frozen provenance:

| Object | SHA-256 / identity |
|---|---|
| Runtime digest | `f70ec7122263ec00da0a4950994dfe483ed29c71bc15e477007ca9ae0a6d2333` |
| Fit manifest | `a8f39d83eb70a32be2c42fabecc56ca4af4d3d40edb97f87242934085fd7ee86` |
| Plan manifest | `f0322718a78d485827c39661e2f1ded4cbde1d8bccd4cd053f4d1fe93dec087d` |
| Refreshed return-blind acceptance record | `adapted_32cell_diagnostic_pending/acceptance.json` (`6e03252b7f6c7f13e0a05b60dd4559bf0b62760b4d148611e88b31c136f0781c`) |
| Ordered matched-input digest | 16 `episodes.csv` + 16 `summary.json` SHA-256 records: `84877e760b813777ecd8c4f310cfa3827aba6252b08f981cd9e5291aadd602c5` |

All 32 MOOR fit tasks (`58391585`) and all 32 MOOR plan/evaluation tasks (`58391647`) completed successfully. The refreshed frozen checker verified all 32 MOOR artifact rows with no failed per-row structural check and recorded `return_fields_opened=false`. Nevertheless, its registered **experiment-level decision is `INCOMPLETE`** because the companion cross-family PLUS plan arm timed out and the checker requires whole-array completion. Its Slurm gate also expects parent-index task identifiers, whereas this cluster reports array elements through their individual raw job IDs, so that gate did not recognize the otherwise successful MOOR elements. The original acceptance job `58391658` had additionally failed before evaluation because its wrapper omitted the scripts import path; rerunning the frozen checker with the correct import context produced the acceptance record above.

The return-blind checkpoint at **2026-07-19 16:48:08 AEST** independently recorded MOOR fits `32/32` complete, MOOR plans `32/32` complete, zero failures at that checkpoint, 78.20 allocated core-hours across the running workflow, approximately 400 MiB peak observed RSS, all resource thresholds still clear, and `return_fields_opened=false`. This confirms successful MOOR execution; it is not a substitute for the later experiment-level acceptance decision.

### Scoped MOOR outcome authorization and matched results

The user subsequently authorized a **scoped outcome waiver for the completed MOOR arm**. The experiment-level acceptance remains `INCOMPLETE`; it has not been relabelled or fabricated. Under that authorization, only corrected MOOR outcomes were opened. The extraction used the 16 cells matching the general-RL domain:

```text
Amur tiger and Egyptian vulture
× Ricker, Allee, Theta, and Regime truth
× sigma_obs {0.1, 0.2}
× safe reward = 16 cells = 320 evaluation episodes
```

All 16 MOOR `dataset_sha256` values exactly match the authoritative general-RL dataset registry. Metrics first average the 20 episodes within each cell and then average the eight cells for each population. `Corrected MOOR` means exactly `moor_adapted_ricker_misspec_pbvi`.

| Population/scope | Method | Cells | Mean return | Collapse | Unsafe | MVP breach | Mean min N | Mean N | Mean final N | Persistence | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Amur tiger / recoverable | RefPlan | 8 | 2.256 | 0.000 | 0.000 | 0.000 | 171.13 | 274.32 | 371.90 | 1.000 | 18.988 |
| Amur tiger / recoverable | OGSRL | 8 | 3.607 | 0.000 | 0.000 | 0.000 | 96.57 | 119.03 | 110.02 | 1.000 | 8.124 |
| Amur tiger / recoverable | BA-MCTS | 8 | 0.665 | 0.000 | 0.000 | 0.000 | 199.32 | 364.87 | 441.61 | 1.000 | 26.110 |
| Amur tiger / recoverable | EVD pessimism | 8 | -12.656 | 1.000 | 0.262 | 1.000 | 10.09 | 59.61 | 10.09 | 0.000 | 4.609 |
| Amur tiger / recoverable | Ricker-only PLUS | 8 | 4.234 | 0.000 | 0.000 | 0.000 | 200.00 | 338.14 | 405.68 | 1.000 | 15.625 |
| Amur tiger / recoverable | **Corrected MOOR** | **8** | **4.234** | **0.000** | **0.000** | **0.000** | **200.00** | **338.14** | **405.68** | **1.000** | **15.625** |
| Egyptian vulture / sink | RefPlan | 8 | -187.013 | 0.000 | 1.000 | 1.000 | 10.98 | 19.93 | 19.81 | 0.000 | 11.372 |
| Egyptian vulture / sink | OGSRL | 8 | -188.451 | 0.000 | 1.000 | 1.000 | 40.71 | 42.57 | 42.83 | 0.000 | 16.219 |
| Egyptian vulture / sink | BA-MCTS | 8 | -189.971 | 0.000 | 1.000 | 1.000 | 6.00 | 15.52 | 9.93 | 0.000 | 16.765 |
| Egyptian vulture / sink | EVD pessimism | 8 | -187.816 | 0.000 | 1.000 | 1.000 | 0.45 | 9.23 | 0.45 | 0.000 | 7.143 |
| Egyptian vulture / sink | Ricker-only PLUS | 8 | -94.953 | 0.000 | 1.000 | 1.000 | 0.45 | 9.23 | 0.45 | 0.000 | 9.375 |
| Egyptian vulture / sink | **Corrected MOOR** | **8** | **-94.953** | **0.000** | **1.000** | **1.000** | **0.45** | **9.23** | **0.45** | **0.000** | **9.375** |

The Egyptian-vulture rows illustrate why collapse entry cannot be interpreted alone: the population is already unsafe, so corrected MOOR has collapse entry 0 while unsafe fraction and MVP breach are both 1 and persistence is 0.

#### Corrected MOOR return by hidden family

Each value averages the two registered noise levels for that population/family.

| Hidden family | Amur tiger | Egyptian vulture |
|---|---:|---:|
| Ricker (in-family) | 4.224 | -94.959 |
| Allee (out-of-family) | 4.224 | -94.959 |
| Theta (out-of-family) | 4.264 | -94.935 |
| Regime (out-of-family) | 4.224 | -94.959 |

For these 16 cells, corrected MOOR and Ricker-only PLUS are exactly tied in every cell. A field-by-field check over all 320 paired episodes and 25 registered return, safety, population, cost, and danger-action fields found maximum absolute difference `0`. The trust audit shows why: both methods choose constant `a10` for Amur tiger and constant `a5` for Egyptian vulture. This slice has no power to distinguish the algorithms and must not be used for a MOOR-versus-PLUS ranking.

Paired cell-mean return gaps below are `other method − corrected MOOR`; wins count cells where the other method has higher return.

| Other method | Mean gap | Median gap | Wins / ties / losses |
|---|---:|---:|---:|
| RefPlan | -47.019 | -47.398 | 0 / 0 / 16 |
| OGSRL | -47.063 | -47.436 | 2 / 0 / 14 |
| BA-MCTS | -49.293 | -49.828 | 0 / 0 / 16 |
| EVD pessimism | -54.877 | -56.351 | 0 / 0 / 16 |
| Ricker-only PLUS | 0.000 | 0.000 | 0 / 16 / 0 |

These pooled gaps are dominated by Egyptian vulture and must not replace the population-stratified table. On Amur tiger alone, the mean return gaps versus corrected MOOR are RefPlan -1.978, OGSRL -0.627, BA-MCTS -3.569, EVD -16.890, and Ricker-only PLUS 0.000. Corrected MOOR recorded no danger-zone exposure in these 320 episodes, so every stored danger-action fraction is zero; no "preferred danger action" should be inferred.

Outcome-access record for this section: `scoped_moor_outcomes_opened=true`; original full-experiment acceptance remains `INCOMPLETE`; the acceptance checker itself recorded `return_fields_opened=false` before the later scoped authorization. No cross-family PLUS outcome was opened from `adapted_32cell_diagnostic_pending`, and no result from `hidden_rk_comparison_20260716` was used.

The frozen general-RL experiment contains 576 method rows:

```text
9 populations × 4 demographic families × 2 observation-noise levels
× 2 reward modes × 4 general methods = 576
```

Each method-cell has 20 registered evaluation episodes (5 seed blocks × 4 episodes), each 50 steps long, with evaluation discount `gamma=0.95`. Headline means below use the 144 scientific cells per method as the aggregation unit; each cell value is its 20-episode mean.

## Authoritative files

Package root:

```text
/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1
```

Frozen design and provenance:

```text
manifests/full_general_sigma01_02_576_rows.csv
manifests/registration.json
manifests/ecological_dataset_reuse_registry_144.csv
provenance/
```

Authoritative outcome root:

```text
quarantine/evaluation/
```

Every one of the 576 method rows has the following files below that root:

```text
summary.json
episodes.csv
training_history.csv
training_history.json
offline_beliefs.npz
manifest_row_resolved.json
validity_receipt.json
```

The directory identity is:

```text
regime_hidden/<population>/<family>/sigma_<noise>/data_real/backend_numpy/
regime_hidden/reward_<safe|yield>/<method>/learned/
```

The ecological merging agent should discover `summary.json` or `episodes.csv` recursively rather than reconstructing paths manually.

Frozen hashes:

| Object | SHA-256 |
|---|---|
| 576-row manifest | `1526ce08dcf1b1d148232c075d41dbd78f0cfa2d7119cff9e330f965b6451df4` |
| Registration | `15fd7aa1c3ddc460fd255e5d66518d49244f81cf3dbbd291a5bdfe52b376b37a` |
| Dataset registry | `474d1a65d2e5ec28c741f5b7ac5691be891712e753ee6a9a6590337d62f5f0c4` |
| Frozen code | `f615d363a525422abfb983f0eee7c433b009fff81a4d0ae4925bfdaba0f3aa72` |

The frozen structural checker accepted all 576 rows. The largest recorded task runtime was approximately 971.6 seconds and peak RSS was approximately 169.8 MiB.

## Metric definitions available in the frozen outputs

The episode-level `episodes.csv` files directly contain:

- `operational_return` and `true_return`: 50-step discounted returns with `gamma=0.95`;
- `collapse_entry`: whether the trajectory entered the safety region from above;
- `collapse_entries` and first `collapse_entry_timestep`;
- `unsafe_fraction` and `mvp_fraction`;
- `mvp_breach` and terminal `persistence`;
- `economic_cost`;
- `min_true_state`, `mean_true_state`, and `final_true_state`;
- action entropy and per-action danger-zone fractions;
- filter diagnostics, uncertainty diagnostic, fallback count, filtering time, and planning time.

The danger zone is defined by the pre-action private state:

```text
safety_threshold < N_t <= 4 × safety_threshold
```

`danger_action_0_fraction` through `danger_action_10_fraction` report the action distribution over those decisions. An episode with no danger-zone step stores zeros for every action, so action fractions averaged over all episodes need not sum to one.

The frozen evaluator did **not** store per-timestep trajectories or episode maximum abundance. Therefore:

- mean, minimum, and final abundance are available for all registered episodes;
- maximum abundance and population/reward/action-over-time curves are not part of the frozen results;
- no ecological or mixed-method reporting job should be used to fill those general-RL fields;
- the completed, separately registered **general-only** diagnostic below supplies them for the three selected populations and validates regenerated returns against the frozen episode returns.

No additional job is required to recover return values: both operational and true returns are already present.

### Completed selected-population general-only trajectory diagnostic

A separate general-only diagnostic completed for two recoverable populations (Amur tiger and Puerto Rican parrot) and one sink population (Egyptian vulture). It covers all four families, both registered noise levels, and both reward modes: 48 cells total. It ran only the four general methods in this report.

```text
capture array: 58398861 (0-47%16, m3h/m3h)
validation/aggregation: 58398862 (afterok:58398861)
```

All 48 array rows and the aggregation job completed with exit code zero. Task elapsed times were 18:12--21:33, maximum task RSS was approximately 150 MiB, and aggregation took 6:24. The resulting 192 method-cells reproduce their frozen mean discounted returns with maximum absolute error `1.1368683772161603e-13`.

Output directory:

```text
analysis/general_only_trajectory_results/
```

It contains:

```text
general_only_trajectories_long.csv.gz
general_only_population_extrema.csv
general_only_trajectory_receipt.json
figures/  # 48 figures, each with population, action, and reward over time
raw_npz/  # per-cell arrays for all four general methods
```

Artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `general_only_trajectories_long.csv.gz` | `067a956edcc7136aa71912147311c12e08928b90ebf278323d505e3bbfd01231` |
| `general_only_population_extrema.csv` | `016656ae9da8042ae61d187307adce149d0a5dff80dcda6e3cd1bf904bf38cdd` |
| `general_only_trajectory_receipt.json` | `b0d840cec5f06c325c0423f194a2bc6584ee65ffde36ac6afdbc0bee0a25f0ca` |

The long-form compressed CSV has 192,000 rows and columns suitable for direct later merging:

```text
population, population_scope, environment, sigma_obs, reward_mode, method,
episode, seed, timestep, state_pre, state_post, observation_pre,
observation_post, belief_mean, action, reward_true, reward_public,
danger, unsafe, mvp, terminated
```

There are 48 corresponding `.npz` files and 48 figures. Every figure overlays the four general methods in three panels: mean true population, median categorical action, and mean immediate reward over 50 timesteps. File names encode cell index, population, family, noise, and reward mode.

#### Population extrema from the selected-population diagnostic

Values below aggregate the eight selected cells per population/reward/method (four families × two noise levels). `Mean max N` and `mean min N` first compute each episode's extrema, then average across cells. `Observed max/min N` are the extrema across all selected registered episodes.

| Population | Reward | Method | Mean max N | Observed max N | Mean min N | Observed min N |
|---|---|---|---:|---:|---:|---:|
| Amur tiger | Safe | RefPlan | 387.157 | 500.000 | 171.126 | 111.074 |
| Amur tiger | Safe | OGSRL | 200.024 | 202.664 | 96.569 | 67.970 |
| Amur tiger | Safe | BA-MCTS | 450.004 | 500.000 | 199.322 | 183.500 |
| Amur tiger | Safe | EVD pessimism | 200.139 | 207.792 | 10.092 | 2.512 |
| Amur tiger | Yield | RefPlan | 200.010 | 201.599 | 24.712 | 2.723 |
| Amur tiger | Yield | OGSRL | 200.024 | 202.664 | 38.130 | 24.814 |
| Amur tiger | Yield | BA-MCTS | 200.840 | 220.269 | 20.277 | 0.003 |
| Amur tiger | Yield | EVD pessimism | 200.000 | 200.000 | 16.062 | 0.758 |
| Puerto Rican parrot | Safe | RefPlan | 495.108 | 529.076 | 77.812 | 69.892 |
| Puerto Rican parrot | Safe | OGSRL | 173.504 | 500.006 | 67.836 | 39.645 |
| Puerto Rican parrot | Safe | BA-MCTS | 490.555 | 537.501 | 77.225 | 35.055 |
| Puerto Rican parrot | Safe | EVD pessimism | 275.390 | 500.000 | 77.547 | 66.131 |
| Puerto Rican parrot | Yield | RefPlan | 120.249 | 508.142 | 41.309 | 11.690 |
| Puerto Rican parrot | Yield | OGSRL | 81.314 | 97.626 | 46.104 | 11.480 |
| Puerto Rican parrot | Yield | BA-MCTS | 111.662 | 335.200 | 44.348 | 5.984 |
| Puerto Rican parrot | Yield | EVD pessimism | 179.846 | 248.213 | 46.855 | 4.790 |
| Egyptian vulture | Safe | RefPlan | 41.000 | 41.000 | 10.978 | 2.843 |
| Egyptian vulture | Safe | OGSRL | 43.292 | 44.947 | 40.712 | 34.765 |
| Egyptian vulture | Safe | BA-MCTS | 41.022 | 41.334 | 5.999 | 0.532 |
| Egyptian vulture | Safe | EVD pessimism | 41.000 | 41.000 | 0.453 | 0.429 |
| Egyptian vulture | Yield | RefPlan | 41.000 | 41.000 | 0.442 | 0.032 |
| Egyptian vulture | Yield | OGSRL | 43.304 | 44.947 | 40.721 | 34.765 |
| Egyptian vulture | Yield | BA-MCTS | 41.005 | 41.334 | 0.907 | 0.004 |
| Egyptian vulture | Yield | EVD pessimism | 41.000 | 41.000 | 0.0005 | 0.0003 |

These diagnostics remain separate from the frozen 576-row experiment while providing validated, mergeable general-method trajectories for the ecological agent.

## Overall general-method results

| Method | Mean return, both modes | Safe return | Yield return | Collapse-entry rate | Unsafe fraction | MVP-breach rate | Mean episode minimum N | Smallest observed N | Mean final N | Persistence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RefPlan | -5.673 | -16.562 | 5.215 | 0.130 | 0.163 | 0.537 | 78.09 | 0.0317 | 195.87 | 0.825 |
| OGSRL | -5.185 | -15.429 | 5.059 | 0.041 | 0.125 | 0.567 | 62.60 | 8.0366 | 76.46 | 0.869 |
| BA-MCTS | -5.310 | -16.780 | 6.159 | 0.141 | 0.182 | 0.530 | 78.49 | 0.0026 | 171.21 | 0.778 |
| EVD pessimism | -8.842 | -23.966 | 6.281 | 0.282 | 0.220 | 0.688 | 31.29 | 0.0003 | 42.87 | 0.613 |

These pooled signed-return means combine recoverable populations with demographic sinks and must not be used alone for biological interpretation.

## Current ecological baseline: Ricker-only adapted PLUS

Authoritative ecological run:

```text
/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/
ricker_only_plus_72_20260720
```

Authoritative files:

```text
production/evaluation/**/summary.json
production/evaluation/**/episodes.csv
production/acceptance.json
runtime_snapshot_routing_fix/experiments/ricker_only_plus_72/
  manifests/ricker_only_plus_plan_72.csv
```

| Ecological object | SHA-256 |
|---|---|
| 72-row plan manifest | `398780b8267af1b53f4092994eb001caa335537910905190dc58acad23595cc2` |
| Final structural acceptance | `0963e219ed09a29aa0936865f804bea971e87c26f86bc7ce267701ce52748593` |

The run is scientifically **incomplete, 71/72**: index 51 (Iberian lynx, Allee, σ=0.2) timed out and was not rerun. All reported comparisons remove that cell from the general methods too. Of the 71 available cells, 69 have byte-identical public datasets. The two Crab-eating fox/Theta cells at σ=0.1 and 0.2 have different dataset hashes and are excluded from the primary exact-data comparison.

### Primary exact-data safe comparison

| Scope | Method | Cells | Mean return | Collapse entry | Unsafe fraction | MVP breach | Mean minimum N | Persistence | Economic cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Recoverable | RefPlan | 53 | 4.913 | 0.008 | 0.001 | 0.264 | 115.24 | 0.999 | 13.467 |
| Recoverable | OGSRL | 53 | **6.496** | 0.016 | 0.001 | 0.300 | 87.85 | 0.998 | 1.474 |
| Recoverable | BA-MCTS | 53 | 5.133 | **0.001** | **0.000** | 0.265 | 120.12 | **0.999** | 12.594 |
| Recoverable | EVD pessimism | 53 | 4.523 | 0.153 | 0.040 | 0.527 | 44.97 | 0.847 | -1.436 |
| Recoverable | Ricker-only PLUS | 53 | 5.102 | 0.034 | 0.025 | 0.310 | 108.18 | 0.966 | 3.593 |
| Sink | RefPlan | 16 | -92.318 | 0.000 | 0.500 | 1.000 | 14.34 | 0.500 | 12.828 |
| Sink | OGSRL | 16 | -92.718 | 0.000 | 0.500 | 1.000 | **26.56** | 0.500 | 12.705 |
| Sink | BA-MCTS | 16 | -94.172 | 0.000 | 0.500 | 1.000 | 12.01 | 0.500 | 16.155 |
| Sink | EVD pessimism | 16 | -124.490 | 0.500 | 0.813 | 1.000 | 0.41 | 0.000 | 4.883 |
| Sink | Ricker-only PLUS | 16 | **-46.247** | 0.000 | 0.500 | 1.000 | 10.41 | 0.500 | 11.930 |

Higher return is better; lower collapse/unsafe is better. Economic cost retains the registered sign convention, including revenue-generating negative costs.

### Direct general-versus-ecological return comparisons

These paired gaps are `general method − Ricker-only PLUS` over the 69 exact-data cells.

| General method | Mean gap | Median gap | Wins / ties / losses |
|---|---:|---:|---:|
| RefPlan | -10.828 | -2.057 | 11 / 0 / 58 |
| OGSRL | -9.705 | -0.052 | 33 / 0 / 36 |
| BA-MCTS | -11.089 | -1.889 | 12 / 0 / 57 |
| EVD pessimism | -18.587 | -1.125 | 20 / 0 / 49 |

The pooled gaps are dominated by the two demographic sinks. On the 53 exact-data recoverable cells, the mean gaps are RefPlan -0.189, OGSRL +1.395, BA-MCTS +0.031, and EVD -0.578. This recoverable/sink split is more informative than the pooled headline.

### Ecological results by species

Each row averages the available safe cells across family and noise. Iberian lynx has seven cells; every other species has eight.

| Population | Class | Cells | Return | Collapse | Unsafe | MVP breach | Mean minimum N | Persistence | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Amur tiger | Recoverable | 8 | 4.234 | 0.000 | 0.000 | 0.000 | 200.00 | 1.000 | 15.625 |
| Asian elephant | Recoverable | 8 | 6.730 | 0.000 | 0.000 | 0.000 | 59.97 | 1.000 | 1.176 |
| Bottlenose dolphin | Sink | 8 | 2.458 | 0.000 | 0.000 | 1.000 | 20.37 | 1.000 | 14.486 |
| Crab-eating fox | Recoverable | 8 | 10.591 | 0.000 | 0.000 | 1.000 | 34.23 | 1.000 | -3.522 |
| Egyptian vulture | Sink | 8 | -94.953 | 0.000 | 1.000 | 1.000 | 0.45 | 0.000 | 9.375 |
| Iberian lynx | Recoverable | 7 | 7.249 | 0.000 | 0.000 | 0.100 | 75.02 | 1.000 | -0.847 |
| Jaguar | Recoverable | 8 | 7.616 | 0.000 | 0.000 | 0.000 | 278.68 | 1.000 | 5.078 |
| Puerto Rican parrot | Recoverable | 8 | -7.040 | 0.219 | 0.164 | 0.219 | 62.79 | 0.781 | 2.922 |
| Spotted turtle | Recoverable | 8 | 7.891 | 0.006 | 0.001 | 1.000 | 21.76 | 0.994 | 2.068 |

The two sink species behave very differently: Bottlenose dolphin remains persistent with positive safe return, whereas Egyptian vulture remains fully unsafe with strongly negative return. They must not be represented only by their pooled average.

### Ecological danger-zone actions

Across all 1,420 completed ecological episodes, the most frequent stored danger-zone action is a0 (do nothing), mean episode fraction 0.219. For recoverable populations it is also a0, fraction 0.282. For sinks it is a10 (translocation), fraction 0.294. Episodes without a danger-zone step contribute zeros, so these fractions do not form a normalized distribution after averaging.

### Family-specific interpretation

The ecological baseline is committed to an eight-candidate Ricker model bank. It has a correct-form advantage on Ricker truth and is intentionally misspecified on Allee, Theta, and Regime truth. Report family-specific results and frame this as a model-form/misspecification comparison—not a universal test of whether general RL plans better than ecological methods.

Available 71-cell mean safe returns by family are:

| Family | RefPlan | OGSRL | BA-MCTS | EVD pessimism | Ricker-only PLUS |
|---|---:|---:|---:|---:|---:|
| Ricker | -16.208 | -15.296 | -16.640 | -24.372 | **-4.891** |
| Allee | -18.073 | -16.741 | -18.192 | -26.089 | **-5.683** |
| Theta | -16.107 | -15.397 | -16.648 | -22.991 | **-4.194** |
| Regime | -17.146 | -15.656 | -16.955 | -24.209 | **-10.493** |

These family means include sink populations and the two non-byte-identical Crab-eating fox/Theta cells; use the exact-data cell files for formal paired claims.

## Recoverable populations versus demographic sinks

Seven populations are registered as recoverable. Bottlenose dolphin and Egyptian vulture are registered as demographic sinks.

| Scope | Method | Safe return | Yield return | Collapse-entry rate | MVP-breach rate | Mean episode minimum N | Persistence |
|---|---|---:|---:|---:|---:|---:|---:|
| Recoverable | RefPlan | 5.082 | 6.201 | 0.096 | 0.404 | 98.02 | 0.958 |
| Recoverable | OGSRL | 6.654 | 6.649 | 0.050 | 0.443 | 72.92 | 0.975 |
| Recoverable | BA-MCTS | 5.332 | 7.322 | 0.110 | 0.395 | 99.02 | 0.929 |
| Recoverable | EVD pessimism | 4.755 | 7.266 | 0.220 | 0.599 | 40.16 | 0.788 |
| Sink | RefPlan | -92.318 | 1.764 | 0.250 | 1.000 | 8.34 | 0.356 |
| Sink | OGSRL | -92.718 | -0.507 | 0.008 | 1.000 | 26.46 | 0.498 |
| Sink | BA-MCTS | -94.172 | 2.089 | 0.250 | 1.000 | 6.64 | 0.250 |
| Sink | EVD pessimism | -124.490 | 2.835 | 0.500 | 1.000 | 0.25 | 0.000 |

Sink returns must remain signed. The strongly negative safe-return aggregate is driven principally by Egyptian vulture. Bottlenose dolphin does not have negative return in every method/reward cell, so the two sink species must also be shown individually.

For sink populations, `collapse_entry` alone is insufficient: a population already below the private safety threshold at reset can remain unsafe without a new entry event. Always report collapse-entry rate together with unsafe/MVP fractions and persistence.

## Individual population results

Each value below averages the 8 matching cells for the named population, reward, and method: 4 demographic families × 2 noise levels. `C` is collapse-entry rate and `min N` is the smallest registered episode minimum in those 8 cells.

| Population | Class | Method | Safe return | Safe C | Safe min N | Yield return | Yield C | Yield min N |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Amur tiger | Recoverable | RefPlan | 2.256 | 0.000 | 111.074 | 3.113 | 0.569 | 2.723 |
| Amur tiger | Recoverable | OGSRL | 3.607 | 0.000 | 67.970 | 2.600 | 0.019 | 24.814 |
| Amur tiger | Recoverable | BA-MCTS | 0.665 | 0.000 | 183.500 | 3.903 | 0.600 | 0.003 |
| Amur tiger | Recoverable | EVD pessimism | -12.656 | 1.000 | 2.512 | 4.371 | 1.000 | 0.758 |
| Asian elephant | Recoverable | RefPlan | 4.060 | 0.000 | 59.940 | 5.713 | 0.000 | 59.940 |
| Asian elephant | Recoverable | OGSRL | 7.865 | 0.000 | 58.519 | 7.953 | 0.000 | 58.519 |
| Asian elephant | Recoverable | BA-MCTS | 4.528 | 0.000 | 59.910 | 7.459 | 0.000 | 59.784 |
| Asian elephant | Recoverable | EVD pessimism | 7.761 | 0.000 | 58.519 | 7.914 | 0.000 | 58.519 |
| Bottlenose dolphin | Sink | RefPlan | 2.377 | 0.000 | 10.805 | 2.990 | 1.000 | 1.886 |
| Bottlenose dolphin | Sink | OGSRL | 3.015 | 0.000 | 10.038 | 2.814 | 0.031 | 8.560 |
| Bottlenose dolphin | Sink | BA-MCTS | 1.627 | 0.000 | 9.341 | 3.677 | 1.000 | 0.056 |
| Bottlenose dolphin | Sink | EVD pessimism | -61.164 | 1.000 | 0.202 | 3.660 | 1.000 | 0.056 |
| Crab-eating fox | Recoverable | RefPlan | 8.121 | 0.050 | 1.037 | 9.195 | 0.038 | 1.667 |
| Crab-eating fox | Recoverable | OGSRL | 10.136 | 0.000 | 21.297 | 10.139 | 0.000 | 21.297 |
| Crab-eating fox | Recoverable | BA-MCTS | 10.080 | 0.000 | 13.326 | 10.169 | 0.000 | 32.117 |
| Crab-eating fox | Recoverable | EVD pessimism | 10.224 | 0.000 | 21.297 | 10.187 | 0.000 | 21.297 |
| Egyptian vulture | Sink | RefPlan | -187.013 | 0.000 | 2.843 | 0.538 | 0.000 | 0.032 |
| Egyptian vulture | Sink | OGSRL | -188.451 | 0.000 | 34.765 | -3.829 | 0.000 | 34.765 |
| Egyptian vulture | Sink | BA-MCTS | -189.971 | 0.000 | 0.532 | 0.500 | 0.000 | 0.004 |
| Egyptian vulture | Sink | EVD pessimism | -187.816 | 0.000 | 0.429 | 2.010 | 0.000 | 0.0003 |
| Iberian lynx | Recoverable | RefPlan | 5.507 | 0.000 | 82.751 | 7.071 | 0.000 | 78.539 |
| Iberian lynx | Recoverable | OGSRL | 7.803 | 0.000 | 20.793 | 6.887 | 0.000 | 21.318 |
| Iberian lynx | Recoverable | BA-MCTS | 6.112 | 0.000 | 85.819 | 8.231 | 0.000 | 76.941 |
| Iberian lynx | Recoverable | EVD pessimism | 6.555 | 0.025 | 15.707 | 6.314 | 0.538 | 14.420 |
| Jaguar | Recoverable | RefPlan | 6.893 | 0.000 | 296.802 | 8.311 | 0.000 | 287.019 |
| Jaguar | Recoverable | OGSRL | 9.107 | 0.000 | 79.172 | 9.394 | 0.000 | 70.023 |
| Jaguar | Recoverable | BA-MCTS | 7.765 | 0.000 | 302.045 | 9.806 | 0.000 | 260.343 |
| Jaguar | Recoverable | EVD pessimism | 9.029 | 0.000 | 70.023 | 8.996 | 0.000 | 70.023 |
| Puerto Rican parrot | Recoverable | RefPlan | 2.508 | 0.000 | 69.892 | 2.924 | 0.688 | 11.690 |
| Puerto Rican parrot | Recoverable | OGSRL | -0.269 | 0.106 | 39.645 | 2.437 | 0.575 | 11.480 |
| Puerto Rican parrot | Recoverable | BA-MCTS | 1.947 | 0.006 | 35.055 | 3.898 | 0.525 | 5.984 |
| Puerto Rican parrot | Recoverable | EVD pessimism | 4.920 | 0.000 | 66.131 | 5.662 | 0.519 | 4.790 |
| Spotted turtle | Recoverable | RefPlan | 6.231 | 0.000 | 26.504 | 7.082 | 0.000 | 10.331 |
| Spotted turtle | Recoverable | OGSRL | 8.327 | 0.000 | 12.298 | 7.134 | 0.000 | 8.037 |
| Spotted turtle | Recoverable | BA-MCTS | 6.230 | 0.000 | 27.254 | 7.784 | 0.413 | 4.770 |
| Spotted turtle | Recoverable | EVD pessimism | 7.453 | 0.000 | 10.014 | 7.419 | 0.000 | 10.014 |

The raw files should be used for uncertainty intervals and family/noise stratification; the rounded values above are a handoff summary, not a replacement for the episode data.

## Actions taken in the danger zone

The most frequent stored danger-zone action by method is shown below. Fractions are averages over all registered episodes, including zero vectors for episodes that never enter the danger zone.

| Scope | Reward | Method | Top action | Mean episode fraction |
|---|---|---|---|---:|
| All | Safe | RefPlan | a10 Translocation | 0.212 |
| All | Safe | OGSRL | a2 Aggressive harvest | 0.298 |
| All | Safe | BA-MCTS | a10 Translocation | 0.202 |
| All | Safe | EVD pessimism | a2 Aggressive harvest | 0.411 |
| All | Yield | RefPlan | a2 Aggressive harvest | 0.164 |
| All | Yield | OGSRL | a2 Aggressive harvest | 0.427 |
| All | Yield | BA-MCTS | a1 Sustainable harvest | 0.202 |
| All | Yield | EVD pessimism | a2 Aggressive harvest | 0.470 |
| Sink | Safe | RefPlan | a10 Translocation | 0.275 |
| Sink | Safe | OGSRL | a3 Predator/disease control | 0.191 |
| Sink | Safe | BA-MCTS | a10 Translocation | 0.238 |
| Sink | Safe | EVD pessimism | a0 Do nothing | 0.250 |
| Sink | Yield | RefPlan | a2 Aggressive harvest | 0.242 |
| Sink | Yield | OGSRL | a10 Translocation | 0.145 |
| Sink | Yield | BA-MCTS | a2 Aggressive harvest | 0.233 |
| Sink | Yield | EVD pessimism | a0 Do nothing | 0.431 |

All 11 action fractions—not just the modal actions above—are available in every `episodes.csv`. The action meanings are:

| ID | Action |
|---:|---|
| a0 | Do nothing |
| a1 | Sustainable harvest |
| a2 | Aggressive harvest |
| a3 | Predator/disease control |
| a4 | Breeding/recruitment support |
| a5 | Moderate restoration |
| a6 | Intensive restoration |
| a7 | Integrated conservation (light) |
| a8 | Adaptive conservation trial |
| a9 | Flagship conservation programme |
| a10 | Translocation |

## Why the experiment looks this way: method trust and policy behavior

This section consolidates the post-outcome trust and policy-degeneracy audit. It is based on read-only inspection of the completed artifacts; no runtime, manifest, fit, dataset, or outcome was modified.

### Trust classification

The exact equality between corrected MOOR and Ricker-only PLUS is caused by **policy degeneracy**, not by shared fit artifacts or an output-directory collision. Both methods chose the same constant action throughout every episode in the 16-cell overlap. With paired datasets, evaluation seeds, and environment randomness, identical action sequences necessarily generated identical ecological outcomes.

| Method/result | Trust classification | What the result supports |
|---|---|---|
| RefPlan | **TRUST** | Registered RefPlan outcomes; complete, behaviorally non-degenerate, and no recorded fallback. |
| BA-MCTS | **TRUST** | Registered BA-MCTS outcomes; complete, behaviorally non-degenerate, and no recorded fallback. |
| EVD pessimism | **TRUST WITH METHOD-SCOPE WARNING** | Outcomes of the registered ensemble-disagreement method. It remains Delphic-motivated, not a full Delphic-compatible-world implementation. |
| OGSRL | **TRUST AS THE DEPLOYED ACTOR + GUARDIAN + FALLBACK COMPOSITE** | Real outcomes of the implemented composite. They are not fallback-free actor-only results. |
| Ricker-only PLUS, available run | **TRUST WITH INCOMPLETENESS WARNING** | Outcomes of the implemented method for 71 completed cells, subject to the exact-data exclusions. The run is not a complete 72-cell experiment. |
| Corrected MOOR diagnostic | **MECHANICALLY VALID, SCIENTIFICALLY NON-DISCRIMINATING** | Outcomes of a nearly constant policy on two populations. It does not demonstrate a general MOOR advantage or family-sensitive adaptation. |
| MOOR versus PLUS on the 16-cell overlap | **DO NOT USE TO DISTINGUISH THE ALGORITHMS** | The outcomes are valid, but both methods execute the same constant actions, leaving the comparison with no behavioral resolving power. |

“Trusted” here means that the stored numbers faithfully represent the registered implementation that ran. It does not mean every implementation provides equally informative evidence about its underlying algorithmic idea.

### Artifact independence: why this is not copied output

For all 16 cells shared by corrected MOOR and Ricker-only PLUS at `sigma_obs` 0.1/0.2:

- identical whole-fit hashes: **0/16**;
- any shared fitted-parameter hash: **0/16**;
- any shared POMDP-model hash: **0/16**;
- cells with different method-uncertainty diagnostics: **16/16**;
- output roots and method identifiers are distinct;
- corrected MOOR uses one fitted Ricker model, while Ricker-only PLUS uses eight fitted Ricker candidates;
- both methods have zero recorded fallback counts.

Thus, there is no evidence that either method silently loaded the other's fit, POMDP, or result file. The internal models and action-value arrays differ even though their maximizing actions agree.

### Exact mechanism behind the MOOR–PLUS tie

Every one of the 320 corrected-MOOR episodes and the corresponding 320 Ricker-only-PLUS episodes has action entropy zero. Economic cost and the registered action costs identify the policies:

- Amur tiger: constant `a10` (translocation) for all 50 steps; `50 × 0.3125 = 15.625` cost per episode;
- Egyptian vulture: constant `a5` (moderate restoration) for all 50 steps; `50 × 0.1875 = 9.375` cost per episode.

The terminal PBVI diagnostics independently agree: corrected MOOR's action-value argmax and PLUS's posterior-weighted argmax are `a10` for every Amur-tiger overlap cell and `a5` for every Egyptian-vulture overlap cell. Their numerical action values and decision margins differ, so the planners are not identical. They simply land on the same action.

Once the action sequence is identical, the paired evaluation seed drives the same environment randomness and therefore the same states, observations, rewards, safety events, and costs. This explains the maximum absolute outcome gap of zero without requiring a caching or routing bug.

### Behavioral-degeneracy census

A deterministic cell is one in which all 20 evaluation episodes have action entropy zero.

| Method | Completed cells | Deterministic cells | Zero-entropy episodes | Mean action entropy | Recorded fallback decisions |
|---|---:|---:|---:|---:|---:|
| RefPlan | 144 | 0 (0.0%) | 0.0% | 1.741 | 0 |
| OGSRL | 144 | 20 (13.9%) | 28.1% | 0.567 | 27,204 |
| BA-MCTS | 144 | 0 (0.0%) | 0.0% | 1.497 | 0 |
| EVD pessimism | 144 | 12 (8.3%) | 16.5% | 0.618 | 0 |
| Ricker-only PLUS | 71 | 16 (22.5%) | 23.6% | 0.347 | 0 |
| Corrected MOOR | 32 | 31 (96.9%) | 96.9% | 0.007 | 0 |

The 16 deterministic Ricker-only-PLUS cells are exactly the corrected-MOOR overlap: Amur tiger and Egyptian vulture across four families and both registered noise levels. The selected slice is therefore unusually uninformative even though 55 of the 71 available Ricker-only-PLUS cells are not wholly deterministic.

Corrected MOOR is much more severely degenerate: 31/32 diagnostic cells are deterministic. Its outputs are valid outcomes of its deployed policy, but they do not show that its fitted ecological model meaningfully changes decisions. This is itself an experimental finding: under the registered reward, surrogate, action set, and PBVI configuration, MOOR usually collapses to a constant intervention.

RefPlan and BA-MCTS show no deterministic cells, providing stronger evidence that their reported differences reflect changing decisions. EVD has some deterministic behavior, but it is not globally collapsed and has no recorded fallback.

### Why OGSRL requires a composite-policy interpretation

The evaluator records 27,204 OGSRL fallback events over 144,000 decisions: **18.89%**, with at least one event in 102/144 cells. The burden is concentrated in demographic sinks:

| Scope/reward | Recorded fallback decisions | Fraction of decisions in stratum |
|---|---:|---:|
| Recoverable / safe | 3,891 | 6.95% |
| Recoverable / yield | 3,093 | 5.52% |
| Sink / safe | 8,609 | 53.81% |
| Sink / yield | 11,611 | 72.57% |

OGSRL intentionally chooses a least-violating action when its guardian finds no feasible action. However, the frozen evaluator increments the same `fallback_count` for:

1. an exception from `policy.act`;
2. an invalid action ID;
3. the policy's intentional `hard_fallback` diagnostic.

The frozen output does not separate these causes. Its Phase-2E validity receipt checks structural identity, fitting/mechanism diagnostics, resources, threads, and successful exit, but does not gate evaluator fallback counts. Therefore the run establishes the performance of the deployed **actor + OOD/safety guardian + least-violating fallback composite**, not fallback-free actor performance.

This does not make the OGSRL outcomes false. It changes the scientific interpretation. The recoverable safe stratum has a much lower 6.95% fallback burden than the sink strata, but even there an actor-only causal claim is unsupported. Sink results are dominated by the regime where the guardian frequently reports no feasible action.

### Experiment-level validity and reporting consequences

- General methods: 576/576 registered method rows completed with dedicated validity receipts and frozen identity. RefPlan, BA-MCTS, and EVD have zero recorded evaluator fallback; OGSRL has the composite-policy limitation above.
- Ricker-only PLUS: 71/72 cells completed. Index 51, Iberian lynx / Allee / `sigma_obs=0.2`, timed out and was neither fabricated nor rerun. Exact-data comparisons retain all registered exclusions.
- Corrected MOOR: 32/32 fits and 32/32 plans completed, and all 32 MOOR rows passed the frozen per-row structural checks. The combined cross-family PLUS/MOOR experiment remains `INCOMPLETE` because the companion PLUS plan arm timed out and its Slurm checker cannot map this cluster's raw array-element identifiers.

The resulting interpretation rules are:

1. RefPlan and BA-MCTS quantitative results can be used as registered.
2. EVD results can be used with the “Delphic-motivated, not full Delphic” label.
3. OGSRL numbers must be labelled as the deployed actor/guardian/fallback composite; avoid unqualified “OGSRL wins” language, especially for sinks.
4. Ricker-only PLUS results remain usable with the 71/72 and exact-dataset limitations.
5. Corrected MOOR numbers should be described as a two-population, nearly constant-policy diagnostic.
6. The corrected-MOOR versus Ricker-only-PLUS tie should be reported as policy degeneracy, not as algorithmic equivalence.
7. A future discriminating ecological comparison should store per-timestep actions and separate counters for policy exceptions, invalid actions, guardian infeasibility, and intended algorithmic fallbacks.

## Comparison keys and future extensions

The current safe-mode cell metrics were compared using the exact scientific key:

```text
population, environment, sigma_obs, reward_mode
```

Retain `model`/method as the comparison dimension. Do not select the best general method before comparison. The current comparison reports RefPlan, OGSRL, BA-MCTS, and EVD pessimism separately against Ricker-only adapted PLUS.

The combined report should include:

1. mean signed operational return, with safe and yield modes separate;
2. recoverable and sink aggregates plus each population individually;
3. collapse-entry rate together with unsafe fraction, MVP breach/fraction, and persistence;
4. minimum, mean, and final abundance from the frozen evaluation;
5. danger-zone action distributions for all 11 actions;
6. family- and noise-specific results;
7. cell-paired method gaps, wins/ties/losses, and uncertainty intervals;
8. method-specific comparisons rather than a best-general-versus-ecological headline;
9. the validated selected-population per-step population/reward/action plots and maximum-abundance diagnostics in `analysis/general_only_trajectory_results/`.

Dataset identity must be preserved in matching. The registry records 140/144 byte-identical reused ecological datasets. The four Crab-eating fox/theta cells use the exact registered 160-complete-episode subset because the ecological source included five partial rows; these four cells are matched in design but are not whole-file-identical.

## Data integrity notes for the merging agent

- Use `operational_return` as the primary return and retain its sign.
- Treat the method-cell, not each episode, as the default unit for across-cell headline uncertainty.
- Keep safe and yield rewards separate before any combined presentation.
- Do not interpret a zero sink `collapse_entry` as safety without checking initial/ongoing unsafe occupancy and persistence.
- Do not use older MOOR/PLUS values or figures formerly placed in this analysis directory.
- Do not use the previously prepared mixed `genrl-traces` workflow as a result source.
- All general methods used one CPU thread with `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.
