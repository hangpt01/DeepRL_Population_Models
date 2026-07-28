# Matched general-RL and Ricker-only PLUS ecological results

## Status and scope

This is the single ecological-side working report for the matched comparison. The authoritative
four-method general-RL results have been imported below. Ecological outcome values and combined
plots remain intentionally blank until the frozen ecological structural acceptance passes and the
user explicitly authorizes ecological return inspection.

The ecological experiment contains only `plus_adapted_ricker_only_pbvi` under the **safe** reward.
It does not contain MOOR, `plus_native`, or an ecological yield experiment. Older PLUS/MOOR values
from `hidden_rk_comparison_20260716` are prohibited.

## Authoritative general-RL sources

General package root:

`/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1`

- Complete handoff:
  `/fs04/scratch2/ce25/general_rl_phase2_iso/docs/fix_implement_general_RL/HANDOFF_GENERAL_RL_RESULTS_FOR_ECOLOGICAL_MERGE.md`
- Canonical general report:
  `/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/analysis/matched_results/MATCHED_GENERAL_ECOLOGICAL_RESULTS.md`
- Frozen 576-row manifest:
  `/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/manifests/full_general_sigma01_02_576_rows.csv`
- Dataset reuse registry:
  `/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/manifests/ecological_dataset_reuse_registry_144.csv`
- Authoritative frozen outcomes:
  `/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/quarantine/evaluation/`

Frozen general hashes:

| Object | SHA-256 |
|---|---|
| 576-row manifest | `1526ce08dcf1b1d148232c075d41dbd78f0cfa2d7119cff9e330f965b6451df4` |
| Registration | `15fd7aa1c3ddc460fd255e5d66518d49244f81cf3dbbd291a5bdfe52b376b37a` |
| Dataset registry | `474d1a65d2e5ec28c741f5b7ac5691be891712e753ee6a9a6590337d62f5f0c4` |
| Frozen general code | `f615d363a525422abfb983f0eee7c433b009fff81a4d0ae4925bfdaba0f3aa72` |

The four general methods must remain separate throughout the comparison:

- `refplan` (RefPlan)
- `ogsrl` (OGSRL)
- `bamcts` (BA-MCTS)
- `ensemble_value_disagreement_pessimism` (EVD pessimism)

The registered general experiment contains 9 populations x 4 families x 2 noise levels x 2
reward modes x 4 methods = 576 rows. All 576 completed and passed the frozen structural checker.
Each method-cell has 20 paired evaluation episodes of 50 steps and discount `gamma=0.95`.

## General-RL metric definitions

The authoritative episode files contain signed operational and true return; collapse entry/count
and first entry time; unsafe and MVP fractions; MVP breach; persistence; economic cost; minimum,
mean and final true population; action entropy; all 11 danger-zone action fractions; filter and
uncertainty diagnostics; fallback count; and filtering/planning time.

The danger zone is `safety_threshold < state_pre <= 4 * safety_threshold`. A zero collapse-entry
indicator does not imply safety when a sink starts below the threshold, so collapse must always be
reported with unsafe occupancy, MVP breach/fraction and persistence.

## Overall general-method results

Headline means use the 144 scientific method-cells as the aggregation unit.

| Method | Both-mode return | Safe return | Yield return | Collapse entry | Unsafe fraction | MVP breach | Mean minimum N | Smallest N | Mean final N | Persistence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RefPlan | -5.673 | -16.562 | 5.215 | 0.130 | 0.163 | 0.537 | 78.09 | 0.0317 | 195.87 | 0.825 |
| OGSRL | -5.185 | -15.429 | 5.059 | 0.041 | 0.125 | 0.567 | 62.60 | 8.0366 | 76.46 | 0.869 |
| BA-MCTS | -5.310 | -16.780 | 6.159 | 0.141 | 0.182 | 0.530 | 78.49 | 0.0026 | 171.21 | 0.778 |
| EVD pessimism | -8.842 | -23.966 | 6.281 | 0.282 | 0.220 | 0.688 | 31.29 | 0.0003 | 42.87 | 0.613 |

These pooled signed returns combine recoverable populations and demographic sinks and are not a
standalone biological headline.

## Recoverable and sink general results

Seven populations are recoverable. Bottlenose dolphin and Egyptian vulture are demographic sinks.

| Scope | Method | Safe return | Yield return | Collapse entry | MVP breach | Mean minimum N | Persistence |
|---|---|---:|---:|---:|---:|---:|---:|
| Recoverable | RefPlan | 5.082 | 6.201 | 0.096 | 0.404 | 98.02 | 0.958 |
| Recoverable | OGSRL | 6.654 | 6.649 | 0.050 | 0.443 | 72.92 | 0.975 |
| Recoverable | BA-MCTS | 5.332 | 7.322 | 0.110 | 0.395 | 99.02 | 0.929 |
| Recoverable | EVD pessimism | 4.755 | 7.266 | 0.220 | 0.599 | 40.16 | 0.788 |
| Sink | RefPlan | -92.318 | 1.764 | 0.250 | 1.000 | 8.34 | 0.356 |
| Sink | OGSRL | -92.718 | -0.507 | 0.008 | 1.000 | 26.46 | 0.498 |
| Sink | BA-MCTS | -94.172 | 2.089 | 0.250 | 1.000 | 6.64 | 0.250 |
| Sink | EVD pessimism | -124.490 | 2.835 | 0.500 | 1.000 | 0.25 | 0.000 |

## Individual-species general results

Each row averages four families x two noise levels. `C` is collapse-entry rate and `min N` is the
smallest episode minimum across those eight cells.

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

## General danger-zone actions and cost

Economic cost is present per episode in the authoritative files and must be included in later
cell-level comparisons. All danger-action columns `danger_action_0_fraction` through
`danger_action_10_fraction` are authoritative; do not collapse them to the modal action for final
analysis. The modal summaries are:

| Scope | Reward | Method | Most frequent danger action | Mean episode fraction |
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

## General selected-species trajectories

Selected populations are Amur tiger and Puerto Rican parrot (recoverable) and Egyptian vulture
(sink). The completed diagnostic covers all four families, sigma 0.1/0.2, safe/yield reward and
all four general methods: 48 cells, 192 method-cells, 20 episodes, 50 timesteps and 192,000 rows.

| Artifact | Absolute path | SHA-256 |
|---|---|---|
| Long trajectories | `/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/analysis/general_only_trajectory_results/general_only_trajectories_long.csv.gz` | `067a956edcc7136aa71912147311c12e08928b90ebf278323d505e3bbfd01231` |
| Population extrema | `/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/analysis/general_only_trajectory_results/general_only_population_extrema.csv` | `016656ae9da8042ae61d187307adce149d0a5dff80dcda6e3cd1bf904bf38cdd` |
| Validation receipt | `/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/analysis/general_only_trajectory_results/general_only_trajectory_receipt.json` | `b0d840cec5f06c325c0423f194a2bc6584ee65ffde36ac6afdbc0bee0a25f0ca` |
| Raw bundles | `/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/analysis/general_only_trajectory_results/raw_npz/` | 48 per-cell files; hashes recorded by the package |
| General-only figures | `/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/analysis/general_only_trajectory_results/figures/` | 48 figures |

The validation receipt records `max_absolute_return_reproduction_gap =
1.1368683772161603e-13` and `return_reproduction_pass = true`.

## Dataset matching

The general registry records 140/144 byte-identical ecological dataset reuses. The four registered
subset exceptions are Crab-eating fox/theta and are outside the three-species trajectory diagnostic.
Exact dataset hashes must still be checked for all 24 ecological trajectory cells before merging.

## Ecological structural acceptance

The frozen checker was rerun after the production array ended. Refreshed result:

| Field | Result |
|---|---|
| Decision | `INCOMPLETE` |
| Fit coverage | 72/72 |
| Plan-completion coverage | 71/72 |
| Slurm states | 71 `COMPLETED`, 1 `TIMEOUT` |
| Structural failures | none |
| Temporary/partial files | none detected |
| Allocated core-hours | 361.36277777777775 |
| Return fields opened | `false` |

The unresolved task is array index 51: Iberian lynx, Allee truth, `sigma_obs=0.2`. Its fit receipt
exists, but its plan-completion receipt does not; Slurm records `TIMEOUT`. This cell is outside the
three-species trajectory subset, but the registered global acceptance gate requires complete 72-row
coverage and zero unresolved task failures. Ecological trajectory capture is therefore blocked.

## Ecological trajectory capture and sealed aggregation

The operator authorized a selected-subset waiver for trajectory generation only. The full 72-row
experiment remains `INCOMPLETE`; no missing result is fabricated and no full-run PASS claim is
permitted.

- `scoped_selected_trajectory_waiver: true`
- `global_acceptance: INCOMPLETE`
- `global_missing_index: 51`
- `global_missing_cell: Iberian lynx / Allee / sigma_obs=0.2`
- `selected_cells_complete: 24/24`
- `missing_global_cell_outside_selected_subset: true`
- `full_run_claims_prohibited: true`

Authorized ecological scope:

- method: `plus_adapted_ricker_only_pbvi`;
- reward: safe only;
- indices: `0-7,16-23,40-47`;
- 3 populations x 4 families x 2 noise levels = 24 cells;
- 20 episodes x 50 steps = 24,000 long-form rows.

All 24 strengthened per-cell checks passed before submission: production Slurm state
`COMPLETED/0:0`, atomic completion receipt, exact artifact hashes, Ricker-only bank, privacy and
planner structure, eight fit-cache hits with no refit, frozen package/config identity, and exact
general-registry dataset reuse.

| Item | Value |
|---|---|
| Capture array | `58399693`, indices `0-7,16-23,40-47%24` |
| Capture start | `2026-07-20 06:54:06 AEST` (all 24 running) |
| Capture resources | `m3h/m3h`, 1 CPU, 8 GiB, `06:30:00`, no requeue |
| Sealed aggregation | `58399718`, `afterok:58399693`, 1 CPU, 8 GiB, `00:30:00` |
| Producer SHA-256 | `bd1cee796a50e52c14f82fc3291c4b221cfd4a0bc072582c0f9bd785e6a89b88` |
| Runner SHA-256 | `8b7cb621cd63db2d130e3c3dc05779598e2a184c3316b4f3219e71b402c33423` |
| Aggregator SHA-256 | `60e27b824025fede731339c4f4d9df4258aba903b01c6a49603fe333706bcb45` |

The outputs remain sealed. Do not insert ecological outcome values until separate return-inspection
authorization is given.

## Authorized ecological return validation

**PLACEHOLDER — REQUIRES EXPLICIT AUTHORIZATION.** After authorization, reconstruct each episode's
discounted public return as `sum(t=0..49) 0.95^t * reward_public[t]`, compare each 20-episode cell
mean against the frozen ecological operational return, and require a tiny numerical gap before
unsealing or plotting.

## Combined comparisons and figures

**PLACEHOLDER — BLOCKED ON ACCEPTANCE, CAPTURE, AND RETURN AUTHORIZATION.** Merge trajectory rows on
`population, environment, sigma_obs, reward_mode, episode, seed, timestep`, retaining `method` as
the comparison dimension. Produce safe-only comparisons for the ecological method. General yield
trajectories remain general-only.

Required combined outputs include 24 three-panel cell figures, species contact sheets, family and
noise views, population extrema, all-action danger-zone composition, collapse/unsafe/MVP/
persistence comparisons, and each general method directly compared with the ecological method.

## Interpretation constraints

Report every family separately. On Ricker truth, the ecological method has a correct-form Ricker
inductive bias; on Allee, theta and regime truth it is deliberately misspecified. Frame results as
form-committed ecological planning versus form-flexible general MBRL under misspecification. Do not
make a broad claim that general RL beats ecological planning, do not pool Ricker/non-Ricker as the
only headline, and do not refer to ecological methods in the plural for this PLUS-only run.
