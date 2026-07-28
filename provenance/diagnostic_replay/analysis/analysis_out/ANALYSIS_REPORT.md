# P=10 diagnostic replay — read-only analysis report

No accepted artifact, replay log, derived input, or receipt was modified. Constant-action rows are evaluator-only references; this report changes no ranking, method, reward, or hyperparameter.

## Execution

- Self-test: **ALL TESTS PASSED** (declared 32 assertions).
- Execution: one local interactive process; no Slurm job submitted.
- Initial uploaded-loader run completed, but its CSV-over-NPZ collision hid every vector array. The corrected pass normalized the existing NPZ arrays in memory without modifying source or data.
- The initial `metrics.json`, `metrics_table.csv`, and `schema_report_initial_loader.csv` are retained only as execution trace. Use `metrics_normalized.json`, `m1_m15_comparison.csv`, and `schema_report.csv` for the corrected results.
- `NEXT_WORK_QUEUE.md` at the replay root: **NOT FOUND**. Prediction text was therefore not inferred.

## Schema report (actual NPZ contents)

The full machine-readable table is `schema_report.csv`. Each of the 24 logs contains 1,000 rows, 20 episodes, and 50 steps per episode. Tier B genuinely omits trajectory utility/penalty/demographic fields and `surrogate_reward_t`; Tier A contains them. The full CSV is reproduced below.

```csv
cell_id,method,source_method,rows,episodes,steps_per_episode,scalar_fields,vector_fields,missing_registered_scalar_fields
A1_crab_eating_fox_ricker_s0p2,moor,moor_adapted_ricker_misspec_pbvi,1000,20,50,"action_t,allee_C_episode,argmax,belief_entropy,belief_mean,cost_t,k_t,m_t,margin_top1_top2,obs_t,oracle_action,penalty_flag_t,r_mort_t,r_pos_t,r_setpoint_t,regime_switched_t,regime_z_t,reward_t,seed,surrogate_reward_t,t,theta_exponent_episode,true_margin_top1_top2,utility_t,x_true_next,x_true_t","q,true_reward_all_actions,x_next_all",-
A1_crab_eating_fox_ricker_s0p2,plus,plus_adapted_ricker_only_pbvi,1000,20,50,"action_t,allee_C_episode,argmax_map_only,argmax_uniform,argmax_weighted,argmax_weighted_matches_deployed,cost_t,k_t,m_t,margin_top1_top2,obs_t,oracle_action,penalty_flag_t,r_mort_t,r_pos_t,r_setpoint_t,regime_switched_t,regime_z_t,reward_t,seed,surrogate_reward_t,t,theta_exponent_episode,true_margin_top1_top2,utility_t,x_true_next,x_true_t","argmax_cand,belief_entropy,belief_mean,q_cand,q_weighted,true_reward_all_actions,w,x_next_all",-
A2_crab_eating_fox_allee_s0p2,moor,moor_adapted_ricker_misspec_pbvi,1000,20,50,"action_t,allee_C_episode,argmax,belief_entropy,belief_mean,cost_t,k_t,m_t,margin_top1_top2,obs_t,oracle_action,penalty_flag_t,r_mort_t,r_pos_t,r_setpoint_t,regime_switched_t,regime_z_t,reward_t,seed,surrogate_reward_t,t,theta_exponent_episode,true_margin_top1_top2,utility_t,x_true_next,x_true_t","q,true_reward_all_actions,x_next_all",-
A2_crab_eating_fox_allee_s0p2,plus,plus_adapted_ricker_only_pbvi,1000,20,50,"action_t,allee_C_episode,argmax_map_only,argmax_uniform,argmax_weighted,argmax_weighted_matches_deployed,cost_t,k_t,m_t,margin_top1_top2,obs_t,oracle_action,penalty_flag_t,r_mort_t,r_pos_t,r_setpoint_t,regime_switched_t,regime_z_t,reward_t,seed,surrogate_reward_t,t,theta_exponent_episode,true_margin_top1_top2,utility_t,x_true_next,x_true_t","argmax_cand,belief_entropy,belief_mean,q_cand,q_weighted,true_reward_all_actions,w,x_next_all",-
A3_crab_eating_fox_regime_s0p2,moor,moor_adapted_ricker_misspec_pbvi,1000,20,50,"action_t,allee_C_episode,argmax,belief_entropy,belief_mean,cost_t,k_t,m_t,margin_top1_top2,obs_t,oracle_action,penalty_flag_t,r_mort_t,r_pos_t,r_setpoint_t,regime_switched_t,regime_z_t,reward_t,seed,surrogate_reward_t,t,theta_exponent_episode,true_margin_top1_top2,utility_t,x_true_next,x_true_t","q,true_reward_all_actions,x_next_all",-
A3_crab_eating_fox_regime_s0p2,plus,plus_adapted_ricker_only_pbvi,1000,20,50,"action_t,allee_C_episode,argmax_map_only,argmax_uniform,argmax_weighted,argmax_weighted_matches_deployed,cost_t,k_t,m_t,margin_top1_top2,obs_t,oracle_action,penalty_flag_t,r_mort_t,r_pos_t,r_setpoint_t,regime_switched_t,regime_z_t,reward_t,seed,surrogate_reward_t,t,theta_exponent_episode,true_margin_top1_top2,utility_t,x_true_next,x_true_t","argmax_cand,belief_entropy,belief_mean,q_cand,q_weighted,true_reward_all_actions,w,x_next_all",-
A4_crab_eating_fox_theta_s0p2,moor,moor_adapted_ricker_misspec_pbvi,1000,20,50,"action_t,allee_C_episode,argmax,belief_entropy,belief_mean,cost_t,k_t,m_t,margin_top1_top2,obs_t,oracle_action,penalty_flag_t,r_mort_t,r_pos_t,r_setpoint_t,regime_switched_t,regime_z_t,reward_t,seed,surrogate_reward_t,t,theta_exponent_episode,true_margin_top1_top2,utility_t,x_true_next,x_true_t","q,true_reward_all_actions,x_next_all",-
A4_crab_eating_fox_theta_s0p2,plus,plus_adapted_ricker_only_pbvi,1000,20,50,"action_t,allee_C_episode,argmax_map_only,argmax_uniform,argmax_weighted,argmax_weighted_matches_deployed,cost_t,k_t,m_t,margin_top1_top2,obs_t,oracle_action,penalty_flag_t,r_mort_t,r_pos_t,r_setpoint_t,regime_switched_t,regime_z_t,reward_t,seed,surrogate_reward_t,t,theta_exponent_episode,true_margin_top1_top2,utility_t,x_true_next,x_true_t","argmax_cand,belief_entropy,belief_mean,q_cand,q_weighted,true_reward_all_actions,w,x_next_all",-
A5_amur_tiger_theta_s0p1,moor,moor_adapted_ricker_misspec_pbvi,1000,20,50,"action_t,allee_C_episode,argmax,belief_entropy,belief_mean,cost_t,k_t,m_t,margin_top1_top2,obs_t,oracle_action,penalty_flag_t,r_mort_t,r_pos_t,r_setpoint_t,regime_switched_t,regime_z_t,reward_t,seed,surrogate_reward_t,t,theta_exponent_episode,true_margin_top1_top2,utility_t,x_true_next,x_true_t","q,true_reward_all_actions,x_next_all",-
A5_amur_tiger_theta_s0p1,plus,plus_adapted_ricker_only_pbvi,1000,20,50,"action_t,allee_C_episode,argmax_map_only,argmax_uniform,argmax_weighted,argmax_weighted_matches_deployed,cost_t,k_t,m_t,margin_top1_top2,obs_t,oracle_action,penalty_flag_t,r_mort_t,r_pos_t,r_setpoint_t,regime_switched_t,regime_z_t,reward_t,seed,surrogate_reward_t,t,theta_exponent_episode,true_margin_top1_top2,utility_t,x_true_next,x_true_t","argmax_cand,belief_entropy,belief_mean,q_cand,q_weighted,true_reward_all_actions,w,x_next_all",-
A6_egyptian_vulture_ricker_s0p1,moor,moor_adapted_ricker_misspec_pbvi,1000,20,50,"action_t,allee_C_episode,argmax,belief_entropy,belief_mean,cost_t,k_t,m_t,margin_top1_top2,obs_t,oracle_action,penalty_flag_t,r_mort_t,r_pos_t,r_setpoint_t,regime_switched_t,regime_z_t,reward_t,seed,surrogate_reward_t,t,theta_exponent_episode,true_margin_top1_top2,utility_t,x_true_next,x_true_t","q,true_reward_all_actions,x_next_all",-
A6_egyptian_vulture_ricker_s0p1,plus,plus_adapted_ricker_only_pbvi,1000,20,50,"action_t,allee_C_episode,argmax_map_only,argmax_uniform,argmax_weighted,argmax_weighted_matches_deployed,cost_t,k_t,m_t,margin_top1_top2,obs_t,oracle_action,penalty_flag_t,r_mort_t,r_pos_t,r_setpoint_t,regime_switched_t,regime_z_t,reward_t,seed,surrogate_reward_t,t,theta_exponent_episode,true_margin_top1_top2,utility_t,x_true_next,x_true_t","argmax_cand,belief_entropy,belief_mean,q_cand,q_weighted,true_reward_all_actions,w,x_next_all",-
B1_crab_eating_fox_regime_s0p2,bamcts,bamcts,1000,20,50,"action_t,cost_t,max_depth,obs_t,reward_t,seed,t,tree_nodes,tree_nodes_trace,unsafe_next,x_true_next,x_true_t","root_model_belief,root_q,root_visit_counts","utility_t,penalty_flag_t,r_setpoint_t,r_pos_t,r_mort_t,k_t,m_t,regime_z_t,regime_switched_t,allee_C_episode,theta_exponent_episode,surrogate_reward_t"
B1_crab_eating_fox_regime_s0p2,evd,ensemble_value_disagreement_pessimism,1000,20,50,"action_t,argmax_lambda0,argmax_pessimistic,cost_t,obs_t,reward_t,seed,t,unsafe_next,x_true_next,x_true_t","pessimistic_q_score,q_ensemble_mean,q_ensemble_variance","utility_t,penalty_flag_t,r_setpoint_t,r_pos_t,r_mort_t,k_t,m_t,regime_z_t,regime_switched_t,allee_C_episode,theta_exponent_episode,surrogate_reward_t"
B1_crab_eating_fox_regime_s0p2,ogsrl,ogsrl,1000,20,50,"action_t,behavior_normalized_cost,cost_horizon_used,cost_t,guardian_override,hard_fallback,lambda_ood,lambda_safety,low_abundance_scale,obs_t,ood_guardian_flag,ood_probability,predicted_C25,reward_t,safety_budget,safety_slack,seed,t,true_s_safe_evaluator_only,unsafe_next,x_true_next,x_true_t",action_probabilities,"utility_t,penalty_flag_t,r_setpoint_t,r_pos_t,r_mort_t,k_t,m_t,regime_z_t,regime_switched_t,allee_C_episode,theta_exponent_episode,surrogate_reward_t"
B1_crab_eating_fox_regime_s0p2,refplan,refplan,1000,20,50,"action_t,argmax_lambda0,belief_entropy,belief_mean,best_sequence_mean,best_sequence_reflected,best_sequence_return_sd,cost_t,model_entropy,obs_t,posterior_max,public_extinction_risk,reward_t,seed,t,unsafe_next,x_true_next,x_true_t","action_mean_lambda0,action_scores,member_returns,model_posterior","utility_t,penalty_flag_t,r_setpoint_t,r_pos_t,r_mort_t,k_t,m_t,regime_z_t,regime_switched_t,allee_C_episode,theta_exponent_episode,surrogate_reward_t"
B2_amur_tiger_ricker_s0p1,bamcts,bamcts,1000,20,50,"action_t,cost_t,max_depth,obs_t,reward_t,seed,t,tree_nodes,tree_nodes_trace,unsafe_next,x_true_next,x_true_t","root_model_belief,root_q,root_visit_counts","utility_t,penalty_flag_t,r_setpoint_t,r_pos_t,r_mort_t,k_t,m_t,regime_z_t,regime_switched_t,allee_C_episode,theta_exponent_episode,surrogate_reward_t"
B2_amur_tiger_ricker_s0p1,evd,ensemble_value_disagreement_pessimism,1000,20,50,"action_t,argmax_lambda0,argmax_pessimistic,cost_t,obs_t,reward_t,seed,t,unsafe_next,x_true_next,x_true_t","pessimistic_q_score,q_ensemble_mean,q_ensemble_variance","utility_t,penalty_flag_t,r_setpoint_t,r_pos_t,r_mort_t,k_t,m_t,regime_z_t,regime_switched_t,allee_C_episode,theta_exponent_episode,surrogate_reward_t"
B2_amur_tiger_ricker_s0p1,ogsrl,ogsrl,1000,20,50,"action_t,behavior_normalized_cost,cost_horizon_used,cost_t,guardian_override,hard_fallback,lambda_ood,lambda_safety,low_abundance_scale,obs_t,ood_guardian_flag,ood_probability,predicted_C25,reward_t,safety_budget,safety_slack,seed,t,true_s_safe_evaluator_only,unsafe_next,x_true_next,x_true_t",action_probabilities,"utility_t,penalty_flag_t,r_setpoint_t,r_pos_t,r_mort_t,k_t,m_t,regime_z_t,regime_switched_t,allee_C_episode,theta_exponent_episode,surrogate_reward_t"
B2_amur_tiger_ricker_s0p1,refplan,refplan,1000,20,50,"action_t,argmax_lambda0,belief_entropy,belief_mean,best_sequence_mean,best_sequence_reflected,best_sequence_return_sd,cost_t,model_entropy,obs_t,posterior_max,public_extinction_risk,reward_t,seed,t,unsafe_next,x_true_next,x_true_t","action_mean_lambda0,action_scores,member_returns,model_posterior","utility_t,penalty_flag_t,r_setpoint_t,r_pos_t,r_mort_t,k_t,m_t,regime_z_t,regime_switched_t,allee_C_episode,theta_exponent_episode,surrogate_reward_t"
B3_amur_tiger_allee_s0p2,bamcts,bamcts,1000,20,50,"action_t,cost_t,max_depth,obs_t,reward_t,seed,t,tree_nodes,tree_nodes_trace,unsafe_next,x_true_next,x_true_t","root_model_belief,root_q,root_visit_counts","utility_t,penalty_flag_t,r_setpoint_t,r_pos_t,r_mort_t,k_t,m_t,regime_z_t,regime_switched_t,allee_C_episode,theta_exponent_episode,surrogate_reward_t"
B3_amur_tiger_allee_s0p2,evd,ensemble_value_disagreement_pessimism,1000,20,50,"action_t,argmax_lambda0,argmax_pessimistic,cost_t,obs_t,reward_t,seed,t,unsafe_next,x_true_next,x_true_t","pessimistic_q_score,q_ensemble_mean,q_ensemble_variance","utility_t,penalty_flag_t,r_setpoint_t,r_pos_t,r_mort_t,k_t,m_t,regime_z_t,regime_switched_t,allee_C_episode,theta_exponent_episode,surrogate_reward_t"
B3_amur_tiger_allee_s0p2,ogsrl,ogsrl,1000,20,50,"action_t,behavior_normalized_cost,cost_horizon_used,cost_t,guardian_override,hard_fallback,lambda_ood,lambda_safety,low_abundance_scale,obs_t,ood_guardian_flag,ood_probability,predicted_C25,reward_t,safety_budget,safety_slack,seed,t,true_s_safe_evaluator_only,unsafe_next,x_true_next,x_true_t",action_probabilities,"utility_t,penalty_flag_t,r_setpoint_t,r_pos_t,r_mort_t,k_t,m_t,regime_z_t,regime_switched_t,allee_C_episode,theta_exponent_episode,surrogate_reward_t"
B3_amur_tiger_allee_s0p2,refplan,refplan,1000,20,50,"action_t,argmax_lambda0,belief_entropy,belief_mean,best_sequence_mean,best_sequence_reflected,best_sequence_return_sd,cost_t,model_entropy,obs_t,posterior_max,public_extinction_risk,reward_t,seed,t,unsafe_next,x_true_next,x_true_t","action_mean_lambda0,action_scores,member_returns,model_posterior","utility_t,penalty_flag_t,r_setpoint_t,r_pos_t,r_mort_t,k_t,m_t,regime_z_t,regime_switched_t,allee_C_episode,theta_exponent_episode,surrogate_reward_t"
```

## M1–M15 comparison

`m1_m15_comparison.csv` has one row per cell×method and one JSON-valued column for each M1–M15. Every unavailable cell contains `{"available":false,"reason":"..."}`. Detailed nested results are also in `metrics_normalized.json`.

## Required surfaced results

### Tier-B trajectory M13

- B1, B2, B3 RefPlan/OGSRL/BA-MCTS: **unavailable** — `surrogate_reward_t` is not logged, so visited-state surrogate error cannot be calculated.
- B1, B2, B3 EVD: **unavailable by design** — EVD uses raw dataset rewards and has no surrogate trajectory.

### Per-fox-cell PLUS M3 unanimity

| cell | pairwise agreement | fraction unanimous | modal share |
|---|---:|---:|---:|
| A1 | 0.651464 | 0.168000 | 0.774125 |
| A2 | 0.404821 | 0.002000 | 0.604750 |
| A3 | 0.464857 | 0.002000 | 0.668000 |
| A4 | 0.516821 | 0.098000 | 0.629625 |

### A5 PLUS M2

- `switch_vs_MAP = 0.000000`
- `switch_vs_uniform = 0.000000`

## Acceptance criterion 4 — independent return reconstruction

- Reconstructed discounted returns directly from per-step `reward_t` logs.
- PASS: **24/24** at `1e-9`.
- Maximum absolute difference from accepted `operational_return_mean`: **5.68434188608e-14**.
- Raw check: `acceptance_criterion_4.csv`.

## Figures and raw plotting data

- `fig_headroom.png` ← `headroom.csv`
- `fig_posterior_entropy_switch.png` ← `posterior_entropy_switch.csv`
- `fig_raw_vs_centred.png` ← `candidate_disagreement.csv`
- `fig_return_decomposition.png` ← `return_decomposition.csv`

Tier B return components are marked unavailable because its logs contain `reward_t` but not trajectory utility/penalty components; they are not inferred.
