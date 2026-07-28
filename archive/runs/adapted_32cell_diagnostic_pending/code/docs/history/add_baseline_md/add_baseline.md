Act as a rigorous Principal Research Software Engineer specializing in Computational Ecology, POMDPs, and adaptive management.

I added these references:

- docs/BioConserv18_Adaptive management of ecological systems under partial observability.pdf
- docs/Supp_BioConserv18_Adaptive management of ecological systems under partial observability.pdf

Goal: implement a BioConserv18 / PLUS-style baseline in `claude_build` so it can be compared fairly with the existing methods: ensemble/MOPO, MOReL, COMBO, CQL, IQL, and hmMDP where applicable.

Important context:
The paper’s method is PLUS: Planning and Learning for Uncertain Systems. It combines:
1. A finite candidate model set over uncertain ecological dynamics.
2. A belief/posterior over candidate models.
3. Bayesian updating after each new observation.
4. Planning by choosing the action maximizing posterior-expected model-specific Q-values, approximately:
   a* = argmax_a E_theta[Q_theta(a, b)]
   where posterior weights over theta are updated by observation likelihood.
5. In the supplement, model posterior update is:
   posterior(theta | z_{t+1}) proportional to P(z_{t+1} | b_t, a_t, theta) * prior(theta).
6. In our current repo, observations are exact discrete abundance bins, so the POMDP state-belief part collapses to the observed state, while the hidden model belief remains meaningful. Implement this as a PLUS-style finite candidate hidden-model baseline, not as a generic black-box learner.

Repository constraints:
- Work only inside `claude_build` unless updating top-level docs is truly necessary.
- Do not modify `baseline_original`.
- Do not commit or regenerate tracked outputs, checkpoints, `.pyc`, or cache files.
- Use the existing `BaseModel` API and registry pattern.
- Respect the new Ricker environment semantics:
  - hidden per-episode `r_base`
  - action records with `delta_r`, `delta_K`, and `cost`
  - reward contract `R(x_t, a_t) = alpha * x_t / x_max - cost[a]`
  - no action masking
- Use `env.reward_contract` as the reward source of truth. Do not duplicate reward tables unless unavoidable, and validate any copied values.

Implementation target:
Add a new model adapter, probably:

- `src/models/bioconserv18_plus_adapter.py`
- config: `config/model/bioconserv18_plus.yaml`
- registry key: `bioconserv18_plus`

The adapter should implement `BaseModel`:

- `fit_offline(...)`: no neural training required. It may precompute transition matrices and value/Q tables from the current env config/reward contract. Return a small metrics dict.
- `select_action(history, planner=None, reward_fn=None)`: ignore external planner; rebuild/update the posterior over candidate models from the observed history; choose the action maximizing posterior-weighted candidate Q-values.
- `predict(history)`: return the posterior-mixture next-state distribution for the last `(state, action)` in history and an uncertainty metric, e.g. posterior variance of expected next state across candidate models.
- `update(...)`: either no-op or update cached posterior if compatible with the online interface; document clearly. Prefer rebuilding posterior from history in `select_action` for consistency with `HmMDPAdapter`.
- `evaluate(...)`: report negative log likelihood / accuracy under the posterior-mixture transition model.

Candidate model design:
- Default candidate grid should be over `r_base`, e.g. 21 evenly spaced values from env `r_base_low` to `r_base_high`.
- Allow explicit `candidate_r_values` in config.
- For each candidate `r_base`, build `P_theta[x, a, x_next]`.
- Transition computation should match `RickerEnv.true_transition_probs(..., r_base=theta)` or the same underlying integration over continuous states inside a bin.
- Use the active env’s action specs, `K_base`, `delta_r`, `delta_K`, `num_states`, `bin_width`, and `max_abundance`.
- Avoid stale duplication: inject the resolved env config or env-derived metadata from `run_pipeline.py` / `train.py` when constructing this model if needed.

Planning:
- For each candidate theta, solve a finite-state MDP over observed abundance bins using value iteration:
  Q_theta[x, a] = R(x, a) + gamma * sum_xnext P_theta[x, a, xnext] V_theta[xnext]
- Use config options:
  - `candidate_r_values` or `num_candidate_models`
  - `discount`
  - `value_iteration_max_iter`
  - `value_iteration_tol`
  - `transition_n_grid`
  - `posterior_floor`
- At decision time:
  - reconstruct posterior over theta from history using Bayes rule and transition likelihoods
  - choose argmax_a sum_theta posterior[theta] * Q_theta[current_state, a]

History reconstruction:
At decision time, evaluator passes `committed + [(current_state, 0)]`.
Use consecutive observed states in this history to reconstruct transitions:
`(x_t, a_t, x_{t+1})` where `x_{t+1}` is the next entry’s state.
Do not use hidden continuous state or hidden true `r_base`.

Baseline naming:
Use clear naming such as `BioConserv18_PLUS_RGrid21` or similar.

Integration:
- Add config/model entry.
- Register in `src/models/__init__.py`.
- Ensure `run_pipeline.py`, `train.py`, and `evaluate.py` can build and evaluate it.
- Ensure action-space validation still applies.
- Ensure hmMDP baseline is skipped if action counts mismatch.
- Add CLI example in README or a short docs note:
  `python scripts/run_pipeline.py active_model=bioconserv18_plus model=bioconserv18_plus`

Tests:
Add focused tests, not huge simulations:

1. Candidate transition matrices have shape `[M, S, A, S]` and rows sum to 1.
2. Posterior update increases weight on the candidate model that assigns higher likelihood to the observed transition.
3. `select_action` returns a valid action and respects the shared reward contract.
4. `predict` returns a normalized mixture distribution and nonnegative uncertainty.
5. A tiny evaluator smoke test runs `bioconserv18_plus` on the 5-action Ricker env.
6. Optional: a 10-action config smoke test with `env=ricker_full model.num_actions=10` or equivalent.
7. Dataset-free operation: `fit_offline` should not require neural training data, but should accept the standard call signature.

Scientific caveat:
Do not claim this is an exact reproduction of the R package or SARSOP implementation unless you actually implement that. Label it as a faithful finite-state PLUS-style baseline adapted to this repo’s exact-observation Ricker setting. The exact POMDP/SARSOP component from the paper reduces to MDP planning here because the current environment exposes exact discrete abundance observations; the remaining uncertainty is the hidden model parameter.

Before finalizing:
- Run the relevant unit tests if the environment has dependencies.
- If dependencies are missing, say exactly what could not be run.
- Report modified files.
- Do not leave generated artifacts or `.pyc` changes in the diff.