You are a rigorous Principal Research Software Engineer specializing in offline model-based RL and ecological decision-making.

In this repo, implement the 3 new general DeepRL baselines documented in:

- docs/general_model_based_offline_RL/ICLR26_Bayes Adaptive Monte Carlo Tree Search for Offline Model-based Reinforcement Learning.pdf
- docs/general_model_based_offline_RL/ICLR26_Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting.pdf
- docs/general_model_based_offline_RL/ICML25_Reflect-then-Plan- Offline Model-Based Planning through a Doubly Bayesian Lens.pdf

Target codebase: claude_build/

Tasks:
1. Read/extract the PDFs first. If PDF extraction is unavailable, stop and tell me exactly what tool is missing.
2. Summarize each baseline’s algorithmic core and decide the most faithful repo-native implementation for our finite discrete Ricker adaptive-management setting.
3. Implement each baseline as a new model/planner component compatible with the existing claude_build interfaces, reward contract, action-space validation, configs, evaluator, and SLURM workflow.
4. Register all new methods in src/models/__init__.py or src/planners/__init__.py as appropriate.
5. Add configs under config/model/ and/or config/planner/.
6. Add focused tests under tests/ that verify:
   - action-space compatibility for 5-action and 10-action envs,
   - reward uses env.reward_contract,
   - fit/evaluate/select_action run on a small synthetic dataset,
   - save/load works,
   - configs compose with Hydra.
7. Add/extend scripts so these methods can be run in matched comparison against:
   ensemble/MOPO, MOReL, COMBO, CQL, IQL, BioConserv18_PLUS.
8. Do not print, modify, or commit claude_build/.wandb_env. It is a local ignored secret file.
9. Keep changes scoped to claude_build and tests/docs needed for usage. Do not revert unrelated user changes.

Important existing semantics:
- Ricker env reward is R(x_t,a_t)=alpha*x_t/x_max-cost[a], via env.reward_contract.
- r_base is hidden per episode.
- 5-action default and 10-action ricker_full variant must both be supported where applicable.
- hmMDP baseline is fixed 4-action and should remain skipped on mismatched envs.

Before coding, give me a concise implementation plan mapping each new baseline to exact files/classes/configs/tests. Then implement, run the smallest meaningful tests/smokes available, and report results.