Act as a rigorous Principal Research Software Engineer specializing in Computational Ecology, model-based offline RL, and POMDP-based adaptive management.

I added this reference:

- docs/ExpertSys23_Model-based offline reinforcement learning for sustainable fishery management.pdf
  (Ju, Kurniawati, Kroese & Ye, *Expert Systems* 2025; the method is called **MOOR**.)

Goal: implement a **MOOR-style mechanistic model-based offline RL baseline** in `claude_build` so it can be compared fairly with the existing methods: ensemble/MOPO, MOReL, COMBO, CQL, IQL, BioConserv18_PLUS, BAMCTS, ROMI, RefPlan, and hmMDP where applicable.

This baseline is **genuinely distinct** from everything already in the repo. Do not collapse it into PLUS or the ensemble; read §"Why this is a new baseline" before coding.

---

## 1. What MOOR actually is (read the PDF, then map it)

MOOR = *Model-based Offline RL* for fishery management. Its pipeline (paper §4, Algorithms 1–2) is:

1. **POMDP formulation.** State = unknown population biomass `B`; action = fishing effort `e`; observation = catch `c`; reward = catch `c`. Transition is a mechanistic growth model minus catch with multiplicative log-normal noise:
   `B_{t+1} = (f(B_t; K, ρ) − g(B_t, e_t; q)) · e^{ε_t}`, `ε_t ~ N(0, σ²)`,
   growth `f` is Beverton–Holt (`BH-POMDP`) or a surplus-production/Schaefer model (`SP-POMDP`); catch `g(B,e;q) = q·e·B`.
2. **Model learning (the paper's signature contribution).** Learn the *physical* parameters `θ = (ρ, K, B₀, q)` (BH) or `(r, K, B₀, q)` (Schaefer) directly from the offline `(catch, effort)` series by **minimizing a sum-of-squared-error (SSE) objective** between predicted and observed catches (Eq. 9–10), optimized with **stochastic L-BFGS** plus two tricks: (a) normalize catch/effort to `[0,1]` and initialize all params near 1; (b) **multiple random restarts**, keep the model with smallest SSE. (Eq. 11 un-normalizes.)
3. **Model discretization (paper §4.5, Algorithm 2; Filar et al. Monte-Carlo).** For each discrete state bin `s` and discrete action `a`, sample `N` continuous states inside the bin, simulate the *learned* transition+observation models forward, and form **empirical** `T̂(s'|s,a)`, `Ô(z|s',a)`, `R̂(s,a)`.
4. **Planning.** Solve the learned discrete POMDP with an online solver (the paper uses DESPOT, γ=0.95), then **evaluate the resulting policy in the ground-truth model**.

The paper's headline findings (these become our reporting targets, §9):
- Policies are **robust to model-learning error** — a *misspecified* (Schaefer) model fit to Beverton–Holt data still yields good policies, because only the *functionally relevant (low-biomass) region* needs to be learned well.
- **Identifiability** requires sufficient **action/effort variability** in the data; otherwise the learned model and policy degrade.

---

## 2. Why this is a new baseline (do not merge it into an existing one)

| Existing method | What it does | Why MOOR differs |
|---|---|---|
| ensemble / MOPO / MOReL | learns a **categorical neural** transition model, plans pessimistically | MOOR learns a **mechanistic, physically-parameterized** dynamics model (a handful of scalars) by **least squares**, not a neural net |
| BioConserv18_PLUS | **oracle finite candidate grid** over `r_base` + Bayesian posterior; **no learning from data** | MOOR **learns a point estimate of the dynamics parameters from the offline dataset**; no candidate grid, no posterior |
| COMBO / CQL / IQL | conservative / model-free deep RL | MOOR is mechanistic model-based |
| BAMCTS / ROMI / RefPlan | tabular Bayes-adaptive / robust / posterior-trajectory planners | none of them *fit a mechanistic growth model by SSE*, and none support a **misspecified model family** |

MOOR fills the **"mechanistic / parametric model-based offline RL"** slot, and — uniquely — adds a **well-specified vs. misspecified model** axis (Ricker-family vs. Schaefer/surplus-production), which is the paper's central experimental contrast.

---

## 3. Faithful adaptation to THIS repo's exact-observation Ricker setting

Read `docs/REPO_BRIEF.md` §3 (the three core contracts) and `src/models/bioconserv18_plus_adapter.py` (closest structural analog) before writing code.

Key environment semantics you MUST respect (see `src/environments/ricker_env.py`, `reward.py`):
- True dynamics: `s_{t+1} = s_t · exp(r_eff·(1 − s_t/K_eff))`, `r_eff = r_base + Δr(a)`, `K_eff = K_base + ΔK(a)`.
- `r_base` is **hidden, drawn per-episode** from `U(r_base_low, r_base_high)`, fixed within an episode, never logged.
- The agent observes only the **exact discrete abundance bin** `x_t = floor(s_t / bin_width)`. There is **no separate catch or effort series** — observations are the population state itself.
- Reward is the shared first-class contract `R(x_t, a_t) = alpha·x_t/x_max − cost[a]` (current state, not next). Use `env.reward_contract` as the single source of truth — **do not** build a catch-based reward.

How MOOR's four steps map here:

1. **POMDP → MDP collapse.** Because observations are exact bins, the state-belief part of the POMDP collapses to the observed bin (same caveat as PLUS). The remaining model uncertainty is the *dynamics parameters*, which MOOR learns. So the planner is **value iteration on the learned discrete MDP**, not DESPOT. Label this honestly (§10).

2. **Borrow MOOR's MODEL-LEARNING methodology, keep the repo's action/reward semantics.** There is no `effort e` or `catch c = q·e·B` here; the action is a discrete management intervention `(Δr, ΔK, cost)` and the reward is abundance-based. So:
   - Keep the repo's discrete actions and `env.reward_contract` unchanged.
   - Replace MOOR's *catch* SSE with the repo-native observed quantity: SSE between the **predicted next abundance** and the **observed next abundance** over the offline `(x_t, a_t, x_{t+1})` dataset.

3. **Two model families (the headline experiment), selected by config `dynamics_model`:**
   - `dynamics_model: ricker` → **well-specified**: learn Ricker growth params; same family as the env.
   - `dynamics_model: schaefer` → **misspecified**: fit a surplus-production / Schaefer-style logistic growth model `f(s) = s + r·s·(1 − s/K)` to Ricker-generated data. Action perturbations apply the **same additive** way: `r_eff = r̂ + Δr(a)`, `K_eff = K̂ + ΔK(a)`. This reproduces the paper's BH-POMDP vs SP-POMDP contrast.
   Default to `ricker` for the primary run; expose `schaefer` for the misspecification ablation.

4. **Parameters to learn.** The paper learns `(ρ/r, K, B₀, q)`. `B₀` (initial biomass) and `q` (catchability) are observation-model params with **no analog** here (exact state observations, no catch). The faithful learnable subset is the **growth parameters**:
   - `learn_params: [r]` → learn growth rate only, hold `K = env.K_base` (simplest; closest to the env's single hidden parameter `r_base`).
   - `learn_params: [r, K]` → learn both growth rate and carrying capacity (default).
   The learned `r` is an **effective point estimate** of the central growth rate (the true `r_base` is hidden and varies per episode — MOOR accepts this model-learning error by design).

---

## 4. Implementation target

Add a new model adapter mirroring the PLUS adapter's structure and constructor contract:

- `src/models/moor_adapter.py` — class `MOORAdapter(BaseModel)`
- `config/model/moor.yaml`
- registry key: `moor`

Constructor signature (match PLUS exactly so the script injection works):
```python
def __init__(self, cfg, env=None, env_cfg=None, reward_contract=None): ...
```
- Resolve `env_cfg` exactly as PLUS does (require it; accept it as a constructor kwarg OR as `cfg["env_cfg"]`; raise a clear `ValueError` mentioning `env_cfg` if missing — see `test_requires_env_cfg`).
- Pull `num_states, max_abundance, bin_width, K_base, r_base_low, r_base_high, actions, reward.{alpha,x_max}` from `env_cfg`.
- Build the reward table from `RewardContract.from_specs(...)`; if an `env`/`reward_contract` is supplied, validate it matches (same strict handshake as PLUS `_assert_reward_contract_matches`) and adopt it as the source of truth.
- Validate action specs, `K_eff > 0` per action, and (if `env` given) that `env` matches `env_cfg` (`num_states`/`num_actions`/`action_specs`).

Implement the `BaseModel` API (`src/interfaces/base_model.py`):

- **`fit_offline(dataset, train_cfg=None, wandb_cfg=None, seed=None, reward_fn=None) -> dict`** — this is where the real work happens (unlike PLUS, which precomputes at `__init__`):
  1. Validate optional `reward_fn` against the env contract (reuse PLUS's `_validate_optional_reward_fn`).
  2. Build continuous-state proxies from discrete bins using bin midpoints: `s_mid(x) = (x + 0.5)·bin_width`. Inputs `s_t = s_mid(x_t)`, targets `y_t = s_mid(x_{t+1})`, with the per-row action `a_t`.
  3. **Least-squares fit** of `learn_params` by minimizing normalized SSE
     `L(θ) = Σ_t ( ŝ_{t+1}(θ; s_t, a_t)/x_max − y_t/x_max )²`
     where `ŝ_{t+1}` is the learned-family one-step map with `r_eff = r̂ + Δr(a)`, `K_eff = K̂ + ΔK(a)`. Optimize with **L-BFGS** (`torch.optim.LBFGS`, closure-based — torch is already a dependency via the ensemble; `scipy.optimize.minimize(method="L-BFGS-B")` is an acceptable alternative) using **`n_restarts` random initializations** near the normalized prior; keep the θ with smallest SSE. This is MOOR's defining step — implement it, don't shortcut to a grid search (you may add a coarse grid only to *seed* restarts).
  4. Store learned θ. Build the discretized transition tensor `T̂[s,a,s']` by Monte-Carlo over a sub-grid of each bin using the **learned** dynamics (reuse PLUS's `_build_transition_matrices` math, substituting `r̂`/`K̂` and the chosen growth family; honor `transition_n_grid`). Optionally inject multiplicative log-normal noise `e^{ε}, ε~N(0,σ²)` during sampling if `transition_noise_sigma > 0` (default 0 — bin-aliasing already yields a non-degenerate `T̂`).
  5. Solve the learned MDP by **value iteration** (reuse PLUS's `_solve_all_mdps` for the single learned model) → `Q̂`, `V̂`, honoring `discount`, `value_iteration_max_iter`, `value_iteration_tol`.
  6. Set a fitted flag and return a small metrics dict, e.g. `{"algo": self.name, "dynamics_model": ..., "learn_params": ..., "learned_r": ..., "learned_K": ..., "train_sse": ..., "n_restarts": ..., "vi_iters": ...}`.

- **`select_action(history, planner=None, reward_fn=None) -> int`** — ignore the external planner; validate `reward_fn`; guard with an `_assert_fitted_for_decision()` that raises if `fit_offline` has not run (mirror BAMCTS/ROMI/RefPlan pre-fit guard, REPO_BRIEF §5); return `int(argmax_a Q̂[x_current, a])` where `x_current = history[-1][0]`.

- **`predict(history) -> (probs[num_states], uncertainty)`** — for the last `(x, a)` in history return the learned `T̂[x, a, :]` (normalized) and an uncertainty scalar. Single point-estimate ⇒ default `uncertainty = 0.0`; optionally report the **restart-disagreement variance** of the predicted expected next state across the kept restarts (nonnegative) so it is comparable to PLUS's posterior-variance metric.

- **`update(states, actions, next_states) -> {"loss": 0.0}`** — no-op. MOOR is purely offline (learn once, then plan); document this clearly, like PLUS.

- **`evaluate(states, actions, next_states) -> {"loss": NLL, "accuracy": acc}`** — one-step held-out NLL and argmax-accuracy of the learned `T̂` (reuse PLUS's `evaluate` structure with the single learned model instead of the prior mixture).

- **`save(path)` / `load(path)`** — persist learned θ + config-defining fields (`dynamics_model`, `learn_params`, grid sizes); on load, validate shapes/finiteness and rebuild `T̂`/`Q̂` (or store and reload them). Mirror PLUS's `save`/`load` validation discipline.

- **Metadata:** `name`, `num_states`, `num_actions`. Name should encode the family and that it is least-squares-learned, e.g.
  `f"MOOR_{self._dynamics_model.capitalize()}_LS"` → `MOOR_Ricker_LS` / `MOOR_Schaefer_LS`.

---

## 5. Config (`config/model/moor.yaml`)

Follow the PLUS yaml's documented style. Suggested keys:
```yaml
_target_: src.models.moor_adapter.MOORAdapter

# Mechanistic growth family fit to the offline data.
#   ricker   -> well-specified (same family as the env)
#   schaefer -> misspecified surplus-production model (paper's SP-POMDP)
dynamics_model: ricker
learn_params: [r, K]          # subset of {r, K}; B0/q have no analog here

# Least-squares model learning (MOOR's signature step).
n_restarts: 30                # paper runs many random restarts; keep best SSE
lbfgs_max_iter: 20
lbfgs_lr: 0.3
init_jitter: 0.25             # random init spread around the normalized prior
transition_noise_sigma: 0.0  # optional log-normal noise during MC discretization

# Planning / discretization (shared with PLUS semantics).
discount: 0.95
transition_n_grid: 2000
value_iteration_max_iter: 1000
value_iteration_tol: 1.0e-8
likelihood_floor: 1.0e-12     # smooths zero transition mass in NLL/Bayes-free eval
# num_states / num_actions are derived from the active env config.
```
Document, as PLUS does, that this is a **repo-native finite-state, least-squares MOOR adaptation**, not an exact reproduction of the authors' PyTorch+DESPOT/R code.

---

## 6. Integration

- Register in `src/models/__init__.py`: import `MOORAdapter`, `MODEL_REGISTRY.register("moor", MOORAdapter)`, add to `__all__`.
- **`env_cfg` injection (critical, REPO_BRIEF §5).** Scripts only inject `env`/`env_cfg`/`reward_contract` for `active_model == "bioconserv18_plus"`. Extend that branch to a membership test in **all four** scripts:
  - `scripts/core/run_pipeline.py` (~line 164)
  - `scripts/core/train.py` (~line 150)
  - `scripts/core/evaluate.py` (~line 65)
  - `scripts/core/online_rollout.py` (the analogous build site)
  Change `if active_model == "bioconserv18_plus":` → `if active_model in ("bioconserv18_plus", "moor"):` so `moor` receives `env=, env_cfg=, reward_contract=`.
- `fit_offline` is already the uniform offline-training entry point in `run_pipeline.py`/`train.py` (they call `model.fit_offline(...)` when present) — no extra wiring needed beyond the injection branch.
- Action-space validation (`validate_action_space`) applies uniformly; MOOR derives `num_actions` from `env_cfg`, so it passes for 5- and 10-action envs. Ensure the hmMDP baseline stays skipped on action-count mismatch (unchanged).
- Add a short usage note (README and/or a `scripts/single_runs/run_moor_single.sh` mirroring `run_bioconserv18_plus_single.sh`), and optionally add `moor` as a matched arm in `scripts/slurm/slurm_general_mbrl_baselines.sh` / `slurm_model_comparison.sh` alongside the other model-based methods.

CLI examples to support:
```bash
# 5-action default
python scripts/core/run_pipeline.py active_model=moor model=moor

# 10-action variant
python scripts/core/run_pipeline.py +experiment=full_action_set active_model=moor model=moor model.num_actions=10

# misspecified-model ablation (paper's SP-POMDP analog)
python scripts/core/run_pipeline.py active_model=moor model=moor model.dynamics_model=schaefer
```

---

## 7. History reconstruction

At decision time the evaluator passes `committed + [(current_state, 0)]` (REPO_BRIEF §3.3). MOOR's policy is the learned MDP's greedy action, so `select_action` only needs `history[-1][0]` (the current bin). Do **not** read hidden continuous state or hidden true `r_base`. (No posterior to rebuild — that is PLUS-specific.)

---

## 8. Tests (`tests/test_moor.py`)

Focused tests, not big simulations. Build the adapter directly with an `ENV_CFG`/`MODEL_CFG` dict pair like `tests/test_bioconserv18_plus.py`. Cover:

1. `fit_offline` on a tiny synthetic Ricker dataset learns a finite `r̂` (and `K̂`) and reports a finite `train_sse`; flips a fitted flag.
2. Learned transition tensor has shape `[S, A, S]` and each `T̂[s,a,:]` sums to 1.
3. **Recovery sanity:** on data generated by `RickerEnv` with a *narrow* `r_base` prior and `dynamics_model: ricker`, the learned `r̂` lands near the prior's central `r_base` (loose tolerance — it is an effective point estimate over a hidden, varying parameter).
4. `select_action` returns a valid action, respects the shared reward contract (identical-dynamics-but-cheaper-action ⇒ prefer it, as in PLUS's test), and **raises before `fit_offline`** (pre-fit guard).
5. `predict` returns a normalized distribution over `num_states` and nonnegative uncertainty.
6. `evaluate` returns `{"loss" ≥ 0, "accuracy" ∈ [0,1]}`.
7. Conflicting `reward_fn` (different costs) raises `ValueError` (strict handshake).
8. `save`/`load` round-trips the learned model; loaded model reproduces the same `select_action` / `T̂`.
9. `dynamics_model: schaefer` fits and runs end-to-end (misspecified family works).
10. A tiny evaluator smoke test runs `moor` on the 5-action Ricker env; optionally a 10-action smoke (`env=ricker_full model.num_actions=10`).
11. `env_cfg` required: omitting it raises a `ValueError` mentioning `env_cfg` (mirror PLUS).

Smoke (cheap) before finalizing:
```bash
PY=/fs04/scratch2/ce25/hphung/conda/envs/pytorchrl/bin/python
$PY claude_build/scripts/core/run_pipeline.py active_model=moor model=moor \
  env.dataset.n_transitions=400 env.dataset.episode_len=10 \
  eval.n_episodes=2 eval.horizon=5 wandb.mode=disabled
$PY -m pytest claude_build/tests/test_moor.py -q
```

---

## 9. Scientific reproduction targets (report these)

These let the baseline actually exercise the paper's claims, not just run:
- **Model error vs. policy quality.** Report both the learned model's held-out NLL/accuracy (`evaluate`) AND the evaluation cumulative reward. The paper's point: policy can be good even when model error is non-trivial.
- **Well-specified vs. misspecified.** Run `dynamics_model: ricker` and `dynamics_model: schaefer` on the same data; report the reward gap. Expectation (paper §5.2.4): the misspecified Schaefer model stays competitive because only the low-abundance region matters.
- **(Optional) Identifiability vs. data variability.** The default `collection_policy: random` gives high action variability (the "good" regime). An optional ablation with a low-variability collection policy should degrade the learned model/policy (paper §5.2.5). Treat as optional — do not add new env code just for this.
- **(Optional) Incomplete data.** The paper studies missing values; the repo's dataset is complete `(x,a,x')`. If you want to mirror it, add an *optional* `fit_offline` path that randomly drops a fraction of transitions before fitting. Optional, clearly labeled.

---

## 10. Repository constraints (do not violate)

- Work only inside `claude_build` plus this `docs/` note. Do not modify `baseline_original`.
- Do not commit or regenerate tracked outputs, checkpoints, `.pyc`, `wandb/`, or cache files. After any smoke run, restore touched artifacts (`git checkout HEAD -- <path>`); keep the diff to source/config/test files only.
- Do not print, modify, or commit `claude_build/.wandb_env` (gitignored secret).
- Use the existing `BaseModel` API, the registry pattern, and `env.reward_contract` as the only reward source. Do not duplicate reward tables; if you copy any value, validate it against the contract.
- Reuse PLUS's MC-discretization and value-iteration math rather than re-deriving it; the novel code is the **least-squares parameter learning** and the **two model families**.

---

## 11. Scientific caveat (state this in the adapter docstring and config)

Label this a **faithful, repo-native, finite-state MOOR-style baseline**, NOT an exact reproduction of Ju et al.'s implementation. Be explicit about what does and does not transfer:
- **Transfers:** the least-squares mechanistic model-learning objective (SSE on the observed quantity), stochastic L-BFGS with normalization + multiple restarts, Monte-Carlo discretization of the learned model, plan-in-learned-model-then-evaluate-in-ground-truth, and the well-specified vs. misspecified contrast.
- **Does not transfer (state why):** there is no DESPOT/SARSOP solver (exact bin observations collapse the POMDP to an MDP, so value iteration is the correct analog); no `catch = q·e·B` observation model and no `B₀`/`q` parameters (observations are exact abundance bins, reward is the repo's abundance-cost contract); no missing-data imputation by default (the offline dataset is complete `(x,a,x')` tuples).

---

## 12. Before finalizing

- Run `tests/test_moor.py` and the tiny `run_pipeline` smoke above. If a dependency (e.g. scipy/torch L-BFGS) is missing, say exactly what could not be run.
- Confirm 5-action and 10-action both build and evaluate, and that hmMDP stays skipped on mismatch.
- Report every modified/created file. Do not leave generated artifacts or `.pyc` changes in the diff.
