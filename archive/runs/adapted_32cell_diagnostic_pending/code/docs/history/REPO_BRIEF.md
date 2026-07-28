# REPO_BRIEF — orientation for any new chat about this repo

**How to use this file.** Start a fresh chat with: *"Read `docs/REPO_BRIEF.md`, then explain / audit / modify `<thing>`."* This document gives just enough to find the right code without re-deriving the architecture. It is intentionally compact; read the linked files for detail.

---

## 1. One-line problem statement

Offline model-based reinforcement learning for ecological adaptive management on a discretized Ricker population model, with **hidden per-episode growth rate `r_base`** and **additive `(Δr, ΔK)` management actions** that carry a **monetary cost**.

The formal definition lives in `docs/problem_setting.tex` (read this first if you need the math). Supplementary detail: `docs/problem_setting_experiment_details.tex`.

---

## 2. Repo layout

```
/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/    # working dir
├── docs/                                                # problem spec + audits + plans
│   ├── problem_setting.tex                              # formal MDP/POMDP definition
│   ├── add_baseline.md, add_baseline_deeprl.md          # specs that codex implemented
│   ├── claude_audit_report.md                           # most recent audit by Claude
│   ├── codex_claude_audit_recheck.md                    # codex's pushback on audit items
│   ├── codex_experiment_plan.md                         # the active experiment plan
│   └── REPO_BRIEF.md                                    # ← this file
└── claude_build/                                        # ALL code lives here
    ├── config/                                          # Hydra configs
    │   ├── config.yaml                                  # top-level: active_env / model / planner
    │   ├── env/                                         # ricker.yaml (5-action), ricker_full.yaml (10-action)
    │   ├── model/                                       # one yaml per registered model
    │   ├── planner/                                     # pessimistic.yaml, morel.yaml
    │   └── experiment/                                  # Hydra experiment groups (full_action_set + ablations)
    ├── src/
    │   ├── environments/ricker_env.py, reward.py        # env + shared reward contract
    │   ├── interfaces/base_env.py, base_model.py        # ABCs
    │   ├── models/                                      # all 9 model adapters + shared tabular utils
    │   ├── planners/pessimistic_planner.py, morel_planner.py
    │   ├── training/offline_trainer.py                  # dynamics-ensemble training loop
    │   ├── evaluation/evaluator.py                      # UnifiedEvaluator
    │   ├── registry.py                                  # MODEL/ENV/PLANNER registries + validate_action_space
    │   └── utils/local_env.py                           # tiny .wandb_env loader
    ├── scripts/
    │   ├── core/{run_pipeline,train,evaluate,online_rollout,generate_dataset}.py
    │   ├── experiments/{make_experiment_manifest,run_manifest_row}.py + submit_experiment_manifest.sh
    │   ├── analysis/{aggregate_experiments,make_figures}.py
    │   └── slurm/                                       # phase[0..5]_*.sh, aggregate_*.sh, manifest runner body
    ├── tests/                                           # 8 test files; see §9
    └── outputs/                                         # generated artifacts (git-ignored)
```

---

## 3. The core contracts — read these first to understand any model

These are the load-bearing abstractions. Every model touches them.

### 3.1 The reward contract (single source of truth)

`src/environments/reward.py` defines `RewardContract(alpha, x_max, costs)` and `ActionSpec(id, name, delta_r, delta_K, cost)`. The reward is:

```
R(x_t, a_t) = alpha * x_t / x_max - cost[a_t]    # depends on CURRENT state, not next_state
```

The env builds the contract from its config and exposes `env.reward_contract`. **Every consumer of reward in the codebase reads this same object.** No duplication, no alternate paths. The planners, the evaluator, the model adapters all take a `reward_fn` parameter that is set to `env.reward_contract`.

### 3.2 The model API (`src/interfaces/base_model.py`)

Every model implements:

```python
predict(history)                                    -> (probs[num_states], uncertainty)
select_action(history, planner=None, reward_fn=None) -> int
fit_offline(dataset, train_cfg, wandb_cfg, seed, reward_fn) -> dict   # most adapters
update(states, actions, next_states)                -> {"loss": float}
evaluate(states, actions, next_states)              -> {"loss": float, "accuracy": float}
save(path) / load(path)
```

### 3.3 The history convention

```
history = committed + [(x_current, placeholder_action=0)]
```
- `committed[i] = (state, action-taken-FROM-that-state)` — NOT `(state, action-that-led-to-state)`.
- The placeholder action at the end is **never read** by `posterior_from_history` (the loop stops one short).
- The evaluator builds this in `_run_episode` ([evaluator.py](../claude_build/src/evaluation/evaluator.py)). If you're confused about a history-related bug, read that function.

### 3.4 Action-space validation

`src/registry.py::validate_action_space(env, model)` raises if `env.num_actions != model.num_actions`. Called in the evaluator constructor for every model. The hmMDP baseline has a fixed 4-action MOMDP and is **explicitly skipped (visibly logged)** when the env has 5 or 10 actions — never silently evaluated.

### 3.5 Hydra config selection

- `env=ricker` (default) vs `env=ricker_full` (10 actions). `active_env` stays `"ricker"` in both cases — it's the **registry key**, not a filename.
- `model=<name>` AND `active_model=<name>` together select an adapter. The `<name>` matches a yaml in `config/model/`.
- For 10-action runs: `env=ricker_full model.num_actions=10` (or `+experiment=full_action_set`).

---

## 4. Methods implemented (9 total, hmMDP exempted from comparisons)

| Method | File | Family | Notes |
|---|---|---|---|
| MOPO/MOReL ensemble | `src/models/ensemble.py` + `network.py` | Model-based | Categorical bootstrap ensemble; the project's flagship dynamics model |
| Pessimistic planner | `src/planners/pessimistic_planner.py` | Planner (MOPO) | Soft `-λ·var` penalty |
| MOReL planner | `src/planners/morel_planner.py` | Planner | Hard USAD halt |
| COMBO | `src/models/combo_model.py` | Model-based | Conservative Q on real + model rollouts |
| CQL | `src/models/cql_adapter.py` | Model-free | d3rlpy DiscreteCQL wrapper |
| IQL | `src/models/iql_agent.py` | Model-free | Native discrete (no d3rlpy DiscreteIQL exists) |
| BioConserv18 PLUS | `src/models/bioconserv18_plus_adapter.py` | Hidden-model planning | Finite candidate `r_base` grid + posterior + per-candidate VI |
| BAMCTS-Tabular | `src/models/bamcts_adapter.py` | Bayes-adaptive MCTS | Real recursive tree (not flat rollouts); coarse belief keys |
| ROMI-Tabular | `src/models/romi_adapter.py` | Robust MDP | Local-min uncertainty set + adaptive `eta_sa` |
| RefPlan-Tabular | `src/models/refplan_adapter.py` | Posterior trajectory opt. | Behavior-prior sequences scored by exp(κ·return) |
| HmMDP (AAAI21) | `src/models/hmmdp_adapter.py` | Baseline | 4-action; skipped on 5/10-action env |

The three "general MBRL" adapters (BAMCTS, ROMI, RefPlan) **share** `src/models/tabular_mbrl.py` — `TabularDynamicsEnsemble`, `value_iteration`, `nll_accuracy`, `require_shape`. The adapters are **honestly labeled tabular** (`BA_MCTS_Tabular_*` etc.) and are not faithful neural reproductions of the original papers. PLUS is also tabular.

---

## 5. Important conventions and gotchas

- **`+dataset_path=…`** is a Hydra override (note the `+`, the key isn't pre-declared). Both `scripts/core/train.py` and `scripts/core/run_pipeline.py` honor it: load if present, generate if not, then validate. Validation is `_validate_dataset_against_env` — catches **legacy datasets** (4-action / next-state-reward) by comparing stored rewards against `env.reward(state, action, next_state)`.
- **Strict reward-contract handshake** in BAMCTS / ROMI / RefPlan: passing a `reward_fn` whose costs differ from the one used at `fit_offline` raises `ValueError`. Object identity is cached, so calling repeatedly with the same `env.reward_contract` is O(1). PLUS uses the same strict check; CQL/IQL accept-and-ignore (they train on dataset rewards).
- **Pre-fit `select_action` raises.** BAMCTS / ROMI / RefPlan all call `_assert_fitted_for_decision()` first, which raises if `_dynamics.fitted` is False.
- **`env_cfg` is injected by scripts ONLY for `active_model=bioconserv18_plus`** — not for all models. The scripts branch on this. PLUS also takes optional `env=, reward_contract=` constructor kwargs which the scripts pass.
- **Generated artifacts** (`outputs/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`) are git-ignored. If a smoke/test touches a tracked one (older `outputs/` files predate the ignore rule), restore with `git checkout HEAD -- <path>`.
- **Working-directory resets between Bash tool calls** — use absolute paths or `cd /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/...` inline.
- **`num_states=100`, `x_max=100`** are coincidentally equal in the default config. The reward uses `x_max`; don't conflate them.
- **Episode boundary:** `terminate_on_extinction: false` by default — bin 0 is NOT terminal (Ricker can recover from `s ∈ (0, 10)`). Set true to use `s ≤ extinction_threshold` as the absorbing boundary.

---

## 6. How to run things

**Python interpreter (the project's conda env):**
```
PY=/fs04/scratch2/ce25/hphung/conda/envs/pytorchrl/bin/python
```

**Single-method pipeline (5-action default):**
```
$PY scripts/core/run_pipeline.py active_model=bamcts model=bamcts
```

**Single-method, 10-action variant:**
```
$PY scripts/core/run_pipeline.py env=ricker_full active_model=bamcts model=bamcts model.num_actions=10
```

**Or use the bundled experiment group:**
```
$PY scripts/core/run_pipeline.py +experiment=full_action_set active_model=bamcts model=bamcts
```

**Tests:**
```
$PY -m pytest claude_build/tests/ -q
$PY -m pytest claude_build/tests/test_<file>.py -v
```

**Manifest-driven experiment (intensive sweeps):**
```
$PY scripts/experiments/make_experiment_manifest.py --phase phase1 --output outputs/manifests/phase1.tsv --base .
$PY scripts/experiments/run_manifest_row.py --manifest outputs/manifests/phase1.tsv --dry-run --base .
$PY scripts/experiments/run_manifest_row.py --manifest outputs/manifests/phase1.tsv --row-index 0 --base .
```

SLURM phase launchers (`scripts/slurm/phase[0..5]_*.sh`) wrap this — they generate the manifest, dry-run, run row 0 inline, then submit the rest as a SLURM array. **They do NOT chain across phases via SLURM dependencies** — the user must wait between `sbatch` submissions or add `--dependency=afterok:$JOBID`.

---

## 7. Environment notes (HPC)

- **Cluster** uses environment modules; `module load miniforge3` then `conda activate pytorchrl`.
- **No PDF tooling** in the conda env — `pdftotext`/`poppler` are absent; for PDF audits use the textual `.md` specs in `docs/` instead.
- **`.wandb_env`** is a gitignored secret file. The Python helper `src/utils/local_env.py` and the shell snippets in `scripts/slurm/*.sh` only export `WANDB_*` / `WB_PROJECT` from it. **Never print or commit `.wandb_env`.**

---

## 8. Audit history — why things are the way they are

These decisions are documented because they're not obvious from grep:

- **Reward is a "first-class shared contract"** because an earlier audit (`docs/claude_audit_report.md` §A2) caught the planner using next-state reward without costs. Now every reward consumer threads `env.reward_contract` and any divergent contract raises.
- **`env_cfg` injection is PLUS-only** because an earlier audit caught it leaking into every model's config. Scripts now branch.
- **BAMCTS uses a real dict-backed tree with coarse belief buckets (`tree_belief_decimals=2`)** because the initial implementation was UCB+flat-MC with 8-decimal belief keys that never collided.
- **The three "deep RL" baselines (BAMCTS, ROMI, RefPlan) are tabular**, not faithful neural reproductions. Their adapter docstrings say so explicitly. They share `TabularDynamicsEnsemble`.
- **`tree_belief_decimals=2`, `prior_count=0.05`, `posterior_floor=1e-12`** are recent defaults — read the relevant adapter docstrings for the trade-offs.

The full back-and-forth: `docs/claude_audit_report.md` (Claude's audit) → `docs/codex_claude_audit_recheck.md` (codex's responses to each point). Read both if you're picking up a contested decision.

---

## 9. Tests (8 files, ~85 tests, ~3 min full run)

| File | Coverage |
|---|---|
| `test_ricker_env.py` | Env contract: reward formula, hidden-`r_base` invariants, dataset reproducibility, boundary modes, marginalized transitions |
| `test_ensemble.py` | `CategoricalMOPOEnsemble` API + bootstrap |
| `test_planner_reward.py` | Regression: identical transitions + different costs → planner prefers lower-cost (catches the pre-audit bug) |
| `test_offline_algos.py` | MOReL planner, CQL, IQL, COMBO smoke/fit/predict/select |
| `test_bioconserv18_plus.py` | PLUS adapter: candidate grid, posterior update, reward-contract match, checkpoint validation, prior shape |
| `test_general_mbrl_baselines.py` | BAMCTS/ROMI/RefPlan: fit/predict/select/save/load + pre-fit guard + contract mismatch + checkpoint shape + BAMCTS depth-3 subtree reuse |
| `test_evaluator.py` | `UnifiedEvaluator` end-to-end |
| `test_adapter.py` | hmMDP adapter (separate baseline) |

---

## 10. How to talk to me about this repo in future chats

**Good queries** (these route directly to a small read):
- *"Read `src/models/bamcts_adapter.py` and explain how `_simulate_tree` builds the tree. Use [REPO_BRIEF.md §3.3] for the history convention."*
- *"Look at `scripts/experiments/make_experiment_manifest.py`, specifically the phase2 branch — what configs does it generate for ROMI?"*
- *"Verify that `src/models/<X>` honors the reward contract per [REPO_BRIEF.md §3.1]."*
- *"What does `_validate_dataset_against_env` in `scripts/core/train.py` check, and why?"*

**Less-good queries** (will burn context):
- *"Read the whole codebase and tell me what's there."* → ask me to read one or two specific files instead.
- *"Audit everything."* → scope it to one adapter, one config, or one script at a time.

**If you want a fresh audit:**
- Cite the specific spec the code is supposed to satisfy (e.g., `docs/add_baseline.md` or §X of `docs/problem_setting.tex`).
- Name the file(s) to read.
- Say what kind of feedback you want (logic bugs / faithfulness to paper / safety / style).

**If you want me to modify code:**
- Confirm the file path.
- Tell me the test that should still pass after the change (almost always `tests/test_<thing>.py`).
- If the change crosses adapters, mention `env.reward_contract` and the history convention as constraints to preserve.

---

## 11. Quick-reference cheat sheet

```
# Working dir
cd /fs04/scratch2/ce25/Claude_DeepRL_Population_Models

# Python
PY=/fs04/scratch2/ce25/hphung/conda/envs/pytorchrl/bin/python

# Smoke any method
$PY claude_build/scripts/core/run_pipeline.py active_model=<m> model=<m> \
  env.dataset.n_transitions=400 env.dataset.episode_len=10 \
  eval.n_episodes=2 eval.horizon=5 wandb.mode=disabled

# All tests
$PY -m pytest claude_build/tests/ -q

# Single test
$PY -m pytest claude_build/tests/test_bioconserv18_plus.py::test_posterior_update_favors_more_likely_candidate -v

# Git status without artifacts
git status --short | grep -vE '\.pyc|__pycache__|outputs/|wandb/'

# Restore touched artifacts (after a smoke run)
git checkout HEAD -- $(git diff --name-only | grep -E 'outputs/|__pycache__|\.pyc$')
```

---

*End of REPO_BRIEF.*
