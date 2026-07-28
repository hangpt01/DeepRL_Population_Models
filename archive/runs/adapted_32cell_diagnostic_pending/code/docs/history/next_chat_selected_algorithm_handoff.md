# Next-Chat Handoff: Short Ecology Benchmark With Selected Algorithms

This note is a compact process representation of the current repo state and the next intended run. It is written so the next chat can start directly from here without rereading the full debugging/audit history.

## Current Repo State

Working repo:

```bash
cd /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/claude_build
```

The main implementation target is `claude_build`, an offline/model-based RL framework for an adaptive-management Ricker ecology problem.

Core ecological setting:

- Environment: 5-action default `ricker`; 10-action `ricker_full` exists for sensitivity.
- State: exact observed discrete abundance bin, 100 bins, bin width 10, max abundance 1000.
- Hidden parameter: per-episode `r_base ~ U(0.95, 1.00)`, fixed within episode, never logged.
- Action semantics: each action has structured metadata `(id, name, delta_r, delta_K, cost)`.
- Dynamics: Ricker growth with `r_eff = r_base + delta_r[a]`, `K_eff = K_base + delta_K[a]`.
- Reward contract: `R(x_t, a_t) = alpha * x_t / x_max - cost[a]`, current-state reward. This is shared through `env.reward_contract`.
- No action masking: all actions available in all states.

Implemented method families now include:

- Ecology-adjacent baselines:
  - `bioconserv18_plus`: PLUS-style finite hidden-model baseline from BioConserv18. It uses a candidate grid over hidden growth rates and posterior-style model weighting; no offline neural training.
  - `moor`: ExpertSys23 MOOR-style mechanistic offline RL baseline. It learns a point-estimate mechanistic model by least squares from offline transitions, supports `dynamics_model=ricker` and `dynamics_model=schaefer`, discretizes the learned model, and plans by value iteration.
- General offline/model-based RL adapted to this ecology setting:
  - `ensemble` + `pessimistic` planner: MOPO-style learned categorical dynamics ensemble.
  - `ensemble` + `morel` planner: MOReL-style pessimistic MDP.
  - `combo`, `cql`, `iql`.
  - `bamcts`, `romi`, `refplan`.

Important validation already completed:

```bash
python -m pytest tests/test_moor.py tests/test_experiment_manifest_moor.py -q
# 31 passed

python -m pytest -q
# 134 passed, 1 pre-existing warning
```

Manifest state:

- `scripts/experiments/make_experiment_manifest.py` includes `moor` in all phases.
- Phase 1 now has 10 methods x 3 seeds = 30 rows.
- Phase 2 now includes MOOR Ricker and Schaefer variants.
- A 10-action MOOR dry-run correctly used `env=ricker_full`.

Slurm note:

- `scripts/slurm/slurm_model_comparison.sh` now has 5 array tasks but limits concurrency to 4 with `--array=0-4%4`.
- The full ladder exists, but the next run should not use the full ladder. The next goal is a short selected-method demonstration.

Working-tree caveat:

- Test runs may leave tracked `.pyc` files dirty under `claude_build/src/models/__pycache__` and `claude_build/tests/__pycache__`. They are generated artifacts, not source changes.
- `docs/audit_report.md` was already deleted before the MOOR work; do not restore it unless the user explicitly asks.

## Next Intended Run

Do not run every implemented algorithm. The next chat should run a short selected-method benchmark whose purpose is:

1. show the framework runs end-to-end,
2. produce a simple initial result table,
3. compare general model-based RL methods against the most related ecology baselines,
4. record wall-clock runtime and GPU usage so the methods can be compared operationally as well as by reward.

Primary selected algorithms for the Ricker setting:

- `MOPO`: `active_model=ensemble`, `model=ensemble`, `active_planner=pessimistic`, `planner=pessimistic`
- `RefPlan`: `active_model=refplan`, `model=refplan`
- `PLUS`: `active_model=bioconserv18_plus`, `model=bioconserv18_plus`
- `MOOR-Ricker`: `active_model=moor`, `model=moor`, `model.dynamics_model=ricker`

Do **not** include `MOOR-Schaefer` in the primary Ricker table. Since the true environment is Ricker, the most relevant ExpertSys23 baseline is MOOR-Ricker.

The desired intuitive story to test is:

> RefPlan and/or MOPO, as general model-based offline RL methods adapted to this ecology framework, can outperform the ecology-specific baselines PLUS and MOOR-Ricker in cumulative reward while maintaining reasonable abundance safety.

Do not state this as a final scientific claim from a tiny run. Treat it as an initial framework/demo result.

## Model-Knowledge Stress Setting

The conceptual advantage of RefPlan and MOPO is that they do not need the user to specify the population model family being solved. They learn a transition model from data. PLUS and MOOR-Ricker are ecology-specific/mechanistic baselines tied to a Ricker-style assumed dynamics model.

To expose this advantage, the next chat should check whether a true Schaefer/logistic environment already exists. Current audit evidence suggests only `RickerEnv` is registered, so this likely requires a small extra env/config if there is time.

Secondary stress-test design:

- True environment/data/evaluation: Schaefer/logistic population dynamics with the same discrete states, actions, reward contract, and action metadata.
- Model-agnostic methods: run the same MOPO and RefPlan configs, unchanged.
- Mechanistic baselines: compare `PLUS` and `MOOR-Ricker` as deliberately model-mismatched ecology baselines.
- Goal: show whether RefPlan/MOPO remain competitive when the assumed ecological population model is wrong.

If adding or activating a true Schaefer environment is too much for the short run, do **not** block the primary Ricker result. Run only the primary Ricker table and explicitly state that the model-mismatch stress setting remains planned.

## Recommended Short Benchmark Settings

Use 5-action `ricker` for the primary table.

Suggested quick settings:

- seeds: start with `42`; if all GPU slots are available and time allows, use `42 123 456`
- offline dataset size: `DATA_N=25000` for a short but nontrivial run; reduce to `10000` if runtime is tight
- collection policy: `random`
- evaluation episodes: `50`; reduce to `20` only for debugging
- evaluation horizon: `50`
- WandB: enabled/online for the demo table
- GPU usage: use all currently available GPU slots for fast turnaround; run methods/seeds concurrently rather than as a serial full ladder
- device:
  - use `cuda` for MOPO and RefPlan
  - PLUS and MOOR-Ricker can run on CPU, but it is acceptable to put them on GPU jobs too if that simplifies parallel Slurm scheduling

Runtime/GPU reporting:

- log `gpu_name`, `elapsed_seconds`, and, if available, `peak_gpu_memory_mb` or Slurm `MaxRSS/AllocGRES`
- use WandB group `quick_selected_ecology_benchmark`
- use one WandB run per method/seed/setting, with method and setting in `WANDB_NAME`
- if running under Slurm, use `sacct` after completion where available and also write local timing logs

Fairness:

- MOPO, RefPlan, and MOOR-Ricker should use the same saved offline dataset path for a given seed/setting.
- PLUS does not train from the offline dataset, but it should use the same environment, evaluation seed, number of episodes, and horizon.
- The table should be interpreted as a smoke/demo comparison, not a tuned final benchmark.

## Exact Commands For One-Seed Demo

Run from:

```bash
cd /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/claude_build
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate pytorchrl
```

Set common variables:

```bash
SEED=42
DATA_N=25000
EVAL_EPISODES=50
EVAL_HORIZON=50
OUT_ROOT=outputs/quick_selected_methods/seed${SEED}_n${DATA_N}
DATASET_PATH=${OUT_ROOT}/dataset/random_5a_seed${SEED}_n${DATA_N}.npz
WB_GROUP=quick_selected_ecology_benchmark
export WANDB_RUN_GROUP=${WB_GROUP}
mkdir -p "${OUT_ROOT}/dataset"
```

Generate one shared dataset:

```bash
WANDB_NAME=quick_dataset_ricker_seed${SEED} \
python scripts/core/generate_dataset.py \
  env=ricker \
  seed=${SEED} \
  wandb.mode=online \
  wandb.project=deeprl_population_models \
  env.dataset.n_transitions=${DATA_N} \
  env.dataset.collection_policy=random \
  +dataset_path=${DATASET_PATH} \
  hydra.run.dir=${OUT_ROOT}/hydra_dataset
```

Run MOPO:

```bash
WANDB_NAME=quick_mopo_ricker_seed${SEED} \
python scripts/core/run_pipeline.py \
  env=ricker \
  seed=${SEED} \
  device=cuda \
  wandb.mode=online \
  wandb.project=deeprl_population_models \
  +dataset_path=${DATASET_PATH} \
  active_model=ensemble \
  model=ensemble \
  active_planner=pessimistic \
  planner=pessimistic \
  model.ensemble_size=5 \
  model.frame_stack_len=2 \
  model.bootstrap_ratio=0.8 \
  model.num_epochs=20 \
  planner.pessimism_lambda=1.0 \
  planner.horizon=5 \
  planner.num_rollouts=20 \
  eval.n_episodes=${EVAL_EPISODES} \
  eval.horizon=${EVAL_HORIZON} \
  hydra.run.dir=${OUT_ROOT}/hydra_mopo \
  paths.dataset_dir=${OUT_ROOT}/mopo/ds \
  paths.checkpoint_dir=${OUT_ROOT}/mopo/ck \
  eval.output_dir=${OUT_ROOT}/mopo/eval \
  eval.csv_name=mopo.csv \
  eval.json_name=mopo.json
```

Run RefPlan:

```bash
WANDB_NAME=quick_refplan_ricker_seed${SEED} \
python scripts/core/run_pipeline.py \
  env=ricker \
  seed=${SEED} \
  device=cuda \
  wandb.mode=online \
  wandb.project=deeprl_population_models \
  +dataset_path=${DATASET_PATH} \
  active_model=refplan \
  model=refplan \
  model.num_actions=5 \
  model.ensemble_size=10 \
  model.planning_horizon=4 \
  model.num_sequences=64 \
  model.latent_samples=4 \
  model.kappa=5.0 \
  model.uncertainty_penalty=0.10 \
  eval.n_episodes=${EVAL_EPISODES} \
  eval.horizon=${EVAL_HORIZON} \
  hydra.run.dir=${OUT_ROOT}/hydra_refplan \
  paths.dataset_dir=${OUT_ROOT}/refplan/ds \
  paths.checkpoint_dir=${OUT_ROOT}/refplan/ck \
  eval.output_dir=${OUT_ROOT}/refplan/eval \
  eval.csv_name=refplan.csv \
  eval.json_name=refplan.json
```

Run PLUS:

```bash
WANDB_NAME=quick_plus_ricker_seed${SEED} \
python scripts/core/evaluate.py \
  env=ricker \
  seed=${SEED} \
  device=cpu \
  wandb.mode=online \
  wandb.project=deeprl_population_models \
  active_model=bioconserv18_plus \
  model=bioconserv18_plus \
  model.num_candidate_models=21 \
  model.discount=0.95 \
  model.transition_n_grid=2000 \
  model.likelihood_floor=1e-12 \
  eval.n_episodes=${EVAL_EPISODES} \
  eval.horizon=${EVAL_HORIZON} \
  hydra.run.dir=${OUT_ROOT}/hydra_plus \
  eval.output_dir=${OUT_ROOT}/plus/eval \
  eval.csv_name=plus.csv \
  eval.json_name=plus.json
```

Run MOOR-Ricker:

```bash
WANDB_NAME=quick_moor_ricker_seed${SEED} \
python scripts/core/run_pipeline.py \
  env=ricker \
  seed=${SEED} \
  device=cpu \
  wandb.mode=online \
  wandb.project=deeprl_population_models \
  +dataset_path=${DATASET_PATH} \
  active_model=moor \
  model=moor \
  model.dynamics_model=ricker \
  'model.learn_params=[r,K]' \
  model.n_restarts=10 \
  model.lbfgs_max_iter=10 \
  model.transition_n_grid=1000 \
  model.discount=0.95 \
  eval.n_episodes=${EVAL_EPISODES} \
  eval.horizon=${EVAL_HORIZON} \
  hydra.run.dir=${OUT_ROOT}/hydra_moor_ricker \
  paths.dataset_dir=${OUT_ROOT}/moor_ricker/ds \
  paths.checkpoint_dir=${OUT_ROOT}/moor_ricker/ck \
  eval.output_dir=${OUT_ROOT}/moor_ricker/eval \
  eval.csv_name=moor_ricker.csv \
  eval.json_name=moor_ricker.json
```

When passing `model.learn_params=[r,K]` in shell context, keep it quoted:

```bash
'model.learn_params=[r,K]'
```

## Make The Simple Result Table

After the five runs finish, create a compact table:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("outputs/quick_selected_methods")
csvs = sorted(root.glob("seed*_n*/**/eval/*.csv"))
rows = []
for path in csvs:
    df = pd.read_csv(path)
    if df.empty:
        continue
    model_name = str(df["model"].iloc[0])
    rows.append({
        "csv": str(path),
        "model": model_name,
        "episodes": len(df),
        "cum_reward_mean": df["cumulative_reward"].mean(),
        "cum_reward_std": df["cumulative_reward"].std(),
        "final_abundance_mean": df["final_abundance"].mean(),
        "min_abundance_mean": df["min_abundance"].mean(),
        "catastrophic_rate": df["catastrophic_low_abundance"].mean(),
        "uncertainty_mean": df["uncertainty_mean"].mean(),
        # Fill these from timing/GPU logs if the next chat runs through Slurm
        # or a timing wrapper; left blank here for direct Python-only runs.
        "elapsed_seconds": None,
        "gpu_name": "",
        "peak_gpu_memory_mb": None,
    })

out = pd.DataFrame(rows)
if not out.empty:
    out = out.sort_values("cum_reward_mean", ascending=False)
    print(out[[
        "model",
        "episodes",
        "cum_reward_mean",
        "cum_reward_std",
        "final_abundance_mean",
        "min_abundance_mean",
        "catastrophic_rate",
        "uncertainty_mean",
        "elapsed_seconds",
        "gpu_name",
        "peak_gpu_memory_mb",
    ]].round(4).to_string(index=False))
    table_path = root / "selected_methods_summary.csv"
    out.to_csv(table_path, index=False)
    print(f"\nSaved: {table_path}")
else:
    print("No CSVs found.")
PY
```

Minimum table columns to report:

| Method | Cumulative reward mean | Cumulative reward std | Final abundance mean | Minimum abundance mean | Catastrophic rate | Runtime | GPU / peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RefPlan | fill after run | fill after run | fill after run | fill after run | fill after run | fill after run | fill after run |
| MOPO | fill after run | fill after run | fill after run | fill after run | fill after run | fill after run | fill after run |
| PLUS | fill after run | fill after run | fill after run | fill after run | fill after run | fill after run | fill after run |
| MOOR-Ricker | fill after run | fill after run | fill after run | fill after run | fill after run | fill after run | fill after run |

## Suggested Next-Chat Prompt

Paste this in the next chat:

```text
We are in /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models. Read docs/next_chat_selected_algorithm_handoff.md first.

I do not want to run all implemented methods. Run a short selected-method benchmark in claude_build comparing RefPlan and MOPO against the most related ecology baselines: BioConserv18 PLUS and ExpertSys23 MOOR-Ricker.

Use the 5-action ricker env, seed 42, DATA_N=25000, EVAL_EPISODES=50, EVAL_HORIZON=50, WandB online/enabled, and the same saved offline dataset path for MOPO, RefPlan, and MOOR-Ricker. Use all currently available GPU slots for fast turnaround and run jobs concurrently where possible. Capture wall-clock runtime and GPU usage/memory per method, either from Slurm logs/sacct or a lightweight timing/GPU wrapper.

Also check whether the repo already has a true Schaefer/logistic environment. If it exists, or if adding a minimal Schaefer true-env variant is quick, run a second stress setting where the true environment is Schaefer/logistic but PLUS and MOOR-Ricker remain Ricker-assumption ecology baselines. This is meant to test the advantage that RefPlan and MOPO do not need the population model family specified. If adding the Schaefer true-env setting is not quick, do not block the primary Ricker result; just report it as planned.

After the runs, produce a compact CSV/Markdown table sorted by cumulative_reward_mean with cumulative reward, final abundance, min abundance, catastrophic rate, uncertainty, runtime, GPU name, and peak GPU memory if available. This is only a short framework/demo result, not a final tuned benchmark.
```

## Interpretation Rules For Tomorrow

- If RefPlan or MOPO beats PLUS/MOOR-Ricker in cumulative reward, write it as an initial indication that general model-based offline RL can be competitive or stronger in this ecology framework.
- If they do not beat the ecology baselines, do not hide it. Report that the framework runs and the first short run suggests more tuning is needed.
- Avoid claiming final superiority from one seed or small compute.
- The strongest valid process claim after this short run is: “The ecology framework can now compare adapted general MBRL algorithms and ecology-specific baselines under shared reward/dynamics semantics.”
