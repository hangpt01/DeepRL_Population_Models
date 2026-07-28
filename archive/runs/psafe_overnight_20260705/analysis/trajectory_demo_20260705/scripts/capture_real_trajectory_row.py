#!/usr/bin/env python3
"""Capture per-timestep trajectories for one real-ecology demo manifest row.

The audited P5 experiment saved per-episode summaries only.  This script reuses
the frozen code snapshot and cached P5 datasets, refits one policy row, then
rolls a small set of paired evaluation episodes and writes true state, action,
and true immediate reward traces for plotting.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import time

import numpy as np


DEMO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = DEMO_ROOT.parents[1]
CODE_ROOT = RUN_ROOT / "code" / "real_ecology_cont_obser"
sys.path.insert(0, str(CODE_ROOT / "src"))

from real_ecology_benchmark.beliefs import BeliefCache, OracleStateFilter  # noqa: E402
from real_ecology_benchmark.config import load_config, real_environment_like  # noqa: E402
from real_ecology_benchmark.envs import make_env  # noqa: E402
from real_ecology_benchmark.pipeline import (  # noqa: E402
    build_method,
    ensure_dataset,
    make_filter_factory,
)
from real_ecology_benchmark.training_monitor import split_train_holdout  # noqa: E402
from real_ecology_benchmark.types import PublicTransition  # noqa: E402


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sigma_slug(value: float | str) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def read_manifest_row(path: Path, index: int) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if int(row["index"]) == index]
    if len(matches) != 1:
        raise SystemExit(f"manifest index {index} matched {len(matches)} rows")
    return matches[0]


def apply_row_config(cfg, row: dict[str, str], source_output_root: Path):
    population = row["population"]
    family = row["environment"]
    reward_mode = row["reward_mode"]
    sigma = float(row["sigma_obs"])
    cfg.environment = real_environment_like(cfg.environment, population, family)
    cfg.environment = replace(
        cfg.environment,
        observation_noise_sigma=sigma,
        reward_mode=reward_mode,
    )
    cfg.validate()
    cell = (
        Path(f"reward_{reward_mode}")
        / slug(population)
        / family
        / f"sigma_{sigma_slug(sigma)}"
    )
    cfg.dataset.output = str(source_output_root / "datasets" / cell / "public.npz")
    cfg.dataset.private_output = str(source_output_root / "private" / cell / "truth.npz")
    return cfg, cell


def roll(policy, cfg, filter_factory, episodes: int):
    horizon = min(cfg.evaluation.horizon, cfg.environment.horizon)
    seeds = [int(cfg.evaluation.seeds[0] + i) for i in range(episodes)]
    env_probe = make_env(cfg.environment)
    num_actions = env_probe.num_actions
    action_names = [a.name for a in env_probe.actions]

    arrays = {
        "state_pre": np.full((episodes, horizon), np.nan),
        "state_post": np.full((episodes, horizon), np.nan),
        "observation": np.full((episodes, horizon), np.nan),
        "belief_mean_pre": np.full((episodes, horizon), np.nan),
        "belief_mean_post": np.full((episodes, horizon), np.nan),
        "action": np.full((episodes, horizon), np.nan),
        "reward": np.full((episodes, horizon), np.nan),
        "reward_true": np.full((episodes, horizon), np.nan),
        "below_safety": np.full((episodes, horizon), np.nan),
        "below_mvp": np.full((episodes, horizon), np.nan),
        "safety_penalty_applied": np.full((episodes, horizon), np.nan),
        "rho": np.full((episodes, horizon), np.nan),
        "K_eff": np.full((episodes, horizon), np.nan),
    }
    summaries = []
    for episode, seed in enumerate(seeds):
        env = make_env(cfg.environment)
        reset = env.reset(seed)
        filt = filter_factory()
        if isinstance(filt, OracleStateFilter):
            belief = filt.set_true_state(
                float(reset.evaluator_info["state"]), 0, reset.public_info
            )
        else:
            belief = filt.reset(reset.observation, seed + 10_000)
        policy.reset(seed + 20_000)
        observation = reset.observation
        cumulative_true_reward = 0.0
        for step in range(horizon):
            arrays["state_pre"][episode, step] = float(env.state)
            arrays["observation"][episode, step] = float(observation)
            arrays["belief_mean_pre"][episode, step] = float(belief.mean_state())
            try:
                action = int(policy.act(belief, observation))
            except (FloatingPointError, ValueError, RuntimeError):
                action = 0
            if not 0 <= action < num_actions:
                action = 0
            result = env.step(action)
            info = result.evaluator_info
            true_reward = float(info["reward_true"])
            cumulative_true_reward += true_reward
            arrays["action"][episode, step] = action
            arrays["reward"][episode, step] = float(result.reward)
            arrays["reward_true"][episode, step] = true_reward
            arrays["state_post"][episode, step] = float(info["state"])
            arrays["below_safety"][episode, step] = float(info["below_safety_region"])
            arrays["below_mvp"][episode, step] = float(info["below_mvp_region"])
            arrays["safety_penalty_applied"][episode, step] = float(
                info["safety_penalty_applied"]
            )
            arrays["rho"][episode, step] = float(info.get("rho", np.nan))
            arrays["K_eff"][episode, step] = float(info.get("K_eff", np.nan))
            policy.observe(
                belief,
                action,
                PublicTransition(
                    observation=result.observation,
                    done=result.done,
                    truncated=result.truncated,
                    public_info=result.public_info.copy(),
                ),
            )
            if isinstance(filt, OracleStateFilter):
                belief = filt.set_true_state(
                    float(info["state"]), step + 1, result.public_info
                )
            else:
                belief = filt.update(belief, action, result.observation)
            arrays["belief_mean_post"][episode, step] = float(belief.mean_state())
            observation = result.observation
            if result.done:
                break
        valid_final = arrays["state_post"][episode]
        valid_final = valid_final[~np.isnan(valid_final)]
        summaries.append(
            {
                "episode": episode,
                "seed": seed,
                "steps": int(len(valid_final)),
                "true_return_undiscounted": cumulative_true_reward,
                "min_state": float(np.nanmin(arrays["state_post"][episode])),
                "final_state": float(valid_final[-1]) if len(valid_final) else np.nan,
                "unsafe_fraction": float(np.nanmean(arrays["below_safety"][episode])),
            }
        )
    return arrays, summaries, seeds, action_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("index", type=int)
    parser.add_argument(
        "--config",
        type=Path,
        default=CODE_ROOT / "configs" / "real_experiment_p5.yaml",
    )
    parser.add_argument(
        "--source-output-root",
        type=Path,
        default=RUN_ROOT / "outputs" / "p5",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=DEMO_ROOT / "traces",
    )
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    start = time.perf_counter()
    row = read_manifest_row(args.manifest, args.index)
    cfg = load_config(args.config)
    cfg, cell = apply_row_config(cfg, row, args.source_output_root)
    dataset = ensure_dataset(cfg, regenerate=False)
    filter_factory, _proposal = make_filter_factory(cfg, dataset, row["filter"])
    cache_path = Path(cfg.dataset.output).with_name(f"public.{row['filter']}.beliefs.npz")
    if cache_path.exists():
        cache = BeliefCache.load(cache_path)
    else:
        raise SystemExit(f"missing cached belief file: {cache_path}")
    train_dataset, train_cache, holdout_dataset, holdout_cache, split_info = split_train_holdout(
        dataset, cache, cfg
    )
    policy, _ = build_method(
        row["method"],
        cfg,
        train_dataset,
        filter_factory,
        train_cache,
        holdout_dataset=holdout_dataset,
        holdout_cache=holdout_cache,
        split_info=split_info,
    )
    arrays, episode_summaries, seeds, action_names = roll(
        policy, cfg, filter_factory, args.episodes
    )
    out_dir = (
        args.trace_root
        / slug(row["population"])
        / row["environment"]
        / f"sigma_{sigma_slug(row['sigma_obs'])}"
        / row["method"]
        / row["filter"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "row": row,
        "cell": str(cell),
        "population": row["population"],
        "population_slug": slug(row["population"]),
        "environment": row["environment"],
        "sigma_obs": float(row["sigma_obs"]),
        "method": row["method"],
        "filter": row["filter"],
        "reward_mode": row["reward_mode"],
        "episodes": args.episodes,
        "seeds": seeds,
        "horizon": min(cfg.evaluation.horizon, cfg.environment.horizon),
        "safety_threshold": float(cfg.environment.safety_threshold),
        "mvp_threshold": float(cfg.environment.mvp_threshold),
        "K_base": float(cfg.environment.K_base),
        "N0": float(cfg.environment.N0),
        "collapse_penalty": float(cfg.environment.collapse_penalty),
        "safety_penalty_mode": cfg.environment.safety_penalty_mode,
        "action_names": action_names,
        "config": str(args.config),
        "source_output_root": str(args.source_output_root),
        "seconds": time.perf_counter() - start,
    }
    np.savez_compressed(out_dir / "trajectory.npz", meta=np.asarray(json.dumps(meta)), **arrays)
    with (out_dir / "episodes_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_summaries[0]))
        writer.writeheader()
        writer.writerows(episode_summaries)
    with (out_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    print(json.dumps({
        "status": "ok",
        "output": str(out_dir / "trajectory.npz"),
        "method": row["method"],
        "population": row["population"],
        "seconds": meta["seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
