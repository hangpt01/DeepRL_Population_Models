"""Paired continuous-POMDP evaluator with public and private metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .backend import get_active_backend
from .config import BenchmarkConfig
from .beliefs import OracleStateFilter
from .envs import make_env
from .types import BeliefFilter, BeliefPolicy, PublicTransition


class ContinuousEvaluator:
    def __init__(
        self,
        cfg: BenchmarkConfig,
        filter_factory: Callable[[], BeliefFilter],
        filter_label: str = "unknown",
    ):
        self.cfg = cfg
        self.filter_factory = filter_factory
        self.filter_label = filter_label

    def run(self, policy: BeliefPolicy) -> list[dict[str, object]]:
        rows = []
        backend_meta = get_active_backend().to_dict()
        env_cfg = self.cfg.environment
        eval_cfg = self.cfg.evaluation
        episode_index = 0
        for block_seed in eval_cfg.seeds:
            for local_episode in range(eval_cfg.episodes_per_seed):
                seed = int(block_seed + local_episode)
                env = make_env(env_cfg)
                reset = env.reset(seed)
                filt = self.filter_factory()
                if isinstance(filt, OracleStateFilter):
                    belief = filt.set_true_state(
                        float(reset.evaluator_info["state"]), 0, reset.public_info
                    )
                else:
                    belief = filt.reset(reset.observation, seed + 10_000)
                policy.reset(seed + 20_000)
                operational_return = 0.0
                true_return = 0.0
                gamma = 1.0
                entries = 0
                unsafe_steps = int(reset.evaluator_info["state"] <= env_cfg.safety_threshold)
                mvp_steps = int(reset.evaluator_info["state"] <= env_cfg.mvp_threshold)
                squared_errors = [(belief.mean_state() - reset.evaluator_info["state"]) ** 2]
                log_squared_errors = [
                    (np.log1p(belief.mean_state()) - np.log1p(reset.evaluator_info["state"])) ** 2
                ]
                filter_seconds = 0.0
                planner_seconds = 0.0
                economic_cost = 0.0
                actions = []
                observed_values = [float(reset.observation)]
                true_values = [float(reset.evaluator_info["state"])]
                coverage90 = []
                ess_fraction = []
                unsafe_brier = []
                danger_action_counts = np.zeros(env.num_actions, dtype=int)
                danger_steps = 0
                uncertainty_values = []
                first_entry_step = -1
                fallback_count = 0
                observation = reset.observation
                result = None

                def record_filter(current_belief, truth):
                    order = np.argsort(current_belief.states)
                    cdf = np.cumsum(current_belief.weights[order])
                    low_idx = min(int(np.searchsorted(cdf, 0.05)), len(order) - 1)
                    high_idx = min(int(np.searchsorted(cdf, 0.95)), len(order) - 1)
                    low = current_belief.states[order[low_idx]]
                    high = current_belief.states[order[high_idx]]
                    coverage90.append(float(low <= truth <= high))
                    ess_fraction.append(current_belief.ess / len(current_belief.states))
                    unsafe_probability = float(np.sum(
                        current_belief.weights[
                            current_belief.states <= env_cfg.safety_threshold
                        ]
                    ))
                    unsafe_brier.append(
                        (unsafe_probability - float(truth <= env_cfg.safety_threshold)) ** 2
                    )

                record_filter(belief, float(reset.evaluator_info["state"]))
                for step in range(min(eval_cfg.horizon, env_cfg.horizon)):
                    t0 = time.perf_counter()
                    try:
                        action = int(policy.act(belief, observation))
                    except (FloatingPointError, ValueError, RuntimeError):
                        action = 0
                        fallback_count += 1
                    planner_seconds += time.perf_counter() - t0
                    if not 0 <= action < env.num_actions:
                        action = 0
                        fallback_count += 1
                    result = env.step(action)
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
                    operational_return += gamma * result.reward
                    true_return += gamma * float(result.evaluator_info["reward_true"])
                    entered = int(result.evaluator_info["entered_safety_region"])
                    entries += entered
                    if entered and first_entry_step < 0:
                        first_entry_step = step
                    truth_state = float(result.evaluator_info["state"])
                    unsafe_steps += int(truth_state <= env_cfg.safety_threshold)
                    mvp_steps += int(truth_state <= env_cfg.mvp_threshold)
                    state_previous = float(result.evaluator_info["state_previous"])
                    if env_cfg.safety_threshold < state_previous <= 4 * env_cfg.safety_threshold:
                        danger_action_counts[action] += 1
                        danger_steps += 1
                    actions.append(action)
                    economic_cost += float(env.actions[action].cost)
                    observed_values.append(float(result.observation))
                    true_values.append(float(result.evaluator_info["state"]))
                    diagnostics = getattr(policy, "last_diagnostics", {})
                    fallback_count += int(bool(diagnostics.get("hard_fallback", False)))
                    for key, value in diagnostics.items():
                        if ("uncertainty" in key or "entropy" in key) and np.isscalar(value):
                            uncertainty_values.append(float(value))
                            break
                    t0 = time.perf_counter()
                    if isinstance(filt, OracleStateFilter):
                        belief = filt.set_true_state(
                            float(result.evaluator_info["state"]), step + 1,
                            result.public_info,
                        )
                    else:
                        belief = filt.update(belief, action, result.observation)
                    filter_seconds += time.perf_counter() - t0
                    truth = truth_state
                    squared_errors.append((belief.mean_state() - truth) ** 2)
                    log_squared_errors.append(
                        (np.log1p(belief.mean_state()) - np.log1p(truth)) ** 2
                    )
                    record_filter(belief, truth)
                    observation = result.observation
                    gamma *= eval_cfg.discount
                    if result.done:
                        break
                n_steps = len(actions)
                counts = np.bincount(actions, minlength=env.num_actions).astype(float)
                probs = counts / max(counts.sum(), 1.0)
                entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
                rows.append(
                    {
                        "model": policy.name,
                        "episode": episode_index,
                        "seed": seed,
                        "block_seed": int(block_seed),
                        "operational_return": float(operational_return),
                        "true_return": float(true_return),
                        "collapse_entry": int(entries > 0),
                        "collapse_entries": int(entries),
                        "collapse_entry_timestep": int(first_entry_step),
                        "unsafe_fraction": float(unsafe_steps / max(n_steps + 1, 1)),
                        "mvp_fraction": float(mvp_steps / max(n_steps + 1, 1)),
                        "mvp_breach": int(mvp_steps > 0),
                        # Reward-agnostic battery (spec E6'): computed from true
                        # states / actions only, so identical across reward_mode
                        # for a fixed trajectory.
                        "economic_cost": float(economic_cost),
                        "min_true_state": float(np.min(true_values)),
                        "final_true_state": float(true_values[-1]),
                        "persistence": int(true_values[-1] > env_cfg.safety_threshold),
                        "mean_observation": float(np.mean(observed_values)),
                        "mean_true_state": float(np.mean(true_values)),
                        "filter_rmse": float(np.sqrt(np.mean(squared_errors))),
                        "filter_log_rmse": float(np.sqrt(np.mean(log_squared_errors))),
                        "filter_coverage90": float(np.mean(coverage90)),
                        "filter_ess_fraction": float(np.mean(ess_fraction)),
                        "filter_unsafe_brier": float(np.mean(unsafe_brier)),
                        "n_steps": n_steps,
                        "action_entropy": entropy,
                        "method_uncertainty_mean": float(np.mean(uncertainty_values))
                        if uncertainty_values else 0.0,
                        "filter_seconds": filter_seconds,
                        "planner_seconds": planner_seconds,
                        "fallback_count": fallback_count,
                        "sigma_obs": env_cfg.observation_noise_sigma,
                        "environment": env_cfg.kind,
                        "data_mode": getattr(env_cfg, "data_mode", "real"),
                        "population": getattr(env_cfg, "population", "n/a"),
                        "reward_mode": getattr(env_cfg, "reward_mode", "n/a"),
                        "safety_penalty_mode": getattr(
                            env_cfg, "safety_penalty_mode", "n/a"
                        ),
                        "safety_threshold": env_cfg.safety_threshold,
                        "mvp_threshold": env_cfg.mvp_threshold,
                        "data_table": str(getattr(env_cfg, "data_dir", "")),
                        "num_actions": env_cfg.num_actions,
                        "filter": self.filter_label,
                        **backend_meta,
                        **{
                            f"danger_action_{a}_fraction": float(
                                danger_action_counts[a] / max(danger_steps, 1)
                            )
                            for a in range(env.num_actions)
                        },
                    }
                )
                episode_index += 1
        return rows

    @staticmethod
    def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
        numeric = (
            "operational_return", "true_return", "collapse_entry", "unsafe_fraction",
            "mvp_fraction", "mvp_breach", "economic_cost", "min_true_state",
            "final_true_state", "persistence",
            "mean_observation", "mean_true_state", "filter_rmse", "filter_log_rmse",
            "filter_coverage90", "filter_ess_fraction", "filter_unsafe_brier",
            "method_uncertainty_mean", "n_steps", "filter_seconds", "planner_seconds",
            "fallback_count",
        )
        summary: dict[str, object] = {
            "model": rows[0]["model"] if rows else "unknown",
            "episodes": len(rows),
        }
        if rows:
            for key in (
                "environment", "data_mode", "population", "reward_mode", "num_actions",
                "sigma_obs", "safety_penalty_mode", "safety_threshold", "mvp_threshold",
                "data_table", "filter", "compute_backend_requested", "compute_backend_effective",
                "compute_device", "compute_backend_device", "compute_backend_strict",
                "compute_backend_fallback_reason", "compute_backend_workload",
                "compute_backend_acceleration_scope",
            ):
                if key in rows[0]:
                    summary[key] = rows[0][key]
        for key in numeric:
            values = np.asarray([float(row[key]) for row in rows])
            summary[f"{key}_mean"] = float(values.mean()) if len(values) else np.nan
            summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        return summary

    @staticmethod
    def save(
        rows: list[dict[str, object]],
        output_dir: str | Path,
        extra_summary: dict[str, object] | None = None,
    ) -> dict[str, object]:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        if rows:
            with (target / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        summary = ContinuousEvaluator.summarize(rows)
        if extra_summary:
            summary.update(extra_summary)
        with (target / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return summary
