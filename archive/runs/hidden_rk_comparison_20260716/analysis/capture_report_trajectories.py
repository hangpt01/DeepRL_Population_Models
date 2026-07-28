#!/usr/bin/env python3
"""Capture small, illustrative full/hidden trajectories for the results report.

This script reads the frozen run snapshot and existing cached datasets. It
refits policies with the registered split, then rolls three fixed episodes.
These traces are report-only and are not added to the headline experiment.
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


ANALYSIS = Path(__file__).resolve().parent
RUN = ANALYSIS.parent
CODE = RUN / "code"
sys.path.insert(0, str(CODE / "src"))

from real_ecology_benchmark.beliefs import (  # noqa: E402
    BeliefCache,
    PublicBeliefCache,
)
from real_ecology_benchmark.config import (  # noqa: E402
    hides_rk,
    load_config,
    real_environment_like,
)
from real_ecology_benchmark.envs import make_env  # noqa: E402
from real_ecology_benchmark.pipeline import (  # noqa: E402
    _hidden_method_context,
    _load_or_fit_public_surrogate,
    build_method,
    ensure_dataset,
    make_filter_factory,
)
from real_ecology_benchmark.training_monitor import split_train_holdout  # noqa: E402
from real_ecology_benchmark.types import PublicTransition  # noqa: E402


METHODS = ("refplan", "bamcts", "ogsrl", "moor_native", "plus_native")
FILTERS = {
    "refplan": "learned",
    "bamcts": "learned",
    "ogsrl": "learned",
    "moor_native": "native_discrete",
    "plus_native": "native_discrete",
}
CELLS = (
    ("Amur tiger", "ricker", 0.2, "safe"),
    ("Puerto Rican parrot", "ricker", 0.2, "safe"),
    ("Iberian lynx", "allee", 0.2, "safe"),
    ("Egyptian vulture", "ricker", 0.2, "safe"),
)
JOBS = tuple((regime, *cell) for regime in ("full", "hidden") for cell in CELLS)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sigma_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def read_manifest_row(regime: str, population: str, family: str, sigma: float,
                      reward_mode: str, method: str) -> dict[str, str]:
    path = RUN / "manifests" / f"manifest_{regime}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["population"] == population
            and row["environment"] == family
            and float(row["sigma_obs"]) == sigma
            and row["reward_mode"] == reward_mode
            and row["method"] == method
        ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one manifest row, found {len(rows)} for {method}")
    return rows[0]


def configure(regime: str, population: str, family: str, sigma: float,
              reward_mode: str):
    cfg = load_config(str(CODE / "configs" / "hidden_rk.yaml"))
    cfg.environment = real_environment_like(cfg.environment, population, family)
    cfg.environment = replace(
        cfg.environment,
        expose_rk=regime,
        observation_noise_sigma=sigma,
        reward_mode=reward_mode,
    )
    cfg.validate()
    cell = (
        Path(f"regime_{regime}")
        / f"reward_{reward_mode}"
        / slug(population)
        / family
        / f"sigma_{sigma_slug(sigma)}"
    )
    source = RUN / "outputs" / regime
    cfg.dataset.output = str(source / "datasets" / cell / "public.npz")
    cfg.dataset.private_output = str(source / "private" / cell / "truth.npz")
    return cfg


def load_cache(cfg, filter_name: str):
    public = Path(cfg.dataset.output)
    if hides_rk(cfg.environment):
        name = f"{public.stem}.regime_hidden.{filter_name}.beliefs.npz"
        return PublicBeliefCache.load(public.with_name(name))
    if filter_name == "native_discrete":
        name = f"{public.stem}.native_discrete_b{cfg.model.native_state_bins}.beliefs.npz"
    else:
        name = f"{public.stem}.regime_full.{filter_name}.beliefs.npz"
    return BeliefCache.load(public.with_name(name))


def diagnostic_number(policy, *names: str) -> float:
    values = getattr(policy, "last_diagnostics", {})
    for name in names:
        value = values.get(name)
        if isinstance(value, (int, float, np.number)):
            return float(value)
    return float("nan")


def roll(policy, cfg, filter_factory, seeds: list[int]) -> dict[str, np.ndarray]:
    episodes = len(seeds)
    horizon = min(cfg.evaluation.horizon, cfg.environment.horizon)
    keys = (
        "state_pre", "state_post", "observation_pre", "observation_post",
        "belief_mean", "action", "reward_true", "reward_public",
        "private_unsafe", "private_penalty", "terminated", "method_risk",
        "method_ood",
    )
    arrays = {key: np.full((episodes, horizon), np.nan) for key in keys}
    for episode, seed in enumerate(seeds):
        env = make_env(cfg.environment)
        reset = env.reset(seed)
        filt = filter_factory()
        belief = filt.reset(reset.observation, seed + 10_000)
        policy.reset(seed + 20_000)
        observation = float(reset.observation)
        for step in range(horizon):
            arrays["state_pre"][episode, step] = float(env.state)
            arrays["observation_pre"][episode, step] = observation
            arrays["belief_mean"][episode, step] = float(belief.mean_state())
            try:
                action = int(policy.act(belief, observation))
            except (FloatingPointError, RuntimeError, ValueError):
                action = 0
            if not 0 <= action < env.num_actions:
                action = 0
            arrays["method_risk"][episode, step] = diagnostic_number(
                policy, "public_extinction_risk", "unsafe_probability"
            )
            arrays["method_ood"][episode, step] = diagnostic_number(
                policy, "ood_probability"
            )
            result = env.step(action)
            info = result.evaluator_info
            arrays["action"][episode, step] = action
            arrays["state_post"][episode, step] = float(info["state"])
            arrays["observation_post"][episode, step] = float(result.observation)
            arrays["reward_true"][episode, step] = float(info["reward_true"])
            arrays["reward_public"][episode, step] = float(result.reward)
            arrays["private_unsafe"][episode, step] = float(info["below_safety_region"])
            arrays["private_penalty"][episode, step] = float(
                info["safety_penalty_applied"]
            )
            arrays["terminated"][episode, step] = float(
                result.done and not result.truncated
            )
            policy.observe(
                belief,
                action,
                PublicTransition(
                    observation=result.observation,
                    done=result.done,
                    truncated=result.truncated,
                    terminated=bool(result.done and not result.truncated),
                    public_info=result.public_info.copy(),
                ),
            )
            belief = filt.update(belief, action, result.observation)
            observation = float(result.observation)
            if result.done:
                break
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int, choices=range(len(JOBS)))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ANALYSIS / "illustrative_trajectories",
    )
    args = parser.parse_args()
    regime, population, family, sigma, reward_mode = JOBS[args.index]
    cfg = configure(regime, population, family, sigma, reward_mode)
    dataset = ensure_dataset(cfg, regenerate=False)
    surrogate = None
    method_context = None
    if hides_rk(cfg.environment):
        surrogate, _path, _status = _load_or_fit_public_surrogate(cfg, dataset)
        method_context = _hidden_method_context(cfg, dataset, surrogate)

    seeds = [7001, 7051, 7101][:args.episodes]
    payload: dict[str, np.ndarray] = {}
    elapsed: dict[str, float] = {}
    for method in METHODS:
        started = time.perf_counter()
        row = read_manifest_row(regime, population, family, sigma, reward_mode, method)
        filter_name = FILTERS[method]
        factory, _proposal = make_filter_factory(
            cfg, dataset, filter_name, method_context
        )
        cache = load_cache(cfg, filter_name)
        train_data, train_cache, holdout_data, holdout_cache, split = split_train_holdout(
            dataset, cache, cfg
        )
        policy, _ = build_method(
            method,
            cfg,
            train_data,
            factory,
            train_cache,
            holdout_dataset=holdout_data,
            holdout_cache=holdout_cache,
            split_info=split,
            method_context=method_context,
        )
        evaluation_factory = (
            policy.hidden_filter_factory()
            if regime == "hidden" and method == "moor_native"
            else factory
        )
        traces = roll(policy, cfg, evaluation_factory, seeds)
        for key, values in traces.items():
            payload[f"{method}_{key}"] = values
        elapsed[method] = time.perf_counter() - started
        print(f"{regime} {population} {method}: {elapsed[method]:.2f}s", flush=True)

    meta = {
        "illustrative_only": True,
        "headline_experiment": False,
        "regime": regime,
        "population": population,
        "population_slug": slug(population),
        "family": family,
        "sigma_obs": sigma,
        "reward_mode": reward_mode,
        "methods": list(METHODS),
        "filters": FILTERS,
        "episodes": len(seeds),
        "seeds": seeds,
        "horizon": min(cfg.evaluation.horizon, cfg.environment.horizon),
        "safety_threshold_private": float(cfg.environment.safety_threshold),
        "mvp_threshold_private": float(cfg.environment.mvp_threshold),
        "elapsed_seconds_by_method": elapsed,
        "source_public_dataset": cfg.dataset.output,
        "source_config": str(CODE / "configs" / "hidden_rk.yaml"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{slug(population)}_{family}_sigma_{sigma_slug(sigma)}_{regime}.npz"
    np.savez_compressed(path, meta=np.asarray(json.dumps(meta)), **payload)
    print(json.dumps({"status": "ok", "output": str(path), **meta}, indent=2), flush=True)


if __name__ == "__main__":
    main()
