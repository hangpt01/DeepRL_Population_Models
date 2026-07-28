#!/usr/bin/env python3
"""Re-fit each method on one cell (learned filter, reusing the saved dataset and
belief cache) and roll a few paired evaluation episodes, capturing per-timestep
true abundance, action, and immediate reward.  Per-step traces are not saved by
the main pipeline, so they must be regenerated here.  Output: one npz per cell.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark.beliefs import BeliefCache, cache_dataset_beliefs  # noqa: E402
from real_ecology_benchmark.config import environment_with_kind_defaults, load_config  # noqa: E402
from real_ecology_benchmark.envs import make_env  # noqa: E402
from real_ecology_benchmark.pipeline import build_method, ensure_dataset, make_filter_factory  # noqa: E402
from real_ecology_benchmark.types import PublicTransition  # noqa: E402

METHODS = [
    "mopo", "refplan", "bamcts", "moor", "plus",
    "ensemble_value_disagreement_pessimism", "ogsrl",
]
CELLS = [(e, a, s) for e in ("allee", "theta", "regime")
         for a in (5, 10) for s in (0.0, 0.1, 0.2, 0.4)]


def roll(env_cfg, policy, factory, seeds, horizon):
    abundance = np.full((len(seeds), horizon), np.nan)
    action = np.full((len(seeds), horizon), np.nan)
    reward = np.full((len(seeds), horizon), np.nan)
    for i, seed in enumerate(seeds):
        env = make_env(env_cfg)
        reset = env.reset(seed)
        filt = factory()
        belief = filt.reset(reset.observation, seed + 10_000)
        policy.reset(seed + 20_000)
        obs = reset.observation
        for t in range(horizon):
            try:
                a = int(policy.act(belief, obs))
            except (FloatingPointError, ValueError, RuntimeError):
                a = 0
            if not 0 <= a < env.num_actions:
                a = 0
            result = env.step(a)
            abundance[i, t] = float(result.evaluator_info["state"])
            action[i, t] = a
            reward[i, t] = float(result.reward)
            policy.observe(belief, a, PublicTransition(
                observation=result.observation,
                done=result.done,
                truncated=result.truncated,
                terminated=bool(result.done and not result.truncated),
                public_info=result.public_info,
            ))
            belief = filt.update(belief, a, result.observation)
            obs = result.observation
            if result.done:
                break
    return abundance, action, reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=int, help="cell index 0-23")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--output-root", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    kind, na, sigma = CELLS[args.index]
    cell = f"{kind}_{na}a_sigma{sigma}"
    root = Path(args.output_root)
    cfg = load_config(str(ROOT / "configs/synthetic_full.yaml"))
    cfg.environment = environment_with_kind_defaults(cfg.environment, kind)
    cfg.environment = replace(cfg.environment, num_actions=na, observation_noise_sigma=sigma)
    cfg.dataset.output = str(root / "data" / cell / "public.npz")
    cfg.dataset.private_output = str(root / "private" / cell / "truth.npz")

    dataset = ensure_dataset(cfg, regenerate=False)
    factory, _ = make_filter_factory(cfg, dataset, "learned")
    cache_path = Path(cfg.dataset.output).with_name("public.learned.beliefs.npz")
    cache = (BeliefCache.load(cache_path) if cache_path.exists()
             else cache_dataset_beliefs(dataset, factory, cfg.seed + 30_000,
                                        cfg.environment.K_ref, cfg.environment.safety_threshold))

    seeds = [cfg.evaluation.seeds[0] + e for e in range(args.episodes)]
    horizon = min(cfg.evaluation.horizon, cfg.environment.horizon)
    payload = {}
    for method in METHODS:
        policy, _ = build_method(method, cfg, dataset, factory, cache)
        ab, ac, rw = roll(cfg.environment, policy, factory, seeds, horizon)
        payload[f"{method}_abundance"] = ab
        payload[f"{method}_action"] = ac
        payload[f"{method}_reward"] = rw
        print(f"{cell} {method}: mean_final_abundance="
              f"{np.nanmean(ab[:, -1]):.1f} mean_reward={np.nanmean(rw):.3f}", flush=True)

    meta = {"cell": cell, "kind": kind, "num_actions": na, "sigma": sigma,
            "horizon": horizon, "episodes": args.episodes, "methods": METHODS,
            "seeds": seeds, "safety_threshold": cfg.environment.safety_threshold,
            "K_base": cfg.environment.K_base}
    tdir = root / "trajectories"
    tdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tdir / f"{cell}.npz", meta=np.asarray(json.dumps(meta)), **payload)
    print(f"saved {tdir / f'{cell}.npz'}", flush=True)


if __name__ == "__main__":
    main()
