#!/usr/bin/env python3
"""Constant-action baseline on Egyptian vulture / ricker / 0.1 (evaluator only, no fitting).

Rolls a fixed action over the registered 20 evaluation seeds and reports the discounted
return_mean plus the M6 decomposition (utility / cost / penalty). Runs a0 (do-nothing) and
a5 (the deployed PLUS/MOOR action) so the a5 result validates the rollout against the accepted
cell (-187.265) and the pair quantifies the surrogate-induced avoidable loss.

Not a method, not a policy fit: this is a raw evaluator rollout. process_noise_sigma=0 so the
true trajectory is deterministic per seed; the constant action ignores observations, so no
filter/belief is involved.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIAGNOSTICS_DIR))
from repo_paths import (  # noqa: E402
    ECOLOGICAL_SCRIPTS,
    ECOLOGICAL_SRC,
    P10_DATA_ROOT,
    P10_PACKAGE,
    require_scientific_tables,
)

sys.path[:0] = [str(ECOLOGICAL_SCRIPTS), str(ECOLOGICAL_SRC)]
import run_real_manifest_row as rrmr
from real_ecology_benchmark.config import load_config
from real_ecology_benchmark.envs import make_env

SEEDS = [b + i for b in (7001, 7051, 7101, 7151, 7201) for i in range(4)]
GAMMA = 0.95


def env_cfg_for(pop, env, sigma):
    row = next(r for r in __import__("csv").DictReader(open(P10_PACKAGE / "manifests" / "moor_p10_plan_24.csv"))
               if r["population"] == pop and r["environment"] == env and r["sigma_obs"] == sigma)
    cfg = load_config(str(P10_PACKAGE / "configs" / "moor_ricker_p10.yaml"))
    cfg, _c, _e = rrmr.apply_row_config(cfg, row, P10_DATA_ROOT / "moor")
    cfg.validate()
    return cfg.environment


def roll(env_cfg, action):
    rets, utils, costs, pens = [], [], [], []
    for seed in SEEDS:
        e = make_env(env_cfg); e.reset(seed)
        g = 1.0; ret = util = cost = pen = 0.0
        for _t in range(50):
            res = e.step(action); info = res.evaluator_info
            ret += g * float(res.reward)
            util += g * float(info["state"] / (info["state"] + env_cfg.K_ref))
            cost += g * float(e.actions[action].cost)
            pen += g * 10.0 * float(bool(info["safety_penalty_applied"]))
            g *= GAMMA
        rets.append(ret); utils.append(util); costs.append(cost); pens.append(pen)
    return (float(np.mean(rets)), float(np.std(rets, ddof=1)),
            float(np.mean(utils)), float(np.mean(costs)), float(np.mean(pens)))


def main():
    require_scientific_tables()
    ec = env_cfg_for("Egyptian vulture", "ricker", "0.1")
    print(f"env: K_ref={ec.K_ref} s_safe={ec.safety_threshold} N0={ec.N0}")
    for a in (0, 5):
        rm, rsd, u, c, p = roll(ec, a)
        print(f"a{a}: return_mean={rm:.4f} (sd {rsd:.3e}) | disc_utility={u:.4f} disc_cost={c:.4f} disc_penalty={p:.4f}")
    r0 = roll(ec, 0)[0]; r5 = roll(ec, 5)[0]
    print(f"\na0 - a5 = {r0 - r5:+.4f} (== a5 discounted cost, since trajectories are identical)")
    print(f"accepted vulture/ricker/0.1 (MOOR/PLUS a5) = -187.26465 ; best accepted method (RefPlan) = -186.768")
    print(f"a0 beats RefPlan by {r0 - (-186.768):+.4f}")


if __name__ == "__main__":
    main()
