#!/usr/bin/env python3
"""Create the authorized Phase 2E 4-row or 64-row manifest; submit nothing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark import realdata


METHODS = (
    "refplan",
    "ogsrl",
    "bamcts",
    "ensemble_value_disagreement_pessimism",
)
POPULATIONS = ("Amur tiger", "Egyptian vulture")
FAMILIES = ("ricker", "allee", "theta", "regime")
SIGMAS = (0.0, 0.4)


def rows_for(mode: str) -> list[dict[str, object]]:
    scenarios = [
        (population, family, sigma)
        for population in POPULATIONS
        for family in FAMILIES
        for sigma in SIGMAS
    ]
    if mode == "timing_preflight":
        scenarios = [("Egyptian vulture", "regime", 0.4)]
    recoverable = set(realdata.recoverable_population_names())
    rows = []
    for population, family, sigma in scenarios:
        for method in METHODS:
            rows.append({
                "index": len(rows),
                "package_role": f"phase2e_{mode}",
                "reward_mode": "safe",
                "population": population,
                "recoverable": population in recoverable,
                "environment": family,
                "num_actions": realdata.NUM_REAL_ACTIONS,
                "sigma_obs": sigma,
                "method": method,
                "filter": "learned",
                "expose_rk": "hidden",
                "target_rows": 4000,
                "actual_rows": "",
                "overshoot_rows": "",
                "episode_count": "",
                "collection_seed": 116,
                "episode_length": 25,
                "evaluation_seed": 9001,
                "evaluation_episodes": 1,
                "evaluation_horizon": 50,
                "horizon": 5,
                "sequences": 96,
                "pessimism": 0.5,
                "bamcts_depth": 8,
                "bamcts_simulations": 256,
                "ogsrl_cost_horizon": 25,
                "ogsrl_deployment_rollouts": 256,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("timing_preflight", "limited_canary"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = rows_for(args.mode)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "mode": args.mode,
        "output": str(target),
        "rows": len(rows),
        "scenarios": len(rows) // len(METHODS),
        "methods": list(METHODS),
        "submitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()
