#!/usr/bin/env python3
"""Manifest for the general-method search-budget audit.

Grid (SERVER_PLAN_general_method_audit.md rev-4):
  cells   144 = 9 populations x 4 families x sigma {0.0, 0.4} x 2 reward modes
  configs  24 new = refplan 11 + bamcts 11 + ogsrl 2   (the as-run config is REUSED
           from the completed motivation run, not re-run)
  rows    3456

Two properties the plan depends on, both asserted here rather than hoped for:

1. ``config_tag`` is UNIQUE per (cell, method).  Without a config dimension in the
   output path, every tuning config for a cell resolves to the same directory and they
   silently overwrite each other.  A collision is a hard error at generation time.

2. Rows are emitted CONFIG-MAJOR, MOST-EXPENSIVE-CONFIG FIRST.  Config-major means a
   packed task's rows share one config, so its cost is predictable instead of a lottery
   over the cheap and expensive corners.  Expensive-first is longest-processing-time-
   first scheduling: the ~5.4 h BA-MCTS tasks start in wave 1 and the short ones
   backfill, rather than a 5.4 h task starting last and appending itself to the tail.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark import realdata

FAMILIES = ("ricker", "allee", "theta", "regime")
# sigma 0.1/0.2 dropped: in the motivation run their general-vs-native gaps were
# -1.119 and -1.118 -- within 0.001 of each other, interpolating between the kept
# endpoints (-0.698 at 0.0, -1.210 at 0.4).  They carry nothing the endpoints don't.
SIGMAS = (0.0, 0.4)
REWARD_MODES = ("safe", "yield")

FIELDS = [
    "index", "reward_mode", "population", "recoverable", "environment", "num_actions",
    "sigma_obs", "method", "filter", "config_tag",
    "horizon", "sequences", "pessimism",
    "bamcts_depth", "bamcts_simulations", "ogsrl_cost_horizon",
]

# As-run settings -- these configs are REUSED from the motivation run and must NOT be
# re-emitted here, or we would pay for them twice and pollute the comparison baseline.
AS_RUN = {
    "refplan": {"horizon": 5, "sequences": 96, "pessimism": 0.5},
    "bamcts": {"bamcts_depth": 5, "bamcts_simulations": 128, "pessimism": 0.5},
    "ogsrl": {"ogsrl_cost_horizon": 25},
}


def num(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def configs_for(method: str) -> list[dict]:
    """Return the NEW configs for a method, each with a cost weight and a unique tag."""

    out: list[dict] = []
    if method == "refplan":
        for horizon in (5, 10, 20):
            for sequences in (96, 256):
                for pessimism in (0.5, 0.0):
                    settings = {
                        "horizon": horizon, "sequences": sequences, "pessimism": pessimism,
                    }
                    if settings == AS_RUN["refplan"]:
                        continue
                    out.append({
                        "settings": settings,
                        "tag": f"h{horizon}_seq{sequences}_pess{num(pessimism)}",
                        "cost": (horizon * sequences) / (5 * 96),
                    })
    elif method == "bamcts":
        for depth in (5, 10, 20):
            for simulations in (128, 256):
                for pessimism in (0.5, 0.0):
                    settings = {
                        "bamcts_depth": depth,
                        "bamcts_simulations": simulations,
                        "pessimism": pessimism,
                    }
                    if settings == AS_RUN["bamcts"]:
                        continue
                    out.append({
                        "settings": settings,
                        "tag": f"d{depth}_sims{simulations}_pess{num(pessimism)}",
                        "cost": (depth * simulations) / (5 * 128),
                    })
    elif method == "ogsrl":
        # Phase 2D registers one common H_cost=25. It is a scientific unit, not
        # a performance-tuning axis, so the former rollout-horizon sweep is retired.
        return []
    else:
        raise ValueError(method)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["refplan", "bamcts", "ogsrl"], required=True,
                        help="one manifest per method: each array gets its own caps/BS/walltime")
    args = parser.parse_args()

    populations = list(realdata.population_names())
    recoverable = set(realdata.recoverable_population_names())
    cells = [
        (mode, pop, family, sigma)
        for mode in REWARD_MODES
        for pop in populations
        for family in FAMILIES
        for sigma in SIGMAS
    ]
    if len(cells) != 144:
        raise SystemExit(f"expected 144 cells, built {len(cells)}")

    configs = configs_for(args.method)
    # expensive-first: longest-processing-time-first scheduling (see module docstring)
    configs.sort(key=lambda c: -c["cost"])

    rows = []
    seen: set[tuple] = set()
    index = 0
    for config in configs:              # config-major
        for mode, pop, family, sigma in cells:
            key = (mode, pop, family, sigma, args.method, config["tag"])
            if key in seen:
                raise SystemExit(f"config_tag collision (would silently overwrite): {key}")
            seen.add(key)
            row = {
                "index": index,
                "reward_mode": mode,
                "population": pop,
                "recoverable": pop in recoverable,
                "environment": family,
                "num_actions": realdata.NUM_REAL_ACTIONS,
                "sigma_obs": sigma,
                "method": args.method,
                "filter": "learned",
                "config_tag": config["tag"],
                **{k: "" for k in FIELDS[10:]},
                **config["settings"],
            }
            rows.append(row)
            index += 1

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "output": str(target),
        "method": args.method,
        "cells": len(cells),
        "new_configs": len(configs),
        "rows": len(rows),
        "config_tags": [c["tag"] for c in configs],
        "cost_weight_total": round(sum(c["cost"] for c in configs), 2),
    }, indent=2))


if __name__ == "__main__":
    main()
