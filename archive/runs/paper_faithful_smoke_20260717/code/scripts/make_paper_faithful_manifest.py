#!/usr/bin/env python3
"""Create registered smoke or blinded-runtime manifests for faithful methods."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = (
    "plus_faithful_pbvi",
    "moor_faithful_ricker_misspec_pbvi",
)


def scientific_cells(mode: str):
    if mode == "smoke":
        return [("Amur tiger", "ricker", 0.1, "safe")]
    populations = (("Amur tiger", True, 0.0), ("Egyptian vulture", False, 0.2))
    return [
        (population, form, sigma, reward_mode)
        for population, _recoverable, sigma in populations
        for form in ("ricker", "allee", "theta", "regime")
        for reward_mode in ("safe", "yield")
    ]


def write_manifest(path: Path, mode: str) -> int:
    rows = []
    for population, form, sigma, reward_mode in scientific_cells(mode):
        for method in METHODS:
            rows.append({
                "index": len(rows),
                "reward_mode": reward_mode,
                "population": population,
                "recoverable": population != "Egyptian vulture",
                "environment": form,
                "num_actions": 11,
                "sigma_obs": sigma,
                "method": method,
                "filter": "faithful_internal",
                "expose_rk": "hidden",
                "target_rows": 160 if mode == "smoke" else 4000,
                "actual_rows": "",
                "overshoot_rows": "",
                "episode_count": "",
                "method_impl_version": "faithful_ecology_v1",
                "planner": "pbvi",
                "candidate_count": 4 if mode == "smoke" else 16,
                "prior": "uniform",
                "fit_budget_id": "smoke_s2_i12_m4" if mode == "smoke" else "registered_s8_i100_m16",
                "discretization_id": "smoke_b15_c5_o15" if mode == "smoke" else "registered_b41_c9_o41",
                "equation_version": "faithful_ecology_v1",
                "run_scope": mode,
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "canary"), default="smoke")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    rows = write_manifest(output, args.mode)
    print(json.dumps({"output": str(output), "rows": rows, "mode": args.mode}, indent=2))


if __name__ == "__main__":
    main()
