#!/usr/bin/env python3
"""Generate all 24 observation-matched gate artifacts before method jobs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark.config import environment_with_kind_defaults, load_config
from real_ecology_benchmark.gate import run_decision_gate, save_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/synthetic_full.yaml"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs/gates"))
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    root = Path(args.output_root)
    hard_failures = []
    for kind in ("allee", "theta", "regime"):
        for num_actions in (5, 10):
            for sigma in (0.0, 0.1, 0.2, 0.4):
                cfg = load_config(args.config)
                cfg.environment = environment_with_kind_defaults(cfg.environment, kind)
                cfg.environment = replace(
                    cfg.environment,
                    num_actions=num_actions,
                    observation_noise_sigma=sigma,
                )
                result = run_decision_gate(cfg, args.episodes)
                cell = f"{kind}_{num_actions}a_sigma{sigma}"
                save_gate(result, root / f"{cell}.json")
                print(cell, "PASS" if result["passed"] else "FAIL", result["reward_gap"], result["collapse_gap"])
                if (
                    kind in {"allee", "regime"}
                    and sigma <= 0.2
                    and not result["passed"]
                ):
                    hard_failures.append(cell)
    if hard_failures:
        raise SystemExit(
            "hard decision gate failed; matrix submission remains held: "
            + ", ".join(hard_failures)
        )


if __name__ == "__main__":
    main()
