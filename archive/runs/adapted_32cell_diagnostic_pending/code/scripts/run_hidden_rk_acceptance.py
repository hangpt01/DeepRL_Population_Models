#!/usr/bin/env python3
"""Run the preregistered two-cell hidden-r/K method smoke battery."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark.config import load_config, real_environment_like
from real_ecology_benchmark.dataset import assert_public_schema
from real_ecology_benchmark.pipeline import ensure_dataset, run_method


CELLS = (
    ("Amur tiger", "ricker", 0.0, "safe"),
    ("Iberian lynx", "allee", 0.2, "safe"),
)
METHODS = (
    ("refplan", "learned"),
    ("bamcts", "learned"),
    ("ogsrl", "learned"),
    ("moor_native", "native_discrete"),
    ("plus_native", "native_discrete"),
)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "real_default.yaml"))
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "outputs" / "hidden_rk" / "acceptance"),
    )
    args = parser.parse_args()
    root = Path(args.output_root)
    records = []
    for population, family, sigma, reward_mode in CELLS:
        cfg = load_config(args.config)
        cfg.environment = real_environment_like(cfg.environment, population, family)
        cfg.environment = replace(
            cfg.environment,
            expose_rk="hidden",
            observation_noise_sigma=sigma,
            reward_mode=reward_mode,
        )
        cfg.dataset.transitions = 4000
        cfg.dataset.episode_length = 25
        cfg.evaluation.seeds = [9001]
        cfg.evaluation.episodes_per_seed = 1
        cfg.evaluation.horizon = 8
        cfg.training.plot = False
        cell = Path("regime_hidden") / f"reward_{reward_mode}" / slug(population) / family / f"sigma_{sigma:g}"
        cfg.dataset.output = str(root / "datasets" / cell / "public.npz")
        cfg.dataset.private_output = str(root / "private" / cell / "truth.npz")
        cfg.evaluation.output_dir = str(root / "evaluation" / cell)
        dataset = ensure_dataset(cfg, regenerate=False)
        assert_public_schema(cfg.dataset.output)
        if not 4000 <= len(dataset) <= 4024:
            raise AssertionError(f"invalid episode-preserving row count: {len(dataset)}")
        for method, filter_mode in METHODS:
            summary = run_method(method, cfg, filter_mode, regenerate=False)
            if summary["expose_rk"] != "hidden" or summary["fallback_count_mean"] < 0:
                raise AssertionError(f"invalid hidden summary for {method}: {summary}")
            records.append(
                {
                    "population": population,
                    "private_family": family,
                    "sigma_obs": sigma,
                    "method": method,
                    "filter": filter_mode,
                    "expose_rk": summary["expose_rk"],
                    "target_rows": summary["target_rows"],
                    "actual_rows": summary["actual_rows"],
                    "overshoot_rows": summary["overshoot_rows"],
                    "risk_fallback": summary["public_surrogate_diagnostics"]["risk_fallback"],
                    "output_dir": summary["output_dir"],
                }
            )
    report = {"passed": True, "rows": records}
    root.mkdir(parents=True, exist_ok=True)
    (root / "acceptance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
