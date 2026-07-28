#!/usr/bin/env python3
"""Run one row from a generated CSV manifest."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark.config import environment_with_kind_defaults, load_config
from real_ecology_benchmark.collector import calibration_summary
from real_ecology_benchmark.dataset import load_private
from real_ecology_benchmark.pipeline import ensure_dataset, run_method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("index", type=int)
    parser.add_argument("--config", default=str(ROOT / "configs/synthetic_full.yaml"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--require-gate", action="store_true")
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="diagnostic override: run even when healthy-start collapse is outside [0.15,0.24]",
    )
    args = parser.parse_args()
    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if int(row["index"]) == args.index]
    if len(matches) != 1:
        raise SystemExit(f"manifest index {args.index} matched {len(matches)} rows")
    row = matches[0]
    cfg = load_config(args.config)
    cfg.environment = environment_with_kind_defaults(cfg.environment, row["environment"])
    cfg.environment = replace(
        cfg.environment,
        num_actions=int(row["num_actions"]),
        observation_noise_sigma=float(row["sigma_obs"]),
    )
    cell = f"{row['environment']}_{row['num_actions']}a_sigma{row['sigma_obs']}"
    root = Path(args.output_root)
    if args.require_gate and row["environment"] in {"allee", "regime"} and float(row["sigma_obs"]) <= 0.2:
        gate_path = root / "gates" / f"{cell}.json"
        if not gate_path.exists():
            raise SystemExit(f"required gate artifact is missing: {gate_path}")
        with gate_path.open("r", encoding="utf-8") as handle:
            gate = json.load(handle)
        if not gate.get("passed", False):
            raise SystemExit(f"hard gate failed for {cell}")
    cfg.dataset.output = str(root / "data" / cell / "public.npz")
    cfg.dataset.private_output = str(root / "private" / cell / "truth.npz")
    cfg.evaluation.output_dir = str(root / "evaluation" / cell)
    dataset = ensure_dataset(cfg, regenerate=False)
    private = load_private(cfg.dataset.private_output)
    calibration = calibration_summary(
        dataset, private, cfg.environment.safety_threshold
    )
    calibration_dir = root / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = calibration_dir / f"{cell}.json"
    temporary = calibration_path.with_name(
        f".{calibration_path.stem}.{row['index']}.tmp"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2, sort_keys=True)
    temporary.replace(calibration_path)
    if not calibration["collapse_band_pass"] and not args.allow_uncalibrated:
        raise SystemExit(
            "calibration failed for "
            f"{cell}: healthy-start incident collapse rate "
            f"{calibration['incident_collapse_rate_healthy_starts']:.4f} "
            "is outside [0.15, 0.24]"
        )
    summary = run_method(row["method"], cfg, row["filter"], regenerate=False)
    print(summary)


if __name__ == "__main__":
    main()
