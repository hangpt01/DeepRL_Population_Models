#!/usr/bin/env python3
"""Regenerate the four failed families' 75k datasets with the calibrated
collector profiles and confirm every cell now satisfies [0.15, 0.24] BEFORE any
method rows are submitted.  Successful families (allee_10a, regime_10a) are not
touched.  Run after the original array has drained.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark.collector import calibration_summary  # noqa: E402
from real_ecology_benchmark.config import environment_with_kind_defaults, load_config  # noqa: E402
from real_ecology_benchmark.dataset import load_private  # noqa: E402
from real_ecology_benchmark.pipeline import ensure_dataset  # noqa: E402

FAILED = [("allee", 5), ("regime", 5), ("theta", 5), ("theta", 10)]
SIGMAS = [0.0, 0.1, 0.2, 0.4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/synthetic_full.yaml"))
    ap.add_argument("--output-root", default=str(ROOT / "outputs"))
    args = ap.parse_args()
    root = Path(args.output_root)
    base = load_config(args.config)
    all_pass = True
    results = []
    for kind, na in FAILED:
        for sigma in SIGMAS:
            cfg = load_config(args.config)
            cfg.environment = environment_with_kind_defaults(base.environment, kind)
            cfg.environment = replace(
                cfg.environment, num_actions=na, observation_noise_sigma=sigma
            )
            cell = f"{kind}_{na}a_sigma{sigma}"
            cfg.dataset.output = str(root / "data" / cell / "public.npz")
            cfg.dataset.private_output = str(root / "private" / cell / "truth.npz")
            dataset = ensure_dataset(cfg, regenerate=True)
            private = load_private(cfg.dataset.private_output)
            summary = calibration_summary(
                dataset, private, cfg.environment.safety_threshold
            )
            rate = summary["incident_collapse_rate_healthy_starts"]
            ok = summary["collapse_band_pass"]
            all_pass &= bool(ok)
            freqs = summary["action_frequency"]
            results.append((cell, rate, ok, min(freqs.values()), len(freqs)))
            print(
                f"{cell:24} rate={rate:.4f} band_pass={ok} "
                f"min_action_freq={min(freqs.values()):.4f} actions={len(freqs)}/{na} "
                f"profile={summary['collector_profile']['name']}",
                flush=True,
            )
    out = root / "calibration" / "regenerated_failed_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"all_pass": all_pass, "cells": [
            {"cell": c, "rate": r, "pass": p, "min_action_freq": m, "n_actions": n}
            for c, r, p, m, n in results
        ]}, indent=2))
    print(f"\nALL 16 CELLS PASS: {all_pass}", flush=True)
    if not all_pass:
        raise SystemExit("at least one regenerated cell is outside [0.15, 0.24]")


if __name__ == "__main__":
    main()
