#!/usr/bin/env python3
"""Run one decision-relevance gate row from a real-ecology manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark.config import load_config
from real_ecology_benchmark.gate import run_decision_gate, save_gate

from run_real_manifest_row import (
    apply_backend_overrides,
    apply_row_config,
    gate_path_for,
    read_manifest_row,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("index", type=int)
    parser.add_argument("--config", default=str(ROOT / "configs" / "real_experiment.yaml"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "real_reward_modes_20260703"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--backend", choices=["numpy", "cupy"], default=None,
                        help="compute backend (default: config, or $BACKEND env)")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--backend-strict", choices=["true", "false"], default=None)
    args = parser.parse_args()

    row = read_manifest_row(args.manifest, args.index)
    cfg = load_config(args.config)
    cfg, cell, _eval_cell = apply_row_config(cfg, row, args.output_root)
    cfg = apply_backend_overrides(cfg, args)
    result = run_decision_gate(cfg, episodes=args.episodes)
    gate_path = gate_path_for(
        args.output_root, cell, str(result.get("compute_backend_effective", "unknown"))
    )
    save_gate(result, gate_path)
    compact = {key: value for key, value in result.items() if key != "records"}
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
