#!/usr/bin/env python3
"""Negative self-test for the accepted-cell parity/cache gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_replay_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "diagnostics" / "replay" / "run_diagnostic_replay.py"
    spec = importlib.util.spec_from_file_location("diagnostic_replay_gate_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    replay = load_replay_module()
    accepted = {
        "return_mean": 1.0,
        "return_sd": 0.0,
        "unsafe_fraction_mean": 0.0,
        "persistence_mean": 1.0,
        "collapse_entry_mean": 0.0,
        "min_population_mean": 2.0,
        "economic_cost_mean": 0.0,
    }
    corrupted = dict(accepted)
    corrupted["return_mean"] += 1e-6
    result = replay.parity(corrupted, accepted)
    if result["verdict"] == "PASS":
        raise AssertionError(f"corrupted comparison unexpectedly passed: {result}")
    replay.enforce_acceptance(result, reuse_ok=True, recomputed_fits=0)
    raise AssertionError("gate returned normally for a corrupted comparison")


if __name__ == "__main__":
    raise SystemExit(main())
