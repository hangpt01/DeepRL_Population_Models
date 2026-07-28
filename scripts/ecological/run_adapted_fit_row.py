#!/usr/bin/env python3
"""Run one return-blind adapted-mechanistic dynamics-fit manifest row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_ecology_benchmark.config import load_config
from real_ecology_benchmark.faithful_fit import (
    build_candidate_bank,
    load_or_fit_mechanistic_model,
)
from real_ecology_benchmark.pipeline import (
    _hidden_method_context,
    _load_or_fit_public_surrogate,
    ensure_dataset,
)
from run_real_manifest_row import apply_row_config, read_manifest_row, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("index", type=int)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    row = read_manifest_row(args.manifest, args.index)
    if row.get("run_stage") != "dynamics_fit":
        raise SystemExit("fit runner accepts only run_stage=dynamics_fit")
    cfg = load_config(args.config)
    cfg, cell, _ = apply_row_config(cfg, row, args.output_root)
    dataset = ensure_dataset(cfg, regenerate=False)
    surrogate, _surrogate_path, _surrogate_status = _load_or_fit_public_surrogate(cfg, dataset)
    context = _hidden_method_context(cfg, dataset, surrogate)
    started = time.perf_counter()
    if row["method"] in {
        "plus_adapted_mechanistic_pbvi",
        "plus_adapted_ricker_only_pbvi",
    }:
        bank, statuses = build_candidate_bank(
            dataset,
            context,
            cfg.faithful.model,
            cfg.faithful.fit,
            cfg.seed + 51_000,
            cfg.faithful.fit_cache_dir,
        )
        fits = bank.fits
    elif row["method"] == "moor_adapted_ricker_misspec_pbvi":
        fit, status = load_or_fit_mechanistic_model(
            dataset,
            context,
            "ricker",
            cfg.faithful.model,
            cfg.faithful.fit,
            cfg.seed + 47_000,
            candidate_id="moor_ricker_00",
            cache_dir=cfg.faithful.fit_cache_dir,
        )
        fits, statuses = (fit,), (status,)
    else:
        raise SystemExit(f"unsupported adapted fit method: {row['method']}")
    payload = {
        "completion_status": "complete",
        "receipt_write": "atomic_replace",
        "run_stage": "dynamics_fit",
        "return_fields_opened": False,
        "method": row["method"],
        "cell": str(cell),
        "transition_data_hash": fits[0].transition_data_hash,
        "fit_cache_keys": [fit.fit_cache_key for fit in fits],
        "parameter_hashes": [fit.model.parameter_hash() for fit in fits],
        "cache_statuses": list(statuses),
        "candidate_count": len(fits),
        "target_rows": dataset.metadata["target_rows"],
        "actual_rows": dataset.metadata["actual_rows"],
        "overshoot_rows": dataset.metadata["overshoot_rows"],
        "episode_count": dataset.metadata["episode_count"],
        "fit_seconds": time.perf_counter() - started,
    }
    target = Path(args.output_root) / "fit_receipts" / cell / row["method"] / "fit_receipt.json"
    write_json_atomic(target, payload, str(args.index))
    print(json.dumps({**payload, "receipt": str(target)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
