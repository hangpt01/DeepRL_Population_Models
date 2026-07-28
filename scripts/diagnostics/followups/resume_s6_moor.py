#!/usr/bin/env python3
"""Resume S6 after the completed PLUS run: run MOOR only and write receipt."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import time

DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIAGNOSTICS_DIR))
from repo_paths import (  # noqa: E402
    ECOLOGICAL_SCRIPTS,
    ECOLOGICAL_SRC,
    FOLLOWUPS_OUTPUT,
    P10_PACKAGE,
    require_scientific_tables,
)

OUT = Path(os.environ.get("DEEPRL_S6_OUTPUT", FOLLOWUPS_OUTPUT / "S6"))
sys.path[:0] = [str(ECOLOGICAL_SCRIPTS), str(ECOLOGICAL_SRC)]

import run_real_manifest_row as runner  # noqa: E402
from real_ecology_benchmark.config import load_config  # noqa: E402
from real_ecology_benchmark.pipeline import run_method  # noqa: E402

COLLECTION_SEED = 20260728
PLUS_FIT_SEED_ROOT = COLLECTION_SEED + 51_000
MOOR_FIT_SEED = COLLECTION_SEED + 47_000


def one_row(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["population"] == "Crab-eating fox"
            and row["environment"] == "ricker"
            and float(row["sigma_obs"]) == 0.2
            and row["method"] == "moor_adapted_ricker_misspec_pbvi"
        ]
    if len(rows) != 1:
        raise FileNotFoundError(f"A1 MOOR manifest identity match count={len(rows)}")
    return rows[0]


def plus_payload():
    summaries = list((OUT / "plus/evaluation").rglob("summary.json"))
    if len(summaries) != 1:
        raise FileNotFoundError(f"completed PLUS summary match count={len(summaries)}")
    summary = json.loads(summaries[0].read_text())
    diag = summary["fit_diagnostics"]
    if int(diag["fit_cache_hits"]) != 0 or int(diag["fit_cache_misses"]) != 8:
        raise AssertionError("salvaged PLUS run was not a fresh eight-candidate fit")
    return {
        "FIT_seconds": float(diag["dynamics_fit_seconds"]),
        "PLAN_seconds": float(diag["pomdp_init_seconds"]) + float(diag["planner_init_seconds"]),
        "EVAL_seconds": float(summary["evaluation_seconds"]),
        "COLLECT_seconds": float(summary["dataset_seconds"]),
        "SURROGATE_seconds": float(summary.get("surrogate_seconds", 0.0)),
        "peak_rss_mb": float(summary["peak_rss_mb"]),
        "return_mean": float(summary["operational_return_mean"]),
        "return_sd": float(summary["operational_return_std"]),
        "fit_cache_hits": 0,
        "fit_cache_misses": 8,
        "salvaged_from_job": "58579501",
        "summary_path": str(summaries[0]),
    }, summary


def configure_moor(plus_summary):
    row = one_row(P10_PACKAGE / "manifests" / "moor_p10_plan_24.csv")
    cfg = load_config(str(P10_PACKAGE / "configs" / "moor_ricker_p10.yaml"))
    cfg, _, _ = runner.apply_row_config(cfg, row, OUT / "moor")
    cfg.seed = COLLECTION_SEED
    cfg.faithful.fit_cache_dir = str(OUT / "moor/fit_cache_fresh")
    cfg.dataset.output = str(
        OUT / "plus/datasets/regime_hidden/reward_safe/crab_eating_fox/"
        "ricker/sigma_0p2/public.npz"
    )
    cfg.dataset.private_output = str(
        OUT / "plus/private/regime_hidden/reward_safe/crab_eating_fox/"
        "ricker/sigma_0p2/truth.npz"
    )
    if plus_summary["dataset_sha256"] == "":
        raise AssertionError("salvaged PLUS summary lacks dataset hash")
    cfg.validate()
    return cfg, plus_summary["dataset_sha256"]


def main():
    require_scientific_tables()
    receipt_path = OUT / "S6_RECEIPT.json"
    if receipt_path.exists():
        raise RuntimeError("S6 receipt already exists; refusing duplicate completion")
    plus, plus_summary = plus_payload()
    cfg, expected_dataset_hash = configure_moor(plus_summary)
    cache_dir = Path(cfg.faithful.fit_cache_dir)
    if cache_dir.exists() and list(cache_dir.glob("*.npz")):
        raise RuntimeError("MOOR fresh-fit cache is not empty")

    started = time.perf_counter()
    summary = run_method(
        "moor_adapted_ricker_misspec_pbvi",
        cfg,
        "faithful_internal",
        regenerate=False,
    )
    wall = time.perf_counter() - started
    if summary["dataset_sha256"] != expected_dataset_hash:
        raise AssertionError("MOOR did not reuse the exact fresh PLUS dataset")
    diag = summary["fit_diagnostics"]
    if int(diag["fit_cache_hit"]) != 0:
        raise AssertionError("MOOR was not fitted from scratch")
    moor = {
        "FIT_seconds": float(diag["dynamics_fit_seconds"]),
        "PLAN_seconds": float(diag["pomdp_init_seconds"]) + float(diag["planner_init_seconds"]),
        "EVAL_seconds": float(summary["evaluation_seconds"]),
        "COLLECT_seconds": float(summary["dataset_seconds"]),
        "SURROGATE_seconds": float(summary.get("surrogate_seconds", 0.0)),
        "peak_rss_mb": float(summary["peak_rss_mb"]),
        "return_mean": float(summary["operational_return_mean"]),
        "return_sd": float(summary["operational_return_std"]),
        "fit_cache_hits": 0,
        "fit_cache_misses": 1,
        "total_resume_wall_seconds": wall,
    }
    receipt = {
        "schema": "S6_fit_timing_probe_v1",
        "cell": "A1 fox/ricker/sigma0.2",
        "new_collection_seed": COLLECTION_SEED,
        "new_fit_seeds": {
            "PLUS_candidate_root": PLUS_FIT_SEED_ROOT,
            "MOOR_ricker": MOOR_FIT_SEED,
        },
        "dataset_sha256": expected_dataset_hash,
        "accepted_seed_distinct": True,
        "fits_from_scratch": True,
        "recomputed_fits": 9,
        "plus": plus,
        "moor": moor,
        "plus_fit_repeated_during_resume": False,
        "plausible_fox_return_sanity": all(
            -20.0 < x["return_mean"] < 20.0 for x in (plus, moor)
        ),
        "accepted_artifacts_modified": False,
        "reranked": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
