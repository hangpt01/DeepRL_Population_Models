#!/usr/bin/env python3
"""S6 one-cell fresh-seed fit/plan/evaluate timing probe."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
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
# build_method seeds the policy at cfg.seed+40_000; the frozen adapted methods
# then derive their demographic fit roots at +11_000 (PLUS) and +7_000 (MOOR).
PLUS_FIT_SEED_ROOT = COLLECTION_SEED + 51_000
MOOR_FIT_SEED = COLLECTION_SEED + 47_000


def one_row(path: Path, method: str):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["population"] == "Crab-eating fox"
            and row["environment"] == "ricker"
            and float(row["sigma_obs"]) == 0.2
            and row["method"] == method
        ]
    if len(rows) != 1:
        raise FileNotFoundError(f"A1 manifest identity match count={len(rows)}")
    return rows[0]


def configure(side: str, row: dict[str, str]):
    config_name = "plus_ricker_only_p10.yaml" if side == "plus" else "moor_ricker_p10.yaml"
    cfg = load_config(str(P10_PACKAGE / "configs" / config_name))
    root = OUT / side
    cfg, _cell, _eval = runner.apply_row_config(cfg, row, root)
    cfg.seed = COLLECTION_SEED
    cfg.faithful.fit_cache_dir = str(root / "fit_cache_fresh")
    cfg.validate()
    return cfg


def split(summary):
    diag = summary["fit_diagnostics"]
    fit = float(diag["dynamics_fit_seconds"])
    plan = float(diag["pomdp_init_seconds"]) + float(diag["planner_init_seconds"])
    return {
        "FIT_seconds": fit,
        "PLAN_seconds": plan,
        "EVAL_seconds": float(summary["evaluation_seconds"]),
        "COLLECT_seconds": float(summary["dataset_seconds"]),
        "SURROGATE_seconds": float(summary.get("surrogate_seconds", 0.0)),
        "peak_rss_mb": float(summary["peak_rss_mb"]),
        "return_mean": float(summary["operational_return_mean"]),
        "return_sd": float(summary["operational_return_sd"]),
        "fit_cache_hits": int(diag["fit_cache_hits"]),
        "fit_cache_misses": int(diag["fit_cache_misses"]),
    }


def main():
    require_scientific_tables()
    OUT.mkdir(parents=True, exist_ok=True)
    marker = OUT / "S6_RECEIPT.json"
    if marker.exists():
        raise RuntimeError("S6 output already exists; refusing a cache-contaminated rerun")
    plus_row = one_row(
        P10_PACKAGE / "manifests" / "plus_p10_plan_24.csv",
        "plus_adapted_ricker_only_pbvi",
    )
    moor_row = one_row(
        P10_PACKAGE / "manifests" / "moor_p10_plan_24.csv",
        "moor_adapted_ricker_misspec_pbvi",
    )
    plus_cfg = configure("plus", plus_row)
    plus_start = time.perf_counter()
    plus_summary = run_method(
        "plus_adapted_ricker_only_pbvi", plus_cfg, "faithful_internal", regenerate=True
    )
    plus_wall = time.perf_counter() - plus_start
    plus = split(plus_summary)
    plus["total_wall_seconds"] = plus_wall
    if plus["fit_cache_hits"] != 0 or plus["fit_cache_misses"] != 8:
        raise AssertionError("PLUS was not fitted from scratch")

    moor_cfg = configure("moor", moor_row)
    moor_cfg.dataset.output = plus_cfg.dataset.output
    moor_cfg.dataset.private_output = plus_cfg.dataset.private_output
    moor_start = time.perf_counter()
    moor_summary = run_method(
        "moor_adapted_ricker_misspec_pbvi",
        moor_cfg,
        "faithful_internal",
        regenerate=False,
    )
    moor_wall = time.perf_counter() - moor_start
    moor = split(moor_summary)
    moor["total_wall_seconds"] = moor_wall
    if moor["fit_cache_hits"] != 0 or moor["fit_cache_misses"] != 1:
        raise AssertionError("MOOR was not fitted from scratch")
    plausible = all(-20.0 < x["return_mean"] < 20.0 for x in (plus, moor))
    receipt = {
        "schema": "S6_fit_timing_probe_v1",
        "cell": "A1 fox/ricker/sigma0.2",
        "new_collection_seed": COLLECTION_SEED,
        "new_fit_seeds": {
            "PLUS_candidate_root": PLUS_FIT_SEED_ROOT,
            "MOOR_ricker": MOOR_FIT_SEED,
        },
        "accepted_seed_distinct": True,
        "fits_from_scratch": True,
        "recomputed_fits": 9,
        "plus": plus,
        "moor": moor,
        "plausible_fox_return_sanity": plausible,
        "accepted_artifacts_modified": False,
    }
    marker.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
