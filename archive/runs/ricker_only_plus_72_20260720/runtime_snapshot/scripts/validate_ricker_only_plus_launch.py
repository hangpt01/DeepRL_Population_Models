#!/usr/bin/env python3
"""Validate and dry-run the fixed 72-cell Ricker-only PLUS launch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


METHOD = "plus_adapted_ricker_only_pbvi"
FAMILIES = {"ricker", "allee", "theta", "regime"}
SIGMAS = {"0.1", "0.2"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate(
    fit_path: Path, plan_path: Path, expected_rows: int = 72
) -> dict[str, object]:
    fit, plan = rows(fit_path), rows(plan_path)
    if len(fit) != expected_rows or len(plan) != expected_rows:
        raise RuntimeError(
            f"fit and plan manifests must each contain exactly {expected_rows} rows"
        )
    expected_positions = list(range(expected_rows))
    for label, values in (("fit", fit), ("plan", plan)):
        positions = [int(row["index"]) for row in values]
        if positions != expected_positions:
            raise RuntimeError(f"{label} positions must be exactly 0..71 in file order")
        if {row["method"] for row in values} != {METHOD}:
            raise RuntimeError(f"{label} contains an unsupported method")
        if {row["candidate_family"] for row in values} != {"ricker"}:
            raise RuntimeError(f"{label} is not Ricker-only")
        if {row["candidate_count"] for row in values} != {"8"}:
            raise RuntimeError(f"{label} candidate count is not eight")
        if {row["prior"] for row in values} != {"uniform"}:
            raise RuntimeError(f"{label} prior is not uniform")
    pair_fields = (
        "population",
        "environment",
        "sigma_obs",
        "method",
        "candidate_count",
        "candidate_construction",
        "candidate_seed_root",
        "fit_cell",
        "fit_receipt_path",
        "runtime_digest_reference",
        "config_sha256",
    )
    for position, (left, right) in enumerate(zip(fit, plan)):
        if any(left[field] != right[field] for field in pair_fields):
            raise RuntimeError(f"fit/plan pair mismatch at position {position}")
        if left["run_stage"] != "dynamics_fit" or right["run_stage"] != "plan_evaluate":
            raise RuntimeError(f"stage mismatch at position {position}")
    populations = {row["population"] for row in fit}
    if expected_rows == 72:
        if len(populations) != 9:
            raise RuntimeError("manifest must contain all nine populations")
        for population in populations:
            block = [row for row in fit if row["population"] == population]
            if len(block) != 8:
                raise RuntimeError(f"population {population!r} does not have eight cells")
            if {row["environment"] for row in block} != FAMILIES:
                raise RuntimeError(f"population {population!r} lacks a four-family block")
            if {row["sigma_obs"] for row in block} != SIGMAS:
                raise RuntimeError(f"population {population!r} lacks both noise levels")
    fit_outputs = {row["fit_receipt_path"] for row in fit}
    plan_outputs = {row["evaluation_artifact_dir"] for row in plan}
    if len(fit_outputs) != expected_rows or len(plan_outputs) != expected_rows:
        raise RuntimeError("output path collision detected")
    return {
        "fit_manifest_sha256": sha256(fit_path),
        "plan_manifest_sha256": sha256(plan_path),
        "fit_rows": expected_rows,
        "plan_rows": expected_rows,
        "population_count": len(populations),
        "families": sorted({row["environment"] for row in fit}),
        "sigma_levels": sorted({row["sigma_obs"] for row in fit}),
        "unique_fit_outputs": len(fit_outputs),
        "unique_plan_outputs": len(plan_outputs),
        "return_fields_opened": False,
    }


def commands(
    fit_path: Path,
    plan_path: Path,
    code_root: Path,
    run_root: Path,
    python_bin: Path,
    config_path: Path | None = None,
) -> list[dict[str, object]]:
    result = []
    for stage, manifest in (("fit", fit_path), ("plan", plan_path)):
        runner = (
            code_root / "scripts/run_ricker_only_fit_locked.py"
            if stage == "fit"
            else code_root / "scripts/run_ricker_only_plan_with_receipt.py"
        )
        for position in range(len(rows(manifest))):
            result.append(
                {
                    "stage": stage,
                    "array_position": position,
                    "command": [
                        str(python_bin),
                        str(code_root / "scripts/ricker_only_thread_entry.py"),
                        str(runner),
                        str(manifest),
                        str(position),
                        "--config",
                        str(
                            config_path
                            or code_root
                            / "configs/paper_faithful_hidden_ricker_only_plus_v1.yaml"
                        ),
                        "--output-root",
                        str(run_root),
                    ],
                    "threads": 1,
                    "return_fields_opened": False,
                }
            )
    return result


def cell_workers(resolved: list[dict[str, object]]) -> list[dict[str, object]]:
    by_position: dict[int, dict[str, object]] = {}
    for item in resolved:
        position = int(item["array_position"])
        by_position.setdefault(position, {})[str(item["stage"])] = item["command"]
    workers = []
    for position in sorted(by_position):
        pair = by_position[position]
        if set(pair) != {"fit", "plan"}:
            raise RuntimeError(f"cell worker {position} lacks an exact fit/plan pair")
        workers.append({
            "array_position": position,
            "fit_command": pair["fit"],
            "plan_command": pair["plan"],
            "dependency": "plan starts only after fit exits zero in the same allocation",
            "allocated_tasks": 1,
            "threads": 1,
            "return_fields_opened": False,
        })
    return workers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-manifest", required=True, type=Path)
    parser.add_argument("--plan-manifest", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--python-bin", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--expected-rows", type=int, default=72)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = {
        "validation": validate(
            args.fit_manifest, args.plan_manifest, args.expected_rows
        )
    }
    if args.dry_run:
        payload["commands"] = commands(
            args.fit_manifest,
            args.plan_manifest,
            args.code_root,
            args.run_root,
            args.python_bin,
            args.config,
        )
        payload["dry_run_count"] = len(payload["commands"])
        payload["cell_workers"] = cell_workers(payload["commands"])
        payload["cell_worker_count"] = len(payload["cell_workers"])
        payload["maximum_simultaneous_plus_tasks"] = min(
            args.expected_rows, 72
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
