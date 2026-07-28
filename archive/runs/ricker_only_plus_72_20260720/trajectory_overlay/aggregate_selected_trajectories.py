#!/usr/bin/env python3
"""Aggregate 24 sealed Ricker-only PLUS trajectory cells without opening frozen returns."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SELECTED = tuple((*range(0, 8), *range(16, 24), *range(40, 48)))
METHOD = "plus_adapted_ricker_only_pbvi"
PRODUCER = ROOT / "capture_ricker_only_plus_trajectories.py"
WAIVER = {
    "scoped_selected_trajectory_waiver": True,
    "global_acceptance": "INCOMPLETE",
    "global_missing_index": 51,
    "global_missing_cell": "Iberian lynx / Allee / sigma_obs=0.2",
    "selected_cells_complete": "24/24",
    "missing_global_cell_outside_selected_subset": True,
    "full_run_claims_prohibited": True,
}
LONG_COLUMNS = (
    "population", "population_scope", "environment", "sigma_obs", "reward_mode",
    "method", "episode", "seed", "timestep", "state_pre", "state_post",
    "observation_pre", "observation_post", "belief_mean", "action", "reward_true",
    "reward_public", "danger", "unsafe", "mvp", "terminated", "belief_low",
    "belief_high", "candidate_weights", "candidate_entropy", "selected_candidate_index",
)
ARRAY_KEYS = (
    "state_pre", "state_post", "observation_pre", "observation_post", "belief_mean",
    "belief_low", "belief_high", "action", "reward_true", "reward_public", "danger",
    "unsafe", "mvp", "terminated", "candidate_entropy", "selected_candidate_index",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_cells() -> list[tuple[dict, dict[str, np.ndarray], Path, Path]]:
    receipts = sorted((RESULTS / "receipts").glob("cell_*.json"))
    if len(receipts) != 24:
        raise SystemExit(f"expected 24 cell receipts, found {len(receipts)}")
    loaded = []
    seen = set()
    for receipt_path in receipts:
        meta = json.loads(receipt_path.read_text(encoding="utf-8"))
        index = int(meta.get("manifest_index", -1))
        if index not in SELECTED or index in seen:
            raise SystemExit(f"unexpected/duplicate selected index {index}")
        seen.add(index)
        if (
            meta.get("schema") != "ricker_only_plus_trajectory_v1"
            or meta.get("sealed") is not True
            or meta.get("return_fields_opened") is not False
            or meta.get("method") != METHOD
            or meta.get("episodes") != 20
            or meta.get("horizon") != 50
            or meta.get("reward_mode") != "safe"
            or meta.get("cell_completion_status") != "complete"
        ):
            raise SystemExit(f"invalid sealed trajectory receipt: {receipt_path}")
        for key, expected in WAIVER.items():
            if meta.get(key) != expected:
                raise SystemExit(f"waiver mismatch {key} in {receipt_path}")
        dataset_gate = meta.get("provenance", {}).get("dataset_registry", {})
        if (
            dataset_gate.get("reuse_status") != "exact_ecological_byte_copy"
            or dataset_gate.get("action_channel_reuse") != "exact_public_array_byte_copy"
        ):
            raise SystemExit(f"dataset reuse gate failed in {receipt_path}")
        matches = list((RESULTS / "raw_npz").glob(f"cell_{index:02d}_*.npz"))
        if len(matches) != 1:
            raise SystemExit(f"expected one NPZ for index {index}, found {len(matches)}")
        npz_path = matches[0]
        if sha256(npz_path) != meta.get("trajectory_npz_sha256"):
            raise SystemExit(f"trajectory NPZ hash mismatch for index {index}")
        with np.load(npz_path, allow_pickle=False) as payload:
            embedded = json.loads(str(payload["meta"].item()))
            if int(embedded.get("manifest_index", -1)) != index:
                raise SystemExit(f"embedded metadata mismatch for index {index}")
            arrays = {key: np.asarray(payload[key]) for key in ARRAY_KEYS}
            arrays["candidate_weights"] = np.asarray(payload["candidate_weights"])
        if any(value.shape != (20, 50) for key, value in arrays.items() if key != "candidate_weights"):
            raise SystemExit(f"trajectory shape mismatch for index {index}")
        if arrays["candidate_weights"].shape != (20, 50, 8):
            raise SystemExit(f"candidate-weight shape mismatch for index {index}")
        loaded.append((meta, arrays, receipt_path, npz_path))
    if seen != set(SELECTED):
        raise SystemExit(f"selected coverage mismatch: {sorted(seen)}")
    return sorted(loaded, key=lambda item: int(item[0]["manifest_index"]))


def write_long(cells: list[tuple[dict, dict[str, np.ndarray], Path, Path]], target: Path) -> int:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    count = 0
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LONG_COLUMNS)
        writer.writeheader()
        for meta, arrays, _receipt, _npz in cells:
            for episode, seed in enumerate(meta["seeds"]):
                for timestep in range(50):
                    row = {
                        "population": meta["population"],
                        "population_scope": meta["population_scope"],
                        "environment": meta["environment"],
                        "sigma_obs": meta["sigma_obs"],
                        "reward_mode": meta["reward_mode"],
                        "method": meta["method"],
                        "episode": episode,
                        "seed": seed,
                        "timestep": timestep,
                    }
                    for key in ARRAY_KEYS:
                        row[key] = float(arrays[key][episode, timestep])
                    row["action"] = int(row["action"]) if np.isfinite(row["action"]) else ""
                    row["candidate_weights"] = json.dumps(
                        arrays["candidate_weights"][episode, timestep].tolist(),
                        separators=(",", ":"),
                    )
                    writer.writerow(row)
                    count += 1
    if count != 24000:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"expected 24000 long rows, wrote {count}")
    os.replace(temporary, target)
    return count


def write_extrema(cells: list[tuple[dict, dict[str, np.ndarray], Path, Path]], target: Path) -> None:
    columns = (
        "population", "population_scope", "environment", "sigma_obs", "reward_mode",
        "method", "episodes", "mean_episode_min_state_post", "mean_episode_max_state_post",
        "observed_min_state_post", "observed_max_state_post",
    )
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for meta, arrays, _receipt, _npz in cells:
            state = arrays["state_post"]
            writer.writerow({
                "population": meta["population"],
                "population_scope": meta["population_scope"],
                "environment": meta["environment"],
                "sigma_obs": meta["sigma_obs"],
                "reward_mode": meta["reward_mode"],
                "method": meta["method"],
                "episodes": 20,
                "mean_episode_min_state_post": float(np.nanmean(np.nanmin(state, axis=1))),
                "mean_episode_max_state_post": float(np.nanmean(np.nanmax(state, axis=1))),
                "observed_min_state_post": float(np.nanmin(state)),
                "observed_max_state_post": float(np.nanmax(state)),
            })
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-job-id", required=True)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    targets = (
        RESULTS / "ecological_only_trajectories_long.csv.gz",
        RESULTS / "ecological_only_population_extrema.csv",
        RESULTS / "ecological_only_trajectory_receipt.json",
    )
    if any(path.exists() for path in targets):
        raise SystemExit("refusing to overwrite an existing sealed aggregate")
    cells = load_cells()
    long_rows = write_long(cells, targets[0])
    write_extrema(cells, targets[1])
    receipt = {
        "schema": "ricker_only_plus_selected_trajectory_aggregate_v1",
        "sealed": True,
        "return_fields_opened": False,
        "frozen_return_comparison_performed": False,
        "capture_job_id": args.capture_job_id,
        "producer_sha256": sha256(PRODUCER),
        "populations": ["Egyptian vulture", "Amur tiger", "Puerto Rican parrot"],
        "selected_indices": list(SELECTED),
        "cells": 24,
        "methods": [METHOD],
        "episodes_per_method_cell": 20,
        "steps_per_episode": 50,
        "long_rows": long_rows,
        "dataset_alignment": "24/24 exact_ecological_byte_copy",
        "cell_receipt_sha256": {
            str(meta["manifest_index"]): sha256(receipt_path)
            for meta, _arrays, receipt_path, _npz in cells
        },
        "cell_npz_sha256": {
            str(meta["manifest_index"]): sha256(npz_path)
            for meta, _arrays, _receipt, npz_path in cells
        },
        "reconstructed_discounted_return_mean_sealed": {
            str(meta["manifest_index"]): meta["reconstructed_discounted_return_mean"]
            for meta, _arrays, _receipt, _npz in cells
        },
        "long_table_sha256": sha256(targets[0]),
        "population_extrema_sha256": sha256(targets[1]),
        **WAIVER,
    }
    atomic_json(targets[2], receipt)
    print(json.dumps({
        "status": "sealed_aggregate_complete",
        "cells": 24,
        "long_rows": long_rows,
        "return_fields_opened": False,
        **WAIVER,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
