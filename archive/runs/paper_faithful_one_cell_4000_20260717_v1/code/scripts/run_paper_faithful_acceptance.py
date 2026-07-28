#!/usr/bin/env python3
"""Audit completed faithful smoke rows without interpreting method returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-rows", type=int, default=2)
    parser.add_argument("--target-rows", type=int)
    parser.add_argument("--require-bootstrap-map", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    fit_files = sorted(root.glob("evaluation/**/faithful_artifacts/faithful_fit.json"))
    planner_files = sorted(root.glob("evaluation/**/faithful_artifacts/planner_provenance.json"))
    privacy_files = sorted(root.glob("evaluation/**/faithful_artifacts/privacy_audit.json"))
    if not (len(fit_files) == len(planner_files) == len(privacy_files) == args.expected_rows):
        raise SystemExit(
            f"incomplete faithful smoke: fits={len(fit_files)}, "
            f"planners={len(planner_files)}, privacy={len(privacy_files)}"
        )
    records = []
    for fit_path, planner_path, privacy_path in zip(fit_files, planner_files, privacy_files):
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        planner = json.loads(planner_path.read_text(encoding="utf-8"))
        privacy = json.loads(privacy_path.read_text(encoding="utf-8"))
        if privacy.get("status") != "passed":
            raise SystemExit(f"privacy audit failed: {privacy_path}")
        if args.require_bootstrap_map and fit["method_impl_version"].startswith("plus_faithful"):
            if fit.get("candidate_construction") != "episode_bootstrap_map_v1":
                raise SystemExit(f"PLUS candidate construction is not registered: {fit_path}")
            if int(fit["candidate_count"]) != 16:
                raise SystemExit(f"PLUS candidate count is not 16: {fit_path}")
        if any(item.get("external_invocation") for item in planner["planners"]):
            raise SystemExit(
                f"PBVI smoke unexpectedly reports an external invocation: {planner_path}"
            )
        artifact_root = fit_path.parent
        pomdp_files = sorted(artifact_root.glob("pomdp_model_*.npz"))
        if len(pomdp_files) != int(fit["candidate_count"]):
            raise SystemExit(f"POMDP artifacts do not match candidate count: {artifact_root}")
        if not (artifact_root / "pbvi_policy_diagnostics.npz").exists():
            raise SystemExit(f"PBVI diagnostics are missing: {artifact_root}")
        records.append(
            {
                "fit_path": str(fit_path),
                "candidate_count": int(fit["candidate_count"]),
                "finite_objectives": all(
                    item["objective"] is not None and item["objective"] < float("inf")
                    for item in fit["fits"]
                ),
                "planner_path": str(planner_path),
                "planner_invocations": sum(
                    int(item["invocation_count"]) for item in planner["planners"]
                ),
                "pomdp_artifact_count": len(pomdp_files),
                "privacy_path": str(privacy_path),
            }
        )
    dataset_records = []
    if args.target_rows is not None:
        dataset_files = sorted(root.glob("datasets/**/public.npz"))
        if len(dataset_files) != 1:
            raise SystemExit(f"expected one shared public dataset, found {len(dataset_files)}")
        with np.load(dataset_files[0], allow_pickle=False) as data:
            actual_rows = int(len(data["actions"]))
            episode_ids = np.asarray(data["episode_id"])
            episode_count = int(len(np.unique(episode_ids)))
            maximum_episode_length = max(
                int(np.sum(episode_ids == episode)) for episode in np.unique(episode_ids)
            )
        overshoot_rows = actual_rows - args.target_rows
        if not 0 <= overshoot_rows < maximum_episode_length:
            raise SystemExit(
                f"dataset violates complete-episode target contract: target={args.target_rows}, "
                f"actual={actual_rows}, max_episode_length={maximum_episode_length}"
            )
        dataset_records.append(
            {
                "path": str(dataset_files[0]),
                "target_rows": args.target_rows,
                "actual_rows": actual_rows,
                "overshoot_rows": overshoot_rows,
                "episode_count": episode_count,
                "maximum_episode_length": maximum_episode_length,
            }
        )
    payload = {
        "passed": all(record["finite_objectives"] for record in records),
        "return_fields_opened": False,
        "rows": records,
        "datasets": dataset_records,
    }
    output = root / "acceptance.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
