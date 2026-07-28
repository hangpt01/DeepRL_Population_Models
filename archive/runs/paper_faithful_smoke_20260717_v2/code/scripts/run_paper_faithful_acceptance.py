#!/usr/bin/env python3
"""Audit completed faithful smoke rows without interpreting method returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-rows", type=int, default=2)
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
        if any(item.get("external_invocation") for item in planner["planners"]):
            raise SystemExit(f"PBVI smoke unexpectedly reports an external invocation: {planner_path}")
        records.append({
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
            "privacy_path": str(privacy_path),
        })
    payload = {
        "passed": all(record["finite_objectives"] for record in records),
        "return_fields_opened": False,
        "rows": records,
    }
    output = root / "acceptance.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
