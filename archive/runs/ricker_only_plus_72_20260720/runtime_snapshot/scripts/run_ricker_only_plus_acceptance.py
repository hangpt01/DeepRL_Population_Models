#!/usr/bin/env python3
"""Return-blind structural acceptance for the 72-cell Ricker-only PLUS run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any


METHOD = "plus_adapted_ricker_only_pbvi"
CONSTRUCTION = "ricker_only_episode_bootstrap_map_1full_7bootstrap_v1"
FORBIDDEN_FILES = {"summary.json", "comparative_summary.json", "ranking.json"}
FORBIDDEN_KEYS = {
    "operational_return",
    "true_return",
    "survival_return",
    "returns",
    "ranking",
    "rankings",
}


def expected_plan_artifact_names() -> set[str]:
    names = {
        "candidate_bank.json", "candidate_bank.npz", "faithful_fit.json",
        "faithful_fit.npz", "planner_provenance.json", "privacy_audit.json",
        "pbvi_policy_diagnostics.npz",
    }
    for index in range(8):
        names.update({
            f"candidate_{index:03d}.json", f"candidate_{index:03d}.npz",
            f"pomdp_model_{index:03d}.json", f"pomdp_model_{index:03d}.npz",
        })
    return names


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_forbidden(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN_KEYS:
                raise RuntimeError(f"forbidden return field at {location}.{key}")
            reject_forbidden(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden(child, f"{location}[{index}]")


def guarded_json(path: Path) -> dict[str, Any]:
    if path.name in FORBIDDEN_FILES:
        raise RuntimeError(f"acceptance cannot open {path.name}")
    raw = path.read_text(encoding="utf-8")
    for key in FORBIDDEN_KEYS:
        if re.search(rf'"{re.escape(key)}"\s*:', raw, flags=re.IGNORECASE):
            raise RuntimeError(f"forbidden return field in {path}: {key}")
    payload = json.loads(raw)
    reject_forbidden(payload)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def temporary_or_partial_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and (path.name.startswith(".") or ".tmp" in path.name)
    )


def accounting(job_ids: list[str]) -> dict[str, object]:
    if not job_ids:
        return {"states": {}, "allocated_core_hours": 0.0}
    result = subprocess.run(
        [
            "sacct",
            "-j",
            ",".join(job_ids),
            "-X",
            "-n",
            "-P",
            "-o",
            "State,ElapsedRaw,AllocCPUS",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    states: dict[str, int] = {}
    seconds = 0
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 3:
            continue
        state = fields[0].split()[0]
        states[state] = states.get(state, 0) + 1
        seconds += int(fields[1] or 0) * int(fields[2] or 0)
    return {"states": states, "allocated_core_hours": seconds / 3600.0}


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    fit_rows, plan_rows = manifest_rows(args.fit_manifest), manifest_rows(args.plan_manifest)
    failures: list[str] = []
    warnings: list[str] = []
    completed_fit = 0
    completed_plan = 0
    partial_files = temporary_or_partial_files(args.run_root)
    if len(fit_rows) != args.expected_rows or len(plan_rows) != args.expected_rows:
        failures.append(
            f"manifest row count is not {args.expected_rows}/{args.expected_rows}"
        )
    for position, row in enumerate(fit_rows):
        receipt_path = args.run_root / row["fit_receipt_path"]
        if not receipt_path.is_file():
            continue
        receipt = guarded_json(receipt_path)
        if (
            receipt.get("completion_status") != "complete"
            or receipt.get("receipt_write") != "atomic_replace"
        ):
            failures.append(f"fit {position}: atomic completion receipt missing")
        if receipt.get("return_fields_opened") is not False:
            failures.append(f"fit {position}: return blindness flag missing")
        if receipt.get("method") != METHOD or receipt.get("candidate_count") != 8:
            failures.append(f"fit {position}: method/candidate count mismatch")
        parameter_hashes = receipt.get("parameter_hashes", [])
        cache_keys = receipt.get("fit_cache_keys", [])
        if len(parameter_hashes) != 8 or len(set(parameter_hashes)) != 8:
            failures.append(f"fit {position}: candidate parameters are missing/duplicated")
        if len(cache_keys) != 8 or len(set(cache_keys)) != 8:
            failures.append(f"fit {position}: cache keys are missing/duplicated")
        for candidate_index, (key, parameter_hash) in enumerate(
            zip(cache_keys, parameter_hashes)
        ):
            metadata_path = args.run_root / "fit_cache" / f"{key}.json"
            arrays_path = args.run_root / "fit_cache" / f"{key}.npz"
            if not metadata_path.is_file() or not arrays_path.is_file():
                failures.append(f"fit {position}: missing cache object {key}")
                continue
            metadata = guarded_json(metadata_path)
            model, fit = metadata.get("model", {}), metadata.get("fit", {})
            if metadata.get("cache_key") != key or model.get("form") != "ricker":
                failures.append(f"fit {position}: cache identity/form mismatch {key}")
            if model.get("parameter_hash") != parameter_hash:
                failures.append(f"fit {position}: parameter hash mismatch {key}")
            if model.get("candidate_id") != f"candidate_00_{candidate_index:02d}":
                failures.append(f"fit {position}: candidate order mismatch {key}")
            if fit.get("transition_data_hash") != receipt.get("transition_data_hash"):
                failures.append(f"fit {position}: transition hash mismatch {key}")
            if not math.isfinite(float(fit.get("objective", float("nan")))):
                failures.append(f"fit {position}: non-finite selected objective {key}")
            fit_ids = set(fit.get("fit_episode_ids", []))
            holdout_ids = set(fit.get("holdout_episode_ids", []))
            if not fit_ids or fit_ids & holdout_ids:
                failures.append(f"fit {position}: invalid episode split {key}")
            if metadata.get("array_hash") != sha256(arrays_path):
                failures.append(f"fit {position}: cache array hash mismatch {key}")
        completed_fit += 1
    for position, row in enumerate(plan_rows):
        root = args.run_root / row["evaluation_artifact_dir"]
        completion_path = (
            args.run_root / "completion_receipts" / row["fit_cell"]
            / row["method"] / "plan_completion.json"
        )
        if not completion_path.is_file():
            continue
        completion = guarded_json(completion_path)
        if (
            completion.get("receipt_schema") != "ricker_only_plan_completion_v1"
            or completion.get("completion_status") != "complete"
            or completion.get("receipt_write") != "atomic_replace"
            or completion.get("return_fields_opened") is not False
            or completion.get("manifest_index") != int(row["index"])
            or completion.get("method") != row["method"]
            or completion.get("fit_cell") != row["fit_cell"]
            or completion.get("artifact_dir") != row["evaluation_artifact_dir"]
        ):
            failures.append(f"plan {position}: invalid atomic completion receipt")
            continue
        recorded_hashes = completion.get("artifact_hashes")
        if not isinstance(recorded_hashes, dict) or not recorded_hashes:
            failures.append(f"plan {position}: completion hashes missing")
            continue
        if set(recorded_hashes) != expected_plan_artifact_names():
            failures.append(f"plan {position}: completion artifact set is not exact")
            continue
        unsafe_names = [
            name for name in recorded_hashes
            if Path(name).name != name or name in FORBIDDEN_FILES or ".tmp" in name
        ]
        if unsafe_names:
            failures.append(f"plan {position}: unsafe completion paths {unsafe_names}")
            continue
        hash_mismatches = [
            name for name, expected in recorded_hashes.items()
            if not (root / name).is_file() or sha256(root / name) != expected
        ]
        if hash_mismatches:
            failures.append(
                f"plan {position}: missing/changed completed artifacts {hash_mismatches}"
            )
            continue
        required = (
            root / "candidate_bank.json",
            root / "faithful_fit.json",
            root / "privacy_audit.json",
            root / "planner_provenance.json",
        )
        if not all(path.is_file() for path in required):
            continue
        bank, fit_artifact, privacy, planner = (guarded_json(path) for path in required)
        if (
            bank.get("candidate_count") != 8
            or bank.get("candidate_construction") != CONSTRUCTION
            or bank.get("prior_type") != "uniform"
        ):
            failures.append(f"plan {position}: candidate-bank identity mismatch")
        hashes = bank.get("parameter_hashes", [])
        if len(hashes) != 8 or len(set(hashes)) != 8:
            failures.append(f"plan {position}: duplicate/missing candidate hashes")
        if privacy.get("status") != "passed" or privacy.get("forbidden_name_hits"):
            failures.append(f"plan {position}: privacy audit failed")
        planners = planner.get("planners", [])
        if len(planners) != 8:
            failures.append(f"plan {position}: expected eight independent planners")
        if (
            fit_artifact.get("candidate_count") != 8
            or fit_artifact.get("candidate_construction") != CONSTRUCTION
            or fit_artifact.get("method_impl_version")
            != "plus_adapted_ricker_only_pbvi_v1"
        ):
            failures.append(f"plan {position}: fitted-artifact routing mismatch")
        for candidate_index in range(8):
            candidate = guarded_json(root / f"candidate_{candidate_index:03d}.json")
            if candidate.get("form") != "ricker":
                failures.append(f"plan {position}: non-Ricker candidate {candidate_index}")
        completed_plan += 1
    acct = accounting([*args.fit_job_id, *args.plan_job_id])
    states = acct["states"]
    bad_states = {
        key: value
        for key, value in states.items()
        if key not in {"COMPLETED"} and value
    }
    if bad_states:
        warnings.append(f"non-completed Slurm states: {bad_states}")
    if failures:
        decision = "FAIL_STRUCTURAL_ACCEPTANCE"
    elif (
        completed_fit < args.expected_rows
        or completed_plan < args.expected_rows
        or bad_states
        or partial_files
    ):
        decision = "INCOMPLETE"
    elif warnings:
        decision = "PASS_WITH_WARNING_STRUCTURAL_ACCEPTANCE"
    else:
        decision = "PASS_LIMITED_STRUCTURAL_ACCEPTANCE"
    return {
        "acceptance_version": "acceptance_ricker_only_plus_structural_v1",
        "decision": decision,
        "fit_manifest_sha256": sha256(args.fit_manifest),
        "plan_manifest_sha256": sha256(args.plan_manifest),
        "completed_fit_rows": completed_fit,
        "completed_plan_rows": completed_plan,
        "failures": failures,
        "warnings": warnings,
        "temporary_or_partial_files": partial_files,
        "not_evaluable": [
            "performance returns (sealed)",
            "run-specific posterior history",
            "run-specific PBVI deterministic repeatability",
            "per-start optimizer termination and bound-aware gradients",
        ],
        "slurm": acct,
        "return_fields_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--fit-manifest", required=True, type=Path)
    parser.add_argument("--plan-manifest", required=True, type=Path)
    parser.add_argument("--fit-job-id", action="append", default=[])
    parser.add_argument("--plan-job-id", action="append", default=[])
    parser.add_argument("--expected-rows", type=int, default=72)
    args = parser.parse_args()
    payload = evaluate(args)
    target = args.run_root / "acceptance.json"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
