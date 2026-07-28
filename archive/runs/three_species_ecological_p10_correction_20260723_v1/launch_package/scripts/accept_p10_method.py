#!/usr/bin/env python3
"""Return-blind method-specific acceptance for the P=10 correction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import yaml


FORBIDDEN_NAMES = {
    "summary.json",
    "episodes.csv",
    "comparative_summary.json",
    "ranking.json",
}
FORBIDDEN_FRAGMENTS = (
    "operational_return",
    "true_return",
    "survival_return",
    "ranking",
)
METHODS = {
    "plus": ("plus_adapted_ricker_only_pbvi", 8),
    "moor": ("moor_adapted_ricker_misspec_pbvi", 1),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def guarded_json(path: Path) -> dict[str, Any]:
    if path.name in FORBIDDEN_NAMES:
        raise RuntimeError(f"acceptance cannot open {path.name}")
    raw = path.read_text(encoding="utf-8")
    for fragment in FORBIDDEN_FRAGMENTS:
        if re.search(rf'"[^"]*{re.escape(fragment)}[^"]*"\s*:', raw, re.I):
            raise RuntimeError(f"forbidden return field in {path}: {fragment}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def accounting(job_id: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "--array",
            "-X",
            "-n",
            "-P",
            "--format=JobID,State,ElapsedRaw,AllocCPUS,ExitCode",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    records, core_seconds = [], 0
    for line in completed.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 5 or "_" not in fields[0]:
            continue
        record = {
            "job_id": fields[0],
            "state": fields[1].split()[0],
            "elapsed_raw": int(fields[2] or 0),
            "alloc_cpus": int(fields[3] or 0),
            "exit_code": fields[4],
        }
        records.append(record)
        core_seconds += record["elapsed_raw"] * record["alloc_cpus"]
    return {"records": records, "allocated_core_hours": core_seconds / 3600.0}


def artifact_root(run_root: Path, row: dict[str, str], group: str) -> Path | None:
    if group == "plus":
        target = run_root / row["evaluation_artifact_dir"]
        return target if target.is_dir() else None
    base = (
        run_root
        / "evaluation/regime_hidden"
        / row["population"].lower().replace("-", "_").replace(" ", "_")
        / row["environment"]
        / f"sigma_{float(row['sigma_obs']):g}".replace(".", "p")
        / row["config_tag"]
    )
    matches = list(
        base.glob(
            f"**/{row['method']}/faithful_internal/faithful_artifacts"
        )
    )
    return matches[0] if len(matches) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--worker-job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    method, candidate_count = METHODS[args.method]
    manifest_rows = rows(args.manifest)
    ledger_rows = rows(args.ledger)
    failures: list[str] = []
    incomplete: list[str] = []
    results = []
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (
        config["environment"]["reward_mode"] != "safe"
        or float(config["environment"]["collapse_penalty"]) != 10.0
        or int(config["evaluation"]["horizon"]) != 50
        or float(config["evaluation"]["discount"]) != 0.95
        or config["evaluation"]["seeds"] != [7001, 7051, 7101, 7151, 7201]
        or int(config["evaluation"]["episodes_per_seed"]) != 4
    ):
        failures.append("P=10 evaluation configuration mismatch")
    if len(manifest_rows) != 24:
        failures.append("manifest row count mismatch")
    for position, row in enumerate(manifest_rows):
        result = {"position": position, "checks": []}

        def check(name: str, passed: bool, evidence: Any, missing: bool = False) -> None:
            result["checks"].append(
                {"name": name, "passed": bool(passed), "evidence": evidence}
            )
            if not passed:
                (incomplete if missing else failures).append(f"row {position}: {name}")

        check(
            "manifest P=10 identity",
            row.get("index") == str(position)
            and row.get("method") == method
            and row.get("run_stage") == "plan_evaluate"
            and row.get("reward_mode") == "safe"
            and row.get("collapse_penalty") == "10.0"
            and row.get("evaluation_horizon") == "50"
            and row.get("evaluation_discount") == "0.95"
            and row.get("evaluation_episodes") == "20"
            and row.get("episodes_per_evaluation_seed") == "4"
            and row.get("new_candidate_fits") == "0"
            and row.get("config_sha256") == sha256(args.config),
            {
                "population": row.get("population"),
                "family": row.get("environment"),
                "sigma_obs": row.get("sigma_obs"),
                "dataset_sha256": row.get("authoritative_dataset_sha256"),
            },
        )
        public = args.run_root / "datasets" / row["fit_cell"] / "public.npz"
        private = args.run_root / "private" / row["fit_cell"] / "truth.npz"
        check(
            "authoritative dataset and hidden environment hashes",
            public.is_file()
            and private.is_file()
            and sha256(public) == row["authoritative_public_file_sha256"]
            and sha256(private) == row["authoritative_private_file_sha256"],
            {"public": str(public), "private": str(private)},
        )
        fit_receipt_path = args.run_root / row["fit_receipt_path"]
        if not fit_receipt_path.is_file():
            check("verified reused-fit receipt exists", False, str(fit_receipt_path), True)
            results.append(result)
            continue
        fit_receipt = guarded_json(fit_receipt_path)
        keys = list(fit_receipt.get("fit_cache_keys") or ())
        parameters = list(fit_receipt.get("parameter_hashes") or ())
        check(
            "reward-independent fit reuse receipt",
            fit_receipt.get("run_stage") == "dynamics_fit"
            and fit_receipt.get("completion_status") == "complete"
            and fit_receipt.get("return_fields_opened") is False
            and fit_receipt.get("reward_independent_fit") is True
            and float(fit_receipt.get("source_reward_penalty")) == 5.0
            and float(fit_receipt.get("target_reward_penalty")) == 10.0
            and fit_receipt.get("method") == method
            and fit_receipt.get("candidate_count") == candidate_count
            and len(keys) == len(parameters) == candidate_count
            and fit_receipt.get("fit_reuse_ledger_sha256") == sha256(args.ledger),
            {"cache_statuses": fit_receipt.get("cache_statuses")},
        )
        matching_ledger = [
            item
            for item in ledger_rows
            if item["method"] == method
            and item["population"] == row["population"]
            and item["environment"] == row["environment"]
            and item["sigma_obs"] == row["sigma_obs"]
        ]
        check(
            "candidate-level provenance ledger coverage",
            len(matching_ledger) == candidate_count
            and [item["cache_key"] for item in matching_ledger] == keys
            and [item["parameter_hash"] for item in matching_ledger] == parameters
            and all(
                item["reuse_status"] == "verified_reward_independent_reuse"
                and item["source_reward_penalty"] == "5.0"
                and item["target_reward_penalty"] == "10.0"
                for item in matching_ledger
            ),
            {"ledger_rows": len(matching_ledger)},
        )
        cache_mismatches = []
        for key, parameter in zip(keys, parameters):
            metadata_path = args.run_root / "fit_cache" / f"{key}.json"
            array_path = args.run_root / "fit_cache" / f"{key}.npz"
            if not metadata_path.is_file() or not array_path.is_file():
                cache_mismatches.append(key)
                continue
            metadata = guarded_json(metadata_path)
            if (
                metadata.get("cache_schema") != "adapted_fit_cache_v1"
                or metadata.get("cache_key") != key
                or metadata.get("array_hash") != sha256(array_path)
                or metadata.get("model", {}).get("parameter_hash") != parameter
            ):
                cache_mismatches.append(key)
        check("reused cache hashes", not cache_mismatches, cache_mismatches)
        root = artifact_root(args.run_root, row, args.method)
        if root is None:
            check("unique P=10 artifact root exists", False, None, True)
            results.append(result)
            continue
        completion_path = (
            args.run_root
            / "completion_receipts"
            / row["fit_cell"]
            / method
            / "plan_completion.json"
        )
        if not completion_path.is_file():
            check("atomic plan completion exists", False, str(completion_path), True)
            results.append(result)
            continue
        completion = guarded_json(completion_path)
        recorded = completion.get("artifact_hashes", {})
        check(
            "atomic return-blind plan completion",
            completion.get("completion_status") == "complete"
            and completion.get("receipt_write") == "atomic_replace"
            and completion.get("return_fields_opened") is False
            and completion.get("manifest_index") == position
            and completion.get("method") == method
            and bool(recorded),
            completion.get("receipt_schema"),
        )
        artifact_mismatches = [
            name
            for name, digest in recorded.items()
            if name in FORBIDDEN_NAMES
            or not (root / name).is_file()
            or sha256(root / name) != digest
        ]
        check("completed structural artifact hashes", not artifact_mismatches, artifact_mismatches)
        privacy = guarded_json(root / "privacy_audit.json")
        fits = guarded_json(root / "faithful_fit.json").get("fits", [])
        planners = guarded_json(root / "planner_provenance.json").get("planners", [])
        check(
            "privacy, fit, and rebuilt PBVI structure",
            privacy.get("status") == "passed"
            and not privacy.get("forbidden_name_hits")
            and len(fits) == candidate_count
            and len(planners) == candidate_count
            and all(item.get("name") == "pbvi" for item in planners),
            {
                "privacy": privacy.get("status"),
                "fits": len(fits),
                "planners": len(planners),
            },
        )
        temporary = [
            str(path)
            for path in root.rglob("*")
            if path.is_file() and (path.name.startswith(".") or ".tmp" in path.name)
        ]
        check("no temporary or partial artifacts", not temporary, temporary)
        results.append(result)
    slurm = accounting(args.worker_job_id)
    bad = [
        record
        for record in slurm["records"]
        if record["state"] != "COMPLETED" or record["exit_code"] != "0:0"
    ]
    if len(slurm["records"]) != 24:
        incomplete.append("Slurm task count mismatch")
    if bad:
        incomplete.append(f"non-successful Slurm tasks: {bad}")
    if failures:
        decision = "FAIL_STRUCTURAL_ACCEPTANCE"
    elif incomplete:
        decision = "INCOMPLETE"
    else:
        decision = "PASS_LIMITED_STRUCTURAL_ACCEPTANCE"
    payload = {
        "acceptance_version": "three_species_p10_method_specific_return_blind_v1",
        "method": method,
        "decision": decision,
        "collapse_penalty": 10.0,
        "manifest_sha256": sha256(args.manifest),
        "config_sha256": sha256(args.config),
        "fit_reuse_ledger_sha256": sha256(args.ledger),
        "completed_rows": sum(
            all(item["passed"] for item in row["checks"]) for row in results
        ),
        "expected_rows": 24,
        "failures": failures,
        "incomplete_reasons": incomplete,
        "rows": results,
        "slurm": slurm,
        "return_fields_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if decision.startswith("PASS_") else 1)


if __name__ == "__main__":
    main()
