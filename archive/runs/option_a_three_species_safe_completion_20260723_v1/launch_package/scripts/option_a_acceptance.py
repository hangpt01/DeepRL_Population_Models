#!/usr/bin/env python3
"""Method-specific, return-blind structural acceptance for Option A."""

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
    "plus": ("plus_adapted_ricker_only_pbvi", 2, 8),
    "moor": ("moor_adapted_ricker_misspec_pbvi", 8, 1),
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
            raise RuntimeError(f"forbidden field in {path}: {fragment}")
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
        core_seconds += record["elapsed_raw"] * record["alloc_cpus"]
        records.append(record)
    return {"records": records, "allocated_core_hours": core_seconds / 3600.0}


def artifact_root(run_root: Path, row: dict[str, str], group: str) -> Path | None:
    if group == "plus":
        target = run_root / row["evaluation_artifact_dir"]
        return target if target.is_dir() else None
    base = (
        run_root
        / "evaluation/regime_hidden/crab_eating_fox"
        / row["environment"]
        / f"sigma_{float(row['sigma_obs']):g}".replace(".", "p")
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
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--worker-job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    method, expected_rows, candidate_count = METHODS[args.method]
    fit_rows, plan_rows = rows(args.fit_manifest), rows(args.plan_manifest)
    failures, row_results = [], []
    if len(fit_rows) != expected_rows or len(plan_rows) != expected_rows:
        failures.append("manifest row count mismatch")
    for position, (fit, plan) in enumerate(zip(fit_rows, plan_rows)):
        result = {"position": position, "checks": []}

        def check(name: str, passed: bool, evidence: Any) -> None:
            result["checks"].append(
                {"name": name, "passed": bool(passed), "evidence": evidence}
            )
            if not passed:
                failures.append(f"row {position}: {name}")

        check(
            "paired row identity",
            fit["index"] == plan["index"] == str(position)
            and fit["method"] == plan["method"] == method
            and fit["population"] == plan["population"] == "Crab-eating fox"
            and fit["reward_mode"] == plan["reward_mode"] == "safe"
            and fit["authoritative_dataset_sha256"]
            == plan["authoritative_dataset_sha256"],
            {
                "family": fit["environment"],
                "sigma_obs": fit["sigma_obs"],
                "dataset_sha256": fit["authoritative_dataset_sha256"],
            },
        )
        public = (
            args.run_root
            / "datasets"
            / fit["fit_cell"]
            / "public.npz"
        )
        private = (
            args.run_root
            / "private"
            / fit["fit_cell"]
            / "truth.npz"
        )
        check(
            "authoritative public/private file hashes",
            public.is_file()
            and private.is_file()
            and sha256(public) == fit["authoritative_public_file_sha256"]
            and sha256(private) == fit["authoritative_private_file_sha256"],
            {"public": str(public), "private": str(private)},
        )
        receipt_path = args.run_root / fit["fit_receipt_path"]
        if not receipt_path.is_file():
            check("fit receipt exists", False, str(receipt_path))
            row_results.append(result)
            continue
        receipt = guarded_json(receipt_path)
        check(
            "return-blind fit receipt",
            receipt.get("run_stage") == "dynamics_fit"
            and receipt.get("return_fields_opened") is False
            and receipt.get("method") == method
            and receipt.get("candidate_count") == candidate_count
            and len(receipt.get("fit_cache_keys", [])) == candidate_count
            and len(receipt.get("parameter_hashes", [])) == candidate_count,
            {"cache_statuses": receipt.get("cache_statuses")},
        )
        root = artifact_root(args.run_root, plan, args.method)
        if root is None:
            check("unique artifact root exists", False, None)
            row_results.append(result)
            continue
        completion = (
            args.run_root
            / "completion_receipts"
            / plan["fit_cell"]
            / method
            / "plan_completion.json"
        )
        if not completion.is_file():
            check("atomic plan completion exists", False, str(completion))
            row_results.append(result)
            continue
        complete = guarded_json(completion)
        recorded = complete.get("artifact_hashes", {})
        check(
            "atomic return-blind plan completion",
            complete.get("completion_status") == "complete"
            and complete.get("receipt_write") == "atomic_replace"
            and complete.get("return_fields_opened") is False
            and complete.get("manifest_index") == position
            and complete.get("method") == method
            and bool(recorded),
            complete.get("receipt_schema"),
        )
        mismatches = [
            name
            for name, digest in recorded.items()
            if name in FORBIDDEN_NAMES
            or not (root / name).is_file()
            or sha256(root / name) != digest
        ]
        check("completed artifact hashes", not mismatches, mismatches)
        privacy = guarded_json(root / "privacy_audit.json")
        fit_artifact = guarded_json(root / "faithful_fit.json")
        planners = guarded_json(root / "planner_provenance.json")
        fits = fit_artifact.get("fits", [])
        check(
            "privacy, fit, and planner structure",
            privacy.get("status") == "passed"
            and not privacy.get("forbidden_name_hits")
            and len(fits) == candidate_count
            and len(planners.get("planners", [])) == candidate_count,
            {
                "privacy": privacy.get("status"),
                "fits": len(fits),
                "planners": len(planners.get("planners", [])),
            },
        )
        row_results.append(result)
    slurm = accounting(args.worker_job_id)
    bad = [
        record
        for record in slurm["records"]
        if record["state"] != "COMPLETED" or record["exit_code"] != "0:0"
    ]
    if len(slurm["records"]) != expected_rows:
        failures.append("Slurm task count mismatch")
    if bad:
        failures.append(f"non-successful Slurm tasks: {bad}")
    decision = (
        "PASS_LIMITED_STRUCTURAL_ACCEPTANCE"
        if not failures
        else "FAIL_STRUCTURAL_ACCEPTANCE"
    )
    payload = {
        "acceptance_version": "option_a_method_specific_return_blind_v1",
        "method": method,
        "decision": decision,
        "fit_manifest_sha256": sha256(args.fit_manifest),
        "plan_manifest_sha256": sha256(args.plan_manifest),
        "completed_rows": sum(
            all(check["passed"] for check in row["checks"]) for row in row_results
        ),
        "expected_rows": expected_rows,
        "failures": failures,
        "rows": row_results,
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
