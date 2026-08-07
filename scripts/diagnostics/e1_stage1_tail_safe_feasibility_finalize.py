#!/usr/bin/env python3
"""Finalize only the registered numerical feasibility evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from typing import Any


REPO = Path("/fs04/scratch2/ce25/DeepRL_Population_Models")
OUT = REPO / ".verification/e1_stage1_tail_safe_feasibility_20260806_v1"
EXPECTED_TASKS = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
    tmp.replace(path)


def main() -> None:
    start = time.perf_counter()
    execution = read(OUT / "EXECUTION_SEAL.json")
    execution_sha = sha256(OUT / "EXECUTION_SEAL.json")
    expected = (OUT / "EXECUTION_SEAL.json.sha256").read_text().split()[0]
    if execution_sha != expected:
        raise AssertionError("execution seal mismatch")
    results = []
    for task in range(EXPECTED_TASKS):
        path = OUT / "tasks" / f"task_{task:02d}" / "CASE_RESULT.json"
        if not path.exists():
            raise AssertionError(f"missing registered task {task}")
        value = read(path)
        if value["task_id"] != task:
            raise AssertionError("task identity mismatch")
        results.append(value)
    comparisons = {
        "schema": "e1_stage1_tail_safe_feasibility_comparisons_v1",
        "label": "NUMERICAL METHOD DEVELOPMENT AFTER FAILED STAGE 1 — NOT SCIENTIFIC RESULTS",
        "cases": [
            {
                "task_id": r["task_id"], "cell": r["cell"], "episode_id": r["episode_id"],
                "timestep": r["timestep"], "source_row_index": r["source_row_index"],
                "status": r["status"], "errors_vs_80_digit_reference": r["errors_vs_80_digit_reference"],
                "reference_60_vs_80_errors": r["reference_60_vs_80_errors"],
                "target_independence": r["target_independence"],
                "mixture_reduction": r["mixture_reduction"], "criteria": r["criteria"]
            } for r in results
        ]
    }
    resources = {
        "schema": "e1_stage1_tail_safe_feasibility_resources_v1",
        "per_case": [{"task_id": r["task_id"], **r["resources"], "reference_60": r["reference"]["60"]["resource"], "reference_80": r["reference"]["80"]["resource"]} for r in results],
        "measurement_policy": "No scientific or holdout metric aggregation; resource measurements only."
    }
    assertions = {
        "schema": "e1_stage1_tail_safe_feasibility_assertions_v1",
        "per_case": [{"task_id": r["task_id"], "score_order_invariant": r["target_independence"]["score_order_invariant"], "synthetic_target_invariant": r["target_independence"]["synthetic_target_invariant"], "subsequent_filter_invariant": r["target_independence"]["subsequent_filter_invariant"]} for r in results]
    }
    all_pass = all(r["status"] == "PASS" for r in results)
    write_new(OUT / "PER_CASE_NUMERICAL_COMPARISONS.json", comparisons)
    write_new(OUT / "RESOURCE_MEASUREMENTS.json", resources)
    write_new(OUT / "TARGET_INDEPENDENCE_ASSERTIONS.json", assertions)
    receipt = {
        "schema": "e1_stage1_tail_safe_feasibility_receipt_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label": "NUMERICAL METHOD DEVELOPMENT AFTER FAILED STAGE 1 — NOT SCIENTIFIC RESULTS",
        "status": "FEASIBILITY_PASS_READY_FOR_INDEPENDENT_AUDIT" if all_pass else "FEASIBILITY_FAIL_DESIGN_STILL_BLOCKED",
        "registered_tasks": EXPECTED_TASKS, "completed_case_files": len(results),
        "all_provisional_criteria_pass": all_pass, "replacement_stage1_authorized": False,
        "stage2_authorized": False, "stage3_authorized": False,
        "execution_seal_sha256": execution_sha,
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", "NOT AVAILABLE"),
        "finalizer_job_id": os.environ.get("SLURM_JOB_ID", "NOT AVAILABLE"),
        "finalizer_wall_seconds": time.perf_counter() - start,
        "finalizer_peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    }
    write_new(OUT / "BENCHMARK_RECEIPT.json", receipt)
    prompt = """# Independent Claude read-only audit prompt\n\nAudit `.verification/e1_stage1_tail_safe_feasibility_20260806_v1/` read-only. Treat the failed Stage 1 namespace as immutable and rejected. Verify seals and hashes; reproduce the recovered sigma=0.2 failure identity; inspect the exact zero-atom handling; reproduce at least one step-0, both known sigma=0.1 failed rows, the sigma=0.2 failed row, one later-step worst row, and both boundary directions. Independently verify that production filtering is target-independent, that reference histories consume no production posterior, and that target-aware adaptation occurs only after posterior freezing. Check maximizers, endpoint margins, omitted-tail estimates, 60/80-digit agreement, all direct errors, cap flags, component dominance, and resource measurements. Reject PASS unless every registered case meets every provisional criterion without a cap. Confirm that no Stage 1 aggregates or scientific conclusions were produced.\n"""
    audit = OUT / "CLAUDE_READ_ONLY_AUDIT_PROMPT.md"
    if audit.exists(): raise FileExistsError(audit)
    audit.write_text(prompt, encoding="utf-8")
    excluded = {"MANIFEST.json", "MANIFEST.json.sha256"}
    manifest = {str(p.relative_to(OUT)): sha256(p) for p in sorted(OUT.rglob("*")) if p.is_file() and str(p.relative_to(OUT)) not in excluded}
    write_new(OUT / "MANIFEST.json", {"schema": "e1_stage1_tail_safe_feasibility_manifest_v1", "files": manifest})
    (OUT / "MANIFEST.json.sha256").write_text(f"{sha256(OUT / 'MANIFEST.json')}  MANIFEST.json\n", encoding="utf-8")


if __name__ == "__main__":
    main()
