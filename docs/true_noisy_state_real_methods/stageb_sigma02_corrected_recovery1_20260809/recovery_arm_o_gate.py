#!/usr/bin/env python3
"""Fail-closed parity/artifact gate between corrected Arm O and Arm T."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from recovery_common import ECO, sha, source_only_hash, strict_json, verify_hash_manifest

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
OUT = ROOT / "outputs/stageb_sigma02_corrected_recovery1_20260809"
RECEIPT = OUT / "gate/CORRECTED_ARM_O_GATE_RECEIPT.json"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> None:
    started = time.time()
    require(not RECEIPT.exists(), "gate receipt collision")
    registration = json.loads((DOC / "CORRECTED_STAGEB_REGISTRATION.json").read_text())
    manifest = json.loads((DOC / "CORRECTED_TASK_MANIFEST.json").read_text())
    checks: list[dict[str, Any]] = []

    def check(name: str, value: bool, evidence: Any) -> None:
        require(value, f"{name}: {evidence}")
        checks.append({"name": name, "result": "PASS", "evidence": evidence})

    check("pre_execution_hashes", verify_hash_manifest(ROOT, DOC / "PRE_EXECUTION_HASHES.sha256") >= 12,
          "all listed identities match")
    check("task_grid", len(manifest["tasks"]) == 12 and
          len({(x["cell"], x["method"]) for x in manifest["tasks"]}) == 12, "2 cells x 6 methods")
    for track in ("ecological", "general"):
        observed = source_only_hash(ROOT, track)
        check(f"frozen_source_{track}", observed == registration["frozen_source_only_hashes"][track], observed)
    for cell, spec in registration["matched_surrogates"].items():
        path = ROOT / spec["path"]
        check(f"matched_surrogate_{cell}", path.is_file() and sha(path) == spec["sha256"], spec)
    receipts = []
    for task in manifest["tasks"]:
        root = OUT / "arm_o_tasks" / task["task_id"]
        path = root / "task_receipt.json"
        require(path.is_file(), f"missing Arm O receipt {task['task_id']}")
        receipt = json.loads(path.read_text())
        require(receipt.get("exit_status") == 0, f"Arm O task nonzero {task['task_id']}")
        require((receipt.get("accepted_arm_o_parity") or {}).get("result") == "PASS",
                f"accepted parity failed {task['task_id']}")
        require(receipt.get("failed_partial_namespace_access") is False,
                f"partial namespace accessed {task['task_id']}")
        require(receipt.get("original_truth_archive_access") is False and
                receipt.get("forbidden_truth_fields_accessed") == [],
                f"truth boundary failed {task['task_id']}")
        artifact = receipt["fitted_artifact"]
        require(sha(Path(artifact["pickle_path"])) == artifact["pickle_sha256"],
                f"policy hash mismatch {task['task_id']}")
        require(sha(Path(artifact["parameter_path"])) == artifact["parameter_sha256"],
                f"parameter hash mismatch {task['task_id']}")
        areceipt = json.loads(Path(artifact["receipt_path"]).read_text())
        require(areceipt["reload_test"] == "PASS" and
                areceipt["prediction_action_parity"] == "PASS" and
                all(areceipt["required_checks"].values()),
                f"fitted artifact incomplete {task['task_id']}")
        component = json.loads(Path(receipt["pre_return_component_receipt"]).read_text())
        require(component["written_before_evaluation"] and not component["arm_t_surrogate_fit_executed"],
                f"pre-return/surrogate contract {task['task_id']}")
        require(component["forbidden_method_fields_accessed"] == [],
                f"method boundary {task['task_id']}")
        if task["method"] in {"refplan", "ogsrl", "bamcts"}:
            require(component["residual_sigma"]["applicable"],
                    f"residual diagnostics absent {task['task_id']}")
        if task["method"] == "refplan":
            dispersion = component["refplan_predictive_dispersion"]
            require(dispersion and sha(Path(dispersion["path"])) == dispersion["sha256"],
                    f"RefPlan dispersion absent {task['task_id']}")
        evidence = json.loads(Path(receipt["timestep_evidence"]).read_text())
        require(evidence["evaluator_only_not_method_visible"] and
                evidence["validation"]["episodes"] == 20 and
                evidence["validation"]["maximum_reward_reconstruction_error"] <= 1e-12,
                f"time evidence invalid {task['task_id']}")
        policy = json.loads(Path(receipt["policy_evidence"]).read_text())
        require(len(policy["episodes"]) == 20, f"policy evidence incomplete {task['task_id']}")
        if task["method"] == "bamcts":
            require(all(ep["initial_posterior"] is not None and
                        all(step.get("posterior_after_observe") is not None for step in ep["steps"])
                        for ep in policy["episodes"]),
                    f"BA posterior trajectory incomplete {task['task_id']}")
        if task["method"] in ECO:
            require(component["shared_ecological_policy"],
                    f"ecological shared-policy contract absent {task['task_id']}")
        require(receipt["matched_surrogate_sha256"] ==
                registration["matched_surrogates"][task["cell"]]["sha256"],
                f"matched surrogate mismatch {task['task_id']}")
        receipts.append(str(path))
    check("twelve_arm_o_rows", len(receipts) == 12, receipts)
    preflight = json.loads((DOC / "PREFLIGHT_TEST_REPORT.json").read_text())
    check("i2a_38", preflight["i2a"] == {"count": 38, "result": "PASS"}, preflight["i2a"])
    check("fasttrack_15", preflight["fasttrack"] == {"count": 15, "result": "PASS"}, preflight["fasttrack"])
    check("existing_total_53", preflight["existing_total"] == 53, preflight["existing_total"])
    check("recovery_contract_tests", preflight["recovery"]["result"] == "PASS", preflight["recovery"])
    check("six_arm_t_specs", len(manifest["method_specs"]) == 6 and
          all({"interpreter", "time_limit", "training_route", "runtime_route", "artifact_contract"}
              <= set(value) for value in manifest["method_specs"].values()),
          sorted(manifest["method_specs"]))
    check("pinned_cpu_config", registration["execution_profile"]["cpu"] ==
          "Intel Xeon Platinum 8452Y", registration["execution_profile"])
    strict_json(RECEIPT, {"schema_version": "corrected_recovery_arm_o_gate_v1",
        "result": "PASS", "started_unix": started, "ended_unix": time.time(),
        "job_id": os.getenv("SLURM_JOB_ID"), "dependency": os.getenv("SLURM_JOB_DEPENDENCY"),
        "checks": checks, "arm_t_tasks_released": 12,
        "repair_retry_rebaseline_or_tuning": False})


if __name__ == "__main__":
    main()
