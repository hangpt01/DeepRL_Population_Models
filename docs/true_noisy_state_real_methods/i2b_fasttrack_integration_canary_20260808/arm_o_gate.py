#!/usr/bin/env python3
"""Fail-closed server-side Arm O gate for the conditional Arm T chain."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
FAST_OUT = ROOT / "outputs/i2b_fasttrack_integration_canary_20260808"
ARM_T_OUT = ROOT / "outputs/i2b_stageb_arm_t_sigma02_20260809"
RECEIPT = ARM_T_OUT / "gate/ARM_O_GATE_RECEIPT.json"
TEST_LOG = ARM_T_OUT / "gate/FASTTRACK_TESTS.log"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_only_hash(track: str) -> tuple[int, str]:
    files = sorted(
        path for path in (ROOT / f"src/tracks/{track}").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    # Reproduce `sha256sum` lines using repository-relative names and two spaces.
    lines = b"".join(
        f"{sha(path)}  {path.relative_to(ROOT)}\n".encode("utf-8") for path in files
    )
    return len(files), hashlib.sha256(lines).hexdigest()


def final_arm_o_receipt(task: dict) -> Path:
    if task["method"] in {
        "plus_adapted_ricker_only_pbvi", "moor_adapted_ricker_misspec_pbvi"
    }:
        return FAST_OUT / "ecological_retry1/arm_o_tasks" / task["task_id"] / "task_receipt.json"
    return FAST_OUT / "arm_o_tasks" / task["task_id"] / "task_receipt.json"


def main() -> None:
    started = time.time()
    require(not RECEIPT.exists(), "gate receipt collision")
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []

    def checked(name: str, condition: bool, evidence: object) -> None:
        require(condition, f"gate failed: {name}: {evidence}")
        checks.append({"name": name, "result": "PASS", "evidence": evidence})

    arm_o = json.loads((DOC / "ARM_O_CANARY_REGISTRATION.json").read_text())
    arm_t = json.loads((DOC / "ARM_T_TASKS.json").read_text())
    checked("arm_o_task_count", len(arm_o["tasks"]) == 12, 12)
    checked("arm_t_task_count", len(arm_t["tasks"]) == 12, 12)
    checked("arm_t_method_spec_count", len(arm_t["method_specs"]) == 6, 6)

    arm_o_receipts = []
    for task in arm_o["tasks"]:
        path = final_arm_o_receipt(task)
        require(path.exists(), f"missing Arm O receipt {path}")
        receipt = json.loads(path.read_text())
        require(receipt.get("exit_status") == 0, f"nonzero Arm O receipt {path}")
        require((receipt.get("parity") or {}).get("result") == "PASS", f"Arm O parity failed {path}")
        require(receipt.get("task") == task, f"Arm O task identity changed {path}")
        require(receipt.get("accepted_episodes_sha256_observed") == task["accepted_episodes_sha256"], f"accepted identity changed {path}")
        require(receipt.get("config_sha256") == arm_o["config_identities"][task["config_path"]], f"config identity changed {path}")
        require(receipt.get("derived_artifact_access") is False, f"Arm O derived access {path}")
        require(receipt.get("original_truth_archive_access") is False, f"Arm O truth access {path}")
        arm_o_receipts.append(str(path))
    checked("all_twelve_arm_o_receipts_and_parity", len(arm_o_receipts) == 12, arm_o_receipts)

    initial_failures = []
    for task in arm_o["tasks"]:
        if task["method"] not in {"plus_adapted_ricker_only_pbvi", "moor_adapted_ricker_misspec_pbvi"}:
            continue
        path = FAST_OUT / "arm_o_tasks" / task["task_id"] / "task_receipt.json"
        require(path.exists(), f"missing preserved wrong-interpreter receipt {path}")
        failure = json.loads(path.read_text())
        require(failure.get("exit_status") != 0 and "PyTorch" in failure.get("failure", ""), f"wrong-interpreter provenance mismatch {path}")
        initial_failures.append(str(path))
    checked("wrong_interpreter_failure_preserved", len(initial_failures) == 4, initial_failures)

    expected_docs = {
        "DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN_REV3_1.md": "701b4509f1dc04b7885891be72e526420ac5558be56bdde2b9628789398fe959",
        "CLAUDE_REV3_1_CONFIRMATION_AUDIT.md": "0f448db89109186aa4bbd64bc000da45ae662386dea79e658bcd802f4880cd17",
    }
    for name, expected in expected_docs.items():
        path = DOC.parent / name
        checked(f"controlling_hash:{name}", sha(path) == expected, expected)
    audit = DOC.parent / "i2_increment_a_exact_state_adapters_20260808/CLAUDE_I2A_REV1_CORRECTIVE_CODE_AUDIT.md"
    checked("i2a_corrective_audit", sha(audit) == "e62682d3756a39215578c77317b995f88416c9cbea2b581b6f69d4f665541a87", sha(audit))
    for track, count, expected in (
        ("ecological", 52, "2b3b8ae6d2f8ff5ffb17c4885ded9e8f1f6b3c0cb662f393186fe4b4706a884e"),
        ("general", 55, "f90cea6f28dcacb910b5e036bf9e09958715d00a2418fd0856a3d5a12856bdbd"),
    ):
        observed_count, observed = source_only_hash(track)
        checked(f"source_only:{track}", (observed_count, observed) == (count, expected), {"count": observed_count, "sha256": observed})

    truth_receipt = json.loads((DOC / "TRUTH_EXTRACTION_RECEIPT.json").read_text())
    checked("truth_extraction_overall", truth_receipt.get("overall_result") == "PASS", truth_receipt.get("overall_result"))
    checked("truth_forbidden_reads", all(not cell["forbidden_truth_payload_fields_materialized"] and cell["forbidden_truth_payload_bytes_read"] == 0 for cell in truth_receipt["cells"]), 0)
    for cell, inputs in arm_t["cell_inputs"].items():
        derived = Path(inputs["derived_offline_path"])
        require(derived.exists() and sha(derived) == inputs["derived_offline_sha256"], f"derived hash mismatch {cell}")
        with np.load(derived, allow_pickle=False) as data:
            require(tuple(data.files) == ("states", "next_states", "row_index", "episode_id", "timestep"), f"derived schema mismatch {cell}")
            require(len(data["states"]) == 4000 and np.isfinite(data["states"]).all() and np.isfinite(data["next_states"]).all(), f"derived content mismatch {cell}")
        checks.append({"name": f"derived:{cell}", "result": "PASS", "evidence": inputs["derived_offline_sha256"]})

    expected_methods = [task["method"] for task in arm_o["tasks"]]
    checked("task_order_and_methods", [task["method"] for task in arm_t["tasks"]] == expected_methods, expected_methods)
    for method, spec in arm_t["method_specs"].items():
        required = {"training_route", "runtime_route", "learned_artifact_change", "interpreter", "time_limit"}
        require(required <= set(spec), f"incomplete Arm T specification {method}")
        require(spec["time_limit"] == "12:00:00", f"unregistered time limit {method}")
    checked("all_six_arm_t_specs_complete", True, sorted(arm_t["method_specs"]))
    checked("arm_t_output_namespace", Path(arm_t["output_namespace"]).resolve() == ARM_T_OUT.resolve(), str(ARM_T_OUT))

    env = os.environ.copy()
    env.update({"LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1"})
    logs = []
    with tempfile.TemporaryDirectory(prefix="i2b_gate_i2a_", dir="/tmp") as tmp_i2a:
        test_env = {**env, "I2A_TEST_TMPDIR": tmp_i2a, "PYTHONPATH": str(DOC.parent / "i2_increment_a_exact_state_adapters_20260808")}
        command = [sys.executable, "-B", "-m", "unittest", "-v", str(DOC.parent / "i2_increment_a_exact_state_adapters_20260808/test_i2a_contracts.py")]
        result = subprocess.run(command, cwd=ROOT, env=test_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        logs.append(result.stdout)
        checked("i2a_38_tests", result.returncode == 0 and "Ran 38 tests" in result.stdout, {"returncode": result.returncode, "count": 38})
    with tempfile.TemporaryDirectory(prefix="i2b_gate_fasttrack_", dir="/tmp") as tmp_i2b:
        test_env = {**env, "I2B_TEST_TMPDIR": tmp_i2b, "PYTHONPATH": str(DOC)}
        command = [sys.executable, "-B", "-m", "unittest", "-v", str(DOC / "test_fasttrack_integration.py")]
        result = subprocess.run(command, cwd=ROOT, env=test_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        logs.append(result.stdout)
        checked("i2b_15_tests", result.returncode == 0 and "Ran 15 tests" in result.stdout, {"returncode": result.returncode, "count": 15})
    TEST_LOG.write_text("\n\n".join(logs), encoding="utf-8")

    receipt = {
        "schema_version": "i2b_arm_o_gate_receipt_v1",
        "result": "PASS",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "dependency": os.environ.get("SLURM_JOB_DEPENDENCY", "afterany:58859482"),
        "started_unix": started,
        "ended_unix": time.time(),
        "checks": checks,
        "test_log": str(TEST_LOG),
        "test_log_sha256": sha(TEST_LOG),
        "arm_t_tasks_authorized": 12,
        "forbidden_truth_payload_reads": 0,
        "repairs_or_retries_performed": 0,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
