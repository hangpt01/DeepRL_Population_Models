#!/usr/bin/env python3
"""Terminal after-any finalizer for the provisional I2B Arm T array."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
OUT = ROOT / "outputs/i2b_stageb_arm_t_sigma02_20260809"
FAST = ROOT / "outputs/i2b_fasttrack_integration_canary_20260808"
LABEL = "PROVISIONAL — NOT YET INDEPENDENTLY AUDITED"
ECO = {"plus_adapted_ricker_only_pbvi", "moor_adapted_ricker_misspec_pbvi"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_write(path: Path, payload: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(payload, encoding="utf-8")


def strict_json(path: Path, value: Any) -> None:
    strict_write(path, json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def arm_o_receipt(task: dict[str, Any]) -> Path:
    if task["method"] in ECO:
        return FAST / "ecological_retry1/arm_o_tasks" / task["task_id"] / "task_receipt.json"
    return FAST / "arm_o_tasks" / task["task_id"] / "task_receipt.json"


def source_only_hash(track: str) -> dict[str, Any]:
    files = sorted(
        path for path in (ROOT / f"src/tracks/{track}").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts and path.suffix not in {".pyc", ".pyo"}
    )
    lines = b"".join(
        f"{sha(path)}  {path.relative_to(ROOT)}\n".encode("utf-8") for path in files
    )
    return {"count": len(files), "sha256": hashlib.sha256(lines).hexdigest()}


def bootstrap_interval(differences: np.ndarray) -> list[float]:
    rng = np.random.Generator(np.random.PCG64(20260808))
    indices = rng.integers(0, len(differences), size=(100_000, len(differences)))
    means = differences[indices].mean(axis=1)
    return [
        float(np.quantile(means, 0.025, method="linear")),
        float(np.quantile(means, 0.975, method="linear")),
    ]


def completed_summary(task: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    o_receipt = json.loads(arm_o_receipt(task).read_text(encoding="utf-8"))
    o_rows = rows(Path(o_receipt["episodes_path"]))
    t_rows = rows(Path(receipt["episodes_path"]))
    if len(o_rows) != 20 or len(t_rows) != 20:
        raise RuntimeError(f"paired cardinality mismatch for {task['task_id']}")
    for left, right in zip(o_rows, t_rows):
        if (left["episode"], left["seed"], left["block_seed"]) != (
            right["episode"], right["seed"], right["block_seed"]
        ):
            raise RuntimeError(f"paired identity mismatch for {task['task_id']}")
    arm_o = np.asarray([float(row["true_return"]) for row in o_rows])
    arm_t = np.asarray([float(row["true_return"]) for row in t_rows])
    differences = arm_t - arm_o
    runtime = json.loads(Path(receipt["runtime_metric_receipt"]).read_text(encoding="utf-8"))
    runtime_rows = runtime["episodes"]
    component = json.loads(
        Path(receipt["pre_return_component_receipt"]).read_text(encoding="utf-8")
    )
    first_collapse = [int(row["collapse_entry_timestep"]) for row in t_rows]
    positive_first = [value for value in first_collapse if value >= 0]
    action_counts = np.sum(
        np.asarray([row["action_counts"] for row in runtime_rows], dtype=np.int64), axis=0
    )
    unsafe_steps = int(round(sum(
        float(row["unsafe_fraction"]) * (int(row["n_steps"]) + 1) for row in t_rows
    )))
    return {
        "status": LABEL,
        "cell": task["cell"],
        "method": task["method"],
        "evd_separate_raw_logged_reward_objective": task["method"] == "ensemble_value_disagreement_pessimism",
        "arm_o_absolute_mean_true_return": float(arm_o.mean()),
        "arm_t_absolute_mean_true_return": float(arm_t.mean()),
        "paired_raw_loss_true_minus_noisy": float(differences.mean()),
        "paired_percentile_bootstrap_95_interval": bootstrap_interval(differences),
        "paired_sampling_units": 20,
        "arm_t_collapse_rate": float(np.mean([float(row["collapse_entry"]) for row in t_rows])),
        "arm_t_first_collapse_timing_mean_when_present": (
            float(np.mean(positive_first)) if positive_first else None
        ),
        "arm_t_unsafe_occupancy_steps": unsafe_steps,
        "arm_t_discounted_safety_penalty_contribution_mean": float(np.mean([
            row["discounted_safety_penalty_contribution"] for row in runtime_rows
        ])),
        "arm_t_minimum_abundance_mean": float(np.mean([
            float(row["min_true_state"]) for row in t_rows
        ])),
        "arm_t_mean_action_entropy": float(np.mean([
            float(row["action_entropy"]) for row in t_rows
        ])),
        "arm_t_action_counts": action_counts.tolist(),
        "arm_t_distinct_action_count": int(np.count_nonzero(action_counts)),
        "arm_t_duration_seconds": receipt.get("duration_seconds"),
        "accepted_arm_o_activity": task["accepted_activity"],
        "constant_policy_non_discrimination_flag": "constant action" in task["accepted_activity"],
        "component_artifact_changes": {
            "learned_artifact_change": component["learned_artifact_change"],
            "interpretation_label": component["interpretation_label"],
            "fit_artifact_identity": component["fit_artifact_identity"],
            "residual_sigma_float64_hex": component["residual_sigma_float64_hex"],
            "initial_model_posterior_entropy": component["initial_model_posterior_entropy"],
            "refplan_sd_disclosure": component["refplan_sd_disclosure"],
            "ogsrl_safety_calibration": component["ogsrl_safety_calibration"],
        },
    }


def main(arm_t_job_id: str, gate_job_id: str) -> None:
    for name in (
        "PROVISIONAL_STAGEB_SUMMARY.json", "PROVISIONAL_STAGEB_SUMMARY.md",
        "STAGEB_FINALIZER_RECEIPT.json", "STAGEB_HASHES.sha256",
    ):
        if (OUT / name).exists():
            raise RuntimeError(f"finalizer output collision: {name}")
    arm_o_reg = json.loads((DOC / "ARM_O_CANARY_REGISTRATION.json").read_text())
    arm_t_reg = json.loads((DOC / "ARM_T_TASKS.json").read_text())
    scheduler = subprocess.run(
        ["sacct", "-j", arm_t_job_id, "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList", "-P"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    completed: dict[str, dict[str, Any]] = {"tiger": {}, "fox": {}}
    failures = []
    for task, task_t in zip(arm_o_reg["tasks"], arm_t_reg["tasks"]):
        if task_t != {"index": task_t["index"], "cell": task["cell"], "method": task["method"]}:
            raise RuntimeError("task registration mismatch in finalizer")
        path = OUT / "arm_t_tasks" / task["task_id"] / "task_receipt.json"
        if not path.exists():
            failures.append({"task": task_t, "reason": "task receipt missing"})
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("exit_status") != 0:
            failures.append({
                "task": task_t, "reason": receipt.get("failure", "nonzero task receipt"),
                "exit_status": receipt.get("exit_status"), "receipt": str(path),
            })
            continue
        completed[task["cell"]][task["method"]] = completed_summary(task, receipt)

    source_hashes = {
        "ecological": source_only_hash("ecological"),
        "general": source_only_hash("general"),
    }
    expected = {
        "ecological": {"count": 52, "sha256": "2b3b8ae6d2f8ff5ffb17c4885ded9e8f1f6b3c0cb662f393186fe4b4706a884e"},
        "general": {"count": 55, "sha256": "f90cea6f28dcacb910b5e036bf9e09958715d00a2418fd0856a3d5a12856bdbd"},
    }
    if source_hashes != expected:
        failures.append({"reason": "frozen source-only hash mismatch", "observed": source_hashes})
    summary = {
        "schema_version": "i2b_provisional_stageb_summary_v1",
        "status": LABEL,
        "scientific_interpretation_performed": False,
        "cross_cell_pooling_performed": False,
        "gate_job_id": gate_job_id,
        "arm_t_array_job_id": arm_t_job_id,
        "cells": completed,
        "failed_or_blocked_tasks": failures,
        "frozen_source_only_hashes": source_hashes,
        "scheduler_accounting_exit": scheduler.returncode,
        "scheduler_accounting": scheduler.stdout,
    }
    json_path = OUT / "PROVISIONAL_STAGEB_SUMMARY.json"
    strict_json(json_path, summary)
    lines = [
        "# Provisional I2B Arm T summary", "", LABEL, "",
        "Fox and tiger are reported separately; EVD is separately labelled. No hypothesis interpretation was performed.", "",
    ]
    for cell in ("tiger", "fox"):
        lines.extend([f"## {cell}", ""])
        if not completed[cell]:
            lines.extend(["No completed task receipts.", ""])
        for method, value in completed[cell].items():
            lines.extend([
                f"### {method}", "", LABEL, "",
                f"- Arm O mean true return: `{value['arm_o_absolute_mean_true_return']}`",
                f"- Arm T mean true return: `{value['arm_t_absolute_mean_true_return']}`",
                f"- Paired raw T−O contrast: `{value['paired_raw_loss_true_minus_noisy']}`",
                f"- Paired 95% bootstrap interval: `{value['paired_percentile_bootstrap_95_interval']}`",
                f"- Collapse rate: `{value['arm_t_collapse_rate']}`",
                f"- Unsafe occupancy steps: `{value['arm_t_unsafe_occupancy_steps']}`",
                f"- Interpretation label: `{value['component_artifact_changes']['interpretation_label']}`",
                "",
            ])
    if failures:
        lines.extend(["## Failed or blocked tasks", "", "```json", json.dumps(failures, indent=2), "```", ""])
    strict_write(OUT / "PROVISIONAL_STAGEB_SUMMARY.md", "\n".join(lines))
    finalizer_receipt = {
        "schema_version": "i2b_stageb_finalizer_receipt_v1",
        "status": LABEL,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "dependency": os.environ.get("SLURM_JOB_DEPENDENCY"),
        "gate_job_id": gate_job_id,
        "arm_t_array_job_id": arm_t_job_id,
        "completed_tasks": sum(len(value) for value in completed.values()),
        "failed_or_blocked_tasks": len(failures),
        "methods_run_by_finalizer": 0,
        "source_hashes": source_hashes,
        "completed_unix": time.time(),
    }
    strict_json(OUT / "STAGEB_FINALIZER_RECEIPT.json", finalizer_receipt)
    manifest_path = OUT / "STAGEB_HASHES.sha256"
    paths = sorted(path for path in OUT.rglob("*") if path.is_file() and path != manifest_path)
    content = "# Self-excluded sealing manifest; paths are repository-relative.\n" + "".join(
        f"{sha(path)}  {path.relative_to(ROOT)}\n" for path in paths
    )
    strict_write(manifest_path, content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm_t_job_id")
    parser.add_argument("gate_job_id")
    arguments = parser.parse_args()
    main(arguments.arm_t_job_id, arguments.gate_job_id)
