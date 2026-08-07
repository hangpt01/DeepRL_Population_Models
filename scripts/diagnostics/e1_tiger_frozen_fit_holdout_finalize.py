#!/usr/bin/env python3
"""Fail-closed finalizer for the two Stage 1 tiger holdout tasks."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO = Path("/fs04/scratch2/ce25/DeepRL_Population_Models")
OUTPUT = REPO / "outputs" / "e1_tiger_frozen_fit_holdout_predictive_20260805_v1"
REGISTRATION = OUTPUT / "STAGE1_REGISTRATION.json"
REGISTRATION_SIDECAR = OUTPUT / "STAGE1_REGISTRATION.json.sha256"
TOKENS = ("sigma_0p1", "sigma_0p2")
COMMIT = "3291eefbe64b5ab19019120bb0ba34a0a9586a53"
PYTHON = Path("/fs04/scratch2/ce25/hphung/conda/envs/poprl/bin/python3.10")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_new(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def verify_sidecar(path: Path, sidecar: Path) -> str:
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise AssertionError(f"invalid sidecar {sidecar}")
    actual = sha256_file(path)
    if actual != fields[0]:
        raise AssertionError(f"hash mismatch for {path}")
    return actual


def validate_task(token: str, registration_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = OUTPUT / "tasks" / token
    manifest = read_json(root / "SHA256_MANIFEST.json")
    for name, expected in manifest["files"].items():
        if sha256_file(root / name) != expected:
            raise AssertionError(f"task file changed: {root / name}")
    receipt = read_json(root / "TASK_RECEIPT.json")
    summary = read_json(root / "CELL_SUMMARY.json")
    zero = read_json(root / "ZERO_REFIT_EVIDENCE.json")
    if receipt.get("status") != "PASS" or summary.get("status") != "PASS":
        raise AssertionError(f"task {token} did not pass")
    if receipt.get("registration_sha256") != registration_hash:
        raise AssertionError(f"task {token} registration mismatch")
    if receipt.get("fits_executed") != 0 or receipt.get("optimizer_calls") != 0:
        raise AssertionError(f"task {token} executed a fit")
    if zero.get("status") != "PASS" or zero.get("fits_executed") != 0:
        raise AssertionError(f"task {token} zero-refit evidence failed")
    if receipt.get("holdout_episode_count") != 32:
        raise AssertionError(f"task {token} holdout count mismatch")
    if receipt.get("excluded_transition_count") != 0:
        raise AssertionError(f"task {token} unexpectedly excluded transitions")
    return receipt, summary


def format_number(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.10g}"
    return str(value)


def report(cell_summaries: dict[str, Any], receipt: dict[str, Any]) -> str:
    lines = [
        "# Tiger frozen-fit holdout predictive error — Stage 1",
        "",
        "Status: **COMPLETED — INDEPENDENT CLAUDE AUDIT REQUIRED BEFORE ACCEPTANCE**",
        "",
        (
            "These results are conditional on one collection log and one frozen fit per "
            "noise cell. The independent statistical unit is the holdout episode (32 per cell)."
        ),
        "",
        (
            "Survey metrics score the noisy next observed survey and are never labelled true "
            "abundance. Latent metrics separately score the evaluator-only untouched true next "
            "abundance. Both are in raw Amur tiger abundance/survey-count units."
        ),
        "",
        "## Exact metrics",
        "",
    ]
    for token in TOKENS:
        cell = cell_summaries[token]
        lines.extend([f"### {token} (sigma={cell['sigma_obs']})", ""])
        for target_name in ("next_observed_survey", "next_true_latent_abundance"):
            target = cell["targets"][target_name]
            pooled = target["transition_pooled_descriptive_only"]
            episode = target["episode_equal_with_ordinary_student_t_95_ci"]
            lines.extend(
                [
                    f"Target: `{target_name}` — {target['target_label']}",
                    "",
                    (
                        "Transition-pooled (descriptive only): "
                        f"MAE={format_number(pooled['mae'])}; "
                        f"RMSE={format_number(pooled['rmse'])}; "
                        f"mean NLPD={format_number(pooled['mean_per_transition_nlpd'])}; "
                        f"total NLPD={format_number(pooled['total_nlpd'])}; "
                        f"mean PIT={format_number(pooled['mean_pit'])}."
                    ),
                    "",
                ]
            )
            for label, key in (
                ("episode MAE", "episode_mae"),
                ("episode RMSE", "episode_rmse"),
                ("episode mean per-transition NLPD", "episode_mean_per_transition_nlpd"),
            ):
                item = episode[key]
                lines.append(
                    f"Episode-equal {label}: mean={format_number(item['mean'])}; "
                    f"ordinary two-sided 95% Student-t CI "
                    f"[{format_number(item['ci_95_lower'])}, "
                    f"{format_number(item['ci_95_upper'])}]."
                )
            lines.append("")
        lines.extend(
            [
                f"Excluded transitions: {cell['exclusions']['count']} (reasons: none).",
                f"Cell elapsed time: {format_number(cell['elapsed_seconds'])} seconds.",
                "",
            ]
        )
    lines.extend(
        [
            "## Integrity and scope",
            "",
            "- Fits executed: 0; optimizer calls: 0.",
            "",
            "- Exact episode-preserving split: 128 fit / 32 holdout; no overlap.",
            "",
            "- Both process and observation uncertainty are integrated under the frozen fitted model.",
            "",
            "- Transition-pooled values are descriptive only; uncertainty intervals use episode-level metrics.",
            "",
            "- Predictive interval coverage: NOT AVAILABLE (not registered); PIT is reported instead.",
            "",
            "- Results are not accepted until the independent read-only Claude audit passes.",
            "",
            f"Total array elapsed task time (sum): {format_number(receipt['sum_task_elapsed_seconds'])} seconds.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    start = time.perf_counter()
    if sys.version_info[:3] != (3, 10, 14) or np.__version__ != "2.2.6":
        raise AssertionError("finalizer runtime mismatch")
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise AssertionError("finalizer Python executable mismatch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if commit != COMMIT:
        raise AssertionError("finalizer source commit mismatch")
    registration_hash = verify_sidecar(REGISTRATION, REGISTRATION_SIDECAR)
    receipts, summaries = {}, {}
    for token in TOKENS:
        receipts[token], summaries[token] = validate_task(token, registration_hash)
    if (OUTPUT / "RUN_RECEIPT.json").exists():
        raise FileExistsError("completed Stage 1 namespace already exists")
    sum_elapsed = float(sum(float(item["elapsed_seconds"]) for item in receipts.values()))
    run_receipt = {
        "schema": "e1_tiger_frozen_fit_holdout_run_receipt_v1",
        "status": "COMPLETED_AWAITING_INDEPENDENT_CLAUDE_AUDIT",
        "created_utc": utc_now(),
        "registration_sha256": registration_hash,
        "source_commit": COMMIT,
        "tasks": {
            token: {
                "task_receipt_sha256": sha256_file(
                    OUTPUT / "tasks" / token / "TASK_RECEIPT.json"
                ),
                "cell_summary_sha256": sha256_file(
                    OUTPUT / "tasks" / token / "CELL_SUMMARY.json"
                ),
                "status": receipts[token]["status"],
                "fits_executed": receipts[token]["fits_executed"],
                "optimizer_calls": receipts[token]["optimizer_calls"],
            }
            for token in TOKENS
        },
        "fits_executed": 0,
        "optimizer_calls": 0,
        "fit_episode_count_per_cell": 128,
        "holdout_episode_count_per_cell": 32,
        "holdout_transition_count_per_cell": 800,
        "excluded_transition_count": 0,
        "sum_task_elapsed_seconds": sum_elapsed,
        "finalizer_elapsed_seconds_before_writes": time.perf_counter() - start,
        "conditionality": "one collection log and one frozen fit per noise cell",
        "acceptance": "requires independent Claude-server read-only audit PASS",
    }
    write_json_new(OUTPUT / "CELL_LEVEL_SUMMARY.json", summaries)
    write_json_new(
        OUTPUT / "ZERO_REFIT_EVIDENCE.json",
        {
            "status": "PASS",
            "fits_executed": 0,
            "optimizer_calls": 0,
            "task_evidence": {
                token: sha256_file(
                    OUTPUT / "tasks" / token / "ZERO_REFIT_EVIDENCE.json"
                )
                for token in TOKENS
            },
        },
    )
    write_json_new(
        OUTPUT / "ENVIRONMENT_RECORD.json",
        {
            "created_utc": utc_now(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "executable": str(Path(sys.executable).resolve()),
            "source_commit": COMMIT,
            "partition": "comp",
            "qos": "normal",
            "constraint": "EPYC9534",
            "cpus_per_task": 1,
            "array": "0-1%2",
            "task_environment_sha256": {
                token: sha256_file(OUTPUT / "tasks" / token / "ENVIRONMENT.json")
                for token in TOKENS
            },
        },
    )
    write_json_new(OUTPUT / "RUN_RECEIPT.json", run_receipt)
    write_text_new(OUTPUT / "RESULTS_REPORT.md", report(summaries, run_receipt))
    manifest = {}
    for path in sorted(OUTPUT.rglob("*")):
        if not path.is_file() or path.name == "SHA256_MANIFEST.json":
            continue
        if "logs" in path.relative_to(OUTPUT).parts:
            continue
        manifest[str(path.relative_to(OUTPUT))] = sha256_file(path)
    write_json_new(
        OUTPUT / "SHA256_MANIFEST.json",
        {
            "schema": "e1_tiger_frozen_fit_holdout_sha256_manifest_v1",
            "created_utc": utc_now(),
            "files": manifest,
        },
    )
    print(json.dumps(run_receipt, sort_keys=True))


if __name__ == "__main__":
    main()
