#!/usr/bin/env python3
"""Position-safe Option A cell dispatcher over frozen scientific runners."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[1]
RUN_ROOT = PACKAGE.parent
WORKSPACE = RUN_ROOT.parents[1]
PYTHON = WORKSPACE / ".venv-paper-faithful/bin/python"
PLUS_CODE = (
    WORKSPACE
    / "real_ecology_runs/ricker_only_plus_72_20260720/runtime_snapshot_routing_fix"
)
MOOR_CODE = WORKSPACE / "real_ecology_runs/adapted_32cell_diagnostic_pending/code"
PLUS_METHOD = "plus_adapted_ricker_only_pbvi"
MOOR_METHOD = "moor_adapted_ricker_misspec_pbvi"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def expected_paths(method: str) -> tuple[Path, Path, Path, Path]:
    manifests = PACKAGE / "manifests"
    if method == "plus":
        return (
            manifests / "plus_alignment_fit_2.csv",
            manifests / "plus_alignment_plan_2.csv",
            RUN_ROOT / "plus_alignment",
            PLUS_CODE,
        )
    if method == "moor":
        return (
            manifests / "moor_crab_fit_8.csv",
            manifests / "moor_crab_plan_8.csv",
            RUN_ROOT / "moor_crab",
            MOOR_CODE,
        )
    raise ValueError(f"unsupported method group: {method}")


def command_specs(method: str, position: int) -> list[dict[str, Any]]:
    fit_manifest, plan_manifest, output_root, code_root = expected_paths(method)
    fit_rows, plan_rows = rows(fit_manifest), rows(plan_manifest)
    if not 0 <= position < len(fit_rows) or len(fit_rows) != len(plan_rows):
        raise ValueError("array position is outside the paired manifest")
    fit, plan = fit_rows[position], plan_rows[position]
    expected_method = PLUS_METHOD if method == "plus" else MOOR_METHOD
    pair_fields = (
        "population",
        "environment",
        "sigma_obs",
        "reward_mode",
        "method",
        "authoritative_dataset_sha256",
        "authoritative_public_file_sha256",
        "authoritative_private_file_sha256",
        "evaluation_seeds",
        "evaluation_horizon",
        "planner_horizon",
        "planner_discount",
        "evaluation_discount",
    )
    if fit["run_stage"] != "dynamics_fit" or plan["run_stage"] != "plan_evaluate":
        raise ValueError("unsupported stage in paired manifest")
    if fit["method"] != expected_method or plan["method"] != expected_method:
        raise ValueError("method routing mismatch")
    if any(fit[field] != plan[field] for field in pair_fields):
        raise ValueError("fit/plan row identity mismatch")
    if fit["index"] != str(position) or plan["index"] != str(position):
        raise ValueError("stored index is not the deterministic row position")
    if method == "plus":
        config = code_root / "configs/paper_faithful_hidden_ricker_only_plus_v1.yaml"
        fit_command = [
            str(PYTHON),
            str(code_root / "scripts/ricker_only_thread_entry.py"),
            str(code_root / "scripts/run_ricker_only_fit_locked.py"),
            str(fit_manifest),
            str(position),
            "--config",
            str(config),
            "--output-root",
            str(output_root),
        ]
        plan_command = [
            str(PYTHON),
            str(code_root / "scripts/ricker_only_thread_entry.py"),
            str(code_root / "scripts/run_ricker_only_plan_with_receipt.py"),
            str(plan_manifest),
            str(position),
            "--config",
            str(config),
            "--output-root",
            str(output_root),
        ]
    else:
        config = code_root / "configs/paper_faithful_hidden.yaml"
        thread_entry = code_root / "scripts/phase2_thread_entry.py"
        fit_command = [
            str(PYTHON),
            str(thread_entry),
            "--threads",
            "1",
            str(code_root / "scripts/run_adapted_fit_row.py"),
            str(fit_manifest),
            str(position),
            "--config",
            str(config),
            "--output-root",
            str(output_root),
        ]
        plan_command = [
            str(PYTHON),
            str(thread_entry),
            "--threads",
            "1",
            str(code_root / "scripts/run_real_manifest_row.py"),
            str(plan_manifest),
            str(position),
            "--config",
            str(config),
            "--output-root",
            str(output_root),
            "--allow-uncalibrated",
        ]
    return [
        {
            "stage": "fit",
            "position": position,
            "stored_manifest_index": int(fit["index"]),
            "method": expected_method,
            "family": fit["environment"],
            "sigma_obs": fit["sigma_obs"],
            "output_root": str(output_root),
            "dataset_sha256": fit["authoritative_dataset_sha256"],
            "command": fit_command,
        },
        {
            "stage": "plan_evaluate",
            "position": position,
            "stored_manifest_index": int(plan["index"]),
            "method": expected_method,
            "family": plan["environment"],
            "sigma_obs": plan["sigma_obs"],
            "output_root": str(output_root),
            "dataset_sha256": plan["authoritative_dataset_sha256"],
            "command": plan_command,
        },
    ]


def moor_completion(position: int) -> None:
    _fit_manifest, plan_manifest, output_root, _code_root = expected_paths("moor")
    row = rows(plan_manifest)[position]
    receipt = output_root / row["fit_receipt_path"]
    if not receipt.is_file():
        raise RuntimeError("MOOR fit receipt is missing after successful fit")
    fit_payload = json.loads(receipt.read_text(encoding="utf-8"))
    if (
        fit_payload.get("return_fields_opened") is not False
        or fit_payload.get("method") != MOOR_METHOD
        or fit_payload.get("candidate_count") != 1
    ):
        raise RuntimeError("MOOR fit receipt failed structural validation")
    cell = (
        output_root
        / "evaluation/regime_hidden/crab_eating_fox"
        / row["environment"]
        / f"sigma_{float(row['sigma_obs']):g}".replace(".", "p")
    )
    matches = list(
        cell.glob(
            f"**/{MOOR_METHOD}/faithful_internal/faithful_artifacts"
        )
    )
    if len(matches) != 1:
        raise RuntimeError("MOOR faithful artifact directory is not unique")
    artifact_root = matches[0]
    required = {
        "candidate_000.json",
        "candidate_000.npz",
        "faithful_fit.json",
        "faithful_fit.npz",
        "planner_provenance.json",
        "privacy_audit.json",
        "pomdp_model_000.json",
        "pomdp_model_000.npz",
        "pbvi_policy_diagnostics.npz",
    }
    missing = sorted(name for name in required if not (artifact_root / name).is_file())
    if missing:
        raise RuntimeError(f"MOOR structural artifacts are missing: {missing}")
    temporary = [
        str(path)
        for path in artifact_root.rglob("*")
        if path.is_file() and (path.name.startswith(".") or ".tmp" in path.name)
    ]
    if temporary:
        raise RuntimeError(f"MOOR temporary artifacts remain: {temporary}")
    hashes = {name: sha256(artifact_root / name) for name in sorted(required)}
    completion = {
        "receipt_schema": "option_a_moor_plan_completion_v1",
        "completion_status": "complete",
        "receipt_write": "atomic_replace",
        "return_fields_opened": False,
        "manifest_index": position,
        "method": MOOR_METHOD,
        "fit_cell": row["fit_cell"],
        "artifact_dir": str(artifact_root.relative_to(output_root)),
        "artifact_hashes": hashes,
    }
    atomic_json(
        output_root
        / "completion_receipts"
        / row["fit_cell"]
        / MOOR_METHOD
        / "plan_completion.json",
        completion,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("plus", "moor"), required=True)
    parser.add_argument("--position", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    fit_manifest, _plan_manifest, _output_root, _code_root = expected_paths(args.method)
    all_positions = (
        [args.position]
        if args.position is not None
        else list(range(len(rows(fit_manifest))))
    )
    resolved = [
        spec for position in all_positions for spec in command_specs(args.method, position)
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "method_group": args.method,
                    "commands": resolved,
                    "dry_run_count": len(resolved),
                    "return_fields_opened": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if len(all_positions) != 1:
        raise SystemExit("execution requires exactly one array position")
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    for spec in resolved:
        subprocess.run(spec["command"], check=True, env=env)
    if args.method == "moor":
        moor_completion(all_positions[0])


if __name__ == "__main__":
    main()
