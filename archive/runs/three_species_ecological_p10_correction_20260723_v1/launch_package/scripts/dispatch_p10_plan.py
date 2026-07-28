#!/usr/bin/env python3
"""Position-safe plan/evaluate dispatcher for the P=10 correction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
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
METHODS = {
    "plus": "plus_adapted_ricker_only_pbvi",
    "moor": "moor_adapted_ricker_misspec_pbvi",
}


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


def paths(group: str) -> tuple[Path, Path, Path, Path]:
    if group == "plus":
        return (
            PACKAGE / "manifests/plus_p10_plan_24.csv",
            PACKAGE / "configs/plus_ricker_only_p10.yaml",
            RUN_ROOT / "plus",
            PLUS_CODE,
        )
    if group == "moor":
        return (
            PACKAGE / "manifests/moor_p10_plan_24.csv",
            PACKAGE / "configs/moor_ricker_p10.yaml",
            RUN_ROOT / "moor",
            MOOR_CODE,
        )
    raise ValueError(f"unsupported method group: {group}")


def command_spec(group: str, position: int) -> dict[str, Any]:
    manifest, config, output_root, code_root = paths(group)
    manifest_rows = rows(manifest)
    if len(manifest_rows) != 24 or not 0 <= position < 24:
        raise ValueError("array position is outside the frozen 24-row manifest")
    row = manifest_rows[position]
    method = METHODS[group]
    expected = {
        "index": str(position),
        "run_stage": "plan_evaluate",
        "method": method,
        "reward_mode": "safe",
        "collapse_penalty": "10.0",
        "evaluation_seeds": "7001;7051;7101;7151;7201",
        "evaluation_episodes": "20",
        "episodes_per_evaluation_seed": "4",
        "evaluation_horizon": "50",
        "planner_discount": "0.95",
        "evaluation_discount": "0.95",
        "fit_cache_hit_required": "True",
        "new_candidate_fits": "0",
    }
    mismatches = {
        key: (row.get(key), value)
        for key, value in expected.items()
        if row.get(key) != value
    }
    if mismatches:
        raise ValueError(f"manifest execution identity mismatch: {mismatches}")
    if row.get("config_sha256") != sha256(config):
        raise ValueError("P=10 configuration hash mismatch")
    if group == "plus":
        command = [
            str(PYTHON),
            str(code_root / "scripts/ricker_only_thread_entry.py"),
            str(code_root / "scripts/run_ricker_only_plan_with_receipt.py"),
            str(manifest),
            str(position),
            "--config",
            str(config),
            "--output-root",
            str(output_root),
        ]
        artifact_dir = output_root / row["evaluation_artifact_dir"]
    else:
        command = [
            str(PYTHON),
            str(code_root / "scripts/phase2_thread_entry.py"),
            "--threads",
            "1",
            str(code_root / "scripts/run_real_manifest_row.py"),
            str(manifest),
            str(position),
            "--config",
            str(config),
            "--output-root",
            str(output_root),
            "--allow-uncalibrated",
        ]
        artifact_dir = (
            output_root
            / "evaluation/regime_hidden"
            / row["population"].lower().replace("-", "_").replace(" ", "_")
            / row["environment"]
            / f"sigma_{float(row['sigma_obs']):g}".replace(".", "p")
            / row["config_tag"]
            / "data_real/backend_numpy/regime_hidden/reward_safe"
            / method
            / "faithful_internal/faithful_artifacts"
        )
    return {
        "group": group,
        "position": position,
        "stored_manifest_index": int(row["index"]),
        "method": method,
        "population": row["population"],
        "environment": row["environment"],
        "sigma_obs": row["sigma_obs"],
        "collapse_penalty": 10.0,
        "dataset_sha256": row["authoritative_dataset_sha256"],
        "fit_receipt": str(output_root / row["fit_receipt_path"]),
        "output_directory": str(artifact_dir),
        "dependency_key": row["fit_cell"],
        "command": command,
    }


def moor_completion(position: int) -> None:
    manifest, _config, output_root, _code_root = paths("moor")
    row = rows(manifest)[position]
    method = METHODS["moor"]
    base = (
        output_root
        / "evaluation/regime_hidden"
        / row["population"].lower().replace("-", "_").replace(" ", "_")
        / row["environment"]
        / f"sigma_{float(row['sigma_obs']):g}".replace(".", "p")
        / row["config_tag"]
    )
    matches = list(
        base.glob(f"**/{method}/faithful_internal/faithful_artifacts")
    )
    if len(matches) != 1:
        raise RuntimeError("MOOR faithful artifact directory is not unique")
    root = matches[0]
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
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise RuntimeError(f"MOOR structural artifacts are missing: {missing}")
    temporary = [
        str(path)
        for path in root.rglob("*")
        if path.is_file() and (path.name.startswith(".") or ".tmp" in path.name)
    ]
    if temporary:
        raise RuntimeError(f"MOOR temporary artifacts remain: {temporary}")
    receipt = {
        "receipt_schema": "three_species_p10_moor_plan_completion_v1",
        "completion_status": "complete",
        "receipt_write": "atomic_replace",
        "return_fields_opened": False,
        "manifest_index": position,
        "method": method,
        "fit_cell": row["fit_cell"],
        "collapse_penalty": 10.0,
        "artifact_dir": str(root.relative_to(output_root)),
        "artifact_hashes": {
            name: sha256(root / name) for name in sorted(required)
        },
    }
    atomic_json(
        output_root
        / "completion_receipts"
        / row["fit_cell"]
        / method
        / "plan_completion.json",
        receipt,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("plus", "moor"), required=True)
    parser.add_argument("--position", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    positions = [args.position] if args.position is not None else list(range(24))
    specs = [command_spec(args.method, position) for position in positions]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "method_group": args.method,
                    "commands": specs,
                    "dry_run_count": len(specs),
                    "return_fields_opened": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if len(positions) != 1:
        raise SystemExit("execution requires exactly one array position")
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    subprocess.run(specs[0]["command"], check=True, env=env)
    if args.method == "moor":
        moor_completion(positions[0])


if __name__ == "__main__":
    main()
