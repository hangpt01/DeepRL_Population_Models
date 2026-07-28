#!/usr/bin/env python3
"""Run the registered plan row and atomically publish a return-blind completion receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_real_manifest_row import read_manifest_row  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temporary_files(root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and (path.name.startswith(".") or ".tmp" in path.name)
    )


def verified_artifact_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise RuntimeError(f"faithful artifact directory is missing: {root}")
    temporary = temporary_files(root)
    if temporary:
        raise RuntimeError(f"temporary artifacts remain: {temporary}")
    required = {
        "candidate_bank.json", "candidate_bank.npz", "faithful_fit.json",
        "faithful_fit.npz", "planner_provenance.json", "privacy_audit.json",
        "pbvi_policy_diagnostics.npz",
    }
    for index in range(8):
        required.update({
            f"candidate_{index:03d}.json", f"candidate_{index:03d}.npz",
            f"pomdp_model_{index:03d}.json", f"pomdp_model_{index:03d}.npz",
        })
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise RuntimeError(f"required structural artifacts are missing: {missing}")
    bank = json.loads((root / "candidate_bank.json").read_text(encoding="utf-8"))
    if bank.get("array_hash") != sha256(root / "candidate_bank.npz"):
        raise RuntimeError("candidate-bank array hash mismatch")
    for prefix in ("candidate", "pomdp_model"):
        for index in range(8):
            metadata = root / f"{prefix}_{index:03d}.json"
            arrays = root / f"{prefix}_{index:03d}.npz"
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            if payload.get("array_hash") != sha256(arrays):
                raise RuntimeError(f"artifact array hash mismatch: {arrays.name}")
    privacy = json.loads((root / "privacy_audit.json").read_text(encoding="utf-8"))
    if privacy.get("status") != "passed" or privacy.get("forbidden_name_hits"):
        raise RuntimeError("privacy audit did not pass")
    return {name: sha256(root / name) for name in sorted(required)}


def write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("index", type=int)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    row = read_manifest_row(args.manifest, args.index)
    if row.get("run_stage") != "plan_evaluate":
        raise SystemExit("plan receipt wrapper accepts only plan_evaluate rows")
    runner = ROOT / "scripts/run_real_manifest_row.py"
    sys.argv = [
        str(runner), args.manifest, str(args.index), "--config", args.config,
        "--output-root", str(args.output_root),
    ]
    runpy.run_path(str(runner), run_name="__main__")
    artifact_root = args.output_root / row["evaluation_artifact_dir"]
    hashes = verified_artifact_hashes(artifact_root)
    receipt = {
        "receipt_schema": "ricker_only_plan_completion_v1",
        "completion_status": "complete",
        "receipt_write": "atomic_replace",
        "return_fields_opened": False,
        "manifest_index": int(row["index"]),
        "method": row["method"],
        "fit_cell": row["fit_cell"],
        "artifact_dir": row["evaluation_artifact_dir"],
        "artifact_hashes": hashes,
    }
    target = (
        args.output_root / "completion_receipts" / row["fit_cell"]
        / row["method"] / "plan_completion.json"
    )
    write_atomic(target, receipt)
    print(json.dumps({"completion_receipt": str(target), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
