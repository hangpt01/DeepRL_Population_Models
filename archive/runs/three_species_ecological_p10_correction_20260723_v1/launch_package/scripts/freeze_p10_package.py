#!/usr/bin/env python3
"""Write the immutable launch-package receipt without hashing itself."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
RECEIPT = PACKAGE / "FINAL_FREEZE_RECEIPT.json"
INCLUDED = (
    "configs/moor_ricker_p10.yaml",
    "configs/plus_ricker_only_p10.yaml",
    "manifests/moor_p10_plan_24.csv",
    "manifests/plus_p10_plan_24.csv",
    "provenance/FIT_REWARD_INDEPENDENCE_AUDIT.json",
    "provenance/fit_reuse_ledger_216.csv",
    "scripts/accept_p10_method.py",
    "scripts/dispatch_p10_plan.py",
    "scripts/prepare_p10_correction.py",
    "scripts/validate_p10_package.py",
    "scripts/freeze_p10_package.py",
    "slurm/run_acceptance_p10.sh",
    "slurm/run_moor_p10.sh",
    "slurm/run_plus_p10.sh",
    "PREPARATION_RECEIPT.json",
    "VALIDATION_RECEIPT.json",
    "DRY_RUN_MOOR.json",
    "DRY_RUN_PLUS.json",
)
RUNTIMES = {
    "plus_ricker_only_routing_fix": (
        ROOT.parents[0]
        / "ricker_only_plus_72_20260720/runtime_snapshot_routing_fix"
    ),
    "moor_corrected": (
        ROOT.parents[0] / "adapted_32cell_diagnostic_pending/code"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def main() -> None:
    files = {name: sha256(PACKAGE / name) for name in INCLUDED}
    payload = {
        "freeze_schema": "three_species_ecological_p10_launch_freeze_v1",
        "experiment_root": str(ROOT),
        "scientific_target": {
            "collapse_penalty": 10.0,
            "methods": [
                "plus_adapted_ricker_only_pbvi",
                "moor_adapted_ricker_misspec_pbvi",
            ],
            "method_cells": 48,
            "fit_candidate_slots_reused": 216,
            "fit_candidate_slots_recomputed": 0,
            "planners_and_evaluations": 48,
        },
        "package_files": files,
        "package_digest_excluding_this_receipt": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "runtime_tree_digests": {
            name: tree_digest(path) for name, path in RUNTIMES.items()
        },
        "return_fields_opened": False,
    }
    temporary = RECEIPT.with_name(f".{RECEIPT.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(RECEIPT)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
