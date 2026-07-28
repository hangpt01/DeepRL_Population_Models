#!/usr/bin/env python3
"""Write the final content-addressed freeze receipt for the Option A overlay."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[1]
RUN_ROOT = PACKAGE.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    preparation = json.loads(
        (PACKAGE / "PREPARATION_RECEIPT.json").read_text(encoding="utf-8")
    )
    paths = sorted(
        [
            *PACKAGE.glob("manifests/*.csv"),
            *PACKAGE.glob("scripts/*.py"),
            *PACKAGE.glob("slurm/*.sh"),
            *PACKAGE.glob("DRY_RUN_*.json"),
            PACKAGE / "PREPARATION_RECEIPT.json",
        ],
        key=lambda path: str(path.relative_to(PACKAGE)),
    )
    files = {
        str(path.relative_to(PACKAGE)): sha256(path)
        for path in paths
        if path.is_file() and path.name != "FINAL_FREEZE_RECEIPT.json"
    }
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "freeze_schema": "option_a_launch_overlay_freeze_v1",
        "overlay_digest": digest,
        "files": files,
        "manifest_sha256": preparation["manifest_sha256"],
        "frozen_inputs": preparation["frozen_inputs"],
        "scope": preparation["scope"],
        "cache_audit": preparation["cache_audit"],
        "return_fields_opened": False,
    }
    target = PACKAGE / "FINAL_FREEZE_RECEIPT.json"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
