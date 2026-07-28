#!/usr/bin/env python3
"""Hash the registered executable launcher overlay, separate from scientific src."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


FILES = (
    "scripts/materialize_ricker_only_reuse.py",
    "scripts/ricker_only_thread_entry.py",
    "scripts/run_adapted_fit_row.py",
    "scripts/run_real_manifest_row.py",
    "scripts/run_ricker_only_fit_locked.py",
    "scripts/run_ricker_only_plan_with_receipt.py",
    "scripts/run_ricker_only_plus_acceptance.py",
    "scripts/slurm/run_ricker_only_plus_cell.sh",
    "scripts/validate_ricker_only_plus_launch.py",
)


def overlay_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(json.dumps({
        "digest_schema": "ricker_only_launcher_overlay_v1",
        "files": list(FILES),
        "launcher_overlay_sha256": overlay_digest(root),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
