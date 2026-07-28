#!/usr/bin/env python3
"""Record external-solver availability without enabling an unverified method ID."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidates = ("despot", "pomdpsol", "sarsop")
    found = {name: shutil.which(name) for name in candidates}
    payload = {
        "despot_enabled": False,
        "sarsop_enabled": False,
        "pbvi_enabled": True,
        "executables": found,
        "reason": (
            "no external solver method ID is registered until a pinned binary, full "
            "license review, repeated fixed-seed probe, and invocation hash gate pass"
        ),
    }
    payload["probe_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
