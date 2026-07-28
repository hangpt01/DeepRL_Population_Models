#!/usr/bin/env python3
"""Copy only hash-verified Ricker fit-cache entries into a new run cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--source-cache", required=True, type=Path)
    parser.add_argument("--target-cache", required=True, type=Path)
    args = parser.parse_args()
    with args.ledger.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 64:
        raise SystemExit("reuse ledger must contain exactly 64 references")
    args.target_cache.mkdir(parents=True, exist_ok=True)
    copied = 0
    for row in rows:
        key = row["cache_key"]
        metadata = args.source_cache / f"{key}.json"
        arrays = args.source_cache / f"{key}.npz"
        if sha256(metadata) != row["metadata_sha256"]:
            raise SystemExit(f"source metadata hash mismatch for {key}")
        if sha256(arrays) != row["arrays_sha256"]:
            raise SystemExit(f"source arrays hash mismatch for {key}")
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        if payload["cache_key"] != key or payload["model"]["form"] != "ricker":
            raise SystemExit(f"source cache identity mismatch for {key}")
        for source in (metadata, arrays):
            target = args.target_cache / source.name
            if target.exists() and sha256(target) != sha256(source):
                raise SystemExit(f"refusing to overwrite different cache entry: {target}")
            if not target.exists():
                shutil.copy2(source, target)
        copied += 1
    print(json.dumps({
        "verified_reuse_references": copied,
        "unique_cache_entries": len({row["cache_key"] for row in rows}),
        "return_fields_opened": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
