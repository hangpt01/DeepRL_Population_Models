#!/usr/bin/env python3
"""Recompute the registered paper-faithful source-snapshot tree digest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if "__pycache__" not in path.parts and path.suffix not in EXCLUDED_SUFFIXES
    ]


def snapshot_tree_hash(root: Path) -> tuple[str, int]:
    """Hash UTF-8 relative paths followed by ASCII per-file SHA-256 digests."""

    digest = hashlib.sha256()
    files = included_files(root)
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest(), len(files)


def repeated_roots_hash(roots: list[Path]) -> tuple[str, int]:
    """Hash several named roots using one reproducible lexical path namespace."""

    resolved = [root.resolve() for root in roots]
    common = Path(os.path.commonpath([str(root) for root in resolved]))
    entries = sorted(
        (path.relative_to(common).as_posix(), path)
        for root in resolved
        for path in included_files(root)
    )
    digest = hashlib.sha256()
    for relative, path in entries:
        digest.update(relative.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest(), len(entries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", help="One or more source roots to verify")
    parser.add_argument("--expected", help="Fail unless the digest matches this value")
    args = parser.parse_args()
    roots = [Path(value).resolve() for value in args.roots]
    for root in roots:
        if not root.is_dir():
            raise SystemExit(f"snapshot root is not a directory: {root}")
    digest, file_count = (
        snapshot_tree_hash(roots[0])
        if len(roots) == 1
        else repeated_roots_hash(roots)
    )
    matched = args.expected is None or digest == args.expected
    print(
        json.dumps(
            {
                "roots": [str(root) for root in roots],
                "included_file_count": file_count,
                "snapshot_tree_sha256": digest,
                "expected": args.expected,
                "matched": matched,
                "recipe": (
                    "lexically sorted recursive files under the supplied roots; exclude "
                    "__pycache__, .pyc, and .pyo; append each UTF-8 POSIX path relative "
                    "to the common root then its ASCII SHA-256 hex digest; single-root "
                    "mode retains the registered legacy relative-path convention"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not matched:
        raise SystemExit("snapshot tree digest mismatch")


if __name__ == "__main__":
    main()
