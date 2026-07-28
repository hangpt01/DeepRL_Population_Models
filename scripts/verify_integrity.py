#!/usr/bin/env python3
"""Verify frozen and load-bearing repository hash ledgers."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def destination_ledger(path: Path) -> dict[str, str]:
    entries = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise AssertionError(f"{path}:{number}: malformed ledger line")
        expected, relative = parts[:2]
        if relative in entries:
            raise AssertionError(f"{path}:{number}: duplicate {relative}")
        entries[relative] = expected
    return entries


def verify_entries(label: str, entries: dict[str, str]) -> None:
    for relative, expected in entries.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise AssertionError(
                f"{label} hash mismatch: {relative}: {actual} != {expected}"
            )
    print(f"PASS {label}={len(entries)}")


def scientific_paths() -> set[str]:
    paths = set()
    for base in (
        ROOT / "scripts" / "ecological",
        ROOT / "scripts" / "general",
        ROOT / "experiments",
    ):
        for path in base.rglob("*"):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            ):
                paths.add(str(path.relative_to(ROOT)))
    for path in (ROOT / "configs" / "ecology").glob("*.csv"):
        paths.add(str(path.relative_to(ROOT)))
    return paths


def main() -> int:
    frozen = destination_ledger(ROOT / "provenance" / "frozen_tracks.sha256")
    verify_entries("frozen_track_files", frozen)

    scientific = destination_ledger(
        ROOT / "provenance" / "scientific_inputs.sha256"
    )
    discovered = scientific_paths()
    if set(scientific) != discovered:
        missing = sorted(discovered - set(scientific))
        stale = sorted(set(scientific) - discovered)
        raise AssertionError(
            f"scientific ledger coverage mismatch: missing={missing}, stale={stale}"
        )
    verify_entries("repo_scientific_inputs", scientific)

    copied = destination_ledger(
        ROOT / "provenance" / "copied_artifacts.sha256"
    )
    verify_entries("copied_result_artifacts", copied)

    accepted = ROOT / "results" / "accepted" / "MATCHED_P10_144_METHOD_CELLS.csv"
    accepted_hash = sha256(accepted)
    expected = "7431318803e468c13ff3008acdc530c0d9f9a919cc29a6fe04951fc8a2cadc1c"
    if accepted_hash != expected:
        raise AssertionError(f"accepted CSV hash mismatch: {accepted_hash}")
    print("PASS accepted_csv=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
