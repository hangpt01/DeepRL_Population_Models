#!/usr/bin/env python3
"""Copy completed H12 artifacts and append their provenance hashes."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


FILES = {
    "H12_out/H12_dose_response.csv": "results/followups/H12/H12_dose_response.csv",
    "H12_out/H12_degradation_slopes.csv": "results/followups/H12/H12_degradation_slopes.csv",
    "H12_out/H12_RECEIPT.json": "results/followups/H12/H12_RECEIPT.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-project", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source_root = (
        Path(args.scratch_project).resolve()
        / "real_ecology_runs"
        / "three_species_diagnostic_followups_20260727_v1"
    )
    missing = [rel for rel in FILES if not (source_root / rel).is_file()]
    if missing:
        print("not ready; missing:")
        for rel in missing:
            print(f"  {source_root / rel}")
        return 2
    ledger = repo / "provenance" / "copied_artifacts.sha256"
    existing = ledger.read_text(encoding="utf-8").splitlines()
    existing_by_destination = {
        line.split(maxsplit=2)[1]: line
        for line in existing
        if line and not line.startswith("#") and len(line.split(maxsplit=2)) == 3
    }
    additions = []
    for source_rel, destination_rel in FILES.items():
        source = source_root / source_rel
        destination = repo / destination_rel
        source_hash = digest(source)
        previous = existing_by_destination.get(destination_rel)
        if previous:
            old_hash, _old_destination, old_source = previous.split(maxsplit=2)
            if old_hash != source_hash or Path(old_source) != source:
                raise RuntimeError(
                    f"ledger conflict for {destination_rel}: {previous}"
                )
            print(f"already recorded; skipping {destination_rel}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if digest(destination) != source_hash:
            raise RuntimeError(f"copy hash mismatch for {destination_rel}")
        line = f"{source_hash}  {destination_rel}  {source}"
        additions.append(line)
        print(line)
    if additions:
        with ledger.open("a", encoding="utf-8") as handle:
            for line in additions:
                handle.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
