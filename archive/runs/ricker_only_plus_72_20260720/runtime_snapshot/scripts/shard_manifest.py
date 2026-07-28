#!/usr/bin/env python3
"""Split a manifest into array-size-safe shards.

Slurm caps job-array indices at ``MaxArraySize`` (this cluster: 1001, so the
maximum index is 1000).  Manifests larger than that (e.g. the full-learned-fast
2016-row and full-all-filters-fast 3744-row arrays) cannot be submitted as a
single ``--array=0-N`` job.  This helper writes contiguous-index shards
``<stem>_partNNN.csv`` (each re-indexed 0..len-1) and prints the ``sbatch``
array ranges to use.  Manifests that already fit are passed through unchanged.

Usage::

    python scripts/shard_manifest.py outputs/.../manifest_full_learned_fast.csv
    # then submit each shard:
    #   sbatch --parsable --array=0-999%60 --time=18:00:00 scripts/slurm/run_real_row.sh <...>_part000.csv
    #   sbatch --parsable --array=0-999%60 --time=18:00:00 scripts/slurm/run_real_row.sh <...>_part001.csv
    #   sbatch --parsable --array=0-15%60  --time=18:00:00 scripts/slurm/run_real_row.sh <...>_part002.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def shard_manifest(path: str | Path, shard_size: int) -> list[tuple[Path, int]]:
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "index" not in fieldnames:
        raise SystemExit(f"{source} has no 'index' column")
    shards: list[tuple[Path, int]] = []
    for shard_no, start in enumerate(range(0, len(rows), shard_size)):
        chunk = rows[start : start + shard_size]
        target = source.with_name(f"{source.stem}_part{shard_no:03d}{source.suffix}")
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for new_index, row in enumerate(chunk):
                writer.writerow({**row, "index": new_index})
        shards.append((target, len(chunk)))
    return shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="rows per shard; keep <= cluster MaxArraySize (1001 here → 1000 safe)",
    )
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--time", default=None, help="optional sbatch walltime, e.g. 18:00:00")
    parser.add_argument("--script", default="scripts/slurm/run_real_row.sh")
    parser.add_argument("--no-parsable", action="store_true", help="omit sbatch --parsable")
    args = parser.parse_args()
    rows_total = sum(1 for _ in Path(args.manifest).open()) - 1
    prefix = ["sbatch"]
    if not args.no_parsable:
        prefix.append("--parsable")
    def command(target: str | Path, count: int) -> str:
        parts = [*prefix, f"--array=0-{count - 1}%{args.concurrency}"]
        if args.time:
            parts.append(f"--time={args.time}")
        parts.extend([args.script, str(target)])
        return " ".join(parts)

    if rows_total <= args.shard_size:
        print(f"{args.manifest}: {rows_total} rows <= shard-size {args.shard_size}; no sharding needed")
        if rows_total > 0:
            print(f"  {command(args.manifest, rows_total)}")
        return
    shards = shard_manifest(args.manifest, args.shard_size)
    print(f"{args.manifest}: {rows_total} rows -> {len(shards)} shards")
    for target, count in shards:
        print(f"  {command(target, count)}")


if __name__ == "__main__":
    main()
