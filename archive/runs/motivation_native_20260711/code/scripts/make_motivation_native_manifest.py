#!/usr/bin/env python3
"""Create the motivation experiment manifest with explicit native filter routing.

``--natives-only`` emits just the two native-ecological methods.  That variant is
what the discretization-resolution replicate runs (acceptance test 7.6): the same
cells re-solved at a coarse and a fine native grid, driven by
``configs/motivation_native_b{31,91}.yaml``.  The general methods do not depend on
``native_state_bins``, so re-running them would burn ~99% of the cost for nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark.manifest import make_manifest, make_motivation_native_manifest

NATIVE_ROUTING = {
    "plus_native": ("native_discrete",),
    "moor_native": ("native_discrete",),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "motivation_native_baselines" / "manifest.csv"),
    )
    parser.add_argument(
        "--natives-only",
        action="store_true",
        help="emit only plus_native/moor_native rows (the resolution replicate)",
    )
    args = parser.parse_args()
    if args.natives_only:
        rows = make_manifest(
            args.output,
            methods=tuple(NATIVE_ROUTING),
            data_mode="real",
            method_filters=NATIVE_ROUTING,
        )
    else:
        rows = make_motivation_native_manifest(args.output)
    print(json.dumps({"output": args.output, "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
