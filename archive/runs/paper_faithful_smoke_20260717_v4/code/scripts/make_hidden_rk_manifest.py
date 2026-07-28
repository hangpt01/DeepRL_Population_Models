#!/usr/bin/env python3
"""Generate the corrected hidden-r/K headline manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark.manifest import make_manifest


ROUTING = {
    "refplan": ("learned",),
    "bamcts": ("learned",),
    "ogsrl": ("learned",),
    "moor_native": ("native_discrete",),
    "plus_native": ("native_discrete",),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "hidden_rk" / "manifest.csv"),
    )
    args = parser.parse_args()
    rows = make_manifest(
        args.output,
        methods=tuple(ROUTING),
        data_mode="real",
        method_filters=ROUTING,
        expose_rk="hidden",
    )
    print(json.dumps({"output": args.output, "rows": rows, "expose_rk": "hidden"}, indent=2))


if __name__ == "__main__":
    main()
