#!/usr/bin/env python3
"""Validate dedicated Phase 2E structural receipts without opening outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark.general_canary_acceptance import summarize_validity_receipts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipts", nargs="+")
    args = parser.parse_args()
    print(json.dumps(summarize_validity_receipts(args.receipts), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
