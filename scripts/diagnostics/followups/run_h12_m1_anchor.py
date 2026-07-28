#!/usr/bin/env python3
"""H12 submission gate: unmodified m=1.0 PLUS/MOOR arms only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIAGNOSTICS_DIR))
from repo_paths import FOLLOWUPS_OUTPUT, REPO_ROOT, require_scientific_tables  # noqa: E402

BASE_DIR = REPO_ROOT / "scripts" / "diagnostics" / "replay"
OUT = Path(os.environ.get("DEEPRL_H12_OUTPUT", FOLLOWUPS_OUTPUT / "H12")) / "m1_anchor"
sys.path.insert(0, str(BASE_DIR))
import run_diagnostic_replay as base  # noqa: E402

base.REPLAY = OUT
base.CELLS = {
    "A1": ("Crab-eating fox", "ricker", "0.2"),
    "A2": ("Crab-eating fox", "allee", "0.2"),
    "A3": ("Crab-eating fox", "regime", "0.2"),
    "A4": ("Crab-eating fox", "theta", "0.2"),
    "B2": ("Amur tiger", "ricker", "0.1"),
    "B3": ("Amur tiger", "allee", "0.2"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cell", choices=tuple(base.CELLS))
    parser.add_argument("side", choices=("plus", "moor"))
    args = parser.parse_args()
    require_scientific_tables()
    if base.sha(base.ACCEPTED_CSV) != base.ACCEPTED_CSV_SHA:
        raise AssertionError("accepted CSV hash changed")
    result = base.run_cell(args.cell, args.side)
    if result["parity"]["max_abs_diff"] != 0.0:
        raise AssertionError(f"H12 m=1.0 gate failed: {result['parity']}")
    label = base.write_outputs(result)
    receipt = {
        "schema": "H12_m1_anchor_v1",
        "cell": args.cell,
        "method": args.side,
        "observation_scale_multiplier": 1.0,
        "parity": result["parity"],
        "recomputed_fits": result["recomputed_fits"],
        "fit_cache_reuse": result["reuse_ok"],
        "full_sweep_authorized": True,
    }
    (OUT / "parity").mkdir(parents=True, exist_ok=True)
    path = OUT / "parity" / f"H12_M1_{args.cell}_{args.side}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
