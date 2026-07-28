#!/usr/bin/env python3
"""Validate and aggregate the nine H14 parity-gated arms."""
import csv
import json
import os
from pathlib import Path
import sys

DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIAGNOSTICS_DIR))
from repo_paths import FOLLOWUPS_OUTPUT  # noqa: E402

OUT = Path(os.environ.get("DEEPRL_H14_OUTPUT", FOLLOWUPS_OUTPUT / "H14"))

receipts = []
for path in sorted((OUT / "parity").glob("PARITY_*.json")):
    record = json.loads(path.read_text())
    if record.get("method") in {"refplan", "ogsrl", "bamcts"}:
        receipts.append((path, record))
if len(receipts) != 9:
    raise AssertionError(f"expected 9 H14 receipts, found {len(receipts)}")
if any(r["parity"]["max_abs_diff"] != 0.0 for _, r in receipts):
    raise AssertionError("H14 parity not exact")
if any(r["recomputed_fits"] != 0 for _, r in receipts):
    raise AssertionError("H14 unexpectedly recomputed a fit")

rows = []
for path, r in receipts:
    for stratum in ("overall", "safe", "unsafe"):
        m = r["M13_surrogate_error_visited"][stratum]
        rows.append({
            "cell": r["cell"], "method": r["method"], "stratum": stratum,
            "n": m["n"], "mean_signed_error": m["mean_signed_error"],
            "rmse": m["rmse"], "parity_max_abs_diff": 0.0,
        })
with (OUT / "H14_M13.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

receipt = {
    "schema": "H14_reinstrument_receipt_v1",
    "cells": ["B1", "B2", "B3"],
    "methods": ["refplan", "ogsrl", "bamcts"],
    "arms": 9, "parity_pass": 9, "parity_max_abs_diff": 0.0,
    "recomputed_fits": 0, "EVD_exempt": True,
    "surrogate_reward_t_present": True,
    "logging_after_act": True, "logging_consumed_rng": False,
    "accepted_artifacts_modified": False, "reranked": False,
    "jobs": {"anchor": "58579344", "remaining_array": "58579415"},
    "outputs": ["H14_M13.csv"],
}
(OUT / "H14_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
