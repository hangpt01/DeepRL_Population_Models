#!/usr/bin/env python3
"""Later-only aggregation for tiger E1; forbidden until independent audit pass."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ARMS = ("A1", "A2", "A3", "A4")
CELLS = ("tiger_ricker_sigma_0.1", "tiger_ricker_sigma_0.2")
EPISODES = tuple(seed + local for seed in (7001, 7051, 7101, 7151, 7201) for local in range(4))
ESTIMANDS = (
    ("A2-A4", "A2", "A4", "observability_primary"),
    ("A1-A2", "A1", "A2", "model_primary"),
    ("A3-A4", "A3", "A4", "model_noisy_secondary"),
    ("A1-A3", "A1", "A3", "numerical_diagnostic_not_estimand"),
)
T_CRITICAL_DF19_975 = 2.093024054408263


def load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--independent-audit-pass-file", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    audit_pass = load(args.independent_audit_pass_file.resolve())
    if audit_pass.get("verdict") != "PASS — SAFE TO AGGREGATE":
        raise AssertionError("independent tiger audit pass is absent")
    output.mkdir(parents=True, exist_ok=False)
    manifest = load(run_dir / "AUDIT_MANIFEST.json")
    if manifest.get("EPYC9534_confirmation_for_every_task") is not True:
        raise AssertionError("audit manifest is incomplete")
    rows, lookup = [], {}
    for cell in CELLS:
        for arm in ARMS:
            for episode in EPISODES:
                path = run_dir / "tasks" / cell / arm / "episodes" / f"episode_{episode}.json"
                record = load(path)
                if record.get("status") != "COMPLETE" or not math.isfinite(float(record["discounted_return"])):
                    raise AssertionError(f"invalid record: {path}")
                lookup[(cell, arm, episode)] = float(record["discounted_return"])
                rows.append({
                    "cell": cell, "arm": arm, "episode": episode,
                    "block_seed": record["block_seed"],
                    "local_episode_index": record["local_episode_index"],
                    "pairing_key": record["pairing_key"],
                    "discounted_return": record["discounted_return"],
                    "undiscounted_return": record["undiscounted_return"],
                    "source_path": str(path), "source_sha256": sha256(path),
                })
    with (output / "raw_episode_returns.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    contrast_rows, paired_rows = [], []
    for cell in CELLS:
        for name, left, right, role in ESTIMANDS:
            differences = [lookup[(cell, left, episode)] - lookup[(cell, right, episode)] for episode in EPISODES]
            mean = sum(differences) / len(differences)
            variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
            standard_error = math.sqrt(variance / len(differences)); half = T_CRITICAL_DF19_975 * standard_error
            contrast_rows.append({
                "cell": cell, "contrast": name, "role": role, "n_paired": len(differences),
                "mean_paired_difference": mean, "sample_standard_deviation": math.sqrt(variance),
                "standard_error": standard_error, "ci_method": "paired Student-t, df=19",
                "ci95_lower": mean - half, "ci95_upper": mean + half,
            })
            paired_rows.extend({"cell": cell, "contrast": name, "role": role, "episode": episode, "paired_difference": difference} for episode, difference in zip(EPISODES, differences))
    for name, data in (("paired_differences.csv", paired_rows), ("paired_contrasts.csv", contrast_rows)):
        with (output / name).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    receipt = {
        "status": "COMPLETE", "run_dir": str(run_dir),
        "independent_audit_pass_file": str(args.independent_audit_pass_file.resolve()),
        "independent_audit_pass_sha256": sha256(args.independent_audit_pass_file.resolve()),
        "ci_method": "paired Student-t, df=19, two-sided 95%", "cells_pooled": False,
        "species_pooled_with_fox": False, "A1_minus_A3_label": "numerical diagnostic, not an estimand",
        "factorial_interaction_estimated": False, "interpretive_decision_applied": False,
        "files": {path.name: sha256(path) for path in sorted(output.glob("*.csv"))},
    }
    with (output / "AGGREGATION_RECEIPT.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True); handle.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
