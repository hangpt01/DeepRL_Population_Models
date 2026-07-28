#!/usr/bin/env python3
"""Validate and aggregate H12 m=1 anchors plus the 96 non-anchor arms."""
import csv
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIAGNOSTICS_DIR))
from repo_paths import FOLLOWUPS_OUTPUT  # noqa: E402

OUT = Path(os.environ.get("DEEPRL_H12_OUTPUT", FOLLOWUPS_OUTPUT / "H12"))

anchors = []
for path in sorted((OUT / "m1_anchor/parity").glob("H12_M1_*.json")):
    anchors.append(json.loads(path.read_text()))
if len(anchors) != 12:
    raise AssertionError(f"expected 12 m=1 anchors, found {len(anchors)}")
if any(a["parity"]["max_abs_diff"] != 0.0 for a in anchors):
    raise AssertionError("m=1 parity gate is not exact")
if any(a["recomputed_fits"] != 0 or not a["fit_cache_reuse"] for a in anchors):
    raise AssertionError("m=1 did not exclusively reuse demographic fits")

records = [json.loads(p.read_text()) for p in sorted((OUT / "arms").glob("*.json"))]
if len(records) != 96:
    raise AssertionError(f"expected 96 dose-response arms, found {len(records)}")
if any(r["recomputed_fits"] != 0 or not r["demographic_fit_cache_only"]
       for r in records):
    raise AssertionError("a dose-response arm recomputed a demographic fit")

rows = []
for r in records:
    rows.append({
        "cell": r["cell"], "population": r["population"],
        "environment": r["environment"], "sigma_obs": r["sigma_obs"],
        "method": r["method"], "arm": r["arm"],
        "multiplier": r["multiplier"],
        "target_observation_scale": r["target_observation_scale"],
        "return_mean": r["summary"]["return_mean"],
        "return_sd": r["summary"]["return_sd"],
        "accepted_m1_return_mean": r["accepted_m1_return_mean"],
        "return_degradation_from_m1": r["return_degradation_from_m1"],
        "surrogate_holdout_R2": r["surrogate_holdout_R2"],
    })
with (OUT / "H12_dose_response.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

slopes = []
for cell in sorted({r["cell"] for r in rows}):
    for method in ("plus", "moor"):
        group = [r for r in rows if r["cell"] == cell and r["method"] == method
                 and r["arm"] != "kref"]
        x = np.asarray([float(r["multiplier"]) for r in group])
        y = np.asarray([float(r["return_mean"]) for r in group])
        slopes.append({
            "cell": cell, "method": method,
            "linear_slope_return_per_multiplier": float(np.polyfit(x, y, 1)[0]),
            "low_side_slope_0p5_to_0p85": float(np.polyfit(x[x < 1], y[x < 1], 1)[0]),
            "high_side_slope_1p2_to_3": float(np.polyfit(x[x > 1], y[x > 1], 1)[0]),
        })
with (OUT / "H12_degradation_slopes.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(slopes[0]))
    writer.writeheader()
    writer.writerows(slopes)

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
for ax, cell in zip(axes.flat, sorted({r["cell"] for r in rows})):
    for method in ("plus", "moor"):
        group = sorted(
            [r for r in rows if r["cell"] == cell and r["method"] == method
             and r["arm"] != "kref"],
            key=lambda r: float(r["multiplier"]))
        x = [float(r["multiplier"]) for r in group]
        y = [float(r["return_mean"]) for r in group]
        anchor = float(group[0]["accepted_m1_return_mean"])
        ax.plot(x[:3] + [1.0] + x[3:], y[:3] + [anchor] + y[3:],
                marker="o", label=method.upper())
    ax.axvline(1.0, color="black", lw=.7, ls="--")
    ax.set_title(cell)
    ax.set_xlabel("observation-scale multiplier")
    ax.set_ylabel("discounted return")
axes[0, 0].legend()
fig.tight_layout()
fig.savefig(OUT / "H12_return_vs_multiplier.png", dpi=180)
plt.close(fig)

receipt = {
    "schema": "H12_dose_response_receipt_v1",
    "m1_anchors": 12, "m1_parity_max_abs_diff": 0.0,
    "non_anchor_arms": 96, "total_arms": 108,
    "cells": 6, "methods": ["plus", "moor"],
    "multipliers": [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0,
                    "K_ref/original_observation_scale"],
    "recomputed_fits": 0, "public_surrogate_refit_each_nonanchor": True,
    "belief_and_surrogate_channels_perturbed_consistently": True,
    "belief_only_variant_run": False,
    "belief_only_variant_reason": (
        "The implementation exposes and rewires the two channels separately; "
        "the coupled requested arm is not an accidental unseparable conflation."),
    "constraint": "xenon-8452Y", "threads": 1,
    "accepted_artifacts_modified": False, "reranked": False,
    "jobs": {
        "m1_gate": "58579502",
        "initial_sweep_moor_outputs": "58579771",
        "plus_repair_array": "58588371",
    },
    "outputs": ["H12_dose_response.csv", "H12_degradation_slopes.csv",
                "H12_return_vs_multiplier.png"],
}
(OUT / "H12_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
