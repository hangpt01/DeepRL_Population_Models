#!/usr/bin/env python3
"""Write the combined receipt only after all five experiment gates pass."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIAGNOSTICS_DIR))
from repo_paths import ACCEPTED_CSV, FOLLOWUPS_OUTPUT, REPO_ROOT  # noqa: E402

ROOT = Path(os.environ.get("DEEPRL_FOLLOWUPS_OUTPUT", FOLLOWUPS_OUTPUT))
REPO = REPO_ROOT
ACCEPTED = ACCEPTED_CSV
EXPECTED_SHA = "7431318803e468c13ff3008acdc530c0d9f9a919cc29a6fe04951fc8a2cadc1c"


def read(relative):
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(f"NOT FOUND: {path}")
    return json.loads(path.read_text())


actual_sha = hashlib.sha256(ACCEPTED.read_bytes()).hexdigest()
if actual_sha != EXPECTED_SHA:
    raise AssertionError("accepted controlling CSV changed")

s2 = read("S2_out/S2_RECEIPT.json")
s2_convergence = read("S2_out/S2_GRID_CONVERGENCE_RECEIPT.json")
job2 = read("Job2_reward_screen_out/JOB2_RECEIPT.json")
h14 = read("H14_out/H14_RECEIPT.json")
s6 = read("S6_out/S6_RECEIPT.json")
h12 = read("H12_out/H12_RECEIPT.json")

if not all(s2["anchors"].values()):
    raise AssertionError("S2 anchor did not pass")
if not s2_convergence["S2_decided"]:
    raise AssertionError("S2 grid convergence remains undecided")
if not job2["current_parity_pass"] or job2["option_C_any_argmax_move"]:
    raise AssertionError("reward-screen control gate failed")
if h14["parity_max_abs_diff"] != 0.0 or h14["recomputed_fits"] != 0:
    raise AssertionError("H14 gate failed")
if not s6["fits_from_scratch"] or s6["recomputed_fits"] != 9:
    raise AssertionError("S6 fresh-fit gate failed")
if h12["m1_parity_max_abs_diff"] != 0.0 or h12["recomputed_fits"] != 0:
    raise AssertionError("H12 gate failed")

receipt = {
    "schema": "three_species_diagnostic_followups_combined_v1",
    "stage": "COMPLETE", "jobs_passed": 5,
    "accepted_csv": str(ACCEPTED.relative_to(REPO)),
    "accepted_csv_sha256": actual_sha,
    "accepted_artifacts_modified": False,
    "accepted_values_used": "parity reconstruction/cell identity only where authorized",
    "reranked": False, "constant_policies": "evaluator-only references",
    "method_reward_hyperparameter_changes": False,
    "recomputed_fits": {"total": 9, "job4_only": True},
    "jobs": {
        "1_S2": {
            "status": "PASS",
            "receipt": "S2_out/S2_RECEIPT.json",
            "grid_convergence_receipt":
                "S2_out/S2_GRID_CONVERGENCE_RECEIPT.json",
        },
        "2_reward_screen": {
            "status": "PASS",
            "receipt": "Job2_reward_screen_out/JOB2_RECEIPT.json",
        },
        "3_H14": {
            "status": "PASS", "receipt": "H14_out/H14_RECEIPT.json",
            "slurm": ["58579344", "58579415"],
        },
        "4_S6": {
            "status": "PASS", "receipt": "S6_out/S6_RECEIPT.json",
            "slurm": {
                "plus_salvaged": "58579501",
                "moor_resume": "58588290",
            },
        },
        "5_H12": {
            "status": "PASS", "receipt": "H12_out/H12_RECEIPT.json",
            "slurm_m1": "58579502",
            "slurm_initial_moor": h12["jobs"].get("initial_sweep_moor_outputs"),
            "slurm_plus_repair": h12["jobs"].get("plus_repair_array"),
        },
    },
    "excluded_launcher_failures": {
        "58579358": "wrong interpreter; no scientific output",
        "58579373": "wrong interpreter; no scientific output",
    },
}
(ROOT / "COMBINED_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")

summary = f"""# Three-species diagnostic follow-ups

All five authorized follow-up experiments passed their construction and
instrumentation gates. They supersede and re-rank nothing. The accepted P=10
table remains unchanged at SHA-256 `{actual_sha}`.

- S2: evaluator-only cross-family identifiability, regret, and VPI.
- Reward screen: exact constant-policy trajectory reconstruction before Option A/C.
- H14: 9/9 general-method arms at exact seven-field parity, with visited-state M13.
- S6: one new A1 dataset and fresh PLUS/MOOR demographic fits (`recomputed_fits=9`).
- H12: 12/12 m=1 anchors at exact parity, followed by 96 cache-only replanning arms.

Constant-action policies remain evaluator-only references. No accepted artifact,
method ranking, demographic model definition, reward used by accepted methods, or
accepted hyperparameter was changed.
"""
(ROOT / "SUMMARY.md").write_text(summary)
print(json.dumps(receipt, indent=2))
