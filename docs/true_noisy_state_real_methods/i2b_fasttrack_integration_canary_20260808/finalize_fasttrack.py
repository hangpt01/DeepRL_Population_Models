#!/usr/bin/env python3
"""Finalize reports only after all twelve successful parity receipts exist."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
OUT = ROOT / "outputs/i2b_fasttrack_integration_canary_20260808"
REG = json.loads((DOC / "ARM_O_CANARY_REGISTRATION.json").read_text(encoding="utf-8"))
GENERAL = {"refplan", "ogsrl", "bamcts", "ensemble_value_disagreement_pessimism"}


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(text, encoding="utf-8")


def receipt_path(task: dict) -> Path:
    if task["method"] in GENERAL:
        return OUT / "arm_o_tasks" / task["task_id"] / "task_receipt.json"
    return OUT / "ecological_retry1/arm_o_tasks" / task["task_id"] / "task_receipt.json"


rows = []
for task in REG["tasks"]:
    path = receipt_path(task)
    if not path.exists():
        raise RuntimeError(f"missing terminal receipt: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    parity = receipt.get("parity") or {}
    if receipt.get("exit_status") != 0 or parity.get("result") != "PASS":
        raise RuntimeError(f"nonpassing receipt: {path}")
    scientific = {
        key: value for key, value in parity["maximum_absolute_deltas"].items()
        if not key.startswith("reconstructed_")
    }
    reconstructed = {
        key: value for key, value in parity["maximum_absolute_deltas"].items()
        if key.startswith("reconstructed_")
    }
    rows.append({
        "task_id": task["task_id"], "cell": task["cell"], "method": task["method"],
        "activity_flag": task["accepted_activity"], "receipt_path": str(path),
        "slurm_job_id": receipt.get("slurm_job_id"), "cpu_model": receipt.get("cpu_model"),
        "duration_seconds": receipt.get("duration_seconds"),
        "accepted_episodes_sha256": task["accepted_episodes_sha256"],
        "new_episodes_sha256": receipt["episodes_sha256"],
        "max_scientific_delta": max(scientific.values(), default=0.0),
        "max_return_reconstruction_delta": max(reconstructed.values(), default=0.0),
        "exact_mismatch_count": len(parity["exact_mismatches"]),
        "historical_full_action_sequences_available": False,
        "new_action_sequences_path": receipt["deployed_trajectories_path"],
        "result": "PASS",
    })

initial_failures = []
for task in REG["tasks"]:
    if task["method"] in GENERAL:
        continue
    path = OUT / "arm_o_tasks" / task["task_id"] / "task_receipt.json"
    failure = json.loads(path.read_text(encoding="utf-8"))
    initial_failures.append({
        "task_id": task["task_id"], "receipt_path": str(path),
        "exit_status": failure.get("exit_status"), "failure": failure.get("failure"),
        "classification": "technical interpreter mismatch; no returns/parity opened",
    })

report = {
    "schema_version": "i2b_arm_o_parity_report_v1",
    "overall_result": "PASS",
    "task_count": 12,
    "passed_tasks": 12,
    "failed_parity_tasks": 0,
    "absolute_tolerance": REG["parity"]["absolute_tolerance"],
    "rows": rows,
    "initial_ecological_technical_attempt": initial_failures,
    "technical_retry": {
        "job_id": next(row["slurm_job_id"] for row in rows if row["method"] == "plus_adapted_ricker_only_pbvi"),
        "interpreter": "/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python",
        "source_evidence": "accepted launch_package/slurm/run_plus_p10.sh and run_moor_p10.sh",
        "overwrote_initial_receipts": False,
    },
    "historical_action_sequence_limitation": "accepted artifacts do not store literal full sequences; every stored action-derived field was compared and new full sequences were sealed",
    "arm_t_real_runs": 0,
    "arm_t_returns_opened": 0,
}
write_new(DOC / "ARM_O_PARITY_REPORT.json", json.dumps(report, indent=2, sort_keys=True) + "\n")

table = [
    "# Arm O canary parity report", "",
    "All twelve registered Arm O method/cell rows pass the accepted 8452Y parity rule.", "",
    "| Cell | Method | Max scientific delta | Max reconstruction delta | Activity flag | Result |",
    "|---|---|---:|---:|---|---|",
]
for row in rows:
    table.append(
        f"| {row['cell']} | `{row['method']}` | {row['max_scientific_delta']:.17g} | "
        f"{row['max_return_reconstruction_delta']:.17g} | {row['activity_flag']} | PASS |"
    )
table.extend([
    "", "The initial four ecological tasks failed before fitting/evaluation because `/usr/bin/python` lacked PyTorch. Their receipts were preserved. A separate non-overwriting retry used the exact interpreter pinned by the accepted PLUS/MOOR Slurm launchers; all four retry rows passed.",
    "", "Historical artifacts contain per-episode returns, event metrics, action entropy/cost and per-action danger fractions, but not literal full action sequences. Every available stored action-derived field was compared. The canary's full deployed sequences are sealed separately; no unavailable accepted hash was fabricated.",
    "", "No derived truth artifact was available to Arm O. No real Arm T policy ran and no Arm T return was opened.", "",
])
write_new(DOC / "ARM_O_PARITY_REPORT.md", "\n".join(table))

completion = f"""# I2B Fast-Track Completion Report

## Executive result

The source-backed interface map, external capability-gated wrappers, two sealed offline
allowlisted derivatives, 53 synthetic tests, and all twelve pinned-CPU Arm O canaries are
complete. Every final Arm O receipt passes the registered `1e-9` rule. Maximum scientific
delta across the twelve rows is `{max(r['max_scientific_delta'] for r in rows):.17g}`;
maximum return-reconstruction delta is
`{max(r['max_return_reconstruction_delta'] for r in rows):.17g}`.

## Integrity and information boundary

- Revision 3.1, I1/addendum/audits, I2A Revision 1 and both source-only hashes were
  verified before writing.
- The truth extractor materialized only binding `metadata_json`, `states`, and offline
  `next_states`; forbidden truth payload reads were zero.
- The two derivatives are each 4,000 rows, 160 × 25, raw abundance, exact-aligned and
  sealed. No runtime future-state file exists.
- Arm O used byte-identical task-local public inputs and a non-truth sentinel. Private
  loading failed closed; original `truth.npz` and derived Arm T files were inaccessible.
- Arm T remained capability-disabled for real evaluation. No real Arm T policy or return
  exists in this package.
- The 38 controlling I2A tests and 15 new integration tests passed without repository
  bytecode/cache output.

## Canary execution

Initial array job `58859435` ran all twelve registered Arm O indices. Four ecological
indices failed before fitting because the generic `/usr/bin/python` lacked PyTorch; no
scientific return or parity result was opened for those attempts. The accepted launch
package source-pins `.venv-paper-faithful/bin/python`, so non-overwriting retry job
`58859482` used that exact environment only for indices 0, 1, 6 and 7. All twelve final
method/cell receipts pass. The failed technical receipts remain preserved.

The accepted package did not store literal action sequences. The parity gate therefore
used every strongest available accepted field: ordered identities, per-episode returns,
action entropy/cost, per-action danger fractions, event counts and all other scientific
columns. New full action/reward sequences were sealed and independently reconstruct both
returns within the registered tolerance.

## Method and interpretation status

The primary future contrast remains end-to-end. Any fitted-artifact or `residual_sigma`
change forces `MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY`. Optional frozen-fit
eligibility remains false until all applicable identities are concrete and equal. EVD
retains separate raw-reward reporting. Constant tiger PLUS/MOOR policies remain flagged
non-discriminating; this report makes no scientific ordering interpretation.

## Authorization status

This canary package is complete and stops here. No source/frozen/accepted file was
modified, no accepted result was rebaselined, no sigma 0.1/other cell ran, no Stage B
scientific interpretation was made, and Stage C remains unauthorized. The next permitted
action is one Claude pre-run audit of this package.

PASS — FAST-TRACK INTEGRATION AND ARM O CANARY COMPLETE; READY FOR ONE CLAUDE PRE-RUN AUDIT
"""
write_new(DOC / "FASTTRACK_COMPLETION_REPORT.md", completion)

# Cover every file in both new namespaces. The manifest covers itself by hashing
# a normalized form in which the recorded digest is replaced by 64 zeroes.
manifest_path = DOC / "FASTTRACK_HASHES.sha256"
files = sorted(
    [path for base in (DOC, OUT) for path in base.rglob("*") if path.is_file() and path != manifest_path],
    key=lambda path: str(path.relative_to(ROOT)),
)
lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}" for path in files]
zero = "0" * 64
self_rel = manifest_path.relative_to(ROOT)
normalized = "\n".join(lines + [f"# SELF-NORMALIZED-SHA256: {zero}  {self_rel}"]) + "\n"
self_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
write_new(manifest_path, normalized.replace(zero, self_digest, 1))
print(json.dumps({"tasks": 12, "result": "PASS", "files_covered": len(files) + 1, "normalized_self_sha256": self_digest}, indent=2))
