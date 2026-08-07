#!/usr/bin/env python3
"""No-interpretation finalization and audit sealing for tiger E1 grid-161."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path("/fs04/scratch2/ce25/DeepRL_Population_Models")
SCRIPT_DIR = REPO / "scripts" / "diagnostics"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e1_tiger_grid161_run as tiger  # noqa: E402

runner = tiger.base


def all_hashes(run_dir: Path, exclusions: set[Path]) -> dict[str, str]:
    rows = {}
    resolved_exclusions = {path.resolve() for path in exclusions}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.resolve() in resolved_exclusions:
            continue
        if ".tmp." in path.name or path.name.startswith("."):
            raise AssertionError(f"temporary/incomplete file present: {path}")
        rows[str(path.resolve())] = runner.sha256_file(path)
    return rows


def live_finalizer_logs(run_dir: Path, final_submission: dict) -> set[Path]:
    job_id = str(final_submission["finalizer_job_id"])
    return {
        run_dir / "logs" / f"finalizer-{job_id}.out",
        run_dir / "logs" / f"finalizer-{job_id}.err",
    }


def claude_prompt(run_dir: Path, registration_hash: str) -> str:
    return f"""Perform an independent READ-ONLY audit of the completed prospective E1 tiger-only grid-161 replication.

Repository: {REPO}
Audit manifest: {run_dir / 'AUDIT_MANIFEST.json'}
Run receipt: {run_dir / 'RUN_RECEIPT.json'}
Registration hash: {registration_hash}

Fox E1 results were known before this tiger extension. Do not inspect fox returns, pool species, modify files, rerun the experiment, aggregate contrasts, or interpret which tiger arm wins.

1. Verify AUDIT_MANIFEST.json, its sidecar, and every listed SHA-256. Independently hash the two live-finalizer logs that were deliberately excluded while open.
2. Verify the immutable tiger registration predates every tiger episode and contains the complete authorising command, no directional tiger prediction, replication/control status, and no-pooling rule.
3. Verify the canonical species identifier is Amur tiger and independently reconstruct N0, K_base, K_max, K_ref, s_safe, action table/costs, registered model, fitted artifact, survey_scale and reset conventions for both cells.
4. Hard fail if fox K_ref=41 or s_safe=10.25 occurs in any tiger environment, policy, reward, episode, task receipt or canonical configuration.
5. Independently reconstruct policy/model/grid/reward/action/code hashes for all eight tasks. Verify state_bins=161, PBVI horizon=5, belief_points=32, seven noisy branches, exact true-state predictive support, gamma=0.95, occupancy penalty, alpha=1 and P=10.
6. Independently reproduce at least one tiger episode per arm/cell from its seed, action sequence, trajectories, explicit true-state reward route and discounted return without saving a replacement.
7. Verify all twenty pairing keys across A1-A4 within each tiger cell, identical paired environmental stream descriptors, and planning RNG independence.
8. Verify Python 3.10.14, NumPy 2.2.6, comp/EPYC9534 for every task, one architecture, zero refits, and A2/A4 fitted-artifact identity within each cell.
9. Verify repaired runtime/lookahead filter counters, zero-evidence/fallback fields, append-only checkpoints, scheduler mapping, no output collision, and that no completed episode was overwritten.
10. Verify src/tracks/**, controlling documents, and outputs/e1_fox_only_grid161/** were unchanged by the tiger run.
11. Confirm no tiger statistical aggregation or scientific interpretation exists.

Report every mismatch before aggregation. Give exactly PASS — SAFE TO AGGREGATE only if every scientific and provenance check independently passes.
"""


def finalize_core(run_dir: Path) -> dict:
    tiger.assert_registered_code(run_dir)
    control = runner.load_json(run_dir / runner.RUN_CONTROL_NAME)
    registration = runner.load_json(Path(control["registration_path"]))
    submission = runner.load_json(run_dir / "SUBMISSION.json")
    final_submission = runner.load_json(run_dir / "FINALIZATION_SUBMISSION.json")
    verification = runner.verify(run_dir, write_outputs=True)
    scheduler = runner.scheduler_status(submission, run_dir / "logs")
    git_after = runner.git_snapshot()
    git_before = registration["git_status_before"]
    if git_after["tracked_src_tracks_digest"] != git_before["tracked_src_tracks_digest"]:
        raise AssertionError("src/tracks tracked digest changed")
    if git_after["src_tracks_git_diff"]:
        raise AssertionError("src/tracks git diff is non-empty")
    if not all(value == 20 for value in verification["actual_episode_count_per_arm"].values()):
        raise AssertionError("episode count failure")
    if set(verification["actual_episode_count_per_arm"]) != {
        f"{cell}/{arm}" for cell, arm in tiger.TASKS
    }:
        raise AssertionError("non-tiger or missing task receipt")
    reproduce = run_dir / "REPRODUCE.md"
    runner.write_text_new(reproduce, tiger.reproduce_markdown(run_dir, control))
    claude = run_dir / "CLAUDE_READ_ONLY_AUDIT_PROMPT.md"
    runner.write_text_new(claude, claude_prompt(run_dir, control["registration_sha256"]))
    live_logs = live_finalizer_logs(run_dir, final_submission)
    exclusions = {
        run_dir / "AUDIT_MANIFEST.json", run_dir / "AUDIT_MANIFEST.sha256",
        run_dir / "RUN_RECEIPT.json", *live_logs,
    }
    scientific_hashes = all_hashes(run_dir, exclusions)
    receipt = {
        "schema": f"{runner.SCHEMA}_run_receipt",
        "status": "COMPLETE",
        "created_utc": runner.utc_now(),
        "result_directory": str(run_dir.resolve()),
        "registration_path": control["registration_path"],
        "registration_sha256": control["registration_sha256"],
        "canonical_configuration": registration["canonical_configuration"],
        "registered_deviation": tiger.REGISTERED_DEVIATION,
        "fox_results_known_before_tiger_registration": True,
        "fox_tiger_pooling_permitted": False,
        "task_completion": {key: "COMPLETE" for key in verification["task_receipts"]},
        "episode_counts": verification["actual_episode_count_per_arm"],
        "pairing_verification": verification["pairing"],
        "aggregate_filter_counters": verification["aggregate_counters"],
        "scientific_and_supporting_file_hashes": scientific_hashes,
        "live_finalizer_logs_excluded_while_open": sorted(str(path.resolve()) for path in live_logs),
        "parent_scheduler": scheduler,
        "finalizer_submission": final_submission,
        "git_status_before": git_before,
        "git_status_after": git_after,
        "src_tracks_unchanged": True,
        "tiger_returns_interpreted_compared_or_statistically_analysed": False,
        "deviations_or_failures": [],
        "no_completed_result_file_overwritten": True,
    }
    path = run_dir / "RUN_RECEIPT.json"
    runner.write_json_new(path, receipt)
    output = {
        "status": "COMPLETE",
        "run_receipt": str(path.resolve()),
        "run_receipt_sha256": runner.sha256_file(path),
        "registration_sha256": control["registration_sha256"],
        "eight_tasks_complete": verification["all_eight_tasks_complete"],
        "episode_counts": verification["actual_episode_count_per_arm"],
        "pairing_verified": True,
        "statistical_analysis_run": False,
        "species_pooling_run": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def seal_audit(run_dir: Path) -> dict:
    tiger.assert_registered_code(run_dir)
    control = runner.load_json(run_dir / runner.RUN_CONTROL_NAME)
    registration = runner.load_json(Path(control["registration_path"]))
    verification = runner.load_json(run_dir / "logs" / "final_completeness_check.json")
    final_submission = runner.load_json(run_dir / "FINALIZATION_SUBMISSION.json")
    run_receipt = run_dir / "RUN_RECEIPT.json"
    if not run_receipt.exists():
        raise AssertionError("run receipt missing")
    git_after = runner.git_snapshot()
    before = registration["git_status_before"]
    if git_after["tracked_src_tracks_digest"] != before["tracked_src_tracks_digest"]:
        raise AssertionError("src/tracks changed before audit seal")
    audit_path = run_dir / "AUDIT_MANIFEST.json"
    sidecar = run_dir / "AUDIT_MANIFEST.sha256"
    live_logs = live_finalizer_logs(run_dir, final_submission)
    output_hashes = all_hashes(run_dir, {audit_path, sidecar, *live_logs})
    task_mapping, task_hashes, history = {}, {}, {}
    for key, receipt_path_text in verification["task_receipts"].items():
        receipt_path = Path(receipt_path_text)
        receipt = runner.load_json(receipt_path)
        task_mapping[key] = {
            "job_id": receipt["scheduler"]["job_id"],
            "array_job_id": receipt["scheduler"]["array_job_id"],
            "array_task_id": receipt["scheduler"]["array_task_id"],
            "cell": receipt["cell"], "arm": receipt["arm"],
        }
        task_hashes[key] = receipt["hashes"]
        history[key] = {
            "checkpoints": sorted(str(path.resolve()) for path in (receipt_path.parent / "checkpoints").glob("*.json")),
            "resume_events": sorted(str(path.resolve()) for path in (receipt_path.parent / "resume_events").glob("*.json")),
        }
    manifest = {
        "schema": f"{runner.SCHEMA}_audit_manifest",
        "created_utc": runner.utc_now(),
        "absolute_result_directory": str(run_dir.resolve()),
        "preregistration": {
            "path": control["registration_path"],
            "timestamp": registration["timestamp_utc"],
            "sha256": control["registration_sha256"],
        },
        "fox_results_known_before_tiger_registration": True,
        "fox_tiger_pooling_permitted": False,
        "repository_commit": registration["repository_commit"],
        "git_status_before": before,
        "git_status_after": git_after,
        "complete_modified_and_untracked_files_before": before["modified_or_untracked_files"],
        "complete_modified_and_untracked_files_after": git_after["modified_or_untracked_files"],
        "src_tracks_unchanged": {
            "confirmed": True,
            "before_digest": before["tracked_src_tracks_digest"],
            "after_digest": git_after["tracked_src_tracks_digest"],
            "git_diff": git_after["src_tracks_git_diff"],
        },
        "environment": registration["environment"],
        "environment_digest": registration["environment_digest"],
        "cpu_architecture": verification["cpu_models"],
        "EPYC9534_confirmation_for_every_task": True,
        "canonical_full_experiment_configuration": registration["canonical_configuration"],
        "planning_seeds": registration["planning_seeds"],
        "evaluation_seeds": registration["evaluation_seeds"],
        "twenty_derived_episode_identities": registration["derived_episode_identities"],
        "slurm_job_to_cell_arm": task_mapping,
        "exact_task_commands": registration["exact_task_commands"],
        "complete_submission_command": registration["complete_submission_command"],
        "finalization_submission": final_submission,
        "task_hashes": task_hashes,
        "expected_episode_count_per_arm": 20,
        "actual_episode_count_per_arm": verification["actual_episode_count_per_arm"],
        "pairing_verification": verification["pairing"],
        "every_output_file_path_and_sha256": output_hashes,
        "live_finalizer_logs_excluded_while_open": sorted(str(path.resolve()) for path in live_logs),
        "checkpoint_resume_history": history,
        "filter_counters": verification["aggregate_counters"],
        "warnings_failures_deviations": [],
        "no_file_was_overwritten": True,
        "run_receipt_path": str(run_receipt.resolve()),
        "run_receipt_sha256": runner.sha256_file(run_receipt),
        "reproduce_path": str((run_dir / "REPRODUCE.md").resolve()),
        "claude_read_only_audit_prompt": str((run_dir / "CLAUDE_READ_ONLY_AUDIT_PROMPT.md").resolve()),
        "tiger_statistical_analysis_or_scientific_interpretation_run": False,
        "species_pooling_run": False,
        "audit_self_hash_excluded_by_definition": True,
    }
    runner.write_json_new(audit_path, manifest)
    digest = runner.sha256_file(audit_path)
    runner.write_text_new(sidecar, f"{digest}  {audit_path.name}\n")
    return {"status": "SEALED", "audit_manifest_sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("finalize", "seal"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "finalize":
        finalize_core(args.run_dir.resolve())
    else:
        seal_audit(args.run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
