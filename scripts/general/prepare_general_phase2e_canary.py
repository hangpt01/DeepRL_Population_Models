#!/usr/bin/env python3
"""Freeze the authorized Phase 2E 4-row and 64-row canary packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hash_paper_faithful_snapshot import repeated_roots_hash, sha256_file, snapshot_tree_hash


DEFAULT_RUN = ROOT / "real_ecology_runs" / "general_phase2e_canary_20260720_v1"
OLD_MANIFEST = ROOT / "real_ecology_runs/general_corrected_prepared/manifest_general_hidden.csv"
CONFIG = "general_phase2e_canary.yaml"


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def copy_snapshot(destination: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache")
    for directory in ("src", "configs", "scripts", "tests", "real_ecology_data"):
        shutil.copytree(ROOT / directory, destination / directory, ignore=ignore)
    shutil.copy2(ROOT / "pyproject.toml", destination / "pyproject.toml")
    lock = ROOT / "docs/fix_implement_ecology_baseline/paper_faithful_fit_requirements.lock"
    (destination / "dependency").mkdir()
    shutil.copy2(lock, destination / "dependency" / lock.name)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def canonical_row(row: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in row.items() if key not in {"index", "package_role"}}


def scheduler_commands(run_root: Path, snapshot_hash: str) -> str:
    code = run_root / "code"
    manifest = run_root / "manifests/timing_preflight_4_rows.csv"
    output = run_root / "quarantine"
    exports = (
        f"ALL,ROOT={code},CONFIG={code / 'configs' / CONFIG},OUTPUT_ROOT={output},"
        f"GENERAL_CANARY_SNAPSHOT_SHA256={snapshot_hash},REGENERATE=false"
    )
    common = (
        "--test-only --cpus-per-task=1 --mem=8G --time=02:00:00 --no-requeue "
        f"--export={exports} {code / 'scripts/slurm/run_real_row.sh'} {manifest}"
    )
    lines = [
        "# Non-submitting probes only. Remove --test-only only under a future explicit authorization.",
        f"sbatch --partition=comp --qos=normal --array=0-3%1 {common}",
        f"sbatch --partition=comp --qos=normal --array=0-3%4 {common}",
        f"sbatch --partition=m3h --qos=m3h --array=0-3%1 {common}",
        f"sbatch --partition=m3h --qos=m3h --array=0-3%4 {common}",
        "",
        "# The 60-row remainder is intentionally not encoded for submission.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(DEFAULT_RUN))
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
    )
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    if run_root.exists():
        raise SystemExit(f"refusing to overwrite an existing run root: {run_root}")
    old_hash_before = sha256_file(OLD_MANIFEST)

    code_root = run_root / "code"
    code_root.mkdir(parents=True)
    copy_snapshot(code_root)
    for relative in ("manifests", "logs", "provenance", "validity", "quarantine/performance"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)

    manifests = {
        "timing_preflight": run_root / "manifests/timing_preflight_4_rows.csv",
        "limited_canary": run_root / "manifests/limited_canary_64_rows.csv",
    }
    expected = {"timing_preflight": 4, "limited_canary": 64}
    for mode, path in manifests.items():
        completed = subprocess.run(
            [
                args.python_bin,
                str(code_root / "scripts/make_general_phase2e_canary_manifests.py"),
                "--mode", mode,
                "--output", str(path),
            ],
            cwd=code_root,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(completed.stdout)
        if int(payload["rows"]) != expected[mode]:
            raise SystemExit(f"unexpected {mode} row count")

    timing_rows = read_rows(manifests["timing_preflight"])
    full_rows = read_rows(manifests["limited_canary"])
    subset = [row for row in full_rows if (
        row["population"] == "Egyptian vulture"
        and row["environment"] == "regime"
        and row["sigma_obs"] == "0.4"
    )]
    if [canonical_row(row) for row in timing_rows] != [canonical_row(row) for row in subset]:
        raise SystemExit("timing rows do not byte-match their limited-canary projections")

    live_hash, live_count = repeated_roots_hash([
        ROOT / "src", ROOT / "configs", ROOT / "scripts", ROOT / "tests"
    ])
    snapshot_hash, snapshot_count = repeated_roots_hash([
        code_root / "src", code_root / "configs", code_root / "scripts", code_root / "tests"
    ])
    if live_hash != snapshot_hash or live_count != snapshot_count:
        raise SystemExit("frozen code snapshot differs from live authorized sources")
    tree_hash, tree_count = snapshot_tree_hash(code_root)

    commands_path = run_root / "provenance/prepared_scheduler_commands.txt"
    commands_path.write_text(scheduler_commands(run_root, tree_hash), encoding="utf-8")
    source_status = command("git", "status", "--short")
    source_diff = command("git", "diff", "--binary", "HEAD")
    registration = {
        "schema": "general_phase2e_frozen_canary_v1",
        "created_date": "2026-07-20",
        "submitted": False,
        "authorized_to_submit": False,
        "source_commit": command("git", "rev-parse", "HEAD"),
        "source_dirty": bool(source_status),
        "source_diff_sha256": hashlib.sha256(source_diff.encode("utf-8")).hexdigest(),
        "live_source_hash": live_hash,
        "live_source_file_count": live_count,
        "snapshot_source_hash": snapshot_hash,
        "snapshot_source_file_count": snapshot_count,
        "snapshot_tree_sha256": tree_hash,
        "snapshot_tree_file_count": tree_count,
        "snapshot_hash_recipe": "scripts/hash_paper_faithful_snapshot.py",
        "config_path": f"code/configs/{CONFIG}",
        "config_sha256": sha256_file(code_root / "configs" / CONFIG),
        "manifest_paths": {key: str(path.relative_to(run_root)) for key, path in manifests.items()},
        "manifest_rows": expected,
        "manifest_sha256": {key: sha256_file(path) for key, path in manifests.items()},
        "timing_rows_match_limited_projection": True,
        "methods": [
            "refplan", "ogsrl", "bamcts", "ensemble_value_disagreement_pessimism"
        ],
        "scenarios": 16,
        "collection_seed": 116,
        "transitions": 4000,
        "episode_length": 25,
        "evaluation_seed": 9001,
        "evaluation_episodes": 1,
        "evaluation_horizon": 50,
        "ogsrl_estimand": (
            "E_belief,bootstrap-member,public-predictive-residual,actor[C_25]"
        ),
        "ogsrl_deployment_rollouts": 256,
        "q_disagreement_penalty": 0.1,
        "q_penalty_influence": "operational but sparse; mechanism evidence only",
        "one_cpu_per_task": True,
        "thread_count": 1,
        "scheduler_commands_path": str(commands_path.relative_to(run_root)),
        "scheduler_commands_sha256": sha256_file(commands_path),
        "automatic_requeue": False,
        "automatic_retry": False,
        "remainder_automatic_submission": False,
        "outcome_files_quarantined": True,
        "acceptance_reader": "code/scripts/check_general_phase2e_canary.py",
        "old_1152_manifest_path": str(OLD_MANIFEST.relative_to(ROOT)),
        "old_1152_manifest_sha256": old_hash_before,
        "old_1152_manifest_modified": False,
        "dependency_lock_sha256": sha256_file(
            code_root / "dependency/paper_faithful_fit_requirements.lock"
        ),
        "python_executable": str(Path(args.python_bin).resolve()),
        "python_version": subprocess.check_output(
            [args.python_bin, "-c", "import sys; print(sys.version)"], text=True
        ).strip(),
    }
    registration_path = run_root / "manifests/registration.json"
    registration_path.write_text(json.dumps(registration, indent=2, sort_keys=True), encoding="utf-8")
    if sha256_file(OLD_MANIFEST) != old_hash_before:
        raise SystemExit("old 1,152-row manifest changed while preparing Phase 2E")
    print(json.dumps({
        "run_root": str(run_root),
        "registration_sha256": sha256_file(registration_path),
        **registration,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
