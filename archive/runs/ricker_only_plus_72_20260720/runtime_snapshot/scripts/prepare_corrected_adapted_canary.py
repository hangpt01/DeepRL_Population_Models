#!/usr/bin/env python3
"""Freeze the registered corrected adapted PLUS/MOOR canary snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from hash_paper_faithful_snapshot import (
    repeated_roots_hash,
    sha256_file,
    snapshot_tree_hash,
)
from prepare_paper_faithful_smoke import command, copy_snapshot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "real_ecology_runs" / "adapted_plus_moor_corrected_canary_20260717_v1"
CONFIG_NAME = "paper_faithful_hidden_one_cell_4000.yaml"
PRE_TOOLING_DIGEST = "8b4c7c44f23b1622e70b580b77cfec9f22e49abb3596690e54010d73df9e56db"
MANIFESTS = {
    "canary_fit": "canary_fit.csv",
    "canary_plan": "canary_plan.csv",
}
METHOD_IMPL_VERSIONS = {
    "plus_adapted_mechanistic_pbvi": "plus_adapted_mechanistic_fixed_pi_pbvi_v2",
    "moor_adapted_ricker_misspec_pbvi": "moor_adapted_ricker_misspec_pbvi_v2",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(DEFAULT_RUN))
    parser.add_argument(
        "--python-bin",
        default=str(ROOT / ".venv-paper-faithful/bin/python"),
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    if run_root.exists():
        raise SystemExit(f"refusing to overwrite an existing run root: {run_root}")

    code_root = run_root / "code"
    code_root.mkdir(parents=True)
    copy_snapshot(code_root)
    (run_root / "manifests").mkdir()
    (run_root / "logs").mkdir()
    (run_root / "provenance").mkdir()

    manifest_hashes: dict[str, str] = {}
    manifest_rows: dict[str, int] = {}
    for mode, filename in MANIFESTS.items():
        manifest_path = run_root / "manifests" / filename
        completed = subprocess.run(
            [
                args.python_bin,
                str(code_root / "scripts/make_paper_faithful_manifest.py"),
                "--mode",
                mode,
                "--output",
                str(manifest_path),
            ],
            check=True,
            cwd=code_root,
            text=True,
            capture_output=True,
        )
        payload = json.loads(completed.stdout)
        manifest_rows[mode] = int(payload["rows"])
        manifest_hashes[mode] = sha256_file(manifest_path)

    subprocess.run(
        [
            args.python_bin,
            str(code_root / "scripts/run_paper_faithful_solver_probe.py"),
            "--output",
            str(run_root / "provenance/solver_probe.json"),
        ],
        check=True,
        cwd=code_root,
    )

    git_status = command("git", "status", "--short")
    diff = command("git", "diff", "--binary", "HEAD")
    live_digest, live_file_count = repeated_roots_hash(
        [ROOT / "src", ROOT / "configs", ROOT / "scripts"]
    )
    snapshot_digest, snapshot_file_count = repeated_roots_hash(
        [code_root / "src", code_root / "configs", code_root / "scripts"]
    )
    registration = {
        "run_kind": "registered_return_blind_corrected_adapted_plus_moor_canary",
        "created_date": "2026-07-17",
        "source_commit": command("git", "rev-parse", "HEAD"),
        "source_dirty": bool(git_status),
        "source_status": git_status.splitlines(),
        "source_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "pre_canary_tooling_source_config_script_sha256": PRE_TOOLING_DIGEST,
        "live_source_config_script_sha256": live_digest,
        "live_source_config_script_file_count": live_file_count,
        "snapshot_source_config_script_sha256": snapshot_digest,
        "snapshot_source_config_script_file_count": snapshot_file_count,
        "snapshot_tree_sha256": snapshot_tree_hash(code_root)[0],
        "snapshot_tree_hash_recipe": "scripts/hash_paper_faithful_snapshot.py",
        "config_path": f"configs/{CONFIG_NAME}",
        "config_sha256": sha256_file(code_root / "configs" / CONFIG_NAME),
        "manifest_paths": {
            mode: f"manifests/{filename}" for mode, filename in MANIFESTS.items()
        },
        "manifest_sha256": manifest_hashes,
        "manifest_rows": manifest_rows,
        "dependency_lock_sha256": sha256_file(
            code_root / "dependency/paper_faithful_fit_requirements.lock"
        ),
        "pyproject_sha256": sha256_file(code_root / "pyproject.toml"),
        "real_ecology_data_tree_sha256": snapshot_tree_hash(
            code_root / "real_ecology_data"
        )[0],
        "python_executable": str(Path(args.python_bin).absolute()),
        "python_version": subprocess.check_output(
            [args.python_bin, "-c", "import sys; print(sys.version)"], text=True
        ).strip(),
        "methods": list(METHOD_IMPL_VERSIONS),
        "method_impl_versions": METHOD_IMPL_VERSIONS,
        "fit_stage_rows": 4,
        "plan_stage_rows": 8,
        "fit_stage_runner": "scripts/run_adapted_fit_row.py",
        "plan_stage_runner": "scripts/run_real_manifest_row.py",
        "fit_receipts_required_for_plan": True,
        "first_stage_cpu_ceiling_hours": 24,
        "cumulative_cpu_ceiling_hours": 48,
        "headline_sweep_authorized": False,
        "diagnostic_subset_authorized": False,
        "sensitivity_suite_authorized": False,
        "return_blinding": (
            "fit stage opens no planner/evaluator/return/action-quality fields; "
            "plan acceptance opens timing, convergence, cache, hash, resource, "
            "and completeness fields only"
        ),
        "preserved_void_run_modified": False,
    }
    (run_root / "manifests/registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"run_root": str(run_root), **registration}, indent=2))


if __name__ == "__main__":
    main()
