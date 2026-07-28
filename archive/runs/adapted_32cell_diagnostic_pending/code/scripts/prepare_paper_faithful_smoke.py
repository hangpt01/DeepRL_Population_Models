#!/usr/bin/env python3
"""Freeze a provenance-complete code snapshot for the registered faithful smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from hash_paper_faithful_snapshot import sha256_file, snapshot_tree_hash


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "real_ecology_runs" / "paper_faithful_smoke_20260717"
MODE_CONFIG = {
    "smoke": "paper_faithful_hidden_smoke.yaml",
    "one_cell_4000": "paper_faithful_hidden_one_cell_4000.yaml",
}


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def copy_snapshot(destination: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    for directory in ("src", "configs", "scripts", "real_ecology_data"):
        shutil.copytree(ROOT / directory, destination / directory, ignore=ignore)
    shutil.copy2(ROOT / "pyproject.toml", destination / "pyproject.toml")
    lock = ROOT / "docs/fix_implement_ecology_baseline/paper_faithful_fit_requirements.lock"
    (destination / "dependency").mkdir(parents=True)
    shutil.copy2(lock, destination / "dependency" / lock.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(DEFAULT_RUN))
    parser.add_argument("--mode", choices=tuple(MODE_CONFIG), default="smoke")
    parser.add_argument(
        "--python-bin",
        default=str(ROOT / ".venv-paper-faithful/bin/python"),
    )
    args = parser.parse_args()
    config_name = MODE_CONFIG[args.mode]
    manifest_name = f"{args.mode}.csv"
    target_transitions = 160 if args.mode == "smoke" else 4000
    run_root = Path(args.run_root).resolve()
    if run_root.exists():
        raise SystemExit(f"refusing to overwrite an existing run root: {run_root}")
    code_root = run_root / "code"
    code_root.mkdir(parents=True)
    copy_snapshot(code_root)
    (run_root / "manifests").mkdir()
    (run_root / "logs").mkdir()
    (run_root / "provenance").mkdir()
    subprocess.run(
        [
            args.python_bin,
            str(code_root / "scripts/make_paper_faithful_manifest.py"),
            "--mode",
            args.mode,
            "--output",
            str(run_root / "manifests" / manifest_name),
        ],
        check=True,
        cwd=code_root,
    )
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
    registration = {
        "run_kind": (
            "registered_interface_smoke_not_headline_experiment"
            if args.mode == "smoke"
            else "registered_4000_target_one_cell_smoke_not_headline_experiment"
        ),
        "created_date": "2026-07-17",
        "source_commit": command("git", "rev-parse", "HEAD"),
        "source_dirty": bool(git_status),
        "source_status": git_status.splitlines(),
        "source_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "snapshot_tree_sha256": snapshot_tree_hash(code_root)[0],
        "snapshot_tree_hash_recipe": "scripts/hash_paper_faithful_snapshot.py",
        "config_path": f"configs/{config_name}",
        "config_sha256": sha256_file(code_root / "configs" / config_name),
        "manifest_path": f"manifests/{manifest_name}",
        "manifest_sha256": sha256_file(run_root / "manifests" / manifest_name),
        "dependency_lock_sha256": sha256_file(
            code_root / "dependency/paper_faithful_fit_requirements.lock"
        ),
        "python_executable": str(Path(args.python_bin).absolute()),
        "python_version": subprocess.check_output(
            [args.python_bin, "-c", "import sys; print(sys.version)"], text=True
        ).strip(),
        "methods": [
            "plus_adapted_mechanistic_pbvi",
            "moor_adapted_ricker_misspec_pbvi",
        ],
        "rows": 2,
        "expose_rk": "hidden",
        "target_transitions": target_transitions,
        "plus_candidate_count": 4 if args.mode == "smoke" else 16,
        "plus_candidate_construction": "episode_bootstrap_map_fixed_pi_v2",
        "moor_candidate_count": 1,
        "evaluation_seed_count": 1,
        "evaluation_episodes_per_seed": 1 if args.mode == "smoke" else 4,
        "headline_sweep_authorized": False,
        "blinded_canary_authorized": False,
        "frozen_20260716_run_modified": False,
    }
    (run_root / "manifests/registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"run_root": str(run_root), **registration}, indent=2))


if __name__ == "__main__":
    main()
