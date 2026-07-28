#!/usr/bin/env python3
"""Freeze a provenance-complete code snapshot for the registered faithful smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "real_ecology_runs" / "paper_faithful_smoke_20260717"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


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
    subprocess.run(
        [
            args.python_bin,
            str(code_root / "scripts/make_paper_faithful_manifest.py"),
            "--mode", "smoke",
            "--output", str(run_root / "manifests/smoke.csv"),
        ],
        check=True,
        cwd=code_root,
    )
    subprocess.run(
        [
            args.python_bin,
            str(code_root / "scripts/run_paper_faithful_solver_probe.py"),
            "--output", str(run_root / "provenance/solver_probe.json"),
        ],
        check=True,
        cwd=code_root,
    )
    git_status = command("git", "status", "--short")
    diff = command("git", "diff", "--binary", "HEAD")
    registration = {
        "run_kind": "registered_interface_smoke_not_headline_experiment",
        "created_date": "2026-07-17",
        "source_commit": command("git", "rev-parse", "HEAD"),
        "source_dirty": bool(git_status),
        "source_status": git_status.splitlines(),
        "source_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "snapshot_tree_sha256": tree_hash(code_root),
        "config_sha256": sha256(code_root / "configs/paper_faithful_hidden_smoke.yaml"),
        "manifest_sha256": sha256(run_root / "manifests/smoke.csv"),
        "dependency_lock_sha256": sha256(
            code_root / "dependency/paper_faithful_fit_requirements.lock"
        ),
        "python_executable": str(Path(args.python_bin).absolute()),
        "python_version": subprocess.check_output(
            [args.python_bin, "-c", "import sys; print(sys.version)"], text=True
        ).strip(),
        "methods": [
            "plus_faithful_pbvi",
            "moor_faithful_ricker_misspec_pbvi",
        ],
        "rows": 2,
        "expose_rk": "hidden",
        "target_transitions": 160,
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
