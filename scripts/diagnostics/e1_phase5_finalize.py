#!/usr/bin/env python3
"""Finalize Phase 5 hashes, reproduction commands, and read-only re-audit prompt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / ".verification" / "e1_phase5_remediation"
SCRIPTS = (
    REPO_ROOT / "scripts" / "diagnostics" / "e1_phase5_remediated_validation.py",
    REPO_ROOT / "scripts" / "diagnostics" / "e1_phase5_fit_provenance_audit.py",
    REPO_ROOT / "scripts" / "diagnostics" / "e1_phase5_grid_characterisation.py",
    Path(__file__).resolve(),
)
HISTORICAL = (
    REPO_ROOT / ".verification" / "e1_py310_numpy226" / "phase1" / "E1_RECEIPT.json",
    REPO_ROOT / ".verification" / "e1_py310_numpy226" / "phase2_core" / "PHASE2_CORE_PHASE3_RECEIPT.json",
    REPO_ROOT / ".verification" / "e1_phase3_remainder" / "V6_DISCRETISATION.json",
    REPO_ROOT / "scripts" / "diagnostics" / "e1_phase1_parity.py",
    REPO_ROOT / "scripts" / "diagnostics" / "e1_phase2_core_validation.py",
    REPO_ROOT / "scripts" / "diagnostics" / "e1_phase3_remainder.py",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Phase 5 output collision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Phase 5 temporary output collision: {temporary}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
    temporary.replace(path)


def git_names(*paths: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *paths], cwd=REPO_ROOT,
        check=True, text=True, capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if (platform.python_version(), np.__version__) != ("3.10.14", "2.2.6"):
        raise AssertionError("finalizer must use the controlling environment")
    v1v5_path = args.root / "v1_v5" / "PHASE5_REMEDIATED_V1_V5_RECEIPT.json"
    provenance_path = args.root / "fit_provenance" / "FIT_PROVENANCE_AUDIT.json"
    grid_path = args.root / "grid_characterisation" / "PHASE5_GRID_CHARACTERISATION_RECEIPT.json"
    v1v5 = load_json(v1v5_path)
    provenance = load_json(provenance_path)
    grid = load_json(grid_path)
    if any(payload.get("status") != "PASS" for payload in (v1v5, provenance, grid)):
        raise AssertionError("Phase 5 component is not passing")
    prompt_path = args.root / "CLAUDE_READ_ONLY_REAUDIT_PROMPT.md"
    command_path = args.root / "CLAUDE_READ_ONLY_REAUDIT_COMMAND.txt"
    prompt = f"""# Independent read-only E1 Phase 5 re-audit

Audit `/fs04/scratch2/ce25/DeepRL_Population_Models` without modifying any file.
Do not run evaluation episodes, inspect scientific returns, submit jobs, or authorize V7/V8.
Use `/fs04/scratch2/ce25/hphung/conda/envs/poprl/bin/python3.10` only for read-only recomputation.

Verify independently:

1. The remediated V1--V5 receipt `{v1v5_path}` and digest `{v1v5['remediation_digest']}`. Confirm the counters instrument actual invoked call sites, the corrected expression is `abs(grid_value * survey_scale - N0)`, no extra initial-belief call constructs receipt fields, all sixteen actions hashes match the controlling receipt, and all runtime/lookahead accounting totals are internally exact.
2. The fit-provenance audit `{provenance_path}` and digest `{provenance['audit_digest']}`. Recompute all four cache keys from the archived frozen config/runtime without accepting E1 hardcoded fit constants. Check the source-to-target receipt, metadata, array, ledger, and faithful-fit artifact hash chains.
3. The grid receipt `{grid_path}` and digest `{grid['characterisation_digest']}` plus every task file it names. Confirm treatment isolation, common replay histories, deployed-41 trajectory control, both prior conventions, all 50 roots, all 5 seeds, all 5 grids, no reward/return accumulation, all Q/ranking/action records, branch-collapse metrics, and union-support Wasserstein/CDF calculations. Never apply direct vector L1/TV across unequal grids.
4. Recompute every action disagreement and the descriptive interpretation labels. Do not select a grid and do not treat this post-failure characterization as preregistered confirmation.
5. Confirm `src/tracks/**`, controlling documents, historical receipts, the genuine V6 failure receipt, and existing Phase 1--3 scripts were not changed by Phase 5.

Report defects and exact file/field references. End without repairs or execution.
"""
    write_new(prompt_path, prompt)
    command = (
        f"cd {REPO_ROOT} && claude --permission-mode plan --allowedTools Read,Glob,Grep "
        f"-p \"$(cat {prompt_path})\"\n"
    )
    write_new(command_path, command)
    output_files = sorted(
        path for path in args.root.rglob("*")
        if path.is_file() and path.name != "PHASE5_COMPLETION_MANIFEST.json"
    )
    file_hashes = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in (*SCRIPTS, *output_files)
    }
    historical_hashes = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in HISTORICAL}
    reproduction = [
        "PYTHONDONTWRITEBYTECODE=1 /fs04/scratch2/ce25/hphung/conda/envs/poprl/bin/python3.10 scripts/diagnostics/e1_phase5_remediated_validation.py",
        "PYTHONDONTWRITEBYTECODE=1 /fs04/scratch2/ce25/hphung/conda/envs/poprl/bin/python3.10 scripts/diagnostics/e1_phase5_fit_provenance_audit.py",
        "for arm in A3 A4; do for seed in 63001 63002 63003 63004 63005; do PYTHONDONTWRITEBYTECODE=1 /fs04/scratch2/ce25/hphung/conda/envs/poprl/bin/python3.10 scripts/diagnostics/e1_phase5_grid_characterisation.py task --arm \"$arm\" --seed \"$seed\"; done; done",
        "PYTHONDONTWRITEBYTECODE=1 /fs04/scratch2/ce25/hphung/conda/envs/poprl/bin/python3.10 scripts/diagnostics/e1_phase5_grid_characterisation.py aggregate",
        "PYTHONDONTWRITEBYTECODE=1 /fs04/scratch2/ce25/hphung/conda/envs/poprl/bin/python3.10 scripts/diagnostics/e1_phase5_finalize.py",
    ]
    payload = {
        "schema": "e1_phase5_completion_manifest_v1",
        "status": "PASS",
        "created_utc": now(),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "executable": str(Path(sys.executable).resolve()),
        },
        "component_digests": {
            "remediated_v1_v5": v1v5["remediation_digest"],
            "fit_provenance": provenance["audit_digest"],
            "grid_characterisation": grid["characterisation_digest"],
        },
        "new_file_hashes": file_hashes,
        "preserved_historical_file_hashes": historical_hashes,
        "scope_protection": {
            "src_tracks_git_diff_names": git_names("src/tracks"),
            "controlling_documents_git_diff_names": git_names(
                "E1_IMPLEMENTATION_BRIEF.md", "RESEARCH_PLAN_05.md"
            ),
            "scientific_returns_read": False,
            "V7_or_V8_run": False,
            "scientific_slurm_jobs_submitted": False,
        },
        "reproduction_commands": reproduction,
        "claude_reaudit": {
            "prompt": str(prompt_path.resolve()),
            "command": str(command_path.resolve()),
            "executed": False,
        },
    }
    if payload["scope_protection"]["src_tracks_git_diff_names"]:
        raise AssertionError("frozen src/tracks diff detected")
    if payload["scope_protection"]["controlling_documents_git_diff_names"]:
        raise AssertionError("controlling-document diff detected")
    digest_payload = dict(payload)
    digest_payload.pop("created_utc")
    payload["completion_digest"] = canonical_digest(digest_payload)
    manifest = args.root / "PHASE5_COMPLETION_MANIFEST.json"
    write_new(manifest, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": "PASS", "completion_digest": payload["completion_digest"],
        "files_hashed": len(file_hashes), "manifest": str(manifest.resolve()),
        "claude_command_executed": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
