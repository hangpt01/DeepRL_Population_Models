"""Return-blind tooling gates for the frozen Phase 2E canary package."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from real_ecology_benchmark import general_canary_acceptance as acceptance


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCRIPT = ROOT / "scripts/make_general_phase2e_canary_manifests.py"
PREPARE_SCRIPT = ROOT / "scripts/prepare_general_phase2e_canary.py"


def _generate(tmp_path: Path, mode: str) -> tuple[Path, list[dict[str, str]]]:
    path = tmp_path / f"{mode}.csv"
    subprocess.run(
        [sys.executable, str(MANIFEST_SCRIPT), "--mode", mode, "--output", str(path)],
        check=True,
        cwd=ROOT,
    )
    with path.open("r", encoding="utf-8") as handle:
        return path, list(csv.DictReader(handle))


def _canonical(row: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in row.items() if key not in {"index", "package_role"}}


def test_phase2e_manifests_have_exact_registered_grid_and_projection(tmp_path):
    _timing_path, timing = _generate(tmp_path, "timing_preflight")
    _full_path, full = _generate(tmp_path, "limited_canary")
    assert len(timing) == 4
    assert len(full) == 64
    assert {row["method"] for row in full} == {
        "refplan", "ogsrl", "bamcts", "ensemble_value_disagreement_pessimism"
    }
    assert {row["population"] for row in full} == {"Amur tiger", "Egyptian vulture"}
    assert {row["environment"] for row in full} == {"ricker", "allee", "theta", "regime"}
    assert {row["sigma_obs"] for row in full} == {"0.0", "0.4"}
    assert {row["reward_mode"] for row in full} == {"safe"}
    assert {row["target_rows"] for row in full} == {"4000"}
    assert {row["collection_seed"] for row in full} == {"116"}
    assert {row["evaluation_seed"] for row in full} == {"9001"}
    assert {row["evaluation_episodes"] for row in full} == {"1"}
    assert {row["ogsrl_deployment_rollouts"] for row in full} == {"256"}
    subset = [row for row in full if (
        row["population"] == "Egyptian vulture"
        and row["environment"] == "regime"
        and row["sigma_obs"] == "0.4"
    )]
    assert [_canonical(row) for row in timing] == [_canonical(row) for row in subset]


def _example_receipt(tmp_path: Path) -> tuple[Path, dict]:
    manifest, rows = _generate(tmp_path, "timing_preflight")
    summary = {
        "fit_diagnostics": {
            "hidden_actor_trained": 1.0,
            "lambda_safety": 1.2,
            "lambda_ood": 0.8,
            "behavior_normalized_cost": 0.07,
            "surrogate_loss": 99.0,
        },
        "dataset_sha256": "d" * 64,
        "target_rows": 4000,
        "actual_rows": 4000,
        "overshoot_rows": 0,
        "episode_count": 160,
        "manifest_row_seconds": 12.0,
        "peak_rss_mb": 256.0,
    }
    row = dict(rows[1])
    payload = acceptance.build_validity_receipt(
        summary, row, manifest, "s" * 64, [100] * 11
    )
    target = tmp_path / "validity_receipt.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target, payload


def test_validity_receipt_contains_only_allowed_structural_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    path, payload = _example_receipt(tmp_path)
    raw = path.read_text(encoding="utf-8").lower()
    assert not any(fragment in raw for fragment in acceptance.FORBIDDEN_FIELD_FRAGMENTS)
    assert "surrogate_loss" not in payload["mechanism"]
    loaded = acceptance.load_validity_receipt(path)
    assert loaded["fit"]["completed"] is True
    assert loaded["mechanism"]["behavior_normalized_cost"] == 0.07


def test_acceptance_reader_rejects_nonreceipt_without_opening(tmp_path):
    forbidden = tmp_path / "summary.json"
    with patch.object(Path, "read_bytes", side_effect=AssertionError("must not open")):
        with pytest.raises(ValueError, match="only opens"):
            acceptance.load_validity_receipt(forbidden)


def test_forbidden_field_is_rejected_before_payload_deserialization(tmp_path):
    path = tmp_path / "validity_receipt.json"
    path.write_text('{"operational_return": 123.0}', encoding="utf-8")
    original = acceptance.json.loads

    def keys_only(value):
        raw = value if isinstance(value, bytes) else str(value).encode()
        assert not raw.lstrip().startswith(b"{"), "forbidden payload was deserialized"
        return original(value)

    with patch.object(acceptance.json, "loads", side_effect=keys_only):
        with pytest.raises(ValueError, match="forbidden field"):
            acceptance.load_validity_receipt(path)


def test_phase2e_preparer_freezes_both_manifests_and_refuses_overwrite(tmp_path):
    run_root = tmp_path / "phase2e"
    subprocess.run(
        [
            sys.executable,
            str(PREPARE_SCRIPT),
            "--run-root", str(run_root),
            "--python-bin", sys.executable,
        ],
        check=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    registration = json.loads(
        (run_root / "manifests/registration.json").read_text(encoding="utf-8")
    )
    assert registration["submitted"] is False
    assert registration["authorized_to_submit"] is False
    assert registration["manifest_rows"] == {"limited_canary": 64, "timing_preflight": 4}
    assert registration["timing_rows_match_limited_projection"] is True
    assert registration["old_1152_manifest_modified"] is False
    commands = (run_root / registration["scheduler_commands_path"]).read_text(encoding="utf-8")
    command_lines = [line for line in commands.splitlines() if line.startswith("sbatch ")]
    assert len(command_lines) == 4
    assert all("--test-only" in line for line in command_lines)
    assert "--array=0-3%1" in commands and "--array=0-3%4" in commands

    refused = subprocess.run(
        [
            sys.executable,
            str(PREPARE_SCRIPT),
            "--run-root", str(run_root),
            "--python-bin", sys.executable,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert refused.returncode != 0
    assert "refusing to overwrite" in refused.stderr
