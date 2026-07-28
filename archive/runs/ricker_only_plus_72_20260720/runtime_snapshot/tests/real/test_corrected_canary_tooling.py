import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCRIPT = ROOT / "scripts" / "make_paper_faithful_manifest.py"
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_corrected_adapted_canary.py"


def _manifest_rows(tmp_path: Path, mode: str) -> list[dict[str, str]]:
    output = tmp_path / f"{mode}.csv"
    subprocess.run(
        [sys.executable, str(MANIFEST_SCRIPT), "--mode", mode, "--output", str(output)],
        check=True,
        cwd=ROOT,
    )
    with output.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_corrected_canary_manifests_split_fit_and_plan_rows(tmp_path):
    fit_rows = _manifest_rows(tmp_path, "canary_fit")
    plan_rows = _manifest_rows(tmp_path, "canary_plan")

    assert len(fit_rows) == 4
    assert len(plan_rows) == 8
    expected_cells = {
        ("Amur tiger", "ricker", "0.1"),
        ("Egyptian vulture", "regime", "0.4"),
    }
    assert {
        (row["population"], row["environment"], row["sigma_obs"])
        for row in fit_rows
    } == expected_cells
    assert {row["method"] for row in fit_rows} == {
        "plus_adapted_mechanistic_pbvi",
        "moor_adapted_ricker_misspec_pbvi",
    }
    assert {row["run_stage"] for row in fit_rows} == {"dynamics_fit"}
    assert {row["reward_mode"] for row in fit_rows} == {""}
    assert {row["collection_reward_mode"] for row in fit_rows} == {"safe"}
    assert {
        row["reward_role"] for row in fit_rows
    } == {"fit_only_reward_excluded_from_fit_identity"}
    assert all("no_planner_no_evaluator" in row["return_blinding"] for row in fit_rows)

    fit_index = {
        (row["population"], row["environment"], row["sigma_obs"], row["method"]): row["index"]
        for row in fit_rows
    }
    assert {row["reward_mode"] for row in plan_rows} == {"safe", "yield"}
    assert {row["run_stage"] for row in plan_rows} == {"plan_evaluate"}
    for row in plan_rows:
        key = (row["population"], row["environment"], row["sigma_obs"], row["method"])
        assert row["fit_manifest_index"] == fit_index[key]
        assert row["fit_receipt_path"].startswith("fit_receipts/regime_hidden/reward_safe/")
        assert row["fit_cache_key_reference"] == "fit_receipt_json.fit_cache_keys"
        assert row["model_hash_reference"] == "fit_receipt_json.parameter_hashes"
        assert row["fit_cache_hit_required"] == "True"
        assert row["reward_role"] == "planner_evaluator_only"
        assert row["requested_cpu_ceiling_hours"] == "48"
        assert row["first_stage_cpu_ceiling_hours"] == "24"
        assert row["cumulative_cpu_ceiling_hours"] == "48"
        assert row["headline_sweep_authorized"] == "False"


def test_corrected_canary_preparer_records_manifest_hashes_and_refuses_overwrite(tmp_path):
    run_root = tmp_path / "corrected_canary"
    subprocess.run(
        [
            sys.executable,
            str(PREPARE_SCRIPT),
            "--run-root",
            str(run_root),
            "--python-bin",
            sys.executable,
        ],
        check=True,
        cwd=ROOT,
    )

    registration_path = run_root / "manifests" / "registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    assert registration["manifest_rows"] == {"canary_fit": 4, "canary_plan": 8}
    assert set(registration["manifest_sha256"]) == {"canary_fit", "canary_plan"}
    assert all(len(value) == 64 for value in registration["manifest_sha256"].values())
    assert registration["headline_sweep_authorized"] is False
    assert registration["fit_receipts_required_for_plan"] is True
    assert registration["first_stage_cpu_ceiling_hours"] == 24
    assert registration["cumulative_cpu_ceiling_hours"] == 48
    assert (run_root / "code" / "src").is_dir()
    assert (run_root / "code" / "configs").is_dir()
    assert (run_root / "code" / "scripts").is_dir()
    assert (run_root / "code" / "real_ecology_data").is_dir()

    refused = subprocess.run(
        [
            sys.executable,
            str(PREPARE_SCRIPT),
            "--run-root",
            str(run_root),
            "--python-bin",
            sys.executable,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert refused.returncode != 0
    assert "refusing to overwrite" in refused.stderr


def test_canary_plan_fit_receipt_gate_requires_existing_cache(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_real_manifest_row import (
        validate_adapted_fit_receipt_gate,
        validate_adapted_plan_cache_hit,
    )

    output_root = tmp_path / "run"
    fit_cell = "regime_hidden/reward_safe/amur_tiger/ricker/sigma_0p1"
    key = "a" * 64
    parameter_hash = "b" * 64
    receipt = output_root / "fit_receipts" / fit_cell / "plus_adapted_mechanistic_pbvi" / "fit_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps(
            {
                "run_stage": "dynamics_fit",
                "return_fields_opened": False,
                "method": "plus_adapted_mechanistic_pbvi",
                "cell": fit_cell,
                "fit_cache_keys": [key],
                "parameter_hashes": [parameter_hash],
            }
        ),
        encoding="utf-8",
    )
    cache_root = output_root / "fit_cache"
    cache_root.mkdir()
    (cache_root / f"{key}.npz").write_bytes(b"placeholder")
    (cache_root / f"{key}.json").write_text(
        json.dumps(
            {
                "cache_schema": "adapted_fit_cache_v1",
                "cache_key": key,
                "model": {"parameter_hash": parameter_hash},
            }
        ),
        encoding="utf-8",
    )
    row = {
        "method": "plus_adapted_mechanistic_pbvi",
        "fit_cell": fit_cell,
        "fit_receipt_path": str(receipt.relative_to(output_root)),
    }
    gate = validate_adapted_fit_receipt_gate(row, output_root)
    assert gate is not None
    summary_update = validate_adapted_plan_cache_hit(
        {
            "fit_diagnostics": {
                "fit_cache_keys": [key],
                "fit_cache_hits": 1.0,
                "fit_cache_misses": 0.0,
            }
        },
        gate,
    )
    assert summary_update["fit_cache_reuse_gate"] == "passed"

    (cache_root / f"{key}.json").unlink()
    with pytest.raises(SystemExit):
        validate_adapted_fit_receipt_gate(row, output_root)
