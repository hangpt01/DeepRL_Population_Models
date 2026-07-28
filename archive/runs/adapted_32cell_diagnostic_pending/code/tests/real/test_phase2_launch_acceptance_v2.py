from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import phase2_acceptance_v2 as acceptance  # noqa: E402
import phase2_launch as launch  # noqa: E402


RUN = ROOT / "real_ecology_runs/adapted_32cell_diagnostic_pending"
FIT = RUN / "manifests/diagnostic_fit_64.csv"
PLAN = RUN / "manifests/diagnostic_plan_safe_64.csv"
PYTHON = ROOT / ".venv-paper-faithful/bin/python"


def external_manifests() -> tuple[Path, Path]:
    fit = Path(__import__("os").environ.get("PHASE2_FIT_MANIFEST", FIT))
    plan = Path(__import__("os").environ.get("PHASE2_PLAN_MANIFEST", PLAN))
    return fit, plan


def test_all_64_pairs_align_and_all_128_commands_resolve(tmp_path):
    fit, plan = external_manifests()
    python_bin = Path(__import__("os").environ.get("PHASE2_PYTHON_BIN", PYTHON))
    report = launch.validate(fit, plan, ROOT, python_bin)
    assert (
        report["pair_count"] == report["unique_fit_outputs"] == report["unique_plan_outputs"] == 64
    )
    derived = launch.write_derived(plan, tmp_path)
    assert len(derived) == 64
    commands = [
        launch.resolved_command(stage, position, fit, plan, tmp_path, RUN, ROOT, python_bin)
        for stage in ("fit", "plan")
        for position in range(64)
    ]
    assert len(commands) == 128
    assert {item["return_fields_opened"] for item in commands} == {False}
    assert len({item["output_key"] for item in commands if item["stage"] == "fit"}) == 64
    assert len({item["output_key"] for item in commands if item["stage"] == "plan"}) == 64
    for item in commands:
        assert item["dependency_key"] == item["position"]
        assert item["method"] in launch.METHODS
        assert item["run_stage"] == ("dynamics_fit" if item["stage"] == "fit" else "plan_evaluate")
        assert item["command"][1].endswith("phase2_thread_entry.py")
        runner = item["command"][4]
        assert (
            runner.endswith("run_adapted_fit_row.py")
            if item["stage"] == "fit"
            else runner.endswith("run_real_manifest_row.py")
        )
    for position, entry in enumerate(derived):
        with Path(entry["path"]).open(encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        assert row["index"] == str(entry["stored_manifest_index"])
        assert row["fit_manifest_index"] == str(position)
        assert row["fit_receipt_path"]
        assert row["fit_cache_hit_required"] == "True"


def test_split_method_arrays_are_an_exact_partition(tmp_path):
    fit, plan = external_manifests()
    python_bin = Path(__import__("os").environ.get("PHASE2_PYTHON_BIN", PYTHON))
    launch.write_derived(plan, tmp_path)
    all_commands = []
    for method_group in ("plus", "moor"):
        positions = launch.positions_for_method(launch.read_rows(fit), method_group)
        assert len(positions) == 32
        for local_position, original_position in enumerate(positions):
            for stage in ("fit", "plan"):
                item = launch.resolved_command(
                    stage, original_position, fit, plan, tmp_path, RUN, ROOT, python_bin, threads=1
                )
                item["array_position"] = local_position
                item["method_group"] = method_group
                all_commands.append(item)
    assert len(all_commands) == 128
    assert len({(item["stage"], item["position"]) for item in all_commands}) == 128
    assert {(item["stage"], item["position"]) for item in all_commands} == {
        (stage, position) for stage in ("fit", "plan") for position in range(64)
    }
    assert all(item["threads"] == 1 for item in all_commands)
    for method_group in ("plus", "moor"):
        fit_items = [
            item
            for item in all_commands
            if item["method_group"] == method_group and item["stage"] == "fit"
        ]
        plan_items = [
            item
            for item in all_commands
            if item["method_group"] == method_group and item["stage"] == "plan"
        ]
        assert [(x["array_position"], x["position"]) for x in fit_items] == [
            (x["array_position"], x["position"]) for x in plan_items
        ]


def test_unsupported_dispatch_stage_fails(tmp_path):
    fit, plan = external_manifests()
    python_bin = Path(__import__("os").environ.get("PHASE2_PYTHON_BIN", PYTHON))
    with pytest.raises(ValueError):
        launch.resolved_command("bogus", 0, fit, plan, tmp_path, RUN, ROOT, python_bin)


def test_acceptance_refuses_forbidden_paths_and_fields(tmp_path):
    for name in ("summary.json", "comparative_summary.json", "ranking.json"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(RuntimeError):
            acceptance.guarded_json(path)
    sentinel = tmp_path / "diagnostic.json"
    for field in ("operational_return", "true_return", "survival_return", "ranking"):
        sentinel.write_text(json.dumps({field: "SENTINEL_MUST_NOT_BE_OPENED"}), encoding="utf-8")
        with pytest.raises(RuntimeError):
            acceptance.guarded_json(sentinel)


def test_acceptance_ignores_summary_sentinel(monkeypatch, tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"operational_return": "SENTINEL"}), encoding="utf-8")
    original = Path.read_text

    def guarded_read(path: Path, *args, **kwargs):
        if path.name == "summary.json":
            raise AssertionError("acceptance attempted to open summary.json")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    allowed = tmp_path / "privacy_audit.json"
    allowed.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    assert acceptance.guarded_json(allowed)["status"] == "passed"


def test_full_acceptance_evaluation_does_not_open_summary(monkeypatch, tmp_path):
    fit, plan = external_manifests()
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"operational_return": "SENTINEL"}), encoding="utf-8")
    sacct = tmp_path / "sacct.txt"
    sacct.write_text("", encoding="utf-8")
    original = Path.read_text

    def guarded_read(path: Path, *args, **kwargs):
        if path.name == "summary.json":
            raise AssertionError("acceptance attempted to open summary.json")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    args = type(
        "Args",
        (),
        {
            "fit_manifest": fit,
            "plan_manifest": plan,
            "code_root": ROOT,
            "run_root": tmp_path,
            "fit_job_id": ["1"],
            "plan_job_id": ["2"],
            "sacct_file": sacct,
        },
    )()
    payload = acceptance.evaluate(args)
    assert payload["decision"] == "INCOMPLETE"
    assert payload["return_fields_opened"] is False
    assert len(payload["original_vs_amended_criterion_mapping"]) == 13


def test_amended_spec_preserves_original_and_classifies_every_criterion():
    original = (
        ROOT / "docs/fix_implement_ecology_baseline/PHASE2_PREREGISTRATION_AND_BUDGET_PACKAGE.md"
    )
    amended = (
        ROOT
        / "docs/fix_implement_ecology_baseline/ACCEPTANCE_V2_FROZEN_RUNTIME_OBSERVABLE_20260719.md"
    )
    assert original.is_file() and amended.is_file()
    text = amended.read_text(encoding="utf-8")
    for number in range(1, 14):
        assert f"| {number} |" in text
    assert "PASS_LIMITED_STRUCTURAL_ACCEPTANCE" in text
    assert "NOT EVALUABLE" in text and "DESCRIPTIVE ONLY" in text and "EVALUABLE GATE" in text
