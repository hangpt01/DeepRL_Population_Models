from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from .. import driver
from ..common import ContractError, canonical_json_bytes, sha256_file, strict_json_loads
from ..parity import (
    PARITY_BASELINE_BINDING_SCHEMA_VERSION,
    PARITY_FAILURE_FILENAME,
    PARITY_FAILURE_NAMESPACE,
    PARITY_IDENTITY_FIELDS,
    REQUIRED_EXECUTION_DIMENSIONS,
    ParityEligibilityError,
    build_parity_comparison_contract,
    canonical_relative_delta,
    compare_accepted_parity,
    validate_parity_baseline_eligibility,
    validate_parity_failure_diagnostic,
)
from ..publication import load_success_receipt
from ..registration import freeze_registration_bundle
from .conftest import artifact_components
from .test_driver import _execution, _interpreter_receipt


def _baseline_case(registration_bundle, index: int = 2):
    descriptor = registration_bundle["stageb_driver_inputs"]["descriptor"]
    accepted = descriptor["evaluator_only_inputs"]["accepted_parity"][index]
    baseline = copy.deepcopy(accepted["baseline"])
    expected = {
        field: copy.deepcopy(baseline[field])
        for field in (
            "repository_commit",
            "source_manifest_sha256",
            "stageb_driver_sha256",
            "rng_contract",
            "interpreter",
            "fitted_object_sha256",
            "component_hashes",
            "artifact_plan_sha256",
            "evaluator",
        )
    }
    return baseline, expected, Path(accepted["episodes_csv"]["path"])


@pytest.mark.parametrize("index", [0, 2])
def test_fully_bound_baseline_is_eligible_for_both_tracks(registration_bundle, index):
    baseline, expected, accepted = _baseline_case(registration_bundle, index)
    result = validate_parity_baseline_eligibility(
        baseline, expected=expected, accepted_csv=accepted
    )
    assert result["assessment"]["result"] == "PASS"
    assert result["binding"]["interpreter"]["track"] in {"ecological", "general"}


@pytest.mark.parametrize("dimension", REQUIRED_EXECUTION_DIMENSIONS)
def test_each_missing_execution_dimension_fails_closed(registration_bundle, dimension):
    baseline, expected, accepted = _baseline_case(registration_bundle)
    baseline.pop(dimension)
    with pytest.raises(ParityEligibilityError) as raised:
        validate_parity_baseline_eligibility(baseline, expected=expected, accepted_csv=accepted)
    assert raised.value.assessment["result"] == "FAIL"
    assert dimension in raised.value.assessment["missing_bindings"]


@pytest.mark.parametrize(
    "dimension",
    [
        "repository_commit",
        "source_manifest_sha256",
        "stageb_driver_sha256",
        "rng_contract",
        "interpreter",
        "fitted_object_sha256",
        "component_hashes",
        "artifact_plan_sha256",
        "evaluator",
    ],
)
def test_each_execution_mismatch_fails_closed(registration_bundle, dimension):
    baseline, expected, accepted = _baseline_case(registration_bundle)
    expected[dimension] = {"deliberate": "mismatch"}
    with pytest.raises(ParityEligibilityError) as raised:
        validate_parity_baseline_eligibility(baseline, expected=expected, accepted_csv=accepted)
    assert dimension in raised.value.assessment["mismatched_bindings"]


def test_fast_track_canary_is_explicitly_legacy_and_ineligible(registration_bundle):
    _baseline, expected, accepted = _baseline_case(registration_bundle)
    legacy = {
        "schema_version": PARITY_BASELINE_BINDING_SCHEMA_VERSION,
        "classification": "LEGACY_INELIGIBLE",
        "producer_label": "2026-08-08 fast-track canary",
        "reason": "not generated under the current registered Stage B execution contract",
        "missing_bindings": list(REQUIRED_EXECUTION_DIMENSIONS),
    }
    with pytest.raises(ParityEligibilityError, match="legacy canary") as raised:
        validate_parity_baseline_eligibility(legacy, expected=expected, accepted_csv=accepted)
    assert raised.value.assessment["classification"] == "LEGACY_INELIGIBLE"
    assert raised.value.assessment["eligible"] is False


def test_fast_track_canary_path_cannot_be_laundered_by_eligible_metadata(
    registration_bundle, tmp_path
):
    baseline, expected, accepted = _baseline_case(registration_bundle)
    canary = tmp_path / "i2b_fasttrack_integration_canary_20260808" / "episodes.csv"
    canary.parent.mkdir()
    canary.write_bytes(accepted.read_bytes())
    with pytest.raises(ParityEligibilityError, match="2026-08-08 fast-track canary") as raised:
        validate_parity_baseline_eligibility(
            baseline,
            expected=expected,
            accepted_csv=canary,
        )
    assert raised.value.assessment["classification"] == "LEGACY_INELIGIBLE"
    assert raised.value.assessment["eligible"] is False


@pytest.mark.parametrize(
    ("observed", "reference", "expected"),
    [
        (0.0, 0.0, 0.0),
        (1e-300, 0.0, 1.0),
        (1_000_000_001.0, 1_000_000_000.0, 1.0 / 1_000_000_001.0),
        (1.0, -1.0, 2.0),
    ],
)
def test_canonical_relative_delta_semantics(observed, reference, expected):
    assert canonical_relative_delta(observed, reference) == pytest.approx(expected)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_relative_delta_rejects_non_finite(value):
    with pytest.raises(ContractError, match="finite"):
        canonical_relative_delta(value, 1.0)


def _write_reference(path: Path, *, true_return: str = "10.0", label: str = "same") -> None:
    path.write_text(
        "episode,seed,block_seed,label,true_return\n"
        + "".join(f"{i},{7001 + i},{7001 + i},{label},{true_return}\n" for i in range(20)),
        encoding="utf-8",
    )


def _comparison_contract():
    columns = (*PARITY_IDENTITY_FIELDS, "label", "true_return")
    return build_parity_comparison_contract(
        reference_columns=columns,
        exact_fields=(*PARITY_IDENTITY_FIELDS, "label"),
        numeric_fields=("true_return",),
        excluded_fields=(),
    )


def _observed_rows(*, label: str = "same", true_return: float = 10.0):
    return [
        {
            "episode": index,
            "seed": 7001 + index,
            "block_seed": 7001 + index,
            "label": label,
            "true_return": true_return,
        }
        for index in range(20)
    ]


def test_eligible_exact_and_numeric_comparison_passes(tmp_path):
    reference = tmp_path / "reference.csv"
    _write_reference(reference)
    result = compare_accepted_parity(_observed_rows(), reference, _comparison_contract())
    assert result["result"] == "PASS"
    assert result["mismatches"] == []


def test_exact_numeric_and_missing_mismatches_are_safe_and_complete(tmp_path):
    reference = tmp_path / "reference.csv"
    _write_reference(reference)
    rows = _observed_rows(label="different", true_return=-5.0)
    del rows[0]["seed"]
    result = compare_accepted_parity(rows, reference, _comparison_contract())
    assert result["result"] == "FAIL"
    assert {item["field_name"] for item in result["mismatches"]} == {
        "seed",
        "label",
        "true_return",
    }
    truth = next(item for item in result["mismatches"] if item["field_name"] == "true_return")
    assert truth["truth_bearing"] is True
    assert truth["absolute_delta"] == 15.0
    assert truth["relative_delta"] == 1.5
    assert "10.0" not in str(truth)
    assert "-5.0" not in str(truth)


def test_numeric_nonfinite_fails_closed_without_raw_value(tmp_path):
    reference = tmp_path / "reference.csv"
    _write_reference(reference, true_return="nan")
    result = compare_accepted_parity(
        _observed_rows(true_return=float("inf")), reference, _comparison_contract()
    )
    assert result["result"] == "FAIL"
    detail = next(item for item in result["mismatches"] if item["field_name"] == "true_return")
    assert detail["comparison_type"] == "numeric_non_finite"
    assert detail["absolute_delta"] is None
    assert "inf" not in str(detail).lower()
    assert "nan" not in str(detail).lower()


def _direct_inputs(registration):
    descriptor = registration.bundle()["stageb_driver_inputs"]["descriptor"]
    return driver.DriverInputs(
        Path("/registered/DRIVER_INPUTS.json"),
        Path(driver.__file__).resolve().parents[3],
        registration.registration_id,
        tuple(
            driver._parse_fit_probe(item, index)
            for index, item in enumerate(descriptor["fit_probes"])
        ),
        {},
        driver.validate_rng_contract_document(
            registration.bundle()["corrected_stageb_registration"]["rng_contract"]
        ),
    )


def _mismatch_publication_arguments(registration_bundle, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    descriptor = registration.bundle()["stageb_driver_inputs"]["descriptor"]
    task_index = 2
    task = driver._task_for_index(registration, "O", task_index)
    reference = tmp_path / "reference.csv"
    _write_reference(reference)
    accepted = copy.deepcopy(descriptor["evaluator_only_inputs"]["accepted_parity"])
    accepted[task_index]["episodes_csv"] = {
        "path": str(reference),
        "sha256": sha256_file(reference),
    }
    accepted[task_index]["baseline"]["comparison_contract"] = _comparison_contract()
    evaluator = driver.EvaluatorOnlyInputs(tuple(accepted))
    execution = replace(
        _execution(registration.sha256),
        evaluator_rows=tuple(_observed_rows(true_return=-5.0)),
    )
    return {
        "registration": registration,
        "inputs": _direct_inputs(registration),
        "evaluator_inputs": evaluator,
        "task": task,
        "task_index": task_index,
        "execution": execution,
        "component_payloads": artifact_components("refplan"),
        "frozen_payload": b"synthetic-frozen",
        "replay_payload": canonical_json_bytes({"synthetic": True}),
        "output_root": tmp_path,
        "interpreter_identity": _interpreter_receipt(
            registration, "arm-o", arm="O", task_index=task_index
        ),
    }


def test_mismatch_diagnostic_is_atomic_collision_safe_and_not_success(
    registration_bundle, tmp_path
):
    arguments = _mismatch_publication_arguments(registration_bundle, tmp_path)
    task_index = arguments["task_index"]
    reference = Path(
        arguments["evaluator_inputs"].accepted_parity[task_index]["episodes_csv"]["path"]
    )
    with pytest.raises(ContractError, match="accepted parity failed"):
        driver.publish_registered_task(**arguments)
    failure = tmp_path / PARITY_FAILURE_NAMESPACE / f"task-{task_index:02d}"
    diagnostic_path = failure / PARITY_FAILURE_FILENAME
    diagnostic = validate_parity_failure_diagnostic(strict_json_loads(diagnostic_path.read_bytes()))
    assert diagnostic["result"] == "FAIL"
    assert diagnostic["qualifies_as_task_result"] is False
    assert diagnostic["comparison"]["result"] == "FAIL"
    assert diagnostic["rng_evidence"]["status"] == "VALIDATED_POST_ROLLOUT"
    assert diagnostic["reference_provenance"]["episodes_csv_absolute_path"] == str(reference)
    assert diagnostic["reference_provenance"]["episodes_csv_registered_sha256"] == sha256_file(
        reference
    )
    assert diagnostic["reference_provenance"]["episodes_csv_observed_sha256"] == sha256_file(
        reference
    )
    manifest = strict_json_loads((failure / "FAILURE_MANIFEST.json").read_bytes())
    assert manifest["files"] == {PARITY_FAILURE_FILENAME: sha256_file(diagnostic_path)}
    assert not (failure / "PUBLICATION_SUCCESS.json").exists()
    assert not (tmp_path / "arm-o" / f"task-{task_index:02d}").exists()
    assert not (tmp_path / "arm-o-receipts" / f"task-{task_index:02d}.json").exists()
    success_shaped = dict(diagnostic)
    success_shaped["result"] = "PASS"
    success_shaped["accepted"] = True
    with pytest.raises(ContractError, match="confused with success"):
        validate_parity_failure_diagnostic(success_shaped)
    with pytest.raises(ContractError):
        load_success_receipt(diagnostic_path)
    with pytest.raises(ContractError, match="exists|retry|collid"):
        driver.publish_registered_task(**arguments)


def test_diagnostic_persistence_failure_itself_fails_closed(
    registration_bundle, tmp_path, monkeypatch
):
    arguments = _mismatch_publication_arguments(registration_bundle, tmp_path)

    def fail_persistence(**_kwargs):
        raise OSError("synthetic diagnostic persistence failure")

    monkeypatch.setattr(driver, "_publish_parity_failure", fail_persistence)
    with pytest.raises(OSError, match="diagnostic persistence"):
        driver.publish_registered_task(**arguments)
    task_index = arguments["task_index"]
    assert not (tmp_path / "arm-o" / f"task-{task_index:02d}").exists()
    assert not (tmp_path / "arm-o-receipts" / f"task-{task_index:02d}.json").exists()
    assert not (tmp_path / PARITY_FAILURE_NAMESPACE / f"task-{task_index:02d}").exists()


def test_failure_namespace_rejects_symlink(registration_bundle, tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / PARITY_FAILURE_NAMESPACE).symlink_to(external, target_is_directory=True)
    arguments = _mismatch_publication_arguments(registration_bundle, tmp_path)
    with pytest.raises(ContractError, match="not a real directory"):
        driver.publish_registered_task(**arguments)
    assert list(external.iterdir()) == []


def test_failure_diagnostic_information_boundary_rejects_forbidden_payload(
    registration_bundle, tmp_path
):
    registration = freeze_registration_bundle(registration_bundle)
    descriptor = registration.bundle()["stageb_driver_inputs"]["descriptor"]
    task_index = 2
    reference = tmp_path / "reference.csv"
    _write_reference(reference)
    accepted = copy.deepcopy(descriptor["evaluator_only_inputs"]["accepted_parity"])
    accepted[task_index]["episodes_csv"] = {
        "path": str(reference),
        "sha256": sha256_file(reference),
    }
    accepted[task_index]["baseline"]["comparison_contract"] = _comparison_contract()
    comparison = compare_accepted_parity(
        _observed_rows(true_return=-5.0), reference, _comparison_contract()
    )
    execution = replace(
        _execution(registration.sha256),
        evaluator_rows=tuple(_observed_rows(true_return=-5.0)),
    )
    driver._publish_parity_failure(
        registration=registration,
        inputs=_direct_inputs(registration),
        accepted=accepted[task_index],
        task=driver._task_for_index(registration, "O", task_index),
        task_index=task_index,
        output_root=tmp_path,
        interpreter_identity=_interpreter_receipt(
            registration, "arm-o", arm="O", task_index=task_index
        ),
        eligibility={
            "schema_version": "corrected_stageb_parity_baseline_eligibility_v1",
            "classification": "ELIGIBLE",
            "eligible": True,
            "missing_bindings": [],
            "mismatched_bindings": [],
            "result": "PASS",
        },
        comparison=driver._safe_parity_comparison_summary(comparison),
        mismatches=comparison["mismatches"],
        execution=execution,
    )
    diagnostic_path = (
        tmp_path / PARITY_FAILURE_NAMESPACE / f"task-{task_index:02d}" / PARITY_FAILURE_FILENAME
    )
    malformed = strict_json_loads(diagnostic_path.read_bytes())
    malformed["mismatches"][0]["next_states"] = [1, 2, 3]
    with pytest.raises(ContractError):
        validate_parity_failure_diagnostic(malformed)


def test_legacy_baseline_fails_before_executor(registration_bundle, tmp_path, monkeypatch):
    registration = freeze_registration_bundle(registration_bundle)
    descriptor = registration.bundle()["stageb_driver_inputs"]["descriptor"]
    legacy = {
        "schema_version": PARITY_BASELINE_BINDING_SCHEMA_VERSION,
        "classification": "LEGACY_INELIGIBLE",
        "producer_label": "2026-08-08 fast-track canary",
        "reason": "insufficient execution provenance",
        "missing_bindings": list(REQUIRED_EXECUTION_DIMENSIONS),
    }
    descriptor["evaluator_only_inputs"]["accepted_parity"][2]["baseline"] = legacy
    called = False

    def forbidden_executor(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("executor must not be constructed for an ineligible baseline")

    loaded = driver.LoadedDriverInputs(
        Path("/registered/DRIVER_INPUTS.json"),
        "a" * 64,
        _direct_inputs(registration),
        driver.EvaluatorOnlyInputs(tuple(descriptor["evaluator_only_inputs"]["accepted_parity"])),
        driver.GateOnlyInputs({}, {}),
    )
    monkeypatch.setattr(driver, "_load_registration", lambda _path: registration)
    monkeypatch.setattr(driver, "load_driver_inputs", lambda *_args: loaded)
    monkeypatch.setattr(driver, "_validate_fit_probe", lambda *_args: ({}, b"f", b"r", {}))
    monkeypatch.setattr(
        driver,
        "require_runtime_interpreter_binding",
        lambda *_args, **_kwargs: _interpreter_receipt(
            registration, "arm-o", arm="O", task_index=2
        ),
    )
    with pytest.raises(ParityEligibilityError, match="legacy canary"):
        driver.run_arm_task(
            registration_path=tmp_path / "registration.json",
            output_root=tmp_path,
            arm="O",
            task_index=2,
            executor=forbidden_executor,
        )
    assert called is False
    assert (tmp_path / PARITY_FAILURE_NAMESPACE / "task-02" / PARITY_FAILURE_FILENAME).is_file()
