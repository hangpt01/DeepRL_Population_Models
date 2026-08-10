from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ..common import ContractError
from ..registration import EvaluatorReturnSink, FrozenRegistration, freeze_registration_bundle
from .conftest import step_records


def test_complete_registration_freezes_before_return(registration_bundle):
    frozen = freeze_registration_bundle(registration_bundle)
    sink = EvaluatorReturnSink(frozen)
    sink.append(step_records(registration_sha256=frozen.sha256))
    assert len(sink.records) == 1


def test_execution_before_freeze_rejected():
    with pytest.raises(ContractError, match="FrozenRegistration"):
        EvaluatorReturnSink({})  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="cannot be constructed"):
        FrozenRegistration(b"forged", "a" * 64, "forged")  # type: ignore[call-arg]
    forged = object.__new__(FrozenRegistration)
    object.__setattr__(forged, "_payload", b"forged")
    object.__setattr__(forged, "_sha256", "a" * 64)
    object.__setattr__(forged, "_registration_id", "forged")
    object.__setattr__(forged, "_seal", b"0" * 32)
    with pytest.raises(ContractError):
        EvaluatorReturnSink(forged)


def test_return_sink_rejects_arbitrary_mapping_and_wrong_registration(registration_bundle):
    frozen = freeze_registration_bundle(registration_bundle)
    sink = EvaluatorReturnSink(frozen)
    with pytest.raises(ContractError, match="StepEvidence"):
        sink.append([{"synthetic_return": 1.0}])
    with pytest.raises(ContractError, match="registration"):
        sink.append(step_records(registration_sha256="f" * 64))


def test_post_freeze_input_mutation_does_not_change_capability(registration_bundle):
    frozen = freeze_registration_bundle(registration_bundle)
    original = frozen.payload
    registration_bundle["execution_authorization"]["authorized"] = False
    assert frozen.payload == original
    assert frozen.bundle()["execution_authorization"]["authorized"] is True


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("execution_authorization", "authorized", False),
        ("execution_authorization", "no_return_exists_at_authorization", False),
        ("corrected_stageb_registration", "status", "TEMPLATE_NOT_FROZEN"),
        ("corrected_stageb_registration", "returns_exist_at_freeze", True),
        ("previous_results_disclosure", "earlier_sigma_0p2_results_known", False),
        ("previous_results_disclosure", "blinded_preregistration", True),
    ],
)
def test_registration_and_disclosure_fail_closed(registration_bundle, section, field, value):
    malformed = copy.deepcopy(registration_bundle)
    malformed[section][field] = value
    with pytest.raises(ContractError):
        freeze_registration_bundle(malformed)


def test_incomplete_task_manifest_rejected(registration_bundle):
    malformed = copy.deepcopy(registration_bundle)
    malformed["task_manifest"]["tasks"].pop()
    with pytest.raises(ContractError, match="count"):
        freeze_registration_bundle(malformed)


def test_duplicate_or_reordered_task_manifest_rejected(registration_bundle):
    malformed = copy.deepcopy(registration_bundle)
    malformed["task_manifest"]["tasks"][1]["method"] = malformed["task_manifest"]["tasks"][0][
        "method"
    ]
    with pytest.raises(ContractError, match="mapping|configuration"):
        freeze_registration_bundle(malformed)


def test_missing_or_placeholder_hash_rejected(registration_bundle):
    malformed = copy.deepcopy(registration_bundle)
    malformed["code_configuration_hashes"]["source_manifest_sha256"] = None
    with pytest.raises(ContractError, match="64 lowercase"):
        freeze_registration_bundle(malformed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ecological_source_only_sha256", "a" * 64),
        ("general_source_only_sha256", "b" * 64),
        ("general_config_sha256", "c" * 64),
        ("plus_config_sha256", "d" * 64),
        ("moor_config_sha256", "e" * 64),
    ],
)
def test_incorrect_controlling_hashes_cannot_freeze(registration_bundle, field, value):
    malformed = copy.deepcopy(registration_bundle)
    malformed["code_configuration_hashes"][field] = value
    with pytest.raises(ContractError, match="controlling|source-only"):
        freeze_registration_bundle(malformed)


def test_fabricated_or_reused_cross_cell_dataset_hash_cannot_freeze(registration_bundle):
    malformed = copy.deepcopy(registration_bundle)
    for task in malformed["task_manifest"]["tasks"]:
        task["dataset_sha256"] = "f" * 64
    with pytest.raises(ContractError, match="dataset hash"):
        freeze_registration_bundle(malformed)


def test_unfrozen_templates_are_structurally_rejected():
    root = Path(__file__).parents[1] / "registration"
    authorization = json.loads((root / "execution_authorization.template.json").read_text())
    assert authorization["authorized"] is False
    assert (
        json.loads((root / "corrected_stageb_registration.template.json").read_text())["status"]
        == "TEMPLATE_NOT_FROZEN"
    )
    hashes = json.loads((root / "code_configuration_hashes.template.json").read_text())
    assert hashes["git_commit_sha"] is None
