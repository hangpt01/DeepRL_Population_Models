from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

import pytest

from .. import driver
from ..common import ContractError, canonical_json_bytes, sha256_bytes, sha256_file
from ..driver_inputs import (
    driver_inputs_sha256,
    produce_registered_driver_inputs,
    stage_driver_inputs,
    validate_registered_driver_inputs,
    validate_output_root_entries,
)
from ..registration import CELLS, METHODS, freeze_registration_bundle
from ..submission import derive_durable_log_plan


@pytest.fixture
def fs04_tmp_path():
    repository = Path(__file__).resolve().parents[4]
    with tempfile.TemporaryDirectory(prefix=".stageb-driver-inputs-test-", dir=repository) as root:
        yield Path(root)


def _stage_registered(tmp_path: Path, registration_bundle):
    descriptor = registration_bundle["stageb_driver_inputs"]["descriptor"]
    output = tmp_path / "scientific-output"
    registration_bundle["scientific_log_plan"] = derive_durable_log_plan(
        f"/fs04/scratch2/ce25/synthetic-stageb-log-evidence-{os.getpid()}", output
    )
    frozen = freeze_registration_bundle(registration_bundle)
    path = stage_driver_inputs(output, descriptor)
    return frozen, output, path


def test_real_canonical_driver_inputs_round_trip(registration_bundle, fs04_tmp_path):
    frozen, output, path = _stage_registered(fs04_tmp_path, registration_bundle)
    loaded = driver.load_driver_inputs(output, frozen)
    assert loaded.descriptor_path == path
    assert loaded.descriptor_sha256 == driver_inputs_sha256(
        registration_bundle["stageb_driver_inputs"]["descriptor"]
    )
    assert loaded.policy.registration_id == frozen.registration_id
    assert len(loaded.policy.fit_probes) == 12


def test_committed_producer_and_validator_round_trip(registration_bundle, fs04_tmp_path):
    registration = fs04_tmp_path / "registration.json"
    output = fs04_tmp_path / "producer-output"
    registration_bundle["scientific_log_plan"] = derive_durable_log_plan(
        f"/fs04/scratch2/ce25/synthetic-stageb-log-evidence-{os.getpid()}", output
    )
    registration.write_bytes(canonical_json_bytes(registration_bundle))
    path = produce_registered_driver_inputs(registration_path=registration, output_root=output)
    assert path == output / driver.DRIVER_INPUTS
    assert (
        validate_registered_driver_inputs(registration_path=registration, output_root=output)
        == registration_bundle["stageb_driver_inputs"]["driver_inputs_sha256"]
    )


def test_policy_view_excludes_evaluator_and_gate_paths(registration_bundle, fs04_tmp_path):
    frozen, output, _path = _stage_registered(fs04_tmp_path, registration_bundle)
    loaded = driver.load_driver_inputs(output, frozen)
    assert not hasattr(loaded.policy, "accepted_parity")
    assert not hasattr(loaded.policy, "inherited_test_receipt")
    assert not hasattr(loaded.policy, "corrected_test_receipt")
    policy_text = repr(loaded.policy)
    assert "accepted-" not in policy_text
    assert "TEST_RECEIPT" not in policy_text


def test_evaluator_parity_is_per_cell_and_method(registration_bundle, fs04_tmp_path):
    frozen, output, _path = _stage_registered(fs04_tmp_path, registration_bundle)
    loaded = driver.load_driver_inputs(output, frozen)
    observed = [(item["cell"], item["method"]) for item in loaded.evaluator_only.accepted_parity]
    assert observed == [(cell, method) for cell in CELLS for method in METHODS]
    assert (
        len({item["episodes_csv"]["path"] for item in loaded.evaluator_only.accepted_parity}) == 12
    )


def test_descriptor_mutation_rejected_by_exact_registration_binding(
    registration_bundle, fs04_tmp_path
):
    frozen, output, path = _stage_registered(fs04_tmp_path, registration_bundle)
    changed = copy.deepcopy(registration_bundle["stageb_driver_inputs"]["descriptor"])
    changed["arm_t_exact_state_allowlist"] = ["current_abundance", "future_abundance"]
    path.write_bytes(canonical_json_bytes(changed))
    with pytest.raises(ContractError, match="registered descriptor"):
        driver.load_driver_inputs(output, frozen)


def test_noncanonical_descriptor_bytes_rejected(registration_bundle, fs04_tmp_path):
    frozen, output, path = _stage_registered(fs04_tmp_path, registration_bundle)
    value = registration_bundle["stageb_driver_inputs"]["descriptor"]
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    with pytest.raises(ContractError, match="canonical JSON"):
        driver.load_driver_inputs(output, frozen)


def test_missing_descriptor_rejected(registration_bundle, fs04_tmp_path):
    output = fs04_tmp_path / "empty-output"
    registration_bundle["scientific_log_plan"] = derive_durable_log_plan(
        f"/fs04/scratch2/ce25/synthetic-stageb-log-evidence-{os.getpid()}", output
    )
    frozen = freeze_registration_bundle(registration_bundle)
    output.mkdir()
    with pytest.raises(ContractError, match="DRIVER_INPUTS|driver inputs"):
        driver.load_driver_inputs(output, frozen)


def test_unregistered_output_root_entry_rejected(registration_bundle, fs04_tmp_path):
    frozen, output, _path = _stage_registered(fs04_tmp_path, registration_bundle)
    (output / "surprise.txt").write_text("collision", encoding="utf-8")
    with pytest.raises(ContractError, match="unregistered output-root"):
        driver.load_driver_inputs(output, frozen)


def test_prerequisite_staging_rejects_existing_content(registration_bundle, fs04_tmp_path):
    output = fs04_tmp_path / "occupied"
    output.mkdir()
    (output / "prior").write_text("x", encoding="utf-8")
    with pytest.raises(ContractError, match="collides"):
        stage_driver_inputs(output, registration_bundle["stageb_driver_inputs"]["descriptor"])


def test_prerequisite_staging_is_owner_only(registration_bundle, fs04_tmp_path):
    _frozen, output, path = _stage_registered(fs04_tmp_path, registration_bundle)
    assert os.stat(output).st_mode & 0o777 == 0o700
    assert os.stat(path).st_mode & 0o777 == 0o600
    validate_output_root_entries(output, pristine=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runtime_next_states_available", True, "next_states"),
        ("original_truth_archive_available_to_policy", True, "truth archive"),
        ("arm_t_refit_permitted", True, "refitting"),
        ("arm_t_exact_state_allowlist", ["current_abundance", "next_states"], "allowlist"),
    ],
)
def test_registration_rejects_information_boundary_mutation(
    registration_bundle, field, value, message
):
    malformed = copy.deepcopy(registration_bundle)
    descriptor = malformed["stageb_driver_inputs"]["descriptor"]
    descriptor[field] = value
    malformed["stageb_driver_inputs"]["driver_inputs_sha256"] = driver_inputs_sha256(descriptor)
    with pytest.raises(ContractError, match=message):
        freeze_registration_bundle(malformed)


def test_registration_rejects_stale_artifact_plan(registration_bundle):
    malformed = copy.deepcopy(registration_bundle)
    for index in (0, 12):
        malformed["task_manifest"]["tasks"][index]["artifact_plan_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="artifact plan"):
        freeze_registration_bundle(malformed)


def test_registration_rejects_reordered_accepted_parity(registration_bundle):
    malformed = copy.deepcopy(registration_bundle)
    parity = malformed["stageb_driver_inputs"]["descriptor"]["evaluator_only_inputs"][
        "accepted_parity"
    ]
    parity[0], parity[1] = parity[1], parity[0]
    malformed["stageb_driver_inputs"]["driver_inputs_sha256"] = driver_inputs_sha256(
        malformed["stageb_driver_inputs"]["descriptor"]
    )
    with pytest.raises(ContractError, match="accepted-parity task binding"):
        freeze_registration_bundle(malformed)


def test_registration_rejects_eligible_label_on_known_legacy_canary_path(
    registration_bundle, fs04_tmp_path
):
    malformed = copy.deepcopy(registration_bundle)
    canary = fs04_tmp_path / "i2b_fasttrack_integration_canary_20260808" / "episodes.csv"
    canary.parent.mkdir()
    canary.write_text("episode,seed,block_seed\n", encoding="utf-8")
    source = malformed["stageb_driver_inputs"]["descriptor"]["evaluator_only_inputs"][
        "accepted_parity"
    ][0]["episodes_csv"]
    source.update({"path": str(canary), "sha256": sha256_file(canary)})
    malformed["stageb_driver_inputs"]["driver_inputs_sha256"] = driver_inputs_sha256(
        malformed["stageb_driver_inputs"]["descriptor"]
    )
    with pytest.raises(ContractError, match="legacy/ineligible"):
        freeze_registration_bundle(malformed)


def test_registration_rejects_stale_source_receipt_hash(registration_bundle):
    malformed = copy.deepcopy(registration_bundle)
    malformed["stageb_driver_inputs"]["descriptor"]["public_inputs"][CELLS[0]]["public_npz"][
        "sha256"
    ] = "0" * 64
    malformed["stageb_driver_inputs"]["driver_inputs_sha256"] = driver_inputs_sha256(
        malformed["stageb_driver_inputs"]["descriptor"]
    )
    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        freeze_registration_bundle(malformed)


def test_descriptor_schema_is_strict_json_and_hash_bound(registration_bundle):
    schema = Path(__file__).parents[1] / "schemas/driver_inputs.schema.json"
    parsed = json.loads(schema.read_text(encoding="utf-8"))
    assert parsed["additionalProperties"] is False
    assert parsed["$id"] == "corrected-stageb-driver-inputs-v4"
    assert parsed["properties"]["rng_contract"] == {"$ref": "#/$defs/rng_contract"}
    assert parsed["properties"]["runtime_next_states_available"] == {"const": False}
    descriptor = registration_bundle["stageb_driver_inputs"]["descriptor"]
    assert registration_bundle["stageb_driver_inputs"]["driver_inputs_sha256"] == sha256_bytes(
        canonical_json_bytes(descriptor)
    )
