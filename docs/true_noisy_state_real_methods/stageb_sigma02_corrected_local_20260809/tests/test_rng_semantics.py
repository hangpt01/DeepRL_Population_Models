from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from .. import driver
from ..canonical_plan import (
    RNG_RECEIPT_SCHEMA_VERSION,
    STEP_EVIDENCE_SCHEMA_VERSION,
    registered_rng_contract_document,
    validate_rng_contract_document,
)
from ..common import ContractError
from ..evidence import RNGReceipt
from ..registration import CONFIG_PATHS, POPULATIONS, freeze_registration_bundle
from .conftest import HASHES, step_records


def _receipt(*, process_sigma: float = 0.0, observation_sigma: float = 0.2) -> RNGReceipt:
    process_required = process_sigma > 0.0
    observation_required = observation_sigma > 0.0
    return RNGReceipt(
        schema_version=RNG_RECEIPT_SCHEMA_VERSION,
        process_noise_sigma=process_sigma,
        observation_noise_sigma=observation_sigma,
        process_draw_required=process_required,
        observation_draw_required=observation_required,
        process_state_advancement_applicable=process_required,
        observation_state_advancement_applicable=observation_required,
        process_draw_invocations_before=4,
        process_draw_invocations_after=4 + int(process_required),
        observation_draw_invocations_before=7,
        observation_draw_invocations_after=7 + int(observation_required),
        process_state_before_sha256=HASHES[0],
        process_state_after_sha256=HASHES[1] if process_required else HASHES[0],
        observation_state_before_sha256=HASHES[2],
        observation_state_after_sha256=(HASHES[3] if observation_required else HASHES[2]),
    )


def test_zero_noise_requires_no_draw_and_identical_state():
    receipt = _receipt()
    receipt.validate()
    assert receipt.process_draw_invocations_before == receipt.process_draw_invocations_after
    assert receipt.process_state_before_sha256 == receipt.process_state_after_sha256
    assert receipt.process_state_advancement_applicable is False


@pytest.mark.parametrize(
    "mutation",
    [
        {"process_draw_invocations_after": 5},
        {"process_state_after_sha256": HASHES[1]},
        {"process_draw_required": True},
        {"process_state_advancement_applicable": True},
    ],
)
def test_zero_noise_false_count_or_state_advancement_rejected(mutation):
    with pytest.raises(ContractError, match="process"):
        replace(_receipt(), **mutation).validate()


def test_positive_noise_requires_one_actual_draw_and_state_advancement():
    receipt = _receipt(process_sigma=0.125)
    receipt.validate(expected_process_noise_sigma=0.125)


@pytest.mark.parametrize(
    "mutation",
    [
        {"process_draw_invocations_after": 4},
        {"process_state_after_sha256": HASHES[0]},
        {"process_draw_invocations_after": 6},
        {"process_draw_required": False},
        {"process_state_advancement_applicable": False},
    ],
)
def test_positive_noise_missing_or_inconsistent_advancement_rejected(mutation):
    with pytest.raises(ContractError, match="process"):
        replace(_receipt(process_sigma=0.125), **mutation).validate(
            expected_process_noise_sigma=0.125
        )


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), float("-inf"), 0, True])
def test_invalid_or_ambiguous_noise_values_rejected(value):
    with pytest.raises(ContractError, match="noise sigma|floating-point"):
        replace(_receipt(), process_noise_sigma=value).validate()  # type: ignore[arg-type]


def test_unregistered_noise_value_rejected():
    with pytest.raises(ContractError, match="registered binding"):
        _receipt(process_sigma=0.125).validate()


def test_negative_zero_normalizes_to_registered_positive_zero():
    contract = registered_rng_contract_document()
    contract["process_noise_sigma"] = -0.0
    normalized = validate_rng_contract_document(contract)
    assert normalized["process_noise_sigma"].hex() == "0x0.0p+0"


def test_observation_zero_noise_has_symmetric_no_draw_semantics():
    receipt = _receipt(observation_sigma=0.0)
    receipt.validate(expected_observation_noise_sigma=0.0)
    with pytest.raises(ContractError, match="observation"):
        replace(receipt, observation_state_after_sha256=HASHES[3]).validate(
            expected_observation_noise_sigma=0.0
        )


def test_counting_generator_counts_actual_draws_and_rejects_other_apis():
    generator = driver._DrawCountingGenerator(
        np.random.default_rng(7001), permitted_method="normal"
    )
    before = driver._hash_rng_state(generator)
    assert generator.draw_invocations == 0
    generator.normal(0.0, 0.2)
    assert generator.draw_invocations == 1
    assert driver._hash_rng_state(generator) != before
    with pytest.raises(ContractError, match="unregistered"):
        generator.lognormal(0.0, 0.2)
    assert generator.draw_invocations == 1


@pytest.mark.parametrize(
    ("track", "method", "config_key"),
    [
        ("ecological", "plus_adapted_ricker_only_pbvi", "plus_config_sha256"),
        ("general", "refplan", "general_config_sha256"),
    ],
)
@pytest.mark.parametrize("process_sigma", [0.0, 0.125])
def test_real_track_environment_records_registered_process_draw_semantics(
    registration_bundle, track, method, config_key, process_sigma
):
    registration = freeze_registration_bundle(registration_bundle)
    task_index = list(driver.METHODS).index(method)
    task = driver._task_for_index(registration, "O", task_index)
    repository = Path(__file__).resolve().parents[4]
    with driver._registered_track_imports(track, repository):
        from real_ecology_benchmark.config import load_config, real_environment_like
        from real_ecology_benchmark.envs import make_env

        cfg = load_config(repository / CONFIG_PATHS[config_key])
        cfg.environment = real_environment_like(cfg.environment, POPULATIONS[task["cell"]], "allee")
        cfg.environment = replace(
            cfg.environment,
            process_noise_sigma=process_sigma,
            observation_noise_sigma=0.2,
        )
        rng_contract = registered_rng_contract_document()
        rng_contract["process_noise_sigma"] = process_sigma
        rng_contract["process_draws_per_step"] = int(process_sigma > 0.0)
        recorder = driver._EvaluatorRecorder(
            registration,
            task,
            driver._ExactStateBridge(),
            rng_contract,
        )
        environment = recorder.wrap(make_env(cfg.environment))
        environment.reset(7001)
        environment.step(0)
    receipt = recorder.episodes[0][0].rng_receipt
    receipt.validate(expected_process_noise_sigma=process_sigma)
    assert receipt.process_draw_invocations_after == int(process_sigma > 0.0)
    assert (receipt.process_state_before_sha256 != receipt.process_state_after_sha256) is (
        process_sigma > 0.0
    )
    assert receipt.observation_draw_invocations_after == 1


def test_registration_and_driver_inputs_bind_identical_rng_contract(registration_bundle):
    registration_rng = registration_bundle["corrected_stageb_registration"]["rng_contract"]
    driver_rng = registration_bundle["stageb_driver_inputs"]["descriptor"]["rng_contract"]
    assert validate_rng_contract_document(registration_rng) == registered_rng_contract_document()
    assert validate_rng_contract_document(driver_rng) == registered_rng_contract_document()
    freeze_registration_bundle(registration_bundle)


@pytest.mark.parametrize(
    ("target", "mutation"),
    [
        ("registration", "missing"),
        ("registration", "negative"),
        ("registration", "nan"),
        ("registration", "infinity"),
        ("driver_inputs", "missing"),
        ("driver_inputs", "mismatch"),
    ],
)
def test_missing_invalid_or_mismatched_registered_rng_contract_rejected(
    registration_bundle, target, mutation
):
    malformed = copy.deepcopy(registration_bundle)
    if target == "registration":
        contract = malformed["corrected_stageb_registration"]["rng_contract"]
    else:
        contract = malformed["stageb_driver_inputs"]["descriptor"]["rng_contract"]
    if mutation == "missing":
        del contract["process_noise_sigma"]
    elif mutation == "negative":
        contract["process_noise_sigma"] = -0.1
    elif mutation == "nan":
        contract["process_noise_sigma"] = float("nan")
    elif mutation == "infinity":
        contract["process_noise_sigma"] = float("inf")
    else:
        contract["process_noise_sigma"] = 0.125
        contract["process_draws_per_step"] = 1
    with pytest.raises(ContractError, match="RNG contract|noise sigma|driver-input"):
        freeze_registration_bundle(malformed)


def test_v1_v2_receipt_and_step_shapes_are_mutually_rejected():
    record = step_records()[0].evaluator_only_dict()
    v2_rng = record["rng_receipt"]
    legacy_rng_keys = {
        "process_calls_before",
        "process_calls_after",
        "observation_calls_before",
        "observation_calls_after",
        "process_state_before_sha256",
        "process_state_after_sha256",
        "observation_state_before_sha256",
        "observation_state_after_sha256",
    }
    assert set(v2_rng) != legacy_rng_keys
    assert set(v2_rng) - legacy_rng_keys
    legacy = dict(record)
    legacy.pop("schema_version")
    legacy["rng_receipt"] = {
        "process_calls_before": 0,
        "process_calls_after": 1,
        "observation_calls_before": 0,
        "observation_calls_after": 1,
        "process_state_before_sha256": HASHES[0],
        "process_state_after_sha256": HASHES[1],
        "observation_state_before_sha256": HASHES[2],
        "observation_state_after_sha256": HASHES[3],
    }
    with pytest.raises(ContractError, match="v2 step evidence|schema"):
        driver._step_from_mapping(legacy)
    assert set(v2_rng) != legacy_rng_keys  # v2 has forbidden extras under v1 additionalProperties


def test_receipt_schema_declares_v2_rng_evidence_and_failure_only_parity_shape():
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas/receipts.schema.json").read_text(encoding="utf-8")
    )
    assert schema["$id"] == "corrected_stageb_receipts_v3.schema.json"
    assert schema["$defs"]["rngReceipt"]["properties"]["schema_version"] == {
        "const": RNG_RECEIPT_SCHEMA_VERSION
    }
    assert schema["$defs"]["stepEvidence"]["properties"]["schema_version"] == {
        "const": STEP_EVIDENCE_SCHEMA_VERSION
    }
    assert "process_calls_before" not in schema["$defs"]["rngReceipt"]["properties"]
    assert "process_draw_invocations_before" in schema["$defs"]["rngReceipt"]["properties"]
    failure = schema["$defs"]["parityFailureDiagnostic"]
    assert failure["properties"]["result"] == {"const": "FAIL"}
    assert failure["properties"]["accepted"] == {"const": False}
    assert failure["properties"]["qualifies_as_task_result"] == {"const": False}


def test_stored_v2_rng_receipt_round_trips_and_revalidates():
    value = step_records()[0].evaluator_only_dict()
    rebuilt = driver._step_from_mapping(value)
    assert asdict(rebuilt) == {key: item for key, item in value.items() if key != "evidence_layer"}
    rebuilt.rng_receipt.validate()
