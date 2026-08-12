"""Canonical scientific identities shared by registration freeze and driver execution.

This is deliberately a leaf module: importing only :mod:`common` prevents the registration,
evidence, and driver modules from acquiring a cycle while keeping each derived identity in one
authoritative recipe.
"""

from __future__ import annotations

from typing import Any, Mapping

from .common import (
    ContractError,
    canonical_json_bytes,
    require_exact_keys,
    require_finite,
    require_sha256,
    sha256_bytes,
)


REGISTERED_HORIZON = 50
REGISTERED_GAMMA = 0.95
REGISTERED_NUM_ACTIONS = 11
REGISTERED_EPISODE_IDS = (
    tuple(range(7001, 7005))
    + tuple(range(7051, 7055))
    + tuple(range(7101, 7105))
    + tuple(range(7151, 7155))
    + tuple(range(7201, 7205))
)

ARTIFACT_PLAN_SCHEMA_VERSION = "corrected_stageb_driver_artifact_plan_v1"
RNG_CONTRACT_SCHEMA_VERSION = "corrected_stageb_rng_contract_v1"
RNG_RECEIPT_SCHEMA_VERSION = "corrected_stageb_rng_receipt_v2"
STEP_EVIDENCE_SCHEMA_VERSION = "corrected_stageb_step_evidence_v2"
EXACT_RETURN_SCHEMA_VERSION = "corrected_stageb_exact_return_reconstruction_v2"
PAIRED_EVIDENCE_SCHEMA_VERSION = "corrected_stageb_paired_evidence_v2"
BOUND_TASK_EVIDENCE_SCHEMA_VERSION = "corrected_stageb_bound_task_evidence_v2"
REGISTERED_PROCESS_NOISE_SIGMA = 0.0
REGISTERED_OBSERVATION_NOISE_SIGMA = 0.2
DRIVER_INPUTS_SCHEMA_VERSION = "corrected_stageb_driver_inputs_v5"
DRIVER_INPUTS_REGISTRATION_SCHEMA_VERSION = "corrected_stageb_driver_inputs_registration_v4"
DRIVER_INPUTS_FILENAME = "DRIVER_INPUTS.json"
ROLE_NAMESPACES = (
    "arm-o",
    "arm-o-receipts",
    "arm-o-failure-diagnostics",
    "arm-t",
    "arm-t-receipts",
    "gate",
    "finalizer",
)


def canonical_noise_sigma(value: Any, label: str) -> float:
    """Return one explicit finite nonnegative floating-point noise binding."""

    if isinstance(value, bool) or not isinstance(value, float):
        raise ContractError(f"{label} must be an explicit floating-point number")
    parsed = require_finite(value, label)
    if parsed < 0.0:
        raise ContractError(f"{label} must be nonnegative")
    if parsed == 0.0:
        return 0.0
    return parsed


def registered_rng_contract_document() -> dict[str, Any]:
    """Return the one prospective environment RNG contract used by every role."""

    return {
        "schema_version": RNG_CONTRACT_SCHEMA_VERSION,
        "process_noise_sigma": REGISTERED_PROCESS_NOISE_SIGMA,
        "observation_noise_sigma": REGISTERED_OBSERVATION_NOISE_SIGMA,
        "process_draws_per_step": 0,
        "observation_draws_per_step": 1,
    }


def validate_rng_contract_document(value: Any) -> dict[str, Any]:
    """Validate exact draw semantics without conflating environment steps and RNG draws."""

    if not isinstance(value, Mapping):
        raise ContractError("registered RNG contract must be an object")
    require_exact_keys(
        value,
        {
            "schema_version",
            "process_noise_sigma",
            "observation_noise_sigma",
            "process_draws_per_step",
            "observation_draws_per_step",
        },
        "registered RNG contract",
    )
    if value["schema_version"] != RNG_CONTRACT_SCHEMA_VERSION:
        raise ContractError("registered RNG contract schema mismatch")
    process = canonical_noise_sigma(value["process_noise_sigma"], "process noise sigma")
    observation = canonical_noise_sigma(value["observation_noise_sigma"], "observation noise sigma")
    expected_process_draws = int(process > 0.0)
    expected_observation_draws = int(observation > 0.0)
    for field, expected in (
        ("process_draws_per_step", expected_process_draws),
        ("observation_draws_per_step", expected_observation_draws),
    ):
        observed = value[field]
        if isinstance(observed, bool) or not isinstance(observed, int) or observed != expected:
            raise ContractError(f"registered RNG contract {field} mismatch")
    return {
        "schema_version": RNG_CONTRACT_SCHEMA_VERSION,
        "process_noise_sigma": process,
        "observation_noise_sigma": observation,
        "process_draws_per_step": expected_process_draws,
        "observation_draws_per_step": expected_observation_draws,
    }


def evaluation_identity_document() -> dict[str, Any]:
    return {
        "discount": REGISTERED_GAMMA,
        "episode_ids": list(REGISTERED_EPISODE_IDS),
        "horizon": REGISTERED_HORIZON,
        "num_actions": REGISTERED_NUM_ACTIONS,
    }


def evaluation_identity_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(evaluation_identity_document()))


EVALUATION_IDENTITY_SHA256 = evaluation_identity_sha256()


def artifact_plan_document(
    *,
    task_index: int,
    cell: str,
    method: str,
    dataset_sha256: str,
    publication_success_sha256: str,
    fit_probe_receipt_sha256: str,
    frozen_object_sha256: str,
    frozen_replay_sha256: str,
    component_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if isinstance(task_index, bool) or not isinstance(task_index, int) or not 0 <= task_index < 12:
        raise ContractError("artifact plan task index must be in 0..11")
    if not isinstance(cell, str) or not cell or not isinstance(method, str) or not method:
        raise ContractError("artifact plan cell and method must be nonempty strings")
    for label, digest in (
        ("dataset", dataset_sha256),
        ("publication success", publication_success_sha256),
        ("fit-probe receipt", fit_probe_receipt_sha256),
        ("frozen object", frozen_object_sha256),
        ("frozen replay", frozen_replay_sha256),
    ):
        require_sha256(digest, f"artifact plan {label}")
    if not isinstance(component_hashes, Mapping) or not component_hashes:
        raise ContractError("artifact plan requires a non-empty component-hash map")
    parsed: dict[str, str] = {}
    for name, digest in component_hashes.items():
        if not isinstance(name, str) or not name:
            raise ContractError("artifact plan component names must be non-empty strings")
        parsed[name] = require_sha256(digest, f"artifact plan component {name}")
    return {
        "schema_version": ARTIFACT_PLAN_SCHEMA_VERSION,
        "fit_probe_task_index": task_index,
        "cell": cell,
        "method": method,
        "dataset_sha256": dataset_sha256,
        "publication_success_sha256": publication_success_sha256,
        "fit_probe_receipt_sha256": fit_probe_receipt_sha256,
        "frozen_object_sha256": frozen_object_sha256,
        "frozen_replay_sha256": frozen_replay_sha256,
        "component_hashes": parsed,
        "arm_t_refit_permitted": False,
        "o_t_fitted_object_byte_identity_required": True,
    }


def artifact_plan_sha256(**fields: Any) -> str:
    return sha256_bytes(canonical_json_bytes(artifact_plan_document(**fields)))
