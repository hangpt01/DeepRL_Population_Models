from __future__ import annotations

import copy
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ..artifacts import CanonicalArtifact, REQUIRED_COMPONENTS
from ..canonical_plan import (
    DRIVER_INPUTS_REGISTRATION_SCHEMA_VERSION,
    DRIVER_INPUTS_SCHEMA_VERSION,
    RNG_RECEIPT_SCHEMA_VERSION,
    STEP_EVIDENCE_SCHEMA_VERSION,
    artifact_plan_sha256,
    registered_rng_contract_document,
)
from ..common import (
    canonical_json_bytes,
    require_registered_cpu_model,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
)
from ..driver_inputs import driver_inputs_sha256
from ..evidence import RNGReceipt, StepEvidence
from ..parity import (
    PARITY_BASELINE_BINDING_SCHEMA_VERSION,
    PARITY_IDENTITY_FIELDS,
    build_parity_comparison_contract,
)
from ..registration import (
    ARMS,
    CELLS,
    METHODS,
    POPULATIONS,
    EXPECTED_CONFIG_HASHES,
    EXPECTED_COMMAND_ROLES,
    EXPECTED_DATASET_HASHES,
    EXPECTED_INTERPRETER_BINDINGS,
    EXPECTED_SOURCE_HASHES,
    _registration_templates_hash,
    require_runtime_interpreter_binding,
)
from ..submission import derive_durable_log_plan


HASHES = tuple(character * 64 for character in "abcdef0123456789")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
EVALUATION_IDS = (
    list(range(7001, 7005))
    + list(range(7051, 7055))
    + list(range(7101, 7105))
    + list(range(7151, 7155))
    + list(range(7201, 7205))
)
SYNTHETIC_EVALUATION_IDENTITY_SHA256 = sha256_bytes(
    canonical_json_bytes(
        {
            "discount": 0.95,
            "episode_ids": EVALUATION_IDS,
            "horizon": 50,
            "num_actions": 11,
        }
    )
)


def _write_synthetic_source(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(payload)


def _synthetic_driver_descriptor(
    repository: Path, *, registration_id: str, commit: str, driver_sha256: str
) -> dict[str, Any]:
    root = Path("/tmp") / f"corrected-stageb-driver-inputs-tests-{os.getpid()}"
    root.mkdir(mode=0o700, parents=False, exist_ok=True)
    probes = []
    parity = []
    manifest = Path(__file__).parents[1] / "SOURCE_TEST_HASHES.sha256"
    manifest_match = re.search(
        r"(?m)^# SELF-NORMALIZED-SHA256: ([0-9a-f]{64})  ",
        manifest.read_text(encoding="utf-8"),
    )
    if manifest_match is None:
        raise RuntimeError("candidate manifest is not sealed")
    source_manifest_sha256 = manifest_match.group(1)
    for index, (cell, method) in enumerate((c, m) for c in CELLS for m in METHODS):
        publication = root / f"fit-probe-{index:02d}"
        frozen = canonical_json_bytes({"frozen": index})
        replay = canonical_json_bytes({"replay": index})
        success = canonical_json_bytes({"publication": index})
        components = {"synthetic_component": sha256_bytes(canonical_json_bytes({"i": index}))}
        receipt = canonical_json_bytes(
            {
                "component_hashes": components,
                "fresh_reload_replay_sha256": sha256_bytes(replay),
                "frozen_fitted_object_sha256": sha256_bytes(frozen),
            }
        )
        for name, payload in (
            ("FROZEN_FITTED_OBJECT.json", frozen),
            ("FRESH_RELOAD_REPLAY.json", replay),
            ("PUBLICATION_SUCCESS.json", success),
            ("FIT_PROBE_RECEIPT.json", receipt),
        ):
            _write_synthetic_source(publication / name, payload)
        probes.append(
            {
                "task_index": index,
                "cell": cell,
                "method": method,
                "publication_dir": str(publication),
                "publication_success_sha256": sha256_bytes(success),
                "fit_probe_receipt_sha256": sha256_bytes(receipt),
                "frozen_object_sha256": sha256_bytes(frozen),
                "frozen_replay_sha256": sha256_bytes(replay),
                "component_hashes": components,
            }
        )
        accepted = root / f"accepted-{index:02d}.csv"
        _write_synthetic_source(
            accepted,
            (
                "episode,seed,block_seed\n"
                + "".join(
                    f"{episode},{episode_id},{episode_id - episode % 4}\n"
                    for episode, episode_id in enumerate(EVALUATION_IDS)
                )
            ).encode(),
        )
        ecological = method.startswith(("plus_", "moor_"))
        config_hash = (
            EXPECTED_CONFIG_HASHES["plus_config_sha256"]
            if method.startswith("plus_")
            else EXPECTED_CONFIG_HASHES["moor_config_sha256"]
            if method.startswith("moor_")
            else EXPECTED_CONFIG_HASHES["general_config_sha256"]
        )
        binding = next(
            item
            for item in EXPECTED_INTERPRETER_BINDINGS
            if item["role"] == ("ecological_paper_faithful" if ecological else "general_registered")
        )
        baseline = {
            "schema_version": PARITY_BASELINE_BINDING_SCHEMA_VERSION,
            "classification": "ELIGIBLE",
            "producer_registration_sha256": HASHES[0],
            "producer_driver_inputs_sha256": HASHES[1],
            "repository_commit": commit,
            "source_manifest_sha256": source_manifest_sha256,
            "stageb_driver_sha256": driver_sha256,
            "rng_contract": registered_rng_contract_document(),
            "interpreter": {
                field: binding[field]
                for field in (
                    "role",
                    "track",
                    "absolute_interpreter_path",
                    "resolved_executable_path",
                    "python_version",
                    "full_python_version",
                    "numpy_version",
                )
            },
            "fitted_object_sha256": probes[index]["frozen_object_sha256"],
            "component_hashes": probes[index]["component_hashes"],
            "artifact_plan_sha256": artifact_plan_sha256(
                task_index=index,
                cell=cell,
                method=method,
                dataset_sha256=EXPECTED_DATASET_HASHES[cell],
                publication_success_sha256=probes[index]["publication_success_sha256"],
                fit_probe_receipt_sha256=probes[index]["fit_probe_receipt_sha256"],
                frozen_object_sha256=probes[index]["frozen_object_sha256"],
                frozen_replay_sha256=probes[index]["frozen_replay_sha256"],
                component_hashes=probes[index]["component_hashes"],
            ),
            "evaluator": {
                "family": "allee",
                "reward_mode": "safe",
                "config_sha256": config_hash,
                "evaluation_identity_sha256": SYNTHETIC_EVALUATION_IDENTITY_SHA256,
                "episode_ids": EVALUATION_IDS,
                "horizon": 50,
                "discount": 0.95,
                "num_actions": 11,
            },
            "comparison_contract": build_parity_comparison_contract(
                reference_columns=PARITY_IDENTITY_FIELDS,
                exact_fields=PARITY_IDENTITY_FIELDS,
                numeric_fields=(),
                excluded_fields=(),
            ),
        }
        parity.append(
            {
                "task_index": index,
                "cell": cell,
                "method": method,
                "historical_canary_identity": "i2b_fasttrack_integration_canary_20260808",
                "episodes_csv": {"path": str(accepted), "sha256": sha256_file(accepted)},
                "baseline": baseline,
            }
        )
    public_inputs = {}
    for cell in CELLS:
        public = root / f"{cell}.public.npz"
        _write_synthetic_source(public, f"synthetic:{cell}\n".encode())
        public_inputs[cell] = {
            "population": POPULATIONS[cell],
            "public_npz": {"path": str(public), "sha256": sha256_file(public)},
            "logical_dataset_sha256": EXPECTED_DATASET_HASHES[cell],
        }
    inherited = root / "INHERITED_TEST_RECEIPT.json"
    corrected = root / "CORRECTED_TEST_RECEIPT.json"
    _write_synthetic_source(inherited, canonical_json_bytes({"synthetic": "inherited"}))
    _write_synthetic_source(corrected, canonical_json_bytes({"synthetic": "corrected"}))
    return {
        "schema_version": DRIVER_INPUTS_SCHEMA_VERSION,
        "registration_id": registration_id,
        "repository_root": str(repository),
        "repository_commit": commit,
        "stageb_driver_sha256": driver_sha256,
        "cpu_profile": "Intel Xeon Platinum 8452Y / xenon-8452Y / one CPU",
        "rng_contract": registered_rng_contract_document(),
        "fit_probes": probes,
        "public_inputs": public_inputs,
        "evaluator_only_inputs": {"accepted_parity": parity},
        "gate_only_inputs": {
            "inherited_test_receipt": {
                "path": str(inherited),
                "sha256": sha256_file(inherited),
            },
            "corrected_test_receipt": {
                "path": str(corrected),
                "sha256": sha256_file(corrected),
            },
        },
        "arm_t_exact_state_allowlist": ["current_abundance"],
        "arm_t_refit_permitted": False,
        "original_truth_archive_available_to_policy": False,
        "runtime_next_states_available": False,
    }


def current_repository_head(repository: Path) -> str:
    """Return the verified commit at HEAD or fail closed."""

    try:
        resolved = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError("git rev-parse HEAD is unavailable") from exc
    if resolved.returncode != 0:
        raise RuntimeError("git rev-parse HEAD failed")
    if resolved.stderr:
        raise RuntimeError("git rev-parse HEAD produced unexpected stderr")
    stdout = resolved.stdout
    if stdout.endswith("\n"):
        stdout = stdout[:-1]
    if GIT_SHA_RE.fullmatch(stdout) is None:
        raise RuntimeError("git rev-parse HEAD did not return exactly one lowercase 40-hex SHA")

    try:
        commit_check = subprocess.run(
            ["git", "cat-file", "-e", f"{stdout}^{{commit}}"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError("git commit verification is unavailable") from exc
    if commit_check.returncode != 0 or commit_check.stdout or commit_check.stderr:
        raise RuntimeError("git rev-parse HEAD did not resolve to a commit")
    return stdout


def complete_bundle() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[4]
    manifest = Path(__file__).parents[1] / "SOURCE_TEST_HASHES.sha256"
    match = re.search(
        r"(?m)^# SELF-NORMALIZED-SHA256: ([0-9a-f]{64})  ",
        manifest.read_text(encoding="utf-8"),
    )
    if match is None:
        raise RuntimeError("candidate manifest is not sealed")
    registration_id = "synthetic-corrected-stageb"
    commit = current_repository_head(repository)
    driver_path = (
        repository
        / "docs/true_noisy_state_real_methods/stageb_sigma02_corrected_local_20260809/driver.py"
    )
    descriptor = _synthetic_driver_descriptor(
        repository,
        registration_id=registration_id,
        commit=commit,
        driver_sha256=sha256_file(driver_path),
    )
    tasks = []
    index = 0
    for arm in ARMS:
        for cell in CELLS:
            for method in METHODS:
                ecological = method.startswith(("plus_", "moor_"))
                config_hash = (
                    EXPECTED_CONFIG_HASHES["plus_config_sha256"]
                    if method.startswith("plus_")
                    else EXPECTED_CONFIG_HASHES["moor_config_sha256"]
                    if method.startswith("moor_")
                    else EXPECTED_CONFIG_HASHES["general_config_sha256"]
                )
                tasks.append(
                    {
                        "task_index": index,
                        "arm": arm,
                        "cell": cell,
                        "method": method,
                        "config_sha256": config_hash,
                        "dataset_sha256": EXPECTED_DATASET_HASHES[cell],
                        "artifact_plan_sha256": artifact_plan_sha256(
                            task_index=index % 12,
                            cell=cell,
                            method=method,
                            dataset_sha256=EXPECTED_DATASET_HASHES[cell],
                            publication_success_sha256=descriptor["fit_probes"][index % 12][
                                "publication_success_sha256"
                            ],
                            fit_probe_receipt_sha256=descriptor["fit_probes"][index % 12][
                                "fit_probe_receipt_sha256"
                            ],
                            frozen_object_sha256=descriptor["fit_probes"][index % 12][
                                "frozen_object_sha256"
                            ],
                            frozen_replay_sha256=descriptor["fit_probes"][index % 12][
                                "frozen_replay_sha256"
                            ],
                            component_hashes=descriptor["fit_probes"][index % 12][
                                "component_hashes"
                            ],
                        ),
                        "evaluation_identity_sha256": SYNTHETIC_EVALUATION_IDENTITY_SHA256,
                        "interpreter_role": (
                            "ecological_paper_faithful" if ecological else "general_registered"
                        ),
                    }
                )
                index += 1
    return {
        "execution_authorization": {
            "schema_version": "corrected_stageb_execution_authorization_v1",
            "authorization_id": "synthetic-local-authorization",
            "authorized": True,
            "scope": "corrected Stage B sigma=0.2 only",
            "authorized_by": "synthetic-test-fixture",
            "authorized_at_utc": "2026-08-09T12:00:00Z",
            "no_return_exists_at_authorization": True,
        },
        "corrected_stageb_registration": {
            "schema_version": "corrected_stageb_registration_v3",
            "registration_id": registration_id,
            "status": "FROZEN_BEFORE_CORRECTED_RETURNS",
            "prospective_corrected_replication": True,
            "blinded_preregistration": False,
            "returns_exist_at_freeze": False,
            "cells": list(CELLS),
            "methods": list(METHODS),
            "arms": list(ARMS),
            "offline_rows": 4000,
            "episodes": 160,
            "episode_length": 25,
            "evaluation_identities": EVALUATION_IDS,
            "horizon": 50,
            "discount": 0.95,
            "num_actions": 11,
            "rng_contract": registered_rng_contract_document(),
            "evaluator_family": "allee",
            "cpu_profile": "Intel Xeon Platinum 8452Y / xenon-8452Y / one CPU",
            "local_status_label": "LOCAL ARM64 DEVELOPMENT TEST — NOT SCIENTIFIC EVIDENCE",
        },
        "analysis_rules": {
            "schema_version": "corrected_stageb_analysis_rules_v2",
            "primary_estimand": "mean_return_T_minus_mean_return_O",
            "bootstrap": {
                "resamples": 100000,
                "rng": "NumPy PCG64",
                "seed": 20260808,
                "quantiles": [0.025, 0.975],
                "quantile_method": "linear",
                "resampling_unit": "intact paired evaluation episode",
            },
            "transition_scale_rule": {
                "numeric_threshold": None,
                "ratio_guard": "compute only when Arm O denominator is positive",
                "general_learned_dynamics_residual_sigma_floor": 0.02,
                "ecological_process_scale_floor": None,
                "floor_semantics": "general_learned_dynamics_only",
                "changed_label": "MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY",
                "bit_identity_required_for_frozen_fit": True,
            },
            "near_constant_rule": {
                "prospectively_declared": True,
                "descriptive_only": True,
                "maximum_action_fraction_gte": 0.98,
                "literal_constant_is_separate": True,
            },
            "collapse_decomposition_rule": "collapse-entry step belongs to post-collapse",
            "evd_objective": "raw logged rewards; separately labelled",
            "cross_species_pooling": False,
        },
        "code_configuration_hashes": {
            "schema_version": "corrected_stageb_code_configuration_hashes_v2",
            "git_commit_sha": commit,
            "source_manifest_sha256": match.group(1),
            "registration_templates_sha256": _registration_templates_hash(repository),
            "stageb_driver_sha256": sha256_file(driver_path),
            **EXPECTED_CONFIG_HASHES,
            **EXPECTED_SOURCE_HASHES,
        },
        "task_manifest": {
            "schema_version": "corrected_stageb_task_manifest_v1",
            "task_count": 24,
            "tasks": tasks,
        },
        "stageb_interpreter_bindings": {
            "schema_version": "corrected_stageb_interpreter_bindings_v1",
            "bindings": copy.deepcopy(list(EXPECTED_INTERPRETER_BINDINGS)),
            "command_roles": dict(EXPECTED_COMMAND_ROLES),
        },
        "stageb_driver_inputs": {
            "schema_version": DRIVER_INPUTS_REGISTRATION_SCHEMA_VERSION,
            "driver_inputs_schema_sha256": sha256_file(
                Path(__file__).parents[1] / "schemas/driver_inputs.schema.json"
            ),
            "driver_inputs_producer_sha256": sha256_file(
                Path(__file__).parents[1] / "driver_inputs.py"
            ),
            "driver_inputs_sha256": driver_inputs_sha256(descriptor),
            "descriptor": descriptor,
        },
        "scientific_log_plan": derive_durable_log_plan(
            f"/fs04/scratch2/ce25/synthetic-stageb-log-evidence-{os.getpid()}",
            f"/fs04/scratch2/ce25/synthetic-stageb-output-{os.getpid()}",
        ),
        "previous_results_disclosure": {
            "schema_version": "corrected_stageb_previous_results_disclosure_v1",
            "earlier_sigma_0p2_results_known": True,
            "prospective_corrected_replication": True,
            "blinded_preregistration": False,
            "exploratory_results_scientifically_accepted": False,
            "required_statement": (
                "Earlier sigma=0.2 exploratory results are known; this is a prospective "
                "corrected replication, not a blinded preregistration."
            ),
        },
    }


@pytest.fixture
def registration_bundle() -> dict[str, Any]:
    return complete_bundle()


def _synthetic_interpreter_receipt(registration, command, *, arm=None, task_index=None):
    bundle = registration.bundle()
    if task_index is None:
        role = bundle["stageb_interpreter_bindings"]["command_roles"][command]
    else:
        manifest_index = task_index if arm == "O" else task_index + 12
        role = bundle["task_manifest"]["tasks"][manifest_index]["interpreter_role"]
    binding = next(
        item for item in bundle["stageb_interpreter_bindings"]["bindings"] if item["role"] == role
    )
    observed = {
        field: binding[field]
        for field in (
            "absolute_interpreter_path",
            "resolved_executable_path",
            "python_version",
            "full_python_version",
            "numpy_version",
        )
    }
    return require_runtime_interpreter_binding(
        registration,
        command=command,
        arm=arm,
        task_index=task_index,
        observed=observed,
    )


@pytest.fixture(autouse=True)
def isolated_orchestration_v2_helpers(request, monkeypatch):
    """Install reversible V2 helpers without cross-test-module import side effects."""

    module = request.module
    if not module.__name__.endswith(".test_orchestration_slurm"):
        return

    original_transition = module.build_arm_transition_diagnostics
    original_receipts = module.arm_o_receipts
    original_parity = module.parity_receipt

    def transition(**kwargs):
        if kwargs["method"] == "ensemble_value_disagreement_pessimism":
            return {
                "schema_version": "corrected_stageb_transition_not_applicable_v1",
                "registration_sha256": kwargs["registration_sha256"],
                "method": kwargs["method"],
                "cell": kwargs["cell"],
                "arm": kwargs["arm"],
                "applicability": "DEFINITIONALLY_NOT_APPLICABLE",
                "source_backed_reason": "EVD has no fitted transition model",
                "artifact_hashes": [],
            }
        return original_transition(**kwargs)

    def arm_o_receipts(registration, evidence_root, **kwargs):
        receipts = original_receipts(registration, evidence_root, **kwargs)
        bound = []
        for index, payload in enumerate(receipts):
            identity = _synthetic_interpreter_receipt(
                registration, "arm-o", arm="O", task_index=index
            )
            receipt = dict(strict_json_loads(payload))
            publication_path = evidence_root / f"task-{index}" / "PUBLICATION_SUCCESS.json"
            publication = dict(strict_json_loads(publication_path.read_bytes()))
            publication["validation"] = dict(publication["validation"])
            publication["validation"]["interpreter_identity"] = identity
            publication["validation"]["cpu_identity"] = require_registered_cpu_model(
                "Intel(R) Xeon(R) Platinum 8452Y"
            )
            publication_payload = canonical_json_bytes(publication)
            publication_path.write_bytes(publication_payload)
            publication_sha = sha256_bytes(publication_payload)
            receipt["interpreter_identity"] = identity
            receipt["cpu_identity"] = require_registered_cpu_model(
                "Intel(R) Xeon(R) Platinum 8452Y"
            )
            receipt["publication_manifest_sha256"] = publication_sha
            receipt["evidence"] = dict(receipt["evidence"])
            receipt["evidence"]["publication_success"] = {
                **receipt["evidence"]["publication_success"],
                "sha256": publication_sha,
            }
            bound.append(canonical_json_bytes(receipt))
        return bound

    def parity_receipt(registration, task_receipts, **kwargs):
        value = dict(strict_json_loads(original_parity(registration, task_receipts, **kwargs)))
        value["interpreter_identity"] = _synthetic_interpreter_receipt(registration, "arm-o-gate")
        return canonical_json_bytes(value)

    monkeypatch.setattr(module, "build_arm_transition_diagnostics", transition)
    monkeypatch.setattr(module, "arm_o_receipts", arm_o_receipts)
    monkeypatch.setattr(module, "parity_receipt", parity_receipt)


def artifact_components(
    method: str, cell: str = "amur_tiger__allee__sigma_0p2"
) -> dict[str, bytes]:
    population_token = "pop_tiger" if cell.startswith("amur_tiger") else "pop_fox"
    reward_features = [
        "standardized_previous_log_observation",
        "standardized_current_log_observation",
        "standardized_current_minus_previous_log_observation",
        "standardized_following_log_observation",
        "standardized_following_minus_current_log_observation",
        "standardized_timestep_fraction",
        *(f"action_{index}" for index in range(11)),
        "standardized_action_cost",
        f"population_token::{population_token}",
    ]
    dynamics_features = [
        "log_abundance",
        "log_abundance_squared",
        *(f"action_{index}" for index in range(11)),
        *(f"action_{index}_times_log_abundance" for index in range(11)),
        *(f"action_{index}_times_log_abundance_squared" for index in range(11)),
    ]
    belief_features = [
        "weighted_mean_log_abundance",
        "weighted_sd_log_abundance",
        "weighted_q10_log_abundance",
        "weighted_q50_log_abundance",
        "weighted_q90_log_abundance",
        "extinction_probability",
        "previous_public_observation",
        "current_public_observation",
        "public_timestep",
        "normalized_effective_sample_size",
    ]
    actor_features = ["log_current_observation", "log_current_observation_squared"]
    guardian_features = [
        "log_current_observation",
        "log_current_observation_squared",
        *(f"action_{index}" for index in range(11)),
    ]

    def action_model(features: list[str]) -> dict[str, Any]:
        return {
            "feature_order": features,
            "action_weights": np.arange(len(features) * 11, dtype=np.float64).reshape(
                len(features), 11
            )
            / 100.0,
            "action_bias": np.arange(11, dtype=np.float64) / 10.0,
        }

    result: dict[str, bytes] = {}

    def add(
        component: str,
        state: dict[str, Any],
        *,
        artifact_method: str = method,
        fit_source: str = "synthetic Arm O fixture",
    ) -> str:
        payload = CanonicalArtifact(
            component=component,
            method=artifact_method,
            fit_source=fit_source,
            state=state,
        ).to_bytes()
        result[component] = payload
        return sha256_bytes(payload)

    if method in {"plus_adapted_ricker_only_pbvi", "moor_adapted_ricker_misspec_pbvi"}:
        count = 8 if method.startswith("plus_") else 1
        candidate_ids = list(range(count))
        action_channels = [
            "none",
            "rate",
            "rate",
            "capacity",
            "rate+capacity",
            "state",
            "none",
            "rate",
            "capacity",
            "state",
            "rate+capacity",
        ]
        growth = np.zeros((count, 11), dtype=np.float64)
        mortality = np.zeros((count, 11), dtype=np.float64)
        growth[:, 0] = np.linspace(0.1, 0.2, count, dtype=np.float64)
        mortality[:, 1] = np.linspace(0.01, 0.02, count, dtype=np.float64)
        capacity_increment = np.zeros((count, 11), dtype=np.float64)
        capacity_increment[:, 3] = 0.5
        stocking = np.zeros((count, 11), dtype=np.float64)
        stocking[:, 5] = 0.25
        initial_capacity = np.linspace(10.0, 20.0, count, dtype=np.float64)
        process_scale = np.linspace(0.001, 0.019, count, dtype=np.float64)
        cache_sha = add(
            "ricker_fit_cache",
            {
                "schema_version": "corrected_stageb_ricker_fit_cache_v2",
                "cell": cell,
                "candidate_ids": candidate_ids,
                "candidate_labels": [f"candidate_{index:03d}" for index in candidate_ids],
                "form": ["ricker"] * count,
                "action_channels": [list(action_channels) for _ in candidate_ids],
                "growth": growth,
                "mortality": mortality,
                "capacity_increment": capacity_increment,
                "stocking": stocking,
                "process_scale": process_scale,
                "observation_scale": np.full(count, 0.2, dtype=np.float64),
                "survey_scale": np.full(count, 10.0, dtype=np.float64),
                "initial_capacity": initial_capacity,
                "capacity_ceiling": initial_capacity + 5.0,
                "reset_log_mean": np.linspace(0.1, 0.2, count, dtype=np.float64),
                "reset_log_scale": np.full(count, 0.3, dtype=np.float64),
                "depensation_thresholds": np.column_stack(
                    [initial_capacity * 0.1, initial_capacity * 0.2]
                ),
                "theta_exponent": np.full(count, 1.5, dtype=np.float64),
                "regime_multipliers": np.tile(np.array([[0.9, 1.1]], dtype=np.float64), (count, 1)),
                "regime_matrix": np.tile(
                    np.array([[[0.9, 0.1], [0.1, 0.9]]], dtype=np.float64),
                    (count, 1, 1),
                ),
                "equation_version": "adapted_mechanistic_v2",
                "regime_law_version": "discrete_current_then_switch_v1",
                "parameter_hashes": [HASHES[index % len(HASHES)] for index in candidate_ids],
            },
        )
        add(
            "residual_process_scales",
            {
                "candidate_ids": candidate_ids,
                "process_scale": process_scale.copy(),
                "ricker_fit_cache_sha256": cache_sha,
            },
        )
        add(
            "reward_surrogate",
            {
                "cell": cell,
                "source_public_view_sha256": EXPECTED_DATASET_HASHES[cell],
                "feature_order": reward_features,
                "weights": np.arange(len(reward_features), dtype=np.float64) / 100.0,
                "bias": 0.1,
            },
        )
        grids_sha = add(
            "pbvi_grids",
            {
                "abundance_grid": np.array([0.0, 0.5, 1.0], dtype=np.float64),
                "capacity_grid": np.array([0.5, 1.0, 1.5], dtype=np.float64),
                "observation_grid": np.array([0.0, 0.6, 1.2], dtype=np.float64),
            },
        )
        candidates_sha = add(
            "pbvi_candidates",
            {"candidate_ids": candidate_ids, "ricker_fit_cache_sha256": cache_sha},
        )
        policy_state = {
            "grids_sha256": grids_sha,
            "candidates_sha256": candidates_sha,
        }
        if method.startswith("plus_"):
            prior_sha = add(
                "pbvi_prior",
                {
                    "candidate_ids": candidate_ids,
                    "probabilities": np.full(count, 1.0 / count, dtype=np.float64),
                },
            )
            policy_state["prior_sha256"] = prior_sha
        add(
            "pbvi_policy",
            policy_state,
        )
        return result

    if method != "ensemble_value_disagreement_pessimism":
        add(
            "reward_surrogate",
            {
                "cell": cell,
                "source_public_view_sha256": EXPECTED_DATASET_HASHES[cell],
                "consumer_methods": ["bamcts", "ogsrl", "refplan"],
                "label": "PROSPECTIVELY RECONSTRUCTED MATCHED ARM-O SURROGATE",
                "feature_order": reward_features,
                "weights": np.arange(len(reward_features), dtype=np.float64) / 100.0,
                "bias": 0.1,
            },
            artifact_method="shared_general_methods",
            fit_source="Arm O public-data view only",
        )
        dynamics_sha = add(
            "dynamics_ensemble",
            {
                "feature_order": dynamics_features,
                "member_ids": list(range(5)),
                "weights": np.arange(5 * len(dynamics_features), dtype=np.float64).reshape(
                    5, len(dynamics_features)
                )
                / 100.0,
                "bias": np.arange(5, dtype=np.float64) / 10.0,
                "residual_sigma": np.full(5, 0.02, dtype=np.float64),
            },
        )
        add(
            "residual_process_scales",
            {
                "member_ids": list(range(5)),
                "residual_sigma": np.full(5, 0.02, dtype=np.float64),
                "dynamics_ensemble_sha256": dynamics_sha,
            },
        )
    if method == "refplan":
        add("refplan_behavior_prior", action_model(belief_features))
        add(
            "planner_configuration",
            {"horizon": 5, "num_sequences": 96, "num_particles": 32, "action_count": 11},
        )
    elif method == "ogsrl":
        add("ogsrl_actor", action_model(actor_features))
        add(
            "ogsrl_guardian",
            {
                "feature_order": guardian_features,
                "reference_points": np.arange(3 * len(guardian_features), dtype=np.float64).reshape(
                    3, len(guardian_features)
                ),
                "support_radius": 0.5,
            },
        )
        add(
            "ogsrl_safety_calibration",
            {"low_abundance_scale": 10.0, "safety_budget": 0.1, "guardian_threshold": 0.3},
        )
    elif method == "bamcts":
        add(
            "bamcts_model_bank",
            {
                "feature_order": dynamics_features,
                "member_ids": list(range(5)),
                "weights": np.arange(5 * len(dynamics_features), dtype=np.float64).reshape(
                    5, len(dynamics_features)
                )
                / 100.0,
                "bias": np.arange(5, dtype=np.float64) / 10.0,
            },
        )
        add(
            "bamcts_search_configuration",
            {
                "depth": 8,
                "simulations": 256,
                "posterior_likelihood_sigma": 0.05,
                "action_count": 11,
            },
        )
    elif method == "ensemble_value_disagreement_pessimism":
        add("evd_behavior_reference", action_model(belief_features))
        add(
            "evd_q_members",
            {
                "feature_order": belief_features,
                "member_ids": list(range(20)),
                "q_weights": np.arange(20 * len(belief_features) * 11, dtype=np.float64).reshape(
                    20, len(belief_features), 11
                )
                / 1000.0,
                "q_bias": np.arange(220, dtype=np.float64).reshape(20, 11) / 1000.0,
            },
        )
        add(
            "evd_policy_configuration",
            {"disagreement_penalty": 0.5, "action_count": 11, "objective": "raw_logged_rewards"},
        )
    assert set(result) == set(REQUIRED_COMPONENTS[method])
    return result


def step_records(
    *,
    collapse_at: int | None = 20,
    arm: str = "O",
    method: str = "refplan",
    cell: str = "amur_tiger__allee__sigma_0p2",
    registration_sha256: str = HASHES[0],
) -> list[StepEvidence]:
    records: list[StepEvidence] = []
    for timestep in range(50):
        discount = 0.95**timestep
        benefit = 2.0 + timestep / 100.0
        cost = -0.25
        penalty = -1.0 if collapse_at is not None and timestep >= collapse_at else 0.0
        total = benefit + cost + penalty
        rng = RNGReceipt(
            schema_version=RNG_RECEIPT_SCHEMA_VERSION,
            process_noise_sigma=0.0,
            observation_noise_sigma=0.2,
            process_draw_required=False,
            observation_draw_required=True,
            process_state_advancement_applicable=False,
            observation_state_advancement_applicable=True,
            process_draw_invocations_before=0,
            process_draw_invocations_after=0,
            observation_draw_invocations_before=timestep,
            observation_draw_invocations_after=timestep + 1,
            process_state_before_sha256=HASHES[0],
            process_state_after_sha256=HASHES[0],
            observation_state_before_sha256=HASHES[(timestep + 2) % len(HASHES)],
            observation_state_after_sha256=HASHES[(timestep + 3) % len(HASHES)],
        )
        records.append(
            StepEvidence(
                schema_version=STEP_EVIDENCE_SCHEMA_VERSION,
                registration_sha256=registration_sha256,
                method=method,
                cell=cell,
                arm=arm,
                episode_id=7001,
                timestep=timestep,
                current_true_abundance=max(0.0, 100.0 - timestep),
                noisy_observation=max(0.0, 101.0 - timestep),
                selected_action=timestep % 11,
                benefit_population_term=benefit,
                action_cost_term=cost,
                safety_penalty_term=penalty,
                total_reward=total,
                discount_factor=discount,
                discounted_benefit_population=benefit * discount,
                discounted_action_cost=cost * discount,
                discounted_safety_penalty=penalty * discount,
                discounted_total_reward=total * discount,
                collapse_indicator=collapse_at is not None and timestep >= collapse_at,
                unsafe_indicator=collapse_at is not None and timestep >= collapse_at,
                process_innovation_sha256=HASHES[(timestep + 4) % len(HASHES)],
                observation_innovation_sha256=HASHES[(timestep + 5) % len(HASHES)],
                rng_receipt=rng,
            )
        )
    return records
