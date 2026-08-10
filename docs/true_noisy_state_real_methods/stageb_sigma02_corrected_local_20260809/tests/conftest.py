from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ..artifacts import CanonicalArtifact, REQUIRED_COMPONENTS
from ..common import sha256_bytes
from ..evidence import RNGReceipt, StepEvidence
from ..registration import (
    ARMS,
    CELLS,
    METHODS,
    EXPECTED_CONFIG_HASHES,
    EXPECTED_DATASET_HASHES,
    EXPECTED_SOURCE_HASHES,
    _registration_templates_hash,
)


HASHES = tuple(character * 64 for character in "abcdef0123456789")


def complete_bundle() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[4]
    manifest = Path(__file__).parents[1] / "SOURCE_TEST_HASHES.sha256"
    match = re.search(
        r"(?m)^# SELF-NORMALIZED-SHA256: ([0-9a-f]{64})  ",
        manifest.read_text(encoding="utf-8"),
    )
    if match is None:
        raise RuntimeError("candidate manifest is not sealed")
    evaluation_ids = (
        list(range(7001, 7005))
        + list(range(7051, 7055))
        + list(range(7101, 7105))
        + list(range(7151, 7155))
        + list(range(7201, 7205))
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
                        "artifact_plan_sha256": HASHES[2],
                        "evaluation_identity_sha256": HASHES[3],
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
            "schema_version": "corrected_stageb_registration_v1",
            "registration_id": "synthetic-corrected-stageb",
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
            "evaluation_identities": evaluation_ids,
            "horizon": 50,
            "discount": 0.95,
            "num_actions": 11,
            "sigma": 0.2,
            "evaluator_family": "allee",
            "cpu_profile": "Intel Xeon Platinum 8452Y / xenon-8452Y / one CPU",
            "local_status_label": "LOCAL ARM64 DEVELOPMENT TEST — NOT SCIENTIFIC EVIDENCE",
        },
        "analysis_rules": {
            "schema_version": "corrected_stageb_analysis_rules_v1",
            "primary_estimand": "mean_return_T_minus_mean_return_O",
            "bootstrap": {
                "resamples": 100000,
                "rng": "NumPy PCG64",
                "seed": 20260808,
                "quantiles": [0.025, 0.975],
                "quantile_method": "linear",
                "resampling_unit": "intact paired evaluation episode",
            },
            "residual_sigma_rule": {
                "numeric_threshold": None,
                "ratio_guard": "compute only when Arm O denominator is positive",
                "residual_sigma_floor": 0.02,
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
            "schema_version": "corrected_stageb_code_configuration_hashes_v1",
            "git_commit_sha": "e44c5931f97f73f4805ac128bc18e904ca68469f",
            "source_manifest_sha256": match.group(1),
            "registration_templates_sha256": _registration_templates_hash(repository),
            **EXPECTED_CONFIG_HASHES,
            **EXPECTED_SOURCE_HASHES,
        },
        "task_manifest": {
            "schema_version": "corrected_stageb_task_manifest_v1",
            "task_count": 24,
            "tasks": tasks,
        },
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


def artifact_components(
    method: str, cell: str = "amur_tiger__allee__sigma_0p2"
) -> dict[str, bytes]:
    features = ["state", "context", "time"]
    action_model = {
        "feature_order": features,
        "action_weights": np.arange(33, dtype=np.float64).reshape(3, 11) / 100.0,
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
        cache_sha = add(
            "ricker_fit_cache",
            {
                "cell": cell,
                "candidate_ids": candidate_ids,
                "r": np.linspace(0.1, 0.2, count, dtype=np.float64),
                "capacity": np.linspace(10.0, 20.0, count, dtype=np.float64),
                "residual_sigma": np.full(count, 0.02, dtype=np.float64),
                "survey_scale": 10.0,
            },
        )
        add(
            "residual_process_scales",
            {
                "candidate_ids": candidate_ids,
                "residual_sigma": np.full(count, 0.02, dtype=np.float64),
                "ricker_fit_cache_sha256": cache_sha,
            },
        )
        add(
            "reward_surrogate",
            {
                "cell": cell,
                "source_public_view_sha256": EXPECTED_DATASET_HASHES[cell],
                "feature_order": features,
                "weights": np.array([0.5, -0.25, 0.75], dtype=np.float64),
                "bias": 0.1,
            },
        )
        grids_sha = add(
            "pbvi_grids",
            {
                "abundance_grid": np.array([0.1, 0.5, 1.0], dtype=np.float64),
                "capacity_grid": np.array([0.5, 1.0, 1.5], dtype=np.float64),
                "observation_grid": np.array([0.1, 0.6, 1.2], dtype=np.float64),
            },
        )
        alpha_sha = add(
            "pbvi_alpha_vectors",
            {
                "alpha_vectors": np.arange(9, dtype=np.float64).reshape(3, 3),
                "action_ids": [0, 5, 10],
                "state_grid_sha256": grids_sha,
            },
        )
        candidates_sha = add(
            "pbvi_candidates",
            {"candidate_ids": candidate_ids, "ricker_fit_cache_sha256": cache_sha},
        )
        prior_sha = add(
            "pbvi_prior",
            {
                "candidate_ids": candidate_ids,
                "probabilities": np.full(count, 1.0 / count, dtype=np.float64),
            },
        )
        add(
            "pbvi_policy",
            {
                "alpha_vectors_sha256": alpha_sha,
                "grids_sha256": grids_sha,
                "candidates_sha256": candidates_sha,
                "prior_sha256": prior_sha,
            },
        )
        return result

    add(
        "preprocessing",
        {
            "feature_order": features,
            "offset": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "scale": np.array([1.0, 2.0, 4.0], dtype=np.float64),
            "dtype": "float64",
        },
    )
    add(
        "feature_transformations",
        {
            "feature_order": features,
            "matrix": np.eye(3, dtype=np.float64),
            "bias": np.array([0.1, 0.2, 0.3], dtype=np.float64),
            "dtype": "float64",
        },
    )
    if method != "ensemble_value_disagreement_pessimism":
        add(
            "reward_surrogate",
            {
                "cell": cell,
                "source_public_view_sha256": EXPECTED_DATASET_HASHES[cell],
                "consumer_methods": ["bamcts", "ogsrl", "refplan"],
                "label": "PROSPECTIVELY RECONSTRUCTED MATCHED ARM-O SURROGATE",
                "feature_order": features,
                "weights": np.array([0.5, -0.25, 0.75], dtype=np.float64),
                "bias": 0.1,
            },
            artifact_method="shared_general_methods",
            fit_source="Arm O public-data view only",
        )
        dynamics_sha = add(
            "dynamics_ensemble",
            {
                "feature_order": features,
                "member_ids": list(range(5)),
                "weights": np.arange(15, dtype=np.float64).reshape(5, 3) / 100.0,
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
        add("refplan_behavior_prior", action_model)
        add(
            "planner_configuration",
            {"horizon": 5, "num_sequences": 96, "num_particles": 32, "action_count": 11},
        )
    elif method == "ogsrl":
        add("ogsrl_actor", action_model)
        add(
            "ogsrl_guardian",
            {
                "feature_order": features,
                "reference_points": np.arange(9, dtype=np.float64).reshape(3, 3),
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
                "feature_order": features,
                "member_ids": list(range(5)),
                "weights": np.arange(15, dtype=np.float64).reshape(5, 3) / 100.0,
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
        add("evd_behavior_reference", action_model)
        add(
            "evd_q_members",
            {
                "feature_order": features,
                "member_ids": list(range(20)),
                "q_weights": np.arange(660, dtype=np.float64).reshape(20, 3, 11) / 1000.0,
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
            process_calls_before=timestep,
            process_calls_after=timestep + 1,
            observation_calls_before=timestep,
            observation_calls_after=timestep + 1,
            process_state_before_sha256=HASHES[timestep % len(HASHES)],
            process_state_after_sha256=HASHES[(timestep + 1) % len(HASHES)],
            observation_state_before_sha256=HASHES[(timestep + 2) % len(HASHES)],
            observation_state_after_sha256=HASHES[(timestep + 3) % len(HASHES)],
        )
        records.append(
            StepEvidence(
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
