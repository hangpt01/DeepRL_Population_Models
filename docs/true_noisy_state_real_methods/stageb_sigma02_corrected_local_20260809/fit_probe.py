"""Tracked, evaluator-free V4 fitted-policy projection and probe publication.

The deployment caller constructs a fitted policy through the frozen method pipeline and then
hands that object to this module. This module never constructs an evaluator, opens truth,
accepts runtime next state, or calculates a return.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import (
    ECOLOGICAL_METHODS,
    EQUATION_VERSION,
    MATCHED_SURROGATE_LABEL,
    REGIME_LAW_VERSION,
    RICKER_FIT_CACHE_SCHEMA,
    SURROGATE_METHODS,
    CanonicalArtifact,
    deterministic_prediction_fixtures,
    fixture_manifest,
    validate_complete_artifact_bundle,
)
from .common import (
    ContractError,
    canonical_json_bytes,
    require_git_sha,
    sha256_bytes,
    sha256_file,
)
from .publication import create_task_staging, publish_once, write_bytes_fsync
from .real_artifacts import (
    scientific_component_for_method,
    validate_frozen_fitted_object_roundtrip_with_diagnostic,
)


EVD_METHOD = "ensemble_value_disagreement_pessimism"
METHODS = (
    "plus_adapted_ricker_only_pbvi",
    "moor_adapted_ricker_misspec_pbvi",
    "refplan",
    "ogsrl",
    "bamcts",
    EVD_METHOD,
)
CELLS = (
    ("amur_tiger__allee__sigma_0p2", "amur_tiger", "Amur tiger"),
    ("crab_eating_fox__allee__sigma_0p2", "crab_eating_fox", "Crab-eating fox"),
)


@dataclass(frozen=True)
class FitProbeTask:
    task_index: int
    cell: str
    cell_slug: str
    population: str
    method: str
    interpreter_role: str


FIT_PROBE_TASKS = tuple(
    FitProbeTask(
        cell_index * len(METHODS) + method_index,
        cell,
        slug,
        population,
        method,
        "ecological_paper_faithful" if method in ECOLOGICAL_METHODS else "general_registered",
    )
    for cell_index, (cell, slug, population) in enumerate(CELLS)
    for method_index, method in enumerate(METHODS)
)


def registered_fit_probe_task(task_index: int) -> FitProbeTask:
    if isinstance(task_index, bool) or not isinstance(task_index, int):
        raise ContractError("fit-probe task index must be an integer")
    if not 0 <= task_index < len(FIT_PROBE_TASKS):
        raise ContractError("fit-probe task index must be in 0..11")
    return FIT_PROBE_TASKS[task_index]


def _artifact(component: str, method: str, fit_source: str, state: Mapping[str, Any]):
    return CanonicalArtifact(component, method, fit_source, state)


def _feature_orders(policy: Any) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    population_token = str(policy.public_context.pop_id)
    reward = [
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
    dynamics = [
        "log_abundance",
        "log_abundance_squared",
        *(f"action_{index}" for index in range(11)),
        *(f"action_{index}_times_log_abundance" for index in range(11)),
        *(f"action_{index}_times_log_abundance_squared" for index in range(11)),
    ]
    belief = [
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
    actor = ["log_current_observation", "log_current_observation_squared"]
    guardian = [*actor, *(f"action_{index}" for index in range(11))]
    return reward, dynamics, belief, actor, guardian


def _standardized_action_projection(model: Any) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(model.weights, dtype=np.float64)
    mean = np.asarray(model.mean, dtype=np.float64)
    scale = np.asarray(model.scale, dtype=np.float64)
    if weights.shape != (11, 11) or mean.shape != (11,) or scale.shape != (11,):
        raise ContractError("real calibrated behavior model shape mismatch")
    action_weights = (weights[:, 1:] / scale[1:]).T
    action_bias = weights[:, 0] / scale[0] - np.sum(weights[:, 1:] * mean[1:] / scale[1:], axis=1)
    return action_weights, action_bias


def _reward_artifact(
    policy: Any, method: str, cell: str, public_view_sha256: str, features: list[str]
) -> CanonicalArtifact:
    surrogate = policy.public_context.surrogate
    state: dict[str, Any] = {
        "cell": cell,
        "source_public_view_sha256": public_view_sha256,
        "feature_order": features,
        "weights": np.asarray(surrogate.reward_coefficients[1:], dtype=np.float64),
        "bias": float(surrogate.reward_coefficients[0]),
    }
    artifact_method = method
    fit_source = "Arm O public-data view only"
    if method in SURROGATE_METHODS:
        artifact_method = "shared_general_methods"
        state.update(
            consumer_methods=sorted(SURROGATE_METHODS),
            label=MATCHED_SURROGATE_LABEL,
        )
    return _artifact("reward_surrogate", artifact_method, fit_source, state)


def project_ecological_fitted_cache(
    method: str, cell: str, models: Sequence[Any]
) -> CanonicalArtifact:
    if method not in ECOLOGICAL_METHODS:
        raise ContractError("ecological fitted-cache projection requires PLUS or MOOR")
    expected_count = 8 if method.startswith("plus_") else 1
    if len(models) != expected_count:
        raise ContractError("real ecological fitted-model count mismatch")
    action_count = len(np.asarray(models[0].growth))
    if action_count < 1 or any(len(np.asarray(model.growth)) != action_count for model in models):
        raise ContractError("real ecological action count mismatch")

    def stack(field: str) -> np.ndarray:
        values = [np.asarray(getattr(model, field), dtype=np.float64) for model in models]
        return np.stack(values).astype(np.float64, copy=False)

    state = {
        "schema_version": RICKER_FIT_CACHE_SCHEMA,
        "cell": cell,
        "candidate_ids": list(range(expected_count)),
        "candidate_labels": [str(model.candidate_id) for model in models],
        "form": [str(model.form) for model in models],
        "action_channels": [list(model.action_channels) for model in models],
        "growth": stack("growth"),
        "mortality": stack("mortality"),
        "capacity_increment": stack("capacity_increment"),
        "stocking": stack("stocking"),
        "process_scale": np.asarray([model.process_scale for model in models], dtype=np.float64),
        "observation_scale": np.asarray(
            [model.observation_scale for model in models], dtype=np.float64
        ),
        "survey_scale": np.asarray([model.survey_scale for model in models], dtype=np.float64),
        "initial_capacity": np.asarray(
            [model.initial_capacity for model in models], dtype=np.float64
        ),
        "capacity_ceiling": np.asarray(
            [model.capacity_ceiling for model in models], dtype=np.float64
        ),
        "reset_log_mean": np.asarray([model.reset_log_mean for model in models], dtype=np.float64),
        "reset_log_scale": np.asarray(
            [model.reset_log_scale for model in models], dtype=np.float64
        ),
        "depensation_thresholds": stack("depensation_thresholds"),
        "theta_exponent": np.asarray([model.theta_exponent for model in models], dtype=np.float64),
        "regime_multipliers": stack("regime_multipliers"),
        "regime_matrix": stack("regime_matrix"),
        "equation_version": EQUATION_VERSION,
        "regime_law_version": REGIME_LAW_VERSION,
        "parameter_hashes": [str(model.parameter_hash()) for model in models],
    }
    return _artifact(
        "ricker_fit_cache", method, "ordered Arm O public survey/action histories", state
    )


def _ecological_components(
    method: str, cell: str, policy: Any, public_view_sha256: str
) -> dict[str, CanonicalArtifact]:
    plus = method.startswith("plus_")
    fits = list(policy.candidate_bank.fits) if plus else [policy.fit_result]
    models = [item.model for item in fits]
    pomdps = list(policy.pomdps) if plus else [policy.pomdp]
    cache = project_ecological_fitted_cache(method, cell, models)
    cache_sha = sha256_bytes(cache.to_bytes())
    candidate_ids = list(range(len(models)))
    process_scale = np.asarray([model.process_scale for model in models], dtype=np.float64)
    result = {
        "ricker_fit_cache": cache,
        "residual_process_scales": _artifact(
            "residual_process_scales",
            method,
            "ordered Arm O public survey/action histories",
            {
                "candidate_ids": candidate_ids,
                "process_scale": process_scale,
                "ricker_fit_cache_sha256": cache_sha,
            },
        ),
    }
    reward_features = _feature_orders(policy)[0]
    result["reward_surrogate"] = _reward_artifact(
        policy, method, cell, public_view_sha256, reward_features
    )
    pomdp = pomdps[0]
    capacity_grid = np.linspace(
        pomdp.model.initial_capacity,
        pomdp.model.capacity_ceiling,
        pomdp.config.capacity_bins,
        dtype=np.float64,
    )
    result["pbvi_grids"] = _artifact(
        "pbvi_grids",
        method,
        "registered discretization of fitted Ricker POMDP",
        {
            "abundance_grid": np.asarray(pomdp.abundance_grid, dtype=np.float64),
            "capacity_grid": capacity_grid,
            "observation_grid": np.asarray(pomdp.abundance_grid, dtype=np.float64)
            * np.float64(pomdp.model.survey_scale),
        },
    )
    result["pbvi_candidates"] = _artifact(
        "pbvi_candidates",
        method,
        "complete fitted Ricker candidate set",
        {"candidate_ids": candidate_ids, "ricker_fit_cache_sha256": cache_sha},
    )
    if plus:
        result["pbvi_prior"] = _artifact(
            "pbvi_prior",
            method,
            "registered candidate prior",
            {
                "candidate_ids": candidate_ids,
                "probabilities": np.asarray(
                    policy.candidate_bank.initial_weights, dtype=np.float64
                ),
            },
        )
    policy_state = {
        "grids_sha256": sha256_bytes(result["pbvi_grids"].to_bytes()),
        "candidates_sha256": sha256_bytes(result["pbvi_candidates"].to_bytes()),
    }
    if plus:
        policy_state["prior_sha256"] = sha256_bytes(result["pbvi_prior"].to_bytes())
    result["pbvi_policy"] = _artifact(
        "pbvi_policy", method, "fitted Ricker POMDP and registered planner", policy_state
    )
    return result


def _general_components(
    method: str, cell: str, policy: Any, public_view_sha256: str
) -> dict[str, CanonicalArtifact]:
    reward_features, dynamics_features, belief_features, actor_features, guardian_features = (
        _feature_orders(policy)
    )
    result: dict[str, CanonicalArtifact] = {}
    if method in SURROGATE_METHODS:
        result["reward_surrogate"] = _reward_artifact(
            policy, method, cell, public_view_sha256, reward_features
        )
        members = list(policy.dynamics.members)
        if len(members) != 5:
            raise ContractError("general dynamics ensemble must contain five members")
        weights = np.stack(
            [np.asarray(member.coefficients[1:], dtype=np.float64) for member in members]
        )
        bias = np.asarray([member.coefficients[0] for member in members], dtype=np.float64)
        residual_sigma = np.asarray([member.residual_sigma for member in members], dtype=np.float64)
        result["dynamics_ensemble"] = _artifact(
            "dynamics_ensemble",
            method,
            "arm-specific public state-input fit",
            {
                "feature_order": dynamics_features,
                "member_ids": list(range(5)),
                "weights": weights,
                "bias": bias,
                "residual_sigma": residual_sigma,
            },
        )
        dynamics_sha = sha256_bytes(result["dynamics_ensemble"].to_bytes())
        result["residual_process_scales"] = _artifact(
            "residual_process_scales",
            method,
            "arm-specific public state-input fit",
            {
                "member_ids": list(range(5)),
                "residual_sigma": residual_sigma.copy(),
                "dynamics_ensemble_sha256": dynamics_sha,
            },
        )
    if method == "refplan":
        weights, bias = _standardized_action_projection(policy.policy_prior)
        result["refplan_behavior_prior"] = _artifact(
            "refplan_behavior_prior",
            method,
            "arm-specific public state-input fit",
            {"feature_order": belief_features, "action_weights": weights, "action_bias": bias},
        )
        result["planner_configuration"] = _artifact(
            "planner_configuration",
            method,
            "registered configuration",
            {
                "horizon": int(policy.planner_cfg.horizon),
                "num_sequences": int(policy.planner_cfg.sequences),
                "num_particles": int(policy.planner_cfg.particles),
                "action_count": 11,
            },
        )
    elif method == "ogsrl":
        actor = np.asarray(policy.actor_weights, dtype=np.float64)
        if actor.shape != (11, 3):
            raise ContractError("real OGSRL actor must have shape (11, 3)")
        result["ogsrl_actor"] = _artifact(
            "ogsrl_actor",
            method,
            "arm-specific public state-input fit",
            {
                "feature_order": actor_features,
                "action_weights": actor[:, 1:].T.copy(),
                "action_bias": actor[:, 0].copy(),
            },
        )
        result["ogsrl_guardian"] = _artifact(
            "ogsrl_guardian",
            method,
            "arm-specific public state-input fit",
            {
                "feature_order": guardian_features,
                "reference_points": np.asarray(policy.guardian.anchors, dtype=np.float64),
                "support_radius": float(policy.guardian.threshold),
            },
        )
        result["ogsrl_safety_calibration"] = _artifact(
            "ogsrl_safety_calibration",
            method,
            "arm-specific public state-input fit",
            {
                "low_abundance_scale": float(policy.s_low),
                "safety_budget": float(policy.safety_budget),
                "guardian_threshold": float(policy.guardian.threshold),
            },
        )
    elif method == "bamcts":
        dynamics = result["dynamics_ensemble"].state
        result["bamcts_model_bank"] = _artifact(
            "bamcts_model_bank",
            method,
            "logical alias of fitted dynamics ensemble",
            {
                "feature_order": list(dynamics["feature_order"]),
                "member_ids": list(range(5)),
                "weights": np.asarray(dynamics["weights"]).copy(),
                "bias": np.asarray(dynamics["bias"]).copy(),
            },
        )
        result["bamcts_search_configuration"] = _artifact(
            "bamcts_search_configuration",
            method,
            "registered configuration",
            {
                "depth": int(policy.depth),
                "simulations": int(policy.simulations),
                "posterior_likelihood_sigma": float(np.mean(dynamics["residual_sigma"])),
                "action_count": 11,
            },
        )
    elif method == EVD_METHOD:
        weights, bias = _standardized_action_projection(policy.behavior_model)
        result["evd_behavior_reference"] = _artifact(
            "evd_behavior_reference",
            method,
            "arm-specific raw logged reward fit",
            {"feature_order": belief_features, "action_weights": weights, "action_bias": bias},
        )
        q_members = [np.asarray(member.q_weights, dtype=np.float64) for member in policy.q_members]
        if len(q_members) != 20 or any(value.shape != (11, 11) for value in q_members):
            raise ContractError("real EVD Q-member shape mismatch")
        result["evd_q_members"] = _artifact(
            "evd_q_members",
            method,
            "arm-specific raw logged reward fit",
            {
                "feature_order": belief_features,
                "member_ids": list(range(20)),
                "q_weights": np.stack([value[:, 1:].T for value in q_members]),
                "q_bias": np.stack([value[:, 0] for value in q_members]),
            },
        )
        result["evd_policy_configuration"] = _artifact(
            "evd_policy_configuration",
            method,
            "registered configuration",
            {
                "disagreement_penalty": float(policy.disagreement_penalty),
                "action_count": 11,
                "objective": "raw_logged_rewards",
            },
        )
    else:
        raise ContractError("unknown registered general fit-probe method")
    return result


def project_fitted_policy_components(
    task_index: int, policy: Any, public_view_sha256: str
) -> dict[str, CanonicalArtifact]:
    task = registered_fit_probe_task(task_index)
    if task.method in ECOLOGICAL_METHODS:
        return _ecological_components(task.method, task.cell, policy, public_view_sha256)
    return _general_components(task.method, task.cell, policy, public_view_sha256)


def _diagnostic_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "data_base64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        }
    if isinstance(value, Mapping):
        return {str(key): _diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def publish_fit_probe(
    *,
    task_index: int,
    policy: Any,
    public_view_sha256: str,
    controlling_commit: str,
    repository_root: Path,
    output_root: Path,
    operation_args: Sequence[Any],
    operation_kwargs: Mapping[str, Any] | None = None,
) -> Path:
    """Project, validate, replay, and atomically publish one evaluator-free fit probe."""

    task = registered_fit_probe_task(task_index)
    commit = require_git_sha(controlling_commit, "fit-probe controlling commit")
    script_sha256 = sha256_file(Path(__file__))
    staging, target = create_task_staging(output_root, f"fit-probe-{task_index:02d}")
    components = project_fitted_policy_components(task_index, policy, public_view_sha256)
    diagnostic = {
        "schema_version": "corrected_stageb_source_projection_diagnostic_v4",
        "task": task.__dict__,
        "controlling_commit": commit,
        "fit_probe_script_sha256": script_sha256,
        "components": {
            name: _diagnostic_value(artifact.state) for name, artifact in components.items()
        },
        "written_before_component_validation": True,
        "evaluator_constructed": False,
        "truth_accessed": False,
        "runtime_next_states_accessed": False,
        "returns_calculated": False,
    }
    write_bytes_fsync(
        staging / "SOURCE_PROJECTION_DIAGNOSTIC.json", canonical_json_bytes(diagnostic)
    )

    payloads = {name: artifact.to_bytes() for name, artifact in components.items()}
    component_hashes = {name: sha256_bytes(payload) for name, payload in payloads.items()}
    fixtures = deterministic_prediction_fixtures(task.method, payloads)
    bundle = validate_complete_artifact_bundle(
        task.method,
        payloads,
        prediction_fixtures=fixtures,
        expected_component_hashes=component_hashes,
    )
    for name, payload in payloads.items():
        write_bytes_fsync(staging / f"artifact-{name}.json", payload)
    fitted_payload, replay, replay_diagnostic = (
        validate_frozen_fitted_object_roundtrip_with_diagnostic(
            scientific_component_for_method(task.method),
            policy,
            operation="act",
            args=operation_args,
            kwargs=operation_kwargs or {},
            repository_root=repository_root,
        )
    )
    write_bytes_fsync(staging / "FROZEN_FITTED_OBJECT.json", fitted_payload)
    write_bytes_fsync(staging / "FRESH_RELOAD_REPLAY.json", canonical_json_bytes(replay))
    write_bytes_fsync(
        staging / "FROZEN_REPLAY_DIAGNOSTIC.json",
        canonical_json_bytes(replay_diagnostic),
    )
    receipt = {
        "schema_version": "corrected_stageb_fit_probe_receipt_v4",
        "task": task.__dict__,
        "controlling_commit": commit,
        "fit_probe_script_sha256": script_sha256,
        "public_view_sha256": public_view_sha256,
        "component_hashes": component_hashes,
        "bundle_sha256": bundle.bundle_sha256,
        "fixture_manifest_sha256": bundle.fixture_manifest_sha256,
        "fixture_manifest": fixture_manifest(components, fixtures),
        "frozen_fitted_object_sha256": sha256_bytes(fitted_payload),
        "fresh_reload_replay_sha256": sha256_bytes(canonical_json_bytes(replay)),
        "evaluator_constructed": False,
        "truth_accessed": False,
        "runtime_next_states_accessed": False,
        "returns_calculated": False,
        "result": "PASS",
    }
    write_bytes_fsync(staging / "FIT_PROBE_RECEIPT.json", canonical_json_bytes(receipt))
    required = [
        "SOURCE_PROJECTION_DIAGNOSTIC.json",
        "FROZEN_FITTED_OBJECT.json",
        "FRESH_RELOAD_REPLAY.json",
        "FROZEN_REPLAY_DIAGNOSTIC.json",
        "FIT_PROBE_RECEIPT.json",
        *(f"artifact-{name}.json" for name in payloads),
    ]
    publish_once(
        staging=staging,
        target=target,
        required_files=required,
        validator=lambda _root: {
            "schema_version": "corrected_stageb_fit_probe_publication_v4",
            "task_index": task.task_index,
            "controlling_commit": commit,
            "fit_probe_script_sha256": script_sha256,
            "result": "PASS",
        },
    )
    return target
