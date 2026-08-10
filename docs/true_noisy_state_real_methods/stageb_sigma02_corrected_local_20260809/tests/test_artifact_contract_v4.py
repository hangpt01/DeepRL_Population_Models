from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ..artifacts import (
    CanonicalArtifact,
    deterministic_prediction_fixtures,
    validate_complete_artifact_bundle,
)
from ..boundary import reject_forbidden_payload
from ..common import ContractError, sha256_bytes
from ..diagnostics import build_arm_transition_diagnostics
from ..fit_probe import FIT_PROBE_TASKS, project_fitted_policy_components
from ..real_artifacts import _registered_track_imports, load_frozen_fitted_object
from .conftest import HASHES, artifact_components
from .frozen_fixtures import frozen_object_evidence


def _artifact_with_state(payload: bytes, state):
    original = CanonicalArtifact.from_bytes(payload)
    return CanonicalArtifact(original.component, original.method, original.fit_source, state)


def _ecological_state(method="moor_adapted_ricker_misspec_pbvi"):
    payload = artifact_components(method)["ricker_fit_cache"]
    return deepcopy(CanonicalArtifact.from_bytes(payload).state)


def test_ecological_subfloor_and_near_zero_process_scales_preserve_exact_float64_bytes():
    components = artifact_components("moor_adapted_ricker_misspec_pbvi")
    cache = CanonicalArtifact.from_bytes(components["ricker_fit_cache"])
    state = deepcopy(cache.state)
    exact = np.array([5.13e-191], dtype=np.float64)
    state["process_scale"] = exact.copy()
    cache = _artifact_with_state(components["ricker_fit_cache"], state)
    components["ricker_fit_cache"] = cache.to_bytes()
    process = CanonicalArtifact.from_bytes(components["residual_process_scales"])
    process_state = deepcopy(process.state)
    process_state["process_scale"] = exact.copy()
    process_state["ricker_fit_cache_sha256"] = sha256_bytes(components["ricker_fit_cache"])
    components["residual_process_scales"] = _artifact_with_state(
        components["residual_process_scales"], process_state
    ).to_bytes()
    candidates = CanonicalArtifact.from_bytes(components["pbvi_candidates"])
    candidate_state = deepcopy(candidates.state)
    candidate_state["ricker_fit_cache_sha256"] = sha256_bytes(components["ricker_fit_cache"])
    components["pbvi_candidates"] = _artifact_with_state(
        components["pbvi_candidates"], candidate_state
    ).to_bytes()
    policy = CanonicalArtifact.from_bytes(components["pbvi_policy"])
    policy_state = deepcopy(policy.state)
    policy_state["candidates_sha256"] = sha256_bytes(components["pbvi_candidates"])
    components["pbvi_policy"] = _artifact_with_state(
        components["pbvi_policy"], policy_state
    ).to_bytes()
    reloaded = CanonicalArtifact.from_bytes(components["ricker_fit_cache"])
    assert reloaded.state["process_scale"].tobytes() == exact.tobytes()
    validate_complete_artifact_bundle(
        "moor_adapted_ricker_misspec_pbvi",
        components,
        prediction_fixtures=deterministic_prediction_fixtures(
            "moor_adapted_ricker_misspec_pbvi", components
        ),
        expected_component_hashes={key: sha256_bytes(value) for key, value in components.items()},
    )


def test_mortality_dominant_action_is_valid_but_simultaneous_growth_is_rejected():
    state = _ecological_state()
    assert state["growth"][0, 1] == 0.0 and state["mortality"][0, 1] > 0.0
    _artifact_with_state(
        artifact_components("moor_adapted_ricker_misspec_pbvi")["ricker_fit_cache"], state
    ).to_bytes()
    state["growth"][0, 1] = np.float64(0.01)
    with pytest.raises(ContractError, match="both be positive"):
        _artifact_with_state(
            artifact_components("moor_adapted_ricker_misspec_pbvi")["ricker_fit_cache"], state
        ).to_bytes()


def test_old_scalar_r_state_and_missing_growth_mortality_are_rejected():
    payload = artifact_components("moor_adapted_ricker_misspec_pbvi")["ricker_fit_cache"]
    state = _ecological_state()
    state["r"] = np.array([0.1], dtype=np.float64)
    with pytest.raises(ContractError, match="key"):
        _artifact_with_state(payload, state).to_bytes()
    for field in ("growth", "mortality"):
        state = _ecological_state()
        del state[field]
        with pytest.raises(ContractError, match="key"):
            _artifact_with_state(payload, state).to_bytes()


@pytest.mark.parametrize("field", ["capacity_increment", "stocking"])
def test_action_channel_constraints_are_enforced(field):
    payload = artifact_components("moor_adapted_ricker_misspec_pbvi")["ricker_fit_cache"]
    state = _ecological_state()
    state[field][0, 0] = np.float64(0.5)
    with pytest.raises(ContractError, match="incompatible"):
        _artifact_with_state(payload, state).to_bytes()


def test_candidate_labels_and_identifiers_are_closed_and_ordered():
    plus = artifact_components("plus_adapted_ricker_only_pbvi")["ricker_fit_cache"]
    for mutation in ("labels", "identifiers"):
        state = deepcopy(CanonicalArtifact.from_bytes(plus).state)
        if mutation == "labels":
            state["candidate_labels"][1] = state["candidate_labels"][0]
        else:
            state["candidate_ids"][1] = 9
        with pytest.raises(ContractError, match="unique|0..7"):
            _artifact_with_state(plus, state).to_bytes()


def test_process_scale_one_ulp_or_cache_hash_mismatch_is_rejected():
    method = "moor_adapted_ricker_misspec_pbvi"
    for mutation in ("ulp", "hash"):
        components = artifact_components(method)
        process = CanonicalArtifact.from_bytes(components["residual_process_scales"])
        state = deepcopy(process.state)
        if mutation == "ulp":
            state["process_scale"][0] = np.nextafter(state["process_scale"][0], np.float64(np.inf))
        else:
            state["ricker_fit_cache_sha256"] = HASHES[7]
        components["residual_process_scales"] = _artifact_with_state(
            components["residual_process_scales"], state
        ).to_bytes()
        with pytest.raises(ContractError, match="bit-identical|bind"):
            validate_complete_artifact_bundle(
                method,
                components,
                prediction_fixtures=deterministic_prediction_fixtures(method, components),
                expected_component_hashes={
                    key: sha256_bytes(value) for key, value in components.items()
                },
            )


def test_regime_matrix_row_sum_failure_is_rejected():
    payload = artifact_components("moor_adapted_ricker_misspec_pbvi")["ricker_fit_cache"]
    state = _ecological_state()
    state["regime_matrix"][0, 0, 0] += np.float64(1e-8)
    with pytest.raises(ContractError, match="row stochastic"):
        _artifact_with_state(payload, state).to_bytes()


def test_method_scoped_0019_diagnostics_accept_ecology_and_reject_general():
    ecological = build_arm_transition_diagnostics(
        registration_sha256=HASHES[0],
        method="moor_adapted_ricker_misspec_pbvi",
        cell="amur_tiger__allee__sigma_0p2",
        arm="O",
        process_scale=[0.019],
        artifact_hashes=[HASHES[1]],
    )
    assert ecological["process_scale"] == [0.019]
    assert ecological["below_general_floor"] == [True]
    assert ecological["applicable_floor"] is None
    assert ecological["floor_semantics"] == "general_learned_dynamics_only"
    with pytest.raises(ContractError, match="general learned-dynamics"):
        build_arm_transition_diagnostics(
            registration_sha256=HASHES[0],
            method="refplan",
            cell="amur_tiger__allee__sigma_0p2",
            arm="O",
            residual_sigma=[0.019] * 5,
            artifact_hashes=list(HASHES[:5]),
        )


class FakeModel:
    def __init__(self, label):
        self.candidate_id = label
        self.form = "ricker"
        self.action_channels = (
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
        )
        self.growth = np.zeros(11, dtype=np.float64)
        self.growth[0] = 0.2
        self.mortality = np.zeros(11, dtype=np.float64)
        self.mortality[1] = 0.03
        self.capacity_increment = np.zeros(11, dtype=np.float64)
        self.capacity_increment[3] = 0.5
        self.stocking = np.zeros(11, dtype=np.float64)
        self.stocking[5] = 0.25
        self.process_scale = np.float64(5.13e-191)
        self.observation_scale = np.float64(0.2)
        self.survey_scale = np.float64(10.0)
        self.initial_capacity = np.float64(20.0)
        self.capacity_ceiling = np.float64(25.0)
        self.reset_log_mean = np.float64(0.1)
        self.reset_log_scale = np.float64(0.3)
        self.depensation_thresholds = np.array([2.0, 4.0], dtype=np.float64)
        self.theta_exponent = np.float64(1.5)
        self.regime_multipliers = np.array([0.9, 1.1], dtype=np.float64)
        self.regime_matrix = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)

    def parameter_hash(self):
        return sha256_bytes(self.candidate_id.encode())


def _surrogate():
    return SimpleNamespace(reward_coefficients=np.arange(20, dtype=np.float64) / 100.0)


def _behavior():
    return SimpleNamespace(
        weights=np.arange(121, dtype=np.float64).reshape(11, 11) / 100.0,
        mean=np.zeros(11, dtype=np.float64),
        scale=np.ones(11, dtype=np.float64),
    )


def _fake_policy(method, cell):
    pop_id = "pop_tiger" if cell.startswith("amur") else "pop_fox"
    context = SimpleNamespace(pop_id=pop_id, surrogate=_surrogate())
    if method.startswith(("plus_", "moor_")):
        count = 8 if method.startswith("plus_") else 1
        models = [FakeModel(f"model_{index:03d}") for index in range(count)]
        fits = [SimpleNamespace(model=model) for model in models]
        pomdps = [
            SimpleNamespace(
                model=model,
                abundance_grid=np.array([0.0, 1.0, 2.0], dtype=np.float64),
                config=SimpleNamespace(capacity_bins=3),
            )
            for model in models
        ]
        if method.startswith("plus_"):
            return SimpleNamespace(
                public_context=context,
                candidate_bank=SimpleNamespace(
                    fits=fits, initial_weights=np.full(count, 1.0 / count, dtype=np.float64)
                ),
                pomdps=pomdps,
            )
        return SimpleNamespace(public_context=context, fit_result=fits[0], pomdp=pomdps[0])
    policy = SimpleNamespace(public_context=context)
    if method != "ensemble_value_disagreement_pessimism":
        members = [
            SimpleNamespace(
                coefficients=np.arange(36, dtype=np.float64) / 100.0,
                residual_sigma=np.float64(0.02 + index * 0.001),
            )
            for index in range(5)
        ]
        policy.dynamics = SimpleNamespace(members=members)
    if method == "refplan":
        policy.policy_prior = _behavior()
        policy.planner_cfg = SimpleNamespace(horizon=5, sequences=96, particles=32)
    elif method == "ogsrl":
        policy.actor_weights = np.arange(33, dtype=np.float64).reshape(11, 3) / 100.0
        policy.guardian = SimpleNamespace(
            anchors=np.arange(39, dtype=np.float64).reshape(3, 13), threshold=0.3
        )
        policy.s_low = 10.0
        policy.safety_budget = 0.1
    elif method == "bamcts":
        policy.depth = 8
        policy.simulations = 256
    else:
        policy.behavior_model = _behavior()
        policy.q_members = [
            SimpleNamespace(q_weights=np.arange(121, dtype=np.float64).reshape(11, 11))
            for _ in range(20)
        ]
        policy.disagreement_penalty = 0.5
    return policy


@pytest.mark.parametrize("task", FIT_PROBE_TASKS)
def test_all_twelve_fit_probe_adapter_paths_are_source_class_compatible(task):
    components = project_fitted_policy_components(
        task.task_index, _fake_policy(task.method, task.cell), HASHES[0]
    )
    payloads = {name: artifact.to_bytes() for name, artifact in components.items()}
    receipt = validate_complete_artifact_bundle(
        task.method,
        payloads,
        prediction_fixtures=deterministic_prediction_fixtures(task.method, payloads),
        expected_component_hashes={name: sha256_bytes(value) for name, value in payloads.items()},
    )
    assert receipt.reload_parity
    if task.method == "ogsrl":
        actor = CanonicalArtifact.from_bytes(payloads["ogsrl_actor"]).state
        source = _fake_policy(task.method, task.cell).actor_weights
        assert np.array_equal(actor["action_weights"], source[:, 1:].T)
        assert np.array_equal(actor["action_bias"], source[:, 0])


@pytest.mark.parametrize("task", FIT_PROBE_TASKS)
def test_all_twelve_fit_probe_paths_project_real_frozen_source_classes(task):
    repository_root = Path(__file__).resolve().parents[4]
    fitted_payload, _ = frozen_object_evidence(task.method, task.cell, HASHES[0])
    track = "ecological" if task.method.startswith(("plus_", "moor_")) else "general"
    with _registered_track_imports(track, repository_root):
        policy = load_frozen_fitted_object(fitted_payload, repository_root=repository_root)
        components = project_fitted_policy_components(task.task_index, policy, HASHES[0])
    payloads = {name: artifact.to_bytes() for name, artifact in components.items()}
    receipt = validate_complete_artifact_bundle(
        task.method,
        payloads,
        prediction_fixtures=deterministic_prediction_fixtures(task.method, payloads),
        expected_component_hashes={name: sha256_bytes(value) for name, value in payloads.items()},
    )
    assert receipt.reload_parity


def test_information_boundary_remains_fail_closed():
    with pytest.raises(ContractError, match="forbidden"):
        reject_forbidden_payload({"runtime_next_states": [1.0]})
