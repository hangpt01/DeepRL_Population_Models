"""Canonical fitted-artifact, matched-surrogate, and shared-policy contracts."""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from docs.true_noisy_state_real_methods.i2b_fasttrack_integration_canary_20260808.fasttrack_wrappers import (  # noqa: E501
    IntegrationViolation,
    direct_point_mass_latent_belief,
)

from .common import (
    ContractError,
    canonical_json_bytes,
    require_exact_keys,
    require_finite,
    require_nonempty_string,
    require_sha256,
    sha256_bytes,
    strict_json_loads,
)


MATCHED_SURROGATE_LABEL = "PROSPECTIVELY RECONSTRUCTED MATCHED ARM-O SURROGATE"
EVD_METHOD = "ensemble_value_disagreement_pessimism"
SURROGATE_METHODS = frozenset({"refplan", "ogsrl", "bamcts"})
ECOLOGICAL_METHODS = frozenset(
    {"plus_adapted_ricker_only_pbvi", "moor_adapted_ricker_misspec_pbvi"}
)

REQUIRED_COMPONENTS: dict[str, frozenset[str]] = {
    "plus_adapted_ricker_only_pbvi": frozenset(
        {
            "ricker_fit_cache",
            "residual_process_scales",
            "reward_surrogate",
            "pbvi_policy",
            "pbvi_grids",
            "pbvi_candidates",
            "pbvi_prior",
        }
    ),
    "moor_adapted_ricker_misspec_pbvi": frozenset(
        {
            "ricker_fit_cache",
            "residual_process_scales",
            "reward_surrogate",
            "pbvi_policy",
            "pbvi_grids",
            "pbvi_candidates",
        }
    ),
    "refplan": frozenset(
        {
            "reward_surrogate",
            "dynamics_ensemble",
            "residual_process_scales",
            "refplan_behavior_prior",
            "planner_configuration",
        }
    ),
    "ogsrl": frozenset(
        {
            "reward_surrogate",
            "dynamics_ensemble",
            "residual_process_scales",
            "ogsrl_actor",
            "ogsrl_guardian",
            "ogsrl_safety_calibration",
        }
    ),
    "bamcts": frozenset(
        {
            "reward_surrogate",
            "dynamics_ensemble",
            "residual_process_scales",
            "bamcts_model_bank",
            "bamcts_search_configuration",
        }
    ),
    EVD_METHOD: frozenset(
        {
            "evd_behavior_reference",
            "evd_q_members",
            "evd_policy_configuration",
        }
    ),
}

GENERAL_METHODS = frozenset({"refplan", "ogsrl", "bamcts", EVD_METHOD})
FEATURE_COMPONENTS = frozenset(
    {
        "reward_surrogate",
        "dynamics_ensemble",
        "refplan_behavior_prior",
        "ogsrl_actor",
        "ogsrl_guardian",
        "bamcts_model_bank",
        "evd_behavior_reference",
        "evd_q_members",
    }
)
FIXTURE_ROW_COUNT = 3
FIXTURE_GENERATOR_RECIPE = (
    "sha256 of UTF-8 corrected-stageb-component-fixture-v1\\0component-name\\0"
    "row-decimal\\0column-decimal\\0feature-name; take the first "
    "eight digest bytes as an unsigned big-endian integer, reduce modulo 2000001, "
    "subtract 1000000, and divide by 65536 (an exactly representable power of two); "
    "serialize arrays as C-order little-endian float64"
)
FIXTURE_MANIFEST_SCHEMA = "corrected_stageb_prediction_fixture_manifest_v1"
RICKER_FIT_CACHE_SCHEMA = "corrected_stageb_ricker_fit_cache_v2"
EQUATION_VERSION = "adapted_mechanistic_v2"
REGIME_LAW_VERSION = "discrete_current_then_switch_v1"
REGISTERED_MODEL_FORMS = frozenset({"ricker", "allee", "theta", "regime"})
REGISTERED_ACTION_CHANNELS = frozenset({"none", "rate", "capacity", "rate+capacity", "state"})


def _state_keys(state: Mapping[str, Any], expected: set[str], component: str) -> None:
    require_exact_keys(state, expected, f"{component} fitted state")


def _array(
    value: Any,
    field: str,
    *,
    ndim: int,
    shape: tuple[int | None, ...] | None = None,
    positive: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype("float64") or array.ndim != ndim or not np.isfinite(array).all():
        raise ContractError(f"{field} must be a finite float64 array with ndim={ndim}")
    if shape is not None:
        if len(shape) != array.ndim or any(
            expected is not None and array.shape[index] != expected
            for index, expected in enumerate(shape)
        ):
            raise ContractError(f"{field} shape mismatch")
    if positive and np.any(array <= 0.0):
        raise ContractError(f"{field} must be strictly positive")
    return array


def _strings(
    value: Any,
    field: str,
    *,
    expected_length: int | None = None,
    unique: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{field} must be a nonempty list")
    parsed = tuple(require_nonempty_string(item, field) for item in value)
    if unique and len(set(parsed)) != len(parsed):
        raise ContractError(f"{field} values must be unique")
    if expected_length is not None and len(parsed) != expected_length:
        raise ContractError(f"{field} length mismatch")
    return parsed


def _member_ids(value: Any, field: str, expected_count: int) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ContractError(f"{field} must be ordered integer identifiers")
    parsed = tuple(value)
    if parsed != tuple(range(expected_count)):
        raise ContractError(f"{field} must equal 0..{expected_count - 1}")
    return parsed


def _feature_order(state: Mapping[str, Any], component: str) -> tuple[str, ...]:
    return _strings(state["feature_order"], f"{component}.feature_order")


def _validate_action_model(state: Mapping[str, Any], component: str) -> None:
    _state_keys(state, {"feature_order", "action_weights", "action_bias"}, component)
    features = _feature_order(state, component)
    _array(
        state["action_weights"],
        f"{component}.action_weights",
        ndim=2,
        shape=(len(features), 11),
    )
    _array(state["action_bias"], f"{component}.action_bias", ndim=1, shape=(11,))


def _validate_artifact_state(artifact: "CanonicalArtifact") -> None:
    component = artifact.component
    method = artifact.method
    state = artifact.state
    if not isinstance(state, Mapping) or not state:
        raise ContractError("fitted artifact state cannot be empty")
    if component == "reward_surrogate":
        common = {"cell", "source_public_view_sha256", "feature_order", "weights", "bias"}
        if method == "shared_general_methods":
            _state_keys(state, common | {"consumer_methods", "label"}, component)
            if tuple(state["consumer_methods"]) != tuple(sorted(SURROGATE_METHODS)):
                raise ContractError("matched surrogate consumer binding mismatch")
            if state["label"] != MATCHED_SURROGATE_LABEL:
                raise ContractError("matched surrogate label mismatch")
        elif method in ECOLOGICAL_METHODS:
            _state_keys(state, common, component)
        else:
            raise ContractError("reward surrogate method binding is invalid")
        require_nonempty_string(state["cell"], "reward_surrogate.cell")
        require_sha256(state["source_public_view_sha256"], "reward surrogate source public view")
        features = _feature_order(state, component)
        _array(state["weights"], "reward_surrogate.weights", ndim=1, shape=(len(features),))
        require_finite(state["bias"], "reward_surrogate.bias")
    elif component == "dynamics_ensemble":
        _state_keys(
            state,
            {"feature_order", "member_ids", "weights", "bias", "residual_sigma"},
            component,
        )
        features = _feature_order(state, component)
        _member_ids(state["member_ids"], "dynamics_ensemble.member_ids", 5)
        _array(state["weights"], "dynamics_ensemble.weights", ndim=2, shape=(5, len(features)))
        _array(state["bias"], "dynamics_ensemble.bias", ndim=1, shape=(5,))
        residuals = _array(
            state["residual_sigma"], "dynamics_ensemble.residual_sigma", ndim=1, shape=(5,)
        )
        if np.any(residuals < 0.02):
            raise ContractError("dynamics residual_sigma below frozen floor")
    elif component == "residual_process_scales":
        if method in ECOLOGICAL_METHODS:
            count = 8 if method.startswith("plus_") else 1
            _state_keys(
                state,
                {"candidate_ids", "process_scale", "ricker_fit_cache_sha256"},
                component,
            )
            _member_ids(state["candidate_ids"], "residual candidate_ids", count)
            require_sha256(state["ricker_fit_cache_sha256"], "residual Ricker cache")
            process_scale = _array(
                state["process_scale"],
                "residual_process_scales.process_scale",
                ndim=1,
                shape=(count,),
            )
            if np.any(process_scale < 0.0):
                raise ContractError("ecological process_scale must be nonnegative")
        else:
            count = 5
            _state_keys(
                state,
                {"member_ids", "residual_sigma", "dynamics_ensemble_sha256"},
                component,
            )
            _member_ids(state["member_ids"], "residual member_ids", count)
            require_sha256(state["dynamics_ensemble_sha256"], "residual dynamics ensemble")
            residuals = _array(
                state["residual_sigma"],
                "residual_process_scales.residual_sigma",
                ndim=1,
                shape=(count,),
            )
            if np.any(residuals < 0.02):
                raise ContractError("general learned-dynamics residual_sigma below frozen floor")
    elif component in {"refplan_behavior_prior", "ogsrl_actor", "evd_behavior_reference"}:
        _validate_action_model(state, component)
    elif component == "planner_configuration":
        _state_keys(state, {"horizon", "num_sequences", "num_particles", "action_count"}, component)
        if state != {"horizon": 5, "num_sequences": 96, "num_particles": 32, "action_count": 11}:
            raise ContractError("RefPlan planner configuration mismatch")
    elif component == "ogsrl_guardian":
        _state_keys(state, {"feature_order", "reference_points", "support_radius"}, component)
        features = _feature_order(state, component)
        points = _array(
            state["reference_points"],
            "ogsrl_guardian.reference_points",
            ndim=2,
            shape=(None, len(features)),
        )
        if (
            points.shape[0] < 2
            or require_finite(
                state["support_radius"], "ogsrl_guardian.support_radius", nonnegative=True
            )
            <= 0.0
        ):
            raise ContractError("OGSRL guardian fitted support is incomplete")
    elif component == "ogsrl_safety_calibration":
        _state_keys(
            state,
            {"low_abundance_scale", "safety_budget", "guardian_threshold"},
            component,
        )
        for field in state:
            if (
                require_finite(state[field], f"ogsrl_safety_calibration.{field}", nonnegative=True)
                <= 0.0
            ):
                raise ContractError("OGSRL safety calibration must be strictly positive")
    elif component == "bamcts_model_bank":
        _state_keys(state, {"feature_order", "member_ids", "weights", "bias"}, component)
        features = _feature_order(state, component)
        _member_ids(state["member_ids"], "bamcts_model_bank.member_ids", 5)
        _array(state["weights"], "bamcts_model_bank.weights", ndim=2, shape=(5, len(features)))
        _array(state["bias"], "bamcts_model_bank.bias", ndim=1, shape=(5,))
    elif component == "bamcts_search_configuration":
        _state_keys(
            state,
            {"depth", "simulations", "posterior_likelihood_sigma", "action_count"},
            component,
        )
        if state["depth"] != 8 or state["simulations"] != 256 or state["action_count"] != 11:
            raise ContractError("BA-MCTS search configuration mismatch")
        if (
            require_finite(
                state["posterior_likelihood_sigma"], "posterior_likelihood_sigma", nonnegative=True
            )
            <= 0.0
        ):
            raise ContractError("BA-MCTS posterior likelihood sigma must be positive")
    elif component == "evd_q_members":
        _state_keys(state, {"feature_order", "member_ids", "q_weights", "q_bias"}, component)
        features = _feature_order(state, component)
        _member_ids(state["member_ids"], "evd_q_members.member_ids", 20)
        _array(state["q_weights"], "evd_q_members.q_weights", ndim=3, shape=(20, len(features), 11))
        _array(state["q_bias"], "evd_q_members.q_bias", ndim=2, shape=(20, 11))
    elif component == "evd_policy_configuration":
        _state_keys(state, {"disagreement_penalty", "action_count", "objective"}, component)
        require_finite(state["disagreement_penalty"], "EVD disagreement penalty", nonnegative=True)
        if state["action_count"] != 11 or state["objective"] != "raw_logged_rewards":
            raise ContractError("EVD policy objective/configuration mismatch")
    elif component == "ricker_fit_cache":
        count = 8 if method.startswith("plus_") else 1
        expected = {
            "schema_version",
            "cell",
            "candidate_ids",
            "candidate_labels",
            "form",
            "action_channels",
            "growth",
            "mortality",
            "capacity_increment",
            "stocking",
            "process_scale",
            "observation_scale",
            "survey_scale",
            "initial_capacity",
            "capacity_ceiling",
            "reset_log_mean",
            "reset_log_scale",
            "depensation_thresholds",
            "theta_exponent",
            "regime_multipliers",
            "regime_matrix",
            "equation_version",
            "regime_law_version",
            "parameter_hashes",
        }
        _state_keys(
            state,
            expected,
            component,
        )
        if state["schema_version"] != RICKER_FIT_CACHE_SCHEMA:
            raise ContractError("ecological fitted-cache state schema mismatch")
        if state["equation_version"] != EQUATION_VERSION:
            raise ContractError("ecological equation version mismatch")
        if state["regime_law_version"] != REGIME_LAW_VERSION:
            raise ContractError("ecological regime-law version mismatch")
        require_nonempty_string(state["cell"], "ricker_fit_cache.cell")
        _member_ids(state["candidate_ids"], "ricker candidate_ids", count)
        _strings(state["candidate_labels"], "ricker candidate_labels", expected_length=count)
        forms = _strings(state["form"], "ricker forms", expected_length=count, unique=False)
        if any(value not in REGISTERED_MODEL_FORMS for value in forms):
            raise ContractError("ecological fitted cache contains an unregistered model form")
        channels = state["action_channels"]
        if not isinstance(channels, list) or len(channels) != count:
            raise ContractError("action_channels candidate dimension mismatch")
        parsed_channels: list[tuple[str, ...]] = []
        action_count: int | None = None
        for index, row in enumerate(channels):
            parsed = _strings(row, f"action_channels[{index}]", unique=False)
            if action_count is None:
                action_count = len(parsed)
            if len(parsed) != action_count or any(
                value not in REGISTERED_ACTION_CHANNELS for value in parsed
            ):
                raise ContractError(
                    "action_channels must use one registered public channel per action"
                )
            parsed_channels.append(parsed)
        assert action_count is not None
        growth = _array(
            state["growth"], "ricker_fit_cache.growth", ndim=2, shape=(count, action_count)
        )
        mortality = _array(
            state["mortality"], "ricker_fit_cache.mortality", ndim=2, shape=(count, action_count)
        )
        capacity_increment = _array(
            state["capacity_increment"],
            "ricker_fit_cache.capacity_increment",
            ndim=2,
            shape=(count, action_count),
        )
        stocking = _array(
            state["stocking"], "ricker_fit_cache.stocking", ndim=2, shape=(count, action_count)
        )
        for field, values in (
            ("growth", growth),
            ("mortality", mortality),
            ("capacity_increment", capacity_increment),
            ("stocking", stocking),
        ):
            if np.any(values < 0.0):
                raise ContractError(f"ricker_fit_cache.{field} must be nonnegative")
        if np.any((growth > 0.0) & (mortality > 0.0)):
            raise ContractError("growth and mortality cannot both be positive for one action")
        for candidate_index, row in enumerate(parsed_channels):
            for action_index, channel in enumerate(row):
                if (
                    channel not in {"capacity", "rate+capacity"}
                    and capacity_increment[candidate_index, action_index] != 0.0
                ):
                    raise ContractError("capacity increment is incompatible with action channel")
                if channel != "state" and stocking[candidate_index, action_index] != 0.0:
                    raise ContractError("stocking is incompatible with action channel")
        process_scale = _array(
            state["process_scale"], "ricker_fit_cache.process_scale", ndim=1, shape=(count,)
        )
        observation_scale = _array(
            state["observation_scale"], "ricker_fit_cache.observation_scale", ndim=1, shape=(count,)
        )
        if np.any(process_scale < 0.0) or np.any(observation_scale < 0.0):
            raise ContractError("ecological process/observation scales must be nonnegative")
        survey_scale = _array(
            state["survey_scale"],
            "ricker_fit_cache.survey_scale",
            ndim=1,
            shape=(count,),
            positive=True,
        )
        initial_capacity = _array(
            state["initial_capacity"],
            "ricker_fit_cache.initial_capacity",
            ndim=1,
            shape=(count,),
            positive=True,
        )
        capacity_ceiling = _array(
            state["capacity_ceiling"],
            "ricker_fit_cache.capacity_ceiling",
            ndim=1,
            shape=(count,),
            positive=True,
        )
        if np.any(capacity_ceiling < initial_capacity):
            raise ContractError("capacity ceiling must be at least initial capacity")
        _array(state["reset_log_mean"], "ricker_fit_cache.reset_log_mean", ndim=1, shape=(count,))
        _array(
            state["reset_log_scale"],
            "ricker_fit_cache.reset_log_scale",
            ndim=1,
            shape=(count,),
            positive=True,
        )
        thresholds = _array(
            state["depensation_thresholds"],
            "ricker_fit_cache.depensation_thresholds",
            ndim=2,
            shape=(count, 2),
            positive=True,
        )
        if np.any(thresholds >= initial_capacity[:, None]):
            raise ContractError("depensation thresholds must be below initial capacity")
        _array(
            state["theta_exponent"],
            "ricker_fit_cache.theta_exponent",
            ndim=1,
            shape=(count,),
            positive=True,
        )
        _array(
            state["regime_multipliers"],
            "ricker_fit_cache.regime_multipliers",
            ndim=2,
            shape=(count, 2),
            positive=True,
        )
        regime = _array(
            state["regime_matrix"], "ricker_fit_cache.regime_matrix", ndim=3, shape=(count, 2, 2)
        )
        if np.any(regime < 0.0) or not np.allclose(regime.sum(axis=2), 1.0, rtol=0.0, atol=1e-10):
            raise ContractError("regime matrix must be nonnegative and row stochastic")
        hashes = _strings(
            state["parameter_hashes"], "ricker parameter hashes", expected_length=count
        )
        for digest in hashes:
            require_sha256(digest, "MechanisticModel parameter hash")
        del survey_scale
    elif component == "pbvi_grids":
        _state_keys(state, {"abundance_grid", "capacity_grid", "observation_grid"}, component)
        for field in ("abundance_grid", "observation_grid"):
            values = _array(state[field], f"pbvi_grids.{field}", ndim=1)
            if values.size < 2 or np.any(values < 0.0) or np.any(np.diff(values) <= 0.0):
                raise ContractError(
                    "PBVI abundance/observation grids must be nonnegative, increasing and complete"
                )
        values = _array(state["capacity_grid"], "pbvi_grids.capacity_grid", ndim=1, positive=True)
        if values.size < 2 or np.any(np.diff(values) <= 0.0):
            raise ContractError("PBVI capacity grid must be positive, increasing and complete")
    elif component == "pbvi_candidates":
        count = 8 if method.startswith("plus_") else 1
        _state_keys(state, {"candidate_ids", "ricker_fit_cache_sha256"}, component)
        _member_ids(state["candidate_ids"], "PBVI candidate_ids", count)
        require_sha256(state["ricker_fit_cache_sha256"], "PBVI candidate Ricker cache")
    elif component == "pbvi_prior":
        count = 8 if method.startswith("plus_") else 1
        _state_keys(state, {"candidate_ids", "probabilities"}, component)
        _member_ids(state["candidate_ids"], "PBVI prior candidate_ids", count)
        probabilities = _array(
            state["probabilities"], "PBVI prior probabilities", ndim=1, shape=(count,)
        )
        if np.any(probabilities < 0.0) or not np.isclose(
            np.sum(probabilities), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ContractError("PBVI prior probabilities must sum to one")
    elif component == "pbvi_policy":
        expected = {"grids_sha256", "candidates_sha256"}
        if method.startswith("plus_"):
            expected.add("prior_sha256")
        _state_keys(state, expected, component)
        for field, value in state.items():
            require_sha256(value, f"PBVI policy {field}")
    else:
        raise ContractError(f"no fitted-state schema exists for component: {component}")


def _encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ContractError("artifact arrays must have a numeric or boolean dtype")
        if not np.isfinite(value).all() if value.dtype.kind in "f" else False:
            raise ContractError("artifact array contains non-finite values")
        contiguous = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "data_base64": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ContractError("artifact state mapping keys must be strings")
        return {key: _encode_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_value(value: Any) -> Any:
    if isinstance(value, Mapping) and value.get("__ndarray__") is True:
        require_exact_keys(value, {"__ndarray__", "dtype", "shape", "data_base64"}, "ndarray")
        try:
            raw = base64.b64decode(value["data_base64"], validate=True)
            dtype = np.dtype(value["dtype"])
            shape = tuple(int(item) for item in value["shape"])
            array = np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
        except (TypeError, ValueError) as exc:
            raise ContractError("malformed serialized ndarray") from exc
        return array
    if isinstance(value, Mapping):
        return {str(key): _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


@dataclass(frozen=True)
class CanonicalArtifact:
    component: str
    method: str
    fit_source: str
    state: Mapping[str, Any]

    def to_bytes(self) -> bytes:
        if "training_history" in self.component.lower() or "training_history" in self.state:
            raise ContractError("training-history files cannot qualify as fitted-artifact identity")
        require_nonempty_string(self.component, "artifact component")
        require_nonempty_string(self.method, "artifact method")
        require_nonempty_string(self.fit_source, "artifact fit_source")
        if self.method in REQUIRED_COMPONENTS:
            if self.component not in REQUIRED_COMPONENTS[self.method]:
                raise ContractError("artifact component is not registered for its method")
        elif not (self.method == "shared_general_methods" and self.component == "reward_surrogate"):
            raise ContractError("artifact method is not registered")
        _validate_artifact_state(self)
        return canonical_json_bytes(
            {
                "schema_version": "corrected_stageb_canonical_artifact_v1",
                "component": self.component,
                "method": self.method,
                "fit_source": self.fit_source,
                "state": _encode_value(self.state),
            }
        )

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.to_bytes())

    @classmethod
    def from_bytes(cls, payload: bytes) -> "CanonicalArtifact":
        value = strict_json_loads(payload)
        if not isinstance(value, Mapping):
            raise ContractError("serialized artifact must be an object")
        require_exact_keys(
            value,
            {"schema_version", "component", "method", "fit_source", "state"},
            "canonical artifact",
        )
        if value["schema_version"] != "corrected_stageb_canonical_artifact_v1":
            raise ContractError("canonical artifact schema mismatch")
        artifact = cls(
            component=value["component"],
            method=value["method"],
            fit_source=value["fit_source"],
            state=_decode_value(value["state"]),
        )
        if artifact.to_bytes() != payload:
            raise ContractError("artifact bytes are not canonical")
        return artifact


@dataclass(frozen=True)
class ArtifactBundleReceipt:
    method: str
    component_hashes: Mapping[str, str]
    bundle_sha256: str
    fixture_manifest_sha256: str
    reload_parity: bool


def applicable_fixture_components(method: str) -> frozenset[str]:
    if method not in REQUIRED_COMPONENTS:
        raise ContractError("unknown registered method")
    return frozenset(REQUIRED_COMPONENTS[method] & FEATURE_COMPONENTS)


def deterministic_component_fixture(
    component: str,
    feature_order: Sequence[str],
    *,
    row_count: int = FIXTURE_ROW_COUNT,
) -> np.ndarray:
    """Derive the registered platform-stable fixture for one logical component."""

    if component not in FEATURE_COMPONENTS:
        raise ContractError("non-feature component cannot have a prediction fixture")
    if row_count != FIXTURE_ROW_COUNT:
        raise ContractError("prediction fixture row count is not the registered fixed value")
    parsed_features = tuple(
        require_nonempty_string(value, f"{component} fixture feature") for value in feature_order
    )
    if not parsed_features or len(set(parsed_features)) != len(parsed_features):
        raise ContractError("prediction fixture feature order must be nonempty and unique")
    fixture = np.empty((row_count, len(parsed_features)), dtype=np.float64)
    recipe_id = b"corrected-stageb-component-fixture-v1"
    for row in range(row_count):
        for column, feature in enumerate(parsed_features):
            seed = b"\0".join(
                (
                    recipe_id,
                    component.encode("utf-8"),
                    str(row).encode("ascii"),
                    str(column).encode("ascii"),
                    feature.encode("utf-8"),
                )
            )
            unsigned = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big")
            fixture[row, column] = np.float64((unsigned % 2_000_001) - 1_000_000) / np.float64(
                65_536.0
            )
    return fixture


def _fixture_little_endian_bytes(fixture: np.ndarray) -> bytes:
    return np.ascontiguousarray(fixture, dtype=np.dtype("<f8")).tobytes(order="C")


def fixture_manifest(
    artifacts: Mapping[str, CanonicalArtifact],
    fixtures: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for component in sorted(fixtures):
        feature_order = list(_feature_order(artifacts[component].state, component))
        fixture = fixtures[component]
        entries[component] = {
            "feature_order": feature_order,
            "dtype": "float64",
            "shape": [FIXTURE_ROW_COUNT, len(feature_order)],
            "little_endian_c_bytes_sha256": sha256_bytes(_fixture_little_endian_bytes(fixture)),
        }
    return {
        "schema_version": FIXTURE_MANIFEST_SCHEMA,
        "generator_recipe": FIXTURE_GENERATOR_RECIPE,
        "row_count": FIXTURE_ROW_COUNT,
        "components": entries,
    }


def deterministic_prediction_fixtures(
    method: str,
    serialized_components: Mapping[str, bytes],
) -> dict[str, np.ndarray]:
    """Build fixtures from each component's exact canonical feature order."""

    expected = applicable_fixture_components(method)
    if set(serialized_components) != set(REQUIRED_COMPONENTS[method]):
        raise ContractError("cannot derive fixtures from an incomplete fitted-artifact bundle")
    result: dict[str, np.ndarray] = {}
    for component in sorted(expected):
        artifact = CanonicalArtifact.from_bytes(serialized_components[component])
        result[component] = deterministic_component_fixture(
            component, _feature_order(artifact.state, component)
        )
    return result


def validate_complete_artifact_bundle(
    method: str,
    serialized_components: Mapping[str, bytes],
    *,
    prediction_fixtures: Mapping[str, np.ndarray],
    expected_component_hashes: Mapping[str, str],
) -> ArtifactBundleReceipt:
    if method not in REQUIRED_COMPONENTS:
        raise ContractError("unknown registered method")
    required = REQUIRED_COMPONENTS[method]
    missing = sorted(required - set(serialized_components))
    extra = sorted(set(serialized_components) - required)
    if missing or extra:
        raise ContractError(f"fitted-artifact bundle incomplete; missing={missing}, extra={extra}")
    if set(expected_component_hashes) != set(required):
        raise ContractError("registered fitted-artifact hash map is incomplete")
    for component, digest in expected_component_hashes.items():
        require_sha256(digest, f"expected artifact hash {component}")
    if not isinstance(prediction_fixtures, Mapping):
        raise ContractError("prediction fixtures must be a component-keyed mapping")
    fixture_domain = applicable_fixture_components(method)
    missing_fixtures = sorted(fixture_domain - set(prediction_fixtures))
    if missing_fixtures:
        raise ContractError(f"prediction fixtures missing components: {missing_fixtures}")
    extra_fixtures = sorted(set(prediction_fixtures) - fixture_domain)
    if extra_fixtures:
        raise ContractError(
            f"prediction fixtures include extra/non-feature components: {extra_fixtures}"
        )
    hashes: dict[str, str] = {}
    artifacts: dict[str, CanonicalArtifact] = {}
    for component in sorted(required):
        payload = serialized_components[component]
        if not isinstance(payload, bytes):
            raise ContractError("serialized fitted-artifact components must be bytes")
        artifact = CanonicalArtifact.from_bytes(payload)
        allowed_method = artifact.method == method or (
            component == "reward_surrogate"
            and method in SURROGATE_METHODS
            and artifact.method == "shared_general_methods"
            and method in artifact.state["consumer_methods"]
        )
        if not allowed_method or artifact.component != component:
            raise ContractError("artifact method/component binding mismatch")
        if (
            component == "reward_surrogate"
            and method in SURROGATE_METHODS
            and artifact.fit_source != "Arm O public-data view only"
        ):
            raise ContractError(
                "matched reward surrogate was not fitted from the Arm O public view"
            )
        artifacts[component] = artifact
        hashes[component] = sha256_bytes(payload)
        if hashes[component] != expected_component_hashes[component]:
            raise ContractError(f"serialized fitted-artifact hash mismatch: {component}")
        reloaded = CanonicalArtifact.from_bytes(bytes(payload))
        if reloaded is artifact:
            raise ContractError("artifact reload did not create a fresh object")
        fixture = None
        if component in fixture_domain:
            provided = prediction_fixtures[component]
            if not isinstance(provided, np.ndarray):
                raise ContractError(f"prediction fixture must be a NumPy array: {component}")
            fixture = provided
            feature_order = _feature_order(artifact.state, component)
            if (
                fixture.dtype != np.dtype("float64")
                or fixture.ndim != 2
                or fixture.shape != (FIXTURE_ROW_COUNT, len(feature_order))
            ):
                raise ContractError(f"prediction fixture dtype/shape mismatch for {component}")
            if not np.isfinite(fixture).all():
                raise ContractError(f"prediction fixture must be finite for {component}")
            expected_fixture = deterministic_component_fixture(component, feature_order)
            if _fixture_little_endian_bytes(fixture) != _fixture_little_endian_bytes(
                expected_fixture
            ):
                raise ContractError(
                    f"prediction fixture deterministic byte mismatch for {component}"
                )
        before = _semantic_probe(artifact, fixture)
        after = _semantic_probe(reloaded, None if fixture is None else fixture.copy())
        if not _semantic_equal(before, after):
            raise ContractError("serialized artifact prediction/action parity failed")
    _validate_bundle_cross_references(method, artifacts, hashes)
    bundle_bytes = canonical_json_bytes(hashes)
    manifest_sha256 = sha256_bytes(
        canonical_json_bytes(fixture_manifest(artifacts, prediction_fixtures))
    )
    return ArtifactBundleReceipt(
        method,
        hashes,
        sha256_bytes(bundle_bytes),
        manifest_sha256,
        True,
    )


def _semantic_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes() == right.tobytes()
        )
    return left == right


def _semantic_probe(artifact: CanonicalArtifact, fixture: np.ndarray | None) -> Any:
    component = artifact.component
    state = artifact.state
    if component in FEATURE_COMPONENTS:
        if fixture is None or fixture.shape[1] != len(state["feature_order"]):
            raise ContractError(f"prediction fixture feature width mismatch for {component}")
    if component == "reward_surrogate":
        return fixture @ state["weights"] + np.float64(state["bias"])
    if component in {"dynamics_ensemble", "bamcts_model_bank"}:
        return fixture @ state["weights"].T + state["bias"]
    if component in {"refplan_behavior_prior", "ogsrl_actor", "evd_behavior_reference"}:
        scores = fixture @ state["action_weights"] + state["action_bias"]
        return np.argmax(scores, axis=1).astype(np.int64)
    if component == "ogsrl_guardian":
        distances = np.linalg.norm(
            fixture[:, None, :] - state["reference_points"][None, :, :], axis=2
        )
        return np.min(distances, axis=1)
    if component == "evd_q_members":
        return np.einsum("bf,mfa->bma", fixture, state["q_weights"]) + state["q_bias"]
    return canonical_json_bytes(_encode_value(state))


def _validate_bundle_cross_references(
    method: str,
    artifacts: Mapping[str, CanonicalArtifact],
    hashes: Mapping[str, str],
) -> None:
    if method in {"refplan", "ogsrl", "bamcts"}:
        residual = artifacts["residual_process_scales"].state
        if residual["dynamics_ensemble_sha256"] != hashes["dynamics_ensemble"]:
            raise ContractError("residual scale does not bind the serialized dynamics ensemble")
    if method in ECOLOGICAL_METHODS:
        cache_state = artifacts["ricker_fit_cache"].state
        process_state = artifacts["residual_process_scales"].state
        if process_state["ricker_fit_cache_sha256"] != hashes["ricker_fit_cache"]:
            raise ContractError("ecological residual scale does not bind the Ricker cache")
        if process_state["candidate_ids"] != cache_state["candidate_ids"]:
            raise ContractError("ecological process-scale candidate identifiers mismatch")
        cache_values = _array(cache_state["process_scale"], "ricker cache process_scale", ndim=1)
        process_values = _array(process_state["process_scale"], "process-scale component", ndim=1)
        if cache_values.dtype != process_values.dtype or cache_values.shape != process_values.shape:
            raise ContractError("ecological process-scale shape/dtype mismatch")
        if cache_values.tobytes(order="C") != process_values.tobytes(order="C"):
            raise ContractError("ecological process-scale values are not bit-identical to cache")
        if (
            artifacts["pbvi_candidates"].state["ricker_fit_cache_sha256"]
            != hashes["ricker_fit_cache"]
        ):
            raise ContractError("PBVI candidates do not bind the Ricker cache")
        policy = artifacts["pbvi_policy"].state
        expected = {
            "grids_sha256": hashes["pbvi_grids"],
            "candidates_sha256": hashes["pbvi_candidates"],
        }
        if method.startswith("plus_"):
            expected["prior_sha256"] = hashes["pbvi_prior"]
        if policy != expected:
            raise ContractError("PBVI policy references do not bind the complete serialized policy")


def linear_prediction(artifact: CanonicalArtifact, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    weights = np.asarray(artifact.state["weights"], dtype=np.float64)
    bias = np.asarray(artifact.state["bias"], dtype=np.float64)
    result = values @ weights + bias
    if not np.isfinite(result).all():
        raise ContractError("artifact prediction is non-finite")
    return np.asarray(result, dtype=np.float64)


def select_action(artifact: CanonicalArtifact, features: np.ndarray) -> int:
    values = np.asarray(features, dtype=np.float64)
    weights = np.asarray(artifact.state["action_weights"], dtype=np.float64)
    bias = np.asarray(artifact.state["action_bias"], dtype=np.float64)
    scores = values @ weights + bias
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ContractError("action scores must be a finite vector")
    return int(np.argmax(scores))


def write_artifact_once(path: Path, artifact: CanonicalArtifact) -> str:
    if path.exists():
        raise ContractError(f"refusing to overwrite serialized artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = artifact.to_bytes()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_bytes(payload)


@dataclass(frozen=True)
class MatchedSurrogateBinding:
    cell: str
    path: Path
    sha256: str
    source_public_view_sha256: str
    label: str = MATCHED_SURROGATE_LABEL

    def load_for_arm(self, arm: str, method: str) -> CanonicalArtifact:
        if arm not in {"O", "T"}:
            raise ContractError("surrogate arm must be O or T")
        if method not in SURROGATE_METHODS:
            raise ContractError("matched surrogate consumer is not registered")
        payload = self.path.read_bytes()
        if sha256_bytes(payload) != self.sha256:
            raise ContractError("matched surrogate bytes changed after serialization")
        artifact = CanonicalArtifact.from_bytes(payload)
        if artifact.component != "reward_surrogate" or artifact.method != "shared_general_methods":
            raise ContractError("matched surrogate artifact component mismatch")
        if artifact.fit_source != "Arm O public-data view only":
            raise ContractError("matched surrogate was not fitted from the Arm O public view")
        if artifact.state["cell"] != self.cell:
            raise ContractError("matched surrogate cross-cell rebinding is prohibited")
        if artifact.state["source_public_view_sha256"] != self.source_public_view_sha256:
            raise ContractError("matched surrogate source-public-view rebinding is prohibited")
        if (
            artifact.state["label"] != self.label
            or method not in artifact.state["consumer_methods"]
        ):
            raise ContractError("matched surrogate consumer/label binding mismatch")
        return artifact


def seal_matched_arm_o_surrogate(
    path: Path,
    *,
    cell: str,
    source_public_view_sha256: str,
    state: Mapping[str, Any],
    fitting_arm: str,
) -> MatchedSurrogateBinding:
    if fitting_arm != "O":
        raise ContractError("Arm T reward-surrogate refitting is structurally prohibited")
    require_sha256(source_public_view_sha256, "source_public_view_sha256")
    require_nonempty_string(cell, "matched surrogate cell")
    _state_keys(state, {"feature_order", "weights", "bias"}, "matched surrogate fit")
    bound_state = {
        "cell": cell,
        "source_public_view_sha256": source_public_view_sha256,
        "consumer_methods": sorted(SURROGATE_METHODS),
        "label": MATCHED_SURROGATE_LABEL,
        "feature_order": state["feature_order"],
        "weights": state["weights"],
        "bias": state["bias"],
    }
    artifact = CanonicalArtifact(
        component="reward_surrogate",
        method="shared_general_methods",
        fit_source="Arm O public-data view only",
        state=bound_state,
    )
    digest = write_artifact_once(path, artifact)
    return MatchedSurrogateBinding(cell, path, digest, source_public_view_sha256)


def verify_matched_surrogate_arms(binding: MatchedSurrogateBinding) -> dict[str, Any]:
    per_consumer: dict[str, dict[str, str]] = {}
    reference_bytes: bytes | None = None
    for method in sorted(SURROGATE_METHODS):
        arm_o_bytes = binding.load_for_arm("O", method).to_bytes()
        arm_t_bytes = binding.load_for_arm("T", method).to_bytes()
        if arm_o_bytes != arm_t_bytes:
            raise ContractError("Arm O/T matched surrogate identity failure")
        if reference_bytes is None:
            reference_bytes = arm_o_bytes
        elif reference_bytes != arm_o_bytes:
            raise ContractError("general methods did not load one shared surrogate byte stream")
        per_consumer[method] = {
            "arm_o_sha256": sha256_bytes(arm_o_bytes),
            "arm_t_sha256": sha256_bytes(arm_t_bytes),
        }
    if reference_bytes is None:
        raise ContractError("matched surrogate verification had no registered consumers")
    arm_o_hash = sha256_bytes(reference_bytes)
    if arm_o_hash != binding.sha256:
        raise ContractError("matched surrogate binding hash mismatch")
    return {
        "label": binding.label,
        "arm_o_sha256": arm_o_hash,
        "arm_t_sha256": arm_o_hash,
        "per_consumer": per_consumer,
        "shared_consumers": sorted(SURROGATE_METHODS),
        "exact_bytes_equal": True,
        "arm_t_refit_permitted": False,
        "cell": binding.cell,
        "source_public_view_sha256": binding.source_public_view_sha256,
    }


def validate_registered_surrogate_set(
    bindings: Sequence[MatchedSurrogateBinding],
) -> dict[str, str]:
    """Require exactly one matched surrogate for each frozen species cell."""

    from .registration import CELLS

    if len(bindings) != len(CELLS):
        raise ContractError("matched surrogate set must contain exactly one binding per cell")
    observed = [binding.cell for binding in bindings]
    if len(set(observed)) != len(observed) or set(observed) != set(CELLS):
        raise ContractError("matched surrogate bindings are duplicated, missing, or cross-cell")
    receipts: dict[str, str] = {}
    for binding in bindings:
        receipt = verify_matched_surrogate_arms(binding)
        receipts[binding.cell] = receipt["arm_o_sha256"]
    return receipts


def require_m3_arm_o_parity(parity_receipt: Mapping[str, Any]) -> None:
    if parity_receipt.get("architecture") != "Intel Xeon Platinum 8452Y":
        raise ContractError("Arm O parity must be established on the registered M3 architecture")
    if parity_receipt.get("per_episode_action_event_parity") is not True:
        raise ContractError("Arm O parity failed; Arm T must remain structurally unavailable")
    require_sha256(parity_receipt.get("accepted_artifact_gate_sha256"), "parity artifact gate")


@dataclass(frozen=True)
class SharedEcologicalPolicy:
    method: str
    cell: str
    serialized_components: Mapping[str, bytes]
    survey_scale: float

    @property
    def ricker_cache_bytes(self) -> bytes:
        return self.serialized_components["ricker_fit_cache"]

    @property
    def reward_surrogate_bytes(self) -> bytes:
        return self.serialized_components["reward_surrogate"]

    @property
    def policy_bytes(self) -> bytes:
        return self.serialized_components["pbvi_policy"]

    @property
    def cache_sha256(self) -> str:
        return sha256_bytes(self.ricker_cache_bytes)

    @property
    def policy_sha256(self) -> str:
        return sha256_bytes(self.policy_bytes)

    @property
    def surrogate_sha256(self) -> str:
        return sha256_bytes(self.reward_surrogate_bytes)

    def validate(self) -> None:
        if self.method not in ECOLOGICAL_METHODS:
            raise ContractError("shared ecological policy is restricted to PLUS/MOOR")
        require_finite(self.survey_scale, "survey_scale", nonnegative=True)
        if self.survey_scale <= 0.0:
            raise ContractError("survey_scale must be strictly positive")
        if set(self.serialized_components) != set(REQUIRED_COMPONENTS[self.method]):
            raise ContractError("shared ecological policy requires the complete artifact bundle")
        hashes = {
            component: sha256_bytes(payload)
            for component, payload in self.serialized_components.items()
        }
        surrogate = CanonicalArtifact.from_bytes(self.reward_surrogate_bytes)
        validate_complete_artifact_bundle(
            self.method,
            self.serialized_components,
            prediction_fixtures=deterministic_prediction_fixtures(
                self.method, self.serialized_components
            ),
            expected_component_hashes=hashes,
        )
        cache = CanonicalArtifact.from_bytes(self.ricker_cache_bytes)
        if cache.state["cell"] != self.cell or surrogate.state["cell"] != self.cell:
            raise ContractError("shared ecological policy cross-cell fitted-artifact binding")
        cached_survey_scales = _array(
            cache.state["survey_scale"], "shared policy cache survey_scale", ndim=1
        )
        expected_scale_bits = np.float64(self.survey_scale).view(np.uint64)
        if not np.all(cached_survey_scales.view(np.uint64) == expected_scale_bits):
            raise ContractError("shared ecological policy survey_scale/cache mismatch")


@dataclass(frozen=True)
class EcologicalArmInterface:
    arm: str
    kind: str
    belief: np.ndarray
    action_history: tuple[int, ...]
    observation_history: tuple[float, ...]
    timestep: int
    latent_grid: np.ndarray
    conversion_receipt: Mapping[str, Any] | None

    def validate(self) -> None:
        if self.arm not in {"O", "T"}:
            raise ContractError("ecological interface arm must be O or T")
        expected_kind = "noisy_survey_belief" if self.arm == "O" else "exact_point_mass"
        if self.kind != expected_kind:
            raise ContractError("ecological abundance-interface kind mismatch")
        belief = np.asarray(self.belief)
        latent_grid = np.asarray(self.latent_grid)
        if (
            belief.dtype != np.dtype("float64")
            or belief.ndim != 1
            or belief.size < 2
            or not np.isfinite(belief).all()
            or np.any(belief < 0.0)
            or not np.isclose(np.sum(belief), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ContractError("ecological abundance belief must be a finite probability vector")
        if (
            latent_grid.dtype != np.dtype("float64")
            or latent_grid.ndim != 1
            or latent_grid.shape != belief.shape
            or not np.isfinite(latent_grid).all()
            or np.any(np.diff(latent_grid) <= 0.0)
        ):
            raise ContractError("ecological latent grid must be aligned finite increasing float64")
        if self.timestep < 0 or len(self.action_history) != self.timestep:
            raise ContractError("ecological action history/timestep mismatch")
        if len(self.observation_history) != self.timestep + 1:
            raise ContractError("ecological observation history/timestep mismatch")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 11
            for item in self.action_history
        ):
            raise ContractError("ecological action history is invalid")
        for item in self.observation_history:
            require_finite(item, "ecological observation history", nonnegative=True)
        if self.arm == "O" and self.conversion_receipt is not None:
            raise ContractError("Arm O cannot carry an exact-abundance conversion receipt")
        if self.arm == "T":
            if self.conversion_receipt is None or np.count_nonzero(belief) != 1:
                raise ContractError("Arm T requires the registered exact point-mass conversion")
            require_exact_keys(
                self.conversion_receipt,
                {
                    "conversion",
                    "raw_abundance_float64_hex",
                    "survey_scale_float64_hex",
                    "latent_abundance_float64_hex",
                    "selected_index",
                    "selected_latent_float64_hex",
                    "selected_raw_float64_hex",
                    "raw_discretisation_absolute_error",
                },
                "ecological conversion receipt",
            )
            if (
                self.conversion_receipt["conversion"]
                != "latent_abundance = raw_abundance / survey_scale"
            ):
                raise ContractError("ecological survey_scale conversion rule mismatch")
            receipt = self.conversion_receipt
            selected_index = receipt["selected_index"]
            if isinstance(selected_index, bool) or not isinstance(selected_index, int):
                raise ContractError("ecological selected_index must be an integer")
            nonzero = np.flatnonzero(belief)
            if (
                selected_index < 0
                or selected_index >= belief.size
                or nonzero.size != 1
                or int(nonzero[0]) != selected_index
                or belief[selected_index].view(np.uint64) != np.float64(1.0).view(np.uint64)
            ):
                raise ContractError("ecological conversion index does not identify the point mass")

            parsed: dict[str, np.float64] = {}
            for field in (
                "raw_abundance_float64_hex",
                "survey_scale_float64_hex",
                "latent_abundance_float64_hex",
                "selected_latent_float64_hex",
                "selected_raw_float64_hex",
            ):
                value = receipt[field]
                if not isinstance(value, str):
                    raise ContractError(f"ecological conversion {field} must be float64 hex")
                try:
                    number = np.float64(float.fromhex(value))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ContractError(f"ecological conversion {field} is invalid") from exc
                if not np.isfinite(number) or number.hex() != value:
                    raise ContractError(
                        f"ecological conversion {field} is not canonical finite hex"
                    )
                parsed[field] = number
            raw = parsed["raw_abundance_float64_hex"]
            scale = parsed["survey_scale_float64_hex"]
            latent = parsed["latent_abundance_float64_hex"]
            selected_latent = parsed["selected_latent_float64_hex"]
            selected_raw = parsed["selected_raw_float64_hex"]
            if raw < 0.0 or scale <= 1e-12:
                raise ContractError("ecological raw abundance or survey_scale is invalid")
            if latent.view(np.uint64) != np.float64(raw / scale).view(np.uint64):
                raise ContractError("ecological latent abundance is not raw/survey_scale")
            if selected_latent.view(np.uint64) != latent_grid[selected_index].view(np.uint64):
                raise ContractError("ecological selected latent value does not match the grid")
            if selected_raw.view(np.uint64) != np.float64(selected_latent * scale).view(np.uint64):
                raise ContractError("ecological selected raw value does not match grid conversion")
            error = require_finite(
                receipt["raw_discretisation_absolute_error"],
                "ecological raw discretisation error",
                nonnegative=True,
            )
            expected_error = np.float64(abs(selected_raw - raw))
            if np.float64(error).view(np.uint64) != expected_error.view(np.uint64):
                raise ContractError("ecological raw discretisation error is inconsistent")


def build_ecological_arm_interfaces(
    *,
    raw_abundance: float,
    noisy_belief: np.ndarray,
    survey_scale: float,
    latent_grid: np.ndarray,
    action_history: Sequence[int],
    observation_history: Sequence[float],
    timestep: int,
) -> tuple[EcologicalArmInterface, EcologicalArmInterface]:
    """Build the two registered interfaces while preserving all public context."""

    try:
        exact_belief, conversion = direct_point_mass_latent_belief(
            raw_abundance,
            survey_scale=survey_scale,
            latent_grid=np.asarray(latent_grid, dtype=np.float64),
        )
    except IntegrationViolation as exc:
        raise ContractError(str(exc)) from exc
    history_actions = tuple(action_history)
    history_observations = tuple(float(item) for item in observation_history)
    registered_grid = np.asarray(latent_grid)
    if registered_grid.dtype != np.dtype("float64"):
        raise ContractError("ecological latent_grid must already be float64")
    arm_o = EcologicalArmInterface(
        "O",
        "noisy_survey_belief",
        np.asarray(noisy_belief, dtype=np.float64).copy(),
        history_actions,
        history_observations,
        timestep,
        registered_grid.copy(),
        None,
    )
    arm_t = EcologicalArmInterface(
        "T",
        "exact_point_mass",
        exact_belief,
        history_actions,
        history_observations,
        timestep,
        registered_grid.copy(),
        conversion,
    )
    arm_o.validate()
    arm_t.validate()
    return arm_o, arm_t


def validate_ecological_arm_pair(
    arm_o: SharedEcologicalPolicy,
    arm_t: SharedEcologicalPolicy,
    *,
    observation_sigma: float,
    arm_o_interface: EcologicalArmInterface,
    arm_t_interface: EcologicalArmInterface,
) -> dict[str, Any]:
    arm_o.validate()
    arm_t.validate()
    if (arm_o.method, arm_o.cell) != (arm_t.method, arm_t.cell):
        raise ContractError("ecological arm policy binding mismatch")
    if arm_o.serialized_components.keys() != arm_t.serialized_components.keys() or any(
        arm_o.serialized_components[component] != arm_t.serialized_components[component]
        for component in arm_o.serialized_components
    ):
        raise ContractError("Arm O/T must load the identical complete ecological artifact bundle")
    if np.float64(arm_o.survey_scale).view(np.uint64) != np.float64(arm_t.survey_scale).view(
        np.uint64
    ):
        raise ContractError("Arm O/T survey_scale must be bit-identical")
    sigma = require_finite(observation_sigma, "observation_sigma", nonnegative=True)
    if sigma <= 1e-12:
        raise ContractError("sigma -> 0 is prohibited")
    arm_o_interface.validate()
    arm_t_interface.validate()
    if arm_o_interface.arm != "O" or arm_t_interface.arm != "T":
        raise ContractError("ecological interfaces must be paired O then T")
    if (
        arm_o_interface.action_history != arm_t_interface.action_history
        or arm_o_interface.observation_history != arm_t_interface.observation_history
        or arm_o_interface.timestep != arm_t_interface.timestep
        or arm_o_interface.belief.shape != arm_t_interface.belief.shape
        or not np.array_equal(
            arm_o_interface.latent_grid.view(np.uint64),
            arm_t_interface.latent_grid.view(np.uint64),
        )
    ):
        raise ContractError("ecological interface changed registered public context")
    conversion = arm_t_interface.conversion_receipt
    if conversion is None:
        raise ContractError("Arm T ecological conversion receipt is missing")
    if conversion["survey_scale_float64_hex"] != np.float64(arm_o.survey_scale).hex():
        raise ContractError("ecological interface used the wrong survey_scale")
    return {
        "method": arm_o.method,
        "cell": arm_o.cell,
        "ricker_cache_sha256_O": arm_o.cache_sha256,
        "ricker_cache_sha256_T": arm_t.cache_sha256,
        "policy_sha256_O": arm_o.policy_sha256,
        "policy_sha256_T": arm_t.policy_sha256,
        "surrogate_sha256_O": arm_o.surrogate_sha256,
        "surrogate_sha256_T": arm_t.surrogate_sha256,
        "identical_policy_bytes": True,
        "only_registered_abundance_interface_differs": True,
        "context_identity_demonstrated": True,
        "arm_o_interface": arm_o_interface.kind,
        "arm_t_interface": arm_t_interface.kind,
        "conversion_receipt_sha256": sha256_bytes(canonical_json_bytes(conversion)),
        "survey_scale_hex": np.float64(arm_o.survey_scale).hex(),
        "sigma": sigma,
    }


def validate_registered_ecological_policy_set(
    pairs: Sequence[tuple[SharedEcologicalPolicy, SharedEcologicalPolicy]],
) -> dict[str, str]:
    """Require one immutable O/T policy pair for every ecological method/species cell."""

    from .registration import CELLS

    expected = {(method, cell) for method in ECOLOGICAL_METHODS for cell in CELLS}
    if len(pairs) != len(expected):
        raise ContractError("ecological policy set is incomplete")
    observed: set[tuple[str, str]] = set()
    hashes: dict[str, str] = {}
    for arm_o, arm_t in pairs:
        arm_o.validate()
        arm_t.validate()
        identity = (arm_o.method, arm_o.cell)
        if identity != (arm_t.method, arm_t.cell) or identity in observed:
            raise ContractError("ecological policy set has a duplicate or cross-arm binding")
        if arm_o.serialized_components.keys() != arm_t.serialized_components.keys() or any(
            arm_o.serialized_components[component] != arm_t.serialized_components[component]
            for component in arm_o.serialized_components
        ):
            raise ContractError("ecological policy-set bytes differ across arms")
        observed.add(identity)
        hashes[f"{identity[0]}::{identity[1]}"] = arm_o.policy_sha256
    if observed != expected:
        raise ContractError("ecological policy set is missing a registered method/cell")
    return hashes


def evd_surrogate_applicability(method: str) -> dict[str, Any]:
    if method != EVD_METHOD:
        raise ContractError("EVD applicability requested for non-EVD method")
    return {
        "status": "DEFINITIONALLY_NOT_APPLICABLE",
        "sha256": None,
        "reason": "EVD trains on raw logged rewards and has no reward-surrogate artifact",
    }
