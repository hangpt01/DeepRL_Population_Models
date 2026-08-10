"""Corrected Stage B information-boundary and context-preservation facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from docs.true_noisy_state_real_methods.i2_increment_a_exact_state_adapters_20260808.adapter_interfaces import (  # noqa: E501
    FEATURE_NAMES,
    ContractViolation as I2AContractViolation,
    ExactStateFeatureAdapter,
)
from docs.true_noisy_state_real_methods.i2b_fasttrack_integration_canary_20260808.fasttrack_wrappers import (  # noqa: E501
    IntegrationViolation,
    exact_general_belief,
    issue_exact_state_capability,
    validate_runtime_boundary,
)

from .common import ContractError, require_exact_keys, require_sha256, sha256_bytes
from .evidence import REGISTERED_EPISODE_IDS, REGISTERED_HORIZON
from .registration import CELLS, METHODS


FORBIDDEN_KEYS = frozenset(
    {
        "next_states",
        "future_states",
        "future_state",
        "family",
        "family_identity",
        "kind",
        "allee",
        "allee_theta",
        "theta",
        "c",
        "r_base",
        "r_eff_true",
        "safety_threshold",
        "mvp_threshold",
        "reward_true",
        "reward_components",
        "entry",
        "initially_unsafe",
        "safety_penalty_applied",
        "benefit_population_term",
        "action_cost_term",
        "safety_penalty_term",
        "process_innovation",
        "observation_innovation",
        "regime",
        "next_regime",
        "private_metadata",
        "private_constants",
        "evaluator_info",
        "evaluator_only",
        "metadata_json",
        "original_truth_path",
        "truth.npz",
        "discounted_total_reward",
        "discounted_benefit_population",
        "discounted_action_cost",
        "discounted_safety_penalty",
        "collapse_indicator",
        "unsafe_indicator",
        "process_innovation_sha256",
        "observation_innovation_sha256",
        "rng_receipt",
    }
)

PUBLIC_METHOD_FIELDS = frozenset(
    {
        "noisy_observation",
        "selected_action",
        "public_observation_history",
        "public_action_history",
        "timestep",
        "rng_call_count",
        "rng_state_sha256",
    }
)
PUBLIC_DIAGNOSTIC_FIELDS = frozenset(
    {"public", "ess", "filter_resampled", "observation_log_likelihood"}
)


def build_task_information_boundary_receipt(
    *,
    registration_sha256: str,
    method: str,
    cell: str,
    arm: str,
    feature_order_sha256: str,
    context_sha256_before: Sequence[str],
    context_sha256_after: Sequence[str],
    observation_history_sha256: Sequence[str],
    action_history_sha256: Sequence[str],
) -> dict[str, Any]:
    """Build the strict task-level receipt for the method/evaluator boundary.

    This Arm-specific receipt deliberately does not claim that O/T interfaces are paired;
    that fact can only be demonstrated after Arm T completes.
    """

    require_sha256(registration_sha256, "information-boundary registration")
    require_sha256(feature_order_sha256, "information-boundary feature order")
    if method not in METHODS or cell not in CELLS or arm not in {"O", "T"}:
        raise ContractError("information-boundary task binding mismatch")
    series = {
        "context_sha256_before": context_sha256_before,
        "context_sha256_after": context_sha256_after,
        "observation_history_sha256": observation_history_sha256,
        "action_history_sha256": action_history_sha256,
    }
    parsed: dict[str, list[str]] = {}
    for field, values in series.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise ContractError(f"information-boundary {field} must be an ordered sequence")
        if len(values) != len(REGISTERED_EPISODE_IDS):
            raise ContractError(f"information-boundary {field} must cover every episode")
        parsed[field] = [require_sha256(value, f"information-boundary {field}") for value in values]
    if parsed["context_sha256_before"] != parsed["context_sha256_after"]:
        raise ContractError("registered public context changed across exact-state adaptation")
    interface = (
        "REGISTERED_ECOLOGICAL_ABUNDANCE_BELIEF_INTERFACE"
        if method.startswith(("plus_", "moor_"))
        else "AUDITED_I2A_EXACT_STATE_ADAPTER"
    )
    return {
        "schema_version": "corrected_stageb_task_information_boundary_v1",
        "registration_sha256": registration_sha256,
        "method": method,
        "cell": cell,
        "arm": arm,
        "episode_ids": list(REGISTERED_EPISODE_IDS),
        "timesteps_per_episode": REGISTERED_HORIZON,
        "observation_noise_sigma": 0.2,
        "feature_dtype": "float64",
        "feature_order_sha256": feature_order_sha256,
        "context_sha256_before": parsed["context_sha256_before"],
        "context_sha256_after": parsed["context_sha256_after"],
        "observation_history_sha256": parsed["observation_history_sha256"],
        "action_history_sha256": parsed["action_history_sha256"],
        "registered_interface": interface,
        "allowed_method_input_fields": sorted(PUBLIC_METHOD_FIELDS),
        "runtime_next_states": "IMPOSSIBLE_NO_ARGUMENT",
        "oracle_state_filter": False,
        "sigma_to_zero": False,
        "history_or_context_deleted": False,
        "evaluator_only_fields_excluded": True,
        "only_registered_abundance_interface_differs": "PENDING_PAIRED_ARM_T_VALIDATION",
        "result": "PASS",
    }


def validate_task_information_boundary_receipt(
    receipt: Mapping[str, Any],
    *,
    registration_sha256: str,
    method: str,
    cell: str,
    arm: str,
) -> None:
    """Rebuild a task boundary receipt and reject unknown or self-declared fields."""

    if not isinstance(receipt, Mapping):
        raise ContractError("task information-boundary receipt must be an object")
    expected_keys = {
        "schema_version",
        "registration_sha256",
        "method",
        "cell",
        "arm",
        "episode_ids",
        "timesteps_per_episode",
        "observation_noise_sigma",
        "feature_dtype",
        "feature_order_sha256",
        "context_sha256_before",
        "context_sha256_after",
        "observation_history_sha256",
        "action_history_sha256",
        "registered_interface",
        "allowed_method_input_fields",
        "runtime_next_states",
        "oracle_state_filter",
        "sigma_to_zero",
        "history_or_context_deleted",
        "evaluator_only_fields_excluded",
        "only_registered_abundance_interface_differs",
        "result",
    }
    require_exact_keys(receipt, expected_keys, "task information-boundary receipt")
    rebuilt = build_task_information_boundary_receipt(
        registration_sha256=registration_sha256,
        method=method,
        cell=cell,
        arm=arm,
        feature_order_sha256=receipt["feature_order_sha256"],
        context_sha256_before=receipt["context_sha256_before"],
        context_sha256_after=receipt["context_sha256_after"],
        observation_history_sha256=receipt["observation_history_sha256"],
        action_history_sha256=receipt["action_history_sha256"],
    )
    if dict(receipt) != rebuilt:
        raise ContractError("task information-boundary receipt is internally inconsistent")


def _walk_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            location = f"{prefix}.{name}" if prefix else name
            found.append(location)
            found.extend(_walk_keys(item, location))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found.extend(_walk_keys(item, f"{prefix}[{index}]"))
    return found


def reject_forbidden_payload(payload: Mapping[str, Any]) -> None:
    """Recursively validate the only public method payload shape."""

    if not isinstance(payload, Mapping):
        raise ContractError("method payload must be a mapping")
    unknown = sorted(set(payload) - PUBLIC_METHOD_FIELDS)
    if unknown:
        raise ContractError(f"unregistered or forbidden method payload keys: {unknown}")
    for path in _walk_keys(payload):
        leaf = path.rsplit(".", 1)[-1].split("[", 1)[0].lower()
        if leaf in FORBIDDEN_KEYS or "truth.npz" in leaf:
            raise ContractError(f"forbidden method payload key: {path}")
    if "noisy_observation" in payload:
        _finite_nonnegative(payload["noisy_observation"], "noisy_observation")
    if "selected_action" in payload:
        action = payload["selected_action"]
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < 11:
            raise ContractError("selected_action is invalid")
    if "public_observation_history" in payload:
        history = payload["public_observation_history"]
        if not isinstance(history, (list, tuple)):
            raise ContractError("public_observation_history must be a numeric sequence")
        for index, item in enumerate(history):
            _finite_nonnegative(item, f"public_observation_history[{index}]")
    if "public_action_history" in payload:
        actions = payload["public_action_history"]
        if not isinstance(actions, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 11
            for item in actions
        ):
            raise ContractError("public_action_history must contain registered action IDs")
    for field in ("timestep", "rng_call_count"):
        if field in payload and (
            isinstance(payload[field], bool)
            or not isinstance(payload[field], int)
            or payload[field] < 0
        ):
            raise ContractError(f"{field} must be a nonnegative integer")
    if "rng_state_sha256" in payload:
        require_sha256(payload["rng_state_sha256"], "method payload RNG state")


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ContractError(f"{field} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ContractError(f"{field} must be finite and nonnegative")
    return result


def _validate_public_belief(base_belief: Any) -> None:
    try:
        attributes = vars(base_belief)
    except TypeError as exc:
        raise ContractError("base belief must expose an inspectable registered interface") from exc
    allowed = {"states", "log_weights", "regimes", "contexts", "observation", "diagnostics"}
    if set(attributes) != allowed:
        raise ContractError(
            f"base belief attribute mismatch; missing={sorted(allowed - set(attributes))}, "
            f"extra={sorted(set(attributes) - allowed)}"
        )
    _finite_nonnegative(attributes["observation"], "base belief observation")
    diagnostics = attributes["diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise ContractError("base belief diagnostics must be a public mapping")
    unknown = sorted(set(diagnostics) - PUBLIC_DIAGNOSTIC_FIELDS)
    if unknown:
        raise ContractError(f"private or unregistered belief diagnostics: {unknown}")
    for field, value in diagnostics.items():
        if field == "filter_resampled":
            if not isinstance(value, bool):
                raise ContractError("filter_resampled must be boolean")
        else:
            _finite_nonnegative(value, f"public diagnostic {field}")


@dataclass(frozen=True)
class PublicContext:
    features: np.ndarray
    feature_names: tuple[str, ...]
    action_history: tuple[int, ...]
    observation_history: tuple[float, ...]
    timestep: int
    rng_call_count: int
    rng_state_sha256: str

    def validate(self) -> None:
        if self.features.dtype != np.dtype("float64"):
            raise ContractError("public context feature dtype must remain float64")
        if self.features.ndim != 2 or self.features.shape[1] != len(FEATURE_NAMES):
            raise ContractError("public context feature shape mismatch")
        if not np.isfinite(self.features).all():
            raise ContractError("public context features must be finite")
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ContractError("public context feature order mismatch")
        if self.timestep < 0 or self.rng_call_count < 0:
            raise ContractError("context timing/RNG counts must be nonnegative")
        if len(self.observation_history) != self.timestep + 1:
            raise ContractError("observation history length/timing mismatch")
        if len(self.action_history) != self.timestep:
            raise ContractError("action history length/timing mismatch")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 11
            for item in self.action_history
        ):
            raise ContractError("public action history contains an invalid action")
        for item in self.observation_history:
            _finite_nonnegative(item, "public observation history")
        require_sha256(self.rng_state_sha256, "public context RNG state")


def _context_identity(context: PublicContext, *, include_state_columns: bool) -> str:
    context.validate()
    columns = context.features if include_state_columns else context.features[:, 6:]
    payload = b"|".join(
        (
            columns.tobytes(order="C"),
            repr(context.feature_names).encode(),
            repr(context.action_history).encode(),
            repr(tuple(np.float64(item).hex() for item in context.observation_history)).encode(),
            str(context.timestep).encode(),
            str(context.rng_call_count).encode(),
            context.rng_state_sha256.encode(),
        )
    )
    return sha256_bytes(payload)


def adapt_exact_features(
    context: PublicContext,
    exact_abundance: np.ndarray,
    *,
    observation_scale: float,
    observation_noise_sigma: float,
) -> tuple[PublicContext, dict[str, Any]]:
    context.validate()
    abundance = np.asarray(exact_abundance)
    if abundance.dtype != np.dtype("float64"):
        raise ContractError("exact_abundance must already be float64; coercion is forbidden")
    before_preserved = _context_identity(context, include_state_columns=False)
    adapter = ExactStateFeatureAdapter()
    try:
        adapted, receipt = adapter.adapt(
            context.features.copy(),
            abundance,
            feature_names=context.feature_names,
            observation_scale=observation_scale,
            observation_noise_sigma=observation_noise_sigma,
            action_history=context.action_history,
            observation_history=context.observation_history,
        )
    except I2AContractViolation as exc:
        raise ContractError(str(exc)) from exc
    emitted = PublicContext(
        features=adapted,
        feature_names=context.feature_names,
        action_history=context.action_history,
        observation_history=context.observation_history,
        timestep=context.timestep,
        rng_call_count=context.rng_call_count,
        rng_state_sha256=context.rng_state_sha256,
    )
    after_preserved = _context_identity(emitted, include_state_columns=False)
    if before_preserved != after_preserved:
        raise ContractError("exact-state adapter mutated registered public context")
    return emitted, {
        "i2a_context_parity": receipt.context_parity,
        "preserved_context_sha256_before": before_preserved,
        "preserved_context_sha256_after": after_preserved,
        "rng_call_count_unchanged": True,
        "rng_state_unchanged": True,
        "runtime_next_states": "IMPOSSIBLE_NO_ARGUMENT",
    }


def validate_external_runtime_request(
    *,
    observation_noise_sigma: float,
    observation_scale: float,
    payload: Mapping[str, Any],
    filter_class_name: str,
) -> None:
    reject_forbidden_payload(payload)
    if filter_class_name not in {
        "PublicObservationFilter",
        "ExactStateFeatureAdapter",
        "RegisteredExactStateBridge",
    }:
        raise ContractError("unregistered or Oracle runtime filter routing is prohibited")
    capability = issue_exact_state_capability(purpose="i2b_synthetic_or_future_authorized_runtime")
    try:
        validate_runtime_boundary(
            capability=capability,
            observation_noise_sigma=observation_noise_sigma,
            observation_scale=observation_scale,
            private_payload=payload,
            oracle_filter=False,
        )
    except IntegrationViolation as exc:
        raise ContractError(str(exc)) from exc


def route_general_current_state(
    base_belief: Any,
    exact_current_abundance: float,
    *,
    observation_noise_sigma: float,
    observation_scale: float,
    payload: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    supplied = payload or {}
    reject_forbidden_payload(supplied)
    _validate_public_belief(base_belief)
    capability = issue_exact_state_capability(purpose="i2b_synthetic_or_future_authorized_runtime")
    try:
        emitted, receipt = exact_general_belief(
            base_belief,
            exact_current_abundance,
            capability=capability,
            observation_noise_sigma=observation_noise_sigma,
            observation_scale=observation_scale,
            private_payload=supplied,
            oracle_filter=False,
        )
        _validate_public_belief(emitted)
        return emitted, receipt
    except IntegrationViolation as exc:
        raise ContractError(str(exc)) from exc


class RuntimeDatasetView:
    """Method-facing current/public view with no original archive or next-state API."""

    def __init__(self, *, current_observation: float, public_history: Sequence[float]) -> None:
        self.current_observation = _finite_nonnegative(
            current_observation, "runtime current observation"
        )
        self.public_history = tuple(
            _finite_nonnegative(item, "runtime public history") for item in public_history
        )

    def open_original_truth(self, _path: Path) -> None:
        raise ContractError("methods must never open the original truth archive")

    def get(self, field: str) -> Any:
        if field in {"next_states", "future_states", "original_truth"}:
            raise ContractError(f"runtime access to {field} is prohibited")
        if field == "current_observation":
            return self.current_observation
        if field == "public_history":
            return self.public_history
        raise ContractError(f"unregistered runtime dataset field: {field}")
