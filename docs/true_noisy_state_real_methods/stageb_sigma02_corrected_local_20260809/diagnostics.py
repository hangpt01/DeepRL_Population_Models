"""Transition, realized-posterior, predictive-dispersion, and activity receipts."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .common import ContractError, require_exact_keys, require_finite, require_sha256
from .evidence import REGISTERED_EPISODE_IDS, REGISTERED_HORIZON
from .registration import CELLS, METHODS


RESIDUAL_SIGMA_FLOOR = 0.02
NEAR_CONSTANT_THRESHOLD = 0.98


def _expected_member_count(method: str) -> int:
    if method.startswith("plus_"):
        return 8
    if method.startswith("moor_"):
        return 1
    return 5


def guarded_ratio(numerator: float, denominator: float) -> dict[str, Any]:
    top = require_finite(numerator, "ratio numerator", nonnegative=True)
    bottom = require_finite(denominator, "ratio denominator", nonnegative=True)
    if bottom <= 0.0:
        return {"value": None, "status": "NA_NONPOSITIVE_ARM_O_DENOMINATOR"}
    return {"value": top / bottom, "status": "COMPUTED"}


def _residual_arm_receipt(
    values: Sequence[float], artifact_hashes: Sequence[str], arm: str
) -> dict[str, Any]:
    if not values:
        raise ContractError(f"Arm {arm} residual_sigma values are missing")
    if len(values) != len(artifact_hashes):
        raise ContractError(f"Arm {arm} residual/artifact member count mismatch")
    parsed = [
        require_finite(value, f"Arm {arm} residual_sigma", nonnegative=True) for value in values
    ]
    if any(value < RESIDUAL_SIGMA_FLOOR for value in parsed):
        raise ContractError("residual_sigma below the frozen 0.02 floor")
    hashes = [require_sha256(value, f"Arm {arm} artifact hash") for value in artifact_hashes]
    return {
        "member_values": parsed,
        "aggregate_mean": float(np.mean(np.asarray(parsed, dtype=np.float64))),
        "aggregate_min": min(parsed),
        "aggregate_max": max(parsed),
        "floor": RESIDUAL_SIGMA_FLOOR,
        "floor_active": [
            bool(
                np.float64(value).view(np.uint64)
                == np.float64(RESIDUAL_SIGMA_FLOOR).view(np.uint64)
            )
            for value in parsed
        ],
        "artifact_hashes": hashes,
    }


def build_transition_diagnostics(
    *,
    registration_sha256: str,
    method: str,
    cell: str,
    arm_o_values: Sequence[float],
    arm_t_values: Sequence[float],
    arm_o_artifact_hashes: Sequence[str],
    arm_t_artifact_hashes: Sequence[str],
) -> dict[str, Any]:
    require_sha256(registration_sha256, "transition diagnostic registration")
    if method not in METHODS or cell not in CELLS:
        raise ContractError("transition diagnostic method/cell binding mismatch")
    expected_count = _expected_member_count(method)
    if len(arm_o_values) != expected_count or len(arm_t_values) != expected_count:
        raise ContractError("transition diagnostic ensemble/member count mismatch")
    arm_o = _residual_arm_receipt(arm_o_values, arm_o_artifact_hashes, "O")
    arm_t = _residual_arm_receipt(arm_t_values, arm_t_artifact_hashes, "T")
    if len(arm_o_values) != len(arm_t_values):
        raise ContractError("Arm O/T residual ensemble sizes differ")
    differences = [float(t_value - o_value) for o_value, t_value in zip(arm_o_values, arm_t_values)]
    ratios = [
        guarded_ratio(t_value, o_value) for o_value, t_value in zip(arm_o_values, arm_t_values)
    ]
    return {
        "schema_version": "corrected_stageb_transition_diagnostics_v1",
        "registration_sha256": registration_sha256,
        "method": method,
        "cell": cell,
        "arm_o": arm_o,
        "arm_t": arm_t,
        "raw_O": list(arm_o["member_values"]),
        "raw_T": list(arm_t["member_values"]),
        "T_minus_O": differences,
        "T_over_O_guarded": ratios,
        "aggregate_T_minus_O": arm_t["aggregate_mean"] - arm_o["aggregate_mean"],
        "aggregate_T_over_O_guarded": guarded_ratio(
            arm_t["aggregate_mean"], arm_o["aggregate_mean"]
        ),
        "bit_identical": all(
            np.float64(o_value).view(np.uint64) == np.float64(t_value).view(np.uint64)
            for o_value, t_value in zip(arm_o_values, arm_t_values)
        ),
    }


def build_arm_transition_diagnostics(
    *,
    registration_sha256: str,
    method: str,
    cell: str,
    arm: str,
    residual_sigma: Sequence[float],
    artifact_hashes: Sequence[str],
) -> dict[str, Any]:
    """Build the complete single-arm receipt available before the Arm O gate."""

    require_sha256(registration_sha256, "single-arm transition registration")
    if method not in METHODS or cell not in CELLS or arm not in {"O", "T"}:
        raise ContractError("single-arm transition binding mismatch")
    if len(residual_sigma) != _expected_member_count(method):
        raise ContractError("single-arm transition ensemble/member count mismatch")
    arm_receipt = _residual_arm_receipt(residual_sigma, artifact_hashes, arm)
    return {
        "schema_version": "corrected_stageb_arm_transition_diagnostics_v1",
        "registration_sha256": registration_sha256,
        "method": method,
        "cell": cell,
        "arm": arm,
        "member_ids": list(range(_expected_member_count(method))),
        "residual_sigma": list(arm_receipt["member_values"]),
        "aggregate_mean": arm_receipt["aggregate_mean"],
        "aggregate_min": arm_receipt["aggregate_min"],
        "aggregate_max": arm_receipt["aggregate_max"],
        "floor": arm_receipt["floor"],
        "floor_active": arm_receipt["floor_active"],
        "artifact_hashes": arm_receipt["artifact_hashes"],
    }


def _entropy(probabilities: np.ndarray) -> float:
    if probabilities.ndim != 1 or probabilities.size < 2:
        raise ContractError("posterior probability vector must have at least two members")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ContractError("posterior probabilities must be finite and nonnegative")
    total = float(np.sum(probabilities, dtype=np.float64))
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
        raise ContractError("posterior probabilities must sum to one")
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive), dtype=np.float64))


def build_realized_posterior_receipt(
    *,
    registration_sha256: str,
    method: str,
    cell: str,
    arm: str,
    episode_id: int,
    member_ids: Sequence[int],
    timesteps: Sequence[int],
    model_bank_sha256: str,
    posterior_probabilities: Sequence[Sequence[float]],
    registered_sharpness: Sequence[float] | None = None,
) -> dict[str, Any]:
    require_sha256(registration_sha256, "posterior registration")
    require_sha256(model_bank_sha256, "posterior model bank")
    if method != "bamcts" or cell not in CELLS or episode_id not in REGISTERED_EPISODE_IDS:
        raise ContractError("posterior method/cell/episode binding mismatch")
    if arm not in {"O", "T"}:
        raise ContractError("posterior arm must be O or T")
    if tuple(member_ids) != tuple(range(5)):
        raise ContractError("BA-MCTS posterior member identities must equal 0..4")
    if tuple(timesteps) != tuple(range(REGISTERED_HORIZON + 1)):
        raise ContractError("BA-MCTS posterior timesteps must cover initial through terminal")
    if len(posterior_probabilities) != REGISTERED_HORIZON + 1:
        raise ContractError("initial-only or partial posterior entropy is forbidden")
    arrays = [np.asarray(item, dtype=np.float64) for item in posterior_probabilities]
    member_count = arrays[0].size
    if member_count != len(member_ids):
        raise ContractError("posterior probability/member identity count mismatch")
    if any(item.size != member_count for item in arrays):
        raise ContractError("posterior member count changed during evaluation")
    entropies = [_entropy(item) for item in arrays]
    if registered_sharpness is not None:
        sharpness = [
            require_finite(item, "registered posterior sharpness", nonnegative=True)
            for item in registered_sharpness
        ]
        if len(sharpness) != len(entropies):
            raise ContractError("posterior sharpness length mismatch")
    else:
        sharpness = [float(math.exp(-value)) for value in entropies]
    return {
        "schema_version": "corrected_stageb_realized_posterior_v1",
        "registration_sha256": registration_sha256,
        "method": method,
        "cell": cell,
        "arm": arm,
        "episode_id": episode_id,
        "member_ids": list(member_ids),
        "timesteps": list(timesteps),
        "model_bank_sha256": model_bank_sha256,
        "posterior_probabilities": [item.tolist() for item in arrays],
        "member_count": member_count,
        "initial_entropy": entropies[0],
        "per_step_entropy": entropies[1:],
        "terminal_entropy": entropies[-1],
        "effective_model_count": [float(math.exp(value)) for value in entropies],
        "registered_sharpness": sharpness,
        "realized_updates_recorded": True,
    }


def pair_posterior_receipts(arm_o: Mapping[str, Any], arm_t: Mapping[str, Any]) -> dict[str, Any]:
    if arm_o.get("arm") != "O" or arm_t.get("arm") != "T":
        raise ContractError("posterior receipts must be paired O then T")
    if not arm_o.get("per_step_entropy") or not arm_t.get("per_step_entropy"):
        raise ContractError("paired posterior receipts require realized per-step entropy")
    identity_fields = (
        "registration_sha256",
        "method",
        "cell",
        "episode_id",
        "member_count",
        "member_ids",
        "timesteps",
    )
    if any(arm_o.get(field) != arm_t.get(field) for field in identity_fields):
        raise ContractError("paired posterior identity/length/member mismatch")
    if (
        len(arm_o["per_step_entropy"]) != REGISTERED_HORIZON
        or len(arm_t["per_step_entropy"]) != REGISTERED_HORIZON
    ):
        raise ContractError("paired posterior series is incomplete")
    return {
        "initial_entropy_O": arm_o["initial_entropy"],
        "initial_entropy_T": arm_t["initial_entropy"],
        "terminal_entropy_O": arm_o["terminal_entropy"],
        "terminal_entropy_T": arm_t["terminal_entropy"],
        "terminal_T_minus_O": arm_t["terminal_entropy"] - arm_o["terminal_entropy"],
        "mean_realized_entropy_O": float(np.mean(arm_o["per_step_entropy"])),
        "mean_realized_entropy_T": float(np.mean(arm_t["per_step_entropy"])),
    }


def build_refplan_predictive_dispersion(
    *,
    registration_sha256: str,
    cell: str,
    episode_id: int,
    timesteps: Sequence[int],
    arm_o_artifact_sha256: str,
    arm_t_artifact_sha256: str,
    arm_o: Sequence[float],
    arm_t: Sequence[float],
) -> dict[str, Any]:
    require_sha256(registration_sha256, "RefPlan dispersion registration")
    require_sha256(arm_o_artifact_sha256, "RefPlan Arm O artifact")
    require_sha256(arm_t_artifact_sha256, "RefPlan Arm T artifact")
    if cell not in CELLS or episode_id not in REGISTERED_EPISODE_IDS:
        raise ContractError("RefPlan dispersion cell/episode binding mismatch")
    if tuple(timesteps) != tuple(range(REGISTERED_HORIZON)):
        raise ContractError("RefPlan predictive dispersion timesteps are incomplete")
    if len(arm_o) != REGISTERED_HORIZON or len(arm_t) != REGISTERED_HORIZON:
        raise ContractError("RefPlan predictive dispersion requires complete paired series")
    parsed_o = [
        require_finite(value, "RefPlan Arm O dispersion", nonnegative=True) for value in arm_o
    ]
    parsed_t = [
        require_finite(value, "RefPlan Arm T dispersion", nonnegative=True) for value in arm_t
    ]
    return {
        "schema_version": "corrected_stageb_refplan_predictive_dispersion_v1",
        "registration_sha256": registration_sha256,
        "method": "refplan",
        "cell": cell,
        "episode_id": episode_id,
        "timesteps": list(timesteps),
        "arm_o_artifact_sha256": arm_o_artifact_sha256,
        "arm_t_artifact_sha256": arm_t_artifact_sha256,
        "per_step_O": parsed_o,
        "per_step_T": parsed_t,
        "per_step_T_minus_O": [t_value - o_value for o_value, t_value in zip(parsed_o, parsed_t)],
        "mean_O": float(np.mean(parsed_o)),
        "mean_T": float(np.mean(parsed_t)),
        "mean_T_minus_O": float(np.mean(parsed_t) - np.mean(parsed_o)),
    }


def build_arm_refplan_predictive_dispersion(
    *,
    registration_sha256: str,
    cell: str,
    arm: str,
    episode_id: int,
    timesteps: Sequence[int],
    artifact_sha256: str,
    values: Sequence[float],
) -> dict[str, Any]:
    """Build a complete RefPlan single-arm series without anticipating Arm T."""

    require_sha256(registration_sha256, "RefPlan single-arm registration")
    require_sha256(artifact_sha256, "RefPlan single-arm artifact")
    if cell not in CELLS or arm not in {"O", "T"}:
        raise ContractError("RefPlan single-arm binding mismatch")
    if episode_id not in REGISTERED_EPISODE_IDS:
        raise ContractError("RefPlan single-arm episode binding mismatch")
    if tuple(timesteps) != tuple(range(REGISTERED_HORIZON)):
        raise ContractError("RefPlan single-arm timesteps are incomplete")
    if len(values) != REGISTERED_HORIZON:
        raise ContractError("RefPlan single-arm dispersion series is incomplete")
    parsed = [
        require_finite(value, "RefPlan single-arm dispersion", nonnegative=True) for value in values
    ]
    return {
        "schema_version": "corrected_stageb_arm_refplan_predictive_dispersion_v1",
        "registration_sha256": registration_sha256,
        "method": "refplan",
        "cell": cell,
        "arm": arm,
        "episode_id": episode_id,
        "timesteps": list(timesteps),
        "artifact_sha256": artifact_sha256,
        "per_step": parsed,
        "mean": float(np.mean(parsed)),
    }


def classify_activity(
    actions: Sequence[int],
    *,
    registration_sha256: str,
    method: str,
    cell: str,
    arm: str,
    episode_id: int,
    num_actions: int = 11,
) -> dict[str, Any]:
    require_sha256(registration_sha256, "activity registration")
    if method not in METHODS or cell not in CELLS or arm not in {"O", "T"}:
        raise ContractError("activity method/cell/arm binding mismatch")
    if episode_id not in REGISTERED_EPISODE_IDS or num_actions != 11:
        raise ContractError("activity episode/action registry mismatch")
    if len(actions) != REGISTERED_HORIZON:
        raise ContractError("complete registered action sequence is required")
    parsed: list[int] = []
    for action in actions:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ContractError("actions must be integer identifiers")
        action_id = int(action)
        if not 0 <= action_id < num_actions:
            raise ContractError("action identifier is outside the registered action set")
        parsed.append(action_id)
    counts = Counter(parsed)
    frequencies = np.asarray(list(counts.values()), dtype=np.float64) / len(parsed)
    entropy = float(-np.sum(frequencies * np.log(frequencies), dtype=np.float64))
    literal_constant = len(counts) == 1
    maximum_fraction = float(np.max(frequencies))
    near_constant = (not literal_constant) and maximum_fraction >= NEAR_CONSTANT_THRESHOLD
    return {
        "schema_version": "corrected_stageb_activity_v1",
        "registration_sha256": registration_sha256,
        "method": method,
        "cell": cell,
        "arm": arm,
        "episode_id": episode_id,
        "actions": parsed,
        "sequence_length": len(parsed),
        "distinct_action_count": len(counts),
        "action_counts": {str(index): counts.get(index, 0) for index in range(num_actions)},
        "entropy_nats": entropy,
        "literal_constant_policy": literal_constant,
        "near_constant_descriptive": near_constant,
        "near_constant_rule": {
            "prospectively_declared": True,
            "maximum_action_fraction_gte": NEAR_CONSTANT_THRESHOLD,
            "descriptive_only": True,
        },
        "maximum_action_fraction": maximum_fraction,
        "discrimination_status": (
            "NON_DISCRIMINATING_LITERAL_CONSTANT" if literal_constant else "DECISION_ACTIVE"
        ),
    }


def _validate_activity_receipt(
    receipt: Mapping[str, Any],
    *,
    registration_sha256: str,
    method: str,
    cell: str,
    arm: str,
) -> None:
    if not isinstance(receipt, Mapping):
        raise ContractError("activity receipt must be an object")
    rebuilt = classify_activity(
        receipt.get("actions", ()),
        registration_sha256=registration_sha256,
        method=method,
        cell=cell,
        arm=arm,
        episode_id=receipt.get("episode_id"),
    )
    if dict(receipt) != rebuilt:
        raise ContractError("activity receipt is not derived from its complete action sequence")


def _validate_posterior_receipt(
    receipt: Mapping[str, Any],
    *,
    registration_sha256: str,
    cell: str,
    arm: str,
) -> None:
    if not isinstance(receipt, Mapping):
        raise ContractError("realized posterior receipt must be an object")
    rebuilt = build_realized_posterior_receipt(
        registration_sha256=registration_sha256,
        method="bamcts",
        cell=cell,
        arm=arm,
        episode_id=receipt.get("episode_id"),
        member_ids=receipt.get("member_ids", ()),
        timesteps=receipt.get("timesteps", ()),
        model_bank_sha256=receipt.get("model_bank_sha256"),
        posterior_probabilities=receipt.get("posterior_probabilities", ()),
        registered_sharpness=receipt.get("registered_sharpness"),
    )
    if dict(receipt) != rebuilt:
        raise ContractError("realized posterior receipt is internally inconsistent")


def validate_arm_task_diagnostics(
    receipts: Mapping[str, Any],
    *,
    registration_sha256: str,
    method: str,
    cell: str,
    arm: str,
) -> None:
    """Validate only recognized, complete task diagnostics for one execution arm."""

    if not isinstance(receipts, Mapping):
        raise ContractError("task diagnostics receipts must be an object")
    expected_keys = {"transition", "activity"}
    if method == "bamcts":
        expected_keys.add("realized_posterior")
    if method == "refplan":
        expected_keys.add("predictive_dispersion")
    require_exact_keys(receipts, expected_keys, "registered task diagnostics receipts")

    transition = receipts["transition"]
    if not isinstance(transition, Mapping):
        raise ContractError("single-arm transition receipt must be an object")
    rebuilt_transition = build_arm_transition_diagnostics(
        registration_sha256=registration_sha256,
        method=method,
        cell=cell,
        arm=arm,
        residual_sigma=transition.get("residual_sigma", ()),
        artifact_hashes=transition.get("artifact_hashes", ()),
    )
    if dict(transition) != rebuilt_transition:
        raise ContractError("single-arm transition receipt is internally inconsistent")

    activities = receipts["activity"]
    if not isinstance(activities, list) or len(activities) != len(REGISTERED_EPISODE_IDS):
        raise ContractError("activity evidence must cover every registered episode")
    for expected_episode, receipt in zip(REGISTERED_EPISODE_IDS, activities):
        if not isinstance(receipt, Mapping) or receipt.get("episode_id") != expected_episode:
            raise ContractError("activity evidence episode identities are incomplete or unordered")
        _validate_activity_receipt(
            receipt,
            registration_sha256=registration_sha256,
            method=method,
            cell=cell,
            arm=arm,
        )

    if method == "bamcts":
        posterior = receipts["realized_posterior"]
        if not isinstance(posterior, list) or len(posterior) != len(REGISTERED_EPISODE_IDS):
            raise ContractError("BA-MCTS posterior evidence must cover every registered episode")
        for expected_episode, receipt in zip(REGISTERED_EPISODE_IDS, posterior):
            if not isinstance(receipt, Mapping) or receipt.get("episode_id") != expected_episode:
                raise ContractError("BA-MCTS posterior episode identities are incomplete")
            _validate_posterior_receipt(
                receipt,
                registration_sha256=registration_sha256,
                cell=cell,
                arm=arm,
            )

    if method == "refplan":
        dispersion = receipts["predictive_dispersion"]
        if not isinstance(dispersion, list) or len(dispersion) != len(REGISTERED_EPISODE_IDS):
            raise ContractError("RefPlan dispersion evidence must cover every registered episode")
        for expected_episode, receipt in zip(REGISTERED_EPISODE_IDS, dispersion):
            if not isinstance(receipt, Mapping) or receipt.get("episode_id") != expected_episode:
                raise ContractError("RefPlan dispersion episode identities are incomplete")
            rebuilt = build_arm_refplan_predictive_dispersion(
                registration_sha256=registration_sha256,
                cell=cell,
                arm=arm,
                episode_id=expected_episode,
                timesteps=receipt.get("timesteps", ()),
                artifact_sha256=receipt.get("artifact_sha256"),
                values=receipt.get("per_step", ()),
            )
            if dict(receipt) != rebuilt:
                raise ContractError("RefPlan dispersion receipt is internally inconsistent")
