"""Fail-closed external Arm O/T seams for the registered I2B fast track.

This module is outside both frozen tracks.  Arm T is disabled unless the caller
holds the process-local capability issued here.  The module contains no truth
archive path and never accepts a future-state argument at runtime.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


END_TO_END_LABEL = "MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY"
METHODS = (
    "plus_adapted_ricker_only_pbvi",
    "moor_adapted_ricker_misspec_pbvi",
    "refplan",
    "ogsrl",
    "bamcts",
    "ensemble_value_disagreement_pessimism",
)
ECOLOGICAL_METHODS = frozenset(METHODS[:2])
FORBIDDEN_RUNTIME_FIELDS = frozenset(
    {
        "C",
        "entry",
        "initially_unsafe",
        "next_regime",
        "next_states",
        "r_base",
        "r_eff_true",
        "regime",
        "reward_true",
        "safety_penalty_applied",
        "theta",
        "family",
        "threshold",
        "parameters",
        "realized_future",
        "evaluator_info",
    }
)
VALID_ARTIFACT_STATUS = frozenset(
    {"APPLICABLE", "DEFINITIONALLY_NOT_APPLICABLE", "UNRESOLVED"}
)
_CAPABILITY_NONCE = object()


class IntegrationViolation(ValueError):
    """A fail-closed integration or information-boundary violation."""


class Arm(str, Enum):
    O = "O"
    T = "T"


@dataclass(frozen=True)
class ExactStateCapability:
    stage: str
    purpose: str
    _nonce: object


def issue_exact_state_capability(*, purpose: str) -> ExactStateCapability:
    if purpose != "i2b_synthetic_or_future_authorized_runtime":
        raise IntegrationViolation("capability purpose is not registered")
    return ExactStateCapability(stage="i2b", purpose=purpose, _nonce=_CAPABILITY_NONCE)


def _require_capability(capability: ExactStateCapability | None) -> None:
    if (
        not isinstance(capability, ExactStateCapability)
        or capability.stage != "i2b"
        or capability._nonce is not _CAPABILITY_NONCE
    ):
        raise IntegrationViolation("exact current state requires the I2B process capability")


def _sha(value: Any) -> str:
    payload = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_runtime_boundary(
    *,
    capability: ExactStateCapability | None,
    observation_noise_sigma: float,
    observation_scale: float,
    private_payload: Mapping[str, Any] | None = None,
    oracle_filter: bool = False,
) -> None:
    _require_capability(capability)
    if not np.isfinite(observation_noise_sigma) or observation_noise_sigma <= 1e-12:
        raise IntegrationViolation("sigma collapse is forbidden")
    if not np.isfinite(observation_scale) or observation_scale <= 1e-12:
        raise IntegrationViolation("observation-scale collapse is forbidden")
    if oracle_filter:
        raise IntegrationViolation("OracleStateFilter shortcuts are forbidden")
    supplied = set((private_payload or {}).keys())
    leaked = sorted(supplied & FORBIDDEN_RUNTIME_FIELDS)
    if leaked:
        raise IntegrationViolation(f"forbidden runtime fields supplied: {leaked}")


def direct_point_mass_latent_belief(
    raw_abundance: float,
    *,
    survey_scale: float,
    latent_grid: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map raw abundance through survey_scale, then choose the nearest latent bin."""

    raw = np.float64(raw_abundance)
    scale = np.float64(survey_scale)
    grid = np.asarray(latent_grid)
    if grid.dtype != np.dtype("float64") or grid.ndim != 1 or not np.isfinite(grid).all():
        raise IntegrationViolation("latent grid must be finite one-dimensional float64")
    if not np.isfinite(raw) or raw < 0.0 or not np.isfinite(scale) or scale <= 1e-12:
        raise IntegrationViolation("raw abundance and survey_scale are invalid")
    latent = raw / scale
    index = int(np.argmin(np.abs(grid - latent)))
    belief = np.zeros(grid.shape, dtype=np.float64)
    belief[index] = 1.0
    selected_raw = np.float64(grid[index] * scale)
    return belief, {
        "conversion": "latent_abundance = raw_abundance / survey_scale",
        "raw_abundance_float64_hex": raw.hex(),
        "survey_scale_float64_hex": scale.hex(),
        "latent_abundance_float64_hex": latent.hex(),
        "selected_index": index,
        "selected_latent_float64_hex": np.float64(grid[index]).hex(),
        "selected_raw_float64_hex": selected_raw.hex(),
        "raw_discretisation_absolute_error": float(abs(selected_raw - raw)),
    }


def exact_general_belief(
    base_belief: Any,
    exact_current_abundance: float,
    *,
    capability: ExactStateCapability,
    observation_noise_sigma: float,
    observation_scale: float,
    private_payload: Mapping[str, Any] | None = None,
    oracle_filter: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Replace only current state particles; preserve history/context and interface type."""

    validate_runtime_boundary(
        capability=capability,
        observation_noise_sigma=observation_noise_sigma,
        observation_scale=observation_scale,
        private_payload=private_payload,
        oracle_filter=oracle_filter,
    )
    states = np.asarray(base_belief.states)
    contexts = np.asarray(base_belief.contexts)
    regimes = np.asarray(base_belief.regimes)
    log_weights = np.asarray(base_belief.log_weights)
    if states.dtype != np.dtype("float64") or states.ndim != 1:
        raise IntegrationViolation("base belief states must be one-dimensional float64")
    raw = np.float64(exact_current_abundance)
    if not np.isfinite(raw) or raw < 0.0:
        raise IntegrationViolation("exact current abundance must be finite and nonnegative")
    n = states.size
    if n == 0 or contexts.shape[0] != n or regimes.shape[0] != n or log_weights.shape != (n,):
        raise IntegrationViolation("base belief particle arrays are misaligned")
    replacement = np.full(n, raw, dtype=np.float64)
    normalized_log_weights = np.full(n, -np.log(np.float64(n)), dtype=np.float64)
    try:
        emitted = replace(
            base_belief,
            states=replacement,
            log_weights=normalized_log_weights,
            contexts=contexts.copy(),
            regimes=regimes.copy(),
        )
    except TypeError as exc:
        raise IntegrationViolation("belief interface is not a replaceable dataclass") from exc
    if not np.array_equal(np.asarray(emitted.contexts), contexts):
        raise IntegrationViolation("history context changed during exact-state routing")
    if not np.array_equal(np.asarray(emitted.regimes), regimes):
        raise IntegrationViolation("public regime particles changed during exact-state routing")
    return emitted, {
        "changed": ["states", "log_weights"],
        "preserved": ["contexts", "regimes", "observation", "diagnostics"],
        "context_sha256": hashlib.sha256(contexts.tobytes()).hexdigest(),
        "runtime_next_states": "IMPOSSIBLE_NO_ARGUMENT",
    }


def assign_ecological_internal_beliefs(
    policy: Any,
    exact_current_abundance: float,
    *,
    capability: ExactStateCapability,
    observation_noise_sigma: float,
    observation_scale: float,
) -> list[dict[str, Any]]:
    """Assign fitted PLUS/MOOR internal beliefs directly; do not call a likelihood update."""

    validate_runtime_boundary(
        capability=capability,
        observation_noise_sigma=observation_noise_sigma,
        observation_scale=observation_scale,
    )
    targets: list[Any]
    if hasattr(policy, "candidates"):
        targets = list(policy.candidates)
    elif hasattr(policy, "pomdp"):
        targets = [policy]
    else:
        raise IntegrationViolation("policy lacks a registered PLUS/MOOR fitted POMDP seam")
    receipts = []
    for target in targets:
        pomdp = target.pomdp
        belief, receipt = direct_point_mass_latent_belief(
            exact_current_abundance,
            survey_scale=float(pomdp.model.survey_scale),
            latent_grid=np.asarray(pomdp.grid, dtype=np.float64),
        )
        target.internal_belief = belief
        receipts.append(receipt)
    return receipts


class ExternalMethodWrapper:
    """A route selector that leaves Arm O object identity and RNG state unchanged."""

    def __init__(self, method: str, *, arm: Arm = Arm.O, exact_state_enabled: bool = False):
        if method not in METHODS:
            raise IntegrationViolation(f"unregistered method: {method}")
        if arm is Arm.T and not exact_state_enabled:
            raise IntegrationViolation("Arm T is disabled by default")
        self.method = method
        self.arm = arm
        self.exact_state_enabled = exact_state_enabled

    def route_general(
        self,
        base_belief: Any,
        *,
        exact_current_abundance: float | None = None,
        capability: ExactStateCapability | None = None,
        observation_noise_sigma: float = 0.2,
        observation_scale: float = 1.0,
        private_payload: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if self.arm is Arm.O:
            if exact_current_abundance is not None or capability is not None or private_payload:
                raise IntegrationViolation("Arm O cannot receive an exact-state capability or data")
            return base_belief, {
                "arm": "O",
                "input_identity_preserved": True,
                "rng_state_sha256": _sha((rng.bit_generator.state if rng else {})),
            }
        if self.method in ECOLOGICAL_METHODS:
            raise IntegrationViolation("use the ecological internal-belief seam")
        if exact_current_abundance is None:
            raise IntegrationViolation("Arm T requires current exact abundance")
        before = _sha(rng.bit_generator.state if rng else {})
        emitted, receipt = exact_general_belief(
            base_belief,
            exact_current_abundance,
            capability=capability,  # type: ignore[arg-type]
            observation_noise_sigma=observation_noise_sigma,
            observation_scale=observation_scale,
            private_payload=private_payload,
        )
        after = _sha(rng.bit_generator.state if rng else {})
        if before != after:
            raise IntegrationViolation("wrapper consumed or changed RNG state")
        receipt.update({"arm": "T", "rng_state_sha256_before": before, "rng_state_sha256_after": after})
        return emitted, receipt


def validate_artifact_axis(axis: Mapping[str, Any]) -> None:
    status = axis.get("status")
    digest = axis.get("sha256")
    if status not in VALID_ARTIFACT_STATUS:
        raise IntegrationViolation("unknown artifact applicability status")
    if status == "APPLICABLE":
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64):
            raise IntegrationViolation("applicable frozen identity must be a real SHA-256")
    elif digest is not None:
        raise IntegrationViolation("non-applicable or unresolved artifact cannot carry a digest")


def interpretation_label(*, fitted_artifacts_equal: bool, residual_sigma_equal: bool) -> str:
    if not fitted_artifacts_equal or not residual_sigma_equal:
        return END_TO_END_LABEL
    return "FROZEN-FIT/STATE-INPUT-ONLY ELIGIBLE SUBJECT TO COMPLETE REGISTRY"
