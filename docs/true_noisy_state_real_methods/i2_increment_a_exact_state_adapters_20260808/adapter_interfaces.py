"""External, non-integrated exact-state adapter interfaces for synthetic I2A tests."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_NAMES = (
    "mean",
    "sd",
    "q10",
    "q50",
    "q90",
    "extinct",
    "ctx_prev_obs",
    "ctx_obs",
    "ctx_t",
    "ess_frac",
)
CONSTANT_THRESHOLD = np.float64(1e-8)
RUNTIME_CONSTANT_TOLERANCE = np.float64(1e-6)
MIN_LIKELIHOOD_SCALE = np.float64(1e-12)
ALIAS_MAP = {2: 0, 3: 0, 4: 0}
END_TO_END_LABEL = "MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY"
VALID_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_DIGEST_FIELDS = (
    "preprocessor_sha256",
    "feature_fit_sha256",
    "dynamics_fit_sha256",
    "surrogate_sha256",
    "reward_model_sha256",
    "safety_calibration_sha256",
    "ensemble_sha256",
    "policy_sha256",
)
FROZEN_FIT_CAPABLE_METHODS = frozenset(
    {
        "refplan",
        "plus_adapted_ricker_only_pbvi",
        "moor_adapted_ricker_misspec_pbvi",
    }
)


class ContractViolation(ValueError):
    """Raised when an adapter, preprocessing, or interpretation contract fails."""


def _require_float64(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype("float64"):
        raise ContractViolation(f"{name} must have dtype float64")
    if not np.isfinite(array).all():
        raise ContractViolation(f"{name} contains non-finite values")
    return array


def _sha256_jsonable(value: Any) -> str:
    payload = _strict_json_dumps(value)
    return hashlib.sha256(payload).hexdigest()


def _reject_nonfinite_json(token: str) -> None:
    raise ContractViolation(f"non-finite JSON token is forbidden: {token}")


def _strict_json_dumps(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractViolation("value is not strict-JSON serializable") from exc


def _strict_json_loads(payload: bytes) -> Any:
    try:
        return json.loads(payload.decode("utf-8"), parse_constant=_reject_nonfinite_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractViolation("serialized artifact is not strict JSON") from exc


@dataclass(frozen=True)
class ContextSnapshot:
    feature_names: tuple[str, ...]
    feature_shape: tuple[int, ...]
    feature_dtype: str
    protected_context_float64_hex: tuple[tuple[str, ...], ...]
    action_history: tuple[int, ...]
    observation_history_float64_hex: tuple[str, ...]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _strict_json_dumps(
                {
                    "feature_names": self.feature_names,
                    "feature_shape": self.feature_shape,
                    "feature_dtype": self.feature_dtype,
                    "protected_context_float64_hex": self.protected_context_float64_hex,
                    "action_history": self.action_history,
                    "observation_history_float64_hex": self.observation_history_float64_hex,
                }
            )
        ).hexdigest()


def capture_context_snapshot(
    features: np.ndarray,
    *,
    feature_names: Sequence[str],
    action_history: Sequence[int],
    observation_history: Sequence[float],
) -> ContextSnapshot:
    values = _require_float64("context features", features)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ContractViolation("context snapshot feature shape must be (n, 10)")
    names = tuple(feature_names)
    if names != FEATURE_NAMES:
        raise ContractViolation("context snapshot feature order mismatch")
    actions: list[int] = []
    for value in action_history:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ContractViolation("action history must contain integer action identifiers")
        actions.append(int(value))
    observations = np.asarray(tuple(observation_history), dtype=np.float64)
    if observations.ndim != 1 or not np.isfinite(observations).all():
        raise ContractViolation("observation history must be a finite one-dimensional sequence")
    return ContextSnapshot(
        feature_names=names,
        feature_shape=tuple(values.shape),
        feature_dtype=str(values.dtype),
        protected_context_float64_hex=tuple(
            tuple(np.float64(item).hex() for item in row) for row in values[:, 6:10]
        ),
        action_history=tuple(actions),
        observation_history_float64_hex=tuple(np.float64(item).hex() for item in observations),
    )


@dataclass(frozen=True)
class AdaptationReceipt:
    feature_names: tuple[str, ...]
    dtype: str
    construction_mode: str
    changed_fields: tuple[str, ...]
    preserved_fields: tuple[str, ...]
    action_history_sha256_before: str
    action_history_sha256_after: str
    observation_history_sha256_before: str
    observation_history_sha256_after: str
    preserved_context_sha256_before: str
    preserved_context_sha256_after: str
    maximum_absolute_assigned_sd: float
    exact_alias_checks: dict[str, bool]
    state_block_rank: int
    context_parity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "dtype": self.dtype,
            "construction_mode": self.construction_mode,
            "changed_fields": list(self.changed_fields),
            "preserved_fields": list(self.preserved_fields),
            "action_history_sha256_before": self.action_history_sha256_before,
            "action_history_sha256_after": self.action_history_sha256_after,
            "observation_history_sha256_before": self.observation_history_sha256_before,
            "observation_history_sha256_after": self.observation_history_sha256_after,
            "preserved_context_sha256_before": self.preserved_context_sha256_before,
            "preserved_context_sha256_after": self.preserved_context_sha256_after,
            "maximum_absolute_assigned_sd": self.maximum_absolute_assigned_sd,
            "exact_alias_checks": self.exact_alias_checks,
            "state_block_rank": self.state_block_rank,
            "context_parity": self.context_parity,
        }


def validate_constructed_features(features: np.ndarray) -> None:
    values = _require_float64("features", features)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ContractViolation("features must have shape (n, 10)")
    if not np.all(values[:, 1].view(np.uint64) == np.uint64(0)):
        raise ContractViolation("assigned SD must be literal positive float64 zero")
    for target, source in ALIAS_MAP.items():
        if not np.array_equal(values[:, target].view(np.uint64), values[:, source].view(np.uint64)):
            raise ContractViolation(f"column {target} must be a bit-exact alias of column {source}")
    if np.std(values[:, 1], dtype=np.float64) > CONSTANT_THRESHOLD:
        raise ContractViolation("assigned SD fails the registered std <= 1e-8 guard")
    if np.linalg.matrix_rank(values[:, :5]) != 1:
        raise ContractViolation("state-derived columns must have registered rank one")


def passes_constant_validation_guard(std_value: float) -> bool:
    value = np.float64(std_value)
    if not np.isfinite(value) or value < 0.0:
        raise ContractViolation("constant-guard standard deviation must be finite and nonnegative")
    return bool(value <= CONSTANT_THRESHOLD)


def validate_context_preservation(
    before: ContextSnapshot,
    after: ContextSnapshot,
) -> None:
    if before.feature_shape != after.feature_shape:
        raise ContractViolation("adapter changed interface shape")
    if before.feature_dtype != after.feature_dtype:
        raise ContractViolation("adapter changed interface dtype")
    if before.feature_names != after.feature_names:
        raise ContractViolation("adapter changed feature ordering")
    if before.protected_context_float64_hex != after.protected_context_float64_hex:
        raise ContractViolation("adapter changed or zeroed registered context features")
    if before.action_history != after.action_history:
        raise ContractViolation("adapter changed action history")
    if before.observation_history_float64_hex != after.observation_history_float64_hex:
        raise ContractViolation("adapter changed observation history")


class ExactStateFeatureAdapter:
    """Replace state-derived public features while preserving caller-supplied context."""

    def adapt(
        self,
        base_features: np.ndarray,
        exact_abundance: np.ndarray,
        *,
        feature_names: Sequence[str] = FEATURE_NAMES,
        observation_scale: float,
        observation_noise_sigma: float,
        action_history: Sequence[int],
        observation_history: Sequence[float],
    ) -> tuple[np.ndarray, AdaptationReceipt]:
        base = _require_float64("base_features", base_features)
        abundance = _require_float64("exact_abundance", exact_abundance)
        if tuple(feature_names) != FEATURE_NAMES:
            raise ContractViolation("feature names/order differ from the registered schema")
        if base.ndim != 2 or base.shape[1] != len(FEATURE_NAMES):
            raise ContractViolation("base feature order/shape must be the fixed ten-column schema")
        if abundance.shape != (base.shape[0],):
            raise ContractViolation("exact abundance must align one-to-one with feature rows")
        if np.any(abundance < 0.0):
            raise ContractViolation("exact abundance must be nonnegative")
        if not np.isfinite(observation_scale) or observation_scale <= MIN_LIKELIHOOD_SCALE:
            raise ContractViolation("observation_scale must remain positive; zero/epsilon routes fail")
        if (
            not np.isfinite(observation_noise_sigma)
            or observation_noise_sigma <= MIN_LIKELIHOOD_SCALE
        ):
            raise ContractViolation("observation_noise_sigma must remain positive; sigma collapse fails")

        before_context = capture_context_snapshot(
            base.copy(),
            feature_names=tuple(feature_names),
            action_history=tuple(action_history),
            observation_history=tuple(observation_history),
        )

        output = base.copy()
        transformed_state = np.log1p(abundance / np.float64(observation_scale))
        output[:, 0] = transformed_state
        output[:, 1] = np.float64(0.0)
        output[:, 2:5] = output[:, 0][:, None]
        output[:, 5] = (abundance <= 0.0).astype(np.float64)

        emitted_actions = tuple(int(value) for value in action_history)
        emitted_observations = tuple(float(value) for value in observation_history)
        after_context = capture_context_snapshot(
            output.copy(),
            feature_names=FEATURE_NAMES,
            action_history=emitted_actions,
            observation_history=emitted_observations,
        )
        validate_context_preservation(before_context, after_context)
        validate_constructed_features(output)
        action_hash_before = _sha256_jsonable(list(before_context.action_history))
        action_hash_after = _sha256_jsonable(list(after_context.action_history))
        observation_hash_before = _sha256_jsonable(
            list(before_context.observation_history_float64_hex)
        )
        observation_hash_after = _sha256_jsonable(
            list(after_context.observation_history_float64_hex)
        )
        receipt = AdaptationReceipt(
            feature_names=FEATURE_NAMES,
            dtype="float64",
            construction_mode="primary_direct_assignment",
            changed_fields=FEATURE_NAMES[:6],
            preserved_fields=FEATURE_NAMES[6:] + ("observation_history", "action_history"),
            action_history_sha256_before=action_hash_before,
            action_history_sha256_after=action_hash_after,
            observation_history_sha256_before=observation_hash_before,
            observation_history_sha256_after=observation_hash_after,
            preserved_context_sha256_before=before_context.sha256,
            preserved_context_sha256_after=after_context.sha256,
            maximum_absolute_assigned_sd=float(np.max(np.abs(output[:, 1]))),
            exact_alias_checks={
                str(target): bool(
                    np.array_equal(
                        output[:, target].view(np.uint64), output[:, source].view(np.uint64)
                    )
                )
                for target, source in ALIAS_MAP.items()
            },
            state_block_rank=int(np.linalg.matrix_rank(output[:, :5])),
            context_parity="PASS",
        )
        return output, receipt


@dataclass(frozen=True)
class PreprocessorArtifact:
    feature_names: tuple[str, ...]
    dtype: str
    offsets: tuple[float, ...]
    scales: tuple[float, ...]
    constant_mask: tuple[bool, ...]
    alias_map: tuple[tuple[int, int], ...]
    constant_threshold: float = float(CONSTANT_THRESHOLD)
    runtime_constant_tolerance: float = float(RUNTIME_CONSTANT_TOLERANCE)

    def to_bytes(self) -> bytes:
        payload = {
            "feature_names": self.feature_names,
            "dtype": self.dtype,
            "offsets_float64_hex": [np.float64(value).hex() for value in self.offsets],
            "scales_float64_hex": [np.float64(value).hex() for value in self.scales],
            "constant_mask": self.constant_mask,
            "alias_map": self.alias_map,
            "constant_threshold": np.float64(self.constant_threshold).hex(),
            "runtime_constant_tolerance": np.float64(self.runtime_constant_tolerance).hex(),
        }
        return _strict_json_dumps(payload)

    @classmethod
    def from_bytes(cls, serialized: bytes) -> "PreprocessorArtifact":
        payload = _strict_json_loads(serialized)
        expected_keys = {
            "feature_names",
            "dtype",
            "offsets_float64_hex",
            "scales_float64_hex",
            "constant_mask",
            "alias_map",
            "constant_threshold",
            "runtime_constant_tolerance",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise ContractViolation("serialized preprocessor fields are incomplete or unexpected")
        try:
            artifact = cls(
                feature_names=tuple(payload["feature_names"]),
                dtype=payload["dtype"],
                offsets=tuple(float.fromhex(value) for value in payload["offsets_float64_hex"]),
                scales=tuple(float.fromhex(value) for value in payload["scales_float64_hex"]),
                constant_mask=tuple(payload["constant_mask"]),
                alias_map=tuple(tuple(pair) for pair in payload["alias_map"]),
                constant_threshold=float.fromhex(payload["constant_threshold"]),
                runtime_constant_tolerance=float.fromhex(payload["runtime_constant_tolerance"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractViolation("serialized preprocessor field type/value is invalid") from exc
        validate_preprocessor_artifact(artifact)
        if artifact.to_bytes() != serialized:
            raise ContractViolation("serialized preprocessor is not in canonical round-trip form")
        return artifact

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def transform(self, features: np.ndarray) -> np.ndarray:
        validate_preprocessor_artifact(self)
        values = _require_float64("features", features)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ContractViolation("runtime feature shape/order mismatch")
        offsets = np.asarray(self.offsets, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        mask = np.asarray(self.constant_mask, dtype=bool)
        for index in np.flatnonzero(mask):
            if np.any(np.abs(values[:, index] - offsets[index]) > RUNTIME_CONSTANT_TOLERANCE):
                raise ContractViolation(f"constant feature {index} exceeds runtime tolerance")
        if values.shape[0] > 1:
            runtime_std = np.std(values, axis=0, dtype=np.float64)
            unexpected = np.flatnonzero((~mask) & (runtime_std <= CONSTANT_THRESHOLD))
            if unexpected.size:
                raise ContractViolation(
                    f"unexpectedly degenerate unmasked runtime columns: {unexpected.tolist()}"
                )
        transformed = np.empty_like(values)
        transformed[:, ~mask] = (values[:, ~mask] - offsets[~mask]) / scales[~mask]
        transformed[:, mask] = np.float64(0.0)
        if not np.isfinite(transformed).all():
            raise ContractViolation("non-finite transformed features")
        return transformed


def validate_preprocessor_artifact(
    artifact: PreprocessorArtifact, expected_sha256: str | None = None
) -> None:
    if artifact.feature_names != FEATURE_NAMES or artifact.dtype != "float64":
        raise ContractViolation("preprocessor feature order or dtype mismatch")
    if len(artifact.offsets) != 10 or len(artifact.scales) != 10 or len(artifact.constant_mask) != 10:
        raise ContractViolation("preprocessor array length mismatch")
    if artifact.alias_map != tuple(sorted(ALIAS_MAP.items())):
        raise ContractViolation("preprocessor alias map mismatch")
    offsets = np.asarray(artifact.offsets, dtype=np.float64)
    scales = np.asarray(artifact.scales, dtype=np.float64)
    mask = np.asarray(artifact.constant_mask, dtype=bool)
    if not np.isfinite(offsets).all() or not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ContractViolation("preprocessor offsets/scales are invalid")
    if not mask[1] or np.any(scales[mask].view(np.uint64) != np.float64(1.0).view(np.uint64)):
        raise ContractViolation("masked columns must use literal float64 scale 1.0")
    if expected_sha256 is not None and artifact.sha256 != expected_sha256:
        raise ContractViolation("preprocessor artifact hash mismatch")


def fit_end_to_end_preprocessor(
    features: np.ndarray, *, feature_names: Sequence[str] = FEATURE_NAMES
) -> PreprocessorArtifact:
    values = _require_float64("features", features)
    if tuple(feature_names) != FEATURE_NAMES:
        raise ContractViolation("preprocessor feature names/order mismatch")
    validate_constructed_features(values)
    offsets = np.mean(values, axis=0, dtype=np.float64)
    raw_std = np.std(values, axis=0, dtype=np.float64)
    constant_mask = raw_std <= CONSTANT_THRESHOLD
    scales = raw_std.copy()
    scales[constant_mask] = np.float64(1.0)
    if not constant_mask[1] or scales[1].view(np.uint64) != np.float64(1.0).view(np.uint64):
        raise ContractViolation("assigned SD must be masked with literal scale 1.0")
    artifact = PreprocessorArtifact(
        feature_names=FEATURE_NAMES,
        dtype="float64",
        offsets=tuple(float(value) for value in offsets),
        scales=tuple(float(value) for value in scales),
        constant_mask=tuple(bool(value) for value in constant_mask),
        alias_map=tuple(sorted(ALIAS_MAP.items())),
    )
    validate_preprocessor_artifact(artifact)
    return artifact


def validate_serialized_preprocessor_parity(
    offline_artifact: PreprocessorArtifact,
    serialized_runtime_artifact: bytes,
    features: np.ndarray,
    *,
    expected_state_block_rank: int,
) -> tuple[PreprocessorArtifact, dict[str, bool]]:
    validate_preprocessor_artifact(offline_artifact)
    runtime_artifact = PreprocessorArtifact.from_bytes(serialized_runtime_artifact)
    validate_preprocessor_artifact(runtime_artifact, expected_sha256=offline_artifact.sha256)
    values = _require_float64("parity features", features)
    offline_output = offline_artifact.transform(values)
    runtime_output = runtime_artifact.transform(values.copy())
    checks = {
        "distinct_reloaded_object": runtime_artifact is not offline_artifact,
        "serialized_bytes_canonical": runtime_artifact.to_bytes() == serialized_runtime_artifact,
        "artifact_sha256": runtime_artifact.sha256 == offline_artifact.sha256,
        "output_bits": np.array_equal(
            offline_output.view(np.uint64), runtime_output.view(np.uint64)
        ),
        "feature_schema": runtime_artifact.feature_names == offline_artifact.feature_names,
        "dtype": runtime_artifact.dtype == offline_artifact.dtype == "float64",
        "constant_mask": runtime_artifact.constant_mask == offline_artifact.constant_mask,
        "offset_bits": np.array_equal(
            np.asarray(runtime_artifact.offsets, dtype=np.float64).view(np.uint64),
            np.asarray(offline_artifact.offsets, dtype=np.float64).view(np.uint64),
        ),
        "scale_bits": np.array_equal(
            np.asarray(runtime_artifact.scales, dtype=np.float64).view(np.uint64),
            np.asarray(offline_artifact.scales, dtype=np.float64).view(np.uint64),
        ),
        "alias_map": runtime_artifact.alias_map == offline_artifact.alias_map,
        "rank_receipt": expected_state_block_rank
        == int(np.linalg.matrix_rank(values[:, :5]))
        == 1,
    }
    if not all(checks.values()):
        raise ContractViolation("serialized fit/runtime preprocessing parity failed")
    return runtime_artifact, checks


@dataclass(frozen=True)
class PointMassBeliefReceipt:
    method: str
    raw_abundance: float
    survey_scale: float
    latent_abundance: float
    selected_bin_index: int
    selected_latent_bin: float
    mapped_raw_abundance: float
    point_mass_sum: float


def adapt_point_mass_belief(
    method: str,
    raw_abundance: float,
    survey_scale: float,
    latent_grid: np.ndarray,
    registered_raw_grid: np.ndarray,
) -> tuple[np.ndarray, PointMassBeliefReceipt]:
    if method not in {"plus_adapted_ricker_only_pbvi", "moor_adapted_ricker_misspec_pbvi"}:
        raise ContractViolation("point-mass latent-grid adapter is restricted to PLUS/MOOR")
    grid = _require_float64("latent_grid", latent_grid)
    raw_grid = _require_float64("registered_raw_grid", registered_raw_grid)
    if grid.ndim != 1 or grid.size == 0 or np.any(np.diff(grid) <= 0.0):
        raise ContractViolation("latent grid must be finite, nonempty, and strictly increasing")
    if raw_grid.shape != grid.shape or np.any(np.diff(raw_grid) <= 0.0):
        raise ContractViolation("registered raw grid must align with the latent grid")
    if not np.isfinite(raw_abundance) or raw_abundance < 0.0:
        raise ContractViolation("raw abundance must be finite and nonnegative")
    if not np.isfinite(survey_scale) or survey_scale <= 0.0:
        raise ContractViolation("survey_scale must remain positive")
    latent = np.float64(raw_abundance) / np.float64(survey_scale)
    reconstructed_raw_grid = grid * np.float64(survey_scale)
    if not np.array_equal(reconstructed_raw_grid.view(np.uint64), raw_grid.view(np.uint64)):
        raise ContractViolation("latent-grid mapping does not match registered raw discretisation")
    selected = int(np.argmin(np.abs(grid - latent)))
    belief = np.zeros(grid.size, dtype=np.float64)
    belief[selected] = np.float64(1.0)
    receipt = PointMassBeliefReceipt(
        method=method,
        raw_abundance=float(raw_abundance),
        survey_scale=float(survey_scale),
        latent_abundance=float(latent),
        selected_bin_index=selected,
        selected_latent_bin=float(grid[selected]),
        mapped_raw_abundance=float(raw_grid[selected]),
        point_mass_sum=float(np.sum(belief)),
    )
    return belief, receipt


def build_preprocessing_receipt(
    *,
    method: str,
    species: str,
    cell: str,
    features: np.ndarray,
    artifact: PreprocessorArtifact,
    adaptation_receipt: AdaptationReceipt,
    arm_o_artifact_sha256: str,
    refplan_disclosure: Mapping[str, Any],
    rng_call_count_parity: bool,
    rng_state_advancement_parity: bool,
) -> dict[str, Any]:
    validate_preprocessor_artifact(artifact)
    values = _require_float64("receipt features", features)
    validate_constructed_features(values)
    serialized_artifact = artifact.to_bytes()
    runtime_artifact, parity_checks = validate_serialized_preprocessor_parity(
        artifact,
        serialized_artifact,
        values,
        expected_state_block_rank=adaptation_receipt.state_block_rank,
    )
    if not refplan_disclosure.get("disclosure_complete"):
        raise ContractViolation("RefPlan disclosure must be complete before receipt emission")
    feature_order_hash = hashlib.sha256("\n".join(FEATURE_NAMES).encode("utf-8")).hexdigest()
    fit_view_hash = hashlib.sha256(values.tobytes(order="C")).hexdigest()
    offsets = np.asarray(artifact.offsets, dtype=np.float64)
    scales = np.asarray(artifact.scales, dtype=np.float64)
    mask = np.asarray(artifact.constant_mask, dtype=bool)
    per_column = []
    for index, name in enumerate(FEATURE_NAMES):
        column = values[:, index]
        per_column.append(
            {
                "index": index,
                "name": name,
                "raw_mean": float(np.mean(column, dtype=np.float64)),
                "raw_std": float(np.std(column, dtype=np.float64)),
                "raw_min": float(np.min(column)),
                "raw_max": float(np.max(column)),
                "constant_mask": bool(mask[index]),
                "offset": float(offsets[index]),
                "scale": float(scales[index]),
                "transformed_constant_value": 0.0 if mask[index] else None,
                "collinear_with": ALIAS_MAP.get(index),
            }
        )
    return {
        "schema_version": "i1_zero_sd_preprocessing_receipt_v1",
        "method": method,
        "species": species,
        "cell": cell,
        "arm": "T",
        "regime": "end_to_end",
        "feature_names": list(FEATURE_NAMES),
        "feature_order_hash": feature_order_hash,
        "dtype": "float64",
        "construction_mode": "primary_direct_assignment",
        "per_column": per_column,
        "constant_mask": [bool(value) for value in mask],
        "alias_rank_collapse_map": {
            "arm_o_state_block_rank": 5,
            "arm_t_state_block_rank": adaptation_receipt.state_block_rank,
            "aliases": {str(target): source for target, source in ALIAS_MAP.items()},
        },
        "collinear_groups": [[0, 2, 3, 4]],
        "design_rank": adaptation_receipt.state_block_rank,
        "design_condition_number": None,
        "constant_threshold": float(CONSTANT_THRESHOLD),
        "threshold_comparison": "<=",
        "runtime_constant_tolerance": float(RUNTIME_CONSTANT_TOLERANCE),
        "finite_check_passed": True,
        "preprocessing_artifact_sha256": artifact.sha256,
        "fit_view_sha256": fit_view_hash,
        "runtime_transform_sha256": artifact.sha256,
        "arm_o_artifact_sha256": arm_o_artifact_sha256,
        "arm_t_artifact_sha256": artifact.sha256,
        "fit_runtime_parity_result": "PASS",
        "serialization_parity": {
            "serialized_artifact_sha256": hashlib.sha256(serialized_artifact).hexdigest(),
            "runtime_reloaded_artifact_sha256": runtime_artifact.sha256,
            "checks": parity_checks,
            "result": "PASS",
        },
        "parity_failures": [],
        "maximum_absolute_assigned_sd": adaptation_receipt.maximum_absolute_assigned_sd,
        "exact_alias_checks": adaptation_receipt.exact_alias_checks,
        "refplan_standardized_sd_offsets": {
            "tiger_reference": -2.7018,
            "fox_reference": -2.7348,
            "arm_o": refplan_disclosure["arm_o_standardized_offset"],
            "arm_t": refplan_disclosure["arm_t_standardized_offset"],
        },
        "refplan_mismatch_disclosure": dict(refplan_disclosure),
        "rng_call_count_parity": bool(rng_call_count_parity),
        "rng_state_advancement_parity": bool(rng_state_advancement_parity),
        "fitted_artifact_hashes": {},
        "available_residual_sigma_values": [],
        "frozen_fit_eligible": False,
        "eligibility_reason": "synthetic primary end-to-end receipt; no learned artifact constructed",
        "model_fit_axis_label": END_TO_END_LABEL,
        "zero_variance_events": [
            {
                "index": int(index),
                "stage": "synthetic_fit",
                "std": float(np.std(values[:, index], dtype=np.float64)),
                "action": "masked" if index == 1 else "masked_preexisting",
            }
            for index in np.flatnonzero(mask)
        ],
        "parity_result": "PASS",
    }


@dataclass(frozen=True)
class LearnedArtifactSnapshot:
    preprocessor_sha256: str
    feature_fit_sha256: str
    dynamics_fit_sha256: str
    residual_sigma_hex: tuple[str, ...]
    surrogate_sha256: str
    reward_model_sha256: str
    safety_calibration_sha256: str
    ensemble_sha256: str
    policy_sha256: str


def _snapshot_mapping(snapshot: LearnedArtifactSnapshot | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(snapshot, LearnedArtifactSnapshot):
        return {name: getattr(snapshot, name) for name in snapshot.__dataclass_fields__}
    if isinstance(snapshot, Mapping):
        return snapshot
    return {}


def _validate_artifact_snapshot(
    snapshot: LearnedArtifactSnapshot | Mapping[str, Any], arm: str
) -> tuple[dict[str, Any] | None, list[str]]:
    values = dict(_snapshot_mapping(snapshot))
    required = set(ARTIFACT_DIGEST_FIELDS) | {"residual_sigma_hex"}
    reasons: list[str] = []
    missing = sorted(required - set(values))
    extra = sorted(set(values) - required)
    if missing:
        reasons.append(f"{arm} omitted required identities: {missing}")
    if extra:
        reasons.append(f"{arm} contains unexpected identity fields: {extra}")
    for field_name in ARTIFACT_DIGEST_FIELDS:
        value = values.get(field_name)
        if not isinstance(value, str) or VALID_SHA256.fullmatch(value) is None:
            reasons.append(
                f"{arm}.{field_name} must be exactly 64 lowercase hexadecimal characters"
            )
    residual_values = values.get("residual_sigma_hex")
    if not isinstance(residual_values, tuple) or not residual_values:
        reasons.append(f"{arm}.residual_sigma_hex must be a nonempty tuple")
    else:
        for index, value in enumerate(residual_values):
            if not isinstance(value, str) or not value.strip():
                reasons.append(f"{arm}.residual_sigma_hex[{index}] is missing or non-string")
                continue
            try:
                parsed = float.fromhex(value)
            except ValueError:
                reasons.append(f"{arm}.residual_sigma_hex[{index}] is not parseable float hex")
                continue
            if not np.isfinite(parsed) or value != np.float64(parsed).hex():
                reasons.append(
                    f"{arm}.residual_sigma_hex[{index}] must be canonical finite float64 hex"
                )
    return (values if not reasons else None), reasons


def classify_interpretation(
    method: str,
    arm_o: LearnedArtifactSnapshot | Mapping[str, Any],
    arm_t: LearnedArtifactSnapshot | Mapping[str, Any],
    *,
    direct_point_mass_replacement: bool = False,
    refplan_frozen_fit_secondary: bool = False,
) -> dict[str, Any]:
    if method not in FROZEN_FIT_CAPABLE_METHODS:
        return {
            "frozen_fit_eligible": False,
            "causal_state_representation_label_permitted": False,
            "changed_artifacts": [],
            "identity_validation": "NOT_APPLICABLE",
            "eligibility_reason": "method has no registered frozen-fit eligibility",
            "label": END_TO_END_LABEL,
        }
    arm_o_values, arm_o_reasons = _validate_artifact_snapshot(arm_o, "Arm O")
    arm_t_values, arm_t_reasons = _validate_artifact_snapshot(arm_t, "Arm T")
    invalid_reasons = arm_o_reasons + arm_t_reasons
    if invalid_reasons:
        return {
            "frozen_fit_eligible": False,
            "causal_state_representation_label_permitted": False,
            "changed_artifacts": [],
            "identity_validation": "FAIL",
            "eligibility_reason": "; ".join(invalid_reasons),
            "label": END_TO_END_LABEL,
        }
    assert arm_o_values is not None and arm_t_values is not None
    changed = [name for name in sorted(arm_o_values) if arm_o_values[name] != arm_t_values[name]]
    if changed:
        return {
            "frozen_fit_eligible": False,
            "causal_state_representation_label_permitted": False,
            "changed_artifacts": changed,
            "identity_validation": "PASS",
            "eligibility_reason": "one or more valid required artifact identities changed",
            "label": END_TO_END_LABEL,
        }
    if method == "refplan":
        eligible = refplan_frozen_fit_secondary
        condition_reason = (
            "all required identities valid/equal and optional frozen-fit secondary requested"
            if eligible
            else "RefPlan optional frozen-fit secondary condition was not explicitly enabled"
        )
    else:
        eligible = direct_point_mass_replacement
        condition_reason = (
            "all required identities valid/equal and direct point-mass replacement demonstrated"
            if eligible
            else "PLUS/MOOR direct point-mass-only replacement was not demonstrated"
        )
    return {
        "frozen_fit_eligible": eligible,
        "causal_state_representation_label_permitted": False,
        "changed_artifacts": [],
        "identity_validation": "PASS",
        "eligibility_reason": condition_reason,
        "label": "FROZEN-FIT/STATE-INPUT-ONLY" if eligible else END_TO_END_LABEL,
    }


REQUIRED_REFPLAN_DISCLOSURE_FIELDS = {
    "fit_time_sd_mean",
    "fit_time_sd_std",
    "planner_root_sd",
    "arm_o_standardized_offset",
    "arm_t_standardized_offset",
    "species",
    "arm_t_removes_preexisting_mismatch",
}


def build_refplan_disclosure(values: Mapping[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_REFPLAN_DISCLOSURE_FIELDS - set(values)
    if missing:
        raise ContractViolation(f"mandatory RefPlan mismatch disclosure missing: {sorted(missing)}")
    extra = set(values) - REQUIRED_REFPLAN_DISCLOSURE_FIELDS
    if extra:
        raise ContractViolation(f"unexpected RefPlan disclosure fields: {sorted(extra)}")
    species = values["species"]
    references = {"amur_tiger": -2.7018, "crab_eating_fox": -2.7348}
    if species not in references:
        raise ContractViolation("RefPlan disclosure species is not a registered pilot species")
    numeric_fields = (
        "fit_time_sd_mean",
        "fit_time_sd_std",
        "planner_root_sd",
        "arm_o_standardized_offset",
        "arm_t_standardized_offset",
    )
    numeric_values: dict[str, float] = {}
    for field_name in numeric_fields:
        value = values[field_name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ContractViolation(f"RefPlan {field_name} must be numeric")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ContractViolation(f"RefPlan {field_name} must be finite")
        numeric_values[field_name] = numeric
    if numeric_values["fit_time_sd_mean"] < 0.0 or numeric_values["fit_time_sd_std"] < 0.0:
        raise ContractViolation("RefPlan fit-time SD mean/std must be nonnegative")
    if np.float64(numeric_values["planner_root_sd"]).view(np.uint64) != np.uint64(0):
        raise ContractViolation("RefPlan planner-root SD must be literal positive zero")
    if numeric_values["arm_o_standardized_offset"] != references[species]:
        raise ContractViolation("RefPlan Arm O offset is inconsistent with the species label")
    removes_mismatch = values["arm_t_removes_preexisting_mismatch"]
    if not isinstance(removes_mismatch, (bool, np.bool_)):
        raise ContractViolation("RefPlan mismatch-removal flag must be boolean")
    if bool(removes_mismatch):
        if np.float64(numeric_values["arm_t_standardized_offset"]).view(np.uint64) != np.uint64(0):
            raise ContractViolation("RefPlan Arm T offset must be literal zero when mismatch is removed")
    elif numeric_values["arm_t_standardized_offset"] != numeric_values[
        "arm_o_standardized_offset"
    ]:
        raise ContractViolation("RefPlan retained-mismatch Arm T offset must equal Arm O")
    receipt: dict[str, Any] = {
        **numeric_values,
        "species": species,
        "arm_t_removes_preexisting_mismatch": bool(removes_mismatch),
    }
    receipt.update(
        {
            "tiger_reference_offset": -2.7018,
            "fox_reference_offset": -2.7348,
            "registered_species_reference_offset": references[species],
            "interpretation_warning": (
                "Arm T may remove a pre-existing train/runtime SD mismatch; the contrast "
                "cannot automatically be attributed solely to improved abundance information."
            ),
            "strict_json_roundtrip": "PASS",
            "disclosure_complete": True,
        }
    )
    serialized = _strict_json_dumps(receipt)
    reloaded = _strict_json_loads(serialized)
    if reloaded != receipt:
        raise ContractViolation("RefPlan disclosure strict JSON round-trip failed")
    return reloaded


class RecordingRNG:
    """Record call order while delegating to a NumPy Generator."""

    def __init__(self, seed: int) -> None:
        self.generator = np.random.default_rng(seed)
        self.calls: list[dict[str, Any]] = []

    @property
    def state(self) -> dict[str, Any]:
        return copy.deepcopy(self.generator.bit_generator.state)

    def normal(self, loc: np.ndarray, scale: float) -> np.ndarray:
        loc_array = _require_float64("rng loc", loc)
        self.calls.append(
            {
                "operation": "normal",
                "shape": list(loc_array.shape),
                "ordinal": len(self.calls),
            }
        )
        return self.generator.normal(loc=loc_array, scale=np.float64(scale))
