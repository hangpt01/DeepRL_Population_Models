"""Fail-closed information boundary for bounded I2 Increment A.

This module never discovers a dataset. Paths and synthetic records must be supplied by
the caller. The only archive reader implemented here reads NumPy headers for all fields
and the payload of ``metadata_json`` only. It cannot return scientific arrays.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REGISTERED_KEY_ORDER = (
    "C",
    "entry",
    "initially_unsafe",
    "metadata_json",
    "next_regime",
    "next_states",
    "r_base",
    "r_eff_true",
    "regime",
    "reward_true",
    "safety_penalty_applied",
    "states",
    "theta",
)

REGISTERED_ROW_COUNT = 4000
RAW_ABUNDANCE_UNIT = "raw_abundance"

FIELD_SPECS: dict[str, tuple[str, tuple[int, ...]]] = {
    "C": ("float64", (REGISTERED_ROW_COUNT,)),
    "entry": ("bool", (REGISTERED_ROW_COUNT,)),
    "initially_unsafe": ("bool", (REGISTERED_ROW_COUNT,)),
    "metadata_json": ("unicode", ()),
    "next_regime": ("int8", (REGISTERED_ROW_COUNT,)),
    "next_states": ("float64", (REGISTERED_ROW_COUNT,)),
    "r_base": ("float64", (REGISTERED_ROW_COUNT,)),
    "r_eff_true": ("float64", (REGISTERED_ROW_COUNT,)),
    "regime": ("int8", (REGISTERED_ROW_COUNT,)),
    "reward_true": ("float64", (REGISTERED_ROW_COUNT,)),
    "safety_penalty_applied": ("bool", (REGISTERED_ROW_COUNT,)),
    "states": ("float64", (REGISTERED_ROW_COUNT,)),
    "theta": ("float64", (REGISTERED_ROW_COUNT,)),
}

FORBIDDEN_FIELDS = frozenset(
    {
        "C",
        "entry",
        "initially_unsafe",
        "next_regime",
        "r_base",
        "r_eff_true",
        "regime",
        "reward_true",
        "safety_penalty_applied",
        "theta",
    }
)

CAPABILITY_METADATA_BINDING = "i2a_metadata_binding"
CAPABILITY_FUTURE_OFFLINE_STATES = "future_offline_states"
CAPABILITY_FUTURE_OFFLINE_NEXT_STATES = "future_offline_next_states"


class BoundaryViolation(ValueError):
    """Raised whenever an information-boundary check fails."""


@dataclass(frozen=True)
class SyntheticProvenance:
    generator: str
    origin: str = "synthetic_generated"
    derived_from_real_arrays: bool = False
    source_paths: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.origin != "synthetic_generated":
            raise BoundaryViolation("fixture origin must be synthetic_generated")
        if self.derived_from_real_arrays or self.source_paths:
            raise BoundaryViolation("synthetic fixtures must be independent of real arrays")
        if not self.generator.strip():
            raise BoundaryViolation("synthetic fixture generator identity is required")


@dataclass
class AccessReceipt:
    schema_version: str = "i2a_information_access_receipt_v1"
    requests: list[dict[str, Any]] = field(default_factory=list)
    header_only_fields: list[str] = field(default_factory=list)
    payload_loaded_fields: list[str] = field(default_factory=list)
    scientific_payload_loaded_fields: list[str] = field(default_factory=list)
    scientific_payload_read_bytes: int = 0
    metadata_keys_returned: list[str] = field(default_factory=list)
    real_scientific_arrays_materialized: int = 0

    def record_request(
        self, field_name: str, purpose: str, capability: str, allowed: bool, reason: str
    ) -> None:
        self.requests.append(
            {
                "field": field_name,
                "purpose": purpose,
                "capability": capability,
                "allowed": allowed,
                "reason": reason,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requests": self.requests,
            "header_only_fields": self.header_only_fields,
            "payload_loaded_fields": self.payload_loaded_fields,
            "scientific_payload_loaded_fields": self.scientific_payload_loaded_fields,
            "scientific_payload_read_bytes": self.scientific_payload_read_bytes,
            "metadata_keys_returned": self.metadata_keys_returned,
            "real_scientific_arrays_materialized": self.real_scientific_arrays_materialized,
            "boundary_result": "PASS"
            if not self.scientific_payload_loaded_fields
            and self.scientific_payload_read_bytes == 0
            and self.real_scientific_arrays_materialized == 0
            else "FAIL",
        }


class AccessController:
    """Explicit capability gate; the I2A stage permits binding metadata only."""

    def __init__(self, stage: str = "i2a", receipt: AccessReceipt | None = None) -> None:
        if stage != "i2a":
            raise BoundaryViolation("this bounded implementation supports stage i2a only")
        self.stage = stage
        self.receipt = receipt or AccessReceipt()

    @staticmethod
    def future_capabilities() -> dict[str, dict[str, Any]]:
        return {
            CAPABILITY_FUTURE_OFFLINE_STATES: {
                "field": "states",
                "runtime_permitted": False,
                "authorized_in_i2a": False,
            },
            CAPABILITY_FUTURE_OFFLINE_NEXT_STATES: {
                "field": "next_states",
                "runtime_permitted": False,
                "offline_fitting_only": True,
                "authorized_in_i2a": False,
            },
        }

    def request(self, field_name: str, purpose: str, capability: str) -> None:
        if field_name not in REGISTERED_KEY_ORDER:
            self.receipt.record_request(
                field_name, purpose, capability, False, "unregistered or additional field"
            )
            raise BoundaryViolation(f"unregistered field requested: {field_name}")
        if field_name in FORBIDDEN_FIELDS:
            self.receipt.record_request(
                field_name, purpose, capability, False, "registered private field is forbidden"
            )
            raise BoundaryViolation(f"forbidden private field requested: {field_name}")
        if field_name == "metadata_json":
            allowed = capability == CAPABILITY_METADATA_BINDING and purpose == "public_hash_binding"
            self.receipt.record_request(
                field_name,
                purpose,
                capability,
                allowed,
                "public_dataset_sha256 binding only" if allowed else "metadata purpose denied",
            )
            if not allowed:
                raise BoundaryViolation("metadata_json is restricted to public hash binding")
            return
        if field_name == "next_states" and purpose in {
            "runtime",
            "deployment",
            "action_selection",
            "planning_root",
        }:
            self.receipt.record_request(
                field_name, purpose, capability, False, "future information forbidden at runtime"
            )
            raise BoundaryViolation("next_states cannot be requested by a runtime interface")
        self.receipt.record_request(
            field_name,
            purpose,
            capability,
            False,
            "future capability represented but not authorized in I2A",
        )
        raise BoundaryViolation(f"{field_name} array access is not authorized in I2A")


def _dtype_matches(key: str, dtype: np.dtype[Any]) -> bool:
    expected, _ = FIELD_SPECS[key]
    if expected == "unicode":
        return dtype.kind == "U"
    return dtype == np.dtype(expected)


def _validate_key_order(keys: Sequence[str]) -> None:
    if tuple(keys) != REGISTERED_KEY_ORDER:
        raise BoundaryViolation(
            f"truth key order/set mismatch: got {tuple(keys)!r}, expected {REGISTERED_KEY_ORDER!r}"
        )


def validate_synthetic_archive(
    archive: Mapping[str, np.ndarray],
    *,
    provenance: SyntheticProvenance,
    declared_units: Mapping[str, str],
    row_alignment_tokens: Mapping[str, str],
    expected_public_dataset_sha256: str,
) -> dict[str, Any]:
    """Validate a caller-supplied synthetic archive; real-derived fixtures are rejected."""

    provenance.validate()
    _validate_key_order(tuple(archive.keys()))
    if declared_units.get("states") != RAW_ABUNDANCE_UNIT:
        raise BoundaryViolation("states unit must be raw_abundance")
    if declared_units.get("next_states") != RAW_ABUNDANCE_UNIT:
        raise BoundaryViolation("next_states unit must be raw_abundance")

    row_tokens: set[str] = set()
    for key in REGISTERED_KEY_ORDER:
        value = archive[key]
        if not isinstance(value, np.ndarray):
            raise BoundaryViolation(f"{key} must be an explicitly supplied ndarray")
        _, expected_shape = FIELD_SPECS[key]
        if value.shape != expected_shape:
            raise BoundaryViolation(f"wrong shape for {key}: {value.shape!r}")
        if not _dtype_matches(key, value.dtype):
            raise BoundaryViolation(f"wrong dtype for {key}: {value.dtype}")
        if key != "metadata_json":
            token = row_alignment_tokens.get(key)
            if not token:
                raise BoundaryViolation(f"missing row-alignment token for {key}")
            row_tokens.add(token)
            if value.dtype.kind in "fci" and not np.isfinite(value).all():
                raise BoundaryViolation(f"non-finite synthetic values in {key}")
    if len(row_tokens) != 1:
        raise BoundaryViolation("row alignment tokens differ")

    try:
        metadata = json.loads(str(archive["metadata_json"].item()))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise BoundaryViolation("metadata_json is not a valid scalar JSON object") from exc
    if set(metadata) != {"public_dataset_sha256"}:
        raise BoundaryViolation("synthetic metadata must expose only public_dataset_sha256")
    if metadata["public_dataset_sha256"] != expected_public_dataset_sha256:
        raise BoundaryViolation("public dataset binding mismatch")
    return {
        "schema_result": "PASS",
        "synthetic_provenance": provenance.__dict__,
        "key_order": list(REGISTERED_KEY_ORDER),
        "row_count": REGISTERED_ROW_COUNT,
        "row_alignment_token": next(iter(row_tokens)),
        "binding": expected_public_dataset_sha256,
        "real_scientific_arrays_materialized": 0,
    }


def _read_npy_header(stream: Any) -> tuple[tuple[int, ...], np.dtype[Any], bool]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise BoundaryViolation(f"unsupported npy header version: {version}")
    return tuple(shape), np.dtype(dtype), bool(fortran_order)


class MetadataOnlyNpzInspector:
    """Inspect archive headers and the allowlisted metadata scalar only."""

    def __init__(self, controller: AccessController | None = None) -> None:
        self.controller = controller or AccessController()

    def inspect(
        self,
        path: str | Path,
        *,
        expected_public_dataset_sha256: str,
        declared_units: Mapping[str, str],
    ) -> dict[str, Any]:
        supplied_path = Path(path)
        if supplied_path.suffix != ".npz":
            raise BoundaryViolation("metadata inspector accepts an explicitly supplied .npz only")
        if declared_units.get("states") != RAW_ABUNDANCE_UNIT or declared_units.get(
            "next_states"
        ) != RAW_ABUNDANCE_UNIT:
            raise BoundaryViolation("registered state units are raw_abundance")

        headers: dict[str, dict[str, Any]] = {}
        with zipfile.ZipFile(supplied_path, "r") as archive:
            member_names = archive.namelist()
            expected_members = [f"{key}.npy" for key in REGISTERED_KEY_ORDER]
            if member_names != expected_members:
                raise BoundaryViolation("missing, additional, or reordered NPZ members")
            for key, member_name in zip(REGISTERED_KEY_ORDER, member_names):
                with archive.open(member_name, "r") as member:
                    shape, dtype, fortran_order = _read_npy_header(member)
                _, expected_shape = FIELD_SPECS[key]
                if shape != expected_shape or not _dtype_matches(key, dtype):
                    raise BoundaryViolation(f"header schema mismatch for {key}")
                headers[key] = {
                    "shape": list(shape),
                    "dtype": str(dtype),
                    "fortran_order": fortran_order,
                }
                self.controller.receipt.header_only_fields.append(key)

            self.controller.request(
                "metadata_json", "public_hash_binding", CAPABILITY_METADATA_BINDING
            )
            metadata_bytes = archive.read("metadata_json.npy")
            metadata_array = np.load(io.BytesIO(metadata_bytes), allow_pickle=False)
            try:
                metadata = json.loads(str(metadata_array.item()))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise BoundaryViolation("metadata_json payload is invalid") from exc
            if metadata.get("public_dataset_sha256") != expected_public_dataset_sha256:
                raise BoundaryViolation("public dataset binding mismatch")

        receipt = self.controller.receipt
        receipt.payload_loaded_fields.append("metadata_json")
        receipt.metadata_keys_returned.append("public_dataset_sha256")
        if receipt.scientific_payload_loaded_fields or receipt.scientific_payload_read_bytes:
            raise BoundaryViolation("scientific payload access instrumentation is nonzero")
        return {
            "public_dataset_sha256": expected_public_dataset_sha256,
            "headers": headers,
            "access_receipt": receipt.to_dict(),
        }
