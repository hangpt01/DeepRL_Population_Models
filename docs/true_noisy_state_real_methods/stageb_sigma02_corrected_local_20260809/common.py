"""Strict, deterministic helpers shared by corrected Stage B contracts."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from pathlib import Path
from typing import Any, Mapping


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
PLACEHOLDER_TOKENS = frozenset({"", "tbd", "todo", "unknown", "placeholder", "unresolved"})
REGISTERED_CPU_MODEL = "Intel Xeon Platinum 8452Y"
_TRADEMARK_MARKER_RE = re.compile(r"\((?:R|TM)\)", re.IGNORECASE)


class ContractError(ValueError):
    """Raised whenever a corrected-workflow contract fails closed."""


def reject_nonfinite_json(token: str) -> None:
    raise ContractError(f"non-finite JSON value is forbidden: {token}")


def strict_json_loads(payload: bytes) -> Any:
    try:
        return json.loads(payload.decode("utf-8"), parse_constant=reject_nonfinite_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("payload is not strict UTF-8 JSON") from exc


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError("value is not canonical strict-JSON serializable") from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{field} must be exactly 64 lowercase hexadecimal characters")
    return value


def require_git_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or GIT_SHA_RE.fullmatch(value) is None:
        raise ContractError(f"{field} must be exactly 40 lowercase hexadecimal characters")
    return value


def require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a nonempty string")
    if value.strip().lower() in PLACEHOLDER_TOKENS or "TO_BE_" in value.upper():
        raise ContractError(f"{field} contains a placeholder")
    return value


def require_finite(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{field} must be finite")
    if nonnegative and result < 0.0:
        raise ContractError(f"{field} must be nonnegative")
    return result


def require_exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ContractError(f"{field} key mismatch; missing={missing}, extra={extra}")


def normalize_cpu_model_identity(value: Any) -> str:
    """Normalize only trademark markers and whitespace in a complete CPU model identity."""

    raw = require_nonempty_string(value, "CPU model identity")
    normalized = " ".join(_TRADEMARK_MARKER_RE.sub(" ", raw).split())
    if not normalized:
        raise ContractError("CPU model identity is empty after normalization")
    return normalized


def require_registered_cpu_model(
    observed: Any,
    *,
    expected: Any = REGISTERED_CPU_MODEL,
) -> Mapping[str, str]:
    """Require complete normalized equality and return the auditable identity receipt."""

    expected_raw = require_nonempty_string(expected, "expected CPU model identity")
    observed_raw = require_nonempty_string(observed, "observed CPU model identity")
    expected_normalized = normalize_cpu_model_identity(expected_raw)
    observed_normalized = normalize_cpu_model_identity(observed_raw)
    if observed_normalized != expected_normalized:
        raise ContractError(
            "corrected Stage B CPU mismatch: "
            f"expected {expected_normalized!r}, observed {observed_normalized!r}"
        )
    return {
        "schema_version": "corrected_stageb_cpu_identity_v1",
        "expected_raw": expected_raw,
        "expected_normalized": expected_normalized,
        "observed_raw": observed_raw,
        "observed_normalized": observed_normalized,
        "result": "PASS",
    }


def validate_cpu_identity_receipt(value: Any) -> Mapping[str, str]:
    """Reconstruct a CPU receipt so raw text cannot be rebound to a passing normalization."""

    if not isinstance(value, Mapping):
        raise ContractError("CPU identity receipt must be an object")
    require_exact_keys(
        value,
        {
            "schema_version",
            "expected_raw",
            "expected_normalized",
            "observed_raw",
            "observed_normalized",
            "result",
        },
        "CPU identity receipt",
    )
    rebuilt = require_registered_cpu_model(
        value["observed_raw"],
        expected=value["expected_raw"],
    )
    if dict(value) != rebuilt:
        raise ContractError("CPU identity receipt is not canonically reconstructable")
    if rebuilt["expected_normalized"] != REGISTERED_CPU_MODEL:
        raise ContractError("CPU identity receipt is not bound to the registered model")
    return rebuilt


def observed_cpu_model() -> str:
    """Read the first kernel-reported CPU model without opening scientific data."""

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def require_current_registered_cpu_model() -> Mapping[str, str]:
    """Apply the canonical registered CPU guard to the current host."""

    return require_registered_cpu_model(observed_cpu_model())
