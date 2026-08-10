"""Unique staging, full-tree validation, and last-operation atomic publication."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path, PurePath
from typing import Any, Callable, Mapping, Sequence

from .common import ContractError, canonical_json_bytes, sha256_file, strict_json_loads


SUCCESS_RECEIPT = "PUBLICATION_SUCCESS.json"
PENDING_SUCCESS_RECEIPT = ".PUBLICATION_SUCCESS.json.pending"
_STAGING_RE = re.compile(r"^(?P<target>[A-Za-z0-9][A-Za-z0-9_.-]*)\.tmp-(?P<nonce>[0-9a-f]{32})$")


def write_bytes_fsync(path: Path, payload: bytes) -> None:
    """Create one file exclusively, flush it, and fsync before returning."""

    if path.exists() or path.is_symlink():
        raise ContractError(f"refusing to overwrite task-local file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short task-local write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_task_staging(root: Path, task_identity: str) -> tuple[Path, Path]:
    """Create a unique, real task-local staging directory and its final target."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task_identity):
        raise ContractError("task publication identity contains unsafe characters")
    if root.is_symlink() or not root.is_dir():
        raise ContractError("publication root must be an existing real directory")
    target = root / task_identity
    staging = root / f"{task_identity}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    if staging.is_symlink() or not staging.is_dir():
        raise ContractError("unique task staging creation failed")
    return staging, target


def _normalized_required_paths(required_files: Sequence[str]) -> tuple[str, ...]:
    if not required_files:
        raise ContractError("required publication file list is empty")
    normalized: list[str] = []
    for raw in required_files:
        if not isinstance(raw, str) or not raw:
            raise ContractError("required publication paths must be nonempty strings")
        pure = PurePath(raw)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise ContractError(f"required publication path escapes staging: {raw}")
        value = PurePath(*pure.parts).as_posix()
        if value in {SUCCESS_RECEIPT, PENDING_SUCCESS_RECEIPT}:
            raise ContractError("publication receipt names are reserved")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ContractError("required publication paths duplicate after normalization")
    return tuple(sorted(normalized))


def _assert_no_linked_ancestor(staging: Path, candidate: Path) -> None:
    current = candidate
    while current != staging:
        if current.is_symlink():
            raise ContractError(f"publication path has a symlinked ancestor: {candidate}")
        current = current.parent
    if staging.is_symlink():
        raise ContractError("task staging root cannot be a symlink")


def validate_staging_tree(staging: Path, required_files: Sequence[str]) -> dict[str, str]:
    if not staging.is_dir() or staging.is_symlink():
        raise ContractError("task staging root must be a real directory")
    required = _normalized_required_paths(required_files)
    staging_real = staging.resolve(strict=True)
    observed: set[str] = set()
    for candidate in staging.rglob("*"):
        _assert_no_linked_ancestor(staging, candidate)
        if candidate.is_symlink():
            raise ContractError(f"publication tree contains a symlink: {candidate}")
        if candidate.is_file():
            real = candidate.resolve(strict=True)
            if not real.is_relative_to(staging_real):
                raise ContractError("publication file resolves outside staging")
            observed.add(candidate.relative_to(staging).as_posix())
        elif not candidate.is_dir():
            raise ContractError(f"publication tree contains an unsupported node: {candidate}")
    if observed != set(required):
        raise ContractError(
            f"publication full-tree coverage mismatch; missing={sorted(set(required) - observed)}, "
            f"extra={sorted(observed - set(required))}"
        )
    hashes: dict[str, str] = {}
    for relative in required:
        candidate = staging / relative
        _assert_no_linked_ancestor(staging, candidate)
        if candidate.is_symlink() or not candidate.is_file():
            raise ContractError(f"required publication file missing or linked: {relative}")
        hashes[relative] = sha256_file(candidate)
    return hashes


def publish_once(
    *,
    staging: Path,
    target: Path,
    required_files: Sequence[str],
    validator: Callable[[Path], Mapping[str, Any]],
) -> Path:
    """Publish with the directory rename as the final fallible filesystem operation."""

    match = _STAGING_RE.fullmatch(staging.name)
    if match is None or match.group("target") != target.name:
        raise ContractError("staging directory is not a unique target-bound temporary namespace")
    if staging.parent.is_symlink() or target.parent.is_symlink():
        raise ContractError("publication parent cannot be a symlink")
    if staging.parent.resolve(strict=True) != target.parent.resolve(strict=True):
        raise ContractError("staging and target must be siblings on one filesystem")
    if target.exists() or target.is_symlink():
        raise ContractError("target namespace already exists; retry/replacement is prohibited")
    lock = target.parent / f".{target.name}.publication.lock"
    write_bytes_fsync(
        lock,
        canonical_json_bytes(
            {
                "schema_version": "corrected_stageb_publication_lock_v1",
                "target": target.name,
                "staging_nonce": match.group("nonce"),
                "automatic_retry_permitted": False,
            }
        ),
    )
    hashes = validate_staging_tree(staging, required_files)
    validation = dict(validator(staging))
    if validation.get("result") != "PASS":
        raise ContractError("task-local validation failed; publication prohibited")
    if validate_staging_tree(staging, required_files) != hashes:
        raise ContractError("task files changed during publication validation")
    receipt = {
        "schema_version": "corrected_stageb_publication_success_v2",
        "result": "PASS",
        "files": hashes,
        "full_tree_coverage": True,
        "validation": validation,
        "automatic_retry_permitted": False,
    }
    pending_receipt = staging / PENDING_SUCCESS_RECEIPT
    write_bytes_fsync(pending_receipt, canonical_json_bytes(receipt))
    _fsync_directory(staging)
    published_success = staging / SUCCESS_RECEIPT
    os.rename(pending_receipt, published_success)
    _fsync_directory(staging)
    _fsync_directory(target.parent)
    # This atomic rename is intentionally the final filesystem operation. If it fails,
    # no target/success receipt is visible. If it succeeds, all content was prevalidated
    # and fsynced and the function has no later failure point.
    os.rename(staging, target)
    return target / SUCCESS_RECEIPT


def load_success_receipt(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContractError("publication success receipt is missing")
    try:
        value = strict_json_loads(path.read_bytes())
    except (OSError, ContractError) as exc:
        raise ContractError("publication success receipt is unreadable") from exc
    if (
        value.get("schema_version") != "corrected_stageb_publication_success_v2"
        or value.get("result") != "PASS"
        or value.get("automatic_retry_permitted") is not False
        or value.get("full_tree_coverage") is not True
    ):
        raise ContractError("publication receipt does not represent a terminal validated success")
    expected = set(value.get("files", {})) | {SUCCESS_RECEIPT}
    observed = {
        candidate.relative_to(path.parent).as_posix()
        for candidate in path.parent.rglob("*")
        if candidate.is_file() and not candidate.is_symlink()
    }
    if observed != expected:
        raise ContractError("published tree no longer matches its full-tree manifest")
    for relative, expected_hash in value["files"].items():
        if sha256_file(path.parent / relative) != expected_hash:
            raise ContractError("published file hash mismatch")
    return value
