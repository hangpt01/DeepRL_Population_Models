"""Fail-closed durable Slurm log-plan construction for corrected Stage B."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .common import ContractError, require_exact_keys, require_nonempty_string


LOG_PLAN_SCHEMA_VERSION = "corrected_stageb_durable_log_plan_v1"
_LOG_NAMES = {
    "arm_o": ("arm-o", "arm-o-%A_%a"),
    "arm_o_gate": ("arm-o-gate", "arm-o-gate-%j"),
    "arm_t": ("arm-t", "arm-t-%A_%a"),
    "finalizer": ("finalizer", "finalizer-%j"),
}


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _absolute_shared_path(value: Any, field: str) -> Path:
    raw = str(value) if isinstance(value, Path) else require_nonempty_string(value, field)
    path = Path(raw)
    if not path.is_absolute():
        raise ContractError(f"{field} must be absolute")
    if path.parts[:2] != ("/", "fs04"):
        raise ContractError(f"{field} must use durable shared /fs04 storage")
    if ".." in path.parts:
        raise ContractError(f"{field} must not contain parent traversal")
    return path


def _absolute_path(value: Any, field: str) -> Path:
    raw = str(value) if isinstance(value, Path) else require_nonempty_string(value, field)
    path = Path(raw)
    if not path.is_absolute():
        raise ContractError(f"{field} must be absolute")
    if ".." in path.parts:
        raise ContractError(f"{field} must not contain parent traversal")
    return path


def derive_durable_log_plan(
    shared_evidence_root: Path | str,
    scientific_output_root: Path | str,
) -> Mapping[str, Any]:
    """Derive the one collision-free log plan permitted by the registered roots."""

    evidence = _absolute_shared_path(shared_evidence_root, "shared evidence root")
    output = _absolute_shared_path(scientific_output_root, "scientific output root")
    if _is_below(evidence, output) or _is_below(output, evidence) or evidence == output:
        raise ContractError("durable logs and scientific results must use disjoint roots")
    templates = {}
    for role, (directory, stem) in _LOG_NAMES.items():
        role_root = evidence / "logs" / directory
        templates[role] = {
            "stdout": str(role_root / f"{stem}.out"),
            "stderr": str(role_root / f"{stem}.err"),
        }
    return {
        "schema_version": LOG_PLAN_SCHEMA_VERSION,
        "shared_evidence_root": str(evidence),
        "scientific_output_root": str(output),
        "templates": templates,
    }


def _require_real_directory(path: Path, field: str, *, writable: bool) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ContractError(f"{field} must be an existing real directory")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ContractError(f"{field} must not traverse symlinks")
    if writable and not os.access(resolved, os.W_OK | os.X_OK):
        raise ContractError(f"{field} is not writable and searchable")
    return resolved


def _reject_symlink_ancestors(path: Path, field: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ContractError(f"{field} must not traverse symlinks")


def validate_durable_log_plan(
    value: Any,
    *,
    require_existing: bool = False,
    require_writable: bool = False,
) -> Mapping[str, Any]:
    """Validate exact derivation, containment, placeholders, collisions, and real parents."""

    if not isinstance(value, Mapping):
        raise ContractError("durable log plan must be an object")
    require_exact_keys(
        value,
        {"schema_version", "shared_evidence_root", "scientific_output_root", "templates"},
        "durable log plan",
    )
    if value["schema_version"] != LOG_PLAN_SCHEMA_VERSION:
        raise ContractError("durable log plan schema mismatch")
    expected = derive_durable_log_plan(
        value["shared_evidence_root"], value["scientific_output_root"]
    )
    if value != expected:
        raise ContractError("durable log templates do not match canonical derivation")
    templates = value["templates"]
    if not isinstance(templates, Mapping):
        raise ContractError("durable log templates must be an object")
    evidence = Path(value["shared_evidence_root"])
    output = Path(value["scientific_output_root"])
    if require_existing:
        evidence = _require_real_directory(
            evidence, "shared evidence root", writable=require_writable
        )
        output = _require_real_directory(output, "scientific output root", writable=False)
    observed_paths: set[str] = set()
    for role in _LOG_NAMES:
        binding = templates.get(role)
        if not isinstance(binding, Mapping):
            raise ContractError(f"durable log binding is missing for {role}")
        require_exact_keys(binding, {"stdout", "stderr"}, f"{role} log binding")
        for stream in ("stdout", "stderr"):
            template = _absolute_shared_path(binding[stream], f"{role} {stream} log")
            if not _is_below(template, evidence):
                raise ContractError(f"{role} {stream} log escapes the shared evidence root")
            if template == output or _is_below(template, output):
                raise ContractError(f"{role} {stream} log enters scientific results")
            rendered = str(template)
            if rendered in observed_paths:
                raise ContractError("durable log templates collide")
            observed_paths.add(rendered)
            if role in {"arm_o", "arm_t"}:
                if "%A_%a" not in template.name or "%j" in template.name:
                    raise ContractError(f"{role} log must use collision-free %A_%a naming")
            elif "%j" not in template.name or "%A" in template.name or "%a" in template.name:
                raise ContractError(f"{role} log must use scalar %j naming")
            if require_existing:
                parent = _require_real_directory(
                    template.parent,
                    f"{role} {stream} log parent",
                    writable=require_writable,
                )
                if not _is_below(parent, evidence):
                    raise ContractError(f"{role} {stream} real parent escapes evidence root")
    return expected


def prepare_durable_log_directories(value: Any) -> Mapping[str, Any]:
    """Create the registered role directories once, rejecting symlinks and path rebinding."""

    plan = validate_durable_log_plan(value)
    evidence = Path(plan["shared_evidence_root"])
    _reject_symlink_ancestors(evidence, "shared evidence root")
    if evidence.exists():
        _require_real_directory(evidence, "shared evidence root", writable=True)
    else:
        evidence.mkdir(mode=0o700, parents=True)
        _require_real_directory(evidence, "shared evidence root", writable=True)
    for binding in plan["templates"].values():
        for stream in ("stdout", "stderr"):
            parent = Path(binding[stream]).parent
            _reject_symlink_ancestors(parent, "durable log parent")
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _require_real_directory(parent, "durable log parent", writable=True)
    return validate_durable_log_plan(plan, require_existing=True, require_writable=True)
