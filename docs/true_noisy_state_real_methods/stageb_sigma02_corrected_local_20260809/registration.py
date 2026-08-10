"""Prospective registration bundle and registration-before-execution guard."""

from __future__ import annotations

import hmac
import platform
import re
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import (
    ContractError,
    canonical_json_bytes,
    require_exact_keys,
    require_git_sha,
    require_nonempty_string,
    require_sha256,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
)


METHODS = (
    "plus_adapted_ricker_only_pbvi",
    "moor_adapted_ricker_misspec_pbvi",
    "refplan",
    "ogsrl",
    "bamcts",
    "ensemble_value_disagreement_pessimism",
)
CELLS = ("amur_tiger__allee__sigma_0p2", "crab_eating_fox__allee__sigma_0p2")
ARMS = ("O", "T")
EXPECTED_TASKS = tuple((arm, cell, method) for arm in ARMS for cell in CELLS for method in METHODS)
REQUIRED_BUNDLE_KEYS = {
    "execution_authorization",
    "corrected_stageb_registration",
    "analysis_rules",
    "code_configuration_hashes",
    "task_manifest",
    "previous_results_disclosure",
    "stageb_interpreter_bindings",
}
INTERPRETER_IDENTITY_FIELDS = {
    "absolute_interpreter_path",
    "resolved_executable_path",
    "python_version",
    "full_python_version",
    "numpy_version",
}
EXPECTED_INTERPRETER_BINDINGS = (
    {
        "role": "ecological_paper_faithful",
        "track": "ecological",
        "absolute_interpreter_path": (
            "/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python"
        ),
        "resolved_executable_path": "/apps/miniforge3/24.3.0-0/miniforge3/bin/python3.10",
        "python_version": "3.10.14",
        "full_python_version": (
            "3.10.14 | packaged by conda-forge | (main, Mar 20 2024, 12:45:18) [GCC 12.3.0]"
        ),
        "numpy_version": "2.2.6",
        "methods": list(METHODS[:2]),
        "task_indices": [0, 1, 6, 7],
    },
    {
        "role": "general_registered",
        "track": "general",
        "absolute_interpreter_path": "/usr/bin/python",
        "resolved_executable_path": "/usr/bin/python3.9",
        "python_version": "3.9.25",
        "full_python_version": (
            "3.9.25 (main, Apr 17 2026, 00:00:00) \n[GCC 11.5.0 20240719 (Red Hat 11.5.0-14)]"
        ),
        "numpy_version": "1.23.5",
        "methods": list(METHODS[2:]),
        "task_indices": [2, 3, 4, 5, 8, 9, 10, 11],
    },
)
EXPECTED_COMMAND_ROLES = {
    "arm-o-gate": "general_registered",
    "finalize-inspection-only": "general_registered",
}
EXPECTED_CONFIG_HASHES = {
    "general_config_sha256": "4e878a9ec7d528af0bfb4948e78036dc864f0d2b158a2fdf5559b24798b1222b",
    "plus_config_sha256": "10a064344e07cac16a7b2d5bd717bdce109c8970148de4295d6ff3532297fa68",
    "moor_config_sha256": "a3655a846d951439177da350ddf40ddc6d4962b42b317582dd29e72ca05323bd",
}
CONFIG_PATHS = {
    "general_config_sha256": "experiments/accepted_general/configs/general_phase2e_full_sigma01_02.yaml",
    "plus_config_sha256": "experiments/accepted_p10/configs/plus_ricker_only_p10.yaml",
    "moor_config_sha256": "experiments/accepted_p10/configs/moor_ricker_p10.yaml",
}
EXPECTED_SOURCE_HASHES = {
    "ecological_source_only_sha256": "2b3b8ae6d2f8ff5ffb17c4885ded9e8f1f6b3c0cb662f393186fe4b4706a884e",
    "general_source_only_sha256": "f90cea6f28dcacb910b5e036bf9e09958715d00a2418fd0856a3d5a12856bdbd",
}
SOURCE_TRACKS = {
    "ecological_source_only_sha256": "src/tracks/ecological",
    "general_source_only_sha256": "src/tracks/general",
}
EXPECTED_DATASET_HASHES = {
    "amur_tiger__allee__sigma_0p2": "7e71172af4bc95b2c31291414af53371c1ea94393e01675d269b2b7c360324c9",
    "crab_eating_fox__allee__sigma_0p2": "688d580f1e47a61dcad9b8a6cb96f1d62835bfbd0b835554cf31c8ab8bdd0c13",
}
CANDIDATE_RELATIVE = Path(
    "docs/true_noisy_state_real_methods/stageb_sigma02_corrected_local_20260809"
)
DRIVER_RELATIVE = CANDIDATE_RELATIVE / "driver.py"
TEMPLATE_NAMES = (
    "analysis_rules.template.json",
    "code_configuration_hashes.template.json",
    "corrected_stageb_registration.template.json",
    "execution_authorization.template.json",
    "previous_results_disclosure.template.json",
    "task_manifest.template.json",
    "stageb_interpreter_bindings.template.json",
)
SELF_HASH_RE = re.compile(
    rb"(?m)^# SELF-NORMALIZED-SHA256: ([0-9a-f]{64})  "
    rb"docs/true_noisy_state_real_methods/stageb_sigma02_corrected_local_20260809/"
    rb"SOURCE_TEST_HASHES\.sha256$"
)


class FrozenRegistration:
    """Process-local capability issued only after complete bundle validation."""

    __slots__ = ("_payload", "_sha256", "_registration_id", "_seal")

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "FrozenRegistration":
        raise ContractError("FrozenRegistration capabilities cannot be constructed directly")

    @classmethod
    def _issue(
        cls,
        payload: bytes,
        digest: str,
        registration_id: str,
        *,
        issuer: object,
    ) -> "FrozenRegistration":
        if issuer is not _REGISTRATION_ISSUER:
            raise ContractError("invalid frozen-registration issuer")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_payload", payload)
        object.__setattr__(instance, "_sha256", digest)
        object.__setattr__(instance, "_registration_id", registration_id)
        object.__setattr__(instance, "_seal", _registration_seal(payload, digest, registration_id))
        return instance

    @property
    def payload(self) -> bytes:
        return self._payload

    @property
    def sha256(self) -> str:
        return self._sha256

    @property
    def registration_id(self) -> str:
        return self._registration_id

    def bundle(self) -> Mapping[str, Any]:
        self.authorize_return_path()
        value = strict_json_loads(self._payload)
        if not isinstance(value, Mapping):
            raise ContractError("frozen registration payload is not an object")
        return value

    def authorize_return_path(self) -> str:
        """Return the immutable bundle identity needed by any evaluator return sink."""

        try:
            payload = self._payload
            digest = self._sha256
            registration_id = self._registration_id
            seal = self._seal
        except AttributeError as exc:
            raise ContractError("frozen registration is not an issued process capability") from exc
        if sha256_bytes(payload) != digest:
            raise ContractError("frozen registration bytes no longer match their SHA-256")
        expected = _registration_seal(payload, digest, registration_id)
        if not hmac.compare_digest(seal, expected):
            raise ContractError("frozen registration is not an issued process capability")
        return digest

    def __reduce__(self) -> Any:
        raise TypeError("FrozenRegistration process capabilities cannot be serialized")


_REGISTRATION_ISSUER = object()
_REGISTRATION_SECRET = secrets.token_bytes(32)


def _registration_seal(payload: bytes, digest: str, registration_id: str) -> bytes:
    message = b"\0".join((payload, digest.encode("ascii"), registration_id.encode("utf-8")))
    return hmac.digest(_REGISTRATION_SECRET, message, "sha256")


def _validate_authorization(value: Mapping[str, Any]) -> None:
    require_exact_keys(
        value,
        {
            "schema_version",
            "authorization_id",
            "authorized",
            "scope",
            "authorized_by",
            "authorized_at_utc",
            "no_return_exists_at_authorization",
        },
        "execution_authorization",
    )
    if value["schema_version"] != "corrected_stageb_execution_authorization_v1":
        raise ContractError("execution authorization schema version mismatch")
    if value["authorized"] is not True:
        raise ContractError("scientific execution is not explicitly authorized")
    if value["no_return_exists_at_authorization"] is not True:
        raise ContractError("authorization must attest that no corrected return exists")
    require_nonempty_string(value["authorization_id"], "authorization_id")
    require_nonempty_string(value["authorized_by"], "authorized_by")
    require_nonempty_string(value["authorized_at_utc"], "authorized_at_utc")
    if value["scope"] != "corrected Stage B sigma=0.2 only":
        raise ContractError("execution authorization scope mismatch")


def _validate_registration(value: Mapping[str, Any]) -> str:
    required = {
        "schema_version",
        "registration_id",
        "status",
        "prospective_corrected_replication",
        "blinded_preregistration",
        "returns_exist_at_freeze",
        "cells",
        "methods",
        "arms",
        "offline_rows",
        "episodes",
        "episode_length",
        "evaluation_identities",
        "horizon",
        "discount",
        "num_actions",
        "sigma",
        "evaluator_family",
        "cpu_profile",
        "local_status_label",
    }
    require_exact_keys(value, required, "corrected_stageb_registration")
    if value["schema_version"] != "corrected_stageb_registration_v2":
        raise ContractError("corrected registration schema version mismatch")
    registration_id = require_nonempty_string(value["registration_id"], "registration_id")
    if value["status"] != "FROZEN_BEFORE_CORRECTED_RETURNS":
        raise ContractError("corrected registration is not frozen")
    if value["prospective_corrected_replication"] is not True:
        raise ContractError("registration must identify a prospective corrected replication")
    if value["blinded_preregistration"] is not False:
        raise ContractError("known earlier results prohibit a blinded-preregistration claim")
    if value["returns_exist_at_freeze"] is not False:
        raise ContractError("a registration cannot freeze after corrected returns exist")
    if tuple(value["cells"]) != CELLS or tuple(value["methods"]) != METHODS:
        raise ContractError("registered cell or method order mismatch")
    if tuple(value["arms"]) != ARMS:
        raise ContractError("registered arm order mismatch")
    scalar_expected = {
        "offline_rows": 4000,
        "episodes": 160,
        "episode_length": 25,
        "horizon": 50,
        "discount": 0.95,
        "num_actions": 11,
        "sigma": 0.2,
        "evaluator_family": "allee",
        "cpu_profile": "Intel Xeon Platinum 8452Y / xenon-8452Y / one CPU",
        "local_status_label": "LOCAL ARM64 DEVELOPMENT TEST — NOT SCIENTIFIC EVIDENCE",
    }
    for field, expected in scalar_expected.items():
        if value[field] != expected:
            raise ContractError(f"registered {field} mismatch")
    expected_ids = (
        tuple(range(7001, 7005))
        + tuple(range(7051, 7055))
        + tuple(range(7101, 7105))
        + tuple(range(7151, 7155))
        + tuple(range(7201, 7205))
    )
    if tuple(value["evaluation_identities"]) != expected_ids:
        raise ContractError("evaluation identities are incomplete or out of order")
    return registration_id


def _validate_analysis_rules(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "primary_estimand",
        "bootstrap",
        "transition_scale_rule",
        "near_constant_rule",
        "collapse_decomposition_rule",
        "evd_objective",
        "cross_species_pooling",
    }
    require_exact_keys(value, required, "analysis_rules")
    if value["schema_version"] != "corrected_stageb_analysis_rules_v2":
        raise ContractError("analysis-rules schema mismatch")
    if value["primary_estimand"] != "mean_return_T_minus_mean_return_O":
        raise ContractError("primary estimand mismatch")
    bootstrap = value["bootstrap"]
    if bootstrap != {
        "resamples": 100000,
        "rng": "NumPy PCG64",
        "seed": 20260808,
        "quantiles": [0.025, 0.975],
        "quantile_method": "linear",
        "resampling_unit": "intact paired evaluation episode",
    }:
        raise ContractError("paired-bootstrap rule mismatch")
    transition_scale = value["transition_scale_rule"]
    if transition_scale != {
        "numeric_threshold": None,
        "ratio_guard": "compute only when Arm O denominator is positive",
        "general_learned_dynamics_residual_sigma_floor": 0.02,
        "ecological_process_scale_floor": None,
        "floor_semantics": "general_learned_dynamics_only",
        "changed_label": "MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY",
        "bit_identity_required_for_frozen_fit": True,
    }:
        raise ContractError("transition-scale rule does not match scoped V4 contract")
    near = value["near_constant_rule"]
    if near != {
        "prospectively_declared": True,
        "descriptive_only": True,
        "maximum_action_fraction_gte": 0.98,
        "literal_constant_is_separate": True,
    }:
        raise ContractError("near-constant descriptive rule mismatch")
    if value["collapse_decomposition_rule"] != "collapse-entry step belongs to post-collapse":
        raise ContractError("collapse decomposition timing is not frozen")
    if value["evd_objective"] != "raw logged rewards; separately labelled":
        raise ContractError("EVD objective rule mismatch")
    if value["cross_species_pooling"] is not False:
        raise ContractError("tiger and fox must never be pooled")


def _source_only_hash(repository_root: Path, relative: str) -> tuple[int, str]:
    root = repository_root / relative
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    lines = b"".join(
        f"{sha256_file(path)}  {path.relative_to(repository_root).as_posix()}\n".encode()
        for path in files
    )
    return len(files), sha256_bytes(lines)


def _registration_templates_hash(repository_root: Path) -> str:
    template_root = repository_root / CANDIDATE_RELATIVE / "registration"
    hashes = {name: sha256_file(template_root / name) for name in TEMPLATE_NAMES}
    return sha256_bytes(canonical_json_bytes(hashes))


def _verify_source_manifest(repository_root: Path, claimed: str) -> None:
    candidate = repository_root / CANDIDATE_RELATIVE
    manifest = candidate / "SOURCE_TEST_HASHES.sha256"
    payload = manifest.read_bytes()
    match = SELF_HASH_RE.search(payload)
    if match is None or len(SELF_HASH_RE.findall(payload)) != 1:
        raise ContractError("source/test manifest has no unique normalized self-hash")
    normalized = payload[: match.start(1)] + b"0" * 64 + payload[match.end(1) :]
    recorded = match.group(1).decode("ascii")
    if sha256_bytes(normalized) != recorded or claimed != recorded:
        raise ContractError("source/test manifest normalized self-hash mismatch")
    standard: dict[str, str] = {}
    for line in payload.decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ContractError("malformed source/test manifest entry") from exc
        require_sha256(digest, "source/test manifest entry")
        if relative in standard:
            raise ContractError("duplicate source/test manifest path")
        standard[relative] = digest
    expected = {
        path.relative_to(repository_root).as_posix()
        for path in candidate.rglob("*")
        if path.is_file() and path != manifest
    }
    if set(standard) != expected:
        raise ContractError("source/test manifest coverage mismatch")
    for relative, digest in standard.items():
        if sha256_file(repository_root / relative) != digest:
            raise ContractError(f"source/test manifest content mismatch: {relative}")


def _validate_hashes(value: Mapping[str, Any], repository_root: Path) -> None:
    required = {
        "schema_version",
        "git_commit_sha",
        "source_manifest_sha256",
        "registration_templates_sha256",
        "stageb_driver_sha256",
        "general_config_sha256",
        "plus_config_sha256",
        "moor_config_sha256",
        "ecological_source_only_sha256",
        "general_source_only_sha256",
    }
    require_exact_keys(value, required, "code_configuration_hashes")
    if value["schema_version"] != "corrected_stageb_code_configuration_hashes_v2":
        raise ContractError("code/configuration hash schema mismatch")
    require_git_sha(value["git_commit_sha"], "code_configuration_hashes.git_commit_sha")
    for field in required - {"schema_version", "git_commit_sha"}:
        require_sha256(value[field], f"code_configuration_hashes.{field}")
    for field, expected in EXPECTED_CONFIG_HASHES.items():
        if (
            value[field] != expected
            or sha256_file(repository_root / CONFIG_PATHS[field]) != expected
        ):
            raise ContractError(f"accepted controlling configuration hash mismatch: {field}")
    for field, expected in EXPECTED_SOURCE_HASHES.items():
        count, observed = _source_only_hash(repository_root, SOURCE_TRACKS[field])
        expected_count = 52 if field.startswith("ecological") else 55
        if value[field] != expected or observed != expected or count != expected_count:
            raise ContractError(f"frozen source-only hash mismatch: {field}")
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("cannot verify the controlling Git commit") from exc
    if completed.stdout.strip() != value["git_commit_sha"]:
        raise ContractError("registered Git commit is not the checked-out controlling commit")
    expected_templates = _registration_templates_hash(repository_root)
    if value["registration_templates_sha256"] != expected_templates:
        raise ContractError("registration-template aggregate hash mismatch")
    if value["stageb_driver_sha256"] != sha256_file(repository_root / DRIVER_RELATIVE):
        raise ContractError("registered Stage B driver hash mismatch")
    _verify_source_manifest(repository_root, value["source_manifest_sha256"])


def _validate_task_manifest(
    value: Mapping[str, Any], code_configuration_hashes: Mapping[str, Any]
) -> None:
    require_exact_keys(value, {"schema_version", "task_count", "tasks"}, "task_manifest")
    if value["schema_version"] != "corrected_stageb_task_manifest_v1":
        raise ContractError("task manifest schema mismatch")
    tasks = value["tasks"]
    if (
        value["task_count"] != len(EXPECTED_TASKS)
        or not isinstance(tasks, list)
        or len(tasks) != len(EXPECTED_TASKS)
    ):
        raise ContractError("task manifest count mismatch")
    observed: list[tuple[str, str, str]] = []
    required_task_keys = {
        "task_index",
        "arm",
        "cell",
        "method",
        "config_sha256",
        "dataset_sha256",
        "artifact_plan_sha256",
        "evaluation_identity_sha256",
        "interpreter_role",
    }
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise ContractError("task manifest entry must be an object")
        require_exact_keys(task, required_task_keys, f"task_manifest.tasks[{index}]")
        if task["task_index"] != index:
            raise ContractError("task indices must be complete, unique, and ordered")
        observed.append((task["arm"], task["cell"], task["method"]))
        for field in (
            "config_sha256",
            "dataset_sha256",
            "artifact_plan_sha256",
            "evaluation_identity_sha256",
        ):
            require_sha256(task[field], f"task[{index}].{field}")
        ecological = task["method"].startswith(("plus_", "moor_"))
        expected_role = "ecological_paper_faithful" if ecological else "general_registered"
        if task["interpreter_role"] != expected_role:
            raise ContractError("task interpreter role does not match its registered method")
        config_field = (
            "plus_config_sha256"
            if task["method"].startswith("plus_")
            else "moor_config_sha256"
            if task["method"].startswith("moor_")
            else "general_config_sha256"
        )
        if task["config_sha256"] != code_configuration_hashes[config_field]:
            raise ContractError("task configuration hash is not bound to the frozen method config")
        if task["dataset_sha256"] != EXPECTED_DATASET_HASHES.get(task["cell"]):
            raise ContractError("task dataset hash does not match the controlling I1 cell binding")
    if tuple(observed) != EXPECTED_TASKS:
        raise ContractError("task manifest mapping is incomplete, duplicated, or reordered")
    for index in range(len(EXPECTED_TASKS) // 2):
        arm_o = tasks[index]
        arm_t = tasks[index + len(EXPECTED_TASKS) // 2]
        for field in (
            "cell",
            "method",
            "config_sha256",
            "dataset_sha256",
            "artifact_plan_sha256",
            "evaluation_identity_sha256",
            "interpreter_role",
        ):
            if arm_o[field] != arm_t[field]:
                raise ContractError(f"paired O/T task binding mismatch: {field}")


def _validate_interpreter_bindings(
    value: Mapping[str, Any], task_manifest: Mapping[str, Any]
) -> None:
    require_exact_keys(
        value,
        {"schema_version", "bindings", "command_roles"},
        "stageb_interpreter_bindings",
    )
    if value["schema_version"] != "corrected_stageb_interpreter_bindings_v1":
        raise ContractError("Stage B interpreter-binding schema mismatch")
    bindings = value["bindings"]
    if not isinstance(bindings, list) or len(bindings) != len(EXPECTED_INTERPRETER_BINDINGS):
        raise ContractError("registered interpreter roles are missing or extra")
    required_binding_keys = {
        "role",
        "track",
        *INTERPRETER_IDENTITY_FIELDS,
        "methods",
        "task_indices",
    }
    observed_by_role: dict[str, Mapping[str, Any]] = {}
    method_coverage: list[str] = []
    task_coverage: list[int] = []
    for index, (binding, expected) in enumerate(zip(bindings, EXPECTED_INTERPRETER_BINDINGS)):
        if not isinstance(binding, Mapping):
            raise ContractError("registered interpreter binding must be an object")
        require_exact_keys(
            binding,
            required_binding_keys,
            f"stageb_interpreter_bindings.bindings[{index}]",
        )
        if dict(binding) != expected:
            raise ContractError("registered interpreter identity or coverage mismatch")
        role = binding["role"]
        if role in observed_by_role:
            raise ContractError("registered interpreter role is duplicated")
        observed_by_role[role] = binding
        method_coverage.extend(binding["methods"])
        task_coverage.extend(binding["task_indices"])
        configured = Path(binding["absolute_interpreter_path"])
        resolved = Path(binding["resolved_executable_path"])
        if not configured.is_absolute() or not resolved.is_absolute():
            raise ContractError("registered interpreter paths must be absolute")
        try:
            actual_resolved = configured.resolve(strict=True)
            registered_resolved = resolved.resolve(strict=True)
        except OSError as exc:
            raise ContractError("registered interpreter path does not exist") from exc
        if not configured.is_file() or not resolved.is_file():
            raise ContractError("registered interpreter path is not a file")
        if actual_resolved != resolved or registered_resolved != resolved:
            raise ContractError("configured interpreter resolves outside its registered identity")
    if method_coverage != list(METHODS) or len(set(method_coverage)) != len(METHODS):
        raise ContractError("registered interpreter method coverage is incomplete or duplicated")
    if sorted(task_coverage) != list(range(12)) or len(set(task_coverage)) != 12:
        raise ContractError("registered interpreter task coverage is incomplete or duplicated")
    command_roles = value["command_roles"]
    if not isinstance(command_roles, Mapping) or dict(command_roles) != EXPECTED_COMMAND_ROLES:
        raise ContractError("registered non-task command-role mapping mismatch")
    tasks = task_manifest["tasks"]
    for task in tasks:
        logical_index = task["task_index"] % 12
        binding = observed_by_role.get(task["interpreter_role"])
        if binding is None:
            raise ContractError("task interpreter role has no registered binding")
        expected_track = (
            "ecological" if task["method"].startswith(("plus_", "moor_")) else "general"
        )
        if (
            binding["track"] != expected_track
            or task["method"] not in binding["methods"]
            or logical_index not in binding["task_indices"]
        ):
            raise ContractError("task role/method/track/index interpreter binding mismatch")


def _binding_by_role(frozen_registration: FrozenRegistration, role: str) -> Mapping[str, Any]:
    if not isinstance(frozen_registration, FrozenRegistration):
        raise ContractError("interpreter resolution requires a FrozenRegistration")
    frozen_registration.authorize_return_path()
    section = frozen_registration.bundle()["stageb_interpreter_bindings"]
    matches = [binding for binding in section["bindings"] if binding["role"] == role]
    if len(matches) != 1:
        raise ContractError("interpreter role does not resolve to exactly one binding")
    return matches[0]


def interpreter_binding_for_task(
    frozen_registration: FrozenRegistration, *, arm: str, task_index: int
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if arm not in ARMS:
        raise ContractError("interpreter task arm must be O or T")
    if isinstance(task_index, bool) or not isinstance(task_index, int) or not 0 <= task_index < 12:
        raise ContractError("interpreter task index must be in 0..11")
    manifest_index = task_index if arm == "O" else task_index + 12
    task = frozen_registration.bundle()["task_manifest"]["tasks"][manifest_index]
    binding = _binding_by_role(frozen_registration, task["interpreter_role"])
    if (
        task["method"] not in binding["methods"]
        or task_index not in binding["task_indices"]
        or task["interpreter_role"] != binding["role"]
    ):
        raise ContractError("task does not resolve to its registered interpreter binding")
    return task, binding


def interpreter_binding_for_command(
    frozen_registration: FrozenRegistration, command: str
) -> Mapping[str, Any]:
    command_roles = frozen_registration.bundle()["stageb_interpreter_bindings"]["command_roles"]
    role = command_roles.get(command)
    if role is None:
        raise ContractError("command has no registered interpreter role")
    return _binding_by_role(frozen_registration, role)


def observe_interpreter_identity() -> Mapping[str, str]:
    import numpy as np

    executable = Path(sys.executable)
    try:
        resolved = executable.resolve(strict=True)
    except OSError as exc:
        raise ContractError("current interpreter executable cannot be resolved") from exc
    return {
        "absolute_interpreter_path": sys.executable,
        "resolved_executable_path": str(resolved),
        "python_version": platform.python_version(),
        "full_python_version": sys.version,
        "numpy_version": np.__version__,
    }


def require_runtime_interpreter_binding(
    frozen_registration: FrozenRegistration,
    *,
    command: str,
    arm: str | None = None,
    task_index: int | None = None,
    observed: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if command in {"arm-o", "arm-t"}:
        expected_arm = "O" if command == "arm-o" else "T"
        if arm != expected_arm or task_index is None:
            raise ContractError("task interpreter context does not match the driver command")
        task, binding = interpreter_binding_for_task(
            frozen_registration, arm=arm, task_index=task_index
        )
        method: str | None = task["method"]
        logical_task_index: int | None = task_index
    else:
        if arm is not None or task_index is not None:
            raise ContractError("non-task interpreter context must not contain a task")
        binding = interpreter_binding_for_command(frozen_registration, command)
        method = None
        logical_task_index = None
    current = observe_interpreter_identity() if observed is None else observed
    if not isinstance(current, Mapping):
        raise ContractError("observed interpreter identity must be an object")
    require_exact_keys(current, INTERPRETER_IDENTITY_FIELDS, "observed interpreter identity")
    expected_identity = {field: binding[field] for field in INTERPRETER_IDENTITY_FIELDS}
    if dict(current) != expected_identity:
        mismatches = sorted(
            field for field in INTERPRETER_IDENTITY_FIELDS if current[field] != binding[field]
        )
        raise ContractError(f"registered interpreter identity mismatch: {mismatches}")
    return {
        "schema_version": "corrected_stageb_interpreter_identity_receipt_v1",
        "command": command,
        "role": binding["role"],
        "track": binding["track"],
        "method": method,
        "task_index": logical_task_index,
        "expected": dict(binding),
        "observed": dict(current),
        "result": "PASS",
    }


def validate_interpreter_identity_receipt(
    receipt: Mapping[str, Any],
    frozen_registration: FrozenRegistration,
    *,
    command: str,
    arm: str | None = None,
    task_index: int | None = None,
) -> None:
    require_exact_keys(
        receipt,
        {
            "schema_version",
            "command",
            "role",
            "track",
            "method",
            "task_index",
            "expected",
            "observed",
            "result",
        },
        "interpreter identity receipt",
    )
    if receipt["schema_version"] != "corrected_stageb_interpreter_identity_receipt_v1":
        raise ContractError("interpreter identity receipt schema mismatch")
    expected = require_runtime_interpreter_binding(
        frozen_registration,
        command=command,
        arm=arm,
        task_index=task_index,
        observed=receipt["observed"],
    )
    if dict(receipt) != expected:
        raise ContractError("interpreter identity receipt binding mismatch")


def _validate_disclosure(value: Mapping[str, Any]) -> None:
    require_exact_keys(
        value,
        {
            "schema_version",
            "earlier_sigma_0p2_results_known",
            "prospective_corrected_replication",
            "blinded_preregistration",
            "exploratory_results_scientifically_accepted",
            "required_statement",
        },
        "previous_results_disclosure",
    )
    if value["schema_version"] != "corrected_stageb_previous_results_disclosure_v1":
        raise ContractError("previous-results disclosure schema mismatch")
    expected = {
        "earlier_sigma_0p2_results_known": True,
        "prospective_corrected_replication": True,
        "blinded_preregistration": False,
        "exploratory_results_scientifically_accepted": False,
        "required_statement": (
            "Earlier sigma=0.2 exploratory results are known; this is a prospective "
            "corrected replication, not a blinded preregistration."
        ),
    }
    for field, required in expected.items():
        if value[field] != required:
            raise ContractError(f"previous-results disclosure mismatch: {field}")


def freeze_registration_bundle(
    bundle: Mapping[str, Any], *, repository_root: Path | None = None
) -> FrozenRegistration:
    """Validate and freeze the complete bundle before any return sink can exist."""

    require_exact_keys(bundle, REQUIRED_BUNDLE_KEYS, "registration bundle")
    _validate_authorization(bundle["execution_authorization"])
    registration_id = _validate_registration(bundle["corrected_stageb_registration"])
    _validate_analysis_rules(bundle["analysis_rules"])
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    _validate_hashes(bundle["code_configuration_hashes"], root)
    _validate_task_manifest(bundle["task_manifest"], bundle["code_configuration_hashes"])
    _validate_interpreter_bindings(bundle["stageb_interpreter_bindings"], bundle["task_manifest"])
    _validate_disclosure(bundle["previous_results_disclosure"])
    payload = canonical_json_bytes(bundle)
    return FrozenRegistration._issue(
        payload,
        sha256_bytes(payload),
        registration_id,
        issuer=_REGISTRATION_ISSUER,
    )


class EvaluatorReturnSink:
    """A return sink that is structurally impossible to construct without a freeze."""

    def __init__(self, frozen_registration: FrozenRegistration) -> None:
        if not isinstance(frozen_registration, FrozenRegistration):
            raise ContractError("return sink requires a FrozenRegistration")
        self.registration_sha256 = frozen_registration.authorize_return_path()
        self._frozen_registration = frozen_registration
        self._records: list[Mapping[str, Any]] = []

    def append(self, records: Sequence[Any]) -> None:
        """Validate and append one complete registered evaluator episode."""

        from .evidence import StepEvidence, reconstruct_episode

        if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
            raise ContractError("return sink accepts only a complete StepEvidence sequence")
        if not records or any(not isinstance(record, StepEvidence) for record in records):
            raise ContractError("return sink accepts only a complete StepEvidence sequence")
        bundle = self._frozen_registration.bundle()
        registration = bundle["corrected_stageb_registration"]
        if any(record.registration_sha256 != self.registration_sha256 for record in records):
            raise ContractError("episode evidence registration binding mismatch")
        receipt = reconstruct_episode(
            records,
            expected_horizon=registration["horizon"],
            expected_gamma=registration["discount"],
            expected_episode_ids=registration["evaluation_identities"],
        )
        self._records.append(receipt)

    @property
    def records(self) -> Sequence[Mapping[str, Any]]:
        return tuple(self._records)
