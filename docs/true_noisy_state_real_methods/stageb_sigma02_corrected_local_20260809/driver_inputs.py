"""Deterministic producer and strict validator for the sealed Stage B driver descriptor."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical_plan import (
    DRIVER_INPUTS_FILENAME,
    DRIVER_INPUTS_SCHEMA_VERSION,
    ROLE_NAMESPACES,
    validate_rng_contract_document,
)
from .common import (
    ContractError,
    canonical_json_bytes,
    require_exact_keys,
    require_git_sha,
    require_sha256,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
)
from .parity import (
    HISTORICAL_CANARY_IDENTITY,
    is_legacy_fast_track_canary_path,
    validate_parity_baseline_binding,
)


PATH_RECEIPT_KEYS = {"path", "sha256"}
FORBIDDEN_DESCRIPTOR_KEYS = frozenset(
    {
        "truth",
        "truth_path",
        "truth.npz",
        "next_states",
        "future_states",
        "hidden_family",
        "hidden_parameters",
        "safety_threshold",
        "reward_true",
        "evaluator_info",
        "signing_seed",
        "gate_signing_seed",
    }
)


def canonical_driver_inputs_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value)


def driver_inputs_sha256(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_driver_inputs_bytes(value))


def validate_no_forbidden_descriptor_keys(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            leaf = str(key).lower()
            if leaf in FORBIDDEN_DESCRIPTOR_KEYS or "truth.npz" in leaf:
                raise ContractError(f"forbidden driver-input field: {path}.{key}")
            validate_no_forbidden_descriptor_keys(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            validate_no_forbidden_descriptor_keys(item, f"{path}[{index}]")


def _absolute_real_path(value: Any, label: str, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} path must be nonempty text")
    candidate = Path(value)
    predicate = candidate.is_dir if directory else candidate.is_file
    if not candidate.is_absolute() or candidate.is_symlink() or not predicate():
        kind = "directory" if directory else "file"
        raise ContractError(f"{label} must be an absolute real {kind}")
    return candidate.resolve(strict=True)


def validate_path_receipt(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} receipt must be an object")
    require_exact_keys(value, PATH_RECEIPT_KEYS, f"{label} receipt")
    path = _absolute_real_path(value["path"], label)
    digest = require_sha256(value["sha256"], f"{label} SHA-256")
    if sha256_file(path) != digest:
        raise ContractError(f"{label} SHA-256 mismatch")
    return {"path": path, "sha256": digest}


def validate_driver_inputs_document(
    value: Mapping[str, Any],
    *,
    registration_id: str,
    repository_root: Path,
    repository_commit: str,
    stageb_driver_sha256: str,
    cells: Sequence[str],
    methods: Sequence[str],
    populations: Mapping[str, str],
    dataset_hashes: Mapping[str, str],
    cpu_profile: str,
    rng_contract: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping):
        raise ContractError("driver inputs must be a JSON object")
    validate_no_forbidden_descriptor_keys(value)
    require_exact_keys(
        value,
        {
            "schema_version",
            "registration_id",
            "repository_root",
            "repository_commit",
            "stageb_driver_sha256",
            "cpu_profile",
            "rng_contract",
            "fit_probes",
            "public_inputs",
            "evaluator_only_inputs",
            "gate_only_inputs",
            "arm_t_exact_state_allowlist",
            "arm_t_refit_permitted",
            "original_truth_archive_available_to_policy",
            "runtime_next_states_available",
        },
        "driver inputs",
    )
    if value["schema_version"] != DRIVER_INPUTS_SCHEMA_VERSION:
        raise ContractError("driver-input schema mismatch")
    if value["registration_id"] != registration_id:
        raise ContractError("driver inputs are not bound to the registered identity")
    repository = _absolute_real_path(value["repository_root"], "driver repository", directory=True)
    if repository != Path(repository_root).resolve(strict=True):
        raise ContractError("driver repository root mismatch")
    if value["repository_commit"] != require_git_sha(repository_commit, "repository commit"):
        raise ContractError("driver-input commit mismatch")
    if value["stageb_driver_sha256"] != require_sha256(
        stageb_driver_sha256, "Stage B driver SHA-256"
    ):
        raise ContractError("driver-input driver hash mismatch")
    if value["cpu_profile"] != cpu_profile:
        raise ContractError("driver-input CPU profile mismatch")
    registered_rng = validate_rng_contract_document(rng_contract)
    driver_rng = validate_rng_contract_document(value["rng_contract"])
    if driver_rng != registered_rng:
        raise ContractError("driver-input RNG contract differs from registration")
    if value["arm_t_exact_state_allowlist"] != ["current_abundance"]:
        raise ContractError("Arm T exact-state allowlist mismatch")
    if value["arm_t_refit_permitted"] is not False:
        raise ContractError("Arm T refitting must remain impossible")
    if value["original_truth_archive_available_to_policy"] is not False:
        raise ContractError("the original truth archive must not be policy-accessible")
    if value["runtime_next_states_available"] is not False:
        raise ContractError("runtime next_states must remain unavailable")

    probes = value["fit_probes"]
    if not isinstance(probes, list) or len(probes) != 12:
        raise ContractError("driver inputs require exactly twelve fit probes")
    expected_pairs = [(cell, method) for cell in cells for method in methods]
    for index, (probe, pair) in enumerate(zip(probes, expected_pairs)):
        if not isinstance(probe, Mapping):
            raise ContractError("fit-probe binding must be an object")
        require_exact_keys(
            probe,
            {
                "task_index",
                "cell",
                "method",
                "publication_dir",
                "publication_success_sha256",
                "fit_probe_receipt_sha256",
                "frozen_object_sha256",
                "frozen_replay_sha256",
                "component_hashes",
            },
            f"fit_probes[{index}]",
        )
        if probe["task_index"] != index or (probe["cell"], probe["method"]) != pair:
            raise ContractError("fit-probe task order/index/cell/method mismatch")
        publication = _absolute_real_path(
            probe["publication_dir"], f"fit-probe {index} publication", directory=True
        )
        sources = {
            "publication_success_sha256": publication / "PUBLICATION_SUCCESS.json",
            "fit_probe_receipt_sha256": publication / "FIT_PROBE_RECEIPT.json",
            "frozen_object_sha256": publication / "FROZEN_FITTED_OBJECT.json",
            "frozen_replay_sha256": publication / "FRESH_RELOAD_REPLAY.json",
        }
        for field, path in sources.items():
            digest = require_sha256(probe[field], f"fit-probe {index} {field}")
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise ContractError(f"fit-probe {index} source binding mismatch: {field}")
        component_hashes = probe["component_hashes"]
        if not isinstance(component_hashes, Mapping) or not component_hashes:
            raise ContractError("fit-probe component hashes must be a non-empty object")
        for name, digest in component_hashes.items():
            if not isinstance(name, str) or not name:
                raise ContractError("fit-probe component name must be nonempty text")
            require_sha256(digest, f"fit-probe component {name}")
        receipt = strict_json_loads(sources["fit_probe_receipt_sha256"].read_bytes())
        if not isinstance(receipt, Mapping) or receipt.get("component_hashes") != dict(
            component_hashes
        ):
            raise ContractError("fit-probe component hashes differ from the sealed receipt")
        if receipt.get("fresh_reload_replay_sha256") != probe["frozen_replay_sha256"]:
            raise ContractError("fit-probe replay hash differs from the sealed receipt")
        if receipt.get("frozen_fitted_object_sha256") != probe["frozen_object_sha256"]:
            raise ContractError("fit-probe frozen-object hash differs from the sealed receipt")

    public_inputs = value["public_inputs"]
    if not isinstance(public_inputs, Mapping) or list(public_inputs) != list(cells):
        raise ContractError("driver public inputs must cover exactly the ordered registered cells")
    for cell in cells:
        item = public_inputs[cell]
        if not isinstance(item, Mapping):
            raise ContractError("cell public-input binding must be an object")
        require_exact_keys(
            item,
            {"population", "public_npz", "logical_dataset_sha256"},
            f"public inputs {cell}",
        )
        if item["population"] != populations[cell]:
            raise ContractError("cell population binding mismatch")
        if item["logical_dataset_sha256"] != dataset_hashes[cell]:
            raise ContractError("cell logical dataset binding mismatch")
        validate_path_receipt(item["public_npz"], f"{cell} public NPZ")

    evaluator = value["evaluator_only_inputs"]
    if not isinstance(evaluator, Mapping):
        raise ContractError("evaluator-only inputs must be an object")
    require_exact_keys(evaluator, {"accepted_parity"}, "evaluator-only inputs")
    parity = evaluator["accepted_parity"]
    if not isinstance(parity, list) or len(parity) != 12:
        raise ContractError("accepted parity must contain twelve per-task bindings")
    for index, (item, pair) in enumerate(zip(parity, expected_pairs)):
        if not isinstance(item, Mapping):
            raise ContractError("accepted-parity binding must be an object")
        require_exact_keys(
            item,
            {
                "task_index",
                "cell",
                "method",
                "episodes_csv",
                "baseline",
                "historical_canary_identity",
            },
            "parity",
        )
        if item["task_index"] != index or (item["cell"], item["method"]) != pair:
            raise ContractError("accepted-parity task binding mismatch")
        source = validate_path_receipt(item["episodes_csv"], f"accepted parity {index}")
        baseline = validate_parity_baseline_binding(item["baseline"])
        if item["historical_canary_identity"] != HISTORICAL_CANARY_IDENTITY:
            raise ContractError("historical canary disclosure identity mismatch")
        legacy_path = is_legacy_fast_track_canary_path(source["path"])
        if legacy_path != (baseline["classification"] == "LEGACY_INELIGIBLE"):
            raise ContractError("legacy canary path/classification binding mismatch")

    gate = value["gate_only_inputs"]
    if not isinstance(gate, Mapping):
        raise ContractError("gate-only inputs must be an object")
    require_exact_keys(
        gate, {"inherited_test_receipt", "corrected_test_receipt"}, "gate-only inputs"
    )
    validate_path_receipt(gate["inherited_test_receipt"], "inherited test receipt")
    validate_path_receipt(gate["corrected_test_receipt"], "corrected test receipt")


def load_canonical_driver_inputs(path: Path) -> Mapping[str, Any]:
    candidate = _absolute_real_path(str(Path(path).resolve()), "driver inputs")
    payload = candidate.read_bytes()
    value = strict_json_loads(payload)
    if not isinstance(value, Mapping):
        raise ContractError("driver inputs must be a JSON object")
    if canonical_driver_inputs_bytes(value) != payload:
        raise ContractError("driver inputs must use canonical JSON bytes")
    return value


def validate_output_root_entries(output_root: Path, *, pristine: bool) -> None:
    root = Path(output_root)
    if root.is_symlink() or not root.is_dir():
        raise ContractError("driver output root must be an existing real directory")
    allowed = {DRIVER_INPUTS_FILENAME}
    if not pristine:
        allowed.update(ROLE_NAMESPACES)
    entries = {path.name for path in root.iterdir()}
    if entries - allowed:
        raise ContractError(f"unregistered output-root entries: {sorted(entries - allowed)}")
    if pristine and entries != {DRIVER_INPUTS_FILENAME}:
        raise ContractError("pristine output root must contain exactly DRIVER_INPUTS.json")
    for name in entries & set(ROLE_NAMESPACES):
        path = root / name
        if path.is_symlink() or not path.is_dir():
            raise ContractError(f"role namespace is not a real directory: {name}")


def stage_driver_inputs(output_root: Path, value: Mapping[str, Any]) -> Path:
    root = Path(output_root)
    if root.exists():
        if root.is_symlink() or not root.is_dir() or any(root.iterdir()):
            raise ContractError("driver output root collides with existing content")
        os.chmod(root, 0o700)
    else:
        root.mkdir(mode=0o700, parents=False)
    path = root / DRIVER_INPUTS_FILENAME
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = canonical_driver_inputs_bytes(value)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ContractError("DRIVER_INPUTS.json already exists") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(descriptor)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    validate_output_root_entries(root, pristine=True)
    return path


def produce_registered_driver_inputs(*, registration_path: Path, output_root: Path) -> Path:
    """Freeze a canonical registration and stage its exact prospectively bound descriptor."""

    from .registration import freeze_registration_bundle
    from .submission import validate_durable_log_plan

    repository_root = Path(__file__).resolve().parents[3]
    registration_payload = Path(registration_path).read_bytes()
    bundle = strict_json_loads(registration_payload)
    if not isinstance(bundle, Mapping) or canonical_json_bytes(bundle) != registration_payload:
        raise ContractError("registration bundle must be a canonical JSON object")
    frozen = freeze_registration_bundle(bundle, repository_root=repository_root)
    if frozen.payload != registration_payload:
        raise ContractError("registration bundle changed during freeze")
    log_plan = validate_durable_log_plan(bundle["scientific_log_plan"])
    if str(Path(output_root)) != log_plan["scientific_output_root"]:
        raise ContractError("DRIVER_INPUTS output root differs from the registered log plan")
    return stage_driver_inputs(output_root, bundle["stageb_driver_inputs"]["descriptor"])


def validate_registered_driver_inputs(*, registration_path: Path, output_root: Path) -> str:
    """Validate the pristine staged descriptor and return its registered canonical digest."""

    from .registration import freeze_registration_bundle
    from .submission import validate_durable_log_plan

    repository_root = Path(__file__).resolve().parents[3]
    registration_payload = Path(registration_path).read_bytes()
    bundle = strict_json_loads(registration_payload)
    if not isinstance(bundle, Mapping) or canonical_json_bytes(bundle) != registration_payload:
        raise ContractError("registration bundle must be a canonical JSON object")
    freeze_registration_bundle(bundle, repository_root=repository_root)
    log_plan = validate_durable_log_plan(bundle["scientific_log_plan"])
    if str(Path(output_root)) != log_plan["scientific_output_root"]:
        raise ContractError("DRIVER_INPUTS output root differs from the registered log plan")
    validate_output_root_entries(output_root, pristine=True)
    observed = load_canonical_driver_inputs(Path(output_root) / DRIVER_INPUTS_FILENAME)
    if dict(observed) != bundle["stageb_driver_inputs"]["descriptor"]:
        raise ContractError("staged DRIVER_INPUTS differs from the frozen registration")
    digest = driver_inputs_sha256(observed)
    if digest != bundle["stageb_driver_inputs"]["driver_inputs_sha256"]:
        raise ContractError("staged DRIVER_INPUTS hash differs from the frozen registration")
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("produce", "validate"):
        item = commands.add_parser(command)
        item.add_argument("--registration", type=Path, required=True)
        item.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "produce":
        path = produce_registered_driver_inputs(
            registration_path=arguments.registration, output_root=arguments.output_root
        )
        print(path)
    else:
        print(
            validate_registered_driver_inputs(
                registration_path=arguments.registration, output_root=arguments.output_root
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"CONTRACT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
