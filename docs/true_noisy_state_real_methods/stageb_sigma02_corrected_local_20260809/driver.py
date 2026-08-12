#!/usr/bin/env python3
"""Registration-bound corrected Stage B scientific task driver.

The four CLI roles in this module are the only supported corrected Stage B entry points.
Fitted objects are loaded from the sealed V4 fit-only publications; neither arm has a fit
path.  The evaluator is the frozen track evaluator, while this external driver owns the
information boundary, exact-state overlay, evaluator-only evidence, gate, and publication.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:  # direct execution from the tracked Slurm templates
    repository_bootstrap = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository_bootstrap))
    __package__ = "docs.true_noisy_state_real_methods.stageb_sigma02_corrected_local_20260809"

from docs.true_noisy_state_real_methods.i2_increment_a_exact_state_adapters_20260808.adapter_interfaces import (  # noqa: E501
    FEATURE_NAMES,
    adapt_point_mass_belief,
)
from docs.true_noisy_state_real_methods.i2b_fasttrack_integration_canary_20260808.fasttrack_wrappers import (  # noqa: E501
    exact_general_belief,
    issue_exact_state_capability,
)

from .artifacts import (
    ECOLOGICAL_METHODS,
    FIXTURE_ROW_COUNT,
    REQUIRED_COMPONENTS,
    CanonicalArtifact,
    deterministic_prediction_fixtures,
    validate_complete_artifact_bundle,
)
from .canonical_plan import (
    BOUND_TASK_EVIDENCE_SCHEMA_VERSION,
    DRIVER_INPUTS_FILENAME,
    EVALUATION_IDENTITY_SHA256,
    RNG_RECEIPT_SCHEMA_VERSION,
    STEP_EVIDENCE_SCHEMA_VERSION,
    artifact_plan_sha256,
    canonical_noise_sigma,
    validate_rng_contract_document,
)
from .boundary import build_task_information_boundary_receipt
from .common import (
    REGISTERED_CPU_MODEL,
    ContractError,
    canonical_json_bytes,
    require_current_registered_cpu_model,
    require_exact_keys,
    require_sha256,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
    validate_cpu_identity_receipt,
)
from .diagnostics import (
    build_arm_refplan_predictive_dispersion,
    build_arm_transition_diagnostics,
    build_realized_posterior_receipt,
    classify_activity,
)
from .driver_inputs import (
    driver_inputs_sha256,
    load_canonical_driver_inputs,
    validate_driver_inputs_document,
    validate_no_forbidden_descriptor_keys as _shared_validate_no_forbidden_descriptor_keys,
    validate_output_root_entries,
    validate_path_receipt,
)
from .evidence import (
    REGISTERED_EPISODE_IDS,
    REGISTERED_GAMMA,
    REGISTERED_HORIZON,
    RNGReceipt,
    StepEvidence,
    pair_episode_evidence,
    reconstruct_episode,
)
from .orchestration import (
    authorize_arm_t_task,
    export_arm_o_gate_receipt,
    inspect_only_finalizer,
    load_gate_signing_key,
    load_gate_verification_key,
    validate_arm_o_gate,
    verify_arm_o_gate_receipt,
)
from .parity import (
    HISTORICAL_CANARY_IDENTITY,
    PARITY_FAILURE_FILENAME,
    PARITY_FAILURE_NAMESPACE,
    PARITY_MODE_DISCLOSURE_ONLY,
    PARITY_MODE_EXTERNAL_BASELINE,
    PARITY_TOLERANCE,
    ParityEligibilityError,
    canonical_reference_rows_sha256,
    compare_accepted_parity,
    is_legacy_fast_track_canary_path,
    reference_file_sha256,
    validate_parity_baseline_binding,
    validate_parity_baseline_eligibility,
    validate_parity_failure_diagnostic,
)
from .publication import (
    SUCCESS_RECEIPT,
    create_task_staging,
    load_success_receipt,
    publish_failure_once,
    publish_once,
    validate_staging_tree,
    write_bytes_fsync,
)
from .real_artifacts import (
    _registered_track_imports,
    load_frozen_fitted_object,
    revalidate_frozen_object_parity,
    scientific_component_for_method,
)
from .registration import (
    CELLS,
    CONFIG_PATHS,
    EXPECTED_DATASET_HASHES,
    METHODS,
    POPULATIONS,
    FrozenRegistration,
    freeze_registration_bundle,
    require_runtime_interpreter_binding,
    validate_interpreter_identity_receipt,
)
from .submission import validate_durable_log_plan


DRIVER_INPUTS = DRIVER_INPUTS_FILENAME
ARM_O_RECEIPT = "ARM_O_TASK_RECEIPT.json"
ARM_T_RECEIPT = "ARM_T_TASK_RECEIPT.json"
GATE_RECEIPT = "ARM_O_GATE_RECEIPT.json"
CPU_MODEL = REGISTERED_CPU_MODEL
CPU_PROFILE = f"{CPU_MODEL} / xenon-8452Y / one CPU"
FILTERS = {
    "plus_adapted_ricker_only_pbvi": "faithful_internal",
    "moor_adapted_ricker_misspec_pbvi": "faithful_internal",
    "refplan": "learned",
    "ogsrl": "learned",
    "bamcts": "learned",
    "ensemble_value_disagreement_pessimism": "learned",
}
TRACKS = {method: "ecological" if method in ECOLOGICAL_METHODS else "general" for method in METHODS}


@dataclass(frozen=True)
class SealedFitProbe:
    task_index: int
    publication_dir: Path
    publication_success_sha256: str
    fit_probe_receipt_sha256: str
    frozen_object_sha256: str
    frozen_replay_sha256: str
    component_hashes: Mapping[str, str]

    def artifact_plan_sha256(self, *, cell: str, method: str, dataset_sha256: str) -> str:
        return artifact_plan_sha256(
            task_index=self.task_index,
            cell=cell,
            method=method,
            dataset_sha256=dataset_sha256,
            publication_success_sha256=self.publication_success_sha256,
            fit_probe_receipt_sha256=self.fit_probe_receipt_sha256,
            frozen_object_sha256=self.frozen_object_sha256,
            frozen_replay_sha256=self.frozen_replay_sha256,
            component_hashes=self.component_hashes,
        )


@dataclass(frozen=True)
class DriverInputs:
    path: Path
    repository_root: Path
    registration_id: str
    fit_probes: tuple[SealedFitProbe, ...]
    public_inputs: Mapping[str, Mapping[str, Any]]
    rng_contract: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluatorOnlyInputs:
    accepted_parity: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class GateOnlyInputs:
    inherited_test_receipt: Mapping[str, Any]
    corrected_test_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedDriverInputs:
    descriptor_path: Path
    descriptor_sha256: str
    policy: DriverInputs
    evaluator_only: EvaluatorOnlyInputs
    gate_only: GateOnlyInputs


@dataclass(frozen=True)
class TaskExecution:
    step_evidence: tuple[tuple[StepEvidence, ...], ...]
    evaluator_rows: tuple[Mapping[str, Any], ...]
    actions: tuple[tuple[int, ...], ...]
    posterior_probabilities: tuple[tuple[tuple[float, ...], ...], ...]
    predictive_dispersion: tuple[tuple[float, ...], ...]
    context_sha256_before: tuple[str, ...]
    context_sha256_after: tuple[str, ...]
    observation_history_sha256: tuple[str, ...]
    action_history_sha256: tuple[str, ...]
    cpu_identity: Mapping[str, str]


def _json_object(payload: bytes, label: str, *, canonical: bool = True) -> Mapping[str, Any]:
    value = strict_json_loads(payload)
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    if canonical and canonical_json_bytes(value) != payload:
        raise ContractError(f"{label} must use canonical JSON bytes")
    return value


def _safe_absolute_file(path: Any, label: str) -> Path:
    if not isinstance(path, str) or not path:
        raise ContractError(f"{label} path must be nonempty text")
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
        raise ContractError(f"{label} must be an absolute real file")
    return candidate.resolve(strict=True)


def _safe_absolute_directory(path: Any, label: str) -> Path:
    if not isinstance(path, str) or not path:
        raise ContractError(f"{label} path must be nonempty text")
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_dir():
        raise ContractError(f"{label} must be an absolute real directory")
    return candidate.resolve(strict=True)


def _load_path_receipt(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} receipt must be an object")
    require_exact_keys(value, {"path", "sha256"}, f"{label} receipt")
    path = _safe_absolute_file(value["path"], label)
    digest = require_sha256(value["sha256"], f"{label} SHA-256")
    if sha256_file(path) != digest:
        raise ContractError(f"{label} SHA-256 mismatch")
    return {"path": path, "sha256": digest}


def _require_scientific_environment() -> Mapping[str, str]:
    cpu_identity = require_current_registered_cpu_model()
    expected = {
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    mismatches = {
        key: os.environ.get(key) for key, value in expected.items() if os.environ.get(key) != value
    }
    if mismatches:
        raise ContractError(f"registered scientific environment mismatch: {mismatches}")
    return cpu_identity


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_registration(path: Path) -> FrozenRegistration:
    registration_path = _safe_absolute_file(str(Path(path).resolve()), "registration bundle")
    payload = registration_path.read_bytes()
    bundle = _json_object(payload, "registration bundle")
    frozen = freeze_registration_bundle(bundle, repository_root=_repository_root())
    if frozen.payload != payload:
        raise ContractError("registration file differs from its canonical frozen bytes")
    return frozen


def _validate_no_forbidden_descriptor_keys(value: Any, path: str = "root") -> None:
    _shared_validate_no_forbidden_descriptor_keys(value, path)


def _parse_fit_probe(value: Any, expected_index: int) -> SealedFitProbe:
    if not isinstance(value, Mapping):
        raise ContractError("fit-probe binding must be an object")
    require_exact_keys(
        value,
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
        "fit-probe binding",
    )
    if value["task_index"] != expected_index:
        raise ContractError("fit-probe task order/index mismatch")
    component_hashes = value["component_hashes"]
    if not isinstance(component_hashes, Mapping):
        raise ContractError("fit-probe component hashes must be an object")
    parsed_hashes = {
        str(name): require_sha256(digest, f"fit-probe component {name}")
        for name, digest in component_hashes.items()
    }
    return SealedFitProbe(
        task_index=expected_index,
        publication_dir=_safe_absolute_directory(value["publication_dir"], "fit-probe publication"),
        publication_success_sha256=require_sha256(
            value["publication_success_sha256"], "fit-probe publication"
        ),
        fit_probe_receipt_sha256=require_sha256(
            value["fit_probe_receipt_sha256"], "fit-probe receipt"
        ),
        frozen_object_sha256=require_sha256(value["frozen_object_sha256"], "frozen object"),
        frozen_replay_sha256=require_sha256(value["frozen_replay_sha256"], "frozen replay"),
        component_hashes=parsed_hashes,
    )


def load_driver_inputs(output_root: Path, registration: FrozenRegistration) -> LoadedDriverInputs:
    root = Path(output_root)
    validate_output_root_entries(root, pristine=False)
    log_plan = validate_durable_log_plan(
        registration.bundle()["scientific_log_plan"],
        require_existing=False,
    )
    if root.resolve(strict=True) != Path(log_plan["scientific_output_root"]):
        raise ContractError("runtime output root differs from the registered durable-log plan")
    path = root / DRIVER_INPUTS
    value = load_canonical_driver_inputs(path)
    bundle = registration.bundle()
    registered = bundle["stageb_driver_inputs"]
    if dict(value) != registered["descriptor"]:
        raise ContractError("DRIVER_INPUTS bytes differ from the registered descriptor")
    observed_digest = driver_inputs_sha256(value)
    if observed_digest != registered["driver_inputs_sha256"]:
        raise ContractError("canonical DRIVER_INPUTS hash mismatch")
    hashes = bundle["code_configuration_hashes"]
    validate_driver_inputs_document(
        value,
        registration_id=registration.registration_id,
        repository_root=_repository_root(),
        repository_commit=hashes["git_commit_sha"],
        stageb_driver_sha256=hashes["stageb_driver_sha256"],
        cells=CELLS,
        methods=METHODS,
        populations=POPULATIONS,
        dataset_hashes=EXPECTED_DATASET_HASHES,
        cpu_profile=CPU_PROFILE,
        rng_contract=bundle["corrected_stageb_registration"]["rng_contract"],
    )
    if sha256_file(Path(__file__)) != value["stageb_driver_sha256"]:
        raise ContractError("executing driver bytes differ from the frozen registration")
    probes = tuple(_parse_fit_probe(item, index) for index, item in enumerate(value["fit_probes"]))
    public_inputs = {
        cell: {
            "population": value["public_inputs"][cell]["population"],
            "logical_dataset_sha256": value["public_inputs"][cell]["logical_dataset_sha256"],
            "public_npz": validate_path_receipt(
                value["public_inputs"][cell]["public_npz"], f"{cell} public NPZ"
            ),
        }
        for cell in CELLS
    }
    evaluator = tuple(
        {
            "task_index": item["task_index"],
            "cell": item["cell"],
            "method": item["method"],
            "historical_canary_identity": item["historical_canary_identity"],
            "episodes_csv": validate_path_receipt(
                item["episodes_csv"], f"accepted parity {item['task_index']}"
            ),
            "baseline": validate_parity_baseline_binding(item["baseline"]),
        }
        for item in value["evaluator_only_inputs"]["accepted_parity"]
    )
    gate = value["gate_only_inputs"]
    return LoadedDriverInputs(
        descriptor_path=path.resolve(strict=True),
        descriptor_sha256=observed_digest,
        policy=DriverInputs(
            path.resolve(strict=True),
            _repository_root(),
            registration.registration_id,
            probes,
            public_inputs,
            validate_rng_contract_document(value["rng_contract"]),
        ),
        evaluator_only=EvaluatorOnlyInputs(evaluator),
        gate_only=GateOnlyInputs(
            validate_path_receipt(gate["inherited_test_receipt"], "inherited test receipt"),
            validate_path_receipt(gate["corrected_test_receipt"], "corrected test receipt"),
        ),
    )


def _task_for_index(
    registration: FrozenRegistration, arm: str, task_index: int
) -> Mapping[str, Any]:
    if arm not in {"O", "T"}:
        raise ContractError("driver arm must be O or T")
    if isinstance(task_index, bool) or not isinstance(task_index, int) or not 0 <= task_index < 12:
        raise ContractError("driver task index must be in 0..11")
    tasks = registration.bundle()["task_manifest"]["tasks"]
    manifest_index = task_index if arm == "O" else task_index + 12
    task = tasks[manifest_index]
    cell_index, method_index = divmod(task_index, len(METHODS))
    expected = {
        "task_index": manifest_index,
        "arm": arm,
        "cell": CELLS[cell_index],
        "method": METHODS[method_index],
    }
    for field, value in expected.items():
        if task[field] != value:
            raise ContractError(f"registered driver task mismatch: {field}")
    if task["evaluation_identity_sha256"] != EVALUATION_IDENTITY_SHA256:
        raise ContractError("registered evaluation-identity hash mismatch")
    if task["config_sha256"] != sha256_file(
        _repository_root()
        / CONFIG_PATHS[
            "plus_config_sha256"
            if task["method"].startswith("plus_")
            else "moor_config_sha256"
            if task["method"].startswith("moor_")
            else "general_config_sha256"
        ]
    ):
        raise ContractError("registered task configuration bytes changed")
    return task


def _validate_fit_probe(
    inputs: DriverInputs,
    task: Mapping[str, Any],
    task_index: int,
) -> tuple[Mapping[str, bytes], bytes, bytes, Mapping[str, Any]]:
    probe = inputs.fit_probes[task_index]
    expected_plan = probe.artifact_plan_sha256(
        cell=task["cell"], method=task["method"], dataset_sha256=task["dataset_sha256"]
    )
    if task["artifact_plan_sha256"] != expected_plan:
        raise ContractError("task artifact plan is not bound to the sealed V4 fit probe")
    success_path = probe.publication_dir / SUCCESS_RECEIPT
    if sha256_file(success_path) != probe.publication_success_sha256:
        raise ContractError("fit-probe publication receipt identity mismatch")
    load_success_receipt(success_path)
    receipt_path = probe.publication_dir / "FIT_PROBE_RECEIPT.json"
    replay_path = probe.publication_dir / "FRESH_RELOAD_REPLAY.json"
    frozen_path = probe.publication_dir / "FROZEN_FITTED_OBJECT.json"
    if sha256_file(receipt_path) != probe.fit_probe_receipt_sha256:
        raise ContractError("fit-probe receipt identity mismatch")
    if sha256_file(replay_path) != probe.frozen_replay_sha256:
        raise ContractError("fit-probe replay identity mismatch")
    if sha256_file(frozen_path) != probe.frozen_object_sha256:
        raise ContractError("fit-probe fitted-object identity mismatch")
    receipt = _json_object(receipt_path.read_bytes(), "fit-probe receipt")
    if (
        receipt.get("schema_version") != "corrected_stageb_fit_probe_receipt_v4"
        or receipt.get("result") != "PASS"
        or receipt.get("task", {}).get("task_index") != task_index
        or receipt.get("task", {}).get("cell") != task["cell"]
        or receipt.get("task", {}).get("method") != task["method"]
        or receipt.get("public_view_sha256") != task["dataset_sha256"]
        or receipt.get("component_hashes") != dict(probe.component_hashes)
        or receipt.get("frozen_fitted_object_sha256") != probe.frozen_object_sha256
        or receipt.get("fresh_reload_replay_sha256") != probe.frozen_replay_sha256
        or receipt.get("evaluator_constructed") is not False
        or receipt.get("truth_accessed") is not False
        or receipt.get("runtime_next_states_accessed") is not False
        or receipt.get("returns_calculated") is not False
    ):
        raise ContractError("fit-probe receipt binding/information boundary mismatch")
    component_payloads: dict[str, bytes] = {}
    for component in REQUIRED_COMPONENTS[task["method"]]:
        path = probe.publication_dir / f"artifact-{component}.json"
        payload = path.read_bytes()
        if sha256_bytes(payload) != probe.component_hashes.get(component):
            raise ContractError(f"fit-probe component identity mismatch: {component}")
        component_payloads[component] = payload
    if set(component_payloads) != set(probe.component_hashes):
        raise ContractError("fit-probe component applicability mismatch")
    fixtures = deterministic_prediction_fixtures(task["method"], component_payloads)
    validated = validate_complete_artifact_bundle(
        task["method"],
        component_payloads,
        prediction_fixtures=fixtures,
        expected_component_hashes=probe.component_hashes,
    )
    if (
        validated.bundle_sha256 != receipt["bundle_sha256"]
        or validated.fixture_manifest_sha256 != receipt["fixture_manifest_sha256"]
    ):
        raise ContractError("fit-probe canonical bundle cannot be reproduced")
    frozen_payload = frozen_path.read_bytes()
    replay_payload = replay_path.read_bytes()
    replay = _json_object(replay_payload, "fit-probe frozen-object replay")
    revalidate_frozen_object_parity(
        frozen_payload,
        replay,
        expected_component=scientific_component_for_method(task["method"]),
        repository_root=inputs.repository_root,
    )
    return component_payloads, frozen_payload, replay_payload, receipt


def _copy_validated_artifacts(
    staging: Path,
    *,
    registration: FrozenRegistration,
    task: Mapping[str, Any],
    component_payloads: Mapping[str, bytes],
    frozen_payload: bytes,
    replay_payload: bytes,
) -> tuple[dict[str, Any], list[str]]:
    component_hashes = {name: sha256_bytes(payload) for name, payload in component_payloads.items()}
    components: dict[str, Any] = {}
    required: list[str] = []
    for component, payload in sorted(component_payloads.items()):
        filename = f"artifact-{component}.json"
        write_bytes_fsync(staging / filename, payload)
        required.append(filename)
        components[component] = {"relative_path": "", "sha256": component_hashes[component]}
    write_bytes_fsync(staging / "FROZEN_FITTED_OBJECT.json", frozen_payload)
    write_bytes_fsync(staging / "FRESH_RELOAD_REPLAY.json", replay_payload)
    required.extend(["FROZEN_FITTED_OBJECT.json", "FRESH_RELOAD_REPLAY.json"])
    fixtures = deterministic_prediction_fixtures(task["method"], component_payloads)
    validated = validate_complete_artifact_bundle(
        task["method"],
        component_payloads,
        prediction_fixtures=fixtures,
        expected_component_hashes=component_hashes,
    )
    artifact = {
        "schema_version": "corrected_stageb_artifact_bundle_evidence_v4",
        "registration_sha256": registration.sha256,
        "task_index": task["task_index"],
        "arm": task["arm"],
        "cell": task["cell"],
        "method": task["method"],
        "artifact_plan_sha256": task["artifact_plan_sha256"],
        "prediction_fixtures": {name: value.tolist() for name, value in fixtures.items()},
        "fixture_row_count": FIXTURE_ROW_COUNT,
        "fixture_manifest_sha256": validated.fixture_manifest_sha256,
        "components": components,
        "component_hashes": component_hashes,
        "bundle_sha256": validated.bundle_sha256,
        "fresh_reload_parity": True,
        "frozen_object_artifact": {
            "relative_path": "",
            "sha256": sha256_bytes(frozen_payload),
        },
        "frozen_object_parity": {
            "relative_path": "",
            "sha256": sha256_bytes(replay_payload),
        },
        "frozen_object_binding": {
            "schema_version": "corrected_stageb_frozen_object_task_binding_v1",
            "registration_sha256": registration.sha256,
            "task_index": task["task_index"],
            "arm": task["arm"],
            "cell": task["cell"],
            "method": task["method"],
            "public_view_sha256": task["dataset_sha256"],
            "frozen_object_component": scientific_component_for_method(task["method"]),
            "frozen_object_sha256": sha256_bytes(frozen_payload),
            "parity_receipt_sha256": sha256_bytes(replay_payload),
            "covered_components": sorted(REQUIRED_COMPONENTS[task["method"]]),
            "logical_component_hashes": component_hashes,
            "result": "PASS",
        },
    }
    return artifact, required


def _hash_rng_state(rng: np.random.Generator) -> str:
    def normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [normalize(item) for item in value]
        return value

    return sha256_bytes(canonical_json_bytes(normalize(rng.bit_generator.state)))


class _ExactStateBridge:
    __slots__ = ("_current",)

    def __init__(self) -> None:
        self._current: float | None = None

    def set_current(self, value: Any) -> None:
        current = float(value)
        if not np.isfinite(current) or current < 0.0:
            raise ContractError("evaluator supplied an invalid current abundance")
        self._current = current

    def current(self) -> float:
        if self._current is None:
            raise ContractError("current exact abundance is unavailable")
        return self._current


class _ExactCurrentFilter:
    __slots__ = ("_base", "_bridge", "_capability", "_sigma", "_scale")

    def __init__(self, base: Any, bridge: _ExactStateBridge, sigma: float, scale: float) -> None:
        self._base = base
        self._bridge = bridge
        self._capability = issue_exact_state_capability(
            purpose="registered_corrected_stageb_runtime_current_state"
        )
        self._sigma = sigma
        self._scale = scale

    def _adapt(self, belief: Any) -> Any:
        emitted, _receipt = exact_general_belief(
            belief,
            self._bridge.current(),
            capability=self._capability,
            observation_noise_sigma=self._sigma,
            observation_scale=self._scale,
            private_payload=None,
            oracle_filter=False,
        )
        return emitted

    def reset(self, observation: float, seed: int) -> Any:
        return self._adapt(self._base.reset(observation, seed))

    def update(self, belief: Any, action: int, observation: float) -> Any:
        return self._adapt(self._base.update(belief, action, observation))


class _RecordingPolicy:
    """Policy-facing wrapper that sees only the frozen public transition type."""

    def __init__(self, policy: Any, method: str, arm: str, bridge: _ExactStateBridge) -> None:
        self._policy = policy
        self._method = method
        self._arm = arm
        self._bridge = bridge
        self._episode_actions: list[list[int]] = []
        self._episode_posteriors: list[list[list[float]]] = []
        self._episode_dispersion: list[list[float]] = []
        self._current_actions: list[int] | None = None
        self._current_posteriors: list[list[float]] | None = None
        self._current_dispersion: list[float] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    @property
    def name(self) -> str:
        return self._policy.name

    def reset(self, seed: int) -> None:
        self._policy.reset(seed)
        self._current_actions = []
        self._current_dispersion = []
        self._episode_actions.append(self._current_actions)
        self._episode_dispersion.append(self._current_dispersion)
        initial = np.asarray(getattr(self._policy, "posterior", ()), dtype=np.float64)
        self._current_posteriors = [initial.tolist()] if self._method == "bamcts" else []
        self._episode_posteriors.append(self._current_posteriors)

    def _assign_ecological_point_mass(self, observation: float) -> None:
        if self._arm != "T" or self._method not in ECOLOGICAL_METHODS:
            return
        raw = self._bridge.current()
        if self._method.startswith("plus_"):
            if self._policy.internal_beliefs is None:
                self._policy._initialize(observation)
            updated = []
            for pomdp, previous in zip(self._policy.pomdps, self._policy.internal_beliefs):
                latent = np.asarray(pomdp.abundance_grid, dtype=np.float64)
                probabilities, _receipt = adapt_point_mass_belief(
                    self._method,
                    raw,
                    float(pomdp.model.survey_scale),
                    latent,
                    latent * np.float64(pomdp.model.survey_scale),
                )
                updated.append(replace(previous, probabilities=probabilities))
            self._policy.internal_beliefs = updated
        else:
            if self._policy.internal_belief is None:
                self._policy.internal_belief = self._policy.pomdp.initial_belief(observation)
            pomdp = self._policy.pomdp
            latent = np.asarray(pomdp.abundance_grid, dtype=np.float64)
            probabilities, _receipt = adapt_point_mass_belief(
                self._method,
                raw,
                float(pomdp.model.survey_scale),
                latent,
                latent * np.float64(pomdp.model.survey_scale),
            )
            self._policy.internal_belief = replace(
                self._policy.internal_belief, probabilities=probabilities
            )

    def act(self, belief: Any, observation: float) -> int:
        if self._current_actions is None or self._current_dispersion is None:
            raise ContractError("policy action occurred before reset")
        self._assign_ecological_point_mass(observation)
        action = int(self._policy.act(belief, observation))
        self._current_actions.append(action)
        if self._method == "refplan":
            scores = np.asarray(self._policy.last_diagnostics.get("action_scores", ()), dtype=float)
            if scores.shape != (11,) or not np.isfinite(scores).all():
                raise ContractError("RefPlan predictive action-value dispersion is unavailable")
            self._current_dispersion.append(float(np.std(scores, dtype=np.float64)))
        return action

    def observe(self, belief: Any, action: int, result: Any) -> None:
        if self._arm == "T" and self._method in ECOLOGICAL_METHODS:
            if self._method.startswith("plus_"):
                updated = []
                for pomdp, previous in zip(self._policy.pomdps, self._policy.internal_beliefs):
                    updated.append(
                        replace(
                            previous,
                            capacity=pomdp.model.next_capacity(previous.capacity, int(action)),
                            previous_observation=previous.current_observation,
                            current_observation=float(result.observation),
                            timestep=previous.timestep + 1,
                        )
                    )
                self._policy.internal_beliefs = updated
            else:
                previous = self._policy.internal_belief
                self._policy.internal_belief = replace(
                    previous,
                    capacity=self._policy.pomdp.model.next_capacity(previous.capacity, int(action)),
                    previous_observation=previous.current_observation,
                    current_observation=float(result.observation),
                    timestep=previous.timestep + 1,
                )
        else:
            visible = result
            if self._arm == "T" and self._method in {"refplan", "bamcts"}:
                visible = replace(result, observation=self._bridge.current())
            self._policy.observe(belief, action, visible)
        if self._method == "bamcts":
            if self._current_posteriors is None:
                raise ContractError("BA-MCTS posterior recording occurred before reset")
            posterior = np.asarray(self._policy.posterior, dtype=np.float64)
            self._current_posteriors.append(posterior.tolist())


class _DrawCountingGenerator:
    """Count only actual evaluator-environment distribution invocations."""

    __slots__ = ("_generator", "_permitted_method", "_draw_invocations")

    def __init__(self, generator: Any, *, permitted_method: str) -> None:
        if permitted_method not in {"normal", "lognormal"}:
            raise ContractError("unregistered evaluator RNG distribution")
        if not hasattr(generator, "bit_generator"):
            raise ContractError("evaluator RNG lacks a bit generator")
        self._generator = generator
        self._permitted_method = permitted_method
        self._draw_invocations = 0

    @property
    def bit_generator(self) -> Any:
        return self._generator.bit_generator

    @property
    def draw_invocations(self) -> int:
        return self._draw_invocations

    def _draw(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if method != self._permitted_method:
            raise ContractError(f"unregistered evaluator RNG draw API: {method}")
        result = getattr(self._generator, method)(*args, **kwargs)
        self._draw_invocations += 1
        return result

    def normal(self, *args: Any, **kwargs: Any) -> Any:
        return self._draw("normal", *args, **kwargs)

    def lognormal(self, *args: Any, **kwargs: Any) -> Any:
        return self._draw("lognormal", *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        raise ContractError(f"unregistered evaluator RNG attribute: {name}")


class _EvaluatorRecorder:
    """Evaluator-only state; no reference to this object reaches a policy."""

    def __init__(
        self,
        registration: FrozenRegistration,
        task: Mapping[str, Any],
        bridge: _ExactStateBridge,
        rng_contract: Mapping[str, Any],
    ) -> None:
        self.registration = registration
        self.task = task
        self.bridge = bridge
        self.rng_contract = validate_rng_contract_document(rng_contract)
        self.episodes: list[list[StepEvidence]] = []
        self._current: list[StepEvidence] | None = None
        self._episode_id: int | None = None
        self._state: float | None = None
        self._observation: float | None = None
        self._collapse_latched = False

    def wrap(self, environment: Any) -> Any:
        recorder = self

        class RecordingEnvironment:
            def __getattr__(self, name: str) -> Any:
                return getattr(environment, name)

            def reset(self, seed: int) -> Any:
                result = environment.reset(seed)
                if seed not in REGISTERED_EPISODE_IDS:
                    raise ContractError("evaluator requested an unregistered episode identity")
                recorder._episode_id = int(seed)
                recorder._state = float(result.evaluator_info["state"])
                recorder._observation = float(result.observation)
                recorder._collapse_latched = False
                process_sigma = canonical_noise_sigma(
                    environment.cfg.process_noise_sigma, "effective process noise sigma"
                )
                observation_sigma = canonical_noise_sigma(
                    environment.cfg.observation_noise_sigma,
                    "effective observation noise sigma",
                )
                if process_sigma.hex() != recorder.rng_contract["process_noise_sigma"].hex():
                    raise ContractError("effective process noise differs from registration")
                if (
                    observation_sigma.hex()
                    != recorder.rng_contract["observation_noise_sigma"].hex()
                ):
                    raise ContractError("effective observation noise differs from registration")
                process_rng = _DrawCountingGenerator(
                    environment._rngs["process"], permitted_method="normal"
                )
                observation_rng = _DrawCountingGenerator(
                    environment._rngs["observation"], permitted_method="lognormal"
                )
                environment._rngs["process"] = process_rng
                environment._rngs["observation"] = observation_rng
                recorder._current = []
                recorder.episodes.append(recorder._current)
                recorder.bridge.set_current(recorder._state)
                return result

            def step(self, action: int) -> Any:
                if (
                    recorder._current is None
                    or recorder._episode_id is None
                    or recorder._state is None
                    or recorder._observation is None
                ):
                    raise ContractError("evaluator step occurred before registered reset")
                process_rng = environment._rngs["process"]
                observation_rng = environment._rngs["observation"]
                process_before = _hash_rng_state(process_rng)
                observation_before = _hash_rng_state(observation_rng)
                process_draws_before = process_rng.draw_invocations
                observation_draws_before = observation_rng.draw_invocations
                state_previous = recorder._state
                observation_previous = recorder._observation
                result = environment.step(action)
                process_after = _hash_rng_state(process_rng)
                observation_after = _hash_rng_state(observation_rng)
                info = result.evaluator_info
                state_next = float(info["state"])
                entered = bool(info["entered_safety_region"])
                recorder._collapse_latched = recorder._collapse_latched or entered
                safety = bool(info["safety_penalty_applied"])
                benefit = float(environment.reward_model.utility(state_next))
                action_cost = -float(environment.actions[int(action)].cost)
                safety_penalty = (
                    -float(environment.reward_model.collapse_penalty) if safety else 0.0
                )
                total = benefit + action_cost + safety_penalty
                if not np.isclose(total, float(info["reward_true"]), rtol=0.0, atol=1e-12):
                    raise ContractError(
                        "evaluator true reward does not match registered components"
                    )
                timestep = len(recorder._current)
                discount = float(np.float64(REGISTERED_GAMMA) ** timestep)
                rng_receipt = RNGReceipt(
                    schema_version=RNG_RECEIPT_SCHEMA_VERSION,
                    process_noise_sigma=recorder.rng_contract["process_noise_sigma"],
                    observation_noise_sigma=recorder.rng_contract["observation_noise_sigma"],
                    process_draw_required=bool(recorder.rng_contract["process_draws_per_step"]),
                    observation_draw_required=bool(
                        recorder.rng_contract["observation_draws_per_step"]
                    ),
                    process_state_advancement_applicable=bool(
                        recorder.rng_contract["process_draws_per_step"]
                    ),
                    observation_state_advancement_applicable=bool(
                        recorder.rng_contract["observation_draws_per_step"]
                    ),
                    process_draw_invocations_before=process_draws_before,
                    process_draw_invocations_after=process_rng.draw_invocations,
                    observation_draw_invocations_before=observation_draws_before,
                    observation_draw_invocations_after=observation_rng.draw_invocations,
                    process_state_before_sha256=process_before,
                    process_state_after_sha256=process_after,
                    observation_state_before_sha256=observation_before,
                    observation_state_after_sha256=observation_after,
                )
                innovation_process = sha256_bytes(
                    canonical_json_bytes({"before": process_before, "after": process_after})
                )
                innovation_observation = sha256_bytes(
                    canonical_json_bytes({"before": observation_before, "after": observation_after})
                )
                recorder._current.append(
                    StepEvidence(
                        schema_version=STEP_EVIDENCE_SCHEMA_VERSION,
                        registration_sha256=recorder.registration.sha256,
                        method=recorder.task["method"],
                        cell=recorder.task["cell"],
                        arm=recorder.task["arm"],
                        episode_id=recorder._episode_id,
                        timestep=timestep,
                        current_true_abundance=state_previous,
                        noisy_observation=observation_previous,
                        selected_action=int(action),
                        benefit_population_term=benefit,
                        action_cost_term=action_cost,
                        safety_penalty_term=safety_penalty,
                        total_reward=total,
                        discount_factor=discount,
                        discounted_benefit_population=discount * benefit,
                        discounted_action_cost=discount * action_cost,
                        discounted_safety_penalty=discount * safety_penalty,
                        discounted_total_reward=discount * total,
                        collapse_indicator=recorder._collapse_latched,
                        unsafe_indicator=bool(info["below_safety_region"]),
                        process_innovation_sha256=innovation_process,
                        observation_innovation_sha256=innovation_observation,
                        rng_receipt=rng_receipt,
                    )
                )
                recorder._state = state_next
                recorder._observation = float(result.observation)
                recorder.bridge.set_current(state_next)
                return result

        return RecordingEnvironment()


def _complete_execution(
    recorder: _EvaluatorRecorder,
    policy: _RecordingPolicy,
    rows: Sequence[Mapping[str, Any]],
    cpu_identity: Mapping[str, str],
) -> TaskExecution:
    if len(recorder.episodes) != 20 or len(rows) != 20:
        raise ContractError("scientific execution must cover exactly twenty paired identities")
    evidence = tuple(tuple(episode) for episode in recorder.episodes)
    for expected_id, episode in zip(REGISTERED_EPISODE_IDS, evidence):
        if len(episode) != REGISTERED_HORIZON or episode[0].episode_id != expected_id:
            raise ContractError("scientific execution did not complete every 50-step identity")
        reconstruct_episode(episode)
    actions = tuple(tuple(item.selected_action for item in episode) for episode in evidence)
    if actions != tuple(tuple(item) for item in policy._episode_actions):
        raise ContractError("policy/evaluator action sequence mismatch")
    posteriors = tuple(
        tuple(tuple(float(value) for value in row) for row in episode)
        for episode in policy._episode_posteriors
    )
    if policy._method == "bamcts" and any(len(episode) != 51 for episode in posteriors):
        raise ContractError("BA-MCTS posterior evidence is not initial-through-terminal")
    dispersion = tuple(
        tuple(float(value) for value in episode) for episode in policy._episode_dispersion
    )
    if policy._method == "refplan" and any(len(episode) != 50 for episode in dispersion):
        raise ContractError("RefPlan predictive dispersion evidence is incomplete")
    context_before: list[str] = []
    context_after: list[str] = []
    observation_hashes: list[str] = []
    action_hashes: list[str] = []
    for episode in evidence:
        observations = [np.float64(item.noisy_observation).hex() for item in episode]
        action_sequence = [item.selected_action for item in episode]
        observation_hash = sha256_bytes(canonical_json_bytes(observations))
        action_hash = sha256_bytes(canonical_json_bytes(action_sequence))
        public_context = sha256_bytes(
            canonical_json_bytes(
                {
                    "observation_history_sha256": observation_hash,
                    "action_history_sha256": action_hash,
                }
            )
        )
        context_before.append(public_context)
        context_after.append(public_context)
        observation_hashes.append(observation_hash)
        action_hashes.append(action_hash)
    return TaskExecution(
        evidence,
        tuple(dict(row) for row in rows),
        actions,
        posteriors,
        dispersion,
        tuple(context_before),
        tuple(context_after),
        tuple(observation_hashes),
        tuple(action_hashes),
        validate_cpu_identity_receipt(cpu_identity),
    )


def execute_registered_task(
    *,
    registration: FrozenRegistration,
    inputs: DriverInputs,
    task: Mapping[str, Any],
    task_index: int,
    frozen_payload: bytes,
) -> TaskExecution:
    """Run the frozen evaluator and a fresh sealed fitted object, without fitting."""

    cpu_identity = _require_scientific_environment()
    if "STAGEB_GATE_SIGNING_SEED_FILE" in os.environ and task["arm"] == "T":
        raise ContractError("Arm T environment must not contain the gate signing-seed path")
    public = inputs.public_inputs[task["cell"]]["public_npz"]["path"]
    config_key = (
        "plus_config_sha256"
        if task["method"].startswith("plus_")
        else "moor_config_sha256"
        if task["method"].startswith("moor_")
        else "general_config_sha256"
    )
    config_path = inputs.repository_root / CONFIG_PATHS[config_key]
    track = TRACKS[task["method"]]
    bridge = _ExactStateBridge()
    with _registered_track_imports(track, inputs.repository_root):
        from real_ecology_benchmark import evaluator as evaluator_module
        from real_ecology_benchmark import pipeline
        from real_ecology_benchmark.backend import resolve_backend_for_workload, set_active_backend
        from real_ecology_benchmark.config import load_config, real_environment_like
        from real_ecology_benchmark.dataset import load_public

        rng_contract = validate_rng_contract_document(inputs.rng_contract)
        cfg = load_config(config_path)
        cfg.environment = real_environment_like(cfg.environment, POPULATIONS[task["cell"]], "allee")
        cfg.environment = replace(
            cfg.environment,
            observation_noise_sigma=rng_contract["observation_noise_sigma"],
            reward_mode="safe",
            expose_rk="hidden",
        )
        cfg.dataset.output = str(public)
        cfg.evaluation.output_dir = "DRIVER_OWNS_ATOMIC_PUBLICATION"
        cfg.validate()
        if (
            tuple(cfg.evaluation.seeds) != (7001, 7051, 7101, 7151, 7201)
            or cfg.evaluation.episodes_per_seed != 4
            or cfg.evaluation.horizon != 50
            or cfg.evaluation.discount != 0.95
            or canonical_noise_sigma(
                cfg.environment.process_noise_sigma, "effective process noise sigma"
            ).hex()
            != rng_contract["process_noise_sigma"].hex()
            or canonical_noise_sigma(
                cfg.environment.observation_noise_sigma, "effective observation noise sigma"
            ).hex()
            != rng_contract["observation_noise_sigma"].hex()
            or cfg.environment.kind != "allee"
            or cfg.environment.num_actions != 11
        ):
            raise ContractError("frozen evaluator configuration mismatch")
        policy = load_frozen_fitted_object(frozen_payload, repository_root=inputs.repository_root)
        dataset = load_public(public)
        if len(dataset) != 4000 or dataset.num_episodes != 160:
            raise ContractError("public input is not the registered 4,000-transition dataset")
        method_context = policy.public_context
        if (
            canonical_noise_sigma(
                method_context.observation_noise_sigma,
                "frozen fitted-object observation noise sigma",
            ).hex()
            != rng_contract["observation_noise_sigma"].hex()
            or method_context.surrogate is not policy.public_context.surrogate
        ):
            raise ContractError("frozen fitted object has an incompatible public context")
        filter_factory, _proposal = pipeline.make_filter_factory(
            cfg, dataset, FILTERS[task["method"]], method_context
        )
        set_active_backend(
            resolve_backend_for_workload(
                cfg.compute, f"corrected-stageb-{task['arm']}:{task['method']}", cfg.environment
            )
        )
        recorder = _EvaluatorRecorder(registration, task, bridge, rng_contract)
        wrapped_policy = _RecordingPolicy(policy, task["method"], task["arm"], bridge)
        base_make_env = evaluator_module.make_env
        evaluator_module.make_env = lambda environment_cfg: recorder.wrap(
            base_make_env(environment_cfg)
        )
        try:
            if task["arm"] == "O":
                scientific_filter_factory: Callable[[], Any] = filter_factory
                label = FILTERS[task["method"]]
            else:

                def scientific_filter_factory() -> Any:
                    return _ExactCurrentFilter(
                        filter_factory(),
                        bridge,
                        float(method_context.observation_noise_sigma),
                        float(method_context.observation_scale),
                    )

                label = "registered_context_preserving_exact_current_state"
            evaluator = evaluator_module.ContinuousEvaluator(cfg, scientific_filter_factory, label)
            rows = evaluator.run(wrapped_policy)
        finally:
            evaluator_module.make_env = base_make_env
    return _complete_execution(recorder, wrapped_policy, rows, cpu_identity)


def _transition_diagnostic(
    registration: FrozenRegistration,
    task: Mapping[str, Any],
    components: Mapping[str, bytes],
) -> Mapping[str, Any]:
    method = task["method"]
    if method == "ensemble_value_disagreement_pessimism":
        return {
            "schema_version": "corrected_stageb_transition_not_applicable_v1",
            "registration_sha256": registration.sha256,
            "method": method,
            "cell": task["cell"],
            "arm": task["arm"],
            "applicability": "DEFINITIONALLY_NOT_APPLICABLE",
            "source_backed_reason": "EVD has no fitted transition model",
            "artifact_hashes": [],
        }
    payload = components.get("residual_process_scales")
    if payload is None:
        raise ContractError("registered transition-scale artifact is missing")
    artifact = CanonicalArtifact.from_bytes(payload)
    artifact_hash = sha256_bytes(payload)
    state = artifact.state
    if method in ECOLOGICAL_METHODS:
        values = np.asarray(state["process_scale"], dtype=np.float64).tolist()
        return build_arm_transition_diagnostics(
            registration_sha256=registration.sha256,
            method=method,
            cell=task["cell"],
            arm=task["arm"],
            process_scale=values,
            artifact_hashes=[artifact_hash] * len(values),
        )
    values = np.asarray(state["residual_sigma"], dtype=np.float64).tolist()
    return build_arm_transition_diagnostics(
        registration_sha256=registration.sha256,
        method=method,
        cell=task["cell"],
        arm=task["arm"],
        residual_sigma=values,
        artifact_hashes=[artifact_hash] * len(values),
    )


def _build_diagnostics(
    registration: FrozenRegistration,
    task: Mapping[str, Any],
    execution: TaskExecution,
    components: Mapping[str, bytes],
) -> Mapping[str, Any]:
    activities = [
        classify_activity(
            actions,
            registration_sha256=registration.sha256,
            method=task["method"],
            cell=task["cell"],
            arm=task["arm"],
            episode_id=episode_id,
        )
        for episode_id, actions in zip(REGISTERED_EPISODE_IDS, execution.actions)
    ]
    receipts: dict[str, Any] = {
        "transition": _transition_diagnostic(registration, task, components),
        "activity": activities,
    }
    if task["method"] == "bamcts":
        model_bank_hash = sha256_bytes(components["bamcts_model_bank"])
        receipts["realized_posterior"] = [
            build_realized_posterior_receipt(
                registration_sha256=registration.sha256,
                method="bamcts",
                cell=task["cell"],
                arm=task["arm"],
                episode_id=episode_id,
                member_ids=range(5),
                timesteps=range(51),
                model_bank_sha256=model_bank_hash,
                posterior_probabilities=posterior,
            )
            for episode_id, posterior in zip(
                REGISTERED_EPISODE_IDS, execution.posterior_probabilities
            )
        ]
    if task["method"] == "refplan":
        dynamics_hash = sha256_bytes(components["dynamics_ensemble"])
        receipts["predictive_dispersion"] = [
            build_arm_refplan_predictive_dispersion(
                registration_sha256=registration.sha256,
                cell=task["cell"],
                arm=task["arm"],
                episode_id=episode_id,
                timesteps=range(50),
                artifact_sha256=dynamics_hash,
                values=values,
            )
            for episode_id, values in zip(REGISTERED_EPISODE_IDS, execution.predictive_dispersion)
        ]
    return {
        "schema_version": BOUND_TASK_EVIDENCE_SCHEMA_VERSION,
        "kind": "diagnostics",
        "registration_sha256": registration.sha256,
        "task_index": task["task_index"],
        "arm": task["arm"],
        "cell": task["cell"],
        "method": task["method"],
        "receipts": receipts,
    }


def _build_boundary(
    registration: FrozenRegistration,
    task: Mapping[str, Any],
    execution: TaskExecution,
) -> Mapping[str, Any]:
    feature_order = (
        ["raw_abundance", "survey_scale", "latent_abundance_grid"]
        if task["method"] in ECOLOGICAL_METHODS
        else list(FEATURE_NAMES)
    )
    receipt = build_task_information_boundary_receipt(
        registration_sha256=registration.sha256,
        method=task["method"],
        cell=task["cell"],
        arm=task["arm"],
        feature_order_sha256=sha256_bytes(canonical_json_bytes(feature_order)),
        context_sha256_before=execution.context_sha256_before,
        context_sha256_after=execution.context_sha256_after,
        observation_history_sha256=execution.observation_history_sha256,
        action_history_sha256=execution.action_history_sha256,
    )
    return {
        "schema_version": BOUND_TASK_EVIDENCE_SCHEMA_VERSION,
        "kind": "information_boundary",
        "registration_sha256": registration.sha256,
        "task_index": task["task_index"],
        "arm": task["arm"],
        "cell": task["cell"],
        "method": task["method"],
        "receipts": {"runtime_information_boundary": receipt},
    }


def _accepted_parity(
    rows: Sequence[Mapping[str, Any]],
    accepted_path: Path,
    comparison_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    return compare_accepted_parity(rows, accepted_path, comparison_contract)


def _parity_receipt_fields(*, eligible_baseline_supplied: bool) -> Mapping[str, Any]:
    if not isinstance(eligible_baseline_supplied, bool):
        raise ContractError("eligible-baseline receipt flag must be boolean")
    return {
        "parity_mode": (
            PARITY_MODE_EXTERNAL_BASELINE
            if eligible_baseline_supplied
            else PARITY_MODE_DISCLOSURE_ONLY
        ),
        "eligible_baseline_supplied": eligible_baseline_supplied,
        "historical_canary_identity": HISTORICAL_CANARY_IDENTITY,
        "historical_canary_used_for_authorization": False,
    }


def _relative_reference(
    output_root: Path, path: Path, digest: str | None = None
) -> Mapping[str, str]:
    root = output_root.resolve(strict=True)
    candidate = path.resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise ContractError("driver evidence path escapes the output root")
    relative = candidate.relative_to(root).as_posix()
    pure = PurePosixPath(relative)
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ContractError("driver evidence reference is unsafe")
    return {"relative_path": relative, "sha256": digest or sha256_file(path)}


def _anticipated_success(
    staging: Path, required_files: Sequence[str], validation: Mapping[str, Any]
) -> bytes:
    hashes = validate_staging_tree(staging, required_files)
    if validation.get("result") != "PASS":
        raise ContractError("task validation did not pass")
    return canonical_json_bytes(
        {
            "schema_version": "corrected_stageb_publication_success_v2",
            "result": "PASS",
            "files": hashes,
            "full_tree_coverage": True,
            "validation": dict(validation),
            "automatic_retry_permitted": False,
        }
    )


def _step_from_mapping(value: Mapping[str, Any]) -> StepEvidence:
    payload = dict(value)
    if payload.pop("evidence_layer", None) != "EVALUATOR_ONLY":
        raise ContractError("stored step evidence lost its evaluator-only label")
    require_exact_keys(
        payload,
        set(StepEvidence.__dataclass_fields__),
        "stored v2 step evidence",
    )
    if payload["schema_version"] != STEP_EVIDENCE_SCHEMA_VERSION:
        raise ContractError("stored step evidence schema mismatch")
    rng = payload.get("rng_receipt")
    if not isinstance(rng, Mapping):
        raise ContractError("stored step evidence RNG receipt is missing")
    require_exact_keys(
        rng,
        set(RNGReceipt.__dataclass_fields__),
        "stored v2 RNG receipt",
    )
    if rng["schema_version"] != RNG_RECEIPT_SCHEMA_VERSION:
        raise ContractError("stored RNG receipt schema mismatch")
    payload["rng_receipt"] = RNGReceipt(**rng)
    return StepEvidence(**payload)


def _load_arm_o_evidence(
    output_root: Path, task_index: int
) -> tuple[tuple[StepEvidence, ...], ...]:
    path = output_root / "arm-o" / f"task-{task_index:02d}" / "STEP_EVIDENCE.json"
    value = strict_json_loads(path.read_bytes())
    if not isinstance(value, list) or len(value) != 20:
        raise ContractError("paired Arm O step evidence is incomplete")
    return tuple(tuple(_step_from_mapping(item) for item in episode) for episode in value)


def _ensure_role_root(output_root: Path, name: str) -> Path:
    root = Path(output_root)
    if root.is_symlink() or not root.is_dir():
        raise ContractError("output root must be an existing real directory")
    role = root / name
    try:
        role.mkdir(mode=0o700)
    except FileExistsError:
        pass
    if role.is_symlink() or not role.is_dir():
        raise ContractError(f"{name} root is not a real directory")
    return role


def _parity_baseline_expectations(
    *,
    registration: FrozenRegistration,
    inputs: DriverInputs,
    task: Mapping[str, Any],
    task_index: int,
    interpreter_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    bundle = registration.bundle()
    hashes = bundle["code_configuration_hashes"]
    registered = bundle["corrected_stageb_registration"]
    probe = inputs.fit_probes[task_index]
    interpreter = interpreter_identity["expected"]
    return {
        "repository_commit": hashes["git_commit_sha"],
        "source_manifest_sha256": hashes["source_manifest_sha256"],
        "stageb_driver_sha256": hashes["stageb_driver_sha256"],
        "rng_contract": validate_rng_contract_document(inputs.rng_contract),
        "interpreter": {
            field: interpreter[field]
            for field in (
                "role",
                "track",
                "absolute_interpreter_path",
                "resolved_executable_path",
                "python_version",
                "full_python_version",
                "numpy_version",
            )
        },
        "fitted_object_sha256": probe.frozen_object_sha256,
        "component_hashes": dict(sorted(probe.component_hashes.items())),
        "artifact_plan_sha256": task["artifact_plan_sha256"],
        "evaluator": {
            "family": registered["evaluator_family"],
            "reward_mode": "safe",
            "config_sha256": task["config_sha256"],
            "evaluation_identity_sha256": task["evaluation_identity_sha256"],
            "episode_ids": list(registered["evaluation_identities"]),
            "horizon": registered["horizon"],
            "discount": registered["discount"],
            "num_actions": registered["num_actions"],
        },
    }


def _parity_current_provenance(
    *,
    registration: FrozenRegistration,
    inputs: DriverInputs,
    task: Mapping[str, Any],
    task_index: int,
    interpreter_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected = dict(
        _parity_baseline_expectations(
            registration=registration,
            inputs=inputs,
            task=task,
            task_index=task_index,
            interpreter_identity=interpreter_identity,
        )
    )
    return {
        "registration_sha256": registration.sha256,
        "driver_inputs_sha256": registration.bundle()["stageb_driver_inputs"][
            "driver_inputs_sha256"
        ],
        **expected,
    }


def _parity_rng_evidence(
    execution: TaskExecution | None, rng_contract: Mapping[str, Any]
) -> Mapping[str, Any]:
    receipts = (
        [asdict(step.rng_receipt) for episode in execution.step_evidence for step in episode]
        if execution is not None
        else []
    )
    return {
        "receipt_schema_version": RNG_RECEIPT_SCHEMA_VERSION,
        "rng_contract": validate_rng_contract_document(rng_contract),
        "receipt_count": len(receipts),
        "receipts_canonical_sha256": sha256_bytes(canonical_json_bytes(receipts)),
        "status": "VALIDATED_POST_ROLLOUT" if execution is not None else "NOT_CONSTRUCTED",
    }


def _safe_parity_comparison_summary(parity: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep comparison metadata and delta magnitudes without truth-named value slots."""

    summary = {
        key: value
        for key, value in parity.items()
        if key
        not in {
            "mismatches",
            "maximum_absolute_deltas",
            "maximum_relative_deltas",
            "failed_numeric_fields",
        }
    }
    summary["stage"] = "VALUE_COMPARISON"
    summary["comparison_type"] = "registered_exact_and_numeric"
    summary["maximum_absolute_deltas"] = [
        {"field_name": field, "delta": delta}
        for field, delta in parity["maximum_absolute_deltas"].items()
    ]
    summary["maximum_relative_deltas"] = [
        {"field_name": field, "delta": delta}
        for field, delta in parity["maximum_relative_deltas"].items()
    ]
    summary["failed_numeric_fields"] = sorted(parity["failed_numeric_fields"])
    return summary


def _publish_parity_failure(
    *,
    registration: FrozenRegistration,
    inputs: DriverInputs,
    accepted: Mapping[str, Any],
    task: Mapping[str, Any],
    task_index: int,
    output_root: Path,
    interpreter_identity: Mapping[str, Any],
    eligibility: Mapping[str, Any],
    comparison: Mapping[str, Any],
    mismatches: Sequence[Mapping[str, Any]],
    execution: TaskExecution | None,
) -> Path:
    reference_path = accepted["episodes_csv"]["path"]
    baseline = accepted.get("baseline")
    reference_provenance: Mapping[str, Any]
    try:
        reference_provenance = validate_parity_baseline_binding(baseline)
    except ContractError:
        reference_provenance = {
            "classification": "UNBOUND_OR_MALFORMED",
            "missing_bindings": list(eligibility.get("missing_bindings", [])),
        }
    diagnostic = {
        "schema_version": "corrected_stageb_arm_o_parity_failure_v1",
        "result": "FAIL",
        "accepted": False,
        "qualifies_as_task_result": False,
        "automatic_retry_permitted": False,
        "task_index": task_index,
        "arm": "O",
        "cell": task["cell"],
        "method": task["method"],
        "current_provenance": _parity_current_provenance(
            registration=registration,
            inputs=inputs,
            task=task,
            task_index=task_index,
            interpreter_identity=interpreter_identity,
        ),
        "reference_provenance": {
            "episodes_csv_absolute_path": str(Path(reference_path)),
            "episodes_csv_registered_sha256": accepted["episodes_csv"]["sha256"],
            "episodes_csv_observed_sha256": reference_file_sha256(reference_path),
            "binding": reference_provenance,
        },
        "eligibility": dict(eligibility),
        "comparison": dict(comparison),
        "mismatches": [dict(item) for item in mismatches],
        "observed_rows_sha256": sha256_bytes(
            canonical_json_bytes(list(execution.evaluator_rows) if execution is not None else [])
        ),
        "reference_rows_sha256": canonical_reference_rows_sha256(reference_path),
        "rng_evidence": _parity_rng_evidence(execution, inputs.rng_contract),
        "information_boundary": {
            "raw_truth_values_published": False,
            "runtime_next_states_published": False,
            "truth_trajectory_published": False,
            "mismatch_values_hashed": True,
        },
    }
    validate_parity_failure_diagnostic(diagnostic)
    root = _ensure_role_root(output_root, PARITY_FAILURE_NAMESPACE)
    staging, target = create_task_staging(root, f"task-{task_index:02d}")
    write_bytes_fsync(staging / PARITY_FAILURE_FILENAME, canonical_json_bytes(diagnostic))
    return publish_failure_once(
        staging=staging,
        target=target,
        diagnostic_filename=PARITY_FAILURE_FILENAME,
        validator=validate_parity_failure_diagnostic,
    )


def _resolve_parity_authorization(
    *,
    registration: FrozenRegistration,
    inputs: DriverInputs,
    accepted: Mapping[str, Any],
    task: Mapping[str, Any],
    task_index: int,
    output_root: Path,
    interpreter_identity: Mapping[str, Any],
    execution: TaskExecution | None,
) -> Mapping[str, Any]:
    if accepted.get("historical_canary_identity") != HISTORICAL_CANARY_IDENTITY:
        raise ContractError("historical canary disclosure identity mismatch")
    baseline = validate_parity_baseline_binding(accepted.get("baseline"))
    if baseline["classification"] == "LEGACY_INELIGIBLE":
        if not is_legacy_fast_track_canary_path(accepted["episodes_csv"]["path"]):
            raise ContractError("legacy disclosure must retain the historical canary path")
        return {
            **_parity_receipt_fields(eligible_baseline_supplied=False),
            "binding": None,
            "assessment": {
                "schema_version": "corrected_stageb_parity_baseline_eligibility_v1",
                "classification": "LEGACY_INELIGIBLE",
                "eligible": False,
                "missing_bindings": list(baseline["missing_bindings"]),
                "mismatched_bindings": [],
                "result": "FAIL",
            },
        }
    expected = _parity_baseline_expectations(
        registration=registration,
        inputs=inputs,
        task=task,
        task_index=task_index,
        interpreter_identity=interpreter_identity,
    )
    if sha256_file(Path(accepted["episodes_csv"]["path"])) != accepted["episodes_csv"]["sha256"]:
        raise ContractError("eligible external parity baseline changed after registration")
    try:
        eligibility = validate_parity_baseline_eligibility(
            baseline,
            expected=expected,
            accepted_csv=accepted["episodes_csv"]["path"],
            current_registration_sha256=registration.sha256,
            current_driver_inputs_sha256=registration.bundle()["stageb_driver_inputs"][
                "driver_inputs_sha256"
            ],
            current_output_root=output_root,
        )
        return {
            **_parity_receipt_fields(eligible_baseline_supplied=True),
            **eligibility,
        }
    except ParityEligibilityError as exc:
        comparison = {
            "stage": "BASELINE_ELIGIBILITY",
            "comparison_type": "execution_contract_binding",
            "absolute_tolerance": PARITY_TOLERANCE,
            "result": "FAIL",
        }
        _publish_parity_failure(
            registration=registration,
            inputs=inputs,
            accepted=accepted,
            task=task,
            task_index=task_index,
            output_root=output_root,
            interpreter_identity=interpreter_identity,
            eligibility=exc.assessment,
            comparison=comparison,
            mismatches=(),
            execution=execution,
        )
        raise


def publish_registered_task(
    *,
    registration: FrozenRegistration,
    inputs: DriverInputs,
    evaluator_inputs: EvaluatorOnlyInputs | None,
    task: Mapping[str, Any],
    task_index: int,
    execution: TaskExecution,
    component_payloads: Mapping[str, bytes],
    frozen_payload: bytes,
    replay_payload: bytes,
    output_root: Path,
    interpreter_identity: Mapping[str, Any],
) -> Path:
    cpu_identity = validate_cpu_identity_receipt(execution.cpu_identity)
    role = "arm-o" if task["arm"] == "O" else "arm-t"
    accepted: Mapping[str, Any] | None = None
    parity_authorization: Mapping[str, Any] | None = None
    if task["arm"] == "O":
        if evaluator_inputs is None:
            raise ContractError("Arm O publication requires evaluator-only accepted parity")
        accepted = evaluator_inputs.accepted_parity[task_index]
        if (
            accepted["task_index"] != task_index
            or accepted["cell"] != task["cell"]
            or accepted["method"] != task["method"]
        ):
            assessment = {
                "schema_version": "corrected_stageb_parity_baseline_eligibility_v1",
                "classification": "TASK_BINDING_MISMATCH",
                "eligible": False,
                "missing_bindings": [],
                "mismatched_bindings": ["task_cell_method"],
                "result": "FAIL",
            }
            _publish_parity_failure(
                registration=registration,
                inputs=inputs,
                accepted=accepted,
                task=task,
                task_index=task_index,
                output_root=output_root,
                interpreter_identity=interpreter_identity,
                eligibility=assessment,
                comparison={
                    "stage": "BASELINE_TASK_BINDING",
                    "comparison_type": "exact",
                    "absolute_tolerance": PARITY_TOLERANCE,
                    "result": "FAIL",
                },
                mismatches=(),
                execution=execution,
            )
            raise ContractError("accepted-parity capability is bound to a different task")
        parity_authorization = _resolve_parity_authorization(
            registration=registration,
            inputs=inputs,
            accepted=accepted,
            task=task,
            task_index=task_index,
            output_root=output_root,
            interpreter_identity=interpreter_identity,
            execution=execution,
        )
    publications = _ensure_role_root(output_root, role)
    receipts_root = _ensure_role_root(output_root, f"{role}-receipts")
    staging, target = create_task_staging(publications, f"task-{task_index:02d}")
    artifact, required = _copy_validated_artifacts(
        staging,
        registration=registration,
        task=task,
        component_payloads=component_payloads,
        frozen_payload=frozen_payload,
        replay_payload=replay_payload,
    )
    target_relative = target.relative_to(output_root).as_posix()
    artifact["components"] = {
        name: {
            "relative_path": f"{target_relative}/artifact-{name}.json",
            "sha256": sha256_bytes(payload),
        }
        for name, payload in sorted(component_payloads.items())
    }
    artifact["frozen_object_artifact"] = {
        "relative_path": f"{target_relative}/FROZEN_FITTED_OBJECT.json",
        "sha256": sha256_bytes(frozen_payload),
    }
    artifact["frozen_object_parity"] = {
        "relative_path": f"{target_relative}/FRESH_RELOAD_REPLAY.json",
        "sha256": sha256_bytes(replay_payload),
    }
    diagnostics = _build_diagnostics(registration, task, execution, component_payloads)
    boundary = _build_boundary(registration, task, execution)
    step_payload = canonical_json_bytes(
        [[item.evaluator_only_dict() for item in episode] for episode in execution.step_evidence]
    )
    reconstructed = canonical_json_bytes(
        [reconstruct_episode(episode) for episode in execution.step_evidence]
    )
    rows_payload = canonical_json_bytes(list(execution.evaluator_rows))
    files = {
        "ARTIFACT_BUNDLE.json": canonical_json_bytes(artifact),
        "DIAGNOSTICS.json": canonical_json_bytes(diagnostics),
        "INFORMATION_BOUNDARY.json": canonical_json_bytes(boundary),
        "STEP_EVIDENCE.json": step_payload,
        "RETURN_RECONSTRUCTION.json": reconstructed,
        "EVALUATOR_ROWS.json": rows_payload,
    }
    parity: Mapping[str, Any] | None = None
    if task["arm"] == "O":
        assert accepted is not None and parity_authorization is not None
        if parity_authorization["eligible_baseline_supplied"]:
            parity = _accepted_parity(
                execution.evaluator_rows,
                accepted["episodes_csv"]["path"],
                parity_authorization["binding"]["comparison_contract"],
            )
            if parity["result"] != "PASS":
                _publish_parity_failure(
                    registration=registration,
                    inputs=inputs,
                    accepted=accepted,
                    task=task,
                    task_index=task_index,
                    output_root=output_root,
                    interpreter_identity=interpreter_identity,
                    eligibility=parity_authorization["assessment"],
                    comparison=_safe_parity_comparison_summary(parity),
                    mismatches=parity["mismatches"],
                    execution=execution,
                )
                raise ContractError("Arm O accepted parity failed")
            files["ACCEPTED_PARITY.json"] = canonical_json_bytes(parity)
    else:
        if evaluator_inputs is not None:
            raise ContractError("Arm T must not receive evaluator-only accepted parity")
        arm_o = _load_arm_o_evidence(output_root, task_index)
        pairings = [
            pair_episode_evidence(o_episode, t_episode)
            for o_episode, t_episode in zip(arm_o, execution.step_evidence)
        ]
        files["PAIRED_RNG_EVIDENCE.json"] = canonical_json_bytes(pairings)
    for filename, payload in files.items():
        write_bytes_fsync(staging / filename, payload)
        required.append(filename)
    validation = {
        "schema_version": "corrected_stageb_task_publication_validation_v3",
        "registration_sha256": registration.sha256,
        "task_index": task["task_index"],
        "arm": task["arm"],
        "cell": task["cell"],
        "method": task["method"],
        "artifact_bundle_sha256": sha256_bytes(files["ARTIFACT_BUNDLE.json"]),
        "diagnostics_sha256": sha256_bytes(files["DIAGNOSTICS.json"]),
        "information_boundary_sha256": sha256_bytes(files["INFORMATION_BOUNDARY.json"]),
        "interpreter_identity": dict(interpreter_identity),
        "cpu_identity": dict(cpu_identity),
        "result": "PASS",
    }
    if task["arm"] == "O":
        assert parity_authorization is not None
        parity_fields = _parity_receipt_fields(
            eligible_baseline_supplied=parity_authorization["eligible_baseline_supplied"]
        )
    else:
        paired_receipt = _json_object(
            (output_root / "arm-o-receipts" / f"task-{task_index:02d}.json").read_bytes(),
            "paired Arm O task receipt",
        )
        parity_fields = {
            field: paired_receipt[field]
            for field in (
                "parity_mode",
                "eligible_baseline_supplied",
                "historical_canary_identity",
                "historical_canary_used_for_authorization",
            )
        }
        expected_parity_fields = _parity_receipt_fields(
            eligible_baseline_supplied=parity_fields["eligible_baseline_supplied"]
        )
        if parity_fields != expected_parity_fields:
            raise ContractError("paired Arm O parity policy receipt is invalid")
    validation.update(parity_fields)
    success_payload = _anticipated_success(staging, required, validation)
    success_sha = sha256_bytes(success_payload)
    receipt_path = receipts_root / f"task-{task_index:02d}.json"
    if task["arm"] == "O":
        receipt = {
            "schema_version": "corrected_stageb_arm_o_task_receipt_v5",
            "task_index": task["task_index"],
            "arm": "O",
            "cell": task["cell"],
            "method": task["method"],
            "registration_sha256": registration.sha256,
            "task_manifest_entry_sha256": sha256_bytes(canonical_json_bytes(task)),
            "config_sha256": task["config_sha256"],
            "dataset_sha256": task["dataset_sha256"],
            "artifact_plan_sha256": task["artifact_plan_sha256"],
            "evaluation_identity_sha256": task["evaluation_identity_sha256"],
            "artifact_bundle_sha256": validation["artifact_bundle_sha256"],
            "diagnostics_sha256": validation["diagnostics_sha256"],
            "information_boundary_sha256": validation["information_boundary_sha256"],
            "publication_manifest_sha256": success_sha,
            "evidence": {
                "artifact_bundle": _relative_reference(
                    output_root,
                    target / "ARTIFACT_BUNDLE.json",
                    validation["artifact_bundle_sha256"],
                ),
                "diagnostics": _relative_reference(
                    output_root, target / "DIAGNOSTICS.json", validation["diagnostics_sha256"]
                ),
                "information_boundary": _relative_reference(
                    output_root,
                    target / "INFORMATION_BOUNDARY.json",
                    validation["information_boundary_sha256"],
                ),
                "publication_success": _relative_reference(
                    output_root, target / SUCCESS_RECEIPT, success_sha
                ),
            },
            "cpu_profile": CPU_PROFILE,
            "cpu_identity": dict(cpu_identity),
            "interpreter_identity": dict(interpreter_identity),
            "result": "PASS",
            **parity_fields,
        }
    else:
        receipt = {
            "schema_version": "corrected_stageb_arm_t_task_receipt_v3",
            "task_index": task["task_index"],
            "arm": "T",
            "cell": task["cell"],
            "method": task["method"],
            "registration_sha256": registration.sha256,
            "artifact_plan_sha256": task["artifact_plan_sha256"],
            "frozen_object_sha256": sha256_bytes(frozen_payload),
            "paired_arm_o_frozen_object_sha256": inputs.fit_probes[task_index].frozen_object_sha256,
            "arm_t_refit_performed": False,
            "fitted_artifacts_byte_identical": True,
            "model_fit_label": "FROZEN-FIT/STATE-INPUT-ONLY",
            "step_evidence_sha256": sha256_bytes(step_payload),
            "paired_rng_evidence_sha256": sha256_bytes(files["PAIRED_RNG_EVIDENCE.json"]),
            "publication_manifest_sha256": success_sha,
            "cpu_profile": CPU_PROFILE,
            "cpu_identity": dict(cpu_identity),
            "interpreter_identity": dict(interpreter_identity),
            "result": "PASS",
            **parity_fields,
        }
        if receipt["frozen_object_sha256"] != receipt["paired_arm_o_frozen_object_sha256"]:
            raise ContractError("Arm T fitted object differs from its paired Arm O object")
    write_bytes_fsync(receipt_path, canonical_json_bytes(receipt))
    publish_once(
        staging=staging,
        target=target,
        required_files=required,
        validator=lambda _root: validation,
    )
    return target


def run_arm_task(
    *,
    arm: str,
    task_index: int,
    registration_path: Path,
    output_root: Path,
    arm_o_gate_path: Path | None = None,
    gate_public_key_path: Path | None = None,
    executor: Callable[..., TaskExecution] | None = None,
) -> Path:
    registration = _load_registration(registration_path)
    task = _task_for_index(registration, arm, task_index)
    interpreter_identity = require_runtime_interpreter_binding(
        registration,
        command="arm-o" if arm == "O" else "arm-t",
        arm=arm,
        task_index=task_index,
    )
    if arm == "T":
        if arm_o_gate_path is None or gate_public_key_path is None:
            raise ContractError("Arm T requires the signed Arm O gate and public key")
        if "STAGEB_GATE_SIGNING_SEED_FILE" in os.environ:
            raise ContractError("the gate signing seed must never enter an Arm T process")
    elif arm_o_gate_path is not None or gate_public_key_path is not None:
        raise ContractError("Arm O must not receive Arm T gate material")
    loaded = load_driver_inputs(output_root, registration)
    inputs = loaded.policy
    if arm == "T":
        assert arm_o_gate_path is not None and gate_public_key_path is not None
        gate_path = _safe_absolute_file(str(Path(arm_o_gate_path).resolve()), "Arm O gate")
        verification_key = load_gate_verification_key(Path(gate_public_key_path))
        token, embedded_registration = verify_arm_o_gate_receipt(
            gate_path.read_bytes(),
            verification_key=verification_key,
            repository_root=inputs.repository_root,
        )
        if embedded_registration.payload != registration.payload:
            raise ContractError("Arm T gate embeds a different frozen registration")
        authorize_arm_t_task(token, registration)
    components, frozen_payload, replay_payload, _probe = _validate_fit_probe(
        inputs, task, task_index
    )
    if arm == "O":
        accepted = loaded.evaluator_only.accepted_parity[task_index]
        _resolve_parity_authorization(
            registration=registration,
            inputs=inputs,
            accepted=accepted,
            task=task,
            task_index=task_index,
            output_root=output_root,
            interpreter_identity=interpreter_identity,
            execution=None,
        )
    implementation = executor or execute_registered_task
    execution = implementation(
        registration=registration,
        inputs=inputs,
        task=task,
        task_index=task_index,
        frozen_payload=frozen_payload,
    )
    if not isinstance(execution, TaskExecution):
        raise ContractError("registered task executor returned an invalid result")
    return publish_registered_task(
        registration=registration,
        inputs=inputs,
        evaluator_inputs=loaded.evaluator_only if arm == "O" else None,
        task=task,
        task_index=task_index,
        execution=execution,
        component_payloads=components,
        frozen_payload=frozen_payload,
        replay_payload=replay_payload,
        output_root=output_root,
        interpreter_identity=interpreter_identity,
    )


def _load_arm_o_task_receipts(output_root: Path) -> tuple[list[bytes], list[Mapping[str, Any]]]:
    payloads: list[bytes] = []
    receipts: list[Mapping[str, Any]] = []
    for index in range(12):
        path = output_root / "arm-o-receipts" / f"task-{index:02d}.json"
        payload = path.read_bytes()
        value = _json_object(payload, f"Arm O task receipt {index}")
        publication = output_root / "arm-o" / f"task-{index:02d}" / SUCCESS_RECEIPT
        if value.get("publication_manifest_sha256") != sha256_file(publication):
            raise ContractError("Arm O task/publication identity mismatch")
        load_success_receipt(publication)
        expected_policy = _parity_receipt_fields(
            eligible_baseline_supplied=value.get("eligible_baseline_supplied")
        )
        if any(value.get(field) != expected for field, expected in expected_policy.items()):
            raise ContractError("Arm O task parity policy binding failed")
        parity_path = output_root / "arm-o" / f"task-{index:02d}" / "ACCEPTED_PARITY.json"
        if value["eligible_baseline_supplied"]:
            parity = _json_object(parity_path.read_bytes(), f"Arm O task parity {index}")
            if parity.get("result") != "PASS":
                raise ContractError("Arm O external baseline parity failed")
        elif parity_path.exists():
            raise ContractError("disclosure-only Arm O task performed a numeric comparison")
        payloads.append(payload)
        receipts.append(value)
    return payloads, receipts


def run_arm_o_gate(
    *,
    registration_path: Path,
    gate_signing_seed_path: Path,
    output_root: Path,
) -> Path:
    registration = _load_registration(registration_path)
    interpreter_identity = require_runtime_interpreter_binding(registration, command="arm-o-gate")
    loaded = load_driver_inputs(output_root, registration)
    task_payloads, task_receipts = _load_arm_o_task_receipts(output_root)
    arm_o_tasks = registration.bundle()["task_manifest"]["tasks"][:12]
    artifact_gate = sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "task_index": task["task_index"],
                    "cell": task["cell"],
                    "method": task["method"],
                    "artifact_plan_sha256": task["artifact_plan_sha256"],
                    "artifact_bundle_evidence_sha256": receipt["artifact_bundle_sha256"],
                }
                for task, receipt in zip(arm_o_tasks, task_receipts)
            ]
        )
    )
    code_hashes = registration.bundle()["code_configuration_hashes"]
    parity_payload = canonical_json_bytes(
        {
            "schema_version": "corrected_stageb_arm_o_parity_v2",
            "architecture": CPU_MODEL,
            "registration_sha256": registration.sha256,
            "git_commit_sha": code_hashes["git_commit_sha"],
            "per_episode_action_event_parity": True,
            "accepted_artifact_gate_sha256": artifact_gate,
            "verified_hashes": {
                field: code_hashes[field]
                for field in code_hashes
                if field not in {"schema_version", "git_commit_sha"}
            },
            "interpreter_identity": dict(interpreter_identity),
        }
    )
    token = validate_arm_o_gate(
        frozen_registration=registration,
        task_receipts=task_payloads,
        inherited_test_receipt=loaded.gate_only.inherited_test_receipt["path"].read_bytes(),
        corrected_test_receipt=loaded.gate_only.corrected_test_receipt["path"].read_bytes(),
        parity_receipt=parity_payload,
        evidence_root=output_root,
    )
    signing_seed = load_gate_signing_key(Path(gate_signing_seed_path))
    signed = export_arm_o_gate_receipt(token, registration, signing_key=signing_seed)
    gate_root = _ensure_role_root(output_root, "gate")
    staging, target = create_task_staging(gate_root, "arm-o-gate")
    files = {
        GATE_RECEIPT: signed,
        "ARM_O_PARITY_RECEIPT.json": parity_payload,
        "ARM_O_TASK_RECEIPT_HASHES.json": canonical_json_bytes(
            [sha256_bytes(payload) for payload in task_payloads]
        ),
    }
    required: list[str] = []
    for filename, payload in files.items():
        write_bytes_fsync(staging / filename, payload)
        required.append(filename)
    validation = {
        "schema_version": "corrected_stageb_gate_publication_v1",
        "registration_sha256": registration.sha256,
        "arm_o_task_count": 12,
        "signed_gate_sha256": sha256_bytes(signed),
        "methods_executed_by_gate": 0,
        "interpreter_identity": dict(interpreter_identity),
        "result": "PASS",
    }
    publish_once(
        staging=staging,
        target=target,
        required_files=required,
        validator=lambda _root: validation,
    )
    return target


def run_inspection_only_finalizer(*, registration_path: Path, output_root: Path) -> Path:
    registration = _load_registration(registration_path)
    interpreter_identity = require_runtime_interpreter_binding(
        registration, command="finalize-inspection-only"
    )
    load_driver_inputs(output_root, registration)
    statuses: list[Mapping[str, Any]] = []
    inspected: list[Mapping[str, Any]] = []
    for index, (cell, method) in enumerate((cell, method) for cell in CELLS for method in METHODS):
        receipt_path = output_root / "arm-t-receipts" / f"task-{index:02d}.json"
        publication_path = output_root / "arm-t" / f"task-{index:02d}" / SUCCESS_RECEIPT
        status = "FAILED"
        identity_payload = canonical_json_bytes(
            {"task_index": index, "cell": cell, "method": method, "status": "MISSING"}
        )
        receipt_sha = sha256_bytes(identity_payload)
        reason = "task receipt or publication missing"
        if receipt_path.is_file() and not receipt_path.is_symlink():
            payload = receipt_path.read_bytes()
            receipt_sha = sha256_bytes(payload)
            receipt = _json_object(payload, f"Arm T task receipt {index}")
            if publication_path.is_file() and not publication_path.is_symlink():
                publication = load_success_receipt(publication_path)
                if (
                    receipt.get("schema_version") == "corrected_stageb_arm_t_task_receipt_v3"
                    and receipt.get("registration_sha256") == registration.sha256
                    and receipt.get("task_index") == index + 12
                    and receipt.get("cell") == cell
                    and receipt.get("method") == method
                    and receipt.get("result") == "PASS"
                    and receipt.get("arm_t_refit_performed") is False
                    and receipt.get("fitted_artifacts_byte_identical") is True
                    and receipt.get("historical_canary_used_for_authorization") is False
                    and receipt.get("historical_canary_identity") == HISTORICAL_CANARY_IDENTITY
                    and receipt.get("parity_mode")
                    in {PARITY_MODE_DISCLOSURE_ONLY, PARITY_MODE_EXTERNAL_BASELINE}
                    and receipt.get("eligible_baseline_supplied")
                    == (receipt.get("parity_mode") == PARITY_MODE_EXTERNAL_BASELINE)
                    and receipt.get("publication_manifest_sha256") == sha256_file(publication_path)
                    and publication.get("result") == "PASS"
                    and publication.get("validation", {}).get("interpreter_identity")
                    == receipt.get("interpreter_identity")
                    and publication.get("validation", {}).get("cpu_identity")
                    == receipt.get("cpu_identity")
                ):
                    try:
                        validate_interpreter_identity_receipt(
                            receipt.get("interpreter_identity", {}),
                            registration,
                            command="arm-t",
                            arm="T",
                            task_index=index,
                        )
                        validate_cpu_identity_receipt(receipt.get("cpu_identity", {}))
                    except ContractError:
                        reason = "task receipt runtime identity binding failed"
                    else:
                        status = "COMPLETED"
                        reason = "validated terminal task receipt and publication"
                else:
                    reason = "task receipt/publication binding failed"
        statuses.append(
            {
                "task_index": index,
                "cell": cell,
                "method": method,
                "status": status,
                "receipt_sha256": receipt_sha,
            }
        )
        inspected.append({**statuses[-1], "reason": reason})
    summary = inspect_only_finalizer(statuses)
    final_root = _ensure_role_root(output_root, "finalizer")
    staging, target = create_task_staging(final_root, "inspection-only")
    files = {
        "FINALIZER_RECEIPT.json": canonical_json_bytes(
            {
                **summary,
                "registration_sha256": registration.sha256,
                "task_statuses": inspected,
                "scientific_values_opened": False,
                "scientific_calculations_performed": False,
                "interpreter_identity": dict(interpreter_identity),
            }
        )
    }
    for filename, payload in files.items():
        write_bytes_fsync(staging / filename, payload)
    publish_once(
        staging=staging,
        target=target,
        required_files=list(files),
        validator=lambda _root: {
            "schema_version": "corrected_stageb_finalizer_publication_v1",
            "registration_sha256": registration.sha256,
            "methods_executed": 0,
            "scientific_calculations_performed": 0,
            "interpreter_identity": dict(interpreter_identity),
            "result": "PASS",
        },
    )
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="driver.py")
    commands = parser.add_subparsers(dest="command", required=True)
    arm_o = commands.add_parser("arm-o")
    arm_o.add_argument("--task-index", type=int, required=True)
    arm_o.add_argument("--registration", type=Path, required=True)
    arm_o.add_argument("--output-root", type=Path, required=True)
    gate = commands.add_parser("arm-o-gate")
    gate.add_argument("--registration", type=Path, required=True)
    gate.add_argument("--gate-signing-seed-file", type=Path, required=True)
    gate.add_argument("--output-root", type=Path, required=True)
    arm_t = commands.add_parser("arm-t")
    arm_t.add_argument("--task-index", type=int, required=True)
    arm_t.add_argument("--registration", type=Path, required=True)
    arm_t.add_argument("--arm-o-gate", type=Path, required=True)
    arm_t.add_argument("--gate-public-key-file", type=Path, required=True)
    arm_t.add_argument("--output-root", type=Path, required=True)
    finalizer = commands.add_parser("finalize-inspection-only")
    finalizer.add_argument("--registration", type=Path, required=True)
    finalizer.add_argument("--output-root", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    executor: Callable[..., TaskExecution] | None = None,
) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "arm-o":
            run_arm_task(
                arm="O",
                task_index=arguments.task_index,
                registration_path=arguments.registration,
                output_root=arguments.output_root,
                executor=executor,
            )
        elif arguments.command == "arm-o-gate":
            run_arm_o_gate(
                registration_path=arguments.registration,
                gate_signing_seed_path=arguments.gate_signing_seed_file,
                output_root=arguments.output_root,
            )
        elif arguments.command == "arm-t":
            run_arm_task(
                arm="T",
                task_index=arguments.task_index,
                registration_path=arguments.registration,
                arm_o_gate_path=arguments.arm_o_gate,
                gate_public_key_path=arguments.gate_public_key_file,
                output_root=arguments.output_root,
                executor=executor,
            )
        elif arguments.command == "finalize-inspection-only":
            run_inspection_only_finalizer(
                registration_path=arguments.registration,
                output_root=arguments.output_root,
            )
        else:  # pragma: no cover - argparse enforces the closed command set
            raise ContractError("unsupported driver subcommand")
    except (ContractError, OSError, ValueError) as exc:
        print(f"STAGEB_DRIVER_FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
