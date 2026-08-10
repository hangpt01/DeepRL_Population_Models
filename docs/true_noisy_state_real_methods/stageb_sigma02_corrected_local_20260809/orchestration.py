"""Fail-closed Arm O gate, issued Arm T capability, and inspection-only finalizer."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import (
    CanonicalArtifact,
    FIXTURE_ROW_COUNT,
    REQUIRED_COMPONENTS,
    SURROGATE_METHODS,
    require_m3_arm_o_parity,
    validate_complete_artifact_bundle,
)
from .common import (
    ContractError,
    canonical_json_bytes,
    require_exact_keys,
    require_git_sha,
    require_sha256,
    sha256_bytes,
    strict_json_loads,
)
from .boundary import validate_task_information_boundary_receipt
from .diagnostics import classify_activity, validate_arm_task_diagnostics
from .evidence import REGISTERED_EPISODE_IDS
from .registration import (
    CELLS,
    METHODS,
    FrozenRegistration,
    validate_interpreter_identity_receipt,
)
from .registration import freeze_registration_bundle
from .publication import load_success_receipt
from .real_artifacts import revalidate_frozen_object_parity, scientific_component_for_method


EXPECTED_ARM_TASKS = tuple((cell, method) for cell in CELLS for method in METHODS)
CORRECTED_TEST_COUNT = 227
ARTIFACT_EVIDENCE_V2 = "corrected_stageb_artifact_bundle_evidence_v2"
ARTIFACT_EVIDENCE_V3 = "corrected_stageb_artifact_bundle_evidence_v3"
ARTIFACT_EVIDENCE_V4 = "corrected_stageb_artifact_bundle_evidence_v4"
_TOKEN_ISSUER = object()
_TOKEN_SECRET = secrets.token_bytes(32)
_ED_Q = 2**255 - 19
_ED_L = 2**252 + 27742317777372353535851937790883648493
_ED_D = (-121665 * pow(121666, _ED_Q - 2, _ED_Q)) % _ED_Q


def require_artifact_evidence_version(
    value: Mapping[str, Any], *, expected: str = ARTIFACT_EVIDENCE_V4
) -> None:
    if expected not in {ARTIFACT_EVIDENCE_V2, ARTIFACT_EVIDENCE_V3, ARTIFACT_EVIDENCE_V4}:
        raise ContractError("unsupported artifact evidence contract version")
    if value.get("schema_version") != expected:
        raise ContractError(f"artifact evidence schema mismatch; expected {expected}")


_ED_I = pow(2, (_ED_Q - 1) // 4, _ED_Q)


def _ed_xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_ED_D * y * y + 1, _ED_Q - 2, _ED_Q) % _ED_Q
    x = pow(xx, (_ED_Q + 3) // 8, _ED_Q)
    if (x * x - xx) % _ED_Q != 0:
        x = x * _ED_I % _ED_Q
    if x & 1:
        x = _ED_Q - x
    return x


_ED_B_Y = 4 * pow(5, _ED_Q - 2, _ED_Q) % _ED_Q
_ED_B = (_ed_xrecover(_ED_B_Y), _ED_B_Y)


def _ed_add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = left
    x2, y2 = right
    product = _ED_D * x1 * x2 * y1 * y2 % _ED_Q
    x3 = (x1 * y2 + x2 * y1) * pow(1 + product, _ED_Q - 2, _ED_Q) % _ED_Q
    y3 = (y1 * y2 + x1 * x2) * pow(1 - product, _ED_Q - 2, _ED_Q) % _ED_Q
    return x3, y3


def _ed_scalar_mult(point: tuple[int, int], scalar: int) -> tuple[int, int]:
    result = (0, 1)
    addend = point
    while scalar:
        if scalar & 1:
            result = _ed_add(result, addend)
        addend = _ed_add(addend, addend)
        scalar >>= 1
    return result


def _ed_encode(point: tuple[int, int]) -> bytes:
    x, y = point
    encoded = y | ((x & 1) << 255)
    return encoded.to_bytes(32, "little")


def _ed_decode(payload: bytes) -> tuple[int, int]:
    if len(payload) != 32:
        raise ContractError("Ed25519 point must contain exactly 32 bytes")
    encoded = int.from_bytes(payload, "little")
    y = encoded & ((1 << 255) - 1)
    if y >= _ED_Q:
        raise ContractError("Ed25519 point has a noncanonical coordinate")
    x = _ed_xrecover(y)
    if (x & 1) != (encoded >> 255):
        x = _ED_Q - x
    if (-x * x + y * y - 1 - _ED_D * x * x * y * y) % _ED_Q != 0:
        raise ContractError("Ed25519 point is not on the registered curve")
    return x, y


def _ed_hash(payload: bytes) -> int:
    return int.from_bytes(hashlib.sha512(payload).digest(), "little")


def _ed_prime_subgroup(point: tuple[int, int]) -> bool:
    return point != (0, 1) and _ed_scalar_mult(point, _ED_L) == (0, 1)


def derive_gate_verification_key(signing_seed: bytes) -> bytes:
    if not isinstance(signing_seed, bytes) or len(signing_seed) != 32:
        raise ContractError("Ed25519 gate signing seed must be exactly 32 bytes")
    digest = hashlib.sha512(signing_seed).digest()
    scalar = int.from_bytes(digest[:32], "little")
    scalar &= (1 << 254) - 8
    scalar |= 1 << 254
    return _ed_encode(_ed_scalar_mult(_ED_B, scalar))


def _ed_sign(seed: bytes, message: bytes) -> bytes:
    public = derive_gate_verification_key(seed)
    digest = hashlib.sha512(seed).digest()
    scalar = int.from_bytes(digest[:32], "little")
    scalar &= (1 << 254) - 8
    scalar |= 1 << 254
    nonce = _ed_hash(digest[32:] + message) % _ED_L
    encoded_r = _ed_encode(_ed_scalar_mult(_ED_B, nonce))
    challenge = _ed_hash(encoded_r + public + message) % _ED_L
    encoded_s = ((nonce + challenge * scalar) % _ED_L).to_bytes(32, "little")
    return encoded_r + encoded_s


def _ed_verify(public: bytes, message: bytes, signature: bytes) -> bool:
    if len(public) != 32 or len(signature) != 64:
        return False
    try:
        public_point = _ed_decode(public)
        r_point = _ed_decode(signature[:32])
    except ContractError:
        return False
    if not _ed_prime_subgroup(public_point) or not _ed_prime_subgroup(r_point):
        return False
    scalar = int.from_bytes(signature[32:], "little")
    if scalar >= _ED_L:
        return False
    challenge = _ed_hash(signature[:32] + public + message) % _ED_L
    return _ed_encode(_ed_scalar_mult(_ED_B, scalar)) == _ed_encode(
        _ed_add(r_point, _ed_scalar_mult(public_point, challenge))
    )


def _canonical_receipt(payload: bytes, label: str) -> Mapping[str, Any]:
    value = strict_json_loads(payload)
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    if canonical_json_bytes(value) != payload:
        raise ContractError(f"{label} must use canonical JSON bytes")
    return value


class ArmTGateToken:
    """Non-serializable process capability issued only by the complete Arm O gate."""

    __slots__ = (
        "_registration_sha256",
        "_gate_receipt_sha256",
        "_gate_payload",
        "_architecture",
        "_seal",
    )

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "ArmTGateToken":
        raise ContractError("ArmTGateToken capabilities cannot be constructed directly")

    @classmethod
    def _issue(
        cls,
        registration_sha256: str,
        gate_payload: bytes,
        architecture: str,
        *,
        issuer: object,
    ) -> "ArmTGateToken":
        if issuer is not _TOKEN_ISSUER:
            raise ContractError("invalid Arm T token issuer")
        instance = object.__new__(cls)
        gate_receipt_sha256 = sha256_bytes(gate_payload)
        object.__setattr__(instance, "_registration_sha256", registration_sha256)
        object.__setattr__(instance, "_gate_receipt_sha256", gate_receipt_sha256)
        object.__setattr__(instance, "_gate_payload", gate_payload)
        object.__setattr__(instance, "_architecture", architecture)
        object.__setattr__(
            instance,
            "_seal",
            _token_seal(registration_sha256, gate_receipt_sha256, architecture),
        )
        return instance

    @property
    def registration_sha256(self) -> str:
        return self._registration_sha256

    @property
    def gate_receipt_sha256(self) -> str:
        return self._gate_receipt_sha256

    @property
    def architecture(self) -> str:
        return self._architecture

    def validate(self) -> None:
        try:
            registration_sha256 = self._registration_sha256
            gate_receipt_sha256 = self._gate_receipt_sha256
            gate_payload = self._gate_payload
            architecture = self._architecture
            seal = self._seal
        except AttributeError as exc:
            raise ContractError("Arm T token is not an issued process capability") from exc
        require_sha256(registration_sha256, "Arm T registration hash")
        require_sha256(gate_receipt_sha256, "Arm T gate receipt")
        if sha256_bytes(gate_payload) != gate_receipt_sha256:
            raise ContractError("Arm T token is not an issued process capability: payload hash")
        if architecture != "Intel Xeon Platinum 8452Y":
            raise ContractError("Arm T gate architecture mismatch")
        expected = _token_seal(
            registration_sha256,
            gate_receipt_sha256,
            architecture,
        )
        if not hmac.compare_digest(seal, expected):
            raise ContractError("Arm T token is not an issued process capability")

    def __reduce__(self) -> Any:
        raise TypeError("ArmTGateToken process capabilities cannot be serialized")


def _token_seal(registration_sha256: str, gate_receipt_sha256: str, architecture: str) -> bytes:
    message = b"\0".join(
        (
            registration_sha256.encode("ascii"),
            gate_receipt_sha256.encode("ascii"),
            architecture.encode("utf-8"),
        )
    )
    return hmac.digest(_TOKEN_SECRET, message, "sha256")


def _validate_task_receipt(
    payload: bytes,
    *,
    registration_sha: str,
    expected_task: Mapping[str, Any],
    evidence_root: Path,
    frozen_registration: FrozenRegistration,
) -> tuple[str, str, Mapping[str, str]]:
    receipt = _canonical_receipt(payload, "Arm O task receipt")
    required = {
        "schema_version",
        "task_index",
        "arm",
        "cell",
        "method",
        "registration_sha256",
        "task_manifest_entry_sha256",
        "config_sha256",
        "dataset_sha256",
        "artifact_plan_sha256",
        "evaluation_identity_sha256",
        "artifact_bundle_sha256",
        "diagnostics_sha256",
        "information_boundary_sha256",
        "publication_manifest_sha256",
        "evidence",
        "cpu_profile",
        "interpreter_identity",
        "result",
    }
    require_exact_keys(receipt, required, "Arm O task receipt")
    expected_scalars = {
        "schema_version": "corrected_stageb_arm_o_task_receipt_v3",
        "task_index": expected_task["task_index"],
        "arm": "O",
        "cell": expected_task["cell"],
        "method": expected_task["method"],
        "registration_sha256": registration_sha,
        "task_manifest_entry_sha256": sha256_bytes(canonical_json_bytes(expected_task)),
        "config_sha256": expected_task["config_sha256"],
        "dataset_sha256": expected_task["dataset_sha256"],
        "artifact_plan_sha256": expected_task["artifact_plan_sha256"],
        "evaluation_identity_sha256": expected_task["evaluation_identity_sha256"],
        "cpu_profile": "Intel Xeon Platinum 8452Y / xenon-8452Y / one CPU",
        "result": "PASS",
    }
    for field, expected in expected_scalars.items():
        if receipt[field] != expected:
            raise ContractError(f"Arm O task receipt binding mismatch: {field}")
    for field in (
        "artifact_bundle_sha256",
        "diagnostics_sha256",
        "information_boundary_sha256",
        "publication_manifest_sha256",
    ):
        require_sha256(receipt[field], f"Arm O task receipt {field}")
    validate_interpreter_identity_receipt(
        receipt["interpreter_identity"],
        frozen_registration,
        command="arm-o",
        arm="O",
        task_index=expected_task["task_index"],
    )
    evidence = receipt["evidence"]
    require_exact_keys(
        evidence,
        {"artifact_bundle", "diagnostics", "information_boundary", "publication_success"},
        "Arm O task evidence references",
    )
    loaded: dict[str, tuple[Path, bytes]] = {}
    for kind, hash_field in (
        ("artifact_bundle", "artifact_bundle_sha256"),
        ("diagnostics", "diagnostics_sha256"),
        ("information_boundary", "information_boundary_sha256"),
        ("publication_success", "publication_manifest_sha256"),
    ):
        path, evidence_payload = _load_evidence_reference(evidence_root, evidence[kind], label=kind)
        if sha256_bytes(evidence_payload) != receipt[hash_field]:
            raise ContractError(f"Arm O {kind} evidence hash does not match the task receipt")
        loaded[kind] = (path, evidence_payload)
    for kind in ("diagnostics", "information_boundary"):
        _validate_bound_task_evidence(
            loaded[kind][1],
            kind=kind,
            registration_sha=registration_sha,
            expected_task=expected_task,
        )
    publication_path = loaded["publication_success"][0]
    publication = load_success_receipt(publication_path)
    component_hashes = _validate_artifact_bundle_evidence(
        loaded["artifact_bundle"][1],
        registration_sha=registration_sha,
        expected_task=expected_task,
        evidence_root=evidence_root,
        publication_path=publication_path,
        publication=publication,
    )
    validation = publication.get("validation")
    if not isinstance(validation, Mapping):
        raise ContractError("published task has no structured validation receipt")
    require_exact_keys(
        validation,
        {
            "schema_version",
            "registration_sha256",
            "task_index",
            "arm",
            "cell",
            "method",
            "artifact_bundle_sha256",
            "diagnostics_sha256",
            "information_boundary_sha256",
            "interpreter_identity",
            "result",
        },
        "published task validation",
    )
    expected_validation = {
        "schema_version": "corrected_stageb_task_publication_validation_v2",
        "registration_sha256": registration_sha,
        "task_index": expected_task["task_index"],
        "arm": "O",
        "cell": expected_task["cell"],
        "method": expected_task["method"],
        "artifact_bundle_sha256": receipt["artifact_bundle_sha256"],
        "diagnostics_sha256": receipt["diagnostics_sha256"],
        "information_boundary_sha256": receipt["information_boundary_sha256"],
        "interpreter_identity": receipt["interpreter_identity"],
        "result": "PASS",
    }
    if validation != expected_validation:
        raise ContractError("published task validation is not registration-bound")
    for kind in ("diagnostics", "information_boundary"):
        path = loaded[kind][0]
        if path.parent != publication_path.parent:
            raise ContractError("task evidence is not contained in its published task tree")
        relative = path.name
        if (
            publication["files"].get(relative)
            != receipt[
                "diagnostics_sha256" if kind == "diagnostics" else "information_boundary_sha256"
            ]
        ):
            raise ContractError("published full-tree manifest does not cover task evidence")
    return sha256_bytes(payload), receipt["artifact_bundle_sha256"], component_hashes


def _validate_artifact_bundle_evidence(
    payload: bytes,
    *,
    registration_sha: str,
    expected_task: Mapping[str, Any],
    evidence_root: Path,
    publication_path: Path,
    publication: Mapping[str, Any],
) -> Mapping[str, str]:
    value = _canonical_receipt(payload, "Arm O fitted-artifact bundle evidence")
    require_artifact_evidence_version(value)
    require_exact_keys(
        value,
        {
            "schema_version",
            "registration_sha256",
            "task_index",
            "arm",
            "cell",
            "method",
            "artifact_plan_sha256",
            "prediction_fixtures",
            "fixture_row_count",
            "fixture_manifest_sha256",
            "components",
            "component_hashes",
            "bundle_sha256",
            "fresh_reload_parity",
            "frozen_object_artifact",
            "frozen_object_parity",
            "frozen_object_binding",
        },
        "Arm O fitted-artifact bundle evidence",
    )
    expected = {
        "schema_version": ARTIFACT_EVIDENCE_V4,
        "registration_sha256": registration_sha,
        "task_index": expected_task["task_index"],
        "arm": "O",
        "cell": expected_task["cell"],
        "method": expected_task["method"],
        "artifact_plan_sha256": expected_task["artifact_plan_sha256"],
        "fresh_reload_parity": True,
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise ContractError(f"fitted-artifact evidence binding mismatch: {field}")
    if value["fixture_row_count"] != FIXTURE_ROW_COUNT:
        raise ContractError("fitted-artifact evidence fixture row count mismatch")
    require_sha256(value["fixture_manifest_sha256"], "prediction fixture manifest")
    raw_fixtures = value["prediction_fixtures"]
    if not isinstance(raw_fixtures, Mapping):
        raise ContractError("fitted-artifact evidence prediction fixtures are invalid")
    fixtures: dict[str, np.ndarray] = {}
    for component, raw_fixture in raw_fixtures.items():
        if not isinstance(component, str) or not component:
            raise ContractError("fitted-artifact evidence fixture component is invalid")
        fixture = np.asarray(raw_fixture)
        if fixture.dtype != np.dtype("float64") or fixture.ndim != 2:
            raise ContractError("fitted-artifact evidence prediction fixture is invalid")
        fixtures[component] = fixture
    components = value["components"]
    component_hashes = value["component_hashes"]
    if not isinstance(components, Mapping) or not isinstance(component_hashes, Mapping):
        raise ContractError("fitted-artifact component references/hashes are missing")
    payloads: dict[str, bytes] = {}
    observed_hashes: dict[str, str] = {}
    for component, reference in components.items():
        if not isinstance(component, str) or not component:
            raise ContractError("fitted-artifact component name is invalid")
        path, component_payload = _load_evidence_reference(
            evidence_root, reference, label=f"fitted artifact {component}"
        )
        if path.parent != publication_path.parent:
            raise ContractError("fitted-artifact component is outside its published task tree")
        digest = sha256_bytes(component_payload)
        if publication["files"].get(path.name) != digest:
            raise ContractError("published full-tree manifest omits a fitted artifact")
        payloads[component] = component_payload
        observed_hashes[component] = digest
    if dict(component_hashes) != observed_hashes:
        raise ContractError("fitted-artifact component hash map does not match referenced bytes")
    validated = validate_complete_artifact_bundle(
        expected_task["method"],
        payloads,
        prediction_fixtures=fixtures,
        expected_component_hashes=observed_hashes,
    )
    if (
        validated.bundle_sha256 != value["bundle_sha256"]
        or validated.fixture_manifest_sha256 != value["fixture_manifest_sha256"]
        or validated.reload_parity is not True
    ):
        raise ContractError("fitted-artifact bundle content/parity validation failed")
    for component in {"reward_surrogate", "ricker_fit_cache"} & set(payloads):
        artifact = CanonicalArtifact.from_bytes(payloads[component])
        if artifact.state.get("cell") != expected_task["cell"]:
            raise ContractError(f"fitted artifact is cross-cell rebound: {component}")
        if (
            component == "reward_surrogate"
            and artifact.state.get("source_public_view_sha256") != expected_task["dataset_sha256"]
        ):
            raise ContractError("reward surrogate is not bound to the registered public view")
    frozen_path, frozen_payload = _load_evidence_reference(
        evidence_root,
        value["frozen_object_artifact"],
        label="real frozen fitted object",
    )
    parity_path, parity_payload = _load_evidence_reference(
        evidence_root,
        value["frozen_object_parity"],
        label="real frozen fitted-object parity",
    )
    for path, evidence_payload, label in (
        (frozen_path, frozen_payload, "real frozen fitted object"),
        (parity_path, parity_payload, "real frozen fitted-object parity"),
    ):
        if path.parent != publication_path.parent:
            raise ContractError(f"{label} is outside its published task tree")
        if publication["files"].get(path.name) != sha256_bytes(evidence_payload):
            raise ContractError(f"published full-tree manifest omits {label}")
    frozen_component = scientific_component_for_method(expected_task["method"])
    binding = value["frozen_object_binding"]
    if not isinstance(binding, Mapping):
        raise ContractError("real frozen fitted-object binding must be an object")
    expected_binding = {
        "schema_version": "corrected_stageb_frozen_object_task_binding_v1",
        "registration_sha256": registration_sha,
        "task_index": expected_task["task_index"],
        "arm": "O",
        "cell": expected_task["cell"],
        "method": expected_task["method"],
        "public_view_sha256": expected_task["dataset_sha256"],
        "frozen_object_component": frozen_component,
        "frozen_object_sha256": sha256_bytes(frozen_payload),
        "parity_receipt_sha256": sha256_bytes(parity_payload),
        "covered_components": sorted(REQUIRED_COMPONENTS[expected_task["method"]]),
        "logical_component_hashes": observed_hashes,
        "result": "PASS",
    }
    require_exact_keys(binding, set(expected_binding), "real frozen fitted-object task binding")
    if dict(binding) != expected_binding:
        raise ContractError("real frozen fitted-object task binding mismatch")
    parity = _canonical_receipt(parity_payload, "real frozen fitted-object parity")
    revalidate_frozen_object_parity(
        frozen_payload,
        parity,
        expected_component=frozen_component,
        repository_root=Path(__file__).resolve().parents[3],
    )
    return observed_hashes


def _load_evidence_reference(
    root: Path, reference: Mapping[str, Any], *, label: str
) -> tuple[Path, bytes]:
    if not isinstance(reference, Mapping):
        raise ContractError(f"{label} evidence reference must be an object")
    require_exact_keys(reference, {"relative_path", "sha256"}, f"{label} evidence reference")
    relative = reference["relative_path"]
    require_sha256(reference["sha256"], f"{label} evidence reference hash")
    if not isinstance(relative, str) or not relative:
        raise ContractError(f"{label} evidence path must be nonempty text")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ContractError(f"{label} evidence path escapes the evidence root")
    root_real = Path(root).resolve(strict=True)
    candidate = root_real.joinpath(*pure.parts)
    current = candidate
    while current != root_real:
        if current.is_symlink():
            raise ContractError(f"{label} evidence has a symlinked path component")
        current = current.parent
    if not candidate.is_file() or candidate.is_symlink():
        raise ContractError(f"{label} evidence file is missing")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root_real):
        raise ContractError(f"{label} evidence resolves outside its root")
    payload = candidate.read_bytes()
    if sha256_bytes(payload) != reference["sha256"]:
        raise ContractError(f"{label} referenced evidence content hash mismatch")
    return candidate, payload


def _validate_bound_task_evidence(
    payload: bytes,
    *,
    kind: str,
    registration_sha: str,
    expected_task: Mapping[str, Any],
) -> None:
    value = _canonical_receipt(payload, f"Arm O {kind} evidence")
    require_exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "registration_sha256",
            "task_index",
            "arm",
            "cell",
            "method",
            "receipts",
        },
        f"Arm O {kind} evidence",
    )
    if value["schema_version"] != "corrected_stageb_bound_task_evidence_v1":
        raise ContractError(f"Arm O {kind} evidence schema mismatch")
    expected = {
        "kind": kind,
        "registration_sha256": registration_sha,
        "task_index": expected_task["task_index"],
        "arm": "O",
        "cell": expected_task["cell"],
        "method": expected_task["method"],
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise ContractError(f"Arm O {kind} evidence binding mismatch: {field}")
    receipts = value["receipts"]
    if not isinstance(receipts, Mapping) or not receipts:
        raise ContractError(f"Arm O {kind} evidence receipts are missing")
    if kind == "diagnostics":
        if expected_task["method"] == "ensemble_value_disagreement_pessimism":
            require_exact_keys(
                receipts,
                {"transition", "activity"},
                "registered EVD task diagnostics receipts",
            )
            transition = receipts["transition"]
            expected_transition = {
                "schema_version": "corrected_stageb_transition_not_applicable_v1",
                "registration_sha256": registration_sha,
                "method": expected_task["method"],
                "cell": expected_task["cell"],
                "arm": "O",
                "applicability": "DEFINITIONALLY_NOT_APPLICABLE",
                "source_backed_reason": "EVD has no fitted transition model",
                "artifact_hashes": [],
            }
            if transition != expected_transition:
                raise ContractError("EVD transition applicability receipt mismatch")
            activities = receipts["activity"]
            if not isinstance(activities, list) or len(activities) != len(REGISTERED_EPISODE_IDS):
                raise ContractError("EVD activity evidence must cover every episode")
            for episode_id, receipt in zip(REGISTERED_EPISODE_IDS, activities):
                if not isinstance(receipt, Mapping):
                    raise ContractError("EVD activity receipt must be an object")
                rebuilt = classify_activity(
                    receipt.get("actions", ()),
                    registration_sha256=registration_sha,
                    method=expected_task["method"],
                    cell=expected_task["cell"],
                    arm="O",
                    episode_id=episode_id,
                )
                if dict(receipt) != rebuilt:
                    raise ContractError("EVD activity receipt is internally inconsistent")
            return
        validate_arm_task_diagnostics(
            receipts,
            registration_sha256=registration_sha,
            method=expected_task["method"],
            cell=expected_task["cell"],
            arm="O",
        )
        return
    if kind == "information_boundary":
        require_exact_keys(
            receipts,
            {"runtime_information_boundary"},
            "registered information-boundary receipts",
        )
        validate_task_information_boundary_receipt(
            receipts["runtime_information_boundary"],
            registration_sha256=registration_sha,
            method=expected_task["method"],
            cell=expected_task["cell"],
            arm="O",
        )
        return
    raise ContractError("unregistered Arm O task evidence kind")


def _validate_test_receipts(
    inherited_payload: bytes,
    corrected_payload: bytes,
    *,
    expected_source_manifest_sha256: str,
) -> tuple[str, str]:
    inherited = _canonical_receipt(inherited_payload, "inherited test receipt")
    require_exact_keys(
        inherited,
        {
            "schema_version",
            "i2a_collected",
            "i2a_passed",
            "fasttrack_collected",
            "fasttrack_passed",
            "skipped",
            "result",
        },
        "inherited test receipt",
    )
    if inherited != {
        "schema_version": "corrected_stageb_inherited_tests_v1",
        "i2a_collected": 38,
        "i2a_passed": 38,
        "fasttrack_collected": 15,
        "fasttrack_passed": 15,
        "skipped": 0,
        "result": "PASS",
    }:
        raise ContractError("registered inherited test total 53 did not pass exactly")
    corrected = _canonical_receipt(corrected_payload, "corrected test receipt")
    require_exact_keys(
        corrected,
        {
            "schema_version",
            "collected",
            "passed",
            "skipped",
            "source_test_manifest_sha256",
            "result",
        },
        "corrected test receipt",
    )
    if corrected["schema_version"] != "corrected_stageb_corrected_tests_v1":
        raise ContractError("corrected test receipt schema mismatch")
    if (
        corrected["collected"] != CORRECTED_TEST_COUNT
        or corrected["passed"] != CORRECTED_TEST_COUNT
        or corrected["skipped"] != 0
        or corrected["result"] != "PASS"
        or corrected["source_test_manifest_sha256"] != expected_source_manifest_sha256
    ):
        raise ContractError("corrected synthetic tests did not pass completely")
    require_sha256(corrected["source_test_manifest_sha256"], "corrected test manifest")
    return sha256_bytes(inherited_payload), sha256_bytes(corrected_payload)


def _validate_parity_receipt(
    payload: bytes,
    *,
    registration_sha: str,
    code_hashes: Mapping[str, Any],
    expected_artifact_gate_sha256: str,
    frozen_registration: FrozenRegistration,
) -> str:
    parity = _canonical_receipt(payload, "Arm O parity receipt")
    require_exact_keys(
        parity,
        {
            "schema_version",
            "architecture",
            "registration_sha256",
            "git_commit_sha",
            "per_episode_action_event_parity",
            "accepted_artifact_gate_sha256",
            "verified_hashes",
            "interpreter_identity",
        },
        "Arm O parity receipt frozen registration binding",
    )
    if parity["schema_version"] != "corrected_stageb_arm_o_parity_v2":
        raise ContractError("Arm O parity receipt schema mismatch")
    if parity["registration_sha256"] != registration_sha:
        raise ContractError("Arm O parity registration binding mismatch")
    require_git_sha(parity["git_commit_sha"], "Arm O parity git commit")
    if parity["git_commit_sha"] != code_hashes["git_commit_sha"]:
        raise ContractError("Arm O parity commit mismatch")
    expected_hashes = {
        field: code_hashes[field]
        for field in code_hashes
        if field not in {"schema_version", "git_commit_sha"}
    }
    if parity["verified_hashes"] != expected_hashes:
        raise ContractError("Arm O parity hashes do not match the frozen registration")
    if parity["accepted_artifact_gate_sha256"] != expected_artifact_gate_sha256:
        raise ContractError("Arm O artifact gate is not derived from frozen task artifacts")
    validate_interpreter_identity_receipt(
        parity["interpreter_identity"],
        frozen_registration,
        command="arm-o-gate",
    )
    require_m3_arm_o_parity(parity)
    return sha256_bytes(payload)


def validate_arm_o_gate(
    *,
    frozen_registration: FrozenRegistration,
    task_receipts: Sequence[bytes],
    inherited_test_receipt: bytes,
    corrected_test_receipt: bytes,
    parity_receipt: bytes,
    evidence_root: Path,
) -> ArmTGateToken:
    """Validate canonical immutable receipts against the frozen registration."""

    if not isinstance(frozen_registration, FrozenRegistration):
        raise ContractError("Arm O gate requires an issued FrozenRegistration")
    registration_sha = frozen_registration.authorize_return_path()
    bundle = frozen_registration.bundle()
    manifest_tasks = bundle["task_manifest"]["tasks"]
    arm_o_tasks = [task for task in manifest_tasks if task["arm"] == "O"]
    if len(task_receipts) != len(EXPECTED_ARM_TASKS) or len(arm_o_tasks) != len(EXPECTED_ARM_TASKS):
        raise ContractError("Arm O gate requires all twelve immutable task receipts")
    validated_tasks = [
        _validate_task_receipt(
            payload,
            registration_sha=registration_sha,
            expected_task=task,
            evidence_root=evidence_root,
            frozen_registration=frozen_registration,
        )
        for payload, task in zip(task_receipts, arm_o_tasks)
    ]
    task_receipt_hashes = [item[0] for item in validated_tasks]
    artifact_bundle_hashes = [item[1] for item in validated_tasks]
    component_hash_maps = [item[2] for item in validated_tasks]
    for cell in CELLS:
        surrogate_hashes = []
        for task, component_hashes in zip(arm_o_tasks, component_hash_maps):
            if task["cell"] == cell and task["method"] in SURROGATE_METHODS:
                surrogate_hash = component_hashes.get("reward_surrogate")
                if surrogate_hash is None:
                    raise ContractError("Arm O task omitted the matched reward surrogate")
                surrogate_hashes.append(surrogate_hash)
        if len(surrogate_hashes) != len(SURROGATE_METHODS) or len(set(surrogate_hashes)) != 1:
            raise ContractError("Arm O methods did not consume identical cell surrogate bytes")
    inherited_sha, corrected_sha = _validate_test_receipts(
        inherited_test_receipt,
        corrected_test_receipt,
        expected_source_manifest_sha256=bundle["code_configuration_hashes"][
            "source_manifest_sha256"
        ],
    )
    parity_sha = _validate_parity_receipt(
        parity_receipt,
        registration_sha=registration_sha,
        code_hashes=bundle["code_configuration_hashes"],
        expected_artifact_gate_sha256=sha256_bytes(
            canonical_json_bytes(
                [
                    {
                        "task_index": task["task_index"],
                        "cell": task["cell"],
                        "method": task["method"],
                        "artifact_plan_sha256": task["artifact_plan_sha256"],
                        "artifact_bundle_evidence_sha256": artifact_bundle_sha,
                    }
                    for task, artifact_bundle_sha in zip(arm_o_tasks, artifact_bundle_hashes)
                ]
            )
        ),
        frozen_registration=frozen_registration,
    )
    gate_payload = canonical_json_bytes(
        {
            "schema_version": "corrected_stageb_arm_o_gate_v2",
            "registration_sha256": registration_sha,
            "task_receipt_sha256": task_receipt_hashes,
            "inherited_test_receipt_sha256": inherited_sha,
            "corrected_test_receipt_sha256": corrected_sha,
            "parity_receipt_sha256": parity_sha,
            "result": "PASS",
            "rebaseline_permitted": False,
            "repair_permitted": False,
            "automatic_retry_permitted": False,
        }
    )
    return ArmTGateToken._issue(
        registration_sha,
        gate_payload,
        "Intel Xeon Platinum 8452Y",
        issuer=_TOKEN_ISSUER,
    )


def load_gate_signing_key(path: Path) -> bytes:
    """Load an Ed25519 private seed from a real owner-only regular file."""

    key_path = Path(path)
    if not key_path.is_absolute() or key_path.is_symlink() or not key_path.is_file():
        raise ContractError("gate signing key must be an absolute real regular file")
    metadata = key_path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
        raise ContractError("gate signing key permissions must exclude group/other access")
    key = key_path.read_bytes()
    if len(key) != 32:
        raise ContractError("Ed25519 gate signing seed must contain exactly 32 bytes")
    return key


def load_gate_verification_key(path: Path) -> bytes:
    """Load the public Ed25519 verification key available to Arm T."""

    key_path = Path(path)
    if not key_path.is_absolute() or key_path.is_symlink() or not key_path.is_file():
        raise ContractError("gate verification key must be an absolute real regular file")
    key = key_path.read_bytes()
    if len(key) != 32:
        raise ContractError("Ed25519 gate verification key must contain exactly 32 bytes")
    if not _ed_prime_subgroup(_ed_decode(key)):
        raise ContractError("Ed25519 gate verification key is not in the prime subgroup")
    return key


def export_arm_o_gate_receipt(
    token: ArmTGateToken,
    registration: FrozenRegistration,
    *,
    signing_key: bytes,
) -> bytes:
    """Create a canonical authenticated receipt for a later Arm T process."""

    authorize_arm_t_task(token, registration)
    verification_key = derive_gate_verification_key(signing_key)
    unsigned = {
        "schema_version": "corrected_stageb_cross_process_arm_o_gate_v1",
        "algorithm": "Ed25519",
        "key_id_sha256": sha256_bytes(verification_key),
        "registration_sha256": registration.sha256,
        "registration_payload_b64": base64.b64encode(registration.payload).decode("ascii"),
        "gate_payload_sha256": token.gate_receipt_sha256,
        "gate_payload_b64": base64.b64encode(token._gate_payload).decode("ascii"),
        "architecture": token.architecture,
    }
    signature = _ed_sign(signing_key, canonical_json_bytes(unsigned))
    return canonical_json_bytes(
        {**unsigned, "signature_b64": base64.b64encode(signature).decode("ascii")}
    )


def verify_arm_o_gate_receipt(
    payload: bytes,
    *,
    verification_key: bytes,
    repository_root: Path | None = None,
) -> tuple[ArmTGateToken, FrozenRegistration]:
    """Authenticate a completed Arm O gate and issue a new process-local Arm T token."""

    if not isinstance(verification_key, bytes) or len(verification_key) != 32:
        raise ContractError("cross-process gate verification key must be exactly 32 bytes")
    if not _ed_prime_subgroup(_ed_decode(verification_key)):
        raise ContractError("cross-process gate verification key is not in the prime subgroup")
    value = _canonical_receipt(payload, "cross-process Arm O gate receipt")
    required = {
        "schema_version",
        "algorithm",
        "key_id_sha256",
        "registration_sha256",
        "registration_payload_b64",
        "gate_payload_sha256",
        "gate_payload_b64",
        "architecture",
        "signature_b64",
    }
    require_exact_keys(value, required, "cross-process Arm O gate receipt")
    unsigned = {key: value[key] for key in required - {"signature_b64"}}
    if (
        value["schema_version"] != "corrected_stageb_cross_process_arm_o_gate_v1"
        or value["algorithm"] != "Ed25519"
        or value["key_id_sha256"] != sha256_bytes(verification_key)
        or value["architecture"] != "Intel Xeon Platinum 8452Y"
    ):
        raise ContractError("cross-process Arm O gate controlling metadata mismatch")
    try:
        signature = base64.b64decode(value["signature_b64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ContractError("cross-process Arm O gate signature encoding is invalid") from exc
    if not _ed_verify(verification_key, canonical_json_bytes(unsigned), signature):
        raise ContractError("cross-process Arm O gate signature mismatch")
    try:
        registration_payload = base64.b64decode(value["registration_payload_b64"], validate=True)
        gate_payload = base64.b64decode(value["gate_payload_b64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ContractError("cross-process Arm O gate base64 payload is invalid") from exc
    if (
        sha256_bytes(registration_payload) != value["registration_sha256"]
        or sha256_bytes(gate_payload) != value["gate_payload_sha256"]
    ):
        raise ContractError("cross-process Arm O gate embedded payload hash mismatch")
    bundle = strict_json_loads(registration_payload)
    if not isinstance(bundle, Mapping):
        raise ContractError("cross-process registration payload is not an object")
    registration = freeze_registration_bundle(bundle, repository_root=repository_root)
    if registration.sha256 != value["registration_sha256"]:
        raise ContractError("cross-process gate registration revalidation mismatch")
    gate = _canonical_receipt(gate_payload, "embedded Arm O gate payload")
    require_exact_keys(
        gate,
        {
            "schema_version",
            "registration_sha256",
            "task_receipt_sha256",
            "inherited_test_receipt_sha256",
            "corrected_test_receipt_sha256",
            "parity_receipt_sha256",
            "result",
            "rebaseline_permitted",
            "repair_permitted",
            "automatic_retry_permitted",
        },
        "embedded Arm O gate payload",
    )
    if (
        gate["schema_version"] != "corrected_stageb_arm_o_gate_v2"
        or gate["registration_sha256"] != registration.sha256
        or not isinstance(gate["task_receipt_sha256"], list)
        or len(gate["task_receipt_sha256"]) != len(EXPECTED_ARM_TASKS)
        or gate["result"] != "PASS"
        or gate["rebaseline_permitted"] is not False
        or gate["repair_permitted"] is not False
        or gate["automatic_retry_permitted"] is not False
    ):
        raise ContractError("embedded Arm O gate did not represent the completed frozen gate")
    for digest in gate["task_receipt_sha256"]:
        require_sha256(digest, "embedded Arm O task receipt")
    for field in (
        "inherited_test_receipt_sha256",
        "corrected_test_receipt_sha256",
        "parity_receipt_sha256",
    ):
        require_sha256(gate[field], f"embedded Arm O gate {field}")
    token = ArmTGateToken._issue(
        registration.sha256,
        gate_payload,
        value["architecture"],
        issuer=_TOKEN_ISSUER,
    )
    return token, registration


def authorize_arm_t_task(token: ArmTGateToken, registration: FrozenRegistration) -> None:
    if not isinstance(token, ArmTGateToken):
        raise ContractError("Arm T requires the completed Arm O gate token")
    if not isinstance(registration, FrozenRegistration):
        raise ContractError("Arm T requires an issued FrozenRegistration")
    token.validate()
    if token.registration_sha256 != registration.authorize_return_path():
        raise ContractError("Arm T gate/registration mismatch")


def inspect_only_finalizer(task_statuses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(task_statuses) != len(EXPECTED_ARM_TASKS):
        raise ContractError("finalizer requires all twelve Arm T statuses")
    required = {"task_index", "cell", "method", "status", "receipt_sha256"}
    statuses: list[str] = []
    for index, (item, expected) in enumerate(zip(task_statuses, EXPECTED_ARM_TASKS)):
        require_exact_keys(item, required, f"finalizer status[{index}]")
        if item["task_index"] != index or (item["cell"], item["method"]) != expected:
            raise ContractError("finalizer status identity mismatch")
        if item["status"] not in {"COMPLETED", "FAILED"}:
            raise ContractError("finalizer encountered a nonterminal task status")
        require_sha256(item["receipt_sha256"], "finalizer task receipt")
        statuses.append(item["status"])
    return {
        "schema_version": "corrected_stageb_finalizer_v2",
        "inspected_task_count": len(statuses),
        "completed": statuses.count("COMPLETED"),
        "failed": statuses.count("FAILED"),
        "other": 0,
        "methods_executed_by_finalizer": 0,
        "repairs_performed": 0,
        "interpretation_status": "PROVISIONAL — NOT YET INDEPENDENTLY AUDITED",
    }
