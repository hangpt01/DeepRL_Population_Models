from __future__ import annotations

from pathlib import Path

import pytest

from ..artifacts import (
    FIXTURE_ROW_COUNT,
    REQUIRED_COMPONENTS,
    deterministic_prediction_fixtures,
    validate_complete_artifact_bundle,
)
from ..boundary import build_task_information_boundary_receipt
from ..common import ContractError, canonical_json_bytes, sha256_bytes, strict_json_loads
from ..diagnostics import (
    build_arm_refplan_predictive_dispersion,
    build_arm_transition_diagnostics,
    build_realized_posterior_receipt,
    classify_activity,
)
from ..evidence import REGISTERED_EPISODE_IDS, REGISTERED_HORIZON
from ..orchestration import (
    ARTIFACT_EVIDENCE_V2,
    ARTIFACT_EVIDENCE_V3,
    ArmTGateToken,
    CORRECTED_TEST_COUNT,
    authorize_arm_t_task,
    derive_gate_verification_key,
    export_arm_o_gate_receipt,
    inspect_only_finalizer,
    load_gate_signing_key,
    load_gate_verification_key,
    require_artifact_evidence_version,
    validate_arm_o_gate,
    verify_arm_o_gate_receipt,
)
from ..publication import create_task_staging, publish_once, write_bytes_fsync
from ..real_artifacts import scientific_component_for_method
from ..registration import CELLS, METHODS, freeze_registration_bundle
from .conftest import HASHES, artifact_components
from .frozen_fixtures import frozen_object_evidence


def _bound_evidence(registration, task, kind, receipt_mutator=None):
    if kind == "diagnostics":
        member_count = (
            8
            if task["method"].startswith("plus_")
            else 1
            if task["method"].startswith("moor_")
            else 5
        )
        receipts = {
            "transition": build_arm_transition_diagnostics(
                registration_sha256=registration.sha256,
                method=task["method"],
                cell=task["cell"],
                arm="O",
                residual_sigma=[0.02 + index * 0.001 for index in range(member_count)],
                artifact_hashes=[HASHES[index % len(HASHES)] for index in range(member_count)],
            ),
            "activity": [
                classify_activity(
                    [index % 11 for index in range(REGISTERED_HORIZON)],
                    registration_sha256=registration.sha256,
                    method=task["method"],
                    cell=task["cell"],
                    arm="O",
                    episode_id=episode_id,
                )
                for episode_id in REGISTERED_EPISODE_IDS
            ],
        }
        if task["method"] == "bamcts":
            receipts["realized_posterior"] = [
                build_realized_posterior_receipt(
                    registration_sha256=registration.sha256,
                    method="bamcts",
                    cell=task["cell"],
                    arm="O",
                    episode_id=episode_id,
                    member_ids=range(5),
                    timesteps=range(REGISTERED_HORIZON + 1),
                    model_bank_sha256=HASHES[1],
                    posterior_probabilities=[[0.2] * 5] * (REGISTERED_HORIZON + 1),
                )
                for episode_id in REGISTERED_EPISODE_IDS
            ]
        if task["method"] == "refplan":
            receipts["predictive_dispersion"] = [
                build_arm_refplan_predictive_dispersion(
                    registration_sha256=registration.sha256,
                    cell=task["cell"],
                    arm="O",
                    episode_id=episode_id,
                    timesteps=range(REGISTERED_HORIZON),
                    artifact_sha256=HASHES[2],
                    values=[0.1] * REGISTERED_HORIZON,
                )
                for episode_id in REGISTERED_EPISODE_IDS
            ]
    elif kind == "information_boundary":
        hashes = [HASHES[index % len(HASHES)] for index in range(len(REGISTERED_EPISODE_IDS))]
        receipts = {
            "runtime_information_boundary": build_task_information_boundary_receipt(
                registration_sha256=registration.sha256,
                method=task["method"],
                cell=task["cell"],
                arm="O",
                feature_order_sha256=HASHES[0],
                context_sha256_before=hashes,
                context_sha256_after=hashes,
                observation_history_sha256=hashes,
                action_history_sha256=hashes,
            )
        }
    else:
        raise AssertionError(f"unsupported evidence kind: {kind}")
    if receipt_mutator is not None:
        receipts = receipt_mutator(receipts)
    return canonical_json_bytes(
        {
            "schema_version": "corrected_stageb_bound_task_evidence_v1",
            "kind": kind,
            "registration_sha256": registration.sha256,
            "task_index": task["task_index"],
            "arm": "O",
            "cell": task["cell"],
            "method": task["method"],
            "receipts": receipts,
        }
    )


def arm_o_receipts(
    registration,
    evidence_root,
    *,
    first_diagnostics_mutator=None,
    first_frozen_parity_mutator=None,
    first_artifact_evidence_mutator=None,
    canonical_only_first=False,
    missing_parity_first=False,
):
    evidence_root.mkdir()
    result = []
    for task in registration.bundle()["task_manifest"]["tasks"][:12]:
        diagnostics = _bound_evidence(
            registration,
            task,
            "diagnostics",
            first_diagnostics_mutator if task["task_index"] == 0 else None,
        )
        boundary = _bound_evidence(registration, task, "information_boundary")
        diagnostics_sha = sha256_bytes(diagnostics)
        boundary_sha = sha256_bytes(boundary)
        staging, target = create_task_staging(evidence_root, f"task-{task['task_index']}")
        write_bytes_fsync(staging / "diagnostics.json", diagnostics)
        write_bytes_fsync(staging / "information_boundary.json", boundary)
        components = artifact_components(task["method"], task["cell"])
        component_hashes = {
            component: sha256_bytes(payload) for component, payload in components.items()
        }
        component_references = {}
        for component, payload in components.items():
            filename = f"artifact-{component}.json"
            write_bytes_fsync(staging / filename, payload)
            component_references[component] = {
                "relative_path": f"task-{task['task_index']}/{filename}",
                "sha256": component_hashes[component],
            }
        prediction_fixtures = deterministic_prediction_fixtures(task["method"], components)
        artifact_validation = validate_complete_artifact_bundle(
            task["method"],
            components,
            prediction_fixtures=prediction_fixtures,
            expected_component_hashes=component_hashes,
        )
        frozen_payload, frozen_parity = frozen_object_evidence(
            task["method"],
            task["cell"],
            task["dataset_sha256"],
        )
        if first_frozen_parity_mutator is not None and task["task_index"] == 0:
            frozen_parity = canonical_json_bytes(
                first_frozen_parity_mutator(dict(strict_json_loads(frozen_parity)))
            )
        frozen_sha = sha256_bytes(frozen_payload)
        frozen_parity_sha = sha256_bytes(frozen_parity)
        write_bytes_fsync(staging / "frozen-object.json", frozen_payload)
        write_bytes_fsync(staging / "frozen-object-parity.json", frozen_parity)
        artifact_evidence_value = {
            "schema_version": "corrected_stageb_artifact_bundle_evidence_v3",
            "registration_sha256": registration.sha256,
            "task_index": task["task_index"],
            "arm": "O",
            "cell": task["cell"],
            "method": task["method"],
            "artifact_plan_sha256": task["artifact_plan_sha256"],
            "prediction_fixtures": {
                component: fixture.tolist() for component, fixture in prediction_fixtures.items()
            },
            "fixture_row_count": FIXTURE_ROW_COUNT,
            "fixture_manifest_sha256": artifact_validation.fixture_manifest_sha256,
            "components": component_references,
            "component_hashes": component_hashes,
            "bundle_sha256": artifact_validation.bundle_sha256,
            "fresh_reload_parity": True,
            "frozen_object_artifact": {
                "relative_path": f"task-{task['task_index']}/frozen-object.json",
                "sha256": frozen_sha,
            },
            "frozen_object_parity": {
                "relative_path": f"task-{task['task_index']}/frozen-object-parity.json",
                "sha256": frozen_parity_sha,
            },
            "frozen_object_binding": {
                "schema_version": "corrected_stageb_frozen_object_task_binding_v1",
                "registration_sha256": registration.sha256,
                "task_index": task["task_index"],
                "arm": "O",
                "cell": task["cell"],
                "method": task["method"],
                "public_view_sha256": task["dataset_sha256"],
                "frozen_object_component": scientific_component_for_method(task["method"]),
                "frozen_object_sha256": frozen_sha,
                "parity_receipt_sha256": frozen_parity_sha,
                "covered_components": sorted(REQUIRED_COMPONENTS[task["method"]]),
                "logical_component_hashes": component_hashes,
                "result": "PASS",
            },
        }
        if first_artifact_evidence_mutator is not None and task["task_index"] == 0:
            artifact_evidence_value = first_artifact_evidence_mutator(artifact_evidence_value)
        if canonical_only_first and task["task_index"] == 0:
            del artifact_evidence_value["frozen_object_artifact"]
            del artifact_evidence_value["frozen_object_parity"]
            del artifact_evidence_value["frozen_object_binding"]
        elif missing_parity_first and task["task_index"] == 0:
            del artifact_evidence_value["frozen_object_parity"]
        artifact_evidence = canonical_json_bytes(artifact_evidence_value)
        artifact_evidence_sha = sha256_bytes(artifact_evidence)
        write_bytes_fsync(staging / "artifact_bundle.json", artifact_evidence)
        required_files = [
            "diagnostics.json",
            "information_boundary.json",
            "artifact_bundle.json",
            "frozen-object.json",
            "frozen-object-parity.json",
            *(f"artifact-{component}.json" for component in components),
        ]
        success = publish_once(
            staging=staging,
            target=target,
            required_files=required_files,
            validator=lambda _root,
            task=task,
            a=artifact_evidence_sha,
            d=diagnostics_sha,
            b=boundary_sha: {
                "schema_version": "corrected_stageb_task_publication_validation_v2",
                "registration_sha256": registration.sha256,
                "task_index": task["task_index"],
                "arm": "O",
                "cell": task["cell"],
                "method": task["method"],
                "artifact_bundle_sha256": a,
                "diagnostics_sha256": d,
                "information_boundary_sha256": b,
                "result": "PASS",
            },
        )
        success_payload = success.read_bytes()
        result.append(
            canonical_json_bytes(
                {
                    "schema_version": "corrected_stageb_arm_o_task_receipt_v3",
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
                    "artifact_bundle_sha256": artifact_evidence_sha,
                    "diagnostics_sha256": diagnostics_sha,
                    "information_boundary_sha256": boundary_sha,
                    "publication_manifest_sha256": sha256_bytes(success_payload),
                    "evidence": {
                        "artifact_bundle": {
                            "relative_path": f"task-{task['task_index']}/artifact_bundle.json",
                            "sha256": artifact_evidence_sha,
                        },
                        "diagnostics": {
                            "relative_path": f"task-{task['task_index']}/diagnostics.json",
                            "sha256": diagnostics_sha,
                        },
                        "information_boundary": {
                            "relative_path": f"task-{task['task_index']}/information_boundary.json",
                            "sha256": boundary_sha,
                        },
                        "publication_success": {
                            "relative_path": f"task-{task['task_index']}/PUBLICATION_SUCCESS.json",
                            "sha256": sha256_bytes(success_payload),
                        },
                    },
                    "cpu_profile": "Intel Xeon Platinum 8452Y / xenon-8452Y / one CPU",
                    "result": "PASS",
                }
            )
        )
    return result


def inherited_receipt():
    return canonical_json_bytes(
        {
            "schema_version": "corrected_stageb_inherited_tests_v1",
            "i2a_collected": 38,
            "i2a_passed": 38,
            "fasttrack_collected": 15,
            "fasttrack_passed": 15,
            "skipped": 0,
            "result": "PASS",
        }
    )


def corrected_receipt(registration):
    return canonical_json_bytes(
        {
            "schema_version": "corrected_stageb_corrected_tests_v1",
            "collected": CORRECTED_TEST_COUNT,
            "passed": CORRECTED_TEST_COUNT,
            "skipped": 0,
            "source_test_manifest_sha256": registration.bundle()["code_configuration_hashes"][
                "source_manifest_sha256"
            ],
            "result": "PASS",
        }
    )


def parity_receipt(registration, task_receipts, *, parity=True):
    hashes = registration.bundle()["code_configuration_hashes"]
    arm_o_tasks = registration.bundle()["task_manifest"]["tasks"][:12]
    artifact_gate_sha = sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "task_index": task["task_index"],
                    "cell": task["cell"],
                    "method": task["method"],
                    "artifact_plan_sha256": task["artifact_plan_sha256"],
                    "artifact_bundle_evidence_sha256": strict_json_loads(receipt)[
                        "artifact_bundle_sha256"
                    ],
                }
                for task, receipt in zip(arm_o_tasks, task_receipts)
            ]
        )
    )
    return canonical_json_bytes(
        {
            "schema_version": "corrected_stageb_arm_o_parity_v2",
            "architecture": "Intel Xeon Platinum 8452Y",
            "registration_sha256": registration.sha256,
            "git_commit_sha": hashes["git_commit_sha"],
            "per_episode_action_event_parity": parity,
            "accepted_artifact_gate_sha256": artifact_gate_sha,
            "verified_hashes": {
                field: hashes[field]
                for field in hashes
                if field not in {"schema_version", "git_commit_sha"}
            },
        }
    )


def gate(registration, evidence_root):
    receipts = arm_o_receipts(registration, evidence_root)
    return validate_arm_o_gate(
        frozen_registration=registration,
        task_receipts=receipts,
        inherited_test_receipt=inherited_receipt(),
        corrected_test_receipt=corrected_receipt(registration),
        parity_receipt=parity_receipt(registration, receipts),
        evidence_root=evidence_root,
    )


def test_arm_t_requires_successful_m3_parity_artifact_gate(registration_bundle, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    token = gate(registration, tmp_path / "evidence")
    authorize_arm_t_task(token, registration)


def test_v2_and_v3_artifact_evidence_are_mutually_rejected():
    require_artifact_evidence_version({"schema_version": ARTIFACT_EVIDENCE_V3})
    with pytest.raises(ContractError, match="schema mismatch"):
        require_artifact_evidence_version({"schema_version": ARTIFACT_EVIDENCE_V2})
    with pytest.raises(ContractError, match="schema mismatch"):
        require_artifact_evidence_version(
            {"schema_version": ARTIFACT_EVIDENCE_V3}, expected=ARTIFACT_EVIDENCE_V2
        )


def test_fixture_manifest_tampering_stops_gate(registration_bundle, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    evidence_root = tmp_path / "evidence"

    def tamper(value):
        result = dict(value)
        result["fixture_manifest_sha256"] = HASHES[0]
        return result

    receipts = arm_o_receipts(
        registration,
        evidence_root,
        first_artifact_evidence_mutator=tamper,
    )
    with pytest.raises(ContractError, match="content/parity"):
        validate_arm_o_gate(
            frozen_registration=registration,
            task_receipts=receipts,
            inherited_test_receipt=inherited_receipt(),
            corrected_test_receipt=corrected_receipt(registration),
            parity_receipt=parity_receipt(registration, receipts),
            evidence_root=evidence_root,
        )


def test_incomplete_v3_artifact_receipt_stops_gate(registration_bundle, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    evidence_root = tmp_path / "evidence"

    def remove_fixture_mapping(value):
        result = dict(value)
        del result["prediction_fixtures"]
        return result

    receipts = arm_o_receipts(
        registration,
        evidence_root,
        first_artifact_evidence_mutator=remove_fixture_mapping,
    )
    with pytest.raises(ContractError, match="key mismatch"):
        validate_arm_o_gate(
            frozen_registration=registration,
            task_receipts=receipts,
            inherited_test_receipt=inherited_receipt(),
            corrected_test_receipt=corrected_receipt(registration),
            parity_receipt=parity_receipt(registration, receipts),
            evidence_root=evidence_root,
        )


def test_cross_process_gate_receipt_reissues_only_after_authentication(
    registration_bundle, tmp_path
):
    registration = freeze_registration_bundle(registration_bundle)
    token = gate(registration, tmp_path / "evidence")
    key = b"K" * 32
    public_key = derive_gate_verification_key(key)
    receipt = export_arm_o_gate_receipt(token, registration, signing_key=key)
    loaded_token, loaded_registration = verify_arm_o_gate_receipt(
        receipt, verification_key=public_key
    )
    authorize_arm_t_task(loaded_token, loaded_registration)
    tampered = bytearray(receipt)
    tampered[-2] = ord("0") if tampered[-2] != ord("0") else ord("1")
    with pytest.raises(ContractError):
        verify_arm_o_gate_receipt(bytes(tampered), verification_key=public_key)


def test_cross_process_gate_key_file_is_owner_only_and_exact_length(tmp_path):
    key_file = tmp_path / "gate.key"
    key_file.write_bytes(b"S" * 32)
    key_file.chmod(0o600)
    assert load_gate_signing_key(key_file) == b"S" * 32
    key_file.chmod(0o644)
    with pytest.raises(ContractError, match="permissions"):
        load_gate_signing_key(key_file)
    public_file = tmp_path / "gate.pub"
    public = derive_gate_verification_key(b"S" * 32)
    public_file.write_bytes(public)
    assert load_gate_verification_key(public_file) == public
    public_file.write_bytes(b"\x01" + b"\x00" * 31)
    with pytest.raises(ContractError, match="prime subgroup"):
        load_gate_verification_key(public_file)


def test_ed25519_implementation_matches_rfc8032_vector():
    from ..orchestration import _ed_sign  # noqa: PLC0415

    seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    expected_public = bytes.fromhex(
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    )
    expected_signature = bytes.fromhex(
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
    )
    assert derive_gate_verification_key(seed) == expected_public
    assert _ed_sign(seed, b"") == expected_signature


def test_arm_t_token_direct_construction_and_fake_object_rejected(registration_bundle):
    registration = freeze_registration_bundle(registration_bundle)
    with pytest.raises(ContractError, match="cannot be constructed"):
        ArmTGateToken(registration.sha256, HASHES[0], "Intel Xeon Platinum 8452Y")  # type: ignore[call-arg]
    with pytest.raises(ContractError, match="requires"):
        authorize_arm_t_task(object(), registration)  # type: ignore[arg-type]
    forged = object.__new__(ArmTGateToken)
    object.__setattr__(forged, "_registration_sha256", registration.sha256)
    object.__setattr__(forged, "_gate_receipt_sha256", HASHES[0])
    object.__setattr__(forged, "_gate_payload", b"forged")
    object.__setattr__(forged, "_architecture", "Intel Xeon Platinum 8452Y")
    object.__setattr__(forged, "_seal", b"0" * 32)
    with pytest.raises(ContractError, match="issued"):
        authorize_arm_t_task(forged, registration)


def test_arm_o_parity_failure_stops_arm_t(registration_bundle, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    evidence_root = tmp_path / "evidence"
    receipts = arm_o_receipts(registration, evidence_root)
    with pytest.raises(ContractError, match="parity"):
        validate_arm_o_gate(
            frozen_registration=registration,
            task_receipts=receipts,
            inherited_test_receipt=inherited_receipt(),
            corrected_test_receipt=corrected_receipt(registration),
            parity_receipt=parity_receipt(registration, receipts, parity=False),
            evidence_root=evidence_root,
        )


def test_missing_or_tampered_task_receipt_stops_gate(registration_bundle, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    evidence_root = tmp_path / "evidence"
    receipts = arm_o_receipts(registration, evidence_root)
    with pytest.raises(ContractError, match="twelve"):
        validate_arm_o_gate(
            frozen_registration=registration,
            task_receipts=receipts[:-1],
            inherited_test_receipt=inherited_receipt(),
            corrected_test_receipt=corrected_receipt(registration),
            parity_receipt=parity_receipt(registration, receipts),
            evidence_root=evidence_root,
        )
    tampered = bytearray(receipts[0])
    tampered[-2] = ord("X")
    receipts[0] = bytes(tampered)
    with pytest.raises(ContractError):
        validate_arm_o_gate(
            frozen_registration=registration,
            task_receipts=receipts,
            inherited_test_receipt=inherited_receipt(),
            corrected_test_receipt=corrected_receipt(registration),
            parity_receipt=parity_receipt(registration, receipts),
            evidence_root=evidence_root,
        )


def test_gate_dereferences_and_rejects_changed_published_evidence(registration_bundle, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    evidence_root = tmp_path / "evidence"
    receipts = arm_o_receipts(registration, evidence_root)
    (evidence_root / "task-0/diagnostics.json").write_bytes(b"{}")
    with pytest.raises(ContractError, match="evidence|published"):
        validate_arm_o_gate(
            frozen_registration=registration,
            task_receipts=receipts,
            inherited_test_receipt=inherited_receipt(),
            corrected_test_receipt=corrected_receipt(registration),
            parity_receipt=parity_receipt(registration, receipts),
            evidence_root=evidence_root,
        )


def test_gate_reloads_referenced_fitted_artifacts_instead_of_trusting_hash_text(
    registration_bundle, tmp_path
):
    registration = freeze_registration_bundle(registration_bundle)
    evidence_root = tmp_path / "evidence"
    receipts = arm_o_receipts(registration, evidence_root)
    artifact = evidence_root / "task-0/artifact-ricker_fit_cache.json"
    original = artifact.read_bytes()
    changed = original.replace(b"synthetic", b"synthetiX", 1)
    assert changed != original
    artifact.write_bytes(changed)
    with pytest.raises(ContractError, match="artifact|evidence|published"):
        validate_arm_o_gate(
            frozen_registration=registration,
            task_receipts=receipts,
            inherited_test_receipt=inherited_receipt(),
            corrected_test_receipt=corrected_receipt(registration),
            parity_receipt=parity_receipt(registration, receipts),
            evidence_root=evidence_root,
        )


def test_gate_rejects_synthetic_diagnostics_placeholder(registration_bundle, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    evidence_root = tmp_path / "evidence"
    receipts = arm_o_receipts(
        registration,
        evidence_root,
        first_diagnostics_mutator=lambda _receipts: {
            "synthetic_probe": {"finite_value": 1.0, "status": "VALIDATED"}
        },
    )
    with pytest.raises(ContractError, match="registered task diagnostics"):
        validate_arm_o_gate(
            frozen_registration=registration,
            task_receipts=receipts,
            inherited_test_receipt=inherited_receipt(),
            corrected_test_receipt=corrected_receipt(registration),
            parity_receipt=parity_receipt(registration, receipts),
            evidence_root=evidence_root,
        )


def test_gate_rejects_unrecognized_diagnostic_receipt(registration_bundle, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    evidence_root = tmp_path / "evidence"

    def add_unrecognized(receipts):
        return {**receipts, "unregistered_summary": {"result": "PASS"}}

    receipts = arm_o_receipts(
        registration,
        evidence_root,
        first_diagnostics_mutator=add_unrecognized,
    )
    with pytest.raises(ContractError, match="registered task diagnostics"):
        validate_arm_o_gate(
            frozen_registration=registration,
            task_receipts=receipts,
            inherited_test_receipt=inherited_receipt(),
            corrected_test_receipt=corrected_receipt(registration),
            parity_receipt=parity_receipt(registration, receipts),
            evidence_root=evidence_root,
        )


def test_gate_rejects_missing_real_frozen_object_parity_receipt(registration_bundle, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    evidence_root = tmp_path / "evidence"
    receipts = arm_o_receipts(
        registration,
        evidence_root,
        missing_parity_first=True,
    )
    with pytest.raises(ContractError, match="fitted-artifact bundle evidence"):
        validate_arm_o_gate(
            frozen_registration=registration,
            task_receipts=receipts,
            inherited_test_receipt=inherited_receipt(),
            corrected_test_receipt=corrected_receipt(registration),
            parity_receipt=parity_receipt(registration, receipts),
            evidence_root=evidence_root,
        )


def test_gate_rejects_canonical_artifact_only_bundle(registration_bundle, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    evidence_root = tmp_path / "evidence"
    receipts = arm_o_receipts(
        registration,
        evidence_root,
        canonical_only_first=True,
    )
    with pytest.raises(ContractError, match="fitted-artifact bundle evidence"):
        validate_arm_o_gate(
            frozen_registration=registration,
            task_receipts=receipts,
            inherited_test_receipt=inherited_receipt(),
            corrected_test_receipt=corrected_receipt(registration),
            parity_receipt=parity_receipt(registration, receipts),
            evidence_root=evidence_root,
        )


def test_gate_replays_and_rejects_handwritten_real_object_parity_receipt(
    registration_bundle, tmp_path
):
    registration = freeze_registration_bundle(registration_bundle)
    evidence_root = tmp_path / "evidence"

    def fabricate_parity(receipt):
        receipt["output_sha256"] = HASHES[7]
        return receipt

    receipts = arm_o_receipts(
        registration,
        evidence_root,
        first_frozen_parity_mutator=fabricate_parity,
    )
    with pytest.raises(ContractError, match="cannot be reproduced"):
        validate_arm_o_gate(
            frozen_registration=registration,
            task_receipts=receipts,
            inherited_test_receipt=inherited_receipt(),
            corrected_test_receipt=corrected_receipt(registration),
            parity_receipt=parity_receipt(registration, receipts),
            evidence_root=evidence_root,
        )


def test_caller_supplied_hash_rebinding_is_rejected(registration_bundle, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    evidence_root = tmp_path / "evidence"
    receipts = arm_o_receipts(registration, evidence_root)
    value = registration.bundle()["code_configuration_hashes"]
    parity = {
        "schema_version": "corrected_stageb_arm_o_parity_v2",
        "architecture": "Intel Xeon Platinum 8452Y",
        "registration_sha256": registration.sha256,
        "git_commit_sha": value["git_commit_sha"],
        "per_episode_action_event_parity": True,
        "accepted_artifact_gate_sha256": HASHES[0],
        "verified_hashes": {"invented": HASHES[1]},
    }
    with pytest.raises(ContractError, match="frozen registration"):
        validate_arm_o_gate(
            frozen_registration=registration,
            task_receipts=receipts,
            inherited_test_receipt=inherited_receipt(),
            corrected_test_receipt=corrected_receipt(registration),
            parity_receipt=canonical_json_bytes(parity),
            evidence_root=evidence_root,
        )


def test_finalizer_is_inspection_only():
    statuses = [
        {
            "task_index": index,
            "cell": cell,
            "method": method,
            "status": "FAILED" if index == 11 else "COMPLETED",
            "receipt_sha256": HASHES[index % len(HASHES)],
        }
        for index, (cell, method) in enumerate((c, m) for c in CELLS for m in METHODS)
    ]
    receipt = inspect_only_finalizer(statuses)
    assert receipt["methods_executed_by_finalizer"] == 0
    assert receipt["repairs_performed"] == 0
    assert receipt["failed"] == 1


def test_slurm_templates_have_correct_dependencies_and_no_mac_path():
    root = Path(__file__).parents[1] / "slurm"
    submit = (root / "submit_chain.template.sh").read_text()
    assert "afterok:${arm_o_job}" in submit
    assert "afterok:${gate_job}" in submit
    assert "afterany:${arm_t_job}" in submit
    for path in root.iterdir():
        if path.is_file():
            text = path.read_text()
            assert "/Users/hphu0007" not in text
    assert "#SBATCH --constraint=xenon-8452Y" in (root / "arm_o_array.sbatch").read_text()
    assert "#SBATCH --array=0-11" in (root / "arm_t_array.sbatch").read_text()
