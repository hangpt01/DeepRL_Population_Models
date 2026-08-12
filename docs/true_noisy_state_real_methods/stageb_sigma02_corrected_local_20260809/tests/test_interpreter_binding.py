from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from .. import driver, real_artifacts
from ..common import (
    ContractError,
    canonical_json_bytes,
    require_registered_cpu_model,
    sha256_bytes,
)
from ..registration import (
    INTERPRETER_IDENTITY_FIELDS,
    freeze_registration_bundle,
    interpreter_binding_for_command,
    interpreter_binding_for_task,
    require_runtime_interpreter_binding,
)
from . import test_driver as _driver_contract_fixture  # noqa: F401


def _observed(binding):
    return {field: binding[field] for field in INTERPRETER_IDENTITY_FIELDS}


def _binding(registration, role):
    return next(
        item
        for item in registration.bundle()["stageb_interpreter_bindings"]["bindings"]
        if item["role"] == role
    )


def test_old_registration_version_is_rejected(registration_bundle):
    malformed = copy.deepcopy(registration_bundle)
    malformed["corrected_stageb_registration"]["schema_version"] = (
        "corrected_stageb_registration_v1"
    )
    with pytest.raises(ContractError, match="registration schema version"):
        freeze_registration_bundle(malformed)


def test_missing_interpreter_binding_section_is_rejected(registration_bundle):
    malformed = copy.deepcopy(registration_bundle)
    del malformed["stageb_interpreter_bindings"]
    with pytest.raises(ContractError, match="registration bundle"):
        freeze_registration_bundle(malformed)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_role",
        "extra_role",
        "missing_method",
        "extra_method",
        "missing_task_index",
        "extra_task_index",
        "wrong_track",
    ],
)
def test_binding_role_method_track_and_index_coverage_is_exact(registration_bundle, mutation):
    malformed = copy.deepcopy(registration_bundle)
    bindings = malformed["stageb_interpreter_bindings"]["bindings"]
    if mutation == "missing_role":
        bindings.pop()
    elif mutation == "extra_role":
        bindings.append(copy.deepcopy(bindings[0]))
    elif mutation == "missing_method":
        bindings[1]["methods"].pop()
    elif mutation == "extra_method":
        bindings[0]["methods"].append("refplan")
    elif mutation == "missing_task_index":
        bindings[1]["task_indices"].pop()
    elif mutation == "extra_task_index":
        bindings[0]["task_indices"].append(2)
    else:
        bindings[0]["track"] = "general"
    with pytest.raises(ContractError, match="interpreter"):
        freeze_registration_bundle(malformed)


def test_task_role_mismatch_is_rejected(registration_bundle):
    malformed = copy.deepcopy(registration_bundle)
    malformed["task_manifest"]["tasks"][0]["interpreter_role"] = "general_registered"
    with pytest.raises(ContractError, match="interpreter role"):
        freeze_registration_bundle(malformed)


@pytest.mark.parametrize("field", sorted(INTERPRETER_IDENTITY_FIELDS))
def test_each_runtime_identity_field_fails_closed_independently(registration_bundle, field):
    registration = freeze_registration_bundle(registration_bundle)
    binding = _binding(registration, "ecological_paper_faithful")
    observed = _observed(binding)
    observed[field] = f"{observed[field]}-mismatch"
    with pytest.raises(ContractError, match=field):
        require_runtime_interpreter_binding(
            registration,
            command="arm-o",
            arm="O",
            task_index=0,
            observed=observed,
        )


@pytest.mark.parametrize(
    ("role", "task_index"),
    [("ecological_paper_faithful", 0), ("general_registered", 2)],
)
def test_correct_ecological_and_general_bindings_pass(registration_bundle, role, task_index):
    registration = freeze_registration_bundle(registration_bundle)
    binding = _binding(registration, role)
    receipt = require_runtime_interpreter_binding(
        registration,
        command="arm-o",
        arm="O",
        task_index=task_index,
        observed=_observed(binding),
    )
    assert receipt["role"] == role
    assert receipt["expected"] == binding
    assert receipt["observed"] == _observed(binding)


@pytest.mark.parametrize(
    ("command", "arm", "task_index", "runner"),
    [
        (
            "arm-o",
            "O",
            0,
            lambda tmp: driver.run_arm_task(
                arm="O",
                task_index=0,
                registration_path=tmp / "registration.json",
                output_root=tmp,
            ),
        ),
        (
            "arm-t",
            "T",
            2,
            lambda tmp: driver.run_arm_task(
                arm="T",
                task_index=2,
                registration_path=tmp / "registration.json",
                output_root=tmp,
            ),
        ),
        (
            "arm-o-gate",
            None,
            None,
            lambda tmp: driver.run_arm_o_gate(
                registration_path=tmp / "registration.json",
                gate_signing_seed_path=tmp / "seed.bin",
                output_root=tmp,
            ),
        ),
        (
            "finalize-inspection-only",
            None,
            None,
            lambda tmp: driver.run_inspection_only_finalizer(
                registration_path=tmp / "registration.json", output_root=tmp
            ),
        ),
    ],
)
def test_all_driver_subcommands_validate_binding_before_inputs_or_publication(
    registration_bundle, monkeypatch, tmp_path, command, arm, task_index, runner
):
    registration = freeze_registration_bundle(registration_bundle)
    monkeypatch.setattr(driver, "_load_registration", lambda _path, **_kwargs: registration)
    calls = []

    class BindingStop(RuntimeError):
        pass

    def stop_at_binding(_registration, **context):
        calls.append(context)
        raise BindingStop

    monkeypatch.setattr(driver, "require_runtime_interpreter_binding", stop_at_binding)
    monkeypatch.setattr(
        driver,
        "load_driver_inputs",
        lambda *_args: pytest.fail("driver inputs loaded before interpreter validation"),
    )
    with pytest.raises(BindingStop):
        runner(tmp_path)
    expected_context = {"command": command}
    if arm is not None:
        expected_context.update({"arm": arm, "task_index": task_index})
    assert calls == [expected_context]
    assert not any(tmp_path.iterdir())


def test_task_and_command_roles_resolve_exactly(registration_bundle):
    registration = freeze_registration_bundle(registration_bundle)
    for arm in ("O", "T"):
        assert (
            interpreter_binding_for_task(registration, arm=arm, task_index=0)[1]["role"]
            == "ecological_paper_faithful"
        )
        assert (
            interpreter_binding_for_task(registration, arm=arm, task_index=2)[1]["role"]
            == "general_registered"
        )
    assert interpreter_binding_for_command(registration, "arm-o-gate")["role"] == (
        "general_registered"
    )
    assert (
        interpreter_binding_for_command(registration, "finalize-inspection-only")["role"]
        == "general_registered"
    )


def test_environment_variables_cannot_override_frozen_binding(registration_bundle, monkeypatch):
    registration = freeze_registration_bundle(registration_bundle)
    binding = _binding(registration, "general_registered")
    monkeypatch.setenv("STAGEB_PYTHON", "/operator/override")
    monkeypatch.setenv("STAGEB_GATE_PYTHON", "/operator/override")
    monkeypatch.setenv("STAGEB_FINALIZER_PYTHON", "/operator/override")
    receipt = require_runtime_interpreter_binding(
        registration,
        command="arm-o",
        arm="O",
        task_index=2,
        observed=_observed(binding),
    )
    assert receipt["expected"]["absolute_interpreter_path"] == "/usr/bin/python"


def test_receipt_records_complete_expected_and_observed_identity(registration_bundle):
    registration = freeze_registration_bundle(registration_bundle)
    binding = _binding(registration, "general_registered")
    receipt = require_runtime_interpreter_binding(
        registration,
        command="arm-t",
        arm="T",
        task_index=11,
        observed=_observed(binding),
    )
    assert set(receipt["observed"]) == INTERPRETER_IDENTITY_FIELDS
    assert set(INTERPRETER_IDENTITY_FIELDS).issubset(receipt["expected"])
    assert receipt["method"] == "ensemble_value_disagreement_pessimism"
    assert receipt["track"] == "general"
    assert receipt["task_index"] == 11


@pytest.mark.parametrize(
    ("task_index", "role"),
    [(0, "ecological_paper_faithful"), (2, "general_registered")],
)
def test_gate_replay_subprocess_uses_exact_registered_task_interpreter(
    registration_bundle, monkeypatch, task_index, role
):
    registration = freeze_registration_bundle(registration_bundle)
    task, binding = interpreter_binding_for_task(registration, arm="O", task_index=task_index)
    payload = canonical_json_bytes({"frozen": task_index})
    receipt = {
        "operation_input_sha256": sha256_bytes(b"operation"),
        "output_sha256": sha256_bytes(b"output"),
        "post_call_state_sha256": sha256_bytes(b"state"),
    }
    observed = _observed(binding)
    interpreter_receipt = require_runtime_interpreter_binding(
        registration,
        command="arm-o",
        arm="O",
        task_index=task_index,
        observed=observed,
    )
    cpu_receipt = require_registered_cpu_model("Intel(R) Xeon(R) Platinum 8452Y")
    seen = {}

    def completed(command, **kwargs):
        request = json.loads(kwargs["input"])
        seen.update({"command": command, "environment": kwargs["env"], "request": request})
        result = {
            "schema_version": "corrected_stageb_registered_gate_replay_result_v1",
            "source_registration_sha256": registration.sha256,
            "source_git_commit_sha": registration.bundle()["code_configuration_hashes"][
                "git_commit_sha"
            ],
            "source_manifest_sha256": registration.bundle()["code_configuration_hashes"][
                "source_manifest_sha256"
            ],
            "task_index": task_index,
            "cell": task["cell"],
            "method": task["method"],
            "component": real_artifacts.scientific_component_for_method(task["method"]),
            "interpreter_identity": interpreter_receipt,
            "cpu_identity": cpu_receipt,
            "serialized_sha256": sha256_bytes(payload),
            "reproduced_payload_sha256": sha256_bytes(payload),
            "stored_receipt_sha256": sha256_bytes(canonical_json_bytes(receipt)),
            "reproduced_receipt_sha256": sha256_bytes(canonical_json_bytes(receipt)),
            "operation_input_sha256": receipt["operation_input_sha256"],
            "output_sha256": receipt["output_sha256"],
            "post_call_state_sha256": receipt["post_call_state_sha256"],
            "payload_exact": True,
            "receipt_exact": True,
            "gate_signing_seed_present": False,
            "evaluator_constructed": False,
            "rollout_executed": False,
            "return_calculated": False,
            "result": "PASS",
        }
        return SimpleNamespace(
            returncode=0,
            stdout=canonical_json_bytes(result),
            stderr=b"",
        )

    monkeypatch.setattr(real_artifacts.subprocess, "run", completed)
    monkeypatch.setenv("STAGEB_GATE_SIGNING_SEED_FILE", "/must/not/propagate")
    result = real_artifacts.revalidate_frozen_object_parity_in_registered_subprocess(
        payload,
        receipt,
        expected_component=real_artifacts.scientific_component_for_method(task["method"]),
        repository_root=Path(__file__).resolve().parents[4],
        frozen_registration=registration,
        task_index=task_index,
    )
    assert result["result"] == "PASS"
    assert seen["command"][0] == binding["absolute_interpreter_path"]
    assert seen["request"]["task_index"] == task_index
    assert seen["request"]["expected_task"]["interpreter_role"] == role
    assert "STAGEB_GATE_SIGNING_SEED_FILE" not in seen["environment"]
    assert "STAGEB_GATE_SIGNING_SEED" not in seen["environment"]


def test_gate_replay_subprocess_fails_closed_on_identity_or_exact_receipt_drift(
    registration_bundle, monkeypatch
):
    registration = freeze_registration_bundle(registration_bundle)
    task, _binding_value = interpreter_binding_for_task(registration, arm="O", task_index=0)
    payload = canonical_json_bytes({"frozen": 0})
    receipt = {
        "operation_input_sha256": sha256_bytes(b"operation"),
        "output_sha256": sha256_bytes(b"output"),
        "post_call_state_sha256": sha256_bytes(b"state"),
    }
    monkeypatch.setattr(
        real_artifacts.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2,
            stdout=b"",
            stderr=(
                b"STAGEB_REGISTERED_GATE_REPLAY_FAIL_CLOSED: "
                b"registered interpreter identity mismatch"
            ),
        ),
    )
    with pytest.raises(ContractError, match="interpreter identity mismatch"):
        real_artifacts.revalidate_frozen_object_parity_in_registered_subprocess(
            payload,
            receipt,
            expected_component=real_artifacts.scientific_component_for_method(task["method"]),
            repository_root=Path(__file__).resolve().parents[4],
            frozen_registration=registration,
            task_index=0,
        )


def _synthetic_carry_forward_bridge(registration, registration_path, task_payloads):
    bundle = registration.bundle()
    hashes = bundle["code_configuration_hashes"]
    registered_inputs = bundle["stageb_driver_inputs"]
    manifest = (
        driver._repository_root()
        / "docs/true_noisy_state_real_methods/stageb_sigma02_corrected_local_20260809/"
        "SOURCE_TEST_HASHES.sha256"
    ).read_bytes()
    marker = b"# SELF-NORMALIZED-SHA256: "
    manifest_sha = next(
        line[len(marker) :].split(b"  ", 1)[0].decode()
        for line in manifest.splitlines()
        if line.startswith(marker)
    )
    return {
        "schema_version": driver.GATE_CARRY_FORWARD_BRIDGE_SCHEMA,
        "purpose": (
            "validate immutable Arm O publications with the mixed-interpreter gate correction"
        ),
        "source_registration": {
            "registration_path": str(registration_path),
            "registration_sha256": registration.sha256,
            "registration_id": registration.registration_id,
            "repository_root": registered_inputs["descriptor"]["repository_root"],
            "git_commit_sha": hashes["git_commit_sha"],
            "source_manifest_sha256": hashes["source_manifest_sha256"],
            "stageb_driver_sha256": hashes["stageb_driver_sha256"],
            "driver_inputs_sha256": registered_inputs["driver_inputs_sha256"],
        },
        "corrected_gate": {
            "repository_root": str(driver._repository_root()),
            "git_commit_sha": driver._git_head(driver._repository_root()),
            "source_manifest_sha256": manifest_sha,
            "stageb_driver_sha256": driver.sha256_file(Path(driver.__file__)),
        },
        "arm_o_publications": {
            "output_root": str(driver._repository_root()),
            "file_count": 197,
            "tree_sha256": sha256_bytes(b"tree"),
            "task_receipt_sha256": [sha256_bytes(item) for item in task_payloads],
            "preserved_from_source_registration": True,
        },
        "authorization": {
            "gate_only": True,
            "arm_o_rerun_permitted": False,
            "arm_o_modification_permitted": False,
            "return_recalculation_permitted": False,
            "independent_audit_required": True,
        },
    }


def test_carry_forward_bridge_is_gate_only_and_binds_immutable_old_registration(
    registration_bundle, tmp_path, monkeypatch
):
    registration = freeze_registration_bundle(registration_bundle)
    registration_path = tmp_path / "registration.json"
    registration_path.write_bytes(registration.payload)
    task_payloads = [canonical_json_bytes({"task": index}) for index in range(12)]
    bridge = _synthetic_carry_forward_bridge(registration, registration_path, task_payloads)
    monkeypatch.setattr(
        driver,
        "_arm_o_tree_identity",
        lambda _root: {
            "file_count": 197,
            "tree_sha256": sha256_bytes(b"tree"),
            "entries": [],
        },
    )
    validation = driver.validate_gate_carry_forward_bridge(
        bridge,
        bridge_payload=canonical_json_bytes(bridge),
        registration_path=registration_path,
        registration=registration,
        driver_inputs_payload=canonical_json_bytes(
            registration.bundle()["stageb_driver_inputs"]["descriptor"]
        ),
        output_root=driver._repository_root(),
        task_payloads=task_payloads,
    )
    assert validation["arm_o_file_count"] == 197
    assert validation["source_registration_sha256"] == registration.sha256
    assert validation["result"] == "PASS"


@pytest.mark.parametrize(
    "mutation",
    ["allow_rerun", "task_receipt_rebinding", "corrected_commit_rebinding"],
)
def test_carry_forward_bridge_rejects_scope_or_identity_rebinding(
    registration_bundle, tmp_path, monkeypatch, mutation
):
    registration = freeze_registration_bundle(registration_bundle)
    registration_path = tmp_path / "registration.json"
    registration_path.write_bytes(registration.payload)
    task_payloads = [canonical_json_bytes({"task": index}) for index in range(12)]
    bridge = _synthetic_carry_forward_bridge(registration, registration_path, task_payloads)
    monkeypatch.setattr(
        driver,
        "_arm_o_tree_identity",
        lambda _root: {
            "file_count": 197,
            "tree_sha256": sha256_bytes(b"tree"),
            "entries": [],
        },
    )
    if mutation == "allow_rerun":
        bridge["authorization"]["arm_o_rerun_permitted"] = True
    elif mutation == "task_receipt_rebinding":
        bridge["arm_o_publications"]["task_receipt_sha256"][0] = sha256_bytes(b"changed")
    else:
        bridge["corrected_gate"]["git_commit_sha"] = "0" * 40
    with pytest.raises(ContractError, match="carry-forward"):
        driver.validate_gate_carry_forward_bridge(
            bridge,
            bridge_payload=canonical_json_bytes(bridge),
            registration_path=registration_path,
            registration=registration,
            driver_inputs_payload=canonical_json_bytes(
                registration.bundle()["stageb_driver_inputs"]["descriptor"]
            ),
            output_root=driver._repository_root(),
            task_payloads=task_payloads,
        )


def test_slurm_launches_are_registration_derived_and_not_environment_overridable():
    root = Path(__file__).parents[1] / "slurm"
    task_scripts = [root / "arm_o_array.sbatch", root / "arm_t_array.sbatch"]
    command_scripts = [root / "arm_o_gate.sbatch", root / "finalizer.sbatch"]
    for path in task_scripts:
        text = path.read_text(encoding="utf-8")
        assert "interpreter_binding_for_task" in text
        assert 'binding["absolute_interpreter_path"]' in text
    for path in command_scripts:
        text = path.read_text(encoding="utf-8")
        assert "interpreter_binding_for_command" in text
        assert 'binding["absolute_interpreter_path"]' in text
    combined = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.sbatch"))
    assert "STAGEB_PYTHON" not in combined
    assert "STAGEB_GATE_PYTHON" not in combined
    assert "STAGEB_FINALIZER_PYTHON" not in combined
