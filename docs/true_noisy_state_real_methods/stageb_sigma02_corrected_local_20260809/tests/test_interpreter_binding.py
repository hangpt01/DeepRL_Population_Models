from __future__ import annotations

import copy
from pathlib import Path

import pytest

from .. import driver
from ..common import ContractError
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
    monkeypatch.setattr(driver, "_load_registration", lambda _path: registration)
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
