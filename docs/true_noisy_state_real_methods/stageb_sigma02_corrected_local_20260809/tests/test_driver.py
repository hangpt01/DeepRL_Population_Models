from __future__ import annotations

import ast
import copy
from dataclasses import replace
from pathlib import Path

import pytest

from .. import driver
from ..common import (
    ContractError,
    canonical_json_bytes,
    require_registered_cpu_model,
    sha256_file,
)
from ..evidence import REGISTERED_EPISODE_IDS
from ..publication import load_success_receipt
from ..registration import (
    CELLS,
    METHODS,
    freeze_registration_bundle,
    require_runtime_interpreter_binding,
)
from .conftest import HASHES, artifact_components, step_records


def _interpreter_receipt(registration, command, *, arm=None, task_index=None):
    if task_index is None:
        role = registration.bundle()["stageb_interpreter_bindings"]["command_roles"][command]
        binding = next(
            item
            for item in registration.bundle()["stageb_interpreter_bindings"]["bindings"]
            if item["role"] == role
        )
    else:
        task = driver._task_for_index(registration, arm, task_index)
        binding = next(
            item
            for item in registration.bundle()["stageb_interpreter_bindings"]["bindings"]
            if item["role"] == task["interpreter_role"]
        )
    observed = {
        field: binding[field]
        for field in (
            "absolute_interpreter_path",
            "resolved_executable_path",
            "python_version",
            "full_python_version",
            "numpy_version",
        )
    }
    return require_runtime_interpreter_binding(
        registration,
        command=command,
        arm=arm,
        task_index=task_index,
        observed=observed,
    )


def _execution(registration_sha256: str, method: str = "refplan", arm: str = "O"):
    episodes = []
    for episode_id in REGISTERED_EPISODE_IDS:
        episodes.append(
            tuple(
                replace(item, episode_id=episode_id, method=method, arm=arm)
                for item in step_records(
                    registration_sha256=registration_sha256, method=method, arm=arm
                )
            )
        )
    actions = tuple(tuple(item.selected_action for item in episode) for episode in episodes)
    rows = tuple(
        {"episode": index, "seed": episode_id, "block_seed": episode_id - index % 4}
        for index, episode_id in enumerate(REGISTERED_EPISODE_IDS)
    )
    dispersion = tuple(tuple(float(index) / 100.0 for index in range(50)) for _ in episodes)
    posteriors = tuple(tuple((0.2, 0.2, 0.2, 0.2, 0.2) for _ in range(51)) for _ in episodes)
    identities = tuple(HASHES[index % len(HASHES)] for index in range(20))
    return driver.TaskExecution(
        tuple(episodes),
        rows,
        actions,
        posteriors,
        dispersion,
        identities,
        identities,
        identities,
        identities,
        require_registered_cpu_model("Intel(R) Xeon(R) Platinum 8452Y"),
    )


def test_synthetic_bundle_upgrades_to_mandatory_driver_hash(registration_bundle):
    hashes = registration_bundle["code_configuration_hashes"]
    assert hashes["schema_version"] == "corrected_stageb_code_configuration_hashes_v2"
    assert hashes["stageb_driver_sha256"] == sha256_file(Path(driver.__file__))
    freeze_registration_bundle(registration_bundle)


@pytest.mark.parametrize("mutation", ["missing", "stale", "schema_v1"])
def test_registration_driver_binding_fails_closed(registration_bundle, mutation):
    malformed = copy.deepcopy(registration_bundle)
    hashes = malformed["code_configuration_hashes"]
    if mutation == "missing":
        del hashes["stageb_driver_sha256"]
    elif mutation == "stale":
        hashes["stageb_driver_sha256"] = "f" * 64
    else:
        hashes["schema_version"] = "corrected_stageb_code_configuration_hashes_v1"
    with pytest.raises(ContractError):
        freeze_registration_bundle(malformed)


def test_all_four_cli_subcommands_dispatch(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(driver, "run_arm_task", lambda **kwargs: calls.append(("task", kwargs)))
    monkeypatch.setattr(driver, "run_arm_o_gate", lambda **kwargs: calls.append(("gate", kwargs)))
    monkeypatch.setattr(
        driver,
        "run_inspection_only_finalizer",
        lambda **kwargs: calls.append(("finalizer", kwargs)),
    )
    registration = tmp_path / "registration.json"
    output = tmp_path / "output"
    gate = tmp_path / "gate.json"
    key = tmp_path / "key.bin"
    seed = tmp_path / "seed.bin"
    assert (
        driver.main(
            [
                "arm-o",
                "--task-index",
                "0",
                "--registration",
                str(registration),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    assert (
        driver.main(
            [
                "arm-o-gate",
                "--registration",
                str(registration),
                "--gate-signing-seed-file",
                str(seed),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    assert (
        driver.main(
            [
                "arm-t",
                "--task-index",
                "11",
                "--registration",
                str(registration),
                "--arm-o-gate",
                str(gate),
                "--gate-public-key-file",
                str(key),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    assert (
        driver.main(
            [
                "finalize-inspection-only",
                "--registration",
                str(registration),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    assert [item[0] for item in calls] == ["task", "gate", "task", "finalizer"]
    assert calls[0][1]["arm"] == "O" and calls[2][1]["arm"] == "T"


@pytest.mark.parametrize("command", ["arm-o", "arm-o-gate", "arm-t", "finalize-inspection-only"])
def test_subcommands_reject_missing_mandatory_arguments(command):
    with pytest.raises(SystemExit):
        driver._parser().parse_args([command])


def test_complete_registered_24_task_traversal(registration_bundle):
    registration = freeze_registration_bundle(registration_bundle)
    observed = []
    for arm in ("O", "T"):
        for index in range(12):
            task = driver._task_for_index(registration, arm, index)
            observed.append((task["arm"], task["cell"], task["method"]))
    expected = [(arm, cell, method) for arm in ("O", "T") for cell in CELLS for method in METHODS]
    assert observed == expected


@pytest.mark.parametrize("index", [-1, 12, True, 1.5])
def test_task_index_fails_closed(registration_bundle, index):
    registration = freeze_registration_bundle(registration_bundle)
    with pytest.raises(ContractError, match="0..11"):
        driver._task_for_index(registration, "O", index)


def test_exact_20_by_50_evidence_and_return_arithmetic(registration_bundle):
    registration = freeze_registration_bundle(registration_bundle)
    execution = _execution(registration.sha256)
    assert len(execution.step_evidence) == 20
    assert all(len(episode) == 50 for episode in execution.step_evidence)
    for episode in execution.step_evidence:
        receipt = driver.reconstruct_episode(episode)
        assert receipt["reconstruction_result"] == "PASS"
        assert receipt["action_sequence"] == list(execution.actions[0])


@pytest.mark.parametrize(
    "forbidden",
    ["next_states", "truth_path", "hidden_family", "safety_threshold", "evaluator_info"],
)
def test_driver_inputs_reject_forbidden_private_fields(forbidden):
    with pytest.raises(ContractError, match="forbidden"):
        driver._validate_no_forbidden_descriptor_keys({"nested": {forbidden: "x"}})


def test_arm_t_refuses_missing_gate_before_executor(registration_bundle, monkeypatch, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    monkeypatch.setattr(driver, "_load_registration", lambda _path: registration)
    monkeypatch.setattr(driver, "load_driver_inputs", lambda *_args: object())
    called = False

    def forbidden_executor(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("executor must remain unreachable")

    with pytest.raises(ContractError, match="requires the signed"):
        driver.run_arm_task(
            arm="T",
            task_index=0,
            registration_path=tmp_path / "registration",
            output_root=tmp_path,
            executor=forbidden_executor,
        )
    assert called is False


def test_arm_t_refuses_signing_seed_environment(registration_bundle, monkeypatch, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    monkeypatch.setattr(driver, "_load_registration", lambda _path: registration)
    monkeypatch.setattr(driver, "load_driver_inputs", lambda *_args: object())
    monkeypatch.setenv("STAGEB_GATE_SIGNING_SEED_FILE", "/forbidden/seed")
    with pytest.raises(ContractError, match="must never enter"):
        driver.run_arm_task(
            arm="T",
            task_index=0,
            registration_path=tmp_path / "registration",
            output_root=tmp_path,
            arm_o_gate_path=tmp_path / "gate",
            gate_public_key_path=tmp_path / "public",
        )


def test_driver_contains_no_fit_call_or_oracle_shortcut():
    source = Path(driver.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "fit" not in called_attributes
    assert "build_method" not in called_attributes
    assert "OracleStateFilter" not in source
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "observation_noise_sigma":
                continue
            assert not (
                isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, (int, float))
                and keyword.value.value == 0
            )


def test_o_t_artifact_plan_and_fitted_object_identity():
    probe = driver.SealedFitProbe(
        0,
        Path("/tmp/not-opened"),
        HASHES[0],
        HASHES[1],
        HASHES[2],
        HASHES[3],
        {"reward_surrogate": HASHES[4]},
    )
    plan_o = probe.artifact_plan_sha256(cell=CELLS[0], method=METHODS[0], dataset_sha256=HASHES[5])
    plan_t = probe.artifact_plan_sha256(cell=CELLS[0], method=METHODS[0], dataset_sha256=HASHES[5])
    assert plan_o == plan_t


def test_evd_transition_is_explicitly_not_applicable(registration_bundle):
    registration = freeze_registration_bundle(registration_bundle)
    task = driver._task_for_index(registration, "O", 5)
    receipt = driver._transition_diagnostic(registration, task, artifact_components(task["method"]))
    assert receipt["applicability"] == "DEFINITIONALLY_NOT_APPLICABLE"
    assert receipt["artifact_hashes"] == []


def test_atomic_task_publication_and_no_retry(registration_bundle, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    task = dict(driver._task_for_index(registration, "O", 2))
    accepted = tmp_path / "accepted.csv"
    accepted.write_text(
        "episode,seed,block_seed\n"
        + "".join(
            f"{index},{episode_id},{episode_id - index % 4}\n"
            for index, episode_id in enumerate(REGISTERED_EPISODE_IDS)
        ),
        encoding="utf-8",
    )
    inputs = driver.DriverInputs(
        tmp_path / driver.DRIVER_INPUTS,
        Path(driver.__file__).resolve().parents[3],
        registration.registration_id,
        tuple(
            driver.SealedFitProbe(index, tmp_path, HASHES[0], HASHES[1], HASHES[2], HASHES[3], {})
            for index in range(12)
        ),
        {},
    )
    evaluator_inputs = driver.EvaluatorOnlyInputs(
        tuple(
            {
                "task_index": index,
                "cell": CELLS[index // len(METHODS)],
                "method": METHODS[index % len(METHODS)],
                "episodes_csv": {"path": accepted},
            }
            for index in range(12)
        )
    )
    target = driver.publish_registered_task(
        registration=registration,
        inputs=inputs,
        evaluator_inputs=evaluator_inputs,
        task=task,
        task_index=2,
        execution=_execution(registration.sha256),
        component_payloads=artifact_components("refplan"),
        frozen_payload=b"synthetic-frozen",
        replay_payload=canonical_json_bytes({"synthetic": True}),
        output_root=tmp_path,
        interpreter_identity=_interpreter_receipt(registration, "arm-o", arm="O", task_index=2),
    )
    load_success_receipt(target / "PUBLICATION_SUCCESS.json")
    assert (tmp_path / "arm-o-receipts/task-02.json").is_file()
    with pytest.raises(ContractError, match="overwrite|exists|retry"):
        driver.publish_registered_task(
            registration=registration,
            inputs=inputs,
            evaluator_inputs=evaluator_inputs,
            task=task,
            task_index=2,
            execution=_execution(registration.sha256),
            component_payloads=artifact_components("refplan"),
            frozen_payload=b"synthetic-frozen",
            replay_payload=canonical_json_bytes({"synthetic": True}),
            output_root=tmp_path,
            interpreter_identity=_interpreter_receipt(registration, "arm-o", arm="O", task_index=2),
        )


def test_finalizer_is_inspection_only(registration_bundle, monkeypatch, tmp_path):
    registration = freeze_registration_bundle(registration_bundle)
    monkeypatch.setattr(driver, "_load_registration", lambda _path: registration)
    monkeypatch.setattr(driver, "load_driver_inputs", lambda *_args: object())
    monkeypatch.setattr(
        driver,
        "require_runtime_interpreter_binding",
        lambda *_args, **_kwargs: _interpreter_receipt(registration, "finalize-inspection-only"),
    )
    target = driver.run_inspection_only_finalizer(
        registration_path=tmp_path / "registration", output_root=tmp_path
    )
    receipt = driver.strict_json_loads((target / "FINALIZER_RECEIPT.json").read_bytes())
    assert receipt["methods_executed_by_finalizer"] == 0
    assert receipt["scientific_values_opened"] is False
    assert receipt["scientific_calculations_performed"] is False
    assert receipt["failed"] == 12
