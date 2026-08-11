from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest

from .. import driver
from ..common import (
    ContractError,
    normalize_cpu_model_identity,
    require_registered_cpu_model,
    validate_cpu_identity_receipt,
)
from ..registration import freeze_registration_bundle
from ..submission import (
    derive_durable_log_plan,
    prepare_durable_log_directories,
    validate_durable_log_plan,
)


ACTUAL_8452Y_PROC_CPUINFO_MODEL = "Intel(R) Xeon(R) Platinum 8452Y"


@pytest.mark.parametrize(
    "observed",
    [
        ACTUAL_8452Y_PROC_CPUINFO_MODEL,
        "Intel Xeon Platinum 8452Y",
        "Intel(TM) Xeon(TM) Platinum 8452Y",
        " Intel(R)   Xeon(R)\tPlatinum   8452Y ",
    ],
)
def test_complete_cpu_identity_accepts_registered_normalizations(observed):
    receipt = require_registered_cpu_model(observed)
    assert receipt["observed_raw"] == observed
    assert receipt["observed_normalized"] == "Intel Xeon Platinum 8452Y"
    assert receipt["expected_raw"] == "Intel Xeon Platinum 8452Y"
    assert receipt["expected_normalized"] == "Intel Xeon Platinum 8452Y"
    assert validate_cpu_identity_receipt(receipt) == receipt


@pytest.mark.parametrize(
    "observed",
    [
        "Intel(R) Xeon(R) Gold 6548Y+",
        "Intel(R) Xeon(R) Platinum 8462Y+",
        "unrelated accelerator 8452Y validated beside Intel Xeon Platinum",
        "Intel Xeon Platinum 8452Y engineering sample",
    ],
)
def test_complete_cpu_identity_rejects_other_or_misleading_models(observed):
    with pytest.raises(ContractError, match="CPU mismatch"):
        require_registered_cpu_model(observed)


def test_cpu_normalizer_removes_only_markers_and_whitespace():
    assert normalize_cpu_model_identity(" Intel(TM)  Xeon(R) Platinum 8452Y ") == (
        "Intel Xeon Platinum 8452Y"
    )
    assert normalize_cpu_model_identity("Intel Xeon Gold 6548Y+") == "Intel Xeon Gold 6548Y+"


def test_driver_environment_guard_uses_canonical_cpu_receipt(monkeypatch):
    from .. import common

    monkeypatch.setattr(common, "observed_cpu_model", lambda: ACTUAL_8452Y_PROC_CPUINFO_MODEL)
    for name, value in {
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }.items():
        monkeypatch.setenv(name, value)
    receipt = driver._require_scientific_environment()
    assert receipt["observed_raw"] == ACTUAL_8452Y_PROC_CPUINFO_MODEL
    assert receipt["observed_normalized"] == driver.CPU_MODEL


def _shared_roots():
    repository = Path(__file__).resolve().parents[4]
    return tempfile.TemporaryDirectory(prefix=".stageb-durable-log-test-", dir=repository)


def test_durable_array_and_scalar_log_derivation():
    plan = derive_durable_log_plan(
        "/fs04/scratch2/ce25/stageb-registered-log-evidence",
        "/fs04/scratch2/ce25/stageb-registered-scientific-output",
    )
    templates = plan["templates"]
    assert templates["arm_o"]["stdout"].endswith("/arm-o/arm-o-%A_%a.out")
    assert templates["arm_t"]["stderr"].endswith("/arm-t/arm-t-%A_%a.err")
    assert templates["arm_o_gate"]["stdout"].endswith("/arm-o-gate-%j.out")
    assert templates["finalizer"]["stderr"].endswith("/finalizer-%j.err")
    assert len({path for binding in templates.values() for path in binding.values()}) == 8


def test_durable_plan_preparation_survives_partial_submission_log():
    with _shared_roots() as root_text:
        root = Path(root_text)
        output = root / "scientific-output"
        output.mkdir(mode=0o700)
        evidence = root / "durable-evidence"
        plan = derive_durable_log_plan(evidence, output)
        prepared = prepare_durable_log_directories(plan)
        first_log = Path(prepared["templates"]["arm_o"]["stdout"])
        rendered = Path(str(first_log).replace("%A_%a", "123_0"))
        rendered.write_text("partial submission log remains durable\n", encoding="utf-8")
        assert rendered.is_file()
        assert (
            validate_durable_log_plan(prepared, require_existing=True, require_writable=True)
            == prepared
        )


def test_durable_plan_rejects_path_escape():
    plan = dict(
        derive_durable_log_plan(
            "/fs04/scratch2/ce25/stageb-registered-log-evidence",
            "/fs04/scratch2/ce25/stageb-registered-scientific-output",
        )
    )
    plan["templates"] = copy.deepcopy(plan["templates"])
    plan["templates"]["arm_o"]["stdout"] = (
        "/fs04/scratch2/ce25/stageb-registered-log-evidence/../escaped-%A_%a.out"
    )
    with pytest.raises(ContractError):
        validate_durable_log_plan(plan)


def test_durable_plan_rejects_symlink_parent():
    with _shared_roots() as root_text:
        root = Path(root_text)
        output = root / "scientific-output"
        output.mkdir(mode=0o700)
        evidence = root / "durable-evidence"
        plan = derive_durable_log_plan(evidence, output)
        prepare_durable_log_directories(plan)
        role = evidence / "logs" / "arm-o"
        target = root / "redirected"
        target.mkdir(mode=0o700)
        role.rmdir()
        role.symlink_to(target, target_is_directory=True)
        with pytest.raises(ContractError, match="real directory|symlink"):
            validate_durable_log_plan(plan, require_existing=True, require_writable=True)


def test_durable_plan_rejects_collision():
    plan = dict(
        derive_durable_log_plan(
            "/fs04/scratch2/ce25/stageb-registered-log-evidence",
            "/fs04/scratch2/ce25/stageb-registered-scientific-output",
        )
    )
    plan["templates"] = copy.deepcopy(plan["templates"])
    plan["templates"]["arm_t"]["stderr"] = plan["templates"]["arm_t"]["stdout"]
    with pytest.raises(ContractError):
        validate_durable_log_plan(plan)


@pytest.mark.parametrize("root", ["/tmp/stageb-logs", "/var/tmp/stageb-logs", "relative/logs"])
def test_durable_plan_rejects_node_local_or_relative_root(root):
    with pytest.raises(ContractError, match="/fs04|absolute"):
        derive_durable_log_plan(root, "/fs04/scratch2/ce25/stageb-output")


def test_registration_rejects_rebound_durable_log_template(registration_bundle):
    malformed = copy.deepcopy(registration_bundle)
    malformed["scientific_log_plan"]["templates"]["finalizer"]["stdout"] = malformed[
        "scientific_log_plan"
    ]["templates"]["arm_o_gate"]["stdout"]
    with pytest.raises(ContractError, match="log"):
        freeze_registration_bundle(malformed)


def test_no_node_local_slurm_log_directive_remains():
    root = Path(__file__).parents[1] / "slurm"
    for path in root.glob("*.sbatch"):
        text = path.read_text(encoding="utf-8")
        assert "--output=/tmp" not in text
        assert "--error=/tmp" not in text
    submit = (root / "submit_chain.template.sh").read_text(encoding="utf-8")
    assert "--output=" in submit
    assert "--error=" in submit
    assert "/tmp" not in submit
