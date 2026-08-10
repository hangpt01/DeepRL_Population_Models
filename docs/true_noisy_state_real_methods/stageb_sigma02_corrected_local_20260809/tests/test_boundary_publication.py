from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pytest

from ..boundary import (
    PublicContext,
    RuntimeDatasetView,
    adapt_exact_features,
    reject_forbidden_payload,
    route_general_current_state,
    validate_external_runtime_request,
)
from ..common import ContractError
from ..publication import (
    SUCCESS_RECEIPT,
    create_task_staging,
    load_success_receipt,
    publish_once,
    validate_staging_tree,
    write_bytes_fsync,
)
from .conftest import HASHES


@dataclass
class FakeBelief:
    states: np.ndarray
    log_weights: np.ndarray
    regimes: np.ndarray
    contexts: np.ndarray
    observation: float
    diagnostics: dict[str, float]


def public_context() -> PublicContext:
    return PublicContext(
        features=np.ones((3, 10), dtype=np.float64),
        feature_names=(
            "mean",
            "sd",
            "q10",
            "q50",
            "q90",
            "extinct",
            "ctx_prev_obs",
            "ctx_obs",
            "ctx_t",
            "ess_frac",
        ),
        action_history=(1, 2),
        observation_history=(10.0, 11.0, 12.0),
        timestep=2,
        rng_call_count=9,
        rng_state_sha256=HASHES[0],
    )


def fake_belief() -> FakeBelief:
    return FakeBelief(
        states=np.array([10.0, 11.0, 12.0], dtype=np.float64),
        log_weights=np.log(np.array([0.2, 0.3, 0.5], dtype=np.float64)),
        regimes=np.array([0, 0, 0], dtype=np.int8),
        contexts=np.array([[8.0, 9.0, 1.0]] * 3, dtype=np.float64),
        observation=9.0,
        diagnostics={"public": 1.0},
    )


def test_exact_feature_overlay_preserves_history_dtype_timing_and_rng():
    before = public_context()
    after, receipt = adapt_exact_features(
        before,
        np.array([13.0, 14.0, 15.0], dtype=np.float64),
        observation_scale=100.0,
        observation_noise_sigma=0.2,
    )
    assert after.features.dtype == np.dtype("float64")
    assert after.action_history == before.action_history
    assert after.observation_history == before.observation_history
    assert after.timestep == before.timestep
    assert after.rng_call_count == before.rng_call_count
    assert receipt["preserved_context_sha256_before"] == receipt["preserved_context_sha256_after"]


def test_exact_feature_overlay_rejects_wrong_dtype_without_coercion():
    with pytest.raises(ContractError, match="already be float64"):
        adapt_exact_features(
            public_context(),
            np.array([13, 14, 15], dtype=np.int32),
            observation_scale=100.0,
            observation_noise_sigma=0.2,
        )


def test_context_deletion_reorder_and_dtype_mutation_rejected():
    context = public_context()
    with pytest.raises(ContractError, match="history length"):
        PublicContext(
            context.features,
            context.feature_names,
            (),
            context.observation_history,
            context.timestep,
            context.rng_call_count,
            context.rng_state_sha256,
        ).validate()
    with pytest.raises(ContractError, match="order"):
        PublicContext(
            context.features,
            tuple(reversed(context.feature_names)),
            context.action_history,
            context.observation_history,
            context.timestep,
            context.rng_call_count,
            context.rng_state_sha256,
        ).validate()
    with pytest.raises(ContractError, match="dtype"):
        PublicContext(
            context.features.astype(np.float32),
            context.feature_names,
            context.action_history,
            context.observation_history,
            context.timestep,
            context.rng_call_count,
            context.rng_state_sha256,
        ).validate()


@pytest.mark.parametrize(
    "payload",
    [
        {"next_states": [1.0]},
        {"nested": {"future_states": [1.0]}},
        {"family": "allee"},
        {"theta": 4.0},
        {"reward_components": {}},
        {"process_innovation": 0.3},
        {"original_truth_path": "/private/truth.npz"},
    ],
)
def test_information_leakage_payloads_rejected(payload):
    with pytest.raises(ContractError, match="forbidden"):
        reject_forbidden_payload(payload)


def test_runtime_next_state_and_original_truth_access_rejected():
    view = RuntimeDatasetView(current_observation=3.0, public_history=[1.0, 2.0, 3.0])
    with pytest.raises(ContractError, match="next_states"):
        view.get("next_states")
    with pytest.raises(ContractError, match="original truth"):
        view.open_original_truth(os.devnull)  # type: ignore[arg-type]


def test_sigma_zero_or_oracle_filter_rejected():
    with pytest.raises(ContractError, match="sigma"):
        validate_external_runtime_request(
            observation_noise_sigma=0.0,
            observation_scale=1.0,
            payload={},
            filter_class_name="PublicObservationFilter",
        )
    with pytest.raises(ContractError, match="Oracle"):
        validate_external_runtime_request(
            observation_noise_sigma=0.2,
            observation_scale=1.0,
            payload={},
            filter_class_name="OracleStateFilter",
        )


def test_general_current_state_route_has_no_next_state_api_and_preserves_context():
    before = fake_belief()
    emitted, receipt = route_general_current_state(
        before,
        13.0,
        observation_noise_sigma=0.2,
        observation_scale=1.0,
    )
    assert np.array_equal(emitted.states, np.full(3, 13.0, dtype=np.float64))
    assert np.array_equal(emitted.contexts, before.contexts)
    assert receipt["runtime_next_states"] == "IMPOSSIBLE_NO_ARGUMENT"


def test_atomic_publication_only_after_validation(tmp_path):
    staging, target = create_task_staging(tmp_path, "task-final")
    write_bytes_fsync(staging / "episodes.json", b"synthetic")
    receipt = publish_once(
        staging=staging,
        target=target,
        required_files=["episodes.json"],
        validator=lambda _: {"result": "PASS", "synthetic": True},
    )
    assert receipt == target / SUCCESS_RECEIPT
    assert target.is_dir()
    assert not staging.exists()
    assert load_success_receipt(receipt)["full_tree_coverage"] is True


def test_no_success_receipt_after_io_failure_and_no_retry(tmp_path, monkeypatch):
    staging, target = create_task_staging(tmp_path, "failed-final")
    (staging / "episodes.json").write_bytes(b"synthetic")

    def fail_fsync(_descriptor):
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="fsync"):
        publish_once(
            staging=staging,
            target=target,
            required_files=["episodes.json"],
            validator=lambda _: {"result": "PASS"},
        )
    assert not (staging / SUCCESS_RECEIPT).exists()
    assert not target.exists()


def test_publication_validation_failure_never_publishes(tmp_path):
    staging, target = create_task_staging(tmp_path, "invalid-final")
    (staging / "episodes.json").write_bytes(b"synthetic")
    with pytest.raises(ContractError, match="validation"):
        publish_once(
            staging=staging,
            target=target,
            required_files=["episodes.json"],
            validator=lambda _: {"result": "FAIL"},
        )
    assert not target.exists()
    assert not (staging / SUCCESS_RECEIPT).exists()


def test_nested_allowlist_values_and_private_belief_diagnostics_rejected():
    with pytest.raises(ContractError):
        reject_forbidden_payload(
            {
                "noisy_observation": 1.0,
                "selected_action": 1,
                "public_observation_history": [{"future_states": [99]}],
                "public_action_history": [],
            }
        )
    belief = fake_belief()
    belief.diagnostics = {"safety_threshold": 17.0}
    with pytest.raises(ContractError, match="diagnostics"):
        route_general_current_state(
            belief,
            13.0,
            observation_noise_sigma=0.2,
            observation_scale=1.0,
        )


@pytest.mark.parametrize("relative", ["../outside.dat", "/absolute.dat", "a/../b.dat"])
def test_publication_rejects_escaping_paths(tmp_path, relative):
    staging, _ = create_task_staging(tmp_path, "escape-final")
    (tmp_path / "outside.dat").write_bytes(b"outside")
    with pytest.raises(ContractError, match="path|coverage"):
        validate_staging_tree(staging, [relative])


def test_publication_rejects_extra_files_and_symlinked_ancestor(tmp_path):
    staging, _ = create_task_staging(tmp_path, "coverage-final")
    (staging / "required.dat").write_bytes(b"required")
    (staging / "extra.dat").write_bytes(b"extra")
    with pytest.raises(ContractError, match="coverage"):
        validate_staging_tree(staging, ["required.dat"])
    linked = staging / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ContractError, match="symlink"):
        validate_staging_tree(staging, ["required.dat", "extra.dat"])


def test_publication_rejects_duplicates_after_path_normalization(tmp_path):
    staging, _ = create_task_staging(tmp_path, "duplicate-final")
    (staging / "nested").mkdir()
    (staging / "nested" / "file.dat").write_bytes(b"value")
    with pytest.raises(ContractError, match="duplicate"):
        validate_staging_tree(staging, ["nested/file.dat", "nested//file.dat"])


def test_failure_after_each_prepublication_fsync_never_exposes_success(tmp_path, monkeypatch):
    from .. import publication

    for failure_call in range(1, 5):
        staging, target = create_task_staging(tmp_path, f"late-{failure_call}")
        (staging / "episodes.json").write_bytes(b"synthetic")
        real_fsync_directory = publication._fsync_directory
        calls = 0

        def fail_selected(path):
            nonlocal calls
            calls += 1
            if calls == failure_call:
                raise OSError("synthetic late failure")
            return real_fsync_directory(path)

        monkeypatch.setattr(publication, "_fsync_directory", fail_selected)
        if failure_call <= 3:
            with pytest.raises(OSError, match="late"):
                publish_once(
                    staging=staging,
                    target=target,
                    required_files=["episodes.json"],
                    validator=lambda _: {"result": "PASS"},
                )
            assert not (target / SUCCESS_RECEIPT).exists()
        else:
            # There are only three directory fsyncs; publication completes and the final
            # atomic rename has no later filesystem operation that can fail.
            receipt = publish_once(
                staging=staging,
                target=target,
                required_files=["episodes.json"],
                validator=lambda _: {"result": "PASS"},
            )
            assert receipt.is_file()
        monkeypatch.setattr(publication, "_fsync_directory", real_fsync_directory)
