from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ..common import ContractError
from ..diagnostics import (
    build_realized_posterior_receipt,
    build_refplan_predictive_dispersion,
    build_transition_diagnostics,
    classify_activity,
    guarded_ratio,
    pair_posterior_receipts,
)
from ..evidence import pair_episode_evidence, reconstruct_episode, validate_method_input_payload
from .conftest import HASHES, step_records


def test_transition_receipt_has_raw_aggregate_floor_ratio_and_hashes():
    receipt = build_transition_diagnostics(
        registration_sha256=HASHES[0],
        method="bamcts",
        cell="amur_tiger__allee__sigma_0p2",
        arm_o_values=[0.02, 0.04, 0.06, 0.08, 0.1],
        arm_t_values=[0.02, 0.03, 0.09, 0.08, 0.11],
        arm_o_artifact_hashes=HASHES[:5],
        arm_t_artifact_hashes=HASHES[5:10],
    )
    assert receipt["arm_o"]["floor_active"] == [True, False, False, False, False]
    assert receipt["raw_O"] == [0.02, 0.04, 0.06, 0.08, 0.1]
    assert receipt["T_minus_O"][1] == pytest.approx(-0.01)
    assert receipt["T_over_O_guarded"][0]["value"] == 1.0


def test_missing_residuals_and_below_floor_rejected():
    with pytest.raises(ContractError, match="member count|missing"):
        build_transition_diagnostics(
            registration_sha256=HASHES[0],
            method="refplan",
            cell="crab_eating_fox__allee__sigma_0p2",
            arm_o_values=[],
            arm_t_values=[],
            arm_o_artifact_hashes=[],
            arm_t_artifact_hashes=[],
        )
    with pytest.raises(ContractError, match="floor"):
        build_transition_diagnostics(
            registration_sha256=HASHES[0],
            method="refplan",
            cell="crab_eating_fox__allee__sigma_0p2",
            arm_o_values=[0.01] * 5,
            arm_t_values=[0.02] * 5,
            arm_o_artifact_hashes=list(HASHES[:5]),
            arm_t_artifact_hashes=list(HASHES[5:10]),
        )


def test_guarded_ratio_nonpositive_denominator_is_na():
    receipt = guarded_ratio(0.4, 0.0)
    assert receipt == {"value": None, "status": "NA_NONPOSITIVE_ARM_O_DENOMINATOR"}


def test_realized_posterior_requires_more_than_uniform_initial_entropy():
    with pytest.raises(ContractError, match="initial|timesteps"):
        build_realized_posterior_receipt(
            registration_sha256=HASHES[0],
            method="bamcts",
            cell="amur_tiger__allee__sigma_0p2",
            arm="O",
            episode_id=7001,
            member_ids=list(range(5)),
            timesteps=[0],
            model_bank_sha256=HASHES[1],
            posterior_probabilities=[[0.2] * 5],
        )


def test_realized_posterior_and_paired_summary():
    arm_o = build_realized_posterior_receipt(
        registration_sha256=HASHES[0],
        method="bamcts",
        cell="amur_tiger__allee__sigma_0p2",
        arm="O",
        episode_id=7001,
        member_ids=list(range(5)),
        timesteps=list(range(51)),
        model_bank_sha256=HASHES[1],
        posterior_probabilities=[[0.2] * 5] + [[0.6, 0.1, 0.1, 0.1, 0.1]] * 50,
    )
    arm_t = build_realized_posterior_receipt(
        registration_sha256=HASHES[0],
        method="bamcts",
        cell="amur_tiger__allee__sigma_0p2",
        arm="T",
        episode_id=7001,
        member_ids=list(range(5)),
        timesteps=list(range(51)),
        model_bank_sha256=HASHES[2],
        posterior_probabilities=[[0.2] * 5] + [[0.4, 0.15, 0.15, 0.15, 0.15]] * 50,
    )
    summary = pair_posterior_receipts(arm_o, arm_t)
    assert len(arm_o["per_step_entropy"]) == 50
    assert summary["terminal_entropy_T"] > summary["terminal_entropy_O"]


def test_nonfinite_posterior_and_dispersion_rejected():
    with pytest.raises(ContractError):
        build_realized_posterior_receipt(
            registration_sha256=HASHES[0],
            method="bamcts",
            cell="amur_tiger__allee__sigma_0p2",
            arm="O",
            episode_id=7001,
            member_ids=list(range(5)),
            timesteps=list(range(51)),
            model_bank_sha256=HASHES[1],
            posterior_probabilities=[[0.2] * 5] * 50 + [[np.nan] * 5],
        )
    with pytest.raises(ContractError):
        build_refplan_predictive_dispersion(
            registration_sha256=HASHES[0],
            cell="amur_tiger__allee__sigma_0p2",
            episode_id=7001,
            timesteps=list(range(50)),
            arm_o_artifact_sha256=HASHES[1],
            arm_t_artifact_sha256=HASHES[2],
            arm_o=[0.1] * 50,
            arm_t=[np.inf] * 50,
        )


def test_refplan_predictive_dispersion_is_paired_and_complete():
    receipt = build_refplan_predictive_dispersion(
        registration_sha256=HASHES[0],
        cell="amur_tiger__allee__sigma_0p2",
        episode_id=7001,
        timesteps=list(range(50)),
        arm_o_artifact_sha256=HASHES[1],
        arm_t_artifact_sha256=HASHES[2],
        arm_o=[0.2] * 50,
        arm_t=[0.1] * 50,
    )
    assert receipt["per_step_T_minus_O"] == pytest.approx([-0.1] * 50)


def test_exact_reward_reconstruction_and_pre_post_collapse_decomposition():
    receipt = reconstruct_episode(step_records(collapse_at=20))
    assert receipt["collapse_timing"] == 20
    assert receipt["collapse_entry_step_partition"] == "post-collapse"
    assert receipt["total_discounted_return"] == pytest.approx(
        receipt["pre_collapse_return"] + receipt["post_collapse_return"]
    )
    assert receipt["safety_contribution"] == pytest.approx(
        sum(item["discounted_safety_penalty"] for item in receipt["discounted_penalty_timing"])
    )


def test_reward_component_mismatch_rejected():
    records = step_records()
    records[3] = replace(records[3], total_reward=records[3].total_reward + 1.0)
    with pytest.raises(ContractError, match="components"):
        reconstruct_episode(records)


def test_rng_call_mismatch_and_nonfinite_receipt_rejected():
    records = step_records()
    bad_rng = replace(records[1].rng_receipt, process_draw_invocations_after=9)
    records[1] = replace(records[1], rng_receipt=bad_rng)
    with pytest.raises(ContractError, match="process RNG"):
        reconstruct_episode(records)
    records = step_records()
    records[2] = replace(records[2], noisy_observation=np.nan)
    with pytest.raises(ContractError, match="finite"):
        reconstruct_episode(records)


def test_evaluator_only_reward_fields_never_enter_method_inputs():
    validate_method_input_payload(
        {
            "noisy_observation": 1.0,
            "selected_action": 2,
            "public_observation_history": [1.0],
            "public_action_history": [],
        }
    )
    with pytest.raises(ContractError, match="forbidden"):
        validate_method_input_payload({"noisy_observation": 1.0, "safety_penalty_term": -10.0})


def test_activity_is_derived_from_complete_actions_not_prose():
    kwargs = {
        "registration_sha256": HASHES[0],
        "method": "refplan",
        "cell": "amur_tiger__allee__sigma_0p2",
        "arm": "O",
        "episode_id": 7001,
    }
    constant = classify_activity([2] * 50, **kwargs)
    assert constant["literal_constant_policy"]
    assert constant["discrimination_status"] == "NON_DISCRIMINATING_LITERAL_CONSTANT"
    near = classify_activity([0] * 49 + [1], **kwargs)
    assert not near["literal_constant_policy"]
    assert near["near_constant_descriptive"]
    assert near["distinct_action_count"] == 2


def test_incorrect_activity_inputs_rejected():
    kwargs = {
        "registration_sha256": HASHES[0],
        "method": "refplan",
        "cell": "amur_tiger__allee__sigma_0p2",
        "arm": "O",
        "episode_id": 7001,
    }
    with pytest.raises(ContractError):
        classify_activity([], **kwargs)
    with pytest.raises(ContractError):
        classify_activity([0] * 49 + [11], **kwargs)


def test_posterior_pair_rejects_member_and_length_mismatch():
    base = dict(
        registration_sha256=HASHES[0],
        method="bamcts",
        cell="amur_tiger__allee__sigma_0p2",
        episode_id=7001,
        member_ids=list(range(5)),
        timesteps=list(range(51)),
        posterior_probabilities=[[0.2] * 5] * 51,
    )
    arm_o = build_realized_posterior_receipt(arm="O", model_bank_sha256=HASHES[1], **base)
    altered = dict(arm_o)
    altered["arm"] = "T"
    altered["member_ids"] = [0, 1, 2, 3, 9]
    with pytest.raises(ContractError, match="mismatch"):
        pair_posterior_receipts(arm_o, altered)


def test_paired_evidence_requires_rng_and_innovation_identity():
    arm_o = step_records(arm="O")
    arm_t = step_records(arm="T")
    assert pair_episode_evidence(arm_o, arm_t)["pairing_result"] == "PASS"
    arm_t[3] = replace(arm_t[3], process_innovation_sha256=HASHES[12])
    with pytest.raises(ContractError, match="RNG/innovation"):
        pair_episode_evidence(arm_o, arm_t)


def test_nonboolean_or_reverting_collapse_indicators_rejected():
    records = step_records()
    records[21] = replace(records[21], collapse_indicator=False)
    with pytest.raises(ContractError, match="revert"):
        reconstruct_episode(records)
    records = step_records()
    records[1] = replace(records[1], unsafe_indicator=7)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="booleans"):
        reconstruct_episode(records)


def test_incorrect_timestep_and_method_cell_arm_episode_bindings_rejected():
    mutations = (
        (3, {"timestep": 4}),
        (3, {"method": "ogsrl"}),
        (3, {"cell": "crab_eating_fox__allee__sigma_0p2"}),
        (3, {"arm": "T"}),
        (3, {"episode_id": 7051}),
    )
    for index, fields in mutations:
        records = step_records()
        records[index] = replace(records[index], **fields)
        with pytest.raises(ContractError):
            reconstruct_episode(records)
