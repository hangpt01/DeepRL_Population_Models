"""Evaluator-only exact return evidence and deterministic reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .common import (
    ContractError,
    canonical_json_bytes,
    require_finite,
    require_sha256,
    sha256_bytes,
)
from .canonical_plan import (
    EXACT_RETURN_SCHEMA_VERSION,
    PAIRED_EVIDENCE_SCHEMA_VERSION,
    REGISTERED_EPISODE_IDS as _REGISTERED_EPISODE_IDS,
    REGISTERED_GAMMA as _REGISTERED_GAMMA,
    REGISTERED_HORIZON as _REGISTERED_HORIZON,
    REGISTERED_OBSERVATION_NOISE_SIGMA,
    REGISTERED_PROCESS_NOISE_SIGMA,
    RNG_RECEIPT_SCHEMA_VERSION,
    STEP_EVIDENCE_SCHEMA_VERSION,
    canonical_noise_sigma,
)
from .registration import ARMS, CELLS, METHODS


REGISTERED_HORIZON = _REGISTERED_HORIZON
REGISTERED_GAMMA = _REGISTERED_GAMMA
REGISTERED_EPISODE_IDS = _REGISTERED_EPISODE_IDS


@dataclass(frozen=True)
class RNGReceipt:
    schema_version: str
    process_noise_sigma: float
    observation_noise_sigma: float
    process_draw_required: bool
    observation_draw_required: bool
    process_state_advancement_applicable: bool
    observation_state_advancement_applicable: bool
    process_draw_invocations_before: int
    process_draw_invocations_after: int
    observation_draw_invocations_before: int
    observation_draw_invocations_after: int
    process_state_before_sha256: str
    process_state_after_sha256: str
    observation_state_before_sha256: str
    observation_state_after_sha256: str

    def validate(
        self,
        *,
        expected_process_noise_sigma: float = REGISTERED_PROCESS_NOISE_SIGMA,
        expected_observation_noise_sigma: float = REGISTERED_OBSERVATION_NOISE_SIGMA,
    ) -> None:
        if self.schema_version != RNG_RECEIPT_SCHEMA_VERSION:
            raise ContractError("RNG receipt schema version mismatch")
        process_sigma = canonical_noise_sigma(self.process_noise_sigma, "process noise sigma")
        observation_sigma = canonical_noise_sigma(
            self.observation_noise_sigma, "observation noise sigma"
        )
        expected_process = canonical_noise_sigma(
            expected_process_noise_sigma, "registered process noise sigma"
        )
        expected_observation = canonical_noise_sigma(
            expected_observation_noise_sigma, "registered observation noise sigma"
        )
        if process_sigma.hex() != expected_process.hex():
            raise ContractError("process noise sigma differs from the registered binding")
        if observation_sigma.hex() != expected_observation.hex():
            raise ContractError("observation noise sigma differs from the registered binding")
        expected_process_required = process_sigma > 0.0
        expected_observation_required = observation_sigma > 0.0
        booleans = (
            ("process_draw_required", self.process_draw_required, expected_process_required),
            (
                "observation_draw_required",
                self.observation_draw_required,
                expected_observation_required,
            ),
            (
                "process_state_advancement_applicable",
                self.process_state_advancement_applicable,
                expected_process_required,
            ),
            (
                "observation_state_advancement_applicable",
                self.observation_state_advancement_applicable,
                expected_observation_required,
            ),
        )
        for field, observed, expected in booleans:
            if not isinstance(observed, bool) or observed is not expected:
                raise ContractError(f"{field} contradicts the registered noise binding")
        counts = (
            self.process_draw_invocations_before,
            self.process_draw_invocations_after,
            self.observation_draw_invocations_before,
            self.observation_draw_invocations_after,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts
        ):
            raise ContractError("RNG draw-invocation counts must be nonnegative integers")
        process_delta = self.process_draw_invocations_after - self.process_draw_invocations_before
        observation_delta = (
            self.observation_draw_invocations_after - self.observation_draw_invocations_before
        )
        if process_delta != int(expected_process_required):
            raise ContractError("process RNG draw-invocation count contradicts noise semantics")
        if observation_delta != int(expected_observation_required):
            raise ContractError("observation RNG draw-invocation count contradicts noise semantics")
        for field in (
            "process_state_before_sha256",
            "process_state_after_sha256",
            "observation_state_before_sha256",
            "observation_state_after_sha256",
        ):
            require_sha256(getattr(self, field), field)
        process_changed = self.process_state_before_sha256 != self.process_state_after_sha256
        observation_changed = (
            self.observation_state_before_sha256 != self.observation_state_after_sha256
        )
        if process_changed is not expected_process_required:
            condition = (
                "advanced when advancement was not applicable"
                if process_changed
                else "did not advance"
            )
            raise ContractError(f"process RNG state {condition}")
        if observation_changed is not expected_observation_required:
            condition = (
                "advanced when advancement was not applicable"
                if observation_changed
                else "did not advance"
            )
            raise ContractError(f"observation RNG state {condition}")


@dataclass(frozen=True)
class StepEvidence:
    schema_version: str
    registration_sha256: str
    method: str
    cell: str
    arm: str
    episode_id: int
    timestep: int
    current_true_abundance: float
    noisy_observation: float
    selected_action: int
    benefit_population_term: float
    action_cost_term: float
    safety_penalty_term: float
    total_reward: float
    discount_factor: float
    discounted_benefit_population: float
    discounted_action_cost: float
    discounted_safety_penalty: float
    discounted_total_reward: float
    collapse_indicator: bool
    unsafe_indicator: bool
    process_innovation_sha256: str
    observation_innovation_sha256: str
    rng_receipt: RNGReceipt

    def validate(
        self,
        *,
        gamma: float,
        expected_timestep: int,
        expected_method: str,
        expected_cell: str,
        expected_arm: str,
        expected_episode_id: int,
        expected_registration_sha256: str,
        expected_process_noise_sigma: float,
        expected_observation_noise_sigma: float,
    ) -> None:
        if self.schema_version != STEP_EVIDENCE_SCHEMA_VERSION:
            raise ContractError("step-evidence schema version mismatch")
        require_sha256(self.registration_sha256, "evidence registration")
        if self.registration_sha256 != expected_registration_sha256:
            raise ContractError("evidence registration binding mismatch")
        if self.method != expected_method or self.method not in METHODS:
            raise ContractError("evidence method binding mismatch")
        if self.cell != expected_cell or self.cell not in CELLS:
            raise ContractError("evidence cell binding mismatch")
        if self.arm != expected_arm or self.arm not in ARMS:
            raise ContractError("evidence arm binding mismatch")
        if self.timestep != expected_timestep:
            raise ContractError("evidence timesteps must be complete and ordered")
        if isinstance(self.episode_id, bool) or not isinstance(self.episode_id, int):
            raise ContractError("episode identity must be an integer")
        if self.episode_id != expected_episode_id:
            raise ContractError("evidence episode identity changed within the sequence")
        if isinstance(self.selected_action, bool) or not isinstance(self.selected_action, int):
            raise ContractError("selected action must be an integer")
        if not 0 <= self.selected_action < 11:
            raise ContractError("selected action is outside the registered action set")
        if not isinstance(self.collapse_indicator, bool) or not isinstance(
            self.unsafe_indicator, bool
        ):
            raise ContractError("collapse and unsafe indicators must be booleans")
        numeric_fields = (
            "current_true_abundance",
            "noisy_observation",
            "benefit_population_term",
            "action_cost_term",
            "safety_penalty_term",
            "total_reward",
            "discount_factor",
            "discounted_benefit_population",
            "discounted_action_cost",
            "discounted_safety_penalty",
            "discounted_total_reward",
        )
        values = {field: require_finite(getattr(self, field), field) for field in numeric_fields}
        if values["current_true_abundance"] < 0.0 or values["noisy_observation"] < 0.0:
            raise ContractError("abundance and observation must be nonnegative")
        expected_total = (
            values["benefit_population_term"]
            + values["action_cost_term"]
            + values["safety_penalty_term"]
        )
        if not np.isclose(values["total_reward"], expected_total, rtol=0.0, atol=1e-12):
            raise ContractError("total reward does not reconstruct from components")
        expected_discount = float(np.float64(gamma) ** self.timestep)
        if not np.isclose(values["discount_factor"], expected_discount, rtol=0.0, atol=1e-15):
            raise ContractError("discount factor does not match gamma**timestep")
        pairs = (
            ("benefit_population_term", "discounted_benefit_population"),
            ("action_cost_term", "discounted_action_cost"),
            ("safety_penalty_term", "discounted_safety_penalty"),
            ("total_reward", "discounted_total_reward"),
        )
        for raw_field, discounted_field in pairs:
            expected = values[raw_field] * values["discount_factor"]
            if not np.isclose(values[discounted_field], expected, rtol=0.0, atol=1e-12):
                raise ContractError(f"{discounted_field} does not reconstruct")
        require_sha256(self.process_innovation_sha256, "process innovation receipt")
        require_sha256(self.observation_innovation_sha256, "observation innovation receipt")
        self.rng_receipt.validate(
            expected_process_noise_sigma=expected_process_noise_sigma,
            expected_observation_noise_sigma=expected_observation_noise_sigma,
        )

    def evaluator_only_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_layer"] = "EVALUATOR_ONLY"
        return value


def reconstruct_episode(
    records: Sequence[StepEvidence],
    *,
    expected_horizon: int = REGISTERED_HORIZON,
    expected_gamma: float = REGISTERED_GAMMA,
    expected_episode_ids: Sequence[int] = REGISTERED_EPISODE_IDS,
    expected_process_noise_sigma: float = REGISTERED_PROCESS_NOISE_SIGMA,
    expected_observation_noise_sigma: float = REGISTERED_OBSERVATION_NOISE_SIGMA,
) -> dict[str, Any]:
    if expected_horizon != REGISTERED_HORIZON or not np.float64(expected_gamma).view(
        np.uint64
    ) == np.float64(REGISTERED_GAMMA).view(np.uint64):
        raise ContractError("return reconstruction must use the frozen horizon and discount")
    if tuple(expected_episode_ids) != REGISTERED_EPISODE_IDS:
        raise ContractError("return reconstruction episode registry mismatch")
    if len(records) != expected_horizon:
        raise ContractError("exact evidence must contain the complete registered horizon")
    if not records or any(not isinstance(record, StepEvidence) for record in records):
        raise ContractError("exact evidence must contain only StepEvidence records")
    episode_ids = {record.episode_id for record in records}
    if len(episode_ids) != 1:
        raise ContractError("episode evidence mixes identities")
    episode_id = next(iter(episode_ids))
    if episode_id not in expected_episode_ids:
        raise ContractError("episode identity is outside the frozen evaluation registry")
    methods = {record.method for record in records}
    cells = {record.cell for record in records}
    arms = {record.arm for record in records}
    registrations = {record.registration_sha256 for record in records}
    if any(len(values) != 1 for values in (methods, cells, arms, registrations)):
        raise ContractError("episode evidence mixes method/cell/arm/registration identities")
    method = next(iter(methods))
    cell = next(iter(cells))
    arm = next(iter(arms))
    registration_sha = next(iter(registrations))
    collapse_seen = False
    for index, record in enumerate(records):
        record.validate(
            gamma=expected_gamma,
            expected_timestep=index,
            expected_method=method,
            expected_cell=cell,
            expected_arm=arm,
            expected_episode_id=episode_id,
            expected_registration_sha256=registration_sha,
            expected_process_noise_sigma=expected_process_noise_sigma,
            expected_observation_noise_sigma=expected_observation_noise_sigma,
        )
        if collapse_seen and not record.collapse_indicator:
            raise ContractError("collapse indicator cannot revert after first entry")
        collapse_seen = collapse_seen or record.collapse_indicator
        if index:
            previous = records[index - 1].rng_receipt
            current = record.rng_receipt
            if previous.process_draw_invocations_after != current.process_draw_invocations_before:
                raise ContractError("process RNG draw-invocation sequence is discontinuous")
            if (
                previous.observation_draw_invocations_after
                != current.observation_draw_invocations_before
            ):
                raise ContractError("observation RNG draw-invocation sequence is discontinuous")
            if previous.process_state_after_sha256 != current.process_state_before_sha256:
                raise ContractError("process RNG state receipt sequence is discontinuous")
            if previous.observation_state_after_sha256 != current.observation_state_before_sha256:
                raise ContractError("observation RNG state receipt sequence is discontinuous")
    total = float(np.sum([record.discounted_total_reward for record in records], dtype=np.float64))
    benefit = float(
        np.sum([record.discounted_benefit_population for record in records], dtype=np.float64)
    )
    action_cost = float(
        np.sum([record.discounted_action_cost for record in records], dtype=np.float64)
    )
    safety = float(
        np.sum([record.discounted_safety_penalty for record in records], dtype=np.float64)
    )
    if not np.isclose(total, benefit + action_cost + safety, rtol=0.0, atol=1e-10):
        raise ContractError("discounted episode return does not reconstruct from components")
    collapse_steps = [record.timestep for record in records if record.collapse_indicator]
    collapse_timing = min(collapse_steps) if collapse_steps else None
    if collapse_timing is None:
        pre_collapse = total
        post_collapse = 0.0
    else:
        pre_collapse = float(
            np.sum(
                [record.discounted_total_reward for record in records[:collapse_timing]],
                dtype=np.float64,
            )
        )
        post_collapse = float(
            np.sum(
                [record.discounted_total_reward for record in records[collapse_timing:]],
                dtype=np.float64,
            )
        )
    if not np.isclose(total, pre_collapse + post_collapse, rtol=0.0, atol=1e-10):
        raise ContractError("pre/post-collapse return decomposition failed")
    penalty_timing = [
        {
            "timestep": record.timestep,
            "raw_safety_penalty": record.safety_penalty_term,
            "discount_factor": record.discount_factor,
            "discounted_safety_penalty": record.discounted_safety_penalty,
        }
        for record in records
        if np.float64(record.safety_penalty_term).view(np.uint64) != np.float64(0.0).view(np.uint64)
    ]
    return {
        "schema_version": EXACT_RETURN_SCHEMA_VERSION,
        "evidence_layer": "EVALUATOR_ONLY",
        "registration_sha256": registration_sha,
        "method": method,
        "cell": cell,
        "arm": arm,
        "episode_id": episode_id,
        "horizon": expected_horizon,
        "gamma": expected_gamma,
        "total_discounted_return": total,
        "benefit_contribution": benefit,
        "action_cost_contribution": action_cost,
        "safety_contribution": safety,
        "pre_collapse_return": pre_collapse,
        "post_collapse_return": post_collapse,
        "collapse_timing": collapse_timing,
        "collapse_entry_step_partition": "post-collapse",
        "discounted_penalty_timing": penalty_timing,
        "unsafe_occupancy_count": sum(record.unsafe_indicator for record in records),
        "action_sequence": [record.selected_action for record in records],
        "process_rng_final_draw_invocation_count": (
            records[-1].rng_receipt.process_draw_invocations_after
        ),
        "observation_rng_final_draw_invocation_count": (
            records[-1].rng_receipt.observation_draw_invocations_after
        ),
        "reconstruction_result": "PASS",
    }


def pair_episode_evidence(
    arm_o_records: Sequence[StepEvidence],
    arm_t_records: Sequence[StepEvidence],
) -> dict[str, Any]:
    """Require exact evaluator RNG and innovation pairing across O/T."""

    receipt_o = reconstruct_episode(arm_o_records)
    receipt_t = reconstruct_episode(arm_t_records)
    identity_fields = ("registration_sha256", "method", "cell", "episode_id", "horizon", "gamma")
    if any(receipt_o[field] != receipt_t[field] for field in identity_fields):
        raise ContractError("paired episode identity mismatch")
    if receipt_o["arm"] != "O" or receipt_t["arm"] != "T":
        raise ContractError("paired evidence must be ordered Arm O then Arm T")
    process_hashes: list[str] = []
    observation_hashes: list[str] = []
    rng_receipt_hashes: list[str] = []
    for record_o, record_t in zip(arm_o_records, arm_t_records):
        if record_o.timestep != record_t.timestep:
            raise ContractError("paired evidence timestep mismatch")
        if (
            record_o.process_innovation_sha256 != record_t.process_innovation_sha256
            or record_o.observation_innovation_sha256 != record_t.observation_innovation_sha256
            or record_o.rng_receipt != record_t.rng_receipt
        ):
            raise ContractError("paired evaluator RNG/innovation identity mismatch")
        process_hashes.append(
            require_sha256(record_o.process_innovation_sha256, "paired process innovation")
        )
        observation_hashes.append(
            require_sha256(record_o.observation_innovation_sha256, "paired observation innovation")
        )
        rng_receipt_hashes.append(sha256_bytes(canonical_json_bytes(asdict(record_o.rng_receipt))))
    return {
        "schema_version": PAIRED_EVIDENCE_SCHEMA_VERSION,
        "registration_sha256": receipt_o["registration_sha256"],
        "method": receipt_o["method"],
        "cell": receipt_o["cell"],
        "episode_id": receipt_o["episode_id"],
        "timesteps": list(range(REGISTERED_HORIZON)),
        "process_innovation_sha256": process_hashes,
        "observation_innovation_sha256": observation_hashes,
        "rng_receipt_sha256": rng_receipt_hashes,
        "pairing_result": "PASS",
    }


METHOD_VISIBLE_FIELDS = frozenset(
    {"noisy_observation", "selected_action", "public_observation_history", "public_action_history"}
)
EVALUATOR_ONLY_FIELDS = frozenset(
    {
        "current_true_abundance",
        "benefit_population_term",
        "action_cost_term",
        "safety_penalty_term",
        "total_reward",
        "discounted_benefit_population",
        "discounted_action_cost",
        "discounted_safety_penalty",
        "discounted_total_reward",
        "collapse_indicator",
        "unsafe_indicator",
        "process_innovation_sha256",
        "observation_innovation_sha256",
        "rng_receipt",
    }
)


def validate_method_input_payload(payload: Mapping[str, Any]) -> None:
    from .boundary import reject_forbidden_payload

    reject_forbidden_payload(payload)
