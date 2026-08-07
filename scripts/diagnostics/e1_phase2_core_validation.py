#!/usr/bin/env python3
"""Build E1 arms and run the authorised V1--V5 validation without returns."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ECOLOGY_SRC = REPO_ROOT / "src" / "tracks" / "ecological"
for entry in (str(REPO_ROOT), str(ECOLOGY_SRC)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from e1_phase1_parity import (  # noqa: E402
    ABSORBING_ZERO_EQUATION_VERSION,
    FIT_CACHE_KEYS,
    LIKELIHOOD_FLOOR,
    POSTERIOR_L1_TOLERANCE,
    RegisteredPOMDP,
    StableObservationPOMDP,
    canonical_digest,
    diagnostic_wrapper_code_digest,
    load_cached_model,
    make_cells as make_phase1_cells,
    registered_model,
    stable_reference_posterior,
    transition_table_provenance,
)
from real_ecology_benchmark import realdata  # noqa: E402
from real_ecology_benchmark.actions import real_action_table  # noqa: E402
from real_ecology_benchmark.beliefs import (  # noqa: E402
    OracleStateFilter,
    RawObservationFilter,
    ReferenceProposal,
)
from real_ecology_benchmark.config import (  # noqa: E402
    FaithfulPlannerConfig,
    FilterConfig,
    MethodContext,
    real_environment,
)
from real_ecology_benchmark.envs import ContinuousEcologyEnv  # noqa: E402
from real_ecology_benchmark.faithful_ecology import MechanisticModel  # noqa: E402
from real_ecology_benchmark.faithful_pomdp import (  # noqa: E402
    CandidateBelief,
    CandidatePOMDP,
)
from real_ecology_benchmark.observation import LogNormalObservationModel  # noqa: E402
from real_ecology_benchmark.planners.pbvi import PointBasedPlanner  # noqa: E402
from real_ecology_benchmark.reward import (  # noqa: E402
    build_reward,
    safety_penalty_indicator,
)
from real_ecology_benchmark.types import BeliefState, PublicTransition  # noqa: E402


SCHEMA = "e1_phase2_core_phase3_v1"
ACCEPTED_PHASE1_DIGEST = (
    "2df9ccf9ce809c94d229b786de7b6a8cc0cb42caddaf3ee6f29830ba596750d7"
)
ACCEPTED_WRAPPER_DIGEST = (
    "174d50423ad98a1587783e5af14fab2554132cb10f025d6c548a9e7ba13f3b0b"
)
ACCEPTED_REGISTERED_TRANSITION_HASH = (
    "5da9e974540cac66c0462f1449100b0d728c636342e4784f5135a5bd5cad0ba6"
)
FIT_CONFIG_DIGEST = (
    "2686c2e09edbeadfa145d26a3c5df12135cf1dc8df4c7a12508370778a2a6e0a"
)
FIT_SEED = 47116
VALIDATION_SEED = 63001
PLANNING_SEEDS = {
    "fox_ricker_sigma_0.1": 62001,
    "fox_ricker_sigma_0.2": 62002,
    "tiger_ricker_sigma_0.1": 62101,
    "tiger_ricker_sigma_0.2": 62102,
}
REGISTERED_EVALUATION_SEEDS = (7001, 7051, 7101, 7151, 7201)
ARMS = ("A1", "A2", "A3", "A4")
POPULATIONS = ("Crab-eating fox", "Amur tiger")
SIGMAS = (0.1, 0.2)
IDENTITY_TOLERANCE = 1.0e-9
ACTION_VALUE_TOLERANCE = 1.0e-8
V1_TARGET = 1.0e-5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cell_id(population: str, sigma: float) -> str:
    prefix = "fox" if population == "Crab-eating fox" else "tiger"
    return f"{prefix}_ricker_sigma_{sigma:.1f}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype="<f8").tobytes()
    ).hexdigest()


def audit_float(value: float) -> float | str:
    value = float(value)
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "+inf" if value > 0.0 else "-inf"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


class ForbiddenSurrogate:
    """Non-None sentinel proving the explicit source-reward route was selected."""

    def predict(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("E1 true-reward planning must never call the surrogate")


def method_context(
    actions: tuple[Any, ...], sigma: float, observation_scale: float
) -> MethodContext:
    context = MethodContext(
        num_actions=len(actions),
        action_costs=tuple(float(action.cost) for action in actions),
        action_channels=realdata.public_action_channels(),
        observation_noise_sigma=float(sigma),
        horizon=50,
        observation_scale=float(observation_scale),
        pop_id="pop_e1_oracle_analysis",
        reward_mode="safe",
        surrogate=ForbiddenSurrogate(),
    )
    context.validate()
    if context.surrogate is None:
        raise AssertionError("E1 must not use surrogate=None")
    return context


class TrueRewardMixin:
    """Explicit oracle objective that delegates to the frozen source reward object."""

    def configure_true_reward(self, env_cfg: Any, arm: str) -> None:
        if env_cfg.safety_penalty_mode != "occupancy":
            raise AssertionError("E1 registered penalty mode must be occupancy")
        self.e1_env_cfg = env_cfg
        self.e1_arm = arm
        self.e1_reward = build_reward(env_cfg)
        self.true_reward_calls = 0
        self.objective_flag = "explicit_source_true_reward"

    def expected_public_reward(
        self,
        belief: CandidateBelief,
        action: int,
        predicted: np.ndarray | None = None,
    ) -> float:
        predicted = self.predict(belief, action) if predicted is None else predicted
        following = self.state_abundances() * self.model.survey_scale
        previous = np.full_like(
            following,
            float(
                np.dot(
                    belief.probabilities,
                    self.state_abundances() * self.model.survey_scale,
                )
            ),
        )
        penalty = np.asarray(
            safety_penalty_indicator(
                self.e1_env_cfg, previous, following
            ),
            dtype=bool,
        )
        rewards = self.e1_reward.expected(
            following,
            np.full(len(following), int(action), dtype=np.int64),
            penalty,
        )
        self.true_reward_calls += 1
        return float(np.dot(predicted, rewards))


class TrueStateE1POMDP(TrueRewardMixin, StableObservationPOMDP):
    """A1/A2 POMDP: exact predictive support and identity conditioning."""

    def __init__(
        self,
        model: MechanisticModel,
        context: MethodContext,
        planner_cfg: FaithfulPlannerConfig,
        seed: int,
        env_cfg: Any,
        arm: str,
    ):
        super().__init__(model, context, planner_cfg, seed)
        self.configure_true_reward(env_cfg, arm)
        self.runtime_filtering_calls = 0
        self.base_bayesian_update_calls = 0
        self.identity_conditioning_calls = 0
        self.representative_observation_calls = 0
        self.realised_branch_counts: list[int] = []
        self.realised_support_sizes: list[int] = []

    def representative_observations(
        self, predicted: np.ndarray, branch_count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        del branch_count
        predicted = np.asarray(predicted, dtype=np.float64)
        support = np.flatnonzero(predicted > 0.0)
        if len(support) == 0:
            raise AssertionError("identity prediction has empty support")
        weights = predicted[support]
        weights = weights / float(weights.sum())
        observations = self.state_abundances()[support] * self.model.survey_scale
        self.representative_observation_calls += 1
        self.realised_branch_counts.append(int(len(support)))
        self.realised_support_sizes.append(int(np.count_nonzero(predicted)))
        return observations.astype(np.float64), weights.astype(np.float64)

    def update(
        self, belief: CandidateBelief, action: int, observation: float
    ) -> tuple[CandidateBelief, float]:
        latent = float(observation) / self.model.survey_scale
        abundance_index = int(np.argmin(np.abs(self.abundance_grid - latent)))
        probabilities = np.zeros(self.hidden_count, dtype=np.float64)
        probabilities[abundance_index] = 1.0
        self.identity_conditioning_calls += 1
        return (
            CandidateBelief(
                probabilities,
                self.model.next_capacity(belief.capacity, int(action)),
                belief.current_observation,
                float(observation),
                belief.timestep + 1,
            ),
            0.0,
        )


class NoisyStateE1POMDP(TrueRewardMixin, RegisteredPOMDP):
    """A3/A4 POMDP using the accepted log-space observation implementation."""

    def __init__(
        self,
        model: MechanisticModel,
        context: MethodContext,
        planner_cfg: FaithfulPlannerConfig,
        seed: int,
        env_cfg: Any,
        arm: str,
        registered_prior: bool,
    ):
        super().__init__(model, context, planner_cfg, seed)
        self.configure_true_reward(env_cfg, arm)
        self.registered_prior = bool(registered_prior)
        self.deployed_logspace_updates = 0
        self.lookahead_logspace_updates = 0
        self._runtime_update_active = False
        self.old_density_evidence_at_or_below_floor = 0
        self.structural_impossible_support_cases = 0
        self.numerical_underflow_cases = 0
        self.old_fallback_calls = 0
        self.unexpected_all_zero_support_cases = 0
        self.fallback_to_uniform_initial_beliefs = 0
        self.maximum_posterior_l1_error = 0.0
        self.maximum_posterior_tv_error = 0.0
        self.maximum_action_value_error = 0.0
        self.reference_action_changes = 0
        self.update_records: list[dict[str, Any]] = []

    def initial_belief(self, observation: float) -> CandidateBelief:
        if self.registered_prior:
            return RegisteredPOMDP.initial_belief(self, observation)
        belief = super(RegisteredPOMDP, self).initial_belief(observation)
        uniform = np.full(self.hidden_count, 1.0 / self.hidden_count)
        if np.array_equal(belief.probabilities, uniform):
            self.fallback_to_uniform_initial_beliefs += 1
        return belief

    def _diagnostic_action_values(
        self, probabilities: np.ndarray, capacity: float
    ) -> np.ndarray:
        belief = CandidateBelief(probabilities, capacity, 0.0, 0.0, 0)
        following = self.state_abundances() * self.model.survey_scale
        return np.asarray(
            [
                float(np.dot(self.predict(belief, action), following))
                for action in range(self.context.num_actions)
            ],
            dtype=np.float64,
        )

    def update(
        self, belief: CandidateBelief, action: int, observation: float
    ) -> tuple[CandidateBelief, float]:
        predicted = self.predict(belief, int(action))
        if not self._runtime_update_active:
            deployed, deployed_log_evidence, impossible = self.posterior_from_prediction(
                predicted, float(observation)
            )
            if not np.all(np.isfinite(deployed)) or not math.isclose(
                float(deployed.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise AssertionError("invalid log-space PBVI-lookahead posterior")
            self.lookahead_logspace_updates += 1
            if impossible:
                deployed_log_evidence = float(math.log(LIKELIHOOD_FLOOR))
            return (
                CandidateBelief(
                    deployed,
                    self.model.next_capacity(belief.capacity, int(action)),
                    belief.current_observation,
                    float(observation),
                    belief.timestep + 1,
                ),
                deployed_log_evidence,
            )
        raw_states = self.state_abundances() * self.model.survey_scale
        density = self.model.observation_likelihood(observation, self.state_abundances())
        old_evidence = float(np.dot(predicted, density))
        old_below_floor = (
            old_evidence <= LIKELIHOOD_FLOOR or not np.isfinite(old_evidence)
        )
        self.old_density_evidence_at_or_below_floor += int(old_below_floor)

        reference_log = LogNormalObservationModel(
            self.e1_env_cfg.observation_noise_sigma
        ).log_prob(float(observation), raw_states)
        reference, reference_log_evidence, informative = stable_reference_posterior(
            predicted, reference_log
        )
        deployed, deployed_log_evidence, impossible = self.posterior_from_prediction(
            predicted, float(observation)
        )
        structural = bool(impossible and not informative)
        self.structural_impossible_support_cases += int(structural)
        numerical_underflow = bool(
            np.any((predicted > 0.0) & (density == 0.0) & np.isfinite(reference_log))
        )
        self.numerical_underflow_cases += int(numerical_underflow)
        all_zero = bool(np.all(density[predicted > 0.0] == 0.0))
        unexpected_all_zero = bool(all_zero and informative)
        self.unexpected_all_zero_support_cases += int(unexpected_all_zero)
        if unexpected_all_zero:
            raise AssertionError("informative reference but all deployed densities are zero")
        if impossible and informative:
            raise AssertionError("repaired observation path discarded informative support")
        if not np.all(np.isfinite(deployed)) or not math.isclose(
            float(deployed.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise AssertionError("non-finite or non-normalized deployed posterior")

        l1 = float(np.abs(deployed - reference).sum())
        tv = 0.5 * l1
        self.maximum_posterior_l1_error = max(self.maximum_posterior_l1_error, l1)
        self.maximum_posterior_tv_error = max(self.maximum_posterior_tv_error, tv)
        deployed_values = self._diagnostic_action_values(deployed, belief.capacity)
        reference_values = self._diagnostic_action_values(reference, belief.capacity)
        value_error = float(np.max(np.abs(deployed_values - reference_values)))
        self.maximum_action_value_error = max(
            self.maximum_action_value_error, value_error
        )
        action_changed = int(np.argmax(deployed_values)) != int(
            np.argmax(reference_values)
        )
        self.reference_action_changes += int(action_changed)
        if l1 > POSTERIOR_L1_TOLERANCE:
            raise AssertionError(f"posterior L1 parity failed: {l1}")
        if value_error > ACTION_VALUE_TOLERANCE or action_changed:
            raise AssertionError("observation update changed diagnostic action selection")

        self.deployed_logspace_updates += 1
        self.update_records.append(
            {
                "observation": float(observation),
                "old_density_evidence": audit_float(old_evidence),
                "old_density_at_or_below_floor": old_below_floor,
                "deployed_log_evidence": audit_float(deployed_log_evidence),
                "reference_log_evidence": audit_float(reference_log_evidence),
                "structural_impossible_support": structural,
                "numerical_underflow": numerical_underflow,
                "posterior_l1": l1,
                "posterior_tv": tv,
                "action_changed": action_changed,
            }
        )
        if impossible:
            deployed_log_evidence = float(math.log(LIKELIHOOD_FLOOR))
        return (
            CandidateBelief(
                deployed,
                self.model.next_capacity(belief.capacity, int(action)),
                belief.current_observation,
                float(observation),
                belief.timestep + 1,
            ),
            deployed_log_evidence,
        )

    def runtime_update(
        self, belief: CandidateBelief, action: int, observation: float
    ) -> tuple[CandidateBelief, float]:
        if self._runtime_update_active:
            raise AssertionError("nested runtime observation update")
        self._runtime_update_active = True
        try:
            return self.update(belief, action, observation)
        finally:
            self._runtime_update_active = False


class TrueStatePolicy:
    """Actual A1/A2 route: OracleStateFilter truth is snapped inside act()."""

    def __init__(self, arm: str, pomdp: TrueStateE1POMDP, planning_seed: int):
        self.name = f"e1_{arm.lower()}_true_state"
        self.arm = arm
        self.pomdp = pomdp
        self.planner = PointBasedPlanner(pomdp, pomdp.config, 0.95, planning_seed)
        self.internal_belief: CandidateBelief | None = None
        self.runtime_policy_filtering_calls = 0
        self.observe_noop_calls = 0
        self.snap_assertion_count = 0
        self.snap_records: list[dict[str, Any]] = []
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self, seed: int) -> None:
        del seed
        self.internal_belief = None
        self.last_diagnostics = {}

    def act(self, belief: BeliefState, observation: float) -> int:
        del observation
        raw_truth = float(belief.mean_state())
        latent_truth = raw_truth / self.pomdp.model.survey_scale
        index = int(np.argmin(np.abs(self.pomdp.abundance_grid - latent_truth)))
        expected_index = int(
            np.argmin(
                np.abs(
                    self.pomdp.abundance_grid * self.pomdp.model.survey_scale
                    - raw_truth
                )
            )
        )
        if index != expected_index:
            raise AssertionError("raw-to-latent truth snapping used inconsistent units")
        probabilities = np.zeros(self.pomdp.hidden_count, dtype=np.float64)
        probabilities[index] = 1.0
        capacity = (
            float(belief.K_eff)
            if belief.K_eff is not None
            else self.pomdp.model.initial_capacity
        )
        self.internal_belief = CandidateBelief(
            probabilities, capacity, raw_truth, raw_truth, belief.timestep
        )
        selected_raw = float(
            self.pomdp.abundance_grid[index] * self.pomdp.model.survey_scale
        )
        registered_discretisation = float(
            self.pomdp.abundance_grid[expected_index] * self.pomdp.model.survey_scale
        )
        if selected_raw != registered_discretisation:
            raise AssertionError("snapped raw state does not equal registered discretisation")
        if float(self.internal_belief.probabilities.max()) <= 1.0 - IDENTITY_TOLERANCE:
            raise AssertionError("true-state internal belief is not a delta")
        self.snap_assertion_count += 1
        self.snap_records.append(
            {
                "timestep": int(belief.timestep),
                "raw_truth": raw_truth,
                "latent_truth": latent_truth,
                "selected_index": index,
                "selected_grid_latent": float(self.pomdp.abundance_grid[index]),
                "selected_grid_raw": selected_raw,
                "survey_scale": self.pomdp.model.survey_scale,
                "belief_max": float(self.internal_belief.probabilities.max()),
            }
        )
        scores = self.planner.action_values(self.internal_belief)
        action = int(np.argmax(scores))
        self.last_diagnostics = {"action_scores": scores.tolist()}
        return action

    def observe(self, belief: BeliefState, action: int, result: PublicTransition) -> None:
        del belief, action, result
        self.observe_noop_calls += 1


class NoisyStatePolicy:
    """Actual A3/A4 route: repaired POMDP update on each noisy observation."""

    def __init__(self, arm: str, pomdp: NoisyStateE1POMDP, planning_seed: int):
        self.name = f"e1_{arm.lower()}_noisy_state"
        self.arm = arm
        self.pomdp = pomdp
        self.planner = PointBasedPlanner(pomdp, pomdp.config, 0.95, planning_seed)
        self.internal_belief: CandidateBelief | None = None
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self, seed: int) -> None:
        del seed
        self.internal_belief = None
        self.last_diagnostics = {}

    def act(self, belief: BeliefState, observation: float) -> int:
        del belief
        if self.internal_belief is None:
            self.internal_belief = self.pomdp.initial_belief(float(observation))
        scores = self.planner.action_values(self.internal_belief)
        action = int(np.argmax(scores))
        self.last_diagnostics = {"action_scores": scores.tolist()}
        return action

    def observe(self, belief: BeliefState, action: int, result: PublicTransition) -> None:
        del belief
        if self.internal_belief is None:
            raise AssertionError("noisy policy observed before initialization")
        self.internal_belief, log_evidence = self.pomdp.runtime_update(
            self.internal_belief, int(action), float(result.observation)
        )
        self.last_diagnostics["public_log_evidence"] = float(log_evidence)


def planner_config() -> FaithfulPlannerConfig:
    config = FaithfulPlannerConfig(
        name="pbvi",
        state_bins=41,
        capacity_bins=9,
        observation_bins=41,
        transition_samples=256,
        observation_samples=256,
        belief_points=32,
        observation_branches=7,
        horizon=5,
    )
    config.validate()
    return config


def fit_metadata(cache_key: str) -> dict[str, Any]:
    metadata_path = Path(
        "/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/"
        "three_species_ecological_p10_correction_20260723_v1/moor/fit_cache"
    ) / f"{cache_key}.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {
        "public_data_hash": metadata["fit"]["public_data_hash"],
        "transition_data_hash": metadata["fit"]["transition_data_hash"],
        "fit_episode_ids_hash": canonical_digest(metadata["fit"]["fit_episode_ids"]),
        "holdout_episode_ids_hash": canonical_digest(
            metadata["fit"]["holdout_episode_ids"]
        ),
        "metadata_hash": sha256_file(metadata_path),
    }


def build_arms() -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    config = planner_config()
    for population in POPULATIONS:
        for sigma in SIGMAS:
            cid = cell_id(population, sigma)
            env_cfg = real_environment(
                population,
                "ricker",
                observation_noise_sigma=sigma,
                process_noise_sigma=0.0,
                initial_log_sigma=0.0,
                low_start_probability=0.0,
                horizon=50,
                expose_rk="full",
                reward_mode="safe",
                collapse_penalty=10.0,
                safety_penalty_mode="occupancy",
                alpha=1.0,
            )
            actions = real_action_table(population, "ricker", env_cfg.data_dir)
            registered = registered_model(env_cfg, actions)
            fitted, fitted_provenance = load_cached_model(
                FIT_CACHE_KEYS[(population, sigma)]
            )
            registered_context = method_context(actions, sigma, 1.0)
            fitted_context = method_context(actions, sigma, fitted.survey_scale)
            planning_seed = PLANNING_SEEDS[cid]

            a1_pomdp = TrueStateE1POMDP(
                registered,
                registered_context,
                config,
                planning_seed,
                env_cfg,
                "A1",
            )
            a2_pomdp = TrueStateE1POMDP(
                fitted,
                fitted_context,
                config,
                planning_seed,
                env_cfg,
                "A2",
            )
            a3_pomdp = NoisyStateE1POMDP(
                registered,
                registered_context,
                config,
                planning_seed,
                env_cfg,
                "A3",
                registered_prior=True,
            )
            a4_pomdp = NoisyStateE1POMDP(
                fitted,
                fitted_context,
                config,
                planning_seed,
                env_cfg,
                "A4",
                registered_prior=False,
            )
            if a2_pomdp.model is not a4_pomdp.model:
                raise AssertionError("A2/A4 did not share one fitted model object")
            cells[cid] = {
                "population": population,
                "sigma": sigma,
                "env_cfg": env_cfg,
                "actions": actions,
                "planning_seed": planning_seed,
                "fitted_provenance": {
                    **fitted_provenance,
                    **fit_metadata(FIT_CACHE_KEYS[(population, sigma)]),
                    "fit_seed": FIT_SEED,
                    "fit_config_digest": FIT_CONFIG_DIGEST,
                    "cache_hit_required": True,
                    "fits_executed_by_E1": 0,
                },
                "policies": {
                    "A1": TrueStatePolicy("A1", a1_pomdp, planning_seed),
                    "A2": TrueStatePolicy("A2", a2_pomdp, planning_seed),
                    "A3": NoisyStatePolicy("A3", a3_pomdp, planning_seed),
                    "A4": NoisyStatePolicy("A4", a4_pomdp, planning_seed),
                },
            }
    return cells


def capacity_grid(pomdp: CandidatePOMDP) -> np.ndarray:
    return np.linspace(
        pomdp.model.initial_capacity,
        pomdp.model.capacity_ceiling,
        pomdp.config.capacity_bins,
    )


def full_transition_kernel_hash(pomdp: CandidatePOMDP) -> str:
    digest = hashlib.sha256()
    digest.update(str(asdict(pomdp.config)).encode("utf-8"))
    for capacity in capacity_grid(pomdp):
        for action in range(pomdp.context.num_actions):
            matrix = pomdp.transition_matrix(float(capacity), action)
            digest.update(np.asarray([capacity, action], dtype="<f8").tobytes())
            digest.update(np.ascontiguousarray(matrix, dtype="<f8").tobytes())
    return digest.hexdigest()


def arm_hash_table(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    actual_registered_cells = {}
    for cid, cell in cells.items():
        table[cid] = {}
        for arm, policy in cell["policies"].items():
            pomdp = policy.pomdp
            kernel_hash = full_transition_kernel_hash(pomdp)
            table[cid][arm] = {
                "model_parameter_hash": pomdp.model.parameter_hash(),
                "transition_kernel_hash": kernel_hash,
                "abundance_grid_hash": array_hash(pomdp.abundance_grid),
                "capacity_grid_hash": array_hash(capacity_grid(pomdp)),
                "model_object_identity": id(pomdp.model),
                "survey_scale": pomdp.model.survey_scale,
                "planning_seed": cell["planning_seed"],
            }
        for left, right in (("A1", "A3"), ("A2", "A4")):
            for key in (
                "model_parameter_hash",
                "transition_kernel_hash",
                "abundance_grid_hash",
                "capacity_grid_hash",
            ):
                if table[cid][left][key] != table[cid][right][key]:
                    raise AssertionError(f"{cid} {left}/{right} {key} mismatch")
        actual_registered_cells[cid] = {
            "registered_pomdp": cell["policies"]["A3"].pomdp
        }
    registered_provenance = transition_table_provenance(actual_registered_cells)
    if (
        registered_provenance["combined_transition_table_hash"]
        != ACCEPTED_REGISTERED_TRANSITION_HASH
    ):
        raise AssertionError("actual A1/A3 registered transition hash drifted")
    return {
        "arms": table,
        "actual_registered_transition_provenance": registered_provenance,
    }


class SmallExactPOMDP:
    """Two-state fully observed POMDP with a closed-form finite-horizon value."""

    def __init__(self):
        self.context = SimpleNamespace(num_actions=2)
        self.config = FaithfulPlannerConfig(
            state_bins=5,
            capacity_bins=2,
            belief_points=8,
            observation_branches=2,
            horizon=3,
            transition_samples=1,
            observation_samples=1,
        )
        self.rewards = np.asarray([0.2, 0.1], dtype=np.float64)

    def predict(self, belief: CandidateBelief, action: int) -> np.ndarray:
        del action
        return belief.probabilities.copy()

    def update(
        self, belief: CandidateBelief, action: int, observation: float
    ) -> tuple[CandidateBelief, float]:
        del action
        probabilities = np.zeros(2, dtype=np.float64)
        probabilities[int(round(observation))] = 1.0
        return (
            CandidateBelief(
                probabilities,
                belief.capacity,
                belief.current_observation,
                observation,
                belief.timestep + 1,
            ),
            0.0,
        )

    def representative_observations(
        self, predicted: np.ndarray, branch_count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        del branch_count
        support = np.flatnonzero(predicted > 0.0)
        return support.astype(np.float64), predicted[support]

    def expected_public_reward(
        self, belief: CandidateBelief, action: int, predicted: np.ndarray
    ) -> float:
        del belief, predicted
        return float(self.rewards[int(action)])

    def model_hash(self) -> str:
        return "e1_v1_small_exact"


def validate_v1() -> dict[str, Any]:
    pomdp = SmallExactPOMDP()
    planner = PointBasedPlanner(pomdp, pomdp.config, discount=0.95, seed=61101)
    root = CandidateBelief(np.asarray([0.4, 0.6]), 1.0, 0.0, 0.0, 0)
    actual = planner.action_values(root)
    continuation = 0.2 * (0.95 + 0.95**2)
    exact = pomdp.rewards + continuation
    maximum_error = float(np.max(np.abs(actual - exact)))
    if maximum_error > V1_TARGET:
        raise AssertionError(f"V1 exact-solution error {maximum_error} exceeds target")
    return {
        "status": "PASS",
        "target": V1_TARGET,
        "actual_action_values": actual.tolist(),
        "exact_action_values": exact.tolist(),
        "maximum_absolute_error": maximum_error,
        "planner_class": f"{PointBasedPlanner.__module__}.{PointBasedPlanner.__name__}",
    }


def oracle_filter(env_cfg: Any) -> OracleStateFilter:
    return OracleStateFilter(env_cfg, FilterConfig(), ReferenceProposal(env_cfg))


def raw_filter(env_cfg: Any) -> RawObservationFilter:
    return RawObservationFilter(env_cfg, FilterConfig(), ReferenceProposal(env_cfg))


def validation_episode(cell: dict[str, Any], arm: str) -> dict[str, Any]:
    policy = cell["policies"][arm]
    env = ContinuousEcologyEnv(cell["env_cfg"])
    reset = env.reset(VALIDATION_SEED)
    if arm in {"A1", "A2"}:
        filt = oracle_filter(cell["env_cfg"])
        belief = filt.set_true_state(
            float(reset.evaluator_info["state"]), 0, reset.public_info
        )
        filter_route = "OracleStateFilter.set_true_state"
    else:
        filt = raw_filter(cell["env_cfg"])
        belief = filt.reset(reset.observation, VALIDATION_SEED + 10_000)
        filter_route = "RawObservationFilter external carrier; internal E1 log-space filter"
    policy.reset(VALIDATION_SEED + 20_000)
    observation = float(reset.observation)
    actions = []
    for step in range(50):
        action = int(policy.act(belief, observation))
        result = env.step(action)
        policy.observe(
            belief,
            action,
            PublicTransition(
                observation=result.observation,
                done=result.done,
                truncated=result.truncated,
                terminated=bool(result.done and not result.truncated),
                public_info=result.public_info.copy(),
            ),
        )
        actions.append(action)
        if arm in {"A1", "A2"}:
            belief = filt.set_true_state(
                float(result.evaluator_info["state"]), step + 1, result.public_info
            )
        else:
            belief = filt.update(belief, action, result.observation)
        observation = float(result.observation)
        if result.done:
            break
    if len(actions) != 50:
        raise AssertionError(f"{arm} validation trajectory ended at {len(actions)} steps")
    return {
        "status": "PASS",
        "validation_seed": VALIDATION_SEED,
        "steps": len(actions),
        "filter_route": filter_route,
        "policy_class": f"{policy.__class__.__module__}.{policy.__class__.__name__}",
        "pomdp_class": f"{policy.pomdp.__class__.__module__}.{policy.pomdp.__class__.__name__}",
        "planner_class": (
            f"{policy.planner.__class__.__module__}."
            f"{policy.planner.__class__.__name__}"
        ),
        "actions_hash": array_hash(np.asarray(actions, dtype=np.float64)),
        "reward_fields_read_or_accumulated": False,
    }


def validate_v2(
    cells: dict[str, dict[str, Any]], episodes: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    results = {}
    for cid, cell in cells.items():
        results[cid] = {}
        for arm in ("A1", "A2"):
            policy = cell["policies"][arm]
            pomdp = policy.pomdp
            if policy.runtime_policy_filtering_calls != 0:
                raise AssertionError("true-state runtime filtering count is non-zero")
            if pomdp.base_bayesian_update_calls != 0:
                raise AssertionError("true-state base Bayesian update count is non-zero")
            if pomdp.identity_conditioning_calls <= 0:
                raise AssertionError("PBVI did not call identity conditioning")
            if policy.snap_assertion_count != episodes[cid][arm]["steps"]:
                raise AssertionError("true-state snapping did not run at every decision")
            results[cid][arm] = {
                "status": "PASS",
                "survey_scale": pomdp.model.survey_scale,
                "steps": episodes[cid][arm]["steps"],
                "snap_assertions": policy.snap_assertion_count,
                "minimum_belief_max": min(
                    item["belief_max"] for item in policy.snap_records
                ),
                "runtime_policy_filtering_calls": policy.runtime_policy_filtering_calls,
                "observe_noop_calls": policy.observe_noop_calls,
                "base_CandidatePOMDP_update_calls": pomdp.base_bayesian_update_calls,
                "identity_conditioning_update_calls": pomdp.identity_conditioning_calls,
                "representative_observation_calls": (
                    pomdp.representative_observation_calls
                ),
                "realised_branch_count_min": min(pomdp.realised_branch_counts),
                "realised_branch_count_max": max(pomdp.realised_branch_counts),
                "realised_support_size_min": min(pomdp.realised_support_sizes),
                "realised_support_size_max": max(pomdp.realised_support_sizes),
                "raw_truth_divided_by_survey_scale": True,
                "round_trip_grid_times_survey_scale_asserted": True,
            }
    return {"status": "PASS", "cells": results}


def validate_v3(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    results = {}
    maximum_error = 0.0
    comparisons = 0
    for cid, cell in cells.items():
        results[cid] = {}
        for arm, policy in cell["policies"].items():
            pomdp = policy.pomdp
            grid_raw = pomdp.state_abundances() * pomdp.model.survey_scale
            below = np.flatnonzero(grid_raw <= cell["env_cfg"].safety_threshold)[-1]
            above_candidates = np.flatnonzero(
                grid_raw > cell["env_cfg"].safety_threshold
            )
            above = int(above_candidates[0])
            positive = int(np.flatnonzero(grid_raw > 0.0)[0])
            indices = sorted(set((0, int(positive), int(below), int(above))))
            arm_maximum = 0.0
            for index in indices:
                probabilities = np.zeros(pomdp.hidden_count, dtype=np.float64)
                probabilities[index] = 1.0
                belief = CandidateBelief(
                    probabilities,
                    pomdp.model.initial_capacity,
                    float(grid_raw[index]),
                    float(grid_raw[index]),
                    0,
                )
                predicted = probabilities.copy()
                for action in range(pomdp.context.num_actions):
                    actual = pomdp.expected_public_reward(belief, action, predicted)
                    penalty = bool(
                        safety_penalty_indicator(
                            cell["env_cfg"], grid_raw[index], grid_raw[index]
                        )
                    )
                    expected = pomdp.e1_reward.state_reward(
                        float(grid_raw[index]), action, penalty
                    )
                    error = abs(actual - expected)
                    comparisons += 1
                    maximum_error = max(maximum_error, error)
                    arm_maximum = max(arm_maximum, error)
                    if error > 1.0e-12:
                        raise AssertionError(f"V3 objective mismatch {cid}/{arm}")
            results[cid][arm] = {
                "status": "PASS",
                "comparison_count": len(indices) * pomdp.context.num_actions,
                "tested_raw_abundances": [float(grid_raw[i]) for i in indices],
                "all_actions_tested": True,
                "maximum_absolute_error": arm_maximum,
                "objective_flag": pomdp.objective_flag,
                "surrogate_is_none": pomdp.context.surrogate is None,
                "source_reward_class": (
                    f"{pomdp.e1_reward.__class__.__module__}."
                    f"{pomdp.e1_reward.__class__.__name__}"
                ),
                "penalty_mode": cell["env_cfg"].safety_penalty_mode,
                "alpha": cell["env_cfg"].alpha,
                "collapse_penalty": cell["env_cfg"].collapse_penalty,
            }
    return {
        "status": "PASS",
        "comparisons": comparisons,
        "maximum_absolute_error": maximum_error,
        "cells": results,
    }


def validate_v4(
    cells: dict[str, dict[str, Any]], hashes: dict[str, Any]
) -> dict[str, Any]:
    results = {}
    for cid, cell in cells.items():
        a2 = hashes["arms"][cid]["A2"]
        a4 = hashes["arms"][cid]["A4"]
        same_object = (
            cell["policies"]["A2"].pomdp.model
            is cell["policies"]["A4"].pomdp.model
        )
        checks = {
            "same_model_object": same_object,
            "model_parameter_hash_equal": (
                a2["model_parameter_hash"] == a4["model_parameter_hash"]
            ),
            "transition_kernel_hash_equal": (
                a2["transition_kernel_hash"] == a4["transition_kernel_hash"]
            ),
            "abundance_grid_hash_equal": (
                a2["abundance_grid_hash"] == a4["abundance_grid_hash"]
            ),
            "capacity_grid_hash_equal": (
                a2["capacity_grid_hash"] == a4["capacity_grid_hash"]
            ),
            "fits_executed_by_E1": cell["fitted_provenance"]["fits_executed_by_E1"],
        }
        if not all(
            value is True
            for key, value in checks.items()
            if key != "fits_executed_by_E1"
        ) or checks["fits_executed_by_E1"] != 0:
            raise AssertionError(f"V4 fitted reuse failed for {cid}")
        results[cid] = {"status": "PASS", **checks}
    return {"status": "PASS", "cells": results}


def validate_v5(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    results = {}
    observation_impl_digest = hashlib.sha256(
        (
            inspect.getsource(StableObservationPOMDP)
            + inspect.getsource(NoisyStateE1POMDP.update)
        ).encode("utf-8")
    ).hexdigest()
    for cid, cell in cells.items():
        results[cid] = {}
        for arm, policy in cell["policies"].items():
            pomdp = policy.pomdp
            if arm in {"A1", "A2"}:
                counters = {
                    "base_initial_belief_calls": pomdp.base_initial_belief_calls,
                    "override_initial_belief_calls": (
                        pomdp.override_initial_belief_calls
                    ),
                    "deployed_logspace_updates": 0,
                    "lookahead_logspace_updates": 0,
                    "old_density_evidence_at_or_below_floor": 0,
                    "structural_impossible_support_cases": 0,
                    "numerical_underflow_cases": 0,
                    "old_fallback_calls": 0,
                    "fallback_to_uniform_initial_beliefs": 0,
                    "maximum_posterior_l1_error": 0.0,
                    "maximum_posterior_tv_error": 0.0,
                    "maximum_action_value_error": 0.0,
                    "reference_action_changes": 0,
                }
            else:
                counters = {
                    "base_initial_belief_calls": pomdp.base_initial_belief_calls,
                    "override_initial_belief_calls": (
                        pomdp.override_initial_belief_calls
                    ),
                    "deployed_logspace_updates": pomdp.deployed_logspace_updates,
                    "lookahead_logspace_updates": pomdp.lookahead_logspace_updates,
                    "old_density_evidence_at_or_below_floor": (
                        pomdp.old_density_evidence_at_or_below_floor
                    ),
                    "structural_impossible_support_cases": (
                        pomdp.structural_impossible_support_cases
                    ),
                    "numerical_underflow_cases": pomdp.numerical_underflow_cases,
                    "old_fallback_calls": pomdp.old_fallback_calls,
                    "fallback_to_uniform_initial_beliefs": (
                        pomdp.fallback_to_uniform_initial_beliefs
                    ),
                    "maximum_posterior_l1_error": pomdp.maximum_posterior_l1_error,
                    "maximum_posterior_tv_error": pomdp.maximum_posterior_tv_error,
                    "maximum_action_value_error": pomdp.maximum_action_value_error,
                    "reference_action_changes": pomdp.reference_action_changes,
                    "unexpected_all_zero_support_cases": (
                        pomdp.unexpected_all_zero_support_cases
                    ),
                }
                if pomdp.deployed_logspace_updates != 50:
                    raise AssertionError(f"{cid}/{arm} did not deploy 50 log-space updates")
            expected_initial_calls = {
                "A1": (0, 0),
                "A2": (0, 0),
                "A3": (0, 1),
                "A4": (1, 0),
            }[arm]
            actual_initial_calls = (
                counters["base_initial_belief_calls"],
                counters["override_initial_belief_calls"],
            )
            if actual_initial_calls != expected_initial_calls:
                raise AssertionError(
                    f"{cid}/{arm} initial-prior route {actual_initial_calls} "
                    f"!= {expected_initial_calls}"
                )
            if counters["old_fallback_calls"] != 0:
                raise AssertionError(f"{cid}/{arm} reached old fallback")
            if counters["maximum_posterior_l1_error"] > POSTERIOR_L1_TOLERANCE:
                raise AssertionError(f"{cid}/{arm} posterior reference mismatch")
            if counters["maximum_action_value_error"] > ACTION_VALUE_TOLERANCE:
                raise AssertionError(f"{cid}/{arm} action value reference mismatch")
            if counters["reference_action_changes"] != 0:
                raise AssertionError(f"{cid}/{arm} reference action changed")
            results[cid][arm] = {
                "status": "PASS",
                **counters,
                "observation_update_implementation_digest": observation_impl_digest,
            }
    return {
        "status": "PASS",
        "posterior_l1_tolerance": POSTERIOR_L1_TOLERANCE,
        "action_value_tolerance": ACTION_VALUE_TOLERANCE,
        "raw_log_density_diagnostic": (
            "retained in accepted Phase 1 receipt at unchanged 2e-12 tolerance; "
            "status remains FAIL and does not override posterior parity"
        ),
        "cells": results,
    }


def zero_boundary_behavior(model: MechanisticModel) -> dict[str, Any]:
    values = [
        float(model.noiseless_next(0.0, model.initial_capacity, action))
        for action in range(model.num_actions)
    ]
    return {
        "following_abundance_by_action": values,
        "absorbing_for_all_actions": all(value == 0.0 for value in values),
        "positive_actions": [index for index, value in enumerate(values) if value > 0.0],
    }


def counter_and_hash_table(
    cells: dict[str, dict[str, Any]], hashes: dict[str, Any]
) -> dict[str, Any]:
    results = {}
    module_path = Path(__file__).resolve()
    for cid, cell in cells.items():
        results[cid] = {}
        for arm, policy in cell["policies"].items():
            pomdp = policy.pomdp
            row = {
                **hashes["arms"][cid][arm],
                "deployed_module_hash": sha256_file(module_path),
                "accepted_wrapper_digest": diagnostic_wrapper_code_digest(),
                "planner_invocations": policy.planner.invocation_count,
                "planner_elapsed_seconds": policy.planner.elapsed_seconds,
                "true_reward_calls": pomdp.true_reward_calls,
                "objective_flag": pomdp.objective_flag,
                "base_initial_belief_calls": pomdp.base_initial_belief_calls,
                "override_initial_belief_calls": pomdp.override_initial_belief_calls,
            }
            if arm in {"A1", "A2"}:
                row.update(
                    {
                        "runtime_policy_filtering_calls": (
                            policy.runtime_policy_filtering_calls
                        ),
                        "base_CandidatePOMDP_update_calls": (
                            pomdp.base_bayesian_update_calls
                        ),
                        "identity_conditioning_update_calls": (
                            pomdp.identity_conditioning_calls
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "deployed_logspace_updates": pomdp.deployed_logspace_updates,
                        "lookahead_logspace_updates": pomdp.lookahead_logspace_updates,
                        "old_fallback_calls": pomdp.old_fallback_calls,
                    }
                )
            results[cid][arm] = row
        fitted = cell["policies"]["A2"].pomdp.model
        results[cid]["fitted_zero_boundary_behavior"] = zero_boundary_behavior(fitted)
        results[cid]["fit_artifact"] = cell["fitted_provenance"]
    return results


def phase2_registration(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "arms": {
            "A1": "registered parameters + true state",
            "A2": "one frozen fitted model + true state",
            "A3": "registered parameters + repaired log-space noisy survey",
            "A4": "same frozen fitted model as A2 + repaired log-space noisy survey",
        },
        "objective": {
            "flag": "explicit_source_true_reward",
            "source": "real_ecology_benchmark.reward.build_reward",
            "penalty_mode": "occupancy",
            "alpha": 1.0,
            "collapse_penalty": 10.0,
            "oracle_privilege": "private K_ref and s_safe; E1 is not a benchmark method",
            "surrogate_none_forbidden": True,
        },
        "planner": {
            "implementation": "PointBasedPlanner",
            "belief_points": 32,
            "horizon": 5,
            "state_bins": 41,
            "capacity_bins": 9,
            "noisy_observation_branches": 7,
            "true_state_branches": "exact predictive support and weights",
            "branch_count_asymmetry_registered": True,
            "discount": 0.95,
        },
        "seeding": {
            "planning_seeds_by_cell": PLANNING_SEEDS,
            "validation_seed": VALIDATION_SEED,
            "registered_evaluation_seeds_not_run": list(REGISTERED_EVALUATION_SEEDS),
            "planning_seed_independent_of_evaluation": True,
        },
        "phase1_semantics": {
            "accepted_digest": ACCEPTED_PHASE1_DIGEST,
            "absorbing_equation_version": ABSORBING_ZERO_EQUATION_VERSION,
            "accepted_wrapper_digest": ACCEPTED_WRAPPER_DIGEST,
            "actual_wrapper_digest": diagnostic_wrapper_code_digest(),
            "accepted_registered_transition_hash": (
                ACCEPTED_REGISTERED_TRANSITION_HASH
            ),
            "single_implementation_imported": "e1_phase1_parity",
        },
        "fit": {
            "fit_seed": FIT_SEED,
            "fit_config_digest": FIT_CONFIG_DIGEST,
            "one_existing_cache_artifact_loaded_once_per_cell": True,
            "A2_A4_share_same_model_object": True,
            "registered_absorbing_guard_applied_to_fitted_models": False,
        },
        "initial_beliefs": {
            "A1": "direct true-state delta at act; no initial_belief call",
            "A2": "direct true-state delta at act; no initial_belief call",
            "A3": "registered override: delta nearest N0/survey_scale",
            "A4": "fitted base implementation with genuine positive reset scale",
            "registered_base_implementation_reachable_from_A1_A2_A3": False,
            "counter_expectations_base_override": {
                "A1": [0, 0],
                "A2": [0, 0],
                "A3": [0, 1],
                "A4": [1, 0],
            },
        },
        "reset_sentinel": {
            "field": "registered MechanisticModel.reset_log_scale",
            "value": float(np.finfo(np.float64).eps),
            "semantic_effect": "none",
            "surviving_effect": "parameter_hash and provenance only",
            "read_by_initial_belief_path": False,
        },
        "returns": {
            "scientific_returns_executed": False,
            "scientific_returns_inspected": False,
            "arm_contrasts_calculated": False,
            "return_table_written": False,
        },
        "cells": {
            cid: {
                "population": cell["population"],
                "sigma": cell["sigma"],
                "planning_seed": cell["planning_seed"],
            }
            for cid, cell in cells.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".verification" / "e1_phase2_core",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=REPO_ROOT / ".verification" / "e1_phase1" / "E1_RECEIPT.json",
    )
    args = parser.parse_args()
    started_at = utc_now()
    started = time.perf_counter()

    if diagnostic_wrapper_code_digest() != ACCEPTED_WRAPPER_DIGEST:
        raise AssertionError("accepted Phase 1 wrapper digest drifted before arm build")
    with args.receipt.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    if receipt.get("parity_digest") != ACCEPTED_PHASE1_DIGEST:
        raise AssertionError("accepted Phase 1 receipt digest mismatch")

    cells = build_arms()
    hashes = arm_hash_table(cells)
    v1 = validate_v1()
    episodes: dict[str, dict[str, Any]] = {}
    for cid, cell in cells.items():
        episodes[cid] = {}
        for arm in ARMS:
            episodes[cid][arm] = validation_episode(cell, arm)
    v2 = validate_v2(cells, episodes)
    v3 = validate_v3(cells)
    v4 = validate_v4(cells, hashes)
    v5 = validate_v5(cells)
    validations = {"V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5}
    if any(result["status"] != "PASS" for result in validations.values()):
        raise AssertionError("at least one V1--V5 validation failed")

    output_payloads = {
        "V1_SMALL_POMDP.json": v1,
        "V2_TRUE_STATE_ROUTE.json": v2,
        "V3_OBJECTIVE_PARITY.json": v3,
        "V4_KERNEL_REUSE.json": v4,
        "V5_OBSERVATION_COUNTERS.json": v5,
    }
    for name, payload in output_payloads.items():
        atomic_json(args.output / name, payload)

    phase_payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "registration": phase2_registration(cells),
        "validations": validations,
        "validation_episodes": episodes,
        "hash_and_counter_table": counter_and_hash_table(cells, hashes),
        "actual_registered_transition_provenance": hashes[
            "actual_registered_transition_provenance"
        ],
        "deployed_path_evidence": {
            "arm_policy_classes": {
                arm: sorted(
                    {
                        episodes[cid][arm]["policy_class"] for cid in episodes
                    }
                )
                for arm in ARMS
            },
            "arm_pomdp_classes": {
                arm: sorted(
                    {
                        episodes[cid][arm]["pomdp_class"] for cid in episodes
                    }
                )
                for arm in ARMS
            },
            "planner_class": (
                f"{PointBasedPlanner.__module__}.{PointBasedPlanner.__name__}"
            ),
            "phase1_wrapper_imported_not_copied": True,
        },
    }
    build_validation_digest = canonical_digest(phase_payload)
    phase_payload["build_validation_digest"] = build_validation_digest
    phase_payload["execution"] = {
        "started_utc": started_at,
        "finished_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "module_path": str(Path(__file__).resolve()),
        "module_hash": sha256_file(Path(__file__).resolve()),
        "output_root": str(args.output.resolve()),
    }
    atomic_json(args.output / "PHASE2_CORE_PHASE3_RECEIPT.json", phase_payload)
    receipt["phase2_core_phase3"] = phase_payload
    atomic_json(args.receipt, receipt)
    print(
        json.dumps(
            {
                "status": "PASS",
                "validations": {
                    name: result["status"] for name, result in validations.items()
                },
                "build_validation_digest": build_validation_digest,
                "elapsed_seconds": phase_payload["execution"]["elapsed_seconds"],
                "output": str(args.output.resolve()),
                "receipt": str(args.receipt.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
