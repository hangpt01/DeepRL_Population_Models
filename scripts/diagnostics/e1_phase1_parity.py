#!/usr/bin/env python3
"""E1 Phase 1 registered-kernel parity gate.

This diagnostic is deliberately independent of the E1 planner and evaluator.  It
constructs only the registered and already-fitted kernels, performs no policy
optimisation, and never evaluates scientific arm returns.  A failed component
produces a failed receipt and a non-zero exit status.
"""

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
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ECOLOGY_SRC = REPO_ROOT / "src" / "tracks" / "ecological"
if str(ECOLOGY_SRC) not in sys.path:
    sys.path.insert(0, str(ECOLOGY_SRC))

from real_ecology_benchmark import realdata  # noqa: E402
from real_ecology_benchmark.actions import (  # noqa: E402
    ActionSpec,
    action_table_hash,
    real_action_table,
)
from real_ecology_benchmark.config import (  # noqa: E402
    FaithfulPlannerConfig,
    MethodContext,
    real_environment,
)
from real_ecology_benchmark.controls import (  # noqa: E402
    advance_public_controls,
)
from real_ecology_benchmark.envs import ContinuousEcologyEnv  # noqa: E402
from real_ecology_benchmark.faithful_ecology import (  # noqa: E402
    MechanisticModel,
    fixed_regime_matrix,
)
from real_ecology_benchmark.faithful_pomdp import (  # noqa: E402
    CandidateBelief,
    CandidatePOMDP,
    LIKELIHOOD_FLOOR,
)


SCHEMA = "e1_phase1_registered_kernel_parity_v2"
ABSORBING_ZERO_EQUATION_VERSION = "e1_registered_absorbing_zero_v1"
STABLE_OBSERVATION_UPDATE_VERSION = "e1_logspace_observation_update_v1"
FLOAT_ATOL = 2.0e-12
FLOAT_RTOL = 2.0e-12
POSTERIOR_L1_TOLERANCE = 1.0e-10
ACTION_VALUE_TOLERANCE = 1.0e-8
FIXED_ACTION_ID = 0
ROLLOUT_STEPS = 50
RESET_SEEDS = (17, 47116)
POPULATIONS = ("Crab-eating fox", "Amur tiger")
SIGMAS = (0.1, 0.2)

FIT_CACHE_ROOT = Path(
    "/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/"
    "three_species_ecological_p10_correction_20260723_v1/moor/fit_cache"
)
FIT_CACHE_KEYS = {
    ("Crab-eating fox", 0.1): "f56dff2ddd98d4ae2dddb302f8e69a34437266a05b04655a4a3bfad095d1d362",
    ("Crab-eating fox", 0.2): "1a69d3abf37220e0e4067a6cf05a648bd0ef378a6b3498abfc3ec75d71789857",
    ("Amur tiger", 0.1): "c8eb8b067aaeaa3c26ebdca2f0175cc59dc2d940589db11e2d056f83bd317f47",
    ("Amur tiger", 0.2): "bab1b4157d333d18b5cdae3d4738c00efc82b70ebd74e2984c19efa628c17e44",
}

# These constants make the hash checks assertions, rather than self-comparisons.
EXPECTED_ACTION_HASHES = {
    "Crab-eating fox": "764bf4a2b1a5a611",
    "Amur tiger": "ededb366daab248e",
}
EXPECTED_ACTION_COST_HASH = (
    "1b4c7b7b3cae69b31ce018ecff04e57c209b4535db9c5c9e8fcb0f204b4bb8a4"
)
EXPECTED_TABLE_HASHES = {
    "actions.csv": "7a9005b143f222e01a4d9501c24f9972f828d6948b0a8c1509185b260d0fae75",
    "species.csv": "10d3e099d0ad0635970198d23b64b60860f09c521093d40797ebf78b8d2c01ab",
    "action_effects_long.csv": (
        "b0944a1e2f00460c9754c4697eb1c2407391e92572c4fca0f713b63985048f14"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit_float(value: float) -> float | str:
    value = float(value)
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "+inf" if value > 0.0 else "-inf"


def cell_id(population: str, sigma: float) -> str:
    label = "fox" if population == "Crab-eating fox" else "tiger"
    return f"{label}_ricker_sigma_{sigma:.1f}"


def action_cost_hash(actions: Iterable[ActionSpec]) -> str:
    costs = np.asarray([action.cost for action in actions], dtype="<f8")
    return hashlib.sha256(costs.tobytes()).hexdigest()


def planner_config() -> FaithfulPlannerConfig:
    cfg = FaithfulPlannerConfig(state_bins=41, capacity_bins=9)
    cfg.validate()
    return cfg


def method_context(actions: tuple[ActionSpec, ...], sigma: float) -> MethodContext:
    return MethodContext(
        num_actions=len(actions),
        action_costs=tuple(float(action.cost) for action in actions),
        action_channels=realdata.public_action_channels(),
        observation_noise_sigma=float(sigma),
        horizon=50,
        observation_scale=1.0,
        pop_id="pop_e1_phase1_parity",
    )


class AbsorbingZeroMechanisticModel(MechanisticModel):
    """E1 registered kernel: extinction is absorbing before every action."""

    def noiseless_next(
        self,
        abundance: np.ndarray | float,
        capacity: float,
        action: int,
        regime: np.ndarray | int = 0,
    ) -> np.ndarray:
        previous = np.asarray(abundance, dtype=np.float64)
        following = super().noiseless_next(previous, capacity, action, regime)
        return np.where(previous == 0.0, 0.0, following)


def absorbing_wrapper_code_digest() -> str:
    source = (
        ABSORBING_ZERO_EQUATION_VERSION
        + inspect.getsource(AbsorbingZeroMechanisticModel)
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def registered_model(cfg: Any, actions: tuple[ActionSpec, ...]) -> MechanisticModel:
    rates = np.asarray([action.delta_r for action in actions], dtype=np.float64)
    return AbsorbingZeroMechanisticModel(
        form="ricker",
        growth=np.maximum(rates, 0.0),
        mortality=np.maximum(-rates, 0.0),
        capacity_increment=np.asarray(
            [action.delta_K for action in actions], dtype=np.float64
        ),
        stocking=np.asarray(
            [action.stocking_delta for action in actions], dtype=np.float64
        ),
        reset_log_mean=float(math.log(cfg.N0)),
        reset_log_scale=float(np.finfo(np.float64).eps),
        initial_capacity=float(cfg.K_base),
        capacity_ceiling=float(cfg.K_max),
        process_scale=0.0,
        observation_scale=float(cfg.observation_noise_sigma),
        survey_scale=1.0,
        depensation_thresholds=np.asarray([cfg.C_low, cfg.C_high]),
        theta_exponent=float(cfg.theta_low),
        regime_multipliers=np.asarray([1.0, cfg.regime_weak_multiplier]),
        regime_matrix=fixed_regime_matrix(cfg.regime_persistence),
        action_channels=realdata.public_action_channels(cfg.data_dir),
        candidate_id=f"registered_{cell_id(cfg.population, cfg.observation_noise_sigma)}",
    )


def load_cached_model(cache_key: str) -> tuple[MechanisticModel, dict[str, Any]]:
    array_path = FIT_CACHE_ROOT / f"{cache_key}.npz"
    metadata_path = FIT_CACHE_ROOT / f"{cache_key}.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("cache_schema") != "adapted_fit_cache_v1":
        raise RuntimeError(f"unexpected fit-cache schema: {metadata_path}")
    if metadata.get("cache_key") != cache_key:
        raise RuntimeError(f"fit-cache key mismatch: {metadata_path}")
    array_hash = sha256_file(array_path)
    if array_hash != metadata.get("array_hash"):
        raise RuntimeError(f"fit-cache array hash mismatch: {array_path}")
    with np.load(array_path, allow_pickle=False) as data:
        scalar = lambda name: float(np.asarray(data[name]).reshape(-1)[0])
        model_metadata = metadata["model"]
        model = MechanisticModel(
            form=str(model_metadata["form"]),
            growth=np.asarray(data["growth"]),
            mortality=np.asarray(data["mortality"]),
            capacity_increment=np.asarray(data["capacity_increment"]),
            stocking=np.asarray(data["stocking"]),
            reset_log_mean=scalar("reset_log_mean"),
            reset_log_scale=scalar("reset_log_scale"),
            initial_capacity=scalar("initial_capacity"),
            capacity_ceiling=scalar("capacity_ceiling"),
            process_scale=scalar("process_scale"),
            observation_scale=scalar("observation_scale"),
            survey_scale=scalar("survey_scale"),
            depensation_thresholds=np.asarray(data["depensation_thresholds"]),
            theta_exponent=scalar("theta_exponent"),
            regime_multipliers=np.asarray(data["regime_multipliers"]),
            regime_matrix=np.asarray(data["regime_matrix"]),
            action_channels=tuple(model_metadata["action_channels"]),
            candidate_id=str(model_metadata["candidate_id"]),
        )
    if model.parameter_hash() != model_metadata.get("parameter_hash"):
        raise RuntimeError(f"fit-cache parameter hash mismatch: {array_path}")
    return model, {
        "cache_key": cache_key,
        "array_path": str(array_path),
        "metadata_path": str(metadata_path),
        "array_hash": array_hash,
        "metadata_hash": sha256_file(metadata_path),
        "parameter_hash": model.parameter_hash(),
    }


class CountingCandidatePOMDP(CandidatePOMDP):
    """Audit the sole authorised call path to the base reset prior (A4)."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.base_initial_belief_calls = 0
        self.override_initial_belief_calls = 0

    def initial_belief(self, observation: float) -> CandidateBelief:
        self.base_initial_belief_calls += 1
        return super().initial_belief(observation)


def stable_model_log_likelihood(
    model: MechanisticModel, observation: float, latent_states: np.ndarray
) -> np.ndarray:
    states = np.asarray(latent_states, dtype=np.float64)
    y = float(observation) / model.survey_scale
    result = np.full(states.shape, -np.inf, dtype=np.float64)
    zero = states == 0.0
    if y == 0.0:
        result[zero] = 0.0
        return result
    positive = states > 0.0
    log_ratio = np.log(y) - np.log(states[positive])
    scale = model.observation_scale
    result[positive] = (
        -0.5 * (log_ratio / scale) ** 2
        - math.log(scale)
        - 0.5 * math.log(2.0 * math.pi)
        - math.log(y)
    )
    return result


class StableObservationPOMDP(CountingCandidatePOMDP):
    """E1 deployed filter: log-space update without informative floor fallback."""

    def posterior_from_prediction(
        self, predicted: np.ndarray, observation: float
    ) -> tuple[np.ndarray, float, bool]:
        log_likelihood = np.tile(
            stable_model_log_likelihood(
                self.model, observation, self.abundance_grid
            ),
            self.regime_count,
        )
        posterior, log_evidence, informative = stable_reference_posterior(
            predicted, log_likelihood
        )
        impossible_support = not np.isfinite(log_evidence)
        return posterior, log_evidence, impossible_support and not informative

    def update(
        self, belief: CandidateBelief, action: int, observation: float
    ) -> tuple[CandidateBelief, float]:
        predicted = self.predict(belief, action)
        posterior, log_evidence, impossible_support = self.posterior_from_prediction(
            predicted, observation
        )
        if impossible_support:
            log_evidence = float(math.log(LIKELIHOOD_FLOOR))
        return (
            CandidateBelief(
                posterior,
                self.model.next_capacity(belief.capacity, action),
                belief.current_observation,
                float(observation),
                belief.timestep + 1,
            ),
            log_evidence,
        )


class RegisteredPOMDP(StableObservationPOMDP):
    """Registered-row POMDP with an explicit, observation-independent reset delta."""

    def initial_belief(self, observation: float) -> CandidateBelief:
        self.override_initial_belief_calls += 1
        latent_n0 = math.exp(self.model.reset_log_mean) / self.model.survey_scale
        abundance_index = int(np.argmin(np.abs(self.abundance_grid - latent_n0)))
        probabilities = np.zeros(self.hidden_count, dtype=np.float64)
        if self.regime_count == 1:
            probabilities[abundance_index] = 1.0
        else:
            bins = len(self.abundance_grid)
            probabilities[abundance_index] = 0.5
            probabilities[bins + abundance_index] = 0.5
        return CandidateBelief(
            probabilities=probabilities,
            capacity=self.model.initial_capacity,
            previous_observation=float(observation),
            current_observation=float(observation),
            timestep=0,
        )


def diagnostic_wrapper_code_digest() -> str:
    source = STABLE_OBSERVATION_UPDATE_VERSION + "".join(
        (
            inspect.getsource(AbsorbingZeroMechanisticModel),
            inspect.getsource(StableObservationPOMDP),
            inspect.getsource(RegisteredPOMDP),
        )
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def component(name: str) -> dict[str, Any]:
    return {"name": name, "status": "PASS", "mismatches": [], "metrics": {}}


def mismatch(record: dict[str, Any], message: str, **details: Any) -> None:
    record["status"] = "FAIL"
    record["mismatches"].append({"message": message, **details})


def close_enough(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, atol=FLOAT_ATOL, rtol=FLOAT_RTOL))


def finite_error(left: float, right: float) -> tuple[float, float]:
    absolute = abs(left - right)
    relative = absolute / max(abs(left), abs(right), np.finfo(np.float64).tiny)
    return float(absolute), float(relative)


def capacity_grid(model: MechanisticModel, cfg: FaithfulPlannerConfig) -> np.ndarray:
    return np.linspace(model.initial_capacity, model.capacity_ceiling, cfg.capacity_bins)


def preregistered_abundances(
    pomdp: CandidatePOMDP,
    env: ContinuousEcologyEnv,
    actions: tuple[ActionSpec, ...],
) -> tuple[np.ndarray, dict[str, int]]:
    grid = np.asarray(pomdp.abundance_grid * pomdp.model.survey_scale)
    midpoints = 0.5 * (grid[:-1] + grid[1:])
    cfg = env.cfg
    induced = []
    for start in (cfg.N0, cfg.safety_threshold):
        for action in actions:
            induced.append(
                env.transition_value(
                    start,
                    action,
                    r_base=0.0,
                    C=cfg.C_low,
                    theta=cfg.theta_low,
                    regime=0,
                    process_noise=0.0,
                    rho=actions[0].delta_r,
                    kappa=0.0,
                )
            )
    combined = np.unique(
        np.asarray(
            [0.0, cfg.N0, cfg.safety_threshold, *grid, *midpoints, *induced],
            dtype=np.float64,
        )
    )
    return combined, {
        "zero": 1,
        "N0": 1,
        "s_safe": 1,
        "grid_nodes": int(len(grid)),
        "adjacent_grid_midpoints": int(len(midpoints)),
        "action_induced_from_N0_and_s_safe": int(len(induced)),
        "unique_total": int(len(combined)),
    }


def check_action_integrity(
    cells: dict[str, dict[str, Any]], table_root: Path
) -> dict[str, Any]:
    record = component("action_table_and_cost_hashes")
    rows: dict[str, Any] = {}
    for population in POPULATIONS:
        actions = cells[cell_id(population, 0.1)]["actions"]
        actual_table = action_table_hash(actions)
        actual_cost = action_cost_hash(actions)
        rows[population] = {
            "action_table_hash": actual_table,
            "expected_action_table_hash": EXPECTED_ACTION_HASHES[population],
            "action_cost_hash_float64_le": actual_cost,
            "expected_action_cost_hash_float64_le": EXPECTED_ACTION_COST_HASH,
            "source_rows": [asdict(action) for action in actions],
        }
        if actual_table != EXPECTED_ACTION_HASHES[population]:
            mismatch(record, "action table hash mismatch", population=population)
        if actual_cost != EXPECTED_ACTION_COST_HASH:
            mismatch(record, "action cost hash mismatch", population=population)
    source_hashes = {}
    for name, expected in EXPECTED_TABLE_HASHES.items():
        actual = sha256_file(table_root / name)
        source_hashes[name] = {"actual": actual, "expected": expected}
        if actual != expected:
            mismatch(record, "source table hash mismatch", table=name)
    record["metrics"] = {"populations": rows, "source_table_hashes": source_hashes}
    return record


def check_transition_and_capacity(
    cells: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    transition_record = component("transition_parity_raw_abundance")
    capacity_record = component("capacity_parity_raw_abundance")
    comparisons = 0
    capacity_comparisons = 0
    max_abs = 0.0
    max_rel = 0.0
    abundance_sets = {}
    for cid, item in cells.items():
        env, model, pomdp, actions = (
            item["env"], item["registered_model"], item["registered_pomdp"], item["actions"]
        )
        abundances, categories = preregistered_abundances(pomdp, env, actions)
        abundance_sets[cid] = categories
        for capacity in capacity_grid(model, pomdp.config):
            kappa = float(capacity - model.initial_capacity)
            for action in actions:
                _rho_next, _kappa_next, expected_capacity_array = advance_public_controls(
                    env.cfg, action.id, actions[0].delta_r, kappa
                )
                expected_capacity = float(expected_capacity_array)
                actual_capacity = model.next_capacity(float(capacity), action.id)
                capacity_comparisons += 1
                if not close_enough(actual_capacity, expected_capacity):
                    mismatch(
                        capacity_record,
                        "capacity mismatch",
                        cell=cid,
                        action=action.id,
                        capacity=float(capacity),
                        candidate=actual_capacity,
                        environment=expected_capacity,
                    )
                candidate_values = np.asarray(
                    model.noiseless_next(abundances, float(capacity), action.id),
                    dtype=np.float64,
                )
                for abundance, candidate_value in zip(abundances, candidate_values):
                    environment_value = env.transition_value(
                        float(abundance),
                        action,
                        r_base=0.0,
                        C=env.cfg.C_low,
                        theta=env.cfg.theta_low,
                        regime=0,
                        process_noise=0.0,
                        rho=actions[0].delta_r,
                        kappa=kappa,
                    )
                    comparisons += 1
                    absolute, relative = finite_error(
                        float(candidate_value), environment_value
                    )
                    max_abs = max(max_abs, absolute)
                    max_rel = max(max_rel, relative)
                    if not close_enough(float(candidate_value), environment_value):
                        if len(transition_record["mismatches"]) < 100:
                            mismatch(
                                transition_record,
                                "transition mismatch",
                                cell=cid,
                                action=action.id,
                                abundance=float(abundance),
                                capacity=float(capacity),
                                candidate=float(candidate_value),
                                environment=float(environment_value),
                                absolute_error=absolute,
                                relative_error=relative,
                            )
                        else:
                            transition_record["status"] = "FAIL"
    shared_metrics = {
        "unit_convention": "raw abundance; registered survey_scale=1.0",
        "float_atol": FLOAT_ATOL,
        "float_rtol": FLOAT_RTOL,
    }
    transition_record["metrics"] = {
        **shared_metrics,
        "transition_comparisons": comparisons,
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "abundance_set_registration": abundance_sets,
        "stored_mismatch_limit": 100,
    }
    capacity_record["metrics"] = {
        **shared_metrics,
        "capacity_comparisons": capacity_comparisons,
    }
    return transition_record, capacity_record


def representative_observations(item: dict[str, Any]) -> np.ndarray:
    env, pomdp = item["env"], item["registered_pomdp"]
    positive_grid = pomdp.abundance_grid[pomdp.abundance_grid > 0.0]
    return np.unique(
        np.asarray(
            [
                0.0,
                env.cfg.safety_threshold,
                env.cfg.N0,
                0.5 * env.cfg.N0,
                2.0 * env.cfg.N0,
                math.sqrt(positive_grid[10] * positive_grid[11]),
                math.sqrt(positive_grid[-2] * positive_grid[-1]),
            ],
            dtype=np.float64,
        )
    )


def preregistered_predictions(item: dict[str, Any]) -> dict[str, np.ndarray]:
    pomdp, cfg = item["registered_pomdp"], item["cfg"]
    grid = pomdp.abundance_grid * pomdp.model.survey_scale
    count = len(grid)
    nearest = int(np.argmin(np.abs(grid - cfg.N0)))
    delta = np.zeros(count, dtype=np.float64)
    delta[nearest] = 1.0

    upper = int(np.searchsorted(grid, cfg.N0, side="left"))
    upper = min(max(upper, 1), count - 1)
    lower = upper - 1
    fraction = (cfg.N0 - grid[lower]) / (grid[upper] - grid[lower])
    barycentric = np.zeros(count, dtype=np.float64)
    barycentric[lower] = 1.0 - fraction
    barycentric[upper] = fraction

    diffuse = np.full(count, 1.0 / count, dtype=np.float64)
    tail = np.zeros(count, dtype=np.float64)
    tail[1] = 0.5
    tail[-1] = 0.5
    return {
        "delta_nearest_N0": delta,
        "two_bin_barycentric_N0": barycentric,
        "diffuse_uniform": diffuse,
        "tail_first_positive_and_upper": tail,
    }


def deployed_posterior(
    predicted: np.ndarray, likelihood: np.ndarray
) -> tuple[np.ndarray, float, bool]:
    evidence = float(np.dot(predicted, likelihood))
    fallback = evidence <= LIKELIHOOD_FLOOR or not np.isfinite(evidence)
    if fallback:
        return predicted.copy(), evidence, True
    weighted = predicted * likelihood
    return weighted / float(weighted.sum()), evidence, False


def stable_reference_posterior(
    predicted: np.ndarray, log_likelihood: np.ndarray
) -> tuple[np.ndarray, float, bool]:
    log_predicted = np.full(len(predicted), -np.inf, dtype=np.float64)
    positive = predicted > 0.0
    log_predicted[positive] = np.log(predicted[positive])
    log_weight = log_predicted + log_likelihood
    finite = np.isfinite(log_weight)
    if not np.any(finite):
        return predicted.copy(), -np.inf, False
    maximum = float(np.max(log_weight[finite]))
    weights = np.zeros(len(predicted), dtype=np.float64)
    weights[finite] = np.exp(log_weight[finite] - maximum)
    normalizer = float(weights.sum())
    posterior = weights / normalizer
    log_evidence = maximum + math.log(normalizer)
    informative = float(np.abs(posterior - predicted).sum()) > POSTERIOR_L1_TOLERANCE
    return posterior, log_evidence, informative


def diagnostic_action_values(
    pomdp: CandidatePOMDP, probabilities: np.ndarray, capacity: float
) -> np.ndarray:
    belief = CandidateBelief(probabilities, capacity, 0.0, 0.0, 0)
    following = pomdp.state_abundances() * pomdp.model.survey_scale
    return np.asarray(
        [
            float(np.dot(pomdp.predict(belief, action), following))
            for action in range(pomdp.model.num_actions)
        ],
        dtype=np.float64,
    )


def check_observation(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    record = component("observation_posterior_parity_deployed_path")
    raw_mismatches: list[dict[str, Any]] = []
    unexpected_underflows: list[dict[str, Any]] = []
    evidence_floor_cases: list[dict[str, Any]] = []
    all_zero_vectors: list[dict[str, Any]] = []
    posterior_cases = 0
    finite_comparisons = 0
    density_underflows = 0
    max_raw_abs = 0.0
    max_raw_rel = 0.0
    max_l1 = 0.0
    max_tv = 0.0
    max_action_value_difference = 0.0
    selected_action_changes: list[dict[str, Any]] = []
    maximum_legacy_l1 = 0.0
    maximum_legacy_tv = 0.0
    maximum_legacy_action_value_difference = 0.0
    legacy_selected_action_changes: list[dict[str, Any]] = []
    smallest_log = math.log(np.nextafter(0.0, 1.0))
    for cid, item in cells.items():
        model, env, pomdp = item["registered_model"], item["env"], item["registered_pomdp"]
        raw_states, _categories = preregistered_abundances(pomdp, env, item["actions"])
        observations = representative_observations(item)

        # Retain the original raw log-density comparison at its unchanged tolerance.
        for observation in observations:
            likelihood = model.observation_likelihood(float(observation), raw_states)
            environment_log = env.observation_model.log_prob(
                float(observation), raw_states
            )
            with np.errstate(divide="ignore"):
                candidate_log = np.log(likelihood)
            both_finite = np.isfinite(candidate_log) & np.isfinite(environment_log)
            for state, candidate_value, environment_value in zip(
                raw_states[both_finite],
                candidate_log[both_finite],
                environment_log[both_finite],
            ):
                finite_comparisons += 1
                absolute, relative = finite_error(
                    float(candidate_value), float(environment_value)
                )
                max_raw_abs = max(max_raw_abs, absolute)
                max_raw_rel = max(max_raw_rel, relative)
                if not close_enough(float(candidate_value), float(environment_value)):
                    raw_mismatches.append(
                        {
                            "kind": "finite log-density mismatch",
                            "cell": cid,
                            "observation": float(observation),
                            "state": float(state),
                            "candidate_log": float(candidate_value),
                            "environment_log": float(environment_value),
                        }
                    )
            underflow = (~np.isfinite(candidate_log)) & np.isfinite(environment_log)
            for state, environment_value in zip(
                raw_states[underflow], environment_log[underflow]
            ):
                density_underflows += 1
                if float(environment_value) > smallest_log + math.log(2.0):
                    finding = {
                        "kind": "candidate density unexpectedly underflowed",
                        "cell": cid,
                        "observation": float(observation),
                        "state": float(state),
                        "environment_log": float(environment_value),
                    }
                    raw_mismatches.append(finding)
                    unexpected_underflows.append(finding)

        grid_states = pomdp.state_abundances() * model.survey_scale
        predictions = preregistered_predictions(item)
        for observation in observations:
            density = model.observation_likelihood(float(observation), grid_states)
            reference_log = env.observation_model.log_prob(
                float(observation), grid_states
            )
            if np.all(density == 0.0):
                all_zero_vectors.append({"cell": cid, "observation": float(observation)})
            for prediction_name, predicted in predictions.items():
                posterior_cases += 1
                legacy, evidence, legacy_fallback = deployed_posterior(predicted, density)
                deployed, deployed_log_evidence, deployed_impossible = (
                    pomdp.posterior_from_prediction(predicted, float(observation))
                )
                stable, stable_log_evidence, informative = stable_reference_posterior(
                    predicted, reference_log
                )
                if evidence <= LIKELIHOOD_FLOOR or not np.isfinite(evidence):
                    evidence_floor_cases.append(
                        {
                            "cell": cid,
                            "observation": float(observation),
                            "prediction": prediction_name,
                            "evidence": audit_float(evidence),
                            "legacy_candidate_fallback": legacy_fallback,
                            "repaired_deployed_log_evidence": audit_float(
                                deployed_log_evidence
                            ),
                            "repaired_deployed_impossible_support": deployed_impossible,
                            "stable_log_evidence": audit_float(stable_log_evidence),
                            "stable_reference_informative": informative,
                        }
                    )
                l1 = float(np.abs(deployed - stable).sum())
                tv = 0.5 * l1
                max_l1, max_tv = max(max_l1, l1), max(max_tv, tv)
                legacy_l1 = float(np.abs(legacy - stable).sum())
                maximum_legacy_l1 = max(maximum_legacy_l1, legacy_l1)
                maximum_legacy_tv = max(maximum_legacy_tv, 0.5 * legacy_l1)
                deployed_values = diagnostic_action_values(
                    pomdp, deployed, model.initial_capacity
                )
                stable_values = diagnostic_action_values(
                    pomdp, stable, model.initial_capacity
                )
                legacy_values = diagnostic_action_values(
                    pomdp, legacy, model.initial_capacity
                )
                action_difference = float(
                    np.max(np.abs(deployed_values - stable_values))
                )
                legacy_action_difference = float(
                    np.max(np.abs(legacy_values - stable_values))
                )
                max_action_value_difference = max(
                    max_action_value_difference, action_difference
                )
                maximum_legacy_action_value_difference = max(
                    maximum_legacy_action_value_difference,
                    legacy_action_difference,
                )
                deployed_action = int(np.argmax(deployed_values))
                stable_action = int(np.argmax(stable_values))
                legacy_action = int(np.argmax(legacy_values))
                if deployed_action != stable_action:
                    selected_action_changes.append(
                        {
                            "cell": cid,
                            "observation": float(observation),
                            "prediction": prediction_name,
                            "deployed_action": deployed_action,
                            "stable_action": stable_action,
                        }
                    )
                if legacy_action != stable_action:
                    legacy_selected_action_changes.append(
                        {
                            "cell": cid,
                            "observation": float(observation),
                            "prediction": prediction_name,
                            "legacy_action": legacy_action,
                            "stable_action": stable_action,
                        }
                    )
                if deployed_impossible and informative:
                    mismatch(
                        record,
                        "repaired deployed path treated informative support as impossible",
                        cell=cid,
                        observation=float(observation),
                        prediction=prediction_name,
                        l1=l1,
                    )
                if l1 > POSTERIOR_L1_TOLERANCE:
                    mismatch(
                        record,
                        "posterior difference exceeds tolerance",
                        cell=cid,
                        observation=float(observation),
                        prediction=prediction_name,
                        l1=l1,
                        total_variation=tv,
                    )
                if action_difference > ACTION_VALUE_TOLERANCE:
                    mismatch(
                        record,
                        "diagnostic action-value difference exceeds tolerance",
                        cell=cid,
                        observation=float(observation),
                        prediction=prediction_name,
                        maximum_absolute_difference=action_difference,
                    )

        zero_states = np.asarray([0.0, env.cfg.N0], dtype=np.float64)
        zero_candidate = model.observation_likelihood(0.0, zero_states)
        zero_environment = env.observation_model.log_prob(0.0, zero_states)
        positive_candidate = model.observation_likelihood(env.cfg.N0, zero_states)
        positive_environment = env.observation_model.log_prob(env.cfg.N0, zero_states)
        supports_match = (
            np.array_equal(zero_candidate, np.asarray([1.0, 0.0]))
            and zero_environment[0] == 0.0
            and np.isneginf(zero_environment[1])
            and positive_candidate[0] == 0.0
            and np.isneginf(positive_environment[0])
            and positive_candidate[1] > 0.0
            and np.isfinite(positive_environment[1])
        )
        if not supports_match:
            mismatch(record, "explicit zero-support mismatch", cell=cid)

    underflow_impacts = []
    for finding in unexpected_underflows:
        item = cells[finding["cell"]]
        pomdp, model, env = (
            item["registered_pomdp"],
            item["registered_model"],
            item["env"],
        )
        grid = pomdp.state_abundances() * model.survey_scale
        matches = np.flatnonzero(grid == finding["state"])
        impact = {**finding, "on_deployed_grid": bool(len(matches))}
        if len(matches):
            index = int(matches[0])
            density = model.observation_likelihood(finding["observation"], grid)
            repaired_density = density.copy()
            repaired_density[index] = math.exp(finding["environment_log"])
            maximum_posterior_l1 = 0.0
            maximum_action_difference = 0.0
            classification_changed = False
            action_changed = False
            for predicted in preregistered_predictions(item).values():
                original, original_evidence, original_fallback = deployed_posterior(
                    predicted, density
                )
                repaired, repaired_evidence, repaired_fallback = deployed_posterior(
                    predicted, repaired_density
                )
                maximum_posterior_l1 = max(
                    maximum_posterior_l1, float(np.abs(original - repaired).sum())
                )
                original_values = diagnostic_action_values(
                    pomdp, original, model.initial_capacity
                )
                repaired_values = diagnostic_action_values(
                    pomdp, repaired, model.initial_capacity
                )
                maximum_action_difference = max(
                    maximum_action_difference,
                    float(np.max(np.abs(original_values - repaired_values))),
                )
                classification_changed |= original_fallback != repaired_fallback
                action_changed |= int(np.argmax(original_values)) != int(
                    np.argmax(repaired_values)
                )
            impact.update(
                {
                    "maximum_posterior_l1_change": maximum_posterior_l1,
                    "maximum_action_value_change": maximum_action_difference,
                    "evidence_classification_changed": classification_changed,
                    "selected_action_changed": action_changed,
                }
            )
        else:
            impact.update(
                {
                    "maximum_posterior_l1_change": 0.0,
                    "maximum_action_value_change": 0.0,
                    "evidence_classification_changed": False,
                    "selected_action_changed": False,
                    "reason": "raw test state is not a deployed 41-bin grid state",
                }
            )
        underflow_impacts.append(impact)

    record["metrics"] = {
        "gate": (
            "posterior parity for repaired E1 StableObservationPOMDP; frozen "
            "CandidatePOMDP density/evidence/fallback retained as legacy impact audit"
        ),
        "posterior_l1_tolerance": POSTERIOR_L1_TOLERANCE,
        "action_value_tolerance_raw_abundance": ACTION_VALUE_TOLERANCE,
        "posterior_cases": posterior_cases,
        "maximum_l1_posterior_difference": max_l1,
        "maximum_total_variation_posterior_difference": max_tv,
        "maximum_diagnostic_action_value_difference": max_action_value_difference,
        "selected_action_changes": selected_action_changes,
        "pre_repair_frozen_candidate_impact": {
            "maximum_l1_posterior_difference": maximum_legacy_l1,
            "maximum_total_variation_posterior_difference": maximum_legacy_tv,
            "maximum_diagnostic_action_value_difference": (
                maximum_legacy_action_value_difference
            ),
            "selected_action_changes": legacy_selected_action_changes,
        },
        "evidence_at_or_below_floor": evidence_floor_cases,
        "all_zero_likelihood_vectors": all_zero_vectors,
        "unexpected_underflow_impacts": underflow_impacts,
        "explicit_zero_support_status": (
            "PASS"
            if not any(
                item["message"] == "explicit zero-support mismatch"
                for item in record["mismatches"]
            )
            else "FAIL"
        ),
        "raw_log_density_diagnostic": {
            "status": "PASS" if not raw_mismatches else "FAIL",
            "tolerance_atol": FLOAT_ATOL,
            "tolerance_rtol": FLOAT_RTOL,
            "finite_comparisons": finite_comparisons,
            "floating_point_density_underflows": density_underflows,
            "maximum_absolute_error": max_raw_abs,
            "maximum_relative_error": max_raw_rel,
            "mismatches": raw_mismatches,
        },
    }
    return record


def check_deterministic_reset(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    record = component("deterministic_reset_and_registered_prior_delta")
    reset_checks = 0
    delta_checks = 0
    details = {}
    for cid, item in cells.items():
        env, pomdp = item["env"], item["registered_pomdp"]
        observations = []
        for seed in RESET_SEEDS:
            reset = env.reset(seed)
            reset_checks += 1
            observations.append(float(reset.observation))
            if reset.evaluator_info["state"] != env.cfg.N0:
                mismatch(
                    record,
                    "environment reset abundance is not exact N0",
                    cell=cid,
                    seed=seed,
                    state=reset.evaluator_info["state"],
                    N0=env.cfg.N0,
                )
        beliefs = [pomdp.initial_belief(value) for value in observations]
        for belief in beliefs:
            delta_checks += 1
            nonzero = np.flatnonzero(belief.probabilities)
            expected_index = int(
                np.argmin(
                    np.abs(
                        pomdp.abundance_grid
                        - env.cfg.N0 / pomdp.model.survey_scale
                    )
                )
            )
            if (
                len(nonzero) != 1
                or int(nonzero[0]) != expected_index
                or belief.probabilities[expected_index] != 1.0
            ):
                mismatch(
                    record,
                    "registered initial belief is not the explicit nearest-bin delta",
                    cell=cid,
                    selected=nonzero.tolist(),
                    expected=expected_index,
                )
        if not np.array_equal(beliefs[0].probabilities, beliefs[1].probabilities):
            mismatch(record, "registered reset prior depends on noisy observation", cell=cid)
        selected_index = int(np.argmax(beliefs[0].probabilities))
        details[cid] = {
            "N0_raw": env.cfg.N0,
            "survey_scale": pomdp.model.survey_scale,
            "target_latent": env.cfg.N0 / pomdp.model.survey_scale,
            "selected_bin_index": selected_index,
            "selected_bin_latent": float(pomdp.abundance_grid[selected_index]),
            "selected_bin_raw": float(
                pomdp.abundance_grid[selected_index] * pomdp.model.survey_scale
            ),
            "reset_observations": observations,
        }
    record["metrics"] = {
        "environment_reset_checks": reset_checks,
        "prior_delta_checks": delta_checks,
        "convention": "delta on bin nearest N0/survey_scale; observation ignored",
        "cells": details,
    }
    return record


def check_rollout(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    record = component("fixed_action_50_step_rollout")
    max_abs = 0.0
    max_rel = 0.0
    comparisons = 0
    finals = {}
    for cid, item in cells.items():
        env, model, actions = item["env"], item["registered_model"], item["actions"]
        action = actions[FIXED_ACTION_ID]
        candidate_state = float(env.cfg.N0)
        environment_state = float(env.cfg.N0)
        candidate_capacity = float(model.initial_capacity)
        rho = float(actions[0].delta_r)
        kappa = 0.0
        candidate_minimum = candidate_state
        environment_minimum = environment_state
        for step in range(1, ROLLOUT_STEPS + 1):
            candidate_state = float(
                model.noiseless_next(candidate_state, candidate_capacity, action.id)
            )
            candidate_capacity = model.next_capacity(candidate_capacity, action.id)
            environment_state = env.transition_value(
                environment_state,
                action,
                r_base=0.0,
                C=env.cfg.C_low,
                theta=env.cfg.theta_low,
                regime=0,
                process_noise=0.0,
                rho=rho,
                kappa=kappa,
            )
            rho_array, kappa_array, environment_capacity_array = advance_public_controls(
                env.cfg, action.id, rho, kappa
            )
            rho, kappa = float(rho_array), float(kappa_array)
            environment_capacity = float(environment_capacity_array)
            candidate_minimum = min(candidate_minimum, candidate_state)
            environment_minimum = min(environment_minimum, environment_state)
            comparisons += 2
            absolute, relative = finite_error(candidate_state, environment_state)
            max_abs, max_rel = max(max_abs, absolute), max(max_rel, relative)
            if not close_enough(candidate_state, environment_state):
                mismatch(
                    record,
                    "rollout abundance mismatch",
                    cell=cid,
                    step=step,
                    candidate=candidate_state,
                    environment=environment_state,
                )
            if not close_enough(candidate_capacity, environment_capacity):
                mismatch(
                    record,
                    "rollout capacity mismatch",
                    cell=cid,
                    step=step,
                    candidate=candidate_capacity,
                    environment=environment_capacity,
                )
        finals[cid] = {
            "candidate_abundance": candidate_state,
            "environment_abundance": environment_state,
            "candidate_capacity": candidate_capacity,
            "environment_capacity": environment_capacity,
            "candidate_minimum_abundance": candidate_minimum,
            "environment_minimum_abundance": environment_minimum,
        }
    record["metrics"] = {
        "fixed_action_id": FIXED_ACTION_ID,
        "steps": ROLLOUT_STEPS,
        "process_noise": 0.0,
        "comparisons": comparisons,
        "max_absolute_abundance_error": max_abs,
        "max_relative_abundance_error": max_rel,
        "final_values": finals,
    }
    return record


def unguarded_model(model: MechanisticModel) -> MechanisticModel:
    """Reconstruct the frozen base equation for an explicit impact comparison."""

    return MechanisticModel(**asdict(model))


def check_registered_trajectory_and_guard_impact(
    cells: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    record = component("registered_noisy_trajectory_and_absorbing_guard_impact")
    per_cell = {}
    representative_any_value_change = False
    representative_any_action_change = False
    boundary_any_value_change = False
    boundary_any_action_change = False
    for cid, item in cells.items():
        guarded = item["registered_pomdp"]
        legacy_model = unguarded_model(item["registered_model"])
        legacy = RegisteredPOMDP(
            legacy_model, item["context"], item["planner_cfg"], seed=47116
        )
        env = item["env"]
        reset = env.reset(47116)
        guarded_belief = guarded.initial_belief(reset.observation)
        legacy_belief = legacy.initial_belief(reset.observation)
        maximum_zero_mass = float(guarded_belief.probabilities[0])
        maximum_representative_value_change = 0.0
        representative_action_change_steps = []
        for step in range(ROLLOUT_STEPS + 1):
            guarded_values = diagnostic_action_values(
                guarded, guarded_belief.probabilities, guarded_belief.capacity
            )
            legacy_values = diagnostic_action_values(
                legacy, legacy_belief.probabilities, legacy_belief.capacity
            )
            value_change = float(np.max(np.abs(guarded_values - legacy_values)))
            maximum_representative_value_change = max(
                maximum_representative_value_change, value_change
            )
            representative_any_value_change |= value_change > ACTION_VALUE_TOLERANCE
            if int(np.argmax(guarded_values)) != int(np.argmax(legacy_values)):
                representative_action_change_steps.append(step)
                representative_any_action_change = True
            if step == ROLLOUT_STEPS:
                break
            result = env.step(FIXED_ACTION_ID)
            guarded_belief, _guarded_log_evidence = guarded.update(
                guarded_belief, FIXED_ACTION_ID, result.observation
            )
            legacy_belief, _legacy_log_evidence = legacy.update(
                legacy_belief, FIXED_ACTION_ID, result.observation
            )
            maximum_zero_mass = max(
                maximum_zero_mass, float(guarded_belief.probabilities[0])
            )

        guarded_zero = np.zeros(guarded.hidden_count, dtype=np.float64)
        guarded_zero[0] = 1.0
        legacy_zero = np.zeros(legacy.hidden_count, dtype=np.float64)
        legacy_zero[0] = 1.0
        guarded_boundary_values = diagnostic_action_values(
            guarded, guarded_zero, guarded.model.initial_capacity
        )
        legacy_boundary_values = diagnostic_action_values(
            legacy, legacy_zero, legacy.model.initial_capacity
        )
        boundary_value_change = float(
            np.max(np.abs(guarded_boundary_values - legacy_boundary_values))
        )
        guarded_boundary_action = int(np.argmax(guarded_boundary_values))
        legacy_boundary_action = int(np.argmax(legacy_boundary_values))
        boundary_action_changed = guarded_boundary_action != legacy_boundary_action
        boundary_any_value_change |= boundary_value_change > ACTION_VALUE_TOLERANCE
        boundary_any_action_change |= boundary_action_changed
        per_cell[cid] = {
            "trajectory": {
                "seed": 47116,
                "fixed_action_id": FIXED_ACTION_ID,
                "steps": ROLLOUT_STEPS,
                "maximum_belief_mass_at_state_zero": maximum_zero_mass,
                "maximum_diagnostic_action_value_change": (
                    maximum_representative_value_change
                ),
                "chosen_action_change_steps": representative_action_change_steps,
            },
            "extinction_boundary_delta": {
                "guarded_action_values_expected_next_raw_abundance": (
                    guarded_boundary_values.tolist()
                ),
                "unguarded_action_values_expected_next_raw_abundance": (
                    legacy_boundary_values.tolist()
                ),
                "maximum_action_value_change": boundary_value_change,
                "guarded_selected_action": guarded_boundary_action,
                "unguarded_selected_action": legacy_boundary_action,
                "selected_action_changed": boundary_action_changed,
            },
        }
    record["metrics"] = {
        "action_value_definition": (
            "one-step expected following raw abundance; diagnostic only, not a "
            "scientific reward or arm return"
        ),
        "representative_trajectory": {
            "any_action_value_changed_above_tolerance": representative_any_value_change,
            "any_chosen_action_changed": representative_any_action_change,
        },
        "extinction_boundary_delta": {
            "any_action_value_changed_above_tolerance": boundary_any_value_change,
            "any_chosen_action_changed": boundary_any_action_change,
        },
        "cells": per_cell,
    }
    return record


def transition_table_provenance(
    cells: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    per_cell = {}
    combined = hashlib.sha256()
    combined.update(ABSORBING_ZERO_EQUATION_VERSION.encode("ascii"))
    for cid in sorted(cells):
        item = cells[cid]
        pomdp = item["registered_pomdp"]
        digest = hashlib.sha256()
        digest.update(cid.encode("utf-8"))
        digest.update(str(asdict(pomdp.config)).encode("utf-8"))
        table_count = 0
        for capacity in capacity_grid(pomdp.model, pomdp.config):
            for action in range(pomdp.model.num_actions):
                table = pomdp.transition_matrix(float(capacity), action)
                digest.update(np.asarray([capacity, action], dtype="<f8").tobytes())
                digest.update(np.ascontiguousarray(table, dtype="<f8").tobytes())
                table_count += 1
        table_hash = digest.hexdigest()
        kernel_hash = canonical_digest(
            {
                "mechanistic_parameter_hash": pomdp.model.parameter_hash(),
                "absorbing_zero_equation_version": ABSORBING_ZERO_EQUATION_VERSION,
                "absorbing_wrapper_code_digest": absorbing_wrapper_code_digest(),
                "diagnostic_wrapper_code_digest": diagnostic_wrapper_code_digest(),
                "stable_observation_update_version": (
                    STABLE_OBSERVATION_UPDATE_VERSION
                ),
                "transition_table_hash": table_hash,
            }
        )
        per_cell[cid] = {
            "transition_table_hash": table_hash,
            "transition_table_count": table_count,
            "registered_kernel_hash": kernel_hash,
            "mechanistic_parameter_hash_not_sufficient": True,
        }
        combined.update(cid.encode("utf-8"))
        combined.update(table_hash.encode("ascii"))
    return {
        "absorbing_zero_equation_version": ABSORBING_ZERO_EQUATION_VERSION,
        "absorbing_wrapper_code_digest": absorbing_wrapper_code_digest(),
        "diagnostic_wrapper_code_digest": diagnostic_wrapper_code_digest(),
        "combined_transition_table_hash": combined.hexdigest(),
        "cells": per_cell,
    }


def check_sentinel_and_counters(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    record = component("reset_sentinel_provenance_and_base_override_counters")
    counters = {}
    fitted = {}
    eps = float(np.finfo(np.float64).eps)
    for cid, item in cells.items():
        model = item["registered_model"]
        if model.reset_log_scale != eps:
            mismatch(record, "registered reset sentinel is not float64 eps", cell=cid)
        alternate = replace(model, reset_log_scale=2.0 * eps)
        hash_changed = model.parameter_hash() != alternate.parameter_hash()
        if not hash_changed:
            mismatch(record, "reset sentinel does not affect parameter_hash", cell=cid)
        probes = np.asarray([0.0, item["env"].cfg.N0, item["env"].cfg.safety_threshold])
        for action in range(model.num_actions):
            if not np.array_equal(
                model.noiseless_next(probes, model.initial_capacity, action),
                alternate.noiseless_next(probes, alternate.initial_capacity, action),
            ):
                mismatch(record, "sentinel changed transition semantics", cell=cid)
        if not np.array_equal(
            model.observation_likelihood(item["env"].cfg.N0, probes),
            alternate.observation_likelihood(item["env"].cfg.N0, probes),
        ):
            mismatch(record, "sentinel changed observation semantics", cell=cid)

        # A1 and A3 each exercise the registered override. A2's direct truth delta
        # does not call initial_belief. A4 alone exercises the base implementation.
        a1 = RegisteredPOMDP(
            model, item["context"], item["planner_cfg"], seed=1101
        )
        a3 = RegisteredPOMDP(
            model, item["context"], item["planner_cfg"], seed=1103
        )
        a1.initial_belief(item["env"].cfg.N0)
        a3.initial_belief(item["env"].cfg.N0)
        fit_model, fit_provenance = load_cached_model(item["fit_cache_key"])
        fit_context = replace(
            item["context"], observation_scale=float(fit_model.survey_scale)
        )
        a4 = StableObservationPOMDP(
            fit_model, fit_context, item["planner_cfg"], seed=1104
        )
        a4_belief = a4.initial_belief(item["env"].cfg.N0)
        arm_counts = {
            "A1": {
                "base": a1.base_initial_belief_calls,
                "override": a1.override_initial_belief_calls,
            },
            "A2": {"base": 0, "override": 0, "route": "direct true-state delta at act()"},
            "A3": {
                "base": a3.base_initial_belief_calls,
                "override": a3.override_initial_belief_calls,
            },
            "A4": {
                "base": a4.base_initial_belief_calls,
                "override": a4.override_initial_belief_calls,
            },
        }
        counters[cid] = arm_counts
        if arm_counts["A1"] != {"base": 0, "override": 1}:
            mismatch(record, "A1 counter assertion failed", cell=cid)
        if arm_counts["A3"] != {"base": 0, "override": 1}:
            mismatch(record, "A3 counter assertion failed", cell=cid)
        if arm_counts["A2"]["base"] != 0 or arm_counts["A2"]["override"] != 0:
            mismatch(record, "A2 reached an initial_belief implementation", cell=cid)
        if arm_counts["A4"] != {"base": 1, "override": 0}:
            mismatch(record, "A4 counter assertion failed", cell=cid)
        if not (
            fit_model.reset_log_scale > 0.0
            and fit_model.reset_log_scale != eps
            and np.all(np.isfinite(a4_belief.probabilities))
            and math.isclose(float(a4_belief.probabilities.sum()), 1.0)
        ):
            mismatch(record, "A4 did not use a genuine positive fitted reset prior", cell=cid)
        fitted[cid] = {
            **fit_provenance,
            "survey_scale": fit_model.survey_scale,
            "reset_log_scale": fit_model.reset_log_scale,
        }
    record["metrics"] = {
        "registered_reset_log_scale": eps,
        "semantic_status": (
            "non-semantic validator sentinel; affects parameter_hash/provenance only; "
            "registered base initial_belief is unreachable"
        ),
        "counters": counters,
        "fitted_A4_models": fitted,
    }
    return record


def make_cells() -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    p_cfg = planner_config()
    for population in POPULATIONS:
        for sigma in SIGMAS:
            cfg = real_environment(
                population,
                "ricker",
                observation_noise_sigma=sigma,
                process_noise_sigma=0.0,
                initial_log_sigma=0.0,
                low_start_probability=0.0,
                horizon=50,
                expose_rk="full",
            )
            actions = real_action_table(population, "ricker", cfg.data_dir)
            context = method_context(actions, sigma)
            model = registered_model(cfg, actions)
            pomdp = RegisteredPOMDP(model, context, p_cfg, seed=47116)
            cells[cell_id(population, sigma)] = {
                "cfg": cfg,
                "actions": actions,
                "context": context,
                "env": ContinuousEcologyEnv(cfg),
                "registered_model": model,
                "registered_pomdp": pomdp,
                "planner_cfg": p_cfg,
                "fit_cache_key": FIT_CACHE_KEYS[(population, sigma)],
            }
    return cells


def receipt_registration(
    cells: dict[str, dict[str, Any]], kernel_provenance: dict[str, Any]
) -> dict[str, Any]:
    cell_rows = {}
    for cid, item in cells.items():
        cfg, model = item["cfg"], item["registered_model"]
        cell_rows[cid] = {
            "population": cfg.population,
            "family": cfg.kind,
            "observation_noise_sigma": cfg.observation_noise_sigma,
            "process_noise_sigma": cfg.process_noise_sigma,
            "N0": cfg.N0,
            "s_safe": cfg.safety_threshold,
            "K_base": cfg.K_base,
            "K_max": cfg.K_max,
            "registered_parameter_hash": model.parameter_hash(),
            "registered_kernel_hash": kernel_provenance["cells"][cid][
                "registered_kernel_hash"
            ],
            "transition_table_hash": kernel_provenance["cells"][cid][
                "transition_table_hash"
            ],
            "fit_cache_key_A2_A4": item["fit_cache_key"],
        }
    return {
        "design": {
            "primary_contrasts": [
                "A2-A4 (observability, fitted row)",
                "A1-A2 (model, true-state column)",
            ],
            "shared_arm": "A2",
            "factorial_interaction": "not estimated",
            "diagnostic_identity_only": "A4-A2-A3+A1; not a factorial estimand",
            "A1_A3": "numerical diagnostic expected approximately zero",
        },
        "red_flag": {
            "unsafe_threshold": "abs(A1-A3) >= 0.05 in either fox cell",
            "categorically_uninterpretable_threshold": "abs(A1-A3) >= 0.10",
            "response": "re-baseline; do not repair a contrast by subtracting the diagnostic",
        },
        "reset_prior": {
            "registered_A1_A3": "explicit delta on bin nearest N0/survey_scale",
            "A2": "direct true-state delta at each act; no initial_belief call",
            "base_initial_belief_reachability": "unreachable from A1/A2/A3",
            "A4": "base initial_belief with fitted model genuine positive reset_log_scale",
        },
        "sentinel": {
            "field": "registered MechanisticModel.reset_log_scale",
            "value": float(np.finfo(np.float64).eps),
            "purpose": "satisfy frozen dataclass positive-parameter validator",
            "semantic_effect": "none",
            "surviving_effect": "parameter_hash and provenance only",
        },
        "registered_model_mapping": {
            "units": "raw abundance",
            "survey_scale": 1.0,
            "absorbing_zero_equation_version": ABSORBING_ZERO_EQUATION_VERSION,
            "absorbing_zero": (
                "if previous abundance == 0, following abundance == 0 for every action"
            ),
            "absorbing_wrapper_code_digest": absorbing_wrapper_code_digest(),
            "diagnostic_wrapper_code_digest": diagnostic_wrapper_code_digest(),
            "stable_observation_update_version": STABLE_OBSERVATION_UPDATE_VERSION,
            "kernel_identity": (
                "registered_kernel_hash combines MechanisticModel.parameter_hash, "
                "absorbing equation/code identity, and transition-table hash"
            ),
            "signed_rate_transform": "growth=max(delta_r,0); mortality=max(-delta_r,0)",
            "capacity": "cumulative delta_K clipped to [K_base,K_max]",
            "stocking": "action stocking_delta added before growth",
            "reset": "environment state exactly N0; registered belief explicit nearest-bin delta",
        },
        "registered_kernel_provenance": kernel_provenance,
        "cells": cell_rows,
    }


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=REPO_ROOT / ".verification" / "e1_phase1" / "E1_RECEIPT.json",
    )
    args = parser.parse_args()
    started_wall = utc_now()
    started = time.perf_counter()
    cells = make_cells()
    table_root = REPO_ROOT / "src" / "tracks" / "real_ecology_data"
    transition_result, capacity_result = check_transition_and_capacity(cells)
    trajectory_result = check_registered_trajectory_and_guard_impact(cells)
    kernel_provenance = transition_table_provenance(cells)
    kernel_identity_result = component("registered_kernel_identity_and_transition_hashes")
    kernel_identity_result["metrics"] = kernel_provenance
    components = [
        check_action_integrity(cells, table_root),
        transition_result,
        capacity_result,
        check_observation(cells),
        check_deterministic_reset(cells),
        check_rollout(cells),
        trajectory_result,
        kernel_identity_result,
        check_sentinel_and_counters(cells),
    ]
    overall = "PASS" if all(item["status"] == "PASS" for item in components) else "FAIL"
    digest_payload = {
        "schema": SCHEMA,
        "registration": receipt_registration(cells, kernel_provenance),
        "components": components,
    }
    receipt = {
        **digest_payload,
        "overall_status": overall,
        "parity_digest": canonical_digest(digest_payload),
        "execution": {
            "started_utc": started_wall,
            "finished_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "diagnostic_path": str(Path(__file__).resolve()),
            "diagnostic_sha256": sha256_file(Path(__file__).resolve()),
            "receipt_path": str(args.receipt.resolve()),
            "scientific_arms_executed": False,
            "planning_executed": False,
        },
    }
    write_receipt(args.receipt, receipt)
    print(json.dumps({
        "overall_status": overall,
        "components": {item["name"]: item["status"] for item in components},
        "parity_digest": receipt["parity_digest"],
        "receipt": str(args.receipt.resolve()),
        "elapsed_seconds": receipt["execution"]["elapsed_seconds"],
    }, indent=2, sort_keys=True))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
